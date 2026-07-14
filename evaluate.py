import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, Dataset

from tooth_dataset import build_transform
from visualize import load_model, load_training_args


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}
METRIC_KEYS = (
    'global_similarity', 'mean_similarity', 'max_similarity', 'inside_similarity', 'outside_similarity', 'contrast',
    'peak_hit', 'top1_recall', 'top5_recall', 'top10_recall', 'oracle_size_iou',
    'center_error_norm', 'target_rank_percentile',
)


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = PIL.Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(image), index


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate crop-to-full-mouth patch similarity using a trained Swin-MAE encoder.'
    )
    parser.add_argument('--checkpoint', default='exp/swinmae_v1/checkpoint-400.pth',
                        help='trained Swin-MAE checkpoint path')
    parser.add_argument('--config', default='config.yaml',
                        help='training YAML used to construct the model')
    parser.add_argument('--data_path', default=None,
                        help='intraoral dataset root; defaults to DATA.DATA_PATH in the YAML')
    parser.add_argument('--collector', default='zeyu', type=str,
                        help='collector subdirectory to evaluate; set to an empty string to use all collectors')
    parser.add_argument('--output_dir', default=None,
                        help='result directory; defaults to <checkpoint parent>/eval')
    parser.add_argument('--categories', nargs='+', default=['single_tooth', 'sextant'],
                        choices=['single_tooth', 'sextant'],
                        help='crop categories to compare against process full-mouth images')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_localization_images', default=20, type=int,
                        help='number of sampled single-tooth localization figures to save; set 0 to disable')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed used when filling visualization samples')
    return parser


def image_files(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def relative_posix_path(path):
    return Path(str(path).replace('\\', '/')).as_posix()


def normalize_fdi(value):
    """Use the same FDI identifier for JSON numbers and crop filenames."""
    text = str(value).strip()
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    return str(int(numeric_value)) if numeric_value.is_integer() else text


def load_tooth_annotations(processed_root):
    """Index zeyu tooth_bbox JSON annotations by (case_id, view, fdi)."""
    tooth_bbox_root = processed_root / 'tooth_bbox'
    if not tooth_bbox_root.is_dir():
        raise FileNotFoundError(f'Missing tooth bbox directory: {tooth_bbox_root}')

    annotations = {}
    for annotation_path in sorted(tooth_bbox_root.rglob('*.json')):
        with annotation_path.open('r', encoding='utf-8') as handle:
            annotation = json.load(handle)
        sample_id = annotation_path.parent.name
        view = annotation_path.stem
        if str(annotation['case_id']) != sample_id:
            raise ValueError(
                f"case_id mismatch in {annotation_path}: {annotation['case_id']} != {sample_id}"
            )
        if str(annotation['view']) != view:
            raise ValueError(f"view mismatch in {annotation_path}: {annotation['view']} != {view}")
        image_path = relative_posix_path(annotation['image_path'])
        for tooth in annotation['teeth']:
            fdi = normalize_fdi(tooth['fdi'])
            key = (sample_id, view, fdi)
            if key in annotations:
                raise ValueError(f'Duplicate FDI entry in tooth annotations: {key}')
            annotations[key] = {
                'annotation_path': annotation_path,
                'fdi': fdi,
                # The single-tooth crops are made from this integer, clamped box.
                'box': tooth['bbox_padded'],
                'image_path': image_path,
            }
    return annotations


def build_pairs(data_path, categories):
    crop_records = []
    skipped_records = []

    for root, dirnames, _ in os.walk(data_path):
        for dirname in dirnames:
            if not dirname.endswith('_process'):
                continue

            processed_root = Path(root) / dirname
            full_images = {}
            process_root = processed_root / 'process'
            if process_root.is_dir():
                for sample_dir in process_root.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    for full_path in image_files(sample_dir):
                        full_images[relative_posix_path(full_path.relative_to(processed_root))] = full_path

            tooth_annotations = load_tooth_annotations(processed_root) if 'single_tooth' in categories else {}

            for category in categories:
                crop_root = processed_root / category
                if not crop_root.is_dir():
                    continue
                for sample_dir in crop_root.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    for view_dir in sample_dir.iterdir():
                        if not view_dir.is_dir():
                            continue
                        for crop_path in image_files(view_dir):
                            crop_relative_path = relative_posix_path(crop_path.relative_to(processed_root))
                            tooth_annotation = None
                            if category == 'single_tooth':
                                tooth_key = (sample_dir.name, view_dir.name, normalize_fdi(crop_path.stem))
                                tooth_annotation = tooth_annotations.get(tooth_key)
                                if tooth_annotation is None:
                                    annotation_path = (
                                        processed_root / 'tooth_bbox' / sample_dir.name / f'{view_dir.name}.json'
                                    )
                                    available_fdis = sorted(
                                        key[2] for key in tooth_annotations
                                        if key[:2] == tooth_key[:2]
                                    )
                                    skipped_records.append({
                                        'category': category,
                                        'sample_id': sample_dir.name,
                                        'view': view_dir.name,
                                        'label': crop_path.stem,
                                        'crop_path': str(crop_path),
                                        'annotation_path': str(annotation_path),
                                        'reason': 'fdi_not_in_tooth_bbox',
                                        'available_fdis': ','.join(available_fdis),
                                    })
                                    continue
                                if crop_path.stem != tooth_annotation['fdi']:
                                    raise ValueError(
                                        f'FDI/crop filename mismatch for {crop_relative_path}: '
                                        f"{tooth_annotation['fdi']} != {crop_path.stem}"
                                    )
                                full_relative_path = tooth_annotation['image_path']
                            else:
                                full_relative_path = f'process/{sample_dir.name}/{view_dir.name}.png'

                            full_path = full_images.get(full_relative_path)
                            if full_path is None:
                                raise FileNotFoundError(
                                    f'Full-mouth image for {crop_relative_path} is absent: {full_relative_path}'
                                )
                            crop_records.append({
                                'category': category,
                                'sample_id': sample_dir.name,
                                'view': view_dir.name,
                                'label': crop_path.stem,
                                'crop_path': crop_path,
                                'full_path': full_path,
                                'tooth_annotation': tooth_annotation,
                            })

    return crop_records, skipped_records


@torch.no_grad()
def extract_encoder_maps(model, image_paths, transform, device, batch_size, num_workers, input_size):
    dataset = ImagePathDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == 'cuda', shuffle=False)
    features = [None] * len(dataset)
    model.eval()
    for images, indices in loader:
        if tuple(images.shape[-2:]) != (input_size, input_size):
            raise RuntimeError(
                f'Evaluation transform produced {tuple(images.shape[-2:])}, '
                f'but the model expects {(input_size, input_size)}.'
            )
        encoded = model.forward_encoder_features(images.to(device, non_blocking=True)).cpu()
        for index, feature in zip(indices.tolist(), encoded):
            features[index] = feature
    return {str(path): feature for path, feature in zip(image_paths, features)}


def transform_box_for_eval(box, original_size, input_size):
    width, height = original_size
    crop_pct = 224 / 256 if input_size <= 224 else 1.0
    resize_size = int(input_size / crop_pct)
    if width < height:
        resized_width = resize_size
        resized_height = int(resize_size * height / width)
    else:
        resized_width = int(resize_size * width / height)
        resized_height = resize_size
    scale_x = resized_width / width
    scale_y = resized_height / height
    crop_left = max(int(round((resized_width - input_size) / 2.0)), 0)
    crop_top = max(int(round((resized_height - input_size) / 2.0)), 0)
    x1, y1, x2, y2 = (float(value) for value in box)
    transformed = (
        x1 * scale_x - crop_left,
        y1 * scale_y - crop_top,
        x2 * scale_x - crop_left,
        y2 * scale_y - crop_top,
    )
    return tuple(max(0.0, min(float(input_size), value)) for value in transformed)


def patch_overlap(box, grid_height, grid_width, input_size, device):
    patch_width = input_size / grid_width
    patch_height = input_size / grid_height
    overlap = torch.zeros((grid_height, grid_width), dtype=torch.float32, device=device)
    x1, y1, x2, y2 = box
    for row in range(grid_height):
        patch_y1, patch_y2 = row * patch_height, (row + 1) * patch_height
        for col in range(grid_width):
            patch_x1, patch_x2 = col * patch_width, (col + 1) * patch_width
            intersection = max(0.0, min(x2, patch_x2) - max(x1, patch_x1)) * max(
                0.0, min(y2, patch_y2) - max(y1, patch_y1)
            )
            overlap[row, col] = intersection / (patch_width * patch_height)
    return overlap.reshape(-1)


def box_iou(box_a, box_b):
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-8)


def predict_box_from_scores(scores, box, grid_height, grid_width, input_size):
    patch_width = input_size / grid_width
    patch_height = input_size / grid_height
    box_width = max(1, min(grid_width, round((box[2] - box[0]) / patch_width)))
    box_height = max(1, min(grid_height, round((box[3] - box[1]) / patch_height)))
    score_grid = scores.reshape(1, 1, grid_height, grid_width)
    kernel = torch.ones((1, 1, box_height, box_width), device=scores.device)
    pooled = F.conv2d(score_grid, kernel) / float(box_height * box_width)
    index = int(pooled.reshape(-1).argmax().item())
    output_width = grid_width - box_width + 1
    row, col = divmod(index, output_width)
    return (
        col * patch_width,
        row * patch_height,
        min((col + box_width) * patch_width, input_size),
        min((row + box_height) * patch_height, input_size),
    )


def load_tooth_box(record, input_size):
    annotation = record['tooth_annotation']
    if annotation is None:
        return None
    with PIL.Image.open(record['full_path']) as image:
        transformed = transform_box_for_eval(annotation['box'], image.size, input_size)
    return transformed


def similarity_metrics(scores, global_similarity, target_box, grid_height, grid_width, input_size):
    metrics = {
        'global_similarity': float(global_similarity.item()),
        'mean_similarity': float(scores.mean().item()),
        'max_similarity': float(scores.max().item()),
    }
    if target_box is None or target_box[2] <= target_box[0] or target_box[3] <= target_box[1]:
        return metrics

    overlap = patch_overlap(target_box, grid_height, grid_width, input_size, scores.device)
    positive = overlap > 0
    outside = overlap == 0
    inside = (scores * overlap).sum() / overlap.sum().clamp_min(1e-8)
    outside_score = scores[outside].mean() if outside.any() else scores.new_tensor(float('nan'))
    ranked = scores.argsort(descending=True)
    best_positive = scores[positive].max() if positive.any() else scores.new_tensor(float('nan'))
    rank = 1 + int((scores > best_positive).sum().item())
    predicted_box = predict_box_from_scores(scores, target_box, grid_height, grid_width, input_size)
    target_center = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    predicted_center = ((predicted_box[0] + predicted_box[2]) / 2, (predicted_box[1] + predicted_box[3]) / 2)
    metrics.update({
        'inside_similarity': float(inside.item()),
        'outside_similarity': float(outside_score.item()),
        'contrast': float((inside - outside_score).item()),
        'peak_hit': int(positive[ranked[0]].item()),
        'top1_recall': int(positive[ranked[:1]].any().item()),
        'top5_recall': int(positive[ranked[:5]].any().item()),
        'top10_recall': int(positive[ranked[:10]].any().item()),
        'oracle_size_iou': box_iou(predicted_box, target_box),
        'center_error_norm': math.dist(target_center, predicted_center) / math.hypot(input_size, input_size),
        'target_rank_percentile': (rank - 1) / max(scores.numel() - 1, 1),
    })
    return metrics


def metric_summary(rows):
    summary = {'count': len(rows)}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        summary[key] = round(sum(values) / len(values), 6) if values else None
        summary[f'{key}_median'] = round(float(torch.tensor(values).median().item()), 6) if values else None
    return summary


def grouped_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {group: metric_summary(items) for group, items in sorted(groups.items())}


def select_localization_samples(records, budget, seed):
    """Select low, median, and high localization-IoU examples per intraoral view."""
    candidates = [
        record for record in records
        if record['category'] == 'single_tooth' and record['bbox_in_view']
    ]
    if budget <= 0 or not candidates:
        return []

    by_view = defaultdict(list)
    for record in candidates:
        by_view[record['view']].append(record)
    views = sorted(by_view)
    quotas = {view: budget // len(views) for view in views}
    for view in views[:budget % len(views)]:
        quotas[view] += 1

    rng = random.Random(seed)
    selected = []
    for view in views:
        items = sorted(by_view[view], key=lambda record: (record['oracle_size_iou'], record['crop_path']))
        chosen = []
        chosen_paths = set()
        for index, reason in ((0, 'low_iou'), (len(items) // 2, 'median_iou'), (len(items) - 1, 'high_iou')):
            if len(chosen) >= quotas[view]:
                break
            candidate = items[index]
            if candidate['crop_path'] not in chosen_paths:
                candidate = dict(candidate)
                candidate['selection_reason'] = reason
                chosen.append(candidate)
                chosen_paths.add(candidate['crop_path'])

        remaining = [item for item in items if item['crop_path'] not in chosen_paths]
        rng.shuffle(remaining)
        for candidate in remaining[:max(0, quotas[view] - len(chosen))]:
            candidate = dict(candidate)
            candidate['selection_reason'] = 'seeded_fill'
            chosen.append(candidate)
        selected.extend(chosen)
    return selected


def tensor_to_rgb(tensor):
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def transformed_rgb(image_path, transform):
    with PIL.Image.open(image_path) as image:
        return tensor_to_rgb(transform(image.convert('RGB')))


def draw_box(axis, box, color, label):
    axis.add_patch(Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=2,
    ))
    axis.text(
        box[0], max(box[1] - 4, 8), label, color=color, fontsize=8,
        bbox={'facecolor': 'black', 'alpha': 0.55, 'pad': 1, 'edgecolor': 'none'},
    )


def safe_name(value):
    return ''.join(character if character.isalnum() or character in '-_.' else '_' for character in str(value))


def save_localization_figure(record, transform, input_size, output_dir):
    full_rgb = transformed_rgb(record['full_path'], transform)
    crop_rgb = transformed_rgb(record['crop_path'], transform)
    score_map = record['scores'].reshape(record['grid_height'], record['grid_width']).numpy()
    normalized_map = (score_map - score_map.min()) / max(float(score_map.max() - score_map.min()), 1e-8)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(crop_rgb)
    axes[0].set_title(f"Tooth query: {record['label']}")
    for axis, heatmap, title in (
        (axes[1], score_map, 'Cosine similarity'),
        (axes[2], normalized_map, 'Min-max similarity'),
    ):
        axis.imshow(full_rgb)
        rendered = axis.imshow(
            heatmap, cmap='magma', alpha=0.55,
            extent=(0, input_size, input_size, 0), aspect='auto',
        )
        draw_box(axis, record['target_box'], 'lime', 'GT')
        draw_box(axis, record['predicted_box'], 'red', 'Prediction')
        axis.set_title(title)
        figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        f"{record['sample_id']} {record['view']} FDI {record['label']} | "
        f"IoU={record['oracle_size_iou']:.3f} | contrast={record['contrast']:.3f} | "
        f"peak hit={record['peak_hit']}",
        fontsize=11,
    )
    figure.tight_layout()
    filename = (
        f"{safe_name(record['sample_id'])}_{safe_name(record['view'])}_"
        f"{safe_name(record['label'])}_{safe_name(record['selection_reason'])}.png"
    )
    figure.savefig(output_dir / filename, dpi=160, bbox_inches='tight')
    plt.close(figure)


def save_localization_visualizations(records, transform, input_size, output_dir, budget, seed):
    if budget <= 0:
        return 0, None
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_localization_samples(records, budget, seed)
    selection_path = output_dir / 'selection.csv'
    fields = [
        'sample_id', 'view', 'label', 'crop_path', 'selection_reason', 'oracle_size_iou',
        'contrast', 'peak_hit', 'top1_recall', 'target_rank_percentile',
    ]
    with selection_path.open('w', newline='', encoding='utf-8') as selection_file:
        writer = csv.DictWriter(selection_file, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            writer.writerow({field: record[field] for field in fields})
            save_localization_figure(record, transform, input_size, output_dir)
    return len(selected), selection_path


def save_results(records, skipped_records, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'per_crop_similarity.csv'
    fields = [
        'category', 'sample_id', 'view', 'label', 'crop_path', 'full_mouth_path',
        'bbox_available', 'bbox_in_view', *METRIC_KEYS,
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    skipped_path = output_dir / 'skipped_samples.csv'
    skipped_fields = [
        'category', 'sample_id', 'view', 'label', 'crop_path', 'annotation_path', 'reason', 'available_fdis',
    ]
    with skipped_path.open('w', newline='', encoding='utf-8') as skipped_file:
        writer = csv.DictWriter(skipped_file, fieldnames=skipped_fields)
        writer.writeheader()
        writer.writerows(skipped_records)

    summary_path = output_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as summary_file:
        json.dump(metadata, summary_file, ensure_ascii=False, indent=2)
    return csv_path, skipped_path, summary_path


def main(args):
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available. Use --device cpu if needed.')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    train_args = load_training_args(args.config)
    data_root = Path(args.data_path or train_args.data_path)
    data_path = data_root / args.collector if args.collector else data_root
    if not data_path.is_dir():
        raise FileNotFoundError(f'Evaluation data directory not found: {data_path}')
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / 'eval')
    device = torch.device(args.device)
    pairs, skipped_records = build_pairs(data_path, args.categories)
    if not pairs:
        raise ValueError(f'No crop/full-mouth pairs found under {data_path}')

    image_paths = {record['full_path'] for record in pairs}
    image_paths.update(record['crop_path'] for record in pairs)
    transform = build_transform(is_train=False, args=train_args)
    model, checkpoint_epoch = load_model(args.checkpoint, train_args, device)
    feature_maps = extract_encoder_maps(
        model,
        sorted(image_paths),
        transform,
        device,
        args.batch_size,
        args.num_workers,
        train_args.input_size,
    )

    output_records = []
    visualization_records = []
    for record in pairs:
        full_map = feature_maps[str(record['full_path'])]
        crop_map = feature_maps[str(record['crop_path'])]
        grid_height, grid_width, _ = full_map.shape
        crop_query = F.normalize(crop_map.mean(dim=(0, 1)), dim=0)
        full_tokens = F.normalize(full_map.reshape(-1, full_map.shape[-1]), dim=1)
        scores = full_tokens @ crop_query
        target_box = load_tooth_box(record, train_args.input_size)
        bbox_available = record['tooth_annotation'] is not None
        has_valid_box = (
            target_box is not None and target_box[2] > target_box[0] and target_box[3] > target_box[1]
        )
        full_global = F.normalize(full_map.mean(dim=(0, 1)), dim=0)
        global_similarity = torch.dot(crop_query, full_global)
        metrics = similarity_metrics(
            scores, global_similarity, target_box, grid_height, grid_width, train_args.input_size
        )
        output_record = {
            'category': record['category'],
            'sample_id': record['sample_id'],
            'view': record['view'],
            'label': record['label'],
            'crop_path': str(record['crop_path']),
            'full_mouth_path': str(record['full_path']),
            'bbox_available': bbox_available,
            'bbox_in_view': has_valid_box,
            **metrics,
        }
        output_records.append(output_record)
        if record['category'] == 'single_tooth' and has_valid_box:
            visualization_records.append({
                **output_record,
                'full_path': str(record['full_path']),
                'scores': scores.detach().cpu(),
                'grid_height': grid_height,
                'grid_width': grid_width,
                'target_box': target_box,
                'predicted_box': predict_box_from_scores(
                    scores, target_box, grid_height, grid_width, train_args.input_size
                ),
            })

    localization_count, localization_selection_path = save_localization_visualizations(
        visualization_records,
        transform,
        train_args.input_size,
        output_dir / 'localization',
        args.num_localization_images,
        args.seed,
    )

    metadata = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_epoch': checkpoint_epoch,
        'config': os.path.abspath(args.config),
        'data_path': str(data_path.resolve()),
        'collector': args.collector or None,
        'feature': 'crop_global_average_pooling_vs_full_encoder_patch_tokens_cosine_similarity',
        'bbox_coverage': {
            'annotation_available': sum(record['bbox_available'] for record in output_records),
            'annotation_missing': sum(not record['bbox_available'] for record in output_records),
            'in_model_input': sum(record['bbox_in_view'] for record in output_records),
            'outside_model_input': sum(
                record['bbox_available'] and not record['bbox_in_view'] for record in output_records
            ),
        },
        'skipped_samples': {
            'count': len(skipped_records),
            'path': str((output_dir / 'skipped_samples.csv').resolve()),
        },
        'localization_visualizations': {
            'count': localization_count,
            'directory': str((output_dir / 'localization').resolve()) if localization_count else None,
            'selection_csv': str(localization_selection_path.resolve()) if localization_selection_path else None,
        },
        'overall': metric_summary(output_records),
        'by_category': grouped_summary(output_records, 'category'),
        'by_view': grouped_summary(output_records, 'view'),
        'by_tooth_label': grouped_summary(
            [record for record in output_records if record['category'] == 'single_tooth'], 'label'
        ),
    }
    csv_path, skipped_path, summary_path = save_results(output_records, skipped_records, output_dir, metadata)
    print(f'Pairs evaluated: {len(output_records)}')
    print(f'Samples skipped for missing tooth bbox: {len(skipped_records)}')
    print(f'Localization visualizations: {localization_count}')
    print(f'Similarities: {csv_path}')
    print(f'Skipped samples: {skipped_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
