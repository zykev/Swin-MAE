import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import PIL.Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tooth_dataset import build_transform
from visualize import load_model, load_training_args


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}
METRIC_KEYS = (
    'mean_similarity', 'max_similarity', 'inside_similarity', 'outside_similarity', 'contrast',
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
    return parser


def image_files(directory):
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def build_pairs(data_path, categories):
    full_images = {}
    crop_records = []

    for root, dirnames, _ in os.walk(data_path):
        for dirname in dirnames:
            if not dirname.endswith('_process'):
                continue

            processed_root = Path(root) / dirname
            process_root = processed_root / 'process'
            if process_root.is_dir():
                for sample_dir in process_root.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    for full_path in image_files(sample_dir):
                        key = (str(processed_root.resolve()), sample_dir.name, full_path.stem)
                        full_images[key] = full_path

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
                        full_key = (str(processed_root.resolve()), sample_dir.name, view_dir.name)
                        full_path = full_images.get(full_key)
                        if full_path is None:
                            continue
                        bbox_path = processed_root / 'tooth_bbox' / sample_dir.name / f'{view_dir.name}.json'
                        for crop_path in image_files(view_dir):
                            crop_records.append({
                                'category': category,
                                'sample_id': sample_dir.name,
                                'view': view_dir.name,
                                'label': crop_path.stem,
                                'crop_path': crop_path,
                                'full_path': full_path,
                                'bbox_path': bbox_path if category == 'single_tooth' and bbox_path.is_file() else None,
                            })

    return crop_records


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
    resize_scale = input_size / min(width, height)
    resized_width = round(width * resize_scale)
    resized_height = round(height * resize_scale)
    crop_left = max((resized_width - input_size) // 2, 0)
    crop_top = max((resized_height - input_size) // 2, 0)
    x1, y1, x2, y2 = (float(value) for value in box)
    transformed = (
        x1 * resize_scale - crop_left,
        y1 * resize_scale - crop_top,
        x2 * resize_scale - crop_left,
        y2 * resize_scale - crop_top,
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
    if record['bbox_path'] is None:
        return None
    with record['bbox_path'].open('r', encoding='utf-8') as handle:
        annotation = json.load(handle)
    entries = annotation.get(record['label'])
    if not entries:
        return None
    with PIL.Image.open(record['full_path']) as image:
        return transform_box_for_eval(entries[0]['box'], image.size, input_size)


def similarity_metrics(scores, target_box, grid_height, grid_width, input_size):
    metrics = {
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


def save_results(records, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'per_crop_similarity.csv'
    fields = ['category', 'sample_id', 'view', 'label', 'crop_path', 'full_mouth_path', *METRIC_KEYS]
    with csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary_path = output_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as summary_file:
        json.dump(metadata, summary_file, ensure_ascii=False, indent=2)
    return csv_path, summary_path


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
    pairs = build_pairs(data_path, args.categories)
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
    for record in pairs:
        full_map = feature_maps[str(record['full_path'])]
        crop_map = feature_maps[str(record['crop_path'])]
        grid_height, grid_width, _ = full_map.shape
        crop_query = F.normalize(crop_map.mean(dim=(0, 1)), dim=0)
        full_tokens = F.normalize(full_map.reshape(-1, full_map.shape[-1]), dim=1)
        scores = full_tokens @ crop_query
        target_box = load_tooth_box(record, train_args.input_size)
        metrics = similarity_metrics(scores, target_box, grid_height, grid_width, train_args.input_size)
        output_records.append({
            'category': record['category'],
            'sample_id': record['sample_id'],
            'view': record['view'],
            'label': record['label'],
            'crop_path': str(record['crop_path']),
            'full_mouth_path': str(record['full_path']),
            **metrics,
        })

    metadata = {
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_epoch': checkpoint_epoch,
        'config': os.path.abspath(args.config),
        'data_path': str(data_path.resolve()),
        'collector': args.collector or None,
        'feature': 'crop_global_average_pooling_vs_full_encoder_patch_tokens_cosine_similarity',
        'overall': metric_summary(output_records),
        'by_category': grouped_summary(output_records, 'category'),
        'by_view': grouped_summary(output_records, 'view'),
        'by_tooth_label': grouped_summary(
            [record for record in output_records if record['category'] == 'single_tooth'], 'label'
        ),
    }
    csv_path, summary_path = save_results(output_records, output_dir, metadata)
    print(f'Pairs evaluated: {len(output_records)}')
    print(f'Similarities: {csv_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main(get_args_parser().parse_args())
