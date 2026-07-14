"""Evaluate crop-to-full alignment checkpoints.

The quantitative pass evaluates every detected crop in the selected split exactly once. A second,
deterministic pass over selected full images writes similarity-localization and reconstruction
figures. Unlike training, this script does not expand annotations into crop chunks.
"""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

from datasets.dataset import _to_tensor
from datasets.split import create_or_load_split, resolve_split_paths
from datasets.transforms import letterbox_bbox, letterbox_image
from loss.alignment import compute_overlap, cosine_sim_matrix
from loss.reconstruction import patchify, reconstruction_loss, unpatchify
from models.heads import make_student_teacher
from train import build_encoder_decoder, debug_json_paths, load_yaml_config
from utils.masking import sample_batch_mask_ratios


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
METRIC_KEYS = (
    "inside_similarity", "outside_similarity", "contrast", "peak_hit", "top1_recall",
    "top5_recall", "top10_recall", "oracle_size_iou", "center_error_norm", "target_rank_percentile",
)


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate tooth-structure pretraining checkpoints")
    parser.add_argument("--checkpoint", required=True, help="pretraining checkpoint saved by train.py")
    parser.add_argument("--config", default=None, help="fallback YAML for checkpoints without saved args")
    parser.add_argument("--data_root", default=None, help="overrides checkpoint data_root")
    parser.add_argument("--split_json", default=None, help="overrides checkpoint split_json")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--debug", action="store_true",
                        help="evaluate the first zeyu/*_process folder, matching train.py --debug")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output_dir", default=None, help="default: <checkpoint_dir>/eval/<checkpoint_stem>")
    parser.add_argument("--num_visual_full", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--similarity_mask_ratio", type=float, default=0.0)
    parser.add_argument("--rec_tooth_mask_ratio", type=float, default=None)
    parser.add_argument("--rec_sextant_mask_ratio", type=float, default=None)
    parser.add_argument("--backbone", choices=("swin", "vit"), default=None,
                        help="only needed to override a checkpoint/config value")
    return parser.parse_args()


def _default_model_args():
    return {
        "backbone": "swin", "full_canvas": None, "tooth_canvas": (224, 224), "sextant_canvas": None,
        "swin_embed_dim": 96, "swin_depths": (2, 2, 6, 2), "swin_num_heads": (3, 6, 12, 24),
        "swin_window_size": 7, "vit_embed_dim": 768, "vit_depth": 12, "vit_num_heads": 12,
        "vit_num_register_tokens": 0, "vit_img_size": 518, "vit_decoder_dim": 384,
        "vit_decoder_depth": 4, "vit_decoder_num_heads": 6, "head_hidden_dim": 512,
        "head_out_dim": 256, "tooth_mask_ratio": (0.25, 0.40), "sextant_mask_ratio": (0.35, 0.55),
        "align_loss_mode": "sigmoid", "test_ratio": 0.2, "split_seed": 0, "data_root": None,
        "split_json": None,
    }


def resolve_training_args(cli_args, checkpoint):
    values = _default_model_args()
    if cli_args.config:
        values.update(load_yaml_config(cli_args.config))
    saved = checkpoint.get("args", {})
    if isinstance(saved, argparse.Namespace):
        saved = vars(saved)
    if not isinstance(saved, dict):
        raise TypeError("checkpoint['args'] must be a dictionary or argparse.Namespace")
    values.update(saved)
    if cli_args.backbone:
        values["backbone"] = cli_args.backbone
    if cli_args.data_root:
        values["data_root"] = cli_args.data_root
    if cli_args.split_json is not None:
        values["split_json"] = cli_args.split_json
    if not values.get("data_root"):
        raise ValueError("data_root is absent from checkpoint; pass --data_root or --config")

    if values.get("full_canvas") is None:
        values["full_canvas"] = (896, 672) if values["backbone"] == "swin" else (518, 518)
    if values.get("sextant_canvas") is None:
        values["sextant_canvas"] = (448, 224) if values["backbone"] == "swin" else (392, 224)
    if values.get("split_seed") is None:
        values["split_seed"] = values.get("seed", 0)
    for key in ("full_canvas", "tooth_canvas", "sextant_canvas", "swin_depths", "swin_num_heads",
                "tooth_mask_ratio", "sextant_mask_ratio"):
        values[key] = tuple(values[key])
    values["swin_pretrained"] = None
    values["vit_pretrained"] = None
    values["no_pretrained"] = True
    return SimpleNamespace(**values)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requests CUDA but CUDA is unavailable")
    return device


def clamp_box(box, width, height):
    x1, y1, x2, y2 = (float(v) for v in box)
    return [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
            max(0.0, min(x2, width)), max(0.0, min(y2, height))]


def valid_box(box):
    return box[2] > box[0] and box[3] > box[1]


def make_crop(full_image, box, canvas_wh):
    crop = full_image.crop(tuple(round(v) for v in box))
    canvas, _, content_mask = letterbox_image(crop, *canvas_wh)
    return canvas, _to_tensor(canvas), content_mask


def load_annotation(json_path, model_args):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as handle:
        ann = json.load(handle)
    base_dir = json_path.parent.parent.parent
    full_image = Image.open(base_dir / ann["image_path"]).convert("RGB")
    image_w, image_h = full_image.size
    full_canvas, (scale, pad_x, pad_y), full_content_mask = letterbox_image(full_image, *model_args.full_canvas)

    def make_records(source_key, label_key, canvas_wh):
        records = []
        for raw in ann.get(source_key, []):
            box = clamp_box(raw["bbox_padded"], image_w, image_h)
            if not valid_box(box):
                continue
            crop_pil, crop_tensor, content_mask = make_crop(full_image, box, canvas_wh)
            label = int(raw[label_key]) if label_key == "fdi" else str(raw[label_key])
            records.append({
                "label": label,
                "box_original": box,
                "box_train": torch.tensor(letterbox_bbox(box, scale, pad_x, pad_y), dtype=torch.float32),
                "crop_pil": crop_pil,
                "crop_tensor": crop_tensor,
                "content_mask": content_mask,
            })
        return records

    return {
        "json_path": str(json_path), "case_id": str(ann.get("case_id", json_path.parent.name)),
        "view": str(ann.get("view", json_path.stem)), "full_pil": full_canvas,
        "full_tensor": _to_tensor(full_canvas), "full_content_mask": full_content_mask,
        "tooth": make_records("teeth", "fdi", model_args.tooth_canvas),
        "sextant": make_records("sextants", "id", model_args.sextant_canvas),
    }


def valid_patch_mask(content_mask, grid_h, grid_w, device):
    mask = F.interpolate(content_mask[None, None].to(device), size=(grid_h, grid_w), mode="area")
    return mask.reshape(-1) > 0.99


def box_iou(box_a, box_b):
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(box_a[2] - box_a[0], 0.0) * max(box_a[3] - box_a[1], 0.0)
    area_b = max(box_b[2] - box_b[0], 0.0) * max(box_b[3] - box_b[1], 0.0)
    return inter / max(area_a + area_b - inter, 1e-8)


def predict_box_from_map(scores, valid, gt_box, grid_h, grid_w, canvas_wh):
    canvas_w, canvas_h = canvas_wh
    patch_w, patch_h = canvas_w / grid_w, canvas_h / grid_h
    win_w = max(1, min(grid_w, round((gt_box[2] - gt_box[0]) / patch_w)))
    win_h = max(1, min(grid_h, round((gt_box[3] - gt_box[1]) / patch_h)))
    score_grid = scores.reshape(1, 1, grid_h, grid_w)
    valid_grid = valid.float().reshape(1, 1, grid_h, grid_w)
    kernel = torch.ones((1, 1, win_h, win_w), device=scores.device)
    score_sum = F.conv2d(score_grid * valid_grid, kernel)
    valid_count = F.conv2d(valid_grid, kernel)
    window_scores = score_sum / valid_count.clamp_min(1)
    window_scores = window_scores.masked_fill(valid_count == 0, -torch.inf)
    index = int(window_scores.reshape(-1).argmax().item())
    out_w = grid_w - win_w + 1
    row, col = divmod(index, out_w)
    return [col * patch_w, row * patch_h, min((col + win_w) * patch_w, canvas_w),
            min((row + win_h) * patch_h, canvas_h)]


def query_metrics(scores, overlap, valid, gt_box, grid_h, grid_w, canvas_wh):
    positive = overlap > 0
    outside = (overlap == 0) & valid
    inside_similarity = (scores * overlap).sum() / overlap.sum().clamp_min(1e-8)
    outside_similarity = scores[outside].mean() if outside.any() else scores.new_tensor(float("nan"))
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    ranked = valid_indices[scores[valid_indices].argsort(descending=True)]
    top_scores = {k: bool(positive[ranked[:min(k, ranked.numel())]].any().item()) for k in (1, 5, 10)}
    peak = int(ranked[0].item()) if ranked.numel() else 0
    best_positive = scores[positive].max() if positive.any() else scores.new_tensor(float("nan"))
    rank = 1 + int(((scores[valid] > best_positive).sum()).item())
    percentile = (rank - 1) / max(int(valid.sum().item()) - 1, 1)
    predicted_box = predict_box_from_map(scores, valid, gt_box, grid_h, grid_w, canvas_wh)
    gt_box = [float(v) for v in gt_box]
    gt_cx, gt_cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
    pred_cx, pred_cy = (predicted_box[0] + predicted_box[2]) / 2, (predicted_box[1] + predicted_box[3]) / 2
    canvas_w, canvas_h = canvas_wh
    center_error = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy) / math.hypot(canvas_w, canvas_h)
    return {
        "inside_similarity": float(inside_similarity.item()), "outside_similarity": float(outside_similarity.item()),
        "contrast": float((inside_similarity - outside_similarity).item()), "peak_hit": int(positive[peak].item()),
        "top1_recall": int(top_scores[1]), "top5_recall": int(top_scores[5]), "top10_recall": int(top_scores[10]),
        "oracle_size_iou": box_iou(predicted_box, gt_box), "center_error_norm": center_error,
        "target_rank_percentile": percentile, "pred_box_train": predicted_box, "gt_box_train": gt_box,
    }


def evaluate_annotation(sample, encoder, decoder, g_student, p_teacher, model_args, device, mask_ratio, keep_payload):
    full_images = sample["full_tensor"].unsqueeze(0).to(device)
    full_tokens, grid_h, grid_w = encoder(full_images)
    full_features = p_teacher(full_tokens)[0]
    valid = valid_patch_mask(sample["full_content_mask"], grid_h, grid_w, device)
    queries = []
    for crop_type in ("tooth", "sextant"):
        records = sample[crop_type]
        if not records:
            continue
        crops = torch.stack([record["crop_tensor"] for record in records]).to(device)
        global_query, _, _, _, _ = encoder(crops, mask_ratio)
        crop_features = g_student(global_query)
        sims = cosine_sim_matrix(crop_features, full_features)
        for index, (record, similarity) in enumerate(zip(records, sims)):
            overlap = compute_overlap(record["box_train"].unsqueeze(0).to(device), grid_h, grid_w,
                                      *model_args.full_canvas)[0]
            metrics = query_metrics(similarity, overlap, valid, record["box_train"].tolist(), grid_h, grid_w,
                                    model_args.full_canvas)
            row = {
                "case_id": sample["case_id"], "view": sample["view"], "json_path": sample["json_path"],
                "crop_type": crop_type, "label": record["label"], "query_index": index,
                **metrics,
            }
            if keep_payload:
                row.update({
                    "similarity": similarity.detach().cpu(),
                    "activated": (torch.sigmoid(similarity) if model_args.align_loss_mode == "sigmoid"
                                  else similarity.relu()).detach().cpu(),
                    "crop_pil": record["crop_pil"], "crop_tensor": record["crop_tensor"],
                    "content_mask": record["content_mask"], "grid_h": grid_h, "grid_w": grid_w,
                })
            queries.append(row)
    return queries


def csv_value(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    return value


def write_rows(rows, path):
    fields = ["case_id", "view", "json_path", "crop_type", "label", "query_index", *METRIC_KEYS,
              "gt_box_train", "pred_box_train"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def metric_summary(rows):
    summary = {"count": len(rows)}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_median"] = float(np.median(values)) if values else None
    return summary


def grouped_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: metric_summary(value) for key, value in sorted(groups.items())}


def write_group_summary(groups, path):
    fields = ["group", "count", *METRIC_KEYS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group, values in groups.items():
            writer.writerow({"group": group, **{key: values.get(key) for key in fields if key != "group"}})


def select_full_images(image_rows, budget, seed):
    if budget <= 0 or not image_rows:
        return []
    by_view = defaultdict(list)
    for row in image_rows:
        by_view[row["view"]].append(row)
    views = sorted(by_view)
    quotas = {view: budget // len(views) for view in views}
    for view in views[:budget % len(views)]:
        quotas[view] += 1
    rng = random.Random(seed)
    selected = []
    for view in views:
        candidates = sorted(by_view[view], key=lambda row: (row["mean_iou"], row["json_path"]))
        quota = min(quotas[view], len(candidates))
        anchors = [0, len(candidates) // 2, len(candidates) - 1]
        chosen = []
        for index in anchors:
            if len(chosen) >= quota:
                break
            candidate = candidates[index]
            if candidate not in chosen:
                candidate = dict(candidate)
                candidate["selection_reason"] = ("low" if index == 0 else "high" if index == len(candidates) - 1 else "median")
                chosen.append(candidate)
        remaining = [candidate for candidate in candidates if candidate["json_path"] not in {item["json_path"] for item in chosen}]
        rng.shuffle(remaining)
        for candidate in remaining[:quota - len(chosen)]:
            candidate = dict(candidate)
            candidate["selection_reason"] = "seeded_fill"
            chosen.append(candidate)
        selected.extend(chosen)
    return selected


def image_array(image):
    return np.asarray(image)


def draw_box(axis, box, color, label):
    axis.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, edgecolor=color, linewidth=2))
    axis.text(box[0], max(box[1] - 4, 8), label, color=color, fontsize=8,
              bbox={"facecolor": "black", "alpha": 0.55, "pad": 1, "edgecolor": "none"})


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def save_localization_figure(sample, row, output_dir):
    width, height = sample["full_pil"].size
    raw_map = row["similarity"].reshape(row["grid_h"], row["grid_w"]).numpy()
    activated_map = row["activated"].reshape(row["grid_h"], row["grid_w"]).numpy()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image_array(row["crop_pil"]))
    axes[0].set_title(f"{row['crop_type']} query: {row['label']}")
    for axis, heatmap, title in ((axes[1], raw_map, "raw cosine"), (axes[2], activated_map, "activated similarity")):
        axis.imshow(image_array(sample["full_pil"]))
        image = axis.imshow(heatmap, cmap="magma", alpha=0.52, extent=(0, width, height, 0), aspect="auto")
        draw_box(axis, row["gt_box_train"], "lime", "GT")
        draw_box(axis, row["pred_box_train"], "red", "prediction")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(
        f"{row['case_id']} {row['view']} | contrast={row['contrast']:.3f} | "
        f"IoU={row['oracle_size_iou']:.3f} | peak_hit={row['peak_hit']}", fontsize=11)
    figure.tight_layout()
    filename = f"{safe_name(row['case_id'])}_{safe_name(row['view'])}_{row['query_index']}_{safe_name(row['label'])}.png"
    path = output_dir / row["crop_type"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def normalized_to_rgb(tensor):
    tensor = tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return tensor.clamp(0, 1).permute(1, 2, 0).numpy()


def restore_pixel_prediction(prediction, crop, patch_size, grid_h, grid_w, norm_pix_loss):
    """Convert decoder patch tokens to the ImageNet-normalized pixel space of ``crop``.

    With ``norm_pix_loss``, the decoder is trained to predict each target patch after that
    patch has been standardized. Follow Swin-MAE's visualization procedure and invert those
    target-derived statistics before unpatchifying, otherwise the result is not displayable RGB.
    """
    if norm_pix_loss:
        target_patches = patchify(crop, patch_size)
        mean = target_patches.mean(dim=-1, keepdim=True)
        std = (target_patches.var(dim=-1, keepdim=True) + 1e-6).sqrt()
        prediction = prediction * std + mean
    return unpatchify(prediction, patch_size, grid_h, grid_w)


def sample_visualization_mask_ratio(mask_ratio, ratio_range, seed):
    """Use the training ratio sampler without perturbing the process-wide Python RNG."""
    if mask_ratio is not None:
        return [float(mask_ratio)]
    state = random.getstate()
    try:
        random.seed(seed)
        return sample_batch_mask_ratios(1, *ratio_range)
    finally:
        random.setstate(state)


def save_reconstruction_figure(row, encoder, decoder, patch_size, mask_ratio, ratio_range, norm_pix_loss, use_l1,
                               device, seed, output_dir):
    crop = row["crop_tensor"].unsqueeze(0).to(device)
    devices = [device.index] if device.type == "cuda" and device.index is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        applied_mask_ratio = sample_visualization_mask_ratio(mask_ratio, ratio_range, seed)
        _, patch_tokens, bool_mask, grid_h, grid_w = encoder(crop, applied_mask_ratio)
        prediction = decoder(patch_tokens, grid_h, grid_w)
    input_grid_h, input_grid_w = crop.shape[-2] // patch_size, crop.shape[-1] // patch_size
    content_mask = row["content_mask"].unsqueeze(0).to(device)
    training_rec_loss = reconstruction_loss(
        prediction, crop, bool_mask, content_mask, patch_size, norm_pix_loss, use_l1,
    )
    pred_image = restore_pixel_prediction(
        prediction, crop, patch_size, input_grid_h, input_grid_w, norm_pix_loss=norm_pix_loss,
    )[0]
    mask = bool_mask[0].reshape(input_grid_h, input_grid_w).repeat_interleave(patch_size, 0).repeat_interleave(patch_size, 1)
    effective_mask = mask.to(dtype=content_mask.dtype) * content_mask[0]
    target_rgb = normalized_to_rgb(crop[0])
    pred_rgb = normalized_to_rgb(pred_image)
    pixel_mask = effective_mask.detach().cpu().numpy()[..., None]
    masked_rgb = target_rgb * (1 - pixel_mask) + 0.5 * pixel_mask
    composite_rgb = target_rgb * (1 - pixel_mask) + pred_rgb * pixel_mask
    mse = (((pred_rgb - target_rgb) ** 2) * pixel_mask).sum() / max(float(pixel_mask.sum() * 3), 1.0)
    psnr = -10 * math.log10(max(float(mse), 1e-12))

    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, image, title in zip(axes, (target_rgb, masked_rgb, pred_rgb, composite_rgb),
                                  ("target", "masked input", "decoder prediction", "masked-patch composite")):
        axis.imshow(image)
        axis.set_title(title)
        axis.set_axis_off()
    rec_name = "L1" if use_l1 else "MSE"
    figure.suptitle(f"{row['case_id']} {row['view']} {row['crop_type']}={row['label']} | "
                     f"mask={applied_mask_ratio[0]:.3f} masked {rec_name}={training_rec_loss.item():.4f} "
                     f"PSNR={psnr:.2f}", fontsize=10)
    figure.tight_layout()
    filename = f"{safe_name(row['case_id'])}_{safe_name(row['view'])}_{row['query_index']}_{safe_name(row['label'])}.png"
    path = output_dir / row["crop_type"] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def selected_reconstruction_rows(rows):
    selected = []
    for crop_type in ("tooth", "sextant"):
        items = sorted((row for row in rows if row["crop_type"] == crop_type), key=lambda row: row["contrast"])
        for index in (0, len(items) // 2, len(items) - 1):
            if items and items[index] not in selected:
                selected.append(items[index])
    return selected


def load_models(checkpoint, model_args, device):
    encoder, decoder, embed_dim, patch_size = build_encoder_decoder(model_args)
    g_student, _ = make_student_teacher(embed_dim, model_args.head_hidden_dim, model_args.head_out_dim)
    _, p_teacher = make_student_teacher(embed_dim, model_args.head_hidden_dim, model_args.head_out_dim)
    for name, module in (("encoder", encoder), ("decoder", decoder), ("G_student", g_student), ("P_teacher", p_teacher)):
        if name not in checkpoint:
            raise KeyError(f"checkpoint has no {name!r} state")
        module.load_state_dict(checkpoint[name], strict=True)
        module.to(device).eval()
    return encoder, decoder, g_student, p_teacher, patch_size


def main():
    cli_args = get_args()
    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    checkpoint_path = Path(cli_args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = resolve_training_args(cli_args, checkpoint)
    device = resolve_device(cli_args.device)
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else checkpoint_path.parent / "eval" / checkpoint_path.stem
    similarity_dir = output_dir / "similarity"
    localization_dir = output_dir / "localization"
    reconstruction_dir = output_dir / "reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cli_args.debug:
        process_dir, json_paths = debug_json_paths(model_args.data_root)
        split_info = {"mode": "debug", "process_dir": str(process_dir), "annotation_json": len(json_paths)}
    else:
        split = create_or_load_split(model_args.data_root, model_args.split_json, model_args.test_ratio,
                                     model_args.split_seed, regenerate=False)
        json_paths = resolve_split_paths(model_args.data_root, split, cli_args.split)
        split_info = {"mode": cli_args.split, "split_counts": split["counts"], "annotation_json": len(json_paths)}
    if not json_paths:
        raise FileNotFoundError("selected evaluation split contains no annotation JSONs")

    encoder, decoder, g_student, p_teacher, patch_size = load_models(checkpoint, model_args, device)
    manifest = {
        "checkpoint": str(checkpoint_path.resolve()), "epoch": checkpoint.get("epoch"), "device": str(device),
        "seed": cli_args.seed, "similarity_mask_ratio": cli_args.similarity_mask_ratio, "split": split_info,
        "model_args": vars(model_args),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)

    rows, image_rows = [], []
    coverage = {"full_images": 0, "tooth": defaultdict(int), "sextant": defaultdict(int)}
    with torch.no_grad():
        for json_path in json_paths:
            sample = load_annotation(json_path, model_args)
            query_rows = evaluate_annotation(sample, encoder, decoder, g_student, p_teacher, model_args, device,
                                             cli_args.similarity_mask_ratio, keep_payload=False)
            rows.extend(query_rows)
            coverage["full_images"] += 1
            for row in query_rows:
                coverage[row["crop_type"]][str(row["label"])] += 1
            ious = [row["oracle_size_iou"] for row in query_rows]
            image_rows.append({"json_path": str(json_path), "case_id": sample["case_id"], "view": sample["view"],
                               "mean_iou": float(np.mean(ious)) if ious else float("nan"), "query_count": len(query_rows)})

        similarity_dir.mkdir(parents=True, exist_ok=True)
        write_rows(rows, similarity_dir / "per_query.csv")
        summary = {
            "overall": metric_summary(rows), "by_crop_type": grouped_summary(rows, "crop_type"),
            "by_view": grouped_summary(rows, "view"),
            "by_fdi": grouped_summary([row for row in rows if row["crop_type"] == "tooth"], "label"),
            "by_sextant": grouped_summary([row for row in rows if row["crop_type"] == "sextant"], "label"),
        }
        with (similarity_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        write_group_summary(summary["by_view"], similarity_dir / "summary_by_view.csv")
        write_group_summary(summary["by_fdi"], similarity_dir / "summary_by_fdi.csv")
        write_group_summary(summary["by_sextant"], similarity_dir / "summary_by_sextant.csv")
        with (output_dir / "coverage.json").open("w", encoding="utf-8") as handle:
            json.dump({"full_images": coverage["full_images"], "tooth": dict(coverage["tooth"]),
                       "sextant": dict(coverage["sextant"])}, handle, indent=2, ensure_ascii=False)

        selected = select_full_images([row for row in image_rows if math.isfinite(row["mean_iou"])],
                                      cli_args.num_visual_full, cli_args.seed)
        localization_dir.mkdir(parents=True, exist_ok=True)
        with (localization_dir / "sampled_full_images.json").open("w", encoding="utf-8") as handle:
            json.dump(selected, handle, indent=2, ensure_ascii=False)

        for sample_index, selected_item in enumerate(selected):
            sample = load_annotation(selected_item["json_path"], model_args)
            query_rows = evaluate_annotation(sample, encoder, decoder, g_student, p_teacher, model_args, device,
                                             cli_args.similarity_mask_ratio, keep_payload=True)
            for row in query_rows:
                save_localization_figure(sample, row, localization_dir)
            for reconstruction_index, row in enumerate(selected_reconstruction_rows(query_rows)):
                mask_ratio = (cli_args.rec_tooth_mask_ratio if row["crop_type"] == "tooth"
                              else cli_args.rec_sextant_mask_ratio)
                ratio_range = (model_args.tooth_mask_ratio if row["crop_type"] == "tooth"
                               else model_args.sextant_mask_ratio)
                save_reconstruction_figure(row, encoder, decoder, patch_size, mask_ratio, ratio_range,
                                           bool(getattr(model_args, "norm_pix_loss", False)),
                                           getattr(model_args, "rec_loss", "mse") == "l1", device,
                                           cli_args.seed + sample_index * 100 + reconstruction_index, reconstruction_dir)

    print(f"evaluated {len(rows)} crops from {len(json_paths)} full images")
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
