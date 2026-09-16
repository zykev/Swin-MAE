"""Dental SSL record builder for Swin-MAE dental training.

Mirror of ``baselines/mae/util/dental_records.py`` (kept self-contained so each
vendored baseline repo has no import-path dependency on the parent workspace).

Reads the public case-level split JSON produced by the workspace root
``datasets/split.py`` (create_or_load_split) and expands each annotation JSON
into crop records:

  full    -> one record per view annotation (the full intraoral image)
  tooth   -> one record per tooth box        (ann["teeth"][].bbox_padded)
  sextant -> one record per sextant box      (ann["sextants"][].bbox_padded)
  group   -> dynamic four-tooth groups       (four_tooth_group_candidates +
             _union_and_pad_boxes on member teeth bbox_padded)

Every record carries a stable ``sample_key``::

    "{case_id}:{view}:{crop_type}:{label}"

so downstream feature evaluation (the ``compare/`` package) can align samples
across models.
"""
import hashlib
import json
import os
from pathlib import Path

PAD_RATIO = 0.1  # must match datasets/dataset.py::_union_and_pad_boxes default

CROP_TYPES = ("full", "tooth", "sextant", "group")

# --- geometry helpers (vendored from datasets/dataset.py) -----------------


def clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [max(0, min(x1, img_w)), max(0, min(y1, img_h)),
            max(0, min(x2, img_w)), max(0, min(y2, img_h))]


def valid_box(box):
    return box[2] > box[0] and box[3] > box[1]


def union_and_pad_boxes(boxes, img_w, img_h, pad_ratio=PAD_RATIO):
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return clamp_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], img_w, img_h)


# --- FDI group helpers (vendored from datasets/fdi_relation.py) -----------

_SEXTANT_TABLE = {
    1: {(1, 3): "S2", (4, 8): "S1"},
    2: {(1, 3): "S2", (4, 8): "S3"},
    3: {(1, 3): "S5", (4, 8): "S4"},
    4: {(1, 3): "S5", (4, 8): "S6"},
}


def parse_fdi(fdi):
    """11 -> (quadrant=1, tooth_index=1)."""
    return fdi // 10, fdi % 10


def sextant_of(fdi):
    q, t = parse_fdi(fdi)
    for (lo, hi), sextant in _SEXTANT_TABLE[q].items():
        if lo <= t <= hi:
            return sextant
    raise ValueError(f"invalid FDI code: {fdi}")


def four_tooth_group_candidates(view, available_fdis):
    """Return strict four-tooth groups whose members are all present."""
    view = str(view).upper()
    if view == "F":
        candidates = [
            (11, 21, 41, 31),
            (11, 12, 41, 42),
            (12, 13, 42, 43),
            (21, 22, 31, 32),
            (22, 23, 32, 33),
        ]
    elif view == "R":
        candidates = [
            (14, 15, 44, 45),
            (15, 16, 45, 46),
            (16, 17, 46, 47),
            (17, 18, 47, 48),
        ]
    elif view == "L":
        candidates = [
            (24, 25, 34, 35),
            (25, 26, 35, 36),
            (26, 27, 36, 37),
            (27, 28, 37, 38),
        ]
    elif view in {"U", "D"}:
        arch = (
            [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
            if view == "U"
            else [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        )
        candidates = [tuple(arch[i:i + 4]) for i in range(len(arch) - 3)]
    else:
        raise ValueError(f"unsupported intraoral view: {view}")

    available = {int(fdi) for fdi in available_fdis}
    return [group for group in candidates if all(fdi in available for fdi in group)]


# --- data root compatibility (intraoral / intraoral1) ---------------------
#
# 不同服务器上数据根目录可能叫 ``.datasets/intraoral``，也可能叫
# ``.datasets/intraoral1``（挂载点不同、符号链接失效等）。规则是【优先用调用方
# 给的路径，只有它不存在时才切换到另一个名字】，因此同一份命令/配置在两种机器
# 上都能跑。两个候选都不存在时返回原路径，由调用方抛出原有的上下文报错。

_COMPATIBLE_DIRECTORY_NAMES = {
    "intraoral": "intraoral1",
    "intraoral1": "intraoral",
}


def compatible_path_candidate(path):
    """Return the counterpart path obtained by swapping one supported directory name."""
    path = Path(path).expanduser()
    parts = list(path.parts)
    for index, part in enumerate(parts):
        replacement = _COMPATIBLE_DIRECTORY_NAMES.get(part)
        if replacement is not None:
            parts[index] = replacement
            return Path(*parts)
    return None


def resolve_compatible_path(path):
    """Use *path* when it exists, otherwise try its ``intraoral``/``intraoral1`` counterpart."""
    path = Path(path).expanduser()
    if path.exists():
        return path
    candidate = compatible_path_candidate(path)
    if candidate is not None and candidate.exists():
        return candidate
    return path


# --- split / annotation loading -------------------------------------------

def load_split(split_json):
    split_path = resolve_compatible_path(split_json)
    if not split_path.exists():
        raise FileNotFoundError(f"split_json does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_annotation_paths(split, split_name):
    """Annotation JSON paths for a partition (train/test).

    Returns absolute paths when the split JSON carries an absolute ``data_root``;
    otherwise returns the raw relative paths so the caller can resolve them
    against the data_root it was invoked with (avoids double-prefixing when
    the split JSON stores a relative ``data_root``).
    """
    data_root = split.get("data_root") or ""
    rel_paths = split["annotations"][split_name]
    if data_root and Path(data_root).is_absolute():
        return [str(Path(data_root) / rel_path) for rel_path in rel_paths]
    return [str(Path(rel_path)) for rel_path in rel_paths]


def load_annotation(json_path):
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def full_image_path(json_path, ann):
    """Full-resolution original image referenced by an annotation JSON."""
    json_path = Path(json_path)
    return json_path.parent.parent.parent / ann["image_path"]


def make_sample_key(case_id, view, crop_type, label):
    return f"{case_id}:{view}:{crop_type}:{label}"


def _build_tooth_records(ann, json_path, case_id, view, partition):
    records = []
    full_path = full_image_path(json_path, ann)
    for raw in ann.get("teeth", []):
        box = clamp_box(raw["bbox_padded"], ann["orig_size"][0], ann["orig_size"][1])
        if not valid_box(box):
            continue
        fdi = int(raw["fdi"])
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "tooth",
            "label": fdi,
            "sample_key": make_sample_key(case_id, view, "tooth", fdi),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in box],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def _build_sextant_records(ann, json_path, case_id, view, partition):
    records = []
    full_path = full_image_path(json_path, ann)
    for raw in ann.get("sextants", []):
        box = clamp_box(raw["bbox_padded"], ann["orig_size"][0], ann["orig_size"][1])
        if not valid_box(box):
            continue
        sextant_id = str(raw["id"])
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "sextant",
            "label": sextant_id,
            "sample_key": make_sample_key(case_id, view, "sextant", sextant_id),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in box],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def _build_group_records(ann, json_path, case_id, view, partition):
    img_w, img_h = ann["orig_size"]
    tooth_boxes = {}
    for raw in ann.get("teeth", []):
        box = clamp_box(raw["bbox_padded"], img_w, img_h)
        if valid_box(box):
            tooth_boxes[int(raw["fdi"])] = box
    full_path = full_image_path(json_path, ann)
    records = []
    for member_fdi in four_tooth_group_candidates(view, tooth_boxes):
        label = "-".join(str(fdi) for fdi in member_fdi)
        records.append({
            "case_id": case_id,
            "view": view,
            "crop_type": "group",
            "label": label,
            "sample_key": make_sample_key(case_id, view, "group", label),
            "source_image_path": str(full_path),
            "box_original": [float(v) for v in union_and_pad_boxes(
                [tooth_boxes[fdi] for fdi in member_fdi], img_w, img_h)],
            "annotation_path": str(json_path),
            "partition": partition,
        })
    return records


def build_records(data_root, split_json, split, categories):
    """Expand split annotations into crop records for SSL pretraining.

    Args:
        data_root: dataset root; used to resolve the split's relative paths
            when the split JSON lacks an absolute ``data_root`` field.
        split_json: path to the case-level split JSON (create_or_load_split).
        split: partition name ("train" or "test").
        categories: iterable of crop types, subset of ("full","tooth","sextant","group").
    """
    split_data = load_split(split_json)
    # data_root 优先用调用方给的路径，失效时回退到 intraoral/intraoral1 的另一个
    data_root = str(resolve_compatible_path(data_root).resolve())
    split_data = dict(split_data)
    if not split_data.get("data_root"):
        split_data["data_root"] = data_root

    categories = set(categories)
    unknown = categories - set(CROP_TYPES)
    if unknown:
        raise ValueError(f"unsupported categories: {sorted(unknown)} (expected subset of {list(CROP_TYPES)})")

    records = []
    for json_path in split_annotation_paths(split_data, split):
        json_path = Path(json_path)
        if not json_path.is_absolute():
            json_path = Path(data_root) / json_path
        # split JSON 里的 data_root 也可能是已失效的 intraoral，逐条记录再回退一次
        json_path = resolve_compatible_path(json_path)
        ann = load_annotation(json_path)
        case_id = str(ann.get("case_id") or json_path.parent.name)
        view = str(ann.get("view") or json_path.stem)
        if "full" in categories:
            records.append({
                "case_id": case_id,
                "view": view,
                "crop_type": "full",
                "label": view,
                "sample_key": make_sample_key(case_id, view, "full", view),
                "source_image_path": str(full_image_path(json_path, ann)),
                "box_original": None,
                "annotation_path": str(json_path),
                "partition": split,
            })
        if "tooth" in categories:
            records.extend(_build_tooth_records(ann, json_path, case_id, view, split))
        if "sextant" in categories:
            records.extend(_build_sextant_records(ann, json_path, case_id, view, split))
        if "group" in categories:
            records.extend(_build_group_records(ann, json_path, case_id, view, split))
    return records


def records_hash(records):
    """Stable content hash over the sorted sample_keys (for checkpoint sidecars)."""
    keys = sorted(r["sample_key"] for r in records)
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
    return digest.hexdigest()[:16]


def write_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path)


def audit_records(records, partition=None):
    """Counts for training/validation records: case, view, crop-type, label stats."""
    from collections import Counter, defaultdict

    stats = {
        "partition": partition,
        "total_records": len(records),
        "cases": len({r["case_id"] for r in records}),
        "views": dict(Counter(r["view"] for r in records)),
        "crop_types": dict(Counter(r["crop_type"] for r in records)),
        "labels_per_crop_type": {},
    }
    by_crop = defaultdict(Counter)
    for r in records:
        by_crop[r["crop_type"]][r["label"]] += 1
    for crop_type, labels in by_crop.items():
        stats["labels_per_crop_type"][crop_type] = {
            "unique_labels": len(labels),
            "total": sum(labels.values()),
        }
    return stats


def print_audit(records, partition=None):
    stats = audit_records(records, partition)
    header = f"records[{stats['partition'] or '?'}]"
    print(f"{header}: total={stats['total_records']} cases={stats['cases']}")
    print(f"  views: {stats['views']}")
    print(f"  crop_types: {stats['crop_types']}")
    for crop_type, info in stats["labels_per_crop_type"].items():
        print(f"  {crop_type}: unique_labels={info['unique_labels']} total={info['total']}")
    return stats
