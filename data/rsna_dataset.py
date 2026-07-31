"""RSNA 2024 Lumbar Spine / LumbarDISC dataset loader.

We build one training sample per study: a single (representative) sagittal-T2 slice
with up to five disc-level bounding boxes derived from the point annotations in
`train_label_coordinates.csv`. Each box is centered on the Spinal Canal Stenosis
coordinate for that level, and its 3-class severity grade comes from `train.csv`.

Design choices (see instructions.md Step 2):
- Task: spinal canal stenosis on sagittal T2 only (the clean, comparable setting).
- 2.5D input: the annotated slice plus its two neighbors, stacked as pseudo-RGB.
- Point coordinates -> fixed-size boxes (box_size in ORIGINAL DICOM pixels), then
  rescaled together with the image to `image_size`.
- All RSNA tokens are disc levels -> level_type == 1 for every box.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import SpineAugmentation

# Canonical lumbar disc levels in anatomical (top -> bottom) order.
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVEL_TO_IDX = {lv: i for i, lv in enumerate(LEVELS)}
# train.csv column suffix form, e.g. "l1_l2".
LEVEL_TO_COL = {lv: lv.lower().replace("/", "_") for lv in LEVELS}

SEVERITY_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
STENOSIS_CONDITION = "Spinal Canal Stenosis"
IGNORE_INDEX = -1


# --------------------------------------------------------------------------------------
# DICOM loading
# --------------------------------------------------------------------------------------
def load_dicom_slice(path: str) -> np.ndarray:
    """Load a DICOM slice as a float32 array normalized to ~[0, 1].

    Applies rescale slope/intercept then window/level if present, else min-max.
    """
    import pydicom

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    def _first(v):
        import pydicom as _pd

        if isinstance(v, _pd.multival.MultiValue):
            return float(v[0])
        return float(v)

    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            center = _first(ds.WindowCenter)
            width = _first(ds.WindowWidth)
            lower, upper = center - width / 2.0, center + width / 2.0
            arr = np.clip(arr, lower, upper)
            arr = (arr - lower) / (upper - lower + 1e-8)
        except Exception:
            arr = _minmax(arr)
    else:
        arr = _minmax(arr)
    return arr


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / (hi - lo + 1e-8)


def coord_to_box(x, y, box_size, img_h, img_w) -> List[float]:
    """Convert a center point (x, y) to a clamped [x1, y1, x2, y2] box."""
    half = box_size / 2.0
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(img_w), x + half)
    y2 = min(float(img_h), y + half)
    return [x1, y1, x2, y2]


# --------------------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------------------
def build_rsna_index(
    data_dir: str,
    task: str = "stenosis",
    condition: str = STENOSIS_CONDITION,
    require_images: bool = True,
) -> List[Dict]:
    """Parse the RSNA CSVs into a lightweight list of per-study samples.

    Each returned dict contains only metadata (no pixels): study_id, series_id, the
    representative instance_number, and an ordered list of levels with their (x, y)
    coordinate and severity target. Studies with no usable sagittal-T2 canal
    annotations are skipped.

    If ``require_images`` is True (default), studies whose ``train_images/{study_id}/``
    folder is absent are also skipped. This makes the loader robust to *partial*
    downloads (e.g. a subset) even when the CSVs still list every study.
    """
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    desc_csv = pd.read_csv(os.path.join(data_dir, "train_series_descriptions.csv"))
    coord_csv = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    image_root = os.path.join(data_dir, "train_images")

    train_csv = train_csv.set_index("study_id")

    # study_id -> set of sagittal-T2 series_ids
    is_sagt2 = desc_csv["series_description"].str.contains("sagittal t2", case=False, na=False)
    sagt2 = desc_csv[is_sagt2]
    study_to_series: Dict[int, List[int]] = (
        sagt2.groupby("study_id")["series_id"].apply(list).to_dict()
    )

    coord_cond = coord_csv[coord_csv["condition"] == condition]

    samples: List[Dict] = []
    for study_id, series_ids in study_to_series.items():
        if require_images and not os.path.isdir(os.path.join(image_root, str(study_id))):
            continue  # images for this study were not downloaded -> skip
        study_coords = coord_cond[
            (coord_cond["study_id"] == study_id) & (coord_cond["series_id"].isin(series_ids))
        ]
        if len(study_coords) == 0:
            continue

        # Pick the sagittal-T2 series with the most canal annotations.
        best_series = study_coords["series_id"].value_counts().idxmax()
        series_coords = study_coords[study_coords["series_id"] == best_series]

        # Representative slice: the most frequently annotated instance in that series.
        instance_number = int(series_coords["instance_number"].value_counts().idxmax())

        # Ensure the exact slice the loader will read is on disk. With partial
        # (rate-limited) downloads a study folder can exist while its representative
        # slice is missing -> skip rather than crash at __getitem__. (2.5D neighbors
        # are optional and handled separately.)
        if require_images and not os.path.exists(
            os.path.join(image_root, str(study_id), str(best_series), f"{instance_number}.dcm")
        ):
            continue

        # Severity grades for this study (may be missing -> ignore index).
        grades = train_csv.loc[study_id] if study_id in train_csv.index else None

        levels = []
        for lv in LEVELS:  # keep anatomical ordering
            rows = series_coords[series_coords["level"] == lv]
            if len(rows) == 0:
                continue  # this level has no coordinate -> skip
            row = rows.iloc[0]
            level_idx = LEVEL_TO_IDX[lv]

            target = IGNORE_INDEX
            if grades is not None:
                col = f"spinal_canal_stenosis_{LEVEL_TO_COL[lv]}"
                if col in grades.index and pd.notna(grades[col]):
                    target = SEVERITY_MAP.get(str(grades[col]).strip(), IGNORE_INDEX)

            levels.append(
                {
                    "level_idx": level_idx,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "target": int(target),
                }
            )

        if len(levels) == 0:
            continue
        samples.append(
            {
                "study_id": int(study_id),
                "series_id": int(best_series),
                "instance_number": instance_number,
                "levels": levels,
            }
        )
    return samples


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class RSNADataset(Dataset):
    """RSNA sagittal-T2 spinal-canal-stenosis dataset (level-wise 3-class grading)."""

    def __init__(
        self,
        data_dir: str,
        samples: Optional[List[Dict]] = None,
        image_size: int = 224,
        box_size: int = 32,  # side length in RESIZED (image_size) pixels (see __getitem__)
        use_25d: bool = True,
        augment: bool = False,
        task: str = "stenosis",
        box_source: str = "oracle",       # "oracle" (annotation coords) | "detected" (learned)
        detected_centers: Optional[Dict] = None,  # {study_id: {level_idx: [x_orig, y_orig]}}
    ):
        self.data_dir = data_dir
        self.image_root = os.path.join(data_dir, "train_images")
        self.image_size = image_size
        self.box_size = box_size
        self.use_25d = use_25d
        self.task = task
        self.box_source = box_source
        self.detected_centers = detected_centers
        self.samples = samples if samples is not None else build_rsna_index(data_dir, task)
        self.aug = SpineAugmentation(image_size=image_size) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    # -- lightweight label access (no image loading) for class-weight computation --
    def get_all_targets(self) -> np.ndarray:
        out = []
        for s in self.samples:
            out.extend(lv["target"] for lv in s["levels"])
        return np.asarray(out, dtype=np.int64)

    def _dicom_path(self, study_id, series_id, instance) -> str:
        return os.path.join(self.image_root, str(study_id), str(series_id), f"{instance}.dcm")

    def _load_image(self, sample: Dict) -> np.ndarray:
        """Return a (3, H0, W0) float32 array at ORIGINAL resolution (2.5D or repeated)."""
        study_id, series_id = sample["study_id"], sample["series_id"]
        inst = sample["instance_number"]
        center = load_dicom_slice(self._dicom_path(study_id, series_id, inst))

        if self.use_25d:
            chans = []
            for off in (-1, 0, 1):
                p = self._dicom_path(study_id, series_id, inst + off)
                if off != 0 and os.path.exists(p):
                    sl = load_dicom_slice(p)
                    if sl.shape != center.shape:  # guard against odd series
                        sl = center
                else:
                    sl = center
                chans.append(sl)
            return np.stack(chans, axis=0)
        return np.stack([center, center, center], axis=0)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = self._load_image(sample)  # (3, H0, W0)
        _, h0, w0 = img.shape

        level_indices = np.array([lv["level_idx"] for lv in sample["levels"]], dtype=np.int64)
        level_types = np.ones(len(sample["levels"]), dtype=np.int64)  # all discs
        targets = np.array([lv["target"] for lv in sample["levels"]], dtype=np.int64)

        # Resize image to image_size, then place a FIXED-size box (box_size px in the
        # resized image) on each annotation center. Only the center comes from the (x, y)
        # coordinate; the extent is constant across studies so every ROI has the same
        # model receptive field — matching the strips baseline and removing the
        # variable-extent confound (raw pixel spacing varies ~3.6x across RSNA sites).
        from .transforms import _resize_chw

        sx = self.image_size / w0
        sy = self.image_size / h0
        img = _resize_chw(img, self.image_size, self.image_size)

        half = self.box_size / 2.0
        S = float(self.image_size)
        # box center source: annotation coords (oracle) or the detector's predictions.
        # Both are in ORIGINAL pixel coords, so the same sx/sy resize applies -> only the
        # center moves; the fixed 32-px extent is identical.
        det = self.detected_centers.get(str(sample["study_id"]), {}) \
            if (self.box_source == "detected" and self.detected_centers is not None) else {}
        boxes = np.zeros((len(sample["levels"]), 4), dtype=np.float32)
        for i, lv in enumerate(sample["levels"]):
            xy = det.get(str(lv["level_idx"]))
            x, y = (xy[0], xy[1]) if xy is not None else (lv["x"], lv["y"])
            cx, cy = x * sx, y * sy
            boxes[i] = [max(0.0, cx - half), max(0.0, cy - half),
                        min(S, cx + half), min(S, cy + half)]

        if self.aug is not None:
            img, boxes = self.aug(img, boxes)

        # Per-image z-score normalization.
        mean, std = float(img.mean()), float(img.std())
        img = (img - mean) / (std + 1e-6)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "boxes": torch.from_numpy(boxes).float(),
            "level_indices": torch.from_numpy(level_indices).long(),
            "level_types": torch.from_numpy(level_types).long(),
            "targets": torch.from_numpy(targets).long(),
            "num_levels": len(sample["levels"]),
            "study_id": sample["study_id"],
        }


# --------------------------------------------------------------------------------------
# Collate + splits
# --------------------------------------------------------------------------------------
def rsna_collate_fn(batch: List[Dict]) -> Dict:
    """Collate variable-level samples into a single batch dict.

    boxes -> (N_total, 5) with a leading batch-index column for ROI-Align.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 3, H, W)

    boxes_list, lvl_idx, lvl_type, targets, num_levels, study_ids = [], [], [], [], [], []
    for bi, b in enumerate(batch):
        k = b["num_levels"]
        bidx = torch.full((k, 1), float(bi))
        boxes_list.append(torch.cat([bidx, b["boxes"]], dim=1))  # (k, 5)
        lvl_idx.append(b["level_indices"])
        lvl_type.append(b["level_types"])
        targets.append(b["targets"])
        num_levels.append(k)
        study_ids.append(b["study_id"])

    return {
        "images": images,
        "boxes": torch.cat(boxes_list, dim=0),          # (N_total, 5)
        "level_indices": torch.cat(lvl_idx, dim=0),      # (N_total,)
        "level_types": torch.cat(lvl_type, dim=0),       # (N_total,)
        "targets": torch.cat(targets, dim=0),            # (N_total,)
        "num_levels": num_levels,                        # list[int], len B
        "study_ids": study_ids,                          # list[int], len B
    }


def make_rsna_splits(data_dir: str, config: Dict):
    """Build patient-level train/val/test RSNADataset splits (70/15/15 by default)."""
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_rsna_index(data_dir, config.get("task", "stenosis"))

    # Split by study_id (patient) so no study appears in two splits.
    study_ids = sorted({s["study_id"] for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(study_ids)
    n = len(study_ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(study_ids[:n_test])
    val_ids = set(study_ids[n_test : n_test + n_val])

    def subset(pred):
        return [s for s in samples if pred(s["study_id"])]

    train_s = subset(lambda i: i not in test_ids and i not in val_ids)
    val_s = subset(lambda i: i in val_ids)
    test_s = subset(lambda i: i in test_ids)

    common = dict(
        image_size=config.get("image_size", 224),
        box_size=config.get("box_size", 32),
        use_25d=config.get("use_25d", True),
        task=config.get("task", "stenosis"),
        box_source=config.get("box_source", "oracle"),
        detected_centers=config.get("detected_centers"),
    )
    train_ds = RSNADataset(data_dir, samples=train_s, augment=True, **common)
    val_ds = RSNADataset(data_dir, samples=val_s, augment=False, **common)
    test_ds = RSNADataset(data_dir, samples=test_s, augment=False, **common)
    return train_ds, val_ds, test_ds
