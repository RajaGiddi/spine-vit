from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import SpineAugmentation

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
LEVEL_TO_IDX = {lv: i for i, lv in enumerate(LEVELS)}
LEVEL_TO_COL = {lv: lv.lower().replace("/", "_") for lv in LEVELS}

SEVERITY_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
STENOSIS_CONDITION = "Spinal Canal Stenosis"
IGNORE_INDEX = -1


def load_dicom_slice(path: str) -> np.ndarray:
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


def build_rsna_index(
    data_dir: str,
    task: str = "stenosis",
    condition: str = STENOSIS_CONDITION,
    require_images: bool = True,
) -> List[Dict]:
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    desc_csv = pd.read_csv(os.path.join(data_dir, "train_series_descriptions.csv"))
    coord_csv = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    image_root = os.path.join(data_dir, "train_images")

    train_csv = train_csv.set_index("study_id")

    is_sagt2 = desc_csv["series_description"].str.contains("sagittal t2", case=False, na=False)
    sagt2 = desc_csv[is_sagt2]
    study_to_series: Dict[int, List[int]] = (
        sagt2.groupby("study_id")["series_id"].apply(list).to_dict()
    )

    coord_cond = coord_csv[coord_csv["condition"] == condition]

    samples: List[Dict] = []
    for study_id, series_ids in study_to_series.items():
        if require_images and not os.path.isdir(os.path.join(image_root, str(study_id))):
            continue
        study_coords = coord_cond[
            (coord_cond["study_id"] == study_id) & (coord_cond["series_id"].isin(series_ids))
        ]
        if len(study_coords) == 0:
            continue

        best_series = study_coords["series_id"].value_counts().idxmax()
        series_coords = study_coords[study_coords["series_id"] == best_series]

        instance_number = int(series_coords["instance_number"].value_counts().idxmax())

        if require_images and not os.path.exists(
            os.path.join(image_root, str(study_id), str(best_series), f"{instance_number}.dcm")
        ):
            continue

        grades = train_csv.loc[study_id] if study_id in train_csv.index else None

        levels = []
        for lv in LEVELS:
            rows = series_coords[series_coords["level"] == lv]
            if len(rows) == 0:
                continue
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


class RSNADataset(Dataset):
    """RSNA sagittal-T2 spinal-canal-stenosis dataset (level-wise 3-class grading)."""

    def __init__(
        self,
        data_dir: str,
        samples: Optional[List[Dict]] = None,
        image_size: int = 224,
        box_size: int = 32,
        use_25d: bool = True,
        augment: bool = False,
        task: str = "stenosis",
        box_source: str = "oracle",
        detected_centers: Optional[Dict] = None,
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
                    if sl.shape != center.shape:
                        sl = center
                else:
                    sl = center
                chans.append(sl)
            return np.stack(chans, axis=0)
        return np.stack([center, center, center], axis=0)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = self._load_image(sample)
        _, h0, w0 = img.shape

        level_indices = np.array([lv["level_idx"] for lv in sample["levels"]], dtype=np.int64)
        level_types = np.ones(len(sample["levels"]), dtype=np.int64)
        targets = np.array([lv["target"] for lv in sample["levels"]], dtype=np.int64)

        from .transforms import _resize_chw

        sx = self.image_size / w0
        sy = self.image_size / h0
        img = _resize_chw(img, self.image_size, self.image_size)

        half = self.box_size / 2.0
        S = float(self.image_size)
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


def rsna_collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([b["image"] for b in batch], dim=0)

    boxes_list, lvl_idx, lvl_type, targets, num_levels, study_ids = [], [], [], [], [], []
    for bi, b in enumerate(batch):
        k = b["num_levels"]
        bidx = torch.full((k, 1), float(bi))
        boxes_list.append(torch.cat([bidx, b["boxes"]], dim=1))
        lvl_idx.append(b["level_indices"])
        lvl_type.append(b["level_types"])
        targets.append(b["targets"])
        num_levels.append(k)
        study_ids.append(b["study_id"])

    return {
        "images": images,
        "boxes": torch.cat(boxes_list, dim=0),
        "level_indices": torch.cat(lvl_idx, dim=0),
        "level_types": torch.cat(lvl_type, dim=0),
        "targets": torch.cat(targets, dim=0),
        "num_levels": num_levels,
        "study_ids": study_ids,
    }


def make_rsna_splits(data_dir: str, config: Dict):
    """Build patient-level train/val/test RSNADataset splits (70/15/15 by default)."""
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_rsna_index(data_dir, config.get("task", "stenosis"))

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
