"""Detector dataset — wraps RSNADataset so the detector sees the EXACT same preprocessed
image the grader sees, plus the per-level centers (224-space), a valid mask, PixelSpacing
-> mm scale, and original dims (for exporting detected centers back to original coords).

Reusing RSNADataset guarantees the oracle-vs-detected comparison differs only in the box
center, never in preprocessing.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .rsna_dataset import RSNADataset, make_rsna_splits


class RSNADetectorDataset(Dataset):
    def __init__(self, base: RSNADataset, num_levels: int = 5):
        self.base = base
        self.num_levels = num_levels

    def __len__(self) -> int:
        return len(self.base)

    def _mm_scale_and_dims(self, sample: Dict):
        import pydicom

        p = self.base._dicom_path(sample["study_id"], sample["series_id"], sample["instance_number"])
        ds = pydicom.dcmread(p, stop_before_pixels=True)
        oh, ow = int(ds.Rows), int(ds.Columns)
        ps = getattr(ds, "PixelSpacing", None)
        row_sp, col_sp = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
        s = self.base.image_size
        # mm per 224-px, per axis (x=col, y=row)
        mm_scale = np.array([(ow / s) * col_sp, (oh / s) * row_sp], dtype=np.float32)
        return mm_scale, np.array([oh, ow], dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict:
        g = self.base[idx]  # verified preprocessing (2.5D, resize, z-score; aug if base.aug)
        boxes = g["boxes"].numpy()
        lvl = g["level_indices"].numpy()

        centers = np.zeros((self.num_levels, 2), dtype=np.float32)
        valid = np.zeros(self.num_levels, dtype=np.float32)
        for i, L in enumerate(lvl):
            if 0 <= L < self.num_levels:
                centers[L, 0] = (boxes[i, 0] + boxes[i, 2]) / 2.0
                centers[L, 1] = (boxes[i, 1] + boxes[i, 3]) / 2.0
                valid[L] = 1.0

        sample = self.base.samples[idx]
        mm_scale, orig_hw = self._mm_scale_and_dims(sample)
        return {
            "image": g["image"],
            "centers": torch.from_numpy(centers),      # (5, 2) in 224-space
            "valid": torch.from_numpy(valid),          # (5,)
            "mm_scale": torch.from_numpy(mm_scale),    # (2,) mm per 224-px
            "orig_hw": torch.from_numpy(orig_hw),      # (2,) original H, W
            "study_id": sample["study_id"],
        }


def detector_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "centers": torch.stack([b["centers"] for b in batch]),
        "valid": torch.stack([b["valid"] for b in batch]),
        "mm_scale": torch.stack([b["mm_scale"] for b in batch]),
        "orig_hw": torch.stack([b["orig_hw"] for b in batch]),
        "study_ids": [b["study_id"] for b in batch],
    }


def make_rsna_detector_splits(data_dir: str, config: Dict):
    """Detector train/val/test wrapping the SAME grader splits (same seed/partition)."""
    train_ds, val_ds, test_ds = make_rsna_splits(data_dir, config)
    nl = config.get("num_levels", 5)
    return (
        RSNADetectorDataset(train_ds, nl),
        RSNADetectorDataset(val_ds, nl),
        RSNADetectorDataset(test_ds, nl),
    )
