from __future__ import annotations

import os
import glob
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import SpineAugmentation, _resize_chw

CANAL_LABEL = 100
DISC_LABEL_OFFSET = 200
VERTEBRA = 0
DISC = 1
IGNORE_INDEX = -1


def load_volume(path: str) -> np.ndarray:
    """Load a 3D volume as (D, H, W) float32 via SimpleITK."""
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.float32)


def _mid_sagittal(vol: np.ndarray, idx: Optional[int] = None) -> int:
    return vol.shape[0] // 2 if idx is None else idx


def bbox_from_mask(mask2d: np.ndarray, label: int, pad: int = 2) -> Optional[List[float]]:
    """Return [x1, y1, x2, y2] for a label in a 2D mask, with `pad` px padding."""
    ys, xs = np.where(mask2d == label)
    if xs.size == 0:
        return None
    x1 = float(xs.min() - pad)
    y1 = float(ys.min() - pad)
    x2 = float(xs.max() + pad)
    y2 = float(ys.max() + pad)
    return [x1, y1, x2, y2]


def intensity_heuristic_regions(
    image2d: np.ndarray, box_h: int = 24, box_w_frac: float = 0.5, sigma: float = 7.0
) -> Tuple[List[List[float]], List[int]]:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = image2d.shape
    cx = w // 2
    half = max(1, w // 40)
    strip = image2d[:, max(0, cx - half) : min(w, cx + half + 1)].mean(axis=1)
    strip = gaussian_filter1d(strip, sigma=sigma)

    peaks, _ = find_peaks(strip, distance=int(sigma * 1.5))
    peaks = list(peaks)
    valleys = [int((peaks[i] + peaks[i + 1]) // 2) for i in range(len(peaks) - 1)]

    box_w = box_w_frac * w
    x1, x2 = cx - box_w / 2, cx + box_w / 2

    def _box(yc):
        return [max(0.0, x1), max(0.0, yc - box_h / 2), min(float(w), x2), min(float(h), yc + box_h / 2)]

    peaks_bu = sorted(peaks, reverse=True)
    valleys_bu = sorted(valleys, reverse=True)

    boxes, types = [], []
    for i, py in enumerate(peaks_bu):
        boxes.append(_box(py))
        types.append(VERTEBRA)
        if i < len(valleys_bu):
            boxes.append(_box(valleys_bu[i]))
            types.append(DISC)
    return boxes, types


def _load_gradings(data_dir: str) -> Dict[Tuple[int, int], int]:
    """Map (patient_id, ivd_label) -> Pfirrmann class in [0, 4], if a CSV is present."""
    import pandas as pd

    candidates = ["radiological_gradings.csv", "overview.csv", "gradings.csv"]
    path = next((os.path.join(data_dir, c) for c in candidates if os.path.exists(os.path.join(data_dir, c))), None)
    if path is None:
        return {}

    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    pat_col = next((cols[k] for k in cols if "patient" in k), None)
    ivd_col = next((cols[k] for k in cols if "ivd" in k), None)
    pf_col = next((cols[k] for k in cols if "pfirrmann" in k), None)
    if not (pat_col and ivd_col and pf_col):
        return {}

    mapping: Dict[Tuple[int, int], int] = {}
    for _, row in df.iterrows():
        try:
            pid = int(row[pat_col])
            ivd = int(row[ivd_col])
            grade = float(row[pf_col])
        except (ValueError, TypeError):
            continue
        if np.isnan(grade):
            continue
        mapping[(pid, ivd)] = int(round(grade)) - 1
    return mapping


def build_spider_index(data_dir: str, sequence: str = "t2") -> List[Dict]:
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    if not os.path.isdir(img_dir):
        img_dir = data_dir
    if not os.path.isdir(mask_dir):
        mask_dir = os.path.join(data_dir, "masks") if os.path.isdir(os.path.join(data_dir, "masks")) else data_dir

    exts = ("*.mha", "*.nii.gz", "*.nii")
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    files = sorted(f for f in files if "mask" not in os.path.basename(f).lower())

    def _seq_ok(name: str) -> bool:
        n = name.lower()
        return sequence in n and "t1" not in n.replace("t1rho", "")

    seq_files = [f for f in files if _seq_ok(os.path.basename(f))]
    if seq_files:
        files = seq_files

    samples: List[Dict] = []
    seen_patient: Dict[int, str] = {}
    for f in files:
        base = os.path.basename(f)
        try:
            pid = int(base.split("_")[0].split(".")[0])
        except ValueError:
            continue
        if pid in seen_patient and "space" in base.lower():
            continue
        mask_path = os.path.join(mask_dir, base)
        if not os.path.exists(mask_path):
            alt = glob.glob(os.path.join(mask_dir, f"{pid}_*mask*")) or glob.glob(os.path.join(mask_dir, base))
            mask_path = alt[0] if alt else None
        samples.append({"patient_id": pid, "image_path": f, "mask_path": mask_path})
        seen_patient[pid] = f

    dedup: Dict[int, Dict] = {}
    for s in samples:
        dedup.setdefault(s["patient_id"], s)
    return list(dedup.values())


class SPIDERDataset(Dataset):
    """SPIDER mid-sagittal dataset with interleaved vertebra/disc tokens."""

    def __init__(
        self,
        data_dir: str,
        samples: Optional[List[Dict]] = None,
        image_size: int = 224,
        use_25d: bool = True,
        use_oracle_regions: bool = True,
        augment: bool = False,
        task: str = "pfirrmann",
    ):
        self.data_dir = data_dir
        self.image_size = image_size
        self.use_25d = use_25d
        self.use_oracle_regions = use_oracle_regions
        self.task = task
        self.samples = samples if samples is not None else build_spider_index(data_dir)
        self.gradings = _load_gradings(data_dir)
        self.aug = SpineAugmentation(image_size=image_size) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def get_all_targets(self) -> np.ndarray:
        out = []
        for s in self.samples:
            out.extend(self._targets_for_patient(s["patient_id"]).values())
        return np.asarray(out if out else [IGNORE_INDEX], dtype=np.int64)

    def _targets_for_patient(self, pid: int) -> Dict[int, int]:
        """ivd_label -> pfirrmann class for this patient (lightweight, from CSV)."""
        return {ivd: g for (p, ivd), g in self.gradings.items() if p == pid}

    def _load_image_slice(self, path: str, mid: int) -> np.ndarray:
        """Return a (3, H0, W0) float32 slice stack (2.5D or repeated)."""
        vol = load_volume(path)
        d = vol.shape[0]
        if self.use_25d:
            idxs = [max(0, mid - 1), mid, min(d - 1, mid + 1)]
            return np.stack([vol[i] for i in idxs], axis=0)
        s = vol[mid]
        return np.stack([s, s, s], axis=0)

    def _oracle_tokens(self, mask2d: np.ndarray, pid: int):
        """Build interleaved (boxes, level_types, level_indices, targets) from the mask."""
        labels = np.unique(mask2d)
        vert_labels = sorted(int(l) for l in labels if 0 < l < CANAL_LABEL)
        disc_labels = sorted(int(l) for l in labels if l > DISC_LABEL_OFFSET)
        targets_map = self._targets_for_patient(pid)

        boxes, types, tgts = [], [], []
        n = max(len(vert_labels), len(disc_labels))
        for i in range(n):
            if i < len(vert_labels):
                b = bbox_from_mask(mask2d, vert_labels[i])
                if b is not None:
                    boxes.append(b)
                    types.append(VERTEBRA)
                    tgts.append(IGNORE_INDEX)
            if i < len(disc_labels):
                dl = disc_labels[i]
                b = bbox_from_mask(mask2d, dl)
                if b is not None:
                    boxes.append(b)
                    types.append(DISC)
                    ivd = dl - DISC_LABEL_OFFSET
                    tgts.append(targets_map.get(ivd, IGNORE_INDEX))
        return boxes, types, tgts

    def _heuristic_tokens(self, image2d: np.ndarray, pid: int):
        boxes, types = intensity_heuristic_regions(image2d)
        targets_map = self._targets_for_patient(pid)
        ivd_sorted = sorted(targets_map.keys())
        tgts, disc_i = [], 0
        for t in types:
            if t == DISC:
                ivd = ivd_sorted[disc_i] if disc_i < len(ivd_sorted) else None
                tgts.append(targets_map.get(ivd, IGNORE_INDEX) if ivd is not None else IGNORE_INDEX)
                disc_i += 1
            else:
                tgts.append(IGNORE_INDEX)
        return boxes, types, tgts

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        pid = sample["patient_id"]

        vol = load_volume(sample["image_path"])
        mid = _mid_sagittal(vol)
        img = self._load_image_slice(sample["image_path"], mid)
        _, h0, w0 = img.shape

        if self.use_oracle_regions and sample.get("mask_path"):
            mask_vol = load_volume(sample["mask_path"])
            mask2d = mask_vol[_mid_sagittal(mask_vol)]
            boxes, types, tgts = self._oracle_tokens(mask2d, pid)
        else:
            boxes, types, tgts = self._heuristic_tokens(img[1], pid)

        if len(boxes) == 0:
            boxes = [[0.0, 0.0, float(w0), float(h0)]]
            types = [DISC]
            tgts = [IGNORE_INDEX]

        boxes = np.asarray(boxes, dtype=np.float32)
        level_types = np.asarray(types, dtype=np.int64)
        level_indices = np.arange(len(types), dtype=np.int64)
        targets = np.asarray(tgts, dtype=np.int64)

        sx = self.image_size / w0
        sy = self.image_size / h0
        img = _resize_chw(img, self.image_size, self.image_size)
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy

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
            "num_levels": len(types),
            "study_id": pid,
        }


def spider_collate_fn(batch: List[Dict]) -> Dict:
    """Identical batch structure to RSNA's collate (boxes carry a batch-index column)."""
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


def make_spider_splits(data_dir: str, config: Dict):
    """Patient-level train/val/test SPIDERDataset splits."""
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_spider_index(data_dir)
    patient_ids = sorted({s["patient_id"] for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(patient_ids[:n_test])
    val_ids = set(patient_ids[n_test : n_test + n_val])

    def subset(pred):
        return [s for s in samples if pred(s["patient_id"])]

    common = dict(
        image_size=config.get("image_size", 224),
        use_25d=config.get("use_25d", True),
        use_oracle_regions=config.get("use_oracle_regions", True),
        task=config.get("task", "pfirrmann"),
    )
    train_ds = SPIDERDataset(
        data_dir, samples=subset(lambda i: i not in test_ids and i not in val_ids), augment=True, **common
    )
    val_ds = SPIDERDataset(data_dir, samples=subset(lambda i: i in val_ids), augment=False, **common)
    test_ds = SPIDERDataset(data_dir, samples=subset(lambda i: i in test_ids), augment=False, **common)
    return train_ds, val_ds, test_ds
