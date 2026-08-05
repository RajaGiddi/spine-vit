from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch

from .rsna_dataset import RSNADataset, rsna_collate_fn, load_dicom_slice
from .transforms import _resize_chw


class RSNAFusionDataset(RSNADataset):
    """RSNADataset (sagittal) + a per-level axial ROI where the axial annotation exists."""

    def __init__(self, *args, axial_index: Optional[Dict] = None, axial_box_size: int = 32,
                 axial_use_25d: bool = True, sag_slices: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.axial_index = axial_index or {}
        self.axial_box_size = axial_box_size
        self.axial_use_25d = axial_use_25d
        self.sag_slices = int(sag_slices)
        if self.aug is not None:
            from .transforms import SpineAugmentation
            self.aug = SpineAugmentation(image_size=self.image_size, p_hflip=0.0)
            self.axial_aug = SpineAugmentation(image_size=self.image_size, p_hflip=0.5)
        else:
            self.axial_aug = None

    def _axial_dicom_path(self, sid, series, instance) -> str:
        return os.path.join(self.image_root, str(sid), str(series), f"{instance}.dcm")

    def _load_axial_25d(self, sid, series, instance) -> np.ndarray:
        """(3, H0, W0) float32 at original resolution: 2.5D axial neighbors or repeat."""
        center = load_dicom_slice(self._axial_dicom_path(sid, series, instance))
        if not self.axial_use_25d:
            return np.stack([center, center, center], axis=0)
        chans = []
        for off in (-1, 0, 1):
            p = self._axial_dicom_path(sid, series, instance + off)
            if off != 0 and os.path.exists(p):
                sl = load_dicom_slice(p)
                sl = sl if sl.shape == center.shape else center
            else:
                sl = center
            chans.append(sl)
        return np.stack(chans, axis=0)

    def _load_sag_2p5d(self, sample, center_inst: int) -> np.ndarray:
        """(3, H0, W0) 2.5D sagittal stack at a parasagittal instance; missing slices fall back."""
        sid, series = sample["study_id"], sample["series_id"]
        cp = self._dicom_path(sid, series, center_inst)
        center = (load_dicom_slice(cp) if os.path.exists(cp)
                  else load_dicom_slice(self._dicom_path(sid, series, sample["instance_number"])))
        if not self.use_25d:
            return np.stack([center, center, center], axis=0)
        chans = []
        for off in (-1, 0, 1):
            p = self._dicom_path(sid, series, center_inst + off)
            if off != 0 and os.path.exists(p):
                sl = load_dicom_slice(p)
                sl = sl if sl.shape == center.shape else center
            else:
                sl = center
            chans.append(sl)
        return np.stack(chans, axis=0)

    def _build_sag_multi(self, sample, item):
        K = self.sag_slices
        inst = sample["instance_number"]
        offs = [j - K // 2 for j in range(K)]
        from .transforms import _resize_chw
        _, h0, w0 = self._load_sag_2p5d(sample, inst).shape
        sx, sy = self.image_size / w0, self.image_size / h0
        half, S = self.box_size / 2.0, float(self.image_size)
        boxes = np.zeros((len(sample["levels"]), 4), dtype=np.float32)
        for i, lv in enumerate(sample["levels"]):
            cx, cy = lv["x"] * sx, lv["y"] * sy
            boxes[i] = [max(0.0, cx - half), max(0.0, cy - half),
                        min(S, cx + half), min(S, cy + half)]
        slices = [_resize_chw(self._load_sag_2p5d(sample, inst + o), self.image_size, self.image_size)
                  for o in offs]
        stack = np.concatenate(slices, axis=0)
        if self.aug is not None:
            stack, boxes = self.aug(stack, boxes)
        multi = stack.reshape(K, 3, self.image_size, self.image_size)
        multi = (multi - multi.mean(axis=(1, 2, 3), keepdims=True)) \
            / (multi.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        item["sag_multi_images"] = torch.from_numpy(np.ascontiguousarray(multi)).float()
        item["boxes"] = torch.from_numpy(np.ascontiguousarray(boxes)).float()
        return item

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        sid = sample["study_id"]
        if self.sag_slices > 1:
            item = self._build_sag_multi(sample, item)
        ax_levels = self.axial_index.get(sid, {})

        S = float(self.image_size)
        half = self.axial_box_size / 2.0
        ax_imgs: List[np.ndarray] = []
        ax_boxes: List[List[float]] = []
        ax_lvls: List[int] = []
        slot: List[int] = []

        for lv in sample["levels"]:
            li = lv["level_idx"]
            info = ax_levels.get(li)
            if info is None:
                slot.append(-1)
                continue
            img = self._load_axial_25d(sid, info["series"], info["instance"])
            _, h0, w0 = img.shape
            sx, sy = self.image_size / w0, self.image_size / h0
            img = _resize_chw(img, self.image_size, self.image_size)
            cx, cy = info["cx"] * sx, info["cy"] * sy
            box = np.array([[max(0.0, cx - half), max(0.0, cy - half),
                             min(S, cx + half), min(S, cy + half)]], dtype=np.float32)
            if self.axial_aug is not None:
                img, box = self.axial_aug(img, box)
            mean, std = float(img.mean()), float(img.std())
            img = (img - mean) / (std + 1e-6)
            slot.append(len(ax_imgs))
            ax_imgs.append(np.ascontiguousarray(img))
            ax_boxes.append([float(v) for v in box[0]])
            ax_lvls.append(li)

        n = len(ax_imgs)
        item["axial_images"] = (torch.from_numpy(np.stack(ax_imgs)).float()
                                if n else torch.zeros(0, 3, self.image_size, self.image_size))
        item["axial_boxes"] = (torch.tensor(ax_boxes, dtype=torch.float32)
                               if n else torch.zeros(0, 4))
        item["axial_level_indices"] = torch.tensor(ax_lvls, dtype=torch.long)
        item["axial_slot"] = torch.tensor(slot, dtype=torch.long)
        item["axial_num"] = n
        return item


def rsna_fusion_collate_fn(batch: List[Dict]) -> Dict:
    """Sagittal collate + a batched axial-image stack with an aligned `axial_slot`."""
    base = rsna_collate_fn(batch)
    H, W = base["images"].shape[-2:]

    ax_imgs, ax_boxes, ax_lvls, ax_num, slot_global = [], [], [], [], []
    ax_offset = 0
    for b in batch:
        n = int(b["axial_num"])
        ax_num.append(n)
        if n > 0:
            ax_imgs.append(b["axial_images"])
            idxcol = torch.arange(n, dtype=torch.float32).unsqueeze(1) + ax_offset
            ax_boxes.append(torch.cat([idxcol, b["axial_boxes"]], dim=1))
            ax_lvls.append(b["axial_level_indices"])
        s = b["axial_slot"]
        slot_global.append(torch.where(s >= 0, s + ax_offset, s))
        ax_offset += n

    base["axial_images"] = torch.cat(ax_imgs, 0) if ax_imgs else torch.zeros(0, 3, H, W)
    base["axial_boxes"] = torch.cat(ax_boxes, 0) if ax_boxes else torch.zeros(0, 5)
    base["axial_level_indices"] = torch.cat(ax_lvls, 0) if ax_lvls else torch.zeros(0, dtype=torch.long)
    base["axial_slot"] = torch.cat(slot_global, 0)
    base["axial_num"] = ax_num

    if "sag_multi_images" in batch[0]:
        base["sag_multi_images"] = torch.stack([b["sag_multi_images"] for b in batch], 0)
        boxes_list = []
        for bi, b in enumerate(batch):
            k = b["num_levels"]
            bidx = torch.full((k, 1), float(bi))
            boxes_list.append(torch.cat([bidx, b["boxes"]], dim=1))
        base["boxes"] = torch.cat(boxes_list, 0)
    return base


def make_rsna_fusion_splits(data_dir: str, config: Dict):
    """Train/val/test RSNAFusionDataset splits reusing the v1 sagittal split logic."""
    from .rsna_dataset import build_rsna_index
    from .rsna_axial import build_axial_index

    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_rsna_index(data_dir, config.get("task", "stenosis"))
    axial_index = build_axial_index(data_dir, posterior_offset=config.get("axial_posterior_offset", 0.0))

    study_ids = sorted({s["study_id"] for s in samples})
    rng = np.random.RandomState(seed)
    rng.shuffle(study_ids)
    n = len(study_ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(study_ids[:n_test])
    val_ids = set(study_ids[n_test:n_test + n_val])

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
        axial_index=axial_index,
        axial_box_size=config.get("axial_box_size", 32),
        axial_use_25d=config.get("axial_use_25d", True),
        sag_slices=config.get("sag_slices", 1),
    )
    augment = bool(config.get("augment", False))
    train_ds = RSNAFusionDataset(data_dir, samples=train_s, augment=augment, **common)
    val_ds = RSNAFusionDataset(data_dir, samples=val_s, augment=False, **common)
    test_ds = RSNAFusionDataset(data_dir, samples=test_s, augment=False, **common)
    return train_ds, val_ds, test_ds
