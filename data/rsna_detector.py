import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset

from .rsna_dataset import make_rsna_splits


class RSNADetectorDataset(Dataset):
    def __init__(self, base, num_levels=5):
        self.base = base
        self.num_levels = num_levels

    def __len__(self):
        return len(self.base)

    def read_mm_scale(self, sample):
        path = self.base.dicom_path(sample["study_id"], sample["series_id"],
                                    sample["instance_number"])
        dicom = pydicom.dcmread(path, stop_before_pixels=True)

        original_height = int(dicom.Rows)
        original_width = int(dicom.Columns)

        spacing = getattr(dicom, "PixelSpacing", None)
        if spacing is not None:
            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])
        else:
            row_spacing = 1.0
            col_spacing = 1.0

        size = self.base.image_size
        mm_scale = np.array([(original_width / size) * col_spacing,
                             (original_height / size) * row_spacing], dtype=np.float32)
        return mm_scale, np.array([original_height, original_width], dtype=np.int64)

    def __getitem__(self, idx):
        item = self.base[idx]
        boxes = item["boxes"].numpy()
        level_indices = item["level_indices"].numpy()

        centers = np.zeros((self.num_levels, 2), dtype=np.float32)
        valid = np.zeros(self.num_levels, dtype=np.float32)

        for i in range(len(level_indices)):
            level = level_indices[i]
            if 0 <= level < self.num_levels:
                centers[level, 0] = (boxes[i, 0] + boxes[i, 2]) / 2.0
                centers[level, 1] = (boxes[i, 1] + boxes[i, 3]) / 2.0
                valid[level] = 1.0

        sample = self.base.samples[idx]
        mm_scale, original_size = self.read_mm_scale(sample)

        return {
            "image": item["image"],
            "centers": torch.from_numpy(centers),
            "valid": torch.from_numpy(valid),
            "mm_scale": torch.from_numpy(mm_scale),
            "orig_hw": torch.from_numpy(original_size),
            "study_id": sample["study_id"],
        }


def detector_collate_fn(batch):
    images = []
    centers = []
    valid = []
    mm_scale = []
    original_size = []
    study_ids = []

    for item in batch:
        images.append(item["image"])
        centers.append(item["centers"])
        valid.append(item["valid"])
        mm_scale.append(item["mm_scale"])
        original_size.append(item["orig_hw"])
        study_ids.append(item["study_id"])

    return {
        "image": torch.stack(images),
        "centers": torch.stack(centers),
        "valid": torch.stack(valid),
        "mm_scale": torch.stack(mm_scale),
        "orig_hw": torch.stack(original_size),
        "study_ids": study_ids,
    }


def make_rsna_detector_splits(data_dir, config):
    train_ds, val_ds, test_ds = make_rsna_splits(data_dir, config)
    num_levels = config.get("num_levels", 5)

    return (
        RSNADetectorDataset(train_ds, num_levels),
        RSNADetectorDataset(val_ds, num_levels),
        RSNADetectorDataset(test_ds, num_levels),
    )
