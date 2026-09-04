import os

import numpy as np
import torch

from .rsna_dataset import RSNADataset, rsna_collate_fn, load_dicom_slice, split_study_ids
from .rsna_dataset import build_rsna_index
from .rsna_axial import build_axial_index
from .transforms import SpineAugmentation, resize_channels


class RSNAFusionDataset(RSNADataset):
    def __init__(self, *args, axial_index=None, axial_box_size=32, axial_use_25d=True,
                 sag_slices=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.axial_index = axial_index or {}
        self.axial_box_size = axial_box_size
        self.axial_use_25d = axial_use_25d
        self.sag_slices = int(sag_slices)

        if self.aug is not None:
            self.aug = SpineAugmentation(image_size=self.image_size, p_hflip=0.0)
            self.axial_aug = SpineAugmentation(image_size=self.image_size, p_hflip=0.5)
        else:
            self.axial_aug = None

    def load_25d_stack(self, study_id, series_id, instance, use_25d):
        center_path = os.path.join(self.image_root, str(study_id), str(series_id),
                                   str(instance) + ".dcm")
        center = load_dicom_slice(center_path)

        if not use_25d:
            return np.stack([center, center, center], axis=0)

        channels = []
        for offset in [-1, 0, 1]:
            path = os.path.join(self.image_root, str(study_id), str(series_id),
                                str(instance + offset) + ".dcm")
            if offset != 0 and os.path.exists(path):
                neighbour = load_dicom_slice(path)
                if neighbour.shape != center.shape:
                    neighbour = center
            else:
                neighbour = center
            channels.append(neighbour)

        return np.stack(channels, axis=0)

    def load_axial_25d(self, study_id, series_id, instance):
        return self.load_25d_stack(study_id, series_id, instance, self.axial_use_25d)

    def load_sag_25d(self, sample, center_instance):
        study_id = sample["study_id"]
        series_id = sample["series_id"]

        path = os.path.join(self.image_root, str(study_id), str(series_id),
                            str(center_instance) + ".dcm")
        if not os.path.exists(path):
            center_instance = sample["instance_number"]

        return self.load_25d_stack(study_id, series_id, center_instance, self.use_25d)

    def build_sag_multi(self, sample, item):
        num_slices = self.sag_slices
        instance = sample["instance_number"]

        offsets = []
        for j in range(num_slices):
            offsets.append(j - num_slices // 2)

        first = self.load_sag_25d(sample, instance)
        original_height = first.shape[1]
        original_width = first.shape[2]
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        half = self.box_size / 2.0
        size = float(self.image_size)
        boxes = np.zeros((len(sample["levels"]), 4), dtype=np.float32)
        for i in range(len(sample["levels"])):
            level = sample["levels"][i]
            center_x = level["x"] * scale_x
            center_y = level["y"] * scale_y
            boxes[i] = [max(0.0, center_x - half), max(0.0, center_y - half),
                        min(size, center_x + half), min(size, center_y + half)]

        slices = []
        for offset in offsets:
            stack = self.load_sag_25d(sample, instance + offset)
            slices.append(resize_channels(stack, self.image_size, self.image_size))
        stacked = np.concatenate(slices, axis=0)

        if self.aug is not None:
            stacked, boxes = self.aug(stacked, boxes)

        multi = stacked.reshape(num_slices, 3, self.image_size, self.image_size)
        mean = multi.mean(axis=(1, 2, 3), keepdims=True)
        std = multi.std(axis=(1, 2, 3), keepdims=True)
        multi = (multi - mean) / (std + 1e-6)

        item["sag_multi_images"] = torch.from_numpy(np.ascontiguousarray(multi)).float()
        item["boxes"] = torch.from_numpy(np.ascontiguousarray(boxes)).float()
        return item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        study_id = sample["study_id"]

        if self.sag_slices > 1:
            item = self.build_sag_multi(sample, item)

        axial_levels = self.axial_index.get(study_id, {})
        size = float(self.image_size)
        half = self.axial_box_size / 2.0

        axial_images = []
        axial_boxes = []
        axial_level_indices = []
        slots = []

        for level in sample["levels"]:
            level_index = level["level_idx"]
            info = axial_levels.get(level_index)

            if info is None:
                slots.append(-1)
                continue

            image = self.load_axial_25d(study_id, info["series"], info["instance"])
            original_height = image.shape[1]
            original_width = image.shape[2]
            scale_x = self.image_size / original_width
            scale_y = self.image_size / original_height
            image = resize_channels(image, self.image_size, self.image_size)

            center_x = info["cx"] * scale_x
            center_y = info["cy"] * scale_y
            box = np.array([[max(0.0, center_x - half), max(0.0, center_y - half),
                             min(size, center_x + half), min(size, center_y + half)]],
                           dtype=np.float32)

            if self.axial_aug is not None:
                image, box = self.axial_aug(image, box)

            mean = float(image.mean())
            std = float(image.std())
            image = (image - mean) / (std + 1e-6)

            slots.append(len(axial_images))
            axial_images.append(np.ascontiguousarray(image))
            axial_boxes.append([float(v) for v in box[0]])
            axial_level_indices.append(level_index)

        count = len(axial_images)
        if count > 0:
            item["axial_images"] = torch.from_numpy(np.stack(axial_images)).float()
            item["axial_boxes"] = torch.tensor(axial_boxes, dtype=torch.float32)
        else:
            item["axial_images"] = torch.zeros(0, 3, self.image_size, self.image_size)
            item["axial_boxes"] = torch.zeros(0, 4)

        item["axial_level_indices"] = torch.tensor(axial_level_indices, dtype=torch.long)
        item["axial_slot"] = torch.tensor(slots, dtype=torch.long)
        item["axial_num"] = count
        return item


def rsna_fusion_collate_fn(batch):
    base = rsna_collate_fn(batch)
    height = base["images"].shape[-2]
    width = base["images"].shape[-1]

    axial_images = []
    axial_boxes = []
    axial_level_indices = []
    axial_num = []
    slots = []
    offset = 0

    for item in batch:
        count = int(item["axial_num"])
        axial_num.append(count)

        if count > 0:
            axial_images.append(item["axial_images"])
            index_column = torch.arange(count, dtype=torch.float32).unsqueeze(1) + offset
            axial_boxes.append(torch.cat([index_column, item["axial_boxes"]], dim=1))
            axial_level_indices.append(item["axial_level_indices"])

        slot = item["axial_slot"]
        slots.append(torch.where(slot >= 0, slot + offset, slot))
        offset = offset + count

    if axial_images:
        base["axial_images"] = torch.cat(axial_images, 0)
    else:
        base["axial_images"] = torch.zeros(0, 3, height, width)

    if axial_boxes:
        base["axial_boxes"] = torch.cat(axial_boxes, 0)
    else:
        base["axial_boxes"] = torch.zeros(0, 5)

    if axial_level_indices:
        base["axial_level_indices"] = torch.cat(axial_level_indices, 0)
    else:
        base["axial_level_indices"] = torch.zeros(0, dtype=torch.long)

    base["axial_slot"] = torch.cat(slots, 0)
    base["axial_num"] = axial_num

    if "sag_multi_images" in batch[0]:
        multi = []
        for item in batch:
            multi.append(item["sag_multi_images"])
        base["sag_multi_images"] = torch.stack(multi, 0)

        all_boxes = []
        for i in range(len(batch)):
            count = batch[i]["num_levels"]
            batch_column = torch.full((count, 1), float(i))
            all_boxes.append(torch.cat([batch_column, batch[i]["boxes"]], dim=1))
        base["boxes"] = torch.cat(all_boxes, 0)

    return base


def make_rsna_fusion_splits(data_dir, config):
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_rsna_index(data_dir, config.get("task", "stenosis"))
    axial_index = build_axial_index(
        data_dir, posterior_offset=config.get("axial_posterior_offset", 0.0))

    val_ids, test_ids = split_study_ids(samples, seed, val_frac, test_frac)

    train_samples = []
    val_samples = []
    test_samples = []
    for sample in samples:
        study_id = sample["study_id"]
        if study_id in test_ids:
            test_samples.append(sample)
        elif study_id in val_ids:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    settings = {
        "image_size": config.get("image_size", 224),
        "box_size": config.get("box_size", 32),
        "use_25d": config.get("use_25d", True),
        "task": config.get("task", "stenosis"),
        "axial_index": axial_index,
        "axial_box_size": config.get("axial_box_size", 32),
        "axial_use_25d": config.get("axial_use_25d", True),
        "sag_slices": config.get("sag_slices", 1),
    }
    augment = bool(config.get("augment", False))

    train_ds = RSNAFusionDataset(data_dir, samples=train_samples, augment=augment, **settings)
    val_ds = RSNAFusionDataset(data_dir, samples=val_samples, augment=False, **settings)
    test_ds = RSNAFusionDataset(data_dir, samples=test_samples, augment=False, **settings)
    return train_ds, val_ds, test_ds
