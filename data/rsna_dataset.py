import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import SpineAugmentation, resize_channels

LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]

LEVEL_TO_IDX = {}
LEVEL_TO_COL = {}
for i in range(len(LEVELS)):
    LEVEL_TO_IDX[LEVELS[i]] = i
    LEVEL_TO_COL[LEVELS[i]] = LEVELS[i].lower().replace("/", "_")

SEVERITY_MAP = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
STENOSIS_CONDITION = "Spinal Canal Stenosis"
IGNORE_INDEX = -1


def minmax(pixels):
    low = float(pixels.min())
    high = float(pixels.max())
    return (pixels - low) / (high - low + 1e-8)


def first_value(value):
    import pydicom

    if isinstance(value, pydicom.multival.MultiValue):
        return float(value[0])
    return float(value)


def load_dicom_slice(path):
    import pydicom

    dicom = pydicom.dcmread(path)
    pixels = dicom.pixel_array.astype(np.float32)

    slope = float(getattr(dicom, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dicom, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    if hasattr(dicom, "WindowCenter") and hasattr(dicom, "WindowWidth"):
        try:
            center = first_value(dicom.WindowCenter)
            width = first_value(dicom.WindowWidth)
            lower = center - width / 2.0
            upper = center + width / 2.0
            pixels = np.clip(pixels, lower, upper)
            pixels = (pixels - lower) / (upper - lower + 1e-8)
        except Exception:
            pixels = minmax(pixels)
    else:
        pixels = minmax(pixels)

    return pixels


def coord_to_box(x, y, box_size, img_h, img_w):
    half = box_size / 2.0
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(img_w), x + half)
    y2 = min(float(img_h), y + half)
    return [x1, y1, x2, y2]


def build_rsna_index(data_dir, task="stenosis", condition=STENOSIS_CONDITION,
                     require_images=True):
    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    desc_csv = pd.read_csv(os.path.join(data_dir, "train_series_descriptions.csv"))
    coord_csv = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    image_root = os.path.join(data_dir, "train_images")

    train_csv = train_csv.set_index("study_id")

    is_sagittal_t2 = desc_csv["series_description"].str.contains("sagittal t2", case=False,
                                                                 na=False)
    sagittal = desc_csv[is_sagittal_t2]
    study_to_series = sagittal.groupby("study_id")["series_id"].apply(list).to_dict()

    condition_coords = coord_csv[coord_csv["condition"] == condition]

    samples = []
    for study_id in study_to_series:
        series_ids = study_to_series[study_id]

        if require_images and not os.path.isdir(os.path.join(image_root, str(study_id))):
            continue

        same_study = condition_coords["study_id"] == study_id
        same_series = condition_coords["series_id"].isin(series_ids)
        study_coords = condition_coords[same_study & same_series]
        if len(study_coords) == 0:
            continue

        best_series = study_coords["series_id"].value_counts().idxmax()
        series_coords = study_coords[study_coords["series_id"] == best_series]
        instance_number = int(series_coords["instance_number"].value_counts().idxmax())

        if require_images:
            slice_path = os.path.join(image_root, str(study_id), str(best_series),
                                      str(instance_number) + ".dcm")
            if not os.path.exists(slice_path):
                continue

        if study_id in train_csv.index:
            grades = train_csv.loc[study_id]
        else:
            grades = None

        levels = []
        for level_name in LEVELS:
            rows = series_coords[series_coords["level"] == level_name]
            if len(rows) == 0:
                continue
            row = rows.iloc[0]

            target = IGNORE_INDEX
            if grades is not None:
                column = "spinal_canal_stenosis_" + LEVEL_TO_COL[level_name]
                if column in grades.index and pd.notna(grades[column]):
                    target = SEVERITY_MAP.get(str(grades[column]).strip(), IGNORE_INDEX)

            levels.append({
                "level_idx": LEVEL_TO_IDX[level_name],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "target": int(target),
            })

        if len(levels) == 0:
            continue

        samples.append({
            "study_id": int(study_id),
            "series_id": int(best_series),
            "instance_number": instance_number,
            "levels": levels,
        })

    return samples


class RSNADataset(Dataset):
    def __init__(self, data_dir, samples=None, image_size=224, box_size=32, use_25d=True,
                 augment=False, task="stenosis", box_source="oracle",
                 detected_centers=None):
        self.data_dir = data_dir
        self.image_root = os.path.join(data_dir, "train_images")
        self.image_size = image_size
        self.box_size = box_size
        self.use_25d = use_25d
        self.task = task
        self.box_source = box_source
        self.detected_centers = detected_centers

        if samples is not None:
            self.samples = samples
        else:
            self.samples = build_rsna_index(data_dir, task)

        if augment:
            self.aug = SpineAugmentation(image_size=image_size)
        else:
            self.aug = None

    def __len__(self):
        return len(self.samples)

    def get_all_targets(self):
        targets = []
        for sample in self.samples:
            for level in sample["levels"]:
                targets.append(level["target"])
        return np.asarray(targets, dtype=np.int64)

    def dicom_path(self, study_id, series_id, instance):
        return os.path.join(self.image_root, str(study_id), str(series_id),
                            str(instance) + ".dcm")

    def load_image(self, sample):
        study_id = sample["study_id"]
        series_id = sample["series_id"]
        instance = sample["instance_number"]

        center = load_dicom_slice(self.dicom_path(study_id, series_id, instance))

        if not self.use_25d:
            return np.stack([center, center, center], axis=0)

        channels = []
        for offset in [-1, 0, 1]:
            path = self.dicom_path(study_id, series_id, instance + offset)
            if offset != 0 and os.path.exists(path):
                neighbour = load_dicom_slice(path)
                if neighbour.shape != center.shape:
                    neighbour = center
            else:
                neighbour = center
            channels.append(neighbour)

        return np.stack(channels, axis=0)

    def get_centers(self, sample):
        if self.box_source == "detected" and self.detected_centers is not None:
            return self.detected_centers.get(str(sample["study_id"]), {})
        return {}

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.load_image(sample)
        original_height = image.shape[1]
        original_width = image.shape[2]

        level_indices = []
        targets = []
        for level in sample["levels"]:
            level_indices.append(level["level_idx"])
            targets.append(level["target"])
        level_indices = np.array(level_indices, dtype=np.int64)
        targets = np.array(targets, dtype=np.int64)
        level_types = np.ones(len(sample["levels"]), dtype=np.int64)

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        image = resize_channels(image, self.image_size, self.image_size)

        half = self.box_size / 2.0
        size = float(self.image_size)
        detected = self.get_centers(sample)

        boxes = np.zeros((len(sample["levels"]), 4), dtype=np.float32)
        for i in range(len(sample["levels"])):
            level = sample["levels"][i]
            point = detected.get(str(level["level_idx"]))
            if point is not None:
                x = point[0]
                y = point[1]
            else:
                x = level["x"]
                y = level["y"]

            center_x = x * scale_x
            center_y = y * scale_y
            boxes[i] = [max(0.0, center_x - half), max(0.0, center_y - half),
                        min(size, center_x + half), min(size, center_y + half)]

        if self.aug is not None:
            image, boxes = self.aug(image, boxes)

        mean = float(image.mean())
        std = float(image.std())
        image = (image - mean) / (std + 1e-6)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "boxes": torch.from_numpy(boxes).float(),
            "level_indices": torch.from_numpy(level_indices).long(),
            "level_types": torch.from_numpy(level_types).long(),
            "targets": torch.from_numpy(targets).long(),
            "num_levels": len(sample["levels"]),
            "study_id": sample["study_id"],
        }


def rsna_collate_fn(batch):
    images = []
    for item in batch:
        images.append(item["image"])
    images = torch.stack(images, dim=0)

    all_boxes = []
    all_level_indices = []
    all_level_types = []
    all_targets = []
    num_levels = []
    study_ids = []

    for i in range(len(batch)):
        item = batch[i]
        count = item["num_levels"]

        batch_column = torch.full((count, 1), float(i))
        all_boxes.append(torch.cat([batch_column, item["boxes"]], dim=1))

        all_level_indices.append(item["level_indices"])
        all_level_types.append(item["level_types"])
        all_targets.append(item["targets"])
        num_levels.append(count)
        study_ids.append(item["study_id"])

    return {
        "images": images,
        "boxes": torch.cat(all_boxes, dim=0),
        "level_indices": torch.cat(all_level_indices, dim=0),
        "level_types": torch.cat(all_level_types, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "num_levels": num_levels,
        "study_ids": study_ids,
    }


def split_study_ids(samples, seed, val_frac, test_frac):
    study_ids = set()
    for sample in samples:
        study_ids.add(sample["study_id"])
    study_ids = sorted(study_ids)

    random_state = np.random.RandomState(seed)
    random_state.shuffle(study_ids)

    total = len(study_ids)
    n_test = int(round(total * test_frac))
    n_val = int(round(total * val_frac))

    test_ids = set(study_ids[:n_test])
    val_ids = set(study_ids[n_test:n_test + n_val])
    return val_ids, test_ids


def make_rsna_splits(data_dir, config):
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_rsna_index(data_dir, config.get("task", "stenosis"))
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
        "box_source": config.get("box_source", "oracle"),
        "detected_centers": config.get("detected_centers"),
    }

    train_ds = RSNADataset(data_dir, samples=train_samples, augment=True, **settings)
    val_ds = RSNADataset(data_dir, samples=val_samples, augment=False, **settings)
    test_ds = RSNADataset(data_dir, samples=test_samples, augment=False, **settings)
    return train_ds, val_ds, test_ds
