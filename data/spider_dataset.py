import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import SpineAugmentation, resize_channels

CANAL_LABEL = 100
DISC_LABEL_OFFSET = 200
VERTEBRA = 0
DISC = 1
IGNORE_INDEX = -1


def load_volume(path):
    import SimpleITK as sitk

    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image).astype(np.float32)


def mid_slice_index(volume):
    return volume.shape[0] // 2


def bbox_from_mask(mask2d, label, pad=2):
    ys, xs = np.where(mask2d == label)
    if xs.size == 0:
        return None
    return [float(xs.min() - pad), float(ys.min() - pad),
            float(xs.max() + pad), float(ys.max() + pad)]


def intensity_heuristic_regions(image2d, box_h=24, box_w_frac=0.5, sigma=7.0):
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    height, width = image2d.shape
    center_x = width // 2
    half_strip = max(1, width // 40)

    left = max(0, center_x - half_strip)
    right = min(width, center_x + half_strip + 1)
    strip = image2d[:, left:right].mean(axis=1)
    strip = gaussian_filter1d(strip, sigma=sigma)

    peaks, _ = find_peaks(strip, distance=int(sigma * 1.5))
    peaks = list(peaks)

    valleys = []
    for i in range(len(peaks) - 1):
        valleys.append(int((peaks[i] + peaks[i + 1]) // 2))

    box_w = box_w_frac * width
    x1 = max(0.0, center_x - box_w / 2)
    x2 = min(float(width), center_x + box_w / 2)

    def make_box(center_y):
        return [x1, max(0.0, center_y - box_h / 2), x2,
                min(float(height), center_y + box_h / 2)]

    peaks = sorted(peaks, reverse=True)
    valleys = sorted(valleys, reverse=True)

    boxes = []
    types = []
    for i in range(len(peaks)):
        boxes.append(make_box(peaks[i]))
        types.append(VERTEBRA)
        if i < len(valleys):
            boxes.append(make_box(valleys[i]))
            types.append(DISC)

    return boxes, types


def find_gradings_file(data_dir):
    for name in ["radiological_gradings.csv", "overview.csv", "gradings.csv"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return None


def find_column(columns, keyword):
    for column in columns:
        if keyword in column.lower().strip():
            return column
    return None


def load_gradings(data_dir):
    import pandas as pd

    path = find_gradings_file(data_dir)
    if path is None:
        return {}

    table = pd.read_csv(path)
    patient_col = find_column(table.columns, "patient")
    ivd_col = find_column(table.columns, "ivd")
    pfirrmann_col = find_column(table.columns, "pfirrmann")

    if not patient_col or not ivd_col or not pfirrmann_col:
        return {}

    gradings = {}
    for _, row in table.iterrows():
        try:
            patient_id = int(row[patient_col])
            ivd = int(row[ivd_col])
            grade = float(row[pfirrmann_col])
        except (ValueError, TypeError):
            continue
        if np.isnan(grade):
            continue
        gradings[(patient_id, ivd)] = int(round(grade)) - 1

    return gradings


def build_spider_index(data_dir, sequence="t2"):
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    if not os.path.isdir(image_dir):
        image_dir = data_dir
    if not os.path.isdir(mask_dir):
        mask_dir = data_dir

    files = []
    for pattern in ["*.mha", "*.nii.gz", "*.nii"]:
        files.extend(glob.glob(os.path.join(image_dir, pattern)))

    image_files = []
    for path in sorted(files):
        if "mask" not in os.path.basename(path).lower():
            image_files.append(path)

    sequence_files = []
    for path in image_files:
        name = os.path.basename(path).lower()
        if sequence in name and "t1" not in name.replace("t1rho", ""):
            sequence_files.append(path)
    if sequence_files:
        image_files = sequence_files

    samples = {}
    for path in image_files:
        base = os.path.basename(path)
        try:
            patient_id = int(base.split("_")[0].split(".")[0])
        except ValueError:
            continue
        if patient_id in samples:
            continue

        mask_path = os.path.join(mask_dir, base)
        if not os.path.exists(mask_path):
            matches = glob.glob(os.path.join(mask_dir, str(patient_id) + "_*mask*"))
            if matches:
                mask_path = matches[0]
            else:
                mask_path = None

        samples[patient_id] = {
            "patient_id": patient_id,
            "image_path": path,
            "mask_path": mask_path,
        }

    return list(samples.values())


class SPIDERDataset(Dataset):
    def __init__(self, data_dir, samples=None, image_size=224, use_25d=True,
                 use_oracle_regions=True, augment=False, task="pfirrmann"):
        self.data_dir = data_dir
        self.image_size = image_size
        self.use_25d = use_25d
        self.use_oracle_regions = use_oracle_regions
        self.task = task

        if samples is not None:
            self.samples = samples
        else:
            self.samples = build_spider_index(data_dir)

        self.gradings = load_gradings(data_dir)

        if augment:
            self.aug = SpineAugmentation(image_size=image_size)
        else:
            self.aug = None

    def __len__(self):
        return len(self.samples)

    def targets_for_patient(self, patient_id):
        targets = {}
        for key in self.gradings:
            if key[0] == patient_id:
                targets[key[1]] = self.gradings[key]
        return targets

    def get_all_targets(self):
        targets = []
        for sample in self.samples:
            patient_targets = self.targets_for_patient(sample["patient_id"])
            targets.extend(patient_targets.values())
        if not targets:
            targets = [IGNORE_INDEX]
        return np.asarray(targets, dtype=np.int64)

    def load_image_slice(self, path, mid):
        volume = load_volume(path)
        depth = volume.shape[0]

        if not self.use_25d:
            single = volume[mid]
            return np.stack([single, single, single], axis=0)

        indices = [max(0, mid - 1), mid, min(depth - 1, mid + 1)]
        slices = []
        for index in indices:
            slices.append(volume[index])
        return np.stack(slices, axis=0)

    def oracle_tokens(self, mask2d, patient_id):
        labels = np.unique(mask2d)

        vertebra_labels = []
        disc_labels = []
        for label in labels:
            if 0 < label < CANAL_LABEL:
                vertebra_labels.append(int(label))
            elif label > DISC_LABEL_OFFSET:
                disc_labels.append(int(label))
        vertebra_labels.sort()
        disc_labels.sort()

        patient_targets = self.targets_for_patient(patient_id)

        boxes = []
        types = []
        targets = []
        count = max(len(vertebra_labels), len(disc_labels))

        for i in range(count):
            if i < len(vertebra_labels):
                box = bbox_from_mask(mask2d, vertebra_labels[i])
                if box is not None:
                    boxes.append(box)
                    types.append(VERTEBRA)
                    targets.append(IGNORE_INDEX)

            if i < len(disc_labels):
                disc_label = disc_labels[i]
                box = bbox_from_mask(mask2d, disc_label)
                if box is not None:
                    boxes.append(box)
                    types.append(DISC)
                    ivd = disc_label - DISC_LABEL_OFFSET
                    targets.append(patient_targets.get(ivd, IGNORE_INDEX))

        return boxes, types, targets

    def heuristic_tokens(self, image2d, patient_id):
        boxes, types = intensity_heuristic_regions(image2d)
        patient_targets = self.targets_for_patient(patient_id)
        ivd_labels = sorted(patient_targets.keys())

        targets = []
        disc_count = 0
        for token_type in types:
            if token_type == DISC and disc_count < len(ivd_labels):
                ivd = ivd_labels[disc_count]
                targets.append(patient_targets.get(ivd, IGNORE_INDEX))
                disc_count = disc_count + 1
            elif token_type == DISC:
                targets.append(IGNORE_INDEX)
                disc_count = disc_count + 1
            else:
                targets.append(IGNORE_INDEX)

        return boxes, types, targets

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_id = sample["patient_id"]

        volume = load_volume(sample["image_path"])
        mid = mid_slice_index(volume)
        image = self.load_image_slice(sample["image_path"], mid)
        original_height = image.shape[1]
        original_width = image.shape[2]

        if self.use_oracle_regions and sample.get("mask_path"):
            mask_volume = load_volume(sample["mask_path"])
            mask2d = mask_volume[mid_slice_index(mask_volume)]
            boxes, types, targets = self.oracle_tokens(mask2d, patient_id)
        else:
            boxes, types, targets = self.heuristic_tokens(image[1], patient_id)

        if len(boxes) == 0:
            boxes = [[0.0, 0.0, float(original_width), float(original_height)]]
            types = [DISC]
            targets = [IGNORE_INDEX]

        boxes = np.asarray(boxes, dtype=np.float32)
        level_types = np.asarray(types, dtype=np.int64)
        level_indices = np.arange(len(types), dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)

        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height
        image = resize_channels(image, self.image_size, self.image_size)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

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
            "num_levels": len(types),
            "study_id": patient_id,
        }


def spider_collate_fn(batch):
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


def make_spider_splits(data_dir, config):
    seed = config.get("seed", 42)
    val_frac = config.get("val_frac", 0.15)
    test_frac = config.get("test_frac", 0.15)

    samples = build_spider_index(data_dir)

    patient_ids = set()
    for sample in samples:
        patient_ids.add(sample["patient_id"])
    patient_ids = sorted(patient_ids)

    random_state = np.random.RandomState(seed)
    random_state.shuffle(patient_ids)

    total = len(patient_ids)
    n_test = int(round(total * test_frac))
    n_val = int(round(total * val_frac))
    test_ids = set(patient_ids[:n_test])
    val_ids = set(patient_ids[n_test:n_test + n_val])

    train_samples = []
    val_samples = []
    test_samples = []
    for sample in samples:
        patient_id = sample["patient_id"]
        if patient_id in test_ids:
            test_samples.append(sample)
        elif patient_id in val_ids:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    settings = {
        "image_size": config.get("image_size", 224),
        "use_25d": config.get("use_25d", True),
        "use_oracle_regions": config.get("use_oracle_regions", True),
        "task": config.get("task", "pfirrmann"),
    }

    train_ds = SPIDERDataset(data_dir, samples=train_samples, augment=True, **settings)
    val_ds = SPIDERDataset(data_dir, samples=val_samples, augment=False, **settings)
    test_ds = SPIDERDataset(data_dir, samples=test_samples, augment=False, **settings)
    return train_ds, val_ds, test_ds
