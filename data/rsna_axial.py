import os

import numpy as np
import pandas as pd
import pydicom

from .rsna_dataset import LEVELS, LEVEL_TO_IDX, load_dicom_slice

LEFT = "Left Subarticular Stenosis"
RIGHT = "Right Subarticular Stenosis"


def axial_dicom_path(data_dir, study_id, series, instance):
    return os.path.join(data_dir, "train_images", str(study_id), str(series),
                        str(instance) + ".dcm")


def build_axial_index(data_dir, posterior_offset=0.0):
    coords = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    subarticular = coords[coords.condition.isin([LEFT, RIGHT])]

    axial = {}
    for study_id, study_rows in subarticular.groupby("study_id"):
        per_level = {}

        for level_name in LEVELS:
            level_rows = study_rows[study_rows.level == level_name]
            if len(level_rows) == 0:
                continue

            points = []
            for condition in [LEFT, RIGHT]:
                matching = level_rows[level_rows.condition == condition]
                if len(matching):
                    row = matching.iloc[0]
                    points.append({
                        "x": float(row.x),
                        "y": float(row.y),
                        "instance": int(row.instance_number),
                        "series": int(row.series_id),
                    })

            if not points:
                continue

            xs = []
            ys = []
            instances = []
            series_ids = []
            for point in points:
                xs.append(point["x"])
                ys.append(point["y"])
                instances.append(point["instance"])
                series_ids.append(point["series"])

            per_level[LEVEL_TO_IDX[level_name]] = {
                "series": points[0]["series"],
                "instance": points[0]["instance"],
                "cx": float(np.mean(xs)),
                "cy": float(np.mean(ys)) + posterior_offset,
                "sided": len(points),
                "inst_lr": instances,
                "series_lr": series_ids,
            }

        if per_level:
            axial[int(study_id)] = per_level

    return axial


def axial_coverage(axial, all_study_ids):
    has_any = 0
    has_all_five = 0
    for study_id in all_study_ids:
        levels = axial.get(study_id, {})
        if levels:
            has_any = has_any + 1
        if len(levels) == 5:
            has_all_five = has_all_five + 1

    per_level = {}
    for level in range(5):
        count = 0
        for study_id in all_study_ids:
            if level in axial.get(study_id, {}):
                count = count + 1
        per_level[LEVELS[level]] = count

    total = len(all_study_ids)
    return {
        "n_studies": total,
        "has_any_axial": has_any,
        "pct_any": 100 * has_any / max(1, total),
        "has_all5_axial": has_all_five,
        "pct_all5": 100 * has_all_five / max(1, total),
        "per_level_count": per_level,
    }


def is_sorted(values):
    increasing = True
    decreasing = True
    for i in range(len(values) - 1):
        if values[i] > values[i + 1]:
            increasing = False
        if values[i] < values[i + 1]:
            decreasing = False
    return increasing or decreasing


def axial_monotonicity_flags(axial):
    flagged = []

    for study_id in axial:
        levels = axial[study_id]

        by_series = {}
        for level in levels:
            series = levels[level]["series"]
            if series not in by_series:
                by_series[series] = []
            by_series[series].append((level, levels[level]["instance"]))

        for series in by_series:
            items = by_series[series]
            if len(items) < 2:
                continue
            items.sort()

            instances = []
            for level, instance in items:
                instances.append(instance)

            if not is_sorted(instances):
                flagged.append(study_id)
                break

    return flagged


def load_axial_slice(data_dir, study_id, series, instance):
    return load_dicom_slice(axial_dicom_path(data_dir, study_id, series, instance))


def axial_box_mm(data_dir, study_id, series, instance, box_px, image_size=224):
    path = axial_dicom_path(data_dir, study_id, series, instance)
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

    mm_x = box_px * (original_width / image_size) * col_spacing
    mm_y = box_px * (original_height / image_size) * row_spacing
    return mm_x, mm_y
