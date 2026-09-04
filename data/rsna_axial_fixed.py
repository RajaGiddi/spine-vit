"""Rule-based axial slice selection, as a control for expert slice choice.

The existing axial configuration takes its slice from the radiologist's
subarticular annotation, so the axial condition confounds the imaging plane
with expert slice selection. This picks the slice by geometry instead: the
axial slice whose plane passes closest to the disc centre, where the disc
centre comes from the sagittal canal-stenosis annotation converted to patient
coordinates.

Everything else is left alone. The series, the in-plane centre (cx, cy), the
box size and the tokenizer are all unchanged, so the only thing that differs
between this and the annotated configuration is which slice is read.

Angling matters here. Most axial lumbar stacks in this dataset are acquired as
per-level angled blocks - median 4 distinct orientations in a single series,
up to 6 - so there is no single series affine to project onto. Each slice is
tested against its own plane, via composition.geometry, which keeps per-slice
orientation throughout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from composition.geometry import GeometryError, SeriesGeometry  # noqa: E402

from .rsna_axial import build_axial_index  # noqa: E402
from .rsna_dataset import LEVELS, LEVEL_TO_IDX  # noqa: E402

CANAL = "Spinal Canal Stenosis"
SAG_T2 = "Sagittal T2/STIR"


def _series_dir(data_dir, study_id, series_id):
    return os.path.join(data_dir, "train_images", str(int(study_id)), str(int(series_id)))


def sagittal_disc_centres(data_dir, coords=None, series=None):
    """{study_id: {level_index: patient-space disc centre}} from the sagittal marks.

    The canal-stenosis point sits at the disc on the sagittal series; converted
    to patient coordinates it gives a level position the axial stack can be
    searched against.
    """
    if coords is None:
        coords = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    if series is None:
        series = pd.read_csv(os.path.join(data_dir, "train_series_descriptions.csv"))

    sag_series = set(series[series.series_description == SAG_T2].series_id)
    rows = coords[(coords.condition == CANAL) & (coords.series_id.isin(sag_series))]

    out = {}
    geometry_cache = {}
    for (study_id, series_id), grp in rows.groupby(["study_id", "series_id"]):
        key = (int(study_id), int(series_id))
        if key not in geometry_cache:
            try:
                geometry_cache[key] = SeriesGeometry.from_dir(
                    _series_dir(data_dir, study_id, series_id))
            except (GeometryError, OSError):
                geometry_cache[key] = None
        geom = geometry_cache[key]
        if geom is None:
            continue
        by_instance = {s.instance_number: s for s in geom}

        for row in grp.itertuples():
            slice_geom = by_instance.get(int(row.instance_number))
            if slice_geom is None or row.level not in LEVEL_TO_IDX:
                continue
            # CSV x is the column, y is the row.
            point = slice_geom.voxel_to_patient(float(row.y), float(row.x))
            out.setdefault(int(study_id), {})[LEVEL_TO_IDX[row.level]] = point
    return out


def nearest_slice(geometry: SeriesGeometry, point, require_in_plane=True):
    """Slice whose own plane passes closest to `point`.

    Each slice is measured against its own normal rather than a series-level
    one, so a stack angled per disc level is handled correctly. Slices whose
    in-plane extent does not contain the point are skipped by default - with an
    angled stack a distant block can otherwise present a deceptively small
    out-of-plane distance.
    """
    point = np.asarray(point, dtype=float)
    best = None
    for k, s in enumerate(geometry):
        row, col, offset = s.patient_to_voxel(point)
        inside = (-0.5 <= row <= s.rows - 0.5) and (-0.5 <= col <= s.cols - 0.5)
        if require_in_plane and not inside:
            continue
        if best is None or abs(offset) < abs(best[1]):
            best = (k, offset, inside)
    if best is None and require_in_plane:
        return nearest_slice(geometry, point, require_in_plane=False)
    return best


def build_axial_index_fixed(data_dir, posterior_offset=0.0, report=False):
    """`build_axial_index` with the slice chosen by rule instead of annotation.

    Only `instance` changes. Series, cx, cy and everything downstream are the
    values the annotated configuration uses, so the comparison isolates slice
    selection.
    """
    annotated = build_axial_index(data_dir, posterior_offset=posterior_offset)
    discs = sagittal_disc_centres(data_dir)

    geometry_cache = {}
    fixed = {}
    rows = []

    for study_id, levels in annotated.items():
        study_discs = discs.get(int(study_id), {})
        for level_index, info in levels.items():
            entry = dict(info)
            target = study_discs.get(level_index)
            chosen, offset_mm, inside = None, None, None

            if target is not None:
                key = (int(study_id), int(info["series"]))
                if key not in geometry_cache:
                    try:
                        geometry_cache[key] = SeriesGeometry.from_dir(
                            _series_dir(data_dir, study_id, info["series"]))
                    except (GeometryError, OSError):
                        geometry_cache[key] = None
                geom = geometry_cache[key]
                if geom is not None:
                    hit = nearest_slice(geom, target)
                    if hit is not None:
                        k, offset_mm, inside = hit
                        chosen = int(geom[k].instance_number)

            if chosen is not None:
                entry["instance"] = chosen
                entry["selection"] = "fixed"
            else:
                # No sagittal anchor or no usable geometry: fall back to the
                # annotated slice and record it, so the count is auditable.
                entry["selection"] = "fallback_annotated"

            fixed.setdefault(int(study_id), {})[level_index] = entry
            rows.append({
                "study_id": int(study_id),
                "level": LEVELS[level_index],
                "series": int(info["series"]),
                "annotated_instance": int(info["instance"]),
                "fixed_instance": chosen if chosen is not None else int(info["instance"]),
                "selection": entry["selection"],
                "offset_mm": offset_mm,
                "in_plane": inside,
            })

    if report:
        return fixed, pd.DataFrame(rows)
    return fixed


def selection_report(table: pd.DataFrame) -> dict:
    """How far the rule departs from the radiologist's choice."""
    used = table[table.selection == "fixed"]
    if used.empty:
        return {"n": 0, "n_fallback": int((table.selection != "fixed").sum())}

    delta = (used.fixed_instance - used.annotated_instance).abs()
    offsets = used.offset_mm.dropna().abs()
    return {
        "n": int(len(used)),
        "n_fallback": int((table.selection != "fixed").sum()),
        "same_slice_frac": float((delta == 0).mean()),
        "within_1_slice_frac": float((delta <= 1).mean()),
        "median_slice_delta": float(delta.median()),
        "max_slice_delta": int(delta.max()),
        "median_offset_mm": float(offsets.median()) if len(offsets) else None,
        "p90_offset_mm": float(np.percentile(offsets, 90)) if len(offsets) else None,
        "out_of_plane_frac": float((~used.in_plane.astype(bool)).mean()),
    }
