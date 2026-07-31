"""Axial ROI derivation from RSNA subarticular-stenosis annotations (v2 Task 2 plumbing).

Central canal stenosis is graded on AXIAL T2 clinically. RSNA annotates Left/Right
Subarticular Stenosis on the axial series, one (x,y) per side per disc level. The canal
center is approximated by the midpoint of the two subarticular points; the axial slice for
that level is the annotation's instance_number.

Three risks this module surfaces (do NOT trust axial fusion until checked):
  1. Anterior-posterior placement: the subarticular recesses sit lateral AND slightly
     posterior; the L/R midpoint may land anterior to the thecal sac in y. `posterior_offset`
     (derived, not eyeballed) can push the center posteriorly. Default 0 -> verify visually.
  2. Slice provenance: axial lumbar is usually angled stacks (one series per level). We take
     series_id + instance_number from each level's OWN annotation, so each level comes from
     its own stack. `axial_monotonicity_flags` checks instance# is monotonic within a series.
  3. Single-sided fallback: if only one recess is annotated, we use it and flag it.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from .rsna_dataset import LEVELS, LEVEL_TO_IDX, load_dicom_slice

LEFT, RIGHT = "Left Subarticular Stenosis", "Right Subarticular Stenosis"


def build_axial_index(data_dir: str, posterior_offset: float = 0.0) -> Dict[int, Dict[int, Dict]]:
    """study_id -> {level_idx: {series, instance, cx, cy, sided(1/2), inst_lr, series_lr}}.

    cx/cy are ORIGINAL-pixel coords of the derived canal center (L/R midpoint, optional
    posterior offset added to y). sided=2 when both recesses annotated, 1 when single-sided.
    """
    coords = pd.read_csv(os.path.join(data_dir, "train_label_coordinates.csv"))
    sub = coords[coords.condition.isin([LEFT, RIGHT])]

    axial: Dict[int, Dict[int, Dict]] = {}
    for sid, g in sub.groupby("study_id"):
        per_level: Dict[int, Dict] = {}
        for lv in LEVELS:
            rows = g[g.level == lv]
            if len(rows) == 0:
                continue
            pts = []  # (x, y, instance, series)
            for cond in (LEFT, RIGHT):
                r = rows[rows.condition == cond]
                if len(r):
                    r = r.iloc[0]
                    pts.append((float(r.x), float(r.y), int(r.instance_number), int(r.series_id)))
            if not pts:
                continue
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts])) + posterior_offset
            per_level[LEVEL_TO_IDX[lv]] = {
                "series": pts[0][3], "instance": pts[0][2], "cx": cx, "cy": cy,
                "sided": len(pts), "inst_lr": [p[2] for p in pts], "series_lr": [p[3] for p in pts],
            }
        if per_level:
            axial[int(sid)] = per_level
    return axial


def axial_coverage(axial: Dict[int, Dict[int, Dict]], all_study_ids: List[int]) -> Dict:
    """Coverage vs the sagittal cohort: fraction with any / all-5 axial levels."""
    have_any = sum(1 for s in all_study_ids if axial.get(s))
    have_all5 = sum(1 for s in all_study_ids if len(axial.get(s, {})) == 5)
    per_level = {LEVELS[L]: sum(1 for s in all_study_ids if L in axial.get(s, {})) for L in range(5)}
    n = len(all_study_ids)
    return {
        "n_studies": n,
        "has_any_axial": have_any, "pct_any": 100 * have_any / max(1, n),
        "has_all5_axial": have_all5, "pct_all5": 100 * have_all5 / max(1, n),
        "per_level_count": per_level,
    }


def axial_monotonicity_flags(axial: Dict[int, Dict[int, Dict]]) -> List[int]:
    """Studies where, within a shared axial series, instance# is NOT monotonic with level
    index — i.e. a level's slice may come from the wrong stack. These need inspection."""
    bad = []
    for sid, levels in axial.items():
        by_series: Dict[int, List] = {}
        for L, info in levels.items():
            by_series.setdefault(info["series"], []).append((L, info["instance"]))
        for series, items in by_series.items():
            if len(items) < 2:
                continue
            items.sort()  # by level index
            insts = [it[1] for it in items]
            if not (all(a <= b for a, b in zip(insts, insts[1:])) or
                    all(a >= b for a, b in zip(insts, insts[1:]))):
                bad.append(sid)
                break
    return bad


def load_axial_slice(data_dir: str, sid: int, series: int, instance: int) -> np.ndarray:
    return load_dicom_slice(os.path.join(data_dir, "train_images", str(sid), str(series), f"{instance}.dcm"))


def axial_box_mm(data_dir: str, sid: int, series: int, instance: int, box_px_224: int, image_size: int = 224):
    """Physical extent (mm) that box_px_224 covers on this axial slice — so the axial box
    size is chosen from mm, not inherited from the sagittal 32px (different FOV/resolution)."""
    import pydicom

    ds = pydicom.dcmread(os.path.join(data_dir, "train_images", str(sid), str(series), f"{instance}.dcm"),
                         stop_before_pixels=True)
    oh, ow = int(ds.Rows), int(ds.Columns)
    ps = getattr(ds, "PixelSpacing", None)
    row_sp, col_sp = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
    mm_x = box_px_224 * (ow / image_size) * col_sp
    mm_y = box_px_224 * (oh / image_size) * row_sp
    return mm_x, mm_y
