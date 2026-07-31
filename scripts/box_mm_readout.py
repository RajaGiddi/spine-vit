"""Box-size -> physical mm readout for BOTH views (v2 Task 2, pre-sweep).

Pixel parity is not physical parity: a 32px box in 224-space covers a different mm
extent on sagittal vs axial (different FOV / PixelSpacing). Before the box-size sweep we
report, per view, what 16/24/32 px covers in mm across studies, so each view's sweep is
interpreted in physical terms and the two views are compared at matched *physical* scale,
not matched pixel scale.

Run:  ./.venv/bin/python scripts/box_mm_readout.py --data_dir data/rsna
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pydicom

from data.rsna_dataset import build_rsna_index
from data.rsna_axial import build_axial_index

BOX_PX = [16, 24, 32]
IMAGE_SIZE = 224


def _slice_mm(data_dir, sid, series, instance, box_px_list):
    """Return {box_px: (mm_x, mm_y)} for one slice, or None if unreadable/off-disk."""
    p = os.path.join(data_dir, "train_images", str(sid), str(series), f"{instance}.dcm")
    if not os.path.exists(p):
        return None
    try:
        ds = pydicom.dcmread(p, stop_before_pixels=True)
    except Exception:
        return None
    oh, ow = int(ds.Rows), int(ds.Columns)
    ps = getattr(ds, "PixelSpacing", None)
    row_sp, col_sp = (float(ps[0]), float(ps[1])) if ps is not None else (1.0, 1.0)
    out = {}
    for bp in box_px_list:
        mm_x = bp * (ow / IMAGE_SIZE) * col_sp
        mm_y = bp * (oh / IMAGE_SIZE) * row_sp
        out[bp] = (mm_x, mm_y)
    return out


def _summarize(view_name, per_study_mm):
    """per_study_mm: list of {box_px: (mm_x, mm_y)}. Report mean side (avg of x,y) stats."""
    print(f"\n=== {view_name}: what each box_px covers in mm (n={len(per_study_mm)} studies) ===")
    print(f"  {'box_px':>7} | {'mean mm':>9} | {'median':>7} | {'min':>6} | {'max':>6} | {'p10-p90':>13}")
    for bp in BOX_PX:
        sides = np.array([0.5 * (d[bp][0] + d[bp][1]) for d in per_study_mm if bp in d])
        p10, p90 = np.percentile(sides, [10, 90])
        print(f"  {bp:>7} | {sides.mean():>9.1f} | {np.median(sides):>7.1f} | "
              f"{sides.min():>6.1f} | {sides.max():>6.1f} | {p10:>5.1f}-{p90:<5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    args = ap.parse_args()

    sag = build_rsna_index(args.data_dir)
    axial = build_axial_index(args.data_dir)

    # --- sagittal: representative slice per study ---
    sag_mm = []
    for s in sag:
        d = _slice_mm(args.data_dir, s["study_id"], s["series_id"], s["instance_number"], BOX_PX)
        if d is not None:
            sag_mm.append(d)

    # --- axial: one representative covered level per study (canal slice) ---
    ax_mm = []
    ax_missing = 0
    for sid, levels in axial.items():
        info = next(iter(levels.values()))  # any covered level's canal slice
        d = _slice_mm(args.data_dir, sid, info["series"], info["instance"], BOX_PX)
        if d is not None:
            ax_mm.append(d)
        else:
            ax_missing += 1

    _summarize("SAGITTAL (v1 tokenizer view)", sag_mm)
    _summarize("AXIAL (fusion second view)", ax_mm)
    if ax_missing:
        print(f"\n  [note] {ax_missing} axial studies had no readable representative slice on disk")

    # matched-physical-scale hint: which axial px ~ 32px sagittal in mm?
    sag32 = np.median([0.5 * (d[32][0] + d[32][1]) for d in sag_mm])
    ax_med = {bp: np.median([0.5 * (d[bp][0] + d[bp][1]) for d in ax_mm]) for bp in BOX_PX}
    print(f"\n=== matched-physical readout ===")
    print(f"  sagittal 32px median ~= {sag32:.1f} mm")
    for bp in BOX_PX:
        print(f"  axial {bp}px median ~= {ax_med[bp]:.1f} mm")
    print("  (interpret the two sweeps at their own optimum; report mm so pixel!=physical is explicit)")


if __name__ == "__main__":
    main()
