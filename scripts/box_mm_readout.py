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


def slice_mm(data_dir, study_id, series, instance, box_px_list):
    p = os.path.join(data_dir, "train_images", str(study_id), str(series), f"{instance}.dcm")
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


def summarize(view_name, per_study_mm):
    print(f"\n=== {view_name}: what each box_px covers in mm (n={len(per_study_mm)} studies) ===")
    print(f"  {'box_px':>7} | {'mean mm':>9} | {'median':>7} | {'min':>6} | {'max':>6} | {'p10-p90':>13}")
    for bp in BOX_PX:
        sides = np.array([0.5 * (dimensions[bp][0] + dimensions[bp][1]) for dimensions in per_study_mm if bp in dimensions])
        p10, p90 = np.percentile(sides, [10, 90])
        print(f"  {bp:>7} | {sides.mean():>9.1f} | {np.median(sides):>7.1f} | "
              f"{sides.min():>6.1f} | {sides.max():>6.1f} | {p10:>5.1f}-{p90:<5.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    sag = build_rsna_index(args.data_dir)
    axial = build_axial_index(args.data_dir)

    sag_mm = []
    for sample in sag:
        dimensions = slice_mm(args.data_dir, sample["study_id"], sample["series_id"], sample["instance_number"], BOX_PX)
        if dimensions is not None:
            sag_mm.append(dimensions)

    ax_mm = []
    ax_missing = 0
    for study_id, levels in axial.items():
        info = next(iter(levels.values()))
        dimensions = slice_mm(args.data_dir, study_id, info["series"], info["instance"], BOX_PX)
        if dimensions is not None:
            ax_mm.append(dimensions)
        else:
            ax_missing += 1

    summarize("SAGITTAL (v1 tokenizer view)", sag_mm)
    summarize("AXIAL (fusion second view)", ax_mm)
    if ax_missing:
        print(f"\n  [note] {ax_missing} axial studies had no readable representative slice on disk")

    sag32 = np.median([0.5 * (dimensions[32][0] + dimensions[32][1]) for dimensions in sag_mm])
    ax_med = {bp: np.median([0.5 * (dimensions[bp][0] + dimensions[bp][1]) for dimensions in ax_mm]) for bp in BOX_PX}
    print(f"\n=== matched-physical readout ===")
    print(f"  sagittal 32px median ~= {sag32:.1f} mm")
    for bp in BOX_PX:
        print(f"  axial {bp}px median ~= {ax_med[bp]:.1f} mm")
    print("  (interpret the two sweeps at their own optimum; report mm so pixel!=physical is explicit)")


if __name__ == "__main__":
    main()
