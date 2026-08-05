import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.rsna_dataset import build_rsna_index, LEVELS
from data.rsna_axial import (
    build_axial_index, axial_coverage, axial_monotonicity_flags, load_axial_slice, axial_box_mm, LEFT, RIGHT,
)
import pandas as pd


def overlay_study(data_dir, sid, levels, coords_sub, out_dir, box_px=64):
    present = sorted(levels)
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.4))
    if len(present) == 1:
        axes = [axes]
    for ax, L in zip(axes, present):
        info = levels[L]
        try:
            img = load_axial_slice(data_dir, sid, info["series"], info["instance"])
        except Exception:
            ax.set_title(f"{LEVELS[L]} (load fail)")
            ax.axis("off")
            continue
        ax.imshow(img, cmap="gray")
        rows = coords_sub[(coords_sub.study_id == sid) & (coords_sub.level == LEVELS[L])]
        for _, row in rows.iterrows():
            ax.plot(row.x, row.y, "r+", markersize=11, markeredgewidth=2)
        cx, cy = info["cx"], info["cy"]
        ax.plot(cx, cy, "bx", markersize=10, markeredgewidth=2)
        ax.add_patch(patches.Rectangle((cx - box_px / 2, cy - box_px / 2), box_px, box_px,
                                       fill=False, edgecolor="lime", linewidth=1.5))
        tag = "1-sided" if info["sided"] == 1 else ""
        ax.set_title(f"{LEVELS[L]} inst{info['instance']} {tag}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"study {sid}  (red=L/R subarticular, blue=derived canal center)", fontsize=9)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"axial_{sid}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out_dir", default="outputs/eda_axial")
    args = parser.parse_args()

    sag = build_rsna_index(args.data_dir)
    sag_ids = sorted({sample["study_id"] for sample in sag})
    axial = build_axial_index(args.data_dir)

    coverage = axial_coverage(axial, sag_ids)
    print(f"\n=== Axial coverage vs sagittal cohort ({coverage['n_studies']} studies) ===")
    print(f"  any axial level : {coverage['has_any_axial']} ({coverage['pct_any']:.0f}%)")
    print(f"  all 5 levels    : {coverage['has_all5_axial']} ({coverage['pct_all5']:.0f}%)")
    print(f"  per level       : {coverage['per_level_count']}")
    if coverage["pct_any"] < 70:
        print("  <70% axial coverage - keep masked fusion, report canal metrics on full + axial subset")

    bad = axial_monotonicity_flags(axial)
    print(f"\n=== Slice provenance: instance# monotonic within series ===")
    print(f"  studies with NON-monotonic instance#/level (inspect): {len(bad)}  e.g. {bad[:5]}")

    on_disk = tot = 0
    for sid, levels in axial.items():
        for L, info in levels.items():
            tot += 1
            p = os.path.join(args.data_dir, "train_images", str(sid), str(info["series"]), f"{info['instance']}.dcm")
            on_disk += os.path.exists(p)
    print(f"\n=== Axial IMAGES on disk: {on_disk}/{tot} referenced slices ===")
    if on_disk == 0:
        print("  axial images not downloaded - the visual gate + box-mm need them.")
        print("  Re-export axial subarticular slices from a Kaggle notebook (as done for sagittal),")
        print("  then re-run this script. Coverage/provenance above are computed from CSVs only.")
        return

    print(f"\n=== What 32px (224-space) covers on AXIAL, mm (choose axial box from this) ===")
    shown = 0
    for sid in sag_ids:
        if sid in axial and axial[sid]:
            L = next(iter(axial[sid]))
            info = axial[sid][L]
            mm_x, mm_y = axial_box_mm(args.data_dir, sid, info["series"], info["instance"], 32)
            print(f"  study {sid}: 32px -> {mm_x:.1f} x {mm_y:.1f} mm")
            shown += 1
            if shown >= 5:
                break

    coords = pd.read_csv(os.path.join(args.data_dir, "train_label_coordinates.csv"))
    sub = coords[coords.condition.isin([LEFT, RIGHT])]
    with_all5 = [sample for sample in sag_ids if len(axial.get(sample, {})) == 5][: args.n]
    print(f"\n=== Rendering {len(with_all5)} axial overlays -> {args.out_dir} (INSPECT ALL) ===")
    for sid in with_all5:
        overlay_study(args.data_dir, sid, axial[sid], sub, args.out_dir)
    print("Done. Inspect every panel: box on central canal? slice at the labeled level?")


if __name__ == "__main__":
    main()
