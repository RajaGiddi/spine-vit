"""EDA for the RSNA / LumbarDISC dataset (instructions.md Phase 1, Step 11.4).

Loads a few samples, prints shapes and grade distributions, and saves MRI slices with
their level boxes overlaid so you can eyeball that boxes land on the discs.

Run:  python scripts/explore_rsna.py --data_dir /path/to/rsna-2024
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data.rsna_dataset import RSNADataset, LEVELS

STENOSIS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


def overlay_sample(sample, idx, out_dir):
    img = sample["image"].numpy()  # (3, H, W)
    disp = img[1]  # center slice
    boxes = sample["boxes"].numpy()
    level_idx = sample["level_indices"].numpy()
    targets = sample["targets"].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(disp, cmap="gray")
    for b, li, t in zip(boxes, level_idx, targets):
        x1, y1, x2, y2 = b
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none"))
        lname = LEVELS[li] if li < len(LEVELS) else f"L{li}"
        grade = STENOSIS_NAMES[t] if 0 <= t < len(STENOSIS_NAMES) else "NA"
        ax.text(x1, y1 - 3, f"{lname}:{grade}", color="yellow", fontsize=8)
    ax.set_title(f"RSNA study {sample['study_id']}")
    ax.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rsna_sample_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="outputs/eda")
    args = ap.parse_args()

    ds = RSNADataset(args.data_dir, augment=False)
    print(f"Total RSNA samples (studies with sagittal-T2 canal annotations): {len(ds)}")

    for i in range(min(args.n, len(ds))):
        s = ds[i]
        print(
            f"\n[sample {i}] study {s['study_id']}  image {tuple(s['image'].shape)}  "
            f"levels {s['num_levels']}"
        )
        print(f"  level_indices {s['level_indices'].tolist()}  types {s['level_types'].tolist()}")
        print(f"  targets       {s['targets'].tolist()}")
        print(f"  boxes:\n{np.round(s['boxes'].numpy(), 1)}")
        p = overlay_sample(s, i, args.out_dir)
        print(f"  overlay -> {p}")

    # Grade distribution across the dataset.
    targets = ds.get_all_targets()
    valid = targets[targets != -1]
    print("\nGrade distribution (spinal canal stenosis):")
    for c, name in enumerate(STENOSIS_NAMES):
        print(f"  {name:12s}: {(valid == c).sum():5d}  ({(valid == c).mean()*100:5.1f}%)")
    print(f"  missing/-1   : {(targets == -1).sum():5d}")


if __name__ == "__main__":
    main()
