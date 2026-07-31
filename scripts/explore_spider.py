"""EDA for the SPIDER dataset (instructions.md Phase 1, Step 11.5).

Loads a few samples, prints shapes and Pfirrmann grade distributions, and saves
mid-sagittal slices with mask-derived vertebra/disc boxes overlaid.

Run:  python scripts/explore_spider.py --data_dir /path/to/spider
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

from data.spider_dataset import SPIDERDataset

PFIRRMANN_NAMES = ["I", "II", "III", "IV", "V"]


def overlay_sample(sample, idx, out_dir):
    img = sample["image"].numpy()
    disp = img[1]
    boxes = sample["boxes"].numpy()
    types = sample["level_types"].numpy()
    targets = sample["targets"].numpy()

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.imshow(disp, cmap="gray")
    for b, t, tg in zip(boxes, types, targets):
        x1, y1, x2, y2 = b
        color = "cyan" if t == 1 else "orange"  # disc vs vertebra
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=color, facecolor="none"))
        if t == 1 and 0 <= tg < len(PFIRRMANN_NAMES):
            ax.text(x1, y1 - 2, PFIRRMANN_NAMES[tg], color="yellow", fontsize=8)
    ax.set_title(f"SPIDER patient {sample['study_id']}  (cyan=disc, orange=vertebra)")
    ax.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"spider_sample_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="outputs/eda")
    ap.add_argument("--no_oracle", action="store_true", help="use the intensity heuristic instead of masks")
    args = ap.parse_args()

    ds = SPIDERDataset(args.data_dir, augment=False, use_oracle_regions=not args.no_oracle)
    print(f"Total SPIDER samples: {len(ds)}  (oracle_regions={not args.no_oracle})")

    for i in range(min(args.n, len(ds))):
        s = ds[i]
        n_disc = int((s["level_types"] == 1).sum())
        n_vert = int((s["level_types"] == 0).sum())
        print(
            f"\n[sample {i}] patient {s['study_id']}  image {tuple(s['image'].shape)}  "
            f"tokens {s['num_levels']} (discs {n_disc}, vertebrae {n_vert})"
        )
        print(f"  level_indices {s['level_indices'].tolist()}")
        print(f"  level_types   {s['level_types'].tolist()}")
        print(f"  targets       {s['targets'].tolist()}")
        p = overlay_sample(s, i, args.out_dir)
        print(f"  overlay -> {p}")

    targets = ds.get_all_targets()
    valid = targets[targets != -1]
    print("\nPfirrmann grade distribution:")
    for c, name in enumerate(PFIRRMANN_NAMES):
        pct = (valid == c).mean() * 100 if valid.size else 0.0
        print(f"  Grade {name:3s}: {(valid == c).sum():5d}  ({pct:5.1f}%)")
    print(f"  missing/-1 : {(targets == -1).sum():5d}")


if __name__ == "__main__":
    main()
