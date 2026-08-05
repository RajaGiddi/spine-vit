from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME_RE = re.compile(r"rsna_anatomy_ordinal_256_2(_det)?(?:_b(\d+))?_s(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vals = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.experiments_dir, "rsna_anatomy_ordinal_256_2*_s*"))):
        m = NAME_RE.match(os.path.basename(d))
        tr = os.path.join(d, "test_results.json")
        if not m or "_ft" in os.path.basename(d) or not os.path.exists(tr):
            continue
        source = "detected" if m.group(1) else "oracle"
        bs = int(m.group(2)) if m.group(2) else 32
        vals[(source, bs)].append(json.load(open(tr))["metrics"]["kappa"])

    sizes = sorted({bs for (_, bs) in vals})
    if not sizes:
        print(f"[warn] no anatomy+ordinal runs found under {args.experiments_dir}")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print(f"{'box':>4} {'oracle κ':>16} {'detected κ':>16} {'gap':>7}")
    for source, color, mark in [("oracle", "#2a7", "o"), ("detected", "#d55", "s")]:
        xs, ys, es = [], [], []
        for bs in sizes:
            v = vals.get((source, bs))
            if v:
                xs.append(bs); ys.append(np.mean(v)); es.append(np.std(v))
        ax.errorbar(xs, ys, yerr=es, marker=mark, color=color, capsize=4, label=source, lw=2)
    for bs in sizes:
        o, det = vals.get(("oracle", bs)), vals.get(("detected", bs))
        om = f"{np.mean(o):.3f}±{np.std(o):.3f}" if o else "-"
        dm = f"{np.mean(det):.3f}±{np.std(det):.3f}" if det else "-"
        gap = f"{np.mean(o)-np.mean(det):+.3f}" if (o and det) else "-"
        print(f"{bs:>4} {om:>16} {dm:>16} {gap:>7}")

    ax.set_xlabel("Box extent (224-space px)")
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Grading κ vs ROI size: oracle vs learned detection")
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(alpha=0.25)
    out = args.out or os.path.join(args.experiments_dir, "evaluation", "boxsize_dose_response.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[info] figure -> {out}")


if __name__ == "__main__":
    main()
