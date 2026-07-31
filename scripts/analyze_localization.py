"""Relate detection quality to grading quality (MICCAI mechanistic result).

Joins per-study localization error (from the detector's localization_per_study.json) with
per-study grading correctness (from the detected-box grading runs' test_predictions.json,
averaged over seeds) and asks: do studies the detector localizes worse also grade worse?

Produces a binned figure (grading accuracy vs localization-error bin) + the correlation.

Usage:
    python analyze_localization.py --experiments_dir outputs_modal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    ap.add_argument("--localization", default=None, help="path to localization_per_study.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    loc_path = args.localization or os.path.join(args.experiments_dir, "detector", "localization_per_study.json")
    if not os.path.exists(loc_path):
        print(f"[error] {loc_path} not found.\n"
              f"  Run the detector pipeline and download its outputs first:\n"
              f"    modal run modal_run.py::detector_pipeline\n"
              f"    modal volume get spine-vit-outputs / ./outputs_modal --force")
        return
    loc = {str(k): v for k, v in json.load(open(loc_path)).items()}

    # detected-box grading runs (all seeds), each with per-study test predictions
    dirs = [d for d in sorted(glob.glob(os.path.join(args.experiments_dir, "*_det_s*")))
            if os.path.exists(os.path.join(d, "test_predictions.json"))]
    if not dirs:
        print(f"[warn] no *_det_s* runs with test_predictions.json under {args.experiments_dir}")
        return
    print(f"[info] {len(dirs)} detected grading run(s): {[os.path.basename(d) for d in dirs]}")

    correct, total = defaultdict(int), defaultdict(int)   # per study, summed over seeds
    for d in dirs:
        tp = json.load(open(os.path.join(d, "test_predictions.json")))
        for sid, tg, pr in zip(tp["studyids"], tp["targets"], tp["preds"]):
            if tg == -1:
                continue
            correct[str(sid)] += int(pr == tg)
            total[str(sid)] += 1

    err, acc = [], []
    for sid, tot in total.items():
        e = loc.get(sid)
        if e is None:
            continue
        err.append(float(e))
        acc.append(correct[sid] / tot)     # per-study exact-grade accuracy (over levels x seeds)
    err, acc = np.array(err), np.array(acc)
    if err.size < 3:
        print("[warn] too few studies to correlate")
        return

    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(err, acc)
    rs, ps = spearmanr(err, acc)
    print(f"\nstudies: {err.size}  |  corr(localization_err, grading_acc): "
          f"Pearson {r:+.3f} (p={p:.2g}), Spearman {rs:+.3f} (p={ps:.2g})")

    edges = [0, 5, 8, 12, np.inf]
    labels = ["≤5mm", "5–8mm", "8–12mm", ">12mm"]
    means, sems, ns = [], [], []
    for i in range(len(edges) - 1):
        m = (err >= edges[i]) & (err < edges[i + 1])
        a = acc[m]
        means.append(float(a.mean()) if a.size else np.nan)
        sems.append(float(a.std() / max(1, np.sqrt(a.size))) if a.size else 0.0)
        ns.append(int(m.sum()))
        print(f"  {labels[i]:7s}: grading_acc {means[-1]:.3f}  (n={ns[-1]})")

    out = args.out or os.path.join(args.experiments_dir, "evaluation", "localization_vs_grading.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, capsize=4, color="#3b6ea5")
    for i, (mn, n) in enumerate(zip(means, ns)):
        if not np.isnan(mn):
            ax.text(i, mn + 0.02, f"{mn:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Per-study localization error")
    ax.set_ylabel("Grading exact-accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Grading accuracy vs detection error  (Spearman {rs:+.2f}, p={ps:.2g})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)

    summary = {"n_studies": int(err.size), "pearson_r": r, "pearson_p": p,
               "spearman_r": rs, "spearman_p": ps,
               "bins": [{"label": labels[i], "mean_acc": means[i], "n": ns[i]} for i in range(len(labels))]}
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] figure -> {out}")


if __name__ == "__main__":
    main()
