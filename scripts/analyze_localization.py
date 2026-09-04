import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--localization", default=None, help="path to localization_per_study.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    loc_path = args.localization or os.path.join(args.experiments_dir, "detector", "localization_per_study.json")
    if not os.path.exists(loc_path):
        print(f"[error] {loc_path} not found.\n"
              f"  Run the detector pipeline and download its outputs first:\n"
              f"    modal run modal_run.py::detector_pipeline\n"
              f"    modal volume get spine-vit-outputs / ./outputs_modal --force")
        return
    loc = {str(k): v for k, v in json.load(open(loc_path)).items()}

    dirs = [run_dir for run_dir in sorted(glob.glob(os.path.join(args.experiments_dir, "*_det_s*")))
            if os.path.exists(os.path.join(run_dir, "test_predictions.json"))]
    if not dirs:
        print(f"[warn] no *_det_s* runs with test_predictions.json under {args.experiments_dir}")
        return
    print(f"[info] {len(dirs)} detected grading run(s): {[os.path.basename(run_dir) for run_dir in dirs]}")

    correct, total = defaultdict(int), defaultdict(int)
    for run_dir in dirs:
        tp = json.load(open(os.path.join(run_dir, "test_predictions.json")))
        for study_id, tg, pr in zip(tp["studyids"], tp["targets"], tp["predictions"]):
            if tg == -1:
                continue
            correct[str(study_id)] += int(pr == tg)
            total[str(study_id)] += 1

    err, accuracy = [], []
    for study_id, tot in total.items():
        error = loc.get(study_id)
        if error is None:
            continue
        err.append(float(error))
        accuracy.append(correct[study_id] / tot)
    err, accuracy = np.array(err), np.array(accuracy)
    if err.size < 3:
        print("too few studies to correlate")
        return

    r, p = pearsonr(err, accuracy)
    rs, ps = spearmanr(err, accuracy)
    print(f"\nstudies: {err.size}  |  corr(localization_err, grading_acc): "
          f"Pearson {r:+.3f} (p={p:.2g}), Spearman {rs:+.3f} (p={ps:.2g})")

    edges = [0, 5, 8, 12, np.inf]
    labels = ["≤5mm", "5-8mm", "8-12mm", ">12mm"]
    means, sems, ns = [], [], []
    for i in range(len(edges) - 1):
        mask = (err >= edges[i]) & (err < edges[i + 1])
        accuracies = accuracy[mask]
        means.append(float(accuracies.mean()) if accuracies.size else np.nan)
        sems.append(float(accuracies.std() / max(1, np.sqrt(accuracies.size))) if accuracies.size else 0.0)
        ns.append(int(mask.sum()))
        print(f"  {labels[i]:7s}: grading_acc {means[-1]:.3f}  (n={ns[-1]})")

    out = args.out or os.path.join(args.experiments_dir, "evaluation", "localization_vs_grading.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, capsize=4, color="#3b6ea5")
    for i, (mean_value, count) in enumerate(zip(means, ns)):
        if not np.isnan(mean_value):
            ax.text(i, mean_value + 0.02, f"{mean_value:.2f}\n(n={count})", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Per-study localization error")
    ax.set_ylabel("Grading exact-accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Grading accuracy vs detection error  (Spearman {rs:+.2f}, p={ps:.2g})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    summary = {"n_studies": int(err.size), "pearson_r": r, "pearson_p": p,
               "spearman_r": rs, "spearman_p": ps,
               "bins": [{"label": labels[i], "mean_acc": means[i], "n": ns[i]} for i in range(len(labels))]}
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[info] figure -> {out}")


if __name__ == "__main__":
    main()
