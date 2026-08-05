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
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    kappa_by_config = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(args.experiments_dir, "rsna_anatomy_ordinal_256_2*_s*"))):
        match = NAME_RE.match(os.path.basename(run_dir))
        test_results_path = os.path.join(run_dir, "test_results.json")
        if not match or "_ft" in os.path.basename(run_dir) or not os.path.exists(test_results_path):
            continue
        source = "detected" if match.group(1) else "oracle"
        box_size = int(match.group(2)) if match.group(2) else 32
        kappa_by_config[(source, box_size)].append(json.load(open(test_results_path))["metrics"]["kappa"])

    sizes = sorted({box_size for (_, box_size) in kappa_by_config})
    if not sizes:
        print(f"[warn] no anatomy+ordinal runs found under {args.experiments_dir}")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    print(f"{'box':>4} {'oracle κ':>16} {'detected κ':>16} {'gap':>7}")
    for source, color, mark in [("oracle", "#2a7", "o"), ("detected", "#d55", "s")]:
        x_values, y_values, error_values = [], [], []
        for box_size in sizes:
            values = kappa_by_config.get((source, box_size))
            if values:
                x_values.append(box_size)
                y_values.append(np.mean(values))
                error_values.append(np.std(values))
        ax.errorbar(x_values, y_values, yerr=error_values, marker=mark, color=color, capsize=4, label=source, lw=2)
    for box_size in sizes:
        oracle_values = kappa_by_config.get(("oracle", box_size))
        detected_values = kappa_by_config.get(("detected", box_size))
        oracle_text = f"{np.mean(oracle_values):.3f}±{np.std(oracle_values):.3f}" if oracle_values else "-"
        detected_text = f"{np.mean(detected_values):.3f}±{np.std(detected_values):.3f}" if detected_values else "-"
        gap = f"{np.mean(oracle_values)-np.mean(detected_values):+.3f}" if (oracle_values and detected_values) else "-"
        print(f"{box_size:>4} {oracle_text:>16} {detected_text:>16} {gap:>7}")

    ax.set_xlabel("Box extent (224-space px)")
    ax.set_ylabel("Cohen's κ")
    ax.set_title("Grading κ vs ROI size: oracle vs learned detection")
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(alpha=0.25)
    out = args.out or os.path.join(args.experiments_dir, "evaluation", "boxsize_dose_response.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[info] figure -> {out}")


if __name__ == "__main__":
    main()
