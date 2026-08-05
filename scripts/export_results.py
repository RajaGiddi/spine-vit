import argparse
import csv
import glob
import json
import os

FIELDS = ["run", "views", "fusion", "sag_slices", "augment", "seed",
          "kappa", "worst_lvl", "macro_f1", "mae", "kappa_axial_subset",
          "n_full", "n_axial_subset", "best_epoch"]


def collect(experiments_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(experiments_dir, "rsna_fusion_*"))):
        tr_p, cf_p = os.path.join(run_dir, "test_results.json"), os.path.join(run_dir, "config.json")
        if not (os.path.exists(tr_p) and os.path.exists(cf_p)
                and os.path.getsize(tr_p) > 0 and os.path.getsize(cf_p) > 0):
            continue
        config, test_results = json.load(open(cf_p)), json.load(open(tr_p))
        mf = test_results.get("metrics_full") or {}
        ms = test_results.get("metrics_axial_subset") or {}
        att = test_results.get("attribution_full") or {}
        views = config.get("views", "?")
        rows.append({
            "run": os.path.basename(run_dir),
            "views": views,
            "fusion": config.get("fusion", "-") if views == "both" else "-",
            "sag_slices": int(config.get("sag_slices", 1)),
            "augment": bool(config.get("augment", False)),
            "seed": config.get("seed"),
            "kappa": mf.get("kappa"),
            "worst_lvl": att.get("worst_level_accuracy"),
            "macro_f1": mf.get("macro_f1"),
            "mae": mf.get("mae"),
            "kappa_axial_subset": ms.get("kappa"),
            "n_full": test_results.get("n_full_tokens"),
            "n_axial_subset": test_results.get("n_axial_subset_tokens"),
            "best_epoch": test_results.get("best_epoch"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--out", default="results/fusion_results.csv")
    args = parser.parse_args()

    rows = collect(args.experiments_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
