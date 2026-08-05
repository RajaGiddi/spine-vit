from __future__ import annotations

import argparse
import csv
import glob
import json
import os

FIELDS = ["run", "views", "fusion", "sag_slices", "augment", "seed",
          "kappa", "worst_lvl", "macro_f1", "mae", "kappa_axial_subset",
          "n_full", "n_axial_subset", "best_epoch"]


def collect(experiments_dir: str):
    rows = []
    for d in sorted(glob.glob(os.path.join(experiments_dir, "rsna_fusion_*"))):
        tr_p, cf_p = os.path.join(d, "test_results.json"), os.path.join(d, "config.json")
        if not (os.path.exists(tr_p) and os.path.exists(cf_p)
                and os.path.getsize(tr_p) > 0 and os.path.getsize(cf_p) > 0):
            continue
        cf, tr = json.load(open(cf_p)), json.load(open(tr_p))
        mf = tr.get("metrics_full") or {}
        ms = tr.get("metrics_axial_subset") or {}
        att = tr.get("attribution_full") or {}
        views = cf.get("views", "?")
        rows.append({
            "run": os.path.basename(d),
            "views": views,
            "fusion": cf.get("fusion", "-") if views == "both" else "-",
            "sag_slices": int(cf.get("sag_slices", 1)),
            "augment": bool(cf.get("augment", False)),
            "seed": cf.get("seed"),
            "kappa": mf.get("kappa"),
            "worst_lvl": att.get("worst_level_accuracy"),
            "macro_f1": mf.get("macro_f1"),
            "mae": mf.get("mae"),
            "kappa_axial_subset": ms.get("kappa"),
            "n_full": tr.get("n_full_tokens"),
            "n_axial_subset": tr.get("n_axial_subset_tokens"),
            "best_epoch": tr.get("best_epoch"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    ap.add_argument("--out", default="results/fusion_results.csv")
    args = ap.parse_args()

    rows = collect(args.experiments_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
