import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

AXIAL_MM = {16: 14.2, 24: 21.3, 32: 28.4}
SAG_MM32 = 40.0


def significance_band(delta):
    magnitude = abs(delta)
    if magnitude >= 0.09:
        return "CLAIM"
    if magnitude >= 0.06:
        return "trend"
    if magnitude < 0.05:
        return "n.s."
    return "borderline"


def load_runs(experiments_dir):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(experiments_dir, "rsna_fusion_*"))):
        test_results_path = os.path.join(run_dir, "test_results.json")
        config_path = os.path.join(run_dir, "config.json")
        if not (os.path.exists(test_results_path) and os.path.exists(config_path)):
            continue
        config = json.load(open(config_path))
        test_results = json.load(open(test_results_path))
        views = config.get("views", "?")
        fusion = config.get("fusion", "?") if views == "both" else "-"
        axial_box = int(config.get("axial_box_size", 32))
        mf, ms = test_results.get("metrics_full") or {}, test_results.get("metrics_axial_subset") or {}
        rows.append({
            "views": views, "fusion": fusion, "abox": axial_box, "seed": config.get("seed"),
            "augment": bool(config.get("augment", False)), "sag_slices": int(config.get("sag_slices", 1)),
            "k_full": mf.get("kappa"), "wl_full": (test_results.get("attribution_full") or {}).get("worst_level_accuracy"),
            "f1_full": mf.get("macro_f1"), "mae_full": mf.get("mae"),
            "k_sub": ms.get("kappa"), "n_sub": test_results.get("n_axial_subset_tokens"), "n_full": test_results.get("n_full_tokens"),
        })
    return rows


def group_rows(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    out = {}
    for group_key, members in groups.items():
        metrics = {}
        for metric in ("k_full", "wl_full", "f1_full", "mae_full", "k_sub"):
            values = [row[metric] for row in members if isinstance(row[metric], (int, float))]
            if values:
                metrics[metric] = (float(np.mean(values)), float(np.std(values)), len(values))
        metrics["n_full"] = members[0]["n_full"]
        metrics["n_sub"] = members[0]["n_sub"]
        out[group_key] = metrics
    return out


def format_mean_std(mean_and_std):
    if mean_and_std is None:
        return "   n/a    "
    mean, std, count = mean_and_std
    return f"{mean:.3f}±{std:.3f}"


def ablation_key(row):
    return (row["views"], row["fusion"], row["sag_slices"])


def axial_box_key(row):
    return row["abox"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--aug", action="store_true", help="report the per-view AUGMENTED runs (default: un-augmented)")
    args = parser.parse_args()
    rows = [row for row in load_runs(args.experiments_dir) if row["augment"] == args.aug]
    tag = "AUGMENTED (per-view, sag-hflip off / axial-hflip on)" if args.aug else "un-augmented"
    print(f"loaded {len(rows)} {tag} fusion runs from {args.experiments_dir}")

    abl_rows = [row for row in rows if row["abox"] == 32]
    ablation = group_rows(abl_rows, ablation_key)
    order = [("sag", "-", 1), ("sag", "-", 5), ("axial", "-", 1), ("both", "concat", 1), ("both", "attn", 1)]
    labels = {("sag", "-", 1): "sag-only (control)", ("sag", "-", 5): "sag-only 5-slice (budget ctrl)",
              ("axial", "-", 1): "axial-only", ("both", "concat", 1): "fusion-A (concat)",
              ("both", "attn", 1): "fusion-B (attn)"}
    ctrl_key = ("sag", "-", 1)
    ctrl = ablation.get(ctrl_key, {}).get("k_full")
    ctrl_mean = ctrl[0] if ctrl else None
    ctrl_wl = ablation.get(ctrl_key, {}).get("wl_full")
    ctrl_wl_mean = ctrl_wl[0] if ctrl_wl else None

    print("\n=== Table 1 - fusion ablation + budget control (axial_box=32 px ~= 28.4mm; sag box 32px ~= 40mm) ===")
    print(f"  n_full={ablation.get(ctrl_key,{}).get('n_full')}  n_axial_subset={ablation.get(ctrl_key,{}).get('n_sub')}")
    print(f"  {'config':<30} | {'seeds':>5} | {'κ full':>12} | {'worst_lvl':>12} | {'Δκ':>14} | {'Δworst_lvl':>14}")
    for key in order:
        if key not in ablation:
            continue
        metrics = ablation[key]
        kappa_full, worst_level = metrics.get("k_full"), metrics.get("wl_full")
        delta_kappa = (kappa_full[0] - ctrl_mean) if (kappa_full and ctrl_mean is not None) else None
        delta_worst_level = (worst_level[0] - ctrl_wl_mean) if (worst_level and ctrl_wl_mean is not None) else None
        is_ctrl = key == ctrl_key
        dktag = "- (ref)" if is_ctrl else (f"{delta_kappa:+.3f} ({significance_band(delta_kappa)})" if delta_kappa is not None else "n/a")
        dwtag = "- (ref)" if is_ctrl else (f"{delta_worst_level:+.3f} ({significance_band(delta_worst_level)})" if delta_worst_level is not None else "n/a")
        nseed = kappa_full[2] if kappa_full else 0
        print(f"  {labels[key]:<30} | {nseed:>5} | {format_mean_std(metrics.get('k_full')):>12} | "
              f"{format_mean_std(metrics.get('wl_full')):>12} | {dktag:>14} | {dwtag:>14}")

    sweep_rows = [row for row in rows if row["views"] == "both" and row["fusion"] == "attn"]
    sweep = group_rows(sweep_rows, axial_box_key)
    print("\n=== Table 2 - axial box-size dose-response (fusion-B attn), 3 seeds ===")
    print(f"  {'axial box':>10} | {'~mm':>6} | {'κ full':>12} | {'κ axial-sub':>12} | {'worst_lvl':>12}")
    for axial_box in (16, 24, 32):
        if axial_box not in sweep:
            continue
        metrics = sweep[axial_box]
        print(f"  {axial_box:>7}px | {AXIAL_MM[axial_box]:>5.1f} | {format_mean_std(metrics.get('k_full')):>12} | "
              f"{format_mean_std(metrics.get('k_sub')):>12} | {format_mean_std(metrics.get('wl_full')):>12}")

    print("\n=== reference ===")
    print(f"  v1 sagittal headline (augmented, separate): κ 0.649 ± 0.031")
    print(f"  sag-only (1-slice) control, matched: κ {format_mean_std(ablation.get(ctrl_key,{}).get('k_full'))}")
    axial_row = ablation.get(("axial", "-", 1), {}).get("wl_full")
    sag_five = ablation.get(("sag", "-", 5), {}).get("wl_full")
    if axial_row and ctrl_wl_mean is not None:
        print(f"  BUDGET vs AXIAL-ACQUISITION: axial-only Δworst_lvl {axial_row[0]-ctrl_wl_mean:+.3f} over sag-1slice.")
        if sag_five:
            frac = (sag_five[0] - ctrl_wl_mean) / (axial_row[0] - ctrl_wl_mean) if (axial_row[0] - ctrl_wl_mean) else float('nan')
            print(f"  5-slice sag closes {100*frac:.0f}% (Δ {sag_five[0]-ctrl_wl_mean:+.3f}). NOTE: this is UNPAIRED "
                  f"(mixes seed sets). Authoritative split is the PAIRED matched-seed decomposition - "
                  f"~47% budget / ~53% axial acquisition; both halves improve level attribution, neither severity.")


if __name__ == "__main__":
    main()
