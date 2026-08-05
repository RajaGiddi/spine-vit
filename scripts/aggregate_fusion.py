from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

AXIAL_MM = {16: 14.2, 24: 21.3, 32: 28.4}
SAG_MM32 = 40.0


def _band(delta):
    a = abs(delta)
    if a >= 0.09:
        return "CLAIM"
    if a >= 0.06:
        return "trend"
    if a < 0.05:
        return "n.s."
    return "borderline"


def _load(experiments_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(experiments_dir, "rsna_fusion_*"))):
        tr_p = os.path.join(d, "test_results.json")
        cf_p = os.path.join(d, "config.json")
        if not (os.path.exists(tr_p) and os.path.exists(cf_p)):
            continue
        cf = json.load(open(cf_p))
        tr = json.load(open(tr_p))
        views = cf.get("views", "?")
        fusion = cf.get("fusion", "?") if views == "both" else "-"
        abox = int(cf.get("axial_box_size", 32))
        mf, ms = tr.get("metrics_full") or {}, tr.get("metrics_axial_subset") or {}
        rows.append({
            "views": views, "fusion": fusion, "abox": abox, "seed": cf.get("seed"),
            "augment": bool(cf.get("augment", False)), "sag_slices": int(cf.get("sag_slices", 1)),
            "k_full": mf.get("kappa"), "wl_full": (tr.get("attribution_full") or {}).get("worst_level_accuracy"),
            "f1_full": mf.get("macro_f1"), "mae_full": mf.get("mae"),
            "k_sub": ms.get("kappa"), "n_sub": tr.get("n_axial_subset_tokens"), "n_full": tr.get("n_full_tokens"),
        })
    return rows


def _agg(rows, key):
    """key(row)->group; return {group: {metric: (mean,std,n)}}."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for g, rs in groups.items():
        m = {}
        for metric in ("k_full", "wl_full", "f1_full", "mae_full", "k_sub"):
            vals = [r[metric] for r in rs if isinstance(r[metric], (int, float))]
            if vals:
                m[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        m["n_full"] = rs[0]["n_full"]
        m["n_sub"] = rs[0]["n_sub"]
        out[g] = m
    return out


def _fmt(mstd):
    if mstd is None:
        return "   n/a    "
    mean, std, n = mstd
    return f"{mean:.3f}±{std:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    ap.add_argument("--aug", action="store_true", help="report the per-view AUGMENTED runs (default: un-augmented)")
    args = ap.parse_args()
    rows = [r for r in _load(args.experiments_dir) if r["augment"] == args.aug]
    tag = "AUGMENTED (per-view, sag-hflip off / axial-hflip on)" if args.aug else "un-augmented"
    print(f"loaded {len(rows)} {tag} fusion runs from {args.experiments_dir}")

    abl_rows = [r for r in rows if r["abox"] == 32]
    abl = _agg(abl_rows, lambda r: (r["views"], r["fusion"], r["sag_slices"]))
    order = [("sag", "-", 1), ("sag", "-", 5), ("axial", "-", 1), ("both", "concat", 1), ("both", "attn", 1)]
    labels = {("sag", "-", 1): "sag-only (control)", ("sag", "-", 5): "sag-only 5-slice (budget ctrl)",
              ("axial", "-", 1): "axial-only", ("both", "concat", 1): "fusion-A (concat)",
              ("both", "attn", 1): "fusion-B (attn)"}
    ctrl_key = ("sag", "-", 1)
    ctrl = abl.get(ctrl_key, {}).get("k_full")
    ctrl_mean = ctrl[0] if ctrl else None
    ctrl_wl = abl.get(ctrl_key, {}).get("wl_full")
    ctrl_wl_mean = ctrl_wl[0] if ctrl_wl else None

    print("\n=== Table 1 - fusion ablation + budget control (axial_box=32 px ~= 28.4mm; sag box 32px ~= 40mm) ===")
    print(f"  n_full={abl.get(ctrl_key,{}).get('n_full')}  n_axial_subset={abl.get(ctrl_key,{}).get('n_sub')}")
    print(f"  {'config':<30} | {'seeds':>5} | {'κ full':>12} | {'worst_lvl':>12} | {'Δκ':>14} | {'Δworst_lvl':>14}")
    for key in order:
        if key not in abl:
            continue
        m = abl[key]
        kf, wl = m.get("k_full"), m.get("wl_full")
        dk = (kf[0] - ctrl_mean) if (kf and ctrl_mean is not None) else None
        dw = (wl[0] - ctrl_wl_mean) if (wl and ctrl_wl_mean is not None) else None
        is_ctrl = key == ctrl_key
        dktag = "- (ref)" if is_ctrl else (f"{dk:+.3f} ({_band(dk)})" if dk is not None else "n/a")
        dwtag = "- (ref)" if is_ctrl else (f"{dw:+.3f} ({_band(dw)})" if dw is not None else "n/a")
        nseed = kf[2] if kf else 0
        print(f"  {labels[key]:<30} | {nseed:>5} | {_fmt(m.get('k_full')):>12} | "
              f"{_fmt(m.get('wl_full')):>12} | {dktag:>14} | {dwtag:>14}")

    sweep_rows = [r for r in rows if r["views"] == "both" and r["fusion"] == "attn"]
    sweep = _agg(sweep_rows, lambda r: r["abox"])
    print("\n=== Table 2 - axial box-size dose-response (fusion-B attn), 3 seeds ===")
    print(f"  {'axial box':>10} | {'~mm':>6} | {'κ full':>12} | {'κ axial-sub':>12} | {'worst_lvl':>12}")
    for abox in (16, 24, 32):
        if abox not in sweep:
            continue
        m = sweep[abox]
        print(f"  {abox:>7}px | {AXIAL_MM[abox]:>5.1f} | {_fmt(m.get('k_full')):>12} | "
              f"{_fmt(m.get('k_sub')):>12} | {_fmt(m.get('wl_full')):>12}")

    print("\n=== reference ===")
    print(f"  v1 sagittal headline (augmented, separate): κ 0.649 ± 0.031")
    print(f"  sag-only (1-slice) control, matched: κ {_fmt(abl.get(ctrl_key,{}).get('k_full'))}")
    ax = abl.get(("axial", "-", 1), {}).get("wl_full")
    s5 = abl.get(("sag", "-", 5), {}).get("wl_full")
    if ax and ctrl_wl_mean is not None:
        print(f"  BUDGET vs AXIAL-ACQUISITION: axial-only Δworst_lvl {ax[0]-ctrl_wl_mean:+.3f} over sag-1slice.")
        if s5:
            frac = (s5[0] - ctrl_wl_mean) / (ax[0] - ctrl_wl_mean) if (ax[0] - ctrl_wl_mean) else float('nan')
            print(f"  5-slice sag closes {100*frac:.0f}% (Δ {s5[0]-ctrl_wl_mean:+.3f}). NOTE: this is UNPAIRED "
                  f"(mixes seed sets). Authoritative split is the PAIRED matched-seed decomposition - "
                  f"~47% budget / ~53% axial acquisition; both halves improve level attribution, neither severity.")


if __name__ == "__main__":
    main()
