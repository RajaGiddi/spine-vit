#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

CFG = {
    "anatomy": "rsna_anatomy_ordinal_256_2",
    "cast_crop": "rsna_cast_crop_ordinal_256_2",
    "patch_query": "rsna_patches_ordinal_256_2",
    "strips": "rsna_strips_ordinal_256_2",
}
METRICS = [("kappa", "quadratic-weighted kappa"), ("worst", "worst-level accuracy"),
           ("macro_f1", "macro F1")]
COMPARISONS = [("anatomy", "strips"), ("anatomy", "patch_query"),
               ("anatomy", "cast_crop"), ("patch_query", "strips")]
BASE_SEEDS = [42, 43, 44]


def load(experiments_dir):
    out = {}
    for name, pre in CFG.items():
        runs = {}
        for d in sorted(glob.glob(os.path.join(experiments_dir, f"{pre}_s*"))):
            tr_p, cf_p = os.path.join(d, "test_results.json"), os.path.join(d, "config.json")
            if not (os.path.exists(tr_p) and os.path.exists(cf_p) and os.path.getsize(tr_p) > 0):
                continue
            cf, tr = json.load(open(cf_p)), json.load(open(tr_p))
            m = tr.get("metrics_full") or tr.get("metrics") or {}
            att = tr.get("attribution_full") or tr.get("attribution") or {}
            runs[int(cf["seed"])] = {"kappa": m.get("kappa"), "worst": att.get("worst_level_accuracy"),
                                     "macro_f1": m.get("macro_f1"), "best_epoch": tr.get("best_epoch"),
                                     "n_path_studies": att.get("n_studies_with_pathology"), "dir": d}
        out[name] = runs
    return out


def paired(runs, a, b, seeds, metric):
    A = np.array([runs[a][s][metric] for s in seeds], dtype=float)
    B = np.array([runs[b][s][metric] for s in seeds], dtype=float)
    d = A - B
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    lo, hi = d.mean() - tc * se, d.mean() + tc * se
    t, p = stats.ttest_rel(A, B)
    return {"A": A, "B": B, "d": d, "n": n, "mean": d.mean(), "sd": sd, "se": se,
            "lo": lo, "hi": hi, "t": t, "p": p,
            "sep": bool(p < 0.05 and (lo > 0 or hi < 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments_dir", default="outputs_modal")
    args = ap.parse_args()
    runs = load(args.experiments_dir)

    print("=== seeds present ===")
    for n, r in runs.items():
        print(f"  {n:<12} {sorted(r)}  (n={len(r)})")

    print("\n=== per-tokenizer aggregates over all seeds present ===")
    print(f"{'tokenizer':<12}{'n':>3}  " + "".join(f"{lab:>28}" for _, lab in METRICS))
    for name in CFG:
        seeds = sorted(runs[name])
        cells = []
        for key, _ in METRICS:
            v = np.array([runs[name][s][key] for s in seeds], dtype=float)
            cells.append(f"{v.mean():.4f} +/- {v.std():.4f} (sd1 {v.std(ddof=1):.4f})")
        print(f"{name:<12}{len(seeds):>3}  " + "".join(f"{c:>28}" for c in cells))

    print("\n=== per-seed values ===")
    for key, lab in METRICS:
        print(f"  -- {lab}")
        allseeds = sorted({s for r in runs.values() for s in r})
        print(f"     {'tokenizer':<12}" + "".join(f"{'s'+str(s):>10}" for s in allseeds))
        for name in CFG:
            row = "".join(f"{runs[name][s][key]:10.4f}" if s in runs[name] else f"{'-':>10}"
                          for s in allseeds)
            print(f"     {name:<12}{row}")

    changed = []
    for a, b in COMPARISONS:
        shared = sorted(set(runs[a]) & set(runs[b]))
        print(f"\n================ {a} vs {b} ================")
        print(f"  seed sets identical: {set(runs[a]) == set(runs[b])}   shared: {shared}")
        for key, lab in METRICS[:2]:
            sets = [("3-seed", BASE_SEEDS)] if len(shared) <= 3 else \
                   [("3-seed", BASE_SEEDS), (f"{len(shared)}-seed", shared)]
            verdicts = {}
            for tag, seeds in sets:
                if not set(seeds) <= set(shared):
                    continue
                r = paired(runs, a, b, seeds, key)
                verdicts[tag] = r["sep"]
                print(f"\n  --- {lab} | {tag} (n={r['n']}) ---")
                for s, av, bv, dv in zip(seeds, r["A"], r["B"], r["d"]):
                    print(f"      s{s}: {a} {av:.6f}  {b} {bv:.6f}   delta {dv:+.6f}")
                print(f"      mean {r['mean']:+.6f}  SD(deltas) {r['sd']:.6f}  SE {r['se']:.6f}")
                print(f"      95% CI [{r['lo']:+.6f}, {r['hi']:+.6f}]   t = {r['t']:.4f}   p = {r['p']:.4f}")
                print(f"      signs {int((r['d'] > 0).sum())}/{r['n']} positive   -> "
                      f"{'SEPARABLE' if r['sep'] else 'not separable'}")
            if len(verdicts) == 2:
                tags = list(verdicts)
                if verdicts[tags[0]] != verdicts[tags[1]]:
                    changed.append(f"{a} vs {b} [{lab}]: {tags[0]} "
                                   f"{'SEPARABLE' if verdicts[tags[0]] else 'not separable'} -> {tags[1]} "
                                   f"{'SEPARABLE' if verdicts[tags[1]] else 'not separable'}")

    print("\n\n================ VERDICT CHANGES vs the 3-seed analysis ================")
    print("\n".join(f"  {c}" for c in changed) if changed else "  (none)")


if __name__ == "__main__":
    main()
