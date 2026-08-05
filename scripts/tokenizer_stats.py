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
        for deltas in sorted(glob.glob(os.path.join(experiments_dir, f"{pre}_s*"))):
            tr_p, cf_p = os.path.join(deltas, "test_results.json"), os.path.join(deltas, "config.json")
            if not (os.path.exists(tr_p) and os.path.exists(cf_p) and os.path.getsize(tr_p) > 0):
                continue
            config, test_results = json.load(open(cf_p)), json.load(open(tr_p))
            metrics = test_results.get("metrics_full") or test_results.get("metrics") or {}
            att = test_results.get("attribution_full") or test_results.get("attribution") or {}
            runs[int(config["seed"])] = {"kappa": metrics.get("kappa"), "worst": att.get("worst_level_accuracy"),
                                     "macro_f1": metrics.get("macro_f1"), "best_epoch": test_results.get("best_epoch"),
                                     "n_path_studies": att.get("n_studies_with_pathology"), "dir": deltas}
        out[name] = runs
    return out


def paired(runs, first_name, second_name, seeds, metric):
    A = np.array([runs[first_name][seed][metric] for seed in seeds], dtype=float)
    B = np.array([runs[second_name][seed][metric] for seed in seeds], dtype=float)
    deltas = A - B
    count = len(deltas)
    sd = deltas.std(ddof=1)
    se = sd / np.sqrt(count)
    tc = stats.t.ppf(0.975, count - 1)
    lo, hi = deltas.mean() - tc * se, deltas.mean() + tc * se
    t, p = stats.ttest_rel(A, B)
    return {"A": A, "B": B, "d": deltas, "n": count, "mean": deltas.mean(), "sd": sd, "se": se,
            "lo": lo, "hi": hi, "t": t, "p": p,
            "sep": bool(p < 0.05 and (lo > 0 or hi < 0))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_dir", default="outputs_modal")
    args = parser.parse_args()
    runs = load(args.experiments_dir)

    print("=== seeds present ===")
    for count, result in runs.items():
        print(f"  {count:<12} {sorted(result)}  (n={len(result)})")

    print("\n=== per-tokenizer aggregates over all seeds present ===")
    print(f"{'tokenizer':<12}{'n':>3}  " + "".join(f"{label:>28}" for _, label in METRICS))
    for name in CFG:
        seeds = sorted(runs[name])
        cells = []
        for key, _ in METRICS:
            values = np.array([runs[name][seed][key] for seed in seeds], dtype=float)
            cells.append(f"{values.mean():.4f} +/- {values.std():.4f} (sd1 {values.std(ddof=1):.4f})")
        print(f"{name:<12}{len(seeds):>3}  " + "".join(f"{c:>28}" for c in cells))

    print("\n=== per-seed values ===")
    for key, label in METRICS:
        print(f"  -- {label}")
        allseeds = sorted({seed for result in runs.values() for seed in result})
        print(f"     {'tokenizer':<12}" + "".join(f"{'s'+str(seed):>10}" for seed in allseeds))
        for name in CFG:
            row = "".join(f"{runs[name][seed][key]:10.4f}" if seed in runs[name] else f"{'-':>10}"
                          for seed in allseeds)
            print(f"     {name:<12}{row}")

    changed = []
    for first_name, second_name in COMPARISONS:
        shared = sorted(set(runs[first_name]) & set(runs[second_name]))
        print(f"\n================ {first_name} vs {second_name} ================")
        print(f"  seed sets identical: {set(runs[first_name]) == set(runs[second_name])}   shared: {shared}")
        for key, label in METRICS[:2]:
            sets = [("3-seed", BASE_SEEDS)] if len(shared) <= 3 else \
                   [("3-seed", BASE_SEEDS), (f"{len(shared)}-seed", shared)]
            verdicts = {}
            for tag, seeds in sets:
                if not set(seeds) <= set(shared):
                    continue
                result = paired(runs, first_name, second_name, seeds, key)
                verdicts[tag] = result["sep"]
                print(f"\n  --- {label} | {tag} (n={result['n']}) ---")
                for seed, av, bv, dv in zip(seeds, result["A"], result["B"], result["d"]):
                    print(f"      s{seed}: {first_name} {av:.6f}  {second_name} {bv:.6f}   delta {dv:+.6f}")
                print(f"      mean {result['mean']:+.6f}  SD(deltas) {result['sd']:.6f}  SE {result['se']:.6f}")
                print(f"      95% CI [{result['lo']:+.6f}, {result['hi']:+.6f}]   t = {result['t']:.4f}   p = {result['p']:.4f}")
                print(f"      signs {int((result['d'] > 0).sum())}/{result['n']} positive   -> "
                      f"{'SEPARABLE' if result['sep'] else 'not separable'}")
            if len(verdicts) == 2:
                tags = list(verdicts)
                if verdicts[tags[0]] != verdicts[tags[1]]:
                    changed.append(f"{first_name} vs {second_name} [{label}]: {tags[0]} "
                                   f"{'SEPARABLE' if verdicts[tags[0]] else 'not separable'} -> {tags[1]} "
                                   f"{'SEPARABLE' if verdicts[tags[1]] else 'not separable'}")

    print("\n\n================ VERDICT CHANGES vs the 3-seed analysis ================")
    print("\n".join(f"  {c}" for c in changed) if changed else "  (none)")


if __name__ == "__main__":
    main()
