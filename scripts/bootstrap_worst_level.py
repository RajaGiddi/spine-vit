"""Bootstrap confidence intervals for worst-level accuracy and the two-factor
decomposition, resampling studies rather than levels.

Design, and why:

*Resampling happens within a seed.* Each seed reseeds the train/val/test split,
so the test sets differ - 27, 27, 33, 34, 27 studies with a pathological level
across seeds 42-46, not the constant 27 the draft states. Pooling studies across
seeds would mix different splits and break the pairing. Within a seed all five
configurations share one study set (verified), so resampling that set and
scoring every configuration on it keeps the comparison paired.

*Paired differences are bootstrapped directly*, not as a difference of two
separately bootstrapped accuracies, so the correlation between configurations
on the same studies is preserved.

*Each iteration averages across seeds*, matching how the point estimate is
formed. The interval therefore reflects study sampling; the seed-based standard
deviation reflects training-run variation. They answer different questions and
are reported side by side.

    python scripts/bootstrap_worst_level.py --iterations 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
LEVELS = ["L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"]
SEEDS = (42, 43, 44, 45, 46)

CONFIGS = {
    "sag-1slice": "rsna_fusion_sag_ordinal_256_2_aug_s{}",
    "sag-5slice": "rsna_fusion_sag_ordinal_256_2_sag5_aug_s{}",
    "axial": "rsna_fusion_axial_ordinal_256_2_aug_s{}",
    "fusion-concat": "rsna_fusion_both_concat_ordinal_256_2_aug_s{}",
    "fusion-attn": "rsna_fusion_both_attn_ordinal_256_2_aug_s{}",
}

# Decomposition from the write-up, plus the fusion comparisons it discusses.
COMPARISONS = {
    "total (axial - sag1)": ("axial", "sag-1slice"),
    "budget (sag5 - sag1)": ("sag-5slice", "sag-1slice"),
    "acquisition (axial - sag5)": ("axial", "sag-5slice"),
    "concat - axial": ("fusion-concat", "axial"),
    "attn - axial": ("fusion-attn", "axial"),
}

PATHOLOGY_THRESHOLD = 1

# train_fusion.py renamed this key from "preds" to "predictions" in the working
# tree, so the existing Table 1 files and any newly trained run disagree. Accept
# either rather than forcing a re-export of results that are already correct.
PRED_KEYS = ("preds", "predictions")


def prediction_array(pred: dict) -> np.ndarray:
    for key in PRED_KEYS:
        if key in pred:
            return np.asarray(pred[key])
    raise SystemExit(
        f"prediction file has none of {PRED_KEYS}; found {sorted(pred)}")



def load_predictions(experiments_dir, pattern, seed):
    path = Path(experiments_dir) / pattern.format(seed) / "test_predictions.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            "  Fetch it from the Modal volume:\n"
            f"    modal volume get spine-vit-outputs {pattern.format(seed)}"
            "/test_predictions.json <dest>")
    return json.loads(path.read_text())


def per_study_worst_level(pred: dict, threshold=PATHOLOGY_THRESHOLD) -> dict:
    """{study_id: 0/1} - did the model point at the right level?

    Mirrors utils.metrics.worst_level_accuracy, including that a tie for worst
    counts as correct for any tied level, and that studies with nothing
    pathological are skipped.
    """
    sid = np.asarray(pred["studyids"])
    lvl = np.asarray(pred["levels"])
    prd = prediction_array(pred)
    tgt = np.asarray(pred["targets"])

    out = {}
    for study in dict.fromkeys(sid.tolist()):
        mask = sid == study
        true, guess, levels = tgt[mask], prd[mask], lvl[mask]
        if true.max() < threshold:
            continue
        worst_true = {int(levels[i]) for i in range(len(levels)) if true[i] == true.max()}
        out[int(study)] = int(int(levels[int(np.argmax(guess))]) in worst_true)
    return out


def per_study_kappa_inputs(pred: dict) -> dict:
    """{study_id: (targets, preds)} over every level, for study-level resampling."""
    sid = np.asarray(pred["studyids"])
    prd = prediction_array(pred)
    tgt = np.asarray(pred["targets"])
    out = {}
    for study in dict.fromkeys(sid.tolist()):
        mask = sid == study
        out[int(study)] = (tgt[mask], prd[mask])
    return out


def kappa_from_studies(inputs: dict, studies) -> float:
    t = np.concatenate([inputs[s][0] for s in studies])
    p = np.concatenate([inputs[s][1] for s in studies])
    if len(np.unique(np.concatenate([t, p]))) < 2:
        return float("nan")
    return float(cohen_kappa_score(t, p, weights="quadratic"))


def worst_level_from_studies(flags: dict, studies) -> float:
    return float(np.mean([flags[s] for s in studies]))


def bootstrap(experiments_dir, iterations=10000, seed=0):
    rng = np.random.default_rng(seed)

    flags, kappas, study_lists = {}, {}, {}
    for name, pattern in CONFIGS.items():
        for s in SEEDS:
            pred = load_predictions(experiments_dir, pattern, s)
            flags[(name, s)] = per_study_worst_level(pred)
            kappas[(name, s)] = per_study_kappa_inputs(pred)

    for s in SEEDS:
        sets = {name: tuple(sorted(flags[(name, s)])) for name in CONFIGS}
        base = sets["sag-1slice"]
        if any(v != base for v in sets.values()):
            raise SystemExit(f"seed {s}: configurations disagree on the study set; "
                             "pairing would be invalid")
        study_lists[s] = np.array(base)

    # Draw resampled study indices once per iteration per seed, then score every
    # configuration on the same draw - that is what preserves the pairing.
    draws = {s: rng.integers(0, len(study_lists[s]), size=(iterations, len(study_lists[s])))
             for s in SEEDS}

    boot_acc = {name: np.empty(iterations) for name in CONFIGS}
    boot_kap = {name: np.empty(iterations) for name in CONFIGS}
    boot_diff = {label: np.empty(iterations) for label in COMPARISONS}
    boot_kdiff = {label: np.empty(iterations) for label in COMPARISONS}

    for b in range(iterations):
        acc_seed = {name: [] for name in CONFIGS}
        kap_seed = {name: [] for name in CONFIGS}
        for s in SEEDS:
            picked = study_lists[s][draws[s][b]]
            for name in CONFIGS:
                acc_seed[name].append(worst_level_from_studies(flags[(name, s)], picked))
                kap_seed[name].append(kappa_from_studies(kappas[(name, s)], picked))
        for name in CONFIGS:
            boot_acc[name][b] = np.mean(acc_seed[name])
            boot_kap[name][b] = np.nanmean(kap_seed[name])
        for label, (a, c) in COMPARISONS.items():
            boot_diff[label][b] = boot_acc[a][b] - boot_acc[c][b]
            boot_kdiff[label][b] = boot_kap[a][b] - boot_kap[c][b]

    point_acc, seed_sd_acc, point_kap, seed_sd_kap = {}, {}, {}, {}
    for name in CONFIGS:
        a = [worst_level_from_studies(flags[(name, s)], study_lists[s]) for s in SEEDS]
        k = [kappa_from_studies(kappas[(name, s)], study_lists[s]) for s in SEEDS]
        point_acc[name], seed_sd_acc[name] = float(np.mean(a)), float(np.std(a, ddof=1))
        point_kap[name], seed_sd_kap[name] = float(np.mean(k)), float(np.std(k, ddof=1))

    point_diff, seed_sd_diff = {}, {}
    for label, (a, c) in COMPARISONS.items():
        per = [worst_level_from_studies(flags[(a, s)], study_lists[s])
               - worst_level_from_studies(flags[(c, s)], study_lists[s]) for s in SEEDS]
        point_diff[label], seed_sd_diff[label] = float(np.mean(per)), float(np.std(per, ddof=1))

    return {
        "study_lists": study_lists, "flags": flags,
        "boot_acc": boot_acc, "boot_kap": boot_kap,
        "boot_diff": boot_diff, "boot_kdiff": boot_kdiff,
        "point_acc": point_acc, "seed_sd_acc": seed_sd_acc,
        "point_kap": point_kap, "seed_sd_kap": seed_sd_kap,
        "point_diff": point_diff, "seed_sd_diff": seed_sd_diff,
    }


def ci(samples, lo=2.5, hi=97.5):
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def seed_ci(mean, sd, n=len(SEEDS)):
    """Seed-based interval, t-distribution on n-1 df - what the draft's +/- implies."""
    from scipy.stats import t
    half = t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean - half, mean + half


def worst_level_distribution(experiments_dir):
    """Which level is actually the worst, per study, per seed.

    A metric over ~30 studies where one level dominates behaves differently
    from a balanced one, so the composition belongs in the paper.
    """
    rows = []
    for s in SEEDS:
        pred = load_predictions(experiments_dir, CONFIGS["sag-1slice"], s)
        sid = np.asarray(pred["studyids"])
        lvl = np.asarray(pred["levels"])
        tgt = np.asarray(pred["targets"])
        for study in dict.fromkeys(sid.tolist()):
            mask = sid == study
            true, levels = tgt[mask], lvl[mask]
            if true.max() < PATHOLOGY_THRESHOLD:
                continue
            worst = [int(levels[i]) for i in range(len(levels)) if true[i] == true.max()]
            rows.append({"seed": s, "study_id": int(study),
                         "n_tied": len(worst),
                         "levels": ",".join(LEVELS[w] for w in sorted(worst))})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments_dir", default="outputs_modal")
    parser.add_argument("--include_fixed", action="store_true",
                        help="add the fixed-slice axial control and the revised "
                             "decomposition that uses it in place of annotated axial")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/bootstrap_worst_level.json")
    args = parser.parse_args()

    if args.include_fixed:
        # Task 2: the control that removes expert slice selection. Adding it here
        # rather than in a separate script keeps one bootstrap draw across every
        # comparison, so the new rows are paired with the existing ones.
        CONFIGS["axial-fixedslice"] = "rsna_fusion_axial_ordinal_256_2_fixedslice_aug_s{}"
        COMPARISONS["selection (axial - axial-fixed)"] = ("axial", "axial-fixedslice")
        COMPARISONS["total, fixed (axial-fixed - sag1)"] = ("axial-fixedslice", "sag-1slice")
        COMPARISONS["acquisition, fixed (axial-fixed - sag5)"] = ("axial-fixedslice", "sag-5slice")

    r = bootstrap(args.experiments_dir, args.iterations, args.seed)

    print(f"Bootstrap over studies, {args.iterations} iterations, resampled within seed.")
    print("Test studies with a pathological level, per seed: "
          + ", ".join(f"s{s}={len(r['study_lists'][s])}" for s in SEEDS)
          + "   (the draft states a constant n=27)\n")

    out = {"iterations": args.iterations,
           "studies_per_seed": {str(s): int(len(r["study_lists"][s])) for s in SEEDS},
           "configs": {}, "comparisons": {}}

    print("Worst-level accuracy")
    print(f"  {'config':16s} {'point':>7s}  {'bootstrap 95% CI':>20s}  {'seed sd':>8s}"
          f"  {'seed 95% CI':>20s}  flag")
    for name in CONFIGS:
        pt, sd = r["point_acc"][name], r["seed_sd_acc"][name]
        blo, bhi = ci(r["boot_acc"][name])
        slo, shi = seed_ci(pt, sd)
        bw, sw = bhi - blo, shi - slo
        flag = "wider by seed" if sw > 1.5 * bw else ("wider by study" if bw > 1.5 * sw else "")
        print(f"  {name:16s} {pt:7.3f}  [{blo:+.3f}, {bhi:+.3f}]  {sd:8.3f}"
              f"  [{slo:+.3f}, {shi:+.3f}]  {flag}")
        out["configs"][name] = {
            "worst_level": {"point": pt, "bootstrap_ci": [blo, bhi], "seed_sd": sd,
                            "seed_ci": [slo, shi], "flag": flag},
            "kappa": {"point": r["point_kap"][name], "seed_sd": r["seed_sd_kap"][name],
                      "bootstrap_ci": list(ci(r["boot_kap"][name]))},
        }

    print("\nPaired differences (bootstrapped directly, so pairing is preserved)")
    print(f"  {'comparison':28s} {'point':>7s}  {'bootstrap 95% CI':>20s}  {'seed sd':>8s}"
          f"  {'seed 95% CI':>20s}  flag")
    for label in COMPARISONS:
        pt, sd = r["point_diff"][label], r["seed_sd_diff"][label]
        blo, bhi = ci(r["boot_diff"][label])
        slo, shi = seed_ci(pt, sd)
        bw, sw = bhi - blo, shi - slo
        flag = "wider by seed" if sw > 1.5 * bw else ("wider by study" if bw > 1.5 * sw else "")
        crosses = "  crosses 0" if blo <= 0 <= bhi else ""
        print(f"  {label:28s} {pt:+7.3f}  [{blo:+.3f}, {bhi:+.3f}]  {sd:8.3f}"
              f"  [{slo:+.3f}, {shi:+.3f}]  {flag}{crosses}")
        out["comparisons"][label] = {
            "worst_level": {"point": pt, "bootstrap_ci": [blo, bhi], "seed_sd": sd,
                            "seed_ci": [slo, shi], "flag": flag,
                            "crosses_zero": bool(blo <= 0 <= bhi)},
            "kappa": {"point": float(np.mean(r["boot_kdiff"][label])),
                      "bootstrap_ci": list(ci(r["boot_kdiff"][label]))},
        }

    dist = worst_level_distribution(args.experiments_dir)
    counts = (dist.levels.str.split(",").explode().value_counts()
              .reindex(LEVELS).fillna(0).astype(int))
    print(f"\nWhich level is worst, pooled over seeds ({len(dist)} study-seed pairs, "
          f"{int((dist.n_tied > 1).sum())} with a tie)")
    for lv in LEVELS:
        n = int(counts[lv])
        print(f"  {lv:7s} {n:4d}  {n / counts.sum():5.1%}  {'#' * int(40 * n / counts.max())}")
    out["worst_level_distribution"] = {lv: int(counts[lv]) for lv in LEVELS}
    out["n_ties"] = int((dist.n_tied > 1).sum())

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    dist.to_csv(path.with_name("worst_level_distribution.csv"), index=False)
    print(f"\nWrote {path}")
    print(f"Wrote {path.with_name('worst_level_distribution.csv')}")


if __name__ == "__main__":
    main()
