"""Test-retest: how reliably can the landmark be found at all?

Experiment 0 reported a 6.49 mm median discrepancy between the sagittal and
axial marks, decomposing into a ~4 mm shared anterior offset and ~5.3 mm of
scatter. That decomposition was fitted to two summary statistics, not measured.
This measures it.

A second, blind marking pass gives the repeatability of the landmark itself.
Because the Exp 0 residual and the retest difference are each the difference of
two independent marks, they are directly comparable - if marking noise is the
whole story, the retest scatter reproduces the Exp 0 residual with no fitting.

Per-axis scatter is the point, not just the magnitude:

  scatter isotropic, offset still A-P  -> the offset is a real difference in how
                                          the landmark reads in profile versus
                                          cross-section
  scatter itself A-P heavy             -> A-P is simply the hard axis to judge,
                                          and the "offset" is marking bias too

Usage:
    python experiments/exp0_retest.py --first landmarks.csv \
        --second landmarks_retest.csv --data_dir data/rsna
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composition.volume import Volume  # noqa: E402
from experiments.exp0_motion import direction_consistency, landmark_centroid  # noqa: E402

AXES = ("x_LR", "y_AP", "z_SI")


def mark_deltas(first: pd.DataFrame, second: pd.DataFrame, images_dir,
                radius_mm: float = 8.0) -> pd.DataFrame:
    """Centroid difference between two marking passes, per study and plane."""
    images_dir = Path(images_dir)
    key = ["study_id", "plane"]
    a = first.set_index(key)
    b = second.set_index(key)
    shared = a.index.intersection(b.index)

    rows = []
    for idx in shared:
        study_id, plane = idx
        ra, rb = a.loc[idx], b.loc[idx]
        if int(ra.series_id) != int(rb.series_id):
            continue  # a different series is not a re-mark of the same thing
        volume = Volume.from_dir(images_dir / str(int(study_id)) / str(int(ra.series_id)))
        try:
            ca = landmark_centroid(volume, ra.instance_number, ra.row, ra.col, radius_mm)
            cb = landmark_centroid(volume, rb.instance_number, rb.row, rb.col, radius_mm)
        except (KeyError, ValueError):
            continue
        d = cb["centroid_mm"] - ca["centroid_mm"]
        rows.append({"study_id": int(study_id), "plane": plane,
                     "dx_mm": d[0], "dy_mm": d[1], "dz_mm": d[2],
                     "dist_mm": float(np.linalg.norm(d))})
    return pd.DataFrame(rows)


MAD_TO_SIGMA = 1.4826  # MAD -> sigma for a Gaussian
OUTLIER_MAD = 4.0      # robust z beyond which a re-mark is a gross error


def robust_sigma(v: np.ndarray) -> np.ndarray:
    """Per-axis SD estimated from the median absolute deviation.

    A single re-mark landing one vertebra away (~35 mm) sets the plain SD at
    n=20 to about 7.8 mm on its own. The SD then describes that one mistake
    rather than the repeatability of the landmark, so it cannot be the basis
    for a precision claim.
    """
    mad = np.median(np.abs(v - np.median(v, axis=0)), axis=0)
    return MAD_TO_SIGMA * mad


def scatter_summary(deltas: pd.DataFrame) -> dict:
    """Per-plane, per-axis repeatability, robust and non-robust side by side.

    `sigma_mark` is the noise on a single mark: a difference of two independent
    marks has sqrt(2) times the SD of one, so divide through.
    """
    out = {}
    for plane, grp in deltas.groupby("plane"):
        v = grp[["dx_mm", "dy_mm", "dz_mm"]].to_numpy()
        if len(v) < 2:
            continue
        sd = v.std(axis=0, ddof=1)
        rsd = robust_sigma(v)

        # Flag re-marks that are gross errors on any axis, then re-measure
        # without them so the scatter describes the landmark, not the mistakes.
        centre, scale = np.median(v, axis=0), np.maximum(rsd, 1e-6)
        z = np.abs(v - centre) / scale
        bad = np.any(z > OUTLIER_MAD, axis=1)
        clean = v[~bad]
        sd_clean = clean.std(axis=0, ddof=1) if len(clean) > 2 else sd

        use = rsd if np.all(rsd > 1e-6) else sd
        others = [use[0], use[2]]
        out[plane] = {
            "n": int(len(v)),
            "n_outliers": int(bad.sum()),
            "outlier_studies": grp.study_id.to_numpy()[bad].tolist(),
            "median_dist_mm": float(np.median(grp.dist_mm)),
            "sd_per_axis_mm": dict(zip(AXES, sd.round(3).tolist())),
            "sd_robust_per_axis_mm": dict(zip(AXES, rsd.round(3).tolist())),
            "sd_excl_outliers_mm": dict(zip(AXES, sd_clean.round(3).tolist())),
            "sigma_mark_per_axis_mm": dict(zip(AXES, (use / np.sqrt(2.0)).round(3).tolist())),
            "sigma_mark_sd_based_mm": dict(zip(AXES, (sd / np.sqrt(2.0)).round(3).tolist())),
            "contamination_ratio": float(np.max(sd / np.maximum(rsd, 1e-6))),
            "mean_per_axis_mm": dict(zip(AXES, v.mean(axis=0).round(3).tolist())),
            "median_per_axis_mm": dict(zip(AXES, np.median(v, axis=0).round(3).tolist())),
            "ap_anisotropy": float(use[1] / max(np.mean(others), 1e-9)),
            "ap_anisotropy_sd_based": float(sd[1] / max(np.mean([sd[0], sd[2]]), 1e-9)),
            "direction": direction_consistency(v),
        }
    return out


def predicted_between_plane(summary: dict) -> dict:
    """What Exp 0's residual would be if marking noise were the only cause.

    The sagittal and axial marks are independent, so their per-axis variances
    add. No fitting: this is a prediction the Exp 0 residual either matches or
    does not.
    """
    if not {"sagittal", "axial"} <= summary.keys():
        return {}
    s = np.array([summary["sagittal"]["sigma_mark_per_axis_mm"][a] for a in AXES])
    x = np.array([summary["axial"]["sigma_mark_per_axis_mm"][a] for a in AXES])
    per_axis = np.sqrt(s**2 + x**2)
    # Median magnitude of a zero-mean Gaussian with these per-axis SDs.
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(20000, 3)) * per_axis
    return {
        "sd_per_axis_mm": dict(zip(AXES, per_axis.round(3).tolist())),
        "median_magnitude_mm": float(np.median(np.linalg.norm(draws, axis=1))),
    }


def interpret(summary: dict, predicted: dict, exp0_residual_median_mm=None,
              exp0_offset_mm=None) -> dict:
    """Turn the numbers into the two readings the protocol cares about."""
    notes = []
    aniso = {p: v["ap_anisotropy"] for p, v in summary.items()}
    ap_heavy = [p for p, r in aniso.items() if r > 1.5]

    # Contamination first: every reading below is meaningless if a handful of
    # gross errors are setting the scale.
    for plane, v in summary.items():
        if v.get("n_outliers"):
            notes.append(
                f"{plane}: {v['n_outliers']} of {v['n']} re-marks are gross errors "
                f"(studies {v['outlier_studies']}). Plain SD is "
                f"{v['contamination_ratio']:.1f}x the robust estimate, so the SD "
                "describes those mistakes, not the landmark. Robust figures used "
                "throughout; inspect those studies before trusting any of this.")
        elif v.get("contamination_ratio", 1.0) > 1.5:
            notes.append(
                f"{plane}: SD is {v['contamination_ratio']:.1f}x the robust estimate "
                "with no single re-mark flagged - the tail is heavy. Treat the "
                "precision figure as approximate.")

    if ap_heavy:
        notes.append(
            f"A-P scatter dominates in {', '.join(ap_heavy)} (ratio "
            f"{', '.join(f'{aniso[p]:.2f}' for p in ap_heavy)}). A-P is the hard "
            "axis to judge, so the shared offset is plausibly marking bias too, "
            "not a landmark-definition difference.")
    elif aniso:
        notes.append(
            "Scatter is close to isotropic (A-P ratio "
            f"{', '.join(f'{p}={r:.2f}' for p, r in aniso.items())}). If the Exp 0 "
            "offset remains A-P, it is a real difference between the views rather "
            "than an artefact of how the mark is placed.")

    verdict = None
    if exp0_residual_median_mm is not None and predicted:
        pred = predicted["median_magnitude_mm"]
        ratio = exp0_residual_median_mm / max(pred, 1e-9)
        if ratio < 1.3:
            verdict = ("marking noise explains the Exp 0 residual; motion is below "
                       "what this landmark can resolve")
        elif ratio < 2.0:
            verdict = ("marking noise explains most of the Exp 0 residual; any motion "
                       "is comparable to or smaller than the measurement precision")
        else:
            verdict = ("marking noise does NOT explain the Exp 0 residual; the excess "
                       "is a real between-acquisition difference")
        notes.append(f"observed residual {exp0_residual_median_mm:.2f} mm vs predicted "
                     f"{pred:.2f} mm from marking noise alone (ratio {ratio:.2f})")

    for plane, v in summary.items():
        # Median, not mean: a single level error drags the mean and would be
        # reported as a drift of the whole pass.
        drift = np.array([v["median_per_axis_mm"][a] for a in AXES])
        if np.linalg.norm(drift) > 0.5 * v["median_dist_mm"]:
            notes.append(f"{plane}: mean difference {drift.round(2).tolist()} mm is "
                         "large relative to the scatter - the second pass drifted, "
                         "which blind re-marking should have prevented.")
    return {"verdict": verdict, "notes": notes, "ap_anisotropy": aniso}


def print_report(deltas, summary, predicted, reading) -> None:
    print(f"\nTest-retest across {len(deltas)} (study, plane) re-marks")
    for plane, v in summary.items():
        sd, rsd, sm = (v["sd_per_axis_mm"], v["sd_robust_per_axis_mm"],
                       v["sigma_mark_per_axis_mm"])
        flag = f"  <-- {v['n_outliers']} gross error(s)" if v["n_outliers"] else ""
        print(f"\n  {plane}  (n={v['n']}){flag}")
        print(f"    median re-mark distance   {v['median_dist_mm']:.2f} mm")
        print("    SD per axis  (plain)      " +
              "  ".join(f"{a} {sd[a]:5.2f}" for a in AXES))
        print("    SD per axis  (robust)     " +
              "  ".join(f"{a} {rsd[a]:5.2f}" for a in AXES))
        if v["n_outliers"]:
            ex = v["sd_excl_outliers_mm"]
            print("    SD excluding outliers     " +
                  "  ".join(f"{a} {ex[a]:5.2f}" for a in AXES))
            print(f"    outlier studies           {v['outlier_studies']}")
        print("    sigma per mark (robust)   " +
              "  ".join(f"{a} {sm[a]:5.2f}" for a in AXES))
        print(f"    A-P anisotropy            {v['ap_anisotropy']:.2f} robust"
              f"  /  {v['ap_anisotropy_sd_based']:.2f} plain"
              "   (>1.5 means A-P is the hard axis)")
        print("    median difference (drift) " +
              "  ".join(f"{a} {v['median_per_axis_mm'][a]:+5.2f}" for a in AXES))

    if predicted:
        p = predicted["sd_per_axis_mm"]
        print("\n  Predicted Exp 0 residual from marking noise alone")
        print("    SD per axis               " +
              "  ".join(f"{a} {p[a]:5.2f}" for a in AXES))
        print(f"    median magnitude          {predicted['median_magnitude_mm']:.2f} mm")

    if reading.get("verdict"):
        print(f"\n  => {reading['verdict']}")
    for n in reading["notes"]:
        print(f"     - {n}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--out", default="results/exp0_retest.json")
    parser.add_argument("--radius_mm", type=float, default=8.0)
    parser.add_argument("--exp0_residual_mm", type=float, default=None,
                        help="residual median from the Exp 0 summary, for comparison")
    args = parser.parse_args()

    images = Path(args.data_dir) / "train_images"
    deltas = mark_deltas(pd.read_csv(args.first), pd.read_csv(args.second),
                         images, args.radius_mm)
    if deltas.empty:
        raise SystemExit("no (study, plane) pairs present in both passes")

    summary = scatter_summary(deltas)
    predicted = predicted_between_plane(summary)
    reading = interpret(summary, predicted, args.exp0_residual_mm)
    print_report(deltas, summary, predicted, reading)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "predicted": predicted,
                               "reading": reading}, indent=2))
    deltas.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nWrote {out}\nWrote {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
