"""Experiment 0: does the patient move between the sagittal and axial acquisition?

For each study, a vertebral body centre is marked once in each acquisition. The
intensity-weighted centroid of a sphere around each mark is computed in that
acquisition's own voxels, converted to patient coordinates, and the two are
compared. If the discrepancy is a large fraction of the slice thickness, then
sagittal and axial are not describing the same anatomy in the same place, and
no amount of care in the composition model will make them agree.

Landmarks come from a CSV (see scripts/mark_landmarks.py):

    study_id,plane,series_id,instance_number,row,col
    109677683,sagittal,714837857,11,418,352
    109677683,axial,107963340,7,160,160

Usage:
    python experiments/exp0_motion.py --landmarks data/rsna/landmarks.csv
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

REQUIRED_COLUMNS = {"study_id", "plane", "series_id", "instance_number", "row", "col"}


def slice_index_for_instance(geometry, instance_number: int) -> int:
    for k, s in enumerate(geometry):
        if s.instance_number == int(instance_number):
            return k
    available = sorted(s.instance_number for s in geometry)
    raise KeyError(f"instance {instance_number} not in series (have {available})")


def landmark_centroid(volume: Volume, instance_number: int, row: int, col: int,
                      radius_mm: float, floor_at_zero: bool = True) -> dict:
    """Intensity-weighted centroid of a sphere seeded at a marked voxel."""
    k = slice_index_for_instance(volume.geometry, instance_number)
    seed = volume.geometry[k].voxel_to_patient(int(row), int(col))

    sample = volume.sample_sphere(seed, radius_mm)
    if len(sample) == 0:
        raise ValueError(f"no voxels within {radius_mm} mm of the mark")

    centroid = sample.weighted_centroid(floor_at_zero=floor_at_zero)
    return {
        "seed_mm": seed,
        "centroid_mm": centroid,
        "n_voxels": len(sample),
        "n_slices": sample.n_slices,
        "shift_from_seed_mm": float(np.linalg.norm(centroid - seed)),
        "thickness_mm": volume.geometry[k].thickness,
        "plane": volume.plane,
    }


def measure_study(sag_dir, ax_dir, sag_mark, ax_mark, radius_mm=8.0,
                  floor_at_zero=True) -> dict:
    sag = Volume.from_dir(sag_dir)
    axi = Volume.from_dir(ax_dir)

    s = landmark_centroid(sag, sag_mark["instance_number"], sag_mark["row"],
                          sag_mark["col"], radius_mm, floor_at_zero)
    a = landmark_centroid(axi, ax_mark["instance_number"], ax_mark["row"],
                          ax_mark["col"], radius_mm, floor_at_zero)

    delta = s["centroid_mm"] - a["centroid_mm"]
    discrepancy = float(np.linalg.norm(delta))
    thickest = max(s["thickness_mm"], a["thickness_mm"])

    return {
        "discrepancy_mm": discrepancy,
        "dx_mm": float(delta[0]),
        "dy_mm": float(delta[1]),
        "dz_mm": float(delta[2]),
        "sag_thickness_mm": s["thickness_mm"],
        "ax_thickness_mm": a["thickness_mm"],
        "ratio_to_thickest": discrepancy / thickest,
        "sag_centroid_mm": s["centroid_mm"].tolist(),
        "ax_centroid_mm": a["centroid_mm"].tolist(),
        "sag_n_voxels": s["n_voxels"],
        "ax_n_voxels": a["n_voxels"],
        "sag_n_slices": s["n_slices"],
        "ax_n_slices": a["n_slices"],
        "sag_shift_from_seed_mm": s["shift_from_seed_mm"],
        "ax_shift_from_seed_mm": a["shift_from_seed_mm"],
    }


def direction_consistency(vectors: np.ndarray) -> dict:
    """Are the discrepancy vectors pointing the same way?

    Uses the Rayleigh test for a preferred direction on the sphere: under
    isotropy 3*n*Rbar^2 is chi-squared with 3 degrees of freedom. An earlier
    version compared Rbar against a 3/sqrt(n) rule of thumb, which is far too
    conservative - it called Rbar=0.56 at n=25 "random" when the Rayleigh test
    puts it at p=3e-5.

    Alignment matters because patient motion is random per patient. A shared
    direction across subjects is a bias in the landmark or the geometry, and
    must be separated out before anything is called motion.
    """
    from scipy.stats import chi2

    norms = np.linalg.norm(vectors, axis=1)
    usable = norms > 1e-9
    if usable.sum() < 2:
        return {"n": int(usable.sum()), "resultant_length": float("nan")}

    units = vectors[usable] / norms[usable, None]
    mean_vec = units.mean(axis=0)
    resultant = float(np.linalg.norm(mean_vec))
    n = int(usable.sum())
    stat = 3.0 * n * resultant**2
    p = float(chi2.sf(stat, 3))
    return {
        "n": n,
        "resultant_length": resultant,
        "mean_direction": (mean_vec / resultant).tolist() if resultant > 1e-12 else None,
        "isotropic_expectation": float(1.0 / np.sqrt(n)),
        "rayleigh_stat": float(stat),
        "rayleigh_p": p,
        "looks_systematic": bool(p < 0.01),
    }


def decompose(vectors: np.ndarray) -> dict:
    """Split the discrepancy into a shared offset and per-study scatter.

    Exp 0 asks whether the patient moved. A component common to every study
    cannot be motion - it is a bias in how the landmark is identified in each
    plane, or in the geometry. The motion estimate is what remains after that
    offset is removed, so the decision rule belongs on the residual.

    The offset is the component-wise median, which shrugs off a bad mark.
    """
    systematic = np.median(vectors, axis=0)
    residual = vectors - systematic
    raw_mag = np.linalg.norm(vectors, axis=1)
    res_mag = np.linalg.norm(residual, axis=1)
    q1, q3 = np.percentile(res_mag, [25, 75])
    return {
        "systematic_mm": systematic.tolist(),
        "systematic_norm_mm": float(np.linalg.norm(systematic)),
        "residual_median_mm": float(np.median(res_mag)),
        "residual_iqr_mm": [float(q1), float(q3)],
        "residual_max_mm": float(res_mag.max()),
        "raw_median_mm": float(np.median(raw_mag)),
        "explained_fraction": float(
            1.0 - np.median(res_mag) / max(np.median(raw_mag), 1e-9)
        ),
        "residual_direction": direction_consistency(residual),
    }


def summarise(table: pd.DataFrame) -> dict:
    d = table["discrepancy_mm"].to_numpy()
    r = table["ratio_to_thickest"].to_numpy()
    vectors = table[["dx_mm", "dy_mm", "dz_mm"]].to_numpy()

    q1, q3 = np.percentile(d, [25, 75])
    rq1, rq3 = np.percentile(r, [25, 75])
    return {
        "n_studies": int(len(table)),
        "discrepancy_mm": {
            "median": float(np.median(d)), "iqr": [float(q1), float(q3)],
            "min": float(d.min()), "max": float(d.max()),
        },
        "ratio_to_thickest": {
            "median": float(np.median(r)), "iqr": [float(rq1), float(rq3)],
            "min": float(r.min()), "max": float(r.max()),
        },
        "per_axis_median_mm": {
            "x_LR": float(np.median(vectors[:, 0])),
            "y_AP": float(np.median(vectors[:, 1])),
            "z_SI": float(np.median(vectors[:, 2])),
        },
        "direction": direction_consistency(vectors),
        "decomposition": decompose(vectors),
        "residual_ratio_to_thickest": _residual_ratio(table, vectors),
    }


def _residual_ratio(table: pd.DataFrame, vectors: np.ndarray) -> dict:
    """Residual discrepancy as a fraction of the larger slice thickness."""
    thickest = np.maximum(table["sag_thickness_mm"].to_numpy(),
                          table["ax_thickness_mm"].to_numpy())
    res = np.linalg.norm(vectors - np.median(vectors, axis=0), axis=1) / thickest
    q1, q3 = np.percentile(res, [25, 75])
    return {"median": float(np.median(res)), "iqr": [float(q1), float(q3)],
            "max": float(res.max())}


def print_summary(summary: dict) -> None:
    d, r, ax, dirn = (summary["discrepancy_mm"], summary["ratio_to_thickest"],
                      summary["per_axis_median_mm"], summary["direction"])
    print(f"\nExperiment 0 - landmark discrepancy across {summary['n_studies']} studies")
    print(f"  discrepancy      median {d['median']:.2f} mm   "
          f"IQR [{d['iqr'][0]:.2f}, {d['iqr'][1]:.2f}]   range [{d['min']:.2f}, {d['max']:.2f}]")
    print(f"  / thickest slice median {r['median']:.2f}      "
          f"IQR [{r['iqr'][0]:.2f}, {r['iqr'][1]:.2f}]   range [{r['min']:.2f}, {r['max']:.2f}]")
    print(f"  per-axis median  x(L+) {ax['x_LR']:+.2f}  y(P+) {ax['y_AP']:+.2f}  "
          f"z(S+) {ax['z_SI']:+.2f} mm")

    if np.isnan(dirn.get("resultant_length", float("nan"))):
        print("  direction        too few usable vectors to assess")
        return
    verdict = "SYSTEMATIC" if dirn["looks_systematic"] else "no consistent direction"
    print(f"  direction        resultant {dirn['resultant_length']:.2f} "
          f"(isotropic ~{dirn['isotropic_expectation']:.2f}), Rayleigh "
          f"p={dirn['rayleigh_p']:.1e} -> {verdict}")
    if dirn.get("mean_direction"):
        m = dirn["mean_direction"]
        print(f"                   mean unit vector [{m[0]:+.2f}, {m[1]:+.2f}, {m[2]:+.2f}]")

    dec = summary.get("decomposition")
    if not dec:
        return
    sysv = dec["systematic_mm"]
    print(f"\n  Shared offset (cannot be motion - it is the same in every study)")
    print(f"    vector         x {sysv[0]:+.2f}   y {sysv[1]:+.2f}   z {sysv[2]:+.2f} mm"
          f"   |{dec['systematic_norm_mm']:.2f} mm|")
    print(f"    accounts for   {dec['explained_fraction']:.0%} of the raw median")
    rr = summary["residual_ratio_to_thickest"]
    print(f"\n  Residual after removing it - THIS is the motion estimate")
    print(f"    discrepancy    median {dec['residual_median_mm']:.2f} mm   "
          f"IQR [{dec['residual_iqr_mm'][0]:.2f}, {dec['residual_iqr_mm'][1]:.2f}]   "
          f"max {dec['residual_max_mm']:.2f}")
    print(f"    / thickest     median {rr['median']:.2f}   "
          f"IQR [{rr['iqr'][0]:.2f}, {rr['iqr'][1]:.2f}]")
    rd = dec["residual_direction"]
    if not np.isnan(rd.get("resultant_length", float("nan"))):
        left = "still SYSTEMATIC" if rd["looks_systematic"] else "isotropic, consistent with motion"
        print(f"    direction      Rayleigh p={rd['rayleigh_p']:.2f} -> {left}")


def load_landmarks(path: Path) -> pd.DataFrame:
    marks = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(marks.columns)
    if missing:
        raise SystemExit(f"{path}: landmark CSV is missing columns {sorted(missing)}")
    marks["plane"] = marks["plane"].str.lower().str.strip()
    bad = set(marks["plane"]) - {"sagittal", "axial"}
    if bad:
        raise SystemExit(f"{path}: unexpected plane values {sorted(bad)}")
    return marks


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--landmarks", required=True, help="CSV of marked vertebral body centres")
    parser.add_argument("--data_dir", default="data/rsna")
    parser.add_argument("--out", default="results/exp0_motion.csv")
    parser.add_argument("--radius_mm", type=float, default=8.0,
                        help="sphere radius around each mark; must stay inside the body")
    parser.add_argument("--raw_weights", action="store_true",
                        help="do not clip negative intensities before weighting")
    args = parser.parse_args()

    images = Path(args.data_dir) / "train_images"
    marks = load_landmarks(Path(args.landmarks))

    rows, skipped = [], []
    for study_id, group in marks.groupby("study_id"):
        planes = {p: g.iloc[0] for p, g in group.groupby("plane")}
        if not {"sagittal", "axial"} <= planes.keys():
            skipped.append((study_id, "needs both a sagittal and an axial mark"))
            continue

        sag_mark, ax_mark = planes["sagittal"], planes["axial"]
        sag_dir = images / str(study_id) / str(int(sag_mark.series_id))
        ax_dir = images / str(study_id) / str(int(ax_mark.series_id))

        try:
            result = measure_study(
                sag_dir, ax_dir,
                {"instance_number": sag_mark.instance_number, "row": sag_mark.row,
                 "col": sag_mark.col},
                {"instance_number": ax_mark.instance_number, "row": ax_mark.row,
                 "col": ax_mark.col},
                radius_mm=args.radius_mm,
                floor_at_zero=not args.raw_weights,
            )
        except (KeyError, ValueError, OSError) as err:
            skipped.append((study_id, str(err)))
            continue

        result["study_id"] = int(study_id)
        rows.append(result)
        print(f"  {study_id}: {result['discrepancy_mm']:6.2f} mm  "
              f"({result['ratio_to_thickest']:.2f} x thickest slice)")

    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for study_id, reason in skipped:
            print(f"  {study_id}: {reason}")

    if not rows:
        raise SystemExit("No studies measured.")

    table = pd.DataFrame(rows)
    lead = ["study_id", "discrepancy_mm", "sag_thickness_mm", "ax_thickness_mm",
            "ratio_to_thickest", "dx_mm", "dy_mm", "dz_mm"]
    table = table[lead + [c for c in table.columns if c not in lead]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    summary = summarise(table)
    summary["radius_mm"] = args.radius_mm
    print_summary(summary)

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
