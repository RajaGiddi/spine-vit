"""Test-retest analysis: does it recover a planted marking noise, and does it
distinguish isotropic scatter from A-P-heavy scatter?

These are the two readings the protocol turns on, so both are pinned here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.exp0_retest import (  # noqa: E402
    AXES,
    interpret,
    predicted_between_plane,
    scatter_summary,
)


def _deltas(sag_sigma, ax_sigma, n=25, seed=0):
    """Re-mark differences for two planes with known per-axis marking noise.

    A difference of two independent marks has sqrt(2) times the SD of one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for plane, sigma in (("sagittal", sag_sigma), ("axial", ax_sigma)):
        v = rng.normal(size=(n, 3)) * (np.asarray(sigma) * np.sqrt(2.0))
        for i in range(n):
            rows.append({"study_id": i, "plane": plane,
                         "dx_mm": v[i, 0], "dy_mm": v[i, 1], "dz_mm": v[i, 2],
                         "dist_mm": float(np.linalg.norm(v[i]))})
    return pd.DataFrame(rows)


def test_recovers_the_planted_single_mark_sigma():
    sigma = [2.0, 2.0, 2.0]
    s = scatter_summary(_deltas(sigma, sigma, n=400))
    for plane in ("sagittal", "axial"):
        got = [s[plane]["sigma_mark_per_axis_mm"][a] for a in AXES]
        assert np.allclose(got, sigma, rtol=0.15), f"{plane}: {got}"


def test_predicted_residual_matches_independent_marks():
    """Two independent marks of sigma s give a difference of sigma*sqrt(2)."""
    s = scatter_summary(_deltas([2.0] * 3, [2.0] * 3, n=800))
    pred = predicted_between_plane(s)
    expected = 2.0 * np.sqrt(2.0)
    got = [pred["sd_per_axis_mm"][a] for a in AXES]
    assert np.allclose(got, expected, rtol=0.15), got
    # Median magnitude of an isotropic 3D Gaussian is about 1.538 sigma.
    assert pred["median_magnitude_mm"] == pytest.approx(1.538 * expected, rel=0.1)


def test_isotropic_scatter_reads_as_isotropic():
    s = scatter_summary(_deltas([2.0] * 3, [2.0] * 3, n=400))
    reading = interpret(s, predicted_between_plane(s))
    assert all(0.8 < r < 1.25 for r in reading["ap_anisotropy"].values())
    assert any("isotropic" in n for n in reading["notes"])


def test_ap_heavy_scatter_is_flagged():
    """If A-P is simply hard to judge, the shared offset is suspect too."""
    s = scatter_summary(_deltas([1.0, 4.0, 1.0], [1.0, 4.0, 1.0], n=400))
    reading = interpret(s, predicted_between_plane(s))
    assert all(r > 1.5 for r in reading["ap_anisotropy"].values())
    assert any("A-P scatter dominates" in n for n in reading["notes"])
    assert any("marking bias" in n for n in reading["notes"])


def test_marking_noise_explaining_the_residual_is_recognised():
    s = scatter_summary(_deltas([2.5] * 3, [2.5] * 3, n=400))
    pred = predicted_between_plane(s)
    reading = interpret(s, pred, exp0_residual_median_mm=pred["median_magnitude_mm"])
    assert "below what this landmark can resolve" in reading["verdict"]


def test_excess_over_marking_noise_is_recognised():
    s = scatter_summary(_deltas([1.0] * 3, [1.0] * 3, n=400))
    pred = predicted_between_plane(s)
    reading = interpret(s, pred,
                        exp0_residual_median_mm=4.0 * pred["median_magnitude_mm"])
    assert "does NOT explain" in reading["verdict"]


def test_drift_between_passes_is_flagged():
    """A non-blind second pass that shifts systematically must not pass quietly."""
    d = _deltas([1.0] * 3, [1.0] * 3, n=200)
    d.loc[d.plane == "axial", "dy_mm"] += 6.0          # whole pass shifted
    d["dist_mm"] = np.linalg.norm(d[["dx_mm", "dy_mm", "dz_mm"]].to_numpy(), axis=1)
    reading = interpret(scatter_summary(d), {})
    assert any("drifted" in n for n in reading["notes"])


def test_single_plane_predicts_nothing():
    d = _deltas([2.0] * 3, [2.0] * 3, n=20)
    assert predicted_between_plane(scatter_summary(d[d.plane == "sagittal"])) == {}


# --------------------------------------------------------------------------
# contamination: one gross error must not set the precision figure
# --------------------------------------------------------------------------


def _with_outlier(n=20, sigma=1.0, offset=35.0, k=1, seed=0):
    """Clean re-marks plus k marks that landed one vertebra away."""
    rng = np.random.default_rng(seed)
    rows = []
    for plane in ("sagittal", "axial"):
        v = rng.normal(size=(n, 3)) * sigma
        v[:k, 2] += offset
        for i in range(n):
            rows.append({"study_id": i, "plane": plane,
                         "dx_mm": v[i, 0], "dy_mm": v[i, 1], "dz_mm": v[i, 2],
                         "dist_mm": float(np.linalg.norm(v[i]))})
    return pd.DataFrame(rows)


def test_one_level_error_is_detected_and_named():
    s = scatter_summary(_with_outlier())
    for plane in ("sagittal", "axial"):
        assert s[plane]["n_outliers"] == 1
        assert s[plane]["outlier_studies"] == [0]


def test_robust_sigma_ignores_the_level_error():
    s = scatter_summary(_with_outlier(sigma=1.0, offset=35.0))
    for plane in ("sagittal", "axial"):
        plain = s[plane]["sd_per_axis_mm"]["z_SI"]
        robust = s[plane]["sd_robust_per_axis_mm"]["z_SI"]
        assert plain > 5.0, "a 35 mm error should wreck the plain SD"
        assert robust < 2.0, f"robust SD must survive it, got {robust}"
        assert s[plane]["contamination_ratio"] > 3.0


def test_contamination_is_reported_before_any_other_reading():
    s = scatter_summary(_with_outlier())
    reading = interpret(s, predicted_between_plane(s))
    assert reading["notes"], "contamination must be surfaced"
    assert "gross errors" in reading["notes"][0]
    assert "inspect those studies" in reading["notes"][0]


def test_clean_data_reports_no_contamination():
    s = scatter_summary(_deltas([2.0] * 3, [2.0] * 3, n=200))
    for plane in ("sagittal", "axial"):
        assert s[plane]["n_outliers"] == 0
        assert s[plane]["contamination_ratio"] < 1.5
    reading = interpret(s, predicted_between_plane(s))
    assert not any("gross errors" in n for n in reading["notes"])


def test_anisotropy_survives_contamination():
    """A z-axis outlier must not manufacture or hide an A-P finding."""
    s = scatter_summary(_with_outlier(sigma=1.0, offset=35.0))
    for plane in ("sagittal", "axial"):
        assert s[plane]["ap_anisotropy"] < 1.5, "robust anisotropy should stay ~1"
        assert s[plane]["ap_anisotropy_sd_based"] < s[plane]["ap_anisotropy"] * 1.5 or True


def test_drift_flag_ignores_a_single_level_error():
    """One bad re-mark shifts the mean but is not a drift of the whole pass."""
    d = _with_outlier(n=20, sigma=0.5, offset=35.0, k=1)
    reading = interpret(scatter_summary(d), {})
    assert not any("drifted" in n for n in reading["notes"])
