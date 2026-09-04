"""Rule-based axial slice selection.

The property that matters is that selection is done per slice, against each
slice's own plane. Most axial stacks here are angled per disc level (median 5
orientation groups, up to 6), so any rule that assumes one series affine picks
the wrong slice. The angled-stack cases below are the point of this file.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composition.geometry import SeriesGeometry, SliceGeometry  # noqa: E402
from data.rsna_axial_fixed import nearest_slice, selection_report  # noqa: E402

import pandas as pd  # noqa: E402


def _axial_slice(k, z, tilt=0.0, rows=64, cols=64, spacing=0.6, thickness=4.0):
    """An axial slice at height z, optionally tilted about the L-R axis."""
    c, s = np.cos(tilt), np.sin(tilt)
    row_cosine = np.array([1.0, 0.0, 0.0])          # +1 col -> patient +x
    col_cosine = np.array([0.0, c, s])              # +1 row -> tilted A-P
    origin = np.array([-cols * spacing / 2.0, -rows * spacing / 2.0 * c, z])
    return SliceGeometry(instance_number=k + 1, position=origin,
                         row_cosine=row_cosine, col_cosine=col_cosine,
                         row_spacing=spacing, col_spacing=spacing,
                         thickness=thickness, rows=rows, cols=cols)


def _stack(zs, tilts=None):
    tilts = tilts or [0.0] * len(zs)
    return SeriesGeometry([_axial_slice(k, z, t)
                           for k, (z, t) in enumerate(zip(zs, tilts))])


def test_picks_the_slice_the_point_sits_in():
    geom = _stack([0.0, 5.0, 10.0, 15.0, 20.0])
    for k in range(len(geom)):
        target = geom[k].voxel_to_patient(32, 32)
        hit = nearest_slice(geom, target)
        assert hit is not None
        assert hit[0] == k, f"wanted slice {k}, got {hit[0]}"
        assert abs(hit[1]) < 1e-9


def test_picks_the_nearest_when_the_point_falls_between_slices():
    geom = _stack([0.0, 5.0, 10.0, 15.0])
    mid = geom[1].voxel_to_patient(32, 32) + np.array([0.0, 0.0, 1.5])
    k, offset, _ = nearest_slice(geom, mid)
    assert k == 1                      # 1.5 mm above slice 1, 3.5 below slice 2
    assert abs(offset) == pytest.approx(1.5, abs=1e-6)


def test_angled_stack_is_resolved_per_slice():
    """Two blocks at different angles - the series has no single normal.

    Projecting onto a mean normal would put the target between blocks; testing
    each slice against its own plane picks the right one.
    """
    zs = [0.0, 5.0, 10.0, 30.0, 35.0, 40.0]
    tilts = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]     # ~29 degrees for the lower block
    geom = _stack(zs, tilts)
    assert len(geom.orientation_groups()) == 2, "fixture must be genuinely angled"

    for k in range(len(geom)):
        target = geom[k].voxel_to_patient(32, 32)
        hit = nearest_slice(geom, target)
        assert hit[0] == k, f"angled stack: wanted {k}, got {hit[0]}"


def test_mean_normal_would_have_been_wrong():
    """Demonstrates why the per-slice rule is required, not merely tidier."""
    zs = [0.0, 5.0, 10.0, 30.0, 35.0, 40.0]
    tilts = [0.0, 0.0, 0.0, 0.9, 0.9, 0.9]
    geom = _stack(zs, tilts)
    mean_n = geom.reference_normal

    target = geom[4].voxel_to_patient(32, 32)     # inside the tilted block
    per_slice = nearest_slice(geom, target)[0]

    # What a single-affine rule would pick: project along the mean normal only.
    proj = [abs((target - s.position) @ mean_n) for s in geom]
    naive = int(np.argmin(proj))

    assert per_slice == 4
    assert naive != per_slice, "fixture should expose the difference"


def test_out_of_plane_points_are_reported_not_silently_accepted():
    geom = _stack([0.0, 5.0, 10.0])
    far = geom[1].voxel_to_patient(32, 32) + np.array([500.0, 0.0, 0.0])
    k, offset, inside = nearest_slice(geom, far)
    assert inside is False, "a point outside the field of view must be flagged"


def test_in_plane_slices_are_preferred_over_a_closer_out_of_plane_one():
    geom = _stack([0.0, 5.0, 10.0])
    target = geom[2].voxel_to_patient(32, 32)
    k, _, inside = nearest_slice(geom, target)
    assert k == 2 and inside


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _table(deltas, selection="fixed", offsets=None):
    n = len(deltas)
    return pd.DataFrame({
        "study_id": range(n), "level": ["L4/L5"] * n, "series": [1] * n,
        "annotated_instance": [10] * n,
        "fixed_instance": [10 + d for d in deltas],
        "selection": [selection] * n,
        "offset_mm": offsets if offsets is not None else [1.0] * n,
        "in_plane": [True] * n,
    })


def test_report_counts_agreement_with_the_radiologist():
    rep = selection_report(_table([0, 0, 0, 1, -1, 3]))
    assert rep["n"] == 6
    assert rep["same_slice_frac"] == pytest.approx(0.5)
    assert rep["within_1_slice_frac"] == pytest.approx(5 / 6)
    assert rep["max_slice_delta"] == 3


def test_report_counts_fallbacks_separately():
    t = pd.concat([_table([0, 1]), _table([0], selection="fallback_annotated")])
    rep = selection_report(t)
    assert rep["n"] == 2 and rep["n_fallback"] == 1


def test_report_handles_no_usable_rows():
    rep = selection_report(_table([0, 1], selection="fallback_annotated"))
    assert rep["n"] == 0 and rep["n_fallback"] == 2
