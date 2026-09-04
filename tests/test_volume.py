"""Stage 2 support: patient-space sampling, and the Exp 0 measurement floor.

The important test here is test_phantom_centroid_agrees_across_acquisitions. It
images one synthetic object with two unrelated acquisition geometries and asks
whether Exp 0 recovers the same centroid from each. With no motion by
construction, whatever discrepancy it reports is pure discretisation - the
noise floor against which real discrepancies have to be judged.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composition.geometry import GeometryError, SeriesGeometry, SliceGeometry  # noqa: E402
from composition.volume import Volume  # noqa: E402
from experiments.exp0_motion import (  # noqa: E402
    decompose,
    direction_consistency,
    landmark_centroid,
    slice_index_for_instance,
)

BLOB_CENTRE = np.array([3.7, -41.3, -228.9])
BLOB_SIGMA = 6.0


def _blob(points):
    """Isotropic Gaussian in patient coordinates - the ground-truth object."""
    d2 = ((points - BLOB_CENTRE) ** 2).sum(axis=-1)
    return np.exp(-d2 / (2 * BLOB_SIGMA**2)).astype(np.float32)


def _phantom(plane, n_slices, row_spacing, col_spacing, thickness, gap,
             rows=96, cols=96):
    """Image the blob with a given acquisition geometry."""
    if plane == "sagittal":
        row_cosine = np.array([0.0, 1.0, 0.0])
        col_cosine = np.array([0.0, 0.0, -1.0])
        stack = np.array([1.0, 0.0, 0.0])
    else:
        row_cosine = np.array([1.0, 0.0, 0.0])
        col_cosine = np.array([0.0, 1.0, 0.0])
        stack = np.array([0.0, 0.0, 1.0])

    # Centre the stack on the blob, and the in-plane grid too.
    slices, pixels = [], []
    for k in range(n_slices):
        offset = (k - (n_slices - 1) / 2.0) * gap
        origin = (
            BLOB_CENTRE
            + offset * stack
            - col_cosine * row_spacing * (rows - 1) / 2.0
            - row_cosine * col_spacing * (cols - 1) / 2.0
        )
        g = SliceGeometry(
            instance_number=k + 1, position=origin,
            row_cosine=row_cosine, col_cosine=col_cosine,
            row_spacing=row_spacing, col_spacing=col_spacing,
            thickness=thickness, rows=rows, cols=cols,
        )
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        slices.append(g)
        pixels.append(_blob(g.voxel_to_patient(rr, cc)))
    return Volume(SeriesGeometry(slices, study_id=1, series_id=2), pixels)


@pytest.fixture
def sag_phantom():
    return _phantom("sagittal", n_slices=17, row_spacing=0.55, col_spacing=0.55,
                    thickness=4.0, gap=4.4)


@pytest.fixture
def ax_phantom():
    # Deliberately unlike the sagittal one: finer in-plane, coarser through-plane.
    return _phantom("axial", n_slices=21, row_spacing=0.31, col_spacing=0.31,
                    thickness=3.5, gap=3.8)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_sphere_respects_its_radius(sag_phantom):
    for radius in (3.0, 7.0, 15.0):
        s = sag_phantom.sample_sphere(BLOB_CENTRE, radius)
        assert len(s) > 0
        assert np.linalg.norm(s.coords - BLOB_CENTRE, axis=1).max() <= radius + 1e-9


def test_sphere_pulls_from_every_overlapping_slice(sag_phantom):
    # A 10 mm sphere spans 20 mm; at a 4.4 mm gap that is 5 slices or so.
    s = sag_phantom.sample_sphere(BLOB_CENTRE, 10.0)
    assert s.n_slices >= 5


def test_sphere_outside_the_field_of_view_is_empty(sag_phantom):
    s = sag_phantom.sample_sphere(BLOB_CENTRE + np.array([500.0, 0.0, 0.0]), 5.0)
    assert len(s) == 0


def test_box_and_sphere_agree_on_the_inscribed_region(sag_phantom):
    radius = 6.0
    sphere = sag_phantom.sample_sphere(BLOB_CENTRE, radius)
    box = sag_phantom.sample_box(BLOB_CENTRE, radius)
    assert len(box) > len(sphere)  # the box circumscribes the sphere
    assert np.all(np.abs(box.coords - BLOB_CENTRE) <= radius + 1e-9)


def test_all_voxels_covers_the_series(sag_phantom):
    assert len(sag_phantom.all_voxels()) == sum(p.size for p in sag_phantom.pixels)


def test_volume_rejects_pixel_geometry_mismatch(sag_phantom):
    bad = [np.zeros((3, 3), dtype=np.float32) for _ in sag_phantom.pixels]
    with pytest.raises(GeometryError, match="pixels"):
        Volume(sag_phantom.geometry, bad)


# --------------------------------------------------------------------------
# centroid recovery
# --------------------------------------------------------------------------


def test_weighted_centroid_recovers_the_blob(sag_phantom):
    s = sag_phantom.sample_sphere(BLOB_CENTRE, 14.0)
    assert np.linalg.norm(s.weighted_centroid() - BLOB_CENTRE) < 0.05


def test_weighted_centroid_beats_geometric_when_seed_is_offset(sag_phantom):
    seed = BLOB_CENTRE + np.array([1.5, 2.0, -1.0])
    s = sag_phantom.sample_sphere(seed, 14.0)
    weighted = np.linalg.norm(s.weighted_centroid() - BLOB_CENTRE)
    geometric = np.linalg.norm(s.geometric_centroid() - BLOB_CENTRE)
    assert weighted < geometric


def test_phantom_centroid_agrees_across_acquisitions(sag_phantom, ax_phantom):
    """The Exp 0 noise floor: same object, two geometries, no motion."""
    radius = 12.0
    results = {}
    for name, volume in (("sagittal", sag_phantom), ("axial", ax_phantom)):
        g = volume.geometry[len(volume) // 2]
        row, col, _ = g.patient_to_voxel(BLOB_CENTRE)
        out = landmark_centroid(volume, g.instance_number, int(round(row)),
                                int(round(col)), radius_mm=radius)
        results[name] = out["centroid_mm"]
        assert np.linalg.norm(out["centroid_mm"] - BLOB_CENTRE) < 0.2, name

    discrepancy = np.linalg.norm(results["sagittal"] - results["axial"])
    # Two acquisitions differing in voxel size, slice gap and orientation still
    # land within a fifth of a millimetre of each other.
    assert discrepancy < 0.2, f"discretisation floor is {discrepancy:.3f} mm"


def test_slice_lookup_rejects_an_unknown_instance(sag_phantom):
    assert slice_index_for_instance(sag_phantom.geometry, 1) >= 0
    with pytest.raises(KeyError, match="not in series"):
        slice_index_for_instance(sag_phantom.geometry, 9999)


# --------------------------------------------------------------------------
# direction statistics
# --------------------------------------------------------------------------


def test_aligned_vectors_read_as_systematic():
    rng = np.random.default_rng(0)
    base = np.array([0.0, 0.0, 1.0])
    vectors = base + rng.normal(scale=0.05, size=(20, 3))
    out = direction_consistency(vectors)
    assert out["resultant_length"] > 0.95
    assert out["looks_systematic"]


def test_isotropic_vectors_do_not_read_as_systematic():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 3))
    out = direction_consistency(vectors)
    assert out["resultant_length"] < 3.0 / np.sqrt(20)
    assert not out["looks_systematic"]


def test_direction_needs_at_least_two_vectors():
    assert np.isnan(direction_consistency(np.zeros((1, 3)))["resultant_length"])


# --------------------------------------------------------------------------
# separating a shared bias from real motion
# --------------------------------------------------------------------------


def test_rayleigh_catches_what_the_old_heuristic_missed():
    """Rbar=0.56 at n=25 is highly non-random, but 3/sqrt(n)=0.60 called it random."""
    rng = np.random.default_rng(3)
    # A-P dominated field like the observed one: shared offset plus scatter.
    # scale 3.5 reproduces the observed Rbar=0.56 at n=25 almost exactly.
    v = np.array([0.0, -4.0, 0.0]) + rng.normal(scale=3.5, size=(25, 3))
    out = direction_consistency(v)
    assert out["resultant_length"] > 0.4
    assert out["rayleigh_p"] < 0.01
    assert out["looks_systematic"], "a shared axis must not read as random"
    assert out["resultant_length"] < 3.0 / np.sqrt(25), "old rule would have missed this"


def test_pure_bias_leaves_almost_no_residual():
    v = np.tile(np.array([0.0, -4.0, 0.0]), (25, 1))
    d = decompose(v)
    assert d["systematic_norm_mm"] == pytest.approx(4.0, abs=1e-9)
    assert d["residual_median_mm"] < 1e-9
    assert d["explained_fraction"] > 0.999


def test_pure_motion_has_no_shared_offset():
    rng = np.random.default_rng(0)
    v = rng.normal(scale=3.0, size=(200, 3))
    d = decompose(v)
    assert d["systematic_norm_mm"] < 1.0
    # Removing a near-zero offset must not shrink the discrepancy much.
    assert d["explained_fraction"] < 0.15
    assert not d["residual_direction"]["looks_systematic"]


def test_decompose_recovers_a_planted_bias():
    rng = np.random.default_rng(7)
    bias = np.array([0.1, -4.0, 0.0])
    v = bias + rng.normal(scale=2.0, size=(400, 3))
    d = decompose(v)
    assert np.allclose(d["systematic_mm"], bias, atol=0.35)
    # What is left should be isotropic - that is the motion component.
    assert not d["residual_direction"]["looks_systematic"]
    assert d["residual_median_mm"] < d["raw_median_mm"]


def test_decompose_is_robust_to_one_bad_mark():
    rng = np.random.default_rng(1)
    v = np.array([0.0, -4.0, 0.0]) + rng.normal(scale=1.5, size=(25, 3))
    v[0] = np.array([0.0, -80.0, 0.0])          # one study marked a level off
    d = decompose(v)
    assert abs(d["systematic_mm"][1] + 4.0) < 1.0, "median must resist the outlier"
