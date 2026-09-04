"""Stage 1 verification. Nothing downstream is trustworthy until these pass.

Three checks are mandated by the spec:
  1. corner voxels -> patient coords; extent matches FOV x matrix size
  2. the same anatomical point resolved via two different slices agrees
  3. a sagittal and an axial series from one study overlap in patient space

Synthetic cases run everywhere; the real-data cases skip when the paired-study
DICOMs are not on disk.
"""

import sys
from pathlib import Path

import numpy as np
import pydicom
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from composition.geometry import (  # noqa: E402
    GeometryError,
    SeriesGeometry,
    SliceGeometry,
    overlap_box,
    plane_trace,
    slice_geometry,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rsna"
IMAGES_DIR = DATA_DIR / "train_images"
PAIRS_CSV = DATA_DIR / "paired_studies.csv"

RTOL = 1e-9
ATOL_MM = 1e-6

# Measured worst case over all 50 paired series on disk is 2.3e-13 mm, so these
# are loose enough for float64 noise and tight enough to catch a real change.
REAL_RTOL = 1e-12
REAL_ATOL_MM = 1e-9


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _synthetic_slice(k, plane="sagittal", rows=64, cols=48,
                     row_spacing=0.6, col_spacing=0.8, thickness=4.0, gap=5.0):
    """A slice with deliberately anisotropic in-plane spacing, so a row/column
    transposition cannot pass unnoticed."""
    if plane == "sagittal":
        row_cosine = np.array([0.0, 1.0, 0.0])  # A->P
        col_cosine = np.array([0.0, 0.0, -1.0])  # S->I
        origin = np.array([-20.0 + k * gap, -100.0, 60.0])
    else:  # axial
        row_cosine = np.array([1.0, 0.0, 0.0])  # R->L
        col_cosine = np.array([0.0, 1.0, 0.0])  # A->P
        origin = np.array([-20.0, -100.0, 40.0 - k * gap])
    return SliceGeometry(
        instance_number=k + 1,
        position=origin,
        row_cosine=row_cosine,
        col_cosine=col_cosine,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        thickness=thickness,
        rows=rows,
        cols=cols,
    )


@pytest.fixture
def synthetic_sagittal():
    return SeriesGeometry([_synthetic_slice(k, "sagittal") for k in range(9)])


@pytest.fixture
def synthetic_axial():
    return SeriesGeometry([_synthetic_slice(k, "axial") for k in range(12)])


def _real_pairs(limit=None):
    """(study_id, sag_dir, ax_dir) for paired studies present on disk."""
    if not PAIRS_CSV.exists():
        return []
    import pandas as pd

    out = []
    for row in pd.read_csv(PAIRS_CSV).itertuples():
        sag = IMAGES_DIR / str(row.study_id) / str(row.sag_series_id)
        axi = IMAGES_DIR / str(row.study_id) / str(row.ax_series_id)
        if any(sag.glob("*.dcm")) and any(axi.glob("*.dcm")):
            out.append((int(row.study_id), sag, axi))
        if limit and len(out) >= limit:
            break
    return out


REAL_PAIRS = _real_pairs(limit=8)
needs_data = pytest.mark.skipif(not REAL_PAIRS, reason="paired-study DICOMs not on disk")


# --------------------------------------------------------------------------
# header parsing
# --------------------------------------------------------------------------


def test_direction_cosines_map_to_the_right_index():
    """+1 column index moves along IOP[0:3]; +1 row index moves along IOP[3:6].

    Spacings pair the other way round: PixelSpacing[0] is the between-row step.
    """
    s = _synthetic_slice(0, "sagittal", row_spacing=0.6, col_spacing=0.8)
    origin = s.voxel_to_patient(0, 0)

    step_c = s.voxel_to_patient(0, 1) - origin
    assert np.allclose(step_c, s.row_cosine * 0.8, atol=ATOL_MM)

    step_r = s.voxel_to_patient(1, 0) - origin
    assert np.allclose(step_r, s.col_cosine * 0.6, atol=ATOL_MM)


def test_rejects_non_orthogonal_cosines():
    ds = pydicom.Dataset()
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    # Both unit length, but 45 degrees apart rather than orthogonal.
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.70710678, 0.70710678, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 3.0
    ds.Rows, ds.Columns = 16, 16
    with pytest.raises(GeometryError, match="orthogonal"):
        slice_geometry(ds)


def test_slices_sort_by_position_not_instance_number():
    slices = [_synthetic_slice(k) for k in (3, 0, 2, 1)]
    series = SeriesGeometry(slices)
    assert np.all(series.slice_gaps() > 0)


# --------------------------------------------------------------------------
# spec check 1: extent matches FOV x matrix size
# --------------------------------------------------------------------------


def _assert_extent(series, rtol=RTOL, atol_mm=ATOL_MM):
    for s in series:
        corners = s.corner_centres()
        assert corners.shape == (4, 3)

        # Span of voxel *centres* is (n - 1) * spacing along each in-plane axis.
        span_row = (corners[2] - corners[0]) @ s.col_cosine
        span_col = (corners[1] - corners[0]) @ s.row_cosine
        assert span_row == pytest.approx((s.rows - 1) * s.row_spacing, rel=rtol)
        assert span_col == pytest.approx((s.cols - 1) * s.col_spacing, rel=rtol)

        # Adding the half-voxel border gives the field of view.
        fov = s.field_of_view()
        assert span_row + s.row_spacing == pytest.approx(fov[0], rel=rtol)
        assert span_col + s.col_spacing == pytest.approx(fov[1], rel=rtol)

        # Corners must be coplanar: no component along the normal.
        off = (corners - s.position) @ s.normal
        assert np.allclose(off, 0.0, atol=atol_mm)


def test_extent_matches_fov_synthetic(synthetic_sagittal, synthetic_axial):
    _assert_extent(synthetic_sagittal)
    _assert_extent(synthetic_axial)


def test_bounds_span_full_voxel_support(synthetic_axial):
    lo, hi = synthetic_axial.bounds()
    s = synthetic_axial[0]
    # Axial synthetic: x is the row-cosine axis, spanned by cols.
    assert (hi - lo)[0] == pytest.approx(s.cols * s.col_spacing, rel=1e-6)
    assert (hi - lo)[1] == pytest.approx(s.rows * s.row_spacing, rel=1e-6)
    # z spans the stack plus half a slice at each end.
    span_z = abs(synthetic_axial.slice_positions()[-1] - synthetic_axial.slice_positions()[0])
    assert (hi - lo)[2] == pytest.approx(span_z + s.thickness, rel=1e-6)


@needs_data
@pytest.mark.parametrize("study_id,sag,axi", REAL_PAIRS)
def test_extent_matches_fov_real(study_id, sag, axi):
    for directory in (sag, axi):
        _assert_extent(SeriesGeometry.from_dir(directory),
                       rtol=REAL_RTOL, atol_mm=REAL_ATOL_MM)


# --------------------------------------------------------------------------
# spec check 2: two slices resolve the same anatomical point identically
# --------------------------------------------------------------------------


def _assert_two_slice_agreement(series, seed=0, atol_mm=ATOL_MM):
    if len(series) < 2:
        pytest.skip("series has a single slice")
    rng = np.random.default_rng(seed)
    a, b = series[0], series[len(series) - 1]

    for _ in range(64):
        # A physical point drawn from inside slice a's field of view.
        row = rng.uniform(0, a.rows - 1)
        col = rng.uniform(0, a.cols - 1)
        point = a.voxel_to_patient(row, col) + rng.uniform(-8, 8) * a.normal

        row_a, col_a, off_a = a.patient_to_voxel(point)
        row_b, col_b, off_b = b.patient_to_voxel(point)

        # Each slice reconstructs the identical patient coordinate.
        rebuilt_a = a.voxel_to_patient(row_a, col_a) + off_a * a.normal
        rebuilt_b = b.voxel_to_patient(row_b, col_b) + off_b * b.normal
        assert np.allclose(rebuilt_a, point, atol=atol_mm)
        assert np.allclose(rebuilt_b, point, atol=atol_mm)
        assert np.allclose(rebuilt_a, rebuilt_b, atol=atol_mm)

        # In-plane projections may differ only along the shared normal.
        if np.allclose(a.normal, b.normal, atol=1e-6):
            delta = a.voxel_to_patient(row_a, col_a) - b.voxel_to_patient(row_b, col_b)
            perp = delta - (delta @ a.normal) * a.normal
            assert np.linalg.norm(perp) < atol_mm


def test_two_slice_agreement_synthetic(synthetic_sagittal, synthetic_axial):
    _assert_two_slice_agreement(synthetic_sagittal)
    _assert_two_slice_agreement(synthetic_axial)


def test_round_trip_is_exact(synthetic_sagittal):
    rng = np.random.default_rng(1)
    for k in range(len(synthetic_sagittal)):
        s = synthetic_sagittal[k]
        rows = rng.uniform(0, s.rows - 1, 200)
        cols = rng.uniform(0, s.cols - 1, 200)
        pts = s.voxel_to_patient(rows, cols)
        back_r, back_c, off = s.patient_to_voxel(pts)
        assert np.allclose(back_r, rows, atol=1e-9)
        assert np.allclose(back_c, cols, atol=1e-9)
        assert np.allclose(off, 0.0, atol=1e-9)


@needs_data
@pytest.mark.parametrize("study_id,sag,axi", REAL_PAIRS)
def test_two_slice_agreement_real(study_id, sag, axi):
    for directory in (sag, axi):
        _assert_two_slice_agreement(SeriesGeometry.from_dir(directory),
                                    atol_mm=REAL_ATOL_MM)


# --------------------------------------------------------------------------
# spec check 3: sagittal and axial from one study overlap
# --------------------------------------------------------------------------


def test_overlap_synthetic(synthetic_sagittal, synthetic_axial):
    box = overlap_box(synthetic_sagittal, synthetic_axial)
    assert box is not None
    lo, hi = box
    assert np.all(hi > lo)


def test_disjoint_series_report_no_overlap(synthetic_axial):
    far = SeriesGeometry([
        SliceGeometry(
            instance_number=1,
            position=np.array([1000.0, 1000.0, 1000.0]),
            row_cosine=np.array([1.0, 0.0, 0.0]),
            col_cosine=np.array([0.0, 1.0, 0.0]),
            row_spacing=1.0, col_spacing=1.0, thickness=3.0, rows=8, cols=8,
        )
    ])
    assert overlap_box(synthetic_axial, far) is None


@needs_data
@pytest.mark.parametrize("study_id,sag,axi", REAL_PAIRS)
def test_sagittal_and_axial_overlap_real(study_id, sag, axi):
    sag_geom = SeriesGeometry.from_dir(sag)
    ax_geom = SeriesGeometry.from_dir(axi)

    assert sag_geom.plane == "sagittal", f"{study_id}: sag series reads as {sag_geom.plane}"
    assert ax_geom.plane == "axial", f"{study_id}: axial series reads as {ax_geom.plane}"

    box = overlap_box(sag_geom, ax_geom)
    assert box is not None, f"{study_id}: sagittal and axial bounding boxes are disjoint"

    lo, hi = box
    extent = hi - lo
    # A lumbar overlap smaller than a centimetre in any axis means one of the
    # two geometries is misread, not that the patient is small.
    assert np.all(extent > 10.0), f"{study_id}: overlap only {extent.round(1)} mm"


@needs_data
def test_sagittal_normal_is_left_right():
    for _, sag, _ in REAL_PAIRS:
        n = SeriesGeometry.from_dir(sag).reference_normal
        assert abs(n[0]) > 0.9, f"sagittal normal not L-R: {n.round(3)}"


@needs_data
def test_axial_normal_is_superior_inferior():
    for _, _, axi in REAL_PAIRS:
        n = SeriesGeometry.from_dir(axi).reference_normal
        assert abs(n[2]) > 0.8, f"axial normal not S-I: {n.round(3)}"


# --------------------------------------------------------------------------
# plane_trace: where one slice's plane crosses another's image
# --------------------------------------------------------------------------


def test_plane_trace_passes_through_the_shared_point(synthetic_sagittal, synthetic_axial):
    """A point on both planes must lie on the drawn line."""
    # SeriesGeometry sorts by position, so the construction index is not the
    # stored index - find an axial slice that actually crosses the image.
    sag = synthetic_sagittal[4]
    axi = next((synthetic_axial[k] for k in range(len(synthetic_axial))
                if plane_trace(sag, synthetic_axial[k]) is not None), None)
    assert axi is not None, "no axial slice crosses the mid-sagittal image"
    trace = plane_trace(sag, axi)

    (r0, c0), (r1, c1) = trace
    # Every point of the trace must sit on the axial plane.
    for t in (0.0, 0.25, 0.5, 1.0):
        row, col = r0 + t * (r1 - r0), c0 + t * (c1 - c0)
        offset = (sag.voxel_to_patient(row, col) - axi.position) @ axi.normal
        assert abs(offset) < 1e-6, f"t={t} is {offset:.3g} mm off the axial plane"


def test_plane_trace_tracks_the_axial_slice(synthetic_sagittal, synthetic_axial):
    """Stepping the axial slice must move the line, monotonically."""
    sag = synthetic_sagittal[4]
    rows = []
    for k in range(len(synthetic_axial)):
        trace = plane_trace(sag, synthetic_axial[k])
        if trace is not None:
            rows.append(0.5 * (trace[0][0] + trace[1][0]))
    assert len(rows) >= 3
    deltas = np.diff(rows)
    assert np.all(deltas > 0) or np.all(deltas < 0), "trace should sweep one way"


def test_plane_trace_is_none_for_parallel_planes(synthetic_axial):
    assert plane_trace(synthetic_axial[0], synthetic_axial[5]) is None


@needs_data
@pytest.mark.parametrize("study_id,sag,axi", REAL_PAIRS)
def test_plane_trace_real(study_id, sag, axi):
    sag_geom = SeriesGeometry.from_dir(sag)
    ax_geom = SeriesGeometry.from_dir(axi)
    s = sag_geom[len(sag_geom) // 2]

    drawn = 0
    for k in range(len(ax_geom)):
        trace = plane_trace(s, ax_geom[k])
        if trace is None:
            continue
        drawn += 1
        (r0, c0), (r1, c1) = trace
        for row, col in ((r0, c0), (r1, c1)):
            off = (s.voxel_to_patient(row, col) - ax_geom[k].position) @ ax_geom[k].normal
            assert abs(off) < REAL_ATOL_MM, f"{study_id}: endpoint {off:.3g} mm off plane"
    assert drawn > 0, f"{study_id}: no axial slice crosses the mid-sagittal image"
