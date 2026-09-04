"""Voxel index -> patient coordinates, in millimetres.

Everything downstream depends on this being right.

Geometry is kept per-slice rather than collapsed into one 3D affine. RSNA
lumbar axial series are routinely acquired as per-level angled stacks, so a
single affine for the whole series is not merely imprecise, it is wrong.

Reference: DICOM PS3.3 C.7.6.2.1.1 (Image Plane Module).

DICOM index conventions, spelled out because they are easy to transpose:
  ImageOrientationPatient[0:3]  direction cosines of the first *row*. Walking
                               along a row means incrementing the COLUMN index.
  ImageOrientationPatient[3:6]  direction cosines of the first *column*.
                               Walking down a column increments the ROW index.
  PixelSpacing[0]              spacing between adjacent rows, i.e. the step
                               taken when the ROW index increments.
  PixelSpacing[1]              spacing between adjacent columns, i.e. the step
                               taken when the COLUMN index increments.

Patient coordinates are DICOM LPS: +x left, +y posterior, +z superior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

LPS_AXES = ("x", "y", "z")
PLANE_BY_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}

# IOP rows are stored to 6-ish decimals; orthonormality holds to about 1e-5.
ORTHONORMAL_TOL = 1e-4


class GeometryError(ValueError):
    pass


@dataclass(frozen=True)
class SliceGeometry:
    """Plane geometry of a single DICOM instance."""

    instance_number: int
    position: np.ndarray  # (3,) ImagePositionPatient: centre of voxel (0, 0)
    row_cosine: np.ndarray  # (3,) IOP[0:3], traversed by incrementing the column index
    col_cosine: np.ndarray  # (3,) IOP[3:6], traversed by incrementing the row index
    row_spacing: float  # PixelSpacing[0], mm between adjacent rows
    col_spacing: float  # PixelSpacing[1], mm between adjacent columns
    thickness: float  # SliceThickness, mm
    rows: int
    cols: int
    path: str = ""

    def __post_init__(self):
        # Columns are the patient-space displacement per +1 of (row, col, normal).
        basis = np.stack([self.step_row, self.step_col, self.normal], axis=1)
        object.__setattr__(self, "_basis", basis)
        object.__setattr__(self, "_basis_inv", np.linalg.inv(basis))

    @property
    def step_row(self) -> np.ndarray:
        """Patient-space displacement per +1 row index."""
        return self.col_cosine * self.row_spacing

    @property
    def step_col(self) -> np.ndarray:
        """Patient-space displacement per +1 column index."""
        return self.row_cosine * self.col_spacing

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.row_cosine, self.col_cosine)

    def voxel_to_patient(self, row, col) -> np.ndarray:
        """Voxel centre(s) in patient mm. Scalars give (3,), arrays give (N, 3)."""
        row = np.asarray(row, dtype=float)
        col = np.asarray(col, dtype=float)
        return (
            self.position
            + row[..., None] * self.step_row
            + col[..., None] * self.step_col
        )

    def patient_to_voxel(self, point) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse of voxel_to_patient.

        Returns (row, col, offset) as continuous values. `offset` is the signed
        distance from the point to this slice's plane, along the normal.

        Solved against the actual basis rather than by projection. Header
        cosines are orthogonal only to ~1e-4, and dot-product projection turns
        that into a round-trip error that grows with distance from the origin;
        inverting the basis is exact whatever the frame.
        """
        delta = np.asarray(point, dtype=float) - self.position
        coords = delta @ self._basis_inv.T
        row, col, offset = coords[..., 0], coords[..., 1], coords[..., 2]
        if row.ndim == 0:  # a single point in, plain floats out
            return float(row), float(col), float(offset)
        return row, col, offset

    def voxel_support_extent(self) -> np.ndarray:
        """Voxel size (mm) along (row axis, column axis, normal)."""
        return np.array([self.row_spacing, self.col_spacing, self.thickness])

    def support_axes(self) -> np.ndarray:
        """(3, 3) unit vectors matching voxel_support_extent, as patient directions."""
        return np.stack([self.col_cosine, self.row_cosine, self.normal])

    def corner_centres(self) -> np.ndarray:
        """(4, 3) centres of the four corner voxels."""
        rr = [0, 0, self.rows - 1, self.rows - 1]
        cc = [0, self.cols - 1, 0, self.cols - 1]
        return self.voxel_to_patient(np.array(rr), np.array(cc))

    def field_of_view(self) -> np.ndarray:
        """(2,) full in-plane extent (mm): matrix size x spacing, edge to edge."""
        return np.array([self.rows * self.row_spacing, self.cols * self.col_spacing])

    def contains(self, point, margin_mm: float = 0.0) -> bool:
        row, col, offset = self.patient_to_voxel(point)
        half = self.thickness / 2.0 + margin_mm
        return (
            -0.5 <= row <= self.rows - 0.5
            and -0.5 <= col <= self.cols - 0.5
            and abs(offset) <= half
        )


class SeriesGeometry:
    """Per-slice geometry for one DICOM series, ordered along the stack normal."""

    def __init__(self, slices: list[SliceGeometry], study_id=None, series_id=None):
        if not slices:
            raise GeometryError("series has no slices")
        self.study_id = study_id
        self.series_id = series_id
        self._slices = _sort_along_normal(slices)

    def __len__(self) -> int:
        return len(self._slices)

    def __getitem__(self, k: int) -> SliceGeometry:
        return self._slices[k]

    def __iter__(self):
        return iter(self._slices)

    @property
    def slices(self) -> list[SliceGeometry]:
        return self._slices

    @property
    def reference_normal(self) -> np.ndarray:
        """Mean slice normal, renormalised. Equals the slice normal when the
        stack is not angled."""
        stacked = np.stack([s.normal for s in self._slices])
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            raise GeometryError("slice normals cancel; stack orientation is incoherent")
        return mean / norm

    @property
    def plane(self) -> str:
        return PLANE_BY_AXIS[int(np.argmax(np.abs(self.reference_normal)))]

    def voxel_to_patient(self, k: int, row, col) -> np.ndarray:
        return self._slices[k].voxel_to_patient(row, col)

    def voxel_centre(self, k: int, row, col) -> np.ndarray:
        return self._slices[k].voxel_to_patient(row, col)

    def voxel_support_extent(self, k: int) -> np.ndarray:
        return self._slices[k].voxel_support_extent()

    def slice_positions(self) -> np.ndarray:
        """(N,) projection of each slice origin onto the reference normal."""
        n = self.reference_normal
        return np.array([s.position @ n for s in self._slices])

    def slice_gaps(self) -> np.ndarray:
        return np.diff(self.slice_positions())

    def orientation_groups(self, tol: float = 1e-3) -> list[list[int]]:
        """Slice indices grouped by shared orientation. More than one group
        means the stack is angled, e.g. a per-level axial acquisition."""
        groups: list[list[int]] = []
        reps: list[np.ndarray] = []
        for k, s in enumerate(self._slices):
            iop = np.concatenate([s.row_cosine, s.col_cosine])
            for gi, rep in enumerate(reps):
                if np.allclose(iop, rep, atol=tol):
                    groups[gi].append(k)
                    break
            else:
                reps.append(iop)
                groups.append([k])
        return groups

    def is_angled(self, tol: float = 1e-3) -> bool:
        return len(self.orientation_groups(tol)) > 1

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned patient-coordinate bounding box of the full voxel
        support, including the half-thickness beyond the first and last slice."""
        pts = []
        for s in self._slices:
            half_r = 0.5 * s.step_row
            half_c = 0.5 * s.step_col
            half_n = 0.5 * s.thickness * s.normal
            for corner in s.corner_centres():
                for dr in (-half_r, half_r):
                    for dc in (-half_c, half_c):
                        for dn in (-half_n, half_n):
                            pts.append(corner + dr + dc + dn)
        pts = np.asarray(pts)
        return pts.min(axis=0), pts.max(axis=0)

    def locate(self, point, margin_mm: float = 0.0):
        """Slice index whose plane the point falls in, or None if uncovered."""
        best = None
        best_offset = np.inf
        for k, s in enumerate(self._slices):
            row, col, offset = s.patient_to_voxel(point)
            if not (-0.5 <= row <= s.rows - 0.5 and -0.5 <= col <= s.cols - 0.5):
                continue
            if abs(offset) <= s.thickness / 2.0 + margin_mm and abs(offset) < best_offset:
                best, best_offset = (k, row, col, offset), abs(offset)
        return best

    def summary(self) -> dict:
        gaps = self.slice_gaps()
        s0 = self._slices[0]
        lo, hi = self.bounds()
        return {
            "study_id": self.study_id,
            "series_id": self.series_id,
            "plane": self.plane,
            "n_slices": len(self),
            "rows": s0.rows,
            "cols": s0.cols,
            "row_spacing_mm": s0.row_spacing,
            "col_spacing_mm": s0.col_spacing,
            "thickness_mm": s0.thickness,
            "median_gap_mm": float(np.median(gaps)) if gaps.size else float("nan"),
            "angled": self.is_angled(),
            "n_orientation_groups": len(self.orientation_groups()),
            "bounds_min": lo.tolist(),
            "bounds_max": hi.tolist(),
        }

    @classmethod
    def from_datasets(cls, datasets, study_id=None, series_id=None) -> "SeriesGeometry":
        return cls([slice_geometry(ds) for ds in datasets], study_id, series_id)

    @classmethod
    def from_dir(cls, directory, study_id=None, series_id=None) -> "SeriesGeometry":
        directory = Path(directory)
        files = sorted(directory.glob("*.dcm"), key=lambda p: int(p.stem))
        if not files:
            raise GeometryError(f"no DICOM files in {directory}")
        slices = []
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            slices.append(slice_geometry(ds, path=str(f)))
        if study_id is None:
            study_id = directory.parent.name
        if series_id is None:
            series_id = directory.name
        return cls(slices, study_id, series_id)


def slice_geometry(ds, path: str = "") -> SliceGeometry:
    """Build SliceGeometry from a pydicom dataset, validating the plane module."""
    for tag in ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"):
        if not hasattr(ds, tag):
            raise GeometryError(f"{path or '<dataset>'}: missing {tag}")

    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    if iop.shape != (6,):
        raise GeometryError(f"{path}: ImageOrientationPatient must have 6 values")
    row_cosine, col_cosine = iop[:3], iop[3:]

    for name, vec in (("row", row_cosine), ("col", col_cosine)):
        if abs(np.linalg.norm(vec) - 1.0) > ORTHONORMAL_TOL:
            raise GeometryError(f"{path}: {name} direction cosine is not unit length")
    if abs(row_cosine @ col_cosine) > ORTHONORMAL_TOL:
        raise GeometryError(f"{path}: direction cosines are not orthogonal")

    # Headers store the cosines rounded to ~6 decimals, so they are a little
    # off unit length. Renormalise: the intent is a unit vector, and leaving
    # the rounding in place puts a ~1e-7 relative scale error on every
    # projection. Orthogonality is checked but not forced - squaring the frame
    # would overwrite what the scanner actually recorded.
    row_cosine = row_cosine / np.linalg.norm(row_cosine)
    col_cosine = col_cosine / np.linalg.norm(col_cosine)

    spacing = np.asarray(ds.PixelSpacing, dtype=float)
    thickness = float(getattr(ds, "SliceThickness", 0.0) or 0.0)
    if thickness <= 0:
        # Fall back to the reconstruction interval when thickness is absent.
        thickness = float(getattr(ds, "SpacingBetweenSlices", 0.0) or 0.0)
    if thickness <= 0:
        raise GeometryError(f"{path}: no usable SliceThickness or SpacingBetweenSlices")

    return SliceGeometry(
        instance_number=int(getattr(ds, "InstanceNumber", 0)),
        position=np.asarray(ds.ImagePositionPatient, dtype=float),
        row_cosine=row_cosine,
        col_cosine=col_cosine,
        row_spacing=float(spacing[0]),
        col_spacing=float(spacing[1]),
        thickness=thickness,
        rows=int(ds.Rows),
        cols=int(ds.Columns),
        path=path,
    )


def _sort_along_normal(slices: list[SliceGeometry]) -> list[SliceGeometry]:
    """Order by position along the stack normal, not by InstanceNumber.

    InstanceNumber ordering is a filesystem convention; position is physical.
    """
    stacked = np.stack([s.normal for s in slices])
    mean = stacked.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-8:
        raise GeometryError("slice normals cancel; stack orientation is incoherent")
    n = mean / norm
    return sorted(slices, key=lambda s: float(s.position @ n))


def plane_trace(target: SliceGeometry, other: SliceGeometry):
    """Where `other`'s plane cuts across `target`'s image, in (row, col).

    Two non-parallel planes meet in a line. Drawing that line on the sagittal
    image shows exactly which vertebra an axial slice passes through - a direct
    check that needs no thresholds and no trust in a distance metric.

    Returns ((row0, col0), (row1, col1)), or None if the planes are parallel or
    the line misses the image.
    """
    n = other.normal
    # A point at (row, col) on `target` lies on `other`'s plane when
    #   c0 + c1*row + c2*col = 0
    c0 = (target.position - other.position) @ n
    c1 = target.step_row @ n
    c2 = target.step_col @ n

    lo_r, hi_r = -0.5, target.rows - 0.5
    lo_c, hi_c = -0.5, target.cols - 0.5
    hits = []
    if abs(c2) > 1e-12:
        for row in (lo_r, hi_r):
            col = -(c0 + c1 * row) / c2
            if lo_c <= col <= hi_c:
                hits.append((row, col))
    if abs(c1) > 1e-12:
        for col in (lo_c, hi_c):
            row = -(c0 + c2 * col) / c1
            if lo_r <= row <= hi_r:
                hits.append((row, col))

    if len(hits) < 2:
        return None
    # Keep the two furthest apart; edge cases can land the same corner twice.
    best, far = None, -1.0
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            d = np.hypot(hits[i][0] - hits[j][0], hits[i][1] - hits[j][1])
            if d > far:
                best, far = (hits[i], hits[j]), d
    return best if far > 1e-9 else None


def overlap_box(a: SeriesGeometry, b: SeriesGeometry):
    """Axis-aligned patient-space intersection of two series, or None."""
    lo_a, hi_a = a.bounds()
    lo_b, hi_b = b.bounds()
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    return (lo, hi) if np.all(hi > lo) else None
