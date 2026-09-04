"""Pixel data attached to per-slice geometry.

Slices are held as a list of 2D arrays, never stacked into a 3D block. Stacking
would presume a common lattice, and 18 of 25 axial series here are angled
per-level stacks for which no such lattice exists. Nothing in this module
resizes, resamples or interpolates: the anisotropy is the object of study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

from .geometry import GeometryError, SeriesGeometry, slice_geometry


@dataclass
class RegionSample:
    """Voxels drawn from a patient-space neighbourhood."""

    intensities: np.ndarray  # (N,)
    coords: np.ndarray  # (N, 3) voxel centres, patient mm
    slice_index: np.ndarray  # (N,) which slice each voxel came from
    rows: np.ndarray  # (N,)
    cols: np.ndarray  # (N,)

    def __len__(self) -> int:
        return int(self.intensities.size)

    @property
    def n_slices(self) -> int:
        return int(np.unique(self.slice_index).size)

    def weighted_centroid(self, floor_at_zero: bool = True) -> np.ndarray:
        """Intensity-weighted centroid in patient mm.

        MR intensity has no calibrated zero, so a large constant pedestal drags
        this toward the plain geometric centre. That is a conservative failure
        for Exp 0 - it shrinks apparent discrepancy rather than inflating it -
        so the raw weighting the spec asks for is kept as the default.
        """
        w = self.intensities.astype(float)
        if floor_at_zero:
            w = np.clip(w, 0.0, None)
        total = w.sum()
        if total <= 0:
            raise ValueError("region has no positive intensity to weight by")
        return (w[:, None] * self.coords).sum(axis=0) / total

    def geometric_centroid(self) -> np.ndarray:
        return self.coords.mean(axis=0)


class Volume:
    """A series' pixel data, indexed by (slice, row, col) but addressed in mm."""

    def __init__(self, geometry: SeriesGeometry, pixels: list[np.ndarray]):
        if len(geometry) != len(pixels):
            raise GeometryError("geometry and pixel slice counts differ")
        for k, (g, p) in enumerate(zip(geometry, pixels)):
            if p.shape != (g.rows, g.cols):
                raise GeometryError(
                    f"slice {k}: pixels {p.shape} != header {(g.rows, g.cols)}"
                )
        self.geometry = geometry
        self.pixels = pixels

    def __len__(self) -> int:
        return len(self.geometry)

    @property
    def study_id(self):
        return self.geometry.study_id

    @property
    def series_id(self):
        return self.geometry.series_id

    @property
    def plane(self) -> str:
        return self.geometry.plane

    def intensity(self, k: int, row: int, col: int) -> float:
        return float(self.pixels[k][row, col])

    def slice_voxel_centres(self, k: int) -> np.ndarray:
        """(rows, cols, 3) patient coordinates of every voxel centre in slice k."""
        g = self.geometry[k]
        rr, cc = np.meshgrid(np.arange(g.rows), np.arange(g.cols), indexing="ij")
        return g.voxel_to_patient(rr, cc)

    def all_voxels(self) -> RegionSample:
        """Every voxel in the series, as a flat patient-coordinate sample."""
        return self._gather(range(len(self)), selector=None)

    def sample_sphere(self, centre_mm, radius_mm: float) -> RegionSample:
        """Voxels whose centres lie within radius_mm of a patient-space point.

        A slice contributes when the point is within radius_mm + half its
        thickness of the slice plane, so an anisotropic stack still supplies
        every slice that physically overlaps the sphere.
        """
        centre = np.asarray(centre_mm, dtype=float)
        return self._gather(range(len(self)), selector=("sphere", centre, radius_mm))

    def sample_box(self, centre_mm, half_extent_mm) -> RegionSample:
        centre = np.asarray(centre_mm, dtype=float)
        half = np.asarray(half_extent_mm, dtype=float)
        if half.ndim == 0:
            half = np.repeat(half, 3)
        return self._gather(range(len(self)), selector=("box", centre, half))

    def _gather(self, slice_indices, selector) -> RegionSample:
        vals, pts, ks, rs, cs = [], [], [], [], []

        for k in slice_indices:
            g = self.geometry[k]
            row_idx, col_idx = self._candidate_indices(g, selector)
            if row_idx is None:
                continue

            coords = g.voxel_to_patient(row_idx, col_idx)
            keep = self._refine(coords, selector)
            if keep is not None:
                if not keep.any():
                    continue
                row_idx, col_idx, coords = row_idx[keep], col_idx[keep], coords[keep]

            vals.append(self.pixels[k][row_idx, col_idx])
            pts.append(coords)
            ks.append(np.full(row_idx.shape, k))
            rs.append(row_idx)
            cs.append(col_idx)

        if not vals:
            empty_f = np.zeros(0)
            return RegionSample(empty_f, np.zeros((0, 3)), np.zeros(0, int),
                                np.zeros(0, int), np.zeros(0, int))
        return RegionSample(
            np.concatenate(vals).astype(float),
            np.concatenate(pts),
            np.concatenate(ks),
            np.concatenate(rs),
            np.concatenate(cs),
        )

    @staticmethod
    def _candidate_indices(g, selector):
        """Index-space bounding box for the selector, or (None, None) to skip."""
        if selector is None:
            rr, cc = np.meshgrid(np.arange(g.rows), np.arange(g.cols), indexing="ij")
            return rr.ravel(), cc.ravel()

        kind, centre, size = selector
        row_c, col_c, offset = g.patient_to_voxel(centre)
        reach = float(size) if kind == "sphere" else float(np.max(size))

        if abs(offset) > reach + g.thickness / 2.0:
            return None, None

        if kind == "sphere":
            # In-plane radius of the sphere's intersection with this plane.
            residual = max(0.0, size**2 - offset**2)
            in_plane = np.sqrt(residual)
            half_r = in_plane / g.row_spacing
            half_c = in_plane / g.col_spacing
        else:
            half_r = reach / g.row_spacing
            half_c = reach / g.col_spacing

        r0 = max(0, int(np.floor(row_c - half_r)))
        r1 = min(g.rows - 1, int(np.ceil(row_c + half_r)))
        c0 = max(0, int(np.floor(col_c - half_c)))
        c1 = min(g.cols - 1, int(np.ceil(col_c + half_c)))
        if r1 < r0 or c1 < c0:
            return None, None

        rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        return rr.ravel(), cc.ravel()

    @staticmethod
    def _refine(coords, selector):
        if selector is None:
            return None
        kind, centre, size = selector
        delta = coords - centre
        if kind == "sphere":
            return (delta**2).sum(axis=1) <= size**2
        return np.all(np.abs(delta) <= size, axis=1)

    @classmethod
    def from_dir(cls, directory, study_id=None, series_id=None) -> "Volume":
        directory = Path(directory)
        files = sorted(directory.glob("*.dcm"), key=lambda p: int(p.stem))
        if not files:
            raise GeometryError(f"no DICOM files in {directory}")

        entries = []
        for f in files:
            ds = pydicom.dcmread(f)
            entries.append((slice_geometry(ds, path=str(f)), _rescaled(ds)))

        geometry = SeriesGeometry(
            [g for g, _ in entries],
            study_id if study_id is not None else directory.parent.name,
            series_id if series_id is not None else directory.name,
        )
        # SeriesGeometry reorders along the normal; follow that ordering.
        by_path = {g.path: px for g, px in entries}
        return cls(geometry, [by_path[g.path] for g in geometry])


def _rescaled(ds) -> np.ndarray:
    """Pixel array with RescaleSlope/Intercept applied where present.

    Only ~1 in 4 of these series carries the rescale tags; the identity default
    is what the standard prescribes when they are absent.
    """
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept
    return arr
