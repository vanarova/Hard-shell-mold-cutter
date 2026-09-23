"""Slice-axis configuration, paths, and coordinate remapping."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


class SliceAxis(str, Enum):
    Z = "Z"
    X = "X"
    Y = "Y"

    @property
    def label(self) -> str:
        return self.value


def output_dir_for_stl(stl_path: Path) -> Path:
    """Return the output directory named after the STL file (without extension)."""
    return stl_path.parent / stl_path.stem


def slice_dir_for_axis(stl_path: Path, axis: SliceAxis) -> Path:
    """Directory holding numbered slice STLs for the given axis."""
    base = output_dir_for_stl(stl_path)
    if axis == SliceAxis.Z:
        return base
    return base / axis.value.lower()


def model_output_dir(stl_path: Path) -> Path:
    return output_dir_for_stl(stl_path)


def combined_model_path(stl_path: Path) -> Path:
    return model_output_dir(stl_path) / "combined.stl"


def permute_points(
    points: np.ndarray,
    axis: SliceAxis,
    *,
    to_canonical: bool,
) -> np.ndarray:
    """
    Remap world XYZ so the slice axis becomes canonical Z.

    Canonical footprint lives in XY; stack depth is Z.
    """
    if axis == SliceAxis.Z:
        return np.asarray(points, dtype=float)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if to_canonical:
        if axis == SliceAxis.X:
            return np.column_stack([y, z, x])
        return np.column_stack([x, z, y])

    if axis == SliceAxis.X:
        return np.column_stack([z, x, y])
    return np.column_stack([x, z, y])


def transform_polydata(
    mesh: pv.PolyData,
    axis: SliceAxis,
    *,
    to_canonical: bool,
) -> pv.PolyData:
    transformed = mesh.copy(deep=True)
    transformed.points = permute_points(mesh.points, axis, to_canonical=to_canonical)
    return transformed


def transform_trimesh(
    mesh: trimesh.Trimesh,
    axis: SliceAxis,
    *,
    to_canonical: bool,
) -> trimesh.Trimesh:
    transformed = mesh.copy()
    transformed.vertices = permute_points(mesh.vertices, axis, to_canonical=to_canonical)
    return transformed


def axis_extent_mm(mesh: pv.PolyData, axis: SliceAxis) -> float:
    bounds = mesh.bounds
    if axis == SliceAxis.X:
        return float(bounds[1] - bounds[0])
    if axis == SliceAxis.Y:
        return float(bounds[3] - bounds[2])
    return float(bounds[5] - bounds[4])
