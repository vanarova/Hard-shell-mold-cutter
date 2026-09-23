"""Mesh decimation utilities to reduce triangle count before slicing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyvista as pv

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class ReduceResult:
    output_path: Path
    original_triangles: int
    reduced_triangles: int
    reduction_percent: float


def reduced_output_path(stl_path: Path) -> Path:
    """Return path for the reduced STL next to the original file."""
    return stl_path.with_name(f"{stl_path.stem}_reduced{stl_path.suffix}")


def reduce_mesh(
    mesh: pv.PolyData,
    reduction_percent: float,
) -> pv.PolyData:
    """
    Reduce mesh triangle count.

    reduction_percent: share of triangles to remove (1–95).
    """
    if reduction_percent <= 0:
        raise ValueError("Reduction must be greater than 0%.")
    if reduction_percent >= 100:
        raise ValueError("Reduction must be less than 100%.")

    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError("Mesh has no triangles to reduce.")

    if surface.n_cells < 4:
        raise ValueError("Mesh is too small to reduce further.")

    target_reduction = min(reduction_percent / 100.0, 0.95)
    reduced = surface.decimate_pro(
        target_reduction,
        preserve_topology=True,
    )
    reduced = reduced.clean().triangulate()

    if reduced.n_cells == 0:
        raise ValueError("Reduction removed all geometry. Use a lower reduction value.")

    return reduced


def reduce_and_save_stl(
    source_path: Path,
    mesh: pv.PolyData,
    reduction_percent: float,
    progress: ProgressCallback | None = None,
) -> ReduceResult:
    """Decimate a mesh and save it as `<name>_reduced.stl`."""
    original_triangles = int(mesh.n_cells)

    if progress:
        progress("Preparing mesh surface", 0, 3)
    reduced = reduce_mesh(mesh, reduction_percent)

    if progress:
        progress("Decimating triangles", 1, 3)

    output_path = reduced_output_path(source_path)
    reduced.save(output_path)

    if progress:
        progress("Saving reduced STL", 3, 3)

    reduced_triangles = int(reduced.n_cells)
    actual_percent = 100.0 * (1.0 - reduced_triangles / original_triangles)

    return ReduceResult(
        output_path=output_path,
        original_triangles=original_triangles,
        reduced_triangles=reduced_triangles,
        reduction_percent=actual_percent,
    )
