"""Build a mold block negative by subtracting the stitched boundary from a box."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import trimesh

from app.boundary_generator_v2 import boundaries_v2_output_dir
from app.boundary_stitcher import stitched_output_path

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class NegativeResult:
    output_path: Path
    margin_mm: float
    box_extents_mm: tuple[float, float, float]
    triangle_count: int


def negative_output_path(slice_dir: Path) -> Path:
    return boundaries_v2_output_dir(slice_dir) / "negative.stl"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )
    return mesh


def bounding_box_mesh(
    mesh: trimesh.Trimesh,
    margin_mm: float,
) -> trimesh.Trimesh:
    """
    Axis-aligned box centered on the mesh, x mm longer/wider/taller than its bounds.
    """
    if margin_mm <= 0:
        raise ValueError("Block margin must be greater than 0 mm.")

    bounds_min, bounds_max = mesh.bounds
    center = (bounds_min + bounds_max) / 2.0
    size = (bounds_max - bounds_min) + margin_mm

    box = trimesh.creation.box(extents=size)
    box.apply_translation(center)
    return box


def _prepare_mesh_for_boolean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean mesh for manifold boolean ops without optional scipy-dependent steps."""
    prepared = mesh.copy()
    prepared.remove_infinite_values()
    prepared.merge_vertices()
    prepared.update_faces(prepared.unique_faces() & prepared.nondegenerate_faces())
    prepared.remove_unreferenced_vertices()
    return prepared


def subtract_stitched_from_box(
    stitched_mesh: trimesh.Trimesh,
    margin_mm: float,
) -> tuple[trimesh.Trimesh, tuple[float, float, float]]:
    """Return box minus stitched mesh and the box extents used."""
    box = bounding_box_mesh(stitched_mesh, margin_mm)
    stitched = _prepare_mesh_for_boolean(stitched_mesh)

    negative = trimesh.boolean.difference(
        [box, stitched],
        engine="manifold",
        check_volume=False,
    )
    if negative.is_empty or len(negative.faces) == 0:
        raise ValueError("Boolean subtraction produced empty geometry.")

    negative = _prepare_mesh_for_boolean(negative)

    extents = tuple(float(value) for value in box.extents)
    return negative, extents


def create_stitched_negative(
    slice_dir: Path,
    margin_mm: float,
    progress: ProgressCallback | None = None,
) -> NegativeResult:
    """Subtract the stitched boundary model from an oversized bounding box."""
    if margin_mm <= 0:
        raise ValueError("Block margin must be greater than 0 mm.")

    stitched_path = stitched_output_path(slice_dir)
    if not stitched_path.is_file():
        raise FileNotFoundError(
            f"Stitched model not found: {stitched_path}\n"
            "Run Stitch Boundaries first."
        )

    if progress:
        progress("Loading stitched model", 0, 3)

    stitched_mesh = _load_mesh(stitched_path)

    if progress:
        progress("Building block and subtracting stitched model", 1, 3)

    negative_mesh, box_extents = subtract_stitched_from_box(stitched_mesh, margin_mm)

    output_path = negative_output_path(slice_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("Saving mold negative", 2, 3)

    negative_mesh.export(output_path)

    if progress:
        progress("Mold negative export complete", 3, 3)

    return NegativeResult(
        output_path=output_path,
        margin_mm=margin_mm,
        box_extents_mm=box_extents,
        triangle_count=int(len(negative_mesh.faces)),
    )
