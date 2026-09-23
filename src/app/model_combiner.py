"""Combine stitched boundary models from all slice axes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import trimesh

from app.boundary_stitcher import stitched_output_path
from app.slice_axis import SliceAxis, combined_model_path, slice_dir_for_axis

ProgressCallback = Callable[[str, int, int], None]

ALL_AXES = (SliceAxis.Z, SliceAxis.X, SliceAxis.Y)


@dataclass
class CombineResult:
    output_path: Path
    axis_paths: dict[str, Path]
    triangle_count: int


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )
    return mesh


def combine_axis_models(
    stl_path: Path,
    progress: ProgressCallback | None = None,
) -> CombineResult:
    """Merge stitched Z, X, and Y boundary models into one STL."""
    missing: list[str] = []
    axis_paths: dict[str, Path] = {}

    for index, axis in enumerate(ALL_AXES):
        slice_dir = slice_dir_for_axis(stl_path, axis)
        stitched_path = stitched_output_path(slice_dir)
        if stitched_path.is_file():
            axis_paths[axis.value] = stitched_path
        else:
            missing.append(axis.value)

        if progress:
            progress(f"Checking {axis.value}-axis stitch", index, len(ALL_AXES))

    if missing:
        raise FileNotFoundError(
            "Missing stitched model(s) for axis: "
            + ", ".join(missing)
            + ".\nRun Create Boundaries and Stitch for each axis first."
        )

    if progress:
        progress("Merging axis models", len(ALL_AXES), len(ALL_AXES))

    parts = [_load_mesh(axis_paths[axis.value]) for axis in ALL_AXES]
    combined = trimesh.util.concatenate(parts)
    combined.merge_vertices()
    combined.update_faces(combined.unique_faces())
    combined.remove_unreferenced_vertices()

    output_path = combined_model_path(stl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path)

    return CombineResult(
        output_path=output_path,
        axis_paths=axis_paths,
        triangle_count=int(len(combined.faces)),
    )
