"""Unified lofted boundary stitching — one continuous shell, no overlapping layers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

from app.boundary_generator import list_slice_files
from app.boundary_generator_v2 import boundaries_v2_output_dir
from app.slice_axis import SliceAxis, transform_polydata, transform_trimesh

ProgressCallback = Callable[[str, int, int], None]

_NUMBER_RE = re.compile(r"^\d+$")
_LOOP_SEGMENTS = 256
_EXCLUDED_STEMS = frozenset({"stitched", "stitched_v2", "negative"})


@dataclass
class StitchResult:
    output_path: Path
    layers_stitched: int
    ring_count: int
    triangle_count: int


def list_boundary_files(boundary_dir: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for path in boundary_dir.glob("*.stl"):
        if _NUMBER_RE.match(path.stem) and path.stem not in _EXCLUDED_STEMS:
            numbered.append((int(path.stem), path))
    return [path for _, path in sorted(numbered)]


def stitched_output_path(slice_dir: Path) -> Path:
    return boundaries_v2_output_dir(slice_dir) / "stitched.stl"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )
    return mesh


def resample_loop(loop: np.ndarray, count: int = _LOOP_SEGMENTS) -> np.ndarray:
    closed = np.vstack([loop, loop[0]])
    line = LineString(closed)
    if line.length <= 0:
        return loop

    distances = np.linspace(0.0, line.length, count, endpoint=False)
    return np.array([line.interpolate(distance).coords[0] for distance in distances], dtype=float)


def _loop_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).sum())


def align_loops(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    best_shift = 0
    best_score = float("inf")

    for shift in range(len(target)):
        rolled = np.roll(target, -shift, axis=0)
        score = _loop_distance(reference, rolled)
        if score < best_score:
            best_score = score
            best_shift = shift

    return np.roll(target, -best_shift, axis=0)


def _linestrings_from_section(section: pv.PolyData) -> list[LineString]:
    points = section.points[:, :2]
    linestrings: list[LineString] = []
    connectivity = section.lines
    index = 0

    while index < len(connectivity):
        count = connectivity[index]
        indices = connectivity[index + 1 : index + 1 + count]
        coords = [tuple(points[point_index]) for point_index in indices]
        if len(coords) >= 2:
            linestrings.append(LineString(coords))
        index += count + 1

    return linestrings


def extract_xy_loop(mesh: trimesh.Trimesh, z_hint: float) -> np.ndarray:
    """Extract the outer XY outline from a boundary mesh."""
    face_sizes = np.full((len(mesh.faces), 1), 3, dtype=np.int64)
    pv_mesh = pv.PolyData(mesh.vertices, np.hstack([face_sizes, mesh.faces]))

    section = pv_mesh.slice(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, z_hint))
    if section.n_points > 0:
        polygons = list(polygonize(_linestrings_from_section(section)))
        if polygons:
            largest = max(polygons, key=lambda poly: poly.area)
            coords = np.array(largest.exterior.coords[:-1], dtype=float)
            if len(coords) >= 3:
                return coords

    xy = mesh.vertices[:, :2]
    hull = Polygon(xy).convex_hull
    return np.array(hull.exterior.coords[:-1], dtype=float)


def _triangulate_cap(loop: np.ndarray, z: float, reverse: bool) -> tuple[np.ndarray, np.ndarray]:
    polygon = Polygon(loop.astype(float))
    if polygon.is_empty:
        raise ValueError("Cannot triangulate empty cap.")

    vertices_2d, faces = trimesh.creation.triangulate_polygon(
        polygon,
        engine="earcut",
    )
    vertices_3d = np.column_stack(
        [vertices_2d, np.full(len(vertices_2d), z, dtype=float)]
    )
    if reverse:
        faces = faces[:, ::-1]
    return vertices_3d, faces


def _build_ring_stack(
    layer_loops: list[np.ndarray],
    z_bottoms: list[float],
    z_tops: list[float],
    loop_segments: int = _LOOP_SEGMENTS,
) -> tuple[list[float], list[np.ndarray]]:
    """Build shared Z levels and one XY ring per level."""
    layer_count = len(layer_loops)
    if layer_count == 0:
        raise ValueError("No boundary layers to stitch.")

    z_levels: list[float] = [float(z_bottoms[0])]
    rings: list[np.ndarray] = [resample_loop(layer_loops[0], loop_segments)]

    for index in range(layer_count - 1):
        z_interface = float(z_tops[index])
        z_levels.append(z_interface)

        lower = resample_loop(layer_loops[index], loop_segments)
        upper = align_loops(lower, resample_loop(layer_loops[index + 1], loop_segments))
        rings.append(lower * 0.5 + upper * 0.5)

    z_levels.append(float(z_tops[-1]))
    rings.append(resample_loop(layer_loops[-1], loop_segments))

    reference = rings[0]
    aligned_rings = [reference]
    for ring in rings[1:]:
        aligned_rings.append(align_loops(reference, ring))

    return z_levels, aligned_rings


def build_unified_lofted_mesh(
    z_levels: list[float],
    rings: list[np.ndarray],
    loop_segments: int = _LOOP_SEGMENTS,
) -> trimesh.Trimesh:
    """Loft aligned rings into one watertight side surface with end caps."""
    if len(z_levels) != len(rings):
        raise ValueError("Z levels and rings must have the same length.")
    if len(z_levels) < 2:
        raise ValueError("At least two rings are required.")

    count = len(rings[0])
    ring_count = len(z_levels)

    side_vertices: list[list[float]] = []
    side_faces: list[list[int]] = []

    for ring_index, z in enumerate(z_levels):
        ring = rings[ring_index]
        for point in ring:
            side_vertices.append([float(point[0]), float(point[1]), float(z)])

    for ring_index in range(ring_count - 1):
        lower_row = ring_index * count
        upper_row = (ring_index + 1) * count
        for point_index in range(count):
            next_point = (point_index + 1) % count
            v00 = lower_row + point_index
            v01 = lower_row + next_point
            v10 = upper_row + point_index
            v11 = upper_row + next_point
            side_faces.append([v00, v10, v11])
            side_faces.append([v00, v11, v01])

    vertices_parts = [np.array(side_vertices, dtype=float)]
    faces_parts = [np.array(side_faces, dtype=int)]
    offset = len(side_vertices)

    bottom_vertices, bottom_faces = _triangulate_cap(rings[0], z_levels[0], reverse=True)
    faces_parts.append(bottom_faces + offset)
    vertices_parts.append(bottom_vertices)
    offset += len(bottom_vertices)

    top_vertices, top_faces = _triangulate_cap(rings[-1], z_levels[-1], reverse=False)
    faces_parts.append(top_faces + offset)
    vertices_parts.append(top_vertices)

    mesh = trimesh.Trimesh(
        vertices=np.vstack(vertices_parts),
        faces=np.vstack(faces_parts),
        process=True,
    )
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def stitch_boundaries(
    slice_dir: Path,
    axis: SliceAxis = SliceAxis.Z,
    progress: ProgressCallback | None = None,
) -> StitchResult:
    """Stitch numbered boundaries into one unified lofted model."""
    boundary_dir = boundaries_v2_output_dir(slice_dir)
    if not boundary_dir.is_dir():
        raise FileNotFoundError(f"Boundary folder not found: {boundary_dir}")

    boundary_files = [
        path
        for path in list_boundary_files(boundary_dir)
        if path.stem not in _EXCLUDED_STEMS
    ]
    if not boundary_files:
        raise FileNotFoundError(
            f"No numbered boundary files found in: {boundary_dir}\n"
            "Run Create Boundaries first."
        )

    slice_by_name = {path.stem: path for path in list_slice_files(slice_dir)}
    layer_loops: list[np.ndarray] = []
    z_bottoms: list[float] = []
    z_tops: list[float] = []
    total = len(boundary_files)

    if progress:
        progress(f"Loading {axis.label}-axis boundaries for stitch", 0, total)

    for index, boundary_path in enumerate(boundary_files):
        slice_path = slice_by_name.get(boundary_path.stem)
        if slice_path is None:
            raise FileNotFoundError(
                f"Missing matching slice file for boundary: {boundary_path.name}"
            )

        boundary_mesh = transform_trimesh(
            _load_mesh(boundary_path),
            axis,
            to_canonical=True,
        )
        slice_mesh = transform_polydata(
            pv.wrap(_load_mesh(slice_path)),
            axis,
            to_canonical=True,
        )
        z_bottom = float(slice_mesh.bounds[4])
        z_top = float(slice_mesh.bounds[5])
        z_mid = (z_bottom + z_top) / 2.0
        loop = extract_xy_loop(boundary_mesh, z_hint=z_mid)

        layer_loops.append(loop)
        z_bottoms.append(z_bottom)
        z_tops.append(z_top)

        if progress:
            progress(
                f"Loaded {axis.label} layer {boundary_path.stem}.stl ({index + 1}/{total})",
                index + 1,
                total,
            )

    if progress:
        progress(f"Building {axis.label}-axis ring stack", total, total)

    z_levels, rings = _build_ring_stack(layer_loops, z_bottoms, z_tops)

    if progress:
        progress(f"Lofting {axis.label}-axis shell", total, total)

    combined = build_unified_lofted_mesh(z_levels, rings)
    combined = transform_trimesh(combined, axis, to_canonical=False)

    output_path = stitched_output_path(slice_dir)
    combined.export(output_path)

    return StitchResult(
        output_path=output_path,
        layers_stitched=len(layer_loops),
        ring_count=len(z_levels),
        triangle_count=int(len(combined.faces)),
    )
