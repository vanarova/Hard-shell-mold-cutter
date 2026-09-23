"""Shared footprint and outline utilities for boundary generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

ProgressCallback = Callable[[str, int, int], None]

# Minimal physical thickness for the boundary wall (not extruded to slice height).
BOUNDARY_WALL_THICKNESS_MM = 0.1


@dataclass
class BoundaryResult:
    output_dir: Path
    boundary_paths: list[Path]
    skipped_slices: int


_SLICE_NAME_RE = re.compile(r"^\d+$")


def list_slice_files(slice_dir: Path) -> list[Path]:
    """Return numbered slice STLs in order (1.stl, 2.stl, ...)."""
    numbered: list[tuple[int, Path]] = []
    for path in slice_dir.glob("*.stl"):
        if _SLICE_NAME_RE.match(path.stem):
            numbered.append((int(path.stem), path))
    return [path for _, path in sorted(numbered)]


def _triangle_polygons(mesh: pv.PolyData) -> list[Polygon]:
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    xy = mesh.points[:, :2]
    polygons: list[Polygon] = []

    for triangle in faces:
        coords = [tuple(xy[index]) for index in triangle]
        if len(set(coords)) < 3:
            continue
        polygon = Polygon(coords)
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)

    return polygons


def _linestrings_from_polydata(lines_mesh: pv.PolyData) -> list[LineString]:
    points = lines_mesh.points[:, :2]
    linestrings: list[LineString] = []
    connectivity = lines_mesh.lines
    index = 0

    while index < len(connectivity):
        count = connectivity[index]
        indices = connectivity[index + 1 : index + 1 + count]
        coords = [tuple(points[point_index]) for point_index in indices]
        if len(coords) >= 2:
            linestrings.append(LineString(coords))
        index += count + 1

    return linestrings


def footprint_from_cross_section(mesh: pv.PolyData) -> Polygon | MultiPolygon | None:
    """Build a footprint by slicing the mesh at mid Z height."""
    zmin, zmax = mesh.bounds[4], mesh.bounds[5]
    z_mid = (zmin + zmax) / 2.0
    section = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z_mid))
    if section.n_points == 0:
        return None

    polygons = list(polygonize(_linestrings_from_polydata(section)))
    if not polygons:
        return None

    footprint = unary_union(polygons).buffer(0)
    if footprint.is_empty:
        return None
    return footprint


def footprint_from_mesh(mesh: pv.PolyData) -> Polygon | MultiPolygon | None:
    """Build a 2D XY footprint from all geometry in a slice."""
    parts: list[Polygon | MultiPolygon] = []

    cross_section = footprint_from_cross_section(mesh)
    if cross_section is not None:
        parts.append(cross_section)

    triangle_parts = _triangle_polygons(mesh)
    if triangle_parts:
        parts.append(unary_union(triangle_parts))

    if not parts:
        return None

    combined = unary_union(parts).buffer(0)
    if combined.is_empty:
        return None
    return combined


def offset_outline(footprint: Polygon | MultiPolygon, offset_mm: float) -> Polygon | MultiPolygon:
    """
    Create a single equidistant outer outline around all objects in the slice.

    All separate bodies are merged first so the boundary wraps the full scene
    without crossing through objects or intersecting itself.
    """
    if offset_mm <= 0:
        raise ValueError("Boundary offset must be greater than 0 mm.")

    combined = unary_union(footprint).buffer(0)
    if combined.is_empty:
        raise ValueError("Slice footprint is empty.")

    # Multiple separate bodies: wrap them with one continuous outer outline.
    if isinstance(combined, MultiPolygon) and len(combined.geoms) > 1:
        source = combined.convex_hull
    else:
        source = combined

    outline = source.buffer(
        offset_mm,
        join_style=1,
        resolution=32,
    )
    if outline.is_empty:
        raise ValueError("Could not build offset outline.")

    return outline.buffer(0)
