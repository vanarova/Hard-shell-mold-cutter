"""Boundary generation v2 — offset in slice plane, stack depth from previous layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from app.boundary_generator import (
    BOUNDARY_WALL_THICKNESS_MM,
    BoundaryResult,
    ProgressCallback,
    footprint_from_mesh,
    list_slice_files,
    offset_outline,
)
from app.slice_axis import SliceAxis, transform_polydata, transform_trimesh

_LOOP_POINTS = 128


@dataclass
class BoundaryV2PlacementStats:
    total_points: int = 0
    valid_points: int = 0


@dataclass
class BoundaryPointValidation:
    xy_ok: bool
    z_stack_ok: bool
    xy_error_mm: float
    z_stack_error_mm: float

    @property
    def ok(self) -> bool:
        return self.xy_ok and self.z_stack_ok


@dataclass
class BoundaryV2Result(BoundaryResult):
    axis: SliceAxis = SliceAxis.Z
    previous_layer_count: int = 1
    placement_stats: BoundaryV2PlacementStats = field(
        default_factory=BoundaryV2PlacementStats
    )


def boundaries_v2_output_dir(slice_dir: Path) -> Path:
    return slice_dir / "boundaries_v2"


def previous_layer_count(offset_mm: float, slice_height_mm: float) -> int:
    """How many slice layers fit in the boundary offset (minimum 1)."""
    if slice_height_mm <= 0:
        raise ValueError("Slice height must be greater than 0 mm.")
    return max(1, int(offset_mm / slice_height_mm))


def _resample_loop(loop: np.ndarray, count: int = _LOOP_POINTS) -> np.ndarray:
    closed = np.vstack([loop, loop[0]])
    line = LineString(closed)
    if line.length <= 0:
        return loop

    distances = np.linspace(0.0, line.length, count, endpoint=False)
    return np.array([line.interpolate(distance).coords[0] for distance in distances], dtype=float)


def _outline_xy_loop(outline) -> np.ndarray:
    if isinstance(outline, Polygon):
        polygon = outline
    else:
        polygon = max(outline.geoms, key=lambda shape: shape.area)

    coords = np.array(polygon.exterior.coords[:-1], dtype=float)
    if len(coords) < 3:
        raise ValueError("Boundary outline is too small.")
    return _resample_loop(coords)


def _local_z_top(mesh: pv.PolyData, x: float, y: float, search_radius: float) -> float:
    points = mesh.points
    distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
    mask = distances <= search_radius
    if np.any(mask):
        return float(points[mask, 2].max())
    return float(mesh.bounds[5])


def _local_z_bottom(mesh: pv.PolyData, x: float, y: float, search_radius: float) -> float:
    points = mesh.points
    distances = np.hypot(points[:, 0] - x, points[:, 1] - y)
    mask = distances <= search_radius
    if np.any(mask):
        return float(points[mask, 2].min())
    return float(mesh.bounds[4])


def _tolerance(offset_mm: float) -> float:
    return max(0.02, offset_mm * 0.05)


def _footprint_shape(footprint: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    if isinstance(footprint, (Polygon, MultiPolygon)):
        return footprint
    return unary_union(footprint)


def _xy_distance_to_current_footprint(
    footprint: Polygon | MultiPolygon,
    x: float,
    y: float,
) -> float:
    shape = _footprint_shape(footprint)
    return float(Point(x, y).distance(shape.boundary))


def _local_z_topmost_from_layers(
    previous_meshes: list[pv.PolyData],
    x: float,
    y: float,
    search_radius: float,
) -> float:
    return max(_local_z_top(mesh, x, y, search_radius) for mesh in previous_meshes)


def _compute_z_from_previous_layers(
    current_mesh: pv.PolyData,
    previous_meshes: list[pv.PolyData],
    x: float,
    y: float,
    offset_mm: float,
    search_radius: float,
) -> float:
    if not previous_meshes:
        return _local_z_bottom(current_mesh, x, y, search_radius) + offset_mm

    z_topmost = _local_z_topmost_from_layers(previous_meshes, x, y, search_radius)
    return z_topmost + offset_mm


def validate_boundary_point(
    footprint: Polygon | MultiPolygon,
    previous_meshes: list[pv.PolyData],
    x: float,
    y: float,
    z: float,
    offset_mm: float,
    search_radius: float,
) -> BoundaryPointValidation:
    tol = _tolerance(offset_mm)
    xy_dist = _xy_distance_to_current_footprint(footprint, x, y)
    xy_error = abs(xy_dist - offset_mm)
    xy_ok = xy_error <= tol

    if not previous_meshes:
        z_stack_error = 0.0
        z_stack_ok = True
    else:
        z_topmost = _local_z_topmost_from_layers(previous_meshes, x, y, search_radius)
        z_stack_error = max(0.0, offset_mm - (z - z_topmost))
        z_stack_ok = z_stack_error <= tol

    return BoundaryPointValidation(
        xy_ok=xy_ok,
        z_stack_ok=z_stack_ok,
        xy_error_mm=xy_error,
        z_stack_error_mm=z_stack_error,
    )


def z_levels_for_boundary_v2(
    current_mesh: pv.PolyData,
    previous_meshes: list[pv.PolyData],
    footprint: Polygon | MultiPolygon,
    loop_xy: np.ndarray,
    offset_mm: float,
) -> tuple[np.ndarray, BoundaryV2PlacementStats]:
    search_radius = max(offset_mm * 2.0, 1.0)
    z_values = np.zeros(len(loop_xy), dtype=float)
    stats = BoundaryV2PlacementStats(total_points=len(loop_xy))

    for index, (x, y) in enumerate(loop_xy):
        z = _compute_z_from_previous_layers(
            current_mesh,
            previous_meshes,
            x,
            y,
            offset_mm,
            search_radius,
        )
        validation = validate_boundary_point(
            footprint,
            previous_meshes,
            x,
            y,
            z,
            offset_mm,
            search_radius,
        )

        if validation.ok:
            stats.valid_points += 1

        z_values[index] = z

    return z_values, stats


def thin_wall_v2_from_loop(
    loop_xy: np.ndarray,
    z_values: np.ndarray,
    wall_thickness_mm: float = BOUNDARY_WALL_THICKNESS_MM,
) -> trimesh.Trimesh:
    half = wall_thickness_mm / 2.0
    half_z = wall_thickness_mm / 2.0
    count = len(loop_xy)

    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    for index in range(count):
        next_index = (index + 1) % count
        p0 = loop_xy[index]
        p1 = loop_xy[next_index]
        z0 = float(z_values[index])
        z1 = float(z_values[next_index])

        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            normal = np.array([1.0, 0.0])
        else:
            tangent = edge / length
            normal = np.array([-tangent[1], tangent[0]])

        offset = normal * half
        a0 = p0 - offset
        a1 = p0 + offset
        b0 = p1 - offset
        b1 = p1 + offset

        base = len(vertices)
        vertices.extend(
            [
                [float(a0[0]), float(a0[1]), z0 - half_z],
                [float(a1[0]), float(a1[1]), z0 - half_z],
                [float(a1[0]), float(a1[1]), z0 + half_z],
                [float(a0[0]), float(a0[1]), z0 + half_z],
                [float(b0[0]), float(b0[1]), z1 - half_z],
                [float(b1[0]), float(b1[1]), z1 - half_z],
                [float(b1[0]), float(b1[1]), z1 + half_z],
                [float(b0[0]), float(b0[1]), z1 + half_z],
            ]
        )
        faces.extend(
            [
                [base + 0, base + 1, base + 2],
                [base + 0, base + 2, base + 3],
                [base + 4, base + 6, base + 5],
                [base + 4, base + 7, base + 6],
                [base + 0, base + 4, base + 5],
                [base + 0, base + 5, base + 1],
                [base + 2, base + 6, base + 7],
                [base + 2, base + 7, base + 3],
                [base + 1, base + 5, base + 6],
                [base + 1, base + 6, base + 2],
                [base + 0, base + 3, base + 7],
                [base + 0, base + 7, base + 4],
            ]
        )

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def create_boundary_v2_for_slice(
    current_slice_path: Path,
    previous_slice_paths: list[Path],
    output_path: Path,
    offset_mm: float,
    axis: SliceAxis = SliceAxis.Z,
) -> tuple[bool, BoundaryV2PlacementStats]:
    empty_stats = BoundaryV2PlacementStats()
    current_mesh = transform_polydata(
        pv.read(str(current_slice_path)),
        axis,
        to_canonical=True,
    )
    previous_meshes = [
        transform_polydata(pv.read(str(path)), axis, to_canonical=True)
        for path in previous_slice_paths
    ]

    footprint = footprint_from_mesh(current_mesh)
    if footprint is None:
        return False, empty_stats

    outline = offset_outline(footprint, offset_mm)
    loop_xy = _outline_xy_loop(outline)
    z_values, stats = z_levels_for_boundary_v2(
        current_mesh,
        previous_meshes,
        footprint,
        loop_xy,
        offset_mm,
    )
    boundary_mesh = thin_wall_v2_from_loop(loop_xy, z_values)
    boundary_mesh = transform_trimesh(boundary_mesh, axis, to_canonical=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary_mesh.export(output_path)
    return True, stats


def create_boundaries_v2_for_slices(
    slice_dir: Path,
    offset_mm: float,
    slice_height_mm: float,
    axis: SliceAxis = SliceAxis.Z,
    progress: ProgressCallback | None = None,
) -> BoundaryV2Result:
    if offset_mm <= 0:
        raise ValueError("Boundary offset must be greater than 0 mm.")
    if not slice_dir.is_dir():
        raise FileNotFoundError(f"Slice folder not found: {slice_dir}")

    layer_count = previous_layer_count(offset_mm, slice_height_mm)
    slice_files = list_slice_files(slice_dir)
    if not slice_files:
        raise FileNotFoundError(
            f"No numbered slice files found in: {slice_dir}\n"
            "Run Slice first to create 1.stl, 2.stl, ..."
        )

    output_dir = boundaries_v2_output_dir(slice_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary_paths: list[Path] = []
    skipped = 0
    placement_stats = BoundaryV2PlacementStats()
    total = len(slice_files)

    if progress:
        progress(
            f"Preparing {axis.label}-axis boundaries ({layer_count} previous layer(s))",
            0,
            total,
        )

    for step, slice_path in enumerate(slice_files, start=1):
        previous_paths = slice_files[max(0, step - 1 - layer_count) : step - 1]
        if progress:
            if previous_paths:
                first = previous_paths[0].stem
                last = previous_paths[-1].stem
                label = (
                    f"{axis.label} boundary {slice_path.stem}.stl "
                    f"(layers {first}–{last}, {step}/{total})"
                )
            else:
                label = f"{axis.label} boundary {slice_path.stem}.stl ({step}/{total})"
            progress(label, step - 1, total)

        output_path = output_dir / f"{slice_path.stem}.stl"
        try:
            created, slice_stats = create_boundary_v2_for_slice(
                slice_path,
                previous_paths,
                output_path,
                offset_mm,
                axis=axis,
            )
        except Exception:
            created = False
            slice_stats = BoundaryV2PlacementStats()

        if created:
            boundary_paths.append(output_path)
            placement_stats.total_points += slice_stats.total_points
            placement_stats.valid_points += slice_stats.valid_points
        else:
            skipped += 1

    if progress:
        progress(f"{axis.label}-axis boundary export complete", total, total)

    return BoundaryV2Result(
        output_dir=output_dir,
        boundary_paths=boundary_paths,
        skipped_slices=skipped,
        axis=axis,
        previous_layer_count=layer_count,
        placement_stats=placement_stats,
    )
