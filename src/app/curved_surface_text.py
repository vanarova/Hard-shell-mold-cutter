"""Curved surface text engraving for mold meshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from app.boundary_generator import ProgressCallback

_MIN_FONT_SIZE_MM = 0.5
_DEPTH_RATIO = 0.08
_MIN_DEPTH_MM = 0.2
_MAX_DEPTH_MM = 2.0
_RAY_SEARCH_MM = 100.0


@dataclass(frozen=True)
class TextPlacement:
    origin: tuple[float, float, float]
    tangent: tuple[float, float, float]
    normal: tuple[float, float, float]


@dataclass
class SurfaceTextResult:
    output_dir: Path
    text_path: Path
    combined_path: Path | None
    triangle_count: int


def text_output_path(output_dir: Path) -> Path:
    return output_dir / "text_engraved.stl"


def combined_text_output_path(output_dir: Path) -> Path:
    return output_dir / "text_combined.stl"


def default_depth_mm(font_size_mm: float) -> float:
    depth = font_size_mm * _DEPTH_RATIO
    return float(min(max(depth, _MIN_DEPTH_MM), _MAX_DEPTH_MM))


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("Zero-length vector.")
    return vector / length


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - np.dot(vector, normal) * normal


def _fallback_tangent(normal: np.ndarray) -> np.ndarray:
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    best = max(axes, key=lambda axis: float(np.linalg.norm(_project_to_plane(axis, normal))))
    tangent = _project_to_plane(best, normal)
    return _normalize(tangent)


def _prepare_target_mesh(mesh: pv.PolyData) -> pv.PolyData:
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError("Target mesh is empty.")
    if surface.cell_normals is None or len(surface.cell_normals) == 0:
        surface = surface.compute_normals(
            cell_normals=True,
            point_normals=True,
            inplace=False,
        )
    return surface


def placement_on_outer_surface(
    mesh: pv.PolyData,
    horizontal_offset_mm: float = 0.0,
    vertical_offset_mm: float = 0.0,
) -> TextPlacement:
    """Place text on the outer vertical wall of a mesh using ray casting."""
    target = _prepare_target_mesh(mesh)
    bounds = target.bounds
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    z_min, z_max = bounds[4], bounds[5]
    z = 0.5 * (z_min + z_max) + float(vertical_offset_mm)

    x_extent = bounds[1] - bounds[0]
    y_extent = bounds[3] - bounds[2]
    margin = max(x_extent, y_extent, 1.0) * 0.25 + 25.0

    if y_extent >= x_extent:
        ray_start = np.array([cx + horizontal_offset_mm, bounds[3] + margin, z], dtype=float)
        ray_end = np.array([cx + horizontal_offset_mm, bounds[2] - margin, z], dtype=float)
        text_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ray_start = np.array([bounds[1] + margin, cy + horizontal_offset_mm, z], dtype=float)
        ray_end = np.array([bounds[0] - margin, cy + horizontal_offset_mm, z], dtype=float)
        text_axis = np.array([0.0, 1.0, 0.0], dtype=float)

    hits, _ = target.ray_trace(ray_start, ray_end, first_point=True)
    if len(hits) > 0:
        hit = np.asarray(hits[0], dtype=float)
    else:
        hit = np.array([cx + horizontal_offset_mm, cy, z], dtype=float)
        closest_idx = int(target.find_closest_point(hit))
        hit = np.asarray(target.points[closest_idx], dtype=float)

    cell_id = int(target.find_closest_cell(hit))
    if cell_id < 0:
        cell_id = int(target.find_closest_cell(ray_start))
    normal = _normalize(np.asarray(target.cell_normals[cell_id], dtype=float))

    tangent = _project_to_plane(text_axis, normal)
    if float(np.linalg.norm(tangent)) < 1e-6:
        tangent = _fallback_tangent(normal)
    else:
        tangent = _normalize(tangent)

    return TextPlacement(
        origin=(float(hit[0]), float(hit[1]), float(hit[2])),
        tangent=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
    )


def surface_frame_at_point(
    mesh: pv.PolyData,
    point: tuple[float, float, float] | np.ndarray,
    direction_hint: tuple[float, float, float] | np.ndarray | None = None,
) -> TextPlacement:
    """Compute a surface-aligned frame (origin, tangent, normal) at a picked point."""
    target = _prepare_target_mesh(mesh)
    query = np.asarray(point, dtype=float)

    closest_idx = int(target.find_closest_point(query))
    origin = np.asarray(target.points[closest_idx], dtype=float)

    cell_id = int(target.find_closest_cell(origin))
    if cell_id < 0:
        cell_id = int(target.find_closest_cell(query))
    normal = _normalize(np.asarray(target.cell_normals[cell_id], dtype=float))

    if direction_hint is not None:
        hint = np.asarray(direction_hint, dtype=float) - origin
        tangent = _project_to_plane(hint, normal)
        if float(np.linalg.norm(tangent)) < 1e-6:
            tangent = _fallback_tangent(normal)
        else:
            tangent = _normalize(tangent)
    else:
        tangent = _fallback_tangent(normal)

    return TextPlacement(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        tangent=(float(tangent[0]), float(tangent[1]), float(tangent[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
    )


def generate_text_mesh(text: str, font_size_mm: float, depth_mm: float) -> pv.PolyData:
    """Create flat 3D text with the baseline corner at the local origin."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text must not be empty.")
    if font_size_mm < _MIN_FONT_SIZE_MM:
        raise ValueError(f"Font size must be at least {_MIN_FONT_SIZE_MM} mm.")

    text_mesh = pv.Text3D(
        cleaned,
        height=float(font_size_mm),
        depth=float(depth_mm),
        center=(0.0, 0.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    ).triangulate()

    # Align so the front face sits on Z=0 and the text starts at X=0, Y=0.
    bounds = text_mesh.bounds
    text_mesh.translate((-bounds[0], -bounds[2], -bounds[4]), inplace=True)
    return text_mesh


def _frame_rotation_matrix(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    tangent_unit = _normalize(np.asarray(tangent, dtype=float))
    normal_unit = _normalize(np.asarray(normal, dtype=float))
    bitangent = _normalize(np.cross(normal_unit, tangent_unit))
    tangent_unit = _normalize(np.cross(bitangent, normal_unit))
    return np.column_stack([tangent_unit, bitangent, normal_unit])


def _orient_text_mesh(text_mesh: pv.PolyData, placement: TextPlacement) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    rotation = _frame_rotation_matrix(
        np.asarray(placement.tangent),
        np.asarray(placement.normal),
    )
    origin = np.asarray(placement.origin, dtype=float)
    oriented = text_mesh.copy(deep=True)
    oriented.points = text_mesh.points @ rotation.T + origin
    return oriented, rotation, origin


def _surface_point_near(
    mesh: pv.PolyData,
    base: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normal_unit = _normalize(np.asarray(normal, dtype=float))
    start = base + normal_unit * _RAY_SEARCH_MM
    end = base - normal_unit * _RAY_SEARCH_MM
    hits, _ = mesh.ray_trace(start, end, first_point=True)
    if len(hits) > 0:
        return np.asarray(hits[0], dtype=float), normal_unit

    closest_idx = int(mesh.find_closest_point(base))
    point = np.asarray(mesh.points[closest_idx], dtype=float)
    if mesh.point_normals is not None and len(mesh.point_normals) > closest_idx:
        hit_normal = _normalize(np.asarray(mesh.point_normals[closest_idx], dtype=float))
    else:
        hit_normal = normal_unit
    return point, hit_normal


def conform_text_to_surface(
    text_mesh: pv.PolyData,
    target_mesh: pv.PolyData,
    placement: TextPlacement,
    depth_mm: float,
) -> pv.PolyData:
    """Orient flat text at the placement frame, then wrap vertices onto the surface."""
    _ = depth_mm  # reserved for future depth-aware conform tuning
    target = _prepare_target_mesh(target_mesh)
    oriented, rotation, origin = _orient_text_mesh(text_mesh, placement)

    tangent = _normalize(np.asarray(placement.tangent, dtype=float))
    normal = _normalize(np.asarray(placement.normal, dtype=float))
    bitangent = _normalize(np.cross(normal, tangent))

    points = oriented.points.copy()
    for index, point in enumerate(points):
        local = rotation.T @ (point - origin)
        u, v, depth_offset = local
        base = origin + u * tangent + v * bitangent
        surface_point, surface_normal = _surface_point_near(target, base, normal)
        points[index] = surface_point + surface_normal * depth_offset

    conformed = oriented.copy(deep=True)
    conformed.points = points
    return conformed.triangulate()


def marker_sphere(
    center: tuple[float, float, float],
    radius_mm: float = 1.0,
) -> pv.PolyData:
    return pv.Sphere(radius=radius_mm, center=center, theta_resolution=16, phi_resolution=16)


def direction_line(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> pv.PolyData:
    return pv.Line(np.asarray(start, dtype=float), np.asarray(end, dtype=float))


def _load_source_mesh(
    source_mesh_path: Path | pv.PolyData,
    progress: ProgressCallback | None,
) -> pv.PolyData:
    if isinstance(source_mesh_path, pv.PolyData):
        return _prepare_target_mesh(source_mesh_path)

    if progress is not None:
        progress("Reading source mesh", 0, 4)
    if not source_mesh_path.is_file():
        raise FileNotFoundError(f"Source mesh not found: {source_mesh_path}")
    mesh = pv.read(str(source_mesh_path))
    return _prepare_target_mesh(mesh)


def _merge_meshes(source: pv.PolyData, text: pv.PolyData) -> trimesh.Trimesh:
    source_tm = trimesh.Trimesh(
        vertices=np.asarray(source.points),
        faces=np.asarray(source.faces.reshape(-1, 4)[:, 1:]),
        process=False,
    )
    text_tm = trimesh.Trimesh(
        vertices=np.asarray(text.points),
        faces=np.asarray(text.faces.reshape(-1, 4)[:, 1:]),
        process=False,
    )
    combined = trimesh.util.concatenate([source_tm, text_tm])
    combined.merge_vertices()
    combined.update_faces(combined.unique_faces() & combined.nondegenerate_faces())
    combined.remove_unreferenced_vertices()
    return combined


def build_text_on_surface(
    source_mesh_path: Path | pv.PolyData,
    text: str,
    placement: TextPlacement,
    font_size_mm: float,
    depth_mm: float | None = None,
) -> pv.PolyData:
    """Generate surface-conformed text without writing files."""
    emboss_depth = default_depth_mm(font_size_mm) if depth_mm is None else float(depth_mm)
    source = _load_source_mesh(source_mesh_path, None)
    text_mesh = generate_text_mesh(text, font_size_mm, emboss_depth)
    return conform_text_to_surface(text_mesh, source, placement, emboss_depth)


def write_text_on_surface(
    source_mesh_path: Path | pv.PolyData,
    text: str,
    placement: TextPlacement,
    font_size_mm: float,
    output_path: Path,
    depth_mm: float | None = None,
    merge_with_source: bool = True,
    progress: ProgressCallback | None = None,
) -> SurfaceTextResult:
    """Generate surface-conformed text and export STL output next to the source model."""
    if font_size_mm < _MIN_FONT_SIZE_MM:
        raise ValueError(f"Font size must be at least {_MIN_FONT_SIZE_MM} mm.")

    emboss_depth = default_depth_mm(font_size_mm) if depth_mm is None else float(depth_mm)
    if emboss_depth <= 0:
        raise ValueError("Text depth must be greater than 0 mm.")

    source = _load_source_mesh(source_mesh_path, progress)

    if progress is not None:
        progress("Generating text mesh", 1, 4)

    text_mesh = generate_text_mesh(text, font_size_mm, emboss_depth)

    if progress is not None:
        progress("Conforming text to surface", 2, 4)

    conformed = conform_text_to_surface(text_mesh, source, placement, emboss_depth)

    if progress is not None:
        progress("Saving text mesh", 3, 4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conformed.save(str(output_path))

    combined_path: Path | None = None
    if merge_with_source:
        combined_path = combined_text_output_path(output_path.parent)
        combined = _merge_meshes(source, conformed)
        combined.export(combined_path)

    if progress is not None:
        progress("Surface text export complete", 4, 4)

    return SurfaceTextResult(
        output_dir=output_path.parent,
        text_path=output_path,
        combined_path=combined_path,
        triangle_count=int(conformed.n_cells),
    )
