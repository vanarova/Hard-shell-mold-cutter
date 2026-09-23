"""Wrap / extrude a start-plane grid onto the model toward a deformed end plane."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from app.plane_deform import sample_deformed_plane_grid
from app.xy_plane import plane_normal_from_rotations

ProgressCallback = Callable[[str, int, int], None]

_RAY_EPSILON = 1e-6
_PLANE_MARGIN_MM = 2.0
DEFAULT_POINT_DENSITY = 100
MIN_POINT_DENSITY = 4
MAX_POINT_DENSITY = 1000
MAX_WRAP_BACK_DENSITY = 316  # 316² ≈ 99,856 ≤ 100,000


@dataclass(frozen=True)
class WrapPlaneResult:
    wrap_mesh: pv.PolyData
    output_path: Path
    point_count: int
    hit_model_count: int
    hit_end_count: int
    density: int


def wrap_plane_output_path(output_dir: Path) -> Path:
    return output_dir / "wrap_plane.stl"


def wrap_back_output_path(output_dir: Path) -> Path:
    return output_dir / "wrap_back_plane.stl"


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vector / length


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = _normalize(np.asarray(normal, dtype=float))
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(n[0])) > 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
    i_axis = _normalize(np.cross(helper, n))
    j_axis = _normalize(np.cross(n, i_axis))
    return i_axis, j_axis


def _plane_half_extents(mesh: pv.PolyData, margin_mm: float = _PLANE_MARGIN_MM) -> float:
    from app.plane_deform import plane_half_extent

    return plane_half_extent(mesh, margin_mm=margin_mm)


def _polydata_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    surface = mesh
    if not surface.is_all_triangles:
        surface = surface.triangulate()
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(
        vertices=np.asarray(surface.points, dtype=float),
        faces=faces,
        process=False,
    )


def _sample_oriented_plane_grid(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_extent: float,
    density: int,
) -> np.ndarray:
    """Return flat plane sample points with shape (density, density, 3)."""
    i_axis, j_axis = _plane_basis(normal)
    u = np.linspace(-half_extent, half_extent, density)
    v = np.linspace(-half_extent, half_extent, density)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return (
        center[None, None, :]
        + uu[:, :, None] * i_axis[None, None, :]
        + vv[:, :, None] * j_axis[None, None, :]
    )


def _sample_deformed_end_grid(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_extent: float,
    density: int,
    end_ctrl_uv: np.ndarray,
    end_ctrl_disp: np.ndarray,
) -> np.ndarray:
    """Sample the curved/deformed end plane on the same parametric grid as start."""
    return sample_deformed_plane_grid(
        center=center,
        normal=normal,
        half_extent=half_extent,
        uv=end_ctrl_uv,
        displacements=end_ctrl_disp,
        density=density,
    )


def _build_extruded_solid(
    bottom: np.ndarray,
    top: np.ndarray,
) -> pv.PolyData:
    """
    Build a solid mesh by extruding the whole start-plane grid to the stopped
    top surface (prism cells between corresponding grid quads).
    """
    if bottom.shape != top.shape or bottom.ndim != 3 or bottom.shape[2] != 3:
        raise ValueError("Bottom and top grids must both have shape (n, n, 3).")

    ny, nx, _ = bottom.shape
    if ny < 2 or nx < 2:
        raise ValueError("Grid density must be at least 2 on each side.")

    n_plane = ny * nx
    vertices = np.vstack([bottom.reshape(-1, 3), top.reshape(-1, 3)])

    def vid(j: int, i: int, layer: int) -> int:
        return layer * n_plane + j * nx + i

    tri_faces: list[list[int]] = []

    for j in range(ny - 1):
        for i in range(nx - 1):
            a = vid(j, i, 0)
            b = vid(j, i + 1, 0)
            c = vid(j + 1, i + 1, 0)
            d = vid(j + 1, i, 0)
            tri_faces.append([a, d, c])
            tri_faces.append([a, c, b])

    for j in range(ny - 1):
        for i in range(nx - 1):
            a = vid(j, i, 1)
            b = vid(j, i + 1, 1)
            c = vid(j + 1, i + 1, 1)
            d = vid(j + 1, i, 1)
            tri_faces.append([a, b, c])
            tri_faces.append([a, c, d])

    for i in range(nx - 1):
        a0, b0 = vid(0, i, 0), vid(0, i + 1, 0)
        a1, b1 = vid(0, i, 1), vid(0, i + 1, 1)
        tri_faces.append([a0, b0, b1])
        tri_faces.append([a0, b1, a1])
        a0, b0 = vid(ny - 1, i, 0), vid(ny - 1, i + 1, 0)
        a1, b1 = vid(ny - 1, i, 1), vid(ny - 1, i + 1, 1)
        tri_faces.append([a0, a1, b1])
        tri_faces.append([a0, b1, b0])

    for j in range(ny - 1):
        a0, b0 = vid(j, 0, 0), vid(j + 1, 0, 0)
        a1, b1 = vid(j, 0, 1), vid(j + 1, 0, 1)
        tri_faces.append([a0, a1, b1])
        tri_faces.append([a0, b1, b0])
        a0, b0 = vid(j, nx - 1, 0), vid(j + 1, nx - 1, 0)
        a1, b1 = vid(j, nx - 1, 1), vid(j + 1, nx - 1, 1)
        tri_faces.append([a0, b0, b1])
        tri_faces.append([a0, b1, a1])

    faces_np = np.asarray(tri_faces, dtype=np.int64)
    pv_faces = np.hstack(
        [
            np.full((faces_np.shape[0], 1), 3, dtype=np.int64),
            faces_np,
        ]
    ).ravel()
    solid = pv.PolyData(vertices, pv_faces)
    if solid.n_cells == 0:
        raise ValueError("Extruded wrap solid produced no faces.")
    return solid.triangulate().clean()


def _ray_stop_on_meshes(
    *,
    origins: np.ndarray,
    targets: np.ndarray,
    model_mesh: pv.PolyData,
    stop_mesh: pv.PolyData | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Extrude each origin toward its target. Stop at the nearest hit on the model
    or optional stop surface (e.g. backend plane); otherwise land on the target.
    """
    point_count = origins.shape[0]
    deltas = targets - origins
    lengths = np.linalg.norm(deltas, axis=1)
    valid_dir = lengths > _RAY_EPSILON
    directions = np.zeros_like(deltas)
    directions[valid_dir] = deltas[valid_dir] / lengths[valid_dir, None]

    wrapped = targets.copy()
    hit_model = np.zeros(point_count, dtype=bool)

    def _apply_hits(tm: trimesh.Trimesh, *, as_model: bool) -> None:
        nonlocal wrapped, hit_model
        if not np.any(valid_dir):
            return
        locations, index_ray, _index_tri = tm.ray.intersects_location(
            ray_origins=origins[valid_dir],
            ray_directions=directions[valid_dir],
            multiple_hits=False,
        )
        if len(index_ray) == 0:
            return
        valid_indices = np.flatnonzero(valid_dir)
        global_rays = valid_indices[np.asarray(index_ray, dtype=int)]
        hit_vectors = np.asarray(locations, dtype=float) - origins[global_rays]
        hit_t = np.einsum("ij,ij->i", hit_vectors, directions[global_rays])
        max_t = lengths[global_rays]
        keep = (hit_t > _RAY_EPSILON) & (hit_t <= max_t + _RAY_EPSILON)
        if not np.any(keep):
            return
        kept_rays = global_rays[keep]
        kept_points = np.asarray(locations, dtype=float)[keep]
        kept_t = hit_t[keep]

        # Only replace a point if this hit is closer than the current stop.
        current_vec = wrapped[kept_rays] - origins[kept_rays]
        current_t = np.einsum("ij,ij->i", current_vec, directions[kept_rays])
        closer = kept_t < current_t - _RAY_EPSILON
        update = kept_rays[closer]
        wrapped[update] = kept_points[closer]
        if as_model:
            hit_model[update] = True
        else:
            hit_model[update] = False

    _apply_hits(_polydata_to_trimesh(model_mesh), as_model=True)
    if stop_mesh is not None and stop_mesh.n_points > 0:
        _apply_hits(_polydata_to_trimesh(stop_mesh), as_model=False)

    hit_model_count = int(np.count_nonzero(hit_model))
    hit_end_count = int(point_count - hit_model_count)
    return wrapped, hit_model_count, hit_end_count


def wrap_start_plane_to_model(
    model_mesh: pv.PolyData,
    output_dir: Path,
    *,
    start_origin: tuple[float, float, float],
    start_rx_deg: float = 0.0,
    start_ry_deg: float = 0.0,
    start_rz_deg: float = 0.0,
    end_origin: tuple[float, float, float],
    end_rx_deg: float = 0.0,
    end_ry_deg: float = 0.0,
    end_rz_deg: float = 0.0,
    end_ctrl_uv: np.ndarray | None = None,
    end_ctrl_disp: np.ndarray | None = None,
    density: int = DEFAULT_POINT_DENSITY,
    progress: ProgressCallback | None = None,
) -> WrapPlaneResult:
    """
    Extrude the start plane toward the curved/deformed end plane as a solid.

    Each sample starts on the flat start plane and travels toward the matching
    point on the deformed end plane. Motion stops at the first model hit, or on
    the deformed end surface if the ray never hits the model first.
    """
    density = int(density)
    if density < MIN_POINT_DENSITY or density > MAX_POINT_DENSITY:
        raise ValueError(
            f"Point density must be between {MIN_POINT_DENSITY} and {MAX_POINT_DENSITY}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    point_count = density * density

    start_center = np.asarray(start_origin, dtype=float)
    end_center = np.asarray(end_origin, dtype=float)
    start_normal = np.asarray(
        plane_normal_from_rotations(start_rx_deg, start_ry_deg, start_rz_deg),
        dtype=float,
    )
    end_normal = np.asarray(
        plane_normal_from_rotations(end_rx_deg, end_ry_deg, end_rz_deg),
        dtype=float,
    )

    if end_ctrl_uv is None:
        end_ctrl_uv = np.zeros((1, 2), dtype=float)
    if end_ctrl_disp is None:
        end_ctrl_disp = np.zeros(len(end_ctrl_uv), dtype=float)
    end_ctrl_uv = np.asarray(end_ctrl_uv, dtype=float)
    end_ctrl_disp = np.asarray(end_ctrl_disp, dtype=float)
    if len(end_ctrl_disp) != len(end_ctrl_uv):
        raise ValueError("end_ctrl_disp length must match end_ctrl_uv.")

    if progress:
        progress("Sampling start and deformed end planes", 0, 4)

    half_extent = _plane_half_extents(model_mesh)
    bottom = _sample_oriented_plane_grid(
        center=start_center,
        normal=start_normal,
        half_extent=half_extent,
        density=density,
    )
    top_end = _sample_deformed_end_grid(
        center=end_center,
        normal=end_normal,
        half_extent=half_extent,
        density=density,
        end_ctrl_uv=end_ctrl_uv,
        end_ctrl_disp=end_ctrl_disp,
    )

    if progress:
        progress("Ray casting against model", 1, 4)

    origins = bottom.reshape(-1, 3)
    targets = top_end.reshape(-1, 3)
    wrapped, hit_model_count, hit_end_count = _ray_stop_on_meshes(
        origins=origins,
        targets=targets,
        model_mesh=model_mesh,
    )

    if progress:
        progress("Extruding start plane to deformed end", 2, 4)

    top = wrapped.reshape(density, density, 3)
    wrap_mesh = _build_extruded_solid(bottom, top)

    output_path = wrap_plane_output_path(output_dir)
    wrap_mesh.save(str(output_path))

    if progress:
        progress("Wrap plane complete", 4, 4)

    return WrapPlaneResult(
        wrap_mesh=wrap_mesh,
        output_path=output_path,
        point_count=point_count,
        hit_model_count=hit_model_count,
        hit_end_count=hit_end_count,
        density=density,
    )


def wrap_back_start_to_backend(
    model_mesh: pv.PolyData,
    output_dir: Path,
    *,
    back_start_origin: tuple[float, float, float],
    back_start_rx_deg: float = 0.0,
    back_start_ry_deg: float = 0.0,
    back_start_rz_deg: float = 0.0,
    backend_origin: tuple[float, float, float],
    backend_rx_deg: float = 0.0,
    backend_ry_deg: float = 0.0,
    backend_rz_deg: float = 0.0,
    backend_ctrl_uv: np.ndarray,
    backend_ctrl_disp: np.ndarray,
    backend_mesh: pv.PolyData | None = None,
    half_extent: float | None = None,
    density: int = DEFAULT_POINT_DENSITY,
    progress: ProgressCallback | None = None,
) -> WrapPlaneResult:
    """
    Extrude Back-start plane points toward the Backend (curved) plane.

    Each sample starts on the flat Back-start plane and travels toward the
    matching UV point on the Backend plane. Motion stops when the ray first
    hits the model or the Backend surface.
    """
    density = int(density)
    if density < 2 or density > MAX_WRAP_BACK_DENSITY:
        raise ValueError(
            f"Wrap-back density must be between 2 and {MAX_WRAP_BACK_DENSITY}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    point_count = density * density

    start_center = np.asarray(back_start_origin, dtype=float)
    end_center = np.asarray(backend_origin, dtype=float)
    start_normal = np.asarray(
        plane_normal_from_rotations(
            back_start_rx_deg, back_start_ry_deg, back_start_rz_deg
        ),
        dtype=float,
    )
    end_normal = np.asarray(
        plane_normal_from_rotations(backend_rx_deg, backend_ry_deg, backend_rz_deg),
        dtype=float,
    )

    backend_ctrl_uv = np.asarray(backend_ctrl_uv, dtype=float)
    backend_ctrl_disp = np.asarray(backend_ctrl_disp, dtype=float)
    if len(backend_ctrl_disp) != len(backend_ctrl_uv):
        raise ValueError("backend_ctrl_disp length must match backend_ctrl_uv.")

    if progress:
        progress("Sampling Back-start and Backend planes", 0, 4)

    if half_extent is None:
        half_extent = _plane_half_extents(model_mesh)

    bottom = _sample_oriented_plane_grid(
        center=start_center,
        normal=start_normal,
        half_extent=half_extent,
        density=density,
    )
    top_end = _sample_deformed_end_grid(
        center=end_center,
        normal=end_normal,
        half_extent=half_extent,
        density=density,
        end_ctrl_uv=backend_ctrl_uv,
        end_ctrl_disp=backend_ctrl_disp,
    )

    if progress:
        progress("Ray casting against model and Backend plane", 1, 4)

    origins = bottom.reshape(-1, 3)
    targets = top_end.reshape(-1, 3)
    wrapped, hit_model_count, hit_end_count = _ray_stop_on_meshes(
        origins=origins,
        targets=targets,
        model_mesh=model_mesh,
        stop_mesh=backend_mesh,
    )

    if progress:
        progress("Extruding Back-start to Backend", 2, 4)

    top = wrapped.reshape(density, density, 3)
    wrap_mesh = _build_extruded_solid(bottom, top)

    output_path = wrap_back_output_path(output_dir)
    wrap_mesh.save(str(output_path))

    if progress:
        progress("Wrap back complete", 4, 4)

    return WrapPlaneResult(
        wrap_mesh=wrap_mesh,
        output_path=output_path,
        point_count=point_count,
        hit_model_count=hit_model_count,
        hit_end_count=hit_end_count,
        density=density,
    )
