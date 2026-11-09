"""Slice a mesh into two parts with a deformable (curved) cutting plane."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon

from app.mesh_ops import polydata_to_trimesh, trimesh_to_polydata
from app.plane_deform import (
    oriented_plane_axes,
    pose_frame,
    sample_deformed_plane_grid,
)

ProgressCallback = Callable[[str, int, int], None]

DEFAULT_GRID_POINTS = 9
MIN_GRID_POINTS = 1
MAX_GRID_POINTS = 200
DEFAULT_SIZE_MARGIN_PERCENT = 10.0
MIN_SIZE_MARGIN_PERCENT = -99.0
MAX_SIZE_MARGIN_PERCENT = 500.0
_MIN_SIZE_SCALE = 0.01
# Reserved for UI compatibility (split is a through-cut on the plane footprint).
DEFAULT_KNIFE_THICKNESS_MM = 0.0
MIN_KNIFE_THICKNESS_MM = 0.0
MAX_KNIFE_THICKNESS_MM = 1_000_000.0
DEFAULT_PATCH_DENSITY = 96
MIN_PATCH_DENSITY = 8
MAX_PATCH_DENSITY = 400
_IDW_POWER = 2.0
_IDW_EPS = 1e-8
_UV_BOUNDS_EPS = 1e-6
_UNCUT_SCALAR = 1.0
# Alias kept for older call sites / UI naming.
DEFAULT_RAY_DENSITY = DEFAULT_PATCH_DENSITY
MIN_RAY_DENSITY = MIN_PATCH_DENSITY
MAX_RAY_DENSITY = MAX_PATCH_DENSITY


def _size_scale(margin_percent: float) -> float:
    """Convert margin % to a scale factor (10 → 1.10, -20 → 0.80)."""
    return max(_MIN_SIZE_SCALE, 1.0 + float(margin_percent) / 100.0)


@dataclass(frozen=True)
class CurveSliceResult:
    part_a: pv.PolyData
    part_b: pv.PolyData
    path_a: Path
    path_b: Path
    point_count: int


@dataclass(frozen=True)
class CurvePlaneFit:
    """Auto-sized / auto-rotated cutting plane relative to the model."""

    half_i: float
    half_j: float
    rz_deg: float
    model_length: float
    model_width: float


def curve_slice_output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / "curve_slice_part_a.stl",
        output_dir / "curve_slice_part_b.stl",
    )


def auto_fit_curve_plane(
    mesh: pv.PolyData,
    *,
    size_margin_percent: float = DEFAULT_SIZE_MARGIN_PERCENT,
) -> CurvePlaneFit:
    """Size/rotate a horizontal cutting plane from the model XY footprint."""
    scale = _size_scale(size_margin_percent)
    xmin, xmax, ymin, ymax, _zmin, _zmax = (float(v) for v in mesh.bounds)
    dx = max(xmax - xmin, 1e-3)
    dy = max(ymax - ymin, 1e-3)
    model_length = max(dx, dy)
    model_width = min(dx, dy)
    half_long = 0.5 * model_length * scale
    half_wide = 0.5 * model_width * scale
    rz_deg = 90.0 if dx >= dy else 0.0
    return CurvePlaneFit(
        half_i=half_long,
        half_j=half_wide,
        rz_deg=rz_deg,
        model_length=model_length,
        model_width=model_width,
    )


def curve_plane_half_extents(
    mesh: pv.PolyData,
    *,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    size_margin_percent: float = DEFAULT_SIZE_MARGIN_PERCENT,
) -> tuple[float, float]:
    """Rectangular half-extents (i, j) for the current plane orientation."""
    scale = _size_scale(size_margin_percent)
    i_axis, j_axis, _normal = oriented_plane_axes(
        rx=float(rx_deg), ry=float(ry_deg), rz=float(rz_deg)
    )
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in mesh.bounds)
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=float,
    )
    ui = corners @ i_axis
    vj = corners @ j_axis
    span_i = max(float(ui.max() - ui.min()), 1e-3)
    span_j = max(float(vj.max() - vj.min()), 1e-3)
    model_length = max(span_i, span_j)
    model_width = min(span_i, span_j)
    half_long = 0.5 * model_length * scale
    half_wide = 0.5 * model_width * scale
    if span_i >= span_j:
        return half_wide, half_long
    return half_long, half_wide


def model_extent_along_normal(
    mesh: pv.PolyData,
    *,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> float:
    """Return model AABB span along the plane normal."""
    _i, _j, normal = oriented_plane_axes(
        rx=float(rx_deg), ry=float(ry_deg), rz=float(rz_deg)
    )
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in mesh.bounds)
    corners = np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=float,
    )
    h = corners @ normal
    return max(float(h.max() - h.min()), 1e-3)


def _idw_displacements(
    query_uv: np.ndarray,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
) -> np.ndarray:
    """IDW pull/push field at parametric UV samples."""
    ctrl = np.asarray(ctrl_uv, dtype=float)
    disp = np.asarray(ctrl_disp, dtype=float)
    query = np.asarray(query_uv, dtype=float)
    if len(ctrl) == 0:
        return np.zeros(query.shape[0], dtype=float)
    if len(ctrl) == 1:
        return np.full(query.shape[0], float(disp[0]), dtype=float)
    delta = query[:, None, :] - ctrl[None, :, :]
    dist2 = np.sum(delta * delta, axis=2)
    exact = dist2 < _IDW_EPS
    weights = 1.0 / np.power(np.maximum(dist2, _IDW_EPS), _IDW_POWER / 2.0)
    field = np.sum(weights * disp[None, :], axis=1) / np.sum(weights, axis=1)
    if np.any(exact):
        hit_rows, hit_cols = np.where(exact)
        field[hit_rows] = disp[hit_cols]
    return field


def _plane_model_intersection_mask(
    points: np.ndarray,
    model_mesh: pv.PolyData,
) -> np.ndarray:
    """True for plane samples that lie inside the model (plane∩solid)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    surface = model_mesh
    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface(algorithm="dataset_surface").triangulate()
    cloud = pv.PolyData(pts.copy())
    for method in ("cell_locator", "signed_distance"):
        try:
            marked = cloud.select_interior_points(
                surface,
                method=method,
                check_surface=False,
            )
            return np.asarray(marked.point_data["selected_points"], dtype=bool)
        except Exception:  # noqa: BLE001
            continue
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in surface.bounds)
    return (
        (pts[:, 0] >= xmin)
        & (pts[:, 0] <= xmax)
        & (pts[:, 1] >= ymin)
        & (pts[:, 1] <= ymax)
        & (pts[:, 2] >= zmin)
        & (pts[:, 2] <= zmax)
    )


def _dilate_bool_grid(mask: np.ndarray, rings: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(rings))):
        padded = np.pad(out, 1, constant_values=False)
        neighbors = (
            padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
        out = out | neighbors
    return out


def _cut_scalars(
    points: np.ndarray,
    *,
    center: np.ndarray,
    i_axis: np.ndarray,
    j_axis: np.ndarray,
    normal: np.ndarray,
    half_i: float,
    half_j: float,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
    cut_uv_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Signed height vs deformed plane; cut only on plane∩model UV samples."""
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    u = (rel @ i_axis) / max(float(half_i), 1e-9)
    v = (rel @ j_axis) / max(float(half_j), 1e-9)
    height = rel @ normal
    inside_uv = (np.abs(u) <= 1.0 + _UV_BOUNDS_EPS) & (
        np.abs(v) <= 1.0 + _UV_BOUNDS_EPS
    )
    if cut_uv_mask is not None:
        res_j, res_i = cut_uv_mask.shape
        ii = np.clip(((u + 1.0) * 0.5 * (res_i - 1)).astype(int), 0, res_i - 1)
        jj = np.clip(((v + 1.0) * 0.5 * (res_j - 1)).astype(int), 0, res_j - 1)
        inside_uv = inside_uv & cut_uv_mask[jj, ii]

    scalar = np.full(len(points), _UNCUT_SCALAR, dtype=float)
    if not np.any(inside_uv):
        return scalar
    query = np.stack([u[inside_uv], v[inside_uv]], axis=1)
    surf_h = _idw_displacements(query, ctrl_uv, ctrl_disp)
    scalar[np.flatnonzero(inside_uv)] = height[inside_uv] - surf_h
    return scalar


def _as_surface(mesh: pv.PolyData) -> pv.PolyData:
    surface = mesh.copy(deep=True)
    surface.clear_data()
    surface = surface.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError("Slice produced an empty mesh part.")
    surface = surface.clean()
    surface.clear_data()
    return surface


def _boundary_vertex_indices(tm: trimesh.Trimesh) -> np.ndarray:
    edges = np.asarray(tm.edges_sorted, dtype=np.int64)
    if len(edges) == 0:
        return np.zeros(0, dtype=np.int64)
    boundary = edges[trimesh.grouping.group_rows(edges, require_count=1)]
    if len(boundary) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(boundary.reshape(-1))


def _mean_edge_length(mesh: pv.PolyData) -> float:
    tm = polydata_to_trimesh(mesh)
    if len(tm.edges) == 0:
        return 1e-3
    return float(np.mean(tm.edges_unique_length))


def _project_points_to_plane(
    points: np.ndarray,
    *,
    center: np.ndarray,
    i_axis: np.ndarray,
    j_axis: np.ndarray,
    normal: np.ndarray,
    half_i: float,
    half_j: float,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    u = np.clip((rel @ i_axis) / max(float(half_i), 1e-9), -1.0, 1.0)
    v = np.clip((rel @ j_axis) / max(float(half_j), 1e-9), -1.0, 1.0)
    h = _idw_displacements(np.column_stack([u, v]), ctrl_uv, ctrl_disp)
    return (
        np.asarray(center, dtype=float)[None, :]
        + u[:, None] * float(half_i) * np.asarray(i_axis, dtype=float)[None, :]
        + v[:, None] * float(half_j) * np.asarray(j_axis, dtype=float)[None, :]
        + h[:, None] * np.asarray(normal, dtype=float)[None, :]
    )


def _signed_heights(
    points: np.ndarray,
    *,
    center: np.ndarray,
    i_axis: np.ndarray,
    j_axis: np.ndarray,
    normal: np.ndarray,
    half_i: float,
    half_j: float,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
) -> np.ndarray:
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    u = (rel @ i_axis) / max(float(half_i), 1e-9)
    v = (rel @ j_axis) / max(float(half_j), 1e-9)
    height = rel @ normal
    surf = _idw_displacements(np.column_stack([u, v]), ctrl_uv, ctrl_disp)
    return height - surf


def _snap_open_cut_to_plane(
    part: pv.PolyData,
    *,
    center: np.ndarray,
    i_axis: np.ndarray,
    j_axis: np.ndarray,
    normal: np.ndarray,
    half_i: float,
    half_j: float,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
    band_mm: float,
) -> pv.PolyData:
    """Snap open-cut / near-plane vertices onto the deformed plane."""
    tm = polydata_to_trimesh(part)
    pts = np.asarray(tm.vertices, dtype=float).copy()
    signed = _signed_heights(
        pts,
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        normal=normal,
        half_i=half_i,
        half_j=half_j,
        ctrl_uv=ctrl_uv,
        ctrl_disp=ctrl_disp,
    )
    near = np.abs(signed) <= max(float(band_mm), 1e-6)
    boundary = _boundary_vertex_indices(tm)
    if len(boundary) > 0:
        near[boundary] = True
    if not np.any(near):
        return part
    pts[near] = _project_points_to_plane(
        pts[near],
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        normal=normal,
        half_i=half_i,
        half_j=half_j,
        ctrl_uv=ctrl_uv,
        ctrl_disp=ctrl_disp,
    )
    snapped = trimesh.Trimesh(
        vertices=pts, faces=np.asarray(tm.faces, dtype=np.int64), process=False
    )
    return trimesh_to_polydata(snapped)


def _ordered_boundary_loops(tm: trimesh.Trimesh) -> list[np.ndarray]:
    edges = np.asarray(tm.edges_sorted, dtype=np.int64)
    if len(edges) == 0:
        return []
    boundary = edges[trimesh.grouping.group_rows(edges, require_count=1)]
    if len(boundary) == 0:
        return []

    adj: dict[int, list[int]] = defaultdict(list)
    for a_raw, b_raw in boundary:
        a, b = int(a_raw), int(b_raw)
        adj[a].append(b)
        adj[b].append(a)

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    used: set[tuple[int, int]] = set()
    loops: list[np.ndarray] = []
    for start in list(adj.keys()):
        for first in adj[start]:
            ek0 = edge_key(start, first)
            if ek0 in used:
                continue
            loop = [start]
            prev, cur = start, first
            used.add(ek0)
            closed = False
            for _ in range(len(boundary) + 2):
                if cur == start:
                    closed = True
                    break
                loop.append(cur)
                nxts = [n for n in adj[cur] if edge_key(cur, n) not in used and n != prev]
                if not nxts:
                    nxts = [n for n in adj[cur] if edge_key(cur, n) not in used]
                if not nxts:
                    break
                nxt = nxts[0]
                used.add(edge_key(cur, nxt))
                prev, cur = cur, nxt
            if closed and len(loop) >= 3:
                loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _shared_interior_points(
    plane_grid: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz, uv) for active plane∩model samples — identical on both caps."""
    ny, nx, _ = plane_grid.shape
    u_lin = np.linspace(-1.0, 1.0, nx)
    v_lin = np.linspace(-1.0, 1.0, ny)
    xyz: list[np.ndarray] = []
    uv: list[list[float]] = []
    for j in range(ny):
        for i in range(nx):
            if not active[j, i]:
                continue
            xyz.append(np.asarray(plane_grid[j, i], dtype=float))
            uv.append([float(u_lin[i]), float(v_lin[j])])
    if not xyz:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 2), dtype=float)
    return np.asarray(xyz, dtype=float), np.asarray(uv, dtype=float)


def _cap_part_with_shared_pattern(
    part: pv.PolyData,
    *,
    interior_xyz: np.ndarray,
    interior_uv: np.ndarray,
    center: np.ndarray,
    i_axis: np.ndarray,
    j_axis: np.ndarray,
    half_i: float,
    half_j: float,
    flip: bool,
) -> pv.PolyData:
    """
    Weld a patterned cap onto the open boundary.

    Boundary vertices stay part of the mesh (true stitch). Interior points are
    the shared deformed-plane samples so both halves get the same pattern.
    """
    tm = polydata_to_trimesh(part)
    loops = _ordered_boundary_loops(tm)
    if not loops:
        return part
    loop = max(loops, key=len)
    if len(loop) < 3:
        raise ValueError("Open cut boundary is too small to cap.")

    loop_pts = np.asarray(tm.vertices[loop], dtype=float)
    rel = loop_pts - np.asarray(center, dtype=float)
    loop_u = (rel @ i_axis) / max(float(half_i), 1e-9)
    loop_v = (rel @ j_axis) / max(float(half_j), 1e-9)
    loop_uv = np.column_stack([loop_u, loop_v])
    poly = Polygon(loop_uv)
    if not poly.is_valid or poly.is_empty or poly.area <= 0.0:
        poly = poly.buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: float(g.area))
    if poly.is_empty or getattr(poly, "area", 0.0) <= 0.0:
        raise ValueError("Could not build a UV polygon for the open cut.")

    keep_interior = np.zeros(len(interior_uv), dtype=bool)
    for i, (uu, vv) in enumerate(interior_uv):
        keep_interior[i] = bool(poly.contains(Point(float(uu), float(vv))))
    interior_xyz_use = interior_xyz[keep_interior]
    interior_uv_use = interior_uv[keep_interior]

    all_uv = loop_uv
    if len(interior_uv_use) > 0:
        all_uv = np.vstack([loop_uv, interior_uv_use])
    if len(all_uv) < 3:
        raise ValueError("Not enough points to triangulate the patterned cap.")

    try:
        delaunay = Delaunay(all_uv)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to triangulate patterned cap: {exc}") from exc

    n_loop = int(len(loop))
    n_old = int(len(tm.vertices))

    def map_idx(local: int) -> int:
        local_i = int(local)
        if local_i < n_loop:
            return int(loop[local_i])
        return n_old + (local_i - n_loop)

    fill_faces: list[list[int]] = []
    for simplex in delaunay.simplices:
        centroid = all_uv[simplex].mean(axis=0)
        if not poly.covers(Point(float(centroid[0]), float(centroid[1]))):
            continue
        a, b, c = map_idx(simplex[0]), map_idx(simplex[1]), map_idx(simplex[2])
        if a == b or b == c or a == c:
            continue
        if flip:
            fill_faces.append([a, c, b])
        else:
            fill_faces.append([a, b, c])

    if not fill_faces:
        raise ValueError("Patterned cap produced no triangles.")

    if len(interior_xyz_use) > 0:
        vertices = np.vstack(
            [np.asarray(tm.vertices, dtype=float), interior_xyz_use]
        )
    else:
        vertices = np.asarray(tm.vertices, dtype=float)
    faces = np.vstack(
        [np.asarray(tm.faces, dtype=np.int64), np.asarray(fill_faces, dtype=np.int64)]
    )
    combined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    combined.merge_vertices()
    combined.update_faces(combined.unique_faces() & combined.nondegenerate_faces())
    combined.remove_unreferenced_vertices()
    try:
        trimesh.repair.fill_holes(combined)
        trimesh.repair.fix_normals(combined)
    except Exception:  # noqa: BLE001
        pass
    if len(combined.faces) == 0:
        raise ValueError("Capped part collapsed to an empty mesh.")

    out = trimesh_to_polydata(combined)
    out.clear_data()
    try:
        extents = np.asarray(out.bounds, dtype=float).reshape(2, 3)
        span = float(np.max(extents[1] - extents[0]))
        out = out.fill_holes(max(span * 0.5, 1.0))
        out.clear_data()
    except Exception:  # noqa: BLE001
        pass
    return out


def slice_mesh_by_curve_plane(
    model_mesh: pv.PolyData,
    output_dir: Path,
    *,
    origin: tuple[float, float, float],
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
    half_extent_i: float | None = None,
    half_extent_j: float | None = None,
    size_margin_percent: float = DEFAULT_SIZE_MARGIN_PERCENT,
    knife_thickness_mm: float = DEFAULT_KNIFE_THICKNESS_MM,
    ray_density: int = DEFAULT_PATCH_DENSITY,
    progress: ProgressCallback | None = None,
) -> CurveSliceResult:
    """
    Cut the model on the plane∩model intersection, then cap both opens.

    1. Sample the deformed plane; keep only samples inside the model.
    2. Split/open the original mesh along the deformed plane (intersection only).
    3. Cap each open side with the same patterned interior points and weld
       on the shared open boundary.
    """
    _ = knife_thickness_mm
    ctrl_uv = np.asarray(ctrl_uv, dtype=float)
    ctrl_disp = np.asarray(ctrl_disp, dtype=float)
    if len(ctrl_uv) == 0:
        raise ValueError("At least one grid point is required.")
    if len(ctrl_disp) != len(ctrl_uv):
        raise ValueError("ctrl_disp length must match ctrl_uv.")
    if len(ctrl_uv) > MAX_GRID_POINTS:
        raise ValueError(f"Grid points must be at most {MAX_GRID_POINTS}.")

    density = int(ray_density)
    if density < MIN_PATCH_DENSITY or density > MAX_PATCH_DENSITY:
        raise ValueError(
            f"Patch density must be between {MIN_PATCH_DENSITY} and {MAX_PATCH_DENSITY}."
        )

    total_steps = 7
    if progress:
        progress("Preparing plane-cut + patterned cap", 0, total_steps)

    if half_extent_i is None or half_extent_j is None:
        half_extent_i, half_extent_j = curve_plane_half_extents(
            model_mesh,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
            size_margin_percent=size_margin_percent,
        )
    half_i = float(half_extent_i)
    half_j = float(half_extent_j)

    center, normal = pose_frame(
        x=float(origin[0]),
        y=float(origin[1]),
        z=float(origin[2]),
        rx=float(rx_deg),
        ry=float(ry_deg),
        rz=float(rz_deg),
    )
    i_axis, j_axis, _n = oriented_plane_axes(
        rx=float(rx_deg), ry=float(ry_deg), rz=float(rz_deg)
    )

    if progress:
        progress(
            f"Sampling deformed plane · {density}×{density} = {density * density:,}",
            1,
            total_steps,
        )

    plane_grid = sample_deformed_plane_grid(
        center=center,
        normal=normal,
        uv=ctrl_uv,
        displacements=ctrl_disp,
        density=density,
        half_extent_i=half_i,
        half_extent_j=half_j,
        i_axis=i_axis,
        j_axis=j_axis,
    )
    flat = plane_grid.reshape(-1, 3)

    if progress:
        progress("Finding plane∩model intersection samples", 2, total_steps)

    active = _plane_model_intersection_mask(flat, model_mesh).reshape(density, density)
    if not np.any(active):
        raise ValueError(
            "Plane does not intersect the model interior. Move the plane so it "
            "crosses the solid."
        )
    active = _dilate_bool_grid(active, rings=1)
    n_active = int(np.count_nonzero(active))
    interior_xyz, interior_uv = _shared_interior_points(plane_grid, active)
    if progress:
        progress(
            f"Intersection samples: {n_active:,} / {density * density:,} · "
            f"shared pattern points: {len(interior_xyz):,}",
            3,
            total_steps,
        )
    if len(interior_xyz) == 0:
        raise ValueError(
            "No shared pattern points on the plane∩model intersection."
        )

    if progress:
        progress("Cutting / opening model along plane intersection", 4, total_steps)

    work = model_mesh.copy(deep=True)
    work.clear_data()
    work["cut_side"] = _cut_scalars(
        np.asarray(work.points, dtype=float),
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        normal=normal,
        half_i=half_i,
        half_j=half_j,
        ctrl_uv=ctrl_uv,
        ctrl_disp=ctrl_disp,
        cut_uv_mask=active,
    )
    part_a_raw = work.clip_scalar(scalars="cut_side", value=0.0, invert=False)
    part_b_raw = work.clip_scalar(scalars="cut_side", value=0.0, invert=True)
    if part_a_raw.n_cells == 0 or part_b_raw.n_cells == 0:
        raise ValueError(
            "Cut left one side empty. Move the plane so it intersects more of "
            "the model."
        )
    part_a = _as_surface(part_a_raw)
    part_b = _as_surface(part_b_raw)

    if progress:
        progress("Snapping open cuts onto deformed plane", 5, total_steps)

    snap_band = max(_mean_edge_length(model_mesh) * 2.5, 1e-3)
    plane_kwargs = dict(
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        normal=normal,
        half_i=half_i,
        half_j=half_j,
        ctrl_uv=ctrl_uv,
        ctrl_disp=ctrl_disp,
        band_mm=snap_band,
    )
    part_a = _snap_open_cut_to_plane(part_a, **plane_kwargs)
    part_b = _snap_open_cut_to_plane(part_b, **plane_kwargs)

    if progress:
        progress("Capping both opens with shared plane pattern + weld", 6, total_steps)

    cap_kwargs = dict(
        interior_xyz=interior_xyz,
        interior_uv=interior_uv,
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        half_i=half_i,
        half_j=half_j,
    )
    part_a = _cap_part_with_shared_pattern(part_a, flip=True, **cap_kwargs)
    part_b = _cap_part_with_shared_pattern(part_b, flip=False, **cap_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    path_a, path_b = curve_slice_output_paths(output_dir)
    part_a.save(str(path_a))
    part_b.save(str(path_b))

    if progress:
        progress("Curve slice complete", 7, total_steps)

    return CurveSliceResult(
        part_a=part_a,
        part_b=part_b,
        path_a=path_a,
        path_b=path_b,
        point_count=int(len(ctrl_uv)),
    )
