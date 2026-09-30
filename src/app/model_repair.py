"""Model gap-fill repair: ray extrusion between repair start/end planes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from app.plane_wrap import (
    DEFAULT_POINT_DENSITY,
    MAX_POINT_DENSITY,
    MIN_POINT_DENSITY,
    _build_extruded_solid,
    _polydata_to_trimesh,
)
from app.xy_plane import (
    plane_normal_from_rotations,
    repair_plane_basis,
    repair_plane_half_extents,
)

ProgressCallback = Callable[[str, int, int], None]

_RAY_EPSILON = 1e-6
# Pull fill endpoints slightly inward so the solid stays inside the model surface.
_INWARD_EPS_MM = 1e-4
# -1 = fill from first model hit to last exit hit (solid inside model).
FILL_TO_EXIT_WALLS = -1
DEFAULT_STOP_AFTER_WALLS = 1
MIN_STOP_AFTER_WALLS = FILL_TO_EXIT_WALLS
MAX_STOP_AFTER_WALLS = 64


@dataclass(frozen=True)
class FillGapsResult:
    fill_mesh: pv.PolyData
    output_path: Path
    point_count: int
    filled_count: int
    skipped_count: int
    density: int
    stop_after_walls: int


def fill_gaps_output_path(output_dir: Path) -> Path:
    return output_dir / "fill_gaps.stl"


def _sample_repair_plane_grid(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_i: float,
    half_j: float,
    density: int,
) -> np.ndarray:
    """Return repair-plane sample points with shape (density, density, 3)."""
    i_axis, j_axis = repair_plane_basis(normal)
    u = np.linspace(-half_i, half_i, density)
    v = np.linspace(-half_j, half_j, density)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return (
        center[None, None, :]
        + uu[:, :, None] * i_axis[None, None, :]
        + vv[:, :, None] * j_axis[None, None, :]
    )


def _validate_stop_after_walls(stop_after_walls: int) -> int:
    value = int(stop_after_walls)
    if value == 0:
        raise ValueError(
            "Stop after walls cannot be 0. Use -1 to fill through the model "
            "to the last exit, or a value >= 1."
        )
    if value < MIN_STOP_AFTER_WALLS or value > MAX_STOP_AFTER_WALLS:
        raise ValueError(
            f"Stop after walls must be {FILL_TO_EXIT_WALLS} (fill to exit) "
            f"or between 1 and {MAX_STOP_AFTER_WALLS}."
        )
    return value


def _ray_fill_gap_segments(
    *,
    origins: np.ndarray,
    targets: np.ndarray,
    model_mesh: pv.PolyData,
    stop_after_walls: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cast rays from origins toward targets and fill only model gaps.

    For each ray:
    - Travel through empty space until the first model surface hit → start fill
    - If ``stop_after_walls`` is -1: stop at the last model hit (exit). All
      intermediate layers are spanned so the solid stays inside the model —
      never on the end plane outside the body.
    - If ``stop_after_walls`` is N >= 1: stop at the N-th wall after entry.
      Intermediate walls before that count are ignored. If fewer than N walls
      remain after entry, stop at the end-plane target.
    - Rays that never hit the model produce no fill.
    """
    stop_after_walls = _validate_stop_after_walls(stop_after_walls)
    fill_to_exit = stop_after_walls == FILL_TO_EXIT_WALLS

    point_count = origins.shape[0]
    deltas = targets - origins
    lengths = np.linalg.norm(deltas, axis=1)
    valid_dir = lengths > _RAY_EPSILON
    directions = np.zeros_like(deltas)
    directions[valid_dir] = deltas[valid_dir] / lengths[valid_dir, None]

    bottom = origins.copy()
    top = origins.copy()
    filled = np.zeros(point_count, dtype=bool)

    if not np.any(valid_dir):
        return bottom, top, filled

    tm = _polydata_to_trimesh(model_mesh)
    locations, index_ray, _index_tri = tm.ray.intersects_location(
        ray_origins=origins[valid_dir],
        ray_directions=directions[valid_dir],
        multiple_hits=True,
    )
    if len(index_ray) == 0:
        return bottom, top, filled

    valid_indices = np.flatnonzero(valid_dir)
    global_rays = valid_indices[np.asarray(index_ray, dtype=int)]
    hit_points = np.asarray(locations, dtype=float)
    hit_vectors = hit_points - origins[global_rays]
    hit_t = np.einsum("ij,ij->i", hit_vectors, directions[global_rays])
    max_t = lengths[global_rays]
    keep = (hit_t > _RAY_EPSILON) & (hit_t <= max_t + _RAY_EPSILON)
    if not np.any(keep):
        return bottom, top, filled

    global_rays = global_rays[keep]
    hit_points = hit_points[keep]
    hit_t = hit_t[keep]

    # Group hits per ray, sorted by distance along the ray.
    order = np.argsort(global_rays, kind="mergesort")
    global_rays = global_rays[order]
    hit_points = hit_points[order]
    hit_t = hit_t[order]

    # Secondary sort by t within each ray group.
    splits = np.flatnonzero(np.diff(global_rays)) + 1
    groups = np.split(np.arange(len(global_rays)), splits)
    for group in groups:
        if len(group) == 0:
            continue
        ray_i = int(global_rays[group[0]])
        local = group[np.argsort(hit_t[group], kind="mergesort")]
        # Deduplicate nearly coincident hits (double-sided / coplanar faces).
        ts = hit_t[local]
        pts = hit_points[local]
        unique_idx = [0]
        for k in range(1, len(local)):
            if ts[k] - ts[unique_idx[-1]] > _RAY_EPSILON * 10:
                unique_idx.append(k)
        pts = pts[unique_idx]
        ts = ts[unique_idx]

        if len(pts) == 0:
            continue

        direction = directions[ray_i]

        if fill_to_exit:
            # Solid interior fill: first entry → last exit. Never leave the model.
            if len(pts) < 2:
                continue
            span = float(ts[-1] - ts[0])
            if span <= _RAY_EPSILON * 10:
                continue
            inset = min(_INWARD_EPS_MM, 0.25 * span)
            bottom[ray_i] = pts[0] + direction * inset
            top[ray_i] = pts[-1] - direction * inset
            filled[ray_i] = True
            continue

        # Entry hit starts the fill. Stop at wall N after entry when available;
        # otherwise continue to the end plane.
        stop_index = stop_after_walls
        bottom[ray_i] = pts[0]
        if len(pts) > stop_index:
            top[ray_i] = pts[stop_index]
        else:
            top[ray_i] = targets[ray_i]
        filled[ray_i] = True

    return bottom, top, filled


def _build_extruded_solid_masked(
    bottom: np.ndarray,
    top: np.ndarray,
    filled: np.ndarray,
) -> pv.PolyData:
    """Extrude only grid quads whose four corners successfully filled a gap."""
    if bottom.shape != top.shape or bottom.ndim != 3 or bottom.shape[2] != 3:
        raise ValueError("Bottom and top grids must both have shape (n, n, 3).")
    if filled.shape != bottom.shape[:2]:
        raise ValueError("Filled mask must match the grid (n, n).")

    ny, nx = filled.shape
    if ny < 2 or nx < 2:
        raise ValueError("Grid density must be at least 2 on each side.")

    # Fall back to full extrusion if every sample filled (faster shared path).
    if bool(np.all(filled)):
        return _build_extruded_solid(bottom, top)

    n_plane = ny * nx
    vertices = np.vstack([bottom.reshape(-1, 3), top.reshape(-1, 3)])

    def vid(j: int, i: int, layer: int) -> int:
        return layer * n_plane + j * nx + i

    tri_faces: list[list[int]] = []

    def quad_filled(j: int, i: int) -> bool:
        return bool(
            filled[j, i]
            and filled[j, i + 1]
            and filled[j + 1, i]
            and filled[j + 1, i + 1]
        )

    for j in range(ny - 1):
        for i in range(nx - 1):
            if not quad_filled(j, i):
                continue
            a = vid(j, i, 0)
            b = vid(j, i + 1, 0)
            c = vid(j + 1, i + 1, 0)
            d = vid(j + 1, i, 0)
            tri_faces.append([a, d, c])
            tri_faces.append([a, c, b])

            a1 = vid(j, i, 1)
            b1 = vid(j, i + 1, 1)
            c1 = vid(j + 1, i + 1, 1)
            d1 = vid(j + 1, i, 1)
            tri_faces.append([a1, b1, c1])
            tri_faces.append([a1, c1, d1])

            # Side walls for this prism cell.
            tri_faces.append([a, b, b1])
            tri_faces.append([a, b1, a1])
            tri_faces.append([b, c, c1])
            tri_faces.append([b, c1, b1])
            tri_faces.append([c, d, d1])
            tri_faces.append([c, d1, c1])
            tri_faces.append([d, a, a1])
            tri_faces.append([d, a1, d1])

    if not tri_faces:
        raise ValueError(
            "No gap segments were filled. Move the repair planes so rays "
            "cross hollow regions, or lower the stop-after-walls count."
        )

    faces_np = np.asarray(tri_faces, dtype=np.int64)
    pv_faces = np.hstack(
        [
            np.full((faces_np.shape[0], 1), 3, dtype=np.int64),
            faces_np,
        ]
    ).ravel()
    solid = pv.PolyData(vertices, pv_faces)
    return solid.triangulate().clean()


def fill_gaps_between_planes(
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
    density: int = DEFAULT_POINT_DENSITY,
    stop_after_walls: int = DEFAULT_STOP_AFTER_WALLS,
    progress: ProgressCallback | None = None,
) -> FillGapsResult:
    """
    Fill model gaps by extruding rays from the repair start plane toward the
    repair end plane. Filling begins at the first model hit and stops after
    ``stop_after_walls`` further wall hits (default 1 = stop at the next layer).
    Use ``-1`` to fill from the first hit through to the last exit hit so the
    solid stays inside the model. If N >= 1 and fewer walls remain after entry,
    the ray stops at the end plane.
    """
    density = int(density)
    stop_after_walls = _validate_stop_after_walls(stop_after_walls)
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

    if progress:
        progress("Sampling repair start and end planes", 0, 4)

    half_i, half_j = repair_plane_half_extents(model_mesh, start_normal)
    bottom_grid = _sample_repair_plane_grid(
        center=start_center,
        normal=start_normal,
        half_i=half_i,
        half_j=half_j,
        density=density,
    )
    # End plane uses the same footprint so rays stay paired across the gap.
    top_end = _sample_repair_plane_grid(
        center=end_center,
        normal=end_normal,
        half_i=half_i,
        half_j=half_j,
        density=density,
    )

    if progress:
        progress("Casting gap-fill rays", 1, 4)

    origins = bottom_grid.reshape(-1, 3)
    targets = top_end.reshape(-1, 3)
    bottom_pts, top_pts, filled = _ray_fill_gap_segments(
        origins=origins,
        targets=targets,
        model_mesh=model_mesh,
        stop_after_walls=stop_after_walls,
    )

    filled_count = int(np.count_nonzero(filled))
    skipped_count = int(point_count - filled_count)

    if progress:
        progress("Extruding filled gap segments", 2, 4)

    bottom = bottom_pts.reshape(density, density, 3)
    top = top_pts.reshape(density, density, 3)
    fill_mask = filled.reshape(density, density)
    fill_mesh = _build_extruded_solid_masked(bottom, top, fill_mask)

    output_path = fill_gaps_output_path(output_dir)
    fill_mesh.save(str(output_path))

    if progress:
        progress("Fill gaps complete", 4, 4)

    return FillGapsResult(
        fill_mesh=fill_mesh,
        output_path=output_path,
        point_count=point_count,
        filled_count=filled_count,
        skipped_count=skipped_count,
        density=density,
        stop_after_walls=stop_after_walls,
    )
