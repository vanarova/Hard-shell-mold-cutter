"""Slice a mesh into two parts with a deformable (curved) cutting plane."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from app.plane_deform import (
    oriented_plane_axes,
    pose_frame,
)

ProgressCallback = Callable[[str, int, int], None]

DEFAULT_GRID_POINTS = 9
MIN_GRID_POINTS = 1
MAX_GRID_POINTS = 200
DEFAULT_SIZE_MARGIN_PERCENT = 10.0
MIN_SIZE_MARGIN_PERCENT = -99.0
MAX_SIZE_MARGIN_PERCENT = 500.0
_MIN_SIZE_SCALE = 0.01
# 0 = cut all the way through under the plane (full depth).
DEFAULT_KNIFE_THICKNESS_MM = 0.0
MIN_KNIFE_THICKNESS_MM = 0.0
MAX_KNIFE_THICKNESS_MM = 1_000_000.0
_IDW_POWER = 2.0
_IDW_EPS = 1e-8
_UV_BOUNDS_EPS = 1e-6
# Outside the knife blade → stay with part A (not cut).
_UNCUT_SCALAR = 1.0


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
    """
    Size and twist a horizontal cutting plane from the model XY footprint.

    - Length / width are the longer / shorter of dx, dy, scaled by
      ``size_margin_percent`` (positive enlarges, negative shrinks)
    - Plane long side (i) is rotated perpendicular to the model's long side
    """
    scale = _size_scale(size_margin_percent)
    xmin, xmax, ymin, ymax, _zmin, _zmax = (float(v) for v in mesh.bounds)
    dx = max(xmax - xmin, 1e-3)
    dy = max(ymax - ymin, 1e-3)
    model_length = max(dx, dy)
    model_width = min(dx, dy)
    half_long = 0.5 * model_length * scale
    half_wide = 0.5 * model_width * scale

    # Default i=+X. If model is longer along X, rotate 90° so i (plane long)
    # lies along Y — perpendicular to the model long side.
    if dx >= dy:
        rz_deg = 90.0
    else:
        rz_deg = 0.0

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
    """
    Rectangular half-extents (i, j) for the current plane orientation.

    Projected model spans on the plane define length/width; the plane's longer
    side is placed perpendicular to the model's longer projected side, then
    scaled by ``size_margin_percent`` (positive enlarges, negative shrinks).
    """
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

    # Longer side of model ⊥ longer side of plane.
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
    """Return model AABB span along the plane normal (for knife defaults)."""
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


def _as_surface(mesh: pv.PolyData) -> pv.PolyData:
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError("Slice produced an empty mesh part.")
    return surface


def _idw_displacements(
    query_uv: np.ndarray,
    ctrl_uv: np.ndarray,
    ctrl_disp: np.ndarray,
) -> np.ndarray:
    """IDW pull/push field at parametric UV samples (same as plane deform)."""
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


def _knife_cut_scalars(
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
    knife_thickness_mm: float,
) -> np.ndarray:
    """
    Knife-style cut scalar: only split where the blade touches the model.

    The blade is the finite plane rectangle. A point is cut only when:
    1. Its projection lies inside the plane UV footprint, and
    2. Its distance to the deformed surface along the normal is within the
       knife thickness band (or thickness <= 0 → cut through full depth).

    Points far from the plane (outside the blade footprint, or outside the
    thickness band along Z/normal) are left uncut and stay with part A.
    """
    rel = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
    u = (rel @ i_axis) / max(float(half_i), 1e-9)
    v = (rel @ j_axis) / max(float(half_j), 1e-9)
    height = rel @ normal
    inside_uv = (np.abs(u) <= 1.0 + _UV_BOUNDS_EPS) & (
        np.abs(v) <= 1.0 + _UV_BOUNDS_EPS
    )

    scalar = np.full(len(points), _UNCUT_SCALAR, dtype=float)
    if not np.any(inside_uv):
        return scalar

    query = np.stack([u[inside_uv], v[inside_uv]], axis=1)
    surf_h = _idw_displacements(query, ctrl_uv, ctrl_disp)
    signed = height[inside_uv] - surf_h

    thickness = float(knife_thickness_mm)
    if thickness <= 0.0:
        # Through-cut under the blade footprint (full depth along normal).
        in_knife = np.ones(np.count_nonzero(inside_uv), dtype=bool)
    else:
        # Thin knife: only the band around the surface is severed.
        in_knife = np.abs(signed) <= (0.5 * thickness + _UV_BOUNDS_EPS)

    cut_idx = np.flatnonzero(inside_uv)[in_knife]
    scalar[cut_idx] = signed[in_knife]
    return scalar


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
    progress: ProgressCallback | None = None,
) -> CurveSliceResult:
    """
    Cut ``model_mesh`` with a deformed plane like a knife.

    Only geometry under the plane rectangle and within the knife thickness
    band (along the plane normal) is split. Points far from the plane are
    not cut and stay with part A. Thickness <= 0 cuts through the full
    depth under the blade.
    """
    ctrl_uv = np.asarray(ctrl_uv, dtype=float)
    ctrl_disp = np.asarray(ctrl_disp, dtype=float)
    if len(ctrl_uv) == 0:
        raise ValueError("At least one grid point is required.")
    if len(ctrl_disp) != len(ctrl_uv):
        raise ValueError("ctrl_disp length must match ctrl_uv.")
    if len(ctrl_uv) > MAX_GRID_POINTS:
        raise ValueError(f"Grid points must be at most {MAX_GRID_POINTS}.")

    if progress:
        progress("Preparing knife cut", 0, 4)

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
        progress("Finding where the knife touches the model", 1, 4)

    work = model_mesh.copy(deep=True)
    scalars = _knife_cut_scalars(
        np.asarray(work.points, dtype=float),
        center=center,
        i_axis=i_axis,
        j_axis=j_axis,
        normal=normal,
        half_i=half_i,
        half_j=half_j,
        ctrl_uv=ctrl_uv,
        ctrl_disp=ctrl_disp,
        knife_thickness_mm=float(knife_thickness_mm),
    )
    work["cut_side"] = scalars

    if progress:
        progress("Clipping into two parts", 2, 4)

    part_a_raw = work.clip_scalar(
        scalars="cut_side",
        value=0.0,
        invert=False,
    )
    part_b_raw = work.clip_scalar(
        scalars="cut_side",
        value=0.0,
        invert=True,
    )
    if part_a_raw.n_cells == 0 or part_b_raw.n_cells == 0:
        raise ValueError(
            "Knife cut left one side empty. Move the plane so it touches the "
            "model, or increase knife thickness / plane size."
        )

    part_a = _as_surface(part_a_raw)
    part_b = _as_surface(part_b_raw)

    if progress:
        progress("Saving slice parts", 3, 4)

    output_dir.mkdir(parents=True, exist_ok=True)
    path_a, path_b = curve_slice_output_paths(output_dir)
    part_a.save(str(path_a))
    part_b.save(str(path_b))

    if progress:
        progress("Curve slice complete", 4, 4)

    return CurveSliceResult(
        part_a=part_a,
        part_b=part_b,
        path_a=path_a,
        path_b=path_b,
        point_count=int(len(ctrl_uv)),
    )
