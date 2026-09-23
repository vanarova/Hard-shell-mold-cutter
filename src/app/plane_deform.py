"""Equidistant control-point deformation for an oriented end plane."""

from __future__ import annotations

import math

import numpy as np
import pyvista as pv

from app.xy_plane import plane_normal_from_rotations

_PLANE_MARGIN_MM = 2.0
_DISPLAY_RESOLUTION = 48
_IDW_POWER = 2.0
_IDW_EPS = 1e-8

DEFAULT_CONTROL_POINTS = 9
MIN_CONTROL_POINTS = 1
MAX_CONTROL_POINTS = 64


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


def plane_half_extent(mesh: pv.PolyData, margin_mm: float = _PLANE_MARGIN_MM) -> float:
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in mesh.bounds)
    diag = math.sqrt(
        (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
    )
    return 0.5 * (max(diag, 1e-3) + 2.0 * margin_mm)


def place_equidistant_uv(n_points: int) -> np.ndarray:
    """
    Place ``n_points`` equidistant samples in the unit square [-1, 1] × [-1, 1].

    Uses a compact rectangular grid (as square as possible).
    """
    n = max(1, int(n_points))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    us = np.linspace(-1.0, 1.0, cols) if cols > 1 else np.array([0.0])
    vs = np.linspace(-1.0, 1.0, rows) if rows > 1 else np.array([0.0])
    coords: list[list[float]] = []
    for v in vs:
        for u in us:
            coords.append([float(u), float(v)])
            if len(coords) >= n:
                return np.asarray(coords, dtype=float)
    return np.asarray(coords, dtype=float)


def place_nxn_uv(n_per_side: int) -> np.ndarray:
    """Place an exact ``n × n`` grid in the unit square [-1, 1] × [-1, 1]."""
    n = max(1, int(n_per_side))
    if n == 1:
        return np.asarray([[0.0, 0.0]], dtype=float)
    u = np.linspace(-1.0, 1.0, n)
    v = np.linspace(-1.0, 1.0, n)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def control_points_world(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_extent: float,
    uv: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Return deformed control-point positions in world space, shape (n, 3)."""
    i_axis, j_axis = _plane_basis(normal)
    n = _normalize(normal)
    rest = (
        center[None, :]
        + (uv[:, 0:1] * half_extent) * i_axis[None, :]
        + (uv[:, 1:2] * half_extent) * j_axis[None, :]
    )
    return rest + displacements[:, None] * n[None, :]


def sample_deformed_plane_grid(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_extent: float,
    uv: np.ndarray,
    displacements: np.ndarray,
    density: int,
) -> np.ndarray:
    """
    Sample the deformed plane on a regular parametric grid.

    Returns points with shape (density, density, 3), matching wrap start-grid UVs.
    """
    i_axis, j_axis = _plane_basis(normal)
    n = _normalize(normal)
    res = max(2, int(density))
    u = np.linspace(-1.0, 1.0, res)
    v = np.linspace(-1.0, 1.0, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    rest = (
        center[None, None, :]
        + (uu[:, :, None] * half_extent) * i_axis[None, None, :]
        + (vv[:, :, None] * half_extent) * j_axis[None, None, :]
    )

    query = np.stack([uu.ravel(), vv.ravel()], axis=1)
    ctrl = np.asarray(uv, dtype=float)
    disp = np.asarray(displacements, dtype=float)
    if len(ctrl) == 0:
        field = np.zeros(query.shape[0], dtype=float)
    elif len(ctrl) == 1:
        field = np.full(query.shape[0], float(disp[0]), dtype=float)
    else:
        delta = query[:, None, :] - ctrl[None, :, :]
        dist2 = np.sum(delta * delta, axis=2)
        exact = dist2 < _IDW_EPS
        weights = 1.0 / np.power(np.maximum(dist2, _IDW_EPS), _IDW_POWER / 2.0)
        field = np.sum(weights * disp[None, :], axis=1) / np.sum(weights, axis=1)
        if np.any(exact):
            hit_rows, hit_cols = np.where(exact)
            field[hit_rows] = disp[hit_cols]

    deformed = rest.reshape(-1, 3) + field[:, None] * n[None, :]
    return deformed.reshape(res, res, 3)


def make_deformed_plane_mesh(
    *,
    center: np.ndarray,
    normal: np.ndarray,
    half_extent: float,
    uv: np.ndarray,
    displacements: np.ndarray,
    resolution: int = _DISPLAY_RESOLUTION,
) -> pv.PolyData:
    """
    Build a dense plane mesh deformed by IDW of control-point pull/push amounts.

    Displacements are along ``normal`` (positive = push along normal).
    """
    pts = sample_deformed_plane_grid(
        center=center,
        normal=normal,
        half_extent=half_extent,
        uv=uv,
        displacements=displacements,
        density=resolution,
    )
    grid = pv.StructuredGrid(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2])
    return grid.extract_surface(algorithm="dataset_surface").triangulate()


def control_point_glyphs(
    points: np.ndarray,
    *,
    selected_index: int,
    radius: float,
) -> tuple[pv.PolyData | None, pv.PolyData | None]:
    """Return (other_points_mesh, selected_point_mesh) as small spheres."""
    if len(points) == 0:
        return None, None
    selected_index = int(selected_index)
    others = []
    selected = None
    for i, p in enumerate(points):
        sphere = pv.Sphere(radius=radius, center=tuple(p), theta_resolution=12, phi_resolution=12)
        if i == selected_index:
            selected = sphere
        else:
            others.append(sphere)
    other_mesh = None
    if others:
        other_mesh = others[0]
        for s in others[1:]:
            other_mesh = other_mesh.merge(s)
    return other_mesh, selected


def point_cloud_glyphs(points: np.ndarray, *, radius: float) -> pv.PolyData | None:
    """Render many points as identical spheres via a single glyph call."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return None
    cloud = pv.PolyData(pts)
    sphere = pv.Sphere(
        radius=max(float(radius), 1e-4),
        theta_resolution=8,
        phi_resolution=8,
    )
    return cloud.glyph(geom=sphere, scale=False, orient=False)


def pose_frame(
    *,
    x: float,
    y: float,
    z: float,
    rx: float,
    ry: float,
    rz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return center and unit normal for a plane pose."""
    center = np.asarray((x, y, z), dtype=float)
    normal = np.asarray(plane_normal_from_rotations(rx, ry, rz), dtype=float)
    return center, _normalize(normal)
