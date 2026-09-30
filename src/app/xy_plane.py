"""XY / oriented plane helpers for mold tools."""

from __future__ import annotations

import math

import numpy as np
import pyvista as pv

_PLANE_MARGIN_MM = 2.0
# Repair planes extend 10% of each in-plane model span beyond both sides
# (e.g. 100 mm → 10 mm each side → 120 mm total).
REPAIR_SIDE_MARGIN_FRACTION = 0.10
_REPAIR_YZ_RX_DEG = 0.0
_REPAIR_YZ_RY_DEG = 90.0
_REPAIR_YZ_RZ_DEG = 0.0


def mesh_xy_bounds(mesh: pv.PolyData) -> tuple[float, float, float, float]:
    """Return xmin, xmax, ymin, ymax for a mesh."""
    xmin, xmax, ymin, ymax, _, _ = mesh.bounds
    return float(xmin), float(xmax), float(ymin), float(ymax)


def mesh_x_bounds(mesh: pv.PolyData) -> tuple[float, float]:
    """Return xmin, xmax for a mesh."""
    xmin, xmax, _, _, _, _ = mesh.bounds
    return float(xmin), float(xmax)


def mesh_y_bounds(mesh: pv.PolyData) -> tuple[float, float]:
    """Return ymin, ymax for a mesh."""
    _, _, ymin, ymax, _, _ = mesh.bounds
    return float(ymin), float(ymax)


def mesh_z_bounds(mesh: pv.PolyData) -> tuple[float, float]:
    """Return zmin, zmax for a mesh."""
    _, _, _, _, zmin, zmax = mesh.bounds
    return float(zmin), float(zmax)


def mesh_center_xy(mesh: pv.PolyData) -> tuple[float, float]:
    """Return the XY center of a mesh bounds box."""
    xmin, xmax, ymin, ymax = mesh_xy_bounds(mesh)
    return (xmin + xmax) * 0.5, (ymin + ymax) * 0.5


def plane_normal_from_rotations(
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
) -> tuple[float, float, float]:
    """
    Unit normal after rotating +Z by Rx, then Ry, then Rz.

    Matches PyVista ``rotate_x`` / ``rotate_y`` / ``rotate_z`` order used by
    oriented planes.
    """
    rx = math.radians(rotate_x_deg)
    ry = math.radians(rotate_y_deg)
    rz = math.radians(rotate_z_deg)
    # +Z after Rx: (0, -sin(rx), cos(rx))
    # then Ry: (sin(ry)*cos(rx), -sin(rx), cos(ry)*cos(rx))
    nx = math.sin(ry) * math.cos(rx)
    ny = -math.sin(rx)
    nz = math.cos(ry) * math.cos(rx)
    # then Rz: rotate XY components
    cos_z = math.cos(rz)
    sin_z = math.sin(rz)
    nx2 = nx * cos_z - ny * sin_z
    ny2 = nx * sin_z + ny * cos_z
    length = math.sqrt(nx2 * nx2 + ny2 * ny2 + nz * nz)
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return (nx2 / length, ny2 / length, nz / length)


def plane_origin(
    mesh: pv.PolyData,
    z: float,
    *,
    x: float | None = None,
    y: float | None = None,
) -> tuple[float, float, float]:
    """Return the plane pivot point (defaults to mesh XY center at ``z``)."""
    cx, cy = mesh_center_xy(mesh)
    return (
        float(cx if x is None else x),
        float(cy if y is None else y),
        float(z),
    )


def make_xy_plane_mesh(
    mesh: pv.PolyData,
    z: float,
    *,
    margin_mm: float = _PLANE_MARGIN_MM,
) -> pv.PolyData:
    """Build a horizontal XY plane at the given Z level sized to the mesh footprint."""
    return make_oriented_plane_mesh(mesh, z, margin_mm=margin_mm)


def make_oriented_plane_mesh(
    mesh: pv.PolyData,
    z: float,
    *,
    center_x: float | None = None,
    center_y: float | None = None,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
    margin_mm: float = _PLANE_MARGIN_MM,
) -> pv.PolyData:
    """
    Build a plane at (center_x, center_y, z), rotated about X, then Y, then Z.

    ``center_x`` / ``center_y`` default to the mesh XY center. Size uses the mesh
    bounding-box diagonal so tilted planes still cover the model.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in mesh.bounds)
    cx = float(center_x) if center_x is not None else (xmin + xmax) * 0.5
    cy = float(center_y) if center_y is not None else (ymin + ymax) * 0.5
    diag = math.sqrt(
        (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
    )
    size = max(diag, 1e-3) + 2.0 * margin_mm
    center = (cx, cy, float(z))

    plane = pv.Plane(
        center=center,
        direction=(0.0, 0.0, 1.0),
        i_size=size,
        j_size=size,
    )
    if abs(rotate_x_deg) > 1e-9:
        plane.rotate_x(float(rotate_x_deg), point=center, inplace=True)
    if abs(rotate_y_deg) > 1e-9:
        plane.rotate_y(float(rotate_y_deg), point=center, inplace=True)
    if abs(rotate_z_deg) > 1e-9:
        plane.rotate_z(float(rotate_z_deg), point=center, inplace=True)
    return plane


def repair_yz_default_rotations() -> tuple[float, float, float]:
    """Default Rx/Ry/Rz that orient a repair plane onto the YZ axis (normal +X)."""
    return (_REPAIR_YZ_RX_DEG, _REPAIR_YZ_RY_DEG, _REPAIR_YZ_RZ_DEG)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return vector / length


def repair_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes for a repair plane (same convention as wrap sampling)."""
    n = _normalize_vector(np.asarray(normal, dtype=float))
    helper = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(n[0])) > 0.9:
        helper = np.array([0.0, 0.0, 1.0], dtype=float)
    i_axis = _normalize_vector(np.cross(helper, n))
    j_axis = _normalize_vector(np.cross(n, i_axis))
    return i_axis, j_axis


def repair_plane_half_extents(
    mesh: pv.PolyData,
    normal: tuple[float, float, float] | np.ndarray,
    *,
    side_margin_fraction: float = REPAIR_SIDE_MARGIN_FRACTION,
) -> tuple[float, float]:
    """
    Half-sizes of a repair plane covering the model in its local i/j axes.

    Each in-plane span is the projected model AABB extent, enlarged by
    ``side_margin_fraction`` on both sides (0.10 → 20% longer total, e.g.
    100 mm → 120 mm).
    """
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
    i_axis, j_axis = repair_plane_basis(np.asarray(normal, dtype=float))
    ui = corners @ i_axis
    vj = corners @ j_axis
    span_i = max(float(ui.max() - ui.min()), 1e-3)
    span_j = max(float(vj.max() - vj.min()), 1e-3)
    scale = 1.0 + 2.0 * float(side_margin_fraction)
    return 0.5 * span_i * scale, 0.5 * span_j * scale


def make_repair_plane_mesh(
    mesh: pv.PolyData,
    *,
    center_x: float,
    center_y: float,
    center_z: float,
    rotate_x_deg: float = _REPAIR_YZ_RX_DEG,
    rotate_y_deg: float = _REPAIR_YZ_RY_DEG,
    rotate_z_deg: float = _REPAIR_YZ_RZ_DEG,
    side_margin_fraction: float = REPAIR_SIDE_MARGIN_FRACTION,
    resolution: int = 2,
) -> pv.PolyData:
    """
    Build a repair plane (default YZ / normal +X) sized to the model footprint
    in the plane, plus ``side_margin_fraction`` beyond both sides on each axis.
    """
    center = np.array(
        [float(center_x), float(center_y), float(center_z)], dtype=float
    )
    normal = np.asarray(
        plane_normal_from_rotations(rotate_x_deg, rotate_y_deg, rotate_z_deg),
        dtype=float,
    )
    half_i, half_j = repair_plane_half_extents(
        mesh,
        normal,
        side_margin_fraction=side_margin_fraction,
    )
    i_axis, j_axis = repair_plane_basis(normal)
    res = max(2, int(resolution))
    u = np.linspace(-half_i, half_i, res)
    v = np.linspace(-half_j, half_j, res)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    points = (
        center[None, None, :]
        + uu[:, :, None] * i_axis[None, None, :]
        + vv[:, :, None] * j_axis[None, None, :]
    ).reshape(-1, 3)

    faces: list[int] = []
    for j in range(res - 1):
        for i in range(res - 1):
            a = j * res + i
            b = a + 1
            c = a + res + 1
            d = a + res
            faces.extend([4, a, b, c, d])
    return pv.PolyData(points, np.asarray(faces, dtype=np.int64))
