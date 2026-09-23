"""XY / oriented plane helpers for mold tools."""

from __future__ import annotations

import math

import pyvista as pv

_PLANE_MARGIN_MM = 2.0


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
