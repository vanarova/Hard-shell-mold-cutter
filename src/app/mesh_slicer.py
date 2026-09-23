"""Axis-aware STL mesh slicing utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

from app.slice_axis import SliceAxis, axis_extent_mm

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class SliceResult:
    output_dir: Path
    piece_paths: list[Path]
    skipped_slices: int
    axis: SliceAxis


def estimate_slice_count(
    mesh: pv.PolyData,
    slice_height_mm: float,
    axis: SliceAxis = SliceAxis.Z,
    *,
    z_bounds: tuple[float, float] | None = None,
) -> int:
    """Estimate how many slices will be produced along an axis."""
    if slice_height_mm <= 0:
        return 0
    if axis == SliceAxis.Z and z_bounds is not None:
        z_lo, z_hi = z_bounds
        extent = max(float(z_hi) - float(z_lo), 0.0)
    else:
        extent = axis_extent_mm(mesh, axis)
    if extent <= 0:
        return 0
    return int(np.ceil(extent / slice_height_mm))


def estimate_z_slice_count(mesh: pv.PolyData, slice_height_mm: float) -> int:
    """Estimate how many Z slices will be produced."""
    return estimate_slice_count(mesh, slice_height_mm, SliceAxis.Z)


def slice_mesh_along_axis(
    mesh: pv.PolyData,
    output_dir: Path,
    slice_height_mm: float,
    axis: SliceAxis = SliceAxis.Z,
    progress: ProgressCallback | None = None,
    *,
    z_bounds: tuple[float, float] | None = None,
    clear_existing_slices: bool = False,
) -> SliceResult:
    """Slice a mesh along X, Y, or Z into fixed-thickness layers."""
    if slice_height_mm <= 0:
        raise ValueError("Slice height must be greater than 0 mm.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing_slices:
        _clear_numbered_slice_files(output_dir)

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    work_mesh = mesh

    if axis == SliceAxis.Z and z_bounds is not None:
        z_lo = float(min(z_bounds))
        z_hi = float(max(z_bounds))
        if z_hi <= z_lo:
            raise ValueError("End plane must be above begin plane.")
        work_mesh = mesh.clip_box([xmin, xmax, ymin, ymax, z_lo, z_hi], invert=False)
        if work_mesh.n_cells == 0:
            return SliceResult(
                output_dir=output_dir,
                piece_paths=[],
                skipped_slices=0,
                axis=axis,
            )
        xmin, xmax, ymin, ymax, _, _ = work_mesh.bounds
        zmin, zmax = z_lo, z_hi

    if axis == SliceAxis.X:
        starts = np.arange(xmin, xmax, slice_height_mm)
        axis_name = "X"
    elif axis == SliceAxis.Y:
        starts = np.arange(ymin, ymax, slice_height_mm)
        axis_name = "Y"
    else:
        starts = np.arange(zmin, zmax, slice_height_mm)
        axis_name = "Z"

    total = len(starts)

    if progress:
        progress(f"Preparing {axis_name}-axis slices", 0, max(total, 1))

    piece_paths: list[Path] = []
    skipped = 0

    for index, start in enumerate(starts, start=1):
        if progress:
            progress(f"Slicing {axis_name} layer {index} of {total}", index - 1, total)

        if axis == SliceAxis.X:
            end = min(float(start) + slice_height_mm, xmax)
            if end <= start:
                continue
            box_bounds = [start, end, ymin, ymax, zmin, zmax]
        elif axis == SliceAxis.Y:
            end = min(float(start) + slice_height_mm, ymax)
            if end <= start:
                continue
            box_bounds = [xmin, xmax, start, end, zmin, zmax]
        else:
            end = min(float(start) + slice_height_mm, zmax)
            if end <= start:
                continue
            box_bounds = [xmin, xmax, ymin, ymax, start, end]

        piece = work_mesh.clip_box(box_bounds, invert=False)
        if piece.n_cells == 0:
            skipped += 1
            continue

        surface = piece.extract_surface(algorithm="dataset_surface").triangulate()
        if surface.n_cells == 0:
            skipped += 1
            continue

        piece_path = output_dir / f"{index}.stl"
        surface.save(piece_path)
        piece_paths.append(piece_path)

    if progress:
        progress("Slice export complete", total, total)

    return SliceResult(
        output_dir=output_dir,
        piece_paths=piece_paths,
        skipped_slices=skipped,
        axis=axis,
    )


def _clear_numbered_slice_files(output_dir: Path) -> None:
    """Remove numbered slice STL files from a slice output folder."""
    for path in output_dir.glob("*.stl"):
        if path.stem.isdigit():
            path.unlink(missing_ok=True)


def slice_single_at_z(
    mesh: pv.PolyData,
    output_dir: Path,
    z_start_mm: float,
    slice_height_mm: float,
    progress: ProgressCallback | None = None,
    *,
    rotate_x_deg: float = 0.0,
    rotate_y_deg: float = 0.0,
    rotate_z_deg: float = 0.0,
) -> SliceResult:
    """Export one slice starting at the marked plane with fixed thickness.

    When any of ``rotate_x_deg`` / ``rotate_y_deg`` / ``rotate_z_deg`` are
    non-zero, the slab is cut along the oriented plane normal (pivot at mesh
    XY center, Z = ``z_start_mm``).
    """
    if slice_height_mm <= 0:
        raise ValueError("Slice height must be greater than 0 mm.")

    output_dir.mkdir(parents=True, exist_ok=True)
    thickness = float(slice_height_mm)
    z_lo = float(z_start_mm)

    if progress:
        progress("Slicing layer at marked plane", 0, 1)

    untilted = (
        abs(rotate_x_deg) <= 1e-9
        and abs(rotate_y_deg) <= 1e-9
        and abs(rotate_z_deg) <= 1e-9
    )
    if untilted:
        xmin, xmax, ymin, ymax, _zmin, zmax = mesh.bounds
        z_hi = min(z_lo + thickness, float(zmax))
        if z_hi <= z_lo:
            raise ValueError(
                f"No geometry above the marked plane (Z = {z_lo:.3f} mm) "
                f"within the slice thickness."
            )
        piece = mesh.clip_box([xmin, xmax, ymin, ymax, z_lo, z_hi], invert=False)
    else:
        from app.xy_plane import plane_normal_from_rotations, plane_origin

        origin = np.asarray(plane_origin(mesh, z_lo), dtype=float)
        normal = np.asarray(
            plane_normal_from_rotations(rotate_x_deg, rotate_y_deg, rotate_z_deg),
            dtype=float,
        )
        # Keep the half-space on the +normal side of the start plane, then
        # cut again so thickness is measured along the normal.
        far_origin = origin + normal * thickness
        piece = mesh.clip(normal=tuple(normal), origin=tuple(origin), invert=False)
        if piece.n_cells > 0:
            piece = piece.clip(
                normal=tuple(-normal),
                origin=tuple(far_origin),
                invert=False,
            )

    if piece.n_cells == 0:
        return SliceResult(
            output_dir=output_dir,
            piece_paths=[],
            skipped_slices=1,
            axis=SliceAxis.Z,
        )

    surface = piece.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        return SliceResult(
            output_dir=output_dir,
            piece_paths=[],
            skipped_slices=1,
            axis=SliceAxis.Z,
        )

    piece_path = output_dir / "1.stl"
    surface.save(piece_path)

    if progress:
        progress("Slice export complete", 1, 1)

    return SliceResult(
        output_dir=output_dir,
        piece_paths=[piece_path],
        skipped_slices=0,
        axis=SliceAxis.Z,
    )


def slice_between_oriented_planes(
    mesh: pv.PolyData,
    output_dir: Path,
    *,
    start_z_mm: float,
    start_x_mm: float | None = None,
    start_y_mm: float | None = None,
    start_rx_deg: float = 0.0,
    start_ry_deg: float = 0.0,
    start_rz_deg: float = 0.0,
    end_z_mm: float,
    end_x_mm: float | None = None,
    end_y_mm: float | None = None,
    end_rx_deg: float = 0.0,
    end_ry_deg: float = 0.0,
    end_rz_deg: float = 0.0,
    progress: ProgressCallback | None = None,
) -> SliceResult:
    """Export one slab kept between a start plane and an end plane.

    Geometry on the +normal side of the start plane and on the -normal side of
    the end plane is kept (normals follow Rx → Ry → Rz from +Z). Plane pivots
    default to the mesh XY center when X/Y are omitted.
    """
    from app.xy_plane import plane_normal_from_rotations, plane_origin

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("Slicing between start and end planes", 0, 1)

    start_origin = np.asarray(
        plane_origin(mesh, start_z_mm, x=start_x_mm, y=start_y_mm),
        dtype=float,
    )
    end_origin = np.asarray(
        plane_origin(mesh, end_z_mm, x=end_x_mm, y=end_y_mm),
        dtype=float,
    )
    start_normal = np.asarray(
        plane_normal_from_rotations(start_rx_deg, start_ry_deg, start_rz_deg),
        dtype=float,
    )
    end_normal = np.asarray(
        plane_normal_from_rotations(end_rx_deg, end_ry_deg, end_rz_deg),
        dtype=float,
    )

    # If the end plane lies behind the start along the start normal, flip the
    # clip sense so the slab between the two pivots is still non-empty.
    mid = 0.5 * (start_origin + end_origin)
    start_sign = 1.0 if float(np.dot(mid - start_origin, start_normal)) >= 0 else -1.0
    end_sign = 1.0 if float(np.dot(mid - end_origin, end_normal)) >= 0 else -1.0

    piece = mesh.clip(
        normal=tuple(start_sign * start_normal),
        origin=tuple(start_origin),
        invert=False,
    )
    if piece.n_cells > 0:
        piece = piece.clip(
            normal=tuple(end_sign * end_normal),
            origin=tuple(end_origin),
            invert=False,
        )

    if piece.n_cells == 0:
        return SliceResult(
            output_dir=output_dir,
            piece_paths=[],
            skipped_slices=1,
            axis=SliceAxis.Z,
        )

    surface = piece.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        return SliceResult(
            output_dir=output_dir,
            piece_paths=[],
            skipped_slices=1,
            axis=SliceAxis.Z,
        )

    piece_path = output_dir / "1.stl"
    surface.save(piece_path)

    if progress:
        progress("Slice export complete", 1, 1)

    return SliceResult(
        output_dir=output_dir,
        piece_paths=[piece_path],
        skipped_slices=0,
        axis=SliceAxis.Z,
    )


def slice_mesh_along_z(
    mesh: pv.PolyData,
    output_dir: Path,
    slice_height_mm: float,
    progress: ProgressCallback | None = None,
    *,
    z_bounds: tuple[float, float] | None = None,
    clear_existing_slices: bool = False,
) -> SliceResult:
    """Slice a mesh along the Z axis into fixed-height layers."""
    return slice_mesh_along_axis(
        mesh,
        output_dir,
        slice_height_mm,
        axis=SliceAxis.Z,
        progress=progress,
        z_bounds=z_bounds,
        clear_existing_slices=clear_existing_slices,
    )
