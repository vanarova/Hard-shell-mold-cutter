"""Clear extra wrap material between planes, preserving model + padding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy import ndimage

from app.model_repair import _build_extruded_solid_masked
from app.plane_wrap import (
    DEFAULT_POINT_DENSITY,
    MAX_POINT_DENSITY,
    MAX_WRAP_BACK_DENSITY,
    MIN_POINT_DENSITY,
    _plane_half_extents,
    _ray_stop_on_meshes,
    _sample_deformed_end_grid,
    _sample_oriented_plane_grid,
)
from app.xy_plane import plane_normal_from_rotations

ProgressCallback = Callable[[str, int, int], None]

DEFAULT_PADDING_MM = 2.0
MIN_PADDING_MM = 0.0
MAX_PADDING_MM = 200.0


@dataclass(frozen=True)
class ClearExtraResult:
    clear_mesh: pv.PolyData
    output_path: Path
    point_count: int
    kept_count: int
    cleared_count: int
    density: int
    padding_mm: float


def clear_start_to_end_output_path(output_dir: Path) -> Path:
    return output_dir / "clear_start_to_end.stl"


def clear_back_to_backend_output_path(output_dir: Path) -> Path:
    return output_dir / "clear_back_start_to_backend.stl"


def _grid_cell_pitch_mm(half_extent: float, density: int) -> float:
    """Spacing between adjacent samples on an N×N plane grid."""
    if density <= 1:
        return max(float(half_extent) * 2.0, 1e-6)
    return (2.0 * float(half_extent)) / float(density - 1)


def _dilate_keep_mask(
    keep: np.ndarray,
    *,
    padding_mm: float,
    half_extent: float,
    density: int,
) -> np.ndarray:
    """
    Expand model-hit columns by padding using fast 2D grid dilation.

    Approximates “keep within padding of the model” on the UV sample grid
    without per-ray nearest-point queries (previous main bottleneck).
    """
    padding = max(0.0, float(padding_mm))
    if padding <= 0.0 or not np.any(keep):
        return np.asarray(keep, dtype=bool)

    pitch = _grid_cell_pitch_mm(half_extent, density)
    radius = int(np.ceil(padding / max(pitch, 1e-9)))
    if radius <= 0:
        return np.asarray(keep, dtype=bool)

    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    structure = (xx * xx + yy * yy) <= (radius * radius)
    return ndimage.binary_dilation(np.asarray(keep, dtype=bool), structure=structure)


def _clear_between_grids(
    *,
    model_mesh: pv.PolyData,
    output_path: Path,
    bottom: np.ndarray,
    top_targets: np.ndarray,
    density: int,
    padding_mm: float,
    half_extent: float,
    stop_mesh: pv.PolyData | None,
    progress: ProgressCallback | None,
) -> ClearExtraResult:
    density = int(density)
    padding_mm = float(padding_mm)
    if padding_mm < MIN_PADDING_MM or padding_mm > MAX_PADDING_MM:
        raise ValueError(
            f"Padding must be between {MIN_PADDING_MM:g} and {MAX_PADDING_MM:g} mm."
        )

    point_count = density * density
    origins = bottom.reshape(-1, 3)
    targets = top_targets.reshape(-1, 3)

    if progress:
        progress("Ray casting wrap stops", 1, 4)

    # Single raycast: reuse model-hit mask (no second cast / nearest queries).
    wrapped, hit_model_count, _hit_end_count, hit_model = _ray_stop_on_meshes(
        origins=origins,
        targets=targets,
        model_mesh=model_mesh,
        stop_mesh=stop_mesh,
        return_hit_mask=True,
    )
    top = wrapped.reshape(density, density, 3)
    keep = np.asarray(hit_model, dtype=bool).reshape(density, density)

    if progress:
        progress("Applying padding (grid dilate)", 2, 4)

    keep = _dilate_keep_mask(
        keep,
        padding_mm=padding_mm,
        half_extent=half_extent,
        density=density,
    )

    kept_count = int(np.count_nonzero(keep))
    cleared_count = int(point_count - kept_count)
    if kept_count == 0:
        raise ValueError(
            "Every ray was cleared. Reduce padding, move the planes, or raise "
            "grid density so some rays hit the model."
        )

    if progress:
        progress("Building cleared solid", 3, 4)

    clear_mesh = _build_extruded_solid_masked(bottom, top, keep)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clear_mesh.save(str(output_path))

    if progress:
        progress("Clear extra complete", 4, 4)

    # hit_model_count kept for sanity; dilated keep may exceed it.
    _ = hit_model_count

    return ClearExtraResult(
        clear_mesh=clear_mesh,
        output_path=output_path,
        point_count=point_count,
        kept_count=kept_count,
        cleared_count=cleared_count,
        density=density,
        padding_mm=padding_mm,
    )


def clear_start_to_end(
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
    padding_mm: float = DEFAULT_PADDING_MM,
    progress: ProgressCallback | None = None,
) -> ClearExtraResult:
    """
    Clear wrap material between Start and End away from the model.

    Rays that hit the model (plus a UV-grid padding halo) are kept.
    All other ray columns are removed from the solid.
    """
    density = int(density)
    if density < MIN_POINT_DENSITY or density > MAX_POINT_DENSITY:
        raise ValueError(
            f"Point density must be between {MIN_POINT_DENSITY} and {MAX_POINT_DENSITY}."
        )

    if end_ctrl_uv is None:
        end_ctrl_uv = np.zeros((1, 2), dtype=float)
    if end_ctrl_disp is None:
        end_ctrl_disp = np.zeros(len(end_ctrl_uv), dtype=float)
    end_ctrl_uv = np.asarray(end_ctrl_uv, dtype=float)
    end_ctrl_disp = np.asarray(end_ctrl_disp, dtype=float)

    if progress:
        progress("Sampling Start and End planes", 0, 4)

    half_extent = _plane_half_extents(model_mesh)
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

    bottom = _sample_oriented_plane_grid(
        center=start_center,
        normal=start_normal,
        half_extent=half_extent,
        density=density,
    )
    top_targets = _sample_deformed_end_grid(
        center=end_center,
        normal=end_normal,
        half_extent=half_extent,
        density=density,
        end_ctrl_uv=end_ctrl_uv,
        end_ctrl_disp=end_ctrl_disp,
    )

    return _clear_between_grids(
        model_mesh=model_mesh,
        output_path=clear_start_to_end_output_path(output_dir),
        bottom=bottom,
        top_targets=top_targets,
        density=density,
        padding_mm=padding_mm,
        half_extent=half_extent,
        stop_mesh=None,
        progress=progress,
    )


def clear_back_start_to_backend(
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
    padding_mm: float = DEFAULT_PADDING_MM,
    progress: ProgressCallback | None = None,
) -> ClearExtraResult:
    """
    Clear wrap-back material between Back-start and Backend away from the model.

    Rays that hit the model (plus a UV-grid padding halo) are kept.
    All other ray columns are removed from the solid.
    """
    density = int(density)
    if density < 2 or density > MAX_WRAP_BACK_DENSITY:
        raise ValueError(
            f"Wrap-back density must be between 2 and {MAX_WRAP_BACK_DENSITY}."
        )

    backend_ctrl_uv = np.asarray(backend_ctrl_uv, dtype=float)
    backend_ctrl_disp = np.asarray(backend_ctrl_disp, dtype=float)

    if progress:
        progress("Sampling Back-start and Backend planes", 0, 4)

    if half_extent is None:
        half_extent = _plane_half_extents(model_mesh)

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

    bottom = _sample_oriented_plane_grid(
        center=start_center,
        normal=start_normal,
        half_extent=half_extent,
        density=density,
    )
    top_targets = _sample_deformed_end_grid(
        center=end_center,
        normal=end_normal,
        half_extent=half_extent,
        density=density,
        end_ctrl_uv=backend_ctrl_uv,
        end_ctrl_disp=backend_ctrl_disp,
    )

    return _clear_between_grids(
        model_mesh=model_mesh,
        output_path=clear_back_to_backend_output_path(output_dir),
        bottom=bottom,
        top_targets=top_targets,
        density=density,
        padding_mm=padding_mm,
        half_extent=float(half_extent),
        stop_mesh=backend_mesh,
        progress=progress,
    )
