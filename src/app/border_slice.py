"""Border ring generation and Z extrusion for single slice exports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyvista as pv
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from app.boundary_generator import ProgressCallback, footprint_from_mesh

_MIN_Z_HEIGHT_MM = 0.1
_BUFFER_KW = {"join_style": 2, "mitre_limit": 10.0, "resolution": 64}


@dataclass
class BorderSliceResult:
    output_dir: Path
    border_path: Path
    triangle_count: int


@dataclass
class BorderExtrudeResult:
    output_dir: Path
    border_path: Path
    extruded_path: Path
    triangle_count: int


@dataclass
class BasePlateResult:
    output_dir: Path
    base_path: Path
    triangle_count: int


@dataclass
class BasePlaneResult:
    output_dir: Path
    plane_path: Path
    triangle_count: int


@dataclass
class AssemblyExportResult:
    output_dir: Path
    assembly_path: Path
    triangle_count: int
    part_count: int


_BASE_MARGIN_MM = 5.0
_BASE_HEIGHT_MM = 2.0


def slice_output_path(output_dir: Path) -> Path:
    return output_dir / "1.stl"


def border_output_path(output_dir: Path) -> Path:
    return output_dir / "border.stl"


def border_extruded_output_path(output_dir: Path) -> Path:
    return output_dir / "border_extruded.stl"


def base_plate_output_path(output_dir: Path) -> Path:
    return output_dir / "base_plate.stl"


def base_plane_output_path(output_dir: Path) -> Path:
    return output_dir / "base_plane.stl"


def assembly_output_path(output_dir: Path) -> Path:
    return output_dir / "assembly.stl"


def _polygons_from_footprint(footprint) -> list[Polygon]:
    if isinstance(footprint, Polygon):
        return [footprint] if not footprint.is_empty and footprint.area > 0 else []
    if isinstance(footprint, MultiPolygon):
        return [poly for poly in footprint.geoms if not poly.is_empty and poly.area > 0]
    raise ValueError("Unsupported footprint geometry.")


def _footprint_for_border(mesh: pv.PolyData):
    """
  Build a footprint that follows the full sliced outline.

  Uses the cross-section contour when possible so the border follows the
  longest outer perimeter instead of shortcuts (e.g. convex hull).
    """
    from app.boundary_generator import footprint_from_cross_section

    footprint = footprint_from_cross_section(mesh)
    if footprint is None:
        footprint = footprint_from_mesh(mesh)
    if footprint is None:
        raise ValueError("Could not build a footprint from the slice mesh.")

    combined = unary_union(footprint).buffer(0)
    if combined.is_empty:
        raise ValueError("Slice footprint is empty.")
    return combined


def _buffer_outline(poly: Polygon, distance_mm: float):
    return poly.buffer(distance_mm, **_BUFFER_KW)


def _ring_for_polygon(
    poly: Polygon,
    offset_mm: float,
    thickness_mm: float,
) -> Polygon | MultiPolygon:
    """
    Build a border ring by offsetting the full exterior perimeter outward.

    Mitre joins keep the offset on the long path around concave corners
    instead of rounding across them.
    """
    inner = _buffer_outline(poly, offset_mm)
    outer = _buffer_outline(poly, offset_mm + thickness_mm)
    ring = outer.difference(inner).buffer(0)
    if ring.is_empty:
        raise ValueError("Could not build border ring from the slice footprint.")
    return ring


def _ring_for_separate_objects(
    polygons: list[Polygon],
    combined,
    offset_mm: float,
    thickness_mm: float,
) -> Polygon | MultiPolygon:
    """
    Build one border around a group of separate slice objects.

    Offset zones from each object are merged first so intersecting border
    sections between objects are discarded. If the objects are too far apart
    for their offset zones to meet, fall back to one outer border around the
    whole group (convex hull envelope).
    """
    inner = unary_union([_buffer_outline(poly, offset_mm) for poly in polygons]).buffer(0)
    outer = unary_union([
        _buffer_outline(poly, offset_mm + thickness_mm) for poly in polygons
    ]).buffer(0)
    ring = outer.difference(inner).buffer(0)
    if ring.is_empty:
        raise ValueError("Could not build border ring from the slice footprint.")

    if isinstance(ring, MultiPolygon):
        ring = _ring_for_polygon(combined.convex_hull, offset_mm, thickness_mm)
    return ring


def _ring_polygon(footprint, offset_mm: float, thickness_mm: float):
    if offset_mm <= 0:
        raise ValueError("Border offset must be greater than 0 mm.")
    if thickness_mm <= 0:
        raise ValueError("Border thickness must be greater than 0 mm.")

    combined = unary_union(footprint).buffer(0)
    polygons = _polygons_from_footprint(combined)
    if not polygons:
        raise ValueError("Slice footprint is empty.")

    if len(polygons) == 1:
        return _ring_for_polygon(polygons[0], offset_mm, thickness_mm)

    return _ring_for_separate_objects(polygons, combined, offset_mm, thickness_mm)


def _mesh_from_ring_polygon(ring, z_lo: float, z_height: float) -> trimesh.Trimesh:
    polygons: list[Polygon] = []
    if isinstance(ring, Polygon):
        polygons = [ring]
    elif isinstance(ring, MultiPolygon):
        polygons = [poly for poly in ring.geoms if poly.area > 0]
    else:
        raise ValueError("Unsupported border ring geometry.")

    if not polygons:
        raise ValueError("Border ring has no usable polygons.")

    parts = [
        trimesh.creation.extrude_polygon(poly, height=z_height)
        for poly in polygons
    ]
    border = trimesh.util.concatenate(parts)
    border.apply_translation([0.0, 0.0, z_lo])
    border.merge_vertices()
    border.update_faces(border.unique_faces() & border.nondegenerate_faces())
    border.remove_unreferenced_vertices()
    return border


def create_border_from_slice(
    slice_path: Path,
    output_path: Path,
    offset_mm: float,
    thickness_mm: float,
    *,
    connected: bool = False,
    progress: ProgressCallback | None = None,
) -> BorderSliceResult:
    """Build a border around a slice STL.

    When ``connected`` is False, builds a ring using offset + thickness.
    When ``connected`` is True, builds a filled plane using offset only
    (thickness is ignored).
    """
    if not slice_path.is_file():
        raise FileNotFoundError(f"Slice file not found: {slice_path}")
    if offset_mm <= 0:
        raise ValueError("Border offset must be greater than 0 mm.")

    if progress is not None:
        progress("Reading slice mesh", 0, 3)

    mesh = pv.read(str(slice_path))
    if mesh.n_cells == 0:
        raise ValueError("Slice mesh is empty.")

    if progress is not None:
        progress(
            "Building connected border plane" if connected else "Building border ring",
            1,
            3,
        )

    footprint = _footprint_for_border(mesh)
    if connected:
        region = _connected_border_polygon(footprint, offset_mm)
    else:
        region = _ring_polygon(footprint, offset_mm, thickness_mm)

    z_lo = float(mesh.bounds[4])
    z_hi = float(mesh.bounds[5])
    z_height = max(z_hi - z_lo, _MIN_Z_HEIGHT_MM)
    border = _mesh_from_ring_polygon(region, z_lo, z_height)

    if progress is not None:
        progress("Saving border mesh", 2, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    border.export(output_path)

    if progress is not None:
        progress("Border export complete", 3, 3)

    return BorderSliceResult(
        output_dir=output_path.parent,
        border_path=output_path,
        triangle_count=int(len(border.faces)),
    )


def _connected_border_polygon(footprint, offset_mm: float):
    """Filled outline expanded by offset only (no ring thickness)."""
    combined = unary_union(footprint).buffer(0)
    solid = _buffer_outline(combined, offset_mm).buffer(0)
    if solid.is_empty:
        raise ValueError("Could not build connected border from the slice footprint.")
    # Close any holes so the result is a solid connected plane.
    return unary_union(_fill_enclosed_polygons(solid)).buffer(0)


def extrude_border_on_z(
    border_path: Path,
    output_path: Path,
    height_mm: float,
    progress: ProgressCallback | None = None,
) -> BorderExtrudeResult:
    """Increase the border mesh height by ``height_mm``, keeping the bottom fixed."""
    if height_mm <= 0:
        raise ValueError("Height increase must be greater than 0 mm.")
    if not border_path.is_file():
        raise FileNotFoundError(f"Border file not found: {border_path}")

    if progress is not None:
        progress("Reading border mesh", 0, 3)

    border = trimesh.load(border_path, force="mesh")
    if border.is_empty or len(border.faces) == 0:
        raise ValueError("Border mesh is empty.")

    if progress is not None:
        progress("Increasing border height", 1, 3)

    vertices = border.vertices.copy()
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())
    z_range = max(z_max - z_min, _MIN_Z_HEIGHT_MM)
    new_range = z_range + float(height_mm)
    vertices[:, 2] = z_min + (vertices[:, 2] - z_min) * (new_range / z_range)

    raised = trimesh.Trimesh(vertices=vertices, faces=border.faces, process=False)
    raised.merge_vertices()
    raised.update_faces(raised.unique_faces() & raised.nondegenerate_faces())
    raised.remove_unreferenced_vertices()

    if progress is not None:
        progress("Saving raised border", 2, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raised.export(output_path)

    if progress is not None:
        progress("Border height update complete", 3, 3)

    return BorderExtrudeResult(
        output_dir=output_path.parent,
        border_path=border_path,
        extruded_path=output_path,
        triangle_count=int(len(raised.faces)),
    )


def _fill_enclosed_polygons(footprint) -> list[Polygon]:
    """Fill holes in border-ring footprints so each region becomes a solid plane."""
    filled: list[Polygon] = []
    for poly in _polygons_from_footprint(unary_union(footprint).buffer(0)):
        solid = Polygon(poly.exterior).buffer(0)
        if isinstance(solid, Polygon) and not solid.is_empty and solid.area > 0:
            filled.append(solid)
        elif isinstance(solid, MultiPolygon):
            filled.extend(
                p for p in solid.geoms if not p.is_empty and p.area > 0
            )
    if not filled:
        raise ValueError("Could not fill the enclosed border space.")
    return filled


def create_base_plane_from_border(
    border_path: Path,
    output_path: Path,
    progress: ProgressCallback | None = None,
) -> BasePlaneResult:
    """
    Copy the sliced border footprint and fill its enclosed hole into a solid plane.

    Uses the outer outline of the border ring (holes closed) and extrudes a thin
    planar slab at the border's bottom Z.
    """
    if not border_path.is_file():
        raise FileNotFoundError(f"Border file not found: {border_path}")

    if progress is not None:
        progress("Reading border mesh", 0, 3)

    mesh = pv.read(str(border_path))
    if mesh.n_cells == 0:
        raise ValueError("Border mesh is empty.")

    if progress is not None:
        progress("Filling enclosed border space", 1, 3)

    footprint = footprint_from_mesh(mesh)
    if footprint is None:
        footprint = _footprint_for_border(mesh)
    filled_polys = _fill_enclosed_polygons(footprint)

    z_lo = float(mesh.bounds[4])
    z_hi = float(mesh.bounds[5])
    z_height = max(z_hi - z_lo, _MIN_Z_HEIGHT_MM)

    parts = [
        trimesh.creation.extrude_polygon(poly, height=z_height)
        for poly in filled_polys
    ]
    plane = trimesh.util.concatenate(parts)
    plane.apply_translation([0.0, 0.0, z_lo])
    plane.merge_vertices()
    plane.update_faces(plane.unique_faces() & plane.nondegenerate_faces())
    plane.remove_unreferenced_vertices()

    if progress is not None:
        progress("Saving base plane", 2, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plane.export(output_path)

    if progress is not None:
        progress("Base plane export complete", 3, 3)

    return BasePlaneResult(
        output_dir=output_path.parent,
        plane_path=output_path,
        triangle_count=int(len(plane.faces)),
    )


def create_base_plate(
    extruded_border_path: Path,
    output_path: Path,
    margin_mm: float = _BASE_MARGIN_MM,
    height_mm: float = _BASE_HEIGHT_MM,
    progress: ProgressCallback | None = None,
) -> BasePlateResult:
    """Add a rectangular base plate below the extruded border footprint."""
    if height_mm <= 0:
        raise ValueError("Base height must be greater than 0 mm.")
    if margin_mm < 0:
        raise ValueError("Base margin must be zero or positive.")
    if not extruded_border_path.is_file():
        raise FileNotFoundError(f"Extruded border not found: {extruded_border_path}")

    if progress is not None:
        progress("Reading extruded border", 0, 3)

    border = trimesh.load(extruded_border_path, force="mesh")
    if border.is_empty or len(border.faces) == 0:
        raise ValueError("Extruded border mesh is empty.")

    xmin, ymin, zmin = border.bounds[0]
    xmax, ymax, _zmax = border.bounds[1]
    width = (xmax - xmin) + 2.0 * margin_mm
    depth = (ymax - ymin) + 2.0 * margin_mm
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    if progress is not None:
        progress("Building base plate", 1, 3)

    base = trimesh.creation.box(extents=[width, depth, height_mm])
    base.apply_translation([cx, cy, zmin - height_mm * 0.5])

    if progress is not None:
        progress("Saving base plate", 2, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.export(output_path)

    if progress is not None:
        progress("Base plate export complete", 3, 3)

    return BasePlateResult(
        output_dir=output_path.parent,
        base_path=output_path,
        triangle_count=int(len(base.faces)),
    )


def export_assembly(
    output_path: Path,
    extruded_border_path: Path,
    text_path: Path | None = None,
    base_path: Path | None = None,
    progress: ProgressCallback | None = None,
) -> AssemblyExportResult:
    """Combine base plate, extruded border, and text into one STL."""
    if not extruded_border_path.is_file():
        raise FileNotFoundError(f"Extruded border not found: {extruded_border_path}")

    parts: list[trimesh.Trimesh] = []
    part_paths = [
        ("base plate", base_path),
        ("extruded border", extruded_border_path),
        ("surface text", text_path),
    ]

    if progress is not None:
        progress("Reading assembly parts", 0, 3)

    for label, path in part_paths:
        if path is None or not path.is_file():
            continue
        mesh = trimesh.load(path, force="mesh")
        if mesh.is_empty or len(mesh.faces) == 0:
            raise ValueError(f"{label} mesh is empty: {path}")
        parts.append(mesh)

    if not parts:
        raise ValueError("No assembly parts available to export.")

    if progress is not None:
        progress("Merging assembly", 1, 3)

    combined = trimesh.util.concatenate(parts)
    combined.merge_vertices()
    combined.update_faces(combined.unique_faces() & combined.nondegenerate_faces())
    combined.remove_unreferenced_vertices()

    if progress is not None:
        progress("Saving assembly", 2, 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(output_path)

    if progress is not None:
        progress("Assembly export complete", 3, 3)

    return AssemblyExportResult(
        output_dir=output_path.parent,
        assembly_path=output_path,
        triangle_count=int(len(combined.faces)),
        part_count=len(parts),
    )
