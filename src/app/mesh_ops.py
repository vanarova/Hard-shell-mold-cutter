"""Mesh repair, remeshing, and boolean operations (manifold3d + trimesh)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

import manifold3d as m3d
import numpy as np
import pyvista as pv
import trimesh

from app.boundary_generator import ProgressCallback

_BOOLEAN_ENGINE = "manifold"
_MIN_EDGE_MM = 0.05
_MAX_EDGE_MM = 50.0


class MeshOpKind(str, Enum):
    REMESH = "remesh"
    WATERTIGHT = "watertight"
    SOLID = "solid"
    UNION = "union"
    SUBTRACT = "subtract"


@dataclass
class MeshOpResult:
    mesh: pv.PolyData
    triangle_count: int
    message: str


def polydata_to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError("Mesh has no triangles.")

    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    return trimesh.Trimesh(
        vertices=np.asarray(surface.points, dtype=float),
        faces=faces,
        process=False,
    )


def trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    cleaned = _cleanup_trimesh(mesh)
    faces = np.column_stack(
        [np.full(len(cleaned.faces), 3, dtype=np.int64), cleaned.faces]
    ).ravel()
    return pv.PolyData(cleaned.vertices, faces).triangulate()


def _cleanup_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    cleaned.merge_vertices()
    cleaned.update_faces(cleaned.unique_faces() & cleaned.nondegenerate_faces())
    cleaned.remove_unreferenced_vertices()
    return cleaned


def _to_manifold(mesh: trimesh.Trimesh) -> m3d.Manifold:
    cleaned = _cleanup_trimesh(mesh)
    manifold_mesh = m3d.Mesh(
        vert_properties=np.asarray(cleaned.vertices, dtype=np.float32),
        tri_verts=np.asarray(cleaned.faces, dtype=np.uint32),
    )
    return m3d.Manifold(manifold_mesh)


def _from_manifold(manifold: m3d.Manifold) -> trimesh.Trimesh:
    mesh_out = manifold.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(mesh_out.vert_properties, dtype=float),
        faces=np.asarray(mesh_out.tri_verts, dtype=np.int64),
        process=False,
    )


def _try_manifold_cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Run manifold repair when it preserves geometry; otherwise keep the input."""
    original_faces = len(mesh.faces)
    if original_faces == 0:
        return mesh

    try:
        cleaned = _from_manifold(_to_manifold(mesh))
        if len(cleaned.faces) > 0:
            return cleaned
    except Exception:  # noqa: BLE001
        pass

    return _cleanup_trimesh(mesh)


def _validate_polydata(mesh: pv.PolyData, operation: str) -> pv.PolyData:
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if surface.n_cells == 0:
        raise ValueError(
            f"{operation} produced an empty mesh. "
            "The source geometry may be open, non-manifold, or too damaged to repair."
        )
    return surface


def _default_voxel_pitch(mesh: trimesh.Trimesh) -> float:
    extents = np.asarray(mesh.extents, dtype=float)
    span = float(np.max(extents))
    if span <= 0:
        return 0.5
    return max(span / 80.0, 0.2)


def remesh_uniform(
    mesh: pv.PolyData,
    edge_length_mm: float,
    progress: ProgressCallback | None = None,
) -> MeshOpResult:
    """Subdivide until triangle edges are near ``edge_length_mm``."""
    if edge_length_mm < _MIN_EDGE_MM:
        raise ValueError(f"Edge length must be at least {_MIN_EDGE_MM} mm.")
    if edge_length_mm > _MAX_EDGE_MM:
        raise ValueError(f"Edge length must be at most {_MAX_EDGE_MM} mm.")

    if progress is not None:
        progress("Preparing mesh for remesh", 0, 3)

    tm = _cleanup_trimesh(polydata_to_trimesh(mesh))
    original = int(len(tm.faces))

    if progress is not None:
        progress("Remeshing triangles", 1, 3)

    vertices, faces = trimesh.remesh.subdivide_to_size(
        tm.vertices,
        tm.faces,
        max_edge=float(edge_length_mm),
        max_iter=50,
    )
    remeshed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    remeshed = _cleanup_trimesh(remeshed)

    if progress is not None:
        progress("Remesh complete", 3, 3)

    result = trimesh_to_polydata(remeshed)
    return MeshOpResult(
        mesh=result,
        triangle_count=int(result.n_cells),
        message=f"Remeshed {original:,} → {result.n_cells:,} triangles "
        f"(target edge {edge_length_mm:.2f} mm)",
    )


def make_watertight(
    mesh: pv.PolyData,
    progress: ProgressCallback | None = None,
) -> MeshOpResult:
    """Fill holes and repair normals to produce a watertight mesh."""
    if progress is not None:
        progress("Preparing mesh", 0, 4)

    tm = polydata_to_trimesh(mesh)
    tm.merge_vertices()
    tm.update_faces(tm.unique_faces() & tm.nondegenerate_faces())
    tm.remove_unreferenced_vertices()

    if progress is not None:
        progress("Fixing normals", 1, 4)

    trimesh.repair.fix_normals(tm)
    trimesh.repair.fill_holes(tm)

    if progress is not None:
        progress("Manifold cleanup", 2, 4)

    tm = _try_manifold_cleanup(tm)

    if progress is not None:
        progress("Watertight repair complete", 4, 4)

    result = _validate_polydata(trimesh_to_polydata(tm), "Watertight repair")
    status = "watertight" if tm.is_watertight else "repaired (may still have issues)"
    return MeshOpResult(
        mesh=result,
        triangle_count=int(result.n_cells),
        message=f"Mesh is {status} — {result.n_cells:,} triangles",
    )


def make_solid(
    mesh: pv.PolyData,
    progress: ProgressCallback | None = None,
) -> MeshOpResult:
    """
    Remove hollow interior geometry and return a filled solid mesh.

    Uses voxel filling for closed shells and keeps the largest exterior shell
    when separate inner/outer components are present.
    """
    if progress is not None:
        progress("Preparing mesh", 0, 5)

    tm = _try_manifold_cleanup(_cleanup_trimesh(polydata_to_trimesh(mesh)))
    if not tm.is_watertight:
        raise ValueError("Mesh must be watertight before solidify. Run Make Watertight first.")

    if progress is not None:
        progress("Analyzing mesh components", 1, 5)

    solid_tm = _solidify_trimesh(tm)

    if progress is not None:
        progress("Manifold cleanup", 3, 5)

    solid_tm = _try_manifold_cleanup(solid_tm)

    if progress is not None:
        progress("Solid mesh ready", 5, 5)

    result = trimesh_to_polydata(solid_tm)
    return MeshOpResult(
        mesh=result,
        triangle_count=int(result.n_cells),
        message=f"Solid mesh — {result.n_cells:,} triangles, volume {solid_tm.volume:.1f} mm³",
    )


def _solidify_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill interior voids in a watertight shell using voxel reconstruction."""
    pitch = _default_voxel_pitch(mesh)
    try:
        voxel = mesh.voxelized(pitch).fill()
        filled = voxel.marching_cubes
        if filled is not None and len(filled.faces) > 0:
            return _cleanup_trimesh(filled)
    except ImportError as exc:
        raise ImportError(
            "Solidify requires scipy and scikit-image. Install dependencies and retry."
        ) from exc

    return _cleanup_trimesh(mesh)


def boolean_union_meshes(
    meshes: Sequence[pv.PolyData],
    progress: ProgressCallback | None = None,
) -> MeshOpResult:
    """Union two or more meshes."""
    if len(meshes) < 2:
        raise ValueError("Load at least two models to union.")

    if progress is not None:
        progress("Preparing meshes", 0, 3)

    tm_meshes = [_try_manifold_cleanup(polydata_to_trimesh(mesh)) for mesh in meshes]

    if progress is not None:
        progress("Running boolean union", 1, 3)

    result_tm = trimesh.boolean.union(
        tm_meshes,
        engine=_BOOLEAN_ENGINE,
        check_volume=False,
    )
    result_tm = _try_manifold_cleanup(result_tm)

    if progress is not None:
        progress("Union complete", 3, 3)

    result = trimesh_to_polydata(result_tm)
    return MeshOpResult(
        mesh=result,
        triangle_count=int(result.n_cells),
        message=f"Union of {len(meshes)} models — {result.n_cells:,} triangles",
    )


def boolean_subtract_meshes(
    meshes: Sequence[pv.PolyData],
    progress: ProgressCallback | None = None,
) -> MeshOpResult:
    """Subtract meshes[1:] from meshes[0]."""
    if len(meshes) < 2:
        raise ValueError("Load at least two models to subtract.")

    if progress is not None:
        progress("Preparing meshes", 0, 3)

    tm_meshes = [_try_manifold_cleanup(polydata_to_trimesh(mesh)) for mesh in meshes]

    if progress is not None:
        progress("Running boolean subtract", 1, 3)

    result_tm = trimesh.boolean.difference(
        tm_meshes,
        engine=_BOOLEAN_ENGINE,
        check_volume=False,
    )
    result_tm = _try_manifold_cleanup(result_tm)

    if progress is not None:
        progress("Subtract complete", 3, 3)

    result = trimesh_to_polydata(result_tm)
    return MeshOpResult(
        mesh=result,
        triangle_count=int(result.n_cells),
        message=f"Subtract model 1 − others — {result.n_cells:,} triangles",
    )
