"""3D STL viewer widget built on PyVista."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QVBoxLayout, QWidget

ProgressCallback = Callable[[str, int, int], None]

MESH_COLOR = "#89b4fa"
SELECTED_COLOR = "#f9e2af"
EDGE_COLOR = "#f9e2af"
WIREFRAME_COLOR = "#cba6f7"
POINT_COLOR = "#a6e3a1"
OVERLAY_COLOR = "#a6e3a1"
WALL_COLOR = "#fab387"


def load_stl_polydata(path: Path, progress: ProgressCallback | None = None) -> pv.PolyData:
    """Load and triangulate an STL file with optional progress reporting."""
    if progress is not None:
        progress("Reading STL file", 0, 3)
    mesh = pv.read(str(path))
    if progress is not None:
        progress("Processing surface mesh", 1, 3)
    surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
    if progress is not None:
        progress("Model loaded", 3, 3)
    return surface


@dataclass(frozen=True)
class OverlayMesh:
    mesh: pv.PolyData
    color: str = OVERLAY_COLOR
    opacity: float = 0.55
    style: str = "surface"
    show_edges: bool = True
    edge_color: str = "#40a870"


class StlViewer(QWidget):
    """Embedded PyVista viewport for displaying STL meshes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mesh: pv.PolyData | None = None
        self._scene_meshes: list[pv.PolyData] = []
        self._mesh_actors: list = []
        self._selected_index = 0
        self._selection_enabled = False
        self._selection_callback: Callable[[int], None] | None = None
        self._overlays: list[OverlayMesh] = []
        self._actor = None
        self._mesh_actor = None
        self._view_mode = "solid_edges"
        self._point_pick_callback: Callable[[tuple[float, float, float]], None] | None = None
        self._display_click_callback: Callable[[float, float], None] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)

        self.plotter.set_background("#1e1e2e")
        self._setup_scene()

    def _setup_scene(self) -> None:
        self.plotter.add_axes()
        self.plotter.show_grid()

    def load_stl(self, path: Path) -> pv.PolyData:
        """Load and display an STL file, replacing the current scene."""
        mesh = pv.read(str(path))
        self.set_mesh(mesh)
        return self._mesh  # type: ignore[return-value]

    def append_stl(self, path: Path) -> pv.PolyData:
        """Add another STL to the scene without removing existing meshes."""
        mesh = pv.read(str(path))
        return self.append_mesh(mesh)

    def append_mesh(self, mesh: pv.PolyData) -> pv.PolyData:
        """Add a mesh to the scene without removing existing meshes."""
        surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
        if self._mesh is None:
            self.set_mesh(surface)
            return surface

        self._scene_meshes.append(surface)
        self._redraw_mesh(preserve_camera=True)
        return surface

    def set_view_mode(self, mode: str) -> None:
        """Change how the loaded mesh is rendered."""
        if mode == self._view_mode and self._actor is not None:
            return

        self._view_mode = mode
        if self._mesh is None:
            return

        self._redraw_mesh(preserve_camera=True)

    def set_scene_meshes(self, meshes: list[pv.PolyData], *, reset_camera: bool = False) -> None:
        """Replace all meshes shown in the scene."""
        surfaces = [
            mesh.extract_surface(algorithm="dataset_surface").triangulate()
            for mesh in meshes
        ]
        if not surfaces:
            self.clear()
            return

        self._scene_meshes = surfaces
        self._mesh = surfaces[0]
        self._redraw_mesh(reset_camera=reset_camera)

    def replace_scene_mesh(self, index: int, mesh: pv.PolyData) -> None:
        """Replace one mesh in the scene by index."""
        if index < 0 or index >= len(self._scene_meshes):
            raise IndexError("Scene mesh index out of range.")

        surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
        self._scene_meshes[index] = surface
        if index == 0:
            self._mesh = surface
        self._redraw_mesh(preserve_camera=True)

    def get_scene_mesh(self, index: int) -> pv.PolyData:
        if index < 0 or index >= len(self._scene_meshes):
            raise IndexError("Scene mesh index out of range.")
        return self._scene_meshes[index]

    def scene_meshes(self) -> list[pv.PolyData]:
        return list(self._scene_meshes)

    def set_selection_enabled(self, enabled: bool) -> None:
        """Enable click-to-select on any loaded model."""
        self._selection_enabled = enabled
        self._restore_point_picking()

    def set_selection_callback(self, callback: Callable[[int], None] | None) -> None:
        self._selection_callback = callback

    def set_selected_index(self, index: int) -> None:
        if not self._scene_meshes:
            return
        clamped = min(max(index, 0), len(self._scene_meshes) - 1)
        if clamped == self._selected_index:
            return
        self._selected_index = clamped
        self._redraw_mesh(preserve_camera=True)

    @property
    def selected_index(self) -> int:
        return self._selected_index

    def move_selected(self, dx: float, dy: float, dz: float) -> None:
        """Translate the selected model in millimeters."""
        if not self._scene_meshes:
            return

        index = self._selected_index
        mesh = self._scene_meshes[index].copy(deep=True)
        mesh.translate((dx, dy, dz), inplace=True)
        self._scene_meshes[index] = mesh
        if index == 0:
            self._mesh = mesh
        self._redraw_mesh(preserve_camera=True)

    def set_mesh(self, mesh: pv.PolyData) -> None:
        """Replace the currently displayed mesh."""
        surface = mesh.extract_surface(algorithm="dataset_surface").triangulate()
        self._mesh = surface
        self._scene_meshes = [surface]
        self._selected_index = 0
        self._overlays = []
        self._redraw_mesh(reset_camera=True)

    def set_overlays(self, overlays: list[OverlayMesh]) -> None:
        """Replace overlay meshes drawn on top of the primary model."""
        self._overlays = overlays
        self._redraw_mesh(preserve_camera=True)

    def clear_overlays(self) -> None:
        self._overlays = []
        self._redraw_mesh(preserve_camera=True)

    def set_point_pick_callback(
        self,
        callback: Callable[[tuple[float, float, float]], None] | None,
    ) -> None:
        """Enable or disable left-click surface picking on the loaded mesh."""
        self._point_pick_callback = callback
        self._restore_point_picking()

    def set_display_click_callback(
        self,
        callback: Callable[[float, float], None] | None,
    ) -> None:
        """Enable or disable left-click callbacks with display (pixel) coordinates."""
        self._display_click_callback = callback
        self._restore_display_click_tracking()

    def world_to_display(self, point: tuple[float, float, float] | np.ndarray) -> tuple[float, float]:
        """Project a world-space point to display coordinates."""
        from vtkmodules.vtkRenderingCore import vtkCoordinate

        coords = np.asarray(point, dtype=float).reshape(3)
        coordinate = vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        coordinate.SetValue(float(coords[0]), float(coords[1]), float(coords[2]))
        display = coordinate.GetComputedDisplayValue(self.plotter.renderer)
        return float(display[0]), float(display[1])

    def pick_surface_frame(
        self,
        point: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the snapped surface point and cell normal nearest ``point``."""
        if self._mesh is None:
            raise ValueError("No mesh loaded.")

        mesh = self._mesh.extract_surface(algorithm="dataset_surface").triangulate()
        if mesh.n_cells == 0:
            raise ValueError("Mesh has no surface cells.")

        if mesh.cell_normals is None or len(mesh.cell_normals) == 0:
            mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                inplace=False,
            )

        query = np.asarray(point, dtype=float)
        closest_idx = int(mesh.find_closest_point(query))
        surface_point = tuple(float(value) for value in mesh.points[closest_idx])

        cell_id = int(mesh.find_closest_cell(surface_point))
        if cell_id < 0:
            cell_id = int(mesh.find_closest_cell(query))
        normal = mesh.cell_normals[cell_id]
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            raise ValueError("Could not determine surface normal.")
        normal_tuple = tuple(float(value / length) for value in normal)
        return surface_point, normal_tuple

    @staticmethod
    def _pick_coords(point) -> tuple[float, float, float] | None:
        if point is None:
            return None
        coords = np.asarray(point, dtype=float).reshape(-1)
        if coords.size < 3:
            return None
        return (float(coords[0]), float(coords[1]), float(coords[2]))

    def _restore_point_picking(self) -> None:
        if self._point_pick_callback is not None:
            self._enable_custom_point_picking()
            return

        if self._selection_enabled and self._scene_meshes:
            self._enable_selection_picking()
            return

        try:
            self.plotter.disable_picking()
        except Exception:  # noqa: BLE001
            pass

    def _enable_custom_point_picking(self) -> None:
        handler = self._point_pick_callback
        target_actor = self._mesh_actor

        def _on_pick(point) -> None:
            picked_actor = getattr(self.plotter, "picked_actor", None)
            if picked_actor is None:
                return
            if target_actor is not None and picked_actor is not target_actor:
                return

            coords = self._pick_coords(point)
            if coords is None:
                return
            handler(coords)

        try:
            self.plotter.enable_surface_point_picking(
                callback=_on_pick,
                show_message=False,
                show_point=False,
                left_clicking=True,
                picker="cell",
                clear_on_no_selection=True,
            )
        except Exception:  # noqa: BLE001
            pass

    def _enable_selection_picking(self) -> None:
        viewer = self

        def _on_pick(_point) -> None:
            picked_actor = getattr(viewer.plotter, "picked_actor", None)
            if picked_actor is None or picked_actor not in viewer._mesh_actors:
                return
            index = viewer._mesh_actors.index(picked_actor)
            if index != viewer._selected_index:
                viewer._selected_index = index
                viewer._redraw_mesh(preserve_camera=True)
                if viewer._selection_callback is not None:
                    viewer._selection_callback(index)

        try:
            self.plotter.enable_surface_point_picking(
                callback=_on_pick,
                show_message=False,
                show_point=False,
                left_clicking=True,
                picker="cell",
                clear_on_no_selection=True,
            )
        except Exception:  # noqa: BLE001
            pass

    def _redraw_mesh(
        self,
        *,
        preserve_camera: bool = False,
        reset_camera: bool = False,
    ) -> None:
        if not self._scene_meshes:
            return

        camera = self.plotter.camera_position if preserve_camera else None

        self.plotter.clear()
        self._setup_scene()
        self._mesh_actor = None
        self._mesh_actors = []
        for index, mesh in enumerate(self._scene_meshes):
            pickable = self._selection_enabled or index == 0
            selected = index == self._selected_index
            actor = self._render_mesh(mesh, pickable=pickable, selected=selected)
            self._mesh_actors.append(actor)
            if index == 0:
                self._actor = actor
                self._mesh_actor = actor
        for overlay in self._overlays:
            self.plotter.add_mesh(
                overlay.mesh,
                color=overlay.color,
                style=overlay.style,
                show_edges=overlay.show_edges,
                edge_color=overlay.edge_color,
                line_width=1,
                opacity=overlay.opacity,
                smooth_shading=True,
                lighting=True,
                pickable=False,
            )

        if reset_camera:
            self.plotter.reset_camera()
            self.plotter.camera_position = "iso"
        elif camera is not None:
            self.plotter.camera_position = camera

        self._restore_point_picking()
        self._restore_display_click_tracking()
        self.plotter.render()

    def _restore_display_click_tracking(self) -> None:
        try:
            self.plotter.untrack_click_position(side="left")
        except Exception:  # noqa: BLE001
            pass
        if self._display_click_callback is None:
            return

        handler = self._display_click_callback

        def _on_click(pos) -> None:
            if handler is None or pos is None:
                return
            coords = np.asarray(pos, dtype=float).reshape(-1)
            if coords.size < 2:
                return
            handler(float(coords[0]), float(coords[1]))

        try:
            self.plotter.track_click_position(
                callback=_on_click,
                side="left",
                viewport=True,
            )
        except Exception:  # noqa: BLE001
            pass

    def _render_mesh(self, mesh: pv.PolyData, *, pickable: bool = True, selected: bool = False):
        mode = self._view_mode
        color = SELECTED_COLOR if selected else MESH_COLOR
        edge_color = "#fab387" if selected else EDGE_COLOR

        if mode == "solid":
            return self.plotter.add_mesh(
                mesh,
                color=color,
                style="surface",
                show_edges=False,
                opacity=1.0,
                smooth_shading=True,
                lighting=True,
                pickable=pickable,
            )

        if mode == "solid_edges":
            return self.plotter.add_mesh(
                mesh,
                color=color,
                style="surface",
                show_edges=True,
                edge_color=edge_color,
                line_width=2,
                opacity=1.0,
                smooth_shading=True,
                lighting=True,
                pickable=pickable,
            )

        if mode == "wireframe":
            edges = mesh.extract_all_edges()
            if edges.n_cells > 0:
                return self.plotter.add_mesh(
                    edges,
                    color=WIREFRAME_COLOR,
                    line_width=2,
                    render_lines_as_tubes=True,
                    lighting=False,
                    pickable=pickable,
                )
            return self.plotter.add_mesh(
                mesh,
                color=WIREFRAME_COLOR,
                style="wireframe",
                show_edges=True,
                line_width=2,
                opacity=1.0,
                lighting=False,
                pickable=pickable,
            )

        if mode == "surface":
            return self.plotter.add_mesh(
                mesh,
                color=MESH_COLOR,
                style="surface",
                show_edges=False,
                opacity=0.35,
                smooth_shading=True,
                lighting=True,
                pickable=pickable,
            )

        if mode == "points":
            return self.plotter.add_mesh(
                mesh,
                color=POINT_COLOR,
                style="points",
                render_points_as_spheres=True,
                point_size=8,
                opacity=1.0,
                lighting=False,
                pickable=pickable,
            )

        return self.plotter.add_mesh(
            mesh,
            color=MESH_COLOR,
            style="surface",
            show_edges=True,
            edge_color=EDGE_COLOR,
            lighting=True,
            pickable=pickable,
        )

    def clear(self) -> None:
        """Remove all meshes from the viewport."""
        self._mesh = None
        self._scene_meshes = []
        self._mesh_actors = []
        self._selected_index = 0
        self._overlays = []
        self._actor = None
        self._mesh_actor = None
        self._point_pick_callback = None
        self._display_click_callback = None
        try:
            self.plotter.untrack_click_position(side="left")
        except Exception:  # noqa: BLE001
            pass
        self.plotter.clear()
        self._setup_scene()

    @property
    def mesh(self) -> pv.PolyData | None:
        return self._mesh

    @property
    def scene_mesh_count(self) -> int:
        return len(self._scene_meshes)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.plotter.close()
        super().closeEvent(event)
