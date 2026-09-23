"""Main window for the 25D-MoldMaker tool (STL load + single-plane Z slicing)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.border_slice import (
    assembly_output_path,
    base_plate_output_path,
    border_extruded_output_path,
    border_output_path,
    create_base_plate,
    create_border_from_slice,
    export_assembly,
    extrude_border_on_z,
    slice_output_path,
)
from app.curved_surface_text import (
    build_text_on_surface,
    placement_on_outer_surface,
    text_output_path,
    write_text_on_surface,
)
from app.footer_bar import FooterBar
from app.main_window import create_app
from app.mesh_slicer import slice_single_at_z
from app.slice_axis import output_dir_for_stl
from app.stl_viewer import OVERLAY_COLOR, OverlayMesh, StlViewer, load_stl_polydata
from app.xy_plane import make_xy_plane_mesh, mesh_z_bounds

T = TypeVar("T")

_SIDEBAR_WIDTH = 340
_PLANE_PREVIEW_COLOR = "#f9e2af"
_PLANE_CONFIRMED_COLOR = "#89b4fa"
_TEXT_PREVIEW_COLOR = "#cba6f7"
_SLIDER_STEPS = 1000


class MoldMakerMainWindow(QMainWindow):
    """STL viewer with a single Z reference plane and one-slice export."""

    def __init__(self) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._preview_z: float | None = None
        self._plane_z: float | None = None
        self._text_preview_mesh = None
        self._slider_sync = False

        self.setWindowTitle("25D-MoldMaker")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        sidebar = self._build_sidebar()
        root_layout.addWidget(sidebar)

        viewer_column = QWidget()
        viewer_layout = QVBoxLayout(viewer_column)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)

        self.viewer = StlViewer()
        viewer_layout.addWidget(self.viewer, stretch=1)

        self.footer = FooterBar()
        self.footer.view_mode_changed.connect(self.viewer.set_view_mode)
        viewer_layout.addWidget(self.footer)

        root_layout.addWidget(viewer_column, stretch=1)
        self.footer.show_message("Open an STL file to begin.")

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setFixedWidth(_SIDEBAR_WIDTH)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 10, 12)
        layout.setSpacing(12)

        title = QLabel("25D-MoldMaker")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        subtitle = QLabel("STL Viewer & Single-Layer Slicing")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(subtitle)

        layout.addWidget(self._divider())

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        self.btn_open = QPushButton("Open STL…")
        self.btn_open.clicked.connect(self._open_stl)
        file_layout.addWidget(self.btn_open)

        self.lbl_size = QLabel("Size: —")
        self.lbl_size.setWordWrap(True)
        self.lbl_size.setObjectName("fileLabel")
        file_layout.addWidget(self.lbl_size)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setObjectName("fileLabel")
        file_layout.addWidget(self.lbl_file)

        layout.addWidget(file_group)

        plane_group = QGroupBox("Slice Plane (Z axis)")
        plane_layout = QVBoxLayout(plane_group)

        plane_info = QLabel(
            "Mark one XY plane along Z. Slide to position, then confirm."
        )
        plane_info.setWordWrap(True)
        plane_info.setObjectName("fileLabel")
        plane_layout.addWidget(plane_info)

        self.lbl_plane_status = QLabel("Plane: not set")
        self.lbl_plane_status.setObjectName("fileLabel")
        plane_layout.addWidget(self.lbl_plane_status)

        self.lbl_plane_z = QLabel("Z: —")
        self.lbl_plane_z.setObjectName("fileLabel")
        plane_layout.addWidget(self.lbl_plane_z)

        self.slider_plane_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_plane_z.setRange(0, _SLIDER_STEPS)
        self.slider_plane_z.setEnabled(False)
        self.slider_plane_z.valueChanged.connect(self._on_plane_slider_changed)
        plane_layout.addWidget(self.slider_plane_z)

        z_row = QHBoxLayout()
        z_label = QLabel("Exact Z (mm)")
        z_label.setWordWrap(True)
        z_row.addWidget(z_label, stretch=1)
        self.spin_plane_z = QDoubleSpinBox()
        self.spin_plane_z.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_plane_z.setDecimals(3)
        self.spin_plane_z.setSingleStep(0.1)
        self.spin_plane_z.setSuffix(" mm")
        self.spin_plane_z.setMinimumWidth(88)
        self.spin_plane_z.setEnabled(False)
        self.spin_plane_z.valueChanged.connect(self._on_plane_spin_changed)
        z_row.addWidget(self.spin_plane_z)
        plane_layout.addLayout(z_row)

        self.btn_confirm_plane = QPushButton("Confirm Plane")
        self.btn_confirm_plane.setEnabled(False)
        self.btn_confirm_plane.clicked.connect(self._confirm_plane)
        plane_layout.addWidget(self.btn_confirm_plane)

        layout.addWidget(plane_group)

        slice_group = QGroupBox("Slice")
        slice_layout = QVBoxLayout(slice_group)

        height_row = QHBoxLayout()
        height_label = QLabel("Slice thickness (mm)")
        height_label.setWordWrap(True)
        height_row.addWidget(height_label, stretch=1)
        self.spin_slice_height = QDoubleSpinBox()
        self.spin_slice_height.setRange(0.1, 10_000.0)
        self.spin_slice_height.setDecimals(2)
        self.spin_slice_height.setSingleStep(1.0)
        self.spin_slice_height.setValue(10.0)
        self.spin_slice_height.setSuffix(" mm")
        self.spin_slice_height.setMinimumWidth(88)
        height_row.addWidget(self.spin_slice_height)
        slice_layout.addLayout(height_row)

        slice_info = QLabel(
            "Export one slice starting at the confirmed plane. "
            "Saved next to the model in a folder with the same name."
        )
        slice_info.setWordWrap(True)
        slice_info.setObjectName("fileLabel")
        slice_layout.addWidget(slice_info)

        self.btn_slice = QPushButton("Slice")
        self.btn_slice.setEnabled(False)
        self.btn_slice.clicked.connect(self._slice_at_plane)
        slice_layout.addWidget(self.btn_slice)

        layout.addWidget(slice_group)

        border_group = QGroupBox("Border")
        border_layout = QVBoxLayout(border_group)

        offset_row = QHBoxLayout()
        offset_label = QLabel("Border offset (mm)")
        offset_label.setWordWrap(True)
        offset_row.addWidget(offset_label, stretch=1)
        self.spin_border_offset = QDoubleSpinBox()
        self.spin_border_offset.setRange(0.1, 10_000.0)
        self.spin_border_offset.setDecimals(2)
        self.spin_border_offset.setSingleStep(0.5)
        self.spin_border_offset.setValue(5.0)
        self.spin_border_offset.setSuffix(" mm")
        self.spin_border_offset.setMinimumWidth(88)
        offset_row.addWidget(self.spin_border_offset)
        border_layout.addLayout(offset_row)

        thickness_row = QHBoxLayout()
        thickness_label = QLabel("Border thickness (mm)")
        thickness_label.setWordWrap(True)
        thickness_row.addWidget(thickness_label, stretch=1)
        self.spin_border_thickness = QDoubleSpinBox()
        self.spin_border_thickness.setRange(0.1, 10_000.0)
        self.spin_border_thickness.setDecimals(2)
        self.spin_border_thickness.setSingleStep(0.5)
        self.spin_border_thickness.setValue(2.0)
        self.spin_border_thickness.setSuffix(" mm")
        self.spin_border_thickness.setMinimumWidth(88)
        thickness_row.addWidget(self.spin_border_thickness)
        border_layout.addLayout(thickness_row)

        border_info = QLabel(
            "Build one border ring around all objects in the slice. Overlapping "
            "border sections between objects are merged into a single outline. "
            "Enable Create connected Border to fill the enclosed area using "
            "offset only (thickness is ignored)."
        )
        border_info.setWordWrap(True)
        border_info.setObjectName("fileLabel")
        border_layout.addWidget(border_info)

        self.chk_connected_border = QCheckBox("Create connected Border")
        self.chk_connected_border.setEnabled(False)
        self.chk_connected_border.toggled.connect(self._on_connected_border_toggled)
        border_layout.addWidget(self.chk_connected_border)

        self.btn_border_slice = QPushButton("Border Slice")
        self.btn_border_slice.setEnabled(False)
        self.btn_border_slice.clicked.connect(self._create_border_slice)
        border_layout.addWidget(self.btn_border_slice)

        extrude_row = QHBoxLayout()
        extrude_label = QLabel("Increase height by (mm)")
        extrude_label.setWordWrap(True)
        extrude_row.addWidget(extrude_label, stretch=1)
        self.spin_border_extrude = QDoubleSpinBox()
        self.spin_border_extrude.setRange(0.1, 10_000.0)
        self.spin_border_extrude.setDecimals(2)
        self.spin_border_extrude.setSingleStep(1.0)
        self.spin_border_extrude.setValue(10.0)
        self.spin_border_extrude.setSuffix(" mm")
        self.spin_border_extrude.setMinimumWidth(88)
        extrude_row.addWidget(self.spin_border_extrude)
        border_layout.addLayout(extrude_row)

        self.btn_extrude_border = QPushButton("Extrude Border")
        self.btn_extrude_border.setEnabled(False)
        self.btn_extrude_border.clicked.connect(self._extrude_border)
        border_layout.addWidget(self.btn_extrude_border)

        layout.addWidget(border_group)

        text_group = QGroupBox("Surface Text")
        text_layout = QVBoxLayout(text_group)

        self.edit_surface_text = QLineEdit()
        self.edit_surface_text.setPlaceholderText("Text to engrave")
        self.edit_surface_text.textChanged.connect(self._on_text_content_changed)
        text_layout.addWidget(self.edit_surface_text)

        text_size_row = QHBoxLayout()
        text_size_label = QLabel("Text size (mm)")
        text_size_label.setWordWrap(True)
        text_size_row.addWidget(text_size_label, stretch=1)
        self.spin_text_size = QDoubleSpinBox()
        self.spin_text_size.setRange(0.5, 500.0)
        self.spin_text_size.setDecimals(2)
        self.spin_text_size.setSingleStep(0.5)
        self.spin_text_size.setValue(5.0)
        self.spin_text_size.setSuffix(" mm")
        self.spin_text_size.setMinimumWidth(88)
        self.spin_text_size.valueChanged.connect(self._update_text_preview)
        text_size_row.addWidget(self.spin_text_size)
        text_layout.addLayout(text_size_row)

        text_info = QLabel(
            "Text is placed on the outer surface of the loaded model. "
            "Use the sliders to move it horizontally and vertically."
        )
        text_info.setWordWrap(True)
        text_info.setObjectName("fileLabel")
        text_layout.addWidget(text_info)

        text_h_row = QHBoxLayout()
        text_h_label = QLabel("Horizontal position")
        text_h_label.setWordWrap(True)
        text_h_row.addWidget(text_h_label, stretch=1)
        self.slider_text_h = QSlider(Qt.Orientation.Horizontal)
        self.slider_text_h.setRange(0, _SLIDER_STEPS)
        self.slider_text_h.setValue(_SLIDER_STEPS // 2)
        self.slider_text_h.valueChanged.connect(self._on_text_position_changed)
        text_h_row.addWidget(self.slider_text_h, stretch=2)
        text_layout.addLayout(text_h_row)

        text_v_row = QHBoxLayout()
        text_v_label = QLabel("Vertical position")
        text_v_label.setWordWrap(True)
        text_v_row.addWidget(text_v_label, stretch=1)
        self.slider_text_v = QSlider(Qt.Orientation.Horizontal)
        self.slider_text_v.setRange(0, _SLIDER_STEPS)
        self.slider_text_v.setValue(_SLIDER_STEPS // 2)
        self.slider_text_v.valueChanged.connect(self._on_text_position_changed)
        text_v_row.addWidget(self.slider_text_v, stretch=2)
        text_layout.addLayout(text_v_row)

        self.lbl_text_position = QLabel("Text position: center")
        self.lbl_text_position.setObjectName("fileLabel")
        text_layout.addWidget(self.lbl_text_position)

        self.btn_write_surface_text = QPushButton("Write Text")
        self.btn_write_surface_text.setEnabled(False)
        self.btn_write_surface_text.clicked.connect(self._write_surface_text)
        text_layout.addWidget(self.btn_write_surface_text)

        self.btn_add_base = QPushButton("Add Base")
        self.btn_add_base.setEnabled(False)
        self.btn_add_base.clicked.connect(self._add_base_plate)
        text_layout.addWidget(self.btn_add_base)

        self.btn_download_assembly = QPushButton("Download Assembly")
        self.btn_download_assembly.setEnabled(False)
        self.btn_download_assembly.clicked.connect(self._download_assembly)
        text_layout.addWidget(self.btn_download_assembly)

        layout.addWidget(text_group)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _progress_callback(self) -> Callable[[str, int, int], None]:
        def report(message: str, step: int = 0, total: int = 0) -> None:
            self.footer.set_stage(message, step, total)
            QApplication.processEvents()

        return report

    def _output_dir(self) -> Path | None:
        if self._current_path is None:
            return None
        return output_dir_for_stl(self._current_path)

    def _has_slice_file(self) -> bool:
        output_dir = self._output_dir()
        return output_dir is not None and slice_output_path(output_dir).is_file()

    def _has_border_file(self) -> bool:
        output_dir = self._output_dir()
        return output_dir is not None and border_output_path(output_dir).is_file()

    def _has_extruded_border_file(self) -> bool:
        output_dir = self._output_dir()
        return output_dir is not None and border_extruded_output_path(output_dir).is_file()

    def _has_text_file(self) -> bool:
        output_dir = self._output_dir()
        return output_dir is not None and text_output_path(output_dir).is_file()

    def _has_base_file(self) -> bool:
        output_dir = self._output_dir()
        return output_dir is not None and base_plate_output_path(output_dir).is_file()

    def _plane_ready(self) -> bool:
        return self._plane_z is not None

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_open.setEnabled(enabled)
        has_mesh = self.viewer.mesh is not None
        plane_ready = self._plane_ready()
        self.spin_slice_height.setEnabled(enabled and plane_ready)
        self.btn_slice.setEnabled(enabled and has_mesh and plane_ready)
        has_slice = self._has_slice_file()
        has_border = self._has_border_file()
        self.spin_border_offset.setEnabled(enabled and has_slice)
        self.spin_border_thickness.setEnabled(
            enabled and has_slice and not self.chk_connected_border.isChecked()
        )
        self.chk_connected_border.setEnabled(enabled and has_slice)
        self.btn_border_slice.setEnabled(enabled and has_slice)
        self.spin_border_extrude.setEnabled(enabled and has_border)
        self.btn_extrude_border.setEnabled(enabled and has_border)
        has_extruded = self._has_extruded_border_file()
        self.spin_text_size.setEnabled(enabled and has_mesh)
        self.edit_surface_text.setEnabled(enabled and has_mesh)
        self.slider_text_h.setEnabled(enabled and has_mesh)
        self.slider_text_v.setEnabled(enabled and has_mesh)
        self.btn_add_base.setEnabled(enabled and has_extruded)
        self.btn_download_assembly.setEnabled(enabled and has_extruded)
        self._update_text_controls(enabled)
        plane_enabled = enabled and has_mesh
        self.slider_plane_z.setEnabled(plane_enabled)
        self.spin_plane_z.setEnabled(plane_enabled)
        self.btn_confirm_plane.setEnabled(plane_enabled)

    def _update_size_label(self, mesh=None) -> None:
        """Show Width × Length × Height (X × Y × Z) for the loaded mesh."""
        if mesh is None:
            mesh = self.viewer.mesh
        if mesh is None:
            self.lbl_size.setText("Size: —")
            return

        bounds = mesh.bounds
        width = float(bounds[1] - bounds[0])
        length = float(bounds[3] - bounds[2])
        height = float(bounds[5] - bounds[4])
        self.lbl_size.setText(
            f"Size: {width:.2f} × {length:.2f} × {height:.2f} mm\n"
            f"(W × L × H)"
        )

    def _show_step_result(self, path: Path, description: str) -> None:
        """Load a generated STL into the viewer and keep plane overlays visible."""
        mesh = load_stl_polydata(path)
        self.viewer.set_mesh(mesh)
        self.viewer.set_view_mode(self.footer.current_view_mode())
        self._refresh_plane_overlays()
        # Keep File panel details on the originally opened STL only.

    def _text_ready(self) -> bool:
        return bool(self.edit_surface_text.text().strip()) and self.viewer.mesh is not None

    def _update_text_controls(self, enabled: bool = True) -> None:
        ready = enabled and self._text_ready()
        if hasattr(self, "btn_write_surface_text"):
            self.btn_write_surface_text.setEnabled(ready)

    def _reset_text_state(self) -> None:
        self._text_preview_mesh = None
        if hasattr(self, "slider_text_h"):
            self.slider_text_h.setValue(_SLIDER_STEPS // 2)
        if hasattr(self, "slider_text_v"):
            self.slider_text_v.setValue(_SLIDER_STEPS // 2)
        self._update_text_position_label()
        self._update_text_controls()

    def _run_busy(self, title: str, operation: Callable[[], T]) -> T:
        self.footer.set_busy(True)
        self.footer.set_stage(title, 0, 0)
        self._set_controls_enabled(False)
        try:
            return operation()
        finally:
            self.footer.set_busy(False)
            self._set_controls_enabled(True)
            self.footer.show_message("Ready")

    def _open_stl(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open STL File",
            "",
            "STL Files (*.stl);;All Files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)

        def load_file():
            self.footer.set_stage("Loading STL file", 0, 1)
            QApplication.processEvents()
            mesh = self.viewer.load_stl(path)
            self.viewer.set_view_mode(self.footer.current_view_mode())
            self.footer.set_stage("STL loaded", 1, 1)
            return mesh

        try:
            mesh = self._run_busy("Opening STL…", load_file)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", f"Could not load STL:\n{exc}")
            return

        self._current_path = path
        self.lbl_file.setText(path.name)
        self._update_size_label(mesh)
        self._reset_plane()
        self._reset_text_state()
        self._init_plane_controls(mesh)
        self._set_controls_enabled(True)
        self.footer.show_message(
            f"Loaded {path.name} — {mesh.n_cells:,} triangles"
        )

    def _reset_plane(self) -> None:
        self._plane_z = None
        self._preview_z = None
        self.viewer.set_point_pick_callback(None)
        self.viewer.clear_overlays()
        self._update_plane_status_label()

    def _init_plane_controls(self, mesh) -> None:
        z_min, z_max = mesh_z_bounds(mesh)
        if z_max <= z_min:
            z_max = z_min + 1.0

        self._slider_sync = True
        self.spin_plane_z.setRange(z_min, z_max)
        preview = z_min + (z_max - z_min) * 0.25
        self.spin_plane_z.setValue(preview)
        self._set_slider_from_z(preview)
        self._slider_sync = False

        self._preview_z = preview
        self._refresh_plane_overlays()
        self._update_plane_z_label()

    def _z_from_slider(self) -> float:
        mesh = self.viewer.mesh
        if mesh is None:
            return 0.0
        z_min, z_max = mesh_z_bounds(mesh)
        if z_max <= z_min:
            return z_min
        fraction = self.slider_plane_z.value() / _SLIDER_STEPS
        return z_min + fraction * (z_max - z_min)

    def _set_slider_from_z(self, z: float) -> None:
        mesh = self.viewer.mesh
        if mesh is None:
            return
        z_min, z_max = mesh_z_bounds(mesh)
        if z_max <= z_min:
            self.slider_plane_z.setValue(0)
            return
        clamped = min(max(z, z_min), z_max)
        fraction = (clamped - z_min) / (z_max - z_min)
        self.slider_plane_z.setValue(int(round(fraction * _SLIDER_STEPS)))

    def _on_plane_slider_changed(self, _value: int) -> None:
        if self._slider_sync:
            return
        self._preview_z = self._z_from_slider()
        self._slider_sync = True
        self.spin_plane_z.setValue(self._preview_z)
        self._slider_sync = False
        self._update_plane_z_label()
        self._refresh_plane_overlays()

    def _on_plane_spin_changed(self, value: float) -> None:
        if self._slider_sync:
            return
        self._preview_z = value
        self._slider_sync = True
        self._set_slider_from_z(value)
        self._slider_sync = False
        self._update_plane_z_label()
        self._refresh_plane_overlays()

    def _update_plane_z_label(self) -> None:
        if self._preview_z is None:
            self.lbl_plane_z.setText("Z: —")
            return
        self.lbl_plane_z.setText(f"Preview Z: {self._preview_z:.3f} mm")

    def _refresh_plane_overlays(self) -> None:
        mesh = self.viewer.mesh
        if mesh is None:
            self.viewer.clear_overlays()
            return

        overlays: list[OverlayMesh] = []

        if self._plane_z is not None:
            overlays.append(
                OverlayMesh(
                    mesh=make_xy_plane_mesh(mesh, self._plane_z),
                    color=_PLANE_CONFIRMED_COLOR,
                    opacity=0.32,
                    style="surface",
                    show_edges=True,
                    edge_color="#74c7ec",
                )
            )

        if self._preview_z is not None:
            overlays.append(
                OverlayMesh(
                    mesh=make_xy_plane_mesh(mesh, self._preview_z),
                    color=_PLANE_PREVIEW_COLOR,
                    opacity=0.42,
                    style="surface",
                    show_edges=True,
                    edge_color=OVERLAY_COLOR,
                )
            )

        if self._text_preview_mesh is not None:
            overlays.append(
                OverlayMesh(
                    mesh=self._text_preview_mesh,
                    color=_TEXT_PREVIEW_COLOR,
                    opacity=0.92,
                    style="surface",
                    show_edges=True,
                    edge_color="#b4befe",
                )
            )

        self.viewer.set_overlays(overlays)

    def _horizontal_text_offset(self) -> float:
        mesh = self.viewer.mesh
        if mesh is None:
            return 0.0
        bounds = mesh.bounds
        span = max(bounds[1] - bounds[0], 1e-6)
        fraction = self.slider_text_h.value() / _SLIDER_STEPS
        return (fraction - 0.5) * span

    def _vertical_text_offset(self) -> float:
        mesh = self.viewer.mesh
        if mesh is None:
            return 0.0
        z_min, z_max = mesh.bounds[4], mesh.bounds[5]
        span = max(z_max - z_min, 1e-6)
        fraction = self.slider_text_v.value() / _SLIDER_STEPS
        return (fraction - 0.5) * span

    def _current_text_placement(self):
        mesh = self.viewer.mesh
        if mesh is None:
            raise ValueError("No mesh loaded.")
        return placement_on_outer_surface(
            mesh,
            self._horizontal_text_offset(),
            self._vertical_text_offset(),
        )

    def _update_text_position_label(self) -> None:
        if not hasattr(self, "lbl_text_position"):
            return
        self.lbl_text_position.setText(
            "Text position: "
            f"H {self._horizontal_text_offset():+.1f} mm, "
            f"V {self._vertical_text_offset():+.1f} mm"
        )

    def _on_text_content_changed(self) -> None:
        self._update_text_controls()
        self._update_text_preview()

    def _on_text_position_changed(self, _value: int) -> None:
        self._update_text_position_label()
        self._update_text_preview()

    def _update_text_preview(self) -> None:
        mesh = self.viewer.mesh
        text = self.edit_surface_text.text().strip()
        if mesh is None or not text:
            self._text_preview_mesh = None
            self._refresh_plane_overlays()
            return

        try:
            placement = self._current_text_placement()
            self._text_preview_mesh = build_text_on_surface(
                mesh,
                text,
                placement,
                self.spin_text_size.value(),
            )
        except Exception:  # noqa: BLE001
            self._text_preview_mesh = None

        self._refresh_plane_overlays()

    def _confirm_plane(self) -> None:
        if self._preview_z is None:
            return

        self._plane_z = self._preview_z
        self._update_plane_status_label()
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(f"Confirmed slice plane at Z = {self._plane_z:.3f} mm")

    def _update_plane_status_label(self) -> None:
        if self._plane_z is None:
            self.lbl_plane_status.setText("Plane: not set")
        else:
            self.lbl_plane_status.setText(f"Plane: Z = {self._plane_z:.3f} mm")

    def _slice_at_plane(self) -> None:
        if self._current_path is None or self.viewer.mesh is None:
            return

        if not self._plane_ready():
            QMessageBox.warning(
                self,
                "Plane Required",
                "Confirm the slice plane before slicing.",
            )
            return

        slice_height_mm = self.spin_slice_height.value()
        output_dir = output_dir_for_stl(self._current_path)
        progress = self._progress_callback()

        def run_slice():
            return slice_single_at_z(
                self.viewer.mesh,
                output_dir,
                self._plane_z,
                slice_height_mm,
                progress=progress,
            )

        try:
            result = self._run_busy("Slicing…", run_slice)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Slice Error", f"Slicing failed:\n{exc}")
            return

        if not result.piece_paths:
            QMessageBox.warning(
                self,
                "No Slice Created",
                "No geometry was found at the marked plane for the given thickness.",
            )
            return

        QMessageBox.information(
            self,
            "Slice Complete",
            f"Saved 1 slice starting at Z = {self._plane_z:.3f} mm "
            f"({slice_height_mm:.2f} mm thick).\n\n"
            f"Output folder:\n{result.output_dir}\n\n"
            f"File: {result.piece_paths[0].name}",
        )
        self.footer.show_message(
            f"Slice saved → {result.output_dir.name}/{result.piece_paths[0].name}"
        )
        self._show_step_result(result.piece_paths[0], "Slice")
        self._set_controls_enabled(True)

    def _create_border_slice(self) -> None:
        output_dir = self._output_dir()
        if output_dir is None:
            return

        slice_path = slice_output_path(output_dir)
        if not slice_path.is_file():
            QMessageBox.warning(
                self,
                "Slice Required",
                "Create a slice first. Expected file:\n" + str(slice_path),
            )
            return

        offset_mm = self.spin_border_offset.value()
        connected = self.chk_connected_border.isChecked()
        thickness_mm = self.spin_border_thickness.value()
        border_path = border_output_path(output_dir)
        progress = self._progress_callback()

        def run_border():
            return create_border_from_slice(
                slice_path,
                border_path,
                offset_mm,
                thickness_mm,
                connected=connected,
                progress=progress,
            )

        try:
            result = self._run_busy("Creating border…", run_border)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Border Error", f"Border creation failed:\n{exc}")
            return

        if connected:
            details = (
                f"Created connected border plane with {result.triangle_count:,} triangles.\n\n"
                f"Offset: {offset_mm:.2f} mm\n"
                f"(Thickness ignored)\n\n"
                f"Saved to:\n{result.border_path}"
            )
        else:
            details = (
                f"Created border with {result.triangle_count:,} triangles.\n\n"
                f"Offset: {offset_mm:.2f} mm\n"
                f"Thickness: {thickness_mm:.2f} mm\n\n"
                f"Saved to:\n{result.border_path}"
            )

        QMessageBox.information(self, "Border Slice Complete", details)
        self.footer.show_message(f"Border saved → {result.border_path.name}")
        self._show_step_result(result.border_path, "Border")
        self._set_controls_enabled(True)

    def _on_connected_border_toggled(self, checked: bool) -> None:
        has_slice = self._has_slice_file()
        self.spin_border_thickness.setEnabled(has_slice and not checked)

    def _extrude_border(self) -> None:
        output_dir = self._output_dir()
        if output_dir is None:
            return

        border_path = border_output_path(output_dir)
        if not border_path.is_file():
            QMessageBox.warning(
                self,
                "Border Required",
                "Create a border first. Expected file:\n" + str(border_path),
            )
            return

        height_mm = self.spin_border_extrude.value()
        extruded_path = border_extruded_output_path(output_dir)
        progress = self._progress_callback()

        def run_extrude():
            return extrude_border_on_z(
                border_path,
                extruded_path,
                height_mm,
                progress=progress,
            )

        try:
            result = self._run_busy("Extruding border…", run_extrude)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Extrude Error", f"Border extrusion failed:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Extrude Border Complete",
            f"Increased border height by {height_mm:.2f} mm.\n\n"
            f"Triangles: {result.triangle_count:,}\n\n"
            f"Saved to:\n{result.extruded_path}",
        )
        self.footer.show_message(f"Extruded border saved → {result.extruded_path.name}")
        self._show_step_result(result.extruded_path, "Border extruded")
        self._reset_text_state()
        self._update_text_preview()
        self._set_controls_enabled(True)

    def _write_surface_text(self) -> None:
        mesh = self.viewer.mesh
        if mesh is None:
            return

        text = self.edit_surface_text.text().strip()
        if not text:
            QMessageBox.warning(self, "Text Required", "Enter text before writing.")
            return

        output_dir = self._output_dir()
        if output_dir is None:
            return

        font_size_mm = self.spin_text_size.value()
        output_path = text_output_path(output_dir)
        placement = self._current_text_placement()
        progress = self._progress_callback()

        def run_write():
            return write_text_on_surface(
                mesh,
                text,
                placement,
                font_size_mm,
                output_path,
                merge_with_source=False,
                progress=progress,
            )

        try:
            result = self._run_busy("Writing surface text…", run_write)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Surface Text Error",
                f"Could not write text on surface:\n{exc}",
            )
            return

        self._text_preview_mesh = build_text_on_surface(
            mesh,
            text,
            placement,
            font_size_mm,
        )
        self._refresh_plane_overlays()

        QMessageBox.information(
            self,
            "Surface Text Complete",
            f"Wrote {result.triangle_count:,} text triangles.\n\n"
            f"Saved to:\n{result.text_path}",
        )
        self.footer.show_message(f"Surface text saved → {result.text_path.name}")
        self._set_controls_enabled(True)

    def _add_base_plate(self) -> None:
        output_dir = self._output_dir()
        if output_dir is None:
            return

        extruded_path = border_extruded_output_path(output_dir)
        if not extruded_path.is_file():
            QMessageBox.warning(
                self,
                "Extruded Border Required",
                "Extrude the border first. Expected file:\n" + str(extruded_path),
            )
            return

        base_path = base_plate_output_path(output_dir)
        progress = self._progress_callback()

        def run_base():
            return create_base_plate(extruded_path, base_path, progress=progress)

        try:
            result = self._run_busy("Adding base plate…", run_base)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Base Plate Error", f"Could not add base plate:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Base Plate Added",
            "Created a 2 mm base plate, 5 mm larger than the extruded border "
            "in X and Y.\n\n"
            f"Triangles: {result.triangle_count:,}\n\n"
            f"Saved to:\n{result.base_path}",
        )
        self.footer.show_message(f"Base plate saved → {result.base_path.name}")
        self._set_controls_enabled(True)

    def _download_assembly(self) -> None:
        output_dir = self._output_dir()
        if output_dir is None:
            return

        extruded_path = border_extruded_output_path(output_dir)
        if not extruded_path.is_file():
            QMessageBox.warning(
                self,
                "Extruded Border Required",
                "Extrude the border first. Expected file:\n" + str(extruded_path),
            )
            return

        default_path = assembly_output_path(output_dir)
        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Download Assembly STL",
            str(default_path),
            "STL Files (*.stl);;All Files (*)",
        )
        if not save_path_str:
            return

        save_path = Path(save_path_str)
        text_path = text_output_path(output_dir)
        base_path = base_plate_output_path(output_dir)
        progress = self._progress_callback()

        def run_export():
            return export_assembly(
                save_path,
                extruded_path,
                text_path=text_path if text_path.is_file() else None,
                base_path=base_path if base_path.is_file() else None,
                progress=progress,
            )

        try:
            result = self._run_busy("Exporting assembly…", run_export)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Assembly Export Error",
                f"Could not export assembly:\n{exc}",
            )
            return

        parts_note = []
        if base_path.is_file():
            parts_note.append("base plate")
        parts_note.append("extruded border")
        if text_path.is_file():
            parts_note.append("surface text")

        QMessageBox.information(
            self,
            "Assembly Exported",
            f"Combined {result.part_count} part(s): {', '.join(parts_note)}.\n\n"
            f"Triangles: {result.triangle_count:,}\n\n"
            f"Saved to:\n{result.assembly_path}",
        )
        self.footer.show_message(f"Assembly saved → {result.assembly_path.name}")
        self._show_step_result(result.assembly_path, "Assembly")
        self._set_controls_enabled(True)


def create_moldmaker_app(argv: list[str] | None = None) -> QApplication:
    return create_app(argv)
