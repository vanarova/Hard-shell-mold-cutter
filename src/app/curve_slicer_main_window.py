"""Main window for Curve Slicer — one curved plane grid, confirm, slice."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.curve_slice import (
    DEFAULT_GRID_POINTS,
    DEFAULT_KNIFE_THICKNESS_MM,
    DEFAULT_SIZE_MARGIN_PERCENT,
    MAX_GRID_POINTS,
    MAX_KNIFE_THICKNESS_MM,
    MAX_SIZE_MARGIN_PERCENT,
    MIN_GRID_POINTS,
    MIN_KNIFE_THICKNESS_MM,
    MIN_SIZE_MARGIN_PERCENT,
    auto_fit_curve_plane,
    curve_plane_half_extents,
    model_extent_along_normal,
    slice_mesh_by_curve_plane,
)
from app.footer_bar import FooterBar
from app.main_window import create_app
from app.plane_deform import (
    control_point_glyphs,
    control_points_world,
    make_deformed_plane_mesh,
    oriented_plane_axes,
    place_equidistant_uv,
    pose_frame,
)
from app.slice_axis import output_dir_for_stl
from app.stl_viewer import OVERLAY_COLOR, OverlayMesh, StlViewer, load_stl_polydata
from app.xy_plane import (
    mesh_center_xy,
    mesh_x_bounds,
    mesh_y_bounds,
    mesh_z_bounds,
)

T = TypeVar("T")
AxisName = Literal["x", "y", "z"]

_SIDEBAR_WIDTH = 340
_SLIDER_STEPS = 1000
_DISP_SLIDER_STEPS = 1000
_CTRL_PICK_PIXEL_RADIUS = 28.0
_PLANE_PREVIEW = "#a6e3a1"
_PLANE_CONFIRMED = "#94e2d5"
_CTRL_COLOR = "#f9e2af"
_CTRL_SELECTED = "#f38ba8"
_PART_A_COLOR = "#89b4fa"
_PART_B_COLOR = "#fab387"


@dataclass
class PlanePose:
    x: float | None = None
    y: float | None = None
    z: float | None = None
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.x is not None and self.y is not None and self.z is not None

    def copy_from(self, other: "PlanePose") -> None:
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.rx = other.rx
        self.ry = other.ry
        self.rz = other.rz

    def clear(self) -> None:
        self.x = None
        self.y = None
        self.z = None
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0

    def reset_preview(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.rx = 0.0
        self.ry = 0.0
        self.rz = 0.0


class CurveSlicerMainWindow(QMainWindow):
    """Minimal tool: one deformable plane grid → confirm → slice into 2 parts."""

    def __init__(self) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._slider_sync = False
        self._deform_sync = False
        self._confirmed = False
        self._preview = PlanePose()
        self._confirmed_pose = PlanePose()
        self._ctrl_uv = place_equidistant_uv(DEFAULT_GRID_POINTS)
        self._ctrl_disp = np.zeros(len(self._ctrl_uv), dtype=float)
        self._ctrl_selected = 0
        self._ctrl_world_points: np.ndarray | None = None
        self._half_i = 50.0
        self._half_j = 50.0
        self._part_a = None
        self._part_b = None

        self.setWindowTitle("Curve Slicer")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_sidebar())

        viewer_column = QWidget()
        viewer_layout = QVBoxLayout(viewer_column)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)
        self.viewer = StlViewer()
        viewer_layout.addWidget(self.viewer, stretch=1)
        self.footer = FooterBar()
        self.footer.view_mode_changed.connect(self.viewer.set_view_mode)
        self.footer.plane_opacity_changed.connect(self.viewer.set_overlay_opacity_scale)
        self.footer.model_opacity_changed.connect(self.viewer.set_model_opacity)
        viewer_layout.addWidget(self.footer)
        root.addWidget(viewer_column, stretch=1)

        self.footer.show_message("Open an STL file to begin.")

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setFixedWidth(_SIDEBAR_WIDTH)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        panel = QFrame()
        panel.setObjectName("sidebar")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 10, 12)
        layout.setSpacing(12)

        title = QLabel("Curve Slicer")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        subtitle = QLabel("One curved plane grid → confirm → cut model in two")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(subtitle)

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        self.btn_open = QPushButton("Open STL…")
        self.btn_open.clicked.connect(self._open_stl)
        file_layout.addWidget(self.btn_open)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setObjectName("fileLabel")
        file_layout.addWidget(self.lbl_file)
        layout.addWidget(file_group)

        layout.addWidget(self._build_plane_group())
        layout.addWidget(self._build_points_group())

        self.btn_confirm = QPushButton("Confirm Grid & Points")
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.clicked.connect(self._confirm)
        layout.addWidget(self.btn_confirm)

        self.btn_slice = QPushButton("Slice Model")
        self.btn_slice.setEnabled(False)
        self.btn_slice.setToolTip(
            "Knife-cut with the confirmed plane: only where the blade touches "
            "the model (inside the plane rectangle and knife thickness). "
            "Points far from the plane are not cut. Saves part A and part B."
        )
        self.btn_slice.clicked.connect(self._slice)
        layout.addWidget(self.btn_slice)

        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _build_plane_group(self) -> QGroupBox:
        group = QGroupBox("Cutting Plane")
        layout = QVBoxLayout(group)
        self.lbl_status = QLabel("Plane: not confirmed")
        self.lbl_status.setObjectName("fileLabel")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.lbl_x, self.slider_x, self.spin_x = self._axis_controls(layout, "x")
        self.lbl_y, self.slider_y, self.spin_y = self._axis_controls(layout, "y")
        self.lbl_z, self.slider_z, self.spin_z = self._axis_controls(layout, "z")
        self.lbl_rx, self.slider_rx = self._rotation_controls(layout, "x")
        self.lbl_ry, self.slider_ry = self._rotation_controls(layout, "y")
        self.lbl_rz, self.slider_rz = self._rotation_controls(layout, "z")

        size_info = QLabel(
            "Plane size vs model footprint. Positive enlarges, negative shrinks "
            "(10% = 1.10×, -20% = 0.80×)."
        )
        size_info.setWordWrap(True)
        size_info.setObjectName("fileLabel")
        layout.addWidget(size_info)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size margin"), stretch=1)
        self.spin_size_margin = QDoubleSpinBox()
        self.spin_size_margin.setDecimals(1)
        self.spin_size_margin.setRange(
            MIN_SIZE_MARGIN_PERCENT, MAX_SIZE_MARGIN_PERCENT
        )
        self.spin_size_margin.setSingleStep(5.0)
        self.spin_size_margin.setValue(DEFAULT_SIZE_MARGIN_PERCENT)
        self.spin_size_margin.setSuffix(" %")
        self.spin_size_margin.setMinimumWidth(88)
        self.spin_size_margin.setEnabled(False)
        self.spin_size_margin.setToolTip(
            "Change plane size relative to the model length/width. "
            "0% = exact footprint, 10% = larger, negative = smaller "
            f"(down to {MIN_SIZE_MARGIN_PERCENT:.0f}%)."
        )
        self.spin_size_margin.valueChanged.connect(self._on_size_margin_changed)
        size_row.addWidget(self.spin_size_margin)
        layout.addLayout(size_row)

        self.lbl_plane_size = QLabel("Plane size: —")
        self.lbl_plane_size.setObjectName("fileLabel")
        self.lbl_plane_size.setWordWrap(True)
        layout.addWidget(self.lbl_plane_size)

        knife_info = QLabel(
            "Knife cut: only geometry near the plane is sliced. Points far "
            "from the plane along its normal are left uncut. "
            "0 mm = cut all the way through under the plane."
        )
        knife_info.setWordWrap(True)
        knife_info.setObjectName("fileLabel")
        layout.addWidget(knife_info)

        knife_row = QHBoxLayout()
        knife_row.addWidget(QLabel("Knife thickness"), stretch=1)
        self.spin_knife_thickness = QDoubleSpinBox()
        self.spin_knife_thickness.setDecimals(2)
        self.spin_knife_thickness.setRange(
            MIN_KNIFE_THICKNESS_MM, MAX_KNIFE_THICKNESS_MM
        )
        self.spin_knife_thickness.setSingleStep(1.0)
        self.spin_knife_thickness.setValue(DEFAULT_KNIFE_THICKNESS_MM)
        self.spin_knife_thickness.setSuffix(" mm")
        self.spin_knife_thickness.setMinimumWidth(96)
        self.spin_knife_thickness.setEnabled(False)
        self.spin_knife_thickness.setToolTip(
            "Band around the plane where the knife cuts. "
            "Small values only sever near the plane; far points stay uncut. "
            "0 = cut through the full depth under the plane rectangle."
        )
        knife_row.addWidget(self.spin_knife_thickness)
        layout.addLayout(knife_row)

        return group

    def _axis_controls(
        self, layout: QVBoxLayout, axis: AxisName
    ) -> tuple[QLabel, QSlider, QDoubleSpinBox]:
        upper = axis.upper()
        lbl = QLabel(f"Preview {upper}: —")
        lbl.setObjectName("fileLabel")
        layout.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, _SLIDER_STEPS)
        slider.setEnabled(False)
        slider.valueChanged.connect(lambda _v, a=axis: self._on_axis_slider(a))
        layout.addWidget(slider)
        row = QHBoxLayout()
        row.addWidget(QLabel(f"Exact {upper}"), stretch=1)
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSuffix(" mm")
        spin.setMinimumWidth(88)
        spin.setEnabled(False)
        spin.valueChanged.connect(lambda v, a=axis: self._on_axis_spin(a, v))
        row.addWidget(spin)
        layout.addLayout(row)
        return lbl, slider, spin

    def _rotation_controls(
        self, layout: QVBoxLayout, axis: AxisName
    ) -> tuple[QLabel, QSlider]:
        upper = axis.upper()
        lbl = QLabel(f"Rotate {upper}: 0°")
        lbl.setObjectName("fileLabel")
        layout.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-90, 90)
        slider.setValue(0)
        slider.setEnabled(False)
        slider.valueChanged.connect(lambda v, a=axis: self._on_rot_slider(a, v))
        layout.addWidget(slider)
        return lbl, slider

    def _build_points_group(self) -> QGroupBox:
        group = QGroupBox("Points Grid")
        layout = QVBoxLayout(group)
        info = QLabel(
            "Equidistant grid points on the plane (up to "
            f"{MAX_GRID_POINTS}). Select a point and pull/push to curve the cut."
        )
        info.setWordWrap(True)
        info.setObjectName("fileLabel")
        layout.addWidget(info)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Grid points (N)"), stretch=1)
        self.spin_point_count = QSpinBox()
        self.spin_point_count.setRange(MIN_GRID_POINTS, MAX_GRID_POINTS)
        self.spin_point_count.setValue(DEFAULT_GRID_POINTS)
        self.spin_point_count.setEnabled(False)
        self.spin_point_count.valueChanged.connect(self._on_point_count_changed)
        count_row.addWidget(self.spin_point_count)
        layout.addLayout(count_row)

        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Selected point"), stretch=1)
        self.combo_point = QComboBox()
        self.combo_point.setEnabled(False)
        self.combo_point.currentIndexChanged.connect(self._on_point_selected)
        sel_row.addWidget(self.combo_point, stretch=2)
        layout.addLayout(sel_row)

        self.lbl_disp = QLabel("Pull / push: 0.000 mm")
        self.lbl_disp.setObjectName("fileLabel")
        layout.addWidget(self.lbl_disp)

        self.slider_disp = QSlider(Qt.Orientation.Horizontal)
        self.slider_disp.setRange(0, _DISP_SLIDER_STEPS)
        self.slider_disp.setValue(_DISP_SLIDER_STEPS // 2)
        self.slider_disp.setEnabled(False)
        self.slider_disp.valueChanged.connect(self._on_disp_slider)
        layout.addWidget(self.slider_disp)

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Exact pull / push"), stretch=1)
        self.spin_disp = QDoubleSpinBox()
        self.spin_disp.setDecimals(3)
        self.spin_disp.setSingleStep(0.5)
        self.spin_disp.setSuffix(" mm")
        self.spin_disp.setMinimumWidth(88)
        self.spin_disp.setEnabled(False)
        self.spin_disp.valueChanged.connect(self._on_disp_spin)
        disp_row.addWidget(self.spin_disp)
        layout.addLayout(disp_row)

        self.btn_reset_points = QPushButton("Reset Points")
        self.btn_reset_points.setEnabled(False)
        self.btn_reset_points.clicked.connect(self._reset_points)
        layout.addWidget(self.btn_reset_points)

        self._rebuild_point_combo()
        return group

    def _progress_callback(self) -> Callable[[str, int, int], None]:
        def report(message: str, step: int = 0, total: int = 0) -> None:
            self.footer.set_stage(message, step, total)
            QApplication.processEvents()

        return report

    def _run_busy(self, title: str, operation: Callable[[], T]) -> T:
        self._set_controls_enabled(False)
        self.footer.set_busy(True)
        self.footer.set_stage(title, 0, 0)
        QApplication.processEvents()
        try:
            return operation()
        finally:
            self.footer.set_busy(False)
            self._set_controls_enabled(True)
            self.footer.show_message("Ready")

    def _set_controls_enabled(self, enabled: bool) -> None:
        has_mesh = self.viewer.mesh is not None
        ready = enabled and has_mesh
        for w in (
            self.slider_x,
            self.spin_x,
            self.slider_y,
            self.spin_y,
            self.slider_z,
            self.spin_z,
            self.slider_rx,
            self.slider_ry,
            self.slider_rz,
            self.spin_size_margin,
            self.spin_knife_thickness,
            self.spin_point_count,
            self.combo_point,
            self.slider_disp,
            self.spin_disp,
            self.btn_reset_points,
            self.btn_confirm,
        ):
            w.setEnabled(ready)
        self.btn_slice.setEnabled(ready and self._confirmed)
        self.btn_open.setEnabled(enabled)

    def _open_stl(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open STL",
            "",
            "STL Files (*.stl);;All Files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)

        def load():
            return load_stl_polydata(path, progress=self._progress_callback())

        try:
            mesh = self._run_busy("Loading STL…", load)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", f"Could not load STL:\n{exc}")
            return

        self._current_path = path
        self.viewer.set_mesh(mesh, reset_camera=True)
        self.viewer.set_view_mode(self.footer.current_view_mode())
        self.lbl_file.setText(path.name)
        self._part_a = None
        self._part_b = None
        self._confirmed = False
        self._confirmed_pose.clear()
        self._init_plane(mesh)
        self._reset_points()
        self._set_controls_enabled(True)
        self.footer.show_message(f"Loaded {path.name} — {mesh.n_cells:,} triangles")

    def _axis_bounds(self, mesh, axis: AxisName) -> tuple[float, float]:
        if axis == "x":
            return mesh_x_bounds(mesh)
        if axis == "y":
            return mesh_y_bounds(mesh)
        return mesh_z_bounds(mesh)

    def _size_margin_percent(self) -> float:
        return float(self.spin_size_margin.value())

    def _update_plane_size_label(self) -> None:
        margin = self._size_margin_percent()
        length = 2.0 * max(self._half_i, self._half_j)
        width = 2.0 * min(self._half_i, self._half_j)
        self.lbl_plane_size.setText(
            f"Plane size: {length:.1f} × {width:.1f} mm ({margin:+.1f}%)"
        )

    def _update_plane_size_from_model(self) -> None:
        """Resize plane to model footprint + margin with long side ⊥ model long side."""
        mesh = self.viewer.mesh
        if mesh is None or not self._preview.is_complete:
            return
        self._half_i, self._half_j = curve_plane_half_extents(
            mesh,
            rx_deg=self._preview.rx,
            ry_deg=self._preview.ry,
            rz_deg=self._preview.rz,
            size_margin_percent=self._size_margin_percent(),
        )
        self._update_plane_size_label()

    def _on_size_margin_changed(self, _value: float) -> None:
        if self.viewer.mesh is None:
            return
        self._confirmed = False
        self._update_plane_size_from_model()
        self._update_labels()
        self._refresh_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(
            f"Plane size margin: {self._size_margin_percent():+.1f}%"
        )

    def _init_plane(self, mesh) -> None:
        cx, cy = mesh_center_xy(mesh)
        z_min, z_max = mesh_z_bounds(mesh)
        z = 0.5 * (z_min + z_max)
        margin = self._size_margin_percent()
        fit = auto_fit_curve_plane(mesh, size_margin_percent=margin)
        self._half_i = fit.half_i
        self._half_j = fit.half_j
        self._slider_sync = True
        for axis, spin, slider, value in (
            ("x", self.spin_x, self.slider_x, cx),
            ("y", self.spin_y, self.slider_y, cy),
            ("z", self.spin_z, self.slider_z, z),
        ):
            lo, hi = self._axis_bounds(mesh, axis)  # type: ignore[arg-type]
            spin.setRange(lo, hi)
            spin.setValue(value)
            if hi <= lo:
                slider.setValue(0)
            else:
                slider.setValue(int(round((value - lo) / (hi - lo) * _SLIDER_STEPS)))
        self.slider_rx.setValue(0)
        self.slider_ry.setValue(0)
        # Auto-rotate so plane long side is perpendicular to model long side.
        self.slider_rz.setValue(int(round(fit.rz_deg)))
        self._slider_sync = False
        self._preview.reset_preview(cx, cy, z)
        self._preview.rz = float(fit.rz_deg)
        # Default knife = thin band (~5% of model depth) so far Z is not cut.
        # User can set 0 mm for a full through-cut under the plane.
        normal_span = model_extent_along_normal(
            mesh, rx_deg=0.0, ry_deg=0.0, rz_deg=float(fit.rz_deg)
        )
        default_knife = max(1.0, 0.05 * normal_span)
        self.spin_knife_thickness.blockSignals(True)
        self.spin_knife_thickness.setValue(default_knife)
        self.spin_knife_thickness.blockSignals(False)
        self._update_plane_size_label()
        self._update_labels()
        self._refresh_overlays()
        scale = 1.0 + margin / 100.0
        self.footer.show_message(
            f"Plane sized to model: "
            f"{fit.model_length * scale:.1f} × {fit.model_width * scale:.1f} mm "
            f"({margin:+.1f}%), Rz={fit.rz_deg:.0f}°, "
            f"knife={default_knife:.2f} mm"
        )

    def _on_axis_slider(self, axis: AxisName) -> None:
        if self._slider_sync or self.viewer.mesh is None:
            return
        slider = {"x": self.slider_x, "y": self.slider_y, "z": self.slider_z}[axis]
        spin = {"x": self.spin_x, "y": self.spin_y, "z": self.spin_z}[axis]
        lo, hi = self._axis_bounds(self.viewer.mesh, axis)
        value = lo if hi <= lo else lo + (slider.value() / _SLIDER_STEPS) * (hi - lo)
        setattr(self._preview, axis, value)
        self._confirmed = False
        self._slider_sync = True
        spin.setValue(value)
        self._slider_sync = False
        self._update_plane_size_from_model()
        self._update_labels()
        self._refresh_overlays()
        self._set_controls_enabled(True)

    def _on_axis_spin(self, axis: AxisName, value: float) -> None:
        if self._slider_sync or self.viewer.mesh is None:
            return
        setattr(self._preview, axis, float(value))
        self._confirmed = False
        lo, hi = self._axis_bounds(self.viewer.mesh, axis)
        slider = {"x": self.slider_x, "y": self.slider_y, "z": self.slider_z}[axis]
        self._slider_sync = True
        if hi <= lo:
            slider.setValue(0)
        else:
            clamped = min(max(value, lo), hi)
            slider.setValue(int(round((clamped - lo) / (hi - lo) * _SLIDER_STEPS)))
        self._slider_sync = False
        self._update_plane_size_from_model()
        self._update_labels()
        self._refresh_overlays()
        self._set_controls_enabled(True)

    def _on_rot_slider(self, axis: AxisName, value: int) -> None:
        if self._slider_sync:
            return
        setattr(self._preview, f"r{axis}", float(value))
        self._confirmed = False
        self._update_plane_size_from_model()
        self._update_labels()
        self._refresh_overlays()
        self._set_controls_enabled(True)

    def _update_labels(self) -> None:
        for axis, lbl in (
            ("x", self.lbl_x),
            ("y", self.lbl_y),
            ("z", self.lbl_z),
        ):
            value = getattr(self._preview, axis)
            if value is None:
                lbl.setText(f"Preview {axis.upper()}: —")
            else:
                lbl.setText(f"Preview {axis.upper()}: {value:.3f} mm")
        self.lbl_rx.setText(f"Rotate X: {self._preview.rx:.0f}°")
        self.lbl_ry.setText(f"Rotate Y: {self._preview.ry:.0f}°")
        self.lbl_rz.setText(f"Rotate Z: {self._preview.rz:.0f}°")
        if self._confirmed and self._confirmed_pose.is_complete:
            p = self._confirmed_pose
            self.lbl_status.setText(
                f"Confirmed: X={p.x:.3f} Y={p.y:.3f} Z={p.z:.3f} mm, "
                f"Rx={p.rx:.0f}° Ry={p.ry:.0f}° Rz={p.rz:.0f}°, "
                f"{len(self._ctrl_uv)} points"
            )
        else:
            self.lbl_status.setText("Plane: not confirmed")

    def _disp_limit(self) -> float:
        return max(self._half_i, self._half_j, 1.0)

    def _rebuild_point_combo(self) -> None:
        self.combo_point.blockSignals(True)
        self.combo_point.clear()
        for i in range(len(self._ctrl_uv)):
            self.combo_point.addItem(f"Point {i + 1}")
        self._ctrl_selected = min(
            max(self._ctrl_selected, 0), max(len(self._ctrl_uv) - 1, 0)
        )
        if len(self._ctrl_uv) > 0:
            self.combo_point.setCurrentIndex(self._ctrl_selected)
        self.combo_point.blockSignals(False)

    def _reset_points(self) -> None:
        n = self.spin_point_count.value()
        self._ctrl_uv = place_equidistant_uv(n)
        self._ctrl_disp = np.zeros(len(self._ctrl_uv), dtype=float)
        self._ctrl_selected = 0
        self._confirmed = False
        self._rebuild_point_combo()
        self._sync_disp_controls()
        self._refresh_overlays()
        self._set_controls_enabled(True)

    def _on_point_count_changed(self, value: int) -> None:
        self._ctrl_uv = place_equidistant_uv(int(value))
        self._ctrl_disp = np.zeros(len(self._ctrl_uv), dtype=float)
        self._ctrl_selected = 0
        self._confirmed = False
        self._rebuild_point_combo()
        self._sync_disp_controls()
        self._refresh_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(f"Grid points: {int(value)}")

    def _on_point_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._ctrl_disp):
            return
        self._ctrl_selected = int(index)
        self._sync_disp_controls()
        self._refresh_overlays()

    def _sync_disp_controls(self) -> None:
        if len(self._ctrl_disp) == 0:
            return
        idx = min(max(self._ctrl_selected, 0), len(self._ctrl_disp) - 1)
        value = float(self._ctrl_disp[idx])
        limit = self._disp_limit()
        self._deform_sync = True
        self.spin_disp.setRange(-limit, limit)
        self.spin_disp.setValue(value)
        fraction = (value + limit) / (2.0 * limit) if limit > 0 else 0.5
        self.slider_disp.setValue(int(round(fraction * _DISP_SLIDER_STEPS)))
        self._deform_sync = False
        self.lbl_disp.setText(f"Pull / push: {value:.3f} mm")

    def _set_selected_disp(self, value: float) -> None:
        if len(self._ctrl_disp) == 0:
            return
        idx = min(max(self._ctrl_selected, 0), len(self._ctrl_disp) - 1)
        limit = self._disp_limit()
        clamped = float(min(max(value, -limit), limit))
        self._ctrl_disp[idx] = clamped
        self._confirmed = False
        self.lbl_disp.setText(f"Pull / push: {clamped:.3f} mm")
        self._refresh_overlays()
        self._set_controls_enabled(True)

    def _on_disp_slider(self, slider_value: int) -> None:
        if self._deform_sync:
            return
        limit = self._disp_limit()
        value = -limit + (slider_value / _DISP_SLIDER_STEPS) * (2.0 * limit)
        self._deform_sync = True
        self.spin_disp.setValue(value)
        self._deform_sync = False
        self._set_selected_disp(value)

    def _on_disp_spin(self, value: float) -> None:
        if self._deform_sync:
            return
        limit = self._disp_limit()
        fraction = (value + limit) / (2.0 * limit) if limit > 0 else 0.5
        self._deform_sync = True
        self.slider_disp.setValue(int(round(fraction * _DISP_SLIDER_STEPS)))
        self._deform_sync = False
        self._set_selected_disp(value)

    def _active_pose(self) -> PlanePose | None:
        if self._preview.is_complete:
            return self._preview
        if self._confirmed_pose.is_complete:
            return self._confirmed_pose
        return None

    def _refresh_overlays(self) -> None:
        mesh = self.viewer.mesh
        if mesh is None:
            self._ctrl_world_points = None
            self.viewer.clear_overlays()
            return

        overlays: list[OverlayMesh] = []
        pose = self._active_pose()
        if pose is not None and pose.is_complete:
            center, normal = pose_frame(
                x=float(pose.x),
                y=float(pose.y),
                z=float(pose.z),
                rx=pose.rx,
                ry=pose.ry,
                rz=pose.rz,
            )
            i_axis, j_axis, _n = oriented_plane_axes(
                rx=pose.rx, ry=pose.ry, rz=pose.rz
            )
            plane_mesh = make_deformed_plane_mesh(
                center=center,
                normal=normal,
                uv=self._ctrl_uv,
                displacements=self._ctrl_disp,
                half_extent_i=self._half_i,
                half_extent_j=self._half_j,
                i_axis=i_axis,
                j_axis=j_axis,
            )
            color = _PLANE_CONFIRMED if self._confirmed else _PLANE_PREVIEW
            overlays.append(
                OverlayMesh(
                    mesh=plane_mesh,
                    color=color,
                    opacity=0.45,
                    style="surface",
                    show_edges=True,
                    edge_color=OVERLAY_COLOR,
                )
            )
            points = control_points_world(
                center=center,
                normal=normal,
                uv=self._ctrl_uv,
                displacements=self._ctrl_disp,
                half_extent_i=self._half_i,
                half_extent_j=self._half_j,
                i_axis=i_axis,
                j_axis=j_axis,
            )
            self._ctrl_world_points = np.asarray(points, dtype=float)
            radius = max(min(self._half_i, self._half_j) * 0.025, 0.4)
            others, selected = control_point_glyphs(
                points,
                selected_index=self._ctrl_selected,
                radius=radius,
            )
            if others is not None:
                overlays.append(
                    OverlayMesh(
                        mesh=others,
                        color=_CTRL_COLOR,
                        opacity=1.0,
                        style="surface",
                        show_edges=False,
                        edge_color=_CTRL_COLOR,
                    )
                )
            if selected is not None:
                overlays.append(
                    OverlayMesh(
                        mesh=selected,
                        color=_CTRL_SELECTED,
                        opacity=1.0,
                        style="surface",
                        show_edges=False,
                        edge_color=_CTRL_SELECTED,
                    )
                )
        else:
            self._ctrl_world_points = None

        if self._part_a is not None:
            overlays.append(
                OverlayMesh(
                    mesh=self._part_a,
                    color=_PART_A_COLOR,
                    opacity=0.85,
                    style="surface",
                    show_edges=True,
                    edge_color="#74c7ec",
                )
            )
        if self._part_b is not None:
            overlays.append(
                OverlayMesh(
                    mesh=self._part_b,
                    color=_PART_B_COLOR,
                    opacity=0.85,
                    style="surface",
                    show_edges=True,
                    edge_color="#fe640b",
                )
            )

        self.viewer.set_overlays(overlays)
        self._sync_point_picking()

    def _sync_point_picking(self) -> None:
        can_pick = (
            self.viewer.mesh is not None
            and self._ctrl_world_points is not None
            and len(self._ctrl_world_points) > 0
        )
        if can_pick:
            self.viewer.set_display_click_callback(self._on_display_click)
        else:
            self.viewer.set_display_click_callback(None)

    def _on_display_click(self, display_x: float, display_y: float) -> None:
        points = self._ctrl_world_points
        if points is None or len(points) == 0:
            return
        best_i = -1
        best_d2 = _CTRL_PICK_PIXEL_RADIUS * _CTRL_PICK_PIXEL_RADIUS
        for i, p in enumerate(points):
            dx, dy = self.viewer.world_to_display(p)
            d2 = (dx - display_x) ** 2 + (dy - display_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        if best_i < 0:
            return
        self._ctrl_selected = best_i
        self.combo_point.blockSignals(True)
        self.combo_point.setCurrentIndex(best_i)
        self.combo_point.blockSignals(False)
        self._sync_disp_controls()
        self._refresh_overlays()
        self.footer.show_message(f"Selected grid point {best_i + 1}")

    def _confirm(self) -> None:
        if not self._preview.is_complete:
            QMessageBox.information(
                self,
                "Confirm",
                "Set the cutting plane position first.",
            )
            return
        self._confirmed_pose.copy_from(self._preview)
        self._confirmed = True
        self._update_labels()
        self._refresh_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(
            f"Confirmed curved plane with {len(self._ctrl_uv)} grid points"
        )

    def _slice(self) -> None:
        if self._current_path is None or self.viewer.mesh is None:
            return
        if not self._confirmed or not self._confirmed_pose.is_complete:
            QMessageBox.warning(
                self,
                "Confirm Required",
                "Confirm the grid and points position before slicing.",
            )
            return

        pose = self._confirmed_pose
        model = self.viewer.mesh
        output_dir = output_dir_for_stl(self._current_path)
        progress = self._progress_callback()
        ctrl_uv = np.asarray(self._ctrl_uv, dtype=float).copy()
        ctrl_disp = np.asarray(self._ctrl_disp, dtype=float).copy()

        margin = self._size_margin_percent()
        knife_mm = float(self.spin_knife_thickness.value())
        half_i, half_j = curve_plane_half_extents(
            model,
            rx_deg=pose.rx,
            ry_deg=pose.ry,
            rz_deg=pose.rz,
            size_margin_percent=margin,
        )

        def run_slice():
            return slice_mesh_by_curve_plane(
                model,
                output_dir,
                origin=(float(pose.x), float(pose.y), float(pose.z)),
                rx_deg=pose.rx,
                ry_deg=pose.ry,
                rz_deg=pose.rz,
                ctrl_uv=ctrl_uv,
                ctrl_disp=ctrl_disp,
                half_extent_i=half_i,
                half_extent_j=half_j,
                size_margin_percent=margin,
                knife_thickness_mm=knife_mm,
                progress=progress,
            )

        try:
            result = self._run_busy("Slicing model…", run_slice)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Slice Error", f"Curve slice failed:\n{exc}")
            return

        self._part_a = result.part_a
        self._part_b = result.part_b
        self._refresh_overlays()
        self._set_controls_enabled(True)
        QMessageBox.information(
            self,
            "Slice Complete",
            f"Cut into two parts with {result.point_count} grid points.\n\n"
            f"Part A ({result.part_a.n_cells:,} tris):\n{result.path_a}\n\n"
            f"Part B ({result.part_b.n_cells:,} tris):\n{result.path_b}",
        )
        self.footer.show_message(
            f"Saved {result.path_a.name} and {result.path_b.name}"
        )


def create_curve_slicer_app(argv: list[str] | None = None) -> QApplication:
    return create_app(argv)
