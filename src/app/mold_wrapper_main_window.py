"""Main window for the 3D Mold Wrapper tool (start/end plane wrapping)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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

from app.footer_bar import FooterBar
from app.main_window import create_app
from app.plane_deform import (
    DEFAULT_CONTROL_POINTS,
    MAX_CONTROL_POINTS,
    MIN_CONTROL_POINTS,
    control_point_glyphs,
    control_points_world,
    make_deformed_plane_mesh,
    place_equidistant_uv,
    plane_half_extent,
    pose_frame,
)
from app.plane_wrap import (
    DEFAULT_POINT_DENSITY,
    MAX_POINT_DENSITY,
    MIN_POINT_DENSITY,
    wrap_back_start_to_backend,
    wrap_start_plane_to_model,
)
from app.slice_axis import output_dir_for_stl
from app.stl_viewer import OVERLAY_COLOR, OverlayMesh, StlViewer
from app.xy_plane import (
    make_oriented_plane_mesh,
    mesh_center_xy,
    mesh_x_bounds,
    mesh_y_bounds,
    mesh_z_bounds,
    plane_origin,
)
import numpy as np

T = TypeVar("T")
AxisName = Literal["x", "y", "z"]

_SIDEBAR_WIDTH = 340
_SLIDER_STEPS = 1000
_ROTATION_DEG_MIN = -90
_ROTATION_DEG_MAX = 90

_START_PREVIEW_COLOR = "#f9e2af"
_START_CONFIRMED_COLOR = "#89b4fa"
_END_PREVIEW_COLOR = "#a6e3a1"
_END_CONFIRMED_COLOR = "#94e2d5"
_WRAP_COLOR = "#cba6f7"
_WRAP_BACK_COLOR = "#89dceb"
_BACKEND_COLOR = "#fab387"
_BACK_START_PREVIEW_COLOR = "#f5c2e7"
_BACK_START_CONFIRMED_COLOR = "#cba6f7"
_CTRL_POINT_COLOR = "#f9e2af"
_CTRL_SELECTED_COLOR = "#f38ba8"
_DISP_SLIDER_STEPS = 1000
_DEFAULT_BACK_START_GRID = 10
_MIN_BACK_START_GRID = 2
_MAX_BACK_START_GRID = 316  # 316² ≈ 99,856 ≤ 100,000
_MAX_BACK_START_POINTS = 100_000
_CTRL_PICK_PIXEL_RADIUS = 28.0


@dataclass
class BackendSnapshot:
    """Frozen curved Backend plane used as the Wrap Back target surface."""

    mesh: object
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    half_extent: float
    ctrl_uv: np.ndarray
    ctrl_disp: np.ndarray


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

    def copy_from(self, other: PlanePose) -> None:
        self.x = other.x
        self.y = other.y
        self.z = other.z
        self.rx = other.rx
        self.ry = other.ry
        self.rz = other.rz

    def clear_confirmed(self) -> None:
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

    def get_axis(self, axis: AxisName) -> float | None:
        return getattr(self, axis)

    def set_axis(self, axis: AxisName, value: float) -> None:
        setattr(self, axis, value)


@dataclass
class PlanePanel:
    """UI + state for one editable oriented plane (start or end)."""

    title: str
    preview_color: str
    confirmed_color: str
    preview: PlanePose = field(default_factory=PlanePose)
    confirmed: PlanePose = field(default_factory=PlanePose)

    group: QGroupBox | None = None
    lbl_status: QLabel | None = None
    lbl_x: QLabel | None = None
    slider_x: QSlider | None = None
    spin_x: QDoubleSpinBox | None = None
    lbl_y: QLabel | None = None
    slider_y: QSlider | None = None
    spin_y: QDoubleSpinBox | None = None
    lbl_z: QLabel | None = None
    slider_z: QSlider | None = None
    spin_z: QDoubleSpinBox | None = None
    lbl_rx: QLabel | None = None
    slider_rx: QSlider | None = None
    lbl_ry: QLabel | None = None
    slider_ry: QSlider | None = None
    lbl_rz: QLabel | None = None
    slider_rz: QSlider | None = None
    btn_confirm: QPushButton | None = None

    def slider_for(self, axis: AxisName) -> QSlider | None:
        return getattr(self, f"slider_{axis}")

    def spin_for(self, axis: AxisName) -> QDoubleSpinBox | None:
        return getattr(self, f"spin_{axis}")

    def label_for(self, axis: AxisName) -> QLabel | None:
        return getattr(self, f"lbl_{axis}")


class MoldWrapperMainWindow(QMainWindow):
    """STL viewer with start/end oriented planes and wrap export."""

    def __init__(self) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._slider_sync = False
        self._wrap_mesh = None
        self._wrap_back_mesh = None
        self._copied_end_mesh = None
        self._backend_snapshot: BackendSnapshot | None = None
        self._end_ctrl_uv = place_equidistant_uv(DEFAULT_CONTROL_POINTS)
        self._end_ctrl_disp = np.zeros(len(self._end_ctrl_uv), dtype=float)
        self._end_ctrl_selected = 0
        self._end_ctrl_world_points: np.ndarray | None = None
        self._deform_sync = False
        self._back_start_grid_n = _DEFAULT_BACK_START_GRID

        self._start = PlanePanel(
            title="Start Plane",
            preview_color=_START_PREVIEW_COLOR,
            confirmed_color=_START_CONFIRMED_COLOR,
        )
        self._end = PlanePanel(
            title="End Plane",
            preview_color=_END_PREVIEW_COLOR,
            confirmed_color=_END_CONFIRMED_COLOR,
        )
        self._back_start = PlanePanel(
            title="Back-start Plane",
            preview_color=_BACK_START_PREVIEW_COLOR,
            confirmed_color=_BACK_START_CONFIRMED_COLOR,
        )

        self.setWindowTitle("3D Mold Wrapper")
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

        title = QLabel("3D Mold Wrapper")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        subtitle = QLabel("STL Viewer & Start/End Plane Wrapping")
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

        layout.addWidget(self._build_plane_group(self._start))
        layout.addWidget(self._build_start_plane_points_group())
        layout.addWidget(self._build_plane_group(self._end))
        layout.addWidget(self._build_end_deform_group())

        backend_group = QGroupBox("Backend Plane")
        backend_layout = QVBoxLayout(backend_group)
        backend_info = QLabel(
            "Snapshot the end plane in its current curved state at the same "
            "position. Shown in a different color from the live end plane."
        )
        backend_info.setWordWrap(True)
        backend_info.setObjectName("fileLabel")
        backend_layout.addWidget(backend_info)

        self.btn_copy_backend = QPushButton("Copy Backend Plane")
        self.btn_copy_backend.setEnabled(False)
        self.btn_copy_backend.clicked.connect(self._copy_curved_end_plane)
        backend_layout.addWidget(self.btn_copy_backend)

        self.chk_show_backend = QCheckBox("Show backend plane")
        self.chk_show_backend.setChecked(True)
        self.chk_show_backend.setEnabled(False)
        self.chk_show_backend.toggled.connect(self._on_show_copied_end_toggled)
        backend_layout.addWidget(self.chk_show_backend)

        layout.addWidget(backend_group)
        layout.addWidget(self._build_plane_group(self._back_start))

        wrap_group = QGroupBox("Wrap")
        wrap_layout = QVBoxLayout(wrap_group)
        wrap_info = QLabel(
            "Extrude the whole start plane toward the end plane as a solid. "
            "Each sample stops on the model surface, or at the end plane if it "
            "misses the body."
        )
        wrap_info.setWordWrap(True)
        wrap_info.setObjectName("fileLabel")
        wrap_layout.addWidget(wrap_info)

        self.btn_wrap_plane = QPushButton("Wrap Plane")
        self.btn_wrap_plane.setEnabled(False)
        self.btn_wrap_plane.clicked.connect(self._wrap_plane)
        wrap_layout.addWidget(self.btn_wrap_plane)

        self.btn_wrap_back = QPushButton("Wrap Back")
        self.btn_wrap_back.setEnabled(False)
        self.btn_wrap_back.clicked.connect(self._wrap_back)
        wrap_layout.addWidget(self.btn_wrap_back)

        layout.addWidget(wrap_group)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _build_start_plane_points_group(self) -> QGroupBox:
        group = QGroupBox("Start Plane Points")
        layout = QVBoxLayout(group)

        info = QLabel(
            "Set an N × N sampling grid on the Start plane for Wrap Plane."
        )
        info.setWordWrap(True)
        info.setObjectName("fileLabel")
        layout.addWidget(info)

        density_row = QHBoxLayout()
        density_label = QLabel("Point density (per side)")
        density_label.setWordWrap(True)
        density_row.addWidget(density_label, stretch=1)
        self.spin_point_density = QSpinBox()
        self.spin_point_density.setRange(MIN_POINT_DENSITY, MAX_POINT_DENSITY)
        self.spin_point_density.setSingleStep(10)
        self.spin_point_density.setValue(DEFAULT_POINT_DENSITY)
        self.spin_point_density.setMinimumWidth(88)
        self.spin_point_density.setEnabled(False)
        self.spin_point_density.valueChanged.connect(self._update_density_label)
        density_row.addWidget(self.spin_point_density)
        layout.addLayout(density_row)

        self.lbl_point_density = QLabel("")
        self.lbl_point_density.setObjectName("fileLabel")
        self.lbl_point_density.setWordWrap(True)
        layout.addWidget(self.lbl_point_density)
        self._update_density_label()

        return group

    def _append_back_start_points_controls(self, layout: QVBoxLayout) -> None:
        points_info = QLabel(
            "N × N sampling grid for Wrap Back "
            f"(invisible markers, up to {_MAX_BACK_START_POINTS:,} points)."
        )
        points_info.setWordWrap(True)
        points_info.setObjectName("fileLabel")
        layout.addWidget(points_info)

        density_row = QHBoxLayout()
        density_label = QLabel("Grid density (N per side)")
        density_label.setWordWrap(True)
        density_row.addWidget(density_label, stretch=1)
        self.spin_back_start_grid = QSpinBox()
        self.spin_back_start_grid.setRange(_MIN_BACK_START_GRID, _MAX_BACK_START_GRID)
        self.spin_back_start_grid.setValue(_DEFAULT_BACK_START_GRID)
        self.spin_back_start_grid.setMinimumWidth(88)
        self.spin_back_start_grid.setEnabled(False)
        self.spin_back_start_grid.valueChanged.connect(self._on_back_start_grid_changed)
        density_row.addWidget(self.spin_back_start_grid)
        layout.addLayout(density_row)

        self.lbl_back_start_grid = QLabel("")
        self.lbl_back_start_grid.setObjectName("fileLabel")
        self.lbl_back_start_grid.setWordWrap(True)
        layout.addWidget(self.lbl_back_start_grid)
        self._update_back_start_grid_label()

    def _add_axis_position_controls(
        self,
        plane: PlanePanel,
        layout: QVBoxLayout,
        axis: AxisName,
    ) -> None:
        axis_upper = axis.upper()
        lbl = QLabel(f"{axis_upper}: —")
        lbl.setObjectName("fileLabel")
        setattr(plane, f"lbl_{axis}", lbl)
        layout.addWidget(lbl)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, _SLIDER_STEPS)
        slider.setEnabled(False)
        slider.valueChanged.connect(
            lambda value, p=plane, a=axis: self._on_plane_axis_slider_changed(p, a, value)
        )
        setattr(plane, f"slider_{axis}", slider)
        layout.addWidget(slider)

        row = QHBoxLayout()
        exact_label = QLabel(f"Exact {axis_upper} (mm)")
        exact_label.setWordWrap(True)
        row.addWidget(exact_label, stretch=1)
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setSuffix(" mm")
        spin.setMinimumWidth(88)
        spin.setEnabled(False)
        spin.valueChanged.connect(
            lambda value, p=plane, a=axis: self._on_plane_axis_spin_changed(p, a, value)
        )
        setattr(plane, f"spin_{axis}", spin)
        row.addWidget(spin)
        layout.addLayout(row)

    def _build_plane_group(self, plane: PlanePanel) -> QGroupBox:
        group = QGroupBox(plane.title)
        plane.group = group
        layout = QVBoxLayout(group)

        if plane is self._end:
            self.btn_match_end_to_start = QPushButton("Match End to Start")
            self.btn_match_end_to_start.setEnabled(False)
            self.btn_match_end_to_start.clicked.connect(self._match_end_to_start)
            layout.addWidget(self.btn_match_end_to_start)
        elif plane is self._back_start:
            self.btn_match_back_start_to_backend = QPushButton(
                "Apply Backend Position to Back-start"
            )
            self.btn_match_back_start_to_backend.setEnabled(False)
            self.btn_match_back_start_to_backend.clicked.connect(
                self._match_back_start_to_backend
            )
            layout.addWidget(self.btn_match_back_start_to_backend)

        info = QLabel(
            f"Set {plane.title.lower()} position (X / Y / Z) and tilt "
            "(rotate X / Y / Z). Slide or pick on the model, then confirm."
        )
        info.setWordWrap(True)
        info.setObjectName("fileLabel")
        layout.addWidget(info)

        plane.lbl_status = QLabel(f"{plane.title}: not set")
        plane.lbl_status.setObjectName("fileLabel")
        plane.lbl_status.setWordWrap(True)
        layout.addWidget(plane.lbl_status)

        for axis in ("x", "y", "z"):
            self._add_axis_position_controls(plane, layout, axis)  # type: ignore[arg-type]

        plane.lbl_rx = QLabel("Rotate X: 0°")
        plane.lbl_rx.setObjectName("fileLabel")
        layout.addWidget(plane.lbl_rx)
        plane.slider_rx = QSlider(Qt.Orientation.Horizontal)
        plane.slider_rx.setRange(_ROTATION_DEG_MIN, _ROTATION_DEG_MAX)
        plane.slider_rx.setValue(0)
        plane.slider_rx.setEnabled(False)
        plane.slider_rx.valueChanged.connect(
            lambda value, p=plane: self._on_plane_rx_changed(p, value)
        )
        layout.addWidget(plane.slider_rx)

        plane.lbl_ry = QLabel("Rotate Y: 0°")
        plane.lbl_ry.setObjectName("fileLabel")
        layout.addWidget(plane.lbl_ry)
        plane.slider_ry = QSlider(Qt.Orientation.Horizontal)
        plane.slider_ry.setRange(_ROTATION_DEG_MIN, _ROTATION_DEG_MAX)
        plane.slider_ry.setValue(0)
        plane.slider_ry.setEnabled(False)
        plane.slider_ry.valueChanged.connect(
            lambda value, p=plane: self._on_plane_ry_changed(p, value)
        )
        layout.addWidget(plane.slider_ry)

        plane.lbl_rz = QLabel("Rotate Z: 0°")
        plane.lbl_rz.setObjectName("fileLabel")
        layout.addWidget(plane.lbl_rz)
        plane.slider_rz = QSlider(Qt.Orientation.Horizontal)
        plane.slider_rz.setRange(_ROTATION_DEG_MIN, _ROTATION_DEG_MAX)
        plane.slider_rz.setValue(0)
        plane.slider_rz.setEnabled(False)
        plane.slider_rz.valueChanged.connect(
            lambda value, p=plane: self._on_plane_rz_changed(p, value)
        )
        layout.addWidget(plane.slider_rz)

        plane.btn_confirm = QPushButton(f"Confirm {plane.title}")
        plane.btn_confirm.setEnabled(False)
        plane.btn_confirm.clicked.connect(lambda _=False, p=plane: self._confirm_plane(p))
        layout.addWidget(plane.btn_confirm)

        if plane is self._back_start:
            self._append_back_start_points_controls(layout)

        return group

    def _build_end_deform_group(self) -> QGroupBox:
        group = QGroupBox("End Plane Deform")
        layout = QVBoxLayout(group)

        info = QLabel(
            "Place N equidistant control points on the end plane. Select a point "
            "from the list or click it in the 3D view, then pull/push it along "
            "the plane normal — the plane deforms with it."
        )
        info.setWordWrap(True)
        info.setObjectName("fileLabel")
        layout.addWidget(info)

        count_row = QHBoxLayout()
        count_label = QLabel("Control points (N)")
        count_label.setWordWrap(True)
        count_row.addWidget(count_label, stretch=1)
        self.spin_end_ctrl_count = QSpinBox()
        self.spin_end_ctrl_count.setRange(MIN_CONTROL_POINTS, MAX_CONTROL_POINTS)
        self.spin_end_ctrl_count.setValue(DEFAULT_CONTROL_POINTS)
        self.spin_end_ctrl_count.setMinimumWidth(88)
        self.spin_end_ctrl_count.setEnabled(False)
        self.spin_end_ctrl_count.valueChanged.connect(self._on_end_ctrl_count_changed)
        count_row.addWidget(self.spin_end_ctrl_count)
        layout.addLayout(count_row)

        select_row = QHBoxLayout()
        select_label = QLabel("Selected point")
        select_label.setWordWrap(True)
        select_row.addWidget(select_label, stretch=1)
        self.combo_end_ctrl_point = QComboBox()
        self.combo_end_ctrl_point.setEnabled(False)
        self.combo_end_ctrl_point.currentIndexChanged.connect(self._on_end_ctrl_selected)
        select_row.addWidget(self.combo_end_ctrl_point, stretch=2)
        layout.addLayout(select_row)

        self.lbl_end_ctrl_disp = QLabel("Pull / push: 0.000 mm")
        self.lbl_end_ctrl_disp.setObjectName("fileLabel")
        layout.addWidget(self.lbl_end_ctrl_disp)

        self.slider_end_ctrl_disp = QSlider(Qt.Orientation.Horizontal)
        self.slider_end_ctrl_disp.setRange(0, _DISP_SLIDER_STEPS)
        self.slider_end_ctrl_disp.setValue(_DISP_SLIDER_STEPS // 2)
        self.slider_end_ctrl_disp.setEnabled(False)
        self.slider_end_ctrl_disp.valueChanged.connect(self._on_end_ctrl_disp_slider)
        layout.addWidget(self.slider_end_ctrl_disp)

        disp_row = QHBoxLayout()
        disp_label = QLabel("Exact pull / push (mm)")
        disp_label.setWordWrap(True)
        disp_row.addWidget(disp_label, stretch=1)
        self.spin_end_ctrl_disp = QDoubleSpinBox()
        self.spin_end_ctrl_disp.setDecimals(3)
        self.spin_end_ctrl_disp.setSingleStep(0.5)
        self.spin_end_ctrl_disp.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_end_ctrl_disp.setSuffix(" mm")
        self.spin_end_ctrl_disp.setMinimumWidth(88)
        self.spin_end_ctrl_disp.setEnabled(False)
        self.spin_end_ctrl_disp.valueChanged.connect(self._on_end_ctrl_disp_spin)
        disp_row.addWidget(self.spin_end_ctrl_disp)
        layout.addLayout(disp_row)

        self.btn_reset_end_deform = QPushButton("Reset Deform")
        self.btn_reset_end_deform.setEnabled(False)
        self.btn_reset_end_deform.clicked.connect(self._reset_end_deform)
        layout.addWidget(self.btn_reset_end_deform)

        self._rebuild_end_ctrl_combo()
        return group

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

    def _update_density_label(self, _value: int | None = None) -> None:
        density = self.spin_point_density.value()
        total = density * density
        self.lbl_point_density.setText(
            f"Plane samples: {density} × {density} = {total:,} points"
        )

    def _match_end_to_start(self) -> None:
        """Copy start plane position and rotation onto the end plane."""
        if self.viewer.mesh is None or not self._start.preview.is_complete:
            return

        source = self._start.preview
        self._apply_pose_to_plane_controls(self._end, source)
        # Clear confirmed end so the user re-confirms after aligning.
        self._end.confirmed.clear_confirmed()
        self._reset_end_deform()
        self._update_plane_status_label(self._end)
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(
            "End plane matched to start plane "
            f"(X = {source.x:.3f}, Y = {source.y:.3f}, Z = {source.z:.3f} mm)"
        )

    def _apply_pose_to_plane_controls(self, plane: PlanePanel, pose: PlanePose) -> None:
        assert pose.is_complete
        assert plane.slider_rx is not None
        assert plane.slider_ry is not None
        assert plane.slider_rz is not None
        self._slider_sync = True
        for axis in ("x", "y", "z"):
            value = pose.get_axis(axis)  # type: ignore[arg-type]
            assert value is not None
            spin = plane.spin_for(axis)  # type: ignore[arg-type]
            assert spin is not None
            spin.setValue(value)
            self._set_slider_from_axis(plane, axis, value)  # type: ignore[arg-type]
        plane.slider_rx.setValue(int(round(pose.rx)))
        plane.slider_ry.setValue(int(round(pose.ry)))
        plane.slider_rz.setValue(int(round(pose.rz)))
        self._slider_sync = False
        plane.preview.copy_from(pose)
        self._update_plane_position_labels(plane)
        self._update_plane_rotation_labels(plane)

    def _planes_ready(self) -> bool:
        return self._start.confirmed.is_complete and self._end.confirmed.is_complete

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_open.setEnabled(enabled)
        has_mesh = self.viewer.mesh is not None
        self.btn_match_end_to_start.setEnabled(enabled and has_mesh)
        self.spin_point_density.setEnabled(enabled and has_mesh)
        self.btn_wrap_plane.setEnabled(enabled and has_mesh and self._planes_ready())
        self.btn_wrap_back.setEnabled(
            enabled
            and has_mesh
            and self._backend_snapshot is not None
            and self._back_start_pose() is not None
        )
        deform_enabled = enabled and has_mesh and self._end.preview.is_complete
        self.spin_end_ctrl_count.setEnabled(deform_enabled)
        self.combo_end_ctrl_point.setEnabled(deform_enabled)
        self.slider_end_ctrl_disp.setEnabled(deform_enabled)
        self.spin_end_ctrl_disp.setEnabled(deform_enabled)
        self.btn_reset_end_deform.setEnabled(deform_enabled)
        can_copy_end = enabled and has_mesh and self._end_pose_for_deform() is not None
        self.btn_copy_backend.setEnabled(can_copy_end)
        self.chk_show_backend.setEnabled(
            enabled and self._copied_end_mesh is not None
        )
        back_start_ready = (
            enabled and has_mesh and self._back_start_pose() is not None
        )
        self.spin_back_start_grid.setEnabled(back_start_ready)
        self.btn_match_back_start_to_backend.setEnabled(
            enabled and has_mesh and self._backend_snapshot is not None
        )
        for plane in (self._start, self._end, self._back_start):
            self._set_plane_controls_enabled(plane, enabled and has_mesh)

    def _set_plane_controls_enabled(self, plane: PlanePanel, enabled: bool) -> None:
        for axis in ("x", "y", "z"):
            slider = plane.slider_for(axis)  # type: ignore[arg-type]
            spin = plane.spin_for(axis)  # type: ignore[arg-type]
            assert slider is not None and spin is not None
            slider.setEnabled(enabled)
            spin.setEnabled(enabled)
        assert plane.slider_rx is not None
        assert plane.slider_ry is not None
        assert plane.slider_rz is not None
        assert plane.btn_confirm is not None
        plane.slider_rx.setEnabled(enabled)
        plane.slider_ry.setEnabled(enabled)
        plane.slider_rz.setEnabled(enabled)
        plane.btn_confirm.setEnabled(enabled)

    def _update_size_label(self, mesh=None) -> None:
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
        self._reset_all_planes()
        self._wrap_mesh = None
        self._wrap_back_mesh = None
        self._copied_end_mesh = None
        self._backend_snapshot = None
        self._reset_end_deform()
        self._init_plane_controls(mesh)
        self._set_controls_enabled(True)
        self.footer.show_message(
            f"Loaded {path.name} — {mesh.n_cells:,} triangles"
        )

    def _reset_all_planes(self) -> None:
        self.viewer.set_point_pick_callback(None)
        self.viewer.clear_overlays()
        for plane in (self._start, self._end, self._back_start):
            plane.confirmed.clear_confirmed()
            plane.preview = PlanePose()
            self._slider_sync = True
            assert plane.slider_rx is not None
            assert plane.slider_ry is not None
            assert plane.slider_rz is not None
            plane.slider_rx.setValue(0)
            plane.slider_ry.setValue(0)
            plane.slider_rz.setValue(0)
            self._slider_sync = False
            self._update_plane_status_label(plane)
            self._update_plane_position_labels(plane)
            self._update_plane_rotation_labels(plane)

    def _model_movement_pad(self, mesh) -> float:
        """Extra travel beyond model bounds: at least half the model width."""
        x_min, x_max = mesh_x_bounds(mesh)
        y_min, y_max = mesh_y_bounds(mesh)
        z_min, z_max = mesh_z_bounds(mesh)
        width = max(x_max - x_min, 1e-3)
        length = max(y_max - y_min, 1e-3)
        height = max(z_max - z_min, 1e-3)
        return 0.5 * max(width, length, height)

    def _axis_bounds(self, mesh, axis: AxisName) -> tuple[float, float]:
        if axis == "x":
            lo, hi = mesh_x_bounds(mesh)
        elif axis == "y":
            lo, hi = mesh_y_bounds(mesh)
        else:
            lo, hi = mesh_z_bounds(mesh)
        if hi <= lo:
            hi = lo + 1.0
        pad = self._model_movement_pad(mesh)
        return lo - pad, hi + pad

    def _init_plane_controls(self, mesh) -> None:
        x_min, x_max = self._axis_bounds(mesh, "x")
        y_min, y_max = self._axis_bounds(mesh, "y")
        z_min, z_max = self._axis_bounds(mesh, "z")

        cx, cy = mesh_center_xy(mesh)
        # Default Z still uses the model body range (without pad) for a sensible start.
        body_z_min, body_z_max = mesh_z_bounds(mesh)
        if body_z_max <= body_z_min:
            body_z_max = body_z_min + 1.0
        start_z = body_z_min + (body_z_max - body_z_min) * 0.25
        end_z = body_z_min + (body_z_max - body_z_min) * 0.75
        back_start_z = body_z_min + (body_z_max - body_z_min) * 0.5

        for plane, z in (
            (self._start, start_z),
            (self._end, end_z),
            (self._back_start, back_start_z),
        ):
            assert plane.slider_rx is not None
            assert plane.slider_ry is not None
            assert plane.slider_rz is not None
            self._slider_sync = True
            for axis, lo, hi, value in (
                ("x", x_min, x_max, cx),
                ("y", y_min, y_max, cy),
                ("z", z_min, z_max, z),
            ):
                spin = plane.spin_for(axis)  # type: ignore[arg-type]
                assert spin is not None
                spin.setRange(lo, hi)
                spin.setValue(value)
                self._set_slider_from_axis(plane, axis, value)  # type: ignore[arg-type]
            plane.slider_rx.setValue(0)
            plane.slider_ry.setValue(0)
            plane.slider_rz.setValue(0)
            self._slider_sync = False
            plane.preview.reset_preview(cx, cy, z)
            self._update_plane_position_labels(plane)
            self._update_plane_rotation_labels(plane)
            self._update_plane_status_label(plane)

        self._refresh_plane_overlays()

    def _value_from_slider(self, plane: PlanePanel, axis: AxisName) -> float:
        mesh = self.viewer.mesh
        slider = plane.slider_for(axis)
        if mesh is None or slider is None:
            return 0.0
        lo, hi = self._axis_bounds(mesh, axis)
        if hi <= lo:
            return lo
        fraction = slider.value() / _SLIDER_STEPS
        return lo + fraction * (hi - lo)

    def _set_slider_from_axis(
        self,
        plane: PlanePanel,
        axis: AxisName,
        value: float,
    ) -> None:
        mesh = self.viewer.mesh
        slider = plane.slider_for(axis)
        if mesh is None or slider is None:
            return
        lo, hi = self._axis_bounds(mesh, axis)
        if hi <= lo:
            slider.setValue(0)
            return
        clamped = min(max(value, lo), hi)
        fraction = (clamped - lo) / (hi - lo)
        slider.setValue(int(round(fraction * _SLIDER_STEPS)))

    def _on_plane_axis_slider_changed(
        self,
        plane: PlanePanel,
        axis: AxisName,
        _value: int,
    ) -> None:
        spin = plane.spin_for(axis)
        if self._slider_sync or spin is None:
            return
        value = self._value_from_slider(plane, axis)
        plane.preview.set_axis(axis, value)
        self._slider_sync = True
        spin.setValue(value)
        self._slider_sync = False
        self._update_plane_position_labels(plane)
        self._refresh_plane_overlays()

    def _on_plane_axis_spin_changed(
        self,
        plane: PlanePanel,
        axis: AxisName,
        value: float,
    ) -> None:
        if self._slider_sync:
            return
        plane.preview.set_axis(axis, value)
        self._slider_sync = True
        self._set_slider_from_axis(plane, axis, value)
        self._slider_sync = False
        self._update_plane_position_labels(plane)
        self._refresh_plane_overlays()

    def _update_plane_position_labels(self, plane: PlanePanel) -> None:
        for axis in ("x", "y", "z"):
            lbl = plane.label_for(axis)  # type: ignore[arg-type]
            assert lbl is not None
            value = plane.preview.get_axis(axis)  # type: ignore[arg-type]
            axis_upper = axis.upper()
            if value is None:
                lbl.setText(f"{axis_upper}: —")
            else:
                lbl.setText(f"Preview {axis_upper}: {value:.3f} mm")

    def _update_plane_rotation_labels(self, plane: PlanePanel) -> None:
        assert plane.lbl_rx is not None
        assert plane.lbl_ry is not None
        assert plane.lbl_rz is not None
        plane.lbl_rx.setText(f"Rotate X: {plane.preview.rx:.0f}°")
        plane.lbl_ry.setText(f"Rotate Y: {plane.preview.ry:.0f}°")
        plane.lbl_rz.setText(f"Rotate Z: {plane.preview.rz:.0f}°")

    def _on_plane_rx_changed(self, plane: PlanePanel, value: int) -> None:
        if self._slider_sync:
            return
        plane.preview.rx = float(value)
        self._update_plane_rotation_labels(plane)
        self._refresh_plane_overlays()

    def _on_plane_ry_changed(self, plane: PlanePanel, value: int) -> None:
        if self._slider_sync:
            return
        plane.preview.ry = float(value)
        self._update_plane_rotation_labels(plane)
        self._refresh_plane_overlays()

    def _on_plane_rz_changed(self, plane: PlanePanel, value: int) -> None:
        if self._slider_sync:
            return
        plane.preview.rz = float(value)
        self._update_plane_rotation_labels(plane)
        self._refresh_plane_overlays()

    def _oriented_plane_mesh(self, mesh, pose: PlanePose):
        assert pose.is_complete
        return make_oriented_plane_mesh(
            mesh,
            pose.z,  # type: ignore[arg-type]
            center_x=pose.x,
            center_y=pose.y,
            rotate_x_deg=pose.rx,
            rotate_y_deg=pose.ry,
            rotate_z_deg=pose.rz,
        )

    def _end_pose_for_deform(self) -> PlanePose | None:
        if self._end.preview.is_complete:
            return self._end.preview
        if self._end.confirmed.is_complete:
            return self._end.confirmed
        return None

    def _end_disp_limit(self) -> float:
        mesh = self.viewer.mesh
        if mesh is None:
            return 50.0
        return max(plane_half_extent(mesh), 1.0)

    def _rebuild_end_ctrl_combo(self) -> None:
        self.combo_end_ctrl_point.blockSignals(True)
        self.combo_end_ctrl_point.clear()
        for i in range(len(self._end_ctrl_uv)):
            self.combo_end_ctrl_point.addItem(f"Point {i + 1}")
        selected = min(max(self._end_ctrl_selected, 0), max(len(self._end_ctrl_uv) - 1, 0))
        self._end_ctrl_selected = selected
        if len(self._end_ctrl_uv) > 0:
            self.combo_end_ctrl_point.setCurrentIndex(selected)
        self.combo_end_ctrl_point.blockSignals(False)

    def _reset_end_deform(self) -> None:
        n = self.spin_end_ctrl_count.value() if hasattr(self, "spin_end_ctrl_count") else DEFAULT_CONTROL_POINTS
        self._end_ctrl_uv = place_equidistant_uv(n)
        self._end_ctrl_disp = np.zeros(len(self._end_ctrl_uv), dtype=float)
        self._end_ctrl_selected = 0
        if hasattr(self, "combo_end_ctrl_point"):
            self._rebuild_end_ctrl_combo()
            self._sync_end_disp_controls_from_selected()
        self._refresh_plane_overlays()

    def _on_end_ctrl_count_changed(self, value: int) -> None:
        self._end_ctrl_uv = place_equidistant_uv(int(value))
        self._end_ctrl_disp = np.zeros(len(self._end_ctrl_uv), dtype=float)
        self._end_ctrl_selected = 0
        self._rebuild_end_ctrl_combo()
        self._sync_end_disp_controls_from_selected()
        self._refresh_plane_overlays()
        self.footer.show_message(f"End plane control points: {int(value)}")

    def _on_end_ctrl_selected(self, index: int) -> None:
        if index < 0 or index >= len(self._end_ctrl_disp):
            return
        self._end_ctrl_selected = int(index)
        self._sync_end_disp_controls_from_selected()
        self._refresh_plane_overlays()

    def _sync_end_disp_controls_from_selected(self) -> None:
        if len(self._end_ctrl_disp) == 0:
            return
        idx = min(max(self._end_ctrl_selected, 0), len(self._end_ctrl_disp) - 1)
        value = float(self._end_ctrl_disp[idx])
        limit = self._end_disp_limit()
        self._deform_sync = True
        self.spin_end_ctrl_disp.setRange(-limit, limit)
        self.spin_end_ctrl_disp.setValue(value)
        fraction = (value + limit) / (2.0 * limit) if limit > 0 else 0.5
        self.slider_end_ctrl_disp.setValue(int(round(fraction * _DISP_SLIDER_STEPS)))
        self._deform_sync = False
        self.lbl_end_ctrl_disp.setText(f"Pull / push: {value:.3f} mm")

    def _set_selected_end_disp(self, value: float) -> None:
        if len(self._end_ctrl_disp) == 0:
            return
        idx = min(max(self._end_ctrl_selected, 0), len(self._end_ctrl_disp) - 1)
        limit = self._end_disp_limit()
        clamped = float(min(max(value, -limit), limit))
        self._end_ctrl_disp[idx] = clamped
        self.lbl_end_ctrl_disp.setText(f"Pull / push: {clamped:.3f} mm")
        self._refresh_plane_overlays()

    def _on_end_ctrl_disp_slider(self, slider_value: int) -> None:
        if self._deform_sync:
            return
        limit = self._end_disp_limit()
        fraction = slider_value / _DISP_SLIDER_STEPS
        value = -limit + fraction * (2.0 * limit)
        self._deform_sync = True
        self.spin_end_ctrl_disp.setValue(value)
        self._deform_sync = False
        self._set_selected_end_disp(value)

    def _on_end_ctrl_disp_spin(self, value: float) -> None:
        if self._deform_sync:
            return
        limit = self._end_disp_limit()
        fraction = (value + limit) / (2.0 * limit) if limit > 0 else 0.5
        self._deform_sync = True
        self.slider_end_ctrl_disp.setValue(int(round(fraction * _DISP_SLIDER_STEPS)))
        self._deform_sync = False
        self._set_selected_end_disp(value)

    def _append_plane_overlay(
        self,
        overlays: list[OverlayMesh],
        mesh,
        pose: PlanePose,
        color: str,
        opacity: float,
        edge_color: str,
        *,
        deformed: bool = False,
    ) -> None:
        if not pose.is_complete:
            return
        if deformed:
            center, normal = pose_frame(
                x=float(pose.x),
                y=float(pose.y),
                z=float(pose.z),
                rx=pose.rx,
                ry=pose.ry,
                rz=pose.rz,
            )
            plane_mesh = make_deformed_plane_mesh(
                center=center,
                normal=normal,
                half_extent=plane_half_extent(mesh),
                uv=self._end_ctrl_uv,
                displacements=self._end_ctrl_disp,
            )
        else:
            plane_mesh = self._oriented_plane_mesh(mesh, pose)
        overlays.append(
            OverlayMesh(
                mesh=plane_mesh,
                color=color,
                opacity=opacity,
                style="surface",
                show_edges=True,
                edge_color=edge_color,
            )
        )

    def _append_end_control_overlays(self, overlays: list[OverlayMesh], mesh) -> None:
        pose = self._end_pose_for_deform()
        if pose is None or len(self._end_ctrl_uv) == 0:
            self._end_ctrl_world_points = None
            return
        center, normal = pose_frame(
            x=float(pose.x),
            y=float(pose.y),
            z=float(pose.z),
            rx=pose.rx,
            ry=pose.ry,
            rz=pose.rz,
        )
        half = plane_half_extent(mesh)
        points = control_points_world(
            center=center,
            normal=normal,
            half_extent=half,
            uv=self._end_ctrl_uv,
            displacements=self._end_ctrl_disp,
        )
        self._end_ctrl_world_points = np.asarray(points, dtype=float)
        radius = max(half * 0.03, 0.5)
        others, selected = control_point_glyphs(
            points,
            selected_index=self._end_ctrl_selected,
            radius=radius,
        )
        if others is not None:
            overlays.append(
                OverlayMesh(
                    mesh=others,
                    color=_CTRL_POINT_COLOR,
                    opacity=0.95,
                    style="surface",
                    show_edges=False,
                    edge_color=_CTRL_POINT_COLOR,
                )
            )
        if selected is not None:
            overlays.append(
                OverlayMesh(
                    mesh=selected,
                    color=_CTRL_SELECTED_COLOR,
                    opacity=1.0,
                    style="surface",
                    show_edges=False,
                    edge_color=_CTRL_SELECTED_COLOR,
                )
            )

    def _refresh_plane_overlays(self) -> None:
        mesh = self.viewer.mesh
        if mesh is None:
            self._end_ctrl_world_points = None
            self.viewer.clear_overlays()
            self._sync_end_ctrl_click_picking()
            return

        overlays: list[OverlayMesh] = []
        # Start plane (flat).
        self._append_plane_overlay(
            overlays,
            mesh,
            self._start.confirmed,
            self._start.confirmed_color,
            0.32,
            "#74c7ec",
        )
        self._append_plane_overlay(
            overlays,
            mesh,
            self._start.preview,
            self._start.preview_color,
            0.42,
            OVERLAY_COLOR,
        )
        # End plane (deformed by control points).
        self._append_plane_overlay(
            overlays,
            mesh,
            self._end.confirmed,
            self._end.confirmed_color,
            0.32,
            "#74c7ec",
            deformed=True,
        )
        self._append_plane_overlay(
            overlays,
            mesh,
            self._end.preview,
            self._end.preview_color,
            0.42,
            OVERLAY_COLOR,
            deformed=True,
        )
        self._append_end_control_overlays(overlays, mesh)
        if (
            self._copied_end_mesh is not None
            and self.chk_show_backend.isChecked()
        ):
            overlays.append(
                OverlayMesh(
                    mesh=self._copied_end_mesh,
                    color=_BACKEND_COLOR,
                    opacity=0.55,
                    style="surface",
                    show_edges=True,
                    edge_color="#fe640b",
                )
            )
        # Back-start plane (flat, movable).
        self._append_plane_overlay(
            overlays,
            mesh,
            self._back_start.confirmed,
            self._back_start.confirmed_color,
            0.32,
            "#b4befe",
        )
        self._append_plane_overlay(
            overlays,
            mesh,
            self._back_start.preview,
            self._back_start.preview_color,
            0.42,
            OVERLAY_COLOR,
        )
        if self._wrap_mesh is not None:
            overlays.append(
                OverlayMesh(
                    mesh=self._wrap_mesh,
                    color=_WRAP_COLOR,
                    opacity=0.85,
                    style="surface",
                    show_edges=True,
                    edge_color="#b4befe",
                )
            )
        if self._wrap_back_mesh is not None:
            overlays.append(
                OverlayMesh(
                    mesh=self._wrap_back_mesh,
                    color=_WRAP_BACK_COLOR,
                    opacity=0.85,
                    style="surface",
                    show_edges=True,
                    edge_color="#74c7ec",
                )
            )
        self.viewer.set_overlays(overlays)
        self._sync_end_ctrl_click_picking()

    def _sync_end_ctrl_click_picking(self) -> None:
        """Enable click-to-select for end control points."""
        can_pick = (
            self._end_pose_for_deform() is not None
            and self._end_ctrl_world_points is not None
            and len(self._end_ctrl_world_points) > 0
        )
        self.viewer.set_display_click_callback(
            self._on_end_ctrl_display_click if can_pick else None
        )

    def _on_end_ctrl_display_click(self, display_x: float, display_y: float) -> None:
        points = self._end_ctrl_world_points
        if points is None or len(points) == 0:
            return

        best_index = -1
        best_dist = float("inf")
        for index, point in enumerate(points):
            dx, dy = self.viewer.world_to_display(point)
            dist = float(np.hypot(dx - display_x, dy - display_y))
            if dist < best_dist:
                best_dist = dist
                best_index = index

        if best_index < 0 or best_dist > _CTRL_PICK_PIXEL_RADIUS:
            return

        self._select_end_ctrl_point(best_index)

    def _select_end_ctrl_point(self, index: int) -> None:
        if index < 0 or index >= len(self._end_ctrl_disp):
            return
        if self.combo_end_ctrl_point.currentIndex() != index:
            self.combo_end_ctrl_point.setCurrentIndex(index)
        else:
            self._end_ctrl_selected = int(index)
            self._sync_end_disp_controls_from_selected()
            self._refresh_plane_overlays()
        self.footer.show_message(f"Selected end control point {index + 1}")

    def _build_curved_end_snapshot(self) -> BackendSnapshot | None:
        """Freeze the current deformed end plane as a Backend snapshot."""
        mesh = self.viewer.mesh
        pose = self._end_pose_for_deform()
        if mesh is None or pose is None:
            return None
        half = plane_half_extent(mesh)
        ctrl_uv = np.asarray(self._end_ctrl_uv, dtype=float).copy()
        ctrl_disp = np.asarray(self._end_ctrl_disp, dtype=float).copy()
        center, normal = pose_frame(
            x=float(pose.x),
            y=float(pose.y),
            z=float(pose.z),
            rx=pose.rx,
            ry=pose.ry,
            rz=pose.rz,
        )
        surface = make_deformed_plane_mesh(
            center=center,
            normal=normal,
            half_extent=half,
            uv=ctrl_uv,
            displacements=ctrl_disp,
        ).copy(deep=True)
        return BackendSnapshot(
            mesh=surface,
            x=float(pose.x),
            y=float(pose.y),
            z=float(pose.z),
            rx=float(pose.rx),
            ry=float(pose.ry),
            rz=float(pose.rz),
            half_extent=float(half),
            ctrl_uv=ctrl_uv,
            ctrl_disp=ctrl_disp,
        )

    def _copy_curved_end_plane(self) -> None:
        snapshot = self._build_curved_end_snapshot()
        if snapshot is None:
            QMessageBox.information(
                self,
                "Copy Backend Plane",
                "Set the end plane position first, then copy its curved state.",
            )
            return
        self._backend_snapshot = snapshot
        self._copied_end_mesh = snapshot.mesh
        self.chk_show_backend.setEnabled(True)
        self.chk_show_backend.setChecked(True)
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message("Copied backend plane at current curved end position")

    def _on_show_copied_end_toggled(self, _checked: bool) -> None:
        self._refresh_plane_overlays()

    def _back_start_pose(self) -> PlanePose | None:
        if self._back_start.preview.is_complete:
            return self._back_start.preview
        if self._back_start.confirmed.is_complete:
            return self._back_start.confirmed
        return None

    def _update_back_start_grid_label(self) -> None:
        n = int(self._back_start_grid_n)
        total = n * n
        self.lbl_back_start_grid.setText(
            f"{n} × {n} = {total:,} sample points "
            f"(max {_MAX_BACK_START_POINTS:,}). Not drawn in the viewer."
        )

    def _on_back_start_grid_changed(self, value: int) -> None:
        self._back_start_grid_n = int(value)
        self._update_back_start_grid_label()

    def _match_back_start_to_backend(self) -> None:
        """Copy Backend plane position and rotation onto the Back-start plane."""
        if self.viewer.mesh is None or self._backend_snapshot is None:
            QMessageBox.information(
                self,
                "Apply Backend Position",
                "Copy a Backend plane first, then apply its position to Back-start.",
            )
            return

        backend = self._backend_snapshot
        pose = PlanePose(
            x=backend.x,
            y=backend.y,
            z=backend.z,
            rx=backend.rx,
            ry=backend.ry,
            rz=backend.rz,
        )
        self._apply_pose_to_plane_controls(self._back_start, pose)
        self._back_start.confirmed.clear_confirmed()
        self._update_plane_status_label(self._back_start)
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(
            "Back-start plane matched to Backend plane "
            f"(X = {pose.x:.3f}, Y = {pose.y:.3f}, Z = {pose.z:.3f} mm)"
        )

    def _confirm_plane(self, plane: PlanePanel) -> None:
        if not plane.preview.is_complete:
            return
        plane.confirmed.copy_from(plane.preview)
        self._update_plane_status_label(plane)
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)
        self.footer.show_message(
            f"Confirmed {plane.title.lower()}: "
            f"X = {plane.confirmed.x:.3f}, Y = {plane.confirmed.y:.3f}, "
            f"Z = {plane.confirmed.z:.3f} mm, "
            f"Rx = {plane.confirmed.rx:.0f}°, Ry = {plane.confirmed.ry:.0f}°, "
            f"Rz = {plane.confirmed.rz:.0f}°"
        )

    def _update_plane_status_label(self, plane: PlanePanel) -> None:
        assert plane.lbl_status is not None
        if not plane.confirmed.is_complete:
            plane.lbl_status.setText(f"{plane.title}: not set")
            return
        plane.lbl_status.setText(
            f"{plane.title}: "
            f"X = {plane.confirmed.x:.3f}, Y = {plane.confirmed.y:.3f}, "
            f"Z = {plane.confirmed.z:.3f} mm, "
            f"Rx = {plane.confirmed.rx:.0f}°, Ry = {plane.confirmed.ry:.0f}°, "
            f"Rz = {plane.confirmed.rz:.0f}°"
        )

    def _wrap_plane(self) -> None:
        if self._current_path is None or self.viewer.mesh is None:
            return

        if not self._planes_ready():
            QMessageBox.warning(
                self,
                "Planes Required",
                "Confirm both the start plane and the end plane before wrapping.",
            )
            return

        start = self._start.confirmed
        end = self._end.confirmed
        assert start.is_complete and end.is_complete

        # Prefer wrapping against the originally loaded model path when available;
        # fall back to the current viewer mesh.
        model_mesh = self.viewer.mesh
        output_dir = output_dir_for_stl(self._current_path)
        progress = self._progress_callback()

        start_origin = plane_origin(
            model_mesh,
            start.z,  # type: ignore[arg-type]
            x=start.x,
            y=start.y,
        )
        end_origin = plane_origin(
            model_mesh,
            end.z,  # type: ignore[arg-type]
            x=end.x,
            y=end.y,
        )

        end_ctrl_uv = np.asarray(self._end_ctrl_uv, dtype=float).copy()
        end_ctrl_disp = np.asarray(self._end_ctrl_disp, dtype=float).copy()

        def run_wrap():
            return wrap_start_plane_to_model(
                model_mesh,
                output_dir,
                start_origin=start_origin,
                start_rx_deg=start.rx,
                start_ry_deg=start.ry,
                start_rz_deg=start.rz,
                end_origin=end_origin,
                end_rx_deg=end.rx,
                end_ry_deg=end.ry,
                end_rz_deg=end.rz,
                density=self.spin_point_density.value(),
                end_ctrl_uv=end_ctrl_uv,
                end_ctrl_disp=end_ctrl_disp,
                progress=progress,
            )

        try:
            result = self._run_busy("Wrapping plane…", run_wrap)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Wrap Error", f"Wrap plane failed:\n{exc}")
            return

        self._wrap_mesh = result.wrap_mesh
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)

        QMessageBox.information(
            self,
            "Wrap Plane Complete",
            f"Extruded start plane solid with {result.point_count:,} samples "
            f"({result.density} × {result.density}).\n\n"
            f"Hit model: {result.hit_model_count:,}\n"
            f"Stopped at end plane: {result.hit_end_count:,}\n\n"
            f"Saved to:\n{result.output_path}",
        )
        self.footer.show_message(f"Wrap plane saved → {result.output_path.name}")

    def _wrap_back(self) -> None:
        if self._current_path is None or self.viewer.mesh is None:
            return

        back_pose = self._back_start_pose()
        backend = self._backend_snapshot
        if back_pose is None or backend is None:
            QMessageBox.warning(
                self,
                "Wrap Back Required",
                "Copy a Backend plane and set the Back-start plane before wrapping back.",
            )
            return

        model_mesh = self.viewer.mesh
        output_dir = output_dir_for_stl(self._current_path)
        progress = self._progress_callback()

        start_origin = plane_origin(
            model_mesh,
            float(back_pose.z),
            x=back_pose.x,
            y=back_pose.y,
        )
        backend_origin = (backend.x, backend.y, backend.z)
        density = int(self._back_start_grid_n)
        ctrl_uv = np.asarray(backend.ctrl_uv, dtype=float).copy()
        ctrl_disp = np.asarray(backend.ctrl_disp, dtype=float).copy()
        backend_mesh = backend.mesh

        def run_wrap_back():
            return wrap_back_start_to_backend(
                model_mesh,
                output_dir,
                back_start_origin=start_origin,
                back_start_rx_deg=back_pose.rx,
                back_start_ry_deg=back_pose.ry,
                back_start_rz_deg=back_pose.rz,
                backend_origin=backend_origin,
                backend_rx_deg=backend.rx,
                backend_ry_deg=backend.ry,
                backend_rz_deg=backend.rz,
                backend_ctrl_uv=ctrl_uv,
                backend_ctrl_disp=ctrl_disp,
                backend_mesh=backend_mesh,
                half_extent=backend.half_extent,
                density=density,
                progress=progress,
            )

        try:
            result = self._run_busy("Wrapping back…", run_wrap_back)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Wrap Back Error", f"Wrap back failed:\n{exc}")
            return

        self._wrap_back_mesh = result.wrap_mesh
        self._refresh_plane_overlays()
        self._set_controls_enabled(True)

        QMessageBox.information(
            self,
            "Wrap Back Complete",
            f"Extruded Back-start points toward Backend with {result.point_count:,} "
            f"samples ({result.density} × {result.density}).\n\n"
            f"Hit model: {result.hit_model_count:,}\n"
            f"Stopped at Backend plane: {result.hit_end_count:,}\n\n"
            f"Saved to:\n{result.output_path}",
        )
        self.footer.show_message(f"Wrap back saved → {result.output_path.name}")


def create_mold_wrapper_app(argv: list[str] | None = None) -> QApplication:
    return create_app(argv)
