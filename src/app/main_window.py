"""Main application window."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QDoubleSpinBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.footer_bar import FooterBar
from app.mesh_ops import (
    boolean_subtract_meshes,
    boolean_union_meshes,
    make_solid,
    make_watertight,
    remesh_uniform,
)
from app.mesh_reducer import reduce_and_save_stl
from app.stl_viewer import StlViewer

T = TypeVar("T")

_SIDEBAR_WIDTH = 340


class MainWindow(QMainWindow):
    """STL viewer with mesh preparation tools."""

    def __init__(self) -> None:
        super().__init__()
        self._current_path: Path | None = None
        self._scene_paths: list[Path] = []

        self.setWindowTitle("Mold Liner — STL Viewer")
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
        self._reset_action_button_styles()

        self.viewer.set_selection_callback(self._on_viewer_model_selected)
        self.viewer.set_selection_enabled(True)
        self.spin_selected_model.valueChanged.connect(self._on_selected_model_spin_changed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.viewer.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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

        title = QLabel("Mold Liner")
        title.setObjectName("appTitle")
        layout.addWidget(title)

        subtitle = QLabel("STL Viewer & Tools")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(subtitle)

        layout.addWidget(self._divider())

        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)

        self.btn_open = QPushButton("Open STL…")
        self.btn_open.clicked.connect(self._open_stl)
        file_layout.addWidget(self.btn_open)

        self.btn_append = QPushButton("Append File")
        self.btn_append.setEnabled(False)
        self.btn_append.clicked.connect(self._append_stl)
        file_layout.addWidget(self.btn_append)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setObjectName("fileLabel")
        file_layout.addWidget(self.lbl_file)

        layout.addWidget(file_group)

        prepare_group = QGroupBox("Prepare Model")
        prepare_layout = QVBoxLayout(prepare_group)

        reduce_row = QHBoxLayout()
        reduce_label = QLabel("Reduce triangles")
        reduce_label.setWordWrap(True)
        reduce_row.addWidget(reduce_label, stretch=1)
        self.spin_reduce_percent = QDoubleSpinBox()
        self.spin_reduce_percent.setRange(1.0, 95.0)
        self.spin_reduce_percent.setDecimals(0)
        self.spin_reduce_percent.setSingleStep(5.0)
        self.spin_reduce_percent.setValue(50.0)
        self.spin_reduce_percent.setSuffix(" %")
        self.spin_reduce_percent.setMinimumWidth(88)
        reduce_row.addWidget(self.spin_reduce_percent)
        prepare_layout.addLayout(reduce_row)

        self.lbl_reduce_info = QLabel("Saves a lighter copy as name_reduced.stl")
        self.lbl_reduce_info.setWordWrap(True)
        self.lbl_reduce_info.setObjectName("fileLabel")
        prepare_layout.addWidget(self.lbl_reduce_info)

        self.btn_reduce = QPushButton("Reduce & Save")
        self.btn_reduce.setEnabled(False)
        self.btn_reduce.clicked.connect(self._reduce_mesh)
        prepare_layout.addWidget(self.btn_reduce)

        layout.addWidget(prepare_group)

        mesh_ops_group = QGroupBox("Mesh Operations")
        mesh_ops_layout = QVBoxLayout(mesh_ops_group)

        select_row = QHBoxLayout()
        select_label = QLabel("Selected model")
        select_label.setWordWrap(True)
        select_row.addWidget(select_label, stretch=1)
        self.spin_selected_model = QSpinBox()
        self.spin_selected_model.setMinimum(1)
        self.spin_selected_model.setMaximum(1)
        self.spin_selected_model.setEnabled(False)
        select_row.addWidget(self.spin_selected_model)
        mesh_ops_layout.addLayout(select_row)

        move_row = QHBoxLayout()
        move_label = QLabel("Move step (mm)")
        move_label.setWordWrap(True)
        move_row.addWidget(move_label, stretch=1)
        self.spin_move_step = QDoubleSpinBox()
        self.spin_move_step.setRange(0.1, 1000.0)
        self.spin_move_step.setDecimals(2)
        self.spin_move_step.setSingleStep(1.0)
        self.spin_move_step.setValue(1.0)
        self.spin_move_step.setSuffix(" mm")
        self.spin_move_step.setMinimumWidth(88)
        self.spin_move_step.setEnabled(False)
        move_row.addWidget(self.spin_move_step)
        mesh_ops_layout.addLayout(move_row)

        move_info = QLabel(
            "Click a model to select it. Arrow keys move X/Y; "
            "Page Up/Down move Z."
        )
        move_info.setWordWrap(True)
        move_info.setObjectName("fileLabel")
        mesh_ops_layout.addWidget(move_info)

        edge_row = QHBoxLayout()
        edge_label = QLabel("Target edge length (mm)")
        edge_label.setWordWrap(True)
        edge_row.addWidget(edge_label, stretch=1)
        self.spin_remesh_edge = QDoubleSpinBox()
        self.spin_remesh_edge.setRange(0.1, 50.0)
        self.spin_remesh_edge.setDecimals(2)
        self.spin_remesh_edge.setSingleStep(0.5)
        self.spin_remesh_edge.setValue(2.0)
        self.spin_remesh_edge.setSuffix(" mm")
        self.spin_remesh_edge.setMinimumWidth(88)
        self.spin_remesh_edge.setEnabled(False)
        edge_row.addWidget(self.spin_remesh_edge)
        mesh_ops_layout.addLayout(edge_row)

        mesh_ops_info = QLabel(
            "Remesh/repair apply to the selected model. Union/subtract use "
            "all loaded models (subtract: model 1 − 2 − 3 …)."
        )
        mesh_ops_info.setWordWrap(True)
        mesh_ops_info.setObjectName("fileLabel")
        mesh_ops_layout.addWidget(mesh_ops_info)

        self.btn_remesh = QPushButton("Remesh Selected")
        self.btn_remesh.setEnabled(False)
        self.btn_remesh.clicked.connect(self._remesh_selected)
        mesh_ops_layout.addWidget(self.btn_remesh)

        self.btn_watertight = QPushButton("Make Watertight")
        self.btn_watertight.setEnabled(False)
        self.btn_watertight.clicked.connect(self._make_watertight)
        mesh_ops_layout.addWidget(self.btn_watertight)

        self.btn_solid = QPushButton("Make Solid")
        self.btn_solid.setEnabled(False)
        self.btn_solid.clicked.connect(self._make_solid)
        mesh_ops_layout.addWidget(self.btn_solid)

        self.btn_union = QPushButton("Union All Models")
        self.btn_union.setEnabled(False)
        self.btn_union.clicked.connect(self._union_models)
        mesh_ops_layout.addWidget(self.btn_union)

        self.btn_subtract = QPushButton("Subtract Models")
        self.btn_subtract.setEnabled(False)
        self.btn_subtract.clicked.connect(self._subtract_models)
        mesh_ops_layout.addWidget(self.btn_subtract)

        layout.addWidget(mesh_ops_group)
        layout.addStretch()

        scroll.setWidget(panel)
        return scroll

    def _action_buttons(self) -> list[QPushButton]:
        return [self.btn_reduce]

    def _refresh_button_style(self, button: QPushButton) -> None:
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def _mark_button_completed(self, button: QPushButton) -> None:
        if button.property("completed"):
            return
        button.setProperty("completed", True)
        self._refresh_button_style(button)

    def _reset_action_button_styles(self) -> None:
        for button in self._action_buttons():
            if not button.property("completed"):
                continue
            button.setProperty("completed", False)
            self._refresh_button_style(button)

    def _progress_callback(self) -> Callable[[str, int, int], None]:
        def report(message: str, step: int = 0, total: int = 0) -> None:
            self.footer.set_stage(message, step, total)
            QApplication.processEvents()

        return report

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.btn_open.setEnabled(enabled)
        self.spin_reduce_percent.setEnabled(enabled)

        has_mesh = self.viewer.mesh is not None
        mesh_count = self.viewer.scene_mesh_count
        multi_mesh = mesh_count >= 2

        self.btn_append.setEnabled(enabled and has_mesh)
        self.btn_reduce.setEnabled(enabled and has_mesh)

        self.spin_selected_model.setEnabled(enabled and has_mesh)
        self.spin_move_step.setEnabled(enabled and has_mesh)
        self.spin_remesh_edge.setEnabled(enabled and has_mesh)
        self.btn_remesh.setEnabled(enabled and has_mesh)
        self.btn_watertight.setEnabled(enabled and has_mesh)
        self.btn_solid.setEnabled(enabled and has_mesh)
        self.btn_union.setEnabled(enabled and multi_mesh)
        self.btn_subtract.setEnabled(enabled and multi_mesh)

    def _sync_model_selector(self) -> None:
        count = self.viewer.scene_mesh_count
        self.spin_selected_model.setMaximum(max(count, 1))
        if count == 0:
            self.spin_selected_model.setValue(1)
            return
        if self.spin_selected_model.value() > count:
            self.spin_selected_model.setValue(count)
        self.viewer.set_selected_index(self.spin_selected_model.value() - 1)

    def _on_selected_model_spin_changed(self, value: int) -> None:
        if self.viewer.scene_mesh_count == 0:
            return
        self.viewer.set_selected_index(value - 1)

    def _on_viewer_model_selected(self, index: int) -> None:
        self.spin_selected_model.blockSignals(True)
        self.spin_selected_model.setValue(index + 1)
        self.spin_selected_model.blockSignals(False)
        self.footer.show_message(f"Selected model {index + 1}")

    def _selected_model_index(self) -> int:
        return self.spin_selected_model.value() - 1

    def _selected_model(self):
        return self.viewer.get_scene_mesh(self._selected_model_index())

    def _all_scene_meshes(self):
        return self.viewer.scene_meshes()

    def _apply_single_mesh_result(self, result_mesh, label: str) -> None:
        index = self._selected_model_index()
        if result_mesh.n_cells == 0:
            raise ValueError("Operation returned an empty mesh.")
        self.viewer.replace_scene_mesh(index, result_mesh)
        self.viewer.set_selected_index(index)
        self.viewer.set_view_mode(self.footer.current_view_mode())
        self.footer.show_message(f"{label} — model {index + 1}")

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.viewer.mesh is None:
            super().keyPressEvent(event)
            return

        step = self.spin_move_step.value()
        key = event.key()
        dx = dy = dz = 0.0

        if key == Qt.Key.Key_Left:
            dx = -step
        elif key == Qt.Key.Key_Right:
            dx = step
        elif key == Qt.Key.Key_Up:
            dy = step
        elif key == Qt.Key.Key_Down:
            dy = -step
        elif key == Qt.Key.Key_PageUp:
            dz = step
        elif key == Qt.Key.Key_PageDown:
            dz = -step
        else:
            super().keyPressEvent(event)
            return

        self.viewer.move_selected(dx, dy, dz)
        index = self.viewer.selected_index + 1
        self.footer.show_message(
            f"Moved model {index} by ({dx:+.2f}, {dy:+.2f}, {dz:+.2f}) mm"
        )
        event.accept()

    def _apply_scene_result(self, result_mesh, label: str, filename_stem: str) -> None:
        self.viewer.set_mesh(result_mesh)
        self.viewer.set_view_mode(self.footer.current_view_mode())
        save_path = self._default_save_path(filename_stem)
        result_mesh.save(str(save_path))
        self._current_path = save_path
        self._scene_paths = [save_path]
        self._update_file_label()
        self._sync_model_selector()
        self.footer.show_message(f"{label} saved → {save_path.name}")

    def _default_save_path(self, stem: str) -> Path:
        if self._current_path is not None:
            return self._current_path.with_name(f"{self._current_path.stem}_{stem}.stl")
        return Path(f"{stem}.stl")

    def _update_file_label(self) -> None:
        if not self._scene_paths:
            self.lbl_file.setText("No file loaded")
            return

        if len(self._scene_paths) == 1:
            self.lbl_file.setText(self._scene_paths[0].name)
            return

        names = ", ".join(path.name for path in self._scene_paths)
        self.lbl_file.setText(f"{len(self._scene_paths)} files: {names}")

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

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

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
        self._scene_paths = [path]
        self._update_file_label()
        self._sync_model_selector()
        self._reset_action_button_styles()
        self.footer.show_message(
            f"Loaded {path.name} — {mesh.n_cells:,} triangles"
        )

    def _append_stl(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Append STL File",
            "",
            "STL Files (*.stl);;All Files (*)",
        )
        if not path_str:
            return

        path = Path(path_str)

        def load_file():
            self.footer.set_stage("Appending STL file", 0, 1)
            QApplication.processEvents()
            mesh = self.viewer.append_stl(path)
            self.viewer.set_view_mode(self.footer.current_view_mode())
            self.footer.set_stage("STL appended", 1, 1)
            return mesh

        try:
            mesh = self._run_busy("Appending STL…", load_file)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Append Error", f"Could not append STL:\n{exc}")
            return

        self._scene_paths.append(path)
        self._update_file_label()
        self._sync_model_selector()
        self.footer.show_message(
            f"Appended {path.name} — {mesh.n_cells:,} triangles "
            f"({self.viewer.scene_mesh_count} in scene)"
        )

    def _reduce_mesh(self) -> None:
        if self._current_path is None or self.viewer.mesh is None:
            return

        reduction_percent = self.spin_reduce_percent.value()
        progress = self._progress_callback()

        def run_reduce():
            return reduce_and_save_stl(
                self._current_path,
                self.viewer.mesh,
                reduction_percent,
                progress=progress,
            )

        try:
            result = self._run_busy("Reducing mesh…", run_reduce)
            self.viewer.load_stl(result.output_path)
            self.viewer.set_view_mode(self.footer.current_view_mode())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Reduce Error", f"Could not reduce mesh:\n{exc}")
            return

        self._current_path = result.output_path
        self._scene_paths = [result.output_path]
        self._update_file_label()
        self._sync_model_selector()

        QMessageBox.information(
            self,
            "Mesh Reduced",
            f"Saved to:\n{result.output_path}\n\n"
            f"Triangles: {result.original_triangles:,} → "
            f"{result.reduced_triangles:,} "
            f"({result.reduction_percent:.0f}% reduced)",
        )
        self.footer.show_message(
            f"Reduced to {result.reduced_triangles:,} triangles — {result.output_path.name}"
        )
        self._mark_button_completed(self.btn_reduce)

    def _remesh_selected(self) -> None:
        if self.viewer.mesh is None:
            return

        edge_mm = self.spin_remesh_edge.value()
        mesh = self._selected_model()
        progress = self._progress_callback()

        def run_remesh():
            return remesh_uniform(mesh, edge_mm, progress=progress)

        try:
            result = self._run_busy("Remeshing…", run_remesh)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Remesh Error", f"Could not remesh model:\n{exc}")
            return

        self._apply_single_mesh_result(result.mesh, "Remesh complete")
        QMessageBox.information(self, "Remesh Complete", result.message)

    def _make_watertight(self) -> None:
        if self.viewer.mesh is None:
            return

        mesh = self._selected_model()
        progress = self._progress_callback()

        def run_repair():
            return make_watertight(mesh, progress=progress)

        try:
            result = self._run_busy("Making watertight…", run_repair)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Watertight Error", f"Could not repair mesh:\n{exc}")
            return

        self._apply_single_mesh_result(result.mesh, "Watertight repair complete")
        QMessageBox.information(self, "Watertight Complete", result.message)

    def _make_solid(self) -> None:
        if self.viewer.mesh is None:
            return

        mesh = self._selected_model()
        progress = self._progress_callback()

        def run_solid():
            return make_solid(mesh, progress=progress)

        try:
            result = self._run_busy("Making solid…", run_solid)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Solidify Error", f"Could not solidify mesh:\n{exc}")
            return

        self._apply_single_mesh_result(result.mesh, "Solid mesh ready")
        QMessageBox.information(self, "Solid Complete", result.message)

    def _union_models(self) -> None:
        meshes = self._all_scene_meshes()
        if len(meshes) < 2:
            return

        progress = self._progress_callback()

        def run_union():
            return boolean_union_meshes(meshes, progress=progress)

        try:
            result = self._run_busy("Unioning models…", run_union)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Union Error", f"Could not union models:\n{exc}")
            return

        self._apply_scene_result(result.mesh, "Union", "union")
        QMessageBox.information(self, "Union Complete", result.message)

    def _subtract_models(self) -> None:
        meshes = self._all_scene_meshes()
        if len(meshes) < 2:
            return

        progress = self._progress_callback()

        def run_subtract():
            return boolean_subtract_meshes(meshes, progress=progress)

        try:
            result = self._run_busy("Subtracting models…", run_subtract)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Subtract Error", f"Could not subtract models:\n{exc}")
            return

        self._apply_scene_result(result.mesh, "Subtract", "subtract")
        QMessageBox.information(self, "Subtract Complete", result.message)


def apply_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background-color: #11111b;
            color: #cdd6f4;
            font-family: "Segoe UI", sans-serif;
            font-size: 13px;
        }
        QFrame#sidebar {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 8px;
        }
        QScrollArea#sidebarScroll {
            background-color: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background-color: #181825;
            width: 10px;
            margin: 2px 0 2px 0;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background-color: #45475a;
            min-height: 24px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #585b70;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
        }
        QLabel#appTitle {
            font-size: 20px;
            font-weight: 700;
            color: #cba6f7;
        }
        QLabel#appSubtitle {
            color: #a6adc8;
            margin-bottom: 4px;
        }
        QLabel#fileLabel {
            color: #a6adc8;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #313244;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #89b4fa;
        }
        QPushButton {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 6px;
            padding: 8px 12px;
            color: #cdd6f4;
        }
        QPushButton:hover:enabled {
            background-color: #45475a;
            border-color: #585b70;
        }
        QPushButton:pressed:enabled {
            background-color: #585b70;
        }
        QPushButton:disabled {
            color: #585b70;
            background-color: #1e1e2e;
        }
        QPushButton[completed="true"] {
            background-color: #1e3a2f;
            border: 1px solid #40a870;
            color: #a6e3a1;
        }
        QPushButton[completed="true"]:hover:enabled {
            background-color: #264a3a;
            border-color: #56c990;
        }
        QPushButton[completed="true"]:pressed:enabled {
            background-color: #2d5a45;
        }
        QPushButton[completed="true"]:disabled {
            color: #7fbf9a;
            background-color: #182820;
            border-color: #356b4f;
        }
        QDoubleSpinBox {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
        }
        QSpinBox {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
            min-width: 56px;
        }
        QFrame#footerBar {
            background-color: #181825;
            border: 1px solid #313244;
            border-radius: 8px;
        }
        QLabel#footerSpinner {
            color: #89b4fa;
            font-size: 16px;
        }
        QLabel#footerStage {
            color: #cdd6f4;
        }
        QLabel#footerViewLabel {
            color: #a6adc8;
        }
        QProgressBar#footerProgress {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            color: #cdd6f4;
            text-align: center;
            min-height: 14px;
        }
        QProgressBar#footerProgress::chunk {
            background-color: #89b4fa;
            border-radius: 3px;
        }
        QComboBox#footerViewMode {
            background-color: #1e1e2e;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px 8px;
            color: #cdd6f4;
            min-width: 130px;
        }
        QComboBox#footerViewMode::drop-down {
            border: none;
        }
        QComboBox#footerViewMode QAbstractItemView {
            background-color: #1e1e2e;
            color: #cdd6f4;
            selection-background-color: #45475a;
        }
        """
    )


def create_app(argv: list[str] | None = None) -> QApplication:
    import sys

    app = QApplication(argv if argv is not None else sys.argv)
    apply_stylesheet(app)
    return app
