"""Application footer with busy animation, stage text, and view controls."""

from __future__ import annotations

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

VIEW_MODE_OPTIONS: list[tuple[str, str]] = [
    ("solid", "Solid"),
    ("solid_edges", "Solid + Edges"),
    ("wireframe", "Wireframe"),
    ("surface", "Surface"),
    ("points", "Points"),
]


class FooterBar(QFrame):
    """Footer showing operation progress and STL view mode controls."""

    view_mode_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("footerBar")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0
        self._busy = False

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(12)

        status_column = QVBoxLayout()
        status_column.setSpacing(4)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)

        self.lbl_spinner = QLabel("●")
        self.lbl_spinner.setObjectName("footerSpinner")
        self.lbl_spinner.setFixedWidth(18)
        status_row.addWidget(self.lbl_spinner)

        self.lbl_stage = QLabel("Ready")
        self.lbl_stage.setObjectName("footerStage")
        status_row.addWidget(self.lbl_stage, stretch=1)

        status_column.addLayout(status_row)

        self.progress = QProgressBar()
        self.progress.setObjectName("footerProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        status_column.addWidget(self.progress)

        root.addLayout(status_column, stretch=1)

        view_row = QHBoxLayout()
        view_row.setSpacing(8)

        view_label = QLabel("View")
        view_label.setObjectName("footerViewLabel")
        view_row.addWidget(view_label)

        self.cmb_view_mode = QComboBox()
        self.cmb_view_mode.setObjectName("footerViewMode")
        for mode_id, label in VIEW_MODE_OPTIONS:
            self.cmb_view_mode.addItem(label, mode_id)
        self.cmb_view_mode.setCurrentIndex(1)
        self.cmb_view_mode.currentIndexChanged.connect(self._emit_view_mode)
        view_row.addWidget(self.cmb_view_mode)

        root.addLayout(view_row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate_spinner)
        self._set_idle_appearance()

    def _emit_view_mode(self) -> None:
        mode = self.cmb_view_mode.currentData()
        if mode:
            self.view_mode_changed.emit(str(mode))

    def current_view_mode(self) -> str:
        return str(self.cmb_view_mode.currentData() or "solid_edges")

    def _animate_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self.lbl_spinner.setText(self._spinner_frames[self._spinner_index])

    def _set_idle_appearance(self) -> None:
        self.lbl_spinner.setText("●")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy:
            self._timer.start(90)
            return

        self._timer.stop()
        self._set_idle_appearance()
        self._repaint_progress()

    def set_stage(self, message: str, step: int = 0, total: int = 0) -> None:
        self.lbl_stage.setText(message)
        if total > 0:
            self.progress.setRange(0, total)
            clamped = min(max(step, 0), total)
            self.progress.setValue(clamped)
            percent = int((clamped / total) * 100)
            self.progress.setFormat(f"{percent}% — {message}")
        elif self._busy:
            self.progress.setRange(0, 0)
            self.progress.setFormat(message or "Working…")
        self._repaint_progress()

    def _repaint_progress(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def show_message(self, message: str) -> None:
        if not self._busy:
            self.lbl_stage.setText(message)
