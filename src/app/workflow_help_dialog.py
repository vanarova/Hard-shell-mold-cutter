"""Help dialog with step-by-step Model Fix and Wrap workflow diagrams."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

_DIAGRAM_CSS = """
body {
  font-family: Segoe UI, sans-serif;
  color: #cdd6f4;
  background: #1e1e2e;
  margin: 8px;
}
h2 { color: #89b4fa; margin: 0 0 8px 0; font-size: 15px; }
p.lead { color: #a6adc8; margin: 0 0 14px 0; }
.flow { margin: 0 auto; max-width: 520px; }
.step {
  background: #313244;
  border: 1px solid #45475a;
  border-left: 4px solid #89b4fa;
  border-radius: 6px;
  padding: 10px 12px;
  margin: 0;
}
.step .n {
  display: inline-block;
  min-width: 22px;
  color: #89b4fa;
  font-weight: 700;
}
.step .btn {
  color: #f9e2af;
  font-weight: 600;
}
.step .hint { color: #a6adc8; font-size: 12px; margin-top: 4px; }
.arrow {
  text-align: center;
  color: #74c7ec;
  font-size: 18px;
  margin: 2px 0;
  line-height: 1;
}
.note {
  margin-top: 16px;
  padding: 10px 12px;
  background: #313244;
  border: 1px dashed #fab387;
  border-radius: 6px;
  color: #f5c2e7;
  font-size: 12px;
}
"""


def _wrap_html(body: str) -> str:
    return f"<html><head><style>{_DIAGRAM_CSS}</style></head><body>{body}</body></html>"


def _step(number: int, title: str, button: str, hint: str) -> str:
    return (
        f'<div class="step">'
        f'<div><span class="n">{number}.</span> {title}</div>'
        f'<div>Press: <span class="btn">{button}</span></div>'
        f'<div class="hint">{hint}</div>'
        f"</div>"
    )


def _arrow() -> str:
    return '<div class="arrow">▼</div>'


def _model_fix_diagram_html() -> str:
    steps = [
        _step(
            1,
            "Load the model",
            "Open STL…",
            "Start from File. Optionally rotate the model afterward.",
        ),
        _arrow(),
        _step(
            2,
            "Show repair tools",
            "Show Model Repair Planes",
            "Opens the Model Repair section in the sidebar.",
        ),
        _arrow(),
        _step(
            3,
            "Place Repair Start plane",
            "Confirm Repair Start Plane",
            "Set X/Y/Z and rotations, then confirm. Set Point density for Fill Gaps.",
        ),
        _arrow(),
        _step(
            4,
            "Place Repair End plane",
            "Confirm Repair End Plane",
            "Position the opposite YZ plane so rays cross the gaps to fill.",
        ),
        _arrow(),
        _step(
            5,
            "Choose stop rule",
            "Stop after walls",
            "Default 1 = stop at the next wall. Use -1 to fill through to the last exit.",
        ),
        _arrow(),
        _step(
            6,
            "Fill hollow regions",
            "Fill Gaps",
            "Rays start→end; material is added only inside model gaps. Saves fill_gaps.stl.",
        ),
        _arrow(),
        _step(
            7,
            "Clean the filled result (optional)",
            "Fix model / Holes",
            "Stitch holes and drop disconnected junk on the Fill Gaps mesh.",
        ),
        _arrow(),
        _step(
            8,
            "Export",
            "Download Filled Model",
            "Save the repaired mesh wherever you want.",
        ),
    ]
    body = (
        "<h2>Model Fix workflow</h2>"
        '<p class="lead">Repair hollow / incomplete regions between two YZ planes.</p>'
        f'<div class="flow">{"".join(steps)}</div>'
        '<div class="note">Tip: Confirm both repair planes before Fill Gaps. '
        "Fix model / Holes stays disabled until Fill Gaps has produced a mesh.</div>"
    )
    return _wrap_html(body)


def _wrap_diagram_html() -> str:
    steps = [
        _step(
            1,
            "Load the model",
            "Open STL…",
            "Same starting point as Model Fix. Rotate if needed.",
        ),
        _arrow(),
        _step(
            2,
            "Show wrap tools",
            "Show Start / End Planes",
            "Opens Start Plane, End Plane, Backend, Back-start, Wrap, and Clear.",
        ),
        _arrow(),
        _step(
            3,
            "Set Start Plane",
            "Confirm Start Plane",
            "Also set Point density (N×N). This density syncs into Backend Plane Grid.",
        ),
        _arrow(),
        _step(
            4,
            "Set End Plane",
            "Confirm End Plane",
            "Optional: Match End to Start, then adjust. Confirm when ready.",
        ),
        _arrow(),
        _step(
            5,
            "Deform the End Plane (optional)",
            "End Plane Deform controls",
            "Add control points, select one (list or click in 3D), pull/push along the normal.",
        ),
        _arrow(),
        _step(
            6,
            "Choose Backend grid density",
            "Backend Plane Grid → Grid density",
            "Usually already matches Start density. Raise it if Wrap Back needs a finer stop surface.",
        ),
        _arrow(),
        _step(
            7,
            "Freeze the curved end",
            "Copy Backend Plane",
            "Snapshots the deformed End plane at the Backend grid density.",
        ),
        _arrow(),
        _step(
            8,
            "Set Back-start Plane",
            "Confirm Back-start Plane",
            "Or use Apply Backend Position to Back-start, then nudge. Set Back-start grid density.",
        ),
        _arrow(),
        _step(
            9,
            "Wrap Start → End",
            "Wrap Plane",
            "Extrudes Start toward End; rays stop on the model or end surface. Saves wrap_plane.stl.",
        ),
        _arrow(),
        _step(
            10,
            "Wrap Back-start → Backend",
            "Wrap Back",
            "Extrudes Back-start toward Backend. Saves wrap_back_plane.stl.",
        ),
        _arrow(),
        _step(
            11,
            "Clear extra material (optional, experimental)",
            "Ray Cast Clear",
            "Removes columns that miss the model + padding. Faster than before, but still scales with Wrap density.",
        ),
    ]
    body = (
        "<h2>Wrap workflow</h2>"
        '<p class="lead">Build wrap solids from Start/End, then Back-start/Backend.</p>'
        f'<div class="flow">{"".join(steps)}</div>'
        '<div class="note">Order matters: Copy Backend Plane before Wrap Back. '
        "Clear Extra Material is experimental and slow — use after Wrap if needed.</div>"
    )
    return _wrap_html(body)


class WorkflowHelpDialog(QDialog):
    """Modal help with Model Fix and Wrap workflow diagrams."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Workflow Help")
        self.setModal(True)
        self.resize(560, 640)
        self.setMinimumSize(480, 480)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("3D Mold Wrapper — what to press, in order")
        title.setObjectName("appTitle")
        root.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._make_tab(_model_fix_diagram_html()), "Model Fix workflow")
        tabs.addTab(self._make_tab(_wrap_diagram_html()), "Wrap workflow")
        root.addWidget(tabs, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        close_btn = buttons.button(QDialogButtonBox.StandardButton.Close)
        if close_btn is not None:
            close_btn.clicked.connect(self.accept)
        root.addWidget(buttons)

    @staticmethod
    def _make_tab(html: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(html)
        browser.setStyleSheet(
            "QTextBrowser { background-color: #1e1e2e; border: 1px solid #45475a; "
            "border-radius: 6px; }"
        )
        layout.addWidget(browser)
        return page
