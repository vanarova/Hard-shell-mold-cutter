"""Entry point for the Curve Slicer application."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.curve_slicer_main_window import (  # noqa: E402
    CurveSlicerMainWindow,
    create_curve_slicer_app,
)


def main() -> int:
    app = create_curve_slicer_app()
    window = CurveSlicerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
