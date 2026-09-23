"""Entry point for the 25D-MoldMaker application."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.moldmaker_main_window import MoldMakerMainWindow, create_moldmaker_app  # noqa: E402


def main() -> int:
    app = create_moldmaker_app()
    window = MoldMakerMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
