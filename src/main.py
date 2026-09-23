"""Entry point for the Mold Liner STL application."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python src/main.py` without installing the package.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.main_window import MainWindow, create_app  # noqa: E402


def main() -> int:
    app = create_app()
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
