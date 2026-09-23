"""Entry point for the 3D Mold Wrapper application."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.mold_wrapper_main_window import (  # noqa: E402
    MoldWrapperMainWindow,
    create_mold_wrapper_app,
)


def main() -> int:
    app = create_mold_wrapper_app()
    window = MoldWrapperMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
