"""Run heavy operations off the UI thread with progress signals."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from PyQt6.QtCore import QObject, QThread, pyqtSignal

T = TypeVar("T")
ProgressCallback = Callable[[str, int, int], None]


class BackgroundWorker(QObject):
    """Execute a callable on a background thread and report progress to the UI."""

    progress = pyqtSignal(str, int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        operation: Callable[[ProgressCallback], T],
    ) -> None:
        super().__init__()
        self._operation = operation

    def run(self) -> None:
        def report(message: str, step: int = 0, total: int = 0) -> None:
            self.progress.emit(message, step, total)

        try:
            result = self._operation(report)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class BackgroundTask:
    """Manage a QThread + BackgroundWorker lifecycle."""

    def __init__(
        self,
        operation: Callable[[ProgressCallback], T],
        *,
        on_progress: Callable[[str, int, int], None],
        on_finished: Callable[[T], None],
        on_error: Callable[[str], None],
        on_started: Callable[[], None] | None = None,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        self._thread = QThread()
        self._worker = BackgroundWorker(operation)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(on_progress)
        self._worker.finished.connect(self._handle_finished)
        self._worker.error.connect(self._handle_error)
        self._on_finished = on_finished
        self._on_error = on_error
        self._on_done = on_done

        if on_started is not None:
            self._thread.started.connect(on_started)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)

    def start(self) -> None:
        self._thread.start()

    def _handle_finished(self, result: object) -> None:
        self._on_finished(result)
        if self._on_done is not None:
            self._on_done()

    def _handle_error(self, message: str) -> None:
        self._on_error(message)
        if self._on_done is not None:
            self._on_done()

    def _cleanup(self) -> None:
        self._worker.deleteLater()
        self._thread.deleteLater()
