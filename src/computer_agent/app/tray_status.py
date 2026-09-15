"""macOS menu-bar status for the Computer Agent runtime."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from computer_agent.app.runtime_ui_state import RuntimeUIState


class AgentTrayStatus(QObject):
    """Small QSystemTrayIcon status surface."""

    def __init__(
        self,
        *,
        parent_window: QWidget,
        controller_getter: Callable[[], object],
    ) -> None:
        super().__init__(parent_window)
        if not callable(controller_getter):
            raise ValueError("controller_getter must be callable")

        self._parent_window = parent_window
        self._controller_getter = controller_getter
        self._tray = QSystemTrayIcon(self)
        self._tray.setIcon(_status_icon("Idle"))
        self._tray.setToolTip("Computer Agent\nIdle")

        self._menu = QMenu(parent_window)
        self._title_action = QAction("Computer Agent", self)
        self._title_action.setEnabled(False)
        self._status_action = QAction("Status: Idle", self)
        self._status_action.setEnabled(False)
        self._step_action = QAction("Step: -", self)
        self._step_action.setEnabled(False)
        self._current_action = QAction("Current: No active task.", self)
        self._current_action.setEnabled(False)
        self._elapsed_action = QAction("Elapsed: 0.0 s", self)
        self._elapsed_action.setEnabled(False)
        self._pause_resume_action = QAction("Pause", self)
        self._stop_action = QAction("Stop", self)
        self._show_action = QAction("Show Main Window", self)

        self._pause_resume_action.triggered.connect(
            self._toggle_pause_resume
        )
        self._stop_action.triggered.connect(self._stop)
        self._show_action.triggered.connect(
            self.show_main_window
        )

        self._menu.addAction(self._title_action)
        self._menu.addSeparator()
        self._menu.addAction(self._status_action)
        self._menu.addAction(self._step_action)
        self._menu.addAction(self._current_action)
        self._menu.addAction(self._elapsed_action)
        self._menu.addSeparator()
        self._menu.addAction(self._pause_resume_action)
        self._menu.addAction(self._stop_action)
        self._menu.addAction(self._show_action)
        self._tray.setContextMenu(self._menu)
        self._tray.show()
        self.update_state(RuntimeUIState())

    @property
    def tray_icon(self) -> QSystemTrayIcon:
        return self._tray

    @property
    def show_action(self) -> QAction:
        return self._show_action

    @property
    def pause_resume_action(self) -> QAction:
        return self._pause_resume_action

    @property
    def stop_action(self) -> QAction:
        return self._stop_action

    def update_state(
        self,
        state: RuntimeUIState,
    ) -> None:
        self._tray.setIcon(_status_icon(state.status))
        elapsed = state.elapsed()
        step_text = "-"
        if state.step_total is not None:
            if state.step_current is None:
                step_text = f"0 / {state.step_total}"
            else:
                step_text = f"{state.step_current} / {state.step_total}"

        self._status_action.setText(f"Status: {state.status}")
        self._step_action.setText(f"Step: {step_text}")
        self._current_action.setText(f"Current: {state.current}")
        self._elapsed_action.setText(f"Elapsed: {elapsed:.1f} s")
        self._pause_resume_action.setText(
            "Resume" if state.status == "Paused" else "Pause"
        )
        controls_enabled = state.status in {
            "Running",
            "Planning",
            "Paused",
        }
        self._pause_resume_action.setEnabled(
            state.status in {"Running", "Paused"}
        )
        self._stop_action.setEnabled(controls_enabled)
        self._tray.setToolTip(
            "Computer Agent\n"
            f"{state.status}\n"
            f"Step {step_text}\n"
            f"{state.current}\n"
            f"Elapsed {elapsed:.1f} s"
        )

    def show_main_window(self) -> None:
        self._parent_window.show()
        self._parent_window.raise_()
        self._parent_window.activateWindow()

    def _toggle_pause_resume(self) -> None:
        controller = self._controller_getter()
        task = getattr(controller, "task", None)
        status = getattr(task, "status", None)
        status_value = getattr(status, "value", status)
        if status_value == "paused":
            controller.resume()
        else:
            controller.pause()

    def _stop(self) -> None:
        self._controller_getter().stop()


def _status_icon(
    status: str,
) -> QIcon:
    color = {
        "Running": QColor("#1f7a4d"),
        "Planning": QColor("#315fbd"),
        "Paused": QColor("#a66a00"),
        "Stopping": QColor("#a66a00"),
        "Completed": QColor("#1f7a4d"),
        "Failed": QColor("#b42318"),
        "Stopped": QColor("#555555"),
    }.get(status, QColor("#555555"))
    pixmap = QPixmap(18, 18)
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(color)
    painter.setPen(color)
    painter.drawEllipse(2, 2, 14, 14)
    painter.end()
    return QIcon(pixmap)
