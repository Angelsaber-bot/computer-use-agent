"""PySide6 Agent Workspace backed by the interactive TaskRuntime."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from computer_agent.app.event_bridge import RuntimeEventBridge
from computer_agent.app.workspace_controller import (
    WorkspaceController,
    create_workspace_demo_worker,
)
from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeWorker,
)


class MainWindow(QMainWindow):
    """Interactive desktop workspace for one computer-agent task."""

    def __init__(
        self,
        *,
        worker_factory: Callable[[], RuntimeWorker] = create_workspace_demo_worker,
    ) -> None:
        super().__init__()

        self._event_bridge = RuntimeEventBridge(self)
        self._controller = WorkspaceController(
            worker_factory=worker_factory,
            event_listener=self._event_bridge.publish,
        )

        self._build_ui()

        self._event_bridge.event_received.connect(
            self._handle_runtime_event,
            Qt.ConnectionType.QueuedConnection,
        )

        self._set_idle_controls()

    @property
    def controller(self) -> WorkspaceController:
        """Return the workspace runtime controller."""
        return self._controller

    def _build_ui(self) -> None:
        self.setWindowTitle("Computer Agent")
        self.resize(780, 640)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("Computer Agent")
        title.setStyleSheet(
            "font-size: 24px; font-weight: 600;"
        )
        layout.addWidget(title)

        subtitle = QLabel(
            "Interactive macOS task workspace"
        )
        subtitle.setStyleSheet(
            "font-size: 13px; color: #666666;"
        )
        layout.addWidget(subtitle)

        task_label = QLabel("Task")
        task_label.setStyleSheet(
            "font-weight: 600;"
        )
        layout.addWidget(task_label)

        self._task_input = QPlainTextEdit()
        self._task_input.setObjectName("taskInput")
        self._task_input.setPlaceholderText(
            "Describe what you want the agent to do..."
        )
        self._task_input.setFixedHeight(95)
        layout.addWidget(self._task_input)

        button_row = QHBoxLayout()

        self._start_button = QPushButton("Start Task")
        self._start_button.setObjectName("startButton")
        self._start_button.clicked.connect(
            self._start_task
        )
        button_row.addWidget(self._start_button)

        self._pause_button = QPushButton("Pause")
        self._pause_button.setObjectName("pauseButton")
        self._pause_button.clicked.connect(
            self._toggle_pause
        )
        button_row.addWidget(self._pause_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("stopButton")
        self._stop_button.clicked.connect(
            self._stop_task
        )
        button_row.addWidget(self._stop_button)

        button_row.addStretch()
        layout.addLayout(button_row)

        status_header = QLabel("Status")
        status_header.setStyleSheet(
            "font-weight: 600;"
        )
        layout.addWidget(status_header)

        self._status_value = QLabel("Idle")
        self._status_value.setObjectName("statusValue")
        self._status_value.setStyleSheet(
            "font-size: 16px;"
        )
        layout.addWidget(self._status_value)

        current_header = QLabel("Current Activity")
        current_header.setStyleSheet(
            "font-weight: 600;"
        )
        layout.addWidget(current_header)

        self._current_value = QLabel(
            "No active task."
        )
        self._current_value.setObjectName("currentValue")
        self._current_value.setWordWrap(True)
        layout.addWidget(self._current_value)

        activity_header = QLabel("Activity")
        activity_header.setStyleSheet(
            "font-weight: 600;"
        )
        layout.addWidget(activity_header)

        self._activity_log = QPlainTextEdit()
        self._activity_log.setObjectName("activityLog")
        self._activity_log.setReadOnly(True)
        self._activity_log.setPlaceholderText(
            "Runtime events will appear here."
        )
        layout.addWidget(self._activity_log, stretch=1)

    @Slot()
    def _start_task(self) -> None:
        goal = self._task_input.toPlainText().strip()

        if not goal:
            QMessageBox.warning(
                self,
                "Task Required",
                "Enter a task before starting the agent.",
            )
            return

        self._activity_log.clear()

        try:
            self._controller.start(goal)
        except (ValueError, RuntimeError) as error:
            QMessageBox.critical(
                self,
                "Unable to Start Task",
                str(error),
            )
            return

        self._task_input.setReadOnly(True)
        self._start_button.setEnabled(False)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

        self._status_value.setText("Starting")
        self._current_value.setText(
            "Waiting for the runtime to begin."
        )

    @Slot()
    def _toggle_pause(self) -> None:
        task = self._controller.task
        if task is None:
            return

        if task.status is RuntimeStatus.RUNNING:
            self._controller.pause()
            return

        if task.status is RuntimeStatus.PAUSED:
            self._controller.resume()

    @Slot()
    def _stop_task(self) -> None:
        if self._controller.stop():
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            self._status_value.setText("Stopping")

    @Slot(object)
    def _handle_runtime_event(
        self,
        event: object,
    ) -> None:
        if not isinstance(event, RuntimeEvent):
            return

        self._append_event(event)

        if event.event_type is RuntimeEventType.TASK_STARTED:
            self._status_value.setText("Running")
            self._current_value.setText(
                "Task runtime started."
            )
            return

        if event.event_type is RuntimeEventType.PROGRESS:
            self._current_value.setText(event.message)
            return

        if event.event_type is RuntimeEventType.TASK_PAUSED:
            self._status_value.setText("Paused")
            self._current_value.setText(
                "Execution paused at a safe checkpoint."
            )
            self._pause_button.setText("Resume")
            return

        if event.event_type is RuntimeEventType.TASK_RESUMED:
            self._status_value.setText("Running")
            self._current_value.setText(
                "Execution resumed."
            )
            self._pause_button.setText("Pause")
            return

        if event.event_type is RuntimeEventType.STOP_REQUESTED:
            self._status_value.setText("Stopping")
            self._current_value.setText(
                "Waiting for the next safe checkpoint."
            )
            return

        if event.event_type is RuntimeEventType.TASK_STOPPED:
            self._status_value.setText("Stopped")
            self._current_value.setText(
                "Task stopped safely."
            )
            self._set_terminal_controls()
            return

        if event.event_type is RuntimeEventType.TASK_COMPLETED:
            self._status_value.setText("Completed")
            self._current_value.setText(
                "Task runtime completed."
            )
            self._set_terminal_controls()
            return

        if event.event_type is RuntimeEventType.TASK_FAILED:
            self._status_value.setText("Failed")

            error = event.data.get("error")
            if isinstance(error, str) and error.strip():
                self._current_value.setText(error)
            else:
                self._current_value.setText(
                    "Task runtime failed."
                )

            self._set_terminal_controls()

    def _append_event(
        self,
        event: RuntimeEvent,
    ) -> None:
        timestamp = (
            event.created_at
            .astimezone()
            .strftime("%H:%M:%S")
        )

        self._activity_log.appendPlainText(
            f"[{timestamp}] "
            f"{event.event_type.value}: "
            f"{event.message}"
        )

    def _set_idle_controls(self) -> None:
        self._start_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._pause_button.setText("Pause")

    def _set_terminal_controls(self) -> None:
        self._task_input.setReadOnly(False)
        self._start_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._pause_button.setText("Pause")

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:
        """Request clean runtime termination before closing."""

        if self._controller.running:
            self._controller.stop()
            self._controller.wait(timeout=1.5)

        event.accept()
