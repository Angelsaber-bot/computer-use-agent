"""PySide6 Agent Workspace backed by the interactive TaskRuntime."""

from __future__ import annotations

from collections.abc import Callable
import re
import time

from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from computer_agent.app.adaptive_decision_bridge import (
    AdaptiveDecisionBridge,
)
from computer_agent.app.adaptive_decision_panel import (
    AdaptiveDecisionPanel,
)
from computer_agent.app.event_bridge import RuntimeEventBridge
from computer_agent.app.evidence_demo import (
    create_evidence_demo_worker,
)
from computer_agent.app.runtime_ui_state import RuntimeUIStateTracker
from computer_agent.app.task_state_bridge import TaskStateBridge
from computer_agent.app.task_state_panel import TaskStatePanel
from computer_agent.app.tray_status import AgentTrayStatus
from computer_agent.app.storage import (
    create_default_task_store,
)
from computer_agent.app.workspace_controller import (
    WorkspaceController,
)
from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeWorker,
)
from computer_agent.reasoning import (
    AdaptiveDecisionSnapshot,
)
from computer_agent.task import (
    TaskStateSnapshot,
    TaskStateStore,
)


class MainWindow(QMainWindow):
    """Interactive desktop workspace for one computer-agent task."""

    def __init__(
        self,
        *,
        worker_factory: Callable[[], RuntimeWorker] | None = None,
        semantic_worker_factory: Callable | None = None,
        task_store: TaskStateStore | None = None,
    ) -> None:
        super().__init__()

        self._event_bridge = RuntimeEventBridge(self)
        self._task_state_bridge = TaskStateBridge(self)
        self._adaptive_decision_bridge = (
            AdaptiveDecisionBridge(self)
        )
        self._ui_state_tracker = RuntimeUIStateTracker()
        self._task_started_at: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(
            self._update_elapsed_display
        )

        if (
            task_store is None
            and worker_factory is None
            and semantic_worker_factory is None
        ):
            task_store = (
                create_default_task_store()
            )

        if (
            worker_factory is not None
            and semantic_worker_factory is not None
        ):
            raise ValueError(
                "provide only one worker factory"
            )

        if (
            worker_factory is None
            and semantic_worker_factory is None
        ):
            self._controller = WorkspaceController(
                semantic_worker_factory=create_evidence_demo_worker,
                event_listener=self._event_bridge.publish,
                task_state_listener=self._task_state_bridge.publish,
                adaptive_decision_listener=(
                    self._adaptive_decision_bridge.publish
                ),
                task_store=task_store,
            )
        elif worker_factory is not None:
            self._controller = WorkspaceController(
                worker_factory=worker_factory,
                event_listener=self._event_bridge.publish,
                task_state_listener=self._task_state_bridge.publish,
                adaptive_decision_listener=(
                    self._adaptive_decision_bridge.publish
                ),
                task_store=task_store,
            )
        else:
            self._controller = WorkspaceController(
                semantic_worker_factory=semantic_worker_factory,
                event_listener=self._event_bridge.publish,
                task_state_listener=self._task_state_bridge.publish,
                adaptive_decision_listener=(
                    self._adaptive_decision_bridge.publish
                ),
                task_store=task_store,
            )

        self._build_ui()
        self._tray_status = AgentTrayStatus(
            parent_window=self,
            controller_getter=lambda: self._controller,
        )

        self._event_bridge.event_received.connect(
            self._handle_runtime_event,
            Qt.ConnectionType.QueuedConnection,
        )
        self._task_state_bridge.snapshot_received.connect(
            self._handle_task_state_snapshot,
            Qt.ConnectionType.QueuedConnection,
        )
        self._adaptive_decision_bridge.snapshot_received.connect(
            self._handle_adaptive_decision_snapshot,
            Qt.ConnectionType.QueuedConnection,
        )

        self._set_idle_controls()

    @property
    def controller(self) -> WorkspaceController:
        """Return the workspace runtime controller."""
        return self._controller

    def _build_ui(self) -> None:
        self.setWindowTitle("Computer Agent")
        self.setMinimumSize(640, 360)
        self.resize(760, 520)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSizeConstraint(
            QLayout.SizeConstraint.SetNoConstraint
        )
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

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
        self._task_input.setFixedHeight(68)
        layout.addWidget(self._task_input)

        button_row = QHBoxLayout()

        self._start_button = QPushButton("Start Task")
        self._start_button.setObjectName("startButton")
        self._start_button.clicked.connect(
            self._start_task
        )
        button_row.addWidget(self._start_button)

        self._resume_last_button = QPushButton(
            "Resume Last Task"
        )
        self._resume_last_button.setObjectName(
            "resumeLastButton"
        )
        self._resume_last_button.clicked.connect(
            self._resume_last_task
        )
        button_row.addWidget(
            self._resume_last_button
        )

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

        self._step_progress = QProgressBar()
        self._step_progress.setObjectName("stepProgress")
        self._step_progress.setRange(0, 1)
        self._step_progress.setValue(0)
        self._step_progress.setFormat("Waiting for task")
        layout.addWidget(self._step_progress)

        self._elapsed_value = QLabel("Elapsed: 0.0 s")
        self._elapsed_value.setObjectName("elapsedValue")
        self._elapsed_value.setStyleSheet(
            "font-size: 12px; color: #666666;"
        )
        layout.addWidget(self._elapsed_value)

        self._timing_value = QLabel("Last observation: -")
        self._timing_value.setObjectName("timingValue")
        self._timing_value.setStyleSheet(
            "font-size: 12px; color: #666666;"
        )
        self._timing_value.setWordWrap(True)
        layout.addWidget(self._timing_value)

        tabs = QTabWidget()
        tabs.setObjectName("workspaceTabs")
        tabs.setMinimumHeight(0)
        tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )

        self._task_state_panel = TaskStatePanel()
        activity_tab = QWidget()
        activity_layout = QVBoxLayout(activity_tab)
        activity_layout.setContentsMargins(8, 8, 8, 8)

        activity_header = QLabel("Activity")
        activity_header.setStyleSheet(
            "font-weight: 600;"
        )
        activity_layout.addWidget(activity_header)

        self._activity_log = QPlainTextEdit()
        self._activity_log.setObjectName("activityLog")
        self._activity_log.setReadOnly(True)
        self._activity_log.setMinimumHeight(0)
        self._activity_log.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Ignored,
        )
        self._activity_log.setPlaceholderText(
            "Runtime events will appear here."
        )
        activity_layout.addWidget(
            self._activity_log,
            stretch=1,
        )

        tabs.addTab(
            activity_tab,
            "Overview / Activity",
        )

        task_state_scroll = QScrollArea()
        task_state_scroll.setObjectName(
            "taskStateScroll"
        )
        task_state_scroll.setWidgetResizable(
            True
        )
        task_state_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        task_state_scroll.setWidget(
            self._task_state_panel
        )

        tabs.addTab(
            task_state_scroll,
            "Task State",
        )

        self._adaptive_decision_panel = (
            AdaptiveDecisionPanel()
        )

        adaptive_scroll = QScrollArea()
        adaptive_scroll.setObjectName(
            "adaptiveDecisionScroll"
        )
        adaptive_scroll.setWidgetResizable(
            True
        )
        adaptive_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        adaptive_scroll.setWidget(
            self._adaptive_decision_panel
        )

        tabs.addTab(
            adaptive_scroll,
            "Decision / Safety",
        )

        layout.addWidget(tabs, stretch=1)

    @Slot(object)
    def _handle_task_state_snapshot(
        self,
        snapshot: object,
    ) -> None:
        if not isinstance(
            snapshot,
            TaskStateSnapshot,
        ):
            return

        self._task_state_panel.render(
            snapshot
        )

    @Slot(object)
    def _handle_adaptive_decision_snapshot(
        self,
        snapshot: object,
    ) -> None:
        if not isinstance(
            snapshot,
            AdaptiveDecisionSnapshot,
        ):
            return

        self._adaptive_decision_panel.render(
            snapshot
        )

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
        self._task_state_panel.clear()
        self._adaptive_decision_panel.clear()

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
        self._resume_last_button.setEnabled(False)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

        self._status_value.setText("Starting")
        self._current_value.setText(
            "Waiting for the runtime to begin."
        )
        self._start_elapsed_display()

    @Slot()
    def _resume_last_task(self) -> None:
        try:
            task_id = (
                self._controller
                .latest_resumable_task_id()
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Unable to Resume Task",
                str(error),
            )
            return

        if task_id is None:
            QMessageBox.information(
                self,
                "No Task to Resume",
                "No resumable task checkpoint was found.",
            )
            self._refresh_resume_last_button()
            return

        self._activity_log.clear()
        self._task_state_panel.clear()
        self._adaptive_decision_panel.clear()

        try:
            task = self._controller.restore_task(
                task_id
            )
        except (
            OSError,
            ValueError,
            RuntimeError,
        ) as error:
            QMessageBox.critical(
                self,
                "Unable to Resume Task",
                str(error),
            )
            return

        self._task_input.setPlainText(
            task.goal
        )
        self._task_input.setReadOnly(True)

        self._start_button.setEnabled(False)
        self._resume_last_button.setEnabled(False)
        self._pause_button.setEnabled(True)
        self._stop_button.setEnabled(True)

        self._status_value.setText(
            "Restoring"
        )
        self._current_value.setText(
            "Loading persisted task state and "
            "re-observing the current environment."
        )
        self._start_elapsed_display()

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
        ui_state = self._ui_state_tracker.handle_event(event)
        self._tray_status.update_state(ui_state)

        if event.event_type is RuntimeEventType.TASK_STARTED:
            self._status_value.setText("Running")
            self._current_value.setText(
                "Task runtime started."
            )
            return

        if event.event_type is RuntimeEventType.PROGRESS:
            if event.message.startswith("Timing: "):
                self._timing_value.setText(
                    event.message.removeprefix("Timing: ")
                )
                return
            self._current_value.setText(event.message)
            self._update_step_progress(event.message)
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
            self._finish_elapsed_display()
            self._set_terminal_controls()
            return

        if event.event_type is RuntimeEventType.TASK_COMPLETED:
            self._status_value.setText("Completed")
            self._current_value.setText(
                "Task runtime completed."
            )
            self._step_progress.setValue(
                self._step_progress.maximum()
            )
            self._step_progress.setFormat("Completed")
            self._finish_elapsed_display()
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

            self._step_progress.setFormat("Failed")
            self._finish_elapsed_display()
            self._set_terminal_controls()

    def _start_elapsed_display(self) -> None:
        self._task_started_at = time.monotonic()
        self._elapsed_value.setText("Elapsed: 0.0 s")
        self._step_progress.setRange(0, 0)
        self._step_progress.setFormat("Preparing durable plan...")
        self._elapsed_timer.start()

    def _finish_elapsed_display(self) -> None:
        self._update_elapsed_display()
        self._elapsed_timer.stop()

    @Slot()
    def _update_elapsed_display(self) -> None:
        if self._task_started_at is None:
            return
        elapsed = max(
            0.0,
            time.monotonic() - self._task_started_at,
        )
        self._elapsed_value.setText(
            f"Elapsed: {elapsed:.1f} s"
        )
        self._tray_status.update_state(
            self._ui_state_tracker.state
        )

    def _update_step_progress(self, message: str) -> None:
        plan_match = re.search(
            r"Plan ready:\s+(\d+) durable step",
            message,
            re.IGNORECASE,
        )
        if plan_match is not None:
            total = int(plan_match.group(1))
            if total > 0:
                self._step_progress.setRange(0, total)
                self._step_progress.setValue(0)
                self._step_progress.setFormat(
                    f"0 of {total} steps verified"
                )
            return

        step_match = re.search(
            r"Step\s+(\d+)\s*/\s*(\d+)",
            message,
            re.IGNORECASE,
        )
        if step_match is None:
            return

        current = int(step_match.group(1))
        total = int(step_match.group(2))
        if total <= 0:
            return

        verified = "VERIFIED" in message.upper()
        self._step_progress.setRange(0, total)
        self._step_progress.setValue(
            min(current if verified else current - 1, total)
        )
        self._step_progress.setFormat(
            f"Step {current} of {total}"
            + (" — verified" if verified else " — running")
        )

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

    def _refresh_resume_last_button(
        self,
    ) -> None:
        try:
            task_id = (
                self._controller
                .latest_resumable_task_id()
            )
        except (OSError, ValueError):
            task_id = None

        self._resume_last_button.setEnabled(
            task_id is not None
            and not self._controller.running
        )

    def _set_idle_controls(self) -> None:
        self._start_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._pause_button.setText("Pause")
        self._refresh_resume_last_button()

    def _set_terminal_controls(self) -> None:
        self._task_input.setReadOnly(False)
        self._start_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._pause_button.setText("Pause")
        self._refresh_resume_last_button()

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:
        """Request clean runtime termination before closing."""

        if self._controller.running:
            self._controller.stop()
            self._controller.wait(timeout=1.5)

        event.accept()
