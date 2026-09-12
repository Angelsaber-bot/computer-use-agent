"""Tests for the PySide6 Agent Workspace."""

from __future__ import annotations

import os
import time
from threading import Event

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QPushButton,
)

from computer_agent.app.event_bridge import RuntimeEventBridge
from computer_agent.app.main_window import MainWindow
from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    yield app


def _wait_until(
    app: QApplication,
    predicate,
    *,
    timeout: float = 1.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        app.processEvents()

        if predicate():
            return True

        time.sleep(0.005)

    app.processEvents()
    return bool(predicate())


def test_runtime_event_bridge_rejects_invalid_event() -> None:
    bridge = RuntimeEventBridge()

    with pytest.raises(
        ValueError,
        match="event must be a RuntimeEvent",
    ):
        bridge.publish("invalid")  # type: ignore[arg-type]


def test_workspace_starts_real_runtime_and_displays_events(
    qapp,
) -> None:
    def worker_factory():
        def worker(task, control, progress) -> None:
            control.checkpoint()
            progress("Observed deterministic workspace state.")
            control.checkpoint()

        return worker

    window = MainWindow(
        worker_factory=worker_factory,
    )

    task_input = window.findChild(
        QPlainTextEdit,
        "taskInput",
    )
    start_button = window.findChild(
        QPushButton,
        "startButton",
    )
    status_value = window.findChild(
        QLabel,
        "statusValue",
    )
    activity_log = window.findChild(
        QPlainTextEdit,
        "activityLog",
    )

    assert task_input is not None
    assert start_button is not None
    assert status_value is not None
    assert activity_log is not None

    task_input.setPlainText(
        "Complete deterministic workspace task"
    )
    start_button.click()

    assert _wait_until(
        qapp,
        lambda: status_value.text() == "Completed",
    )

    task = window.controller.task

    assert task is not None
    assert task.status is RuntimeStatus.COMPLETED

    log_text = activity_log.toPlainText()

    assert "task_started" in log_text
    assert (
        "Observed deterministic workspace state."
        in log_text
    )
    assert "task_completed" in log_text

    window.close()


def test_workspace_pause_and_resume_control_real_runtime(
    qapp,
) -> None:
    ready = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            progress("Waiting before checkpoint.")
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError(
                    "checkpoint release timeout"
                )

            control.checkpoint()

            passed_checkpoint.set()
            progress("Continued after resume.")

        return worker

    window = MainWindow(
        worker_factory=worker_factory,
    )

    task_input = window.findChild(
        QPlainTextEdit,
        "taskInput",
    )
    start_button = window.findChild(
        QPushButton,
        "startButton",
    )
    pause_button = window.findChild(
        QPushButton,
        "pauseButton",
    )
    status_value = window.findChild(
        QLabel,
        "statusValue",
    )

    assert task_input is not None
    assert start_button is not None
    assert pause_button is not None
    assert status_value is not None

    task_input.setPlainText(
        "Pause and resume workspace task"
    )
    start_button.click()

    assert ready.wait(timeout=1.0)

    pause_button.click()

    assert _wait_until(
        qapp,
        lambda: status_value.text() == "Paused",
    )
    assert pause_button.text() == "Resume"

    allow_checkpoint.set()

    assert passed_checkpoint.wait(
        timeout=0.05
    ) is False

    pause_button.click()

    assert passed_checkpoint.wait(timeout=1.0)

    assert _wait_until(
        qapp,
        lambda: status_value.text() == "Completed",
    )

    assert pause_button.text() == "Pause"

    window.close()


def test_workspace_stop_controls_real_runtime(
    qapp,
) -> None:
    ready = Event()
    allow_checkpoint = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError(
                    "checkpoint release timeout"
                )

            control.checkpoint()

        return worker

    window = MainWindow(
        worker_factory=worker_factory,
    )

    task_input = window.findChild(
        QPlainTextEdit,
        "taskInput",
    )
    start_button = window.findChild(
        QPushButton,
        "startButton",
    )
    pause_button = window.findChild(
        QPushButton,
        "pauseButton",
    )
    stop_button = window.findChild(
        QPushButton,
        "stopButton",
    )
    status_value = window.findChild(
        QLabel,
        "statusValue",
    )

    assert task_input is not None
    assert start_button is not None
    assert pause_button is not None
    assert stop_button is not None
    assert status_value is not None

    task_input.setPlainText(
        "Stop workspace task"
    )
    start_button.click()

    assert ready.wait(timeout=1.0)

    pause_button.click()

    assert _wait_until(
        qapp,
        lambda: status_value.text() == "Paused",
    )

    allow_checkpoint.set()
    stop_button.click()

    assert _wait_until(
        qapp,
        lambda: status_value.text() == "Stopped",
    )

    task = window.controller.task

    assert task is not None
    assert task.status is RuntimeStatus.STOPPED

    window.close()


def test_workspace_event_bridge_delivers_runtime_event(
    qapp,
) -> None:
    bridge = RuntimeEventBridge()
    received: list[RuntimeEvent] = []

    bridge.event_received.connect(
        received.append
    )

    event = RuntimeEvent(
        event_type=RuntimeEventType.PROGRESS,
        task_id="task-1",
        message="Bridge event.",
    )

    bridge.publish(event)

    assert _wait_until(
        qapp,
        lambda: received == [event],
    )
