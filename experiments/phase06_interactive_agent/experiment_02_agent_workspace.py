"""Phase 06 Experiment 02: deterministic Agent Workspace acceptance."""

from __future__ import annotations

import os
import time
from threading import Event

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QPushButton,
)

from computer_agent.app.main_window import MainWindow
from computer_agent.runtime import (
    RuntimeControl,
    RuntimeStatus,
    RuntimeTask,
)


TITLE = "Phase 06 Experiment 02: Agent Workspace"

WAIT_TIMEOUT_SECONDS = 1.0
POLL_INTERVAL_SECONDS = 0.005


def wait_until(
    app: QApplication,
    predicate,
    *,
    timeout: float = WAIT_TIMEOUT_SECONDS,
) -> bool:
    """Process Qt events until a condition becomes true or times out."""

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        app.processEvents()

        if predicate():
            return True

        time.sleep(POLL_INTERVAL_SECONDS)

    app.processEvents()
    return bool(predicate())


def run_experiment() -> tuple[bool, tuple[str, ...]]:
    """Run deterministic workspace acceptance checks."""

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    failures: list[str] = []

    ready = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker_factory():
        def worker(
            task: RuntimeTask,
            control: RuntimeControl,
            progress,
        ) -> None:
            progress("Workspace worker reached checkpoint.")
            ready.set()

            if not allow_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
                raise RuntimeError("checkpoint release timeout")

            control.checkpoint()

            passed_checkpoint.set()
            progress("Workspace worker resumed successfully.")

        return worker

    window = MainWindow(worker_factory=worker_factory)

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
    activity_log = window.findChild(
        QPlainTextEdit,
        "activityLog",
    )

    required_widgets = {
        "task input": task_input,
        "start button": start_button,
        "pause button": pause_button,
        "stop button": stop_button,
        "status value": status_value,
        "activity log": activity_log,
    }

    for name, widget in required_widgets.items():
        if widget is None:
            failures.append(f"missing required widget: {name}")

    if failures:
        window.close()
        return False, tuple(failures)

    assert task_input is not None
    assert start_button is not None
    assert pause_button is not None
    assert stop_button is not None
    assert status_value is not None
    assert activity_log is not None

    task_input.setPlainText(
        "Demonstrate the Phase 06 Agent Workspace."
    )
    start_button.click()

    if not ready.wait(timeout=WAIT_TIMEOUT_SECONDS):
        failures.append("worker did not reach checkpoint")

    if not wait_until(
        app,
        lambda: status_value.text() == "Running",
    ):
        failures.append("workspace did not enter Running state")

    pause_button.click()

    if not wait_until(
        app,
        lambda: status_value.text() == "Paused",
    ):
        failures.append("workspace did not enter Paused state")

    if pause_button.text() != "Resume":
        failures.append(
            "pause button did not change to Resume"
        )

    allow_checkpoint.set()

    if passed_checkpoint.wait(timeout=0.05):
        failures.append(
            "worker crossed safe checkpoint while paused"
        )

    pause_button.click()

    if not passed_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
        failures.append(
            "worker did not continue after resume"
        )

    if not wait_until(
        app,
        lambda: status_value.text() == "Completed",
    ):
        failures.append(
            "workspace did not reach Completed state"
        )

    task = window.controller.task

    if task is None:
        failures.append("workspace controller lost current task")
    elif task.status is not RuntimeStatus.COMPLETED:
        failures.append(
            f"runtime final status was {task.status.value}, not completed"
        )

    log_text = activity_log.toPlainText()

    required_log_fragments = (
        "task_started",
        "task_paused",
        "task_resumed",
        "Workspace worker resumed successfully.",
        "task_completed",
    )

    for fragment in required_log_fragments:
        if fragment not in log_text:
            failures.append(
                f"activity log missing: {fragment}"
            )

    if not start_button.isEnabled():
        failures.append(
            "Start Task was not re-enabled after completion"
        )

    if pause_button.isEnabled():
        failures.append(
            "Pause remained enabled after completion"
        )

    if stop_button.isEnabled():
        failures.append(
            "Stop remained enabled after completion"
        )

    window.close()
    app.processEvents()

    return not failures, tuple(failures)


def main() -> int:
    print(TITLE)

    passed, failures = run_experiment()

    if passed:
        print("Workspace creation: passed")
        print("Runtime event delivery: passed")
        print("Pause checkpoint blocking: passed")
        print("Resume continuation: passed")
        print("Terminal UI controls: passed")
        print("Experiment acceptance: passed")
        return 0

    print("Experiment acceptance: failed")

    for failure in failures:
        print(f"- {failure}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
