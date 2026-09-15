import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from computer_agent.app.runtime_ui_state import RuntimeUIStateTracker
from computer_agent.app.tray_status import AgentTrayStatus
from computer_agent.runtime import RuntimeEvent, RuntimeEventType, RuntimeStatus, RuntimeTask


def _app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _event(event_type, message="message", **data):
    return RuntimeEvent(
        event_type=event_type,
        task_id="task-id",
        message=message,
        data=data,
    )


class FakeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.show_calls = 0
        self.raise_calls = 0
        self.activate_calls = 0

    def show(self):
        self.show_calls += 1

    def raise_(self):
        self.raise_calls += 1

    def activateWindow(self):
        self.activate_calls += 1


class FakeController:
    def __init__(self):
        self.task = RuntimeTask(goal="task")
        self.pause_calls = 0
        self.resume_calls = 0
        self.stop_calls = 0

    def pause(self):
        self.pause_calls += 1
        return True

    def resume(self):
        self.resume_calls += 1
        return True

    def stop(self):
        self.stop_calls += 1
        return True


def test_runtime_ui_state_tracks_lifecycle_and_progress():
    now = iter([10.0, 12.0])
    tracker = RuntimeUIStateTracker(clock=lambda: next(now))

    state = tracker.handle_event(
        _event(RuntimeEventType.TASK_STARTED, "Task runtime started.")
    )
    assert state.status == "Running"
    assert state.elapsed(now=11.5) == 1.5

    state = tracker.handle_event(
        _event(RuntimeEventType.PROGRESS, "Plan ready: 3 durable steps.")
    )
    assert state.step_total == 3
    assert state.verified_steps == 0

    state = tracker.handle_event(
        _event(
            RuntimeEventType.PROGRESS,
            "Step 2/3 - Open link: VERIFIED.",
        )
    )
    assert state.step_current == 2
    assert state.verified_steps == 2

    state = tracker.handle_event(
        _event(RuntimeEventType.TASK_PAUSED, "paused")
    )
    assert state.status == "Paused"

    state = tracker.handle_event(
        _event(RuntimeEventType.TASK_RESUMED, "resumed")
    )
    assert state.status == "Running"

    state = tracker.handle_event(
        _event(
            RuntimeEventType.PROGRESS,
            "Timing: Last observation 1.20 s.",
        )
    )
    assert state.latest_timing == "Last observation 1.20 s."
    assert state.current == "Execution resumed."

    state = tracker.handle_event(
        _event(RuntimeEventType.TASK_COMPLETED, "completed")
    )
    assert state.status == "Completed"
    assert state.finished is True


def test_runtime_ui_state_tracks_failure_message():
    tracker = RuntimeUIStateTracker(clock=lambda: 1.0)
    tracker.handle_event(
        _event(RuntimeEventType.TASK_STARTED, "started")
    )
    state = tracker.handle_event(
        _event(
            RuntimeEventType.TASK_FAILED,
            "failed",
            error="deterministic failure",
        )
    )

    assert state.status == "Failed"
    assert state.current == "deterministic failure"


def test_tray_actions_call_controller_methods():
    _app()
    window = FakeWindow()
    controller = FakeController()
    tray = AgentTrayStatus(
        parent_window=window,
        controller_getter=lambda: controller,
    )

    controller.task.set_status(RuntimeStatus.RUNNING)
    tray.pause_resume_action.setEnabled(True)
    tray.pause_resume_action.trigger()
    assert controller.pause_calls == 1

    controller.task.set_status(RuntimeStatus.PAUSED)
    tray.pause_resume_action.setEnabled(True)
    tray.pause_resume_action.trigger()
    assert controller.resume_calls == 1

    tray.stop_action.setEnabled(True)
    tray.stop_action.trigger()
    assert controller.stop_calls == 1


def test_tray_progress_update_does_not_focus_main_window():
    _app()
    window = FakeWindow()
    controller = FakeController()
    tray = AgentTrayStatus(
        parent_window=window,
        controller_getter=lambda: controller,
    )
    tracker = RuntimeUIStateTracker(clock=lambda: 1.0)
    state = tracker.handle_event(
        _event(
            RuntimeEventType.PROGRESS,
            "Step 1/2 - checking browser state.",
        )
    )

    tray.update_state(state)

    assert window.show_calls == 0
    assert window.raise_calls == 0
    assert window.activate_calls == 0


def test_tray_show_main_window_is_explicit():
    _app()
    window = FakeWindow()
    tray = AgentTrayStatus(
        parent_window=window,
        controller_getter=FakeController,
    )

    tray.show_action.trigger()

    assert window.show_calls == 1
    assert window.raise_calls == 1
    assert window.activate_calls == 1
