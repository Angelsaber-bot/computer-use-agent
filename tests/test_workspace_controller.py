"""Tests for the Agent Workspace runtime controller."""

from __future__ import annotations

from threading import Event

import pytest

from computer_agent.app.workspace_controller import WorkspaceController
from computer_agent.runtime import (
    RuntimeControl,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
)


def test_workspace_controller_starts_runtime_and_forwards_events() -> None:
    events: list[RuntimeEvent] = []

    def worker_factory():
        def worker(task, control, progress) -> None:
            progress("Workspace worker executed.")

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=events.append,
    )

    task = controller.start("Complete workspace task")

    assert controller.wait(timeout=1.0)
    assert task.status is RuntimeStatus.COMPLETED

    assert tuple(
        event.event_type
        for event in events
    ) == (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    )


def test_workspace_controller_pause_resume_controls_real_runtime() -> None:
    events: list[RuntimeEvent] = []

    ready = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker_factory():
        def worker(
            task: RuntimeTask,
            control: RuntimeControl,
            progress,
        ) -> None:
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError("checkpoint release timeout")

            control.checkpoint()

            passed_checkpoint.set()

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=events.append,
    )

    task = controller.start("Pause workspace task")

    assert ready.wait(timeout=1.0)

    assert controller.pause() is True
    assert task.status is RuntimeStatus.PAUSED

    allow_checkpoint.set()

    assert passed_checkpoint.wait(timeout=0.05) is False

    assert controller.resume() is True

    assert passed_checkpoint.wait(timeout=1.0)
    assert controller.wait(timeout=1.0)

    assert task.status is RuntimeStatus.COMPLETED

    assert RuntimeEventType.TASK_PAUSED in tuple(
        event.event_type
        for event in events
    )

    assert RuntimeEventType.TASK_RESUMED in tuple(
        event.event_type
        for event in events
    )


def test_workspace_controller_stop_controls_real_runtime() -> None:
    ready = Event()
    allow_checkpoint = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError("checkpoint release timeout")

            control.checkpoint()

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=lambda event: None,
    )

    task = controller.start("Stop workspace task")

    assert ready.wait(timeout=1.0)

    assert controller.pause() is True

    allow_checkpoint.set()

    assert controller.stop() is True

    assert controller.wait(timeout=1.0)
    assert task.status is RuntimeStatus.STOPPED


def test_workspace_controller_rejects_empty_goal() -> None:
    controller = WorkspaceController(
        worker_factory=lambda: lambda task, control, progress: None,
        event_listener=lambda event: None,
    )

    with pytest.raises(
        ValueError,
        match="goal must be a non-empty string",
    ):
        controller.start("   ")


def test_workspace_controller_rejects_second_active_task() -> None:
    release = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            while not release.is_set():
                control.checkpoint()
                release.wait(timeout=0.01)

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=lambda event: None,
    )

    controller.start("First task")

    with pytest.raises(
        RuntimeError,
        match="cannot start a new task",
    ):
        controller.start("Second task")

    release.set()

    assert controller.wait(timeout=1.0)


def test_workspace_controller_allows_new_task_after_completion() -> None:
    controller = WorkspaceController(
        worker_factory=lambda: lambda task, control, progress: None,
        event_listener=lambda event: None,
    )

    first = controller.start("First task")
    assert controller.wait(timeout=1.0)
    assert first.status is RuntimeStatus.COMPLETED

    second = controller.start("Second task")
    assert controller.wait(timeout=1.0)
    assert second.status is RuntimeStatus.COMPLETED

    assert first.task_id != second.task_id
