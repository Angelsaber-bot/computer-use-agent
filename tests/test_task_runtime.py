"""Tests for the Phase 06 interactive task runtime."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
    TaskRuntime,
)


def test_runtime_completes_worker_and_emits_ordered_events() -> None:
    task = RuntimeTask(goal="Complete deterministic task")
    event_bus = RuntimeEventBus()
    events: list[RuntimeEvent] = []

    event_bus.subscribe(events.append)

    def worker(task, control, progress) -> None:
        assert task.status is RuntimeStatus.RUNNING

        control.checkpoint()
        progress("Observed environment.")

        control.checkpoint()
        progress("Verified deterministic result.")

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    runtime.run()

    assert task.status is RuntimeStatus.COMPLETED
    assert task.last_error is None

    assert [
        event.event_type
        for event in events
    ] == [
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    ]

    assert events[1].message == "Observed environment."
    assert events[2].message == "Verified deterministic result."

    assert all(
        event.task_id == task.task_id
        for event in events
    )


def test_runtime_records_worker_failure() -> None:
    task = RuntimeTask(goal="Fail deterministic task")
    events: list[RuntimeEvent] = []
    event_bus = RuntimeEventBus()
    event_bus.subscribe(events.append)

    def worker(task, control, progress) -> None:
        raise RuntimeError("deterministic worker failure")

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    runtime.run()

    assert task.status is RuntimeStatus.FAILED
    assert task.last_error == "deterministic worker failure"

    assert [
        event.event_type
        for event in events
    ] == [
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.TASK_FAILED,
    ]

    assert events[-1].data == {
        "error": "deterministic worker failure",
    }


def test_runtime_pause_blocks_progress_until_resume() -> None:
    task = RuntimeTask(goal="Pause and resume deterministic task")
    events: list[RuntimeEvent] = []
    event_bus = RuntimeEventBus()
    event_bus.subscribe(events.append)

    ready_for_checkpoint = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker(task, control, progress) -> None:
        progress("Worker reached pre-checkpoint stage.")
        ready_for_checkpoint.set()

        assert allow_checkpoint.wait(timeout=1.0)

        control.checkpoint()

        passed_checkpoint.set()
        progress("Worker passed checkpoint.")

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    thread = Thread(target=runtime.run)
    thread.start()

    assert ready_for_checkpoint.wait(timeout=1.0)

    assert runtime.pause() is True
    assert task.status is RuntimeStatus.PAUSED

    allow_checkpoint.set()

    assert passed_checkpoint.wait(timeout=0.05) is False

    assert runtime.resume() is True
    assert task.status is RuntimeStatus.RUNNING

    assert passed_checkpoint.wait(timeout=1.0)

    thread.join(timeout=1.0)
    assert thread.is_alive() is False

    assert task.status is RuntimeStatus.COMPLETED

    assert [
        event.event_type
        for event in events
    ] == [
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.TASK_RESUMED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    ]


def test_runtime_stop_while_running_terminates_at_checkpoint() -> None:
    task = RuntimeTask(goal="Stop running deterministic task")
    events: list[RuntimeEvent] = []
    event_bus = RuntimeEventBus()
    event_bus.subscribe(events.append)

    ready_for_checkpoint = Event()
    allow_checkpoint = Event()
    after_checkpoint = Event()

    def worker(task, control, progress) -> None:
        ready_for_checkpoint.set()

        assert allow_checkpoint.wait(timeout=1.0)

        control.checkpoint()

        after_checkpoint.set()

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    thread = Thread(target=runtime.run)
    thread.start()

    assert ready_for_checkpoint.wait(timeout=1.0)

    assert runtime.stop() is True

    allow_checkpoint.set()

    thread.join(timeout=1.0)
    assert thread.is_alive() is False

    assert after_checkpoint.is_set() is False
    assert task.status is RuntimeStatus.STOPPED

    assert [
        event.event_type
        for event in events
    ] == [
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.STOP_REQUESTED,
        RuntimeEventType.TASK_STOPPED,
    ]


def test_runtime_stop_while_paused_wakes_and_terminates() -> None:
    task = RuntimeTask(goal="Stop paused deterministic task")
    events: list[RuntimeEvent] = []
    event_bus = RuntimeEventBus()
    event_bus.subscribe(events.append)

    ready_for_checkpoint = Event()
    allow_checkpoint = Event()

    def worker(task, control, progress) -> None:
        ready_for_checkpoint.set()

        assert allow_checkpoint.wait(timeout=1.0)

        control.checkpoint()

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    thread = Thread(target=runtime.run)
    thread.start()

    assert ready_for_checkpoint.wait(timeout=1.0)

    assert runtime.pause() is True
    assert task.status is RuntimeStatus.PAUSED

    allow_checkpoint.set()

    assert runtime.stop() is True

    thread.join(timeout=1.0)
    assert thread.is_alive() is False

    assert task.status is RuntimeStatus.STOPPED

    assert [
        event.event_type
        for event in events
    ] == [
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.STOP_REQUESTED,
        RuntimeEventType.TASK_STOPPED,
    ]


def test_runtime_rejects_second_run() -> None:
    task = RuntimeTask(goal="Run only once")

    def worker(task, control, progress) -> None:
        control.checkpoint()

    runtime = TaskRuntime(
        task=task,
        worker=worker,
    )

    runtime.run()

    assert task.status is RuntimeStatus.COMPLETED

    with pytest.raises(
        RuntimeError,
        match="cannot run task with status completed",
    ):
        runtime.run()


def test_pause_resume_and_stop_reject_invalid_lifecycle_states() -> None:
    task = RuntimeTask(goal="Lifecycle validation")

    def worker(task, control, progress) -> None:
        pass

    runtime = TaskRuntime(
        task=task,
        worker=worker,
    )

    assert runtime.pause() is False
    assert runtime.resume() is False
    assert runtime.stop() is False

    runtime.run()

    assert task.status is RuntimeStatus.COMPLETED

    assert runtime.pause() is False
    assert runtime.resume() is False
    assert runtime.stop() is False


def test_event_bus_supports_subscribe_and_unsubscribe() -> None:
    bus = RuntimeEventBus()
    events: list[RuntimeEvent] = []

    bus.subscribe(events.append)

    first = RuntimeEvent(
        event_type=RuntimeEventType.PROGRESS,
        task_id="task-1",
        message="First event.",
    )

    bus.publish(first)

    assert events == [first]

    bus.unsubscribe(events.append)

    second = RuntimeEvent(
        event_type=RuntimeEventType.PROGRESS,
        task_id="task-1",
        message="Second event.",
    )

    bus.publish(second)

    assert events == [first]
