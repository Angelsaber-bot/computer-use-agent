"""Tests for interactive runtime lifecycle and event models."""

from __future__ import annotations

from datetime import datetime

import pytest

from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
)


def test_runtime_task_defaults_to_created() -> None:
    task = RuntimeTask(goal="Complete a demo task")

    assert task.goal == "Complete a demo task"
    assert task.task_id
    assert task.status is RuntimeStatus.CREATED
    assert task.last_error is None
    assert isinstance(task.created_at, datetime)
    assert isinstance(task.updated_at, datetime)


@pytest.mark.parametrize(
    "goal",
    [
        "",
        "   ",
    ],
)
def test_runtime_task_rejects_empty_goal(goal: str) -> None:
    with pytest.raises(
        ValueError,
        match="goal must be a non-empty string",
    ):
        RuntimeTask(goal=goal)


def test_runtime_task_set_status_updates_status() -> None:
    task = RuntimeTask(goal="Complete a demo task")
    original_updated_at = task.updated_at

    task.set_status(RuntimeStatus.RUNNING)

    assert task.status is RuntimeStatus.RUNNING
    assert task.updated_at >= original_updated_at


def test_runtime_task_rejects_invalid_status() -> None:
    task = RuntimeTask(goal="Complete a demo task")

    with pytest.raises(
        ValueError,
        match="status must be a RuntimeStatus",
    ):
        task.set_status("running")  # type: ignore[arg-type]


def test_runtime_event_preserves_structured_data() -> None:
    event = RuntimeEvent(
        event_type=RuntimeEventType.PROGRESS,
        task_id="task-1",
        message="Observed current application.",
        data={
            "application": "Google Chrome",
        },
    )

    assert event.event_type is RuntimeEventType.PROGRESS
    assert event.task_id == "task-1"
    assert event.message == "Observed current application."
    assert event.data == {
        "application": "Google Chrome",
    }
    assert isinstance(event.created_at, datetime)


def test_runtime_event_rejects_empty_task_id() -> None:
    with pytest.raises(
        ValueError,
        match="task_id must be a non-empty string",
    ):
        RuntimeEvent(
            event_type=RuntimeEventType.PROGRESS,
            task_id="",
            message="Progress.",
        )


def test_runtime_event_rejects_empty_message() -> None:
    with pytest.raises(
        ValueError,
        match="message must be a non-empty string",
    ):
        RuntimeEvent(
            event_type=RuntimeEventType.PROGRESS,
            task_id="task-1",
            message="",
        )


def test_runtime_event_rejects_invalid_event_type() -> None:
    with pytest.raises(
        ValueError,
        match="event_type must be a RuntimeEventType",
    ):
        RuntimeEvent(
            event_type="progress",  # type: ignore[arg-type]
            task_id="task-1",
            message="Progress.",
        )
