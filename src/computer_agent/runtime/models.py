"""Runtime lifecycle and event models for interactive agent tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class RuntimeStatus(str, Enum):
    """Lifecycle status for one interactive task runtime."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeEventType(str, Enum):
    """Externally observable runtime event kinds."""

    TASK_STARTED = "task_started"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    STOP_REQUESTED = "stop_requested"
    TASK_STOPPED = "task_stopped"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PROGRESS = "progress"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One structured event emitted by the interactive runtime."""

    event_type: RuntimeEventType
    task_id: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, RuntimeEventType):
            raise ValueError("event_type must be a RuntimeEventType")

        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")

        if not isinstance(self.data, dict):
            raise ValueError("data must be a dict")


@dataclass(slots=True)
class RuntimeTask:
    """Runtime-owned lifecycle state for one user task."""

    goal: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: RuntimeStatus = RuntimeStatus.CREATED
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string")

    def set_status(self, status: RuntimeStatus) -> None:
        """Set lifecycle status and update the modification timestamp."""
        if not isinstance(status, RuntimeStatus):
            raise ValueError("status must be a RuntimeStatus")

        self.status = status
        self.updated_at = utc_now()
