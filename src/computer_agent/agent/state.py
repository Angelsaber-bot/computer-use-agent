"""State transitions and execution history for one user task."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from computer_agent.core.models import (
    Action,
    Observation,
    StepRecord,
    ToolResult,
    utc_now,
)


class AgentStatus(str, Enum):
    """Lifecycle states for one agent task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class AgentState:
    """All information required to continue or inspect one task."""

    user_task: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: AgentStatus = AgentStatus.PENDING
    steps: list[StepRecord] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.user_task.strip():
            raise ValueError("user_task cannot be empty")

    @property
    def next_step_number(self) -> int:
        """Return the number that will be assigned to the next step."""

        return len(self.steps) + 1

    def start(self) -> None:
        """Move a pending task into the running state."""

        if self.status is not AgentStatus.PENDING:
            raise ValueError(
                f"cannot start a task with status {self.status.value}"
            )

        self.status = AgentStatus.RUNNING
        self._touch()

    def record_step(
        self,
        action: Action,
        result: ToolResult,
        observation: Observation | None = None,
    ) -> StepRecord:
        """Add one completed tool attempt to the task history."""

        if self.status is not AgentStatus.RUNNING:
            raise ValueError(
                "steps can only be recorded while a task is running"
            )

        record = StepRecord(
            step_number=self.next_step_number,
            action=action,
            result=result,
            observation=observation,
        )

        self.steps.append(record)

        if result.success:
            self.last_error = None
        else:
            self.last_error = result.error

        self._touch()

        return record

    def succeed(self) -> None:
        """Mark a running task as successfully completed."""

        if self.status is not AgentStatus.RUNNING:
            raise ValueError("only a running task can succeed")

        self.status = AgentStatus.SUCCEEDED
        self.last_error = None
        self._touch()

    def fail(self, error: str) -> None:
        """Mark a pending or running task as failed."""

        if self.status not in (
            AgentStatus.PENDING,
            AgentStatus.RUNNING,
        ):
            raise ValueError("a completed task cannot fail")

        if not error.strip():
            raise ValueError("failure error cannot be empty")

        self.status = AgentStatus.FAILED
        self.last_error = error
        self._touch()

    def _touch(self) -> None:
        """Update the time of the most recent state change."""

        self.updated_at = utc_now()