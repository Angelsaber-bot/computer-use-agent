"""Data contracts shared by planners, tools, and agent state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return a timezone-aware timestamp."""

    return datetime.now(timezone.utc)


@dataclass(slots=True)
class Action:
    """A structured request to invoke one registered tool."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    action_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty")


@dataclass(slots=True)
class ToolResult:
    """The normalized result of a tool execution."""

    action_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("a successful result cannot contain an error")

        if not self.success and not self.error:
            raise ValueError("a failed result must contain an error")

        if self.duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")


@dataclass(slots=True)
class Observation:
    """Information read from the computer environment."""

    source: str
    data: Any
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("observation source cannot be empty")


@dataclass(slots=True)
class StepRecord:
    """One recorded action, result, and optional observation."""

    step_number: int
    action: Action
    result: ToolResult
    observation: Observation | None = None

    def __post_init__(self) -> None:
        if self.step_number < 1:
            raise ValueError("step_number must be at least 1")

        if self.action.action_id != self.result.action_id:
            raise ValueError("action and result IDs must match")

        if self.action.tool_name != self.result.tool_name:
            raise ValueError("action and result tool names must match")