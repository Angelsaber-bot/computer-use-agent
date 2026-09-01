"""Data models for deterministic structured planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.grounding.models import TargetSpec


MAX_PLAN_STEP_ATTEMPTS = 3
MAX_STRUCTURED_PLAN_STEPS = 20


class PlanOperation(str, Enum):
    """Supported semantic plan operations."""

    CLICK_TARGET = "click_target"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One ordered semantic step in a structured plan."""

    goal: str
    operation: PlanOperation
    action_target: TargetSpec
    verification_target: TargetSpec
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("goal must be a non-empty string")

        if not isinstance(self.operation, PlanOperation):
            raise ValueError("operation must be a PlanOperation")

        if not isinstance(self.action_target, TargetSpec):
            raise ValueError("action_target must be a TargetSpec")

        if not isinstance(self.verification_target, TargetSpec):
            raise ValueError("verification_target must be a TargetSpec")

        if isinstance(self.max_attempts, bool) or not isinstance(
            self.max_attempts,
            int,
        ):
            raise ValueError(
                "max_attempts must be a non-boolean integer"
            )

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        if self.max_attempts > MAX_PLAN_STEP_ATTEMPTS:
            raise ValueError(
                "max_attempts must be less than or equal to "
                f"{MAX_PLAN_STEP_ATTEMPTS}"
            )


@dataclass(frozen=True, slots=True)
class StructuredPlan:
    """An immutable ordered semantic plan for a user task."""

    task_goal: str
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_goal, str) or not self.task_goal.strip():
            raise ValueError("task_goal must be a non-empty string")

        if not isinstance(self.steps, tuple):
            raise ValueError("steps must be a tuple of PlanStep objects")

        if not self.steps:
            raise ValueError("steps must contain at least one PlanStep")

        if len(self.steps) > MAX_STRUCTURED_PLAN_STEPS:
            raise ValueError(
                "steps must contain no more than "
                f"{MAX_STRUCTURED_PLAN_STEPS} PlanStep objects"
            )

        for step in self.steps:
            if not isinstance(step, PlanStep):
                raise ValueError(
                    "steps must contain PlanStep objects"
                )
