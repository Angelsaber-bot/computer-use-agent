"""Data models for deterministic structured planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import TypeAlias

from computer_agent.grounding.models import TargetSpec


MAX_PLAN_STEP_ATTEMPTS = 3
MAX_STRUCTURED_PLAN_STEPS = 20
_VALUE_KEY_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
_VALUE_KEY_RE = re.compile(_VALUE_KEY_PATTERN)


class PlanOperation(str, Enum):
    """Supported semantic plan operations."""

    CLICK_TARGET = "click_target"
    READ_CLIPBOARD = "read_clipboard"
    ACTIVATE_APP = "activate_app"
    INSERT_TEXT = "insert_text"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One ordered semantic step in a structured plan."""

    goal: str
    operation: PlanOperation
    action_target: TargetSpec
    verification_target: TargetSpec
    max_attempts: int = 1

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.goal, "goal")

        if not isinstance(self.operation, PlanOperation):
            raise ValueError("operation must be a PlanOperation")

        if self.operation is not PlanOperation.CLICK_TARGET:
            raise ValueError("PlanStep operation must be click_target")

        if not isinstance(self.action_target, TargetSpec):
            raise ValueError("action_target must be a TargetSpec")

        if not isinstance(self.verification_target, TargetSpec):
            raise ValueError("verification_target must be a TargetSpec")

        _validate_max_attempts(self.max_attempts)


@dataclass(frozen=True, slots=True)
class ReadClipboardStep:
    """Read and verify a bounded runtime text value from the clipboard."""

    goal: str
    value_key: str
    expected_text: str
    max_attempts: int = 1
    operation: PlanOperation = field(
        init=False,
        default=PlanOperation.READ_CLIPBOARD,
    )

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.goal, "goal")
        _validate_value_key(self.value_key)
        _validate_non_empty_string(self.expected_text, "expected_text")
        _validate_max_attempts(self.max_attempts)


@dataclass(frozen=True, slots=True)
class ActivateAppStep:
    """Activate an application by semantic application name."""

    goal: str
    app_name: str
    max_attempts: int = 1
    operation: PlanOperation = field(
        init=False,
        default=PlanOperation.ACTIVATE_APP,
    )

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.goal, "goal")
        _validate_non_empty_string(self.app_name, "app_name")
        _validate_max_attempts(self.max_attempts)


@dataclass(frozen=True, slots=True)
class InsertTextStep:
    """Insert a previously verified runtime text value into the active app."""

    goal: str
    value_key: str
    max_attempts: int = 1
    operation: PlanOperation = field(
        init=False,
        default=PlanOperation.INSERT_TEXT,
    )

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.goal, "goal")
        _validate_value_key(self.value_key)
        _validate_max_attempts(self.max_attempts)


SemanticPlanStep: TypeAlias = (
    PlanStep
    | ReadClipboardStep
    | ActivateAppStep
    | InsertTextStep
)
_SEMANTIC_PLAN_STEP_TYPES = (
    PlanStep,
    ReadClipboardStep,
    ActivateAppStep,
    InsertTextStep,
)


@dataclass(frozen=True, slots=True)
class StructuredPlan:
    """An immutable ordered semantic plan for a user task."""

    task_goal: str
    steps: tuple[SemanticPlanStep, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.task_goal, "task_goal")

        if not isinstance(self.steps, tuple):
            raise ValueError(
                "steps must be a tuple of PlanStep objects or typed "
                "semantic step objects"
            )

        if not self.steps:
            raise ValueError(
                "steps must contain at least one PlanStep object or "
                "typed semantic step object"
            )

        if len(self.steps) > MAX_STRUCTURED_PLAN_STEPS:
            raise ValueError(
                "steps must contain no more than "
                f"{MAX_STRUCTURED_PLAN_STEPS} PlanStep objects or typed "
                "semantic step objects"
            )

        for step in self.steps:
            if not isinstance(step, _SEMANTIC_PLAN_STEP_TYPES):
                raise ValueError(
                    "steps must contain PlanStep objects or typed "
                    "semantic step objects"
                )


def _validate_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_value_key(value: object) -> None:
    _validate_non_empty_string(value, "value_key")

    if _VALUE_KEY_RE.fullmatch(value) is None:
        raise ValueError(
            "value_key must match "
            f"{_VALUE_KEY_PATTERN}"
        )


def _validate_max_attempts(value: object) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ValueError(
            "max_attempts must be a non-boolean integer"
        )

    if value < 1:
        raise ValueError("max_attempts must be at least 1")

    if value > MAX_PLAN_STEP_ATTEMPTS:
        raise ValueError(
            "max_attempts must be less than or equal to "
            f"{MAX_PLAN_STEP_ATTEMPTS}"
        )
