"""Data models for provider-neutral LLM reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.planning.models import StructuredPlan


SUPPORTED_REASONING_ELEMENT_TYPES = (
    "button",
    "checkbox",
    "popup_button",
    "radio_button",
    "text_field",
    "text",
)


class ReasoningStatus(str, Enum):
    """Possible outcomes of converting task intent into a structured plan."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Explicit result of one LLM reasoning attempt."""

    status: ReasoningStatus
    plan: StructuredPlan | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ReasoningStatus):
            raise ValueError("status must be a ReasoningStatus")

        if self.plan is not None and not isinstance(
            self.plan,
            StructuredPlan,
        ):
            raise ValueError("plan must be a StructuredPlan or None")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if self.status is ReasoningStatus.READY and self.plan is None:
            raise ValueError("READY results require a StructuredPlan")

        if self.status is ReasoningStatus.BLOCKED and self.plan is not None:
            raise ValueError("BLOCKED results must not contain a plan")
