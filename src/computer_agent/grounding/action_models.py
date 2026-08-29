"""Data models for deterministic action grounding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.core.models import Action


class ActionGroundingStatus(str, Enum):
    """Possible outcomes of converting UI grounding into an action."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ActionGroundingResult:
    """Explicit result of deterministic action grounding."""

    status: ActionGroundingStatus
    action: Action | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionGroundingStatus):
            raise ValueError(
                "status must be an ActionGroundingStatus"
            )

        if self.action is not None and not isinstance(self.action, Action):
            raise ValueError(
                "action must be an Action or None"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "reason must be a non-empty string"
            )

        if (
            self.status is ActionGroundingStatus.READY
            and self.action is None
        ):
            raise ValueError(
                "READY results require an Action"
            )

        if (
            self.status is ActionGroundingStatus.BLOCKED
            and self.action is not None
        ):
            raise ValueError(
                "BLOCKED results must not contain an Action"
            )
