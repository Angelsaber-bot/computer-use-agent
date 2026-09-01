"""Data models for deterministic action recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.grounding.action_models import (
    ActionGroundingResult,
    ActionGroundingStatus,
)
from computer_agent.grounding.models import (
    GroundingResult,
    GroundingStatus,
)


class RecoveryStatus(str, Enum):
    """Possible outcomes of deterministic action recovery."""

    RETRY_READY = "retry_ready"
    NOT_NEEDED = "not_needed"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Explicit result of preparing one deterministic retry."""

    status: RecoveryStatus
    grounding_result: GroundingResult | None
    action_grounding_result: ActionGroundingResult | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, RecoveryStatus):
            raise ValueError(
                "status must be a RecoveryStatus"
            )

        if (
            self.grounding_result is not None
            and not isinstance(self.grounding_result, GroundingResult)
        ):
            raise ValueError(
                "grounding_result must be a GroundingResult or None"
            )

        if (
            self.action_grounding_result is not None
            and not isinstance(
                self.action_grounding_result,
                ActionGroundingResult,
            )
        ):
            raise ValueError(
                "action_grounding_result must be an "
                "ActionGroundingResult or None"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "reason must be a non-empty string"
            )

        if self.status is RecoveryStatus.RETRY_READY:
            if self.grounding_result is None:
                raise ValueError(
                    "RETRY_READY results require a grounding_result"
                )

            if self.grounding_result.status is not GroundingStatus.RESOLVED:
                raise ValueError(
                    "RETRY_READY results require resolved UI grounding"
                )

            if self.action_grounding_result is None:
                raise ValueError(
                    "RETRY_READY results require an action_grounding_result"
                )

            if (
                self.action_grounding_result.status
                is not ActionGroundingStatus.READY
            ):
                raise ValueError(
                    "RETRY_READY results require ready action grounding"
                )

        if self.status in (
            RecoveryStatus.NOT_NEEDED,
            RecoveryStatus.EXHAUSTED,
        ):
            if self.grounding_result is not None:
                raise ValueError(
                    f"{self.status.name} results must not contain "
                    "a grounding_result"
                )

            if self.action_grounding_result is not None:
                raise ValueError(
                    f"{self.status.name} results must not contain "
                    "an action_grounding_result"
                )

        if (
            self.status is RecoveryStatus.BLOCKED
            and self.action_grounding_result is not None
        ):
            if self.grounding_result is None:
                raise ValueError(
                    "BLOCKED results with action grounding require "
                    "a grounding_result"
                )

            if (
                self.grounding_result.status
                is not GroundingStatus.RESOLVED
            ):
                raise ValueError(
                    "BLOCKED results with unresolved UI grounding must not "
                    "contain action grounding"
                )

            if (
                self.action_grounding_result.status
                is ActionGroundingStatus.READY
            ):
                raise ValueError(
                    "BLOCKED results must not contain ready action grounding"
                )

        if (
            self.status is RecoveryStatus.BLOCKED
            and self.grounding_result is not None
            and self.grounding_result.status is GroundingStatus.RESOLVED
            and self.action_grounding_result is None
        ):
            raise ValueError(
                "BLOCKED results with resolved UI grounding require "
                "blocked action grounding"
            )
