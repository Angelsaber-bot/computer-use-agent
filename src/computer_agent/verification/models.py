"""Data models for deterministic action verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.grounding.models import GroundingResult, GroundingStatus


class ActionVerificationStatus(str, Enum):
    """Possible outcomes of verifying an action postcondition."""

    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class StateVerificationStatus(str, Enum):
    """Possible outcomes of verifying observed application state."""

    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ActionVerificationResult:
    """Explicit result of deterministic action verification."""

    status: ActionVerificationStatus
    before_grounding: GroundingResult
    after_grounding: GroundingResult
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActionVerificationStatus):
            raise ValueError(
                "status must be an ActionVerificationStatus"
            )

        if not isinstance(self.before_grounding, GroundingResult):
            raise ValueError(
                "before_grounding must be a GroundingResult"
            )

        if not isinstance(self.after_grounding, GroundingResult):
            raise ValueError(
                "after_grounding must be a GroundingResult"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "reason must be a non-empty string"
            )

        if self.status is ActionVerificationStatus.VERIFIED:
            if self.before_grounding.status is not GroundingStatus.NOT_FOUND:
                raise ValueError(
                    "VERIFIED results require before_grounding to be NOT_FOUND"
                )

            if self.after_grounding.status is not GroundingStatus.RESOLVED:
                raise ValueError(
                    "VERIFIED results require after_grounding to be RESOLVED"
                )


@dataclass(frozen=True, slots=True)
class StateVerificationResult:
    """Explicit result of deterministic state verification."""

    status: StateVerificationStatus
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StateVerificationStatus):
            raise ValueError(
                "status must be a StateVerificationStatus"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "reason must be a non-empty string"
            )
