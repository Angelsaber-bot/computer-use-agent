"""Data models for deterministic action verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.grounding.models import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
)
from computer_agent.verification.state_observation import StateObservationResult


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


class PresenceExpectation(str, Enum):
    """Expected target presence in one UI state snapshot."""

    PRESENT = "present"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class UIStateCondition:
    """One generic target presence requirement."""

    target: TargetSpec
    expectation: PresenceExpectation

    def __post_init__(self) -> None:
        if not isinstance(self.target, TargetSpec):
            raise ValueError("target must be a TargetSpec")

        if not isinstance(self.expectation, PresenceExpectation):
            raise ValueError("expectation must be a PresenceExpectation")


@dataclass(frozen=True, slots=True)
class VerificationSpec:
    """Generic before/after UI state contract."""

    before_conditions: tuple[UIStateCondition, ...] = ()
    after_conditions: tuple[UIStateCondition, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.before_conditions, tuple):
            raise ValueError("before_conditions must be a tuple")

        if not isinstance(self.after_conditions, tuple):
            raise ValueError("after_conditions must be a tuple")

        for condition in self.before_conditions:
            if not isinstance(condition, UIStateCondition):
                raise ValueError(
                    "before_conditions must contain UIStateCondition objects"
                )

        for condition in self.after_conditions:
            if not isinstance(condition, UIStateCondition):
                raise ValueError(
                    "after_conditions must contain UIStateCondition objects"
                )

        if not self.before_conditions and not self.after_conditions:
            raise ValueError(
                "VerificationSpec requires at least one UIStateCondition"
            )


@dataclass(frozen=True, slots=True)
class UIStateConditionEvaluation:
    """Result for one evaluated UI state condition."""

    condition: UIStateCondition
    grounding: GroundingResult | None
    status: StateVerificationStatus
    reason: str
    observation: StateObservationResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.condition, UIStateCondition):
            raise ValueError("condition must be a UIStateCondition")

        has_grounding = self.grounding is not None
        has_observation = self.observation is not None
        if has_grounding == has_observation:
            raise ValueError(
                "UIStateConditionEvaluation requires exactly one of "
                "grounding or observation"
            )

        if has_grounding and not isinstance(self.grounding, GroundingResult):
            raise ValueError("grounding must be a GroundingResult or None")

        if has_observation and not isinstance(
            self.observation,
            StateObservationResult,
        ):
            raise ValueError(
                "observation must be a StateObservationResult or None"
            )

        if not isinstance(self.status, StateVerificationStatus):
            raise ValueError("status must be a StateVerificationStatus")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


@dataclass(frozen=True, slots=True)
class StateTransitionVerificationResult:
    """Result of evaluating a generic before/after UI state contract."""

    status: StateVerificationStatus
    before_evaluations: tuple[UIStateConditionEvaluation, ...]
    after_evaluations: tuple[UIStateConditionEvaluation, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StateVerificationStatus):
            raise ValueError("status must be a StateVerificationStatus")

        if not isinstance(self.before_evaluations, tuple):
            raise ValueError("before_evaluations must be a tuple")

        if not isinstance(self.after_evaluations, tuple):
            raise ValueError("after_evaluations must be a tuple")

        for evaluation in self.before_evaluations:
            if not isinstance(evaluation, UIStateConditionEvaluation):
                raise ValueError(
                    "before_evaluations must contain "
                    "UIStateConditionEvaluation objects"
                )

        for evaluation in self.after_evaluations:
            if not isinstance(evaluation, UIStateConditionEvaluation):
                raise ValueError(
                    "after_evaluations must contain "
                    "UIStateConditionEvaluation objects"
                )

        if not self.before_evaluations and not self.after_evaluations:
            raise ValueError(
                "StateTransitionVerificationResult requires at least "
                "one evaluation"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


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
