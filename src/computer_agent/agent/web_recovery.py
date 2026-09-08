"""Pure cause-aware web recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.grounding import GroundingResult, GroundingStatus


class WebRecoveryDecision(str, Enum):
    """Allowed web recovery decisions for failed preconditions."""

    VIEWPORT_SEARCH = "viewport_search"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class WebRecoveryDecisionResult:
    """Deterministic decision for one failed grounding precondition."""

    decision: WebRecoveryDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, WebRecoveryDecision):
            raise ValueError("decision must be a WebRecoveryDecision")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def decide_failed_grounding_recovery(
    grounding_result: GroundingResult,
) -> WebRecoveryDecisionResult:
    """Return whether bounded viewport search may recover grounding."""

    if not isinstance(grounding_result, GroundingResult):
        raise ValueError("grounding_result must be a GroundingResult")

    if grounding_result.status is GroundingStatus.NOT_FOUND:
        return WebRecoveryDecisionResult(
            decision=WebRecoveryDecision.VIEWPORT_SEARCH,
            reason=(
                "semantic target was not found; bounded viewport search "
                "is eligible"
            ),
        )

    if grounding_result.status is GroundingStatus.UNSAFE:
        if (
            grounding_result.candidates
            and all(
                candidate.rejection_reasons == ("outside_viewport",)
                for candidate in grounding_result.candidates
            )
        ):
            return WebRecoveryDecisionResult(
                decision=WebRecoveryDecision.VIEWPORT_SEARCH,
                reason=(
                    "all unsafe candidates were only outside the viewport; "
                    "bounded viewport search is eligible"
                ),
            )

        return WebRecoveryDecisionResult(
            decision=WebRecoveryDecision.BLOCK,
            reason=(
                "unsafe grounding was not recoverable by viewport search"
            ),
        )

    if grounding_result.status is GroundingStatus.AMBIGUOUS:
        return WebRecoveryDecisionResult(
            decision=WebRecoveryDecision.BLOCK,
            reason="ambiguous grounding is not eligible for viewport search",
        )

    if grounding_result.status is GroundingStatus.RESOLVED:
        return WebRecoveryDecisionResult(
            decision=WebRecoveryDecision.BLOCK,
            reason=(
                "resolved grounding does not justify failed-precondition "
                "viewport search"
            ),
        )

    return WebRecoveryDecisionResult(
        decision=WebRecoveryDecision.BLOCK,
        reason="grounding status is not eligible for viewport search",
    )
