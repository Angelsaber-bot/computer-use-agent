"""Bounded adaptive decision orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from computer_agent.reasoning.adaptive_models import (
    AdaptiveReasoningContext,
    NextStepReasoningResult,
    NextStepReasoningStatus,
)
from computer_agent.reasoning.next_step_reasoner import (
    NextStepReasoner,
    UNSAFE_RETRY_BLOCK_REASON,
)


_SAFETY_REPLAN_FEEDBACK = (
    "The previous proposal was rejected by the deterministic "
    "safety gate because it attempted a blocked unresolved "
    "non-idempotent action. Do not repeat any action listed in "
    "blocked_action_keys. Use the current observation to choose "
    "a safe inspection or reconciliation action. If no safe "
    "reconciliation action exists, ask the user."
)


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionOutcome:
    """Final result of bounded adaptive decision orchestration."""

    result: NextStepReasoningResult
    attempts: int
    safety_replan_used: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.result,
            NextStepReasoningResult,
        ):
            raise ValueError(
                "result must be a NextStepReasoningResult"
            )

        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 1
            or self.attempts > 2
        ):
            raise ValueError(
                "attempts must be an integer from 1 to 2"
            )

        if not isinstance(
            self.safety_replan_used,
            bool,
        ):
            raise ValueError(
                "safety_replan_used must be a bool"
            )


class AdaptiveDecisionEngine:
    """Perform one decision plus at most one safety-triggered replan."""

    def __init__(
        self,
        *,
        reasoner: NextStepReasoner,
    ) -> None:
        if not isinstance(
            reasoner,
            NextStepReasoner,
        ):
            raise ValueError(
                "reasoner must be a NextStepReasoner"
            )

        self._reasoner = reasoner

    def decide(
        self,
        context: AdaptiveReasoningContext,
    ) -> AdaptiveDecisionOutcome:
        """Return a validated decision with one bounded safety replan."""

        if not isinstance(
            context,
            AdaptiveReasoningContext,
        ):
            raise ValueError(
                "context must be an AdaptiveReasoningContext"
            )

        first = self._reasoner.reason(
            context
        )

        if (
            first.status
            is NextStepReasoningStatus.READY
        ):
            return AdaptiveDecisionOutcome(
                result=first,
                attempts=1,
                safety_replan_used=False,
            )

        if (
            first.reason
            != UNSAFE_RETRY_BLOCK_REASON
        ):
            return AdaptiveDecisionOutcome(
                result=first,
                attempts=1,
                safety_replan_used=False,
            )

        retry_context = replace(
            context,
            decision_feedback=(
                *context.decision_feedback,
                _SAFETY_REPLAN_FEEDBACK,
            ),
        )

        second = self._reasoner.reason(
            retry_context
        )

        return AdaptiveDecisionOutcome(
            result=second,
            attempts=2,
            safety_replan_used=True,
        )
