"""Immutable UI-safe snapshots for adaptive next-step decisions."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.reasoning.adaptive_decision_engine import (
    AdaptiveDecisionOutcome,
)
from computer_agent.reasoning.adaptive_models import (
    AdaptiveReasoningContext,
    NextStepDecision,
    NextStepReasoningResult,
    NextStepReasoningStatus,
)


@dataclass(frozen=True, slots=True)
class DecisionAttemptSnapshot:
    """Presentation-safe summary of one adaptive decision attempt."""

    attempt_number: int
    status: str
    decision_type: str | None
    operation: str | None
    target_text: str | None
    result: str
    reason: str


@dataclass(frozen=True, slots=True)
class AdaptiveDecisionSnapshot:
    """Presentation-safe view of one adaptive decision outcome."""

    observation_application: str | None
    observation_window: str | None
    observation_text: tuple[str, ...]
    decision_source: str
    model_attempts: int
    safety_replan_used: bool
    final_status: str
    final_decision_type: str | None
    operation: str | None
    target_text: str | None
    expected_effect: str | None
    question: str | None
    completion_summary: str | None
    blocked_reason: str | None
    blocked_action_keys: tuple[str, ...]
    attempts: tuple[DecisionAttemptSnapshot, ...] = ()

    @classmethod
    def from_outcome(
        cls,
        *,
        context: AdaptiveReasoningContext,
        outcome: AdaptiveDecisionOutcome,
    ) -> "AdaptiveDecisionSnapshot":
        """Build a UI-safe snapshot from production reasoning objects."""

        if not isinstance(
            context,
            AdaptiveReasoningContext,
        ):
            raise ValueError(
                "context must be an AdaptiveReasoningContext"
            )

        if not isinstance(
            outcome,
            AdaptiveDecisionOutcome,
        ):
            raise ValueError(
                "outcome must be an AdaptiveDecisionOutcome"
            )

        final_decision = outcome.result.decision
        display_decision = (
            final_decision
            if final_decision is not None
            else outcome.result.rejected_decision
        )

        return cls(
            observation_application=(
                context.observation.application_name
            ),
            observation_window=(
                context.observation.window_title
            ),
            observation_text=tuple(
                context.observation.visible_text
            ),
            decision_source="Adaptive model",
            model_attempts=outcome.attempts,
            safety_replan_used=(
                outcome.safety_replan_used
            ),
            final_status=_result_label(
                outcome.result
            ),
            final_decision_type=(
                _decision_type_label(final_decision)
                if final_decision is not None
                else "BLOCKED"
            ),
            operation=_operation(display_decision),
            target_text=_target_text(display_decision),
            expected_effect=(
                final_decision.expected_effect
                if final_decision is not None
                else None
            ),
            question=(
                final_decision.question
                if final_decision is not None
                else None
            ),
            completion_summary=(
                final_decision.completion_summary
                if final_decision is not None
                else None
            ),
            blocked_reason=(
                None
                if outcome.result.status
                is NextStepReasoningStatus.READY
                else outcome.result.reason
            ),
            blocked_action_keys=tuple(
                context.blocked_action_keys
            ),
            attempts=tuple(
                _attempt_snapshot(
                    index=index,
                    result=result,
                )
                for index, result
                in enumerate(
                    outcome.attempt_results,
                    start=1,
                )
            ),
        )


def _attempt_snapshot(
    *,
    index: int,
    result: NextStepReasoningResult,
) -> DecisionAttemptSnapshot:
    decision = (
        result.decision
        if result.decision is not None
        else result.rejected_decision
    )

    return DecisionAttemptSnapshot(
        attempt_number=index,
        status=result.status.value.upper(),
        decision_type=_decision_type_label(decision),
        operation=_operation(decision),
        target_text=_target_text(decision),
        result=_result_label(result),
        reason=result.reason,
    )


def _result_label(
    result: NextStepReasoningResult,
) -> str:
    if (
        result.status
        is NextStepReasoningStatus.READY
    ):
        return "ACCEPTED"

    if result.rejected_decision is not None:
        return "REJECTED"

    return "BLOCKED"


def _decision_type_label(
    decision: NextStepDecision | None,
) -> str | None:
    if decision is None:
        return None

    return decision.decision_type.value.upper()


def _operation(
    decision: NextStepDecision | None,
) -> str | None:
    if decision is None or decision.action is None:
        return None

    operation = getattr(
        decision.action,
        "operation",
        None,
    )

    if operation is None:
        return None

    value = getattr(
        operation,
        "value",
        operation,
    )

    return str(value)


def _target_text(
    decision: NextStepDecision | None,
) -> str | None:
    if decision is None or decision.action is None:
        return None

    action = decision.action

    for attribute in (
        "action_target",
        "target",
    ):
        target = getattr(
            action,
            attribute,
            None,
        )

        text = getattr(
            target,
            "text",
            None,
        )

        if isinstance(text, str) and text.strip():
            return text

    for attribute in (
        "app_name",
        "value_key",
    ):
        value = getattr(
            action,
            attribute,
            None,
        )

        if isinstance(value, str) and value.strip():
            return value

    return None
