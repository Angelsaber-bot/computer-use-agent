"""Build compact adaptive-reasoning context from TaskState."""

from __future__ import annotations

from computer_agent.reasoning.adaptive_models import (
    AdaptiveReasoningContext,
    ObservationContext,
)
from computer_agent.task import (
    EvidenceFreshness,
    SideEffectState,
    SubgoalStatus,
    TaskState,
    TaskStateTransitions,
)


def build_adaptive_reasoning_context(
    *,
    state: TaskState,
    observation: ObservationContext,
) -> AdaptiveReasoningContext:
    """Build one bounded next-step reasoning input."""

    if not isinstance(state, TaskState):
        raise ValueError(
            "state must be a TaskState"
        )

    if not isinstance(
        observation,
        ObservationContext,
    ):
        raise ValueError(
            "observation must be an ObservationContext"
        )

    verified_subgoals = tuple(
        item.description
        for item in state.subgoals.values()
        if item.status is SubgoalStatus.VERIFIED
    )

    unresolved_subgoals = tuple(
        f"{item.status.value}: {item.description}"
        for item in state.subgoals.values()
        if item.status is not SubgoalStatus.VERIFIED
    )

    current_evidence = tuple(
        f"{item.summary} | source={item.source}"
        for item in state.evidence.values()
        if (
            item.freshness
            is EvidenceFreshness.CURRENT
        )
    )

    stale_or_unknown_evidence = tuple(
        (
            f"{item.freshness.value}: "
            f"{item.summary} | source={item.source}"
        )
        for item in state.evidence.values()
        if (
            item.freshness
            is not EvidenceFreshness.CURRENT
        )
    )

    unresolved_side_effects = tuple(
        (
            f"{item.state.value}: "
            f"{item.description} | "
            f"idempotent={item.idempotent}"
        )
        for item in state.side_effects.values()
        if item.state is not SideEffectState.CONFIRMED
    )

    blocked_action_keys = tuple(
        item.action_key
        for item in state.side_effects.values()
        if (
            item.action_key is not None
            and item.idempotent is False
            and item.state
            in (
                SideEffectState.EXECUTED,
                SideEffectState.UNKNOWN,
            )
        )
    )

    transitions = TaskStateTransitions(state)
    blockers = transitions.completion_blockers()

    return AdaptiveReasoningContext(
        goal=state.goal,
        constraints=state.constraints,
        task_status=state.status.value,
        verified_subgoals=verified_subgoals,
        unresolved_subgoals=unresolved_subgoals,
        current_evidence=current_evidence,
        stale_or_unknown_evidence=(
            stale_or_unknown_evidence
        ),
        unresolved_side_effects=(
            unresolved_side_effects
        ),
        pending_questions=state.pending_questions,
        completion_allowed=not blockers,
        completion_blockers=blockers,
        observation=observation,
        blocked_action_keys=blocked_action_keys,
    )
