"""Restart-safe preparation for persisted semantic task state."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.task.models import (
    ClaimStatus,
    EvidenceFreshness,
    EvidenceKind,
    SideEffectState,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
)
from computer_agent.task.transitions import (
    TaskStateTransitions,
)


_RESTART_VOLATILE_EVIDENCE_KINDS = frozenset(
    {
        EvidenceKind.OBSERVATION,
        EvidenceKind.VERIFICATION,
        EvidenceKind.ARTIFACT,
        EvidenceKind.EXTERNAL_STATE,
    }
)

_NON_RESUMABLE_STATUSES = frozenset(
    {
        TaskStateStatus.COMPLETED,
        TaskStateStatus.CANCELLED,
    }
)


class TaskResumeError(RuntimeError):
    """Raised when persisted task state cannot safely be resumed."""


@dataclass(frozen=True, slots=True)
class ResumePreparationReport:
    """Deterministic summary of restart preparation."""

    task_id: str
    previous_status: TaskStateStatus
    resume_status: TaskStateStatus
    stale_evidence_ids: tuple[str, ...]
    invalidated_claim_ids: tuple[str, ...]
    invalidated_subgoal_ids: tuple[str, ...]
    preserved_unknown_side_effect_ids: tuple[str, ...]
    blocked_action_keys: tuple[str, ...]


def prepare_state_for_resume(
    state: TaskState,
) -> ResumePreparationReport:
    """Prepare loaded semantic state for safe post-restart reasoning."""
    if not isinstance(state, TaskState):
        raise ValueError(
            "state must be a TaskState"
        )

    if state.status in _NON_RESUMABLE_STATUSES:
        raise TaskResumeError(
            "task cannot be resumed from terminal status: "
            f"{state.status.value}"
        )

    transitions = TaskStateTransitions(state)

    previous_status = state.status

    verified_claim_ids = {
        claim.claim_id
        for claim in state.claims.values()
        if claim.status is ClaimStatus.VERIFIED
    }
    verified_subgoal_ids = {
        subgoal.subgoal_id
        for subgoal in state.subgoals.values()
        if subgoal.status is SubgoalStatus.VERIFIED
    }

    stale_evidence_ids: list[str] = []

    for evidence_id, evidence in tuple(
        state.evidence.items()
    ):
        if (
            evidence.freshness
            is not EvidenceFreshness.CURRENT
        ):
            continue

        if (
            evidence.kind
            not in _RESTART_VOLATILE_EVIDENCE_KINDS
        ):
            continue

        transitions.set_evidence_freshness(
            evidence_id,
            EvidenceFreshness.STALE,
        )
        stale_evidence_ids.append(
            evidence_id
        )

    invalidated_claim_ids = tuple(
        claim_id
        for claim_id in verified_claim_ids
        if (
            state.claims[claim_id].status
            is not ClaimStatus.VERIFIED
        )
    )

    invalidated_subgoal_ids = tuple(
        subgoal_id
        for subgoal_id in verified_subgoal_ids
        if (
            state.subgoals[subgoal_id].status
            is not SubgoalStatus.VERIFIED
        )
    )

    preserved_unknown_side_effect_ids = tuple(
        effect.side_effect_id
        for effect in state.side_effects.values()
        if effect.state is SideEffectState.UNKNOWN
    )

    blocked_action_keys = tuple(
        effect.action_key
        for effect in state.side_effects.values()
        if (
            effect.action_key is not None
            and effect.idempotent is False
            and effect.state
            in (
                SideEffectState.EXECUTED,
                SideEffectState.UNKNOWN,
            )
        )
    )

    if state.status is not TaskStateStatus.PAUSED:
        state.status = TaskStateStatus.PAUSED
        state.touch()

    return ResumePreparationReport(
        task_id=state.task_id,
        previous_status=previous_status,
        resume_status=state.status,
        stale_evidence_ids=tuple(
            stale_evidence_ids
        ),
        invalidated_claim_ids=(
            invalidated_claim_ids
        ),
        invalidated_subgoal_ids=(
            invalidated_subgoal_ids
        ),
        preserved_unknown_side_effect_ids=(
            preserved_unknown_side_effect_ids
        ),
        blocked_action_keys=blocked_action_keys,
    )
