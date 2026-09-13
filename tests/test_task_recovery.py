"""Tests for restart-safe semantic task recovery."""

from __future__ import annotations

import pytest

from computer_agent.reasoning import (
    ObservationContext,
    build_adaptive_reasoning_context,
)
from computer_agent.task import (
    ClaimRecord,
    ClaimStatus,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskResumeError,
    TaskState,
    TaskStateStatus,
    prepare_state_for_resume,
)


def _build_resumable_state(
    *,
    side_effect_state: SideEffectState = (
        SideEffectState.UNKNOWN
    ),
) -> TaskState:
    live_evidence = EvidenceRecord(
        summary=(
            "The registration form currently "
            "contains the intended values."
        ),
        source="Current registration form",
        kind=EvidenceKind.VERIFICATION,
        evidence_id="evidence-live-form",
    )

    user_evidence = EvidenceRecord(
        summary=(
            "The user authorized exactly one "
            "registration submission."
        ),
        source="User confirmation",
        kind=EvidenceKind.USER_CONFIRMATION,
        evidence_id="evidence-user-authorization",
    )

    claim = ClaimRecord(
        statement=(
            "The registration form is ready "
            "to submit."
        ),
        status=ClaimStatus.VERIFIED,
        evidence_ids=(
            live_evidence.evidence_id,
        ),
        claim_id="claim-form-ready",
    )

    subgoal = SubgoalRecord(
        description=(
            "Prepare the registration form."
        ),
        status=SubgoalStatus.VERIFIED,
        claim_ids=(claim.claim_id,),
        subgoal_id="subgoal-prepare",
    )

    effect = SideEffectRecord(
        description=(
            "Submit the registration form."
        ),
        state=side_effect_state,
        side_effect_id="effect-submit",
        idempotent=False,
        action_key="click_target:submit",
    )

    return TaskState(
        goal=(
            "Submit the registration exactly once "
            "and confirm the result."
        ),
        constraints=(
            "Do not submit more than once.",
        ),
        task_id="task-restart-test",
        status=TaskStateStatus.RUNNING,
        evidence={
            live_evidence.evidence_id: (
                live_evidence
            ),
            user_evidence.evidence_id: (
                user_evidence
            ),
        },
        claims={
            claim.claim_id: claim,
        },
        subgoals={
            subgoal.subgoal_id: subgoal,
        },
        side_effects={
            effect.side_effect_id: effect,
        },
    )


def test_resume_stales_environment_evidence_and_invalidates_progress() -> None:
    state = _build_resumable_state()

    report = prepare_state_for_resume(
        state
    )

    assert (
        state.status
        is TaskStateStatus.PAUSED
    )

    assert (
        state.evidence[
            "evidence-live-form"
        ].freshness
        is EvidenceFreshness.STALE
    )

    assert (
        state.evidence[
            "evidence-user-authorization"
        ].freshness
        is EvidenceFreshness.CURRENT
    )

    assert (
        state.claims[
            "claim-form-ready"
        ].status
        is ClaimStatus.UNKNOWN
    )

    assert (
        state.subgoals[
            "subgoal-prepare"
        ].status
        is SubgoalStatus.UNKNOWN
    )

    assert report.stale_evidence_ids == (
        "evidence-live-form",
    )
    assert report.invalidated_claim_ids == (
        "claim-form-ready",
    )
    assert report.invalidated_subgoal_ids == (
        "subgoal-prepare",
    )


def test_resume_preserves_unknown_non_idempotent_side_effect() -> None:
    state = _build_resumable_state()

    report = prepare_state_for_resume(
        state
    )

    effect = state.side_effects[
        "effect-submit"
    ]

    assert (
        effect.state
        is SideEffectState.UNKNOWN
    )
    assert effect.idempotent is False
    assert (
        effect.action_key
        == "click_target:submit"
    )

    assert (
        report.preserved_unknown_side_effect_ids
        == ("effect-submit",)
    )
    assert report.blocked_action_keys == (
        "click_target:submit",
    )


def test_adaptive_context_keeps_unknown_submit_blocked_after_resume() -> None:
    state = _build_resumable_state()

    prepare_state_for_resume(state)

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            application_name="Google Chrome",
            window_title="Registration",
            visible_text=(
                "Submit",
                "Check status",
            ),
        ),
    )

    assert (
        "click_target:submit"
        in context.blocked_action_keys
    )
    assert any(
        item.startswith("unknown:")
        for item
        in context.unresolved_side_effects
    )
    assert context.completion_allowed is False


def test_confirmed_side_effect_remains_resolved_after_restart() -> None:
    state = _build_resumable_state(
        side_effect_state=(
            SideEffectState.CONFIRMED
        )
    )

    report = prepare_state_for_resume(
        state
    )

    effect = state.side_effects[
        "effect-submit"
    ]

    assert (
        effect.state
        is SideEffectState.CONFIRMED
    )

    assert (
        "click_target:submit"
        not in report.blocked_action_keys
    )

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=(
                "Submit",
                "Check status",
            ),
        ),
    )

    assert (
        "click_target:submit"
        not in context.blocked_action_keys
    )

    assert not any(
        item.startswith("confirmed:")
        for item
        in context.unresolved_side_effects
    )


@pytest.mark.parametrize(
    "status",
    (
        TaskStateStatus.COMPLETED,
        TaskStateStatus.CANCELLED,
    ),
)
def test_terminal_task_cannot_be_prepared_for_resume(
    status: TaskStateStatus,
) -> None:
    state = TaskState(
        goal="Already terminal task.",
        status=status,
    )

    with pytest.raises(
        TaskResumeError,
        match="cannot be resumed",
    ):
        prepare_state_for_resume(state)
