"""Tests for deterministic evidence-grounded task transitions."""

from __future__ import annotations

import pytest

from computer_agent.task import (
    ArtifactRecord,
    ClaimRecord,
    ClaimStatus,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
)


def make_state() -> tuple[
    TaskState,
    TaskStateTransitions,
]:
    state = TaskState(
        goal="Complete the deterministic workflow."
    )
    return state, TaskStateTransitions(state)


def test_add_evidence_records_it_in_state() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Current page shows confirmation.",
        source="Browser Accessibility",
        kind=EvidenceKind.OBSERVATION,
    )

    transitions.add_evidence(evidence)

    assert state.evidence[evidence.evidence_id] == evidence


def test_duplicate_evidence_is_rejected() -> None:
    _, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Observed current state.",
        source="Browser",
        kind=EvidenceKind.OBSERVATION,
    )

    transitions.add_evidence(evidence)

    with pytest.raises(
        ValueError,
        match="duplicate evidence_id",
    ):
        transitions.add_evidence(evidence)


def test_claim_verification_requires_current_evidence() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="File exists.",
        source="Local filesystem",
        kind=EvidenceKind.ARTIFACT,
    )
    claim = ClaimRecord(
        statement="The requested file exists."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)

    verified = transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    assert verified.status is ClaimStatus.VERIFIED
    assert verified.evidence_ids == (
        evidence.evidence_id,
    )
    assert (
        state.claims[claim.claim_id].status
        is ClaimStatus.VERIFIED
    )


def test_claim_verification_rejects_stale_evidence() -> None:
    _, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Old page state.",
        source="Browser",
        kind=EvidenceKind.OBSERVATION,
        freshness=EvidenceFreshness.STALE,
    )
    claim = ClaimRecord(
        statement="The page is still current."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)

    with pytest.raises(
        ValueError,
        match="evidence is not current",
    ):
        transitions.verify_claim(
            claim.claim_id,
            (evidence.evidence_id,),
        )


def test_verified_subgoal_requires_verified_claim() -> None:
    _, transitions = make_state()

    claim = ClaimRecord(
        statement="Document exists."
    )
    subgoal = SubgoalRecord(
        description="Obtain the document.",
        claim_ids=(claim.claim_id,),
    )

    transitions.add_claim(claim)
    transitions.add_subgoal(subgoal)

    with pytest.raises(
        ValueError,
        match="claim is not verified",
    ):
        transitions.verify_subgoal(
            subgoal.subgoal_id
        )


def test_current_evidence_can_verify_claim_and_subgoal() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Official PDF exists.",
        source="Local filesystem",
        kind=EvidenceKind.ARTIFACT,
    )
    claim = ClaimRecord(
        statement="Official PDF has been obtained."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)

    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Obtain the official PDF.",
        claim_ids=(claim.claim_id,),
    )

    transitions.add_subgoal(subgoal)

    verified = transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    assert verified.status is SubgoalStatus.VERIFIED
    assert (
        state.subgoals[subgoal.subgoal_id].status
        is SubgoalStatus.VERIFIED
    )


def test_stale_evidence_invalidates_claim_and_subgoal() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Tracker row is complete.",
        source="Spreadsheet row 4",
        kind=EvidenceKind.VERIFICATION,
    )
    claim = ClaimRecord(
        statement="Tracker row 4 is complete."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)

    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Complete tracker row 4.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    transitions.set_evidence_freshness(
        evidence.evidence_id,
        EvidenceFreshness.STALE,
    )

    assert (
        state.claims[claim.claim_id].status
        is ClaimStatus.UNKNOWN
    )
    assert (
        state.subgoals[subgoal.subgoal_id].status
        is SubgoalStatus.UNKNOWN
    )


def test_stale_evidence_cannot_be_revived_as_current() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Page shows result.",
        source="Browser",
        kind=EvidenceKind.OBSERVATION,
    )
    claim = ClaimRecord(
        statement="Result is present."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    transitions.set_evidence_freshness(
        evidence.evidence_id,
        EvidenceFreshness.STALE,
    )

    with pytest.raises(
        ValueError,
        match="cannot become current",
    ):
        transitions.set_evidence_freshness(
            evidence.evidence_id,
            EvidenceFreshness.CURRENT,
        )

    assert (
        state.claims[claim.claim_id].status
        is ClaimStatus.UNKNOWN
    )
    assert (
        state.evidence[evidence.evidence_id].freshness
        is EvidenceFreshness.STALE
    )


def test_side_effect_can_move_to_unknown_after_execution() -> None:
    state, transitions = make_state()

    effect = SideEffectRecord(
        description="Submit registration form.",
        idempotent=False,
    )

    transitions.add_side_effect(effect)

    executed = transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    unknown = transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    assert executed.state is SideEffectState.EXECUTED
    assert unknown.state is SideEffectState.UNKNOWN
    assert (
        state.side_effects[
            effect.side_effect_id
        ].state
        is SideEffectState.UNKNOWN
    )


def test_unknown_side_effect_can_be_confirmed_with_evidence() -> None:
    state, transitions = make_state()

    effect = SideEffectRecord(
        description="Submit registration form.",
        idempotent=False,
    )
    confirmation = EvidenceRecord(
        summary="Server shows submission ID 42.",
        source="Server-visible confirmation page",
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    transitions.add_evidence(confirmation)

    confirmed = transitions.confirm_side_effect(
        effect.side_effect_id,
        (confirmation.evidence_id,),
    )

    assert confirmed.state is SideEffectState.CONFIRMED
    assert confirmed.evidence_ids == (
        confirmation.evidence_id,
    )
    assert (
        state.side_effects[
            effect.side_effect_id
        ].state
        is SideEffectState.CONFIRMED
    )


def test_side_effect_confirmation_rejects_stale_evidence() -> None:
    _, transitions = make_state()

    effect = SideEffectRecord(
        description="Submit form."
    )
    evidence = EvidenceRecord(
        summary="Old confirmation.",
        source="Browser",
        kind=EvidenceKind.VERIFICATION,
        freshness=EvidenceFreshness.STALE,
    )

    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.add_evidence(evidence)

    with pytest.raises(
        ValueError,
        match="evidence is not current",
    ):
        transitions.confirm_side_effect(
            effect.side_effect_id,
            (evidence.evidence_id,),
        )


def test_artifact_requires_current_evidence_when_provided() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="PDF exists at expected path.",
        source="Local filesystem",
        kind=EvidenceKind.ARTIFACT,
    )
    transitions.add_evidence(evidence)

    artifact = ArtifactRecord(
        description="Official PDF",
        location="/tmp/official.pdf",
        evidence_ids=(evidence.evidence_id,),
    )

    transitions.add_artifact(artifact)

    assert (
        state.artifacts[artifact.artifact_id]
        == artifact
    )


def test_pending_question_add_and_resolve() -> None:
    state, transitions = make_state()

    question = (
        "Should the existing nonempty value be overwritten?"
    )

    transitions.add_pending_question(question)
    transitions.add_pending_question(question)

    assert state.pending_questions == (
        question,
    )

    assert (
        transitions.resolve_pending_question(
            question
        )
        is True
    )
    assert state.pending_questions == ()

    assert (
        transitions.resolve_pending_question(
            question
        )
        is False
    )


def test_completion_rejects_unverified_subgoal() -> None:
    state, transitions = make_state()

    claim = ClaimRecord(
        statement="Required work is complete."
    )
    transitions.add_claim(claim)

    subgoal = SubgoalRecord(
        description="Complete required work.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)

    assert transitions.can_complete() is False

    with pytest.raises(
        RuntimeError,
        match="subgoal is not verified",
    ):
        transitions.complete_task()

    assert (
        state.status
        is not TaskStateStatus.COMPLETED
    )


def test_completion_rejects_unknown_side_effect() -> None:
    _, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Required work is verified.",
        source="Current external state",
        kind=EvidenceKind.VERIFICATION,
    )
    claim = ClaimRecord(
        statement="Required work is complete."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Complete required work.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    effect = SideEffectRecord(
        description="Submit final result.",
        idempotent=False,
    )
    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    blockers = transitions.completion_blockers()

    assert any(
        "side effect is unresolved" in blocker
        for blocker in blockers
    )

    with pytest.raises(
        RuntimeError,
        match="side effect is unresolved",
    ):
        transitions.complete_task()


def test_completion_rejects_pending_question() -> None:
    _, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Required work is verified.",
        source="Current external state",
        kind=EvidenceKind.VERIFICATION,
    )
    claim = ClaimRecord(
        statement="Required work is complete."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Complete required work.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    transitions.add_pending_question(
        "Should the conflicting value be overwritten?"
    )

    assert transitions.can_complete() is False

    with pytest.raises(
        RuntimeError,
        match="pending user question",
    ):
        transitions.complete_task()


def test_completion_succeeds_only_after_verified_progress() -> None:
    state, transitions = make_state()

    evidence = EvidenceRecord(
        summary="Required work is verified.",
        source="Current external state",
        kind=EvidenceKind.VERIFICATION,
    )
    claim = ClaimRecord(
        statement="Required work is complete."
    )

    transitions.add_evidence(evidence)
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Complete required work.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    effect = SideEffectRecord(
        description="Submit final result.",
        idempotent=False,
    )
    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    confirmation = EvidenceRecord(
        summary="Server confirms exactly one submission.",
        source="Server-visible result",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(confirmation)
    transitions.confirm_side_effect(
        effect.side_effect_id,
        (confirmation.evidence_id,),
    )

    assert transitions.can_complete() is True

    transitions.complete_task()

    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
