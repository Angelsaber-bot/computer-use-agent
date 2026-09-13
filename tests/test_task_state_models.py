"""Tests for evidence-grounded semantic task-state models."""

from __future__ import annotations

from datetime import datetime

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
)


def test_task_state_defaults_to_pending() -> None:
    state = TaskState(
        goal="Complete the current administrative workflow."
    )

    assert state.task_id
    assert state.status is TaskStateStatus.PENDING
    assert state.subgoals == {}
    assert state.claims == {}
    assert state.evidence == {}
    assert state.side_effects == {}
    assert state.artifacts == {}
    assert state.pending_questions == ()
    assert isinstance(state.created_at, datetime)
    assert isinstance(state.updated_at, datetime)


def test_task_state_preserves_immutable_constraints() -> None:
    state = TaskState(
        goal="Update the tracker.",
        constraints=(
            "Do not overwrite conflicting values.",
            "Do not create duplicate rows.",
        ),
    )

    assert state.constraints == (
        "Do not overwrite conflicting values.",
        "Do not create duplicate rows.",
    )


@pytest.mark.parametrize(
    "goal",
    [
        "",
        "   ",
    ],
)
def test_task_state_rejects_empty_goal(
    goal: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="goal must be a non-empty string",
    ):
        TaskState(goal=goal)


def test_evidence_records_source_time_and_freshness() -> None:
    evidence = EvidenceRecord(
        summary="Tracker row is marked Research Needed.",
        source="College Tracker / row 4",
        kind=EvidenceKind.OBSERVATION,
    )

    assert evidence.evidence_id
    assert (
        evidence.freshness
        is EvidenceFreshness.CURRENT
    )
    assert evidence.observed_at.tzinfo is not None


def test_evidence_can_preserve_uncertainty() -> None:
    evidence = EvidenceRecord(
        summary="Submission button was clicked.",
        source="Browser observation",
        kind=EvidenceKind.OBSERVATION,
        freshness=EvidenceFreshness.CURRENT,
        uncertainty=(
            "No server-visible confirmation has been observed."
        ),
    )

    assert evidence.uncertainty is not None


def test_claim_can_reference_supporting_evidence() -> None:
    evidence = EvidenceRecord(
        summary="Downloaded PDF exists in target folder.",
        source="Local filesystem",
        kind=EvidenceKind.ARTIFACT,
    )

    claim = ClaimRecord(
        statement="The requested PDF is present.",
        status=ClaimStatus.VERIFIED,
        evidence_ids=(evidence.evidence_id,),
    )

    assert claim.status is ClaimStatus.VERIFIED
    assert claim.evidence_ids == (
        evidence.evidence_id,
    )


def test_subgoal_can_reference_claims() -> None:
    claim = ClaimRecord(
        statement="Official document has been obtained."
    )

    subgoal = SubgoalRecord(
        description="Obtain the official document.",
        status=SubgoalStatus.ACTIVE,
        claim_ids=(claim.claim_id,),
    )

    assert subgoal.status is SubgoalStatus.ACTIVE
    assert subgoal.claim_ids == (
        claim.claim_id,
    )


def test_side_effect_defaults_to_intended() -> None:
    effect = SideEffectRecord(
        description="Submit the registration form."
    )

    assert effect.state is SideEffectState.INTENDED
    assert effect.external_reference is None
    assert effect.idempotent is None


def test_side_effect_can_be_unknown_after_execution() -> None:
    effect = SideEffectRecord(
        description="Submit the registration form.",
        state=SideEffectState.UNKNOWN,
        external_reference="registration-42",
        idempotent=False,
    )

    assert effect.state is SideEffectState.UNKNOWN
    assert effect.idempotent is False


def test_artifact_tracks_external_location() -> None:
    artifact = ArtifactRecord(
        description="Official admissions PDF",
        location=(
            "/Users/demo/College Research/"
            "Example University/admissions.pdf"
        ),
    )

    assert artifact.artifact_id
    assert artifact.location.endswith(
        "admissions.pdf"
    )


def test_records_are_immutable() -> None:
    evidence = EvidenceRecord(
        summary="Observed current page.",
        source="Accessibility snapshot",
        kind=EvidenceKind.OBSERVATION,
    )

    with pytest.raises(
        AttributeError,
    ):
        evidence.summary = "Changed"  # type: ignore[misc]


def test_task_state_touch_updates_timestamp() -> None:
    state = TaskState(goal="Test timestamp update.")
    original = state.updated_at

    state.touch()

    assert state.updated_at >= original
