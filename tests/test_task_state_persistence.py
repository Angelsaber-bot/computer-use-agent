"""Tests for versioned TaskState persistence encoding."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from computer_agent.task.models import (
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
from computer_agent.task.persistence import (
    TASK_STATE_SCHEMA_VERSION,
    TaskStatePersistenceError,
    task_state_from_payload,
    task_state_to_payload,
)


def _build_complete_state() -> TaskState:
    observed_at = datetime(
        2026,
        9,
        13,
        14,
        0,
        tzinfo=timezone.utc,
    )
    created_at = datetime(
        2026,
        9,
        13,
        13,
        0,
        tzinfo=timezone.utc,
    )
    updated_at = datetime(
        2026,
        9,
        13,
        14,
        5,
        tzinfo=timezone.utc,
    )

    evidence = EvidenceRecord(
        summary=(
            "The registration form contains "
            "the intended values."
        ),
        source="Current registration form",
        kind=EvidenceKind.VERIFICATION,
        freshness=EvidenceFreshness.CURRENT,
        observed_at=observed_at,
        evidence_id="evidence-form-ready",
        uncertainty="Server submission state is not known.",
    )

    claim = ClaimRecord(
        statement="The registration form is ready.",
        status=ClaimStatus.VERIFIED,
        evidence_ids=(evidence.evidence_id,),
        claim_id="claim-form-ready",
    )

    subgoal = SubgoalRecord(
        description="Prepare the registration form.",
        status=SubgoalStatus.VERIFIED,
        claim_ids=(claim.claim_id,),
        subgoal_id="subgoal-prepare-form",
    )

    effect = SideEffectRecord(
        description="Submit the registration form.",
        state=SideEffectState.UNKNOWN,
        side_effect_id="effect-submit",
        external_reference="pending-registration",
        evidence_ids=(),
        idempotent=False,
        action_key="click_target:submit",
    )

    artifact = ArtifactRecord(
        description="Prepared registration data.",
        location="/tmp/registration.json",
        artifact_id="artifact-registration",
        evidence_ids=(evidence.evidence_id,),
    )

    return TaskState(
        goal=(
            "Submit the registration exactly once "
            "and confirm the result."
        ),
        constraints=(
            "Do not submit more than once.",
            "Do not declare completion without evidence.",
        ),
        task_id="task-phase06-05",
        status=TaskStateStatus.PAUSED,
        subgoals={
            subgoal.subgoal_id: subgoal,
        },
        claims={
            claim.claim_id: claim,
        },
        evidence={
            evidence.evidence_id: evidence,
        },
        side_effects={
            effect.side_effect_id: effect,
        },
        artifacts={
            artifact.artifact_id: artifact,
        },
        pending_questions=(
            "Is server confirmation available?",
        ),
        created_at=created_at,
        updated_at=updated_at,
    )


def test_task_state_round_trip_preserves_complete_state() -> None:
    state = _build_complete_state()

    payload = task_state_to_payload(state)
    restored = task_state_from_payload(payload)

    assert payload["schema_version"] == (
        TASK_STATE_SCHEMA_VERSION
    )
    assert restored == state
    assert restored is not state


def test_round_trip_preserves_unknown_non_idempotent_effect() -> None:
    state = _build_complete_state()

    restored = task_state_from_payload(
        task_state_to_payload(state)
    )

    effect = restored.side_effects[
        "effect-submit"
    ]

    assert effect.state is SideEffectState.UNKNOWN
    assert effect.idempotent is False
    assert effect.action_key == "click_target:submit"


def test_round_trip_preserves_timestamps_and_freshness() -> None:
    state = _build_complete_state()

    restored = task_state_from_payload(
        task_state_to_payload(state)
    )

    evidence = restored.evidence[
        "evidence-form-ready"
    ]

    assert restored.created_at == state.created_at
    assert restored.updated_at == state.updated_at
    assert (
        evidence.observed_at
        == state.evidence[
            "evidence-form-ready"
        ].observed_at
    )
    assert (
        evidence.freshness
        is EvidenceFreshness.CURRENT
    )


def test_unsupported_schema_version_is_rejected() -> None:
    payload = task_state_to_payload(
        _build_complete_state()
    )
    payload["schema_version"] = 999

    with pytest.raises(
        TaskStatePersistenceError,
        match="unsupported task-state schema version",
    ):
        task_state_from_payload(payload)


def test_missing_reference_is_rejected() -> None:
    payload = task_state_to_payload(
        _build_complete_state()
    )

    payload["task_state"]["evidence"] = []

    with pytest.raises(
        TaskStatePersistenceError,
        match="claim references missing evidence",
    ):
        task_state_from_payload(payload)


def test_store_saves_and_loads_complete_state(
    tmp_path,
) -> None:
    from computer_agent.task.persistence import (
        TaskStateStore,
    )

    state = _build_complete_state()
    store = TaskStateStore(tmp_path)

    checkpoint = store.save(state)

    assert checkpoint.is_file()
    assert store.exists(state.task_id)

    restored = store.load(state.task_id)

    assert restored == state
    assert restored is not state


def test_store_overwrites_checkpoint_with_latest_state(
    tmp_path,
) -> None:
    from computer_agent.task.persistence import (
        TaskStateStore,
    )

    state = _build_complete_state()
    store = TaskStateStore(tmp_path)

    first_path = store.save(state)

    state.pending_questions = (
        "Use the latest server status?",
    )

    second_path = store.save(state)

    assert second_path == first_path

    restored = store.load(state.task_id)

    assert restored.pending_questions == (
        "Use the latest server status?",
    )


def test_store_rejects_corrupt_json(
    tmp_path,
) -> None:
    from computer_agent.task.persistence import (
        TaskStateStore,
    )

    state = _build_complete_state()
    store = TaskStateStore(tmp_path)

    checkpoint = store.save(state)
    checkpoint.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    with pytest.raises(
        TaskStatePersistenceError,
        match="invalid JSON",
    ):
        store.load(state.task_id)


def test_store_rejects_mismatched_task_id(
    tmp_path,
) -> None:
    import json

    from computer_agent.task.persistence import (
        TaskStateStore,
    )

    state = _build_complete_state()
    store = TaskStateStore(tmp_path)

    checkpoint = store.save(state)

    payload = json.loads(
        checkpoint.read_text(
            encoding="utf-8"
        )
    )
    payload["task_state"]["task_id"] = (
        "different-task"
    )

    checkpoint.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        TaskStatePersistenceError,
        match="task_id does not match",
    ):
        store.load(state.task_id)


def test_failed_atomic_replace_preserves_old_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    import computer_agent.task.persistence as persistence_module

    from computer_agent.task.persistence import (
        TaskStateStore,
    )

    state = _build_complete_state()
    store = TaskStateStore(tmp_path)

    store.save(state)

    original_questions = (
        state.pending_questions
    )

    state.pending_questions = (
        "This new state must not partially replace "
        "the previous checkpoint.",
    )

    def fail_replace(
        source,
        destination,
    ) -> None:
        raise OSError(
            "simulated atomic replace failure"
        )

    monkeypatch.setattr(
        persistence_module.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        TaskStatePersistenceError,
        match="failed to save task checkpoint",
    ):
        store.save(state)

    restored = store.load(state.task_id)

    assert (
        restored.pending_questions
        == original_questions
    )

    assert not list(
        tmp_path.glob(".task-state-*.tmp")
    )


def test_store_finds_latest_resumable_task(
    tmp_path,
) -> None:
    from datetime import (
        datetime,
        timezone,
    )

    from computer_agent.task import (
        TaskState,
        TaskStateStatus,
        TaskStateStore,
    )

    store = TaskStateStore(
        tmp_path
    )

    older = TaskState(
        goal="Older resumable task",
        task_id="older-resumable",
        status=TaskStateStatus.PAUSED,
        updated_at=datetime(
            2026,
            9,
            13,
            10,
            0,
            tzinfo=timezone.utc,
        ),
    )

    newer_resumable = TaskState(
        goal="Newer resumable task",
        task_id="newer-resumable",
        status=TaskStateStatus.WAITING_USER,
        updated_at=datetime(
            2026,
            9,
            13,
            10,
            30,
            tzinfo=timezone.utc,
        ),
    )

    newest_terminal = TaskState(
        goal="Newest completed task",
        task_id="newest-terminal",
        status=TaskStateStatus.COMPLETED,
        updated_at=datetime(
            2026,
            9,
            13,
            11,
            0,
            tzinfo=timezone.utc,
        ),
    )

    store.save(older)
    store.save(newest_terminal)
    store.save(newer_resumable)

    assert (
        store.latest_resumable_task_id()
        == "newer-resumable"
    )


def test_store_latest_resumable_returns_none_without_checkpoints(
    tmp_path,
) -> None:
    from computer_agent.task import (
        TaskStateStore,
    )

    store = TaskStateStore(
        tmp_path
    )

    assert (
        store.latest_resumable_task_id()
        is None
    )
