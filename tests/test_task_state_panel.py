"""Tests for the visible evidence-grounded Task State workspace."""

from __future__ import annotations

import os

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
)

from computer_agent.app.task_state_panel import TaskStatePanel
from computer_agent.task import (
    ClaimRecord,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SubgoalRecord,
    TaskState,
    TaskStateSnapshot,
    TaskStateStatus,
    TaskStateTransitions,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    yield app


def _build_verified_state() -> tuple[
    TaskState,
    TaskStateTransitions,
    EvidenceRecord,
]:
    state = TaskState(
        goal="Complete the visible semantic workflow."
    )
    transitions = TaskStateTransitions(state)

    evidence = EvidenceRecord(
        summary="Current external state satisfies the requirement.",
        source="Visible test observation",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(evidence)

    claim = ClaimRecord(
        statement="The requirement is satisfied."
    )
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description="Verify the required external state.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    return state, transitions, evidence


def test_task_state_panel_renders_verified_progress(
    qapp,
) -> None:
    state, _, _ = _build_verified_state()

    panel = TaskStatePanel()
    panel.render(
        TaskStateSnapshot.from_state(state)
    )

    progress = panel.findChild(
        QPlainTextEdit,
        "progressView",
    )
    evidence = panel.findChild(
        QPlainTextEdit,
        "evidenceView",
    )

    assert progress is not None
    assert evidence is not None

    assert "[VERIFIED]" in progress.toPlainText()
    assert (
        "Verify the required external state."
        in progress.toPlainText()
    )

    assert "[CURRENT]" in evidence.toPlainText()
    assert (
        "Current external state satisfies the requirement."
        in evidence.toPlainText()
    )


def test_task_state_panel_shows_stale_invalidation(
    qapp,
) -> None:
    state, transitions, evidence = (
        _build_verified_state()
    )

    transitions.set_evidence_freshness(
        evidence.evidence_id,
        EvidenceFreshness.STALE,
    )

    panel = TaskStatePanel()
    panel.render(
        TaskStateSnapshot.from_state(state)
    )

    progress = panel.findChild(
        QPlainTextEdit,
        "progressView",
    )
    evidence_view = panel.findChild(
        QPlainTextEdit,
        "evidenceView",
    )
    completion = panel.findChild(
        QLabel,
        "completionStatus",
    )

    assert progress is not None
    assert evidence_view is not None
    assert completion is not None

    assert "[UNKNOWN]" in progress.toPlainText()
    assert "[STALE]" in evidence_view.toPlainText()
    assert completion.text() == "BLOCKED"


def test_task_state_panel_shows_unknown_side_effect_blocker(
    qapp,
) -> None:
    state, transitions, _ = _build_verified_state()

    effect = SideEffectRecord(
        description="Commit external result.",
        idempotent=False,
    )

    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    panel = TaskStatePanel()
    panel.render(
        TaskStateSnapshot.from_state(state)
    )

    effects = panel.findChild(
        QPlainTextEdit,
        "sideEffectsView",
    )
    completion = panel.findChild(
        QLabel,
        "completionStatus",
    )
    completion_view = panel.findChild(
        QPlainTextEdit,
        "completionView",
    )

    assert effects is not None
    assert completion is not None
    assert completion_view is not None

    assert "[UNKNOWN]" in effects.toPlainText()
    assert (
        "Commit external result."
        in effects.toPlainText()
    )

    assert completion.text() == "BLOCKED"
    assert (
        "side effect is unresolved"
        in completion_view.toPlainText()
    )


def test_task_state_panel_shows_confirmed_completion(
    qapp,
) -> None:
    state, transitions, _ = _build_verified_state()

    effect = SideEffectRecord(
        description="Commit external result.",
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
        summary="Exactly one committed result is visible.",
        source="Visible reconciliation test",
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

    panel = TaskStatePanel()
    panel.render(
        TaskStateSnapshot.from_state(state)
    )

    effects = panel.findChild(
        QPlainTextEdit,
        "sideEffectsView",
    )
    completion = panel.findChild(
        QLabel,
        "completionStatus",
    )
    completion_view = panel.findChild(
        QPlainTextEdit,
        "completionView",
    )

    assert effects is not None
    assert completion is not None
    assert completion_view is not None

    assert "[CONFIRMED]" in effects.toPlainText()
    assert completion.text() == "ALLOWED"
    assert (
        "No semantic completion blockers remain."
        in completion_view.toPlainText()
    )


def test_task_state_bridge_delivers_snapshot(
    qapp,
) -> None:
    from computer_agent.app.task_state_bridge import (
        TaskStateBridge,
    )

    state, _, _ = _build_verified_state()
    snapshot = TaskStateSnapshot.from_state(state)

    bridge = TaskStateBridge()
    received = []

    bridge.snapshot_received.connect(
        received.append
    )

    bridge.publish(snapshot)
    qapp.processEvents()

    assert received == [snapshot]
