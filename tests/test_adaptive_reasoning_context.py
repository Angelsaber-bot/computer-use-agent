"""Tests for observation-conditioned adaptive reasoning context."""

from __future__ import annotations

from computer_agent.reasoning.adaptive_context import (
    build_adaptive_reasoning_context,
)
from computer_agent.reasoning.adaptive_models import (
    ObservationContext,
    ObservedElement,
)
from computer_agent.task import (
    ClaimRecord,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SubgoalRecord,
    TaskState,
    TaskStateTransitions,
)


def test_observation_context_preserves_current_ui() -> None:
    observation = ObservationContext(
        application_name="Google Chrome",
        window_title="Example Form",
        visible_text=(
            "Submit",
            "Email",
        ),
        elements=(
            ObservedElement(
                text="Submit",
                element_type="button",
                enabled=True,
            ),
        ),
    )

    assert (
        observation.application_name
        == "Google Chrome"
    )
    assert observation.visible_text == (
        "Submit",
        "Email",
    )
    assert (
        observation.elements[0].element_type
        == "button"
    )


def test_context_includes_verified_and_unresolved_progress() -> None:
    state = TaskState(
        goal="Complete the form.",
        constraints=(
            "Do not submit twice.",
        ),
    )
    transitions = TaskStateTransitions(state)

    evidence = EvidenceRecord(
        summary="Required fields are filled.",
        source="Current form",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(evidence)

    claim = ClaimRecord(
        statement="Required fields are ready."
    )
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (evidence.evidence_id,),
    )

    verified = SubgoalRecord(
        description="Fill required fields.",
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(verified)
    transitions.verify_subgoal(
        verified.subgoal_id
    )

    pending = SubgoalRecord(
        description="Submit form."
    )
    transitions.add_subgoal(pending)

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=("Submit",),
        ),
    )

    assert context.goal == "Complete the form."
    assert context.constraints == (
        "Do not submit twice.",
    )
    assert context.verified_subgoals == (
        "Fill required fields.",
    )
    assert any(
        "Submit form." in item
        for item in context.unresolved_subgoals
    )


def test_context_separates_current_and_stale_evidence() -> None:
    state = TaskState(
        goal="Inspect current state."
    )
    transitions = TaskStateTransitions(state)

    current = EvidenceRecord(
        summary="Current result is visible.",
        source="Current page",
        kind=EvidenceKind.OBSERVATION,
    )
    stale = EvidenceRecord(
        summary="Old result was visible.",
        source="Previous page",
        kind=EvidenceKind.OBSERVATION,
        freshness=EvidenceFreshness.STALE,
    )

    transitions.add_evidence(current)
    transitions.add_evidence(stale)

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(),
    )

    assert any(
        "Current result is visible."
        in item
        for item in context.current_evidence
    )

    assert any(
        "stale:" in item
        and "Old result was visible." in item
        for item
        in context.stale_or_unknown_evidence
    )


def test_context_exposes_unknown_side_effect() -> None:
    state = TaskState(
        goal="Submit once."
    )
    transitions = TaskStateTransitions(state)

    effect = SideEffectRecord(
        description="Submit the form.",
        idempotent=False,
        action_key="click_target:submit",
    )
    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=(
                "Submission status",
            ),
        ),
    )

    assert context.unresolved_side_effects == (
        "unknown: Submit the form. | idempotent=False",
    )
    assert context.blocked_action_keys == (
        "click_target:submit",
    )


def test_confirmed_side_effect_does_not_block_action_key() -> None:
    state = TaskState(
        goal="Submit once."
    )
    transitions = TaskStateTransitions(state)

    evidence = EvidenceRecord(
        summary="Submission is confirmed.",
        source="Status page",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(evidence)

    effect = SideEffectRecord(
        description="Submit the form.",
        idempotent=False,
        action_key="click_target:submit",
    )
    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.confirm_side_effect(
        effect.side_effect_id,
        (evidence.evidence_id,),
    )

    context = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=(
                "Submission confirmed",
            ),
        ),
    )

    assert context.blocked_action_keys == ()


def test_context_changes_when_observation_changes() -> None:
    state = TaskState(
        goal="Continue based on current UI."
    )

    first = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=("Continue",),
        ),
    )

    second = build_adaptive_reasoning_context(
        state=state,
        observation=ObservationContext(
            visible_text=("Confirmation",),
        ),
    )

    assert (
        first.observation
        != second.observation
    )
