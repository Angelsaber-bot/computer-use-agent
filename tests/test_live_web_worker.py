"""Tests for the bounded production live-web workspace worker."""

from __future__ import annotations

import pytest

from computer_agent.app.live_web_worker import (
    BROWSER_WINDOW_ARTIFACT_ID,
    BROWSER_WINDOW_MARKER_PREFIX,
    GO_BUTTON,
    RESULTS_TARGET,
    RESULT_CLAIM_ID,
    RESULT_SUBGOAL_ID,
    SEARCH_FIELD,
    SEARCH_QUERY,
    QUERY_CLAIM_ID,
    QUERY_SUBGOAL_ID,
    _browser_window_marker_from_state,
    _ensure_live_task_structure,
    _search_field_with_expected_value,
    _should_prepare_live_task_segment,
    build_prepare_plan,
    build_resume_plan,
    create_live_web_worker,
    goal_is_supported,
)
from computer_agent.perception import (
    BoundingBox,
    UIElement,
)
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
    WebTextInputStep,
)
from computer_agent.task import (
    ArtifactRecord,
    ClaimStatus,
    EvidenceKind,
    EvidenceRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
    prepare_state_for_resume,
)


GOAL = "Search python.org for typing."


def _element(
    *,
    text: str | None,
    value: str | None,
    element_type: str = "text_field",
    confidence: float = 0.95,
    enabled: bool | None = True,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        text=text,
        value=value,
        confidence=confidence,
        enabled=enabled,
        bounding_box=BoundingBox(
            x=10,
            y=10,
            width=100,
            height=20,
        ),
        source="accessibility",
    )


def test_live_web_goal_support() -> None:
    assert goal_is_supported(
        GOAL
    )

    assert not goal_is_supported(
        "Search Wikipedia for typing."
    )


def test_prepare_plan_contains_only_verified_text_input() -> None:
    plan = build_prepare_plan(
        GOAL
    )

    assert len(plan.steps) == 1

    step = plan.steps[0]

    assert isinstance(
        step,
        WebTextInputStep,
    )

    assert (
        step.operation
        is PlanOperation.TYPE_INTO_TARGET
    )

    assert step.target == SEARCH_FIELD
    assert step.input_text == SEARCH_QUERY
    assert step.max_attempts == 1


def test_resume_plan_contains_only_submit_click() -> None:
    plan = build_resume_plan(
        GOAL
    )

    assert len(plan.steps) == 1

    step = plan.steps[0]

    assert isinstance(
        step,
        PlanStep,
    )

    assert (
        step.operation
        is PlanOperation.CLICK_TARGET
    )

    assert (
        step.action_target
        == GO_BUTTON
    )

    assert (
        step.verification_target
        == RESULTS_TARGET
    )


def test_live_worker_rejects_unsupported_goal_without_actions() -> None:
    state = TaskState(
        goal="Search somewhere else."
    )

    with pytest.raises(
        RuntimeError,
        match="supports only the bounded task",
    ):
        create_live_web_worker(
            state,
            lambda: None,
            lambda snapshot: None,
        )


def test_live_task_structure_is_preregistered_with_pending_result() -> None:
    state = TaskState(
        goal=GOAL
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_live_task_structure(
        transitions
    )

    assert {
        QUERY_CLAIM_ID,
        RESULT_CLAIM_ID,
    } == set(state.claims)
    assert {
        QUERY_SUBGOAL_ID,
        RESULT_SUBGOAL_ID,
    } == set(state.subgoals)
    assert (
        state.claims[QUERY_CLAIM_ID].status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.claims[RESULT_CLAIM_ID].status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.subgoals[QUERY_SUBGOAL_ID].status
        is SubgoalStatus.PENDING
    )
    assert (
        state.subgoals[RESULT_SUBGOAL_ID].status
        is SubgoalStatus.PENDING
    )


def test_live_completion_waits_for_result_subgoal() -> None:
    state = TaskState(
        goal=GOAL
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_live_task_structure(
        transitions
    )

    query_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains typing.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        QUERY_CLAIM_ID,
        (query_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
    )

    blockers = transitions.completion_blockers()

    assert blockers
    assert not transitions.can_complete()
    assert any(
        "subgoal is not verified: "
        "Submit the search and verify the results page."
        in blocker
        for blocker in blockers
    )

    result_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Results marker is visible.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        RESULT_CLAIM_ID,
        (result_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        RESULT_SUBGOAL_ID
    )

    assert transitions.completion_blockers() == ()
    assert transitions.can_complete()


def test_live_stage_selection_uses_persisted_browser_artifact_after_recovery() -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-123",
        status=TaskStateStatus.WAITING_USER,
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_live_task_structure(
        transitions
    )
    query_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains typing.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        QUERY_CLAIM_ID,
        (query_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
    )
    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=(
                BROWSER_WINDOW_ARTIFACT_ID
            ),
            description=(
                "Agent-owned Google Chrome "
                "task window."
            ),
            location=(
                BROWSER_WINDOW_MARKER_PREFIX
                + state.task_id
            ),
        )
    )

    prepare_state_for_resume(
        state
    )

    assert (
        state.claims[QUERY_CLAIM_ID].status
        is ClaimStatus.UNKNOWN
    )
    assert (
        state.subgoals[QUERY_SUBGOAL_ID].status
        is SubgoalStatus.UNKNOWN
    )
    assert not _should_prepare_live_task_segment(
        state
    )


def test_browser_window_marker_round_trip_from_task_state() -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-123",
    )
    marker_url = (
        BROWSER_WINDOW_MARKER_PREFIX
        + state.task_id
    )

    state.artifacts[
        BROWSER_WINDOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=(
            BROWSER_WINDOW_ARTIFACT_ID
        ),
        description=(
            "Agent-owned Chrome window."
        ),
        location=marker_url,
    )

    assert (
        _browser_window_marker_from_state(
            state
        )
        == marker_url
    )


def test_browser_window_marker_requires_persisted_artifact() -> None:
    state = TaskState(
        goal=GOAL
    )

    with pytest.raises(
        RuntimeError,
        match="no owned Chrome window identity",
    ):
        _browser_window_marker_from_state(
            state
        )


def test_browser_window_marker_rejects_wrong_task() -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-123",
    )

    state.artifacts[
        BROWSER_WINDOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=(
            BROWSER_WINDOW_ARTIFACT_ID
        ),
        description=(
            "Agent-owned Chrome window."
        ),
        location=(
            BROWSER_WINDOW_MARKER_PREFIX
            + "task-456"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="does not belong to this task",
    ):
        _browser_window_marker_from_state(
            state
        )


def test_browser_window_marker_rejects_legacy_numeric_checkpoint() -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-123",
    )

    state.artifacts[
        BROWSER_WINDOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=(
            BROWSER_WINDOW_ARTIFACT_ID
        ),
        description=(
            "Agent-owned Chrome window."
        ),
        location="chrome-window-id:48291",
    )

    with pytest.raises(
        RuntimeError,
        match="legacy Chrome window identity format",
    ):
        _browser_window_marker_from_state(
            state
        )


def test_search_field_uses_semantic_grounding_when_value_matches() -> None:
    semantic = _element(
        text="Search This Site",
        value=SEARCH_QUERY,
    )
    other = _element(
        text=None,
        value=SEARCH_QUERY,
    )

    assert (
        _search_field_with_expected_value(
            (semantic, other),
            expected_value=SEARCH_QUERY,
        )
        is semantic
    )


def test_search_field_falls_back_to_unique_current_value_match() -> None:
    field = _element(
        text="typing",
        value=SEARCH_QUERY,
    )

    assert (
        _search_field_with_expected_value(
            (field,),
            expected_value=SEARCH_QUERY,
        )
        is field
    )


def test_search_field_returns_none_when_no_value_matches() -> None:
    assert (
        _search_field_with_expected_value(
            (
                _element(
                    text="typing",
                    value="different",
                ),
            ),
            expected_value=SEARCH_QUERY,
        )
        is None
    )


def test_search_field_returns_none_for_ambiguous_value_matches() -> None:
    assert (
        _search_field_with_expected_value(
            (
                _element(
                    text="typing",
                    value=SEARCH_QUERY,
                ),
                _element(
                    text=None,
                    value=SEARCH_QUERY,
                ),
            ),
            expected_value=SEARCH_QUERY,
        )
        is None
    )


def test_search_field_wrong_semantic_value_does_not_pass() -> None:
    assert (
        _search_field_with_expected_value(
            (
                _element(
                    text="Search This Site",
                    value="different",
                ),
            ),
            expected_value=SEARCH_QUERY,
        )
        is None
    )
