"""Tests for the bounded production live-web workspace worker."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from computer_agent.agent import (
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    TextInputObservation,
)
from computer_agent.app.live_web_worker import (
    BROWSER_WINDOW_ARTIFACT_ID,
    BROWSER_WINDOW_MARKER_PREFIX,
    CRASH_ENV_VAR,
    GO_BUTTON,
    LiveCrashPoint,
    RESULTS_TARGET,
    RESULT_CLAIM_ID,
    RESULT_SUBGOAL_ID,
    SEARCH_FIELD,
    SEARCH_QUERY,
    SUBMIT_ACTION_KEY,
    SUBMIT_SIDE_EFFECT_ID,
    QUERY_CLAIM_ID,
    QUERY_SUBGOAL_ID,
    _browser_window_marker_from_state,
    _maybe_inject_process_crash,
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
    PerceptionSnapshot,
    ScreenFrame,
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
    SideEffectRecord,
    SideEffectState,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
    prepare_state_for_resume,
)
from computer_agent.runtime import (
    RuntimeTask,
)


GOAL = "Search python.org for typing."


class InjectedCrash(RuntimeError):
    """Raised by tests instead of terminating pytest."""


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


def _button(
    text: str,
) -> UIElement:
    return _element(
        text=text,
        value=None,
        element_type="button",
    )


def _snapshot(
    elements: tuple[UIElement, ...],
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("fake.png"),
            pixel_width=10,
            pixel_height=10,
            screen_width=10,
            screen_height=10,
        ),
        image=Image.new(
            "RGB",
            (10, 10),
        ),
        accessibility_elements=elements,
        ocr_elements=(),
        fused_elements=elements,
        warnings=(),
    )


class FakeControl:
    def __init__(self) -> None:
        self.checkpoints = 0

    def checkpoint(self) -> None:
        self.checkpoints += 1


class FakeLiveWebEnvironment:
    def __init__(
        self,
        *,
        capture_path,
        mode: str = "empty",
    ) -> None:
        del capture_path
        self.mode = mode
        self.open_count = 0
        self.activate_count = 0
        self.type_count = 0
        self.click_count = 0
        self.markers: list[str] = []
        self.perception_engine = object()
        self.executor = object()

    def observe(self) -> TextInputObservation:
        return TextInputObservation(
            application_name="Google Chrome",
            viewport=None,
            snapshot=_snapshot(
                self._elements()
            ),
            semantic_elements=(),
        )

    def open_python_org(
        self,
        task_marker: str,
    ) -> str:
        self.open_count += 1
        marker = (
            BROWSER_WINDOW_MARKER_PREFIX
            + task_marker
        )
        self.markers.append(
            marker
        )
        return marker

    def activate_task_chrome_window(
        self,
        marker_url: str,
    ) -> None:
        self.activate_count += 1
        self.markers.append(
            marker_url
        )

    def execute_plan(
        self,
        plan,
    ) -> AgentLoopResult:
        step = plan.steps[0]

        if isinstance(
            step,
            WebTextInputStep,
        ):
            self.type_count += 1
            self.mode = "query"
        elif (
            step.operation
            is PlanOperation.CLICK_TARGET
        ):
            self.click_count += 1
            self.mode = "results"
        else:
            raise AssertionError(
                f"unexpected plan step: {step!r}"
            )

        agent_state = AgentState(
            user_task=plan.task_goal
        )
        agent_state.start()
        agent_state.succeed()

        return AgentLoopResult(
            status=AgentLoopStatus.COMPLETED,
            plan=plan,
            state=agent_state,
            completed_plan_steps=1,
            reason="fake success",
        )

    def _elements(
        self,
    ) -> tuple[UIElement, ...]:
        if self.mode == "empty":
            return (
                _element(
                    text="Search This Site",
                    value="",
                ),
                _button("GO"),
            )

        if self.mode == "query":
            return (
                _element(
                    text="Search This Site",
                    value=SEARCH_QUERY,
                ),
                _button("GO"),
            )

        if self.mode == "results":
            return (
                _element(
                    text="Search This Site",
                    value=SEARCH_QUERY,
                ),
                _button("GO"),
                _element(
                    text="Results",
                    value=None,
                    element_type="heading",
                ),
            )

        if self.mode == "ambiguous":
            return (
                _element(
                    text="Other page",
                    value=None,
                    element_type="static_text",
                ),
            )

        raise AssertionError(
            f"unknown fake mode: {self.mode}"
        )


def _fake_environment_factory(
    environment: FakeLiveWebEnvironment,
):
    def factory(
        *,
        capture_path,
    ):
        del capture_path
        return environment

    return factory


def _run_worker_with_fake_environment(
    monkeypatch,
    state: TaskState,
    environment: FakeLiveWebEnvironment,
):
    published_statuses: list[TaskStateStatus] = []
    decisions = []
    progress_messages: list[str] = []
    control = FakeControl()

    def publish_state() -> None:
        published_statuses.append(
            state.status
        )

    monkeypatch.setattr(
        "computer_agent.app.live_web_worker._execute_agent_plan",
        lambda env, plan: env.execute_plan(
            plan
        ),
    )

    worker = create_live_web_worker(
        state,
        publish_state,
        decisions.append,
        environment_factory=(
            _fake_environment_factory(
                environment
            )
        ),
    )

    worker(
        RuntimeTask(
            goal=state.goal,
            task_id=state.task_id,
        ),
        control,
        progress_messages.append,
    )

    return (
        published_statuses,
        decisions,
        progress_messages,
        control,
    )


def _create_fake_worker_run(
    monkeypatch,
    state: TaskState,
    environment: FakeLiveWebEnvironment,
):
    published_statuses: list[TaskStateStatus] = []
    decisions = []
    progress_messages: list[str] = []
    control = FakeControl()

    def publish_state() -> None:
        published_statuses.append(
            state.status
        )

    monkeypatch.setattr(
        "computer_agent.app.live_web_worker._execute_agent_plan",
        lambda env, plan: env.execute_plan(
            plan
        ),
    )

    worker = create_live_web_worker(
        state,
        publish_state,
        decisions.append,
        environment_factory=(
            _fake_environment_factory(
                environment
            )
        ),
    )

    return (
        worker,
        control,
        progress_messages,
        published_statuses,
        decisions,
    )


def _run_worker_until_injected_crash(
    monkeypatch,
    state: TaskState,
    environment: FakeLiveWebEnvironment,
):
    crash_points: list[str] = []

    def terminate() -> None:
        crash_points.append("terminated")
        raise InjectedCrash(
            "patched process termination"
        )

    monkeypatch.setattr(
        "computer_agent.app.live_web_worker._terminate_process_for_test",
        terminate,
    )

    (
        worker,
        control,
        progress_messages,
        published_statuses,
        decisions,
    ) = _create_fake_worker_run(
        monkeypatch,
        state,
        environment,
    )

    with pytest.raises(
        InjectedCrash,
    ):
        worker(
            RuntimeTask(
                goal=state.goal,
                task_id=state.task_id,
            ),
            control,
            progress_messages.append,
        )

    return (
        crash_points,
        published_statuses,
        decisions,
        progress_messages,
        control,
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


def test_live_crash_env_unset_does_not_terminate(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.delenv(
        CRASH_ENV_VAR,
        raising=False,
    )
    monkeypatch.setattr(
        "computer_agent.app.live_web_worker._terminate_process_for_test",
        lambda: calls.append("terminated"),
    )

    _maybe_inject_process_crash(
        LiveCrashPoint.QUERY_CHECKPOINT
    )

    assert calls == []


def test_live_crash_env_rejects_unsupported_value(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        CRASH_ENV_VAR,
        "query-chekpoint",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Unsupported "
            "COMPUTER_AGENT_CRASH_AFTER"
        ),
    ):
        create_live_web_worker(
            TaskState(
                goal=GOAL
            ),
            lambda: None,
            lambda snapshot: None,
        )


def test_live_query_checkpoint_crash_happens_before_go(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        CRASH_ENV_VAR,
        LiveCrashPoint.QUERY_CHECKPOINT.value,
    )
    state = TaskState(
        goal=GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
    )

    (
        crash_points,
        published_statuses,
        decisions,
        _progress_messages,
        _control,
    ) = _run_worker_until_injected_crash(
        monkeypatch,
        state,
        environment,
    )

    assert crash_points == ["terminated"]
    assert environment.open_count == 1
    assert environment.type_count == 1
    assert environment.click_count == 0
    assert (
        state.status
        is TaskStateStatus.RUNNING
    )
    assert (
        TaskStateStatus.WAITING_USER
        not in published_statuses
    )
    assert (
        state.claims[QUERY_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.subgoals[QUERY_SUBGOAL_ID]
        .status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.claims[RESULT_CLAIM_ID]
        .status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.subgoals[RESULT_SUBGOAL_ID]
        .status
        is SubgoalStatus.PENDING
    )
    assert (
        SUBMIT_SIDE_EFFECT_ID
        not in state.side_effects
    )
    assert decisions == []


def test_live_submit_execution_crash_happens_after_go_before_results(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        CRASH_ENV_VAR,
        LiveCrashPoint.SUBMIT_EXECUTION.value,
    )
    state = TaskState(
        goal=GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
    )

    (
        crash_points,
        published_statuses,
        decisions,
        _progress_messages,
        _control,
    ) = _run_worker_until_injected_crash(
        monkeypatch,
        state,
        environment,
    )

    assert crash_points == ["terminated"]
    assert environment.open_count == 1
    assert environment.type_count == 1
    assert environment.click_count == 1
    assert (
        TaskStateStatus.WAITING_USER
        not in published_statuses
    )
    assert (
        state.claims[QUERY_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.claims[RESULT_CLAIM_ID]
        .status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.subgoals[RESULT_SUBGOAL_ID]
        .status
        is SubgoalStatus.PENDING
    )
    assert (
        state.side_effects[
            SUBMIT_SIDE_EFFECT_ID
        ].state
        is SideEffectState.EXECUTED
    )
    assert any(
        decision.final_decision_type
        == "ACTION"
        and decision.target_text == "GO"
        for decision in decisions
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


def test_live_worker_runs_continuously_without_waiting_for_resume(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
    )

    (
        published_statuses,
        decisions,
        _progress_messages,
        control,
    ) = _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 1
    assert environment.activate_count == 0
    assert environment.type_count == 1
    assert environment.click_count == 1
    assert control.checkpoints >= 3
    assert (
        TaskStateStatus.WAITING_USER
        not in published_statuses
    )
    assert (
        TaskStateStatus.COMPLETED
        in published_statuses
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
    assert (
        state.subgoals[QUERY_SUBGOAL_ID]
        .status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.subgoals[RESULT_SUBGOAL_ID]
        .status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.side_effects[
            SUBMIT_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert len(published_statuses) >= 5
    assert any(
        decision.final_decision_type
        == "ACTION"
        and decision.target_text == "GO"
        for decision in decisions
    )
    assert (
        decisions[-1].final_decision_type
        == "COMPLETE"
    )


def test_live_worker_resume_after_query_checkpoint_does_not_retype(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-query-checkpoint",
        status=TaskStateStatus.PAUSED,
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

    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="query",
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 0
    assert environment.activate_count == 1
    assert environment.type_count == 0
    assert environment.click_count == 1
    assert (
        state.claims[QUERY_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.claims[RESULT_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_live_worker_resume_after_go_click_reconciles_results_without_click(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-after-go-click",
        status=TaskStateStatus.PAUSED,
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
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                SUBMIT_SIDE_EFFECT_ID
            ),
            description=(
                "Submit the python.org search "
                "query through the GO control."
            ),
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=SUBMIT_ACTION_KEY,
        )
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

    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="results",
    )

    (
        _published_statuses,
        decisions,
        _progress_messages,
        _control,
    ) = _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 0
    assert environment.activate_count == 1
    assert environment.type_count == 0
    assert environment.click_count == 0
    assert (
        state.claims[QUERY_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.claims[RESULT_CLAIM_ID]
        .status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.side_effects[
            SUBMIT_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
    assert all(
        decision.target_text != "GO"
        for decision in decisions
        if decision.final_decision_type
        == "ACTION"
    )
    assert (
        decisions[-1].final_decision_type
        == "COMPLETE"
    )


def test_live_worker_fails_closed_on_ambiguous_external_state(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL,
        task_id="task-ambiguous",
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_live_task_structure(
        transitions
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
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="ambiguous",
    )

    with pytest.raises(
        RuntimeError,
        match="ambiguous",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.type_count == 0
    assert environment.click_count == 0
    assert (
        state.status
        is TaskStateStatus.RUNNING
    )


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
