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
    DurableWebSearchSpec,
    GO_BUTTON,
    LiveCrashPoint,
    PYTHON_WORKFLOW,
    QUERY_ARTIFACT_ID,
    QueryVerificationMode,
    RESULTS_TARGET,
    RESULT_CLAIM_ID,
    RESULT_SUBGOAL_ID,
    SEARCH_FIELD,
    SUBMIT_ACTION_KEY,
    SUBMIT_SIDE_EFFECT_ID,
    ResolvedDurableWebSearchTask,
    WIKIPEDIA_WORKFLOW,
    WORKFLOW_ARTIFACT_ID,
    QUERY_CLAIM_ID,
    QUERY_SUBGOAL_ID,
    _browser_window_marker_from_state,
    _ensure_query_identity,
    _ensure_workflow_identity,
    _maybe_inject_process_crash,
    _ensure_live_task_structure,
    _search_field_with_expected_value,
    _query_condition_satisfied,
    _results_visible,
    _should_prepare_live_task_segment,
    build_prepare_plan,
    build_resume_plan,
    create_live_web_worker,
    goal_is_supported,
    parse_live_web_search_goal,
    resolve_live_web_task,
    resolve_live_web_workflow,
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
SEARCH_QUERY = "typing"
WIKIPEDIA_GOAL = (
    "Search Wikipedia for computer use agent."
)
WIKIPEDIA_QUERY = "computer use agent"


def _resolved_task(
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
    query_text: str = SEARCH_QUERY,
) -> ResolvedDurableWebSearchTask:
    return ResolvedDurableWebSearchTask(
        spec=spec,
        query_text=query_text,
    )


class InjectedCrash(RuntimeError):
    """Raised by tests instead of terminating pytest."""


def _element(
    *,
    text: str | None,
    value: str | None,
    element_type: str = "text_field",
    confidence: float = 0.95,
    enabled: bool | None = True,
    bounding_box: BoundingBox | None = None,
    source: str | None = "accessibility",
) -> UIElement:
    return UIElement(
        element_type=element_type,
        text=text,
        value=value,
        confidence=confidence,
        enabled=enabled,
        bounding_box=bounding_box
        or BoundingBox(
            x=10,
            y=10,
            width=100,
            height=20,
        ),
        source=source,
    )


def _button(
    text: str,
    *,
    bounding_box: BoundingBox | None = None,
) -> UIElement:
    return _element(
        text=text,
        value=None,
        element_type="button",
        bounding_box=bounding_box,
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


def _observation(
    elements: tuple[UIElement, ...],
) -> TextInputObservation:
    return TextInputObservation(
        application_name="Google Chrome",
        viewport=None,
        snapshot=_snapshot(
            elements
        ),
        semantic_elements=(),
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
        spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
        query_text: str | None = None,
    ) -> None:
        del capture_path
        self.mode = mode
        self.spec = spec
        self.query_text = (
            query_text
            if query_text is not None
            else (
                WIKIPEDIA_QUERY
                if spec is WIKIPEDIA_WORKFLOW
                else SEARCH_QUERY
            )
        )
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

    def open_task_site(
        self,
        start_url: str,
        task_marker: str,
    ) -> str:
        del start_url
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
        working_url_prefix: str = "",
    ) -> None:
        del working_url_prefix
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
        submit_box = BoundingBox(
            x=430,
            y=10,
            width=70,
            height=24,
        )

        if self.mode == "empty":
            return (
                _element(
                    text=self.spec.search_field.text,
                    value="",
                ),
                _button(
                    self.spec.submit_target.text
                    or "",
                    bounding_box=submit_box,
                ),
            )

        if self.mode == "query":
            if (
                self.spec.query_verification_mode
                is QueryVerificationMode.VISIBLE_SEARCH_UI
            ):
                return (
                    _element(
                        text=self.query_text,
                        value=self.query_text,
                        element_type="text",
                        bounding_box=BoundingBox(
                            x=120,
                            y=12,
                            width=200,
                            height=20,
                        ),
                    ),
                    _button(
                        self.spec.submit_target.text
                        or "",
                        bounding_box=submit_box,
                    ),
                )

            return (
                _element(
                    text=self.spec.search_field.text,
                    value=self.query_text,
                ),
                _button(
                    self.spec.submit_target.text
                    or "",
                    bounding_box=submit_box,
                ),
            )

        if self.mode == "results":
            return (
                _element(
                    text=self.spec.search_field.text,
                    value=self.query_text,
                ),
                _button(
                    self.spec.submit_target.text
                    or "",
                    bounding_box=submit_box,
                ),
                _element(
                    text=self.spec.result_target.text,
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


class FailedTextInputBookkeepingEnvironment(
    FakeLiveWebEnvironment
):
    def __init__(
        self,
        *,
        capture_path,
        mode: str = "empty",
        spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
        text_input_external_mode: str,
    ) -> None:
        super().__init__(
            capture_path=capture_path,
            mode=mode,
            spec=spec,
        )
        self.text_input_external_mode = (
            text_input_external_mode
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
            self.mode = self.text_input_external_mode
            agent_state = AgentState(
                user_task=plan.task_goal
            )
            agent_state.fail(
                "fake text input bookkeeping failure"
            )
            return AgentLoopResult(
                status=AgentLoopStatus.EXHAUSTED,
                plan=plan,
                state=agent_state,
                completed_plan_steps=0,
                reason=(
                    "fake exhausted text input "
                    "bookkeeping"
                ),
            )

        return super().execute_plan(
            plan
        )


class SubmitBookkeepingEnvironment(
    FakeLiveWebEnvironment
):
    def __init__(
        self,
        *,
        capture_path,
        mode: str = "query",
        spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
        submit_external_mode: str,
        submit_loop_status: AgentLoopStatus,
    ) -> None:
        super().__init__(
            capture_path=capture_path,
            mode=mode,
            spec=spec,
        )
        self.submit_external_mode = submit_external_mode
        self.submit_loop_status = submit_loop_status

    def execute_plan(
        self,
        plan,
    ) -> AgentLoopResult:
        step = plan.steps[0]

        if (
            not isinstance(step, WebTextInputStep)
            and step.operation
            is PlanOperation.CLICK_TARGET
        ):
            self.click_count += 1
            self.mode = self.submit_external_mode
            agent_state = AgentState(
                user_task=plan.task_goal
            )

            if (
                self.submit_loop_status
                is AgentLoopStatus.COMPLETED
            ):
                agent_state.start()
                agent_state.succeed()
                completed_steps = 1
            else:
                agent_state.fail(
                    "fake submit bookkeeping failure"
                )
                completed_steps = 0

            return AgentLoopResult(
                status=self.submit_loop_status,
                plan=plan,
                state=agent_state,
                completed_plan_steps=completed_steps,
                reason="fake submit bookkeeping",
            )

        return super().execute_plan(
            plan
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

    assert goal_is_supported(
        WIKIPEDIA_GOAL
    )

    assert goal_is_supported(
        "Search Wikipedia for typing."
    )

    assert not goal_is_supported(
        "Search Google for typing."
    )


@pytest.mark.parametrize(
    ("goal", "site", "query"),
    (
        (
            "Search Wikipedia for Claude Shannon.",
            "wikipedia",
            "Claude Shannon",
        ),
        (
            "Search Wikipedia for Alan Turing.",
            "wikipedia",
            "Alan Turing",
        ),
        (
            "Search Wikipedia for reinforcement learning",
            "wikipedia",
            "reinforcement learning",
        ),
        (
            "Search python.org for asyncio.",
            "python.org",
            "asyncio",
        ),
        (
            "Search python.org for dataclasses.",
            "python.org",
            "dataclasses",
        ),
        (
            "Search python.org for virtual environments.",
            "python.org",
            "virtual environments",
        ),
        (
            "search wikipedia for alan turing",
            "wikipedia",
            "alan turing",
        ),
    ),
)
def test_live_web_goal_parser(
    goal: str,
    site: str,
    query: str,
) -> None:
    assert parse_live_web_search_goal(
        goal
    ) == (site, query)


@pytest.mark.parametrize(
    "goal",
    (
        "Search Wikipedia for .",
        "Search Wikipedia Claude Shannon.",
        "Search for Claude Shannon.",
    ),
)
def test_live_web_goal_parser_rejects_malformed_or_empty(
    goal: str,
) -> None:
    with pytest.raises(RuntimeError):
        parse_live_web_search_goal(goal)


def test_live_web_goal_parser_rejects_unsupported_site() -> None:
    with pytest.raises(
        RuntimeError,
        match="Unsupported live web search site",
    ):
        parse_live_web_search_goal(
            "Search Google for OpenAI."
        )


def test_python_task_resolves_correctly() -> None:
    resolved_task = resolve_live_web_task(
        "Search python.org for asyncio."
    )
    assert resolved_task is not None
    spec = resolved_task.spec

    assert spec.workflow_id == "python-org-search"
    assert resolved_task.query_text == "asyncio"
    assert spec.search_field == SEARCH_FIELD
    assert spec.submit_target == GO_BUTTON
    assert spec.result_target == RESULTS_TARGET


def test_wikipedia_task_resolves_correctly() -> None:
    resolved_task = resolve_live_web_task(
        "Search Wikipedia for Claude Shannon."
    )
    assert resolved_task is not None
    spec = resolved_task.spec

    assert spec is WIKIPEDIA_WORKFLOW
    assert spec.start_url == (
        "https://en.wikipedia.org/wiki/Main_Page"
    )
    assert spec.working_url_prefix == (
        "https://en.wikipedia.org/"
    )
    assert resolved_task.query_text == "Claude Shannon"
    assert spec.search_field.text == "Search Wikipedia"
    assert spec.submit_target.text == "Search"
    assert spec.result_target.text == "Search results"


def test_resolve_live_web_workflow_returns_static_spec() -> None:
    assert (
        resolve_live_web_workflow(
            "Search python.org for dataclasses."
        )
        is PYTHON_WORKFLOW
    )


def test_unsupported_goal_rejected() -> None:
    with pytest.raises(
        RuntimeError,
        match="Unsupported live web search site",
    ):
        resolve_live_web_task(
            "Search Google for OpenAI."
        )


def test_wikipedia_prepare_plan_uses_spec_values() -> None:
    plan = build_prepare_plan(
        WIKIPEDIA_GOAL,
        WIKIPEDIA_WORKFLOW,
    )

    step = plan.steps[0]

    assert isinstance(
        step,
        WebTextInputStep,
    )
    assert step.target == WIKIPEDIA_WORKFLOW.search_field
    assert step.input_text == WIKIPEDIA_QUERY


def test_wikipedia_resume_plan_uses_spec_values() -> None:
    plan = build_resume_plan(
        WIKIPEDIA_GOAL,
        WIKIPEDIA_WORKFLOW,
    )

    step = plan.steps[0]

    assert isinstance(
        step,
        PlanStep,
    )
    assert (
        step.action_target
        == WIKIPEDIA_WORKFLOW.submit_target
    )
    assert (
        step.verification_target
        == WIKIPEDIA_WORKFLOW.result_target
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
        goal="Search Google for OpenAI."
    )

    with pytest.raises(
        RuntimeError,
        match="Unsupported live web search site",
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


def test_wikipedia_query_checkpoint_crash_happens_before_submit(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        CRASH_ENV_VAR,
        LiveCrashPoint.QUERY_CHECKPOINT.value,
    )
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
    )

    (
        crash_points,
        _published_statuses,
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
    assert decisions == []
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.query_claim_id
        ].status
        is ClaimStatus.VERIFIED
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


def test_generic_structure_creation_uses_spec_ids() -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_live_task_structure(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )

    assert {
        WIKIPEDIA_WORKFLOW.query_claim_id,
        WIKIPEDIA_WORKFLOW.result_claim_id,
    } == set(state.claims)
    assert {
        WIKIPEDIA_WORKFLOW.query_subgoal_id,
        WIKIPEDIA_WORKFLOW.result_subgoal_id,
    } == set(state.subgoals)


def test_workflow_identity_persists() -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_workflow_identity(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )

    artifact = state.artifacts[
        WORKFLOW_ARTIFACT_ID
    ]
    assert artifact.location == (
        "workflow:wikipedia-search"
    )


def test_query_identity_persists_exact_runtime_query() -> None:
    state = TaskState(
        goal="Search Wikipedia for Claude Shannon."
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None

    _ensure_query_identity(
        transitions,
        resolved_task,
    )

    artifact = state.artifacts[
        QUERY_ARTIFACT_ID
    ]
    assert artifact.location == "Claude Shannon"


def test_mismatched_persisted_query_fails_closed() -> None:
    state = TaskState(
        goal="Search Wikipedia for Claude Shannon."
    )
    state.artifacts[
        QUERY_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=QUERY_ARTIFACT_ID,
        description="Durable live web runtime query text.",
        location="Alan Turing",
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None

    with pytest.raises(
        RuntimeError,
        match="query identity",
    ):
        _ensure_query_identity(
            transitions,
            resolved_task,
        )


def test_missing_query_identity_on_resume_fails_closed() -> None:
    state = TaskState(
        goal="Search python.org for asyncio.",
        task_id="missing-query-identity",
    )
    state.artifacts[
        BROWSER_WINDOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=BROWSER_WINDOW_ARTIFACT_ID,
        description="Agent-owned Chrome window.",
        location=(
            BROWSER_WINDOW_MARKER_PREFIX
            + state.task_id
        ),
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None

    with pytest.raises(
        RuntimeError,
        match="no durable web query identity",
    ):
        _ensure_query_identity(
            transitions,
            resolved_task,
        )


def test_mismatched_persisted_workflow_fails_closed() -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    state.artifacts[
        WORKFLOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=WORKFLOW_ARTIFACT_ID,
        description="Durable live web workflow identity.",
        location="workflow:python-org-search",
    )
    transitions = TaskStateTransitions(
        state
    )

    with pytest.raises(
        RuntimeError,
        match="workflow identity",
    ):
        _ensure_workflow_identity(
            transitions,
            WIKIPEDIA_WORKFLOW,
        )


def test_live_worker_mismatched_persisted_workflow_fails_closed(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    state.artifacts[
        WORKFLOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=WORKFLOW_ARTIFACT_ID,
        description="Durable live web workflow identity.",
        location="workflow:python-org-search",
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
    )

    with pytest.raises(
        RuntimeError,
        match="workflow identity",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.open_count == 0
    assert environment.type_count == 0
    assert environment.click_count == 0


def test_live_worker_mismatched_persisted_query_fails_closed(
    monkeypatch,
) -> None:
    state = TaskState(
        goal="Search Wikipedia for Claude Shannon."
    )
    state.artifacts[
        QUERY_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=QUERY_ARTIFACT_ID,
        description="Durable live web runtime query text.",
        location="Alan Turing",
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text="Claude Shannon",
    )

    with pytest.raises(
        RuntimeError,
        match="query identity",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.open_count == 0
    assert environment.type_count == 0
    assert environment.click_count == 0


def test_live_completion_waits_for_result_subgoal() -> None:
    state = TaskState(
        goal=GOAL
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_query_identity(
        transitions,
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
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


def test_wikipedia_worker_runs_through_shared_loop(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 1
    assert environment.type_count == 1
    assert environment.click_count == 1
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.query_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.result_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )
    effect = state.side_effects[
        WIKIPEDIA_WORKFLOW.submit_side_effect_id
    ]
    assert (
        effect.state
        is SideEffectState.CONFIRMED
    )
    assert effect.action_key == (
        WIKIPEDIA_WORKFLOW.submit_action_key
    )
    assert (
        state.artifacts[
            WORKFLOW_ARTIFACT_ID
        ].location
        == "workflow:wikipedia-search"
    )
    assert (
        state.artifacts[
            QUERY_ARTIFACT_ID
        ].location
        == WIKIPEDIA_QUERY
    )


@pytest.mark.parametrize(
    ("goal", "spec", "query_text"),
    (
        (
            "Search Wikipedia for Claude Shannon.",
            WIKIPEDIA_WORKFLOW,
            "Claude Shannon",
        ),
        (
            "Search python.org for asyncio.",
            PYTHON_WORKFLOW,
            "asyncio",
        ),
    ),
)
def test_live_worker_uses_dynamic_runtime_query(
    monkeypatch,
    goal: str,
    spec: DurableWebSearchSpec,
    query_text: str,
) -> None:
    state = TaskState(
        goal=goal
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=spec,
        query_text=query_text,
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.type_count == 1
    assert environment.click_count == 1
    assert (
        state.artifacts[
            QUERY_ARTIFACT_ID
        ].location
        == query_text
    )
    assert any(
        query_text in evidence.summary
        for evidence in state.evidence.values()
    )


def test_dynamic_queries_reuse_workflow_side_effect_identity(
    monkeypatch,
) -> None:
    action_keys: list[str | None] = []

    for query_text in ("Claude Shannon", "Alan Turing"):
        state = TaskState(
            goal=f"Search Wikipedia for {query_text}."
        )
        environment = FakeLiveWebEnvironment(
            capture_path=None,
            mode="empty",
            spec=WIKIPEDIA_WORKFLOW,
            query_text=query_text,
        )

        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )
        action_keys.append(
            state.side_effects[
                WIKIPEDIA_WORKFLOW.submit_side_effect_id
            ].action_key
        )

    assert action_keys == [
        WIKIPEDIA_WORKFLOW.submit_action_key,
        WIKIPEDIA_WORKFLOW.submit_action_key,
    ]


@pytest.mark.parametrize(
    "loop_status",
    (
        AgentLoopStatus.BLOCKED,
        AgentLoopStatus.EXHAUSTED,
    ),
)
def test_submission_loop_failure_accepts_fresh_result_postcondition(
    monkeypatch,
    loop_status: AgentLoopStatus,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = SubmitBookkeepingEnvironment(
        capture_path=None,
        mode="query",
        spec=WIKIPEDIA_WORKFLOW,
        submit_external_mode="results",
        submit_loop_status=loop_status,
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.click_count == 1
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.result_claim_id
        ].status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.side_effects[
            WIKIPEDIA_WORKFLOW.submit_side_effect_id
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_submission_loop_failure_still_fails_when_result_absent(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = SubmitBookkeepingEnvironment(
        capture_path=None,
        mode="query",
        spec=WIKIPEDIA_WORKFLOW,
        submit_external_mode="ambiguous",
        submit_loop_status=AgentLoopStatus.BLOCKED,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "real-web search submission failed: "
            "loop=blocked, state=failed, "
            "completed_steps=0"
        ),
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.result_claim_id
        ].status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.side_effects[
            WIKIPEDIA_WORKFLOW.submit_side_effect_id
        ].state
        is SideEffectState.UNKNOWN
    )


def test_submission_success_still_fails_when_result_absent(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = SubmitBookkeepingEnvironment(
        capture_path=None,
        mode="query",
        spec=WIKIPEDIA_WORKFLOW,
        submit_external_mode="ambiguous",
        submit_loop_status=AgentLoopStatus.COMPLETED,
    )

    with pytest.raises(
        RuntimeError,
        match="did not resolve Results",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert (
        state.side_effects[
            WIKIPEDIA_WORKFLOW.submit_side_effect_id
        ].state
        is SideEffectState.UNKNOWN
    )
    assert (
        state.status
        is TaskStateStatus.RUNNING
    )


def test_wikipedia_query_accepts_fresh_external_postcondition_after_loop_failure(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = FailedTextInputBookkeepingEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        text_input_external_mode="query",
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.type_count == 1
    assert environment.click_count == 1
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.query_claim_id
        ].status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.query_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_query_loop_failure_still_fails_without_fresh_external_postcondition(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    environment = FailedTextInputBookkeepingEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        text_input_external_mode="empty",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "real-web query entry failed: "
            "loop=exhausted, state=failed, "
            "completed_steps=0"
        ),
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.type_count == 1
    assert environment.click_count == 0
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.query_claim_id
        ].status
        is ClaimStatus.UNVERIFIED
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
    _ensure_query_identity(
        transitions,
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
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


def test_wikipedia_restart_after_query_checkpoint_avoids_retyping(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL,
        task_id="wiki-query-checkpoint",
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_workflow_identity(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )
    _ensure_query_identity(
        transitions,
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            WIKIPEDIA_QUERY,
        ),
    )
    _ensure_live_task_structure(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )
    query_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains computer use agent.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        WIKIPEDIA_WORKFLOW.query_claim_id,
        (query_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        WIKIPEDIA_WORKFLOW.query_subgoal_id
    )
    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=(
                WIKIPEDIA_WORKFLOW.workspace_artifact_id
            ),
            description=(
                WIKIPEDIA_WORKFLOW.workspace_description
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
        spec=WIKIPEDIA_WORKFLOW,
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
    _ensure_query_identity(
        transitions,
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
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


def test_wikipedia_restart_after_submit_avoids_duplicate_submit(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_GOAL,
        task_id="wiki-after-submit",
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )
    _ensure_workflow_identity(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )
    _ensure_query_identity(
        transitions,
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            WIKIPEDIA_QUERY,
        ),
    )
    _ensure_live_task_structure(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )
    query_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains computer use agent.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        WIKIPEDIA_WORKFLOW.query_claim_id,
        (query_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        WIKIPEDIA_WORKFLOW.query_subgoal_id
    )
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                WIKIPEDIA_WORKFLOW.submit_side_effect_id
            ),
            description=(
                WIKIPEDIA_WORKFLOW.submit_description
            ),
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=(
                WIKIPEDIA_WORKFLOW.submit_action_key
            ),
        )
    )
    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=(
                WIKIPEDIA_WORKFLOW.workspace_artifact_id
            ),
            description=(
                WIKIPEDIA_WORKFLOW.workspace_description
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
        spec=WIKIPEDIA_WORKFLOW,
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 0
    assert environment.activate_count == 1
    assert environment.type_count == 0
    assert environment.click_count == 0
    assert (
        state.side_effects[
            WIKIPEDIA_WORKFLOW.submit_side_effect_id
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
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
    _ensure_query_identity(
        transitions,
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
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
    _ensure_query_identity(
        transitions,
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
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


def test_search_field_accepts_character_spaced_accessibility_value() -> None:
    field = _element(
        text="Search This Site",
        value="a s y n ci o",
    )

    assert (
        _search_field_with_expected_value(
            (field,),
            expected_value="asyncio",
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


def test_search_field_rejects_non_character_spaced_phrase_compaction() -> None:
    assert (
        _search_field_with_expected_value(
            (
                _element(
                    text="Search This Site",
                    value="virtual environments",
                ),
            ),
            expected_value="virtualenvironments",
        )
        is None
    )


def test_query_condition_field_value_accepts_current_python_field() -> None:
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text="Search This Site",
                    value=SEARCH_QUERY,
                ),
                _button("GO"),
            )
        ),
        _resolved_task(
            PYTHON_WORKFLOW,
            SEARCH_QUERY,
        ),
    )

    assert result is not None
    assert "search field contains 'typing'" in result.summary


def test_wikipedia_query_condition_accepts_autocomplete_visible_ui() -> None:
    query_text = "Claude Shannon"
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text="Address and search bar",
                    value=(
                        "en.wikipedia.org/wiki/"
                        "Main_Page"
                    ),
                    bounding_box=BoundingBox(
                        x=0,
                        y=0,
                        width=400,
                        height=24,
                    ),
                ),
                _element(
                    text=(
                        "Search for pages containing "
                        f"{query_text}"
                    ),
                    value=(
                        "Search for pages containing "
                        f"{query_text}"
                    ),
                    element_type="link",
                    bounding_box=BoundingBox(
                        x=90,
                        y=42,
                        width=330,
                        height=20,
                    ),
                ),
                _button(
                    "Search",
                    bounding_box=BoundingBox(
                        x=430,
                        y=10,
                        width=70,
                        height=24,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            query_text,
        ),
    )

    assert result is not None
    assert (
        "confirms 'Claude Shannon'"
        in result.summary
    )


def test_wikipedia_query_condition_accepts_closed_autocomplete_query_text() -> None:
    query_text = "reinforcement learning"
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text=query_text,
                    value=query_text,
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=12,
                        width=210,
                        height=18,
                    ),
                ),
                _button(
                    "Search",
                    bounding_box=BoundingBox(
                        x=430,
                        y=10,
                        width=70,
                        height=24,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            query_text,
        ),
    )

    assert result is not None


def test_wikipedia_query_condition_rejects_body_text_far_from_search() -> None:
    query_text = "Claude Shannon"
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text=query_text,
                    value=query_text,
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=260,
                        width=210,
                        height=18,
                    ),
                ),
                _button(
                    "Search",
                    bounding_box=BoundingBox(
                        x=430,
                        y=10,
                        width=70,
                        height=24,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            query_text,
        ),
    )

    assert result is None


def test_wikipedia_query_condition_rejects_different_query_near_search() -> None:
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text="Alan Turing",
                    value="Alan Turing",
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=12,
                        width=210,
                        height=18,
                    ),
                ),
                _button(
                    "Search",
                    bounding_box=BoundingBox(
                        x=430,
                        y=10,
                        width=70,
                        height=24,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            "Claude Shannon",
        ),
    )

    assert result is None


def test_wikipedia_query_condition_rejects_when_submit_missing() -> None:
    query_text = "Claude Shannon"
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text=query_text,
                    value=query_text,
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=12,
                        width=210,
                        height=18,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            query_text,
        ),
    )

    assert result is None


def test_wikipedia_results_accept_dynamic_article_heading() -> None:
    assert _results_visible(
        _observation(
            (
                _element(
                    text="Claude Shannon",
                    value="Claude Shannon",
                    element_type="heading",
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            "Claude Shannon",
        ),
    )


def test_wikipedia_query_condition_rejects_ambiguous_near_matches() -> None:
    query_text = "Claude Shannon"
    result = _query_condition_satisfied(
        _observation(
            (
                _element(
                    text=query_text,
                    value=query_text,
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=12,
                        width=210,
                        height=18,
                    ),
                ),
                _element(
                    text=query_text,
                    value=query_text,
                    element_type="text",
                    bounding_box=BoundingBox(
                        x=120,
                        y=44,
                        width=210,
                        height=18,
                    ),
                ),
                _button(
                    "Search",
                    bounding_box=BoundingBox(
                        x=430,
                        y=10,
                        width=70,
                        height=24,
                    ),
                ),
            )
        ),
        _resolved_task(
            WIKIPEDIA_WORKFLOW,
            query_text,
        ),
    )

    assert result is None
