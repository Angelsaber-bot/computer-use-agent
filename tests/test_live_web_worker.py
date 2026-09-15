"""Tests for the bounded production live-web workspace worker."""

from __future__ import annotations

import json
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
    DURABLE_PLAN_ARTIFACT_ID,
    DurablePostconditionKind,
    DurableStepKind,
    DurableWebSearchSpec,
    FOLLOWUP_DESTINATION_ARTIFACT_ID,
    FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID,
    FOLLOWUP_TARGET_ARTIFACT_ID,
    GO_BUTTON,
    LiveCrashPoint,
    LiveWebEnvironment,
    LiveWebPerformanceMetrics,
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
    _ensure_durable_plan_identity,
    _ensure_expected_workspace_observation,
    _ensure_followup_identity,
    _ensure_query_identity,
    _ensure_workflow_identity,
    _followup_destination_postcondition,
    _ground_followup_link,
    _maybe_inject_process_crash,
    _ensure_live_task_structure,
    _upgrade_observation_with_ocr_if_needed,
    _search_field_with_expected_value,
    _query_condition_satisfied,
    _results_visible,
    _should_prepare_live_task_segment,
    compile_durable_task_plan,
    durable_plan_canonical_json,
    build_prepare_plan,
    build_followup_plan,
    build_resume_plan,
    create_live_web_worker,
    goal_is_supported,
    parse_live_web_search_goal,
    resolve_live_web_task,
    resolve_live_web_workflow,
)
from computer_agent.perception import (
    AccessibilitySnapshot,
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
    Viewport,
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
WIKIPEDIA_FOLLOWUP_GOAL = (
    "Search Wikipedia for Claude Shannon "
    "and open Information theory."
)
WIKIPEDIA_FOLLOWUP_QUERY = "Claude Shannon"
WIKIPEDIA_FOLLOWUP_TARGET = "Information theory"


def _resolved_task(
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
    query_text: str = SEARCH_QUERY,
) -> ResolvedDurableWebSearchTask:
    return ResolvedDurableWebSearchTask(
        spec=spec,
        query_text=query_text,
    )


def _ensure_plan_identity_for_goal(
    transitions: TaskStateTransitions,
    goal: str,
) -> None:
    resolved_task = resolve_live_web_task(
        goal
    )
    assert resolved_task is not None
    durable_plan = compile_durable_task_plan(
        resolved_task
    )
    _ensure_durable_plan_identity(
        transitions,
        resolved_task,
        durable_plan,
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


class FakeLivePerceptionEngine:
    def __init__(
        self,
        snapshots: tuple[PerceptionSnapshot, ...],
    ) -> None:
        self.snapshots = list(snapshots)
        self.include_ocr_values: list[bool] = []
        self.upgraded_snapshots: list[PerceptionSnapshot] = []

    def observe(
        self,
        *,
        include_ocr: bool = True,
    ) -> PerceptionSnapshot:
        self.include_ocr_values.append(include_ocr)
        return self.snapshots.pop(0)

    def with_ocr(
        self,
        snapshot: PerceptionSnapshot,
    ) -> PerceptionSnapshot:
        self.upgraded_snapshots.append(snapshot)
        ocr_element = _element(
            text="Results",
            value=None,
            element_type="text",
            source="ocr",
        )
        return PerceptionSnapshot(
            frame=snapshot.frame,
            image=snapshot.image,
            accessibility_elements=snapshot.accessibility_elements,
            ocr_elements=(ocr_element,),
            fused_elements=(
                *snapshot.accessibility_elements,
                ocr_element,
            ),
            warnings=snapshot.warnings,
            timings={
                **snapshot.timings,
                "ocr": 0.25,
                "ocr_executed": 1.0,
                "fusion": snapshot.timings.get("fusion", 0.0) + 0.05,
                "ocr_upgrade_total": 0.30,
            },
            accessibility_snapshot=snapshot.accessibility_snapshot,
        )


class FailingSeparateAccessibilityReads:
    def read_frontmost_application_name(self):
        raise AssertionError("separate app read should not run")

    def read_frontmost_viewport(self):
        raise AssertionError("separate viewport read should not run")

    def read_frontmost_semantic_elements(self):
        raise AssertionError("separate semantic read should not run")


def _live_environment_shell(
    tmp_path,
    snapshot: PerceptionSnapshot,
) -> tuple[LiveWebEnvironment, FakeLivePerceptionEngine]:
    environment = object.__new__(
        LiveWebEnvironment
    )
    environment.capture_path = tmp_path / "capture.png"
    environment.stabilization_seconds = 0.0
    engine = FakeLivePerceptionEngine(
        (snapshot,)
    )
    environment.perception_engine = engine
    environment.accessibility = FailingSeparateAccessibilityReads()
    environment.last_observation_timings = {}
    environment.performance_metrics = LiveWebPerformanceMetrics()
    return environment, engine


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
        followup_target_text: str | None = None,
        followup_link_value: str | None = None,
        destination_heading_text: str | None = None,
        address_bar_value: str | None = None,
        click_external_modes: tuple[str, ...] = (),
        click_loop_statuses: tuple[AgentLoopStatus, ...] = (),
        frontmost_sequence: tuple[str | None, ...] = (),
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
        self.followup_target_text = (
            followup_target_text
        )
        self.followup_link_value = followup_link_value
        self.destination_heading_text = (
            destination_heading_text
        )
        self.address_bar_value = address_bar_value
        self.click_external_modes = list(
            click_external_modes
        )
        self.click_loop_statuses = list(
            click_loop_statuses
        )
        self.frontmost_sequence = list(
            frontmost_sequence
        )
        self.open_count = 0
        self.activate_count = 0
        self.frontmost_probe_count = 0
        self.type_count = 0
        self.click_count = 0
        self.submit_click_count = 0
        self.followup_click_count = 0
        self.click_reference_points: list[
            tuple[float, float] | None
        ] = []
        self.markers: list[str] = []
        self.perception_engine = object()
        self.executor = object()
        self.last_observation_timings = {
            "observe_total": 0.0,
            "accessibility_controls": 0.0,
            "ocr": 0.0,
            "fusion": 0.0,
            "stabilization": 0.0,
        }

    def frontmost_application_name(self) -> str | None:
        self.frontmost_probe_count += 1
        if self.frontmost_sequence:
            return self.frontmost_sequence.pop(0)
        return "Google Chrome"

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
            loop_status = AgentLoopStatus.COMPLETED
            completed_steps = 1
        elif (
            step.operation
            is PlanOperation.CLICK_TARGET
        ):
            target_text = getattr(
                step.action_target,
                "text",
                None,
            )
            self.click_reference_points.append(
                getattr(
                    step.action_target,
                    "reference_point",
                    None,
                )
            )
            self.click_count += 1
            if (
                target_text
                == self.followup_target_text
            ):
                self.followup_click_count += 1
                default_mode = "destination"
            else:
                self.submit_click_count += 1
                default_mode = "results"
            self.mode = (
                self.click_external_modes.pop(0)
                if self.click_external_modes
                else default_mode
            )
            loop_status = (
                self.click_loop_statuses.pop(0)
                if self.click_loop_statuses
                else AgentLoopStatus.COMPLETED
            )
            completed_steps = (
                1
                if loop_status
                is AgentLoopStatus.COMPLETED
                else 0
            )
        else:
            raise AssertionError(
                f"unexpected plan step: {step!r}"
            )

        agent_state = AgentState(
            user_task=plan.task_goal
        )
        if loop_status is AgentLoopStatus.COMPLETED:
            agent_state.start()
            agent_state.succeed()
        else:
            agent_state.fail(
                "fake click bookkeeping failure"
            )

        return AgentLoopResult(
            status=loop_status,
            plan=plan,
            state=agent_state,
            completed_plan_steps=completed_steps,
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
            elements = [
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
            ]
            if self.followup_target_text is not None:
                elements.append(
                    _element(
                        text=self.followup_target_text,
                        value=self.followup_link_value,
                        element_type="link",
                        bounding_box=BoundingBox(
                            x=30,
                            y=100,
                            width=220,
                            height=20,
                        ),
                    )
                )
            return tuple(elements)

        if self.mode == "missing_link":
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

        if self.mode == "ambiguous_link":
            assert self.followup_target_text is not None
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
                _element(
                    text=self.followup_target_text,
                    value=None,
                    element_type="link",
                    bounding_box=BoundingBox(
                        x=30,
                        y=100,
                        width=220,
                        height=20,
                    ),
                ),
                _element(
                    text=self.followup_target_text,
                    value=None,
                    element_type="link",
                    bounding_box=BoundingBox(
                        x=30,
                        y=140,
                        width=220,
                        height=20,
                    ),
                ),
            )

        if self.mode == "unsafe_link":
            assert self.followup_target_text is not None
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
                _element(
                    text=self.followup_target_text,
                    value=(
                        "https://en.wikipedia.org/wiki/"
                        "Information_theory"
                    ),
                    element_type="link",
                    enabled=False,
                    bounding_box=BoundingBox(
                        x=30,
                        y=100,
                        width=220,
                        height=20,
                    ),
                ),
            )

        if self.mode in (
            "ambiguous_link_same_url",
            "ambiguous_link_different_url",
        ):
            assert self.followup_target_text is not None
            if self.mode == "ambiguous_link_same_url":
                first_url = second_url = (
                    "https://en.wikipedia.org/wiki/"
                    "Information_theory"
                )
            else:
                first_url = (
                    "https://en.wikipedia.org/wiki/"
                    "Information_theory"
                )
                second_url = (
                    "https://en.wikipedia.org/wiki/"
                    "Information"
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
                _element(
                    text=self.spec.result_target.text,
                    value=None,
                    element_type="heading",
                ),
                _element(
                    text=self.followup_target_text,
                    value=first_url,
                    element_type="link",
                    bounding_box=BoundingBox(
                        x=80,
                        y=140,
                        width=220,
                        height=20,
                    ),
                ),
                _element(
                    text=self.followup_target_text,
                    value=second_url,
                    element_type="link",
                    bounding_box=BoundingBox(
                        x=30,
                        y=100,
                        width=220,
                        height=20,
                    ),
                ),
            )

        if self.mode == "destination":
            assert self.followup_target_text is not None
            return (
                _element(
                    text=(
                        self.destination_heading_text
                        or self.followup_target_text
                    ),
                    value=None,
                    element_type="heading",
                    bounding_box=BoundingBox(
                        x=10,
                        y=60,
                        width=280,
                        height=32,
                    ),
                ),
            )

        if self.mode == "canonical_destination":
            return (
                _element(
                    text="Address and search bar",
                    value=(
                        self.address_bar_value
                        or "https://en.wikipedia.org/wiki/"
                        "Information_theory"
                    ),
                    element_type="text_field",
                    bounding_box=BoundingBox(
                        x=100,
                        y=10,
                        width=300,
                        height=20,
                    ),
                ),
                _element(
                    text=(
                        self.destination_heading_text
                        or "Information theory"
                    ),
                    value=None,
                    element_type="heading",
                    bounding_box=BoundingBox(
                        x=10,
                        y=60,
                        width=280,
                        height=32,
                    ),
                ),
            )

        if self.mode == "wrong_destination":
            return (
                _element(
                    text="Wrong destination",
                    value=None,
                    element_type="heading",
                    bounding_box=BoundingBox(
                        x=10,
                        y=60,
                        width=280,
                        height=32,
                    ),
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


class WrongApplicationEnvironment(FakeLiveWebEnvironment):
    def observe(self) -> TextInputObservation:
        observation = super().observe()
        return TextInputObservation(
            application_name="TextEdit",
            viewport=observation.viewport,
            snapshot=observation.snapshot,
            semantic_elements=observation.semantic_elements,
        )


class DelayedReadyEnvironment(FakeLiveWebEnvironment):
    def __init__(
        self,
        *,
        capture_path,
        mode: str = "empty",
        ready_mode: str = "empty",
        initial_unready_observes: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            capture_path=capture_path,
            mode=mode,
            **kwargs,
        )
        self.ready_mode = ready_mode
        self.initial_unready_observes = initial_unready_observes
        self.observe_count = 0

    def observe(self) -> TextInputObservation:
        self.observe_count += 1
        if self.observe_count > self.initial_unready_observes:
            if self.mode == "ambiguous":
                self.mode = self.ready_mode
        return super().observe()


class DelayedPostActionEnvironment(FakeLiveWebEnvironment):
    def __init__(
        self,
        *,
        capture_path,
        delayed_mode: str,
        final_mode: str,
        **kwargs,
    ) -> None:
        super().__init__(
            capture_path=capture_path,
            **kwargs,
        )
        self.delayed_mode = delayed_mode
        self.final_mode = final_mode
        self.delayed_observed = False

    def execute_plan(
        self,
        plan,
    ) -> AgentLoopResult:
        result = super().execute_plan(plan)
        step = plan.steps[0]
        if (
            not isinstance(step, WebTextInputStep)
            and step.operation is PlanOperation.CLICK_TARGET
        ):
            self.mode = self.delayed_mode
            self.delayed_observed = False
        return result

    def observe(self) -> TextInputObservation:
        observation = super().observe()
        if self.mode == self.delayed_mode:
            if self.delayed_observed:
                self.mode = self.final_mode
            else:
                self.delayed_observed = True
        return observation


class DelayedFollowupLinkEnvironment(
    FakeLiveWebEnvironment
):
    def __init__(
        self,
        *,
        capture_path,
        ready_mode: str,
        missing_link_observes: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            capture_path=capture_path,
            **kwargs,
        )
        self.ready_mode = ready_mode
        self.missing_link_observes = missing_link_observes
        self.missing_link_observe_count = 0

    def observe(self) -> TextInputObservation:
        if self.mode == "missing_link":
            self.missing_link_observe_count += 1
            if (
                self.missing_link_observe_count
                > self.missing_link_observes
            ):
                self.mode = self.ready_mode

        return super().observe()


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


def _state_with_browser_artifact(
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> TaskState:
    state = TaskState(
        goal=GOAL
    )
    state.artifacts[
        spec.workspace_artifact_id
    ] = ArtifactRecord(
        artifact_id=spec.workspace_artifact_id,
        description=spec.workspace_description,
        location=BROWSER_WINDOW_MARKER_PREFIX + state.task_id,
    )
    return state


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
    ("goal", "site", "query", "followup"),
    (
        (
            "Search Wikipedia for Claude Shannon.",
            "wikipedia",
            "Claude Shannon",
            None,
        ),
        (
            "Search Wikipedia for Alan Turing.",
            "wikipedia",
            "Alan Turing",
            None,
        ),
        (
            "Search Wikipedia for reinforcement learning",
            "wikipedia",
            "reinforcement learning",
            None,
        ),
        (
            "Search python.org for asyncio.",
            "python.org",
            "asyncio",
            None,
        ),
        (
            "Search python.org for dataclasses.",
            "python.org",
            "dataclasses",
            None,
        ),
        (
            "Search python.org for virtual environments.",
            "python.org",
            "virtual environments",
            None,
        ),
        (
            "search wikipedia for alan turing",
            "wikipedia",
            "alan turing",
            None,
        ),
        (
            "Search Wikipedia for Claude Shannon and open Information theory.",
            "wikipedia",
            "Claude Shannon",
            "Information theory",
        ),
        (
            "Search Wikipedia for Alan Turing and open Turing machine.",
            "wikipedia",
            "Alan Turing",
            "Turing machine",
        ),
        (
            "search wikipedia for alan turing and open turing machine",
            "wikipedia",
            "alan turing",
            "turing machine",
        ),
    ),
)
def test_live_web_goal_parser(
    goal: str,
    site: str,
    query: str,
    followup: str | None,
) -> None:
    assert parse_live_web_search_goal(
        goal
    ) == (site, query, followup)


@pytest.mark.parametrize(
    "goal",
    (
        "Search Wikipedia for .",
        "Search Wikipedia Claude Shannon.",
        "Search for Claude Shannon.",
        "Search Wikipedia for Claude Shannon and open .",
        "Search Wikipedia for Claude Shannon and open.",
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


def test_live_web_goal_parser_rejects_python_followup() -> None:
    with pytest.raises(
        RuntimeError,
        match="Follow-up navigation",
    ):
        parse_live_web_search_goal(
            "Search python.org for asyncio and open documentation."
        )


def test_expected_workspace_observation_does_not_activate_when_chrome_frontmost():
    state = _state_with_browser_artifact()
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        frontmost_sequence=("Google Chrome",),
    )
    observation = environment.observe()
    messages: list[str] = []

    result = _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=PYTHON_WORKFLOW,
        observation=observation,
        refresh=False,
        progress=messages.append,
    )

    assert result is observation
    assert environment.activate_count == 0
    assert not messages


def test_expected_workspace_observation_reacquires_when_ui_steals_focus():
    state = _state_with_browser_artifact()
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        frontmost_sequence=("Python",),
    )
    messages: list[str] = []

    _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=PYTHON_WORKFLOW,
        observation=environment.observe(),
        refresh=False,
        progress=messages.append,
    )

    assert environment.activate_count == 1
    assert environment.markers[-1] == (
        BROWSER_WINDOW_MARKER_PREFIX + state.task_id
    )
    assert any("exact Agent-owned Chrome" in message for message in messages)


def test_expected_workspace_post_action_reacquires_before_refresh():
    state = _state_with_browser_artifact()
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="results",
        frontmost_sequence=("Python",),
    )

    _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=PYTHON_WORKFLOW,
        observation=None,
        refresh=True,
        progress=lambda message: None,
    )

    assert environment.activate_count == 1


def test_expected_workspace_reacquisition_failure_fails_closed():
    state = _state_with_browser_artifact()
    environment = WrongApplicationEnvironment(
        capture_path=None,
        mode="empty",
        frontmost_sequence=("Python",),
    )

    with pytest.raises(
        RuntimeError,
        match="expected 'Google Chrome'",
    ):
        _ensure_expected_workspace_observation(
            state=state,
            environment=environment,
            spec=PYTHON_WORKFLOW,
            observation=environment.observe(),
            refresh=False,
            progress=lambda message: None,
        )

    assert environment.activate_count == 1


def test_live_worker_reobserves_once_when_new_workspace_not_ready(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL
    )
    environment = DelayedReadyEnvironment(
        capture_path=None,
        mode="ambiguous",
        ready_mode="empty",
        initial_unready_observes=1,
    )

    (
        _published_statuses,
        _decisions,
        progress_messages,
        _control,
    ) = _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.type_count == 1
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
    assert any(
        "re-observing before any action" in message
        for message in progress_messages
    )


def test_live_worker_reobserves_post_action_without_reclicking(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=GOAL
    )
    environment = DelayedPostActionEnvironment(
        capture_path=None,
        mode="empty",
        delayed_mode="ambiguous",
        final_mode="results",
    )

    (
        _published_statuses,
        _decisions,
        progress_messages,
        _control,
    ) = _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.submit_click_count == 1
    assert (
        state.side_effects[
            SUBMIT_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert any(
        "post-action state is not verified yet" in message
        for message in progress_messages
    )


def test_live_environment_observe_uses_single_consolidated_ax_snapshot(
    tmp_path,
) -> None:
    search = _element(
        text="Search Wikipedia",
        value="Alan Turing",
        element_type="text_field",
        source="accessibility",
    )
    semantic = AccessibilitySnapshot(
        application_name="Google Chrome",
        controls=(search,),
        semantic_elements=(),
        web_areas=(
            BoundingBox(
                x=0,
                y=100,
                width=1200,
                height=700,
            ),
        ),
        viewport=Viewport(
            BoundingBox(
                x=0,
                y=100,
                width=1200,
                height=700,
            )
        ),
        focused_window_bounds=BoundingBox(
            x=0,
            y=0,
            width=1200,
            height=800,
        ),
        traversed_nodes=42,
        tree_traversals=1,
    )
    snapshot = PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=tmp_path / "screen.png",
            pixel_width=10,
            pixel_height=10,
            screen_width=10,
            screen_height=10,
        ),
        image=Image.new("RGB", (10, 10)),
        accessibility_elements=(search,),
        ocr_elements=(),
        fused_elements=(search,),
        warnings=(),
        timings={
            "accessibility_controls": 0.4,
            "accessibility_tree_traversals": 1.0,
            "accessibility_nodes": 42.0,
            "ocr": 0.0,
            "ocr_executed": 0.0,
            "fusion": 0.02,
            "perception_total": 0.5,
        },
        accessibility_snapshot=semantic,
    )
    environment, engine = _live_environment_shell(
        tmp_path,
        snapshot,
    )

    observation = environment.observe(
        include_ocr=False,
    )

    assert engine.include_ocr_values == [False]
    assert observation.application_name == "Google Chrome"
    assert observation.viewport == semantic.viewport
    assert observation.snapshot.fused_elements == (search,)
    assert (
        environment.performance_metrics.observation_count
        == 1
    )
    assert (
        environment.performance_metrics.ax_tree_traversal_count
        == 1
    )
    assert (
        environment.performance_metrics.ocr_call_count
        == 0
    )


def test_live_environment_same_capture_ocr_upgrade_preserves_fresh_snapshot(
    tmp_path,
) -> None:
    ax_element = _element(
        text="Search This Site",
        value="asyncio",
        element_type="text_field",
        source="accessibility",
    )
    ax_snapshot = AccessibilitySnapshot(
        application_name="Google Chrome",
        controls=(ax_element,),
        semantic_elements=(),
        web_areas=(),
        viewport=None,
        focused_window_bounds=None,
        traversed_nodes=3,
        tree_traversals=1,
    )
    snapshot = PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=tmp_path / "screen.png",
            pixel_width=10,
            pixel_height=10,
            screen_width=10,
            screen_height=10,
        ),
        image=Image.new("RGB", (10, 10)),
        accessibility_elements=(ax_element,),
        ocr_elements=(),
        fused_elements=(ax_element,),
        warnings=(),
        timings={
            "accessibility_controls": 0.1,
            "accessibility_tree_traversals": 1.0,
            "ocr": 0.0,
            "ocr_executed": 0.0,
            "fusion": 0.01,
            "observe_total": 0.2,
        },
        accessibility_snapshot=ax_snapshot,
    )
    environment, engine = _live_environment_shell(
        tmp_path,
        snapshot,
    )
    observation = environment.observe()
    messages: list[str] = []

    upgraded = _upgrade_observation_with_ocr_if_needed(
        environment,
        observation,
        progress=messages.append,
    )

    assert engine.upgraded_snapshots == [
        observation.snapshot,
    ]
    assert upgraded is not observation
    assert upgraded.snapshot.frame is observation.snapshot.frame
    assert upgraded.snapshot.image is observation.snapshot.image
    assert upgraded.snapshot.ocr_elements
    assert (
        environment.performance_metrics.observation_count
        == 1
    )
    assert (
        environment.performance_metrics.ocr_call_count
        == 1
    )
    assert any(
        "AX-only browser evidence was insufficient" in message
        for message in messages
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


def test_wikipedia_followup_task_resolves_runtime_target() -> None:
    resolved_task = resolve_live_web_task(
        WIKIPEDIA_FOLLOWUP_GOAL
    )
    assert resolved_task is not None

    assert resolved_task.spec is WIKIPEDIA_WORKFLOW
    assert (
        resolved_task.query_text
        == WIKIPEDIA_FOLLOWUP_QUERY
    )
    assert (
        resolved_task.followup_target_text
        == WIKIPEDIA_FOLLOWUP_TARGET
    )


def test_wikipedia_search_only_durable_plan_shape() -> None:
    resolved_task = resolve_live_web_task(
        "Search Wikipedia for Claude Shannon."
    )
    assert resolved_task is not None

    plan = compile_durable_task_plan(
        resolved_task
    )

    assert plan.version == 1
    assert plan.workflow_id == "wikipedia-search"
    assert [step.step_id for step in plan.steps] == [
        "enter-query",
        "submit-search",
    ]
    assert [step.kind for step in plan.steps] == [
        DurableStepKind.ENTER_TEXT,
        DurableStepKind.ACTIVATE_CONTROL,
    ]
    assert plan.steps[0].input_text == "Claude Shannon"
    assert plan.steps[0].claim_id == (
        WIKIPEDIA_WORKFLOW.query_claim_id
    )
    assert plan.steps[0].postcondition is (
        DurablePostconditionKind.QUERY_MATCHES
    )
    assert plan.steps[1].side_effect_id == (
        WIKIPEDIA_WORKFLOW.submit_side_effect_id
    )
    assert plan.steps[1].side_effect_action_key == (
        WIKIPEDIA_WORKFLOW.submit_action_key
    )
    assert plan.steps[1].postcondition is (
        DurablePostconditionKind.SEARCH_OUTCOME_VISIBLE
    )


def test_wikipedia_followup_durable_plan_shape() -> None:
    resolved_task = resolve_live_web_task(
        WIKIPEDIA_FOLLOWUP_GOAL
    )
    assert resolved_task is not None

    plan = compile_durable_task_plan(
        resolved_task
    )

    assert [step.step_id for step in plan.steps] == [
        "enter-query",
        "submit-search",
        "open-followup-link",
    ]
    assert [step.kind for step in plan.steps] == [
        DurableStepKind.ENTER_TEXT,
        DurableStepKind.ACTIVATE_CONTROL,
        DurableStepKind.OPEN_LINK,
    ]
    assert plan.steps[2].action_target is not None
    assert plan.steps[2].action_target.text == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )
    assert plan.steps[2].action_target.element_types == (
        "link",
    )
    assert plan.steps[2].verification_target is not None
    assert plan.steps[2].verification_target.text == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )
    assert plan.steps[2].verification_target.element_types == (
        "heading",
    )
    assert plan.steps[2].side_effect_id == (
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
    )
    assert plan.steps[2].postcondition is (
        DurablePostconditionKind.DESTINATION_HEADING_VISIBLE
    )


def test_python_search_only_durable_plan_shape() -> None:
    resolved_task = resolve_live_web_task(
        "Search python.org for asyncio."
    )
    assert resolved_task is not None

    plan = compile_durable_task_plan(
        resolved_task
    )

    assert plan.workflow_id == "python-org-search"
    assert [step.step_id for step in plan.steps] == [
        "enter-query",
        "submit-search",
    ]
    assert plan.steps[0].input_text == "asyncio"
    assert plan.steps[1].action_target == GO_BUTTON
    assert all(
        step.step_id != "open-followup-link"
        for step in plan.steps
    )


def test_durable_plan_persists_deterministically() -> None:
    resolved_task = resolve_live_web_task(
        WIKIPEDIA_FOLLOWUP_GOAL
    )
    assert resolved_task is not None
    plan = compile_durable_task_plan(
        resolved_task
    )
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_durable_plan_identity(
        transitions,
        resolved_task,
        plan,
    )

    artifact = state.artifacts[
        DURABLE_PLAN_ARTIFACT_ID
    ]
    assert artifact.location == durable_plan_canonical_json(
        plan
    )
    payload = json.loads(
        artifact.location
    )
    assert payload["steps"][2]["action_target"]["text"] == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["steps"].reverse(),
        lambda payload: payload["steps"].pop(),
        lambda payload: payload["steps"][0].__setitem__(
            "input_text",
            "Alan Turing",
        ),
        lambda payload: payload["steps"][2]["action_target"].__setitem__(
            "text",
            "Turing machine",
        ),
        lambda payload: payload.__setitem__(
            "version",
            999,
        ),
    ),
)
def test_durable_plan_tampering_fails_closed(mutate) -> None:
    resolved_task = resolve_live_web_task(
        WIKIPEDIA_FOLLOWUP_GOAL
    )
    assert resolved_task is not None
    plan = compile_durable_task_plan(
        resolved_task
    )
    payload = json.loads(
        durable_plan_canonical_json(plan)
    )
    mutate(payload)
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    state.artifacts[
        DURABLE_PLAN_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=DURABLE_PLAN_ARTIFACT_ID,
        description="Canonical persisted durable task plan.",
        location=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    transitions = TaskStateTransitions(
        state
    )

    with pytest.raises(
        RuntimeError,
        match="durable task plan",
    ):
        _ensure_durable_plan_identity(
            transitions,
            resolved_task,
            plan,
        )


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


def test_followup_plan_clicks_runtime_link_and_verifies_heading() -> None:
    resolved_task = resolve_live_web_task(
        WIKIPEDIA_FOLLOWUP_GOAL
    )
    assert resolved_task is not None

    plan = build_followup_plan(
        WIKIPEDIA_FOLLOWUP_GOAL,
        resolved_task,
    )

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert (
        step.operation
        is PlanOperation.CLICK_TARGET
    )
    assert step.action_target.text == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )
    assert step.action_target.element_types == (
        "link",
    )
    assert step.verification_target.text == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )
    assert step.verification_target.element_types == (
        "heading",
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


def test_followup_execution_crash_happens_after_link_before_heading(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        CRASH_ENV_VAR,
        LiveCrashPoint.FOLLOWUP_EXECUTION.value,
    )
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
    )

    (
        crash_points,
        _published_statuses,
        _decisions,
        _progress_messages,
        _control,
    ) = _run_worker_until_injected_crash(
        monkeypatch,
        state,
        environment,
    )

    assert crash_points == ["terminated"]
    assert environment.type_count == 1
    assert environment.submit_click_count == 1
    assert environment.followup_click_count == 1
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.result_claim_id
        ].status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.followup_claim_id
        ].status
        is ClaimStatus.UNVERIFIED
    )
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.EXECUTED
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


def test_followup_structure_is_created_only_for_multistep_task() -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None

    _ensure_live_task_structure(
        transitions,
        resolved_task,
    )

    assert tuple(state.subgoals) == (
        WIKIPEDIA_WORKFLOW.query_subgoal_id,
        WIKIPEDIA_WORKFLOW.result_subgoal_id,
        WIKIPEDIA_WORKFLOW.followup_subgoal_id,
    )
    assert (
        WIKIPEDIA_WORKFLOW.followup_claim_id
        in state.claims
    )
    assert (
        WIKIPEDIA_FOLLOWUP_TARGET
        in state.subgoals[
            WIKIPEDIA_WORKFLOW.followup_subgoal_id
        ].description
    )


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


def test_followup_identity_persists_exact_runtime_target() -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None

    _ensure_followup_identity(
        transitions,
        resolved_task,
    )

    artifact = state.artifacts[
        FOLLOWUP_TARGET_ARTIFACT_ID
    ]
    assert artifact.location == (
        WIKIPEDIA_FOLLOWUP_TARGET
    )


def test_mismatched_persisted_followup_fails_closed() -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    state.artifacts[
        FOLLOWUP_TARGET_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_TARGET_ARTIFACT_ID,
        description="Durable live web follow-up target text.",
        location="Turing machine",
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
        match="follow-up target identity",
    ):
        _ensure_followup_identity(
            transitions,
            resolved_task,
        )


def test_missing_followup_identity_on_resume_fails_closed() -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL,
        task_id="missing-followup-identity",
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
        match="no durable web follow-up target identity",
    ):
        _ensure_followup_identity(
            transitions,
            resolved_task,
        )


def test_search_only_task_does_not_require_followup_identity() -> None:
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

    _ensure_followup_identity(
        transitions,
        resolved_task,
    )

    assert (
        FOLLOWUP_TARGET_ARTIFACT_ID
        not in state.artifacts
    )


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


def test_wikipedia_followup_worker_runs_through_shared_loop(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
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

    assert environment.type_count == 1
    assert environment.submit_click_count == 1
    assert environment.followup_click_count == 1
    assert (
        state.artifacts[
            FOLLOWUP_TARGET_ARTIFACT_ID
        ].location
        == WIKIPEDIA_FOLLOWUP_TARGET
    )
    assert (
        state.claims[
            WIKIPEDIA_WORKFLOW.followup_claim_id
        ].status
        is ClaimStatus.VERIFIED
    )
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.followup_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
    assert any(
        decision.final_decision_type == "ACTION"
        and decision.target_text
        == WIKIPEDIA_FOLLOWUP_TARGET
        for decision in decisions
    )
    assert (
        decisions[-1].final_decision_type
        == "COMPLETE"
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


def test_followup_not_found_reobserves_then_uses_fresh_link_coordinates(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = DelayedFollowupLinkEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=(
            "missing_link",
            "destination",
        ),
        ready_mode="results",
        missing_link_observes=1,
    )

    (
        _published_statuses,
        _decisions,
        progress_messages,
        _control,
    ) = _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.followup_click_count == 1
    assert environment.missing_link_observe_count == 2
    assert environment.click_reference_points[-1] == (
        140.0,
        110.0,
    )
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )
    assert any(
        "Requested link is not exposed yet" in message
        for message in progress_messages
    )


def test_followup_persistent_not_found_uses_bounded_reobservations(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = DelayedFollowupLinkEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("missing_link",),
        ready_mode="missing_link",
        missing_link_observes=99,
    )

    with pytest.raises(
        RuntimeError,
        match="not_found",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.followup_click_count == 0
    assert environment.missing_link_observe_count == 3
    assert (
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        not in state.side_effects
    )
    assert (
        state.status
        is TaskStateStatus.RUNNING
    )


def test_followup_ambiguous_does_not_use_readiness_reobservation(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = DelayedFollowupLinkEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("ambiguous_link_different_url",),
        ready_mode="results",
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

    assert environment.followup_click_count == 0
    assert environment.missing_link_observe_count == 0


def test_followup_unsafe_does_not_use_readiness_reobservation(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = DelayedFollowupLinkEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("unsafe_link",),
        ready_mode="results",
    )

    with pytest.raises(
        RuntimeError,
        match="unsafe",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.followup_click_count == 0
    assert environment.missing_link_observe_count == 0


def test_followup_readiness_reconciles_destination_without_clicking(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = DelayedFollowupLinkEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("missing_link",),
        ready_mode="destination",
        missing_link_observes=1,
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.followup_click_count == 0
    assert environment.missing_link_observe_count == 2
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.followup_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )


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
        match="did not resolve the configured search outcome",
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


def test_followup_link_absent_fails_closed(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("missing_link",),
    )

    with pytest.raises(
        RuntimeError,
        match="did not resolve to one actionable link",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.followup_click_count == 0
    assert (
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        not in state.side_effects
    )


def test_followup_link_ambiguous_fails_closed(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("ambiguous_link",),
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

    assert environment.followup_click_count == 0


def test_duplicate_followup_links_same_axurl_resolve_and_persist_destination(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=(
            "ambiguous_link_same_url",
            "canonical_destination",
        ),
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.followup_click_count == 1
    assert (
        state.artifacts[
            FOLLOWUP_DESTINATION_ARTIFACT_ID
        ].location
        == "https://en.wikipedia.org/wiki/Information_theory"
    )
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_duplicate_followup_links_different_axurls_stay_ambiguous(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=("ambiguous_link_different_url",),
    )

    with pytest.raises(
        RuntimeError,
        match="Equivalent destination: no",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.followup_click_count == 0


def test_followup_canonical_destination_behavior_remains_unchanged(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=(
            "Search Wikipedia for computer science "
            "and open data structures."
        )
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text="computer science",
        followup_target_text="data structures",
        followup_link_value=(
            "https://en.wikipedia.org/wiki/"
            "Data_structure"
        ),
        destination_heading_text="Data structure",
        address_bar_value=(
            "https://en.wikipedia.org/wiki/"
            "Data_structure"
        ),
        click_external_modes=(
            "results",
            "canonical_destination",
        ),
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.followup_click_count == 1
    assert (
        state.artifacts[
            FOLLOWUP_DESTINATION_ARTIFACT_ID
        ].location
        == "https://en.wikipedia.org/wiki/Data_structure"
    )
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_followup_success_bookkeeping_still_fails_when_heading_absent(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=(
            "results",
            "wrong_destination",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="did not resolve the requested destination heading",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert environment.followup_click_count == 1
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.UNKNOWN
    )
    assert (
        state.status
        is TaskStateStatus.RUNNING
    )


def test_followup_loop_failure_accepts_fresh_destination(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_loop_statuses=(
            AgentLoopStatus.COMPLETED,
            AgentLoopStatus.BLOCKED,
        ),
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.followup_click_count == 1
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.status
        is TaskStateStatus.COMPLETED
    )


def test_followup_loop_failure_still_fails_when_heading_absent(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
        click_external_modes=(
            "results",
            "wrong_destination",
        ),
        click_loop_statuses=(
            AgentLoopStatus.COMPLETED,
            AgentLoopStatus.BLOCKED,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="real-web follow-up navigation failed",
    ):
        _run_worker_with_fake_environment(
            monkeypatch,
            state,
            environment,
        )

    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.UNKNOWN
    )


def test_followup_destination_direct_heading_match_passes_without_url():
    task = ResolvedDurableWebSearchTask(
        spec=WIKIPEDIA_WORKFLOW,
        query_text="Alan Turing",
        followup_target_text="Turing machine",
    )

    result = _followup_destination_postcondition(
        _observation(
            (
                _element(
                    text="Turing machine",
                    value=None,
                    element_type="heading",
                ),
            )
        ),
        task,
        state=None,
    )

    assert result is not None
    assert "directly matches" in result.summary


def test_followup_destination_canonical_url_and_heading_pass():
    task = ResolvedDurableWebSearchTask(
        spec=WIKIPEDIA_WORKFLOW,
        query_text="computer science",
        followup_target_text="data structures",
    )
    state = TaskState(
        goal="Search Wikipedia for computer science and open data structures"
    )
    state.artifacts[
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_DESTINATION_ARTIFACT_ID,
        description="Destination",
        location="https://en.wikipedia.org/wiki/Data_structure",
    )

    result = _followup_destination_postcondition(
        _observation(
            (
                _element(
                    text="Address and search bar",
                    value="https://en.wikipedia.org/wiki/Data_structure",
                    element_type="text_field",
                ),
                _element(
                    text="Data structure",
                    value=None,
                    element_type="heading",
                ),
            )
        ),
        task,
        state=state,
    )

    assert result is not None
    assert "fresh address-bar URL" in result.summary


def test_followup_destination_url_mismatch_fails():
    task = ResolvedDurableWebSearchTask(
        spec=WIKIPEDIA_WORKFLOW,
        query_text="computer science",
        followup_target_text="data structures",
    )
    state = TaskState(goal="goal")
    state.artifacts[
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_DESTINATION_ARTIFACT_ID,
        description="Destination",
        location="https://en.wikipedia.org/wiki/Data_structure",
    )

    result = _followup_destination_postcondition(
        _observation(
            (
                _element(
                    text="Address and search bar",
                    value="https://en.wikipedia.org/wiki/Computer_science",
                    element_type="text_field",
                ),
                _element(
                    text="Data structure",
                    value=None,
                    element_type="heading",
                ),
            )
        ),
        task,
        state=state,
    )

    assert result is None


def test_followup_destination_heading_mismatch_fails():
    task = ResolvedDurableWebSearchTask(
        spec=WIKIPEDIA_WORKFLOW,
        query_text="computer science",
        followup_target_text="data structures",
    )
    state = TaskState(goal="goal")
    state.artifacts[
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_DESTINATION_ARTIFACT_ID,
        description="Destination",
        location="https://en.wikipedia.org/wiki/Data_structure",
    )

    result = _followup_destination_postcondition(
        _observation(
            (
                _element(
                    text="Address and search bar",
                    value="https://en.wikipedia.org/wiki/Data_structure",
                    element_type="text_field",
                ),
                _element(
                    text="Wrong heading",
                    value=None,
                    element_type="heading",
                ),
            )
        ),
        task,
        state=state,
    )

    assert result is None


def test_followup_destination_missing_address_bar_fails_canonical_path():
    task = ResolvedDurableWebSearchTask(
        spec=WIKIPEDIA_WORKFLOW,
        query_text="computer science",
        followup_target_text="data structures",
    )
    state = TaskState(goal="goal")
    state.artifacts[
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_DESTINATION_ARTIFACT_ID,
        description="Destination",
        location="https://en.wikipedia.org/wiki/Data_structure",
    )

    result = _followup_destination_postcondition(
        _observation(
            (
                _element(
                    text="Data structure",
                    value=None,
                    element_type="heading",
                ),
            )
        ),
        task,
        state=state,
    )

    assert result is None


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
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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


def test_wikipedia_followup_restart_at_destination_avoids_duplicate_click(
    monkeypatch,
) -> None:
    state = TaskState(
        goal=WIKIPEDIA_FOLLOWUP_GOAL,
        task_id="wiki-after-followup",
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )
    resolved_task = resolve_live_web_task(
        state.goal
    )
    assert resolved_task is not None
    _ensure_workflow_identity(
        transitions,
        WIKIPEDIA_WORKFLOW,
    )
    _ensure_query_identity(
        transitions,
        resolved_task,
    )
    _ensure_followup_identity(
        transitions,
        resolved_task,
    )
    _ensure_live_task_structure(
        transitions,
        resolved_task,
    )
    query_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains Claude Shannon.",
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
    result_evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Claude Shannon result visible.",
            source="test",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        WIKIPEDIA_WORKFLOW.result_claim_id,
        (result_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        WIKIPEDIA_WORKFLOW.result_subgoal_id
    )
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
            ),
            description=(
                "Open Wikipedia link "
                "'Information theory'."
            ),
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=(
                "click_target:wikipedia_followup_link"
            ),
        )
    )
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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
        mode="destination",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_FOLLOWUP_QUERY,
        followup_target_text=(
            WIKIPEDIA_FOLLOWUP_TARGET
        ),
    )

    _run_worker_with_fake_environment(
        monkeypatch,
        state,
        environment,
    )

    assert environment.open_count == 0
    assert environment.activate_count == 1
    assert environment.type_count == 0
    assert environment.submit_click_count == 0
    assert environment.followup_click_count == 0
    assert (
        state.side_effects[
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        ].state
        is SideEffectState.CONFIRMED
    )
    assert (
        state.subgoals[
            WIKIPEDIA_WORKFLOW.followup_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
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
    _ensure_plan_identity_for_goal(
        transitions,
        state.goal,
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
