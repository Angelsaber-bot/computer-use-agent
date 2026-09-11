from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.agent import (
    AgentLoop,
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    AgentStatus,
    TextInputObservation,
)
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import GroundingResult, TargetSpec, UIGrounder
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    SemanticAXElement,
    UIElement,
    Viewport,
)
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from computer_agent.reasoning import ReasoningResult, ReasoningStatus
from computer_agent.verification import (
    PresenceExpectation,
    StateObservationStatus,
    StateObserver,
    StateVerificationStatus,
    StateTransitionVerifier,
    UIStateCondition,
    VerificationSpec,
)
from experiments.phase05_real_web_autonomy import (
    experiment_10_cross_site_generalization as experiment,
)


def _box(
    *,
    x: int = 100,
    y: int = 120,
    width: int = 220,
    height: int = 32,
) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def _element(
    *,
    text: str | None,
    element_type: str,
    confidence: float = 1.0,
    identifier: str | None = None,
    value=None,
    source: str = "accessibility",
    x: int = 100,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x=x),
        confidence=confidence,
        text=text,
        identifier=identifier,
        value=value,
        enabled=True,
        focused=False,
        source=source,
    )


def _snapshot(*, fused_elements=(), warnings=()) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic-05-10.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=datetime(
                2026,
                9,
            8,
            12,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        ),
        image=Image.new("RGB", (1000, 700)),
        accessibility_elements=tuple(fused_elements),
        ocr_elements=(),
        fused_elements=tuple(fused_elements),
        warnings=tuple(warnings),
    )


def _snapshot_at(
    second: int,
    *,
    elements=(),
    warnings=(),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"synthetic-05-10-{second}.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=datetime(
                2026,
                9,
                8,
                12,
                0,
                second,
                tzinfo=timezone.utc,
            ),
        ),
        image=Image.new("RGB", (1000, 700)),
        accessibility_elements=tuple(elements),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=tuple(warnings),
    )


def _snapshot_with_channels(
    *,
    second: int = 0,
    accessibility_elements=(),
    ocr_elements=(),
    fused_elements=(),
    warnings=(),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"synthetic-khan-{second}.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=datetime(
                2026,
                9,
                8,
                12,
                30,
                second,
                tzinfo=timezone.utc,
            ),
        ),
        image=Image.new("RGB", (1000, 700)),
        accessibility_elements=tuple(accessibility_elements),
        ocr_elements=tuple(ocr_elements),
        fused_elements=tuple(fused_elements),
        warnings=tuple(warnings),
    )


_DEFAULT_VIEWPORT = object()


def _observation(
    *,
    snapshot=None,
    semantic_elements=(),
    app="Google Chrome",
    viewport=_DEFAULT_VIEWPORT,
    elements=(),
    warnings=(),
):
    return experiment.WikipediaObservation(
        frontmost_app=app,
        viewport=Viewport(_box(x=0, y=100, width=1000, height=600))
        if viewport is _DEFAULT_VIEWPORT
        else viewport,
        snapshot=snapshot
        if snapshot is not None
        else _snapshot(fused_elements=elements, warnings=warnings),
        semantic_elements=tuple(semantic_elements),
    )


def _plan(
    *,
    steps=None,
    search_text: str = experiment.SEARCH_FIELD_TEXT,
    search_types: tuple[str, ...] = ("text_field",),
    input_text: str = experiment.SEARCH_INPUT_TEXT,
    submit_text: str = experiment.SUBMIT_TARGET_TEXT,
    submit_types: tuple[str, ...] = ("button",),
    verification_text: str = experiment.ARTICLE_HEADING_TEXT,
    verification_types: tuple[str, ...] = ("heading",),
) -> StructuredPlan:
    if steps is None:
        steps = (
            WebTextInputStep(
                goal="Enter the Wikipedia search query",
                target=TargetSpec(
                    text=search_text,
                    element_types=search_types,
                ),
                input_text=input_text,
                max_attempts=1,
            ),
            PlanStep(
                goal="Submit the Wikipedia search query",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=submit_text,
                    element_types=submit_types,
                ),
                verification_target=TargetSpec(
                    text=verification_text,
                    element_types=verification_types,
                ),
                max_attempts=1,
            ),
        )

    return StructuredPlan(
        task_goal=experiment.TASK_INTENT,
        steps=tuple(steps),
    )


def _step_1() -> WebTextInputStep:
    return _plan().steps[0]


def _step_2() -> PlanStep:
    return _plan().steps[1]


def _execution_plan(plan: StructuredPlan | None = None) -> StructuredPlan:
    return experiment._build_generic_execution_plan(plan or _plan())


def _generic_spec(
    *,
    before_conditions=None,
    after_conditions=None,
) -> VerificationSpec:
    if before_conditions is None:
        before_conditions = (
            UIStateCondition(
                target=TargetSpec(
                    text="Wikipedia The Free Encyclopedia",
                    element_types=("heading",),
                ),
                expectation=PresenceExpectation.PRESENT,
            ),
        )
    if after_conditions is None:
        after_conditions = (
            UIStateCondition(
                target=TargetSpec(
                    text="Computer vision",
                    element_types=("heading",),
                ),
                expectation=PresenceExpectation.PRESENT,
            ),
            UIStateCondition(
                target=TargetSpec(
                    text="Wikipedia The Free Encyclopedia",
                    element_types=("heading",),
                ),
                expectation=PresenceExpectation.ABSENT,
            ),
        )
    return VerificationSpec(
        before_conditions=tuple(before_conditions),
        after_conditions=tuple(after_conditions),
    )


def _assert_exact_generic_spec(spec: VerificationSpec) -> None:
    assert spec.before_conditions == (
        UIStateCondition(
            target=TargetSpec(
                text="Wikipedia The Free Encyclopedia",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.PRESENT,
        ),
    )
    assert spec.after_conditions == (
        UIStateCondition(
            target=TargetSpec(
                text="Computer vision",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.PRESENT,
        ),
        UIStateCondition(
            target=TargetSpec(
                text="Wikipedia The Free Encyclopedia",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.ABSENT,
        ),
    )


class RecordingReasoner:
    def __init__(self, result: ReasoningResult) -> None:
        self.result = result
        self.calls: list[str] = []

    def reason(self, task: str) -> ReasoningResult:
        self.calls.append(task)
        return self.result


class RecordingReasonerBuilder:
    def __init__(self, reasoner: RecordingReasoner) -> None:
        self.reasoner = reasoner
        self.calls = 0

    def __call__(self) -> RecordingReasoner:
        self.calls += 1
        return self.reasoner


class FakeAccessibility:
    available = True
    trusted = True

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def is_trusted(cls):
        return cls.trusted


class FakeEnvironment:
    def __init__(
        self,
        *,
        capture_path,
        initial_observation=None,
        final_observation=None,
        write_candidate=True,
    ) -> None:
        self.capture_path = Path(capture_path)
        self.observations = [
            initial_observation or _observation(elements=_initial_elements()),
            final_observation or _observation(elements=_final_elements()),
        ]
        self.write_candidate = write_candidate
        self.observe_calls = 0
        self.text_input_observe_calls = 0
        self.perception_engine = object()

    def observe(self):
        self.observe_calls += 1
        if self.write_candidate:
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            self.capture_path.write_text("candidate evidence", encoding="utf-8")
        if len(self.observations) > 1:
            return self.observations.pop(0)
        return self.observations[0]

    def text_input_observe(self):
        self.text_input_observe_calls += 1
        return _observation(elements=_initial_elements())


class FakeEnvironmentBuilder:
    def __init__(self, environment: FakeEnvironment | None = None, **kwargs):
        self.environment = environment
        self.kwargs = kwargs
        self.calls = 0
        self.capture_paths = []

    def __call__(
        self,
        *,
        capture_path,
        sleeper=None,
        stabilization_wait_seconds=0.0,
    ):
        self.calls += 1
        self.capture_paths.append(Path(capture_path))
        if self.environment is not None:
            return self.environment
        return FakeEnvironment(capture_path=capture_path, **self.kwargs)


class FakeExecutorBuilder:
    def __init__(self):
        self.calls = 0
        self.executor = object()

    def __call__(self):
        self.calls += 1
        return self.executor


class FakeAgentLoop:
    result: AgentLoopResult | None = None
    instances = []

    def __init__(
        self,
        *,
        perception_engine,
        grounder,
        executor,
        state_transition_verifier,
        web_text_input_observer,
    ) -> None:
        self.perception_engine = perception_engine
        self.grounder = grounder
        self.executor = executor
        self.state_transition_verifier = state_transition_verifier
        self.web_text_input_observer = web_text_input_observer
        self.run_calls = []
        type(self).instances.append(self)

    def run(self, plan):
        self.run_calls.append(plan)
        if type(self).result is None:
            type(self).result = _loop_result(plan)
        return type(self).result


class FakeLiveExecutionRunner:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.report


class SequencePerception:
    def __init__(self, snapshots) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0

    def observe(self):
        self.calls += 1
        if not self.snapshots:
            raise AssertionError("unexpected perception observe")
        return self.snapshots.pop(0)


class RecordingExecutor:
    def __init__(self, *, fail_tool_call: int | None = None) -> None:
        self.calls = []
        self.fail_tool_call = fail_tool_call

    def execute(self, action):
        self.calls.append(action)
        success = self.fail_tool_call != len(self.calls)
        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            error=None if success else "synthetic tool failure",
        )


class ExplodingActionVerifier:
    def __init__(self) -> None:
        self.calls = []

    def verify_target_appeared(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("legacy ActionVerifier should not be used")


class ExplodingRecovery:
    def __init__(self) -> None:
        self.calls = []

    def prepare_retry(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("ActionRecovery should not be used")


class RecordingStateTransitionVerifier:
    def __init__(self) -> None:
        self.verifier = StateTransitionVerifier(
            state_observer=StateObserver()
        )
        self.calls = []

    def verify(self, **kwargs):
        result = self.verifier.verify(**kwargs)
        self.calls.append({**kwargs, "result": result})
        return result


class RecordingUIGrounder(UIGrounder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[TargetSpec, tuple[UIElement, ...], GroundingResult]] = []

    def ground(self, target_spec, elements, *, viewport=None):
        element_tuple = tuple(elements)
        result = super().ground(
            target_spec,
            element_tuple,
            viewport=viewport,
        )
        self.calls.append((target_spec, element_tuple, result))
        return result


def _ready_result(plan: StructuredPlan | None = None) -> ReasoningResult:
    return ReasoningResult(
        status=ReasoningStatus.READY,
        plan=plan if plan is not None else _plan(),
        reason="structured plan ready",
    )


def _blocked_result() -> ReasoningResult:
    return ReasoningResult(
        status=ReasoningStatus.BLOCKED,
        plan=None,
        reason="fake reasoning failure",
    )


def _identity_heading() -> UIElement:
    return _element(
        text="Wikipedia The Free Encyclopedia",
        element_type="heading",
        x=80,
    )


def _search_field(*, value="") -> UIElement:
    return _element(
        text="Search Wikipedia",
        element_type="text_field",
        value=value,
        x=110,
    )


def _search_button() -> UIElement:
    return _element(text="Search", element_type="button", x=360)


def _destination_heading() -> UIElement:
    return _element(text="Computer vision", element_type="heading", x=70)


def _identity_link() -> UIElement:
    return _element(
        text="Wikipedia The Free Encyclopedia",
        element_type="link",
        x=80,
    )


def _initial_elements(*, search_value="") -> tuple[UIElement, ...]:
    return (
        _identity_heading(),
        _search_field(value=search_value),
        _search_button(),
    )


def _final_elements() -> tuple[UIElement, ...]:
    return (_destination_heading(),)


def _khan_logged_out_elements() -> tuple[UIElement, ...]:
    return (
        _element(text="Khan Academy", element_type="heading", x=40),
        _element(text="Search", element_type="button", x=280),
        _element(text="Courses", element_type="link", x=360),
        _element(text="Sign up", element_type="button", x=450),
        _element(text="Log in", element_type="link", x=540),
        _element(text="Learn math", element_type="link", x=620),
    )


def _khan_logged_in_elements() -> tuple[UIElement, ...]:
    return (
        _element(text="Learner home", element_type="heading", x=40),
        _element(text="Search", element_type="text_field", x=280),
        _element(text="Continue learning", element_type="button", x=360),
        _element(text="My courses", element_type="link", x=520),
        _element(text="Practice math", element_type="link", x=620),
    )


def _text_input_observation(*, value="", second=0) -> TextInputObservation:
    field = _search_field(value=value)
    return TextInputObservation(
        application_name=experiment.EXPECTED_APPLICATION_NAME,
        viewport=Viewport(_box(x=0, y=100, width=1000, height=600)),
        snapshot=_snapshot_at(second, elements=(field,)),
        semantic_elements=(),
    )


def _run_synthetic_agent_loop(
    *,
    before_click_elements,
    after_click_elements,
    before_click_warnings=(),
    after_click_warnings=(),
    fail_tool_call: int | None = None,
):
    text_observations = [
        _text_input_observation(value="", second=0),
        _text_input_observation(
            value=experiment.SEARCH_INPUT_TEXT,
            second=1,
        ),
    ]
    perception = SequencePerception(
        (
            _snapshot_at(
                2,
                elements=before_click_elements,
                warnings=before_click_warnings,
            ),
            _snapshot_at(
                3,
                elements=after_click_elements,
                warnings=after_click_warnings,
            ),
        )
    )
    executor = RecordingExecutor(fail_tool_call=fail_tool_call)
    verifier = ExplodingActionVerifier()
    recovery = ExplodingRecovery()
    state_transition_verifier = RecordingStateTransitionVerifier()
    grounder = RecordingUIGrounder()
    state_transition_verifier.action_grounder = grounder
    plan = _execution_plan()

    result = AgentLoop(
        perception_engine=perception,
        grounder=grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
        state_transition_verifier=state_transition_verifier,
        web_text_input_observer=lambda: text_observations.pop(0),
        post_action_settle_timeout_seconds=0.0,
    ).run(plan)

    return result, executor, verifier, recovery, state_transition_verifier


def _planning_report(plan: StructuredPlan | None = None):
    return experiment.LiveOpenAIPlanningReport(
        live_openai_request=True,
        browser_actions=False,
        result=_ready_result(plan if plan is not None else _plan()),
        failures=(),
    )


def _preconditions(tmp_path, *, environment=None):
    if environment is None:
        environment = FakeEnvironment(capture_path=tmp_path / "candidate.png")
    return experiment.live_precondition_failures(
        environment=environment,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
    )


def _final_report(*, observation=None):
    return experiment._final_observation_report(
        observation or _observation(elements=_final_elements())
    )


def _action(tool_name: str, *, success: bool = True) -> tuple[Action, ToolResult]:
    action = Action(
        tool_name=tool_name,
        arguments={"text": experiment.SEARCH_INPUT_TEXT}
        if tool_name == "type_text"
        else {"x": 10, "y": 20},
        reason="fake action",
    )
    result = ToolResult(
        action_id=action.action_id,
        tool_name=tool_name,
        success=success,
        error=None if success else "fake failure",
    )
    return action, result


def _state_with_actions(
    plan: StructuredPlan,
    tools: tuple[str, ...],
    *,
    failed_record: int | None = None,
    final_status: AgentStatus = AgentStatus.SUCCEEDED,
) -> AgentState:
    state = AgentState(user_task=plan.task_goal)
    state.start()
    for index, tool_name in enumerate(tools, start=1):
        action, result = _action(
            tool_name,
            success=failed_record != index,
        )
        state.record_step(action, result)
    if final_status is AgentStatus.SUCCEEDED:
        state.succeed()
    else:
        state.fail("fake loop failure")
    return state


def _loop_result(
    plan: StructuredPlan,
    *,
    status: AgentLoopStatus = AgentLoopStatus.COMPLETED,
    completed_plan_steps: int = experiment.EXPECTED_PLAN_STEPS,
    tools: tuple[str, ...] = experiment.EXPECTED_ACTION_ORDER,
    failed_record: int | None = None,
    final_state_status: AgentStatus | None = None,
) -> AgentLoopResult:
    if final_state_status is None:
        final_state_status = (
            AgentStatus.SUCCEEDED
            if status is AgentLoopStatus.COMPLETED
            else AgentStatus.FAILED
        )
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=_state_with_actions(
            plan,
            tools,
            failed_record=failed_record,
            final_status=final_state_status,
        ),
        completed_plan_steps=completed_plan_steps,
        reason="fake loop result",
    )


def _execution_failures(
    tmp_path,
    *,
    planning_plan: StructuredPlan | None = None,
    execution_plan: StructuredPlan | None = None,
    agent_result: AgentLoopResult | None = None,
    final_report=None,
    candidate_exists: bool = True,
):
    if planning_plan is None:
        planning_plan = _plan()
    if execution_plan is None:
        execution_plan = _execution_plan(planning_plan)
    candidate_path = tmp_path / "candidate.png"
    if candidate_exists:
        candidate_path.write_text("candidate evidence", encoding="utf-8")
    if agent_result is None:
        agent_result = _loop_result(execution_plan)
    return experiment.execution_acceptance_failures(
        planning=_planning_report(planning_plan),
        execution_plan=execution_plan,
        preconditions=_preconditions(
            tmp_path,
            environment=FakeEnvironment(
                capture_path=candidate_path,
                write_candidate=candidate_exists,
            ),
        ),
        agent_result=agent_result,
        final_report=_final_report()
        if final_report is None
        else final_report,
        candidate_path=candidate_path,
    )


def test_candidate_records_include_raw_accessibility_and_production_fields():
    semantic = SemanticAXElement(
        role="AXTextField",
        text=None,
        value="computer vision",
        bounds=_box(),
    )
    fused = _element(
        text="Search Wikipedia",
        element_type="text_field",
        value="",
    )

    records = experiment.candidate_records(
        snapshot=_snapshot(fused_elements=(fused,)),
        semantic_elements=(semantic,),
    )

    raw, production = records
    assert raw.source == "accessibility_raw"
    assert raw.raw_role == "AXTextField"
    assert raw.production_element_type == "text_field"
    assert raw.value == "computer vision"
    assert raw.enabled is None
    assert raw.focused is None

    assert production.source == "accessibility"
    assert production.raw_role is None
    assert production.production_element_type == "text_field"
    assert production.text == "Search Wikipedia"
    assert production.value == ""
    assert production.enabled is True
    assert production.focused is False


def test_matching_candidates_finds_terms_in_text_value_role_and_type():
    candidates = (
        experiment.CandidateRecord(
            source="accessibility_raw",
            raw_role="AXButton",
            production_element_type="button",
            text=None,
            value=None,
            bounds=_box(),
            enabled=None,
            focused=None,
        ),
        experiment.CandidateRecord(
            source="accessibility",
            raw_role=None,
            production_element_type="text_field",
            text=None,
            value="computer vision",
            bounds=_box(),
            enabled=True,
            focused=True,
        ),
        experiment.CandidateRecord(
            source="ocr",
            raw_role=None,
            production_element_type="text",
            text="Wikipedia",
            value=None,
            bounds=_box(),
            enabled=None,
            focused=None,
        ),
    )

    assert experiment.matching_candidates(candidates, ("button",)) == (
        candidates[0],
    )
    assert experiment.matching_candidates(candidates, ("computer vision",)) == (
        candidates[1],
    )
    assert experiment.matching_candidates(candidates, ("wikipedia",)) == (
        candidates[2],
    )


def test_matching_candidates_ignores_empty_terms():
    candidate = experiment.CandidateRecord(
        source="ocr",
        raw_role=None,
        production_element_type="text",
        text="Wikipedia",
        value=None,
        bounds=_box(),
        enabled=None,
        focused=None,
    )

    assert experiment.matching_candidates((candidate,), ("", "   ")) == ()


def test_print_candidate_section_includes_required_candidate_fields(capsys):
    candidate = experiment.CandidateRecord(
        source="accessibility_raw",
        raw_role="AXTextField",
        production_element_type="text_field",
        text="Search Wikipedia",
        value="",
        bounds=_box(),
        enabled=True,
        focused=False,
    )

    experiment.print_candidate_section("Likely candidates", (candidate,))
    output = capsys.readouterr().out

    assert "Likely candidates: 1" in output
    assert "source: accessibility_raw" in output
    assert "raw role: AXTextField" in output
    assert "production element_type: text_field" in output
    assert "text: 'Search Wikipedia'" in output
    assert "value: ''" in output
    assert "bounds: x=100, y=120, width=220, height=32" in output
    assert "enabled: True" in output
    assert "focused: False" in output


def test_run_read_only_audit_uses_injected_observer_and_prints_no_actions(
    monkeypatch,
    capsys,
):
    calls = []
    sleeps = []
    semantic = SemanticAXElement(
        role="AXStaticText",
        text="Wikipedia",
        value="Wikipedia",
        bounds=_box(),
    )
    fused = _element(
        text="Search Wikipedia",
        element_type="text_field",
        value="",
    )

    def observer():
        calls.append("observed")
        return _observation(
            snapshot=_snapshot(fused_elements=(fused,)),
            semantic_elements=(semantic,),
        )

    monkeypatch.setattr(experiment.sys, "platform", "darwin")

    code = experiment.run_read_only_audit(
        observer=observer,
        sleeper=sleeps.append,
        wait_seconds=2,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert sleeps == [1, 1]
    assert calls == ["observed"]
    assert "Phase 05 Experiment 10: Cross-Site Generalization" in output
    assert "Experiment increment: read-only site qualification" in output
    assert "Target site: Wikipedia" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Frontmost app: Google Chrome" in output
    assert "Perception warnings: ()" in output
    assert "Likely search/Wikipedia candidates" in output


def test_run_read_only_audit_blocks_non_macos_before_observer(monkeypatch):
    calls = []

    def observer():
        calls.append("observed")
        raise AssertionError("observer should not be called")

    monkeypatch.setattr(experiment.sys, "platform", "linux")

    code = experiment.run_read_only_audit(
        observer=observer,
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert code == 1
    assert calls == []


def test_khan_qualification_takes_exactly_one_observation_after_countdown():
    calls = []
    sleeps = []
    observation = _observation(elements=_khan_logged_out_elements())

    def observer():
        calls.append("observed")
        return observation

    report = experiment.run_khan_qualification(
        observer=observer,
        sleeper=sleeps.append,
    )

    assert report.observation is observation
    assert report.snapshot is observation.snapshot
    assert calls == ["observed"]
    assert sleeps == [1] * experiment.DEFAULT_WAIT_SECONDS
    assert report.browser_actions is False
    assert report.openai_request is False
    assert report.agent_loop_execution is False
    assert report.evidence_promoted is False


def test_khan_qualification_has_no_execution_openai_or_promotion_side_effects(
    monkeypatch,
    capsys,
):
    calls = []

    def blocker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("Khan read-only qualification must not mutate")

    monkeypatch.setattr(experiment, "_build_executor", blocker)
    monkeypatch.setattr(experiment, "_build_agent_loop_cls", blocker)
    monkeypatch.setattr(experiment, "_build_live_openai_reasoner", blocker)
    monkeypatch.setattr(experiment, "_promote_evidence", blocker)

    code = experiment.run_khan_read_only_qualification(
        observer=lambda: _observation(elements=_khan_logged_out_elements()),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert calls == []
    assert "OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "AgentLoop execution: no" in output
    assert "Evidence promoted: no" in output


def test_khan_qualification_preserves_snapshot_warnings_and_page_totals(capsys):
    accessibility = (
        _element(text="Khan Academy", element_type="heading"),
        _element(text="Courses", element_type="link"),
    )
    ocr = (_element(text="OCR Search", element_type="text", source="ocr"),)
    fused = (
        *accessibility,
        _element(text="Search", element_type="button"),
        _element(text="Practice math", element_type="link"),
    )
    snapshot = _snapshot_with_channels(
        second=4,
        accessibility_elements=accessibility,
        ocr_elements=ocr,
        fused_elements=fused,
        warnings=("synthetic warning",),
    )

    report = experiment.run_khan_qualification(
        observer=lambda: _observation(
            snapshot=snapshot,
            semantic_elements=(),
        ),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_khan_qualification_report(report)
    output = capsys.readouterr().out

    assert report.warnings == ("synthetic warning",)
    assert report.snapshot is snapshot
    assert len(report.relevant_elements) == 4
    assert "Snapshot timestamp: 2026-09-08 12:30:04+00:00" in output
    assert "Snapshot warnings: ('synthetic warning',)" in output
    assert "Screenshot identity/path: synthetic-khan-4.png" in output
    assert "Total snapshot accessibility elements: 2" in output
    assert "Total OCR elements: 1" in output
    assert "Total fused elements: 4" in output


def test_khan_qualification_reports_actual_semantic_elements_and_grounding(
    capsys,
):
    search = _element(
        text="Search",
        identifier="site-search",
        element_type="button",
        x=210,
    )
    practice = _element(text="Practice math", element_type="link", x=410)
    decorative = _element(text="Decorative copy", element_type="text", x=610)
    snapshot = _snapshot_with_channels(
        fused_elements=(search, practice, decorative),
    )

    report = experiment.run_khan_qualification(
        observer=lambda: _observation(snapshot=snapshot),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_khan_qualification_report(report)
    output = capsys.readouterr().out

    assert tuple(item.text for item in report.relevant_elements) == (
        "Search",
        "Practice math",
    )
    assert tuple(item.identifier for item in report.relevant_elements) == (
        "site-search",
        None,
    )
    assert tuple(item.text for item in report.highlighted_elements) == (
        "Search",
        "Practice math",
    )
    assert [diagnostic.target.text for diagnostic in report.grounding_diagnostics] == [
        "Search",
        "Practice math",
    ]
    assert "Relevant Khan semantic elements: 2" in output
    assert "text: 'Search'" in output
    assert "identifier: site-search" in output
    assert "element_type: button" in output
    assert "text: 'Practice math'" in output
    assert "element_type: link" in output
    assert "Read-only UIGrounder diagnostics: 2" in output
    assert "GroundingStatus: resolved" in output
    assert "candidate count: 1" in output
    assert "eligible: True" in output
    assert "rejection_reasons: ()" in output


def test_khan_logged_out_looking_snapshot_reports_semantic_signals(capsys):
    report = experiment.run_khan_qualification(
        observer=lambda: _observation(elements=_khan_logged_out_elements()),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_khan_qualification_report(report)
    output = capsys.readouterr().out

    assert any(
        item.startswith("logged-out-like signals:")
        for item in report.page_evidence
    )
    assert not any(
        item == "login-state semantic evidence: inconclusive"
        for item in report.page_evidence
    )
    assert "logged-out-like signals:" in output
    assert "Sign up" in output
    assert "Log in" in output


def test_khan_logged_in_looking_snapshot_reports_semantic_signals(capsys):
    report = experiment.run_khan_qualification(
        observer=lambda: _observation(elements=_khan_logged_in_elements()),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_khan_qualification_report(report)
    output = capsys.readouterr().out

    assert any(
        item.startswith("logged-in/dashboard-like signals:")
        for item in report.page_evidence
    )
    assert not any(
        item == "login-state semantic evidence: inconclusive"
        for item in report.page_evidence
    )
    assert "logged-in/dashboard-like signals:" in output
    assert "Learner home" in output
    assert "Continue learning" in output


def test_khan_unknown_page_shape_reports_inconclusive_login_state(capsys):
    report = experiment.run_khan_qualification(
        observer=lambda: _observation(
            elements=(
                _element(text="Unexpected Khan page", element_type="heading"),
            )
        ),
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_khan_qualification_report(report)
    output = capsys.readouterr().out

    assert report.page_evidence == (
        "login-state semantic evidence: inconclusive",
    )
    assert "login-state semantic evidence: inconclusive" in output


def test_khan_read_only_cli_is_explicit_and_mutually_exclusive(capsys):
    code = experiment.main(["--khan-read-only", "--live-openai"])
    output = capsys.readouterr().out

    assert code == 2
    assert "Khan qualification: failed" in output
    assert "--khan-read-only cannot be combined" in output


def test_wait_seconds_validation():
    assert experiment._wait_seconds("0") == 0
    assert experiment._wait_seconds("30") == 30


def test_offline_wikipedia_plan_is_exact_and_passes(capsys):
    report = experiment.run_offline_planning()

    assert report.live_openai_request is False
    assert report.browser_actions is False
    assert report.failures == ()
    assert report.plan.task_goal == experiment.TASK_INTENT
    assert report.execution_plan is not None
    assert report.execution_plan is not report.plan
    assert report.execution_plan.steps[1].verification_target is None
    _assert_exact_generic_spec(report.execution_plan.steps[1].verification_spec)
    assert report.plan.task_goal == (
        'On Wikipedia, type "computer vision" into the "Search Wikipedia" '
        'search field, click the Search button, and verify that the '
        '"Computer vision" article heading appears.'
    )
    assert len(report.plan.steps) == 2

    first, second = report.plan.steps
    assert isinstance(first, WebTextInputStep)
    assert first.operation is PlanOperation.TYPE_INTO_TARGET
    assert first.target.text == "Search Wikipedia"
    assert first.target.element_types == ("text_field",)
    assert first.input_text == "computer vision"
    assert first.max_attempts == 1

    assert isinstance(second, PlanStep)
    assert second.operation is PlanOperation.CLICK_TARGET
    assert second.action_target.text == "Search"
    assert second.action_target.element_types == ("button",)
    assert second.verification_target.text == "Computer vision"
    assert second.verification_target.element_types == ("heading",)
    assert second.max_attempts == 1

    code = experiment.run_offline_acceptance()
    output = capsys.readouterr().out

    assert code == 0
    assert "Phase 05 Experiment 10: Cross-Site Generalization" in output
    assert (
        "Experiment increment: deterministic offline Wikipedia plan"
        in output
    )
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Plan steps: 2" in output
    assert "Step 1 class: WebTextInputStep" in output
    assert "Step 1 operation: type_into_target" in output
    assert "Step 1 target text: Search Wikipedia" in output
    assert "Step 1 input_text: computer vision" in output
    assert "Step 2 class: PlanStep" in output
    assert "Step 2 operation: click_target" in output
    assert "Step 2 action target text: Search" in output
    assert "Step 2 verification target text: Computer vision" in output
    assert "Step 2 verification target element_types: ('heading',)" in output
    assert "Planning acceptance: passed" in output
    assert "Execution plan conversion: passed" in output


def test_accepted_legacy_plan_converts_to_exact_generic_execution_plan():
    accepted_plan = _plan()

    execution_plan = experiment._build_generic_execution_plan(accepted_plan)

    assert execution_plan is not accepted_plan
    assert execution_plan.task_goal == accepted_plan.task_goal
    assert execution_plan.steps[0] is not accepted_plan.steps[0]
    assert execution_plan.steps[1] is not accepted_plan.steps[1]
    assert accepted_plan.steps[1].verification_target == TargetSpec(
        text="Computer vision",
        element_types=("heading",),
    )
    assert accepted_plan.steps[1].verification_spec is None

    accepted_first, accepted_second = accepted_plan.steps
    execution_first, execution_second = execution_plan.steps
    assert execution_first == accepted_first
    assert execution_second.goal == accepted_second.goal
    assert execution_second.operation is accepted_second.operation
    assert execution_second.action_target == accepted_second.action_target
    assert execution_second.max_attempts == accepted_second.max_attempts
    assert execution_second.verification_target is None
    _assert_exact_generic_spec(execution_second.verification_spec)
    assert experiment.execution_plan_acceptance_failures(execution_plan) == ()


def test_generic_execution_plan_has_exact_before_condition():
    spec = _execution_plan().steps[1].verification_spec

    assert spec.before_conditions == (
        UIStateCondition(
            target=TargetSpec(
                text="Wikipedia The Free Encyclopedia",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.PRESENT,
        ),
    )


def test_generic_execution_plan_has_exact_after_conditions():
    spec = _execution_plan().steps[1].verification_spec

    assert spec.after_conditions == (
        UIStateCondition(
            target=TargetSpec(
                text="Computer vision",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.PRESENT,
        ),
        UIStateCondition(
            target=TargetSpec(
                text="Wikipedia The Free Encyclopedia",
                element_types=("heading",),
            ),
            expectation=PresenceExpectation.ABSENT,
        ),
    )


def test_recording_state_transition_verifier_delegates_to_real_verifier():
    spec = _generic_spec()
    before = _snapshot_at(
        1,
        elements=(
            _identity_heading(),
            _search_button(),
        ),
    )
    after = _snapshot_at(2, elements=_final_elements())
    expected = StateTransitionVerifier(
        state_observer=StateObserver()
    ).verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=StateObserver())
    )

    result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )

    assert result == expected
    assert recorder.records[0].result is result
    assert recorder.records[0].verification_spec is spec
    assert result.before_evaluations[0].grounding is None
    assert result.before_evaluations[0].observation is not None


def test_recording_state_transition_verifier_retains_calls_in_order():
    spec = _generic_spec()
    before = _snapshot_at(1, elements=(_identity_heading(),))
    after_first = _snapshot_at(2, elements=())
    after_second = _snapshot_at(3, elements=_final_elements())
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=StateObserver())
    )

    first_result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after_first,
        verification_spec=spec,
    )
    second_result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after_second,
        verification_spec=spec,
    )

    assert tuple(record.result for record in recorder.records) == (
        first_result,
        second_result,
    )
    assert tuple(
        record.after_snapshot_captured_at for record in recorder.records
    ) == (
        after_first.frame.captured_at,
        after_second.frame.captured_at,
    )
    assert tuple(
        record.after_snapshot_identity for record in recorder.records
    ) == (
        str(after_first.frame.image_path),
        str(after_second.frame.image_path),
    )


def test_execution_report_retains_generic_evaluation_details():
    spec = _generic_spec()
    before = _snapshot_at(1, elements=(_identity_heading(),))
    after = _snapshot_at(2, elements=_final_elements())
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=StateObserver())
    )
    result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )
    report = experiment.ExecutionReport(
        planning=_planning_report(),
        preconditions=None,
        agent_result=None,
        final_report=None,
        generic_verification_records=recorder.records,
        execution_failures=(),
        evidence_promoted=False,
    )

    record = report.generic_verification_records[0]
    assert record.result is result
    assert record.result.before_evaluations[0].condition.target.text == (
        "Wikipedia The Free Encyclopedia"
    )
    assert record.result.after_evaluations[0].condition.target.text == (
        "Computer vision"
    )
    assert (
        record.result.before_evaluations[0].status
        is StateVerificationStatus.VERIFIED
    )
    assert (
        record.result.after_evaluations[0].observation.status
        is StateObservationStatus.PRESENT
    )
    assert record.result.after_evaluations[0].grounding is None


def test_print_execution_report_includes_observation_verification_diagnostics(
    capsys,
):
    spec = _generic_spec()
    before = _snapshot_at(1, elements=(_identity_heading(),))
    after = _snapshot_at(2, elements=(*_final_elements(), _identity_link()))
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=StateObserver())
    )
    result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )
    report = experiment.ExecutionReport(
        planning=_planning_report(),
        preconditions=None,
        agent_result=None,
        final_report=None,
        generic_verification_records=recorder.records,
        execution_failures=(),
        evidence_promoted=False,
    )

    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    assert "Generic verification calls: 1" in output
    assert "overall status: verified" in output
    assert "target text: Wikipedia The Free Encyclopedia" in output
    assert "target text: Computer vision" in output
    assert "target element_types: ('heading',)" in output
    assert "expectation: present" in output
    assert "expectation: absent" in output
    assert "evidence backend: state_observation" in output
    assert "observation status: present" in output
    assert "observation status: absent" in output
    assert "condition status: verified" in output
    assert "condition reason: target text 'Computer vision' expected present" in output
    assert result.after_evaluations[1].observation.candidates[0].element is (
        after.fused_elements[1]
    )


def test_print_execution_report_includes_legacy_grounding_candidate_details(
    capsys,
):
    spec = _generic_spec()
    before = _snapshot_at(1, elements=(_identity_heading(),))
    unsafe_landing_candidate = _element(
        text="Wikipedia The Free Encyclopedia",
        element_type="text",
        source="ocr",
        x=240,
    )
    after = _snapshot_at(
        2,
        elements=(
            _destination_heading(),
            unsafe_landing_candidate,
        ),
    )
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier()
    )
    result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )
    report = experiment.ExecutionReport(
        planning=_planning_report(),
        preconditions=None,
        agent_result=None,
        final_report=None,
        generic_verification_records=recorder.records,
        execution_failures=(),
        evidence_promoted=False,
    )

    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    assert result.after_evaluations[1].grounding.status.value == "unsafe"
    assert "grounding status: unsafe" in output
    assert "condition status: inconclusive" in output
    assert "grounding reason: text candidates were unsafe" in output
    assert "candidate count: 1" in output
    assert "candidate 1:" in output
    assert "match_basis: text" in output
    assert "eligible: False" in output
    assert "rejection_reasons: ('incompatible_element_type',)" in output
    assert "distance: None" in output
    assert "element text: Wikipedia The Free Encyclopedia" in output
    assert "element identifier: None" in output
    assert "element element_type: text" in output
    assert "element source: ocr" in output
    assert "element confidence: 1.0" in output
    assert "element enabled: True" in output
    assert "element bounds: x=240, y=120, width=220, height=32" in output


def test_print_execution_report_includes_wrong_role_observation_candidate(
    capsys,
):
    spec = _generic_spec()
    before = _snapshot_at(1, elements=(_identity_heading(),))
    landing_link = _identity_link()
    after = _snapshot_at(
        2,
        elements=(
            _destination_heading(),
            landing_link,
        ),
    )
    recorder = experiment.RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=StateObserver())
    )
    result = recorder.verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=spec,
    )
    report = experiment.ExecutionReport(
        planning=_planning_report(),
        preconditions=None,
        agent_result=None,
        final_report=None,
        generic_verification_records=recorder.records,
        execution_failures=(),
        evidence_promoted=False,
    )

    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    landing_evaluation = result.after_evaluations[1]
    assert landing_evaluation.observation.status is StateObservationStatus.ABSENT
    assert landing_evaluation.observation.candidates[0].element is landing_link
    assert landing_evaluation.observation.candidates[0].predicate_match is False
    assert landing_evaluation.observation.candidates[0].mismatch_reasons == (
        "incompatible_element_type",
    )
    assert "evidence backend: state_observation" in output
    assert "candidate 1:" in output
    assert "predicate_match: False" in output
    assert "mismatch_reasons: ('incompatible_element_type',)" in output
    assert "element text: Wikipedia The Free Encyclopedia" in output
    assert "element element_type: link" in output


def test_generic_execution_conversion_rejects_wrong_accepted_task_goal():
    plan = _plan()
    object.__setattr__(plan, "task_goal", "Search Wikipedia.")

    with pytest.raises(ValueError, match="task_goal"):
        experiment._build_generic_execution_plan(plan)


def test_generic_execution_conversion_rejects_wrong_action_target():
    with pytest.raises(ValueError, match="action target text"):
        experiment._build_generic_execution_plan(_plan(submit_text="Go"))


def test_generic_execution_conversion_rejects_wrong_article_verification_target():
    with pytest.raises(ValueError, match="verification target text"):
        experiment._build_generic_execution_plan(
            _plan(verification_text="Computer Vision")
        )


def test_generic_execution_conversion_rejects_wrong_article_verification_role():
    with pytest.raises(ValueError, match="verification target element_types"):
        experiment._build_generic_execution_plan(
            _plan(verification_types=("text",))
        )


def test_generic_execution_conversion_rejects_max_attempts_other_than_one():
    second = _step_2()
    object.__setattr__(second, "max_attempts", 2)

    with pytest.raises(ValueError, match="step 2 max_attempts"):
        experiment._build_generic_execution_plan(
            _plan(steps=(_step_1(), second))
        )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            _generic_spec(before_conditions=()),
            "before_conditions",
        ),
        (
            _generic_spec(
                before_conditions=(
                    UIStateCondition(
                        target=TargetSpec(
                            text="Wikipedia The Free Encyclopedia",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.ABSENT,
                    ),
                )
            ),
            "before_conditions",
        ),
        (
            _generic_spec(
                after_conditions=(
                    UIStateCondition(
                        target=TargetSpec(
                            text="Computer vision",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.PRESENT,
                    ),
                )
            ),
            "after_conditions",
        ),
        (
            _generic_spec(
                after_conditions=(
                    UIStateCondition(
                        target=TargetSpec(
                            text="Computer vision",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.ABSENT,
                    ),
                    UIStateCondition(
                        target=TargetSpec(
                            text="Wikipedia The Free Encyclopedia",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.ABSENT,
                    ),
                )
            ),
            "after_conditions",
        ),
        (
            _generic_spec(
                after_conditions=(
                    UIStateCondition(
                        target=TargetSpec(
                            text="Computer vision",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.PRESENT,
                    ),
                    UIStateCondition(
                        target=TargetSpec(
                            text="Wikipedia The Free Encyclopedia",
                            element_types=("heading",),
                        ),
                        expectation=PresenceExpectation.ABSENT,
                    ),
                    UIStateCondition(
                        target=TargetSpec(
                            text="Search",
                            element_types=("button",),
                        ),
                        expectation=PresenceExpectation.ABSENT,
                    ),
                )
            ),
            "after_conditions",
        ),
    ],
)
def test_execution_acceptance_rejects_malformed_generic_spec(
    tmp_path,
    spec,
    message,
):
    execution_plan = _execution_plan()
    object.__setattr__(execution_plan.steps[1], "verification_spec", spec)

    failures = _execution_failures(
        tmp_path,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan),
    )

    assert any(message in failure for failure in failures)


def test_execution_acceptance_rejects_legacy_target_in_execution_plan(tmp_path):
    execution_plan = _execution_plan()
    object.__setattr__(
        execution_plan.steps[1],
        "verification_target",
        TargetSpec(text="Computer vision", element_types=("heading",)),
    )

    failures = _execution_failures(
        tmp_path,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan),
    )

    assert any("verification_target was present" in failure for failure in failures)


def test_execution_acceptance_rejects_missing_generic_spec(tmp_path):
    execution_plan = _execution_plan()
    object.__setattr__(execution_plan.steps[1], "verification_spec", None)

    failures = _execution_failures(
        tmp_path,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan),
    )

    assert any("verification_spec" in failure for failure in failures)


def test_execution_acceptance_rejects_generic_step_max_attempts_other_than_one(
    tmp_path,
):
    execution_plan = _execution_plan()
    object.__setattr__(execution_plan.steps[1], "max_attempts", 2)

    failures = _execution_failures(
        tmp_path,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan),
    )

    assert any("max_attempts" in failure for failure in failures)


def test_synthetic_successful_execution_uses_generic_agent_loop_verification():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(*_final_elements(), _identity_link()),
        )
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.completed_plan_steps == 2
    assert [record.action.tool_name for record in result.state.steps] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert executor.calls[0].arguments == {"x": 220, "y": 136}
    assert executor.calls[1].arguments == {
        "text": experiment.SEARCH_INPUT_TEXT
    }
    assert executor.calls[2].arguments == {"x": 470, "y": 136}
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    assert transition_verifier.verifier.grounder is None
    assert isinstance(transition_verifier.verifier.state_observer, StateObserver)
    assert [
        call[0]
        for call in transition_verifier.action_grounder.calls
    ] == [result.plan.steps[1].action_target]
    assert (
        transition_verifier.calls[0]["verification_spec"].before_conditions
        == result.plan.steps[1].verification_spec.before_conditions
    )
    assert (
        transition_verifier.calls[1]["verification_spec"]
        is result.plan.steps[1].verification_spec
    )
    before_result = transition_verifier.calls[0]["result"]
    transition_result = transition_verifier.calls[1]["result"]
    assert before_result.status is StateVerificationStatus.VERIFIED
    assert transition_result.status is StateVerificationStatus.VERIFIED
    assert (
        before_result.before_evaluations[0].observation.status
        is StateObservationStatus.PRESENT
    )
    assert (
        transition_result.after_evaluations[0].observation.status
        is StateObservationStatus.PRESENT
    )
    landing_absence = transition_result.after_evaluations[1]
    assert landing_absence.observation.status is StateObservationStatus.ABSENT
    assert landing_absence.status is StateVerificationStatus.VERIFIED
    assert landing_absence.observation.candidates[0].element.element_type == "link"
    assert (
        landing_absence.observation.candidates[0].predicate_match is False
    )


def test_generic_precondition_failure_prevents_search_click():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=(
                _search_field(value=experiment.SEARCH_INPUT_TEXT),
                _search_button(),
            ),
            after_click_elements=_final_elements(),
        )
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
    ]
    assert [record.action.tool_name for record in result.state.steps] == [
        "click_mouse",
        "type_text",
    ]
    assert "preconditions were not verified" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 1


def test_after_article_heading_missing_fails_without_retry():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(),
        )
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert sum(
        action.tool_name == "click_mouse" for action in executor.calls
    ) == 2
    assert "generic state transition verification failed" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    article_evaluation = (
        transition_verifier.calls[1]["result"].after_evaluations[0]
    )
    assert article_evaluation.observation.status is StateObservationStatus.ABSENT
    assert article_evaluation.status is StateVerificationStatus.FAILED


def test_low_confidence_article_heading_blocks_closed():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(
                _element(
                    text="Computer vision",
                    element_type="heading",
                    confidence=0.2,
                ),
            ),
        )
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert "generic state transition verification was inconclusive" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    article_evaluation = (
        transition_verifier.calls[1]["result"].after_evaluations[0]
    )
    assert (
        article_evaluation.observation.status
        is StateObservationStatus.INCONCLUSIVE
    )
    assert article_evaluation.status is StateVerificationStatus.INCONCLUSIVE
    assert article_evaluation.observation.candidates[0].uncertainty_reasons == (
        "low_confidence",
    )


def test_warning_bearing_article_absence_blocks_closed():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(),
            after_click_warnings=("synthetic observation warning",),
        )
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert "generic state transition verification was inconclusive" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    article_evaluation = (
        transition_verifier.calls[1]["result"].after_evaluations[0]
    )
    assert (
        article_evaluation.observation.status
        is StateObservationStatus.INCONCLUSIVE
    )
    assert article_evaluation.status is StateVerificationStatus.INCONCLUSIVE
    assert "warnings" in article_evaluation.observation.reason


def test_landing_identity_still_present_fails_transition_contract():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(
                _destination_heading(),
                _identity_heading(),
            ),
        )
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert "generic state transition verification failed" in result.reason
    assert "1 failed" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    landing_evaluation = (
        transition_verifier.calls[1]["result"].after_evaluations[1]
    )
    assert landing_evaluation.observation.status is StateObservationStatus.PRESENT
    assert landing_evaluation.status is StateVerificationStatus.FAILED


def test_multiple_valid_article_headings_succeed_without_ambiguity():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(
                _destination_heading(),
                _element(
                    text="Computer vision",
                    element_type="heading",
                    x=300,
                ),
            ),
        )
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 2
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 2
    transition_result = transition_verifier.calls[1]["result"]
    article_evaluation = transition_result.after_evaluations[0]
    assert transition_result.status is StateVerificationStatus.VERIFIED
    assert article_evaluation.observation.status is StateObservationStatus.PRESENT
    assert len(article_evaluation.observation.candidates) == 2


def test_failed_generic_click_tool_result_remains_exhausted_without_recovery():
    result, executor, verifier, recovery, transition_verifier = (
        _run_synthetic_agent_loop(
            before_click_elements=_initial_elements(
                search_value=experiment.SEARCH_INPUT_TEXT
            ),
            after_click_elements=(*_final_elements(), _identity_link()),
            fail_tool_call=3,
        )
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "click_mouse",
        "type_text",
        "click_mouse",
    ]
    assert result.state.steps[-1].result.success is False
    assert "generic click tool failed" in result.reason
    assert verifier.calls == []
    assert recovery.calls == []
    assert len(transition_verifier.calls) == 1


def test_offline_acceptance_rejects_altered_task_goal():
    plan = _plan()
    object.__setattr__(
        plan,
        "task_goal",
        "Search Wikipedia for machine vision.",
    )

    failures = experiment.plan_acceptance_failures(plan)

    assert failures
    assert any("task_goal" in failure for failure in failures)


def test_offline_acceptance_rejects_empty_task_goal():
    plan = _plan()
    object.__setattr__(plan, "task_goal", "")

    failures = experiment.plan_acceptance_failures(plan)

    assert failures
    assert any("task_goal" in failure for failure in failures)


def test_offline_acceptance_rejects_wrong_search_target():
    failures = experiment.plan_acceptance_failures(
        _plan(search_text="Search This Site")
    )

    assert any("step 1 target text" in failure for failure in failures)


def test_offline_acceptance_rejects_search_text_field_target():
    failures = experiment.plan_acceptance_failures(
        _plan(search_text="Search", search_types=("text_field",))
    )

    assert any("step 1 target text was Search" in failure for failure in failures)


def test_offline_acceptance_rejects_wrong_submit_target():
    failures = experiment.plan_acceptance_failures(
        _plan(submit_text="GO")
    )

    assert any("step 2 action target text" in failure for failure in failures)


def test_offline_acceptance_rejects_article_heading_as_click_target():
    failures = experiment.plan_acceptance_failures(
        _plan(submit_text="Computer vision", submit_types=("heading",))
    )

    assert any("step 2 action target text" in failure for failure in failures)
    assert any(
        "step 2 action target element_types" in failure
        for failure in failures
    )


def test_offline_acceptance_rejects_wrong_input_text():
    failures = experiment.plan_acceptance_failures(
        _plan(input_text="machine vision")
    )

    assert any("step 1 input_text" in failure for failure in failures)


def test_offline_acceptance_rejects_wrong_verification_target():
    failures = experiment.plan_acceptance_failures(
        _plan(verification_text="Computer Vision")
    )

    assert any(
        "step 2 verification target text" in failure
        for failure in failures
    )


def test_offline_acceptance_rejects_empty_verification_role():
    failures = experiment.plan_acceptance_failures(
        _plan(verification_types=())
    )

    assert any(
        "step 2 verification target element_types" in failure
        for failure in failures
    )


def test_offline_acceptance_rejects_text_verification_role():
    failures = experiment.plan_acceptance_failures(
        _plan(verification_types=("text",))
    )

    assert any(
        "step 2 verification target element_types" in failure
        for failure in failures
    )


def test_offline_acceptance_rejects_link_verification_role():
    failures = experiment.plan_acceptance_failures(
        _plan(verification_types=("link",))
    )

    assert any(
        "step 2 verification target element_types" in failure
        for failure in failures
    )


def test_offline_acceptance_rejects_max_attempts_other_than_one():
    first = _step_1()
    second = _step_2()
    object.__setattr__(first, "max_attempts", 2)
    object.__setattr__(second, "max_attempts", 2)
    plan = _plan(steps=(first, second))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("step 1 max_attempts" in failure for failure in failures)
    assert any("step 2 max_attempts" in failure for failure in failures)


def test_offline_acceptance_rejects_extra_or_missing_steps():
    missing = _plan(steps=(_step_1(),))
    extra = _plan(steps=(_step_1(), _step_2(), _step_2()))

    missing_failures = experiment.plan_acceptance_failures(missing)
    extra_failures = experiment.plan_acceptance_failures(extra)

    assert any("plan step count" in failure for failure in missing_failures)
    assert any("plan step count" in failure for failure in extra_failures)


def test_offline_acceptance_rejects_incorrect_operation_ordering():
    plan = _plan(steps=(_step_2(), _step_1()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("step 1" in failure for failure in failures)
    assert any("WebTextInputStep" in failure for failure in failures)
    assert any("step 2" in failure for failure in failures)
    assert any("PlanStep" in failure for failure in failures)


def test_offline_acceptance_rejects_unexpected_target_payloads():
    plan = _plan()
    first, second = plan.steps
    object.__setattr__(
        first,
        "target",
        TargetSpec(
            text=experiment.SEARCH_FIELD_TEXT,
            identifier="search-input",
            element_types=("text_field",),
            reference_point=(1.0, 2.0),
        ),
    )
    object.__setattr__(
        second,
        "action_target",
        TargetSpec(
            text=experiment.SUBMIT_TARGET_TEXT,
            identifier="submit",
            element_types=("button",),
            reference_point=(3.0, 4.0),
        ),
    )

    failures = experiment.plan_acceptance_failures(plan)

    assert any("identifier" in failure for failure in failures)
    assert any("reference_point" in failure for failure in failures)


def test_offline_mode_has_no_browser_or_openai_side_effects(monkeypatch, capsys):
    calls = []

    def blocker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("offline mode should not call live dependencies")

    monkeypatch.setattr(experiment, "observe_wikipedia", blocker)
    monkeypatch.setattr(experiment, "_build_live_dependencies", blocker)
    monkeypatch.setattr(experiment, "run_read_only_audit", blocker)

    code = experiment.main(["--offline-plan"])
    output = capsys.readouterr().out

    assert code == 0
    assert calls == []
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Planning acceptance: passed" in output


def test_live_openai_routes_to_planning_only_mode(monkeypatch, capsys):
    calls = []
    reasoner = RecordingReasoner(_ready_result())
    builder = RecordingReasonerBuilder(reasoner)

    def blocker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("live planning should not inspect the browser")

    monkeypatch.setattr(experiment, "run_read_only_audit", blocker)
    monkeypatch.setattr(experiment, "_build_live_dependencies", blocker)

    code = experiment.main(["--live-openai"], reasoner_builder=builder)
    output = capsys.readouterr().out

    assert code == 0
    assert builder.calls == 1
    assert reasoner.calls == [experiment.TASK_INTENT]
    assert calls == []
    assert "Experiment increment: live OpenAI Wikipedia planning" in output
    assert "Live OpenAI request: yes" in output
    assert "Browser actions: no" in output
    assert "Planning acceptance: passed" in output


def test_live_openai_makes_exactly_one_fake_reasoner_call():
    reasoner = RecordingReasoner(_ready_result())
    builder = RecordingReasonerBuilder(reasoner)

    report = experiment.run_live_openai_planning(reasoner_builder=builder)

    assert report.live_openai_request is True
    assert report.browser_actions is False
    assert report.failures == ()
    assert builder.calls == 1
    assert reasoner.calls == [experiment.TASK_INTENT]


def test_live_openai_valid_returned_plan_passes(capsys):
    reasoner = RecordingReasoner(_ready_result(_plan()))

    code = experiment.run_live_openai_acceptance(
        reasoner_builder=RecordingReasonerBuilder(reasoner)
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "ReasoningStatus: ready" in output
    assert f"task_goal: {experiment.TASK_INTENT}" in output
    assert "Plan steps: 2" in output
    assert "Step 1 class: WebTextInputStep" in output
    assert "Step 2 verification target element_types: ('heading',)" in output
    assert "Planning acceptance: passed" in output


def test_live_openai_invalid_returned_plan_fails_closed(capsys):
    invalid_plan = _plan(search_text="Search This Site")
    reasoner = RecordingReasoner(_ready_result(invalid_plan))

    code = experiment.run_live_openai_acceptance(
        reasoner_builder=RecordingReasonerBuilder(reasoner)
    )
    output = capsys.readouterr().out

    assert code == 1
    assert "Planning acceptance: failed" in output
    assert "step 1 target text" in output
    assert "Browser actions: no" in output


def test_live_openai_reasoning_failure_fails_without_browser_action(
    monkeypatch,
    capsys,
):
    calls = []
    reasoner = RecordingReasoner(_blocked_result())

    def blocker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("reasoning failure should not inspect browser")

    monkeypatch.setattr(experiment, "run_read_only_audit", blocker)
    monkeypatch.setattr(experiment, "_build_live_dependencies", blocker)

    code = experiment.main(
        ["--live-openai"],
        reasoner_builder=RecordingReasonerBuilder(reasoner),
    )
    output = capsys.readouterr().out

    assert code == 1
    assert reasoner.calls == [experiment.TASK_INTENT]
    assert calls == []
    assert "ReasoningStatus: blocked" in output
    assert "fake reasoning failure" in output
    assert "Planning acceptance: failed" in output
    assert "Browser actions: no" in output


def test_live_openai_does_not_import_execution_stack(monkeypatch):
    reasoner = RecordingReasoner(_ready_result())
    blocked_imports = []
    original_import = __import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name.startswith("computer_agent.agent")
            or name.startswith("computer_agent.tools.executor")
        ):
            blocked_imports.append(name)
            raise AssertionError(f"unexpected execution import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guarded_import)

    report = experiment.run_live_openai_planning(
        reasoner_builder=RecordingReasonerBuilder(reasoner)
    )

    assert report.failures == ()
    assert blocked_imports == []


def test_offline_plan_behavior_remains_unchanged(capsys):
    code = experiment.main(["--offline-plan"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment increment: deterministic offline Wikipedia plan" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Plan steps: 2" in output
    assert "Planning acceptance: passed" in output


def test_default_mode_remains_read_only(monkeypatch):
    calls = []

    def read_only_audit(*, wait_seconds):
        calls.append(wait_seconds)
        return 17

    monkeypatch.setattr(experiment, "run_read_only_audit", read_only_audit)

    assert experiment.main([]) == 17
    assert calls == [experiment.DEFAULT_WAIT_SECONDS]


def test_offline_and_live_openai_flags_are_rejected(capsys):
    code = experiment.main(["--offline-plan", "--live-openai"])
    output = capsys.readouterr().out

    assert code == 2
    assert "Experiment increment: rejected" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Planning acceptance: failed" in output
    assert "--offline-plan and --live-openai are mutually exclusive" in output


def test_execute_without_live_openai_is_rejected(capsys):
    code = experiment.main(["--execute"])
    output = capsys.readouterr().out

    assert code == 2
    assert "Experiment increment: rejected" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "--execute requires --live-openai" in output


def test_planning_failure_prevents_countdown_environment_and_executor(tmp_path):
    sleeps = []
    reasoner = RecordingReasoner(_blocked_result())
    environment_builder = FakeEnvironmentBuilder()
    executor_builder = FakeExecutorBuilder()

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
        environment_builder=environment_builder,
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=sleeps.append,
    )

    assert report.execution_failures
    assert any("ReasoningStatus" in failure for failure in report.execution_failures)
    assert sleeps == []
    assert environment_builder.calls == 0
    assert executor_builder.calls == 0


def test_plan_acceptance_failure_prevents_execution(tmp_path):
    invalid_plan = _plan(submit_text="GO")
    reasoner = RecordingReasoner(_ready_result(invalid_plan))
    environment_builder = FakeEnvironmentBuilder()
    executor_builder = FakeExecutorBuilder()

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
        environment_builder=environment_builder,
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
    )

    assert report.execution_failures
    assert any("action target text" in failure for failure in report.execution_failures)
    assert environment_builder.calls == 0
    assert executor_builder.calls == 0


@pytest.mark.parametrize(
    ("platform_name", "available", "trusted", "message"),
    [
        ("linux", True, True, "platform is not macOS"),
        ("darwin", False, True, "Accessibility is unavailable"),
        ("darwin", True, False, "Accessibility is not trusted"),
    ],
)
def test_platform_accessibility_gates_prevent_executor_construction(
    tmp_path,
    platform_name,
    available,
    trusted,
    message,
):
    class Access:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def is_trusted():
            return trusted

    environment_builder = FakeEnvironmentBuilder()
    executor_builder = FakeExecutorBuilder()

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(RecordingReasoner(_ready_result())),
        environment_builder=environment_builder,
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name=platform_name,
        accessibility_cls=Access,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert any(message in failure for failure in report.execution_failures)
    assert environment_builder.calls == 0
    assert executor_builder.calls == 0


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (_observation(app="Safari", elements=_initial_elements()), "frontmost app"),
        (
            _observation(viewport=None, elements=_initial_elements()),
            "viewport was unavailable",
        ),
        (
            _observation(elements=_initial_elements(), warnings=("warn",)),
            "perception warnings",
        ),
        (
            _observation(elements=(_search_field(), _search_button())),
            "Wikipedia landing identity grounding",
        ),
        (
            _observation(elements=(_identity_heading(), _search_button())),
            "Search Wikipedia grounding",
        ),
        (
            _observation(elements=_initial_elements(search_value="computer")),
            "field was not empty",
        ),
        (
            _observation(elements=(_identity_heading(), _search_field())),
            "Search button grounding",
        ),
        (
            _observation(elements=(*_initial_elements(), _destination_heading())),
            "Computer vision destination grounding",
        ),
    ],
)
def test_live_precondition_failures_prevent_executor_construction(
    tmp_path,
    observation,
    message,
):
    executor_builder = FakeExecutorBuilder()

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(RecordingReasoner(_ready_result())),
        environment_builder=FakeEnvironmentBuilder(
            initial_observation=observation,
        ),
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert any(message in failure for failure in report.execution_failures)
    assert executor_builder.calls == 0


def test_executor_constructed_only_after_all_gates(tmp_path):
    executor_builder = FakeExecutorBuilder()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(RecordingReasoner(_ready_result())),
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.execution_failures == ()
    assert executor_builder.calls == 1
    assert len(FakeAgentLoop.instances) == 1


def test_agent_loop_called_exactly_once(tmp_path):
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None

    report = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(RecordingReasoner(_ready_result())),
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.execution_failures == ()
    assert len(FakeAgentLoop.instances) == 1
    instance = FakeAgentLoop.instances[0]
    assert isinstance(
        instance.state_transition_verifier,
        experiment.RecordingStateTransitionVerifier,
    )
    assert isinstance(instance.grounder, UIGrounder)
    assert instance.state_transition_verifier.verifier.grounder is None
    assert isinstance(
        instance.state_transition_verifier.verifier.state_observer,
        StateObserver,
    )
    run_plan = instance.run_calls[0]
    assert run_plan is not report.planning.result.plan
    assert experiment.execution_plan_acceptance_failures(run_plan) == ()
    assert run_plan.steps[1].verification_target is None
    _assert_exact_generic_spec(run_plan.steps[1].verification_spec)


def test_default_state_transition_verifier_stays_legacy_grounder_backed():
    verifier = StateTransitionVerifier()

    assert isinstance(verifier.grounder, UIGrounder)
    assert verifier.state_observer is None


def test_live_openai_execute_routes_to_execution_runner(capsys):
    runner = FakeLiveExecutionRunner(
        experiment.ExecutionReport(
            planning=_planning_report(),
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=(),
            evidence_promoted=True,
        )
    )

    code = experiment.main(
        ["--live-openai", "--execute"],
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_result())
        ),
        live_execution_runner=runner,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert len(runner.calls) == 1
    assert "Execution mode: live OpenAI Wikipedia execution" in output
    assert "Execution acceptance: passed" in output


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        (("click_mouse", "type_text"), "executed action count"),
        (("click_mouse", "click_mouse", "type_text"), "action tool order"),
        (("click_mouse", "type_text", "scroll"), "scroll"),
        (("type_text", "click_mouse", "click_mouse"), "action tool order"),
    ],
)
def test_execution_acceptance_requires_exact_action_order(tmp_path, tools, message):
    planning_plan = _plan()
    execution_plan = _execution_plan(planning_plan)

    failures = _execution_failures(
        tmp_path,
        planning_plan=planning_plan,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan, tools=tools),
    )

    assert any(message in failure for failure in failures)


def test_execution_acceptance_rejects_failed_tool_result(tmp_path):
    planning_plan = _plan()
    execution_plan = _execution_plan(planning_plan)

    failures = _execution_failures(
        tmp_path,
        planning_plan=planning_plan,
        execution_plan=execution_plan,
        agent_result=_loop_result(execution_plan, failed_record=2),
    )

    assert any("ToolResult failed" in failure for failure in failures)


def test_execution_acceptance_rejects_wrong_agent_loop_status(tmp_path):
    planning_plan = _plan()
    execution_plan = _execution_plan(planning_plan)

    failures = _execution_failures(
        tmp_path,
        planning_plan=planning_plan,
        execution_plan=execution_plan,
        agent_result=_loop_result(
            execution_plan,
            status=AgentLoopStatus.EXHAUSTED,
            tools=experiment.EXPECTED_ACTION_ORDER,
        ),
    )

    assert any("AgentLoopResult status" in failure for failure in failures)


def test_execution_acceptance_rejects_wrong_agent_state(tmp_path):
    planning_plan = _plan()
    execution_plan = _execution_plan(planning_plan)
    agent_result = _loop_result(execution_plan)
    agent_result.state.status = AgentStatus.FAILED

    failures = _execution_failures(
        tmp_path,
        planning_plan=planning_plan,
        execution_plan=execution_plan,
        agent_result=agent_result,
    )

    assert any("AgentState status" in failure for failure in failures)


def test_execution_acceptance_rejects_wrong_completed_step_count(tmp_path):
    planning_plan = _plan()
    execution_plan = _execution_plan(planning_plan)
    agent_result = _loop_result(execution_plan)
    object.__setattr__(agent_result, "completed_plan_steps", 1)

    failures = _execution_failures(
        tmp_path,
        planning_plan=planning_plan,
        execution_plan=execution_plan,
        agent_result=agent_result,
    )

    assert any("completed plan steps" in failure for failure in failures)


def test_execution_acceptance_rejects_final_destination_not_resolved(tmp_path):
    failures = _execution_failures(
        tmp_path,
        final_report=_final_report(
            observation=_observation(elements=(_search_button(),))
        ),
    )

    assert any("final Computer vision grounding" in failure for failure in failures)


def test_execution_acceptance_rejects_final_app_mismatch(tmp_path):
    failures = _execution_failures(
        tmp_path,
        final_report=_final_report(
            observation=_observation(app="Safari", elements=_final_elements())
        ),
    )

    assert any("final frontmost app" in failure for failure in failures)


def test_execution_acceptance_rejects_final_warnings(tmp_path):
    failures = _execution_failures(
        tmp_path,
        final_report=_final_report(
            observation=_observation(
                elements=_final_elements(),
                warnings=("warn",),
            )
        ),
    )

    assert any("final perception warnings" in failure for failure in failures)


def test_evidence_promotion_only_on_full_success(tmp_path):
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None

    success = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(RecordingReasoner(_ready_result())),
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "success-candidate.png",
        formal_evidence_path=tmp_path / "success-formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert success.execution_failures == ()
    assert success.evidence_promoted is True
    assert (tmp_path / "success-formal.png").read_text(encoding="utf-8") == (
        "candidate evidence"
    )

    FakeAgentLoop.instances = []
    failed_plan = _plan()
    FakeAgentLoop.result = _loop_result(
        failed_plan,
        status=AgentLoopStatus.EXHAUSTED,
        completed_plan_steps=1,
        tools=("click_mouse", "type_text"),
    )

    failure = experiment.run_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_result(failed_plan))
        ),
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "failed-candidate.png",
        formal_evidence_path=tmp_path / "failed-formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert failure.execution_failures
    assert failure.evidence_promoted is False
    assert not (tmp_path / "failed-formal.png").exists()


def test_planning_only_mode_remains_zero_browser_actions(monkeypatch, capsys):
    calls = []

    def blocker(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("planning-only mode should not execute")

    monkeypatch.setattr(experiment, "run_live_execution", blocker)
    monkeypatch.setattr(experiment, "_build_executor", blocker)
    monkeypatch.setattr(experiment, "_build_agent_loop_cls", blocker)

    code = experiment.main(
        ["--live-openai"],
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_result())
        ),
    )
    output = capsys.readouterr().out

    assert code == 0
    assert calls == []
    assert "Experiment increment: live OpenAI Wikipedia planning" in output
    assert "Browser actions: no" in output
