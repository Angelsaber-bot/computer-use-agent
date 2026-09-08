from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys

from PIL import Image
import pytest

from computer_agent.agent import (
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    AgentStatus,
)
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import TargetSpec
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    SemanticAXElement,
    UIElement,
    Viewport,
)
from computer_agent.planning import (
    InsertTextStep,
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from computer_agent.reasoning import LLMReasoner, ReasoningStatus
from experiments.phase05_real_web_autonomy import (
    experiment_09_live_openai_web_agent as experiment,
)


SCRIPT_PATH = Path(experiment.__file__).resolve()
DEFAULT_VIEWPORT = object()


class RecordingReasoner:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[str] = []

    def reason(self, task: str):
        self.calls.append(task)
        return self.result


class RecordingReasonerBuilder:
    def __init__(self, reasoner: RecordingReasoner) -> None:
        self.reasoner = reasoner
        self.calls = 0

    def __call__(self) -> RecordingReasoner:
        self.calls += 1
        return self.reasoner


class RecordingLLMClient:
    def __init__(self, response: str | None = None) -> None:
        self.response = response if response is not None else experiment._offline_response()
        self.calls: list[dict[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.response


@dataclass(frozen=True, slots=True)
class CoordinateStep:
    goal: str = "Coordinate action"
    operation: str = "click_target"
    x: int = 10
    y: int = 20


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
        return FakeEnvironment(
            capture_path=capture_path,
            **self.kwargs,
        )


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
        executor,
        web_text_input_observer,
    ) -> None:
        self.perception_engine = perception_engine
        self.executor = executor
        self.web_text_input_observer = web_text_input_observer
        self.run_calls = []
        type(self).instances.append(self)

    def run(self, plan):
        self.run_calls.append(plan)
        if type(self).result is None:
            type(self).result = _loop_result(plan)
        return type(self).result


def _path_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(SCRIPT_PATH.parents[2] / "src")
        + os.pathsep
        + str(SCRIPT_PATH.parents[2])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return env


def _time(seconds: int = 0) -> datetime:
    return datetime(
        2026,
        9,
        8,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _box(
    *,
    x: int = 100,
    y: int = 120,
    width: int = 240,
    height: int = 36,
) -> BoundingBox:
    return BoundingBox(x=x, y=y, width=width, height=height)


def _element(
    *,
    text: str,
    element_type: str,
    value=None,
    x: int = 100,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x=x),
        confidence=0.95,
        text=text,
        value=value,
        enabled=True,
        source="accessibility",
    )


def _search_field(*, value=None, element_type: str = "text_field") -> UIElement:
    return _element(
        text=experiment.SEARCH_FIELD_TEXT,
        element_type=element_type,
        value=value,
        x=100,
    )


def _go_button(*, element_type: str = "button") -> UIElement:
    return _element(
        text=experiment.SUBMIT_TARGET_TEXT,
        element_type=element_type,
        x=360,
    )


def _results_marker() -> UIElement:
    return _element(
        text=experiment.RESULT_MARKER_TEXT,
        element_type="heading",
        x=100,
    )


def _initial_elements(
    *,
    search_value=None,
    search_type: str = "text_field",
    go_type: str = "button",
):
    return (
        _search_field(value=search_value, element_type=search_type),
        _go_button(element_type=go_type),
    )


def _final_elements():
    return (
        _go_button(),
        _results_marker(),
    )


def _viewport() -> Viewport:
    return Viewport(BoundingBox(x=0, y=0, width=1000, height=700))


def _snapshot(*, elements=(), warnings=(), seconds: int = 0):
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"synthetic-05-09-{seconds}.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=_time(seconds),
        ),
        image=Image.new("RGB", (1000, 700)),
        accessibility_elements=tuple(elements),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=tuple(warnings),
    )


def _observation(
    *,
    app: str | None = experiment.EXPECTED_APPLICATION_NAME,
    viewport=DEFAULT_VIEWPORT,
    elements=(),
    warnings=(),
    seconds: int = 0,
):
    return experiment.WebAgentObservation(
        application_name=app,
        viewport=_viewport() if viewport is DEFAULT_VIEWPORT else viewport,
        snapshot=_snapshot(
            elements=elements,
            warnings=warnings,
            seconds=seconds,
        ),
        semantic_elements=tuple(
            SemanticAXElement(
                role="AXTextField",
                text=element.text,
                bounds=element.bounding_box,
                value=element.value,
            )
            for element in elements
        ),
    )


def _trusted_result(plan: StructuredPlan | None = None):
    client = RecordingLLMClient()
    reasoner = LLMReasoner(client=client)
    if plan is None:
        return reasoner.reason(experiment.TASK_INTENT)

    result = reasoner.reason(experiment.TASK_INTENT)
    object.__setattr__(result, "plan", plan)
    return result


def _reasoner_with_client(response: str | None = None):
    client = RecordingLLMClient(response=response)
    reasoner = LLMReasoner(client=client)
    return reasoner, client


def _plan(
    *,
    steps=None,
    search_text: str = experiment.SEARCH_FIELD_TEXT,
    search_types: tuple[str, ...] = ("text_field",),
    input_text: str = experiment.SEARCH_INPUT_TEXT,
    go_text: str = experiment.SUBMIT_TARGET_TEXT,
    go_types: tuple[str, ...] = ("button",),
    results_text: str = experiment.RESULT_MARKER_TEXT,
    results_types: tuple[str, ...] = (),
) -> StructuredPlan:
    if steps is None:
        steps = (
            WebTextInputStep(
                goal="Enter the search query",
                target=TargetSpec(
                    text=search_text,
                    element_types=search_types,
                ),
                input_text=input_text,
                max_attempts=1,
            ),
            PlanStep(
                goal="Submit the search query",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=go_text,
                    element_types=go_types,
                ),
                verification_target=TargetSpec(
                    text=results_text,
                    element_types=results_types,
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
) -> AgentLoopResult:
    state_status = (
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
            final_status=state_status,
        ),
        completed_plan_steps=completed_plan_steps,
        reason="fake loop result",
    )


def _planning_report(plan: StructuredPlan):
    return experiment.PlanningReport(
        mode=experiment.LIVE_OPENAI_MODE,
        live_openai_request=True,
        browser_actions=False,
        result=_trusted_result(plan),
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


def _execution_failures(
    tmp_path,
    *,
    plan: StructuredPlan | None = None,
    agent_result: AgentLoopResult | None = None,
    final_report=None,
    candidate_exists: bool = True,
):
    if plan is None:
        plan = _plan()
    candidate_path = tmp_path / "candidate.png"
    if candidate_exists:
        candidate_path.write_text("candidate evidence", encoding="utf-8")
    if agent_result is None:
        agent_result = _loop_result(plan)
    return experiment.execution_acceptance_failures(
        planning=_planning_report(plan),
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


def test_default_direct_run_is_offline_and_makes_no_openai_request():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=SCRIPT_PATH.parents[2],
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Experiment mode: offline" in result.stdout
    assert "Live OpenAI request: no" in result.stdout
    assert "Browser actions: no" in result.stdout
    assert "Experiment acceptance: passed" in result.stdout
    assert result.stderr == ""


def test_offline_fake_plan_passes_acceptance():
    report = experiment.run_planning()

    assert report.mode == experiment.OFFLINE_MODE
    assert report.live_openai_request is False
    assert report.browser_actions is False
    assert report.result.status is ReasoningStatus.READY
    assert report.failures == ()


def test_exactly_two_semantic_steps_required():
    plan = _plan(steps=(_step_1(),))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("plan step count" in failure for failure in failures)


def test_exact_operation_order_required():
    plan = _plan(steps=(_step_2(), _step_1()))

    failures = experiment.plan_acceptance_failures(plan)

    assert failures
    assert any("step 1" in failure for failure in failures)
    assert any("step 2" in failure for failure in failures)


def test_type_into_target_required_for_step_1():
    plan = _plan(steps=(_step_2(), _step_2()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("step 1" in failure for failure in failures)
    assert any("WebTextInputStep" in failure for failure in failures)


def test_exact_input_text_required():
    plan = _plan(input_text="Typing")

    failures = experiment.plan_acceptance_failures(plan)

    assert any("step 1 input_text" in failure for failure in failures)


def test_exact_search_this_site_target_required():
    plan = _plan(search_text="Search")

    failures = experiment.plan_acceptance_failures(plan)

    assert any("step 1 target text" in failure for failure in failures)


def test_go_action_target_required():
    plan = _plan(go_text="Search")

    failures = experiment.plan_acceptance_failures(plan)

    assert any("action target text" in failure for failure in failures)


def test_results_verification_target_required():
    plan = _plan(results_text="Search Results")

    failures = experiment.plan_acceptance_failures(plan)

    assert any("verification target text" in failure for failure in failures)


def test_no_executable_action_payload_is_accepted():
    plan = _plan()
    object.__setattr__(plan, "steps", (Action(tool_name="click_mouse"), _step_2()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("executable Action" in failure for failure in failures)


def test_no_coordinates_are_accepted():
    plan = _plan()
    object.__setattr__(plan, "steps", (CoordinateStep(), _step_2()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("forbidden fields" in failure for failure in failures)
    assert any("x" in failure for failure in failures)


def test_extra_step_is_rejected():
    plan = _plan(steps=(_step_1(), _step_2(), _step_2()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("plan step count" in failure for failure in failures)


def test_incorrect_operation_is_rejected():
    plan = _plan(
        steps=(
            InsertTextStep(
                goal="Insert text",
                value_key="typing",
                max_attempts=1,
            ),
            _step_2(),
        )
    )

    failures = experiment.plan_acceptance_failures(plan)

    assert any("WebTextInputStep" in failure for failure in failures)


def test_unsafe_malformed_fake_provider_response_is_rejected():
    client = experiment.DeterministicFakeLLMClient(response="{")
    reasoner = LLMReasoner(client=client)

    report = experiment.run_planning(
        reasoner_builder=lambda: reasoner,
    )

    assert report.result.status is ReasoningStatus.BLOCKED
    assert report.failures
    assert len(client.calls) == 1


def test_help_makes_no_request():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=SCRIPT_PATH.parents[2],
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--live-openai" in result.stdout
    assert "Experiment acceptance:" not in result.stdout
    assert result.stderr == ""


def test_live_mode_can_be_injected_and_calls_reason_once():
    client = RecordingLLMClient()
    reasoner = LLMReasoner(client=client)
    builder = RecordingReasonerBuilder(RecordingReasoner(reasoner.reason(experiment.TASK_INTENT)))

    report = experiment.run_planning(
        live_openai=True,
        reasoner_builder=builder,
    )

    assert report.mode == experiment.LIVE_OPENAI_MODE
    assert report.live_openai_request is True
    assert report.browser_actions is False
    assert builder.calls == 1
    assert builder.reasoner.calls == [experiment.TASK_INTENT]
    assert report.failures == ()


def test_live_mode_with_real_reasoner_and_fake_provider_makes_one_provider_call():
    client = RecordingLLMClient()
    reasoner = LLMReasoner(client=client)

    report = experiment.run_planning(
        live_openai=True,
        reasoner_builder=lambda: reasoner,
    )

    assert report.live_openai_request is True
    assert report.failures == ()
    assert len(client.calls) == 1
    assert client.calls[0]["user_prompt"] == f"Task intent:\n{experiment.TASK_INTENT}"


def test_results_heading_role_is_rejected_by_acceptance():
    plan = _plan(results_types=("heading",))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("verification target element_types" in failure for failure in failures)


def test_results_text_role_is_rejected_by_acceptance():
    plan = _plan(results_types=("text",))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("verification target element_types" in failure for failure in failures)


def test_identifier_and_reference_point_are_rejected():
    plan = _plan()
    target = TargetSpec(
        text=experiment.SEARCH_FIELD_TEXT,
        identifier="search-field",
        element_types=("text_field",),
        reference_point=(10.0, 20.0),
    )
    step = WebTextInputStep(
        goal="Enter the search query",
        target=target,
        input_text=experiment.SEARCH_INPUT_TEXT,
        max_attempts=1,
    )
    object.__setattr__(plan, "steps", (step, _step_2()))

    failures = experiment.plan_acceptance_failures(plan)

    assert any("identifier" in failure for failure in failures)
    assert any("reference_point" in failure for failure in failures)


def test_live_openai_remains_planning_only_with_injected_reasoner(capsys):
    reasoner, client = _reasoner_with_client()

    code = experiment.main(
        ["--live-openai"],
        reasoner_builder=lambda: reasoner,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment mode: live-openai" in output
    assert "Live OpenAI request: yes" in output
    assert "Browser actions: no" in output
    assert "Experiment acceptance: passed" in output
    assert "Execution acceptance:" not in output
    assert len(client.calls) == 1


def test_execute_without_live_openai_is_rejected_before_actions(capsys):
    reasoner, client = _reasoner_with_client()

    code = experiment.main(
        ["--execute"],
        reasoner_builder=lambda: reasoner,
    )
    output = capsys.readouterr().out

    assert code == 2
    assert "Experiment mode: rejected" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "--execute requires --live-openai" in output
    assert len(client.calls) == 0


@pytest.mark.parametrize(
    ("platform_name", "available", "trusted", "message"),
    [
        ("linux", True, True, "platform is not macOS"),
        ("darwin", False, True, "Accessibility is unavailable"),
        ("darwin", True, False, "Accessibility is not trusted"),
    ],
)
def test_platform_accessibility_failures_block_before_environment_and_executor(
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

    reasoner, client = _reasoner_with_client()
    environment_builder = FakeEnvironmentBuilder()
    executor_builder = FakeExecutorBuilder()

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=environment_builder,
        executor_builder=executor_builder,
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
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            _observation(app="Safari", elements=_initial_elements()),
            "frontmost app",
        ),
        (
            _observation(viewport=None, elements=_initial_elements()),
            "viewport was unavailable",
        ),
        (
            _observation(elements=_initial_elements(), warnings=("warn",)),
            "perception warnings",
        ),
        (
            _observation(elements=(_go_button(),)),
            "Search This Site grounding",
        ),
        (
            _observation(elements=_initial_elements(search_value="typing")),
            "field was not empty",
        ),
        (
            _observation(elements=(_search_field(),)),
            "GO grounding",
        ),
        (
            _observation(elements=(*_initial_elements(), _results_marker())),
            "Results grounding",
        ),
    ],
)
def test_live_precondition_failures_block_before_executor(
    tmp_path,
    observation,
    message,
):
    reasoner, client = _reasoner_with_client()
    executor_builder = FakeExecutorBuilder()
    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(
            initial_observation=observation
        ),
        executor_builder=executor_builder,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert any(message in failure for failure in report.execution_failures)
    assert executor_builder.calls == 0
    assert len(client.calls) == 1


def test_live_execution_uses_one_agent_loop_run_and_promotes_evidence(
    monkeypatch,
    tmp_path,
):
    reasoner, client = _reasoner_with_client()
    executor_builder = FakeExecutorBuilder()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=executor_builder,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.execution_failures == ()
    assert report.evidence_promoted is True
    assert (tmp_path / "formal.png").read_text(encoding="utf-8") == (
        "candidate evidence"
    )
    assert len(FakeAgentLoop.instances) == 1
    loop = FakeAgentLoop.instances[0]
    assert loop.run_calls == [report.planning.result.plan]
    assert executor_builder.calls == 1
    assert len(client.calls) == 1


def test_successful_execution_reporting_uses_stage_acceptance_labels(
    monkeypatch,
    capsys,
    tmp_path,
):
    reasoner, _client = _reasoner_with_client()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    assert "Planning acceptance: passed" in output
    assert "Execution acceptance: passed" in output
    assert "Experiment acceptance: passed" not in output


def test_failed_execution_reporting_does_not_print_experiment_success(
    monkeypatch,
    capsys,
    tmp_path,
):
    reasoner, _client = _reasoner_with_client()
    plan = _plan()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = _loop_result(
        plan,
        status=AgentLoopStatus.EXHAUSTED,
        completed_plan_steps=1,
        tools=("click_mouse", "type_text"),
    )
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    assert "Planning acceptance: passed" in output
    assert "Execution acceptance: failed" in output
    assert "Experiment acceptance: passed" not in output


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        (("click_mouse", "type_text"), "executed action count"),
        (("click_mouse", "click_mouse", "type_text"), "action tool order"),
        (("click_mouse", "type_text", "scroll"), "scroll"),
        (("click_mouse", "type_text", "type_text"), "type_text"),
        (("type_text", "click_mouse", "click_mouse"), "action tool order"),
    ],
)
def test_execution_acceptance_rejects_wrong_action_sequences(
    tmp_path,
    tools,
    message,
):
    plan = _plan()

    failures = _execution_failures(
        tmp_path,
        plan=plan,
        agent_result=_loop_result(plan, tools=tools),
    )

    assert any(message in failure for failure in failures)


def test_execution_acceptance_rejects_duplicate_or_missing_clicks(tmp_path):
    plan = _plan()

    failures = _execution_failures(
        tmp_path,
        plan=plan,
        agent_result=_loop_result(
            plan,
            tools=("click_mouse", "type_text", "type_text"),
        ),
    )

    assert any("click_mouse" in failure for failure in failures)


def test_execution_acceptance_rejects_failed_tool_result(tmp_path):
    plan = _plan()

    failures = _execution_failures(
        tmp_path,
        plan=plan,
        agent_result=_loop_result(plan, failed_record=2),
    )

    assert any("ToolResult failed" in failure for failure in failures)


def test_execution_acceptance_rejects_incomplete_agent_loop(tmp_path):
    plan = _plan()

    failures = _execution_failures(
        tmp_path,
        plan=plan,
        agent_result=_loop_result(
            plan,
            status=AgentLoopStatus.EXHAUSTED,
            completed_plan_steps=1,
            tools=("click_mouse", "type_text"),
        ),
    )

    assert any("AgentLoopResult status" in failure for failure in failures)


def test_execution_acceptance_rejects_wrong_completed_step_count(tmp_path):
    plan = _plan()

    failures = _execution_failures(
        tmp_path,
        plan=plan,
        agent_result=_loop_result(
            plan,
            status=AgentLoopStatus.EXHAUSTED,
            completed_plan_steps=1,
            tools=experiment.EXPECTED_ACTION_ORDER,
        ),
    )

    assert any("completed plan steps" in failure for failure in failures)


def test_execution_acceptance_rejects_unresolved_final_results(tmp_path):
    failures = _execution_failures(
        tmp_path,
        final_report=_final_report(
            observation=_observation(elements=(_go_button(),))
        ),
    )

    assert any("final Results grounding" in failure for failure in failures)


def test_execution_acceptance_rejects_wrong_final_app(tmp_path):
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


def test_missing_candidate_evidence_fails_acceptance(tmp_path):
    failures = _execution_failures(tmp_path, candidate_exists=False)

    assert any("candidate evidence file was missing" in failure for failure in failures)


def test_missing_candidate_reporting_does_not_print_experiment_success(
    monkeypatch,
    capsys,
    tmp_path,
):
    reasoner, _client = _reasoner_with_client()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(write_candidate=False),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    experiment.print_execution_report(report)
    output = capsys.readouterr().out

    assert "Planning acceptance: passed" in output
    assert "Execution acceptance: failed" in output
    assert "candidate evidence file was missing" in output
    assert "Experiment acceptance: passed" not in output


def test_failed_execution_does_not_promote_formal_evidence(monkeypatch, tmp_path):
    reasoner, _client = _reasoner_with_client()
    FakeAgentLoop.instances = []
    plan = _plan()
    FakeAgentLoop.result = _loop_result(
        plan,
        status=AgentLoopStatus.EXHAUSTED,
        completed_plan_steps=1,
        tools=("click_mouse", "type_text"),
    )
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.execution_failures
    assert report.evidence_promoted is False
    assert not (tmp_path / "formal.png").exists()


def test_live_execution_counts_down_before_live_observation(monkeypatch, tmp_path):
    sleeps = []
    reasoner, _client = _reasoner_with_client()
    FakeAgentLoop.instances = []
    FakeAgentLoop.result = None
    monkeypatch.setattr(experiment, "AgentLoop", FakeAgentLoop)

    report = experiment.run_live_execution(
        reasoner_builder=lambda: reasoner,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=sleeps.append,
        wait_seconds=2,
        stabilization_wait_seconds=0.25,
    )

    assert report.execution_failures == ()
    assert sleeps[:2] == [1, 1]
