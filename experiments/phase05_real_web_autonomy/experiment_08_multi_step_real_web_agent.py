"""Phase 05 Experiment 08: multi-step real-web AgentLoop harness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time
from typing import Callable, Sequence
from numbers import Real

from PIL import Image

from computer_agent.agent import (
    AgentLoop,
    AgentLoopResult,
    AgentLoopStatus,
    AgentStatus,
    TextInputObservation,
)
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import GroundingResult, GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception import (
    BoundingBox,
    MacOSAccessibility,
    PerceptionEngine,
    PerceptionSnapshot,
    ScreenFrame,
    SemanticAXElement,
    UIElement,
    Viewport,
    normalize_ui_text,
)
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase05_real_web_autonomy"
CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_08_multi_step_real_web_agent_candidate.png"
)
FORMAL_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_08_multi_step_real_web_agent.png"
)

TITLE = "Phase 05 Experiment 08: Multi-Step Real-Web Agent"
TASK_GOAL = "Complete the deterministic multi-step real-web workflow"
TARGET_URL = "https://www.python.org/"
EXPECTED_APPLICATION_NAME = "Google Chrome"
SEARCH_FIELD_TEXT = "Search This Site"
SEARCH_INPUT_TEXT = "typing"
SUBMIT_TARGET_TEXT = "GO"
RESULT_MARKER_TEXT = "Results"
SEARCH_FIELD_TARGET = TargetSpec(
    text=SEARCH_FIELD_TEXT,
    element_types=("text_field",),
    minimum_confidence=0.70,
)
SUBMIT_TARGET_SPEC = TargetSpec(
    text=SUBMIT_TARGET_TEXT,
    element_types=("button",),
    minimum_confidence=0.70,
)
RESULT_MARKER_SPEC = TargetSpec(
    text=RESULT_MARKER_TEXT,
    element_types=("heading",),
    minimum_confidence=0.70,
)
WEB_TEXT_GOAL = "Enter the deterministic search query"
SUBMIT_GOAL = "Submit the deterministic search query"
EXPECTED_ACTION_ORDER = ("click_mouse", "type_text", "click_mouse")
EXPECTED_PLAN_STEPS = 2
EXPECTED_ACTION_EXECUTIONS = 3
DEFAULT_WAIT_SECONDS = 8
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.5
_DEFAULT_VIEWPORT = object()


@dataclass(frozen=True, slots=True)
class WebAgentObservation:
    """One live web observation used by harness checks."""

    application_name: str | None
    viewport: Viewport | None
    snapshot: PerceptionSnapshot
    semantic_elements: tuple[SemanticAXElement, ...]


@dataclass(frozen=True, slots=True)
class LivePreconditionReport:
    """Read-only live precondition evidence."""

    observation: WebAgentObservation | None
    search_grounding: GroundingResult | None
    submit_grounding: GroundingResult | None
    results_grounding: GroundingResult | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalObservationReport:
    """Final post-run observation and result-page grounding."""

    observation: WebAgentObservation
    results_grounding: GroundingResult


class LiveWebEnvironment:
    """Shared production Accessibility and perception observation stack."""

    def __init__(
        self,
        *,
        capture_path: str | Path,
        accessibility: MacOSAccessibility | None = None,
        perception_engine: PerceptionEngine | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
    ) -> None:
        if not callable(sleeper):
            raise ValueError("sleeper must be callable")

        self.capture_path = Path(capture_path)
        self.accessibility = (
            accessibility if accessibility is not None else MacOSAccessibility()
        )
        self.perception_engine = (
            perception_engine
            if perception_engine is not None
            else _build_perception_engine(
                self.capture_path,
                self.accessibility,
            )
        )
        self.sleeper = sleeper
        self.stabilization_wait_seconds = _validate_stabilization_wait_seconds(
            stabilization_wait_seconds
        )

    def observe(self) -> WebAgentObservation:
        self.sleeper(self.stabilization_wait_seconds)
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.perception_engine.observe()
        return WebAgentObservation(
            application_name=self.accessibility.read_frontmost_application_name(),
            viewport=self.accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(
                self.accessibility.read_frontmost_semantic_elements()
            ),
        )

    def text_input_observe(self) -> TextInputObservation:
        observation = self.observe()
        return TextInputObservation(
            application_name=observation.application_name,
            viewport=observation.viewport,
            snapshot=observation.snapshot,
            semantic_elements=observation.semantic_elements,
        )


class SyntheticPerceptionEngine:
    """Return queued synthetic snapshots for the submit click step."""

    def __init__(self, snapshots: Sequence[PerceptionSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self.calls = 0

    def observe(self) -> PerceptionSnapshot:
        self.calls += 1
        if not self._snapshots:
            raise AssertionError("unexpected synthetic perception call")

        return self._snapshots.pop(0)


class SyntheticTextInputObserver:
    """Return queued browser observations for TextInputController."""

    def __init__(self, observations: Sequence[TextInputObservation]) -> None:
        self._observations = list(observations)
        self.calls = 0

    def observe(self) -> TextInputObservation:
        self.calls += 1
        if not self._observations:
            raise AssertionError("unexpected synthetic text-input observe")

        return self._observations.pop(0)


class SyntheticExecutor:
    """Execute synthetic actions with configurable tool failures."""

    def __init__(
        self,
        *,
        fail_tool_call: int | None = None,
    ) -> None:
        self.fail_tool_call = fail_tool_call
        self.actions: list[Action] = []

    def execute(self, action: Action) -> ToolResult:
        self.actions.append(action)
        success = self.fail_tool_call != len(self.actions)
        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            output=dict(action.arguments) if success else None,
            error=None if success else "synthetic tool failure",
        )


def build_structured_plan() -> StructuredPlan:
    """Build the deterministic two-step plan for Experiment 05.08."""

    return StructuredPlan(
        task_goal=TASK_GOAL,
        steps=(
            WebTextInputStep(
                goal=WEB_TEXT_GOAL,
                target=SEARCH_FIELD_TARGET,
                input_text=SEARCH_INPUT_TEXT,
                max_attempts=1,
            ),
            PlanStep(
                goal=SUBMIT_GOAL,
                operation=PlanOperation.CLICK_TARGET,
                action_target=SUBMIT_TARGET_SPEC,
                verification_target=RESULT_MARKER_SPEC,
                max_attempts=1,
            ),
        ),
    )


def run_synthetic_agent_loop(
    plan: StructuredPlan,
    *,
    text_input_observations: Sequence[TextInputObservation] | None = None,
    submit_snapshots: Sequence[PerceptionSnapshot] | None = None,
    executor: SyntheticExecutor | None = None,
) -> AgentLoopResult:
    """Run the production AgentLoop against deterministic fake web state."""

    text_observer = SyntheticTextInputObserver(
        text_input_observations
        if text_input_observations is not None
        else _successful_text_input_observations()
    )
    perception_engine = SyntheticPerceptionEngine(
        submit_snapshots
        if submit_snapshots is not None
        else _successful_submit_snapshots()
    )
    if executor is None:
        executor = SyntheticExecutor()

    return AgentLoop(
        perception_engine=perception_engine,
        executor=executor,
        web_text_input_observer=text_observer.observe,
    ).run(plan)


def run_live_agent_loop(
    plan: StructuredPlan,
    *,
    environment: LiveWebEnvironment,
    executor: object,
) -> AgentLoopResult:
    """Run the production AgentLoop against live python.org state."""

    return AgentLoop(
        perception_engine=environment.perception_engine,
        executor=executor,
        web_text_input_observer=environment.text_input_observe,
    ).run(plan)


def live_precondition_failures(
    *,
    platform_name: str = sys.platform,
    accessibility_cls=MacOSAccessibility,
    environment: object | None = None,
    observer: Callable[[], WebAgentObservation] | None = None,
) -> LivePreconditionReport:
    """Check all live execute-mode preconditions without actions."""

    failures = list(
        _platform_accessibility_failures(
            platform_name=platform_name,
            accessibility_cls=accessibility_cls,
        )
    )
    if failures:
        return LivePreconditionReport(
            observation=None,
            search_grounding=None,
            submit_grounding=None,
            results_grounding=None,
            failures=tuple(failures),
        )

    if observer is None:
        if environment is None:
            raise ValueError("environment or observer is required")
        observer = environment.observe

    observation = observer()
    grounder = UIGrounder()
    search_grounding = grounder.ground(
        SEARCH_FIELD_TARGET,
        observation.snapshot.fused_elements,
    )
    submit_grounding = grounder.ground(
        SUBMIT_TARGET_SPEC,
        observation.snapshot.fused_elements,
    )
    results_grounding = grounder.ground(
        RESULT_MARKER_SPEC,
        observation.snapshot.fused_elements,
    )

    if observation.application_name != EXPECTED_APPLICATION_NAME:
        failures.append(
            "frontmost app was "
            f"{observation.application_name}, not {EXPECTED_APPLICATION_NAME}"
        )

    if observation.viewport is None:
        failures.append("viewport was unavailable")

    if observation.snapshot.warnings:
        failures.append(
            f"perception warnings were {observation.snapshot.warnings}"
        )

    if search_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "Search This Site grounding was "
            f"{search_grounding.status.value}"
        )
    elif normalize_ui_text(search_grounding.element.element_type) != "text field":
        failures.append(
            "Search This Site resolved role was "
            f"{search_grounding.element.element_type}"
        )
    elif _value_is_non_empty(search_grounding.element.value):
        failures.append("Search This Site field was not empty")

    if submit_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(f"GO grounding was {submit_grounding.status.value}")
    elif normalize_ui_text(submit_grounding.element.element_type) != "button":
        failures.append(f"GO resolved role was {submit_grounding.element.element_type}")

    if results_grounding.status is not GroundingStatus.NOT_FOUND:
        failures.append(
            "Results grounding was "
            f"{results_grounding.status.value}, not not_found"
        )

    return LivePreconditionReport(
        observation=observation,
        search_grounding=search_grounding,
        submit_grounding=submit_grounding,
        results_grounding=results_grounding,
        failures=tuple(failures),
    )


def acceptance_failures(
    result: object,
    plan: StructuredPlan,
    *,
    final_report: FinalObservationReport | None = None,
    require_final_report: bool = False,
) -> tuple[str, ...]:
    """Return deterministic Experiment 05.08 acceptance failures."""

    failures = list(plan_acceptance_failures(plan))
    if not isinstance(result, AgentLoopResult):
        failures.append(
            "AgentLoop result was not an AgentLoopResult: "
            f"{type(result).__name__}"
        )
        return tuple(failures)

    records = tuple(result.state.steps)
    action_order = tuple(record.action.tool_name for record in records)

    if result.plan is not plan:
        failures.append("AgentLoopResult plan was not the input plan")

    if result.status is not AgentLoopStatus.COMPLETED:
        failures.append(f"AgentLoopResult status was {result.status.value}")

    if result.state.status is not AgentStatus.SUCCEEDED:
        failures.append(f"AgentState status was {result.state.status.value}")

    if result.completed_plan_steps != EXPECTED_PLAN_STEPS:
        failures.append(
            "completed plan steps were "
            f"{result.completed_plan_steps}, not {EXPECTED_PLAN_STEPS}"
        )

    if len(records) != EXPECTED_ACTION_EXECUTIONS:
        failures.append(
            "AgentState record count was "
            f"{len(records)}, not {EXPECTED_ACTION_EXECUTIONS}"
        )

    if action_order != EXPECTED_ACTION_ORDER:
        failures.append(
            "action tool order was "
            f"{action_order}, not {EXPECTED_ACTION_ORDER}"
        )

    for index, record in enumerate(records, start=1):
        if not record.result.success:
            failures.append(
                f"record {index} ToolResult failed: {record.result.error}"
            )

    if action_order.count("scroll") != 0:
        failures.append("deterministic happy path recorded scroll actions")

    if action_order[:2] != ("click_mouse", "type_text"):
        failures.append("text-input actions were not recorded exactly once")

    if action_order.count("type_text") != 1:
        failures.append("type_text was not recorded exactly once")

    if action_order.count("click_mouse") != 2:
        failures.append("click_mouse was not recorded exactly twice")

    if len(records) == EXPECTED_ACTION_EXECUTIONS:
        focus_click = records[0]
        submit_click = records[2]
        if focus_click.action.tool_name != "click_mouse":
            failures.append("focus click was not recorded exactly once")
        if submit_click.action.tool_name != "click_mouse":
            failures.append("submit click was not recorded exactly once")
        if focus_click.action.action_id == submit_click.action.action_id:
            failures.append("focus and submit clicks shared an action_id")

    if final_report is None:
        if require_final_report:
            failures.append("final live observation report was missing")
        return tuple(failures)

    if final_report.results_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "final Results grounding was "
            f"{final_report.results_grounding.status.value}"
        )

    if final_report.observation.application_name != EXPECTED_APPLICATION_NAME:
        failures.append(
            "final frontmost app was "
            f"{final_report.observation.application_name}, "
            f"not {EXPECTED_APPLICATION_NAME}"
        )

    if final_report.observation.snapshot.warnings:
        failures.append(
            "final perception warnings were "
            f"{final_report.observation.snapshot.warnings}"
        )

    return tuple(failures)


def plan_acceptance_failures(plan: StructuredPlan) -> tuple[str, ...]:
    """Return deterministic plan-shape failures."""

    failures: list[str] = []

    if not isinstance(plan, StructuredPlan):
        return (f"plan was not a StructuredPlan: {type(plan).__name__}",)

    if len(plan.steps) != EXPECTED_PLAN_STEPS:
        failures.append(
            f"plan step count was {len(plan.steps)}, not {EXPECTED_PLAN_STEPS}"
        )
        return tuple(failures)

    first, second = plan.steps
    if not isinstance(first, WebTextInputStep):
        failures.append(f"step 1 was {type(first).__name__}")
    elif first.operation is not PlanOperation.TYPE_INTO_TARGET:
        failures.append(f"step 1 operation was {first.operation}")
    else:
        if first.goal != WEB_TEXT_GOAL:
            failures.append(f"step 1 goal was {first.goal}")
        if first.target != SEARCH_FIELD_TARGET:
            failures.append(f"step 1 target was {first.target}")
        if first.input_text != SEARCH_INPUT_TEXT:
            failures.append(f"step 1 input_text was {first.input_text}")
        if first.max_attempts != 1:
            failures.append(f"step 1 max_attempts was {first.max_attempts}")

    if not isinstance(second, PlanStep):
        failures.append(f"step 2 was {type(second).__name__}")
    elif second.operation is not PlanOperation.CLICK_TARGET:
        failures.append(f"step 2 operation was {second.operation}")
    else:
        if second.goal != SUBMIT_GOAL:
            failures.append(f"step 2 goal was {second.goal}")
        if second.action_target != SUBMIT_TARGET_SPEC:
            failures.append(f"step 2 action target was {second.action_target}")
        if second.verification_target != RESULT_MARKER_SPEC:
            failures.append(
                f"step 2 verification target was {second.verification_target}"
            )
        if second.max_attempts != 1:
            failures.append(f"step 2 max_attempts was {second.max_attempts}")

    failures.extend(_plan_payload_failures(plan))
    return tuple(failures)


def print_acceptance_result(
    result: object,
    plan: StructuredPlan,
    *,
    mode: str,
    preconditions: LivePreconditionReport | None = None,
    final_report: FinalObservationReport | None = None,
    require_final_report: bool = False,
    extra_failures: tuple[str, ...] = (),
) -> int:
    """Print the deterministic Experiment 05.08 acceptance summary."""

    print(TITLE)
    print(f"Experiment mode: {mode}")
    print(f"Live browser actions: {_yes_no(mode == 'live')}")
    print("OpenAI request: no")
    print(f"Synthetic execution: {_yes_no(mode == 'synthetic')}")
    print(f"Task goal: {plan.task_goal}")
    print(f"Plan steps: {len(plan.steps)}")
    for index, step in enumerate(plan.steps, start=1):
        print(
            f"Step {index}: {step.goal} "
            f"({step.operation.value}, max_attempts={step.max_attempts})"
        )

    if preconditions is not None:
        _print_preconditions(preconditions)

    if isinstance(result, AgentLoopResult):
        _print_agent_loop_result(result, plan)

    if final_report is not None:
        print(
            "Final Results grounding status: "
            f"{final_report.results_grounding.status.value}"
        )
        print(f"Final app: {final_report.observation.application_name}")
        print(f"Final warnings: {final_report.observation.snapshot.warnings}")

    failures = (
        acceptance_failures(
            result,
            plan,
            final_report=final_report,
            require_final_report=require_final_report,
        )
        + extra_failures
    )
    if failures:
        print("Experiment acceptance: failed")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("Experiment acceptance: passed")
    return 0


def run_acceptance(
    *,
    runner=run_synthetic_agent_loop,
) -> int:
    """Run synthetic acceptance through the production AgentLoop path."""

    plan = build_structured_plan()
    result = runner(plan)
    return print_acceptance_result(
        result,
        plan,
        mode="synthetic",
    )


def run_live_acceptance(
    *,
    environment_builder=LiveWebEnvironment,
    executor_builder=None,
    platform_name: str = sys.platform,
    accessibility_cls=MacOSAccessibility,
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = FORMAL_EVIDENCE_PATH,
    sleeper: Callable[[float], None] = time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
) -> int:
    """Run live acceptance through one production AgentLoop call."""

    plan = build_structured_plan()
    print(TITLE)
    print("Experiment mode: live")
    print(f"Target URL: {TARGET_URL}")
    print("Required initial state: Google Chrome frontmost")
    print("Required initial state: python.org open at the top")
    print("Required initial state: Search This Site field empty")
    _countdown(wait_seconds, sleeper=sleeper)

    early_failures = _platform_accessibility_failures(
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
    )
    if early_failures:
        preconditions = LivePreconditionReport(
            observation=None,
            search_grounding=None,
            submit_grounding=None,
            results_grounding=None,
            failures=early_failures,
        )
        _print_precondition_failure(plan, preconditions)
        return 1

    environment = environment_builder(
        capture_path=capture_path,
        sleeper=sleeper,
        stabilization_wait_seconds=stabilization_wait_seconds,
    )
    preconditions = live_precondition_failures(
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
        environment=environment,
    )
    if preconditions.failures:
        _print_precondition_failure(plan, preconditions)
        return 1

    if executor_builder is None:
        executor_builder = _build_executor
    executor = executor_builder()

    result = run_live_agent_loop(
        plan,
        environment=environment,
        executor=executor,
    )
    final_report = _final_observation_report(environment.observe())
    evidence_failure = ()
    if not Path(capture_path).exists():
        evidence_failure = ("candidate evidence file was missing",)

    code = print_acceptance_result(
        result,
        plan,
        mode="live",
        preconditions=preconditions,
        final_report=final_report,
        require_final_report=True,
        extra_failures=evidence_failure,
    )
    if code == 0:
        _promote_evidence(
            candidate_path=Path(capture_path),
            formal_path=Path(formal_evidence_path),
        )

    return code


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 05.08 multi-step real-web AgentLoop acceptance. "
            "Default mode is offline synthetic execution."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the live python.org workflow through production tools.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to count down before live observations/actions.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    synthetic_runner=run_synthetic_agent_loop,
    live_runner=run_live_acceptance,
) -> int:
    args = _parse_args(argv)
    if not args.execute:
        return run_acceptance(runner=synthetic_runner)

    return live_runner(wait_seconds=args.wait_seconds)


def _build_perception_engine(
    capture_path: str | Path,
    accessibility: MacOSAccessibility,
) -> PerceptionEngine:
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import ScreenCapture, TesseractOCR, UIElementFusion

    return PerceptionEngine(
        screen_capture=ScreenCapture(ComputerController()),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=0.05,
            page_segmentation_mode=6,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )


def _build_executor():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.tools.computer import create_computer_tools
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    return ToolExecutor(ToolRegistry(create_computer_tools(ComputerController())))


def _successful_text_input_observations() -> tuple[TextInputObservation, ...]:
    return (
        _text_input_observation(value=None, seconds=0),
        _text_input_observation(value=SEARCH_INPUT_TEXT, seconds=1),
    )


def _successful_submit_snapshots() -> tuple[PerceptionSnapshot, ...]:
    return (
        _snapshot(
            elements=(_submit_element(),),
            seconds=2,
        ),
        _snapshot(
            elements=(
                _submit_element(),
                _result_marker_element(),
            ),
            seconds=3,
        ),
    )


def _text_input_observation(
    *,
    value: str | None,
    application_name: str | None = EXPECTED_APPLICATION_NAME,
    viewport: object = _DEFAULT_VIEWPORT,
    warnings: tuple[str, ...] = (),
    seconds: int = 0,
) -> TextInputObservation:
    field = _search_field_element(value=value)
    return TextInputObservation(
        application_name=application_name,
        viewport=_viewport() if viewport is _DEFAULT_VIEWPORT else viewport,
        snapshot=_snapshot(
            elements=(field,),
            warnings=warnings,
            seconds=seconds,
        ),
        semantic_elements=(
            SemanticAXElement(
                role="AXTextField",
                text=SEARCH_FIELD_TEXT,
                bounds=field.bounding_box,
                value=value,
            ),
        ),
    )


def _web_observation(
    *,
    app: str | None = EXPECTED_APPLICATION_NAME,
    viewport: object = _DEFAULT_VIEWPORT,
    elements: Sequence[UIElement] = (),
    warnings: tuple[str, ...] = (),
    seconds: int = 0,
) -> WebAgentObservation:
    return WebAgentObservation(
        application_name=app,
        viewport=_viewport() if viewport is _DEFAULT_VIEWPORT else viewport,
        snapshot=_snapshot(
            elements=elements,
            warnings=warnings,
            seconds=seconds,
        ),
        semantic_elements=tuple(
            SemanticAXElement(
                role=_role_for_element(element),
                text=element.text,
                bounds=element.bounding_box,
                value=element.value,
            )
            for element in elements
        ),
    )


def _snapshot(
    *,
    elements: Sequence[UIElement],
    warnings: tuple[str, ...] = (),
    seconds: int,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"synthetic-05-08-{seconds}.png"),
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
        warnings=warnings,
    )


def _search_field_element(
    *,
    value: str | None,
) -> UIElement:
    return UIElement(
        element_type="text_field",
        bounding_box=BoundingBox(x=100, y=120, width=240, height=36),
        confidence=0.95,
        text=SEARCH_FIELD_TEXT,
        value=value,
        enabled=True,
        source="accessibility",
    )


def _submit_element() -> UIElement:
    return UIElement(
        element_type="button",
        bounding_box=BoundingBox(x=360, y=120, width=90, height=36),
        confidence=0.95,
        text=SUBMIT_TARGET_TEXT,
        enabled=True,
        source="accessibility",
    )


def _result_marker_element() -> UIElement:
    return UIElement(
        element_type="heading",
        bounding_box=BoundingBox(x=100, y=220, width=260, height=40),
        confidence=0.95,
        text=RESULT_MARKER_TEXT,
        enabled=True,
        source="accessibility",
    )


def _viewport() -> Viewport:
    return Viewport(BoundingBox(x=0, y=0, width=1000, height=700))


def _time(seconds: int) -> datetime:
    return datetime(
        2026,
        9,
        8,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _final_observation_report(
    observation: WebAgentObservation,
) -> FinalObservationReport:
    grounding = UIGrounder().ground(
        RESULT_MARKER_SPEC,
        observation.snapshot.fused_elements,
    )
    return FinalObservationReport(
        observation=observation,
        results_grounding=grounding,
    )


def _print_preconditions(preconditions: LivePreconditionReport) -> None:
    observation = preconditions.observation
    if observation is not None:
        print(f"Initial app: {observation.application_name}")
        print(f"Initial viewport: {observation.viewport}")
    if preconditions.search_grounding is not None:
        print(
            "Initial Search This Site grounding status: "
            f"{preconditions.search_grounding.status.value}"
        )
    if preconditions.submit_grounding is not None:
        print(
            "Initial GO grounding status: "
            f"{preconditions.submit_grounding.status.value}"
        )
    if preconditions.results_grounding is not None:
        print(
            "Initial Results grounding status: "
            f"{preconditions.results_grounding.status.value}"
        )


def _print_precondition_failure(
    plan: StructuredPlan,
    preconditions: LivePreconditionReport,
) -> None:
    print(f"Task goal: {plan.task_goal}")
    print(f"Plan steps: {len(plan.steps)}")
    _print_preconditions(preconditions)
    print("Experiment acceptance: failed")
    for failure in preconditions.failures:
        print(f"  {failure}")


def _print_agent_loop_result(
    result: AgentLoopResult,
    plan: StructuredPlan,
) -> None:
    print(f"AgentLoopResult reason: {result.reason}")
    print(f"Agent loop status: {result.status.value}")
    print(f"Agent state: {result.state.status.value}")
    print(
        "Completed plan steps: "
        f"{result.completed_plan_steps} / {len(plan.steps)}"
    )
    print(f"Action executions: {len(result.state.steps)}")
    print(
        "Action tool order: "
        f"{tuple(record.action.tool_name for record in result.state.steps)}"
    )
    for index, record in enumerate(result.state.steps, start=1):
        print(f"Record {index} tool: {record.action.tool_name}")
        print(f"Record {index} arguments: {record.action.arguments}")
        print(f"Record {index} ToolResult success: {record.result.success}")
        print(f"Record {index} ToolResult error: {record.result.error}")


def _plan_payload_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []

    if any(
        isinstance(value, (Action, ToolResult))
        for step in plan.steps
        for value in _step_values(step)
    ):
        failures.append("plan contained Action or ToolResult objects")

    if any(
        getattr(step, "target", None) is not None
        and getattr(step.target, "reference_point", None) is not None
        for step in plan.steps
    ) or any(
        isinstance(step, PlanStep)
        and (
            step.action_target.reference_point is not None
            or step.verification_target.reference_point is not None
        )
        for step in plan.steps
    ):
        failures.append("StructuredPlan contained runtime coordinates")

    return tuple(failures)


def _platform_accessibility_failures(
    *,
    platform_name: str,
    accessibility_cls,
) -> tuple[str, ...]:
    failures: list[str] = []
    if platform_name != "darwin":
        failures.append("platform is not macOS")

    if not accessibility_cls.is_available():
        failures.append("macOS Accessibility is unavailable")

    if not accessibility_cls.is_trusted():
        failures.append("macOS Accessibility is not trusted")

    return tuple(failures)


def _step_values(step: object) -> tuple[object, ...]:
    return tuple(getattr(step, field.name) for field in fields(step))


def _value_is_non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _role_for_element(element: UIElement) -> str:
    normalized = normalize_ui_text(element.element_type)
    if normalized == "text field":
        return "AXTextField"
    if normalized == "button":
        return "AXButton"
    if normalized == "heading":
        return "AXHeading"
    return "AXStaticText"


def _countdown(seconds: int, *, sleeper: Callable[[float], None]) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"{remaining}...")
        sleeper(1)


def _wait_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be an integer"
        ) from error

    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be from 0 through 30"
        )

    return seconds


def _promote_evidence(
    *,
    candidate_path: Path,
    formal_path: Path,
) -> bool:
    if not candidate_path.exists():
        return False
    formal_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, formal_path)
    return True


def _validate_stabilization_wait_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("stabilization_wait_seconds must be numeric")

    seconds = float(value)
    if seconds < 0:
        raise ValueError(
            "stabilization_wait_seconds must be non-negative"
        )

    return seconds


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
