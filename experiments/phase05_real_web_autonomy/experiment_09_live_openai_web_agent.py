"""Phase 05 Experiment 09: live OpenAI web-agent planning harness."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import json
from numbers import Real
from pathlib import Path
import shutil
import sys
import time
from typing import Callable, Sequence

from computer_agent.agent import (
    AgentLoop,
    AgentLoopResult,
    AgentLoopStatus,
    AgentStatus,
    TextInputObservation,
)
from computer_agent.core.models import Action
from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    MacOSAccessibility,
    PerceptionEngine,
    PerceptionSnapshot,
    SemanticAXElement,
    Viewport,
    normalize_ui_text,
)
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from computer_agent.reasoning import (
    LLMReasoner,
    ReasoningResult,
    ReasoningStatus,
)
from computer_agent.reasoning.openai_client import OpenAILLMClient


TITLE = "Phase 05 Experiment 09: Live OpenAI Web-Agent Planning"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase05_real_web_autonomy"
CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_09_live_openai_web_agent_candidate.png"
)
FORMAL_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_09_live_openai_web_agent.png"
)
TASK_INTENT = (
    "On python.org, enter the text 'typing' into the 'Search This Site' "
    "field, click the 'GO' button, and verify that 'Results' appears."
)
OFFLINE_MODE = "offline"
LIVE_OPENAI_MODE = "live-openai"
LIVE_EXECUTION_MODE = "live-openai-execute"
EXPECTED_APPLICATION_NAME = "Google Chrome"
SEARCH_FIELD_TEXT = "Search This Site"
SEARCH_INPUT_TEXT = "typing"
SUBMIT_TARGET_TEXT = "GO"
RESULT_MARKER_TEXT = "Results"
EXPECTED_PLAN_STEPS = 2
EXPECTED_ACTION_ORDER = ("click_mouse", "type_text", "click_mouse")
EXPECTED_ACTION_EXECUTIONS = 3
DEFAULT_WAIT_SECONDS = 8
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.5
_SEARCH_FIELD_ELEMENT_TYPES = ((), ("text_field",))
_SUBMIT_ELEMENT_TYPES = ((), ("button",))
_RESULT_ELEMENT_TYPES = ((),)
_SEARCH_FIELD_TARGET = TargetSpec(
    text=SEARCH_FIELD_TEXT,
    element_types=("text_field",),
)
_SUBMIT_TARGET = TargetSpec(
    text=SUBMIT_TARGET_TEXT,
    element_types=("button",),
)
_RESULT_TARGET = TargetSpec(
    text=RESULT_MARKER_TEXT,
    element_types=(),
)
_FORBIDDEN_STEP_FIELD_NAMES = frozenset(
    (
        "x",
        "y",
        "coordinates",
        "tool_name",
        "arguments",
        "raw_arguments",
        "tool_arguments",
        "browser_action",
        "semantic_extraction",
    )
)


@dataclass(frozen=True, slots=True)
class PlanningReport:
    """Result of one planning-only Experiment 05.09 run."""

    mode: str
    live_openai_request: bool
    browser_actions: bool
    result: ReasoningResult
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WebAgentObservation:
    """One read-only browser observation used by execution gates."""

    application_name: str | None
    viewport: Viewport | None
    snapshot: PerceptionSnapshot
    semantic_elements: tuple[SemanticAXElement, ...]


@dataclass(frozen=True, slots=True)
class LivePreconditionReport:
    """Read-only precondition evidence before any browser action."""

    observation: WebAgentObservation | None
    search_grounding: GroundingResult | None
    submit_grounding: GroundingResult | None
    results_grounding: GroundingResult | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalObservationReport:
    """Read-only post-execution browser evidence."""

    observation: WebAgentObservation
    results_grounding: GroundingResult


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Result of one gated live execution attempt."""

    planning: PlanningReport
    preconditions: LivePreconditionReport | None
    agent_result: AgentLoopResult | None
    final_report: FinalObservationReport | None
    execution_failures: tuple[str, ...]
    evidence_promoted: bool


class DeterministicFakeLLMClient:
    """Offline provider that returns the expected semantic plan JSON."""

    def __init__(self, response: str | None = None) -> None:
        self.response = response if response is not None else _offline_response()
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.response


class LiveWebEnvironment:
    """Production read-only observation stack for the live browser."""

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


def run_planning(
    *,
    live_openai: bool = False,
    reasoner_builder: Callable[[], object] | None = None,
) -> PlanningReport:
    """Make one planning-only reasoning call and return acceptance evidence."""

    mode = LIVE_OPENAI_MODE if live_openai else OFFLINE_MODE
    reasoner = (
        reasoner_builder()
        if reasoner_builder is not None
        else _build_reasoner(live_openai=live_openai)
    )
    result = reasoner.reason(TASK_INTENT)
    failures = acceptance_failures(result)
    return PlanningReport(
        mode=mode,
        live_openai_request=live_openai,
        browser_actions=False,
        result=result,
        failures=failures,
    )


def run_acceptance(
    *,
    live_openai: bool = False,
    reasoner_builder: Callable[[], object] | None = None,
) -> int:
    """Run planning-only acceptance and print deterministic evidence."""

    report = run_planning(
        live_openai=live_openai,
        reasoner_builder=reasoner_builder,
    )
    print_report(report)
    return 1 if report.failures else 0


def run_live_execution(
    *,
    reasoner_builder: Callable[[], object] | None = None,
    environment_builder=LiveWebEnvironment,
    executor_builder=None,
    platform_name: str = sys.platform,
    accessibility_cls=MacOSAccessibility,
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = FORMAL_EVIDENCE_PATH,
    sleeper: Callable[[float], None] = time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
) -> ExecutionReport:
    """Run live planning, gate execution, then call AgentLoop exactly once."""

    planning = run_planning(
        live_openai=True,
        reasoner_builder=reasoner_builder,
    )
    if planning.failures or planning.result.plan is None:
        return ExecutionReport(
            planning=planning,
            preconditions=None,
            agent_result=None,
            final_report=None,
            execution_failures=planning.failures,
            evidence_promoted=False,
        )

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
        return ExecutionReport(
            planning=planning,
            preconditions=preconditions,
            agent_result=None,
            final_report=None,
            execution_failures=early_failures,
            evidence_promoted=False,
        )

    environment = environment_builder(
        capture_path=capture_path,
        sleeper=sleeper,
        stabilization_wait_seconds=stabilization_wait_seconds,
    )
    preconditions = live_precondition_failures(
        environment=environment,
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
    )
    if preconditions.failures:
        return ExecutionReport(
            planning=planning,
            preconditions=preconditions,
            agent_result=None,
            final_report=None,
            execution_failures=preconditions.failures,
            evidence_promoted=False,
        )

    if executor_builder is None:
        executor_builder = _build_executor
    executor = executor_builder()
    agent_result = AgentLoop(
        perception_engine=environment.perception_engine,
        executor=executor,
        web_text_input_observer=environment.text_input_observe,
    ).run(planning.result.plan)
    final_report = _final_observation_report(environment.observe())
    execution_failures = execution_acceptance_failures(
        planning=planning,
        preconditions=preconditions,
        agent_result=agent_result,
        final_report=final_report,
        candidate_path=Path(capture_path),
    )
    evidence_promoted = False
    if not execution_failures:
        evidence_promoted = _promote_evidence(
            candidate_path=Path(capture_path),
            formal_path=Path(formal_evidence_path),
        )
        if not evidence_promoted:
            execution_failures = ("formal evidence promotion failed",)

    return ExecutionReport(
        planning=planning,
        preconditions=preconditions,
        agent_result=agent_result,
        final_report=final_report,
        execution_failures=execution_failures,
        evidence_promoted=evidence_promoted,
    )


def live_precondition_failures(
    *,
    environment: object,
    platform_name: str = sys.platform,
    accessibility_cls=MacOSAccessibility,
) -> LivePreconditionReport:
    """Check all read-only execution preconditions."""

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

    observation = environment.observe()
    grounder = UIGrounder()
    search_grounding = grounder.ground(
        _SEARCH_FIELD_TARGET,
        observation.snapshot.fused_elements,
    )
    submit_grounding = grounder.ground(
        _SUBMIT_TARGET,
        observation.snapshot.fused_elements,
    )
    results_grounding = grounder.ground(
        _RESULT_TARGET,
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
        failures.append(
            f"GO resolved role was {submit_grounding.element.element_type}"
        )

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


def acceptance_failures(result: object) -> tuple[str, ...]:
    """Return fail-closed semantic acceptance failures."""

    failures: list[str] = []
    if not isinstance(result, ReasoningResult):
        return (f"result was not ReasoningResult: {type(result).__name__}",)

    if result.status is not ReasoningStatus.READY:
        failures.append(f"ReasoningStatus was {result.status.value}")

    if not isinstance(result.plan, StructuredPlan):
        failures.append(
            "plan was not a StructuredPlan: "
            f"{type(result.plan).__name__}"
        )
        return tuple(failures)

    failures.extend(plan_acceptance_failures(result.plan))
    return tuple(failures)


def plan_acceptance_failures(plan: object) -> tuple[str, ...]:
    """Return deterministic plan-shape failures for the trusted plan."""

    if not isinstance(plan, StructuredPlan):
        return (f"plan was not a StructuredPlan: {type(plan).__name__}",)

    failures: list[str] = []
    if len(plan.steps) != EXPECTED_PLAN_STEPS:
        failures.append(
            f"plan step count was {len(plan.steps)}, not {EXPECTED_PLAN_STEPS}"
        )
        return tuple(failures)

    first, second = plan.steps
    failures.extend(_first_step_failures(first))
    failures.extend(_second_step_failures(second))
    failures.extend(_plan_payload_failures(plan))
    return tuple(failures)


def execution_acceptance_failures(
    *,
    planning: PlanningReport,
    preconditions: LivePreconditionReport | None,
    agent_result: object,
    final_report: FinalObservationReport | None,
    candidate_path: Path,
) -> tuple[str, ...]:
    """Return deterministic live execution acceptance failures."""

    failures = list(planning.failures)
    if planning.result.status is not ReasoningStatus.READY:
        failures.append(f"ReasoningStatus was {planning.result.status.value}")
    if planning.result.plan is None:
        failures.append("trusted plan was missing")
        return tuple(failures)

    failures.extend(plan_acceptance_failures(planning.result.plan))

    if preconditions is None:
        failures.append("live preconditions were missing")
    else:
        failures.extend(preconditions.failures)

    if not isinstance(agent_result, AgentLoopResult):
        failures.append(
            "AgentLoop result was not AgentLoopResult: "
            f"{type(agent_result).__name__}"
        )
        return tuple(failures)

    records = tuple(agent_result.state.steps)
    action_order = tuple(record.action.tool_name for record in records)

    if agent_result.plan is not planning.result.plan:
        failures.append("AgentLoopResult plan was not the trusted plan")
    if agent_result.status is not AgentLoopStatus.COMPLETED:
        failures.append(f"AgentLoopResult status was {agent_result.status.value}")
    if agent_result.state.status is not AgentStatus.SUCCEEDED:
        failures.append(f"AgentState status was {agent_result.state.status.value}")
    if agent_result.completed_plan_steps != EXPECTED_PLAN_STEPS:
        failures.append(
            "completed plan steps were "
            f"{agent_result.completed_plan_steps}, not {EXPECTED_PLAN_STEPS}"
        )
    if len(records) != EXPECTED_ACTION_EXECUTIONS:
        failures.append(
            "executed action count was "
            f"{len(records)}, not {EXPECTED_ACTION_EXECUTIONS}"
        )
    if action_order != EXPECTED_ACTION_ORDER:
        failures.append(
            f"action tool order was {action_order}, not {EXPECTED_ACTION_ORDER}"
        )
    if action_order.count("type_text") != 1:
        failures.append("type_text was not executed exactly once")
    if action_order.count("click_mouse") != 2:
        failures.append("click_mouse was not executed exactly twice")
    if "scroll" in action_order:
        failures.append("scroll was executed on the validated happy path")

    for index, record in enumerate(records, start=1):
        if not record.result.success:
            failures.append(
                f"record {index} ToolResult failed: {record.result.error}"
            )

    if final_report is None:
        failures.append("final observation report was missing")
    else:
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

    if not candidate_path.exists():
        failures.append("candidate evidence file was missing")

    return tuple(failures)


def print_report(
    report: PlanningReport,
    *,
    acceptance_label: str = "Experiment acceptance",
) -> None:
    """Print complete trusted-plan evidence in a stable text format."""

    print(TITLE)
    print(f"Experiment mode: {report.mode}")
    print(f"Live OpenAI request: {_yes_no(report.live_openai_request)}")
    print(f"Browser actions: {_yes_no(report.browser_actions)}")
    print(f"ReasoningStatus: {report.result.status.value}")
    print(f"Reasoning reason: {report.result.reason}")

    plan = report.result.plan
    if plan is None:
        print("Plan task_goal: <none>")
        print("Plan steps: 0")
    else:
        print(f"Plan task_goal: {plan.task_goal}")
        print(f"Plan steps: {len(plan.steps)}")
        for index, step in enumerate(plan.steps, start=1):
            _print_step(index, step)

    if report.failures:
        print(f"{acceptance_label}: failed")
        for failure in report.failures:
            print(f"  {failure}")
        return

    print(f"{acceptance_label}: passed")


def print_execution_report(report: ExecutionReport) -> None:
    """Print complete live execution evidence in a stable text format."""

    print_report(report.planning, acceptance_label="Planning acceptance")
    print(f"Execution mode: {LIVE_EXECUTION_MODE}")
    print("Execution browser actions: yes")
    if report.preconditions is not None:
        _print_preconditions(report.preconditions)
    if report.agent_result is not None:
        _print_agent_result(report.agent_result)
    if report.final_report is not None:
        print(
            "Final Results grounding status: "
            f"{report.final_report.results_grounding.status.value}"
        )
        print(f"Final app: {report.final_report.observation.application_name}")
        print(
            "Final warnings: "
            f"{report.final_report.observation.snapshot.warnings}"
        )
    print(f"Evidence promoted: {_yes_no(report.evidence_promoted)}")

    if report.execution_failures:
        print("Execution acceptance: failed")
        for failure in report.execution_failures:
            print(f"  {failure}")
        return

    print("Execution acceptance: passed")


def _build_reasoner(*, live_openai: bool) -> LLMReasoner:
    client = OpenAILLMClient() if live_openai else DeterministicFakeLLMClient()
    return LLMReasoner(client=client)


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


def _offline_response() -> str:
    return json.dumps(
        {
            "task_goal": TASK_INTENT,
            "steps": [
                {
                    "goal": "Enter the python.org search query",
                    "operation": PlanOperation.TYPE_INTO_TARGET.value,
                    "target": {
                        "text": SEARCH_FIELD_TEXT,
                        "element_types": ["text_field"],
                    },
                    "input_text": SEARCH_INPUT_TEXT,
                    "max_attempts": 1,
                },
                {
                    "goal": "Submit the python.org search query",
                    "operation": PlanOperation.CLICK_TARGET.value,
                    "action_target": {
                        "text": SUBMIT_TARGET_TEXT,
                        "element_types": ["button"],
                    },
                    "verification_target": {
                        "text": RESULT_MARKER_TEXT,
                        "element_types": [],
                    },
                    "max_attempts": 1,
                },
            ],
        }
    )


def _first_step_failures(step: object) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(step, WebTextInputStep):
        return (f"step 1 was {type(step).__name__}, not WebTextInputStep",)

    if step.operation is not PlanOperation.TYPE_INTO_TARGET:
        failures.append(f"step 1 operation was {step.operation}")
    if not step.goal.strip():
        failures.append("step 1 goal was empty")
    if step.target.text != SEARCH_FIELD_TEXT:
        failures.append(f"step 1 target text was {step.target.text}")
    if step.target.element_types not in _SEARCH_FIELD_ELEMENT_TYPES:
        failures.append(
            "step 1 target element_types were "
            f"{step.target.element_types}"
        )
    if step.input_text != SEARCH_INPUT_TEXT:
        failures.append(f"step 1 input_text was {step.input_text}")
    if step.max_attempts != 1:
        failures.append(f"step 1 max_attempts was {step.max_attempts}")

    failures.extend(_target_payload_failures("step 1 target", step.target))
    return tuple(failures)


def _second_step_failures(step: object) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(step, PlanStep):
        return (f"step 2 was {type(step).__name__}, not PlanStep",)

    if step.operation is not PlanOperation.CLICK_TARGET:
        failures.append(f"step 2 operation was {step.operation}")
    if not step.goal.strip():
        failures.append("step 2 goal was empty")
    if step.action_target.text != SUBMIT_TARGET_TEXT:
        failures.append(
            f"step 2 action target text was {step.action_target.text}"
        )
    if step.action_target.element_types not in _SUBMIT_ELEMENT_TYPES:
        failures.append(
            "step 2 action target element_types were "
            f"{step.action_target.element_types}"
        )
    if step.verification_target.text != RESULT_MARKER_TEXT:
        failures.append(
            "step 2 verification target text was "
            f"{step.verification_target.text}"
        )
    if step.verification_target.element_types not in _RESULT_ELEMENT_TYPES:
        failures.append(
            "step 2 verification target element_types were "
            f"{step.verification_target.element_types}"
        )

    failures.extend(
        _target_payload_failures("step 2 action target", step.action_target)
    )
    failures.extend(
        _target_payload_failures(
            "step 2 verification target",
            step.verification_target,
        )
    )
    return tuple(failures)


def _plan_payload_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []
    for item_name, item in _trusted_plan_items(plan):
        if isinstance(item, Action):
            failures.append(f"{item_name} contained executable Action")
        if is_dataclass(item):
            field_names = {field.name for field in fields(item)}
            forbidden = field_names & _FORBIDDEN_STEP_FIELD_NAMES
            if forbidden:
                failures.append(
                    f"{item_name} exposed forbidden fields {tuple(forbidden)}"
                )
    return tuple(failures)


def _trusted_plan_items(plan: StructuredPlan) -> tuple[tuple[str, object], ...]:
    items: list[tuple[str, object]] = [("plan", plan)]
    for index, step in enumerate(plan.steps, start=1):
        items.append((f"step {index}", step))
        if isinstance(step, WebTextInputStep):
            items.append((f"step {index} target", step.target))
        if isinstance(step, PlanStep):
            items.append((f"step {index} action target", step.action_target))
            items.append(
                (f"step {index} verification target", step.verification_target)
            )
    return tuple(items)


def _target_payload_failures(
    label: str,
    target: TargetSpec,
) -> tuple[str, ...]:
    failures: list[str] = []
    if target.identifier is not None:
        failures.append(f"{label} identifier was {target.identifier}")
    if target.reference_point is not None:
        failures.append(f"{label} reference_point was {target.reference_point}")
    if target.minimum_confidence != 0.70:
        failures.append(
            f"{label} minimum_confidence was {target.minimum_confidence}"
        )
    return tuple(failures)


def _print_step(index: int, step: object) -> None:
    print(f"Step {index} class: {type(step).__name__}")
    print(f"Step {index} goal: {getattr(step, 'goal', '<missing>')}")
    operation = getattr(step, "operation", "<missing>")
    if isinstance(operation, PlanOperation):
        operation = operation.value
    print(f"Step {index} operation: {operation}")
    print(f"Step {index} max_attempts: {getattr(step, 'max_attempts', '<missing>')}")

    if isinstance(step, WebTextInputStep):
        _print_target(f"Step {index} target", step.target)
        print(f"Step {index} input_text: {step.input_text}")
    elif isinstance(step, PlanStep):
        _print_target(f"Step {index} action target", step.action_target)
        _print_target(
            f"Step {index} verification target",
            step.verification_target,
        )


def _print_target(label: str, target: TargetSpec) -> None:
    print(f"{label} text: {target.text}")
    print(f"{label} element_types: {target.element_types}")
    print(f"{label} identifier: {target.identifier}")
    print(f"{label} reference_point: {target.reference_point}")


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


def _print_agent_result(result: AgentLoopResult) -> None:
    print(f"AgentLoopResult reason: {result.reason}")
    print(f"Agent loop status: {result.status.value}")
    print(f"Agent state: {result.state.status.value}")
    print(f"Completed plan steps: {result.completed_plan_steps}")
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


def _final_observation_report(
    observation: WebAgentObservation,
) -> FinalObservationReport:
    grounding = UIGrounder().ground(
        _RESULT_TARGET,
        observation.snapshot.fused_elements,
    )
    return FinalObservationReport(
        observation=observation,
        results_grounding=grounding,
    )


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


def _value_is_non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _countdown(seconds: int, *, sleeper: Callable[[float], None]) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"{remaining}...")
        sleeper(1)


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
        raise ValueError("stabilization_wait_seconds must be non-negative")
    return seconds


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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 05.09 LLM reasoning acceptance. "
            "Default mode is offline and uses a deterministic fake provider."
        )
    )
    parser.add_argument(
        "--live-openai",
        action="store_true",
        help="Make exactly one OpenAI planning request and perform no actions.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "With --live-openai, run the trusted plan through gated live "
            "AgentLoop execution."
        ),
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
    reasoner_builder: Callable[[], object] | None = None,
    live_execution_runner=run_live_execution,
) -> int:
    args = _parse_args(argv)
    if args.execute and not args.live_openai:
        print(TITLE)
        print("Experiment mode: rejected")
        print("Live OpenAI request: no")
        print("Browser actions: no")
        print("Experiment acceptance: failed")
        print("  --execute requires --live-openai")
        return 2

    if args.execute:
        report = live_execution_runner(
            reasoner_builder=reasoner_builder,
            wait_seconds=args.wait_seconds,
        )
        print_execution_report(report)
        return 1 if report.execution_failures else 0

    return run_acceptance(
        live_openai=args.live_openai,
        reasoner_builder=reasoner_builder,
    )


if __name__ == "__main__":
    raise SystemExit(main())
