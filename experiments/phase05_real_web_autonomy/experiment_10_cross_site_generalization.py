"""Phase 05 Experiment 10: read-only cross-site semantic qualification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Iterable, Sequence

from computer_agent.core.models import Action
from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
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
from computer_agent.reasoning import (
    LLMReasoner,
    ReasoningResult,
    ReasoningStatus,
)
from computer_agent.verification import (
    PresenceExpectation,
    StateObserver,
    StateTransitionVerificationResult,
    StateTransitionVerifier,
    UIStateCondition,
    VerificationSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase05_real_web_autonomy"
CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_10_cross_site_generalization_candidate.png"
)
FORMAL_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_10_cross_site_generalization.png"
)
TARGET_URL = "https://www.wikipedia.org/"
TARGET_SITE = "Wikipedia"
KHAN_TARGET_URL = "https://www.khanacademy.org/"
KHAN_TARGET_SITE = "Khan Academy"
EXPECTED_APPLICATION_NAME = "Google Chrome"
TASK_INTENT = (
    'On Wikipedia, type "computer vision" into the "Search Wikipedia" search '
    'field, click the Search button, and verify that the "Computer vision" '
    "article heading appears."
)
SEARCH_FIELD_TEXT = "Search Wikipedia"
SEARCH_INPUT_TEXT = "computer vision"
SUBMIT_TARGET_TEXT = "Search"
ARTICLE_HEADING_TEXT = "Computer vision"
EXPECTED_PLAN_STEPS = 2
EXPECTED_ACTION_ORDER = ("click_mouse", "type_text", "click_mouse")
EXPECTED_ACTION_EXECUTIONS = 3
DEFAULT_WAIT_SECONDS = 8
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.5
OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6

RELEVANT_TERMS = (
    "search",
    "wikipedia",
)
POSTCONDITION_TERMS = (
    "computer vision",
    "computer",
    "vision",
    "results",
)
KHAN_HIGHLIGHT_TERMS = (
    "search",
    "course",
    "learn",
    "practice",
    "math",
    "computer",
    "continue",
)
KHAN_LOGGED_OUT_TERMS = (
    "log in",
    "sign up",
    "start learning",
    "join",
)
KHAN_LOGGED_IN_TERMS = (
    "dashboard",
    "learner home",
    "my courses",
    "continue",
    "profile",
)
KHAN_RELEVANT_ELEMENT_TYPES = (
    "heading",
    "text_field",
    "text_area",
    "button",
    "link",
    "checkbox",
    "radio",
    "radio_button",
    "combobox",
    "popup_button",
    "tab",
    "menu",
    "menu_item",
    "menuitem",
)
KHAN_GROUNDING_PROBE_LIMIT = 8
_ROLE_TO_ELEMENT_TYPE = {
    "AXTextField": "text_field",
    "AXTextArea": "text_area",
    "AXButton": "button",
    "AXCheckBox": "checkbox",
    "AXPopUpButton": "popup_button",
    "AXRadioButton": "radio_button",
    "AXLink": "link",
    "AXHeading": "heading",
    "AXStaticText": "text",
}
_SEARCH_FIELD_ELEMENT_TYPES = ("text_field",)
_SUBMIT_ELEMENT_TYPES = ("button",)
_LANDING_IDENTITY_ELEMENT_TYPES = ("heading",)
_DESTINATION_HEADING_ELEMENT_TYPES = ("heading",)
_ARTICLE_HEADING_ELEMENT_TYPES = ("heading",)
_LANDING_IDENTITY_TARGET = TargetSpec(
    text="Wikipedia The Free Encyclopedia",
    element_types=_LANDING_IDENTITY_ELEMENT_TYPES,
)
_SEARCH_FIELD_TARGET = TargetSpec(
    text=SEARCH_FIELD_TEXT,
    element_types=_SEARCH_FIELD_ELEMENT_TYPES,
)
_SUBMIT_TARGET = TargetSpec(
    text=SUBMIT_TARGET_TEXT,
    element_types=_SUBMIT_ELEMENT_TYPES,
)
_DESTINATION_HEADING_TARGET = TargetSpec(
    text=ARTICLE_HEADING_TEXT,
    element_types=_DESTINATION_HEADING_ELEMENT_TYPES,
)
_FINAL_DESTINATION_TARGET = TargetSpec(
    text=ARTICLE_HEADING_TEXT,
    element_types=_ARTICLE_HEADING_ELEMENT_TYPES,
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
class CandidateRecord:
    """Printable read-only semantic candidate evidence."""

    source: str
    raw_role: str | None
    production_element_type: str
    text: str | None
    value: str | int | float | bool | None
    bounds: BoundingBox | None
    enabled: bool | None
    focused: bool | None


@dataclass(frozen=True, slots=True)
class KhanElementSummary:
    """Read-only summary of one relevant Khan semantic element."""

    text: str | None
    identifier: str | None
    element_type: str
    source: str | None
    confidence: float
    enabled: bool | None
    bounds: BoundingBox


@dataclass(frozen=True, slots=True)
class KhanGroundingDiagnostic:
    """Read-only UIGrounder diagnostic for one observed Khan candidate."""

    target: TargetSpec
    result: GroundingResult


@dataclass(frozen=True, slots=True)
class WikipediaObservation:
    """One read-only live observation of the frontmost browser state."""

    frontmost_app: str | None
    viewport: Viewport | None
    snapshot: PerceptionSnapshot
    semantic_elements: tuple[SemanticAXElement, ...]


@dataclass(frozen=True, slots=True)
class KhanQualificationReport:
    """One read-only Khan Academy qualification observation."""

    observation: WikipediaObservation
    snapshot: PerceptionSnapshot
    viewport: Viewport | None
    warnings: tuple[str, ...]
    relevant_elements: tuple[KhanElementSummary, ...]
    highlighted_elements: tuple[KhanElementSummary, ...]
    page_evidence: tuple[str, ...]
    grounding_diagnostics: tuple[KhanGroundingDiagnostic, ...]
    browser_actions: bool
    openai_request: bool
    agent_loop_execution: bool
    evidence_promoted: bool


@dataclass(frozen=True, slots=True)
class OfflinePlanningReport:
    """Deterministic offline plan construction and acceptance evidence."""

    live_openai_request: bool
    browser_actions: bool
    plan: StructuredPlan
    execution_plan: StructuredPlan | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveOpenAIPlanningReport:
    """Real OpenAI planning-only result and acceptance evidence."""

    live_openai_request: bool
    browser_actions: bool
    result: ReasoningResult
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LivePreconditionReport:
    """Read-only execution gate evidence before constructing the executor."""

    observation: WikipediaObservation | None
    landing_identity_grounding: GroundingResult | None
    search_grounding: GroundingResult | None
    submit_grounding: GroundingResult | None
    destination_grounding: GroundingResult | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalObservationReport:
    """Read-only post-execution semantic verification evidence."""

    observation: WikipediaObservation
    destination_grounding: GroundingResult


@dataclass(frozen=True, slots=True)
class GenericVerificationRecord:
    """One generic state-transition verification call made during execution."""

    verification_spec: VerificationSpec
    before_snapshot_captured_at: object | None
    before_snapshot_identity: str | None
    after_snapshot_captured_at: object | None
    after_snapshot_identity: str | None
    result: StateTransitionVerificationResult


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Result of one gated live OpenAI Wikipedia execution attempt."""

    planning: LiveOpenAIPlanningReport
    preconditions: LivePreconditionReport | None
    agent_result: object | None
    final_report: FinalObservationReport | None
    generic_verification_records: tuple[GenericVerificationRecord, ...]
    execution_failures: tuple[str, ...]
    evidence_promoted: bool


class RecordingStateTransitionVerifier:
    """Experiment-local recorder for production generic verification calls."""

    def __init__(
        self,
        verifier: StateTransitionVerifier | None = None,
    ) -> None:
        self._verifier = verifier if verifier is not None else (
            StateTransitionVerifier()
        )
        self._records: list[GenericVerificationRecord] = []

    @property
    def records(self) -> tuple[GenericVerificationRecord, ...]:
        """Return recorded generic verification calls in execution order."""

        return tuple(self._records)

    @property
    def verifier(self) -> StateTransitionVerifier:
        """Return the wrapped production verifier."""

        return self._verifier

    def verify(
        self,
        *,
        before_snapshot: PerceptionSnapshot,
        after_snapshot: PerceptionSnapshot,
        verification_spec: VerificationSpec,
    ) -> StateTransitionVerificationResult:
        """Delegate to production verification and retain the exact result."""

        result = self._verifier.verify(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            verification_spec=verification_spec,
        )
        self._records.append(
            GenericVerificationRecord(
                verification_spec=verification_spec,
                before_snapshot_captured_at=_snapshot_captured_at(
                    before_snapshot
                ),
                before_snapshot_identity=_snapshot_identity(
                    before_snapshot
                ),
                after_snapshot_captured_at=_snapshot_captured_at(
                    after_snapshot
                ),
                after_snapshot_identity=_snapshot_identity(after_snapshot),
                result=result,
            )
        )
        return result


class LiveWikipediaEnvironment:
    """Production browser observation stack for live Wikipedia execution."""

    def __init__(
        self,
        *,
        capture_path: str | Path,
        accessibility=None,
        perception_engine=None,
        sleeper=time.sleep,
        stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
    ) -> None:
        if not callable(sleeper):
            raise ValueError("sleeper must be callable")

        self.capture_path = Path(capture_path)
        self.accessibility = (
            accessibility
            if accessibility is not None
            else _build_macos_accessibility()
        )
        self.perception_engine = (
            perception_engine
            if perception_engine is not None
            else _build_perception_engine(
                capture_path=self.capture_path,
                accessibility=self.accessibility,
            )
        )
        self.sleeper = sleeper
        self.stabilization_wait_seconds = stabilization_wait_seconds

    def observe(self) -> WikipediaObservation:
        self.sleeper(self.stabilization_wait_seconds)
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.perception_engine.observe()
        return WikipediaObservation(
            frontmost_app=self.accessibility.read_frontmost_application_name(),
            viewport=self.accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(
                self.accessibility.read_frontmost_semantic_elements()
            ),
        )

    def text_input_observe(self):
        from computer_agent.agent.text_input import TextInputObservation

        observation = self.observe()
        return TextInputObservation(
            application_name=observation.frontmost_app,
            viewport=observation.viewport,
            snapshot=observation.snapshot,
            semantic_elements=observation.semantic_elements,
        )


def wikipedia_offline_plan() -> StructuredPlan:
    """Construct the exact trusted plan for the bounded Wikipedia task."""

    return StructuredPlan(
        task_goal=TASK_INTENT,
        steps=(
            WebTextInputStep(
                goal="Enter the Wikipedia search query",
                target=TargetSpec(
                    text=SEARCH_FIELD_TEXT,
                    element_types=_SEARCH_FIELD_ELEMENT_TYPES,
                ),
                input_text=SEARCH_INPUT_TEXT,
                max_attempts=1,
            ),
            PlanStep(
                goal="Submit the Wikipedia search query",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=SUBMIT_TARGET_TEXT,
                    element_types=_SUBMIT_ELEMENT_TYPES,
                ),
                verification_target=TargetSpec(
                    text=ARTICLE_HEADING_TEXT,
                    element_types=_ARTICLE_HEADING_ELEMENT_TYPES,
                ),
                max_attempts=1,
            ),
        ),
    )


def run_offline_planning(
    *,
    plan_builder=wikipedia_offline_plan,
) -> OfflinePlanningReport:
    """Build and accept the offline Wikipedia plan without side effects."""

    plan = plan_builder()
    failures = list(plan_acceptance_failures(plan))
    execution_plan = None
    if not failures:
        try:
            execution_plan = _build_generic_execution_plan(plan)
        except ValueError as error:
            failures.append(str(error))
        else:
            failures.extend(
                execution_plan_acceptance_failures(execution_plan)
            )

    return OfflinePlanningReport(
        live_openai_request=False,
        browser_actions=False,
        plan=plan,
        execution_plan=execution_plan,
        failures=tuple(failures),
    )


def run_offline_acceptance(
    *,
    plan_builder=wikipedia_offline_plan,
) -> int:
    """Print the deterministic offline planning report."""

    report = run_offline_planning(plan_builder=plan_builder)
    print_offline_planning_report(report)
    return 1 if report.failures else 0


def run_live_openai_planning(
    *,
    reasoner_builder=None,
) -> LiveOpenAIPlanningReport:
    """Make exactly one OpenAI planning request and accept the plan only."""

    reasoner = (
        reasoner_builder()
        if reasoner_builder is not None
        else _build_live_openai_reasoner()
    )
    result = reasoner.reason(TASK_INTENT)
    failures = reasoning_acceptance_failures(result)
    return LiveOpenAIPlanningReport(
        live_openai_request=True,
        browser_actions=False,
        result=result,
        failures=failures,
    )


def run_live_openai_acceptance(
    *,
    reasoner_builder=None,
) -> int:
    """Print the live OpenAI planning-only report."""

    report = run_live_openai_planning(reasoner_builder=reasoner_builder)
    print_live_openai_planning_report(report)
    return 1 if report.failures else 0


def run_live_execution(
    *,
    reasoner_builder=None,
    environment_builder=LiveWikipediaEnvironment,
    executor_builder=None,
    agent_loop_cls=None,
    platform_name: str = sys.platform,
    accessibility_cls=None,
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = FORMAL_EVIDENCE_PATH,
    sleeper=time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
) -> ExecutionReport:
    """Run live planning, gate execution, then call AgentLoop exactly once."""

    planning = run_live_openai_planning(reasoner_builder=reasoner_builder)
    if planning.failures or planning.result.plan is None:
        return ExecutionReport(
            planning=planning,
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=planning.failures,
            evidence_promoted=False,
        )

    try:
        execution_plan = _build_generic_execution_plan(planning.result.plan)
    except ValueError as error:
        return ExecutionReport(
            planning=planning,
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=(str(error),),
            evidence_promoted=False,
        )

    _countdown(wait_seconds, sleeper=sleeper)

    if accessibility_cls is None:
        accessibility_cls = _build_macos_accessibility_cls()

    early_failures = _platform_accessibility_failures(
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
    )
    if early_failures:
        preconditions = LivePreconditionReport(
            observation=None,
            landing_identity_grounding=None,
            search_grounding=None,
            submit_grounding=None,
            destination_grounding=None,
            failures=early_failures,
        )
        return ExecutionReport(
            planning=planning,
            preconditions=preconditions,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
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
            generic_verification_records=(),
            execution_failures=preconditions.failures,
            evidence_promoted=False,
        )

    if executor_builder is None:
        executor_builder = _build_executor
    if agent_loop_cls is None:
        agent_loop_cls = _build_agent_loop_cls()

    executor = executor_builder()
    grounder = UIGrounder()
    state_observer = StateObserver()
    state_transition_verifier = RecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=state_observer)
    )
    agent_result = agent_loop_cls(
        perception_engine=environment.perception_engine,
        grounder=grounder,
        executor=executor,
        state_transition_verifier=state_transition_verifier,
        web_text_input_observer=environment.text_input_observe,
    ).run(execution_plan)
    final_report = _final_observation_report(environment.observe())
    generic_verification_records = state_transition_verifier.records
    execution_failures = execution_acceptance_failures(
        planning=planning,
        execution_plan=execution_plan,
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
        generic_verification_records=generic_verification_records,
        execution_failures=execution_failures,
        evidence_promoted=evidence_promoted,
    )


def live_precondition_failures(
    *,
    environment: object,
    platform_name: str = sys.platform,
    accessibility_cls=None,
) -> LivePreconditionReport:
    """Check all read-only live execution preconditions."""

    if accessibility_cls is None:
        accessibility_cls = _build_macos_accessibility_cls()

    failures = list(
        _platform_accessibility_failures(
            platform_name=platform_name,
            accessibility_cls=accessibility_cls,
        )
    )
    if failures:
        return LivePreconditionReport(
            observation=None,
            landing_identity_grounding=None,
            search_grounding=None,
            submit_grounding=None,
            destination_grounding=None,
            failures=tuple(failures),
        )

    observation = environment.observe()
    grounder = UIGrounder()
    landing_identity_grounding = grounder.ground(
        _LANDING_IDENTITY_TARGET,
        observation.snapshot.fused_elements,
    )
    search_grounding = grounder.ground(
        _SEARCH_FIELD_TARGET,
        observation.snapshot.fused_elements,
    )
    submit_grounding = grounder.ground(
        _SUBMIT_TARGET,
        observation.snapshot.fused_elements,
    )
    destination_grounding = grounder.ground(
        _DESTINATION_HEADING_TARGET,
        observation.snapshot.fused_elements,
    )

    if observation.frontmost_app != EXPECTED_APPLICATION_NAME:
        failures.append(
            "frontmost app was "
            f"{observation.frontmost_app}, not {EXPECTED_APPLICATION_NAME}"
        )

    if observation.viewport is None:
        failures.append("viewport was unavailable")

    if observation.snapshot.warnings:
        failures.append(
            f"perception warnings were {observation.snapshot.warnings}"
        )

    if landing_identity_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "Wikipedia landing identity grounding was "
            f"{landing_identity_grounding.status.value}"
        )

    if search_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "Search Wikipedia grounding was "
            f"{search_grounding.status.value}"
        )
    elif normalize_ui_text(search_grounding.element.element_type) != "text field":
        failures.append(
            "Search Wikipedia resolved role was "
            f"{search_grounding.element.element_type}"
        )
    elif _value_is_non_empty(search_grounding.element.value):
        failures.append("Search Wikipedia field was not empty")

    if submit_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            f"Search button grounding was {submit_grounding.status.value}"
        )
    elif normalize_ui_text(submit_grounding.element.element_type) != "button":
        failures.append(
            f"Search button resolved role was {submit_grounding.element.element_type}"
        )

    if destination_grounding.status is not GroundingStatus.NOT_FOUND:
        failures.append(
            "Computer vision destination grounding was "
            f"{destination_grounding.status.value}, not not_found"
        )

    return LivePreconditionReport(
        observation=observation,
        landing_identity_grounding=landing_identity_grounding,
        search_grounding=search_grounding,
        submit_grounding=submit_grounding,
        destination_grounding=destination_grounding,
        failures=tuple(failures),
    )


def execution_acceptance_failures(
    *,
    planning: LiveOpenAIPlanningReport,
    execution_plan: StructuredPlan,
    preconditions: LivePreconditionReport | None,
    agent_result: object,
    final_report: FinalObservationReport | None,
    candidate_path: Path,
) -> tuple[str, ...]:
    """Return fail-closed live execution acceptance failures."""

    from computer_agent.agent import AgentLoopResult, AgentLoopStatus, AgentStatus

    failures = list(planning.failures)
    if planning.result.status is not ReasoningStatus.READY:
        failures.append(f"ReasoningStatus was {planning.result.status.value}")
    if planning.result.plan is None:
        failures.append("trusted plan was missing")
        return tuple(failures)

    failures.extend(plan_acceptance_failures(planning.result.plan))
    failures.extend(execution_plan_acceptance_failures(execution_plan))

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

    if agent_result.plan is not execution_plan:
        failures.append(
            "AgentLoopResult plan was not the trusted execution plan"
        )
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
        if final_report.destination_grounding.status is not GroundingStatus.RESOLVED:
            failures.append(
                "final Computer vision grounding was "
                f"{final_report.destination_grounding.status.value}"
            )
        if final_report.observation.frontmost_app != EXPECTED_APPLICATION_NAME:
            failures.append(
                "final frontmost app was "
                f"{final_report.observation.frontmost_app}, "
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


def reasoning_acceptance_failures(result: object) -> tuple[str, ...]:
    """Return fail-closed reasoning and plan acceptance failures."""

    if not isinstance(result, ReasoningResult):
        return (f"result was not ReasoningResult: {type(result).__name__}",)

    failures: list[str] = []
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
    """Return fail-closed acceptance failures for the Wikipedia plan."""

    if not isinstance(plan, StructuredPlan):
        return (f"plan was not a StructuredPlan: {type(plan).__name__}",)

    failures: list[str] = []
    if plan.task_goal != TASK_INTENT:
        failures.append(f"plan task_goal was {plan.task_goal!r}")

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


def execution_plan_acceptance_failures(plan: object) -> tuple[str, ...]:
    """Return fail-closed acceptance failures for the generic execution plan."""

    if not isinstance(plan, StructuredPlan):
        return (f"execution plan was {type(plan).__name__}, not StructuredPlan",)

    failures: list[str] = []
    if plan.task_goal != TASK_INTENT:
        failures.append(f"execution plan task_goal was {plan.task_goal!r}")

    if len(plan.steps) != EXPECTED_PLAN_STEPS:
        failures.append(
            "execution plan step count was "
            f"{len(plan.steps)}, not {EXPECTED_PLAN_STEPS}"
        )
        return tuple(failures)

    first, second = plan.steps
    failures.extend(_first_step_failures(first))
    failures.extend(_generic_second_step_failures(second))
    failures.extend(_plan_payload_failures(plan))
    return tuple(failures)


def print_offline_planning_report(report: OfflinePlanningReport) -> None:
    """Print stable deterministic planning evidence."""

    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: deterministic offline Wikipedia plan")
    print(f"Live OpenAI request: {_yes_no(report.live_openai_request)}")
    print(f"Browser actions: {_yes_no(report.browser_actions)}")
    print(f"Plan steps: {len(report.plan.steps)}")
    for index, step in enumerate(report.plan.steps, start=1):
        _print_step(index, step)

    if report.failures:
        print("Planning acceptance: failed")
        for failure in report.failures:
            print(f"  {failure}")
        return

    print("Planning acceptance: passed")
    print("Execution plan conversion: passed")


def print_live_openai_planning_report(report: LiveOpenAIPlanningReport) -> None:
    """Print stable live OpenAI planning-only evidence."""

    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: live OpenAI Wikipedia planning")
    print(f"Live OpenAI request: {_yes_no(report.live_openai_request)}")
    print(f"Browser actions: {_yes_no(report.browser_actions)}")
    print(f"ReasoningStatus: {report.result.status.value}")
    print(f"Reasoning reason: {report.result.reason}")

    plan = report.result.plan
    if plan is None:
        print("task_goal: <none>")
        print("Plan steps: 0")
    else:
        print(f"task_goal: {plan.task_goal}")
        print(f"Plan steps: {len(plan.steps)}")
        for index, step in enumerate(plan.steps, start=1):
            _print_step(index, step)

    if report.failures:
        print("Planning acceptance: failed")
        for failure in report.failures:
            print(f"  {failure}")
        return

    print("Planning acceptance: passed")


def print_execution_report(report: ExecutionReport) -> None:
    """Print stable live execution evidence."""

    print_live_openai_planning_report(report.planning)
    print("Execution mode: live OpenAI Wikipedia execution")
    print("Execution browser actions: yes")
    if report.preconditions is not None:
        _print_preconditions(report.preconditions)
    if report.agent_result is not None:
        _print_agent_result(report.agent_result)
    _print_generic_verification_records(report.generic_verification_records)
    if report.final_report is not None:
        print(
            "Final Computer vision grounding status: "
            f"{report.final_report.destination_grounding.status.value}"
        )
        print(f"Final app: {report.final_report.observation.frontmost_app}")
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


def _print_generic_verification_records(
    records: Sequence[GenericVerificationRecord],
) -> None:
    print(f"Generic verification calls: {len(records)}")
    for call_index, record in enumerate(records, start=1):
        result = record.result
        print(f"Generic verification call {call_index}:")
        print(f"  overall status: {result.status.value}")
        print(f"  overall reason: {result.reason}")
        print(
            "  before snapshot timestamp: "
            f"{record.before_snapshot_captured_at}"
        )
        print(
            "  before snapshot identity: "
            f"{record.before_snapshot_identity}"
        )
        print(
            "  after snapshot timestamp: "
            f"{record.after_snapshot_captured_at}"
        )
        print(
            "  after snapshot identity: "
            f"{record.after_snapshot_identity}"
        )
        _print_condition_evaluations(
            "before",
            result.before_evaluations,
        )
        _print_condition_evaluations(
            "after",
            result.after_evaluations,
        )


def _print_condition_evaluations(
    phase: str,
    evaluations: Sequence[object],
) -> None:
    for index, evaluation in enumerate(evaluations, start=1):
        condition = evaluation.condition
        target = condition.target
        print(f"  {phase} evaluation {index}:")
        print(f"    target text: {target.text}")
        print(f"    target element_types: {target.element_types}")
        print(f"    expectation: {condition.expectation.value}")
        print(f"    condition status: {evaluation.status.value}")
        print(f"    condition reason: {evaluation.reason}")
        if evaluation.grounding is not None:
            _print_grounding_evidence(evaluation.grounding)
        elif evaluation.observation is not None:
            _print_observation_evidence(evaluation.observation)
        else:
            print("    evidence backend: <missing>")


def _print_grounding_evidence(grounding: object) -> None:
    print("    evidence backend: grounding")
    print(f"    grounding status: {grounding.status.value}")
    print(f"    grounding reason: {grounding.reason}")
    print(f"    candidate count: {len(grounding.candidates)}")
    for candidate_index, candidate in enumerate(
        grounding.candidates,
        start=1,
    ):
        element = candidate.element
        print(f"    candidate {candidate_index}:")
        print(f"      match_basis: {candidate.match_basis}")
        print(f"      eligible: {candidate.eligible}")
        print(f"      rejection_reasons: {candidate.rejection_reasons}")
        print(f"      distance: {candidate.distance}")
        print(f"      element text: {getattr(element, 'text', None)}")
        print(
            "      element identifier: "
            f"{getattr(element, 'identifier', None)}"
        )
        print(
            "      element element_type: "
            f"{getattr(element, 'element_type', None)}"
        )
        print(f"      element source: {getattr(element, 'source', None)}")
        print(
            "      element confidence: "
            f"{getattr(element, 'confidence', None)}"
        )
        print(
            "      element enabled: "
            f"{getattr(element, 'enabled', None)}"
        )
        print(
            "      element bounds: "
            f"{_format_bounds(getattr(element, 'bounding_box', None))}"
        )


def _print_observation_evidence(observation: object) -> None:
    print("    evidence backend: state_observation")
    print(f"    observation status: {observation.status.value}")
    print(f"    observation reason: {observation.reason}")
    print(f"    candidate count: {len(observation.candidates)}")
    for candidate_index, candidate in enumerate(
        observation.candidates,
        start=1,
    ):
        element = candidate.element
        print(f"    candidate {candidate_index}:")
        print(f"      match_basis: {candidate.match_basis}")
        print(f"      predicate_match: {candidate.predicate_match}")
        print(f"      mismatch_reasons: {candidate.mismatch_reasons}")
        print(f"      uncertainty_reasons: {candidate.uncertainty_reasons}")
        print(f"      distance: {candidate.distance}")
        print(f"      element text: {getattr(element, 'text', None)}")
        print(
            "      element identifier: "
            f"{getattr(element, 'identifier', None)}"
        )
        print(
            "      element element_type: "
            f"{getattr(element, 'element_type', None)}"
        )
        print(f"      element source: {getattr(element, 'source', None)}")
        print(
            "      element confidence: "
            f"{getattr(element, 'confidence', None)}"
        )
        print(
            "      element enabled: "
            f"{getattr(element, 'enabled', None)}"
        )
        print(
            "      element bounds: "
            f"{_format_bounds(getattr(element, 'bounding_box', None))}"
        )


def candidate_records(
    *,
    snapshot: PerceptionSnapshot,
    semantic_elements: Iterable[SemanticAXElement],
) -> tuple[CandidateRecord, ...]:
    """Return raw Accessibility and production perception candidates."""

    records: list[CandidateRecord] = []
    for element in semantic_elements:
        records.append(
            CandidateRecord(
                source="accessibility_raw",
                raw_role=element.role,
                production_element_type=_ROLE_TO_ELEMENT_TYPE.get(
                    element.role,
                    "unknown",
                ),
                text=element.text,
                value=element.value,
                bounds=element.bounds,
                enabled=None,
                focused=None,
            )
        )

    for element in snapshot.fused_elements:
        records.append(
            CandidateRecord(
                source=element.source or "production",
                raw_role=None,
                production_element_type=element.element_type,
                text=element.text,
                value=element.value,
                bounds=element.bounding_box,
                enabled=element.enabled,
                focused=element.focused,
            )
        )

    return tuple(records)


def matching_candidates(
    candidates: Iterable[CandidateRecord],
    terms: Sequence[str],
) -> tuple[CandidateRecord, ...]:
    """Return candidates whose semantic text or value contains any term."""

    normalized_terms = tuple(
        normalize_ui_text(term)
        for term in terms
        if isinstance(term, str) and term.strip()
    )
    if not normalized_terms:
        return ()

    matches = []
    for candidate in candidates:
        haystack = normalize_ui_text(_candidate_search_text(candidate))
        if any(term in haystack for term in normalized_terms):
            matches.append(candidate)

    return tuple(matches)


def print_candidate_section(
    title: str,
    candidates: Sequence[CandidateRecord],
) -> None:
    """Print one deterministic candidate section."""

    print()
    print(f"{title}: {len(candidates)}")
    if not candidates:
        print("  none")
        return

    for index, candidate in enumerate(candidates, start=1):
        print(f"Candidate {index}:")
        print(f"  source: {candidate.source}")
        print(f"  raw role: {candidate.raw_role}")
        print(
            "  production element_type: "
            f"{candidate.production_element_type}"
        )
        print(f"  text: {candidate.text!r}")
        print(f"  value: {candidate.value!r}")
        print(f"  bounds: {_format_bounds(candidate.bounds)}")
        print(f"  enabled: {candidate.enabled!r}")
        print(f"  focused: {candidate.focused!r}")


def observe_wikipedia() -> WikipediaObservation:
    """Observe the frontmost browser state without taking any action."""

    accessibility, engine = _build_live_dependencies()
    return WikipediaObservation(
        frontmost_app=accessibility.read_frontmost_application_name(),
        viewport=accessibility.read_frontmost_viewport(),
        snapshot=engine.observe(),
        semantic_elements=tuple(
            accessibility.read_frontmost_semantic_elements()
        ),
    )


def run_read_only_audit(
    *,
    observer=observe_wikipedia,
    sleeper=time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
) -> int:
    """Run one read-only semantic qualification observation."""

    if not callable(sleeper):
        raise ValueError("sleeper must be callable")

    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: read-only site qualification")
    print(f"Target site: {TARGET_SITE}")
    print(f"Target URL: {TARGET_URL}")
    print("Live OpenAI request: no")
    print("Browser actions: no")
    print(
        "Open the target URL manually in Google Chrome, or manually "
        "prepare the Wikipedia result page for a computer vision search."
    )

    if sys.platform != "darwin":
        print("Read-only audit failed: macOS is required.")
        return 1

    _countdown(wait_seconds, sleeper=sleeper)
    observation = observer()
    print(f"Frontmost app: {observation.frontmost_app}")
    print(f"Viewport: {observation.viewport}")
    if observation.snapshot.warnings:
        print(f"Perception warnings: {observation.snapshot.warnings}")
    else:
        print("Perception warnings: ()")

    candidates = candidate_records(
        snapshot=observation.snapshot,
        semantic_elements=observation.semantic_elements,
    )
    print(f"Raw semantic elements: {len(observation.semantic_elements)}")
    print(f"Production fused elements: {len(observation.snapshot.fused_elements)}")
    print(f"Candidate records: {len(candidates)}")

    print_candidate_section(
        "Likely search/Wikipedia candidates",
        matching_candidates(candidates, RELEVANT_TERMS),
    )
    print_candidate_section(
        "Manual computer vision result-page candidates",
        matching_candidates(candidates, POSTCONDITION_TERMS),
    )

    return 0


def run_khan_qualification(
    *,
    observer=observe_wikipedia,
    sleeper=time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
) -> KhanQualificationReport:
    """Take exactly one read-only Khan Academy qualification observation."""

    if not callable(observer):
        raise ValueError("observer must be callable")

    if not callable(sleeper):
        raise ValueError("sleeper must be callable")

    _countdown(wait_seconds, sleeper=sleeper)
    observation = observer()
    return _khan_qualification_report(observation)


def run_khan_read_only_qualification(
    *,
    observer=observe_wikipedia,
    sleeper=time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
) -> int:
    """Print one read-only Khan Academy qualification report."""

    report = run_khan_qualification(
        observer=observer,
        sleeper=sleeper,
        wait_seconds=wait_seconds,
    )
    print_khan_qualification_report(report)
    return 0


def _khan_qualification_report(
    observation: WikipediaObservation,
) -> KhanQualificationReport:
    snapshot = observation.snapshot
    relevant_elements = _khan_relevant_element_summaries(
        snapshot.fused_elements
    )
    highlighted_elements = _khan_highlighted_element_summaries(
        snapshot.fused_elements
    )
    return KhanQualificationReport(
        observation=observation,
        snapshot=snapshot,
        viewport=observation.viewport,
        warnings=snapshot.warnings,
        relevant_elements=relevant_elements,
        highlighted_elements=highlighted_elements,
        page_evidence=_khan_page_evidence(snapshot.fused_elements),
        grounding_diagnostics=_khan_grounding_diagnostics(
            snapshot.fused_elements
        ),
        browser_actions=False,
        openai_request=False,
        agent_loop_execution=False,
        evidence_promoted=False,
    )


def print_khan_qualification_report(
    report: KhanQualificationReport,
) -> None:
    """Print stable read-only Khan qualification evidence."""

    snapshot = report.snapshot
    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: Khan Academy read-only qualification")
    print(f"Target site: {KHAN_TARGET_SITE}")
    print(f"Target URL: {KHAN_TARGET_URL}")
    print(f"OpenAI request: {_yes_no(report.openai_request)}")
    print(f"Browser actions: {_yes_no(report.browser_actions)}")
    print(
        "AgentLoop execution: "
        f"{_yes_no(report.agent_loop_execution)}"
    )
    print(f"Evidence promoted: {_yes_no(report.evidence_promoted)}")
    print(f"Frontmost application: {report.observation.frontmost_app}")
    print(f"Viewport: {report.viewport}")
    print(f"Snapshot timestamp: {_snapshot_captured_at(snapshot)}")
    print(f"Snapshot warnings: {report.warnings}")
    print(f"Screenshot identity/path: {_snapshot_identity(snapshot)}")
    print(
        "Total raw accessibility semantic elements: "
        f"{len(report.observation.semantic_elements)}"
    )
    print(
        "Total snapshot accessibility elements: "
        f"{len(snapshot.accessibility_elements)}"
    )
    print(f"Total OCR elements: {len(snapshot.ocr_elements)}")
    print(f"Total fused elements: {len(snapshot.fused_elements)}")
    print("Khan page semantic evidence:")
    for item in report.page_evidence:
        print(f"  {item}")

    _print_khan_element_summaries(
        "Relevant Khan semantic elements",
        report.relevant_elements,
    )
    _print_khan_element_summaries(
        "Search/course/practice-like Khan candidates",
        report.highlighted_elements,
    )
    _print_khan_grounding_diagnostics(report.grounding_diagnostics)


def _khan_relevant_element_summaries(
    elements: Iterable[UIElement],
) -> tuple[KhanElementSummary, ...]:
    return tuple(
        _khan_element_summary(element)
        for element in elements
        if _is_khan_relevant_element(element)
    )


def _khan_highlighted_element_summaries(
    elements: Iterable[UIElement],
) -> tuple[KhanElementSummary, ...]:
    return tuple(
        _khan_element_summary(element)
        for element in elements
        if _khan_text_matches_terms(element, KHAN_HIGHLIGHT_TERMS)
    )


def _khan_element_summary(element: UIElement) -> KhanElementSummary:
    return KhanElementSummary(
        text=element.text,
        identifier=element.identifier,
        element_type=element.element_type,
        source=element.source,
        confidence=element.confidence,
        enabled=element.enabled,
        bounds=element.bounding_box,
    )


def _khan_page_evidence(
    elements: Iterable[UIElement],
) -> tuple[str, ...]:
    logged_out = _matched_khan_terms(elements, KHAN_LOGGED_OUT_TERMS)
    logged_in = _matched_khan_terms(elements, KHAN_LOGGED_IN_TERMS)
    evidence: list[str] = []
    if logged_out:
        evidence.append(f"logged-out-like signals: {logged_out}")
    if logged_in:
        evidence.append(f"logged-in/dashboard-like signals: {logged_in}")
    if not evidence:
        evidence.append("login-state semantic evidence: inconclusive")
    return tuple(evidence)


def _khan_grounding_diagnostics(
    elements: Iterable[UIElement],
) -> tuple[KhanGroundingDiagnostic, ...]:
    element_tuple = tuple(elements)
    grounder = UIGrounder()
    diagnostics: list[KhanGroundingDiagnostic] = []
    seen: set[tuple[str, str]] = set()
    for element in element_tuple:
        if not _is_khan_relevant_element(element):
            continue
        if not _khan_text_matches_terms(element, KHAN_HIGHLIGHT_TERMS):
            continue
        if element.text is None or not element.text.strip():
            continue

        key = (
            normalize_ui_text(element.text),
            _normalized_element_type(element.element_type),
        )
        if key in seen:
            continue
        seen.add(key)

        target = TargetSpec(
            text=element.text,
            element_types=(element.element_type,),
        )
        diagnostics.append(
            KhanGroundingDiagnostic(
                target=target,
                result=grounder.ground(target, element_tuple),
            )
        )
        if len(diagnostics) >= KHAN_GROUNDING_PROBE_LIMIT:
            break

    return tuple(diagnostics)


def _print_khan_element_summaries(
    title: str,
    elements: Sequence[KhanElementSummary],
) -> None:
    print(f"{title}: {len(elements)}")
    if not elements:
        print("  none")
        return

    for index, element in enumerate(elements, start=1):
        print(f"  Element {index}:")
        print(f"    text: {element.text!r}")
        print(f"    identifier: {element.identifier}")
        print(f"    element_type: {element.element_type}")
        print(f"    source: {element.source}")
        print(f"    confidence: {element.confidence}")
        print(f"    enabled: {element.enabled}")
        print(f"    bounds: {_format_bounds(element.bounds)}")


def _print_khan_grounding_diagnostics(
    diagnostics: Sequence[KhanGroundingDiagnostic],
) -> None:
    print(f"Read-only UIGrounder diagnostics: {len(diagnostics)}")
    if not diagnostics:
        print("  none")
        return

    for index, diagnostic in enumerate(diagnostics, start=1):
        result = diagnostic.result
        print(f"  Probe {index}:")
        print(f"    target text: {diagnostic.target.text}")
        print(f"    target element_types: {diagnostic.target.element_types}")
        print(f"    GroundingStatus: {result.status.value}")
        print(f"    reason: {result.reason}")
        print(f"    candidate count: {len(result.candidates)}")
        for candidate_index, candidate in enumerate(
            result.candidates,
            start=1,
        ):
            print(f"    candidate {candidate_index}:")
            print(f"      eligible: {candidate.eligible}")
            print(f"      rejection_reasons: {candidate.rejection_reasons}")


def _build_live_dependencies():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import (
        MacOSAccessibility,
        PerceptionEngine,
        ScreenCapture,
        TesseractOCR,
        UIElementFusion,
    )

    accessibility = MacOSAccessibility()
    controller = ComputerController()
    engine = PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=(
            Path(tempfile.gettempdir())
            / "computer_agent_experiment_10_cross_site_generalization.png"
        ),
    )
    return accessibility, engine


def _build_macos_accessibility():
    return _build_macos_accessibility_cls()()


def _build_macos_accessibility_cls():
    from computer_agent.perception import MacOSAccessibility

    return MacOSAccessibility


def _build_perception_engine(
    *,
    capture_path: str | Path,
    accessibility,
):
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import (
        PerceptionEngine,
        ScreenCapture,
        TesseractOCR,
        UIElementFusion,
    )

    return PerceptionEngine(
        screen_capture=ScreenCapture(ComputerController()),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )


def _build_live_openai_reasoner() -> LLMReasoner:
    from computer_agent.reasoning.openai_client import OpenAILLMClient

    return LLMReasoner(client=OpenAILLMClient())


def _build_agent_loop_cls():
    from computer_agent.agent import AgentLoop

    return AgentLoop


def _build_executor():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.tools.computer import create_computer_tools
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    return ToolExecutor(ToolRegistry(create_computer_tools(ComputerController())))


def _final_observation_report(
    observation: WikipediaObservation,
) -> FinalObservationReport:
    grounding = UIGrounder().ground(
        _FINAL_DESTINATION_TARGET,
        observation.snapshot.fused_elements,
    )
    return FinalObservationReport(
        observation=observation,
        destination_grounding=grounding,
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


def _candidate_search_text(candidate: CandidateRecord) -> str:
    parts = (
        candidate.text,
        str(candidate.value) if candidate.value is not None else None,
        candidate.production_element_type,
        candidate.raw_role,
    )
    return " ".join(part for part in parts if part)


def _is_khan_relevant_element(element: UIElement) -> bool:
    return _normalized_element_type(element.element_type) in {
        _normalized_element_type(element_type)
        for element_type in KHAN_RELEVANT_ELEMENT_TYPES
    }


def _khan_text_matches_terms(
    element: UIElement,
    terms: Sequence[str],
) -> bool:
    text = normalize_ui_text(element.text)
    if not text:
        return False
    return any(normalize_ui_text(term) in text for term in terms)


def _matched_khan_terms(
    elements: Iterable[UIElement],
    terms: Sequence[str],
) -> tuple[str, ...]:
    matched: list[str] = []
    normalized_texts = tuple(
        normalize_ui_text(element.text)
        for element in elements
        if element.text is not None
    )
    for term in terms:
        normalized_term = normalize_ui_text(term)
        if any(normalized_term in text for text in normalized_texts):
            matched.append(term)
    return tuple(matched)


def _normalized_element_type(element_type: str) -> str:
    return normalize_ui_text(element_type).replace(" ", "_")


def _format_bounds(bounds: BoundingBox | None) -> str:
    if bounds is None:
        return "None"
    return (
        f"x={bounds.x}, y={bounds.y}, "
        f"width={bounds.width}, height={bounds.height}"
    )


def _snapshot_captured_at(snapshot: object) -> object | None:
    frame = getattr(snapshot, "frame", None)
    return getattr(frame, "captured_at", None)


def _snapshot_identity(snapshot: object) -> str | None:
    frame = getattr(snapshot, "frame", None)
    image_path = getattr(frame, "image_path", None)
    if image_path is None:
        return None
    return str(image_path)


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
    if step.target.element_types != _SEARCH_FIELD_ELEMENT_TYPES:
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
    if step.action_target.element_types != _SUBMIT_ELEMENT_TYPES:
        failures.append(
            "step 2 action target element_types were "
            f"{step.action_target.element_types}"
        )
    if not isinstance(step.verification_target, TargetSpec):
        failures.append(
            "step 2 verification target was "
            f"{type(step.verification_target).__name__}, not TargetSpec"
        )
        return tuple(failures)
    if step.verification_spec is not None:
        failures.append("step 2 verification_spec was present")
    if step.verification_target.text != ARTICLE_HEADING_TEXT:
        failures.append(
            "step 2 verification target text was "
            f"{step.verification_target.text}"
        )
    if step.verification_target.element_types != _ARTICLE_HEADING_ELEMENT_TYPES:
        failures.append(
            "step 2 verification target element_types were "
            f"{step.verification_target.element_types}"
        )
    if step.max_attempts != 1:
        failures.append(f"step 2 max_attempts was {step.max_attempts}")

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


def _generic_second_step_failures(step: object) -> tuple[str, ...]:
    failures: list[str] = []
    if not isinstance(step, PlanStep):
        return (f"execution step 2 was {type(step).__name__}, not PlanStep",)

    if step.operation is not PlanOperation.CLICK_TARGET:
        failures.append(f"execution step 2 operation was {step.operation}")
    if not step.goal.strip():
        failures.append("execution step 2 goal was empty")
    if step.action_target.text != SUBMIT_TARGET_TEXT:
        failures.append(
            "execution step 2 action target text was "
            f"{step.action_target.text}"
        )
    if step.action_target.element_types != _SUBMIT_ELEMENT_TYPES:
        failures.append(
            "execution step 2 action target element_types were "
            f"{step.action_target.element_types}"
        )
    if step.verification_target is not None:
        failures.append("execution step 2 verification_target was present")
    if not isinstance(step.verification_spec, VerificationSpec):
        failures.append(
            "execution step 2 verification_spec was "
            f"{type(step.verification_spec).__name__}, not VerificationSpec"
        )
    else:
        failures.extend(
            _verification_spec_failures(
                "execution step 2",
                step.verification_spec,
            )
        )
    if step.max_attempts != 1:
        failures.append(
            f"execution step 2 max_attempts was {step.max_attempts}"
        )

    failures.extend(
        _target_payload_failures(
            "execution step 2 action target",
            step.action_target,
        )
    )
    return tuple(failures)


def _build_generic_execution_plan(plan: StructuredPlan) -> StructuredPlan:
    """Convert an accepted Experiment 10 plan to a generic execution plan."""

    failures = plan_acceptance_failures(plan)
    if failures:
        raise ValueError(
            "accepted planning plan failed trusted conversion: "
            + "; ".join(failures)
        )

    first, second = plan.steps
    execution_plan = StructuredPlan(
        task_goal=plan.task_goal,
        steps=(
            WebTextInputStep(
                goal=first.goal,
                target=TargetSpec(
                    text=first.target.text,
                    identifier=first.target.identifier,
                    element_types=first.target.element_types,
                    reference_point=first.target.reference_point,
                    minimum_confidence=first.target.minimum_confidence,
                ),
                input_text=first.input_text,
                max_attempts=first.max_attempts,
            ),
            PlanStep(
                goal=second.goal,
                operation=second.operation,
                action_target=TargetSpec(
                    text=second.action_target.text,
                    identifier=second.action_target.identifier,
                    element_types=second.action_target.element_types,
                    reference_point=second.action_target.reference_point,
                    minimum_confidence=second.action_target.minimum_confidence,
                ),
                verification_target=None,
                verification_spec=_wikipedia_execution_verification_spec(),
                max_attempts=second.max_attempts,
            ),
        ),
    )

    execution_failures = execution_plan_acceptance_failures(execution_plan)
    if execution_failures:
        raise ValueError(
            "generic execution plan failed trusted acceptance: "
            + "; ".join(execution_failures)
        )

    return execution_plan


def _wikipedia_execution_verification_spec() -> VerificationSpec:
    return VerificationSpec(
        before_conditions=(
            UIStateCondition(
                target=_LANDING_IDENTITY_TARGET,
                expectation=PresenceExpectation.PRESENT,
            ),
        ),
        after_conditions=(
            UIStateCondition(
                target=_FINAL_DESTINATION_TARGET,
                expectation=PresenceExpectation.PRESENT,
            ),
            UIStateCondition(
                target=_LANDING_IDENTITY_TARGET,
                expectation=PresenceExpectation.ABSENT,
            ),
        ),
    )


def _verification_spec_failures(
    label: str,
    spec: VerificationSpec,
) -> tuple[str, ...]:
    failures: list[str] = []
    expected = _wikipedia_execution_verification_spec()
    if spec.before_conditions != expected.before_conditions:
        failures.append(
            f"{label} before_conditions were {spec.before_conditions!r}"
        )
    if spec.after_conditions != expected.after_conditions:
        failures.append(
            f"{label} after_conditions were {spec.after_conditions!r}"
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
            if step.verification_target is not None:
                items.append(
                    (
                        f"step {index} verification target",
                        step.verification_target,
                    )
                )
            if step.verification_spec is not None:
                for condition_index, condition in enumerate(
                    (
                        *step.verification_spec.before_conditions,
                        *step.verification_spec.after_conditions,
                    ),
                    start=1,
                ):
                    items.append(
                        (
                            "step "
                            f"{index} verification condition "
                            f"{condition_index} target",
                            condition.target,
                        )
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
        print(f"Initial app: {observation.frontmost_app}")
        print(f"Initial viewport: {observation.viewport}")
    if preconditions.landing_identity_grounding is not None:
        print(
            "Initial Wikipedia identity grounding status: "
            f"{preconditions.landing_identity_grounding.status.value}"
        )
    if preconditions.search_grounding is not None:
        print(
            "Initial Search Wikipedia grounding status: "
            f"{preconditions.search_grounding.status.value}"
        )
    if preconditions.submit_grounding is not None:
        print(
            "Initial Search button grounding status: "
            f"{preconditions.submit_grounding.status.value}"
        )
    if preconditions.destination_grounding is not None:
        print(
            "Initial Computer vision grounding status: "
            f"{preconditions.destination_grounding.status.value}"
        )


def _print_agent_result(result: object) -> None:
    print(f"AgentLoopResult reason: {getattr(result, 'reason', '<missing>')}")
    status = getattr(result, "status", "<missing>")
    if hasattr(status, "value"):
        status = status.value
    print(f"Agent loop status: {status}")
    state = getattr(result, "state", None)
    state_status = getattr(state, "status", "<missing>")
    if hasattr(state_status, "value"):
        state_status = state_status.value
    print(f"Agent state: {state_status}")
    print(
        "Completed plan steps: "
        f"{getattr(result, 'completed_plan_steps', '<missing>')}"
    )
    records = tuple(getattr(state, "steps", ())) if state is not None else ()
    print(f"Action executions: {len(records)}")
    print(
        "Action tool order: "
        f"{tuple(record.action.tool_name for record in records)}"
    )
    for index, record in enumerate(records, start=1):
        print(f"Record {index} tool: {record.action.tool_name}")
        print(f"Record {index} arguments: {record.action.arguments}")
        print(f"Record {index} ToolResult success: {record.result.success}")
        print(f"Record {index} ToolResult error: {record.result.error}")


def _countdown(seconds: int, *, sleeper) -> None:
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


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only semantic qualification of wikipedia.org through "
            "production perception."
        )
    )
    parser.add_argument(
        "--offline-plan",
        action="store_true",
        help="Construct and accept the deterministic offline Wikipedia plan.",
    )
    parser.add_argument(
        "--live-openai",
        action="store_true",
        help=(
            "Make exactly one OpenAI planning request and perform no "
            "browser actions."
        ),
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
        "--khan-read-only",
        action="store_true",
        help=(
            "Take one read-only Khan Academy qualification observation "
            "after the countdown."
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to count down before the read-only observation.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    reasoner_builder=None,
    live_execution_runner=run_live_execution,
) -> int:
    args = _parse_args(argv)
    if args.khan_read_only and (
        args.offline_plan or args.live_openai or args.execute
    ):
        print("Phase 05 Experiment 10: Cross-Site Generalization")
        print("Experiment increment: rejected")
        print("OpenAI request: no")
        print("Browser actions: no")
        print("Khan qualification: failed")
        print(
            "  --khan-read-only cannot be combined with --offline-plan, "
            "--live-openai, or --execute"
        )
        return 2

    if args.execute and not args.live_openai:
        print("Phase 05 Experiment 10: Cross-Site Generalization")
        print("Experiment increment: rejected")
        print("Live OpenAI request: no")
        print("Browser actions: no")
        print("Planning acceptance: failed")
        print("  --execute requires --live-openai")
        return 2

    if args.offline_plan and args.live_openai:
        print("Phase 05 Experiment 10: Cross-Site Generalization")
        print("Experiment increment: rejected")
        print("Live OpenAI request: no")
        print("Browser actions: no")
        print("Planning acceptance: failed")
        print("  --offline-plan and --live-openai are mutually exclusive")
        return 2

    if args.offline_plan:
        return run_offline_acceptance()
    if args.khan_read_only:
        return run_khan_read_only_qualification(wait_seconds=args.wait_seconds)
    if args.live_openai:
        if args.execute:
            report = live_execution_runner(
                reasoner_builder=reasoner_builder,
                wait_seconds=args.wait_seconds,
            )
            print_execution_report(report)
            return 1 if report.execution_failures else 0
        return run_live_openai_acceptance(
            reasoner_builder=reasoner_builder,
        )
    return run_read_only_audit(wait_seconds=args.wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
