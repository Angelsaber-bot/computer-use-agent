"""Phase 05 Experiment 10: Khan Academy cross-site validation helpers.

The trusted Khan Academy SAT Math navigation contract is validated offline
first, then reused by a planning-only OpenAI qualification mode.  Live
planning never observes or operates the browser.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import shutil
import sys
import time
from typing import Sequence

from computer_agent.core.models import Action
from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    PerceptionSnapshot,
    SemanticAXElement,
    Viewport,
)
from computer_agent.planning import PlanOperation, PlanStep, StructuredPlan
from computer_agent.reasoning import LLMReasoner, ReasoningResult, ReasoningStatus
from computer_agent.verification import (
    PresenceExpectation,
    StateObserver,
    StateTransitionVerificationResult,
    StateTransitionVerifier,
    StateVerificationStatus,
    UIStateCondition,
    VerificationSpec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase05_real_web_autonomy"
KHAN_CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_10_khan_agent_loop_execution_candidate.png"
)
KHAN_FORMAL_EVIDENCE_PATH = (
    SCREENSHOT_DIR / "experiment_10_khan_agent_loop_execution.png"
)
KHAN_TASK_INTENT = (
    'On the Khan Academy SAT Math course page, open "UNIT 2 Foundations: '
    'Algebra" and verify that the "Unit 2: Foundations: Algebra" heading '
    'appears.'
)
KHAN_COURSE_HEADING_TEXT = "SAT Math"
KHAN_ACTION_TARGET_TEXT = "UNIT 2 Foundations: Algebra"
KHAN_DESTINATION_HEADING_TEXT = "Unit 2: Foundations: Algebra"
KHAN_EXPECTED_PLAN_STEPS = 1
KHAN_EXPECTED_ACTION_ORDER = ("click_mouse",)
KHAN_EXPECTED_ACTION_EXECUTIONS = 1
KHAN_EXPECTED_APPLICATION_NAME = "Google Chrome"
DEFAULT_WAIT_SECONDS = 8
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.5
OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6

# The current provider-neutral reasoner does not expose a link role.  For this
# audited Experiment 10 interaction, the action target therefore uses the
# unique exact visible text with an empty role tuple.  Runtime action grounding
# still resolves the actual element through UIGrounder before execution.
_KHAN_ACTION_ELEMENT_TYPES: tuple[str, ...] = ()
_KHAN_HEADING_ELEMENT_TYPES = ("heading",)

_KHAN_COURSE_HEADING_TARGET = TargetSpec(
    text=KHAN_COURSE_HEADING_TEXT,
    element_types=_KHAN_HEADING_ELEMENT_TYPES,
)
_KHAN_ACTION_TARGET = TargetSpec(
    text=KHAN_ACTION_TARGET_TEXT,
    element_types=_KHAN_ACTION_ELEMENT_TYPES,
)
_KHAN_DESTINATION_HEADING_TARGET = TargetSpec(
    text=KHAN_DESTINATION_HEADING_TEXT,
    element_types=_KHAN_HEADING_ELEMENT_TYPES,
)

_FORBIDDEN_PLAN_FIELDS = frozenset(
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
class KhanOfflinePlanningReport:
    """Offline trusted-plan and generic-conversion evidence for Khan."""

    live_openai_request: bool
    browser_actions: bool
    plan: StructuredPlan
    execution_plan: StructuredPlan | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KhanLiveOpenAIPlanningReport:
    """Exactly one OpenAI planning request with no browser side effects."""

    live_openai_request: bool
    browser_actions: bool
    agent_loop_execution: bool
    result: ReasoningResult
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KhanLiveObservation:
    """One read-only live observation of the frontmost Khan browser state."""

    frontmost_app: str | None
    viewport: Viewport | None
    snapshot: PerceptionSnapshot
    semantic_elements: tuple[SemanticAXElement, ...]


@dataclass(frozen=True, slots=True)
class KhanLivePreconditionReport:
    """Read-only execution gate evidence before executor construction."""

    observation: KhanLiveObservation | None
    course_heading_grounding: GroundingResult | None
    course_heading_observation: object | None
    destination_heading_observation: object | None
    action_target_grounding: GroundingResult | None
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KhanFinalObservationReport:
    """Read-only post-execution destination evidence."""

    observation: KhanLiveObservation
    destination_heading_grounding: GroundingResult


@dataclass(frozen=True, slots=True)
class KhanGenericVerificationRecord:
    """One full generic state-transition verification call."""

    verification_spec: VerificationSpec
    before_snapshot_captured_at: object | None
    before_snapshot_identity: str | None
    after_snapshot_captured_at: object | None
    after_snapshot_identity: str | None
    result: StateTransitionVerificationResult


@dataclass(frozen=True, slots=True)
class KhanExecutionReport:
    """Result of one gated live Khan AgentLoop execution attempt."""

    planning: KhanLiveOpenAIPlanningReport
    execution_plan: StructuredPlan | None
    preconditions: KhanLivePreconditionReport | None
    agent_result: object | None
    final_report: KhanFinalObservationReport | None
    generic_verification_records: tuple[KhanGenericVerificationRecord, ...]
    execution_failures: tuple[str, ...]
    evidence_promoted: bool


class KhanRecordingStateTransitionVerifier:
    """Delegate to production verification and record full contract calls."""

    def __init__(
        self,
        verifier: StateTransitionVerifier | None = None,
    ) -> None:
        self._verifier = (
            verifier if verifier is not None else StateTransitionVerifier()
        )
        self._records: list[KhanGenericVerificationRecord] = []

    @property
    def records(self) -> tuple[KhanGenericVerificationRecord, ...]:
        """Return recorded before/after contract calls in execution order."""

        return tuple(self._records)

    def verify(
        self,
        *,
        before_snapshot: PerceptionSnapshot,
        after_snapshot: PerceptionSnapshot,
        verification_spec: VerificationSpec,
    ) -> StateTransitionVerificationResult:
        """Run production generic verification and retain full contracts."""

        result = self._verifier.verify(
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            verification_spec=verification_spec,
        )
        if (
            verification_spec.before_conditions
            and verification_spec.after_conditions
        ):
            self._records.append(
                KhanGenericVerificationRecord(
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


class KhanLiveEnvironment:
    """Production browser observation stack for live Khan execution."""

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

    def observe(self) -> KhanLiveObservation:
        """Return the minimum read-only observation needed for the task."""

        self.sleeper(self.stabilization_wait_seconds)
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.perception_engine.observe()
        return KhanLiveObservation(
            frontmost_app=self.accessibility.read_frontmost_application_name(),
            viewport=self.accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(
                self.accessibility.read_frontmost_semantic_elements()
            ),
        )


def khan_offline_plan() -> StructuredPlan:
    """Return the exact audited one-click Khan Academy navigation plan."""

    return StructuredPlan(
        task_goal=KHAN_TASK_INTENT,
        steps=(
            PlanStep(
                goal="Open the Foundations: Algebra unit in SAT Math",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=KHAN_ACTION_TARGET_TEXT,
                    element_types=_KHAN_ACTION_ELEMENT_TYPES,
                ),
                verification_target=TargetSpec(
                    text=KHAN_DESTINATION_HEADING_TEXT,
                    element_types=_KHAN_HEADING_ELEMENT_TYPES,
                ),
                max_attempts=1,
            ),
        ),
    )


def khan_plan_acceptance_failures(plan: object) -> tuple[str, ...]:
    """Return fail-closed acceptance failures for the audited Khan plan."""

    if not isinstance(plan, StructuredPlan):
        return (f"plan was not a StructuredPlan: {type(plan).__name__}",)

    failures: list[str] = []
    if plan.task_goal != KHAN_TASK_INTENT:
        failures.append(f"plan task_goal was {plan.task_goal!r}")

    if len(plan.steps) != KHAN_EXPECTED_PLAN_STEPS:
        failures.append(
            f"plan step count was {len(plan.steps)}, not "
            f"{KHAN_EXPECTED_PLAN_STEPS}"
        )
        return tuple(failures)

    step = plan.steps[0]
    failures.extend(_khan_planning_step_failures(step))
    failures.extend(_plan_payload_failures(plan))
    return tuple(failures)


def khan_execution_plan_acceptance_failures(plan: object) -> tuple[str, ...]:
    """Return fail-closed failures for the trusted generic Khan plan."""

    if not isinstance(plan, StructuredPlan):
        return (
            f"execution plan was {type(plan).__name__}, not StructuredPlan",
        )

    failures: list[str] = []
    if plan.task_goal != KHAN_TASK_INTENT:
        failures.append(f"execution plan task_goal was {plan.task_goal!r}")

    if len(plan.steps) != KHAN_EXPECTED_PLAN_STEPS:
        failures.append(
            "execution plan step count was "
            f"{len(plan.steps)}, not {KHAN_EXPECTED_PLAN_STEPS}"
        )
        return tuple(failures)

    step = plan.steps[0]
    failures.extend(_khan_execution_step_failures(step))
    failures.extend(_plan_payload_failures(plan))
    return tuple(failures)


def build_khan_generic_execution_plan(plan: StructuredPlan) -> StructuredPlan:
    """Convert an accepted Khan planning plan into its state contract."""

    failures = khan_plan_acceptance_failures(plan)
    if failures:
        raise ValueError(
            "accepted Khan planning plan failed trusted conversion: "
            + "; ".join(failures)
        )

    planning_step = plan.steps[0]
    execution_plan = StructuredPlan(
        task_goal=plan.task_goal,
        steps=(
            PlanStep(
                goal=planning_step.goal,
                operation=planning_step.operation,
                action_target=_copy_target(planning_step.action_target),
                verification_target=None,
                verification_spec=khan_execution_verification_spec(),
                max_attempts=planning_step.max_attempts,
            ),
        ),
    )

    execution_failures = khan_execution_plan_acceptance_failures(
        execution_plan
    )
    if execution_failures:
        raise ValueError(
            "generic Khan execution plan failed trusted acceptance: "
            + "; ".join(execution_failures)
        )

    return execution_plan


def khan_execution_verification_spec() -> VerificationSpec:
    """Return the audited before/after semantic state transition contract."""

    return VerificationSpec(
        before_conditions=(
            UIStateCondition(
                target=_KHAN_COURSE_HEADING_TARGET,
                expectation=PresenceExpectation.PRESENT,
            ),
            UIStateCondition(
                target=_KHAN_DESTINATION_HEADING_TARGET,
                expectation=PresenceExpectation.ABSENT,
            ),
        ),
        after_conditions=(
            UIStateCondition(
                target=_KHAN_DESTINATION_HEADING_TARGET,
                expectation=PresenceExpectation.PRESENT,
            ),
            UIStateCondition(
                target=_KHAN_COURSE_HEADING_TARGET,
                expectation=PresenceExpectation.ABSENT,
            ),
        ),
    )


def run_khan_offline_planning(
    *,
    plan_builder=khan_offline_plan,
) -> KhanOfflinePlanningReport:
    """Construct and validate the Khan plan without any live side effects."""

    plan = plan_builder()
    failures = list(khan_plan_acceptance_failures(plan))
    execution_plan = None
    if not failures:
        try:
            execution_plan = build_khan_generic_execution_plan(plan)
        except ValueError as error:
            failures.append(str(error))
        else:
            failures.extend(
                khan_execution_plan_acceptance_failures(execution_plan)
            )

    return KhanOfflinePlanningReport(
        live_openai_request=False,
        browser_actions=False,
        plan=plan,
        execution_plan=execution_plan,
        failures=tuple(failures),
    )


def khan_reasoning_acceptance_failures(result: object) -> tuple[str, ...]:
    """Return fail-closed reasoning and plan failures for Khan planning."""

    if not isinstance(result, ReasoningResult):
        return (
            f"result was not ReasoningResult: {type(result).__name__}",
        )

    failures: list[str] = []
    if result.status is not ReasoningStatus.READY:
        failures.append(f"ReasoningStatus was {result.status.value}")

    if not isinstance(result.plan, StructuredPlan):
        failures.append(
            "plan was not a StructuredPlan: "
            f"{type(result.plan).__name__}"
        )
        return tuple(failures)

    failures.extend(khan_plan_acceptance_failures(result.plan))
    return tuple(failures)


def run_khan_live_openai_planning(
    *,
    reasoner_builder=None,
) -> KhanLiveOpenAIPlanningReport:
    """Make exactly one OpenAI planning request and never touch the browser."""

    reasoner = (
        reasoner_builder()
        if reasoner_builder is not None
        else _build_live_openai_reasoner()
    )
    result = reasoner.reason(KHAN_TASK_INTENT)
    return KhanLiveOpenAIPlanningReport(
        live_openai_request=True,
        browser_actions=False,
        agent_loop_execution=False,
        result=result,
        failures=khan_reasoning_acceptance_failures(result),
    )


def print_khan_live_openai_planning_report(
    report: KhanLiveOpenAIPlanningReport,
) -> None:
    """Print stable planning-only evidence for the live Khan request."""

    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: live OpenAI Khan planning")
    print("Live OpenAI request: yes")
    print("Browser actions: no")
    print("AgentLoop execution: no")
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
            print(f"Step {index} class: {type(step).__name__}")
            print(f"Step {index} goal: {step.goal}")
            if isinstance(step, PlanStep):
                print(f"Step {index} operation: {step.operation.value}")
                print(f"Step {index} max_attempts: {step.max_attempts}")
                print(
                    f"Step {index} action target text: "
                    f"{step.action_target.text}"
                )
                print(
                    f"Step {index} action target element_types: "
                    f"{step.action_target.element_types}"
                )
                verification_target = step.verification_target
                if verification_target is None:
                    print(
                        f"Step {index} verification target: <none>"
                    )
                else:
                    print(
                        f"Step {index} verification target text: "
                        f"{verification_target.text}"
                    )
                    print(
                        f"Step {index} verification target element_types: "
                        f"{verification_target.element_types}"
                    )

    if report.failures:
        print("Planning acceptance: failed")
        for failure in report.failures:
            print(f"  {failure}")
        return

    print("Planning acceptance: passed")


def run_khan_live_openai_acceptance(
    *,
    reasoner_builder=None,
) -> int:
    """Run and print the Khan planning-only OpenAI qualification."""

    report = run_khan_live_openai_planning(
        reasoner_builder=reasoner_builder,
    )
    print_khan_live_openai_planning_report(report)
    return 1 if report.failures else 0


def run_khan_live_execution(
    *,
    reasoner_builder=None,
    environment_builder=KhanLiveEnvironment,
    executor_builder=None,
    agent_loop_cls=None,
    platform_name: str = sys.platform,
    accessibility_cls=None,
    capture_path: str | Path = KHAN_CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = KHAN_FORMAL_EVIDENCE_PATH,
    sleeper=time.sleep,
    wait_seconds: int = DEFAULT_WAIT_SECONDS,
    stabilization_wait_seconds: float = DEFAULT_STABILIZATION_WAIT_SECONDS,
) -> KhanExecutionReport:
    """Run gated live Khan planning and execute once through AgentLoop."""

    planning = run_khan_live_openai_planning(
        reasoner_builder=reasoner_builder,
    )
    if planning.failures or planning.result.plan is None:
        return KhanExecutionReport(
            planning=planning,
            execution_plan=None,
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=planning.failures,
            evidence_promoted=False,
        )

    try:
        execution_plan = build_khan_generic_execution_plan(
            planning.result.plan
        )
    except ValueError as error:
        return KhanExecutionReport(
            planning=planning,
            execution_plan=None,
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=(str(error),),
            evidence_promoted=False,
        )

    execution_failures = khan_execution_plan_acceptance_failures(
        execution_plan
    )
    if execution_failures:
        return KhanExecutionReport(
            planning=planning,
            execution_plan=execution_plan,
            preconditions=None,
            agent_result=None,
            final_report=None,
            generic_verification_records=(),
            execution_failures=execution_failures,
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
        preconditions = KhanLivePreconditionReport(
            observation=None,
            course_heading_grounding=None,
            course_heading_observation=None,
            destination_heading_observation=None,
            action_target_grounding=None,
            failures=early_failures,
        )
        return KhanExecutionReport(
            planning=planning,
            execution_plan=execution_plan,
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
    preconditions = khan_live_precondition_failures(
        environment=environment,
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
    )
    if preconditions.failures:
        return KhanExecutionReport(
            planning=planning,
            execution_plan=execution_plan,
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
    state_transition_verifier = KhanRecordingStateTransitionVerifier(
        StateTransitionVerifier(state_observer=state_observer)
    )
    agent_result = agent_loop_cls(
        perception_engine=environment.perception_engine,
        grounder=grounder,
        executor=executor,
        state_transition_verifier=state_transition_verifier,
        post_action_settle_timeout_seconds=0.0,
    ).run(execution_plan)
    final_report = _khan_final_observation_report(environment.observe())
    generic_records = state_transition_verifier.records
    execution_failures = khan_execution_acceptance_failures(
        planning=planning,
        execution_plan=execution_plan,
        preconditions=preconditions,
        agent_result=agent_result,
        final_report=final_report,
        generic_verification_records=generic_records,
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

    return KhanExecutionReport(
        planning=planning,
        execution_plan=execution_plan,
        preconditions=preconditions,
        agent_result=agent_result,
        final_report=final_report,
        generic_verification_records=generic_records,
        execution_failures=execution_failures,
        evidence_promoted=evidence_promoted,
    )


def khan_live_precondition_failures(
    *,
    environment: object,
    platform_name: str = sys.platform,
    accessibility_cls=None,
) -> KhanLivePreconditionReport:
    """Check all read-only live Khan execution preconditions."""

    if accessibility_cls is None:
        accessibility_cls = _build_macos_accessibility_cls()

    failures = list(
        _platform_accessibility_failures(
            platform_name=platform_name,
            accessibility_cls=accessibility_cls,
        )
    )
    if failures:
        return KhanLivePreconditionReport(
            observation=None,
            course_heading_grounding=None,
            course_heading_observation=None,
            destination_heading_observation=None,
            action_target_grounding=None,
            failures=tuple(failures),
        )

    observation = environment.observe()
    grounder = UIGrounder()
    state_observer = StateObserver()
    course_heading_grounding = grounder.ground(
        _KHAN_COURSE_HEADING_TARGET,
        observation.snapshot.fused_elements,
    )
    course_heading_observation = state_observer.observe(
        snapshot=observation.snapshot,
        target=_KHAN_COURSE_HEADING_TARGET,
    )
    destination_heading_observation = state_observer.observe(
        snapshot=observation.snapshot,
        target=_KHAN_DESTINATION_HEADING_TARGET,
    )
    action_target_grounding = grounder.ground(
        _KHAN_ACTION_TARGET,
        observation.snapshot.fused_elements,
    )

    if observation.frontmost_app != KHAN_EXPECTED_APPLICATION_NAME:
        failures.append(
            "frontmost app was "
            f"{observation.frontmost_app}, not {KHAN_EXPECTED_APPLICATION_NAME}"
        )

    if observation.snapshot.warnings:
        failures.append(
            f"perception warnings were {observation.snapshot.warnings}"
        )

    if course_heading_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "SAT Math heading grounding was "
            f"{course_heading_grounding.status.value}"
        )

    if course_heading_observation.status.value != "present":
        failures.append(
            "SAT Math heading state was "
            f"{course_heading_observation.status.value}, not present"
        )

    if destination_heading_observation.status.value != "absent":
        failures.append(
            "Unit 2 destination heading state was "
            f"{destination_heading_observation.status.value}, not absent"
        )

    if action_target_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(
            "UNIT 2 Foundations: Algebra action target grounding was "
            f"{action_target_grounding.status.value}"
        )

    return KhanLivePreconditionReport(
        observation=observation,
        course_heading_grounding=course_heading_grounding,
        course_heading_observation=course_heading_observation,
        destination_heading_observation=destination_heading_observation,
        action_target_grounding=action_target_grounding,
        failures=tuple(failures),
    )


def khan_execution_acceptance_failures(
    *,
    planning: KhanLiveOpenAIPlanningReport,
    execution_plan: StructuredPlan,
    preconditions: KhanLivePreconditionReport | None,
    agent_result: object,
    final_report: KhanFinalObservationReport | None,
    generic_verification_records: Sequence[KhanGenericVerificationRecord],
    candidate_path: Path,
) -> tuple[str, ...]:
    """Return fail-closed live Khan execution acceptance failures."""

    from computer_agent.agent import AgentLoopResult, AgentLoopStatus, AgentStatus

    failures = list(planning.failures)
    if planning.result.status is not ReasoningStatus.READY:
        failures.append(f"ReasoningStatus was {planning.result.status.value}")
    if planning.result.plan is None:
        failures.append("trusted plan was missing")
        return tuple(failures)

    failures.extend(khan_plan_acceptance_failures(planning.result.plan))
    failures.extend(khan_execution_plan_acceptance_failures(execution_plan))

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
    if agent_result.completed_plan_steps != KHAN_EXPECTED_PLAN_STEPS:
        failures.append(
            "completed plan steps were "
            f"{agent_result.completed_plan_steps}, not "
            f"{KHAN_EXPECTED_PLAN_STEPS}"
        )
    if len(records) != KHAN_EXPECTED_ACTION_EXECUTIONS:
        failures.append(
            "executed action count was "
            f"{len(records)}, not {KHAN_EXPECTED_ACTION_EXECUTIONS}"
        )
    if action_order != KHAN_EXPECTED_ACTION_ORDER:
        failures.append(
            f"action tool order was {action_order}, not "
            f"{KHAN_EXPECTED_ACTION_ORDER}"
        )
    if action_order.count("click_mouse") != 1:
        failures.append("click_mouse was not executed exactly once")
    if any(tool_name != "click_mouse" for tool_name in action_order):
        failures.append("non-click browser action was executed")

    for index, record in enumerate(records, start=1):
        if not record.result.success:
            failures.append(
                f"record {index} ToolResult failed: {record.result.error}"
            )

    generic_records = tuple(generic_verification_records)
    if len(generic_records) != 1:
        failures.append(
            "generic StateTransitionVerifier call count was "
            f"{len(generic_records)}, not 1"
        )
    elif generic_records[0].result.status is not StateVerificationStatus.VERIFIED:
        failures.append(
            "generic StateTransitionVerifier result was "
            f"{generic_records[0].result.status.value}"
        )

    if final_report is None:
        failures.append("final observation report was missing")
    else:
        if (
            final_report.destination_heading_grounding.status
            is not GroundingStatus.RESOLVED
        ):
            failures.append(
                "final Unit 2 heading grounding was "
                f"{final_report.destination_heading_grounding.status.value}"
            )
        if final_report.observation.frontmost_app != KHAN_EXPECTED_APPLICATION_NAME:
            failures.append(
                "final frontmost app was "
                f"{final_report.observation.frontmost_app}, not "
                f"{KHAN_EXPECTED_APPLICATION_NAME}"
            )
        if final_report.observation.snapshot.warnings:
            failures.append(
                "final perception warnings were "
                f"{final_report.observation.snapshot.warnings}"
            )

    if not candidate_path.exists():
        failures.append("candidate evidence file was missing")

    return tuple(failures)


def print_khan_offline_planning_report(
    report: KhanOfflinePlanningReport,
) -> None:
    """Print stable offline evidence for the Khan trusted contract."""

    print("Phase 05 Experiment 10: Cross-Site Generalization")
    print("Experiment increment: deterministic offline Khan plan")
    print("Live OpenAI request: no")
    print("Browser actions: no")
    print(f"task_goal: {report.plan.task_goal}")
    print(f"Plan steps: {len(report.plan.steps)}")
    step = report.plan.steps[0]
    print(f"Step 1 class: {type(step).__name__}")
    print(f"Step 1 goal: {step.goal}")
    print(f"Step 1 operation: {step.operation.value}")
    print(f"Step 1 max_attempts: {step.max_attempts}")
    print(f"Step 1 action target text: {step.action_target.text}")
    print(
        "Step 1 action target element_types: "
        f"{step.action_target.element_types}"
    )
    print(
        "Step 1 verification target text: "
        f"{step.verification_target.text}"
    )
    print(
        "Step 1 verification target element_types: "
        f"{step.verification_target.element_types}"
    )

    if report.execution_plan is not None:
        execution_step = report.execution_plan.steps[0]
        print("Trusted generic conversion: yes")
        print(
            "Execution verification_target: "
            f"{execution_step.verification_target}"
        )
        print(
            "Execution before_conditions: "
            f"{execution_step.verification_spec.before_conditions}"
        )
        print(
            "Execution after_conditions: "
            f"{execution_step.verification_spec.after_conditions}"
        )
    else:
        print("Trusted generic conversion: no")

    print(
        "Offline acceptance: "
        f"{'passed' if not report.failures else 'failed'}"
    )
    for failure in report.failures:
        print(f"  {failure}")


def _khan_planning_step_failures(step: object) -> tuple[str, ...]:
    if not isinstance(step, PlanStep):
        return (f"step 1 was {type(step).__name__}, not PlanStep",)

    failures: list[str] = []
    if step.operation is not PlanOperation.CLICK_TARGET:
        failures.append(f"step 1 operation was {step.operation}")
    if not step.goal.strip():
        failures.append("step 1 goal was empty")
    if step.action_target.text != KHAN_ACTION_TARGET_TEXT:
        failures.append(
            f"step 1 action target text was {step.action_target.text}"
        )
    if step.action_target.element_types != _KHAN_ACTION_ELEMENT_TYPES:
        failures.append(
            "step 1 action target element_types were "
            f"{step.action_target.element_types}"
        )
    if step.verification_spec is not None:
        failures.append("step 1 verification_spec was present")
    if not isinstance(step.verification_target, TargetSpec):
        failures.append(
            "step 1 verification target was "
            f"{type(step.verification_target).__name__}, not TargetSpec"
        )
        return tuple(failures)
    if step.verification_target.text != KHAN_DESTINATION_HEADING_TEXT:
        failures.append(
            "step 1 verification target text was "
            f"{step.verification_target.text}"
        )
    if step.verification_target.element_types != _KHAN_HEADING_ELEMENT_TYPES:
        failures.append(
            "step 1 verification target element_types were "
            f"{step.verification_target.element_types}"
        )
    if step.max_attempts != 1:
        failures.append(f"step 1 max_attempts was {step.max_attempts}")

    failures.extend(_target_payload_failures("step 1 action target", step.action_target))
    failures.extend(
        _target_payload_failures(
            "step 1 verification target",
            step.verification_target,
        )
    )
    return tuple(failures)


def _khan_execution_step_failures(step: object) -> tuple[str, ...]:
    if not isinstance(step, PlanStep):
        return (
            f"execution step 1 was {type(step).__name__}, not PlanStep",
        )

    failures: list[str] = []
    if step.operation is not PlanOperation.CLICK_TARGET:
        failures.append(f"execution step 1 operation was {step.operation}")
    if not step.goal.strip():
        failures.append("execution step 1 goal was empty")
    if step.action_target.text != KHAN_ACTION_TARGET_TEXT:
        failures.append(
            "execution step 1 action target text was "
            f"{step.action_target.text}"
        )
    if step.action_target.element_types != _KHAN_ACTION_ELEMENT_TYPES:
        failures.append(
            "execution step 1 action target element_types were "
            f"{step.action_target.element_types}"
        )
    if step.verification_target is not None:
        failures.append("execution step 1 verification_target was present")
    if step.verification_spec != khan_execution_verification_spec():
        failures.append(
            "execution step 1 verification_spec was "
            f"{step.verification_spec!r}"
        )
    if step.max_attempts != 1:
        failures.append(
            f"execution step 1 max_attempts was {step.max_attempts}"
        )
    failures.extend(
        _target_payload_failures(
            "execution step 1 action target",
            step.action_target,
        )
    )
    return tuple(failures)


def _copy_target(target: TargetSpec) -> TargetSpec:
    return TargetSpec(
        text=target.text,
        identifier=target.identifier,
        element_types=target.element_types,
        minimum_confidence=target.minimum_confidence,
        reference_point=target.reference_point,
    )


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


def _plan_payload_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []
    items: list[tuple[str, object]] = [("plan", plan)]
    for index, step in enumerate(plan.steps, start=1):
        items.append((f"step {index}", step))
        if isinstance(step, PlanStep):
            items.append((f"step {index} action target", step.action_target))
            if step.verification_target is not None:
                items.append(
                    (f"step {index} verification target", step.verification_target)
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
                            f"step {index} verification condition "
                            f"{condition_index} target",
                            condition.target,
                        )
                    )

    for item_name, item in items:
        if isinstance(item, Action):
            failures.append(f"{item_name} contained executable Action")
        if is_dataclass(item):
            field_names = {field.name for field in fields(item)}
            forbidden = field_names & _FORBIDDEN_PLAN_FIELDS
            if forbidden:
                failures.append(
                    f"{item_name} exposed forbidden fields {tuple(forbidden)}"
                )
    return tuple(failures)

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

    controller = ComputerController()
    return PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
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


def _khan_final_observation_report(
    observation: KhanLiveObservation,
) -> KhanFinalObservationReport:
    grounding = UIGrounder().ground(
        _KHAN_DESTINATION_HEADING_TARGET,
        observation.snapshot.fused_elements,
    )
    return KhanFinalObservationReport(
        observation=observation,
        destination_heading_grounding=grounding,
    )


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


def _snapshot_captured_at(snapshot: object) -> object | None:
    frame = getattr(snapshot, "frame", None)
    return getattr(frame, "captured_at", None)


def _snapshot_identity(snapshot: object) -> str | None:
    frame = getattr(snapshot, "frame", None)
    image_path = getattr(frame, "image_path", None)
    if image_path is None:
        return None
    return str(image_path)


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


def print_khan_execution_report(report: KhanExecutionReport) -> None:
    """Print stable live Khan execution evidence."""

    print_khan_live_openai_planning_report(report.planning)
    print("Execution mode: live OpenAI Khan AgentLoop execution")
    print("Execution browser actions: yes")
    if report.preconditions is not None:
        _print_khan_preconditions(report.preconditions)
    if report.agent_result is not None:
        _print_agent_result(report.agent_result)
    _print_khan_generic_verification_records(
        report.generic_verification_records
    )
    if report.final_report is not None:
        print(
            "Final Unit 2 heading grounding status: "
            f"{report.final_report.destination_heading_grounding.status.value}"
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


def _print_khan_preconditions(
    preconditions: KhanLivePreconditionReport,
) -> None:
    observation = preconditions.observation
    if observation is not None:
        print(f"Initial app: {observation.frontmost_app}")
        print(f"Initial viewport: {observation.viewport}")
        print(f"Initial warnings: {observation.snapshot.warnings}")
        print(
            "Initial snapshot timestamp: "
            f"{_snapshot_captured_at(observation.snapshot)}"
        )
        print(
            "Initial snapshot identity: "
            f"{_snapshot_identity(observation.snapshot)}"
        )
    if preconditions.course_heading_grounding is not None:
        print(
            "Initial SAT Math heading grounding status: "
            f"{preconditions.course_heading_grounding.status.value}"
        )
    if preconditions.course_heading_observation is not None:
        print(
            "Initial SAT Math heading observation status: "
            f"{preconditions.course_heading_observation.status.value}"
        )
    if preconditions.destination_heading_observation is not None:
        print(
            "Initial Unit 2 heading observation status: "
            f"{preconditions.destination_heading_observation.status.value}"
        )
    if preconditions.action_target_grounding is not None:
        print(
            "Initial action target grounding status: "
            f"{preconditions.action_target_grounding.status.value}"
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
        print(f"Record {index} ToolResult success: {record.result.success}")
        print(f"Record {index} ToolResult error: {record.result.error}")


def _print_khan_generic_verification_records(
    records: Sequence[KhanGenericVerificationRecord],
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
        _print_condition_evaluations("before", result.before_evaluations)
        _print_condition_evaluations("after", result.after_evaluations)


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
        observation = getattr(evaluation, "observation", None)
        grounding = getattr(evaluation, "grounding", None)
        if observation is not None:
            print("    evidence backend: state_observation")
            print(f"    observation status: {observation.status.value}")
            print(f"    observation reason: {observation.reason}")
            print(f"    candidate count: {len(observation.candidates)}")
            for candidate_index, candidate in enumerate(
                observation.candidates,
                start=1,
            ):
                _print_observation_candidate(candidate_index, candidate)
        elif grounding is not None:
            print("    evidence backend: grounding")
            print(f"    grounding status: {grounding.status.value}")
            print(f"    grounding reason: {grounding.reason}")
            print(f"    candidate count: {len(grounding.candidates)}")
        else:
            print("    evidence backend: <missing>")


def _print_observation_candidate(index: int, candidate: object) -> None:
    element = candidate.element
    print(f"    candidate {index}:")
    print(f"      match_basis: {candidate.match_basis}")
    print(f"      predicate_match: {candidate.predicate_match}")
    print(f"      mismatch_reasons: {candidate.mismatch_reasons}")
    print(f"      uncertainty_reasons: {candidate.uncertainty_reasons}")
    print(f"      distance: {candidate.distance}")
    print(f"      element text: {getattr(element, 'text', None)}")
    print(f"      element identifier: {getattr(element, 'identifier', None)}")
    print(f"      element element_type: {getattr(element, 'element_type', None)}")
    print(f"      element source: {getattr(element, 'source', None)}")
    print(f"      element confidence: {getattr(element, 'confidence', None)}")
    print(f"      element enabled: {getattr(element, 'enabled', None)}")
    print(f"      element bounds: {getattr(element, 'bounding_box', None)}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Khan Academy Experiment 10 trusted plan and "
            "planning-only OpenAI qualification."
        )
    )
    parser.add_argument(
        "--live-openai",
        action="store_true",
        help=(
            "Make exactly one OpenAI planning request with no browser "
            "observation or action."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "With --live-openai, run the trusted Khan plan through gated "
            "live AgentLoop execution."
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to count down before the live precondition observation.",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    reasoner_builder=None,
    live_execution_runner=run_khan_live_execution,
) -> int:
    args = _parse_args(argv)
    if args.execute and not args.live_openai:
        print("Phase 05 Experiment 10: Cross-Site Generalization")
        print("Experiment increment: rejected")
        print("Live OpenAI request: no")
        print("Browser actions: no")
        print("Planning acceptance: failed")
        print("  --execute requires --live-openai")
        return 2

    if args.live_openai:
        if args.execute:
            report = live_execution_runner(
                reasoner_builder=reasoner_builder,
                wait_seconds=args.wait_seconds,
            )
            print_khan_execution_report(report)
            return 1 if report.execution_failures else 0
        return run_khan_live_openai_acceptance(
            reasoner_builder=reasoner_builder,
        )

    report = run_khan_offline_planning()
    print_khan_offline_planning_report(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
