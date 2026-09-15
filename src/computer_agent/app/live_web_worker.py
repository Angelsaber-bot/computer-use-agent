"""Bounded live-web worker for the persistent Agent Workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import NoReturn
from urllib.parse import unquote, urlparse

from computer_agent.agent import (
    AgentLoop,
    AgentLoopStatus,
    AgentStatus,
    TextInputObservation,
)
from computer_agent.app.durable_planning import (
    DeterministicDurablePlanner,
    DurablePlanner,
    LLMDurablePlanner,
    default_durable_planning_context,
)
from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.core.models import Action
from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    SemanticTargetResolver,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    MacOSAccessibility,
    PerceptionEngine,
    ScreenCapture,
    TesseractOCR,
    UIElement,
    UIElementFusion,
    normalize_ui_text,
)
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from computer_agent.reasoning import (
    AdaptiveDecisionSnapshot,
    DecisionAttemptSnapshot,
)
from computer_agent.reasoning.openai_client import (
    OpenAILLMClient,
)

from computer_agent.runtime import (
    RuntimeControl,
    RuntimeTask,
    RuntimeWorker,
)
from computer_agent.task import (
    ArtifactRecord,
    ClaimRecord,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
)
from computer_agent.tools.computer import (
    create_computer_tools,
)
from computer_agent.tools.executor import (
    ToolExecutor,
)
from computer_agent.tools.registry import (
    ToolRegistry,
)


BROWSER_WINDOW_ARTIFACT_ID = (
    "live-web-browser-window"
)

WORKFLOW_ARTIFACT_ID = "live-web-workflow"
QUERY_ARTIFACT_ID = "live-web-query"
FOLLOWUP_TARGET_ARTIFACT_ID = (
    "live-web-followup-target"
)
FOLLOWUP_DESTINATION_ARTIFACT_ID = (
    "live-web-followup-destination"
)
DURABLE_PLAN_ARTIFACT_ID = (
    "live-web-durable-plan"
)
DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID = (
    "live-web-durable-planner-provenance"
)
DURABLE_PLAN_VERSION = 1
DURABLE_PLANNER_PROVENANCE_VERSION = 1

DURABLE_PLANNER_ENV_VAR = (
    "COMPUTER_AGENT_DURABLE_PLANNER"
)

FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID = (
    "live-web-followup-navigation-side-effect"
)
FOLLOWUP_NAVIGATION_ACTION_KEY = (
    "click_target:wikipedia_followup_link"
)
FOLLOWUP_LINK_READINESS_REOBSERVATIONS = 2

BROWSER_WINDOW_MARKER_PREFIX = (
    "about:blank#computer-agent-task="
)

CRASH_ENV_VAR = (
    "COMPUTER_AGENT_CRASH_AFTER"
)
CRASH_EXIT_CODE = 86


TaskStatePublisher = Callable[[], None]
AdaptiveDecisionPublisher = Callable[
    [object],
    None,
]


class LiveCrashPoint(StrEnum):
    """Development-only live-worker process crash injection points."""

    QUERY_CHECKPOINT = "query_checkpoint"
    SUBMIT_EXECUTION = "submit_execution"
    FOLLOWUP_EXECUTION = "followup_execution"


class DurablePlannerMode(StrEnum):
    """Selectable durable-planning modes for live web tasks."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


class QueryVerificationMode(StrEnum):
    """Postcondition strategy for verifying query entry."""

    FIELD_VALUE = "field_value"
    VISIBLE_SEARCH_UI = "visible_search_ui"


class DurableStepKind(StrEnum):
    """Semantic durable step operation kinds."""

    ENTER_TEXT = "enter_text"
    ACTIVATE_CONTROL = "activate_control"
    OPEN_LINK = "open_link"


class DurablePostconditionKind(StrEnum):
    """Semantic postconditions for durable plan steps."""

    QUERY_MATCHES = "query_matches"
    SEARCH_OUTCOME_VISIBLE = "search_outcome_visible"
    DESTINATION_HEADING_VISIBLE = "destination_heading_visible"


@dataclass(frozen=True, slots=True)
class QueryVerificationResult:
    """Fresh evidence that the configured query condition is satisfied."""

    element: UIElement
    summary: str
    source: str


@dataclass(frozen=True, slots=True)
class StepPostconditionResult:
    """Fresh evidence summary for one durable step postcondition."""

    summary: str
    source: str


@dataclass(frozen=True, slots=True)
class FollowupLinkReadinessResult:
    """Grounded follow-up link state after bounded read-only refreshes."""

    observation: TextInputObservation
    grounding: GroundingResult | None
    already_at_destination: bool = False


@dataclass(frozen=True, slots=True)
class DurableWebSearchSpec:
    """Configuration for one bounded durable search workflow."""

    workflow_id: str
    start_url: str
    working_url_prefix: str
    expected_application: str
    search_field: TargetSpec
    submit_target: TargetSpec
    result_target: TargetSpec
    workspace_artifact_id: str
    workspace_description: str
    query_claim_id: str
    query_subgoal_id: str
    query_claim_text: str
    query_subgoal_text: str
    result_claim_id: str
    result_subgoal_id: str
    result_claim_text: str
    result_subgoal_text: str
    submit_side_effect_id: str
    submit_action_key: str
    submit_description: str
    prepare_step_goal: str
    submit_step_goal: str
    query_verification_mode: QueryVerificationMode = (
        QueryVerificationMode.FIELD_VALUE
    )
    query_confirmation_target_factory: Callable[
        [str],
        tuple[TargetSpec, ...],
    ] | None = None
    result_target_factory: Callable[
        [str],
        tuple[TargetSpec, ...],
    ] | None = None
    followup_claim_id: str | None = None
    followup_subgoal_id: str | None = None
    followup_claim_text_factory: Callable[
        [str],
        str,
    ] | None = None
    followup_subgoal_text_factory: Callable[
        [str],
        str,
    ] | None = None
    followup_step_goal_factory: Callable[
        [str],
        str,
    ] | None = None


@dataclass(frozen=True, slots=True)
class ResolvedDurableWebSearchTask:
    """One static workflow plus per-task runtime parameters."""

    spec: DurableWebSearchSpec
    query_text: str
    followup_target_text: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.query_text, str)
            or not self.query_text.strip()
        ):
            raise ValueError(
                "query_text must be a non-empty string"
            )
        if self.followup_target_text is not None and (
            not isinstance(self.followup_target_text, str)
            or not self.followup_target_text.strip()
        ):
            raise ValueError(
                "followup_target_text must be a non-empty string or None"
            )


@dataclass(frozen=True, slots=True)
class DurableTaskStep:
    """One persisted semantic step in a durable task plan."""

    step_id: str
    kind: DurableStepKind
    description: str
    postcondition: DurablePostconditionKind
    claim_id: str
    claim_text: str
    subgoal_id: str
    subgoal_text: str
    action_target: TargetSpec | None = None
    verification_target: TargetSpec | None = None
    input_text: str | None = None
    side_effect_id: str | None = None
    side_effect_action_key: str | None = None
    side_effect_description: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string")
        if not isinstance(self.kind, DurableStepKind):
            raise ValueError("kind must be a DurableStepKind")
        if not isinstance(
            self.postcondition,
            DurablePostconditionKind,
        ):
            raise ValueError(
                "postcondition must be a DurablePostconditionKind"
            )


@dataclass(frozen=True, slots=True)
class DurableTaskPlan:
    """Ordered persisted semantic plan for one live-web task."""

    plan_id: str
    workflow_id: str
    version: int
    steps: tuple[DurableTaskStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be a non-empty string")
        if (
            not isinstance(self.workflow_id, str)
            or not self.workflow_id.strip()
        ):
            raise ValueError("workflow_id must be a non-empty string")
        if self.version != DURABLE_PLAN_VERSION:
            raise ValueError("unsupported durable plan version")
        if not self.steps:
            raise ValueError("durable plan must contain steps")
        seen: set[str] = set()
        for step in self.steps:
            if not isinstance(step, DurableTaskStep):
                raise ValueError("steps must contain DurableTaskStep")
            if step.step_id in seen:
                raise ValueError(f"duplicate durable step: {step.step_id}")
            seen.add(step.step_id)


@dataclass(frozen=True, slots=True)
class DurablePlannerProvenance:
    """Persisted source metadata for an accepted durable plan."""

    planner_type: str
    model_identifier: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.planner_type, str)
            or not self.planner_type.strip()
        ):
            raise ValueError("planner_type must be a non-empty string")
        if self.model_identifier is not None and (
            not isinstance(self.model_identifier, str)
            or not self.model_identifier.strip()
        ):
            raise ValueError(
                "model_identifier must be a non-empty string or None"
            )


EXPECTED_APP = "Google Chrome"

PYTHON_URL = "https://www.python.org/"


def _wikipedia_query_confirmation_targets(
    query_text: str,
) -> tuple[TargetSpec, ...]:
    containing = (
        f"Search for pages containing {query_text}"
    )
    return (
        TargetSpec(
            text=query_text,
            element_types=("text",),
            minimum_confidence=0.70,
        ),
        TargetSpec(
            text=containing,
            element_types=("text",),
            minimum_confidence=0.70,
        ),
        TargetSpec(
            text=containing,
            element_types=("link",),
            minimum_confidence=0.70,
        ),
    )


def _wikipedia_result_targets(
    query_text: str,
) -> tuple[TargetSpec, ...]:
    return (
        TargetSpec(
            text=query_text,
            element_types=("heading",),
            minimum_confidence=0.70,
        ),
    )


def _wikipedia_followup_claim_text(
    target_text: str,
) -> str:
    return (
        "The requested Wikipedia destination "
        f"{target_text!r} is visible."
    )


def _wikipedia_followup_subgoal_text(
    target_text: str,
) -> str:
    return (
        "Open the requested Wikipedia link "
        f"{target_text!r} and verify its destination."
    )


def _wikipedia_followup_step_goal(
    target_text: str,
) -> str:
    return (
        "Open the requested Wikipedia link "
        f"{target_text!r}."
    )


PYTHON_WORKFLOW = DurableWebSearchSpec(
    workflow_id="python-org-search",
    start_url=PYTHON_URL,
    working_url_prefix=PYTHON_URL,
    expected_application=EXPECTED_APP,
    search_field=TargetSpec(
        text="Search This Site",
        element_types=("text_field",),
        minimum_confidence=0.70,
    ),
    submit_target=TargetSpec(
        text="GO",
        element_types=("button",),
        minimum_confidence=0.70,
    ),
    result_target=TargetSpec(
        text="Results",
        element_types=(),
        minimum_confidence=0.70,
    ),
    workspace_artifact_id=BROWSER_WINDOW_ARTIFACT_ID,
    workspace_description=(
        "Agent-owned Google Chrome task window."
    ),
    query_claim_id="live-web-query-entered",
    query_subgoal_id="live-web-enter-query",
    query_claim_text=(
        "The intended python.org query has been entered."
    ),
    query_subgoal_text=(
        "Enter the intended python.org search query."
    ),
    result_claim_id="live-web-results-visible",
    result_subgoal_id="live-web-submit-query",
    result_claim_text=(
        "The python.org search completed successfully."
    ),
    result_subgoal_text=(
        "Submit the search and verify the results page."
    ),
    submit_side_effect_id=(
        "live-web-submit-query-side-effect"
    ),
    submit_action_key=(
        "click_target:python_org_search_go"
    ),
    submit_description=(
        "Submit the python.org search query through the GO control."
    ),
    prepare_step_goal=(
        "Enter the intended search query on python.org."
    ),
    submit_step_goal=(
        "Submit the persisted python.org search query."
    ),
)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Main_Page"

WIKIPEDIA_WORKFLOW = DurableWebSearchSpec(
    workflow_id="wikipedia-search",
    start_url=WIKIPEDIA_URL,
    working_url_prefix="https://en.wikipedia.org/",
    expected_application=EXPECTED_APP,
    search_field=TargetSpec(
        text="Search Wikipedia",
        element_types=("text_field",),
        minimum_confidence=0.70,
    ),
    submit_target=TargetSpec(
        text="Search",
        element_types=("button",),
        minimum_confidence=0.70,
    ),
    result_target=TargetSpec(
        text="Search results",
        element_types=("heading",),
        minimum_confidence=0.70,
    ),
    workspace_artifact_id=BROWSER_WINDOW_ARTIFACT_ID,
    workspace_description=(
        "Agent-owned Google Chrome task window."
    ),
    query_claim_id="live-web-wikipedia-query-entered",
    query_subgoal_id="live-web-wikipedia-enter-query",
    query_claim_text=(
        "The intended Wikipedia query has been entered."
    ),
    query_subgoal_text=(
        "Enter the intended Wikipedia search query."
    ),
    result_claim_id="live-web-wikipedia-results-visible",
    result_subgoal_id="live-web-wikipedia-submit-query",
    result_claim_text=(
        "The Wikipedia search results page is visible."
    ),
    result_subgoal_text=(
        "Submit the Wikipedia search and verify the results page."
    ),
    submit_side_effect_id=(
        "live-web-wikipedia-submit-query-side-effect"
    ),
    submit_action_key=(
        "click_target:wikipedia_search_submit"
    ),
    submit_description=(
        "Submit the Wikipedia search query through the Search control."
    ),
    prepare_step_goal=(
        "Enter the intended search query on Wikipedia."
    ),
    submit_step_goal=(
        "Submit the persisted Wikipedia search query."
    ),
    query_verification_mode=(
        QueryVerificationMode.VISIBLE_SEARCH_UI
    ),
    query_confirmation_target_factory=(
        _wikipedia_query_confirmation_targets
    ),
    result_target_factory=_wikipedia_result_targets,
    followup_claim_id=(
        "live-web-wikipedia-followup-destination-visible"
    ),
    followup_subgoal_id=(
        "live-web-wikipedia-open-followup"
    ),
    followup_claim_text_factory=(
        _wikipedia_followup_claim_text
    ),
    followup_subgoal_text_factory=(
        _wikipedia_followup_subgoal_text
    ),
    followup_step_goal_factory=(
        _wikipedia_followup_step_goal
    ),
)

WORKFLOWS = (
    PYTHON_WORKFLOW,
    WIKIPEDIA_WORKFLOW,
)

SEARCH_FIELD = PYTHON_WORKFLOW.search_field
GO_BUTTON = PYTHON_WORKFLOW.submit_target
RESULTS_TARGET = PYTHON_WORKFLOW.result_target
QUERY_CLAIM_ID = PYTHON_WORKFLOW.query_claim_id
QUERY_SUBGOAL_ID = PYTHON_WORKFLOW.query_subgoal_id
RESULT_CLAIM_ID = PYTHON_WORKFLOW.result_claim_id
RESULT_SUBGOAL_ID = PYTHON_WORKFLOW.result_subgoal_id
SUBMIT_SIDE_EFFECT_ID = (
    PYTHON_WORKFLOW.submit_side_effect_id
)
SUBMIT_ACTION_KEY = PYTHON_WORKFLOW.submit_action_key


class LiveWebEnvironment:
    """Production Chrome perception and execution stack."""

    def __init__(
        self,
        *,
        capture_path: str | Path,
        stabilization_seconds: float = 0.6,
    ) -> None:
        self.capture_path = Path(
            capture_path
        )
        self.stabilization_seconds = (
            stabilization_seconds
        )

        self.controller = (
            ComputerController()
        )

        self.accessibility = (
            MacOSAccessibility()
        )

        self.perception_engine = (
            PerceptionEngine(
                screen_capture=ScreenCapture(
                    self.controller
                ),
                accessibility_reader=(
                    self.accessibility
                ),
                ocr=TesseractOCR(
                    minimum_confidence=0.05,
                    page_segmentation_mode=6,
                    group_words_by_line=True,
                ),
                fusion=UIElementFusion(),
                capture_path=(
                    self.capture_path
                ),
            )
        )

        self.executor = ToolExecutor(
            ToolRegistry(
                create_computer_tools(
                    self.controller
                )
            )
        )
        self.last_observation_timings: dict[str, float] = {}

    def frontmost_application_name(self) -> str | None:
        """Return the current frontmost app without a full perception pass."""
        return (
            self.accessibility
            .read_frontmost_application_name()
        )

    def observe(
        self,
    ) -> TextInputObservation:
        observe_started = time.perf_counter()
        timings: dict[str, float] = {}
        stabilization_started = time.perf_counter()
        time.sleep(
            self.stabilization_seconds
        )
        timings["stabilization"] = (
            time.perf_counter() - stabilization_started
        )

        self.capture_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        perception_started = time.perf_counter()
        snapshot = (
            self.perception_engine.observe()
        )
        timings["perception_engine"] = (
            time.perf_counter() - perception_started
        )
        timings.update(snapshot.timings)

        application_started = time.perf_counter()
        application_name = (
            self.accessibility
            .read_frontmost_application_name()
        )
        timings["frontmost_application"] = (
            time.perf_counter() - application_started
        )

        viewport_started = time.perf_counter()
        viewport = (
            self.accessibility
            .read_frontmost_viewport()
        )
        timings["viewport"] = time.perf_counter() - viewport_started

        semantic_started = time.perf_counter()
        semantic_elements = tuple(
            self.accessibility
            .read_frontmost_semantic_elements()
        )
        timings["accessibility_semantic_elements"] = (
            time.perf_counter() - semantic_started
        )
        timings["observe_total"] = time.perf_counter() - observe_started
        self.last_observation_timings = timings

        return TextInputObservation(
            application_name=application_name,
            viewport=viewport,
            snapshot=snapshot,
            semantic_elements=semantic_elements,
        )

    def open_task_site(
        self,
        start_url: str,
        task_marker: str,
    ) -> str:
        marker_url = (
            self.controller
            .open_task_chrome_window(
                start_url,
                task_marker,
            )
        )

        time.sleep(1.0)

        return marker_url

    def open_python_org(
        self,
        task_marker: str,
    ) -> str:
        return self.open_task_site(
            PYTHON_URL,
            task_marker,
        )

    def activate_task_chrome_window(
        self,
        marker_url: str,
        working_url_prefix: str = PYTHON_URL,
    ) -> None:
        self.controller.activate_task_chrome_window(
            marker_url,
            working_url_prefix,
        )

        time.sleep(0.8)

    def activate_chrome(
        self,
    ) -> None:
        result = self.executor.execute(
            Action(
                tool_name="activate_app",
                arguments={
                    "app_name": EXPECTED_APP,
                },
                reason=(
                    "Return to the browser "
                    "before restart recovery."
                ),
            )
        )

        _require_tool_success(
            result,
            "activate Chrome",
        )

        time.sleep(0.8)


def build_prepare_plan(
    goal: str,
    resolved_task: (
        ResolvedDurableWebSearchTask
        | DurableWebSearchSpec
        | None
    ) = None,
) -> StructuredPlan:
    """Build the first real-browser segment."""
    resolved_task = _coerce_resolved_task(
        goal,
        resolved_task,
    )
    spec = resolved_task.spec

    return StructuredPlan(
        task_goal=goal,
        steps=(
            WebTextInputStep(
                goal=spec.prepare_step_goal,
                target=spec.search_field,
                input_text=resolved_task.query_text,
                max_attempts=1,
            ),
        ),
    )


def build_resume_plan(
    goal: str,
    resolved_task: (
        ResolvedDurableWebSearchTask
        | DurableWebSearchSpec
        | None
    ) = None,
) -> StructuredPlan:
    """Build the post-restart browser segment."""
    resolved_task = _coerce_resolved_task(
        goal,
        resolved_task,
    )
    spec = resolved_task.spec

    return StructuredPlan(
        task_goal=goal,
        steps=(
            PlanStep(
                goal=spec.submit_step_goal,
                operation=(
                    PlanOperation.CLICK_TARGET
                ),
                action_target=spec.submit_target,
                verification_target=(
                    spec.result_target
                ),
                max_attempts=1,
            ),
        ),
    )


def build_followup_plan(
    goal: str,
    resolved_task: ResolvedDurableWebSearchTask | None = None,
    *,
    action_target: TargetSpec | None = None,
) -> StructuredPlan:
    """Build the bounded follow-up navigation segment."""
    resolved_task = _coerce_resolved_task(
        goal,
        resolved_task,
    )
    if resolved_task.followup_target_text is None:
        raise RuntimeError(
            "Follow-up plan requires a follow-up target."
        )

    target_text = resolved_task.followup_target_text
    spec = resolved_task.spec
    step_goal = (
        spec.followup_step_goal_factory(target_text)
        if spec.followup_step_goal_factory is not None
        else f"Open the requested link {target_text!r}."
    )

    return StructuredPlan(
        task_goal=goal,
        steps=(
            PlanStep(
                goal=step_goal,
                operation=(
                    PlanOperation.CLICK_TARGET
                ),
                action_target=(
                    action_target
                    if action_target is not None
                    else _followup_link_target(
                        resolved_task
                    )
                ),
                verification_target=(
                    _followup_destination_target(
                        resolved_task
                    )
                ),
                max_attempts=1,
            ),
        ),
    )


def compile_durable_task_plan(
    resolved_task: ResolvedDurableWebSearchTask,
) -> DurableTaskPlan:
    """Compile a resolved bounded goal into a persisted durable plan."""
    spec = resolved_task.spec
    steps: list[DurableTaskStep] = [
        DurableTaskStep(
            step_id="enter-query",
            kind=DurableStepKind.ENTER_TEXT,
            description=spec.prepare_step_goal,
            action_target=spec.search_field,
            input_text=resolved_task.query_text,
            verification_target=None,
            postcondition=(
                DurablePostconditionKind.QUERY_MATCHES
            ),
            claim_id=spec.query_claim_id,
            claim_text=spec.query_claim_text,
            subgoal_id=spec.query_subgoal_id,
            subgoal_text=spec.query_subgoal_text,
        ),
        DurableTaskStep(
            step_id="submit-search",
            kind=DurableStepKind.ACTIVATE_CONTROL,
            description=spec.submit_step_goal,
            action_target=spec.submit_target,
            input_text=None,
            verification_target=spec.result_target,
            postcondition=(
                DurablePostconditionKind.SEARCH_OUTCOME_VISIBLE
            ),
            claim_id=spec.result_claim_id,
            claim_text=spec.result_claim_text,
            subgoal_id=spec.result_subgoal_id,
            subgoal_text=spec.result_subgoal_text,
            side_effect_id=spec.submit_side_effect_id,
            side_effect_action_key=spec.submit_action_key,
            side_effect_description=spec.submit_description,
        ),
    ]

    if resolved_task.followup_target_text is not None:
        steps.append(
            DurableTaskStep(
                step_id="open-followup-link",
                kind=DurableStepKind.OPEN_LINK,
                description=(
                    _followup_subgoal_text(
                        resolved_task
                    )
                ),
                action_target=(
                    _followup_link_target(
                        resolved_task
                    )
                ),
                input_text=None,
                verification_target=(
                    _followup_destination_target(
                        resolved_task
                    )
                ),
                postcondition=(
                    DurablePostconditionKind
                    .DESTINATION_HEADING_VISIBLE
                ),
                claim_id=(
                    _followup_claim_id(
                        resolved_task
                    )
                ),
                claim_text=(
                    _followup_claim_text(
                        resolved_task
                    )
                ),
                subgoal_id=(
                    _followup_subgoal_id(
                        resolved_task
                    )
                ),
                subgoal_text=(
                    _followup_subgoal_text(
                        resolved_task
                    )
                ),
                side_effect_id=(
                    FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
                ),
                side_effect_action_key=(
                    FOLLOWUP_NAVIGATION_ACTION_KEY
                ),
                side_effect_description=(
                    _followup_side_effect_description(
                        resolved_task
                    )
                ),
            )
        )

    return DurableTaskPlan(
        plan_id=(
            f"live-web:{spec.workflow_id}:"
            f"v{DURABLE_PLAN_VERSION}"
        ),
        workflow_id=spec.workflow_id,
        version=DURABLE_PLAN_VERSION,
        steps=tuple(steps),
    )


def durable_plan_canonical_json(
    plan: DurableTaskPlan,
) -> str:
    """Return a deterministic JSON durable plan representation."""
    return json.dumps(
        _durable_plan_payload(plan),
        sort_keys=True,
        separators=(",", ":"),
    )


def durable_planner_provenance_canonical_json(
    provenance: DurablePlannerProvenance,
) -> str:
    """Return deterministic JSON durable-planner provenance."""
    return json.dumps(
        {
            "version": DURABLE_PLANNER_PROVENANCE_VERSION,
            "planner_type": provenance.planner_type,
            "model_identifier": provenance.model_identifier,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def durable_plan_from_canonical_json(
    value: str,
) -> DurableTaskPlan:
    """Load a persisted canonical durable task plan."""
    try:
        payload = json.loads(
            value
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Persisted durable task plan is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Persisted durable task plan must be a JSON object."
        )

    expected_keys = {
        "plan_id",
        "version",
        "workflow_id",
        "steps",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "Persisted durable task plan contains unsupported fields."
        )
    if payload["version"] != DURABLE_PLAN_VERSION:
        raise RuntimeError(
            "Persisted durable task plan version is unsupported."
        )
    plan_id = _require_persisted_string(
        payload["plan_id"],
        "plan_id",
    )
    workflow_id = _require_persisted_string(
        payload["workflow_id"],
        "workflow_id",
    )
    steps_payload = payload["steps"]
    if not isinstance(steps_payload, list) or not steps_payload:
        raise RuntimeError(
            "Persisted durable task plan must contain steps."
        )

    plan = DurableTaskPlan(
        plan_id=plan_id,
        workflow_id=workflow_id,
        version=DURABLE_PLAN_VERSION,
        steps=tuple(
            _durable_step_from_payload(step_payload)
            for step_payload in steps_payload
        ),
    )
    if durable_plan_canonical_json(plan) != value:
        raise RuntimeError(
            "Persisted durable task plan is not canonical."
        )
    return plan


def _durable_step_from_payload(
    payload: object,
) -> DurableTaskStep:
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Persisted durable task plan contains an invalid step."
        )
    expected_keys = {
        "step_id",
        "kind",
        "description",
        "postcondition",
        "claim_id",
        "claim_text",
        "subgoal_id",
        "subgoal_text",
        "action_target",
        "verification_target",
        "input_text",
        "side_effect_id",
        "side_effect_action_key",
        "side_effect_description",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "Persisted durable task plan contains an invalid step."
        )

    try:
        kind = DurableStepKind(
            payload["kind"]
        )
        postcondition = DurablePostconditionKind(
            payload["postcondition"]
        )
    except ValueError as exc:
        raise RuntimeError(
            "Persisted durable task plan contains an invalid step."
        ) from exc

    return DurableTaskStep(
        step_id=_require_persisted_string(
            payload["step_id"],
            "step_id",
        ),
        kind=kind,
        description=_require_persisted_string(
            payload["description"],
            "description",
        ),
        postcondition=postcondition,
        claim_id=_require_persisted_string(
            payload["claim_id"],
            "claim_id",
        ),
        claim_text=_require_persisted_string(
            payload["claim_text"],
            "claim_text",
        ),
        subgoal_id=_require_persisted_string(
            payload["subgoal_id"],
            "subgoal_id",
        ),
        subgoal_text=_require_persisted_string(
            payload["subgoal_text"],
            "subgoal_text",
        ),
        action_target=_target_spec_from_payload(
            payload["action_target"]
        ),
        verification_target=_target_spec_from_payload(
            payload["verification_target"]
        ),
        input_text=_optional_persisted_string(
            payload["input_text"],
            "input_text",
        ),
        side_effect_id=_optional_persisted_string(
            payload["side_effect_id"],
            "side_effect_id",
        ),
        side_effect_action_key=_optional_persisted_string(
            payload["side_effect_action_key"],
            "side_effect_action_key",
        ),
        side_effect_description=_optional_persisted_string(
            payload["side_effect_description"],
            "side_effect_description",
        ),
    )


def _target_spec_from_payload(
    payload: object,
) -> TargetSpec | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Persisted durable task plan contains an invalid target."
        )
    expected_keys = {
        "text",
        "identifier",
        "element_types",
        "minimum_confidence",
        "reference_point",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "Persisted durable task plan contains an invalid target."
        )

    element_types = payload["element_types"]
    if not isinstance(element_types, list) or not all(
        isinstance(item, str)
        for item in element_types
    ):
        raise RuntimeError(
            "Persisted durable task plan contains an invalid target."
        )
    reference_point = payload["reference_point"]
    if reference_point is not None:
        if (
            not isinstance(reference_point, list)
            or len(reference_point) != 2
            or not all(
                isinstance(item, (int, float))
                for item in reference_point
            )
        ):
            raise RuntimeError(
                "Persisted durable task plan contains an invalid target."
            )
        reference_point_value = (
            float(reference_point[0]),
            float(reference_point[1]),
        )
    else:
        reference_point_value = None

    return TargetSpec(
        text=_optional_persisted_string(
            payload["text"],
            "target.text",
        ),
        identifier=_optional_persisted_string(
            payload["identifier"],
            "target.identifier",
        ),
        element_types=tuple(element_types),
        minimum_confidence=_require_persisted_float(
            payload["minimum_confidence"],
            "target.minimum_confidence",
        ),
        reference_point=reference_point_value,
    )


def _durable_plan_payload(
    plan: DurableTaskPlan,
) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "version": plan.version,
        "workflow_id": plan.workflow_id,
        "steps": [
            _durable_step_payload(step)
            for step in plan.steps
        ],
    }


def _durable_step_payload(
    step: DurableTaskStep,
) -> dict[str, object]:
    return {
        "step_id": step.step_id,
        "kind": step.kind.value,
        "description": step.description,
        "postcondition": step.postcondition.value,
        "claim_id": step.claim_id,
        "claim_text": step.claim_text,
        "subgoal_id": step.subgoal_id,
        "subgoal_text": step.subgoal_text,
        "action_target": _target_spec_payload(
            step.action_target
        ),
        "verification_target": _target_spec_payload(
            step.verification_target
        ),
        "input_text": step.input_text,
        "side_effect_id": step.side_effect_id,
        "side_effect_action_key": (
            step.side_effect_action_key
        ),
        "side_effect_description": (
            step.side_effect_description
        ),
    }


def _target_spec_payload(
    target: TargetSpec | None,
) -> dict[str, object] | None:
    if target is None:
        return None
    return {
        "text": target.text,
        "identifier": target.identifier,
        "element_types": list(target.element_types),
        "minimum_confidence": target.minimum_confidence,
        "reference_point": (
            list(target.reference_point)
            if target.reference_point is not None
            else None
        ),
    }


def _require_persisted_string(
    value: object,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"Persisted durable task plan field {field_name} "
            "must be a non-empty string."
        )
    return value


def _optional_persisted_string(
    value: object,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(
            f"Persisted durable task plan field {field_name} "
            "must be a string or null."
        )
    return value


def _require_persisted_float(
    value: object,
    field_name: str,
) -> float:
    if not isinstance(value, (int, float)):
        raise RuntimeError(
            f"Persisted durable task plan field {field_name} "
            "must be numeric."
        )
    return float(value)


def goal_is_supported(
    goal: str,
) -> bool:
    """Return whether the bounded live worker supports this task."""
    return (
        resolve_live_web_task(
            goal,
            raise_on_unsupported=False,
        )
        is not None
    )


def _coerce_resolved_task(
    goal: str,
    resolved_task: (
        ResolvedDurableWebSearchTask
        | DurableWebSearchSpec
        | None
    ),
) -> ResolvedDurableWebSearchTask:
    parsed = resolve_live_web_task(
        goal
    )

    if resolved_task is None:
        return parsed

    if isinstance(
        resolved_task,
        ResolvedDurableWebSearchTask,
    ):
        if resolved_task != parsed:
            raise RuntimeError(
                "Resolved task does not match the goal."
            )
        return resolved_task

    if resolved_task is parsed.spec:
        return parsed

    raise RuntimeError(
        "Workflow spec does not match the goal."
    )


_SEARCH_GOAL_RE = re.compile(
    r"^\s*search\s+(?P<site>wikipedia|python\.org)"
    r"\s+for\s+(?P<query>.*?)"
    r"(?:\s+and\s+open\s+(?P<target>.*?))?"
    r"\s*$",
    re.IGNORECASE,
)


def parse_live_web_search_goal(
    goal: str,
) -> tuple[str, str, str | None]:
    """Parse the deterministic live search command grammar."""
    if not isinstance(goal, str):
        raise RuntimeError(
            "Live web search goal must be a string."
        )

    match = _SEARCH_GOAL_RE.match(goal)
    if match is None:
        site_match = re.match(
            r"^\s*search\s+(?P<site>\S+)",
            goal,
            re.IGNORECASE,
        )
        if site_match is not None:
            site = site_match.group("site")
            if site.lower() not in {
                "wikipedia",
                "python.org",
            }:
                raise RuntimeError(
                    "Unsupported live web search site "
                    f"{site!r}; supported sites are "
                    "python.org and Wikipedia."
                )

        raise RuntimeError(
            "Malformed live web search goal. Use "
            "'Search python.org for <query>.', "
            "'Search Wikipedia for <query>.', or "
            "'Search Wikipedia for <query> and open <target>.'"
        )

    site = match.group("site").lower()
    query_text = match.group("query").strip()
    target_text = (
        match.group("target").strip()
        if match.group("target") is not None
        else None
    )

    if target_text is None and query_text.endswith("."):
        query_text = query_text[:-1].rstrip()
    elif target_text is not None and target_text.endswith("."):
        target_text = target_text[:-1].rstrip()

    if not query_text:
        raise RuntimeError(
            "Live web search query must be non-empty."
        )

    if (
        target_text is None
        and re.search(
            r"\band\s+open\s*$",
            query_text,
            re.IGNORECASE,
        )
    ):
        raise RuntimeError(
            "Live web follow-up target must be non-empty."
        )

    if target_text is not None and not target_text:
        raise RuntimeError(
            "Live web follow-up target must be non-empty."
        )

    if target_text is not None and site != "wikipedia":
        raise RuntimeError(
            "Follow-up navigation is supported only "
            "for Wikipedia in this experiment."
        )

    return site, query_text, target_text


def resolve_live_web_task(
    goal: str,
    *,
    raise_on_unsupported: bool = True,
) -> ResolvedDurableWebSearchTask | None:
    """Resolve a bounded user goal to a workflow and runtime query."""
    try:
        (
            site,
            query_text,
            followup_target_text,
        ) = parse_live_web_search_goal(
            goal
        )
    except RuntimeError:
        if not raise_on_unsupported:
            return None
        raise

    if site == "python.org":
        return ResolvedDurableWebSearchTask(
            spec=PYTHON_WORKFLOW,
            query_text=query_text,
            followup_target_text=(
                followup_target_text
            ),
        )

    if site == "wikipedia":
        return ResolvedDurableWebSearchTask(
            spec=WIKIPEDIA_WORKFLOW,
            query_text=query_text,
            followup_target_text=(
                followup_target_text
            ),
        )

    if not raise_on_unsupported:
        return None

    raise RuntimeError(
        "Unsupported live web search site "
        f"{site!r}; supported sites are "
        "python.org and Wikipedia."
    )


def resolve_live_web_workflow(
    goal: str,
    *,
    raise_on_unsupported: bool = True,
) -> DurableWebSearchSpec | None:
    """Resolve a bounded user goal to its static web workflow."""
    resolved_task = resolve_live_web_task(
        goal,
        raise_on_unsupported=raise_on_unsupported,
    )
    if resolved_task is None:
        return None
    return resolved_task.spec


def _configured_durable_planner_mode() -> DurablePlannerMode:
    raw_value = os.environ.get(
        DURABLE_PLANNER_ENV_VAR,
        DurablePlannerMode.DETERMINISTIC.value,
    )
    try:
        return DurablePlannerMode(
            raw_value.strip().lower()
        )
    except ValueError as exc:
        supported = ", ".join(
            mode.value
            for mode in DurablePlannerMode
        )
        raise RuntimeError(
            f"Unsupported {DURABLE_PLANNER_ENV_VAR} value "
            f"{raw_value!r}; supported values: {supported}"
        ) from exc


def _create_durable_planner(
    mode: DurablePlannerMode,
) -> DurablePlanner:
    if mode is DurablePlannerMode.DETERMINISTIC:
        return DeterministicDurablePlanner()
    if mode is DurablePlannerMode.LLM:
        return LLMDurablePlanner(
            client=OpenAILLMClient()
        )
    raise RuntimeError(
        f"Unsupported durable planner mode: {mode}"
    )


def _planner_provenance_for(
    planner: DurablePlanner,
    mode: DurablePlannerMode,
) -> DurablePlannerProvenance:
    planner_type = getattr(
        planner,
        "planner_type",
        mode.value,
    )
    if not isinstance(planner_type, str) or not planner_type.strip():
        planner_type = mode.value
    model_identifier = getattr(
        planner,
        "model_identifier",
        None,
    )
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        model_identifier = None
    return DurablePlannerProvenance(
        planner_type=planner_type,
        model_identifier=model_identifier,
    )


def _select_durable_plan(
    *,
    goal: str,
    state: TaskState,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_planner: DurablePlanner | None,
) -> tuple[DurableTaskPlan, DurablePlannerProvenance | None]:
    expected_plan = compile_durable_task_plan(
        resolved_task
    )
    artifact = state.artifacts.get(
        DURABLE_PLAN_ARTIFACT_ID
    )
    if artifact is not None:
        persisted_plan = durable_plan_from_canonical_json(
            artifact.location
        )
        if durable_plan_canonical_json(
            persisted_plan
        ) != durable_plan_canonical_json(
            expected_plan
        ):
            raise RuntimeError(
                "Persisted durable task plan does not "
                "match the resolved goal."
            )
        return persisted_plan, _read_durable_planner_provenance(
            state,
            allow_missing=True,
        )

    mode = _configured_durable_planner_mode()
    planner = (
        durable_planner
        if durable_planner is not None
        else _create_durable_planner(mode)
    )
    provenance = _planner_provenance_for(
        planner,
        mode,
    )
    context = default_durable_planning_context()
    try:
        durable_plan = planner.plan(
            goal,
            context,
        )
    except Exception as exc:
        raise RuntimeError(
            "Durable planning failed before browser action."
        ) from exc

    if durable_plan_canonical_json(
        durable_plan
    ) != durable_plan_canonical_json(
        expected_plan
    ):
        raise RuntimeError(
            "Accepted durable plan does not match "
            "the resolved goal."
        )
    return durable_plan, provenance


def create_live_web_worker(
    state: TaskState,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    *,
    capture_path: str | Path | None = None,
    environment_factory=LiveWebEnvironment,
    durable_planner: DurablePlanner | None = None,
) -> RuntimeWorker:
    """Create the first persistent real-browser workspace worker."""
    if not isinstance(
        state,
        TaskState,
    ):
        raise ValueError(
            "state must be a TaskState"
        )

    if not callable(
        publish_state
    ):
        raise ValueError(
            "publish_state must be callable"
        )

    if not callable(
        publish_decision
    ):
        raise ValueError(
            "publish_decision must be callable"
        )

    resolved_task = resolve_live_web_task(
        state.goal
    )
    durable_plan, planner_provenance = _select_durable_plan(
        goal=state.goal,
        state=state,
        resolved_task=resolved_task,
        durable_planner=durable_planner,
    )

    _configured_crash_point()

    if capture_path is None:
        capture_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Computer Agent"
            / "live"
            / "current.png"
        )

    def worker(
        task: RuntimeTask,
        control: RuntimeControl,
        progress: Callable[[str], None],
    ) -> None:
        del task

        transitions = (
            TaskStateTransitions(
                state
            )
        )

        _ensure_workflow_identity(
            transitions,
            resolved_task.spec,
        )

        _ensure_query_identity(
            transitions,
            resolved_task,
        )

        _ensure_followup_identity(
            transitions,
            resolved_task,
        )

        _ensure_durable_planner_provenance(
            transitions,
            planner_provenance,
        )

        _ensure_durable_plan_identity(
            transitions,
            resolved_task,
            durable_plan,
        )

        _ensure_live_task_structure(
            transitions,
            durable_plan,
        )

        state.status = (
            TaskStateStatus.RUNNING
        )

        if not state.constraints:
            state.constraints = (
                (
                    "Use fresh UI observations "
                    "after restart."
                ),
                (
                    "Do not reuse pre-restart "
                    "screen coordinates."
                ),
                (
                    "Do not declare completion "
                    "without fresh result evidence."
                ),
                (
                    "Do not declare follow-up completion "
                    "without fresh destination evidence."
                ),
            )

        state.touch()
        publish_state()

        environment = (
            environment_factory(
                capture_path=capture_path
            )
        )

        _run_live_task(
            state=state,
            transitions=transitions,
            environment=environment,
            resolved_task=resolved_task,
            durable_plan=durable_plan,
            control=control,
            publish_state=publish_state,
            publish_decision=publish_decision,
            progress=progress,
        )

    return worker


def _configured_crash_point() -> LiveCrashPoint | None:
    raw_value = os.environ.get(
        CRASH_ENV_VAR
    )

    if raw_value is None or not raw_value.strip():
        return None

    try:
        return LiveCrashPoint(
            raw_value.strip()
        )
    except ValueError as exc:
        supported = ", ".join(
            point.value
            for point in LiveCrashPoint
        )
        raise RuntimeError(
            f"Unsupported {CRASH_ENV_VAR} value "
            f"{raw_value!r}; supported values: "
            f"{supported}"
        ) from exc


def _maybe_inject_process_crash(
    point: LiveCrashPoint,
) -> None:
    configured = _configured_crash_point()

    if configured is not point:
        return

    print(
        "[crash-injection] terminating after "
        f"{point.value}",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    _terminate_process_for_test()


def _terminate_process_for_test() -> NoReturn:
    os._exit(
        CRASH_EXIT_CODE
    )


def _run_live_task(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_plan: DurableTaskPlan,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    """Converge the durable task state and live browser state."""
    spec = resolved_task.spec
    control.checkpoint()

    progress(
        f"Plan ready: {len(durable_plan.steps)} durable step"
        f"{'s' if len(durable_plan.steps) != 1 else ''}."
    )

    observation = _ensure_browser_workspace(
        state=state,
        transitions=transitions,
        environment=environment,
        spec=spec,
        publish_state=publish_state,
        progress=progress,
    )

    final_observation = _reconcile_durable_plan(
        state=state,
        transitions=transitions,
        environment=environment,
        resolved_task=resolved_task,
        durable_plan=durable_plan,
        control=control,
        publish_state=publish_state,
        publish_decision=publish_decision,
        progress=progress,
        observation=observation,
    )

    if not transitions.can_complete():
        raise RuntimeError(
            "The live task cannot complete: "
            + "; ".join(
                transitions.completion_blockers()
            )
        )

    transitions.complete_task()

    publish_state()

    _publish_live_decision(
        publish_decision,
        observation=final_observation,
        decision_type="COMPLETE",
        operation=None,
        target_text=None,
        expected_effect=None,
        reason=(
            "Fresh live evidence verified "
            "all semantic completion requirements."
        ),
        completion_summary=(
            _completion_summary(
                resolved_task
            )
        ),
    )

    progress(
        "Real browser task completed."
    )


def _reconcile_durable_plan(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_plan: DurableTaskPlan,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    """Reconcile one ordered durable plan from fresh browser state."""
    observation = _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=resolved_task.spec,
        observation=observation,
        refresh=False,
        progress=progress,
    )

    observation, next_index = (
        _reconcile_most_advanced_satisfied_step(
            transitions=transitions,
            resolved_task=resolved_task,
            durable_plan=durable_plan,
            observation=observation,
            publish_state=publish_state,
            progress=progress,
        )
    )

    for index in range(
        next_index,
        len(durable_plan.steps),
    ):
        step = durable_plan.steps[index]
        observation = _reconcile_durable_step(
            state=state,
            transitions=transitions,
            environment=environment,
            resolved_task=resolved_task,
            durable_plan=durable_plan,
            step=step,
            step_index=index,
            control=control,
            publish_state=publish_state,
            publish_decision=publish_decision,
            progress=progress,
            observation=observation,
        )

    return observation


def _reconcile_most_advanced_satisfied_step(
    *,
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_plan: DurableTaskPlan,
    observation: TextInputObservation,
    publish_state: TaskStatePublisher,
    progress: Callable[[str], None],
) -> tuple[TextInputObservation, int]:
    for index in range(
        len(durable_plan.steps) - 1,
        -1,
        -1,
    ):
        step = durable_plan.steps[index]
        postcondition = _step_postcondition_satisfied(
            step,
            observation,
            resolved_task,
            state=transitions.state,
        )
        if postcondition is None:
            continue

        if not _prior_steps_have_durable_history(
            transitions.state,
            durable_plan,
            index,
        ):
            continue

        _reconcile_prior_steps_from_history(
            transitions,
            durable_plan,
            index,
            observation_summary=(
                "Fresh advanced browser state reconciles "
                "previous durable plan history after restart."
            ),
        )
        evidence = _verify_step_from_postcondition(
            transitions,
            step,
            postcondition,
        )
        _confirm_step_side_effect(
            transitions,
            resolved_task,
            step,
            evidence.evidence_id,
        )
        publish_state()
        progress(
            "Fresh browser state verifies durable "
            f"plan step {step.step_id!r}."
        )
        return observation, index + 1

    return observation, 0


def _reconcile_durable_step(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_plan: DurableTaskPlan,
    step: DurableTaskStep,
    step_index: int,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    progress(
        f"Step {step_index + 1}/{len(durable_plan.steps)} — "
        f"{step.description}: checking browser state."
    )

    observation = _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=resolved_task.spec,
        observation=observation,
        refresh=False,
        progress=progress,
    )

    postcondition = _step_postcondition_satisfied(
        step,
        observation,
        resolved_task,
        state=state,
    )
    if postcondition is not None:
        evidence = _verify_step_from_postcondition(
            transitions,
            step,
            postcondition,
        )
        _confirm_step_side_effect(
            transitions,
            resolved_task,
            step,
            evidence.evidence_id,
        )
        publish_state()
        progress(
            f"Step {step_index + 1}/{len(durable_plan.steps)} — "
            f"{step.description}: VERIFIED from fresh browser state."
        )
        return observation

    _require_prior_steps_verified(
        transitions.state,
        durable_plan,
        step_index,
    )

    grounded_action_target = None
    observation = _ensure_step_precondition_with_fresh_reads(
        state=state,
        environment=environment,
        resolved_task=resolved_task,
        step=step,
        observation=observation,
        progress=progress,
    )

    if step.kind is DurableStepKind.OPEN_LINK:
        readiness = (
            _ground_followup_link_with_readiness(
                state=state,
                transitions=transitions,
                environment=environment,
                resolved_task=resolved_task,
                observation=observation,
                publish_state=publish_state,
                progress=progress,
            )
        )
        observation = readiness.observation
        if readiness.already_at_destination:
            return observation
        if readiness.grounding is None:
            raise RuntimeError(
                "Follow-up readiness finished without "
                "a grounded link or destination evidence."
            )
        _persist_followup_destination_identity(
            transitions,
            resolved_task,
            readiness.grounding,
        )
        publish_state()
        if readiness.grounding.element is not None:
            grounded_action_target = TargetSpec(
                text=_followup_target_text(
                    resolved_task
                ),
                element_types=("link",),
                minimum_confidence=0.70,
                reference_point=(
                    readiness.grounding.element.center
                ),
            )

    side_effect = _ensure_step_side_effect(
        transitions,
        resolved_task,
        step,
    )
    if side_effect is not None:
        publish_state()

    _publish_step_decision(
        publish_decision,
        observation=observation,
        resolved_task=resolved_task,
        step=step,
    )

    _mark_step_execution_attempt(
        transitions,
        resolved_task,
        step,
    )
    if side_effect is not None:
        publish_state()

    control.checkpoint()

    progress(
        f"Step {step_index + 1}/{len(durable_plan.steps)} — "
        + _step_execution_progress(step)
    )

    result = _execute_agent_plan(
        environment,
        _structured_plan_for_durable_step(
            state.goal,
            resolved_task,
            step,
            observation=observation,
            action_target=grounded_action_target,
        ),
    )

    _maybe_inject_step_crash(
        step
    )

    progress(
        f"Step {step_index + 1}/{len(durable_plan.steps)} — "
        "verifying fresh post-action browser state."
    )

    refreshed = _ensure_expected_workspace_observation(
        state=state,
        environment=environment,
        spec=resolved_task.spec,
        observation=None,
        refresh=True,
        progress=progress,
    )

    postcondition = _step_postcondition_satisfied(
        step,
        refreshed,
        resolved_task,
        state=state,
    )
    for _ in range(2):
        if postcondition is not None:
            break
        progress(
            f"Step {step_index + 1}/{len(durable_plan.steps)} — "
            "fresh post-action state is not verified yet; "
            "re-observing before deciding outcome."
        )
        refreshed = _ensure_expected_workspace_observation(
            state=state,
            environment=environment,
            spec=resolved_task.spec,
            observation=None,
            refresh=True,
            progress=progress,
        )
        postcondition = _step_postcondition_satisfied(
            step,
            refreshed,
            resolved_task,
            state=state,
        )
    if postcondition is None:
        _mark_step_outcome_unknown(
            transitions,
            resolved_task,
            step,
        )
        publish_state()
        _require_agent_success(
            result,
            _step_failure_label(step),
        )
        raise RuntimeError(
            _missing_postcondition_message(
                step
            )
        )

    evidence = _verify_step_from_postcondition(
        transitions,
        step,
        postcondition,
    )
    _confirm_step_side_effect(
        transitions,
        resolved_task,
        step,
        evidence.evidence_id,
    )
    publish_state()

    progress(
        f"Step {step_index + 1}/{len(durable_plan.steps)} — "
        f"{step.description}: VERIFIED."
    )

    _maybe_inject_checkpoint_crash(
        step
    )

    return refreshed


def _step_postcondition_satisfied(
    step: DurableTaskStep,
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    *,
    state: TaskState | None = None,
) -> StepPostconditionResult | None:
    if (
        step.postcondition
        is DurablePostconditionKind.QUERY_MATCHES
    ):
        result = _query_condition_satisfied(
            observation,
            resolved_task,
        )
        if result is None:
            return None
        return StepPostconditionResult(
            summary=result.summary,
            source=result.source,
        )

    if (
        step.postcondition
        is DurablePostconditionKind.SEARCH_OUTCOME_VISIBLE
    ):
        if not _results_visible(
            observation,
            resolved_task,
        ):
            return None
        return StepPostconditionResult(
            summary=(
                "Fresh Chrome Accessibility shows "
                "the configured search outcome."
            ),
            source=(
                "Live Chrome Accessibility observation"
            ),
        )

    if (
        step.postcondition
        is DurablePostconditionKind.DESTINATION_HEADING_VISIBLE
    ):
        return _followup_destination_postcondition(
            observation,
            resolved_task,
            state=state,
        )

    return None


def _verify_step_from_postcondition(
    transitions: TaskStateTransitions,
    step: DurableTaskStep,
    postcondition: StepPostconditionResult,
) -> EvidenceRecord:
    evidence = EvidenceRecord(
        summary=postcondition.summary,
        source=postcondition.source,
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(
        evidence
    )
    transitions.verify_claim(
        step.claim_id,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        step.subgoal_id
    )
    return evidence


def _prior_steps_have_durable_history(
    state: TaskState,
    durable_plan: DurableTaskPlan,
    step_index: int,
) -> bool:
    for prior_step in durable_plan.steps[:step_index]:
        claim = state.claims.get(
            prior_step.claim_id
        )
        if claim is None or not claim.evidence_ids:
            return False
    return True


def _reconcile_prior_steps_from_history(
    transitions: TaskStateTransitions,
    durable_plan: DurableTaskPlan,
    step_index: int,
    *,
    observation_summary: str,
) -> None:
    for prior_step in durable_plan.steps[:step_index]:
        subgoal = transitions.state.subgoals.get(
            prior_step.subgoal_id
        )
        if (
            subgoal is not None
            and subgoal.status
            is SubgoalStatus.VERIFIED
        ):
            continue

        evidence = transitions.add_evidence(
            EvidenceRecord(
                summary=(
                    observation_summary
                    + f" Step {prior_step.step_id!r} "
                    "had persisted durable history."
                ),
                source=(
                    "Live Chrome observation plus "
                    "persisted durable plan history"
                ),
                kind=EvidenceKind.VERIFICATION,
            )
        )
        transitions.verify_claim(
            prior_step.claim_id,
            (evidence.evidence_id,),
        )
        transitions.verify_subgoal(
            prior_step.subgoal_id
        )


def _require_prior_steps_verified(
    state: TaskState,
    durable_plan: DurableTaskPlan,
    step_index: int,
) -> None:
    for prior_step in durable_plan.steps[:step_index]:
        subgoal = state.subgoals.get(
            prior_step.subgoal_id
        )
        if (
            subgoal is None
            or subgoal.status
            is not SubgoalStatus.VERIFIED
        ):
            raise RuntimeError(
                "Cannot execute durable step "
                f"{durable_plan.steps[step_index].step_id!r}; "
                "prior step is not verified: "
                f"{prior_step.step_id!r}."
            )


def _ensure_step_precondition(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> None:
    if _step_precondition_satisfied(
        observation,
        resolved_task,
        step,
    ):
        return

    raise RuntimeError(
        _missing_step_precondition_message(step)
    )


def _ensure_step_precondition_with_fresh_reads(
    *,
    state: TaskState,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
    observation: TextInputObservation,
    progress: Callable[[str], None],
) -> TextInputObservation:
    current = observation
    for attempt in range(3):
        if _step_precondition_satisfied(
            current,
            resolved_task,
            step,
        ):
            return current
        if attempt == 2:
            break

        progress(
            "Fresh browser state is not ready for "
            f"durable step {step.step_id!r}; "
            "re-observing before any action."
        )
        current = _ensure_expected_workspace_observation(
            state=state,
            environment=environment,
            spec=resolved_task.spec,
            observation=None,
            refresh=True,
            progress=progress,
        )

    raise RuntimeError(
        _missing_step_precondition_message(step)
    )


def _step_precondition_satisfied(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> bool:
    spec = resolved_task.spec
    if step.kind is DurableStepKind.ENTER_TEXT:
        return _search_field_available(
            observation,
            spec,
        )

    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return _pre_submit_query_state(
            observation,
            resolved_task,
        )

    if step.kind is DurableStepKind.OPEN_LINK:
        return _results_visible(
            observation,
            resolved_task,
        )

    return False


def _missing_step_precondition_message(
    step: DurableTaskStep,
) -> str:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return (
            "The live browser state is ambiguous: "
            "the durable query-entry target is not available."
        )
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return (
            "The live browser state is ambiguous: "
            "it is not verified pre-submit search state."
        )
    if step.kind is DurableStepKind.OPEN_LINK:
        return (
            "The live browser state is ambiguous: "
            "it is not verified search outcome state."
        )

    return f"Unsupported durable step kind: {step.kind}"


def _structured_plan_for_durable_step(
    goal: str,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
    *,
    observation: TextInputObservation | None = None,
    action_target: TargetSpec | None = None,
) -> StructuredPlan:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return build_prepare_plan(
            goal,
            resolved_task,
        )
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return build_resume_plan(
            goal,
            resolved_task,
        )
    if step.kind is DurableStepKind.OPEN_LINK:
        if observation is not None:
            if action_target is None:
                grounding = _ground_followup_link(
                    observation,
                    resolved_task,
                )
                if grounding.element is not None:
                    action_target = TargetSpec(
                        text=_followup_target_text(
                            resolved_task
                        ),
                        element_types=("link",),
                        minimum_confidence=0.70,
                        reference_point=(
                            grounding.element.center
                        ),
                    )
        return build_followup_plan(
            goal,
            resolved_task,
            action_target=action_target,
        )
    raise RuntimeError(
        f"Unsupported durable step kind: {step.kind}"
    )


def _ensure_step_side_effect(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> SideEffectRecord | None:
    if step.side_effect_id is None:
        return None
    if step.side_effect_id == resolved_task.spec.submit_side_effect_id:
        return _ensure_submit_side_effect(
            transitions,
            resolved_task.spec,
        )
    if step.side_effect_id == FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID:
        return _ensure_followup_side_effect(
            transitions,
            resolved_task,
        )
    raise RuntimeError(
        "Unsupported durable step side effect: "
        f"{step.side_effect_id}"
    )


def _mark_step_execution_attempt(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> None:
    if step.side_effect_id is None:
        return
    if step.side_effect_id == resolved_task.spec.submit_side_effect_id:
        _mark_submit_execution_attempt(
            transitions,
            resolved_task.spec,
        )
        return
    if step.side_effect_id == FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID:
        _mark_followup_execution_attempt(
            transitions,
            resolved_task,
        )
        return
    raise RuntimeError(
        "Unsupported durable step side effect: "
        f"{step.side_effect_id}"
    )


def _mark_step_outcome_unknown(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> None:
    if step.side_effect_id is None:
        return
    if step.side_effect_id == resolved_task.spec.submit_side_effect_id:
        _mark_submit_outcome_unknown(
            transitions,
            resolved_task.spec,
        )
        return
    if step.side_effect_id == FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID:
        _mark_followup_outcome_unknown(
            transitions,
        )
        return
    raise RuntimeError(
        "Unsupported durable step side effect: "
        f"{step.side_effect_id}"
    )


def _confirm_step_side_effect(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
    evidence_id: str,
) -> None:
    if step.side_effect_id is None:
        return
    if step.side_effect_id == resolved_task.spec.submit_side_effect_id:
        _confirm_submit_side_effect(
            transitions,
            resolved_task.spec,
            evidence_id,
        )
        return
    if step.side_effect_id == FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID:
        _confirm_followup_side_effect(
            transitions,
            resolved_task,
            evidence_id,
        )
        return
    raise RuntimeError(
        "Unsupported durable step side effect: "
        f"{step.side_effect_id}"
    )


def _publish_step_decision(
    publish_decision: AdaptiveDecisionPublisher,
    *,
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    step: DurableTaskStep,
) -> None:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return

    target_text = (
        step.action_target.text
        if step.action_target is not None
        else None
    )
    expected_effect = (
        "Satisfy durable step postcondition "
        f"{step.postcondition.value!r}."
    )
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        expected_effect = (
            "Submit the verified query and navigate "
            "to the configured search outcome."
        )
    elif step.kind is DurableStepKind.OPEN_LINK:
        expected_effect = (
            "Open the requested Wikipedia link "
            "and verify the destination heading."
        )

    _publish_live_decision(
        publish_decision,
        observation=observation,
        decision_type="ACTION",
        operation="click_target",
        target_text=target_text,
        expected_effect=expected_effect,
        reason=(
            "Fresh browser observation confirms "
            "the next unfinished durable plan step "
            f"{step.step_id!r} is actionable."
        ),
    )
    del resolved_task


def _maybe_inject_step_crash(
    step: DurableTaskStep,
) -> None:
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        _maybe_inject_process_crash(
            LiveCrashPoint.SUBMIT_EXECUTION
        )
    elif step.kind is DurableStepKind.OPEN_LINK:
        _maybe_inject_process_crash(
            LiveCrashPoint.FOLLOWUP_EXECUTION
        )


def _maybe_inject_checkpoint_crash(
    step: DurableTaskStep,
) -> None:
    if step.kind is DurableStepKind.ENTER_TEXT:
        _maybe_inject_process_crash(
            LiveCrashPoint.QUERY_CHECKPOINT
        )


def _step_execution_progress(
    step: DurableTaskStep,
) -> str:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return (
            "Typing the intended query into "
            "the freshly grounded search field."
        )
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return (
            "Submitting through newly grounded "
            "live UI coordinates."
        )
    if step.kind is DurableStepKind.OPEN_LINK:
        return (
            "Opening the requested Wikipedia link "
            "through newly grounded live UI coordinates."
        )
    return f"Executing durable step {step.step_id!r}."


def _step_failure_label(
    step: DurableTaskStep,
) -> str:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return "real-web query entry"
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return "real-web search submission"
    if step.kind is DurableStepKind.OPEN_LINK:
        return "real-web follow-up navigation"
    return f"durable step {step.step_id}"


def _missing_postcondition_message(
    step: DurableTaskStep,
) -> str:
    if step.kind is DurableStepKind.ENTER_TEXT:
        return (
            "The live search field did not uniquely "
            "verify the intended query."
        )
    if step.kind is DurableStepKind.ACTIVATE_CONTROL:
        return (
            "Fresh post-submit observation did not "
            "resolve the configured search outcome."
        )
    if step.kind is DurableStepKind.OPEN_LINK:
        return (
            "Fresh post-follow-up observation did "
            "not resolve the requested destination heading."
        )
    return (
        "Fresh post-action observation did not "
        f"resolve durable step {step.step_id!r}."
    )


def _ensure_expected_workspace_observation(
    *,
    state: TaskState,
    environment: LiveWebEnvironment,
    spec: DurableWebSearchSpec,
    observation: TextInputObservation | None,
    refresh: bool,
    progress: Callable[[str], None],
) -> TextInputObservation:
    """Return a valid observation from the exact marker-owned workspace."""
    probe = getattr(
        environment,
        "frontmost_application_name",
        None,
    )

    # If the environment exposes a cheap frontmost-app probe, use it to
    # reacquire before an expensive full observation. Unknown focus is not
    # treated as wrong focus.
    if callable(probe):
        frontmost = probe()

        if (
            frontmost is not None
            and frontmost != spec.expected_application
        ):
            progress(
                "Browser focus changed; re-acquiring the exact "
                "Agent-owned Chrome task window."
            )

            environment.activate_task_chrome_window(
                _browser_window_marker_from_state(
                    state,
                    spec,
                ),
                spec.working_url_prefix,
            )

            current = environment.observe()
            _publish_observation_timing(
                environment,
                progress,
            )

            _require_expected_app(
                current,
                spec,
            )

            return current

    # If we do not already have a usable observation, observe first.
    # Only reactivate Chrome when that fresh observation proves that the
    # wrong application is frontmost.
    current = observation

    if current is None or refresh:
        current = environment.observe()
        _publish_observation_timing(
            environment,
            progress,
        )

    if current.application_name != spec.expected_application:
        progress(
            "Browser focus changed; re-acquiring the exact "
            "Agent-owned Chrome task window."
        )

        environment.activate_task_chrome_window(
            _browser_window_marker_from_state(
                state,
                spec,
            ),
            spec.working_url_prefix,
        )

        current = environment.observe()
        _publish_observation_timing(
            environment,
            progress,
        )

    _require_expected_app(
        current,
        spec,
    )

    return current


def _publish_observation_timing(
    environment: object,
    progress: Callable[[str], None],
) -> None:
    timings = getattr(
        environment,
        "last_observation_timings",
        None,
    )
    if not isinstance(timings, dict) or not timings:
        return

    observe_total = _timing_value(timings, "observe_total")
    if observe_total is None:
        return

    accessibility = sum(
        value
        for key, value in timings.items()
        if key.startswith("accessibility")
        or key in (
            "frontmost_application",
            "viewport",
        )
    )
    ocr = _timing_value(timings, "ocr") or 0.0
    fusion = _timing_value(timings, "fusion") or 0.0
    stabilization = _timing_value(timings, "stabilization") or 0.0
    known = accessibility + ocr + fusion + stabilization
    other = max(0.0, observe_total - known)
    progress(
        "Timing: Last observation "
        f"{observe_total:.2f} s "
        f"(Accessibility {accessibility:.2f} s, "
        f"OCR {ocr:.2f} s, Fusion {fusion:.2f} s, "
        f"Stabilization {stabilization:.2f} s, "
        f"Other {other:.2f} s)."
    )


def _timing_value(
    timings: dict,
    key: str,
) -> float | None:
    value = timings.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return float(value)

def _reconcile_followup_navigation_condition(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    spec = resolved_task.spec
    _require_expected_app(
        observation,
        spec,
    )

    already_final = (
        _reconcile_followup_if_destination_visible(
            transitions=transitions,
            resolved_task=resolved_task,
            observation=observation,
            publish_state=publish_state,
            progress=progress,
        )
    )
    if already_final is not None:
        return already_final

    if not _results_visible(
        observation,
        resolved_task,
    ):
        observation = _observe_or_reactivate_workspace(
            state=state,
            environment=environment,
            spec=spec,
        )
        already_final = (
            _reconcile_followup_if_destination_visible(
                transitions=transitions,
                resolved_task=resolved_task,
                observation=observation,
                publish_state=publish_state,
                progress=progress,
            )
        )
        if already_final is not None:
            return already_final

    if not _results_visible(
        observation,
        resolved_task,
    ):
        raise RuntimeError(
            "The live browser state is ambiguous: "
            "it is neither verified search outcome "
            "state nor the requested destination."
        )

    readiness = _ground_followup_link_with_readiness(
        state=state,
        transitions=transitions,
        environment=environment,
        resolved_task=resolved_task,
        observation=observation,
        publish_state=publish_state,
        progress=progress,
    )
    observation = readiness.observation
    if readiness.already_at_destination:
        return observation
    if readiness.grounding is None:
        raise RuntimeError(
            "Follow-up readiness finished without "
            "a grounded link or destination evidence."
        )

    _persist_followup_destination_identity(
        transitions,
        resolved_task,
        readiness.grounding,
    )
    publish_state()

    _ensure_followup_side_effect(
        transitions,
        resolved_task,
    )
    publish_state()

    _publish_live_decision(
        publish_decision,
        observation=observation,
        decision_type="ACTION",
        operation="click_target",
        target_text=resolved_task.followup_target_text,
        expected_effect=(
            "Open the requested Wikipedia link "
            "and verify the destination heading."
        ),
        reason=(
            "Fresh search outcome state contains "
            "a unique actionable link for the "
            "requested follow-up target."
        ),
    )

    _mark_followup_execution_attempt(
        transitions,
        resolved_task,
    )
    publish_state()

    control.checkpoint()

    progress(
        "Opening the requested Wikipedia link "
        "through newly grounded live UI coordinates."
    )

    action_target = None
    if readiness.grounding.element is not None:
        action_target = TargetSpec(
            text=_followup_target_text(
                resolved_task
            ),
            element_types=("link",),
            minimum_confidence=0.70,
            reference_point=(
                readiness.grounding.element.center
            ),
        )
    result = _execute_agent_plan(
        environment,
        build_followup_plan(
            state.goal,
            resolved_task,
            action_target=action_target,
        ),
    )

    _maybe_inject_process_crash(
        LiveCrashPoint.FOLLOWUP_EXECUTION
    )

    final_observation = environment.observe()
    _publish_observation_timing(
        environment,
        progress,
    )
    _require_expected_app(
        final_observation,
        spec,
    )

    destination = _followup_destination_postcondition(
        final_observation,
        resolved_task,
        state=transitions.state,
    )
    if destination is None:
        _mark_followup_outcome_unknown(
            transitions,
        )
        publish_state()

        _require_agent_success(
            result,
            "real-web follow-up navigation",
        )

        raise RuntimeError(
            "Fresh post-follow-up observation did "
            "not resolve the requested destination heading."
        )

    evidence = _verify_followup_from_current_observation(
        transitions,
        resolved_task,
        summary=destination.summary,
        source=destination.source,
    )
    _confirm_followup_side_effect(
        transitions,
        resolved_task,
        evidence.evidence_id,
    )
    publish_state()

    return final_observation


def _ensure_browser_workspace(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    spec: DurableWebSearchSpec,
    publish_state: TaskStatePublisher,
    progress: Callable[[str], None],
) -> TextInputObservation:
    if _should_prepare_live_task_segment(
        state,
        spec,
    ):
        progress(
            "Opening the configured website "
            "in a dedicated Chrome task window."
        )

        browser_window_marker = (
            environment.open_task_site(
                spec.start_url,
                state.task_id
            )
        )

        transitions.add_artifact(
            ArtifactRecord(
                artifact_id=(
                    spec.workspace_artifact_id
                ),
                description=(
                    spec.workspace_description
                ),
                location=(
                    browser_window_marker
                ),
            )
        )

        publish_state()

        progress(
            "Created dedicated persistent "
            "Chrome task window."
        )
    else:
        browser_window_marker = (
            _browser_window_marker_from_state(
                state,
                spec,
            )
        )

        progress(
            "Re-activating the persisted "
            "Agent-owned Chrome task window."
        )

        environment.activate_task_chrome_window(
            browser_window_marker,
            spec.working_url_prefix,
        )

    observation = environment.observe()
    if (
        observation.application_name
        != spec.expected_application
    ):
        environment.activate_task_chrome_window(
            browser_window_marker,
            spec.working_url_prefix,
        )
        observation = environment.observe()

    _require_expected_app(
        observation,
        spec,
    )
    return observation


def _reconcile_query_condition(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    spec = resolved_task.spec
    _require_expected_app(
        observation,
        spec,
    )

    query_verification = (
        _query_condition_satisfied(
            observation,
            resolved_task,
        )
    )

    if query_verification is not None:
        _verify_query_from_current_observation(
            transitions,
            spec,
            summary=query_verification.summary,
            source=query_verification.source,
        )
        progress(
            "Fresh browser state verifies "
            "the intended query."
        )
        return observation

    if _results_visible(
        observation,
        resolved_task,
    ):
        if not _query_has_durable_history(
            state,
            spec,
        ):
            raise RuntimeError(
                "Results are visible, but the "
                "task has no durable query history "
                "to reconcile."
            )

        _verify_query_from_current_observation(
            transitions,
            spec,
            summary=(
                "Fresh Results state reconciles "
                "the previously verified "
                "query after restart."
            ),
            source=(
                "Post-submit live Chrome "
                "observation"
            ),
        )
        progress(
            "Fresh Results state reconciles "
            "the previously entered query."
        )
        return observation

    if not _search_field_available(
        observation,
        spec,
    ):
        raise RuntimeError(
            "The live browser state is ambiguous: "
            "neither the verified query field nor "
            "Results are visible."
        )

    progress(
        "Typing the intended query into "
        "the freshly grounded search field."
    )

    result = _execute_agent_plan(
        environment,
        build_prepare_plan(
            state.goal,
            resolved_task,
        ),
    )

    refreshed = environment.observe()
    _require_expected_app(
        refreshed,
        spec,
    )

    query_verification = (
        _query_condition_satisfied(
            refreshed,
            resolved_task,
        )
    )

    if query_verification is None:
        _require_agent_success(
            result,
            "real-web query entry",
        )
        raise RuntimeError(
            "The live search field did not "
            "uniquely verify the intended query."
        )

    _verify_query_from_current_observation(
        transitions,
        spec,
        summary=query_verification.summary,
        source=query_verification.source,
    )

    return refreshed


def _reconcile_submission_result_condition(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> TextInputObservation:
    spec = resolved_task.spec
    observation = _observe_or_reactivate_workspace(
        state=state,
        environment=environment,
        spec=spec,
    )

    if _results_visible(
        observation,
        resolved_task,
    ):
        evidence = _verify_results_from_current_observation(
            transitions,
            spec,
            summary=(
                "Fresh Chrome Accessibility "
                "shows the configured results "
                "marker."
            ),
            source=(
                "Live Chrome Accessibility "
                "observation"
            ),
        )
        _confirm_submit_side_effect(
            transitions,
            spec,
            evidence.evidence_id,
        )
        publish_state()
        progress(
            "Fresh browser state already "
            "shows the result target; "
            f"{spec.submit_target.text!r} was not clicked."
        )
        return observation

    if not _pre_submit_query_state(
        observation,
        resolved_task,
    ):
        raise RuntimeError(
            "The live browser state is ambiguous: "
            "it is neither verified pre-submit "
            "search state nor verified Results."
        )

    _ensure_submit_side_effect(
        transitions,
        spec,
    )
    publish_state()

    _mark_submit_execution_attempt(
        transitions,
        spec,
    )
    publish_state()

    _publish_live_decision(
        publish_decision,
        observation=observation,
        decision_type="ACTION",
        operation="click_target",
        target_text=spec.submit_target.text,
        expected_effect=(
            "Submit the verified query and navigate "
            "to the configured search results."
        ),
        reason=(
            "Fresh browser observation confirms "
            "the pre-submit state with the "
            "expected query."
        ),
    )

    control.checkpoint()

    progress(
        "Submitting through newly grounded "
        "live UI coordinates."
    )

    result = _execute_agent_plan(
        environment,
        build_resume_plan(
            state.goal,
            resolved_task,
        ),
    )

    _maybe_inject_process_crash(
        LiveCrashPoint.SUBMIT_EXECUTION
    )

    final_observation = environment.observe()
    _require_expected_app(
        final_observation,
        spec,
    )

    if not _results_visible(
        final_observation,
        resolved_task,
    ):
        _mark_submit_outcome_unknown(
            transitions,
            spec,
        )
        publish_state()

        _require_agent_success(
            result,
            "real-web search submission",
        )

        raise RuntimeError(
            "Fresh post-submit observation "
            "did not resolve Results."
        )

    results_evidence = (
        _verify_results_from_current_observation(
            transitions,
            spec,
            summary=(
                "The configured results "
                "marker is visible after "
                "submission."
            ),
            source=(
                "Post-submit live Chrome "
                "observation"
            ),
        )
    )

    _confirm_submit_side_effect(
        transitions,
        spec,
        results_evidence.evidence_id,
    )

    publish_state()

    return final_observation


def _observe_or_reactivate_workspace(
    *,
    state: TaskState,
    environment: LiveWebEnvironment,
    spec: DurableWebSearchSpec,
) -> TextInputObservation:
    observation = environment.observe()
    if (
        observation.application_name
        != spec.expected_application
    ):
        environment.activate_task_chrome_window(
            _browser_window_marker_from_state(
                state,
                spec,
            ),
            spec.working_url_prefix,
        )
        observation = environment.observe()

    _require_expected_app(
        observation,
        spec,
    )
    return observation


def _execute_agent_plan(
    environment: LiveWebEnvironment,
    plan: StructuredPlan,
):
    fake_executor = getattr(
        environment,
        "execute_plan",
        None,
    )

    if callable(
        fake_executor
    ):
        return fake_executor(
            plan
        )

    return AgentLoop(
        perception_engine=(
            environment.perception_engine
        ),
        executor=(
            environment.executor
        ),
        web_text_input_observer=(
            environment.observe
        ),
    ).run(
        plan
    )


def _verify_query_from_current_observation(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec,
    *,
    summary: str,
    source: str,
) -> EvidenceRecord:
    evidence = EvidenceRecord(
        summary=summary,
        source=source,
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        evidence
    )

    transitions.verify_claim(
        spec.query_claim_id,
        (evidence.evidence_id,),
    )

    transitions.verify_subgoal(
        spec.query_subgoal_id
    )

    return evidence


def _verify_results_from_current_observation(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec,
    *,
    summary: str,
    source: str,
) -> EvidenceRecord:
    evidence = EvidenceRecord(
        summary=summary,
        source=source,
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        evidence
    )

    transitions.verify_claim(
        spec.result_claim_id,
        (evidence.evidence_id,),
    )

    transitions.verify_subgoal(
        spec.result_subgoal_id
    )

    return evidence


def _verify_followup_from_current_observation(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    *,
    summary: str,
    source: str,
) -> EvidenceRecord:
    claim_id = _followup_claim_id(
        resolved_task
    )
    subgoal_id = _followup_subgoal_id(
        resolved_task
    )
    evidence = EvidenceRecord(
        summary=summary,
        source=source,
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        evidence
    )

    transitions.verify_claim(
        claim_id,
        (evidence.evidence_id,),
    )

    transitions.verify_subgoal(
        subgoal_id
    )

    return evidence


def _reconcile_followup_if_destination_visible(
    *,
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    observation: TextInputObservation,
    publish_state: TaskStatePublisher,
    progress: Callable[[str], None],
) -> TextInputObservation | None:
    if not _task_has_followup(
        resolved_task
    ):
        return None

    destination = _followup_destination_postcondition(
        observation,
        resolved_task,
        state=transitions.state,
    )
    if destination is None:
        return None

    _reconcile_search_history_from_final_destination(
        transitions,
        resolved_task,
    )

    evidence = _verify_followup_from_current_observation(
        transitions,
        resolved_task,
        summary=destination.summary,
        source=destination.source,
    )
    _confirm_followup_side_effect(
        transitions,
        resolved_task,
        evidence.evidence_id,
    )
    publish_state()
    progress(
        "Fresh browser state already shows "
        "the requested Wikipedia destination; "
        "the follow-up link was not clicked again."
    )
    return observation


def _reconcile_search_history_from_final_destination(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
) -> None:
    state = transitions.state
    spec = resolved_task.spec
    query_claim = state.claims.get(
        spec.query_claim_id
    )
    result_claim = state.claims.get(
        spec.result_claim_id
    )

    if (
        query_claim is None
        or result_claim is None
        or not query_claim.evidence_ids
        or not result_claim.evidence_ids
    ):
        return

    if (
        state.subgoals[
            spec.query_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
        and state.subgoals[
            spec.result_subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    ):
        return

    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary=(
                "Fresh final destination state "
                "reconciles previously durable "
                "query and search-result history "
                "after restart."
            ),
            source=(
                "Live Chrome destination observation "
                "plus persisted task history"
            ),
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        spec.query_claim_id,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        spec.query_subgoal_id
    )
    transitions.verify_claim(
        spec.result_claim_id,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        spec.result_subgoal_id
    )


def _ensure_submit_side_effect(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> SideEffectRecord:
    state = transitions.state
    effect = state.side_effects.get(
        spec.submit_side_effect_id
    )

    if effect is not None:
        return effect

    return transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                spec.submit_side_effect_id
            ),
            description=(
                spec.submit_description
            ),
            external_reference=spec.start_url,
            idempotent=True,
            action_key=spec.submit_action_key,
        )
    )


def _mark_submit_execution_attempt(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> None:
    effect = _ensure_submit_side_effect(
        transitions,
        spec,
    )

    if (
        effect.state
        is SideEffectState.INTENDED
    ):
        transitions.mark_side_effect_executed(
            spec.submit_side_effect_id
        )


def _mark_submit_outcome_unknown(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> None:
    effect = transitions.state.side_effects.get(
        spec.submit_side_effect_id
    )

    if (
        effect is not None
        and effect.state
        is SideEffectState.EXECUTED
    ):
        transitions.mark_side_effect_unknown(
            spec.submit_side_effect_id
        )


def _confirm_submit_side_effect(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec,
    evidence_id: str,
) -> None:
    effect = transitions.state.side_effects.get(
        spec.submit_side_effect_id
    )

    if effect is None:
        transitions.add_side_effect(
            SideEffectRecord(
                side_effect_id=(
                    spec.submit_side_effect_id
                ),
                description=(
                    spec.submit_description
                ),
                state=(
                    SideEffectState.CONFIRMED
                ),
                external_reference=spec.start_url,
                evidence_ids=(evidence_id,),
                idempotent=True,
                action_key=spec.submit_action_key,
            )
        )
        return

    if (
        effect.state
        is SideEffectState.CONFIRMED
    ):
        return

    if (
        effect.state
        is SideEffectState.INTENDED
    ):
        transitions.mark_side_effect_executed(
            spec.submit_side_effect_id
        )

    transitions.confirm_side_effect(
        spec.submit_side_effect_id,
        (evidence_id,),
    )


def _ensure_followup_side_effect(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
) -> SideEffectRecord:
    state = transitions.state
    effect = state.side_effects.get(
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
    )

    if effect is not None:
        return effect

    return transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
            ),
            description=(
                _followup_side_effect_description(
                    resolved_task
                )
            ),
            external_reference=(
                resolved_task.spec.working_url_prefix
            ),
            idempotent=True,
            action_key=(
                FOLLOWUP_NAVIGATION_ACTION_KEY
            ),
        )
    )


def _mark_followup_execution_attempt(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
) -> None:
    effect = _ensure_followup_side_effect(
        transitions,
        resolved_task,
    )

    if (
        effect.state
        is SideEffectState.INTENDED
    ):
        transitions.mark_side_effect_executed(
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        )


def _mark_followup_outcome_unknown(
    transitions: TaskStateTransitions,
) -> None:
    effect = transitions.state.side_effects.get(
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
    )

    if (
        effect is not None
        and effect.state
        is SideEffectState.EXECUTED
    ):
        transitions.mark_side_effect_unknown(
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        )


def _confirm_followup_side_effect(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    evidence_id: str,
) -> None:
    effect = transitions.state.side_effects.get(
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
    )

    if effect is None:
        transitions.add_side_effect(
            SideEffectRecord(
                side_effect_id=(
                    FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
                ),
                description=(
                    _followup_side_effect_description(
                        resolved_task
                    )
                ),
                state=(
                    SideEffectState.CONFIRMED
                ),
                external_reference=(
                    resolved_task.spec.working_url_prefix
                ),
                evidence_ids=(evidence_id,),
                idempotent=True,
                action_key=(
                    FOLLOWUP_NAVIGATION_ACTION_KEY
                ),
            )
        )
        return

    if (
        effect.state
        is SideEffectState.CONFIRMED
    ):
        return

    if (
        effect.state
        is SideEffectState.INTENDED
    ):
        transitions.mark_side_effect_executed(
            FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
        )

    transitions.confirm_side_effect(
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID,
        (evidence_id,),
    )


def _try_ground_followup_link(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> GroundingResult:
    resolver = SemanticTargetResolver(
        destination_normalizer=_wikipedia_article_identity,
    )
    return resolver.ground(
        _followup_link_target(resolved_task),
        observation.snapshot.fused_elements,
    )


def _ground_followup_link(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> GroundingResult:
    grounding = _try_ground_followup_link(
        observation,
        resolved_task,
    )

    if grounding.status is GroundingStatus.RESOLVED:
        return grounding

    _raise_followup_grounding_error(
        resolved_task,
        grounding,
    )


def _ground_followup_link_with_readiness(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    observation: TextInputObservation,
    publish_state: TaskStatePublisher,
    progress: Callable[[str], None],
) -> FollowupLinkReadinessResult:
    current = observation
    grounding = _try_ground_followup_link(
        current,
        resolved_task,
    )
    if grounding.status is GroundingStatus.RESOLVED:
        return FollowupLinkReadinessResult(
            observation=current,
            grounding=grounding,
        )
    if grounding.status is not GroundingStatus.NOT_FOUND:
        _raise_followup_grounding_error(
            resolved_task,
            grounding,
        )

    for _ in range(
        FOLLOWUP_LINK_READINESS_REOBSERVATIONS
    ):
        progress(
            "Requested link is not exposed yet; "
            "refreshing browser state."
        )
        current = _ensure_expected_workspace_observation(
            state=state,
            environment=environment,
            spec=resolved_task.spec,
            observation=None,
            refresh=True,
            progress=progress,
        )

        already_final = (
            _reconcile_followup_if_destination_visible(
                transitions=transitions,
                resolved_task=resolved_task,
                observation=current,
                publish_state=publish_state,
                progress=progress,
            )
        )
        if already_final is not None:
            return FollowupLinkReadinessResult(
                observation=already_final,
                grounding=None,
                already_at_destination=True,
            )

        grounding = _try_ground_followup_link(
            current,
            resolved_task,
        )
        if grounding.status is GroundingStatus.RESOLVED:
            return FollowupLinkReadinessResult(
                observation=current,
                grounding=grounding,
            )
        if grounding.status is not GroundingStatus.NOT_FOUND:
            _raise_followup_grounding_error(
                resolved_task,
                grounding,
            )

    _raise_followup_grounding_error(
        resolved_task,
        grounding,
    )


def _raise_followup_grounding_error(
    resolved_task: ResolvedDurableWebSearchTask,
    grounding: GroundingResult,
) -> NoReturn:
    raise RuntimeError(
        "The requested Wikipedia follow-up link "
        f"{resolved_task.followup_target_text!r} "
        "did not resolve to one actionable link: "
        f"{grounding.status.value}. "
        + _followup_ambiguity_diagnostics(grounding)
    )


def _followup_ambiguity_diagnostics(
    grounding: GroundingResult,
) -> str:
    eligible = tuple(
        candidate
        for candidate in grounding.candidates
        if candidate.eligible
    )
    if not eligible:
        return "No eligible actionable candidates were found."

    lines = [
        f"Eligible candidates: {len(eligible)}.",
    ]
    identities = []
    for index, candidate in enumerate(eligible, start=1):
        element = candidate.element
        identity = _wikipedia_article_identity(element.value)
        identities.append(identity)
        box = element.bounding_box
        lines.append(
            "Candidate "
            f"{index}: text={element.text!r}, "
            f"type={element.element_type!r}, "
            f"bounds=({box.left},{box.top},{box.width},{box.height}), "
            f"destination={element.value!r}."
        )

    canonical = {
        identity[0]
        for identity in identities
        if identity is not None
    }
    equivalent = (
        len(identities) == len(eligible)
        and all(identity is not None for identity in identities)
        and len(canonical) == 1
    )
    lines.append(
        "Equivalent destination: "
        + ("yes." if equivalent else "no.")
    )
    return " ".join(lines)


def _results_visible(
    observation: TextInputObservation,
    target: DurableWebSearchSpec | ResolvedDurableWebSearchTask,
) -> bool:
    if isinstance(
        target,
        ResolvedDurableWebSearchTask,
    ):
        spec = target.spec
        dynamic_targets = (
            ()
            if spec.result_target_factory is None
            else spec.result_target_factory(
                target.query_text
            )
        )
    else:
        spec = target
        dynamic_targets = ()

    for result_target in (
        spec.result_target,
        *dynamic_targets,
    ):
        grounding = UIGrounder().ground(
            result_target,
            observation.snapshot.fused_elements,
        )

        if (
            grounding.status
            is GroundingStatus.RESOLVED
        ):
            return True

    return False


def _followup_destination_visible(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    *,
    state: TaskState | None = None,
) -> bool:
    return (
        _followup_destination_postcondition(
            observation,
            resolved_task,
            state=state,
        )
        is not None
    )


def _followup_destination_postcondition(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
    *,
    state: TaskState | None,
) -> StepPostconditionResult | None:
    direct = UIGrounder().ground(
        _followup_destination_target(
            resolved_task
        ),
        observation.snapshot.fused_elements,
    )
    if direct.status is GroundingStatus.RESOLVED:
        return StepPostconditionResult(
            summary=(
                "Fresh destination heading directly matches "
                f"the requested Wikipedia link "
                f"{resolved_task.followup_target_text!r}."
            ),
            source="Live Chrome destination heading observation",
        )

    if state is None:
        return None

    artifact = state.artifacts.get(
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    )
    if artifact is None:
        return None

    expected_identity = _wikipedia_article_identity(
        artifact.location
    )
    current_url = _chrome_address_bar_url(
        observation
    )
    current_identity = _wikipedia_article_identity(
        current_url
    )
    if (
        expected_identity is None
        or current_identity is None
        or expected_identity[0] != current_identity[0]
    ):
        return None

    canonical_path, canonical_title = current_identity
    heading = UIGrounder().ground(
        TargetSpec(
            text=canonical_title,
            element_types=("heading",),
            minimum_confidence=0.70,
        ),
        observation.snapshot.fused_elements,
    )
    if heading.status is not GroundingStatus.RESOLVED:
        return None

    return StepPostconditionResult(
        summary=(
            "Fresh Wikipedia navigation verifies requested link "
            f"{resolved_task.followup_target_text!r}: persisted "
            f"destination {canonical_path!r} agrees with the fresh "
            f"address-bar URL and heading {canonical_title!r}."
        ),
        source=(
            "Persisted grounded-link AXURL plus fresh Chrome address bar "
            "and Wikipedia heading"
        ),
    )


def _ensure_followup_destination_identity(
    transitions: TaskStateTransitions,
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> None:
    grounding = _ground_followup_link(
        observation,
        resolved_task,
    )
    _persist_followup_destination_identity(
        transitions,
        resolved_task,
        grounding,
    )


def _persist_followup_destination_identity(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    grounding,
) -> None:
    element = grounding.element
    raw_url = (
        getattr(element, "value", None)
        if element is not None
        else None
    )
    identity = _wikipedia_article_identity(
        raw_url
    )
    if identity is None:
        # Direct requested-heading verification remains available when AXURL
        # is genuinely unavailable. Never invent a destination URL.
        return

    canonical_path, _ = identity
    canonical_url = (
        "https://en.wikipedia.org"
        + canonical_path
    )
    existing = transitions.state.artifacts.get(
        FOLLOWUP_DESTINATION_ARTIFACT_ID
    )
    if existing is not None:
        existing_identity = _wikipedia_article_identity(
            existing.location
        )
        if (
            existing_identity is None
            or existing_identity[0] != canonical_path
        ):
            raise RuntimeError(
                "Persisted Wikipedia follow-up destination does not "
                "match the freshly grounded link."
            )
        return

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=FOLLOWUP_DESTINATION_ARTIFACT_ID,
            description=(
                "Canonical Wikipedia destination of the uniquely "
                "grounded follow-up link."
            ),
            location=canonical_url,
        )
    )


def _chrome_address_bar_url(
    observation: TextInputObservation,
) -> str | None:
    expected_label = normalize_ui_text(
        "Address and search bar"
    )
    values = []
    for element in observation.snapshot.fused_elements:
        if element.element_type != "text_field":
            continue
        if normalize_ui_text(element.text or "") != expected_label:
            continue
        value = element.value
        if isinstance(value, str) and value.strip():
            values.append(value.strip())

    unique = tuple(dict.fromkeys(values))
    if len(unique) != 1:
        return None
    return unique[0]


def _wikipedia_article_identity(
    raw_url: object,
) -> tuple[str, str] | None:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None

    value = raw_url.strip()
    if value.startswith("/wiki/"):
        value = "https://en.wikipedia.org" + value
    elif "://" not in value:
        value = "https://" + value

    try:
        parsed = urlparse(value)
    except ValueError:
        return None

    if (parsed.hostname or "").lower() != "en.wikipedia.org":
        return None

    path = unquote(parsed.path or "")
    prefix = "/wiki/"
    if not path.startswith(prefix):
        return None

    slug = path[len(prefix):].strip()
    if not slug or slug.startswith("Special:"):
        return None

    canonical_path = prefix + slug
    canonical_title = slug.replace("_", " ").strip()
    if not canonical_title:
        return None

    return canonical_path, canonical_title


def _followup_link_target(
    resolved_task: ResolvedDurableWebSearchTask,
) -> TargetSpec:
    target_text = _followup_target_text(
        resolved_task
    )
    return TargetSpec(
        text=target_text,
        element_types=("link",),
        minimum_confidence=0.70,
    )


def _followup_destination_target(
    resolved_task: ResolvedDurableWebSearchTask,
) -> TargetSpec:
    target_text = _followup_target_text(
        resolved_task
    )
    return TargetSpec(
        text=target_text,
        element_types=("heading",),
        minimum_confidence=0.70,
    )


def _task_has_followup(
    resolved_task: ResolvedDurableWebSearchTask,
) -> bool:
    return (
        resolved_task.followup_target_text
        is not None
    )


def _followup_target_text(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    if resolved_task.followup_target_text is None:
        raise RuntimeError(
            "Resolved task has no follow-up target."
        )
    return resolved_task.followup_target_text


def _followup_claim_id(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    claim_id = resolved_task.spec.followup_claim_id
    if claim_id is None:
        raise RuntimeError(
            "Workflow has no follow-up claim configured."
        )
    return claim_id


def _followup_subgoal_id(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    subgoal_id = resolved_task.spec.followup_subgoal_id
    if subgoal_id is None:
        raise RuntimeError(
            "Workflow has no follow-up subgoal configured."
        )
    return subgoal_id


def _followup_claim_text(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    target_text = _followup_target_text(
        resolved_task
    )
    factory = (
        resolved_task.spec.followup_claim_text_factory
    )
    if factory is None:
        return (
            "The requested destination "
            f"{target_text!r} is visible."
        )
    return factory(
        target_text
    )


def _followup_subgoal_text(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    target_text = _followup_target_text(
        resolved_task
    )
    factory = (
        resolved_task.spec.followup_subgoal_text_factory
    )
    if factory is None:
        return (
            "Open the requested link "
            f"{target_text!r} and verify its destination."
        )
    return factory(
        target_text
    )


def _followup_side_effect_description(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    return (
        "Open Wikipedia link "
        f"{_followup_target_text(resolved_task)!r}."
    )


def _query_condition_satisfied(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> QueryVerificationResult | None:
    spec = resolved_task.spec
    if (
        spec.query_verification_mode
        is QueryVerificationMode.FIELD_VALUE
    ):
        search_field = (
            _search_field_with_expected_value(
                observation.snapshot.fused_elements,
                spec=spec,
                expected_value=(
                    resolved_task.query_text
                ),
            )
        )

        if search_field is None:
            return None

        return QueryVerificationResult(
            element=search_field,
            summary=(
                "The current search field contains "
                f"{resolved_task.query_text!r}."
            ),
            source=(
                "Live Chrome Accessibility "
                "observation"
            ),
        )

    if (
        spec.query_verification_mode
        is QueryVerificationMode.VISIBLE_SEARCH_UI
    ):
        return _visible_search_ui_query_condition(
            observation,
            resolved_task,
        )

    return None


def _visible_search_ui_query_condition(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> QueryVerificationResult | None:
    spec = resolved_task.spec
    elements = tuple(
        observation.snapshot.fused_elements
    )
    submit_grounding = UIGrounder().ground(
        spec.submit_target,
        elements,
    )

    if (
        submit_grounding.status
        is not GroundingStatus.RESOLVED
        or submit_grounding.element is None
    ):
        return None

    for target in _query_confirmation_targets(
        resolved_task
    ):
        associated = tuple(
            element
            for element in _visible_query_evidence_matches(
                elements,
                target,
            )
            if _is_query_evidence_near_submit(
                element,
                submit_grounding.element,
            )
        )
        associated = _collapse_equivalent_query_evidence(
            associated
        )

        if len(associated) == 1:
            return QueryVerificationResult(
                element=associated[0],
                summary=(
                    "Fresh visible search UI "
                    f"confirms {resolved_task.query_text!r} "
                    "adjacent to the Search control."
                ),
                source=(
                    "Live Chrome visible search UI "
                    "observation"
                ),
            )

        if len(associated) > 1:
            return None

    return None


def _query_confirmation_targets(
    resolved_task: ResolvedDurableWebSearchTask,
) -> tuple[TargetSpec, ...]:
    spec = resolved_task.spec
    configured_targets: tuple[TargetSpec, ...] = ()
    if spec.query_confirmation_target_factory is not None:
        configured_targets = (
            spec.query_confirmation_target_factory(
                resolved_task.query_text
            )
        )

    targets = (
        TargetSpec(
            text=resolved_task.query_text,
            element_types=("text",),
            minimum_confidence=(
                spec.submit_target.minimum_confidence
            ),
        ),
        *configured_targets,
    )
    unique: list[TargetSpec] = []
    seen: set[tuple[str | None, tuple[str, ...]]] = set()

    for target in targets:
        key = (
            normalize_ui_text(
                target.text
            )
            if target.text is not None
            else None,
            tuple(
                normalize_ui_text(
                    element_type
                )
                for element_type in target.element_types
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )
        unique.append(
            target
        )

    return tuple(unique)


def _visible_query_evidence_matches(
    elements: tuple[UIElement, ...],
    target: TargetSpec,
) -> tuple[UIElement, ...]:
    if target.text is None:
        return ()

    expected = normalize_ui_text(
        target.text
    )

    return tuple(
        element
        for element in elements
        if element.enabled is not False
        and element.confidence
        >= target.minimum_confidence
        and _target_element_type_matches(
            element,
            target,
        )
        and (
            normalize_ui_text(
                element.text
            )
            == expected
            or normalize_ui_text(
                str(element.value or "")
            )
            == expected
        )
    )


def _target_element_type_matches(
    element: UIElement,
    target: TargetSpec,
) -> bool:
    if not target.element_types:
        return True

    element_type = normalize_ui_text(
        element.element_type
    )
    return element_type in {
        normalize_ui_text(
            expected_type
        )
        for expected_type in target.element_types
    }


def _collapse_equivalent_query_evidence(
    elements: tuple[UIElement, ...],
) -> tuple[UIElement, ...]:
    unique: list[UIElement] = []

    for element in elements:
        if any(
            _query_evidence_equivalent(
                element,
                existing,
            )
            for existing in unique
        ):
            continue

        unique.append(
            element
        )

    return tuple(unique)


def _query_evidence_equivalent(
    first: UIElement,
    second: UIElement,
) -> bool:
    if _element_visible_text(
        first
    ) != _element_visible_text(
        second
    ):
        return False

    return first.bounding_box.intersects(
        second.bounding_box
    )


def _element_visible_text(
    element: UIElement,
) -> str:
    text = normalize_ui_text(
        element.text
    )
    if text:
        return text

    return normalize_ui_text(
        str(element.value or "")
    )


def _is_query_evidence_near_submit(
    query_element: UIElement,
    submit_element: UIElement,
) -> bool:
    query_box = query_element.bounding_box
    submit_box = submit_element.bounding_box

    submit_height = submit_box.height
    submit_width = submit_box.width
    vertical_overlap = min(
        query_box.bottom,
        submit_box.bottom,
    ) - max(
        query_box.top,
        submit_box.top,
    )
    same_row = vertical_overlap >= min(
        query_box.height,
        submit_height,
    ) * 0.35
    horizontal_gap = (
        submit_box.left - query_box.right
    )

    if (
        same_row
        and query_box.left < submit_box.left
        and horizontal_gap
        <= submit_width * 8
    ):
        return True

    vertical_gap = (
        query_box.top - submit_box.bottom
    )
    horizontally_related = (
        query_box.left <= submit_box.right
        and query_box.right
        >= submit_box.left - submit_width * 8
    )

    return (
        0 <= vertical_gap <= submit_height * 4
        and horizontally_related
    )


def _search_field_available(
    observation: TextInputObservation,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> bool:
    grounding = UIGrounder().ground(
        spec.search_field,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _pre_submit_query_state(
    observation: TextInputObservation,
    resolved_task: ResolvedDurableWebSearchTask,
) -> bool:
    spec = resolved_task.spec
    if _results_visible(
        observation,
        resolved_task,
    ):
        return False

    if (
        _query_condition_satisfied(
            observation,
            resolved_task,
        )
        is None
    ):
        return False

    grounding = UIGrounder().ground(
        spec.submit_target,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _query_has_durable_history(
    state: TaskState,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> bool:
    claim = state.claims.get(
        spec.query_claim_id
    )

    return (
        claim is not None
        and bool(claim.evidence_ids)
    )


def _run_prepare_segment(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    durable_plan = compile_durable_task_plan(
        resolved_task
    )
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
        resolved_task=resolved_task,
        durable_plan=durable_plan,
        control=control,
        publish_state=publish_state,
        publish_decision=publish_decision,
        progress=progress,
    )


def _run_resume_segment(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    resolved_task: ResolvedDurableWebSearchTask,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    durable_plan = compile_durable_task_plan(
        resolved_task
    )
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
        resolved_task=resolved_task,
        durable_plan=durable_plan,
        control=control,
        publish_state=publish_state,
        publish_decision=publish_decision,
        progress=progress,
    )


def _ensure_live_task_structure(
    transitions: TaskStateTransitions,
    target: (
        DurableTaskPlan
        | ResolvedDurableWebSearchTask
        | DurableWebSearchSpec
    ) = PYTHON_WORKFLOW,
) -> None:
    """Pre-register the complete live-task semantic structure."""
    state = transitions.state
    if isinstance(
        target,
        DurableTaskPlan,
    ):
        for step in target.steps:
            if step.claim_id not in state.claims:
                transitions.add_claim(
                    ClaimRecord(
                        claim_id=step.claim_id,
                        statement=step.claim_text,
                    )
                )
            if step.subgoal_id not in state.subgoals:
                transitions.add_subgoal(
                    SubgoalRecord(
                        subgoal_id=step.subgoal_id,
                        description=step.subgoal_text,
                        claim_ids=(step.claim_id,),
                    )
                )
        return

    if isinstance(
        target,
        ResolvedDurableWebSearchTask,
    ):
        spec = target.spec
        resolved_task = target
    else:
        spec = target
        resolved_task = None

    if spec.query_claim_id not in state.claims:
        transitions.add_claim(
            ClaimRecord(
                claim_id=spec.query_claim_id,
                statement=spec.query_claim_text,
            )
        )

    if spec.result_claim_id not in state.claims:
        transitions.add_claim(
            ClaimRecord(
                claim_id=spec.result_claim_id,
                statement=spec.result_claim_text,
            )
        )

    if spec.query_subgoal_id not in state.subgoals:
        transitions.add_subgoal(
            SubgoalRecord(
                subgoal_id=spec.query_subgoal_id,
                description=spec.query_subgoal_text,
                claim_ids=(
                    spec.query_claim_id,
                ),
            )
        )

    if spec.result_subgoal_id not in state.subgoals:
        transitions.add_subgoal(
            SubgoalRecord(
                subgoal_id=spec.result_subgoal_id,
                description=spec.result_subgoal_text,
                claim_ids=(
                    spec.result_claim_id,
                ),
            )
        )

    if (
        resolved_task is not None
        and _task_has_followup(
            resolved_task
        )
    ):
        claim_id = _followup_claim_id(
            resolved_task
        )
        subgoal_id = _followup_subgoal_id(
            resolved_task
        )
        if claim_id not in state.claims:
            transitions.add_claim(
                ClaimRecord(
                    claim_id=claim_id,
                    statement=(
                        _followup_claim_text(
                            resolved_task
                        )
                    ),
                )
            )

        if subgoal_id not in state.subgoals:
            transitions.add_subgoal(
                SubgoalRecord(
                    subgoal_id=subgoal_id,
                    description=(
                        _followup_subgoal_text(
                            resolved_task
                        )
                    ),
                    claim_ids=(
                        claim_id,
                    ),
                )
            )


def _ensure_workflow_identity(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec,
) -> None:
    """Persist and validate the selected durable workflow identity."""
    state = transitions.state
    expected_location = _workflow_location(
        spec
    )
    artifact = state.artifacts.get(
        WORKFLOW_ARTIFACT_ID
    )

    if artifact is not None:
        if artifact.location != expected_location:
            raise RuntimeError(
                "Persisted live web workflow identity "
                "does not match the resolved goal."
            )
        return

    if (
        spec.workspace_artifact_id
        in state.artifacts
        and not _task_state_matches_workflow(
            state,
            spec,
        )
    ):
        raise RuntimeError(
            "Persisted task has no durable web workflow "
            "identity and cannot be safely resumed."
        )

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=WORKFLOW_ARTIFACT_ID,
            description=(
                "Durable live web workflow identity."
            ),
            location=expected_location,
        )
    )


def _ensure_query_identity(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
) -> None:
    """Persist and validate the exact runtime query identity."""
    state = transitions.state
    artifact = state.artifacts.get(
        QUERY_ARTIFACT_ID
    )

    if artifact is not None:
        if artifact.location != resolved_task.query_text:
            raise RuntimeError(
                "Persisted live web query identity "
                "does not match the resolved goal."
            )
        return

    if (
        resolved_task.spec.workspace_artifact_id
        in state.artifacts
    ):
        raise RuntimeError(
            "Persisted task has no durable web query "
            "identity and cannot be safely resumed."
        )

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=QUERY_ARTIFACT_ID,
            description=(
                "Durable live web runtime query text."
            ),
            location=resolved_task.query_text,
        )
    )


def _ensure_followup_identity(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
) -> None:
    """Persist and validate the requested follow-up target identity."""
    state = transitions.state
    artifact = state.artifacts.get(
        FOLLOWUP_TARGET_ARTIFACT_ID
    )

    if not _task_has_followup(
        resolved_task
    ):
        return

    expected_target = _followup_target_text(
        resolved_task
    )

    if artifact is not None:
        if artifact.location != expected_target:
            raise RuntimeError(
                "Persisted live web follow-up target "
                "identity does not match the resolved goal."
            )
        return

    if (
        resolved_task.spec.workspace_artifact_id
        in state.artifacts
    ):
        raise RuntimeError(
            "Persisted multi-step task has no durable "
            "web follow-up target identity and cannot "
            "be safely resumed."
        )

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=(
                FOLLOWUP_TARGET_ARTIFACT_ID
            ),
            description=(
                "Durable live web follow-up target text."
            ),
            location=expected_target,
        )
    )


def _ensure_durable_planner_provenance(
    transitions: TaskStateTransitions,
    provenance: DurablePlannerProvenance | None,
) -> None:
    """Persist and validate the accepted durable planner provenance."""
    state = transitions.state
    artifact = state.artifacts.get(
        DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID
    )
    if artifact is not None:
        _read_durable_planner_provenance(
            state,
            allow_missing=False,
        )
        return

    if provenance is None:
        return

    if DURABLE_PLAN_ARTIFACT_ID in state.artifacts:
        return

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=(
                DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID
            ),
            description=(
                "Durable planner provenance for "
                "the accepted task plan."
            ),
            location=durable_planner_provenance_canonical_json(
                provenance
            ),
        )
    )


def _read_durable_planner_provenance(
    state: TaskState,
    *,
    allow_missing: bool,
) -> DurablePlannerProvenance | None:
    artifact = state.artifacts.get(
        DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID
    )
    if artifact is None:
        if allow_missing:
            return None
        raise RuntimeError(
            "Persisted task has no durable planner provenance."
        )

    try:
        payload = json.loads(
            artifact.location
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Persisted durable planner provenance "
            "is not valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Persisted durable planner provenance "
            "must be a JSON object."
        )
    expected_keys = {
        "version",
        "planner_type",
        "model_identifier",
    }
    if set(payload) != expected_keys:
        raise RuntimeError(
            "Persisted durable planner provenance "
            "contains unsupported fields."
        )
    if payload["version"] != DURABLE_PLANNER_PROVENANCE_VERSION:
        raise RuntimeError(
            "Persisted durable planner provenance "
            "version is unsupported."
        )
    planner_type = payload["planner_type"]
    if not isinstance(planner_type, str) or not planner_type.strip():
        raise RuntimeError(
            "Persisted durable planner provenance "
            "has invalid planner type."
        )
    model_identifier = payload["model_identifier"]
    if model_identifier is not None and (
        not isinstance(model_identifier, str)
        or not model_identifier.strip()
    ):
        raise RuntimeError(
            "Persisted durable planner provenance "
            "has invalid model identifier."
        )

    provenance = DurablePlannerProvenance(
        planner_type=planner_type,
        model_identifier=model_identifier,
    )
    if (
        durable_planner_provenance_canonical_json(
            provenance
        )
        != artifact.location
    ):
        raise RuntimeError(
            "Persisted durable planner provenance "
            "is not canonical."
        )
    return provenance


def _ensure_durable_plan_identity(
    transitions: TaskStateTransitions,
    resolved_task: ResolvedDurableWebSearchTask,
    durable_plan: DurableTaskPlan,
) -> None:
    """Persist and validate the exact ordered durable task plan."""
    state = transitions.state
    expected_location = durable_plan_canonical_json(
        durable_plan
    )
    artifact = state.artifacts.get(
        DURABLE_PLAN_ARTIFACT_ID
    )

    if artifact is not None:
        _validate_durable_plan_json(
            artifact.location
        )
        if artifact.location != expected_location:
            raise RuntimeError(
                "Persisted durable task plan does not "
                "match the resolved goal."
            )
        return

    if (
        resolved_task.spec.workspace_artifact_id
        in state.artifacts
    ):
        raise RuntimeError(
            "Persisted task has no durable task plan "
            "identity and cannot be safely resumed."
        )

    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=DURABLE_PLAN_ARTIFACT_ID,
            description=(
                "Canonical persisted durable task plan."
            ),
            location=expected_location,
        )
    )


def _validate_durable_plan_json(
    value: str,
) -> None:
    durable_plan_from_canonical_json(
        value
    )


def _workflow_location(
    spec: DurableWebSearchSpec,
) -> str:
    return f"workflow:{spec.workflow_id}"


def _completion_summary(
    resolved_task: ResolvedDurableWebSearchTask,
) -> str:
    if resolved_task.spec is PYTHON_WORKFLOW:
        site = "python.org"
    elif resolved_task.spec is WIKIPEDIA_WORKFLOW:
        site = "Wikipedia"
    else:
        site = resolved_task.spec.workflow_id

    if resolved_task.followup_target_text is None:
        return (
            f"{site} search for "
            f"{resolved_task.query_text!r} "
            "completed and was verified."
        )

    return (
        f"{site} search for "
        f"{resolved_task.query_text!r} "
        "completed and opened "
        f"{resolved_task.followup_target_text!r} "
        "with destination verification."
    )


def _task_state_matches_workflow(
    state: TaskState,
    spec: DurableWebSearchSpec,
) -> bool:
    known_ids = (
        spec.query_claim_id,
        spec.result_claim_id,
        spec.query_subgoal_id,
        spec.result_subgoal_id,
        spec.submit_side_effect_id,
        spec.followup_claim_id,
        spec.followup_subgoal_id,
        FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID,
    )
    return any(
        known_id is not None
        and (
            known_id in state.claims
        or known_id in state.subgoals
        or known_id in state.side_effects
        )
        for known_id in known_ids
    )


def _should_prepare_live_task_segment(
    state: TaskState,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> bool:
    """Return whether this run still needs to create the browser workspace."""
    return (
        spec.workspace_artifact_id
        not in state.artifacts
    )


def _browser_window_marker_from_state(
    state: TaskState,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> str:
    artifact = state.artifacts.get(
        spec.workspace_artifact_id
    )

    if artifact is None:
        raise RuntimeError(
            "Persisted task has no owned "
            "Chrome window identity."
        )

    marker_url = artifact.location

    if marker_url.startswith(
        "chrome-window-id:"
    ):
        raise RuntimeError(
            "This checkpoint uses the legacy "
            "Chrome window identity format and "
            "cannot be safely resumed. Start a "
            "new task."
        )

    if not marker_url.startswith(
        BROWSER_WINDOW_MARKER_PREFIX
    ):
        raise RuntimeError(
            "Persisted Chrome task-window "
            "marker is invalid."
        )

    expected_marker = (
        BROWSER_WINDOW_MARKER_PREFIX
        + state.task_id
    )

    if marker_url != expected_marker:
        raise RuntimeError(
            "Persisted Chrome task-window "
            "marker does not belong to this task."
        )

    return marker_url


def _search_field_with_expected_value(
    elements,
    *,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
    expected_value: str,
) -> UIElement | None:
    """Find the current search field only when its live value matches."""
    element_tuple = tuple(elements)
    expected = normalize_ui_text(
        expected_value
    )

    grounding = UIGrounder().ground(
        spec.search_field,
        element_tuple,
    )

    if (
        grounding.status
        is GroundingStatus.RESOLVED
        and grounding.element is not None
        and _element_value_matches(
            grounding.element,
            expected,
        )
    ):
        return grounding.element

    matches = tuple(
        element
        for element in element_tuple
        if _is_enabled_text_field(element)
        and element.confidence
        >= spec.search_field.minimum_confidence
        and _element_value_matches(
            element,
            expected,
        )
    )

    if len(matches) != 1:
        return None

    return matches[0]


def _is_enabled_text_field(
    element: UIElement,
) -> bool:
    return (
        normalize_ui_text(
            element.element_type
        )
        == normalize_ui_text(
            "text_field"
        )
        and element.enabled is not False
    )


def _element_value_matches(
    element: UIElement,
    expected: str,
) -> bool:
    value = getattr(
        element,
        "value",
        None,
    )

    observed = normalize_ui_text(
        str(value or "")
    )

    return observed == expected or _character_spaced_value_matches(
        observed,
        expected,
    )


def _character_spaced_value_matches(
    observed: str,
    expected: str,
) -> bool:
    if " " not in observed:
        return False

    fragments = observed.split()
    if len(fragments) < 2:
        return False

    short_fragments = sum(
        1 for fragment in fragments if len(fragment) <= 2
    )
    if short_fragments < len(fragments) * 0.75:
        return False

    return observed.replace(" ", "") == expected.replace(" ", "")


def _visible_observation_text(
    observation: TextInputObservation,
    *,
    limit: int = 16,
) -> tuple[str, ...]:
    """Return a compact unique text summary for the workspace UI."""
    values: list[str] = []
    seen: set[str] = set()

    for element in (
        observation.snapshot.fused_elements
    ):
        text = getattr(
            element,
            "text",
            None,
        )

        if (
            not isinstance(text, str)
            or not text.strip()
        ):
            continue

        normalized = normalize_ui_text(
            text
        )

        if (
            not normalized
            or normalized in seen
        ):
            continue

        seen.add(
            normalized
        )
        values.append(
            text.strip()
        )

        if len(values) >= limit:
            break

    return tuple(values)


def _publish_live_decision(
    publish_decision: AdaptiveDecisionPublisher,
    *,
    observation: TextInputObservation,
    decision_type: str,
    operation: str | None,
    target_text: str | None,
    expected_effect: str | None,
    reason: str,
    completion_summary: str | None = None,
) -> None:
    """Publish one truthful deterministic live-workspace decision."""
    attempt = DecisionAttemptSnapshot(
        attempt_number=1,
        status="DETERMINISTIC",
        decision_type=decision_type,
        operation=operation,
        target_text=target_text,
        result="ACCEPTED",
        reason=reason,
    )

    snapshot = AdaptiveDecisionSnapshot(
        observation_application=(
            observation.application_name
        ),
        observation_window=None,
        observation_text=(
            _visible_observation_text(
                observation
            )
        ),
        decision_source="Deterministic live policy",
        model_attempts=0,
        safety_replan_used=False,
        final_status="ACCEPTED",
        final_decision_type=decision_type,
        operation=operation,
        target_text=target_text,
        expected_effect=expected_effect,
        question=None,
        completion_summary=completion_summary,
        blocked_reason=None,
        blocked_action_keys=(),
        attempts=(attempt,),
    )

    publish_decision(
        snapshot
    )


def _require_expected_app(
    observation: TextInputObservation,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> None:
    if (
        observation.application_name
        != spec.expected_application
    ):
        raise RuntimeError(
            "Frontmost application was "
            f"{observation.application_name!r}; "
            f"expected {spec.expected_application!r}."
        )

    if observation.snapshot.warnings:
        raise RuntimeError(
            "Live perception returned warnings: "
            f"{observation.snapshot.warnings}"
        )


def _require_agent_success(
    result,
    label: str,
) -> None:
    if (
        result.status
        is not AgentLoopStatus.COMPLETED
        or result.state.status
        is not AgentStatus.SUCCEEDED
        or result.completed_plan_steps != 1
    ):
        raise RuntimeError(
            f"{label} failed: "
            f"loop={result.status.value}, "
            f"state={result.state.status.value}, "
            "completed_steps="
            f"{result.completed_plan_steps}"
        )


def _require_tool_success(
    result,
    label: str,
) -> None:
    if not result.success:
        raise RuntimeError(
            f"{label} failed: "
            f"{result.error}"
        )
