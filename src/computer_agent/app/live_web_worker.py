"""Bounded live-web worker for the persistent Agent Workspace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import os
from pathlib import Path
import sys
import time
from typing import NoReturn

from computer_agent.agent import (
    AgentLoop,
    AgentLoopStatus,
    AgentStatus,
    TextInputObservation,
)
from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.core.models import Action
from computer_agent.grounding import (
    GroundingStatus,
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


class QueryVerificationMode(StrEnum):
    """Postcondition strategy for verifying query entry."""

    FIELD_VALUE = "field_value"
    VISIBLE_SEARCH_UI = "visible_search_ui"


@dataclass(frozen=True, slots=True)
class QueryVerificationResult:
    """Fresh evidence that the configured query condition is satisfied."""

    element: UIElement
    summary: str
    source: str


@dataclass(frozen=True, slots=True)
class DurableWebSearchSpec:
    """Configuration for one bounded durable search workflow."""

    workflow_id: str
    supported_goal: str
    start_url: str
    working_url_prefix: str
    expected_application: str
    query_text: str
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
    completion_summary: str
    query_verification_mode: QueryVerificationMode = (
        QueryVerificationMode.FIELD_VALUE
    )
    query_confirmation_targets: tuple[TargetSpec, ...] = ()


EXPECTED_APP = "Google Chrome"

PYTHON_URL = "https://www.python.org/"

PYTHON_WORKFLOW = DurableWebSearchSpec(
    workflow_id="python-org-search",
    supported_goal="Search python.org for typing.",
    start_url=PYTHON_URL,
    working_url_prefix=PYTHON_URL,
    expected_application=EXPECTED_APP,
    query_text="typing",
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
    completion_summary=(
        "python.org search for 'typing' completed and was verified."
    ),
)

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Main_Page"

WIKIPEDIA_WORKFLOW = DurableWebSearchSpec(
    workflow_id="wikipedia-search",
    supported_goal="Search Wikipedia for computer use agent.",
    start_url=WIKIPEDIA_URL,
    working_url_prefix="https://en.wikipedia.org/",
    expected_application=EXPECTED_APP,
    query_text="computer use agent",
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
    completion_summary=(
        "Wikipedia search for 'computer use agent' completed "
        "and was verified."
    ),
    query_verification_mode=(
        QueryVerificationMode.VISIBLE_SEARCH_UI
    ),
    query_confirmation_targets=(
        TargetSpec(
            text="computer use agent",
            element_types=("text",),
            minimum_confidence=0.70,
        ),
        TargetSpec(
            text="Search for pages containing computer use agent",
            element_types=("text",),
            minimum_confidence=0.70,
        ),
        TargetSpec(
            text="Search for pages containing computer use agent",
            element_types=("link",),
            minimum_confidence=0.70,
        ),
    ),
)

WORKFLOWS = (
    PYTHON_WORKFLOW,
    WIKIPEDIA_WORKFLOW,
)

SEARCH_QUERY = PYTHON_WORKFLOW.query_text
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

    def observe(
        self,
    ) -> TextInputObservation:
        time.sleep(
            self.stabilization_seconds
        )

        self.capture_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot = (
            self.perception_engine.observe()
        )

        return TextInputObservation(
            application_name=(
                self.accessibility
                .read_frontmost_application_name()
            ),
            viewport=(
                self.accessibility
                .read_frontmost_viewport()
            ),
            snapshot=snapshot,
            semantic_elements=tuple(
                self.accessibility
                .read_frontmost_semantic_elements()
            ),
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
    spec: DurableWebSearchSpec | None = None,
) -> StructuredPlan:
    """Build the first real-browser segment."""
    if spec is None:
        spec = resolve_live_web_workflow(
            goal
        )

    return StructuredPlan(
        task_goal=goal,
        steps=(
            WebTextInputStep(
                goal=spec.prepare_step_goal,
                target=spec.search_field,
                input_text=spec.query_text,
                max_attempts=1,
            ),
        ),
    )


def build_resume_plan(
    goal: str,
    spec: DurableWebSearchSpec | None = None,
) -> StructuredPlan:
    """Build the post-restart browser segment."""
    if spec is None:
        spec = resolve_live_web_workflow(
            goal
        )

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


def goal_is_supported(
    goal: str,
) -> bool:
    """Return whether the bounded live worker supports this task."""
    return (
        resolve_live_web_workflow(
            goal,
            raise_on_unsupported=False,
        )
        is not None
    )


def resolve_live_web_workflow(
    goal: str,
    *,
    raise_on_unsupported: bool = True,
) -> DurableWebSearchSpec | None:
    """Resolve a bounded user goal to a deterministic web workflow."""
    normalized = (
        " ".join(
            goal.lower().split()
        )
    )

    if (
        "search" in normalized
        and "python.org" in normalized
        and "typing" in normalized
    ):
        return PYTHON_WORKFLOW

    if (
        "search" in normalized
        and "wikipedia" in normalized
        and "computer use agent" in normalized
    ):
        return WIKIPEDIA_WORKFLOW

    if not raise_on_unsupported:
        return None

    supported = ", ".join(
        spec.supported_goal
        for spec in WORKFLOWS
    )
    raise RuntimeError(
        "The current live workspace supports only "
        f"these bounded tasks: {supported}"
    )


def create_live_web_worker(
    state: TaskState,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    *,
    capture_path: str | Path | None = None,
    environment_factory=LiveWebEnvironment,
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

    spec = resolve_live_web_workflow(
        state.goal
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
            spec,
        )

        _ensure_live_task_structure(
            transitions,
            spec,
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
            spec=spec,
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
    spec: DurableWebSearchSpec,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    """Converge the durable task state and live browser state."""
    control.checkpoint()

    observation = _ensure_browser_workspace(
        state=state,
        transitions=transitions,
        environment=environment,
        spec=spec,
        publish_state=publish_state,
        progress=progress,
    )

    observation = _reconcile_query_condition(
        state=state,
        transitions=transitions,
        environment=environment,
        spec=spec,
        progress=progress,
        observation=observation,
    )

    publish_state()

    _maybe_inject_process_crash(
        LiveCrashPoint.QUERY_CHECKPOINT
    )

    if not _results_visible(
        observation,
        spec,
    ):
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
                "The durable checkpoint is complete; "
                "execution can continue without waiting "
                "for a process restart."
            ),
        )

    control.checkpoint()

    final_observation = (
        _reconcile_submission_result_condition(
            state=state,
            transitions=transitions,
            environment=environment,
            spec=spec,
            control=control,
            publish_state=publish_state,
            publish_decision=publish_decision,
            progress=progress,
        )
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
            "Fresh live result evidence verified "
            "all semantic completion requirements."
        ),
        completion_summary=(
            spec.completion_summary
        ),
    )

    progress(
        "Real browser task completed."
    )


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
    spec: DurableWebSearchSpec,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    _require_expected_app(
        observation,
        spec,
    )

    query_verification = (
        _query_condition_satisfied(
            observation,
            spec,
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
        spec,
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
            spec,
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
            spec,
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
    spec: DurableWebSearchSpec,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> TextInputObservation:
    observation = environment.observe()
    _require_expected_app(
        observation,
        spec,
    )

    if _results_visible(
        observation,
        spec,
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
        spec,
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
            spec,
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
        spec,
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


def _results_visible(
    observation: TextInputObservation,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> bool:
    grounding = UIGrounder().ground(
        spec.result_target,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _query_condition_satisfied(
    observation: TextInputObservation,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> QueryVerificationResult | None:
    if (
        spec.query_verification_mode
        is QueryVerificationMode.FIELD_VALUE
    ):
        search_field = (
            _search_field_with_expected_value(
                observation.snapshot.fused_elements,
                spec=spec,
                expected_value=spec.query_text,
            )
        )

        if search_field is None:
            return None

        return QueryVerificationResult(
            element=search_field,
            summary=(
                "The current search field contains "
                f"{spec.query_text!r}."
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
            spec,
        )

    return None


def _visible_search_ui_query_condition(
    observation: TextInputObservation,
    spec: DurableWebSearchSpec,
) -> QueryVerificationResult | None:
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
        spec
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
                    "Fresh Wikipedia search UI "
                    f"visibly confirms {spec.query_text!r} "
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
    spec: DurableWebSearchSpec,
) -> tuple[TargetSpec, ...]:
    targets = (
        TargetSpec(
            text=spec.query_text,
            element_types=("text",),
            minimum_confidence=(
                spec.submit_target.minimum_confidence
            ),
        ),
        *spec.query_confirmation_targets,
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
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> bool:
    if _results_visible(
        observation,
        spec,
    ):
        return False

    if (
        _query_condition_satisfied(
            observation,
            spec,
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
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
        spec=spec,
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
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
        spec=spec,
        control=control,
        publish_state=publish_state,
        publish_decision=publish_decision,
        progress=progress,
    )


def _ensure_live_task_structure(
    transitions: TaskStateTransitions,
    spec: DurableWebSearchSpec = PYTHON_WORKFLOW,
) -> None:
    """Pre-register the complete live-task semantic structure."""
    state = transitions.state

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


def _workflow_location(
    spec: DurableWebSearchSpec,
) -> str:
    return f"workflow:{spec.workflow_id}"


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
    )
    return any(
        known_id in state.claims
        or known_id in state.subgoals
        or known_id in state.side_effects
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

    return (
        normalize_ui_text(
            str(value or "")
        )
        == expected
    )


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
