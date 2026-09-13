"""Bounded live-web worker for the persistent Agent Workspace."""

from __future__ import annotations

from collections.abc import Callable
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


PYTHON_URL = "https://www.python.org/"
EXPECTED_APP = "Google Chrome"

SEARCH_QUERY = "typing"

SEARCH_FIELD = TargetSpec(
    text="Search This Site",
    element_types=("text_field",),
    minimum_confidence=0.70,
)

GO_BUTTON = TargetSpec(
    text="GO",
    element_types=("button",),
    minimum_confidence=0.70,
)

RESULTS_TARGET = TargetSpec(
    text="Results",
    element_types=(),
    minimum_confidence=0.70,
)

QUERY_CLAIM_ID = "live-web-query-entered"
QUERY_SUBGOAL_ID = "live-web-enter-query"

RESULT_CLAIM_ID = "live-web-results-visible"
RESULT_SUBGOAL_ID = "live-web-submit-query"

SUBMIT_SIDE_EFFECT_ID = (
    "live-web-submit-query-side-effect"
)
SUBMIT_ACTION_KEY = (
    "click_target:python_org_search_go"
)

BROWSER_WINDOW_ARTIFACT_ID = (
    "live-web-browser-window"
)

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

    def open_python_org(
        self,
        task_marker: str,
    ) -> str:
        marker_url = (
            self.controller
            .open_task_chrome_window(
                PYTHON_URL,
                task_marker,
            )
        )

        time.sleep(1.0)

        return marker_url

    def activate_task_chrome_window(
        self,
        marker_url: str,
    ) -> None:
        self.controller.activate_task_chrome_window(
            marker_url,
            PYTHON_URL,
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
) -> StructuredPlan:
    """Build the first real-browser segment."""
    return StructuredPlan(
        task_goal=goal,
        steps=(
            WebTextInputStep(
                goal=(
                    "Enter the intended search "
                    "query on python.org."
                ),
                target=SEARCH_FIELD,
                input_text=SEARCH_QUERY,
                max_attempts=1,
            ),
        ),
    )


def build_resume_plan(
    goal: str,
) -> StructuredPlan:
    """Build the post-restart browser segment."""
    return StructuredPlan(
        task_goal=goal,
        steps=(
            PlanStep(
                goal=(
                    "Submit the persisted "
                    "python.org search query."
                ),
                operation=(
                    PlanOperation.CLICK_TARGET
                ),
                action_target=GO_BUTTON,
                verification_target=(
                    RESULTS_TARGET
                ),
                max_attempts=1,
            ),
        ),
    )


def goal_is_supported(
    goal: str,
) -> bool:
    """Return whether the bounded live worker supports this task."""
    normalized = (
        " ".join(
            goal.lower().split()
        )
    )

    return (
        "search" in normalized
        and "python.org" in normalized
        and "typing" in normalized
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

    if not goal_is_supported(
        state.goal
    ):
        raise RuntimeError(
            "The current live workspace "
            "supports only the bounded task: "
            "'Search python.org for typing.'"
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

        _ensure_live_task_structure(
            transitions
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
        publish_state=publish_state,
        progress=progress,
    )

    observation = _reconcile_query_condition(
        state=state,
        transitions=transitions,
        environment=environment,
        progress=progress,
        observation=observation,
    )

    publish_state()

    _maybe_inject_process_crash(
        LiveCrashPoint.QUERY_CHECKPOINT
    )

    if not _results_visible(
        observation
    ):
        _publish_live_decision(
            publish_decision,
            observation=observation,
            decision_type="ACTION",
            operation="click_target",
            target_text="GO",
            expected_effect=(
                "Submit the verified 'typing' query "
                "and navigate to python.org Results."
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
            "Fresh live Results evidence verified "
            "all semantic completion requirements."
        ),
        completion_summary=(
            "python.org search for 'typing' "
            "completed and was verified."
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
    publish_state: TaskStatePublisher,
    progress: Callable[[str], None],
) -> TextInputObservation:
    if _should_prepare_live_task_segment(
        state
    ):
        progress(
            "Opening real python.org "
            "in a dedicated Chrome task window."
        )

        browser_window_marker = (
            environment.open_python_org(
                state.task_id
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
                state
            )
        )

        progress(
            "Re-activating the persisted "
            "Agent-owned Chrome task window."
        )

        environment.activate_task_chrome_window(
            browser_window_marker
        )

    observation = environment.observe()
    _require_expected_app(
        observation
    )
    return observation


def _reconcile_query_condition(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    progress: Callable[[str], None],
    observation: TextInputObservation,
) -> TextInputObservation:
    _require_expected_app(
        observation
    )

    search_field = (
        _search_field_with_expected_value(
            observation.snapshot.fused_elements,
            expected_value=SEARCH_QUERY,
        )
    )

    if search_field is not None:
        _verify_query_from_current_observation(
            transitions,
            summary=(
                "Fresh Chrome Accessibility "
                "shows query 'typing' in the "
                "python.org search field."
            ),
            source=(
                "Live Chrome Accessibility "
                "observation"
            ),
        )
        progress(
            "Fresh browser state verifies "
            "the intended query."
        )
        return observation

    if _results_visible(
        observation
    ):
        if not _query_has_durable_history(
            state
        ):
            raise RuntimeError(
                "Results are visible, but the "
                "task has no durable query history "
                "to reconcile."
            )

        _verify_query_from_current_observation(
            transitions,
            summary=(
                "Fresh Results state reconciles "
                "the previously verified "
                "'typing' query after restart."
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
        observation
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
            state.goal
        ),
    )

    _require_agent_success(
        result,
        "real-web query entry",
    )

    refreshed = environment.observe()
    _require_expected_app(
        refreshed
    )

    if (
        _search_field_with_expected_value(
            refreshed.snapshot.fused_elements,
            expected_value=SEARCH_QUERY,
        )
        is None
    ):
        raise RuntimeError(
            "The live search field did not "
            "uniquely verify the intended query."
        )

    _verify_query_from_current_observation(
        transitions,
        summary=(
            "The current real python.org "
            "search field contains 'typing'."
        ),
        source=(
            "Live Chrome Accessibility "
            "observation"
        ),
    )

    return refreshed


def _reconcile_submission_result_condition(
    *,
    state: TaskState,
    transitions: TaskStateTransitions,
    environment: LiveWebEnvironment,
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> TextInputObservation:
    observation = environment.observe()
    _require_expected_app(
        observation
    )

    if _results_visible(
        observation
    ):
        evidence = _verify_results_from_current_observation(
            transitions,
            summary=(
                "Fresh Chrome Accessibility "
                "shows the python.org Results "
                "marker."
            ),
            source=(
                "Live Chrome Accessibility "
                "observation"
            ),
        )
        _confirm_submit_side_effect(
            transitions,
            evidence.evidence_id,
        )
        publish_state()
        progress(
            "Fresh browser state already "
            "shows Results; GO was not clicked."
        )
        return observation

    if not _pre_submit_query_state(
        observation
    ):
        raise RuntimeError(
            "The live browser state is ambiguous: "
            "it is neither verified pre-submit "
            "search state nor verified Results."
        )

    _ensure_submit_side_effect(
        transitions
    )
    publish_state()

    _mark_submit_execution_attempt(
        transitions
    )
    publish_state()

    _publish_live_decision(
        publish_decision,
        observation=observation,
        decision_type="ACTION",
        operation="click_target",
        target_text="GO",
        expected_effect=(
            "Submit the verified 'typing' query "
            "and navigate to python.org Results."
        ),
        reason=(
            "Fresh browser observation confirms "
            "the pre-submit state with query "
            "'typing'."
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
            state.goal
        ),
    )

    _require_agent_success(
        result,
        "real-web search submission",
    )

    _maybe_inject_process_crash(
        LiveCrashPoint.SUBMIT_EXECUTION
    )

    final_observation = environment.observe()
    _require_expected_app(
        final_observation
    )

    if not _results_visible(
        final_observation
    ):
        _mark_submit_outcome_unknown(
            transitions
        )
        publish_state()
        raise RuntimeError(
            "Fresh post-submit observation "
            "did not resolve Results."
        )

    results_evidence = (
        _verify_results_from_current_observation(
            transitions,
            summary=(
                "The real python.org Results "
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
        QUERY_CLAIM_ID,
        (evidence.evidence_id,),
    )

    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
    )

    return evidence


def _verify_results_from_current_observation(
    transitions: TaskStateTransitions,
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
        RESULT_CLAIM_ID,
        (evidence.evidence_id,),
    )

    transitions.verify_subgoal(
        RESULT_SUBGOAL_ID
    )

    return evidence


def _ensure_submit_side_effect(
    transitions: TaskStateTransitions,
) -> SideEffectRecord:
    state = transitions.state
    effect = state.side_effects.get(
        SUBMIT_SIDE_EFFECT_ID
    )

    if effect is not None:
        return effect

    return transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                SUBMIT_SIDE_EFFECT_ID
            ),
            description=(
                "Submit the python.org search "
                "query through the GO control."
            ),
            external_reference=PYTHON_URL,
            idempotent=True,
            action_key=SUBMIT_ACTION_KEY,
        )
    )


def _mark_submit_execution_attempt(
    transitions: TaskStateTransitions,
) -> None:
    effect = _ensure_submit_side_effect(
        transitions
    )

    if (
        effect.state
        is SideEffectState.INTENDED
    ):
        transitions.mark_side_effect_executed(
            SUBMIT_SIDE_EFFECT_ID
        )


def _mark_submit_outcome_unknown(
    transitions: TaskStateTransitions,
) -> None:
    effect = transitions.state.side_effects.get(
        SUBMIT_SIDE_EFFECT_ID
    )

    if (
        effect is not None
        and effect.state
        is SideEffectState.EXECUTED
    ):
        transitions.mark_side_effect_unknown(
            SUBMIT_SIDE_EFFECT_ID
        )


def _confirm_submit_side_effect(
    transitions: TaskStateTransitions,
    evidence_id: str,
) -> None:
    effect = transitions.state.side_effects.get(
        SUBMIT_SIDE_EFFECT_ID
    )

    if effect is None:
        transitions.add_side_effect(
            SideEffectRecord(
                side_effect_id=(
                    SUBMIT_SIDE_EFFECT_ID
                ),
                description=(
                    "Submit the python.org search "
                    "query through the GO control."
                ),
                state=(
                    SideEffectState.CONFIRMED
                ),
                external_reference=PYTHON_URL,
                evidence_ids=(evidence_id,),
                idempotent=True,
                action_key=SUBMIT_ACTION_KEY,
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
            SUBMIT_SIDE_EFFECT_ID
        )

    transitions.confirm_side_effect(
        SUBMIT_SIDE_EFFECT_ID,
        (evidence_id,),
    )


def _results_visible(
    observation: TextInputObservation,
) -> bool:
    grounding = UIGrounder().ground(
        RESULTS_TARGET,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _search_field_available(
    observation: TextInputObservation,
) -> bool:
    grounding = UIGrounder().ground(
        SEARCH_FIELD,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _pre_submit_query_state(
    observation: TextInputObservation,
) -> bool:
    if _results_visible(
        observation
    ):
        return False

    if (
        _search_field_with_expected_value(
            observation.snapshot.fused_elements,
            expected_value=SEARCH_QUERY,
        )
        is None
    ):
        return False

    grounding = UIGrounder().ground(
        GO_BUTTON,
        observation.snapshot.fused_elements,
    )

    return (
        grounding.status
        is GroundingStatus.RESOLVED
    )


def _query_has_durable_history(
    state: TaskState,
) -> bool:
    claim = state.claims.get(
        QUERY_CLAIM_ID
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
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
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
    control: RuntimeControl,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> None:
    _run_live_task(
        state=state,
        transitions=transitions,
        environment=environment,
        control=control,
        publish_state=publish_state,
        publish_decision=publish_decision,
        progress=progress,
    )


def _ensure_live_task_structure(
    transitions: TaskStateTransitions,
) -> None:
    """Pre-register the complete live-task semantic structure."""
    state = transitions.state

    if QUERY_CLAIM_ID not in state.claims:
        transitions.add_claim(
            ClaimRecord(
                claim_id=QUERY_CLAIM_ID,
                statement=(
                    "The intended python.org "
                    "query has been entered."
                ),
            )
        )

    if RESULT_CLAIM_ID not in state.claims:
        transitions.add_claim(
            ClaimRecord(
                claim_id=RESULT_CLAIM_ID,
                statement=(
                    "The python.org search "
                    "completed successfully."
                ),
            )
        )

    if QUERY_SUBGOAL_ID not in state.subgoals:
        transitions.add_subgoal(
            SubgoalRecord(
                subgoal_id=QUERY_SUBGOAL_ID,
                description=(
                    "Enter the intended "
                    "python.org search query."
                ),
                claim_ids=(
                    QUERY_CLAIM_ID,
                ),
            )
        )

    if RESULT_SUBGOAL_ID not in state.subgoals:
        transitions.add_subgoal(
            SubgoalRecord(
                subgoal_id=RESULT_SUBGOAL_ID,
                description=(
                    "Submit the search and "
                    "verify the results page."
                ),
                claim_ids=(
                    RESULT_CLAIM_ID,
                ),
            )
        )


def _should_prepare_live_task_segment(
    state: TaskState,
) -> bool:
    """Return whether this run still needs to create the browser workspace."""
    return (
        BROWSER_WINDOW_ARTIFACT_ID
        not in state.artifacts
    )


def _browser_window_marker_from_state(
    state: TaskState,
) -> str:
    artifact = state.artifacts.get(
        BROWSER_WINDOW_ARTIFACT_ID
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
    expected_value: str,
) -> UIElement | None:
    """Find the current search field only when its live value matches."""
    element_tuple = tuple(elements)
    expected = normalize_ui_text(
        expected_value
    )

    grounding = UIGrounder().ground(
        SEARCH_FIELD,
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
        >= SEARCH_FIELD.minimum_confidence
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
) -> None:
    if (
        observation.application_name
        != EXPECTED_APP
    ):
        raise RuntimeError(
            "Frontmost application was "
            f"{observation.application_name!r}; "
            f"expected {EXPECTED_APP!r}."
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
