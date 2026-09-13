"""Bounded live-web worker for the persistent Agent Workspace."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

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

BROWSER_WINDOW_ARTIFACT_ID = (
    "live-web-browser-window"
)

BROWSER_WINDOW_MARKER_PREFIX = (
    "about:blank#computer-agent-task="
)


TaskStatePublisher = Callable[[], None]
AdaptiveDecisionPublisher = Callable[
    [object],
    None,
]


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

        if _should_prepare_live_task_segment(
            state
        ):
            _run_prepare_segment(
                state=state,
                transitions=transitions,
                environment=environment,
                control=control,
                publish_state=(
                    publish_state
                ),
                publish_decision=(
                    publish_decision
                ),
                progress=progress,
            )
            return

        _run_resume_segment(
            state=state,
            transitions=transitions,
            environment=environment,
            control=control,
            publish_state=publish_state,
            publish_decision=publish_decision,
            progress=progress,
        )

    return worker


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
    control.checkpoint()

    progress(
        "Opening real python.org "
        "in Google Chrome."
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

    progress(
        "Executing verified real-web "
        "text input."
    )

    result = AgentLoop(
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
        build_prepare_plan(
            state.goal
        )
    )

    _require_agent_success(
        result,
        "real-web query entry",
    )

    observation = (
        environment.observe()
    )

    _require_expected_app(
        observation
    )

    search_field = (
        _search_field_with_expected_value(
            observation.snapshot.fused_elements,
            expected_value=SEARCH_QUERY,
        )
    )

    if search_field is None:
        raise RuntimeError(
            "The live search field did not "
            "uniquely verify the intended query."
        )

    evidence = EvidenceRecord(
        summary=(
            "The current real python.org "
            "search field contains 'typing'."
        ),
        source=(
            "Live Chrome Accessibility "
            "observation"
        ),
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

    _publish_live_decision(
        publish_decision,
        observation=observation,
        decision_type="WAIT",
        operation="resume_after_restart",
        target_text="Resume Last Task",
        expected_effect=(
            "Reload the persistent checkpoint, "
            "take a fresh browser observation, "
            "and continue toward search submission."
        ),
        reason=(
            "The live query was verified and "
            "the experiment intentionally pauses "
            "at the persistent process boundary."
        ),
    )

    state.status = (
        TaskStateStatus.WAITING_USER
    )
    state.touch()

    publish_state()

    progress(
        "Persistent checkpoint saved. "
        "Close the Computer Agent app now "
        "to test a real process restart."
    )

    progress(
        "Do not click GO manually. "
        "After reopening the app, use "
        "Resume Last Task."
    )

    while True:
        control.checkpoint()
        time.sleep(0.10)


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
    control.checkpoint()

    progress(
        "Loaded the persisted task. "
        "Activating Chrome."
    )

    browser_window_marker = (
        _browser_window_marker_from_state(
            state
        )
    )

    environment.activate_task_chrome_window(
        browser_window_marker
    )

    progress(
        "Re-activated the persisted "
        "Agent-owned Chrome task window."
    )

    progress(
        "Re-observing the real browser "
        "instead of reusing old coordinates."
    )

    observation = (
        environment.observe()
    )

    _require_expected_app(
        observation
    )

    search_field = (
        _search_field_with_expected_value(
            observation.snapshot.fused_elements,
            expected_value=SEARCH_QUERY,
        )
    )

    if search_field is None:
        raise RuntimeError(
            "Restart recovery could not "
            "uniquely verify the live search value."
        )

    fresh_query_evidence = (
        EvidenceRecord(
            summary=(
                "After process restart, "
                "fresh Chrome Accessibility "
                "still shows query 'typing'."
            ),
            source=(
                "Post-restart live Chrome "
                "observation"
            ),
            kind=(
                EvidenceKind.VERIFICATION
            ),
        )
    )

    transitions.add_evidence(
        fresh_query_evidence
    )

    transitions.verify_claim(
        QUERY_CLAIM_ID,
        (
            fresh_query_evidence
            .evidence_id,
        ),
    )

    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
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
            "and navigate to python.org results."
        ),
        reason=(
            "Fresh post-restart Accessibility "
            "evidence confirms the intended query "
            "is still present."
        ),
    )

    control.checkpoint()

    progress(
        "Fresh query evidence verified. "
        "Submitting through newly grounded "
        "live UI coordinates."
    )

    result = AgentLoop(
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
        build_resume_plan(
            state.goal
        )
    )

    _require_agent_success(
        result,
        "real-web resumed submission",
    )

    final_observation = (
        environment.observe()
    )

    _require_expected_app(
        final_observation
    )

    result_grounding = (
        UIGrounder().ground(
            RESULTS_TARGET,
            final_observation.snapshot
            .fused_elements,
        )
    )

    if (
        result_grounding.status
        is not GroundingStatus.RESOLVED
    ):
        raise RuntimeError(
            "Fresh post-submit observation "
            "did not resolve Results."
        )

    results_evidence = EvidenceRecord(
        summary=(
            "The real python.org Results "
            "marker is visible after resumed "
            "submission."
        ),
        source=(
            "Post-submit live Chrome "
            "observation"
        ),
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        results_evidence
    )

    transitions.verify_claim(
        RESULT_CLAIM_ID,
        (
            results_evidence
            .evidence_id,
        ),
    )

    transitions.verify_subgoal(
        RESULT_SUBGOAL_ID
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
        "Real browser task completed "
        "after process restart."
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
