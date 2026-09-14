"""Phase 06 Experiment 07: continuous durable execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from computer_agent.agent import (
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    TextInputObservation,
)
from computer_agent.app.live_web_worker import (
    BROWSER_WINDOW_ARTIFACT_ID,
    BROWSER_WINDOW_MARKER_PREFIX,
    PYTHON_WORKFLOW,
    QUERY_CLAIM_ID,
    QUERY_SUBGOAL_ID,
    RESULT_CLAIM_ID,
    RESULT_SUBGOAL_ID,
    ResolvedDurableWebSearchTask,
    SUBMIT_ACTION_KEY,
    SUBMIT_SIDE_EFFECT_ID,
    create_live_web_worker,
    _ensure_query_identity,
    _ensure_live_task_structure,
)
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.planning import (
    PlanOperation,
    WebTextInputStep,
)
from computer_agent.runtime import (
    RuntimeTask,
)
from computer_agent.task import (
    ArtifactRecord,
    ClaimStatus,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
    prepare_state_for_resume,
)


TITLE = (
    "Phase 06 Experiment 07: "
    "Continuous Durable Execution"
)
GOAL = "Search python.org for typing."
SEARCH_QUERY = "typing"


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result of one deterministic browser reconciliation path."""

    completed: bool
    opened_workspace: bool
    reacquired_workspace: bool
    typed_count: int
    clicked_count: int
    saw_waiting_user: bool
    query_verified: bool
    result_verified: bool
    side_effect_confirmed: bool
    intermediate_publish_count: int
    final_decision_type: str | None


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Acceptance report for continuous durable execution."""

    uninterrupted_passed: bool
    restart_after_query_passed: bool
    restart_after_submit_passed: bool
    failures: tuple[str, ...]


class FakeControl:
    """Cooperative runtime control used by the headless experiment."""

    def checkpoint(self) -> None:
        return


class FakeLiveWebEnvironment:
    """Small deterministic browser model for python.org search states."""

    def __init__(
        self,
        *,
        capture_path,
        mode: str,
    ) -> None:
        del capture_path
        self.mode = mode
        self.open_count = 0
        self.activate_count = 0
        self.type_count = 0
        self.click_count = 0

    def open_task_site(
        self,
        start_url: str,
        task_marker: str,
    ) -> str:
        del start_url
        self.open_count += 1
        return (
            BROWSER_WINDOW_MARKER_PREFIX
            + task_marker
        )

    def activate_task_chrome_window(
        self,
        marker_url: str,
        working_url_prefix: str = "",
    ) -> None:
        del working_url_prefix
        if not marker_url.startswith(
            BROWSER_WINDOW_MARKER_PREFIX
        ):
            raise RuntimeError(
                "invalid task marker"
            )
        self.activate_count += 1

    def observe(self) -> TextInputObservation:
        return TextInputObservation(
            application_name="Google Chrome",
            viewport=None,
            snapshot=_snapshot(
                self._elements()
            ),
            semantic_elements=(),
        )

    def execute_plan(
        self,
        plan,
    ) -> AgentLoopResult:
        step = plan.steps[0]

        if isinstance(
            step,
            WebTextInputStep,
        ):
            self.type_count += 1
            self.mode = "query"
        elif (
            step.operation
            is PlanOperation.CLICK_TARGET
        ):
            self.click_count += 1
            self.mode = "results"
        else:
            raise RuntimeError(
                f"unexpected operation: {step!r}"
            )

        agent_state = AgentState(
            user_task=plan.task_goal
        )
        agent_state.start()
        agent_state.succeed()

        return AgentLoopResult(
            status=AgentLoopStatus.COMPLETED,
            plan=plan,
            state=agent_state,
            completed_plan_steps=1,
            reason="deterministic fake execution",
        )

    def _elements(
        self,
    ) -> tuple[UIElement, ...]:
        if self.mode == "empty":
            return (
                _field("Search This Site", ""),
                _button("GO"),
            )

        if self.mode == "query":
            return (
                _field(
                    "Search This Site",
                    SEARCH_QUERY,
                ),
                _button("GO"),
            )

        if self.mode == "results":
            return (
                _field(
                    "Search This Site",
                    SEARCH_QUERY,
                ),
                _button("GO"),
                _element(
                    text="Results",
                    value=None,
                    element_type="heading",
                ),
            )

        raise RuntimeError(
            f"unknown fake browser mode: {self.mode}"
        )


def run_experiment() -> ExperimentReport:
    """Run the three deterministic durable execution scenarios."""
    failures: list[str] = []

    uninterrupted = _run_uninterrupted()
    restart_after_query = _run_restart_after_query()
    restart_after_submit = _run_restart_after_submit()

    uninterrupted_passed = (
        uninterrupted.completed
        and uninterrupted.opened_workspace
        and not uninterrupted.reacquired_workspace
        and uninterrupted.typed_count == 1
        and uninterrupted.clicked_count == 1
        and not uninterrupted.saw_waiting_user
        and uninterrupted.query_verified
        and uninterrupted.result_verified
        and uninterrupted.side_effect_confirmed
        and uninterrupted.intermediate_publish_count >= 1
    )
    if not uninterrupted_passed:
        failures.append(
            "uninterrupted execution did not "
            "continue through checkpoints"
        )

    restart_after_query_passed = (
        restart_after_query.completed
        and not restart_after_query.opened_workspace
        and restart_after_query.reacquired_workspace
        and restart_after_query.typed_count == 0
        and restart_after_query.clicked_count == 1
        and restart_after_query.query_verified
        and restart_after_query.result_verified
        and restart_after_query.side_effect_confirmed
    )
    if not restart_after_query_passed:
        failures.append(
            "restart after query checkpoint did "
            "not reverify query without retyping"
        )

    restart_after_submit_passed = (
        restart_after_submit.completed
        and not restart_after_submit.opened_workspace
        and restart_after_submit.reacquired_workspace
        and restart_after_submit.typed_count == 0
        and restart_after_submit.clicked_count == 0
        and restart_after_submit.query_verified
        and restart_after_submit.result_verified
        and restart_after_submit.side_effect_confirmed
        and restart_after_submit.final_decision_type == "COMPLETE"
    )
    if not restart_after_submit_passed:
        failures.append(
            "restart after GO click did not "
            "reconcile Results without clicking"
        )

    return ExperimentReport(
        uninterrupted_passed=uninterrupted_passed,
        restart_after_query_passed=(
            restart_after_query_passed
        ),
        restart_after_submit_passed=(
            restart_after_submit_passed
        ),
        failures=tuple(failures),
    )


def print_report(
    report: ExperimentReport,
) -> None:
    """Print a compact human-readable report."""
    print(TITLE)
    print(
        "Uninterrupted continuous run: "
        f"{_result(report.uninterrupted_passed)}"
    )
    print(
        "Restart after query checkpoint: "
        f"{_result(report.restart_after_query_passed)}"
    )
    print(
        "Restart after GO before Results checkpoint: "
        f"{_result(report.restart_after_submit_passed)}"
    )
    print(
        "Experiment acceptance: "
        f"{'passed' if not report.failures else 'failed'}"
    )

    for failure in report.failures:
        print(
            f"Failure: {failure}"
        )


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 0 if not report.failures else 1


def _run_uninterrupted() -> ScenarioResult:
    state = TaskState(
        goal=GOAL
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
    )
    return _run_scenario(
        state,
        environment,
    )


def _run_restart_after_query() -> ScenarioResult:
    state = _state_with_workspace(
        task_id="phase06-07-query"
    )
    transitions = TaskStateTransitions(
        state
    )
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains typing.",
            source="pre-crash observation",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        QUERY_CLAIM_ID,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
    )
    prepare_state_for_resume(
        state
    )

    return _run_scenario(
        state,
        FakeLiveWebEnvironment(
            capture_path=None,
            mode="query",
        ),
    )


def _run_restart_after_submit() -> ScenarioResult:
    state = _state_with_workspace(
        task_id="phase06-07-submit"
    )
    transitions = TaskStateTransitions(
        state
    )
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains typing.",
            source="pre-crash observation",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        QUERY_CLAIM_ID,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        QUERY_SUBGOAL_ID
    )
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                SUBMIT_SIDE_EFFECT_ID
            ),
            description=(
                "Submit the python.org search "
                "query through the GO control."
            ),
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=SUBMIT_ACTION_KEY,
        )
    )
    prepare_state_for_resume(
        state
    )

    return _run_scenario(
        state,
        FakeLiveWebEnvironment(
            capture_path=None,
            mode="results",
        ),
    )


def _run_scenario(
    state: TaskState,
    environment: FakeLiveWebEnvironment,
) -> ScenarioResult:
    published: list[TaskStateStatus] = []
    decisions = []

    worker = create_live_web_worker(
        state,
        lambda: published.append(
            state.status
        ),
        decisions.append,
        environment_factory=lambda *, capture_path: environment,
    )

    worker(
        RuntimeTask(
            goal=state.goal,
            task_id=state.task_id,
        ),
        FakeControl(),
        lambda message: None,
    )

    effect = state.side_effects.get(
        SUBMIT_SIDE_EFFECT_ID
    )

    return ScenarioResult(
        completed=(
            state.status
            is TaskStateStatus.COMPLETED
        ),
        opened_workspace=(
            environment.open_count == 1
        ),
        reacquired_workspace=(
            environment.activate_count == 1
        ),
        typed_count=environment.type_count,
        clicked_count=environment.click_count,
        saw_waiting_user=(
            TaskStateStatus.WAITING_USER
            in published
        ),
        query_verified=(
            state.claims[QUERY_CLAIM_ID].status
            is ClaimStatus.VERIFIED
            and state.subgoals[
                QUERY_SUBGOAL_ID
            ].status
            is SubgoalStatus.VERIFIED
        ),
        result_verified=(
            state.claims[RESULT_CLAIM_ID].status
            is ClaimStatus.VERIFIED
            and state.subgoals[
                RESULT_SUBGOAL_ID
            ].status
            is SubgoalStatus.VERIFIED
        ),
        side_effect_confirmed=(
            effect is not None
            and effect.state
            is SideEffectState.CONFIRMED
        ),
        intermediate_publish_count=sum(
            1
            for status in published
            if status is TaskStateStatus.RUNNING
        ),
        final_decision_type=(
            None
            if not decisions
            else decisions[-1].final_decision_type
        ),
    )


def _state_with_workspace(
    *,
    task_id: str,
) -> TaskState:
    state = TaskState(
        goal=GOAL,
        task_id=task_id,
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_query_identity(
        transitions,
        ResolvedDurableWebSearchTask(
            spec=PYTHON_WORKFLOW,
            query_text=SEARCH_QUERY,
        ),
    )
    _ensure_live_task_structure(
        transitions
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
                BROWSER_WINDOW_MARKER_PREFIX
                + state.task_id
            ),
        )
    )
    return state


def _snapshot(
    elements: tuple[UIElement, ...],
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("fake.png"),
            pixel_width=10,
            pixel_height=10,
            screen_width=10,
            screen_height=10,
        ),
        image=Image.new(
            "RGB",
            (10, 10),
        ),
        accessibility_elements=elements,
        ocr_elements=(),
        fused_elements=elements,
        warnings=(),
    )


def _field(
    text: str,
    value: str,
) -> UIElement:
    return _element(
        text=text,
        value=value,
        element_type="text_field",
    )


def _button(
    text: str,
) -> UIElement:
    return _element(
        text=text,
        value=None,
        element_type="button",
    )


def _element(
    *,
    text: str | None,
    value: str | None,
    element_type: str,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        text=text,
        value=value,
        confidence=0.95,
        enabled=True,
        bounding_box=BoundingBox(
            x=10,
            y=10,
            width=100,
            height=20,
        ),
        source="accessibility",
    )


def _result(
    value: bool,
) -> str:
    return "passed" if value else "failed"


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
