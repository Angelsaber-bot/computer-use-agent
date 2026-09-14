"""Phase 06 Experiment 07.02: configurable durable web workflow."""

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
    BROWSER_WINDOW_MARKER_PREFIX,
    DurableWebSearchSpec,
    PYTHON_WORKFLOW,
    QueryVerificationMode,
    ResolvedDurableWebSearchTask,
    WIKIPEDIA_WORKFLOW,
    WORKFLOW_ARTIFACT_ID,
    compile_durable_task_plan,
    create_live_web_worker,
    _ensure_durable_plan_identity,
    _ensure_live_task_structure,
    _ensure_query_identity,
    _ensure_workflow_identity,
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
    "Phase 06 Experiment 07.02: "
    "Configurable Durable Web Workflow"
)
PYTHON_GOAL = "Search python.org for typing."
PYTHON_QUERY = "typing"
WIKIPEDIA_GOAL = (
    "Search Wikipedia for computer use agent."
)
WIKIPEDIA_QUERY = "computer use agent"


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result of one deterministic browser reconciliation path."""

    completed: bool
    opened_workspace: bool
    reacquired_workspace: bool
    typed_count: int
    clicked_count: int
    query_verified: bool
    result_verified: bool
    side_effect_confirmed: bool
    side_effect_action_key: str | None
    workflow_identity: str | None
    failed_closed: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Acceptance report for configurable durable web workflows."""

    python_uninterrupted_passed: bool
    wikipedia_uninterrupted_passed: bool
    wikipedia_restart_after_query_passed: bool
    wikipedia_restart_after_submit_passed: bool
    workflow_mismatch_passed: bool
    failures: tuple[str, ...]


class FakeControl:
    """Cooperative runtime control used by the headless experiment."""

    def checkpoint(self) -> None:
        return


class FakeLiveWebEnvironment:
    """Small deterministic browser model driven by a workflow spec."""

    def __init__(
        self,
        *,
        capture_path,
        mode: str,
        spec: DurableWebSearchSpec,
        query_text: str,
    ) -> None:
        del capture_path
        self.mode = mode
        self.spec = spec
        self.query_text = query_text
        self.open_count = 0
        self.activate_count = 0
        self.type_count = 0
        self.click_count = 0

    def open_task_site(
        self,
        start_url: str,
        task_marker: str,
    ) -> str:
        if start_url != self.spec.start_url:
            raise RuntimeError(
                "unexpected workflow start URL"
            )

        self.open_count += 1
        return (
            BROWSER_WINDOW_MARKER_PREFIX
            + task_marker
        )

    def activate_task_chrome_window(
        self,
        marker_url: str,
        working_url_prefix: str,
    ) -> None:
        if not marker_url.startswith(
            BROWSER_WINDOW_MARKER_PREFIX
        ):
            raise RuntimeError(
                "invalid task marker"
            )

        if working_url_prefix != self.spec.working_url_prefix:
            raise RuntimeError(
                "unexpected workflow working URL prefix"
            )

        self.activate_count += 1

    def observe(self) -> TextInputObservation:
        return TextInputObservation(
            application_name=self.spec.expected_application,
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
        submit_box = BoundingBox(
            x=430,
            y=10,
            width=70,
            height=24,
        )

        if self.mode == "empty":
            return (
                _field(
                    self.spec.search_field.text or "",
                    "",
                ),
                _button(
                    self.spec.submit_target.text or "",
                    bounding_box=submit_box,
                ),
            )

        if self.mode == "query":
            if (
                self.spec.query_verification_mode
                is QueryVerificationMode.VISIBLE_SEARCH_UI
            ):
                return (
                    _element(
                        text=self.query_text,
                        value=self.query_text,
                        element_type="text",
                        bounding_box=BoundingBox(
                            x=120,
                            y=12,
                            width=200,
                            height=20,
                        ),
                    ),
                    _button(
                        self.spec.submit_target.text
                        or "",
                        bounding_box=submit_box,
                    ),
                )

            return (
                _field(
                    self.spec.search_field.text or "",
                    self.query_text,
                ),
                _button(
                    self.spec.submit_target.text or "",
                    bounding_box=submit_box,
                ),
            )

        if self.mode == "results":
            return (
                _field(
                    self.spec.search_field.text or "",
                    self.query_text,
                ),
                _button(
                    self.spec.submit_target.text or "",
                    bounding_box=submit_box,
                ),
                _element(
                    text=(
                        self.spec.result_target.text
                        or ""
                    ),
                    value=None,
                    element_type="heading",
                ),
            )

        raise RuntimeError(
            f"unknown fake browser mode: {self.mode}"
        )


def run_experiment() -> ExperimentReport:
    """Run deterministic configurable durable workflow scenarios."""
    failures: list[str] = []

    python_uninterrupted = _run_uninterrupted(
        PYTHON_GOAL,
        PYTHON_WORKFLOW,
    )
    wikipedia_uninterrupted = _run_uninterrupted(
        WIKIPEDIA_GOAL,
        WIKIPEDIA_WORKFLOW,
    )
    wikipedia_restart_after_query = (
        _run_restart_after_query(
            WIKIPEDIA_GOAL,
            WIKIPEDIA_WORKFLOW,
        )
    )
    wikipedia_restart_after_submit = (
        _run_restart_after_submit(
            WIKIPEDIA_GOAL,
            WIKIPEDIA_WORKFLOW,
        )
    )
    workflow_mismatch = _run_workflow_mismatch()

    python_uninterrupted_passed = _completed_with_one_type_and_submit(
        python_uninterrupted,
        PYTHON_WORKFLOW,
    )
    if not python_uninterrupted_passed:
        failures.append(
            "python.org uninterrupted configurable workflow failed"
        )

    wikipedia_uninterrupted_passed = _completed_with_one_type_and_submit(
        wikipedia_uninterrupted,
        WIKIPEDIA_WORKFLOW,
    )
    if not wikipedia_uninterrupted_passed:
        failures.append(
            "Wikipedia uninterrupted configurable workflow failed"
        )

    wikipedia_restart_after_query_passed = (
        wikipedia_restart_after_query.completed
        and not wikipedia_restart_after_query.opened_workspace
        and wikipedia_restart_after_query.reacquired_workspace
        and wikipedia_restart_after_query.typed_count == 0
        and wikipedia_restart_after_query.clicked_count == 1
        and wikipedia_restart_after_query.query_verified
        and wikipedia_restart_after_query.result_verified
        and wikipedia_restart_after_query.side_effect_confirmed
    )
    if not wikipedia_restart_after_query_passed:
        failures.append(
            "Wikipedia restart after query checkpoint retyped or failed"
        )

    wikipedia_restart_after_submit_passed = (
        wikipedia_restart_after_submit.completed
        and not wikipedia_restart_after_submit.opened_workspace
        and wikipedia_restart_after_submit.reacquired_workspace
        and wikipedia_restart_after_submit.typed_count == 0
        and wikipedia_restart_after_submit.clicked_count == 0
        and wikipedia_restart_after_submit.query_verified
        and wikipedia_restart_after_submit.result_verified
        and wikipedia_restart_after_submit.side_effect_confirmed
    )
    if not wikipedia_restart_after_submit_passed:
        failures.append(
            "Wikipedia restart after submit clicked again or failed"
        )

    workflow_mismatch_passed = (
        workflow_mismatch.failed_closed
        and not workflow_mismatch.opened_workspace
        and workflow_mismatch.typed_count == 0
        and workflow_mismatch.clicked_count == 0
    )
    if not workflow_mismatch_passed:
        failures.append(
            "workflow identity mismatch did not fail closed"
        )

    return ExperimentReport(
        python_uninterrupted_passed=python_uninterrupted_passed,
        wikipedia_uninterrupted_passed=wikipedia_uninterrupted_passed,
        wikipedia_restart_after_query_passed=(
            wikipedia_restart_after_query_passed
        ),
        wikipedia_restart_after_submit_passed=(
            wikipedia_restart_after_submit_passed
        ),
        workflow_mismatch_passed=workflow_mismatch_passed,
        failures=tuple(failures),
    )


def print_report(
    report: ExperimentReport,
) -> None:
    """Print a compact human-readable report."""
    print(TITLE)
    print(
        "Python uninterrupted: "
        f"{_result(report.python_uninterrupted_passed)}"
    )
    print(
        "Wikipedia uninterrupted: "
        f"{_result(report.wikipedia_uninterrupted_passed)}"
    )
    print(
        "Wikipedia restart after query checkpoint: "
        f"{_result(report.wikipedia_restart_after_query_passed)}"
    )
    print(
        "Wikipedia restart after submit before result checkpoint: "
        f"{_result(report.wikipedia_restart_after_submit_passed)}"
    )
    print(
        "Workflow identity mismatch: "
        f"{_result(report.workflow_mismatch_passed)}"
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


def _run_uninterrupted(
    goal: str,
    spec: DurableWebSearchSpec,
) -> ScenarioResult:
    query_text = _query_for_spec(
        spec
    )
    return _run_scenario(
        TaskState(
            goal=goal
        ),
        FakeLiveWebEnvironment(
            capture_path=None,
            mode="empty",
            spec=spec,
            query_text=query_text,
        ),
        spec,
    )


def _run_restart_after_query(
    goal: str,
    spec: DurableWebSearchSpec,
) -> ScenarioResult:
    query_text = _query_for_spec(
        spec
    )
    state = _state_with_workspace(
        goal=goal,
        spec=spec,
        query_text=query_text,
        task_id=f"{spec.workflow_id}-query",
    )
    transitions = TaskStateTransitions(
        state
    )
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains configured query.",
            source="pre-crash observation",
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
    prepare_state_for_resume(
        state
    )

    return _run_scenario(
        state,
        FakeLiveWebEnvironment(
            capture_path=None,
            mode="query",
            spec=spec,
            query_text=query_text,
        ),
        spec,
    )


def _run_restart_after_submit(
    goal: str,
    spec: DurableWebSearchSpec,
) -> ScenarioResult:
    query_text = _query_for_spec(
        spec
    )
    state = _state_with_workspace(
        goal=goal,
        spec=spec,
        query_text=query_text,
        task_id=f"{spec.workflow_id}-submit",
    )
    transitions = TaskStateTransitions(
        state
    )
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search field contains configured query.",
            source="pre-crash observation",
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
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=spec.submit_side_effect_id,
            description=spec.submit_description,
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=spec.submit_action_key,
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
            spec=spec,
            query_text=query_text,
        ),
        spec,
    )


def _run_workflow_mismatch() -> ScenarioResult:
    state = TaskState(
        goal=WIKIPEDIA_GOAL
    )
    state.artifacts[
        WORKFLOW_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=WORKFLOW_ARTIFACT_ID,
        description="Durable live web workflow identity.",
        location="workflow:python-org-search",
    )
    environment = FakeLiveWebEnvironment(
        capture_path=None,
        mode="empty",
        spec=WIKIPEDIA_WORKFLOW,
        query_text=WIKIPEDIA_QUERY,
    )

    try:
        return _run_scenario(
            state,
            environment,
            WIKIPEDIA_WORKFLOW,
        )
    except RuntimeError:
        return ScenarioResult(
            completed=False,
            opened_workspace=(
                environment.open_count == 1
            ),
            reacquired_workspace=(
                environment.activate_count == 1
            ),
            typed_count=environment.type_count,
            clicked_count=environment.click_count,
            query_verified=False,
            result_verified=False,
            side_effect_confirmed=False,
            side_effect_action_key=None,
            workflow_identity=(
                state.artifacts[
                    WORKFLOW_ARTIFACT_ID
                ].location
            ),
            failed_closed=True,
        )


def _run_scenario(
    state: TaskState,
    environment: FakeLiveWebEnvironment,
    spec: DurableWebSearchSpec,
) -> ScenarioResult:
    published: list[TaskStateStatus] = []

    worker = create_live_web_worker(
        state,
        lambda: published.append(
            state.status
        ),
        lambda decision: None,
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
        spec.submit_side_effect_id
    )
    workflow = state.artifacts.get(
        WORKFLOW_ARTIFACT_ID
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
        query_verified=(
            state.claims[spec.query_claim_id].status
            is ClaimStatus.VERIFIED
            and state.subgoals[
                spec.query_subgoal_id
            ].status
            is SubgoalStatus.VERIFIED
        ),
        result_verified=(
            state.claims[spec.result_claim_id].status
            is ClaimStatus.VERIFIED
            and state.subgoals[
                spec.result_subgoal_id
            ].status
            is SubgoalStatus.VERIFIED
        ),
        side_effect_confirmed=(
            effect is not None
            and effect.state
            is SideEffectState.CONFIRMED
        ),
        side_effect_action_key=(
            None
            if effect is None
            else effect.action_key
        ),
        workflow_identity=(
            None
            if workflow is None
            else workflow.location
        ),
    )


def _state_with_workspace(
    *,
    goal: str,
    spec: DurableWebSearchSpec,
    query_text: str,
    task_id: str,
) -> TaskState:
    state = TaskState(
        goal=goal,
        task_id=task_id,
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(
        state
    )

    _ensure_workflow_identity(
        transitions,
        spec,
    )
    resolved_task = ResolvedDurableWebSearchTask(
        spec=spec,
        query_text=query_text,
    )
    durable_plan = compile_durable_task_plan(
        resolved_task
    )
    _ensure_query_identity(
        transitions,
        resolved_task,
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
    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=spec.workspace_artifact_id,
            description=spec.workspace_description,
            location=(
                BROWSER_WINDOW_MARKER_PREFIX
                + state.task_id
            ),
        )
    )
    return state


def _query_for_spec(
    spec: DurableWebSearchSpec,
) -> str:
    if spec is WIKIPEDIA_WORKFLOW:
        return WIKIPEDIA_QUERY
    return PYTHON_QUERY


def _completed_with_one_type_and_submit(
    result: ScenarioResult,
    spec: DurableWebSearchSpec,
) -> bool:
    return (
        result.completed
        and result.opened_workspace
        and not result.reacquired_workspace
        and result.typed_count == 1
        and result.clicked_count == 1
        and result.query_verified
        and result.result_verified
        and result.side_effect_confirmed
        and result.side_effect_action_key
        == spec.submit_action_key
        and result.workflow_identity
        == f"workflow:{spec.workflow_id}"
    )


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
    *,
    bounding_box: BoundingBox | None = None,
) -> UIElement:
    return _element(
        text=text,
        value=None,
        element_type="button",
        bounding_box=bounding_box,
    )


def _element(
    *,
    text: str,
    value: str | None,
    element_type: str,
    bounding_box: BoundingBox | None = None,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        text=text,
        value=value,
        confidence=0.95,
        enabled=True,
        bounding_box=bounding_box
        or BoundingBox(
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
