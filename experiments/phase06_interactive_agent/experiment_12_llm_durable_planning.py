"""Phase 06 Experiment 08.04: LLM-generated durable planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable

from computer_agent.app.durable_planning import (
    DurablePlanProposal,
    DurableStepProposal,
    LLMDurablePlanner,
    compile_plan_proposal,
    default_durable_planning_context,
)
from computer_agent.app.live_web_worker import (
    CRASH_ENV_VAR,
    DURABLE_PLAN_ARTIFACT_ID,
    LiveCrashPoint,
    create_live_web_worker,
    resolve_live_web_task,
)
import computer_agent.app.live_web_worker as live_web_worker
from computer_agent.runtime import RuntimeTask
from computer_agent.task import TaskState

try:
    from experiments.phase06_interactive_agent.durable_search_fake import (
        FakeControl,
        FakeSearchEnvironment,
        run_fake_worker_state,
    )
except ModuleNotFoundError:
    from durable_search_fake import (  # type: ignore
        FakeControl,
        FakeSearchEnvironment,
        run_fake_worker_state,
    )


WIKI_GOAL = "Search Wikipedia for Claude Shannon."
WIKI_FOLLOWUP_GOAL = (
    "Search Wikipedia for Claude Shannon and open Information theory."
)
PYTHON_GOAL = "Search python.org for asyncio."


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    wikipedia_two_step_plan: bool
    wikipedia_three_step_plan: bool
    python_plan: bool
    mismatch_rejected_before_browser_action: bool
    invalid_ordering_rejected: bool
    persisted_plan_resumes_without_planner_call: bool
    three_step_execution_uses_reconciler: bool
    crash_recovery_does_not_replan: bool
    planner_call_count: int
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures and all(
            (
                self.wikipedia_two_step_plan,
                self.wikipedia_three_step_plan,
                self.python_plan,
                self.mismatch_rejected_before_browser_action,
                self.invalid_ordering_rejected,
                self.persisted_plan_resumes_without_planner_call,
                self.three_step_execution_uses_reconciler,
                self.crash_recovery_does_not_replan,
            )
        )


class FakeLLMClient:
    def __init__(
        self,
        response: str,
        *,
        model: str = "fake-durable-planner",
    ) -> None:
        self.response = response
        self.model = model
        self.calls = 0

    def generate_json(self, **kwargs: object) -> str:
        del kwargs
        self.calls += 1
        return self.response


class FailingPlanner:
    planner_type = "llm"
    model_identifier = "failing-planner"

    def __init__(self) -> None:
        self.calls = 0

    def plan(self, goal, context):
        del goal, context
        self.calls += 1
        raise RuntimeError("planner should not be called on resume")


def run_experiment() -> AcceptanceReport:
    failures: list[str] = []
    total_planner_calls = 0

    def check(
        label: str,
        func: Callable[[], bool],
    ) -> bool:
        try:
            passed = bool(func())
        except Exception as exc:
            failures.append(f"{label}: {exc}")
            return False
        if not passed:
            failures.append(label)
        return passed

    wikipedia_two_step_plan = check(
        "LLM proposal compiles to Wikipedia 2-step durable plan",
        _wikipedia_two_step_plan,
    )
    wikipedia_three_step_plan = check(
        "LLM proposal compiles to Wikipedia 3-step durable plan",
        _wikipedia_three_step_plan,
    )
    python_plan = check(
        "LLM proposal compiles to python.org durable plan",
        _python_plan,
    )
    mismatch_rejected = check(
        "mismatch rejected before browser action",
        _mismatch_rejected_before_browser_action,
    )
    invalid_ordering_rejected = check(
        "invalid ordering rejected",
        _invalid_ordering_rejected,
    )
    no_replan_resume, resume_calls = (
        _persisted_plan_resumes_without_planner_call()
    )
    if not no_replan_resume:
        failures.append("persisted plan resumes without planner call")
    total_planner_calls += resume_calls
    three_step_execution = check(
        "3-step execution passes through generic reconciler",
        _three_step_execution_uses_reconciler,
    )
    no_replan_crash, crash_calls = (
        _crash_recovery_does_not_replan()
    )
    if not no_replan_crash:
        failures.append("crash recovery does not re-plan")
    total_planner_calls += crash_calls

    return AcceptanceReport(
        wikipedia_two_step_plan=wikipedia_two_step_plan,
        wikipedia_three_step_plan=wikipedia_three_step_plan,
        python_plan=python_plan,
        mismatch_rejected_before_browser_action=mismatch_rejected,
        invalid_ordering_rejected=invalid_ordering_rejected,
        persisted_plan_resumes_without_planner_call=no_replan_resume,
        three_step_execution_uses_reconciler=three_step_execution,
        crash_recovery_does_not_replan=no_replan_crash,
        planner_call_count=total_planner_calls,
        failures=tuple(failures),
    )


def print_report(
    report: AcceptanceReport,
) -> None:
    rows = (
        ("Wikipedia 2-step LLM durable plan", report.wikipedia_two_step_plan),
        ("Wikipedia 3-step LLM durable plan", report.wikipedia_three_step_plan),
        ("python.org LLM durable plan", report.python_plan),
        (
            "Mismatch rejected before browser action",
            report.mismatch_rejected_before_browser_action,
        ),
        ("Invalid ordering rejected", report.invalid_ordering_rejected),
        (
            "Persisted accepted plan resumes without planner call",
            report.persisted_plan_resumes_without_planner_call,
        ),
        (
            "3-step execution uses generic reconciler",
            report.three_step_execution_uses_reconciler,
        ),
        (
            "Crash recovery does not re-plan",
            report.crash_recovery_does_not_replan,
        ),
    )
    print("Phase 06.08.04 - LLM-Generated Durable Planning")
    for label, passed in rows:
        print(f"{label}: {'passed' if passed else 'failed'}")
    print(f"Planner calls observed: {report.planner_call_count}")
    if report.failures:
        print("Failures:")
        for failure in report.failures:
            print(f"- {failure}")
    print(
        "Experiment acceptance: "
        + ("passed" if report.passed else "failed")
    )


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 0 if report.passed else 1


def _wikipedia_two_step_plan() -> bool:
    plan = _plan_from_response(
        WIKI_GOAL,
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
            ],
        ),
    )
    return len(plan.steps) == 2 and plan.workflow_id == "wikipedia-search"


def _wikipedia_three_step_plan() -> bool:
    plan = _plan_from_response(
        WIKI_FOLLOWUP_GOAL,
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
                {
                    "kind": "open_link",
                    "target_text": "Information theory",
                },
            ],
        ),
    )
    return len(plan.steps) == 3 and plan.steps[2].step_id == "open-followup-link"


def _python_plan() -> bool:
    plan = _plan_from_response(
        PYTHON_GOAL,
        _proposal(
            "python-org-search",
            [
                {
                    "kind": "enter_text",
                    "text": "asyncio",
                },
                {
                    "kind": "submit_search",
                },
            ],
        ),
    )
    return len(plan.steps) == 2 and plan.workflow_id == "python-org-search"


def _mismatch_rejected_before_browser_action() -> bool:
    state = TaskState(
        goal=WIKI_GOAL
    )
    factory_calls = 0

    def factory(*, capture_path):
        nonlocal factory_calls
        del capture_path
        factory_calls += 1
        raise RuntimeError("browser action should not start")

    try:
        create_live_web_worker(
            state,
            lambda: None,
            lambda decision: None,
            environment_factory=factory,
            durable_planner=_planner(
                _proposal(
                    "wikipedia-search",
                    [
                        {
                            "kind": "enter_text",
                            "text": "Alan Turing",
                        },
                        {
                            "kind": "submit_search",
                        },
                    ],
                )
            ),
        )
    except RuntimeError:
        return factory_calls == 0
    return False


def _invalid_ordering_rejected() -> bool:
    try:
        compile_plan_proposal(
            WIKI_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                    DurableStepProposal(
                        kind="enter_text",
                        text="Claude Shannon",
                    ),
                ),
            ),
        )
    except ValueError:
        return True
    return False


def _persisted_plan_resumes_without_planner_call() -> tuple[bool, int]:
    response = _proposal(
        "wikipedia-search",
        [
            {
                "kind": "enter_text",
                "text": "Claude Shannon",
            },
            {
                "kind": "submit_search",
            },
        ],
    )
    client = FakeLLMClient(
        response
    )
    state = TaskState(
        goal=WIKI_GOAL
    )
    first_result = run_fake_worker_state(
        state,
        "empty",
        durable_planner=LLMDurablePlanner(
            client=client
        ),
    )
    failing = FailingPlanner()
    second_result = run_fake_worker_state(
        state,
        "results",
        durable_planner=failing,
    )
    return (
        first_result.completed
        and second_result.completed
        and failing.calls == 0
        and DURABLE_PLAN_ARTIFACT_ID in state.artifacts,
        client.calls,
    )


def _three_step_execution_uses_reconciler() -> bool:
    state = TaskState(
        goal=WIKI_FOLLOWUP_GOAL
    )
    result = run_fake_worker_state(
        state,
        "empty",
        durable_planner=_planner(
            _proposal(
                "wikipedia-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "Claude Shannon",
                    },
                    {
                        "kind": "submit_search",
                    },
                    {
                        "kind": "open_link",
                        "target_text": "Information theory",
                    },
                ],
            )
        ),
    )
    return (
        result.completed
        and result.typed == 1
        and result.clicked == 2
        and result.followup_clicked == 1
    )


def _crash_recovery_does_not_replan() -> tuple[bool, int]:
    class TestCrash(RuntimeError):
        pass

    old_env_value = os.environ.get(
        CRASH_ENV_VAR
    )
    old_terminate = live_web_worker._terminate_process_for_test
    os.environ[CRASH_ENV_VAR] = LiveCrashPoint.QUERY_CHECKPOINT.value
    live_web_worker._terminate_process_for_test = (
        lambda: (_ for _ in ()).throw(TestCrash())
    )

    client = FakeLLMClient(
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
                {
                    "kind": "open_link",
                    "target_text": "Information theory",
                },
            ],
        )
    )
    state = TaskState(
        goal=WIKI_FOLLOWUP_GOAL
    )
    resolved = resolve_live_web_task(
        WIKI_FOLLOWUP_GOAL
    )
    assert resolved is not None
    try:
        first_environment = FakeSearchEnvironment(
            capture_path=None,
            spec=resolved.spec,
            query_text=resolved.query_text,
            followup_target_text=resolved.followup_target_text,
            mode="empty",
        )
        worker = create_live_web_worker(
            state,
            lambda: None,
            lambda decision: None,
            environment_factory=lambda *, capture_path: first_environment,
            durable_planner=LLMDurablePlanner(
                client=client
            ),
        )
        try:
            worker(
                RuntimeTask(
                    goal=state.goal,
                    task_id=state.task_id,
                ),
                FakeControl(),
                lambda message: None,
            )
        except TestCrash:
            pass
        else:
            return False, client.calls
    finally:
        if old_env_value is None:
            os.environ.pop(
                CRASH_ENV_VAR,
                None,
            )
        else:
            os.environ[CRASH_ENV_VAR] = old_env_value
        live_web_worker._terminate_process_for_test = old_terminate

    failing = FailingPlanner()
    restart_environment = FakeSearchEnvironment(
        capture_path=None,
        spec=resolved.spec,
        query_text=resolved.query_text,
        followup_target_text=resolved.followup_target_text,
        mode="query",
    )
    restart_worker = create_live_web_worker(
        state,
        lambda: None,
        lambda decision: None,
        environment_factory=lambda *, capture_path: restart_environment,
        durable_planner=failing,
    )
    restart_worker(
        RuntimeTask(
            goal=state.goal,
            task_id=state.task_id,
        ),
        FakeControl(),
        lambda message: None,
    )
    return (
        failing.calls == 0
        and restart_environment.type_count == 0
        and restart_environment.click_count == 2,
        client.calls,
    )


def _plan_from_response(
    goal: str,
    response: str,
):
    return _planner(
        response
    ).plan(
        goal,
        default_durable_planning_context(),
    )


def _planner(
    response: str,
) -> LLMDurablePlanner:
    return LLMDurablePlanner(
        client=FakeLLMClient(
            response
        )
    )


def _proposal(
    workflow_id: str,
    steps: list[dict[str, object]],
) -> str:
    return json.dumps(
        {
            "version": 1,
            "workflow_id": workflow_id,
            "steps": steps,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
