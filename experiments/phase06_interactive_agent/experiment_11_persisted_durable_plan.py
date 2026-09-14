"""Phase 06 Experiment 08.03: persisted dynamic durable plan."""

from __future__ import annotations

from dataclasses import dataclass
import json

from computer_agent.app.live_web_worker import (
    DURABLE_PLAN_ARTIFACT_ID,
    FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID,
    compile_durable_task_plan,
    durable_plan_canonical_json,
    resolve_live_web_task,
)
from computer_agent.task import (
    ArtifactRecord,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    TaskState,
    TaskStateTransitions,
    prepare_state_for_resume,
)

try:
    from experiments.phase06_interactive_agent.durable_search_fake import (
        FakeRunResult,
        run_fake_worker_state,
        state_with_workspace,
    )
except ModuleNotFoundError:
    from durable_search_fake import (
        FakeRunResult,
        run_fake_worker_state,
        state_with_workspace,
    )


TITLE = (
    "Phase 06 Experiment 08.03: "
    "Persisted Dynamic Durable Plan"
)

CLAUDE_GOAL = (
    "Search Wikipedia for Claude Shannon "
    "and open Information theory."
)
WIKI_SEARCH_GOAL = "Search Wikipedia for Claude Shannon."
PYTHON_GOAL = "Search python.org for asyncio."


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    wikipedia_search_plan_two_steps: bool
    wikipedia_followup_plan_three_steps: bool
    python_plan_two_steps: bool
    uninterrupted_three_step_execution: bool
    resume_from_step_1: bool
    resume_from_step_2: bool
    resume_at_step_3_destination: bool
    persisted_plan_tampering_failed_closed: bool
    search_only_regression: bool
    failures: tuple[str, ...]


def run_experiment() -> ExperimentReport:
    checks = {
        "wikipedia_search_plan_two_steps": (
            _step_ids(WIKI_SEARCH_GOAL)
            == ("enter-query", "submit-search")
        ),
        "wikipedia_followup_plan_three_steps": (
            _step_ids(CLAUDE_GOAL)
            == (
                "enter-query",
                "submit-search",
                "open-followup-link",
            )
        ),
        "python_plan_two_steps": (
            _step_ids(PYTHON_GOAL)
            == ("enter-query", "submit-search")
        ),
        "uninterrupted_three_step_execution": _matches(
            _run(CLAUDE_GOAL),
            opened=1,
            activated=0,
            typed=1,
            clicked=2,
            followup_clicked=1,
        ),
        "resume_from_step_1": _matches(
            _run_query_resume(CLAUDE_GOAL),
            opened=0,
            activated=1,
            typed=0,
            clicked=2,
            followup_clicked=1,
        ),
        "resume_from_step_2": _matches(
            _run_result_resume(CLAUDE_GOAL),
            opened=0,
            activated=1,
            typed=0,
            clicked=1,
            followup_clicked=1,
        ),
        "resume_at_step_3_destination": _matches(
            _run_destination_resume(CLAUDE_GOAL),
            opened=0,
            activated=1,
            typed=0,
            clicked=0,
            followup_clicked=0,
        ),
        "persisted_plan_tampering_failed_closed": (
            _tampered_plan_fails_closed()
        ),
        "search_only_regression": _matches(
            _run(PYTHON_GOAL),
            opened=1,
            activated=0,
            typed=1,
            clicked=1,
            followup_clicked=0,
        ),
    }
    failures = tuple(
        name for name, passed in checks.items() if not passed
    )
    return ExperimentReport(
        wikipedia_search_plan_two_steps=checks[
            "wikipedia_search_plan_two_steps"
        ],
        wikipedia_followup_plan_three_steps=checks[
            "wikipedia_followup_plan_three_steps"
        ],
        python_plan_two_steps=checks["python_plan_two_steps"],
        uninterrupted_three_step_execution=checks[
            "uninterrupted_three_step_execution"
        ],
        resume_from_step_1=checks["resume_from_step_1"],
        resume_from_step_2=checks["resume_from_step_2"],
        resume_at_step_3_destination=checks[
            "resume_at_step_3_destination"
        ],
        persisted_plan_tampering_failed_closed=checks[
            "persisted_plan_tampering_failed_closed"
        ],
        search_only_regression=checks["search_only_regression"],
        failures=failures,
    )


def print_report(report: ExperimentReport) -> None:
    print(TITLE)
    rows = (
        (
            "Wikipedia search-only plan has 2 steps",
            report.wikipedia_search_plan_two_steps,
        ),
        (
            "Wikipedia follow-up plan has 3 steps",
            report.wikipedia_followup_plan_three_steps,
        ),
        (
            "python.org search-only plan has 2 steps",
            report.python_plan_two_steps,
        ),
        (
            "Uninterrupted 3-step execution",
            report.uninterrupted_three_step_execution,
        ),
        (
            "Resume from Step 1",
            report.resume_from_step_1,
        ),
        (
            "Resume from Step 2",
            report.resume_from_step_2,
        ),
        (
            "Resume already at Step 3 destination",
            report.resume_at_step_3_destination,
        ),
        (
            "Persisted plan tampering fails closed",
            report.persisted_plan_tampering_failed_closed,
        ),
        (
            "Search-only regression",
            report.search_only_regression,
        ),
    )
    for label, passed in rows:
        print(f"{label}: {_result(passed)}")
    print(
        "Experiment acceptance: "
        f"{'passed' if not report.failures else 'failed'}"
    )
    for failure in report.failures:
        print(f"Failure: {failure}")


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 0 if not report.failures else 1


def _step_ids(goal: str) -> tuple[str, ...]:
    resolved = resolve_live_web_task(goal)
    assert resolved is not None
    return tuple(
        step.step_id
        for step in compile_durable_task_plan(resolved).steps
    )


def _run(goal: str) -> FakeRunResult:
    return run_fake_worker_state(
        TaskState(goal=goal),
        mode="empty",
    )


def _run_query_resume(goal: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    _verify_query_history(state)
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode="query")


def _run_result_resume(goal: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    _verify_query_history(state)
    _verify_result_history(state)
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode="results")


def _run_destination_resume(goal: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    _verify_query_history(state)
    _verify_result_history(state)
    transitions = TaskStateTransitions(state)
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID
            ),
            description="Open Wikipedia link 'Information theory'.",
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key="click_target:wikipedia_followup_link",
        )
    )
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode="destination")


def _tampered_plan_fails_closed() -> bool:
    state = state_with_workspace(CLAUDE_GOAL)
    resolved = resolve_live_web_task(CLAUDE_GOAL)
    assert resolved is not None
    plan = compile_durable_task_plan(resolved)
    payload = json.loads(
        durable_plan_canonical_json(plan)
    )
    payload["steps"].reverse()
    state.artifacts[
        DURABLE_PLAN_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=DURABLE_PLAN_ARTIFACT_ID,
        description="Canonical persisted durable task plan.",
        location=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    try:
        run_fake_worker_state(state, mode="empty")
    except RuntimeError as exc:
        return "durable task plan" in str(exc)
    return False


def _verify_query_history(state: TaskState) -> None:
    resolved = resolve_live_web_task(state.goal)
    assert resolved is not None
    transitions = TaskStateTransitions(state)
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary=f"Search field contains {resolved.query_text!r}.",
            source="pre-crash fake observation",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        resolved.spec.query_claim_id,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        resolved.spec.query_subgoal_id
    )


def _verify_result_history(state: TaskState) -> None:
    resolved = resolve_live_web_task(state.goal)
    assert resolved is not None
    transitions = TaskStateTransitions(state)
    evidence = transitions.add_evidence(
        EvidenceRecord(
            summary="Search outcome is visible.",
            source="pre-crash fake observation",
            kind=EvidenceKind.VERIFICATION,
        )
    )
    transitions.verify_claim(
        resolved.spec.result_claim_id,
        (evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        resolved.spec.result_subgoal_id
    )


def _matches(
    result: FakeRunResult,
    *,
    opened: int,
    activated: int,
    typed: int,
    clicked: int,
    followup_clicked: int,
) -> bool:
    return (
        result.completed
        and result.opened == opened
        and result.activated == activated
        and result.typed == typed
        and result.clicked == clicked
        and result.followup_clicked == followup_clicked
    )


def _result(value: bool) -> str:
    return "passed" if value else "failed"


if __name__ == "__main__":
    raise SystemExit(main())
