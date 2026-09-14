"""Phase 06 Experiment 08.01: parameterized durable search."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.app.live_web_worker import (
    parse_live_web_search_goal,
    resolve_live_web_task,
)
from computer_agent.task import (
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
    "Phase 06 Experiment 08.01: "
    "Parameterized Durable Search"
)


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    wikipedia_claude_uninterrupted: bool
    wikipedia_reinforcement_restart: bool
    wikipedia_alan_submit_restart: bool
    python_asyncio_uninterrupted: bool
    python_dataclasses_restart: bool
    malformed_unsupported_rejected: bool
    failures: tuple[str, ...]


def run_experiment() -> ExperimentReport:
    checks = {
        "wikipedia_claude_uninterrupted": _matches(
            _run("Search Wikipedia for Claude Shannon."),
            query="Claude Shannon",
            opened=1,
            activated=0,
            typed=1,
            clicked=1,
        ),
        "wikipedia_reinforcement_restart": _matches(
            _run_restart(
                "Search Wikipedia for reinforcement learning.",
                mode="query",
            ),
            query="reinforcement learning",
            opened=0,
            activated=1,
            typed=0,
            clicked=1,
        ),
        "wikipedia_alan_submit_restart": _matches(
            _run_restart(
                "Search Wikipedia for Alan Turing.",
                mode="results",
            ),
            query="Alan Turing",
            opened=0,
            activated=1,
            typed=0,
            clicked=0,
        ),
        "python_asyncio_uninterrupted": _matches(
            _run("Search python.org for asyncio."),
            query="asyncio",
            opened=1,
            activated=0,
            typed=1,
            clicked=1,
        ),
        "python_dataclasses_restart": _matches(
            _run_restart(
                "Search python.org for dataclasses.",
                mode="query",
            ),
            query="dataclasses",
            opened=0,
            activated=1,
            typed=0,
            clicked=1,
        ),
        "malformed_unsupported_rejected": _rejects_bad_goals(),
    }
    failures = tuple(
        name for name, passed in checks.items() if not passed
    )
    return ExperimentReport(
        wikipedia_claude_uninterrupted=checks[
            "wikipedia_claude_uninterrupted"
        ],
        wikipedia_reinforcement_restart=checks[
            "wikipedia_reinforcement_restart"
        ],
        wikipedia_alan_submit_restart=checks[
            "wikipedia_alan_submit_restart"
        ],
        python_asyncio_uninterrupted=checks[
            "python_asyncio_uninterrupted"
        ],
        python_dataclasses_restart=checks[
            "python_dataclasses_restart"
        ],
        malformed_unsupported_rejected=checks[
            "malformed_unsupported_rejected"
        ],
        failures=failures,
    )


def print_report(report: ExperimentReport) -> None:
    print(TITLE)
    rows = (
        (
            "Wikipedia Claude Shannon uninterrupted",
            report.wikipedia_claude_uninterrupted,
        ),
        (
            "Wikipedia reinforcement learning query restart",
            report.wikipedia_reinforcement_restart,
        ),
        (
            "Wikipedia Alan Turing submit restart",
            report.wikipedia_alan_submit_restart,
        ),
        (
            "python.org asyncio uninterrupted",
            report.python_asyncio_uninterrupted,
        ),
        (
            "python.org dataclasses restart",
            report.python_dataclasses_restart,
        ),
        (
            "Malformed/unsupported goal rejected",
            report.malformed_unsupported_rejected,
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


def _run(goal: str) -> FakeRunResult:
    return run_fake_worker_state(
        TaskState(goal=goal),
        mode="empty",
    )


def _run_restart(goal: str, *, mode: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    resolved = resolve_live_web_task(goal)
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
    if mode == "results":
        transitions.add_side_effect(
            SideEffectRecord(
                side_effect_id=(
                    resolved.spec.submit_side_effect_id
                ),
                description=(
                    resolved.spec.submit_description
                ),
                state=SideEffectState.EXECUTED,
                idempotent=True,
                action_key=resolved.spec.submit_action_key,
            )
        )
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode=mode)


def _matches(
    result: FakeRunResult,
    *,
    query: str,
    opened: int,
    activated: int,
    typed: int,
    clicked: int,
) -> bool:
    return (
        result.completed
        and result.opened == opened
        and result.activated == activated
        and result.typed == typed
        and result.clicked == clicked
        and result.query_identity == query
    )


def _rejects_bad_goals() -> bool:
    for goal in (
        "Search Wikipedia for .",
        "Search Google for OpenAI.",
        "Search Wikipedia Claude Shannon.",
    ):
        try:
            parse_live_web_search_goal(goal)
        except RuntimeError:
            continue
        return False
    return True


def _result(value: bool) -> str:
    return "passed" if value else "failed"


if __name__ == "__main__":
    raise SystemExit(main())
