"""Phase 06 Experiment 08.02: durable follow-up navigation."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.app.live_web_worker import (
    FOLLOWUP_NAVIGATION_SIDE_EFFECT_ID,
    FOLLOWUP_TARGET_ARTIFACT_ID,
    QUERY_ARTIFACT_ID,
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
    "Phase 06 Experiment 08.02: "
    "Durable Follow-Up Navigation"
)

CLAUDE_GOAL = (
    "Search Wikipedia for Claude Shannon "
    "and open Information theory."
)
ALAN_GOAL = (
    "Search Wikipedia for Alan Turing "
    "and open Turing machine."
)


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    claude_uninterrupted: bool
    alan_query_restart: bool
    claude_submit_restart: bool
    claude_followup_restart: bool
    missing_link_failed_closed: bool
    followup_identity_mismatch_failed_closed: bool
    search_only_regression: bool
    failures: tuple[str, ...]


def run_experiment() -> ExperimentReport:
    checks = {
        "claude_uninterrupted": _matches(
            _run(CLAUDE_GOAL),
            query="Claude Shannon",
            followup="Information theory",
            opened=1,
            activated=0,
            typed=1,
            clicked=2,
            followup_clicked=1,
        ),
        "alan_query_restart": _matches(
            _run_query_restart(ALAN_GOAL),
            query="Alan Turing",
            followup="Turing machine",
            opened=0,
            activated=1,
            typed=0,
            clicked=2,
            followup_clicked=1,
        ),
        "claude_submit_restart": _matches(
            _run_submit_restart(CLAUDE_GOAL),
            query="Claude Shannon",
            followup="Information theory",
            opened=0,
            activated=1,
            typed=0,
            clicked=1,
            followup_clicked=1,
        ),
        "claude_followup_restart": _matches(
            _run_followup_restart(CLAUDE_GOAL),
            query="Claude Shannon",
            followup="Information theory",
            opened=0,
            activated=1,
            typed=0,
            clicked=0,
            followup_clicked=0,
        ),
        "missing_link_failed_closed": _missing_link_fails_closed(),
        "followup_identity_mismatch_failed_closed": (
            _identity_mismatch_fails_closed()
        ),
        "search_only_regression": _search_only_regression(),
    }
    failures = tuple(
        name for name, passed in checks.items() if not passed
    )
    return ExperimentReport(
        claude_uninterrupted=checks["claude_uninterrupted"],
        alan_query_restart=checks["alan_query_restart"],
        claude_submit_restart=checks["claude_submit_restart"],
        claude_followup_restart=checks["claude_followup_restart"],
        missing_link_failed_closed=checks[
            "missing_link_failed_closed"
        ],
        followup_identity_mismatch_failed_closed=checks[
            "followup_identity_mismatch_failed_closed"
        ],
        search_only_regression=checks["search_only_regression"],
        failures=failures,
    )


def print_report(report: ExperimentReport) -> None:
    print(TITLE)
    rows = (
        (
            "Claude Shannon to Information theory uninterrupted",
            report.claude_uninterrupted,
        ),
        (
            "Alan Turing to Turing machine query restart",
            report.alan_query_restart,
        ),
        (
            "Claude Shannon submit restart",
            report.claude_submit_restart,
        ),
        (
            "Claude Shannon follow-up restart",
            report.claude_followup_restart,
        ),
        (
            "Requested link missing fails closed",
            report.missing_link_failed_closed,
        ),
        (
            "Persisted follow-up identity mismatch fails closed",
            report.followup_identity_mismatch_failed_closed,
        ),
        (
            "Search-only arbitrary query regression",
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


def _run(goal: str) -> FakeRunResult:
    return run_fake_worker_state(
        TaskState(goal=goal),
        mode="empty",
    )


def _run_query_restart(goal: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    _verify_query_history(state)
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode="query")


def _run_submit_restart(goal: str) -> FakeRunResult:
    state = state_with_workspace(goal)
    resolved = resolve_live_web_task(goal)
    assert resolved is not None
    _verify_query_history(state)
    transitions = TaskStateTransitions(state)
    transitions.add_side_effect(
        SideEffectRecord(
            side_effect_id=(
                resolved.spec.submit_side_effect_id
            ),
            description=resolved.spec.submit_description,
            state=SideEffectState.EXECUTED,
            idempotent=True,
            action_key=resolved.spec.submit_action_key,
        )
    )
    prepare_state_for_resume(state)
    return run_fake_worker_state(state, mode="results")


def _run_followup_restart(goal: str) -> FakeRunResult:
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


def _missing_link_fails_closed() -> bool:
    state = state_with_workspace(CLAUDE_GOAL)
    _verify_query_history(state)
    _verify_result_history(state)
    prepare_state_for_resume(state)
    try:
        run_fake_worker_state(state, mode="missing_link")
    except RuntimeError as exc:
        return "actionable link" in str(exc)
    return False


def _identity_mismatch_fails_closed() -> bool:
    state = state_with_workspace(CLAUDE_GOAL)
    state.artifacts[
        FOLLOWUP_TARGET_ARTIFACT_ID
    ] = ArtifactRecord(
        artifact_id=FOLLOWUP_TARGET_ARTIFACT_ID,
        description="Durable live web follow-up target text.",
        location="Turing machine",
    )
    try:
        run_fake_worker_state(state, mode="empty")
    except RuntimeError as exc:
        return "follow-up target identity" in str(exc)
    return False


def _search_only_regression() -> bool:
    result = _run("Search python.org for asyncio.")
    return (
        result.completed
        and result.query_identity == "asyncio"
        and result.followup_identity is None
        and result.clicked == 1
        and result.followup_clicked == 0
    )


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
    query: str,
    followup: str,
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
        and result.query_identity == query
        and result.followup_identity == followup
    )


def _result(value: bool) -> str:
    return "passed" if value else "failed"


if __name__ == "__main__":
    raise SystemExit(main())
