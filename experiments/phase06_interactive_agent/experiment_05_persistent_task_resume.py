"""Phase 06 Experiment 05: persistent restart-safe task resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from tempfile import TemporaryDirectory

from computer_agent.reasoning import (
    AdaptiveDecisionEngine,
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
    build_adaptive_reasoning_context,
)
from computer_agent.task import (
    ClaimRecord,
    ClaimStatus,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
    TaskStateStore,
    TaskStateTransitions,
    prepare_state_for_resume,
)


GOAL = (
    "Submit the registration form exactly once "
    "and confirm the result."
)

TASK_ID = "phase06-experiment05-restart"

OLD_EVIDENCE_ID = "evidence-pre-crash-form"
CLAIM_ID = "claim-registration-data"
SUBGOAL_ID = "subgoal-prepare-registration"
SIDE_EFFECT_ID = "effect-submit-registration"
FRESH_EVIDENCE_ID = "evidence-post-restart-confirmation"

SUBMIT_ACTION_KEY = "click_target:submit"


class RestartRecoveryDemoClient:
    """Deterministic model boundary for restart recovery."""

    def __init__(self) -> None:
        self.prompts: list[dict[str, object]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        prefix = "Current task context:\n"

        if not user_prompt.startswith(prefix):
            raise RuntimeError(
                "unexpected adaptive reasoning prompt"
            )

        context = json.loads(
            user_prompt[len(prefix):]
        )
        self.prompts.append(context)

        if context["completion_allowed"]:
            return json.dumps(
                {
                    "decision": "complete",
                    "summary": (
                        "The registration submission "
                        "is confirmed exactly once."
                    ),
                }
            )

        if context["decision_feedback"]:
            return json.dumps(
                {
                    "decision": "action",
                    "expected_effect": (
                        "The current server-side "
                        "submission status should "
                        "become visible."
                    ),
                    "action": {
                        "goal": (
                            "Reconcile the previous "
                            "submission outcome."
                        ),
                        "operation": "click_target",
                        "action_target": {
                            "text": "Check status",
                            "element_types": [
                                "button"
                            ],
                        },
                        "verification_target": {
                            "text": "Submission REG-42",
                            "element_types": [],
                        },
                        "max_attempts": 1,
                    },
                }
            )

        return json.dumps(
            {
                "decision": "action",
                "expected_effect": (
                    "The registration should be "
                    "submitted."
                ),
                "action": {
                    "goal": (
                        "Submit the registration form."
                    ),
                    "operation": "click_target",
                    "action_target": {
                        "text": "Submit",
                        "element_types": [
                            "button"
                        ],
                    },
                    "verification_target": {
                        "text": "Submission status",
                        "element_types": [],
                    },
                    "max_attempts": 1,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    checkpoint_round_trip_passed: bool
    restart_invalidated_old_evidence_passed: bool
    unknown_side_effect_survived_passed: bool
    duplicate_submit_blocked_passed: bool
    bounded_replan_passed: bool
    fresh_reverification_passed: bool
    completion_gate_passed: bool
    final_checkpoint_passed: bool
    same_goal_passed: bool
    failures: tuple[str, ...]


def _build_pre_crash_state() -> TaskState:
    state = TaskState(
        goal=GOAL,
        constraints=(
            "Do not submit more than once.",
            (
                "Do not declare completion without "
                "fresh confirmation evidence."
            ),
        ),
        task_id=TASK_ID,
        status=TaskStateStatus.RUNNING,
    )

    transitions = TaskStateTransitions(state)

    old_evidence = EvidenceRecord(
        summary=(
            "The registration form contains "
            "the intended registration data."
        ),
        source="Current registration form",
        kind=EvidenceKind.VERIFICATION,
        evidence_id=OLD_EVIDENCE_ID,
    )
    transitions.add_evidence(
        old_evidence
    )

    claim = ClaimRecord(
        statement=(
            "The registration data matches "
            "the intended values."
        ),
        claim_id=CLAIM_ID,
    )
    transitions.add_claim(
        claim
    )
    transitions.verify_claim(
        CLAIM_ID,
        (OLD_EVIDENCE_ID,),
    )

    subgoal = SubgoalRecord(
        description=(
            "Prepare the intended registration data."
        ),
        claim_ids=(CLAIM_ID,),
        subgoal_id=SUBGOAL_ID,
    )
    transitions.add_subgoal(
        subgoal
    )
    transitions.verify_subgoal(
        SUBGOAL_ID
    )

    effect = SideEffectRecord(
        description=(
            "Submit the registration form."
        ),
        side_effect_id=SIDE_EFFECT_ID,
        idempotent=False,
        action_key=SUBMIT_ACTION_KEY,
    )
    transitions.add_side_effect(
        effect
    )
    transitions.mark_side_effect_executed(
        SIDE_EFFECT_ID
    )
    transitions.mark_side_effect_unknown(
        SIDE_EFFECT_ID
    )

    return state


def _restart_observation() -> ObservationContext:
    return ObservationContext(
        application_name="Google Chrome",
        window_title="Registration",
        visible_text=(
            "Registration",
            "Submit",
            "Check status",
            "Submission status",
        ),
        elements=(
            ObservedElement(
                text="Submit",
                element_type="button",
                enabled=True,
            ),
            ObservedElement(
                text="Check status",
                element_type="button",
                enabled=True,
            ),
        ),
    )


def _action_target_text(
    result,
) -> str | None:
    decision = result.decision

    if (
        decision is None
        or decision.action is None
    ):
        return None

    target = getattr(
        decision.action,
        "action_target",
        None,
    )

    if target is None:
        return None

    return target.text


def _rejected_action_target_text(
    result,
) -> str | None:
    decision = result.rejected_decision

    if (
        decision is None
        or decision.action is None
    ):
        return None

    target = getattr(
        decision.action,
        "action_target",
        None,
    )

    if target is None:
        return None

    return target.text


def run_experiment() -> ExperimentReport:
    failures: list[str] = []

    with TemporaryDirectory() as directory:
        store = TaskStateStore(directory)

        pre_crash_state = (
            _build_pre_crash_state()
        )

        store.save(
            pre_crash_state
        )

        restored = store.load(
            TASK_ID
        )

        checkpoint_round_trip_passed = (
            restored == pre_crash_state
            and restored is not pre_crash_state
        )

        if not checkpoint_round_trip_passed:
            failures.append(
                "checkpoint did not restore an "
                "independent equivalent TaskState"
            )

        resume_report = (
            prepare_state_for_resume(
                restored
            )
        )

        restart_invalidated_old_evidence_passed = (
            restored.evidence[
                OLD_EVIDENCE_ID
            ].freshness
            is EvidenceFreshness.STALE
            and restored.claims[
                CLAIM_ID
            ].status
            is ClaimStatus.UNKNOWN
            and restored.subgoals[
                SUBGOAL_ID
            ].status
            is SubgoalStatus.UNKNOWN
            and OLD_EVIDENCE_ID
            in resume_report.stale_evidence_ids
        )

        if not restart_invalidated_old_evidence_passed:
            failures.append(
                "restart did not invalidate stale "
                "environment-dependent progress"
            )

        effect = restored.side_effects[
            SIDE_EFFECT_ID
        ]

        unknown_side_effect_survived_passed = (
            effect.state
            is SideEffectState.UNKNOWN
            and effect.idempotent is False
            and effect.action_key
            == SUBMIT_ACTION_KEY
            and SUBMIT_ACTION_KEY
            in resume_report.blocked_action_keys
        )

        if not unknown_side_effect_survived_passed:
            failures.append(
                "UNKNOWN non-idempotent side effect "
                "did not survive restart safely"
            )

        restored.status = (
            TaskStateStatus.RUNNING
        )
        restored.touch()

        client = RestartRecoveryDemoClient()

        engine = AdaptiveDecisionEngine(
            reasoner=NextStepReasoner(
                client=client
            )
        )

        restart_context = (
            build_adaptive_reasoning_context(
                state=restored,
                observation=(
                    _restart_observation()
                ),
            )
        )

        outcome = engine.decide(
            restart_context
        )

        first_attempt = (
            outcome.attempt_results[0]
        )

        duplicate_submit_blocked_passed = (
            first_attempt.status
            is NextStepReasoningStatus.BLOCKED
            and _rejected_action_target_text(
                first_attempt
            )
            == "Submit"
            and SUBMIT_ACTION_KEY
            in restart_context.blocked_action_keys
        )

        if not duplicate_submit_blocked_passed:
            failures.append(
                "duplicate Submit was not "
                "deterministically rejected"
            )

        bounded_replan_passed = (
            outcome.attempts == 2
            and outcome.safety_replan_used
            and outcome.result.status
            is NextStepReasoningStatus.READY
            and outcome.result.decision
            is not None
            and outcome.result.decision.decision_type
            is NextStepDecisionType.ACTION
            and _action_target_text(
                outcome.result
            )
            == "Check status"
        )

        if not bounded_replan_passed:
            failures.append(
                "bounded safety replan did not "
                "choose Check status"
            )

        transitions = (
            TaskStateTransitions(
                restored
            )
        )

        fresh_evidence = EvidenceRecord(
            summary=(
                "Submission REG-42 is confirmed "
                "exactly once and shows the "
                "intended registration data."
            ),
            source=(
                "Current registration status page"
            ),
            kind=EvidenceKind.VERIFICATION,
            evidence_id=FRESH_EVIDENCE_ID,
        )
        transitions.add_evidence(
            fresh_evidence
        )

        transitions.confirm_side_effect(
            SIDE_EFFECT_ID,
            (FRESH_EVIDENCE_ID,),
        )

        transitions.verify_claim(
            CLAIM_ID,
            (FRESH_EVIDENCE_ID,),
        )

        transitions.verify_subgoal(
            SUBGOAL_ID
        )

        fresh_reverification_passed = (
            restored.evidence[
                OLD_EVIDENCE_ID
            ].freshness
            is EvidenceFreshness.STALE
            and restored.evidence[
                FRESH_EVIDENCE_ID
            ].freshness
            is EvidenceFreshness.CURRENT
            and restored.claims[
                CLAIM_ID
            ].status
            is ClaimStatus.VERIFIED
            and restored.subgoals[
                SUBGOAL_ID
            ].status
            is SubgoalStatus.VERIFIED
            and restored.side_effects[
                SIDE_EFFECT_ID
            ].state
            is SideEffectState.CONFIRMED
            and transitions.can_complete()
        )

        if not fresh_reverification_passed:
            failures.append(
                "fresh post-restart evidence did "
                "not restore verified progress"
            )

        final_observation = (
            ObservationContext(
                application_name=(
                    "Google Chrome"
                ),
                window_title=(
                    "Registration Status"
                ),
                visible_text=(
                    "Submission REG-42",
                    "Confirmed",
                ),
            )
        )

        final_context = (
            build_adaptive_reasoning_context(
                state=restored,
                observation=final_observation,
            )
        )

        final_outcome = engine.decide(
            final_context
        )

        completion_gate_passed = (
            final_context.completion_allowed
            and final_outcome.attempts == 1
            and final_outcome.result.status
            is NextStepReasoningStatus.READY
            and final_outcome.result.decision
            is not None
            and (
                final_outcome.result
                .decision.decision_type
                is NextStepDecisionType.COMPLETE
            )
        )

        if not completion_gate_passed:
            failures.append(
                "completion was not proposed "
                "after fresh reconciliation"
            )
        else:
            transitions.complete_task()

        store.save(
            restored
        )

        final_loaded = store.load(
            TASK_ID
        )

        final_checkpoint_passed = (
            final_loaded.status
            is TaskStateStatus.COMPLETED
            and final_loaded.side_effects[
                SIDE_EFFECT_ID
            ].state
            is SideEffectState.CONFIRMED
            and final_loaded.evidence[
                OLD_EVIDENCE_ID
            ].freshness
            is EvidenceFreshness.STALE
            and final_loaded.evidence[
                FRESH_EVIDENCE_ID
            ].freshness
            is EvidenceFreshness.CURRENT
        )

        if not final_checkpoint_passed:
            failures.append(
                "final reconciled state did not "
                "survive a second checkpoint load"
            )

        same_goal_passed = (
            len(client.prompts) == 3
            and all(
                prompt["goal"] == GOAL
                for prompt in client.prompts
            )
        )

        if not same_goal_passed:
            failures.append(
                "restart reasoning did not "
                "preserve the original goal"
            )

    return ExperimentReport(
        checkpoint_round_trip_passed=(
            checkpoint_round_trip_passed
        ),
        restart_invalidated_old_evidence_passed=(
            restart_invalidated_old_evidence_passed
        ),
        unknown_side_effect_survived_passed=(
            unknown_side_effect_survived_passed
        ),
        duplicate_submit_blocked_passed=(
            duplicate_submit_blocked_passed
        ),
        bounded_replan_passed=(
            bounded_replan_passed
        ),
        fresh_reverification_passed=(
            fresh_reverification_passed
        ),
        completion_gate_passed=(
            completion_gate_passed
        ),
        final_checkpoint_passed=(
            final_checkpoint_passed
        ),
        same_goal_passed=(
            same_goal_passed
        ),
        failures=tuple(failures),
    )


def print_report(
    report: ExperimentReport,
) -> None:
    def result(value: bool) -> str:
        return (
            "passed"
            if value
            else "failed"
        )

    print(
        "Phase 06 Experiment 05: "
        "Persistent Restart-Safe Task Resume"
    )
    print(
        "Checkpoint round trip: "
        f"{result(report.checkpoint_round_trip_passed)}"
    )
    print(
        "Restart evidence invalidation: "
        f"{result(report.restart_invalidated_old_evidence_passed)}"
    )
    print(
        "UNKNOWN side-effect persistence: "
        f"{result(report.unknown_side_effect_survived_passed)}"
    )
    print(
        "Duplicate-submit safety block: "
        f"{result(report.duplicate_submit_blocked_passed)}"
    )
    print(
        "Bounded reconciliation replan: "
        f"{result(report.bounded_replan_passed)}"
    )
    print(
        "Fresh re-verification: "
        f"{result(report.fresh_reverification_passed)}"
    )
    print(
        "Completion-gate recovery: "
        f"{result(report.completion_gate_passed)}"
    )
    print(
        "Final checkpoint recovery: "
        f"{result(report.final_checkpoint_passed)}"
    )
    print(
        "Same-goal preservation: "
        f"{result(report.same_goal_passed)}"
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

    return (
        1
        if report.failures
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
