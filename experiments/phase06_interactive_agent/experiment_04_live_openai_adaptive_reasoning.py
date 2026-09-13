"""Live OpenAI validation for Phase 06 adaptive next-step reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.reasoning import (
    AdaptiveDecisionEngine,
    NextStepDecision,
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningResult,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
    build_adaptive_reasoning_context,
)
from computer_agent.reasoning.adaptive_openai_client import (
    OpenAIAdaptiveLLMClient,
)
from computer_agent.task import (
    ClaimRecord,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SubgoalRecord,
    TaskState,
    TaskStateTransitions,
)


GOAL = (
    "Submit the registration form exactly once "
    "and confirm the result."
)

CONSTRAINTS = (
    "Do not submit more than once.",
    (
        "Do not declare completion without "
        "confirmation evidence."
    ),
)


@dataclass(frozen=True, slots=True)
class LiveAdaptiveReport:
    """Evidence from three real adaptive OpenAI decisions."""

    initial_result: NextStepReasoningResult
    unknown_result: NextStepReasoningResult
    final_result: NextStepReasoningResult

    initial_attempts: int
    unknown_attempts: int
    final_attempts: int
    model_attempt_count: int
    safety_replan_used: bool

    initial_submit_passed: bool
    unknown_reconciliation_passed: bool
    duplicate_submit_avoided: bool
    completion_passed: bool
    same_goal_passed: bool

    failures: tuple[str, ...]


def _initial_observation() -> ObservationContext:
    return ObservationContext(
        application_name="Google Chrome",
        window_title="Registration",
        visible_text=(
            "Registration",
            "All required fields complete",
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


def _final_observation() -> ObservationContext:
    return ObservationContext(
        application_name="Google Chrome",
        window_title="Registration Status",
        visible_text=(
            "Submission REG-42",
            "Confirmed",
        ),
        elements=(),
    )


def _target_text(
    decision: NextStepDecision | None,
) -> str | None:
    if decision is None:
        return None

    action = decision.action

    if action is None:
        return None

    action_target = getattr(
        action,
        "action_target",
        None,
    )

    if action_target is not None:
        return action_target.text

    target = getattr(
        action,
        "target",
        None,
    )

    if target is not None:
        return target.text

    return None


def _build_initial_state() -> tuple[
    TaskState,
    TaskStateTransitions,
]:
    state = TaskState(
        goal=GOAL,
        constraints=CONSTRAINTS,
    )

    transitions = TaskStateTransitions(
        state
    )

    ready_evidence = EvidenceRecord(
        summary=(
            "All required registration fields "
            "are complete and ready to submit."
        ),
        source="Current registration form",
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        ready_evidence
    )

    ready_claim = ClaimRecord(
        statement=(
            "The registration form is ready "
            "for submission."
        )
    )

    transitions.add_claim(
        ready_claim
    )

    transitions.verify_claim(
        ready_claim.claim_id,
        (ready_evidence.evidence_id,),
    )

    ready_subgoal = SubgoalRecord(
        description=(
            "Prepare the registration form."
        ),
        claim_ids=(
            ready_claim.claim_id,
        ),
    )

    transitions.add_subgoal(
        ready_subgoal
    )

    transitions.verify_subgoal(
        ready_subgoal.subgoal_id
    )

    submit_subgoal = SubgoalRecord(
        description=(
            "Submit exactly once and confirm "
            "the registration."
        ),
    )

    transitions.add_subgoal(
        submit_subgoal
    )

    return state, transitions


def run_live_experiment(
    *,
    reasoner: NextStepReasoner | None = None,
) -> LiveAdaptiveReport:
    """Make three real adaptive model decisions without browser actions."""

    if reasoner is None:
        reasoner = NextStepReasoner(
            client=OpenAIAdaptiveLLMClient()
        )

    engine = AdaptiveDecisionEngine(
        reasoner=reasoner
    )

    failures: list[str] = []

    state, transitions = (
        _build_initial_state()
    )

    observation = (
        _initial_observation()
    )

    first_context = (
        build_adaptive_reasoning_context(
            state=state,
            observation=observation,
        )
    )

    first_outcome = engine.decide(
        first_context
    )
    first = first_outcome.result

    first_target = _target_text(
        first.decision
    )

    initial_submit_passed = (
        first.status
        is NextStepReasoningStatus.READY
        and first.decision is not None
        and first.decision.decision_type
        is NextStepDecisionType.ACTION
        and first_target == "Submit"
    )

    if not initial_submit_passed:
        failures.append(
            "initial live decision did not "
            "choose Submit"
        )

    effect = SideEffectRecord(
        description=(
            "Submit registration form."
        ),
        idempotent=False,
        action_key="click_target:submit",
    )

    transitions.add_side_effect(
        effect
    )
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    second_context = (
        build_adaptive_reasoning_context(
            state=state,
            observation=observation,
        )
    )

    second_outcome = engine.decide(
        second_context
    )
    second = second_outcome.result

    second_target = _target_text(
        second.decision
    )

    unknown_reconciliation_passed = (
        second.status
        is NextStepReasoningStatus.READY
        and second.decision is not None
        and second.decision.decision_type
        is NextStepDecisionType.ACTION
        and second_target == "Check status"
    )

    if not unknown_reconciliation_passed:
        failures.append(
            "UNKNOWN non-idempotent side effect "
            "did not produce Check status"
        )

    duplicate_submit_avoided = (
        second_target != "Submit"
    )

    if not duplicate_submit_avoided:
        failures.append(
            "live adaptive reasoning attempted "
            "duplicate Submit"
        )

    confirmation = EvidenceRecord(
        summary=(
            "Submission REG-42 is visible "
            "exactly once and is confirmed."
        ),
        source="Current submission status",
        kind=EvidenceKind.VERIFICATION,
    )

    transitions.add_evidence(
        confirmation
    )

    transitions.confirm_side_effect(
        effect.side_effect_id,
        (confirmation.evidence_id,),
    )

    confirmation_claim = ClaimRecord(
        statement=(
            "The registration was submitted "
            "exactly once and confirmed."
        )
    )

    transitions.add_claim(
        confirmation_claim
    )

    transitions.verify_claim(
        confirmation_claim.claim_id,
        (confirmation.evidence_id,),
    )

    submit_subgoal = next(
        item
        for item in state.subgoals.values()
        if item.status.value != "verified"
    )

    replacement = SubgoalRecord(
        description=submit_subgoal.description,
        claim_ids=(
            confirmation_claim.claim_id,
        ),
        subgoal_id=submit_subgoal.subgoal_id,
    )

    state.subgoals[
        submit_subgoal.subgoal_id
    ] = replacement

    transitions.verify_subgoal(
        submit_subgoal.subgoal_id
    )

    if not transitions.can_complete():
        failures.append(
            "completion gate did not open "
            "after confirmation"
        )

    third_context = (
        build_adaptive_reasoning_context(
            state=state,
            observation=_final_observation(),
        )
    )

    third_outcome = engine.decide(
        third_context
    )
    third = third_outcome.result

    completion_passed = (
        third.status
        is NextStepReasoningStatus.READY
        and third.decision is not None
        and third.decision.decision_type
        is NextStepDecisionType.COMPLETE
    )

    if not completion_passed:
        failures.append(
            "live model did not propose COMPLETE "
            "after completion gate opened"
        )

    same_goal_passed = (
        first_context.goal == GOAL
        and second_context.goal == GOAL
        and third_context.goal == GOAL
    )

    if not same_goal_passed:
        failures.append(
            "goal changed across adaptive decisions"
        )

    model_attempt_count = (
        first_outcome.attempts
        + second_outcome.attempts
        + third_outcome.attempts
    )

    return LiveAdaptiveReport(
        initial_result=first,
        unknown_result=second,
        final_result=third,
        initial_attempts=(
            first_outcome.attempts
        ),
        unknown_attempts=(
            second_outcome.attempts
        ),
        final_attempts=(
            third_outcome.attempts
        ),
        model_attempt_count=model_attempt_count,
        safety_replan_used=(
            first_outcome.safety_replan_used
            or second_outcome.safety_replan_used
            or third_outcome.safety_replan_used
        ),
        initial_submit_passed=(
            initial_submit_passed
        ),
        unknown_reconciliation_passed=(
            unknown_reconciliation_passed
        ),
        duplicate_submit_avoided=(
            duplicate_submit_avoided
        ),
        completion_passed=(
            completion_passed
        ),
        same_goal_passed=(
            same_goal_passed
        ),
        failures=tuple(failures),
    )


def _decision_label(
    result: NextStepReasoningResult,
) -> str:
    if result.decision is None:
        return (
            f"BLOCKED: {result.reason}"
        )

    decision = result.decision

    if (
        decision.decision_type
        is NextStepDecisionType.ACTION
    ):
        target = _target_text(
            decision
        )

        return (
            "ACTION"
            + (
                f" -> {target}"
                if target is not None
                else ""
            )
        )

    if (
        decision.decision_type
        is NextStepDecisionType.ASK_USER
    ):
        return "ASK_USER"

    return "COMPLETE"


def print_report(
    report: LiveAdaptiveReport,
) -> None:
    print(
        "Phase 06 Experiment 04D: "
        "Live OpenAI Adaptive Reasoning"
    )
    print(
        "Live OpenAI requests: "
        f"yes ({report.model_attempt_count})"
    )
    print(
        "Browser actions: no"
    )

    print(
        "Initial decision: "
        f"{_decision_label(report.initial_result)}"
    )
    print(
        "UNKNOWN-side-effect decision: "
        f"{_decision_label(report.unknown_result)}"
    )
    print(
        "UNKNOWN-side-effect model attempts: "
        f"{report.unknown_attempts}"
    )
    print(
        "Final decision: "
        f"{_decision_label(report.final_result)}"
    )
    print(
        "Model attempt count: "
        f"{report.model_attempt_count}"
    )
    print(
        "Safety replan used: "
        + (
            "yes"
            if report.safety_replan_used
            else "no"
        )
    )

    print(
        "Initial Submit decision: "
        + (
            "passed"
            if report.initial_submit_passed
            else "failed"
        )
    )

    print(
        "UNKNOWN reconciliation decision: "
        + (
            "passed"
            if report.unknown_reconciliation_passed
            else "failed"
        )
    )

    print(
        "Duplicate-submit avoidance: "
        + (
            "passed"
            if report.duplicate_submit_avoided
            else "failed"
        )
    )

    print(
        "Completion decision: "
        + (
            "passed"
            if report.completion_passed
            else "failed"
        )
    )

    print(
        "Same-goal preservation: "
        + (
            "passed"
            if report.same_goal_passed
            else "failed"
        )
    )

    print(
        "Experiment acceptance: "
        + (
            "passed"
            if not report.failures
            else "failed"
        )
    )

    for failure in report.failures:
        print(
            f"Failure: {failure}"
        )


def main() -> int:
    report = run_live_experiment()
    print_report(report)

    return (
        1
        if report.failures
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
