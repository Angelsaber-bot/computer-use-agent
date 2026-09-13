"""Phase 06 Experiment 04: adaptive next-step reasoning."""

from __future__ import annotations

import json
from dataclasses import dataclass

from computer_agent.reasoning import (
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
    build_adaptive_reasoning_context,
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


class ContextAwareDemoClient:
    """Deterministic adaptive policy behind the LLMClient boundary."""

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

        unresolved_effects = context[
            "unresolved_side_effects"
        ]
        completion_allowed = context[
            "completion_allowed"
        ]
        visible_text = context[
            "observation"
        ]["visible_text"]

        if completion_allowed:
            return json.dumps(
                {
                    "decision": "complete",
                    "summary": (
                        "The registration is confirmed "
                        "exactly once."
                    ),
                }
            )

        if any(
            item.startswith("unknown:")
            and "idempotent=False" in item
            for item in unresolved_effects
        ):
            if "Check status" not in visible_text:
                return json.dumps(
                    {
                        "decision": "ask_user",
                        "question": (
                            "The previous submission outcome "
                            "is unknown and cannot be safely "
                            "repeated. Should I continue?"
                        ),
                    }
                )

            return json.dumps(
                {
                    "decision": "action",
                    "expected_effect": (
                        "A submission confirmation should "
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

        if "Submit" in visible_text:
            return json.dumps(
                {
                    "decision": "action",
                    "expected_effect": (
                        "The registration should enter "
                        "a submitted or pending state."
                    ),
                    "action": {
                        "goal": (
                            "Submit the completed "
                            "registration form."
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

        return json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "The required next control is not "
                    "visible. Should I continue?"
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    initial_action_passed: bool
    state_adaptation_passed: bool
    no_duplicate_submit_passed: bool
    reconciliation_passed: bool
    completion_gate_passed: bool
    same_goal_passed: bool
    failures: tuple[str, ...]


def _observation() -> ObservationContext:
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


def run_experiment() -> ExperimentReport:
    failures: list[str] = []

    state = TaskState(
        goal=GOAL,
        constraints=(
            "Do not submit more than once.",
            (
                "Do not declare completion without "
                "confirmation evidence."
            ),
        ),
    )
    transitions = TaskStateTransitions(state)

    client = ContextAwareDemoClient()
    reasoner = NextStepReasoner(
        client=client
    )

    observation = _observation()

    first_context = build_adaptive_reasoning_context(
        state=state,
        observation=observation,
    )
    first = reasoner.reason(
        first_context
    )

    initial_action_passed = (
        first.status
        is NextStepReasoningStatus.READY
        and first.decision is not None
        and first.decision.decision_type
        is NextStepDecisionType.ACTION
        and first.decision.action is not None
        and getattr(
            first.decision.action,
            "action_target",
            None,
        ) is not None
        and first.decision.action.action_target.text
        == "Submit"
    )

    if not initial_action_passed:
        failures.append(
            "initial state did not choose Submit"
        )

    effect = SideEffectRecord(
        description="Submit registration form.",
        idempotent=False,
        action_key="click_target:submit",
    )
    transitions.add_side_effect(effect)
    transitions.mark_side_effect_executed(
        effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        effect.side_effect_id
    )

    second_context = build_adaptive_reasoning_context(
        state=state,
        observation=observation,
    )
    second = reasoner.reason(
        second_context
    )

    state_adaptation_passed = (
        second.status
        is NextStepReasoningStatus.READY
        and second.decision is not None
        and second.decision.decision_type
        is NextStepDecisionType.ACTION
        and second.decision.action is not None
        and getattr(
            second.decision.action,
            "action_target",
            None,
        ) is not None
        and second.decision.action.action_target.text
        == "Check status"
    )

    if not state_adaptation_passed:
        failures.append(
            "UNKNOWN side effect did not change "
            "the next decision to reconciliation"
        )

    no_duplicate_submit_passed = (
        second.decision is not None
        and second.decision.action is not None
        and getattr(
            second.decision.action,
            "action_target",
            None,
        ) is not None
        and second.decision.action.action_target.text
        != "Submit"
    )

    if not no_duplicate_submit_passed:
        failures.append(
            "adaptive decision repeated Submit "
            "after an UNKNOWN non-idempotent effect"
        )

    confirmation = EvidenceRecord(
        summary=(
            "Submission REG-42 is visible exactly once."
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

    claim = ClaimRecord(
        statement=(
            "The registration was submitted "
            "exactly once."
        )
    )
    transitions.add_claim(claim)
    transitions.verify_claim(
        claim.claim_id,
        (confirmation.evidence_id,),
    )

    subgoal = SubgoalRecord(
        description=(
            "Submit and confirm the registration."
        ),
        claim_ids=(claim.claim_id,),
    )
    transitions.add_subgoal(subgoal)
    transitions.verify_subgoal(
        subgoal.subgoal_id
    )

    reconciliation_passed = (
        transitions.can_complete()
    )

    if not reconciliation_passed:
        failures.append(
            "confirmed side effect did not clear "
            "completion blockers"
        )

    final_observation = ObservationContext(
        application_name="Google Chrome",
        window_title="Registration Status",
        visible_text=(
            "Submission REG-42",
            "Confirmed",
        ),
        elements=(),
    )

    third_context = build_adaptive_reasoning_context(
        state=state,
        observation=final_observation,
    )
    third = reasoner.reason(
        third_context
    )

    completion_gate_passed = (
        third.status
        is NextStepReasoningStatus.READY
        and third.decision is not None
        and third.decision.decision_type
        is NextStepDecisionType.COMPLETE
    )

    if not completion_gate_passed:
        failures.append(
            "reasoner did not propose completion "
            "after the deterministic gate opened"
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
            "adaptive decisions did not preserve "
            "the same task goal"
        )

    return ExperimentReport(
        initial_action_passed=(
            initial_action_passed
        ),
        state_adaptation_passed=(
            state_adaptation_passed
        ),
        no_duplicate_submit_passed=(
            no_duplicate_submit_passed
        ),
        reconciliation_passed=(
            reconciliation_passed
        ),
        completion_gate_passed=(
            completion_gate_passed
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
        return "passed" if value else "failed"

    print(
        "Phase 06 Experiment 04: "
        "Adaptive Next-Step Reasoning"
    )
    print(
        "Initial next-step action: "
        f"{result(report.initial_action_passed)}"
    )
    print(
        "Task-state adaptation: "
        f"{result(report.state_adaptation_passed)}"
    )
    print(
        "Duplicate-submit avoidance: "
        f"{result(report.no_duplicate_submit_passed)}"
    )
    print(
        "Reconciliation state update: "
        f"{result(report.reconciliation_passed)}"
    )
    print(
        "Completion-gate reasoning: "
        f"{result(report.completion_gate_passed)}"
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
        print(f"Failure: {failure}")


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
