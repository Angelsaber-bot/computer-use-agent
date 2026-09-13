"""Tests for bounded safety-triggered adaptive replanning."""

from __future__ import annotations

import json

from computer_agent.reasoning import (
    AdaptiveDecisionEngine,
    AdaptiveReasoningContext,
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
)


class ScriptedClient:
    def __init__(
        self,
        responses: tuple[str, ...],
    ) -> None:
        self.responses = responses
        self.calls: list[
            dict[str, str]
        ] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        index = len(self.calls) - 1

        if index >= len(
            self.responses
        ):
            raise RuntimeError(
                "unexpected additional model call"
            )

        return self.responses[index]


def _submit_response() -> str:
    return json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "Submission status changes."
            ),
            "action": {
                "goal": "Submit registration.",
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


def _check_status_response() -> str:
    return json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "Existing submission state becomes visible."
            ),
            "action": {
                "goal": (
                    "Reconcile the prior submission."
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


def _context() -> AdaptiveReasoningContext:
    return AdaptiveReasoningContext(
        goal=(
            "Submit the registration exactly once."
        ),
        constraints=(
            "Do not submit twice.",
        ),
        task_status="running",
        verified_subgoals=(),
        unresolved_subgoals=(
            "unknown: Submit registration.",
        ),
        current_evidence=(),
        stale_or_unknown_evidence=(),
        unresolved_side_effects=(
            (
                "unknown: Submit registration form. "
                "| idempotent=False"
            ),
        ),
        pending_questions=(),
        completion_allowed=False,
        completion_blockers=(
            (
                "side effect is unresolved: "
                "Submit registration form."
            ),
        ),
        observation=ObservationContext(
            visible_text=(
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
        ),
        blocked_action_keys=(
            "click_target:submit",
        ),
    )


def test_unsafe_proposal_triggers_one_safe_replan() -> None:
    client = ScriptedClient(
        (
            _submit_response(),
            _check_status_response(),
        )
    )

    engine = AdaptiveDecisionEngine(
        reasoner=NextStepReasoner(
            client=client
        )
    )

    outcome = engine.decide(
        _context()
    )

    assert outcome.attempts == 2
    assert (
        outcome.safety_replan_used
        is True
    )

    assert (
        outcome.result.status
        is NextStepReasoningStatus.READY
    )

    assert outcome.result.decision is not None
    assert (
        outcome.result.decision.decision_type
        is NextStepDecisionType.ACTION
    )

    action = outcome.result.decision.action

    assert action is not None
    assert (
        action.action_target.text
        == "Check status"
    )

    assert len(client.calls) == 2
    assert (
        "decision_feedback"
        in client.calls[1]["user_prompt"]
    )


def test_second_unsafe_proposal_is_not_retried_again() -> None:
    client = ScriptedClient(
        (
            _submit_response(),
            _submit_response(),
        )
    )

    engine = AdaptiveDecisionEngine(
        reasoner=NextStepReasoner(
            client=client
        )
    )

    outcome = engine.decide(
        _context()
    )

    assert outcome.attempts == 2
    assert len(client.calls) == 2

    assert (
        outcome.result.status
        is NextStepReasoningStatus.BLOCKED
    )


def test_non_safety_failure_does_not_trigger_replan() -> None:
    client = ScriptedClient(
        (
            "not-json",
        )
    )

    engine = AdaptiveDecisionEngine(
        reasoner=NextStepReasoner(
            client=client
        )
    )

    outcome = engine.decide(
        _context()
    )

    assert outcome.attempts == 1
    assert (
        outcome.safety_replan_used
        is False
    )
    assert len(client.calls) == 1
