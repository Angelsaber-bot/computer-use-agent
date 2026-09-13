"""Tests for observation-conditioned one-step reasoning."""

from __future__ import annotations

import json

from computer_agent.planning import (
    PlanOperation,
    PlanStep,
)
from computer_agent.reasoning import (
    AdaptiveReasoningContext,
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
)


class FakeLLMClient:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


class FailingLLMClient:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise RuntimeError(
            "provider unavailable"
        )


def make_context(
    *,
    completion_allowed: bool = False,
) -> AdaptiveReasoningContext:
    blockers = (
        ()
        if completion_allowed
        else ("subgoal is not verified: Submit form.",)
    )

    return AdaptiveReasoningContext(
        goal="Submit the form exactly once.",
        constraints=(
            "Do not submit twice.",
        ),
        task_status="running",
        verified_subgoals=(
            "Fill required fields.",
        ),
        unresolved_subgoals=(
            "pending: Submit form.",
        ),
        current_evidence=(
            "Required fields are complete. "
            "| source=Current form",
        ),
        stale_or_unknown_evidence=(),
        unresolved_side_effects=(),
        pending_questions=(),
        completion_allowed=(
            completion_allowed
        ),
        completion_blockers=blockers,
        observation=ObservationContext(
            application_name="Google Chrome",
            window_title="Registration",
            visible_text=(
                "Registration",
                "Submit",
            ),
            elements=(
                ObservedElement(
                    text="Submit",
                    element_type="button",
                    enabled=True,
                ),
            ),
        ),
    )


def test_action_decision_returns_exactly_one_step() -> None:
    response = json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "A confirmation element should appear."
            ),
            "action": {
                "goal": "Submit the completed form.",
                "operation": "click_target",
                "action_target": {
                    "text": "Submit",
                    "element_types": [
                        "button"
                    ],
                },
                "verification_target": {
                    "text": "Confirmation",
                    "element_types": [],
                },
                "max_attempts": 1,
            },
        }
    )

    client = FakeLLMClient(response)
    reasoner = NextStepReasoner(
        client=client
    )

    result = reasoner.reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.READY
    )
    assert result.decision is not None
    assert (
        result.decision.decision_type
        is NextStepDecisionType.ACTION
    )

    action = result.decision.action

    assert isinstance(
        action,
        PlanStep,
    )
    assert (
        action.operation
        is PlanOperation.CLICK_TARGET
    )
    assert (
        action.action_target.text
        == "Submit"
    )
    assert action.max_attempts == 1


def test_reasoning_prompt_contains_current_observation() -> None:
    response = json.dumps(
        {
            "decision": "ask_user",
            "question": (
                "Should I continue with this form?"
            ),
        }
    )

    client = FakeLLMClient(response)
    reasoner = NextStepReasoner(
        client=client
    )

    reasoner.reason(
        make_context()
    )

    assert client.user_prompt is not None

    assert "Google Chrome" in client.user_prompt
    assert "Registration" in client.user_prompt
    assert "Submit" in client.user_prompt
    assert "Do not submit twice." in client.user_prompt


def test_reasoning_prompt_contains_blocked_action_keys() -> None:
    from dataclasses import replace

    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "Should I inspect the current status?"
                ),
            }
        )
    )
    reasoner = NextStepReasoner(
        client=client
    )

    reasoner.reason(
        replace(
            make_context(),
            blocked_action_keys=(
                "click_target:submit",
            ),
        )
    )

    assert client.user_prompt is not None
    assert (
        '"blocked_action_keys":["click_target:submit"]'
        in client.user_prompt
    )


def test_reasoning_prompt_contains_decision_feedback() -> None:
    from dataclasses import replace

    feedback = (
        "Previous proposal was rejected.",
    )
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "Should I inspect the current status?"
                ),
            }
        )
    )
    reasoner = NextStepReasoner(
        client=client
    )

    reasoner.reason(
        replace(
            make_context(),
            decision_feedback=feedback,
        )
    )

    assert client.user_prompt is not None
    assert (
        '"decision_feedback":["Previous proposal was rejected."]'
        in client.user_prompt
    )


def test_ask_user_decision_is_supported() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "Should I overwrite the existing value?"
                ),
            }
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.READY
    )
    assert result.decision is not None
    assert (
        result.decision.decision_type
        is NextStepDecisionType.ASK_USER
    )
    assert (
        result.decision.question
        == "Should I overwrite the existing value?"
    )


def test_complete_decision_allowed_when_gate_is_open() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "complete",
                "summary": (
                    "All required task state is verified."
                ),
            }
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context(
            completion_allowed=True
        )
    )

    assert (
        result.status
        is NextStepReasoningStatus.READY
    )
    assert result.decision is not None
    assert (
        result.decision.decision_type
        is NextStepDecisionType.COMPLETE
    )


def test_complete_decision_is_rejected_when_gate_blocked() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "complete",
                "summary": "The task looks complete.",
            }
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context(
            completion_allowed=False
        )
    )

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )
    assert result.decision is None
    assert (
        "completion gate is blocked"
        in result.reason
    )


def test_extra_top_level_keys_are_rejected() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "ask_user",
                "question": "Continue?",
                "reasoning": (
                    "I think this is best."
                ),
            }
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )
    assert result.decision is None


def test_adaptive_action_rejects_multiple_attempts() -> None:
    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "action",
                "expected_effect": (
                    "Confirmation appears."
                ),
                "action": {
                    "goal": "Submit form.",
                    "operation": "click_target",
                    "action_target": {
                        "text": "Submit",
                        "element_types": [
                            "button"
                        ],
                    },
                    "verification_target": {
                        "text": "Confirmation",
                        "element_types": [],
                    },
                    "max_attempts": 2,
                },
            }
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )


def test_duplicate_json_keys_are_rejected() -> None:
    client = FakeLLMClient(
        (
            '{"decision":"ask_user",'
            '"decision":"complete",'
            '"question":"Continue?"}'
        )
    )

    result = NextStepReasoner(
        client=client
    ).reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )


def test_generation_failure_fails_closed() -> None:
    reasoner = NextStepReasoner(
        client=FailingLLMClient()
    )

    result = reasoner.reason(
        make_context()
    )

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )
    assert result.decision is None
    assert result.reason == (
        "LLM generation failed"
    )


def test_unknown_non_idempotent_effect_is_in_context() -> None:
    context = AdaptiveReasoningContext(
        goal="Submit exactly once.",
        constraints=(
            "Never duplicate submission.",
        ),
        task_status="running",
        verified_subgoals=(),
        unresolved_subgoals=(
            "unknown: Submit form.",
        ),
        current_evidence=(),
        stale_or_unknown_evidence=(),
        unresolved_side_effects=(
            (
                "unknown: Submit form. "
                "| idempotent=False"
            ),
        ),
        pending_questions=(),
        completion_allowed=False,
        completion_blockers=(
            (
                "side effect is unresolved: "
                "Submit form."
            ),
        ),
        observation=ObservationContext(
            visible_text=(
                "Submission status",
            ),
        ),
    )

    client = FakeLLMClient(
        json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "I cannot safely confirm the prior "
                    "submission. Should I continue?"
                ),
            }
        )
    )

    NextStepReasoner(
        client=client
    ).reason(context)

    assert client.user_prompt is not None
    assert (
        "idempotent=False"
        in client.user_prompt
    )


def test_unknown_non_idempotent_retry_is_deterministically_blocked() -> None:
    from dataclasses import replace

    response = json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "The form should submit."
            ),
            "action": {
                "goal": "Submit again.",
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

    context = replace(
        make_context(),
        blocked_action_keys=(
            "click_target:submit",
        ),
        unresolved_side_effects=(
            (
                "unknown: Submit the form. "
                "| idempotent=False"
            ),
        ),
    )

    result = NextStepReasoner(
        client=FakeLLMClient(response)
    ).reason(context)

    assert (
        result.status
        is NextStepReasoningStatus.BLOCKED
    )
    assert result.decision is None
    assert (
        "unsafe retry"
        in result.reason
    )
