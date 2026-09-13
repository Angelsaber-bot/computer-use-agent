"""Offline tests for the live OpenAI adaptive harness."""

from __future__ import annotations

import json

from computer_agent.reasoning import (
    NextStepReasoner,
)
from experiments.phase06_interactive_agent import (
    experiment_04_live_openai_adaptive_reasoning
    as experiment,
)


class FakeAdaptiveClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            user_prompt
        )

        prefix = (
            "Current task context:\n"
        )

        context = json.loads(
            user_prompt[
                len(prefix):
            ]
        )

        if context[
            "completion_allowed"
        ]:
            return json.dumps(
                {
                    "decision": "complete",
                    "summary": (
                        "Confirmed complete."
                    ),
                }
            )

        unresolved = context[
            "unresolved_side_effects"
        ]

        if any(
            item.startswith(
                "unknown:"
            )
            for item in unresolved
        ):
            if not context[
                "decision_feedback"
            ]:
                return json.dumps(
                    {
                        "decision": "action",
                        "expected_effect": (
                            "Submission status changes."
                        ),
                        "action": {
                            "goal": (
                                "Submit registration."
                            ),
                            "operation": (
                                "click_target"
                            ),
                            "action_target": {
                                "text": "Submit",
                                "element_types": [
                                    "button"
                                ],
                            },
                            "verification_target": {
                                "text": (
                                    "Submission status"
                                ),
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
                        "Confirmation appears."
                    ),
                    "action": {
                        "goal": (
                            "Check prior submission."
                        ),
                        "operation": (
                            "click_target"
                        ),
                        "action_target": {
                            "text": (
                                "Check status"
                            ),
                            "element_types": [
                                "button"
                            ],
                        },
                        "verification_target": {
                            "text": (
                                "Submission REG-42"
                            ),
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
                    "Submission status changes."
                ),
                "action": {
                    "goal": (
                        "Submit registration."
                    ),
                    "operation": (
                        "click_target"
                    ),
                    "action_target": {
                        "text": "Submit",
                        "element_types": [
                            "button"
                        ],
                    },
                    "verification_target": {
                        "text": (
                            "Submission status"
                        ),
                        "element_types": [],
                    },
                    "max_attempts": 1,
                },
            }
        )


def test_live_harness_with_fake_reasoner() -> None:
    client = FakeAdaptiveClient()
    reasoner = NextStepReasoner(
        client=client
    )

    report = experiment.run_live_experiment(
        reasoner=reasoner
    )

    assert (
        report.initial_submit_passed
        is True
    )
    assert (
        report.unknown_reconciliation_passed
        is True
    )
    assert (
        report.duplicate_submit_avoided
        is True
    )
    assert (
        report.completion_passed
        is True
    )
    assert (
        report.same_goal_passed
        is True
    )
    assert (
        report.safety_replan_used
        is True
    )
    assert report.unknown_attempts == 2
    assert report.model_attempt_count == 4
    assert len(client.calls) == 4
    assert (
        "decision_feedback"
        in client.calls[2]
    )
    assert report.failures == ()
