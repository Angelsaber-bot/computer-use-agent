"""Deterministic in-app Phase 06 evidence and reasoning demo."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import replace

from computer_agent.reasoning import (
    AdaptiveDecisionEngine,
    AdaptiveDecisionOutcome,
    AdaptiveDecisionSnapshot,
    NextStepDecisionType,
    NextStepReasoner,
    NextStepReasoningStatus,
    ObservationContext,
    ObservedElement,
    build_adaptive_reasoning_context,
)
from computer_agent.runtime import (
    RuntimeControl,
    RuntimeTask,
    RuntimeWorker,
)
from computer_agent.task import (
    ArtifactRecord,
    ClaimRecord,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SubgoalRecord,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
)


TaskStatePublisher = Callable[[], None]
AdaptiveDecisionPublisher = Callable[
    [AdaptiveDecisionSnapshot],
    None,
]

_DEMO_CONSTRAINTS = (
    "Do not submit more than once.",
    "Do not declare completion without confirmation evidence.",
)


class DeterministicAdaptiveDemoClient:
    """Deterministic fake LLM behind the production reasoner boundary."""

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

        if context["completion_allowed"]:
            return json.dumps(
                {
                    "decision": "complete",
                    "summary": (
                        "The registration is confirmed "
                        "exactly once."
                    ),
                }
            )

        blocked = set(
            context["blocked_action_keys"]
        )
        feedback = context["decision_feedback"]
        visible_text = context["observation"][
            "visible_text"
        ]

        if "click_target:submit" in blocked:
            if feedback and "Check status" in visible_text:
                return _check_status_response()

            return _submit_response()

        if "Submit" in visible_text:
            return _submit_response()

        return json.dumps(
            {
                "decision": "ask_user",
                "question": (
                    "The next safe control is not visible. "
                    "Should I continue?"
                ),
            }
        )


def create_evidence_demo_worker(
    state: TaskState,
    publish_state: TaskStatePublisher,
    publish_decision: AdaptiveDecisionPublisher,
    *,
    stage_delay: float = 0.75,
) -> RuntimeWorker:
    """Create the integrated 06.02 + 06.03 + 06.04 demo worker."""

    if not isinstance(state, TaskState):
        raise ValueError("state must be a TaskState")

    if not callable(publish_state):
        raise ValueError(
            "publish_state must be callable"
        )

    if not callable(publish_decision):
        raise ValueError(
            "publish_decision must be callable"
        )

    if (
        isinstance(stage_delay, bool)
        or not isinstance(stage_delay, int | float)
        or stage_delay < 0
    ):
        raise ValueError(
            "stage_delay must be a non-negative number"
        )

    def worker(
        task: RuntimeTask,
        control: RuntimeControl,
        progress: Callable[[str], None],
    ) -> None:
        transitions = TaskStateTransitions(state)
        engine = AdaptiveDecisionEngine(
            reasoner=NextStepReasoner(
                client=DeterministicAdaptiveDemoClient()
            )
        )

        state.status = TaskStateStatus.RUNNING
        state.constraints = (
            state.constraints
            or _DEMO_CONSTRAINTS
        )
        state.touch()
        publish_state()

        control.checkpoint()

        subgoal = SubgoalRecord(
            description=(
                "Submit registration exactly once "
                "and confirm result."
            )
        )
        transitions.add_subgoal(subgoal)

        ready_evidence = EvidenceRecord(
            summary=(
                "Registration controls are visible: "
                "Submit, Check status, Submission status."
            ),
            source="Deterministic current observation",
            kind=EvidenceKind.OBSERVATION,
        )
        transitions.add_evidence(ready_evidence)
        publish_state()

        observation = _initial_observation()

        first = _decide_and_publish(
            state=state,
            observation=observation,
            engine=engine,
            publish_decision=publish_decision,
            progress=progress,
        )
        _pauseable_delay(control, stage_delay)

        _require_action_target(
            first,
            "Submit",
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
        publish_state()
        progress(
            "Submit registration form state is UNKNOWN; "
            "click_target:submit is blocked."
        )
        _pauseable_delay(control, stage_delay)

        second = _decide_and_publish(
            state=state,
            observation=observation,
            engine=engine,
            publish_decision=publish_decision,
            progress=progress,
        )
        progress(
            "Duplicate Submit executed = 0."
        )
        _pauseable_delay(control, stage_delay)

        _require_action_target(
            second,
            "Check status",
        )

        confirmation = EvidenceRecord(
            summary=(
                "Submission REG-42 is visible exactly "
                "once and confirmed."
            ),
            source="Current submission status",
            kind=EvidenceKind.VERIFICATION,
        )
        transitions.add_evidence(confirmation)
        transitions.confirm_side_effect(
            effect.side_effect_id,
            (confirmation.evidence_id,),
        )

        claim = ClaimRecord(
            statement=(
                "The registration was submitted "
                "exactly once and confirmed."
            )
        )
        transitions.add_claim(claim)
        transitions.verify_claim(
            claim.claim_id,
            (confirmation.evidence_id,),
        )

        state.subgoals[subgoal.subgoal_id] = replace(
            state.subgoals[subgoal.subgoal_id],
            claim_ids=(claim.claim_id,),
        )
        transitions.verify_subgoal(
            subgoal.subgoal_id
        )

        artifact = ArtifactRecord(
            description="Confirmed registration result",
            location="demo://submission/REG-42",
            evidence_ids=(
                confirmation.evidence_id,
            ),
        )
        transitions.add_artifact(artifact)

        progress(
            "Confirmation evidence recorded."
        )
        progress(
            "Completion gate opened."
        )
        transitions.complete_task()
        publish_state()
        _pauseable_delay(control, stage_delay)

        third = _decide_and_publish(
            state=state,
            observation=_final_observation(),
            engine=engine,
            publish_decision=publish_decision,
            progress=progress,
        )
        _require_decision_type(
            third,
            NextStepDecisionType.COMPLETE,
        )
        _pauseable_delay(control, stage_delay)

    return worker


def _decide_and_publish(
    *,
    state: TaskState,
    observation: ObservationContext,
    engine: AdaptiveDecisionEngine,
    publish_decision: AdaptiveDecisionPublisher,
    progress: Callable[[str], None],
) -> AdaptiveDecisionOutcome:
    progress("Observed current UI.")

    context = build_adaptive_reasoning_context(
        state=state,
        observation=observation,
    )
    progress("Built adaptive reasoning context.")

    outcome = engine.decide(context)

    for index, result in enumerate(
        outcome.attempt_results,
        start=1,
    ):
        if (
            index == 2
            and outcome.safety_replan_used
        ):
            progress("Safety replan attempt 2.")

        progress(
            "Model proposed "
            f"{_decision_label(result)}."
        )

        if (
            result.status
            is NextStepReasoningStatus.READY
        ):
            progress(
                "Accepted "
                f"{_decision_label(result)}."
            )
        elif result.rejected_decision is not None:
            progress(
                "Safety gate rejected blocked action."
            )
        else:
            progress(
                "Adaptive decision blocked: "
                f"{result.reason}."
            )

    publish_decision(
        AdaptiveDecisionSnapshot.from_outcome(
            context=context,
            outcome=outcome,
        )
    )

    return outcome


def _initial_observation() -> ObservationContext:
    return ObservationContext(
        application_name="Google Chrome",
        window_title="Registration",
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


def _submit_response() -> str:
    return json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "The registration should enter "
                "a submitted or pending state."
            ),
            "action": {
                "goal": (
                    "Submit the completed registration form."
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


def _check_status_response() -> str:
    return json.dumps(
        {
            "decision": "action",
            "expected_effect": (
                "The existing submission state should "
                "become visible."
            ),
            "action": {
                "goal": (
                    "Reconcile the previous submission outcome."
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


def _decision_label(
    result,
) -> str:
    decision = (
        result.decision
        if result.decision is not None
        else result.rejected_decision
    )

    if decision is None:
        return "BLOCKED"

    label = decision.decision_type.value.upper()
    target = _target_text(decision)

    if target is None:
        return label

    return f"{label} -> {target}"


def _target_text(
    decision,
) -> str | None:
    action = decision.action

    if action is None:
        return None

    target = getattr(
        action,
        "action_target",
        None,
    )

    text = getattr(
        target,
        "text",
        None,
    )

    if isinstance(text, str):
        return text

    return None


def _require_action_target(
    outcome: AdaptiveDecisionOutcome,
    expected: str,
) -> None:
    result = outcome.result

    if (
        result.status
        is not NextStepReasoningStatus.READY
        or result.decision is None
        or result.decision.decision_type
        is not NextStepDecisionType.ACTION
        or _target_text(result.decision) != expected
    ):
        raise RuntimeError(
            "deterministic demo decision mismatch: "
            f"expected ACTION -> {expected}"
        )


def _require_decision_type(
    outcome: AdaptiveDecisionOutcome,
    expected: NextStepDecisionType,
) -> None:
    result = outcome.result

    if (
        result.status
        is not NextStepReasoningStatus.READY
        or result.decision is None
        or result.decision.decision_type is not expected
    ):
        raise RuntimeError(
            "deterministic demo decision mismatch: "
            f"expected {expected.value}"
        )


def _pauseable_delay(
    control: RuntimeControl,
    seconds: float,
) -> None:
    deadline = time.monotonic() + seconds

    while True:
        control.checkpoint()

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return

        time.sleep(
            min(remaining, 0.05)
        )
