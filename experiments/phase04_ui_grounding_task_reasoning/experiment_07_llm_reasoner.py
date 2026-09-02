"""Phase 04 experiment for headless LLM semantic reasoning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from typing import Callable

from computer_agent.core.models import Action
from computer_agent.planning import PlanOperation, StructuredPlan
from computer_agent.reasoning import LLMReasoner, ReasoningResult, ReasoningStatus


TITLE = "Phase 04 Experiment 07: LLM Reasoner Formal Harness"
TASK = "Complete the deterministic LLM reasoning workflow"
EXPECTED_TASK_GOAL = TASK

STEP_1_GOAL = "Activate the first reasoning target"
STEP_1_ACTION_TARGET_TEXT = "STEP_1_TARGET_07"
STEP_1_VERIFICATION_TARGET_TEXT = "STEP_1_COMPLETE_07"

STEP_2_GOAL = "Activate the second reasoning target"
STEP_2_ACTION_TARGET_TEXT = "STEP_2_TARGET_07"
STEP_2_VERIFICATION_TARGET_TEXT = "TASK_COMPLETE_07"

EXPECTED_OPERATION = PlanOperation.CLICK_TARGET
EXPECTED_ELEMENT_TYPES: tuple[str, ...] = ()
STEP_MAX_ATTEMPTS = 3

LLM_PROVIDER = "deterministic fake"
LIVE_API_REQUEST = False
OBSERVATION_COUNT = 0
ACTION_EXECUTION_COUNT = 0


@dataclass(frozen=True, slots=True)
class ReasoningRun:
    result: ReasoningResult
    client_call_count: int
    provider: str = LLM_PROVIDER
    live_api_request: bool = LIVE_API_REQUEST
    observation_count: int = OBSERVATION_COUNT
    action_execution_count: int = ACTION_EXECUTION_COUNT


RunBuilder = Callable[[], ReasoningRun]
ExpectedStep = tuple[str, str, str]


EXPECTED_STEPS: tuple[ExpectedStep, ...] = (
    (STEP_1_GOAL, STEP_1_ACTION_TARGET_TEXT, STEP_1_VERIFICATION_TARGET_TEXT),
    (STEP_2_GOAL, STEP_2_ACTION_TARGET_TEXT, STEP_2_VERIFICATION_TARGET_TEXT),
)


class DeterministicFakeLLMClient:
    def __init__(self, response: str | Exception | None = None) -> None:
        if response is None:
            response = build_fake_response_json()

        self._response = response
        self.calls: list[dict[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
        )
        if isinstance(self._response, Exception):
            raise self._response

        return self._response


def build_fake_response_payload() -> dict[str, object]:
    return {
        "task_goal": EXPECTED_TASK_GOAL,
        "steps": [_step_payload(step) for step in EXPECTED_STEPS],
    }


def build_fake_response_json() -> str:
    return json.dumps(build_fake_response_payload())


def run_deterministic_reasoning(
    *,
    client: DeterministicFakeLLMClient | None = None,
) -> ReasoningRun:
    if client is None:
        client = DeterministicFakeLLMClient()

    result = LLMReasoner(client=client).reason(TASK)
    return ReasoningRun(result=result, client_call_count=client.call_count)


def acceptance_failures(run: object) -> tuple[str, ...]:
    if not isinstance(run, ReasoningRun):
        return (
            "run was not a ReasoningRun: "
            f"{type(run).__name__}",
        )

    failures: list[str] = []
    if not isinstance(run.result, ReasoningResult):
        failures.append(
            f"result was not a ReasoningResult: {type(run.result).__name__}"
        )
    else:
        failures.extend(_reasoning_result_failures(run.result))

    expected_run_values = (
        ("fake client call count", run.client_call_count, 1),
        ("LLM provider", run.provider, LLM_PROVIDER),
        ("live API request", run.live_api_request, LIVE_API_REQUEST),
        ("observation count", run.observation_count, OBSERVATION_COUNT),
        (
            "action execution count",
            run.action_execution_count,
            ACTION_EXECUTION_COUNT,
        ),
    )
    for label, actual, expected in expected_run_values:
        _append_mismatch(failures, label, actual, expected)

    return tuple(failures)


def run_acceptance(
    *,
    client: DeterministicFakeLLMClient | None = None,
    run_builder: RunBuilder | None = None,
) -> int:
    print(TITLE)

    try:
        if run_builder is not None:
            run = run_builder()
        else:
            run = run_deterministic_reasoning(client=client)
    except Exception as error:
        print("Experiment acceptance: failed")
        print(
            "Failed condition: reasoning run raised "
            f"{type(error).__name__}: {error}"
        )
        _print_execution_summary()
        return 1

    if isinstance(run, ReasoningRun) and isinstance(run.result.plan, StructuredPlan):
        _print_plan_summary(run.result.plan)

    failures = acceptance_failures(run)
    if failures:
        print("Experiment acceptance: failed")
        for failure in failures:
            print(f"Failed condition: {failure}")
        _print_execution_summary(run)
        return 1

    print("Experiment acceptance: passed")
    _print_execution_summary(run)
    return 0


def _step_payload(step: ExpectedStep) -> dict[str, object]:
    goal, action_text, verification_text = step
    return {
        "goal": goal,
        "operation": EXPECTED_OPERATION.value,
        "action_target": _target_payload(action_text),
        "verification_target": _target_payload(verification_text),
        "max_attempts": STEP_MAX_ATTEMPTS,
    }


def _target_payload(text: str) -> dict[str, object]:
    return {"text": text, "element_types": list(EXPECTED_ELEMENT_TYPES)}


def _reasoning_result_failures(result: ReasoningResult) -> tuple[str, ...]:
    failures: list[str] = []

    if result.status is not ReasoningStatus.READY:
        failures.append(
            "reasoning status was "
            f"{result.status.value}, expected {ReasoningStatus.READY.value}"
        )

    plan = result.plan
    if not isinstance(plan, StructuredPlan):
        failures.append(f"plan was not a StructuredPlan: {type(plan).__name__}")
        return tuple(failures)

    failures.extend(_structured_plan_failures(plan))
    failures.extend(_action_coordinate_failures(plan))
    return tuple(failures)


def _structured_plan_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []

    if plan.task_goal != EXPECTED_TASK_GOAL:
        failures.append(
            f"task goal was {plan.task_goal!r}, "
            f"expected {EXPECTED_TASK_GOAL!r}"
        )

    if len(plan.steps) != len(EXPECTED_STEPS):
        failures.append(
            f"step count was {len(plan.steps)}, "
            f"expected {len(EXPECTED_STEPS)}"
        )

    for index, expected_step in enumerate(EXPECTED_STEPS, start=1):
        if len(plan.steps) < index:
            failures.append(f"step {index} was missing")
            continue

        step = plan.steps[index - 1]
        goal, action_text, verification_text = expected_step
        actual_values = (
            ("goal", step.goal, goal),
            ("operation", step.operation, EXPECTED_OPERATION),
            ("action target text", step.action_target.text, action_text),
            (
                "verification target text",
                step.verification_target.text,
                verification_text,
            ),
            ("max_attempts", step.max_attempts, STEP_MAX_ATTEMPTS),
            (
                "action element_types",
                step.action_target.element_types,
                EXPECTED_ELEMENT_TYPES,
            ),
            (
                "verification element_types",
                step.verification_target.element_types,
                EXPECTED_ELEMENT_TYPES,
            ),
        )
        for label, actual, expected in actual_values:
            _append_mismatch(failures, f"step {index} {label}", actual, expected)

    return tuple(failures)


def _action_coordinate_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []

    if _has_coordinate_authority(plan):
        failures.append("StructuredPlan unexpectedly exposed coordinates")

    for index, step in enumerate(plan.steps, start=1):
        values = (
            step.goal,
            step.operation,
            step.action_target,
            step.verification_target,
            step.max_attempts,
        )
        if any(isinstance(value, Action) for value in values):
            failures.append(f"step {index} unexpectedly contained an Action")

        coordinate_sources = (
            ("step", step),
            ("action target", step.action_target),
            ("verification target", step.verification_target),
        )
        for label, value in coordinate_sources:
            if _has_coordinate_authority(value):
                failures.append(
                    f"step {index} {label} unexpectedly exposed coordinates"
                )

    return tuple(failures)


def _has_coordinate_authority(value: object) -> bool:
    fields = getattr(value, "__dataclass_fields__", {})
    return any(field in fields for field in ("x", "y", "coordinates")) or (
        getattr(value, "reference_point", None) is not None
    )


def _value_text(value: object) -> str:
    if isinstance(value, PlanOperation):
        return value.value

    return repr(value)


def _append_mismatch(
    failures: list[str],
    label: str,
    actual: object,
    expected: object,
) -> None:
    if actual != expected:
        failures.append(
            f"{label} was {_value_text(actual)}, expected {_value_text(expected)}"
        )


def _print_plan_summary(plan: StructuredPlan) -> None:
    print(f"Task goal: {plan.task_goal}")
    print(f"Step count: {len(plan.steps)}")

    for index, step in enumerate(plan.steps, start=1):
        print()
        print(f"Step {index}")
        summary_fields = (
            ("Goal", step.goal),
            ("Operation", step.operation.value),
            ("Action target", step.action_target.text),
            ("Action element types", step.action_target.element_types),
            ("Verification target", step.verification_target.text),
            (
                "Verification element types",
                step.verification_target.element_types,
            ),
            ("Max attempts", step.max_attempts),
        )
        for label, value in summary_fields:
            print(f"{label}: {value}")

    print()


def _print_execution_summary(run: ReasoningRun | None = None) -> None:
    provider = run.provider if run is not None else LLM_PROVIDER
    live_api_request = run.live_api_request if run else LIVE_API_REQUEST
    observation_count = run.observation_count if run else OBSERVATION_COUNT
    action_execution_count = (
        run.action_execution_count if run else ACTION_EXECUTION_COUNT
    )

    if run is not None:
        print(f"LLM client call count: {run.client_call_count}")
    print(f"LLM provider: {provider}")
    print(f"Live API request: {'yes' if live_api_request else 'no'}")
    print(f"Observation count: {observation_count}")
    print(f"Action execution count: {action_execution_count}")


def _parse_args(argv: tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline Experiment 07 LLM reasoner harness.",
    )
    return parser.parse_args(argv)


def main(argv: tuple[str, ...] | None = None) -> int:
    _parse_args(argv)
    return run_acceptance()


if __name__ == "__main__":
    raise SystemExit(main())
