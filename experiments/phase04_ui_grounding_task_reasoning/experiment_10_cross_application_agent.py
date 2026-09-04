"""Phase 04 experiment for deterministic cross-application execution."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time

from computer_agent.agent import AgentLoop, AgentLoopResult, AgentLoopStatus
from computer_agent.agent import AgentStatus
from computer_agent.core.models import Action
from computer_agent.planning import (
    ActivateAppStep,
    InsertTextStep,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    StructuredPlan,
)
from computer_agent.reasoning import LLMReasoner, ReasoningResult
from computer_agent.reasoning import ReasoningStatus

if __package__:
    from .live_harness_utils import (
        build_live_perception_engine,
        build_live_tool_executor,
        live_prerequisites_available,
        wait_for_focus,
        wait_seconds,
    )
else:
    from live_harness_utils import (
        build_live_perception_engine,
        build_live_tool_executor,
        live_prerequisites_available,
        wait_for_focus,
        wait_seconds,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase04_ui_grounding_task_reasoning"
    / "experiment_10_cross_application_agent.html"
)
CAPTURE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase04_ui_grounding_task_reasoning"
    / "experiment_10_cross_application_agent.png"
)

TITLE = "Phase 04 Experiment 10: Cross-Application Agent"
TASK = (
    "Transfer the deterministic browser fixture value "
    "CROSS_APP_TRANSFER_10 into the blank TextEdit document."
)
EXPECTED_TASK_GOAL = TASK
TRANSFER_VALUE = "CROSS_APP_TRANSFER_10"
VALUE_KEY = "transfer_value"
TEXTEDIT_APP_NAME = "TextEdit"
COPY_TARGET_TEXT = "COPY_TRANSFER_VALUE_10"
COPIED_VERIFICATION_TEXT = "TRANSFER_COPIED_10"
EXPECTED_ELEMENT_TYPES = ("button",)
STEP_MAX_ATTEMPTS = 1
EXPECTED_PLAN_STEPS = 4
EXPECTED_ACTION_TOOLS = (
    "click_mouse",
    "read_from_clipboard",
    "activate_app",
    "paste_text",
)
EXPECTED_OPERATIONS = (
    PlanOperation.CLICK_TARGET,
    PlanOperation.READ_CLIPBOARD,
    PlanOperation.ACTIVATE_APP,
    PlanOperation.INSERT_TEXT,
)
EXPECTED_STEP_SPECS = (
    {
        "type": PlanStep,
        "goal": "Copy the transfer value from the browser fixture",
        "operation": PlanOperation.CLICK_TARGET,
        "action_target": COPY_TARGET_TEXT,
        "verification_target": COPIED_VERIFICATION_TEXT,
    },
    {
        "type": ReadClipboardStep,
        "goal": "Read and verify the copied transfer value",
        "operation": PlanOperation.READ_CLIPBOARD,
        "value_key": VALUE_KEY,
        "expected_text": TRANSFER_VALUE,
    },
    {
        "type": ActivateAppStep,
        "goal": "Switch to TextEdit",
        "operation": PlanOperation.ACTIVATE_APP,
        "app_name": TEXTEDIT_APP_NAME,
    },
    {
        "type": InsertTextStep,
        "goal": "Insert the verified transfer value",
        "operation": PlanOperation.INSERT_TEXT,
        "value_key": VALUE_KEY,
    },
)
ALLOWED_APP_NAMES = frozenset((TEXTEDIT_APP_NAME,))
LLM_PROVIDER = "deterministic fake"
LIVE_API_REQUEST = False
DEFAULT_WAIT_SECONDS = 8


@dataclass(frozen=True, slots=True)
class ReasoningRun:
    result: ReasoningResult
    client_call_count: int
    provider: str = LLM_PROVIDER
    live_api_request: bool = LIVE_API_REQUEST


class DeterministicFakeLLMClient:
    def __init__(self, response: str | Exception | None = None) -> None:
        self._response = response if response is not None else build_fake_response_json()
        self.calls: list[dict[str, str]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def build_fake_response_payload() -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for spec in EXPECTED_STEP_SPECS:
        step = {
            "goal": spec["goal"],
            "operation": spec["operation"].value,
            "max_attempts": STEP_MAX_ATTEMPTS,
        }
        if spec["operation"] is PlanOperation.CLICK_TARGET:
            step["action_target"] = _target_payload(spec["action_target"])
            step["verification_target"] = _target_payload(spec["verification_target"])
        elif spec["operation"] is PlanOperation.READ_CLIPBOARD:
            step["value_key"] = VALUE_KEY
            step["expected_text"] = TRANSFER_VALUE
        elif spec["operation"] is PlanOperation.ACTIVATE_APP:
            step["app_name"] = TEXTEDIT_APP_NAME
        elif spec["operation"] is PlanOperation.INSERT_TEXT:
            step["value_key"] = VALUE_KEY
        steps.append(step)
    return {"task_goal": EXPECTED_TASK_GOAL, "steps": steps}


def build_fake_response_json() -> str:
    return json.dumps(build_fake_response_payload())


def run_deterministic_reasoning(
    client: DeterministicFakeLLMClient | None = None,
) -> ReasoningRun:
    if client is None:
        client = DeterministicFakeLLMClient()
    result = LLMReasoner(client=client).reason(TASK)
    return ReasoningRun(result=result, client_call_count=client.call_count)


def build_live_state_verifier():
    from computer_agent.verification import StateVerifier

    return StateVerifier()


def run_live_agent_loop(
    plan: StructuredPlan,
    *,
    capture_path: str | Path = CAPTURE_PATH,
    perception_engine_builder=build_live_perception_engine,
    executor_builder=build_live_tool_executor,
    state_verifier_builder=build_live_state_verifier,
) -> AgentLoopResult:
    capture = Path(capture_path)
    capture.parent.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        perception_engine=perception_engine_builder(capture),
        executor=executor_builder(),
        state_verifier=state_verifier_builder(),
        allowed_app_names=ALLOWED_APP_NAMES,
    ).run(plan)


def reasoning_acceptance_failures(run: object) -> tuple[str, ...]:
    if not isinstance(run, ReasoningRun):
        return (f"run was not a ReasoningRun: {type(run).__name__}",)

    failures: list[str] = []
    result = run.result
    if not isinstance(result, ReasoningResult):
        failures.append(f"result was not a ReasoningResult: {type(result).__name__}")
    elif result.status is not ReasoningStatus.READY:
        failures.append(
            "reasoning status was "
            f"{result.status.value}, expected {ReasoningStatus.READY.value}"
        )
    elif not isinstance(result.plan, StructuredPlan):
        failures.append(f"plan was not a StructuredPlan: {type(result.plan).__name__}")
    else:
        failures.extend(_plan_failures(result.plan))

    for label, actual, expected in (
        ("fake client call count", run.client_call_count, 1),
        ("LLM provider", run.provider, LLM_PROVIDER),
        ("live API request", run.live_api_request, LIVE_API_REQUEST),
    ):
        _append_mismatch(failures, label, actual, expected)
    return tuple(failures)


def agent_loop_acceptance_failures(
    result: object,
    plan: StructuredPlan,
) -> tuple[str, ...]:
    if not isinstance(result, AgentLoopResult):
        return (
            "AgentLoop result was not an AgentLoopResult: "
            f"{type(result).__name__}",
        )

    failures: list[str] = []
    records = tuple(result.state.steps)
    _append_identity_failure(failures, result.plan, plan)
    if result.status is not AgentLoopStatus.COMPLETED:
        failures.append(f"Agent loop status was {result.status.value}")
    if result.state.status is not AgentStatus.SUCCEEDED:
        failures.append(f"Agent state was {result.state.status.value}")
    if result.completed_plan_steps != EXPECTED_PLAN_STEPS:
        failures.append(
            "completed plan steps were "
            f"{result.completed_plan_steps}, not {EXPECTED_PLAN_STEPS}"
        )
    if result.completed_plan_steps != len(plan.steps):
        failures.append(
            "completed plan steps did not match plan length: "
            f"{result.completed_plan_steps} != {len(plan.steps)}"
        )

    tool_order = tuple(record.action.tool_name for record in records)
    if tool_order != EXPECTED_ACTION_TOOLS:
        failures.append(
            f"action tool order was {tool_order}, expected {EXPECTED_ACTION_TOOLS}"
        )
    if len(records) != len(EXPECTED_ACTION_TOOLS):
        failures.append(
            "Action execution count was "
            f"{len(records)}, not {len(EXPECTED_ACTION_TOOLS)}"
        )
    for index, record in enumerate(records, start=1):
        if not record.result.success:
            failures.append(f"attempt {index} ToolResult failed: {record.result.error}")

    runtime_value = _runtime_transfer_value(result)
    _append_mismatch(failures, "runtime transfer value", runtime_value, TRANSFER_VALUE)
    if len(records) >= 4:
        paste_value = records[3].action.arguments.get("text")
        if paste_value != runtime_value:
            failures.append(
                "paste_text value was "
                f"{_value(paste_value)}, expected runtime value {_value(runtime_value)}"
            )
    return tuple(failures)


def print_agent_loop_acceptance_result(
    result: object,
    plan: StructuredPlan,
) -> int:
    if isinstance(result, AgentLoopResult):
        print(f"AgentLoopResult reason: {result.reason}")
        print(f"Agent loop status: {result.status.value}")
        print(f"Agent state: {result.state.status.value}")
        print(f"Completed plan steps: {result.completed_plan_steps} / {len(plan.steps)}")
        print(f"Action executions: {len(result.state.steps)}")
        for index, record in enumerate(result.state.steps, start=1):
            print(f"Attempt {index} tool: {record.action.tool_name}")
            print(f"Attempt {index} arguments: {record.action.arguments}")
            print(f"Attempt {index} ToolResult success: {record.result.success}")
            print(f"Attempt {index} ToolResult error: {record.result.error}")
        print(f"Runtime transfer value: {_runtime_transfer_value(result)}")

    failures = agent_loop_acceptance_failures(result, plan)
    if failures:
        print("AgentLoop acceptance: failed")
        _print_failures(failures)
        print("Experiment acceptance: failed")
        return 1

    print("AgentLoop acceptance: passed")
    print("Experiment acceptance: passed")
    return 0


def run_acceptance(
    *,
    client: DeterministicFakeLLMClient | None = None,
    run_builder: Callable[[], object] | None = None,
) -> int:
    return main([], client=client, run_builder=run_builder)


def _target_payload(text: object) -> dict[str, object]:
    return {"text": text, "element_types": list(EXPECTED_ELEMENT_TYPES)}


def _plan_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []
    _append_mismatch(failures, "task goal", plan.task_goal, EXPECTED_TASK_GOAL)
    if len(plan.steps) != EXPECTED_PLAN_STEPS:
        failures.append(f"step count was {len(plan.steps)}, expected {EXPECTED_PLAN_STEPS}")

    for index, spec in enumerate(EXPECTED_STEP_SPECS, start=1):
        if len(plan.steps) < index:
            failures.append(f"step {index} was missing")
            continue
        step = plan.steps[index - 1]
        expected_type = spec["type"]
        if not isinstance(step, expected_type):
            failures.append(
                f"step {index} type was {type(step).__name__}, "
                f"expected {expected_type.__name__}"
            )
        _append_mismatch(failures, f"step {index} goal", step.goal, spec["goal"])
        _append_mismatch(failures, f"step {index} operation", step.operation, spec["operation"])
        _append_mismatch(
            failures,
            f"step {index} max_attempts",
            step.max_attempts,
            STEP_MAX_ATTEMPTS,
        )
        _check_specific_step_fields(failures, index, step, spec)
        _check_no_actions_or_coordinates(failures, index, step)

    if len(plan.steps) >= 4 and all(
        isinstance(plan.steps[position], expected)
        for position, expected in ((1, ReadClipboardStep), (3, InsertTextStep))
    ) and plan.steps[1].value_key != plan.steps[3].value_key:
        failures.append("step 2 and step 4 value_key did not match")
    return tuple(failures)


def _check_specific_step_fields(
    failures: list[str],
    index: int,
    step: object,
    spec: dict[str, object],
) -> None:
    if isinstance(step, PlanStep):
        _append_mismatch(
            failures,
            f"step {index} action target text",
            step.action_target.text,
            spec.get("action_target"),
        )
        _append_mismatch(
            failures,
            f"step {index} verification target text",
            step.verification_target.text,
            spec.get("verification_target"),
        )
        for label, target in (("action", step.action_target), ("verification", step.verification_target)):
            _append_mismatch(
                failures,
                f"step {index} {label} element_types",
                target.element_types,
                EXPECTED_ELEMENT_TYPES,
            )
            if target.reference_point is not None:
                failures.append(f"step {index} {label} target exposed coordinates")
    elif isinstance(step, ReadClipboardStep):
        _append_mismatch(failures, f"step {index} value_key", step.value_key, spec.get("value_key"))
        _append_mismatch(
            failures,
            f"step {index} expected_text",
            step.expected_text,
            spec.get("expected_text"),
        )
    elif isinstance(step, ActivateAppStep):
        _append_mismatch(failures, f"step {index} app_name", step.app_name, spec.get("app_name"))
    elif isinstance(step, InsertTextStep):
        _append_mismatch(failures, f"step {index} value_key", step.value_key, spec.get("value_key"))
        for field in ("text", "expected_text", "action_target", "arguments"):
            if hasattr(step, field):
                failures.append(f"step {index} unexpectedly exposed {field}")


def _check_no_actions_or_coordinates(
    failures: list[str],
    index: int,
    step: object,
) -> None:
    fields = getattr(step, "__dataclass_fields__", {})
    if any(field in fields for field in ("x", "y", "coordinates")):
        failures.append(f"step {index} unexpectedly exposed coordinates")
    if any(isinstance(getattr(step, field), Action) for field in fields):
        failures.append(f"step {index} unexpectedly contained an Action")


def _runtime_transfer_value(result: AgentLoopResult) -> object:
    values = result.state.context.get("values")
    if not isinstance(values, dict):
        return None
    return values.get(VALUE_KEY)


def _append_identity_failure(
    failures: list[str],
    actual: object,
    expected: object,
) -> None:
    if actual is not expected:
        failures.append("AgentLoopResult plan was not the reasoned plan object")


def _append_mismatch(
    failures: list[str],
    label: str,
    actual: object,
    expected: object,
) -> None:
    if actual != expected:
        failures.append(f"{label} was {_value(actual)}, expected {_value(expected)}")


def _value(value: object) -> str:
    return value.value if isinstance(value, PlanOperation) else repr(value)


def _print_plan(plan: StructuredPlan) -> None:
    print(f"Task goal: {plan.task_goal}")
    print(f"Step count: {len(plan.steps)}")
    for index, step in enumerate(plan.steps, start=1):
        print(f"\nStep {index}")
        print(f"Goal: {step.goal}")
        print(f"Operation: {step.operation.value}")
        if isinstance(step, PlanStep):
            print(f"Action target: {step.action_target.text}")
            print(f"Action element types: {step.action_target.element_types}")
            print(f"Verification target: {step.verification_target.text}")
            print(f"Verification element types: {step.verification_target.element_types}")
        elif isinstance(step, ReadClipboardStep):
            print(f"Value key: {step.value_key}")
            print(f"Expected text: {step.expected_text}")
        elif isinstance(step, ActivateAppStep):
            print(f"Application name: {step.app_name}")
        elif isinstance(step, InsertTextStep):
            print(f"Value key: {step.value_key}")
        print(f"Max attempts: {step.max_attempts}")
    print()


def _print_reasoning_summary(run: ReasoningRun) -> None:
    print(f"LLM client call count: {run.client_call_count}")
    print(f"LLM provider: {run.provider}")
    print(f"Live API request: {'yes' if run.live_api_request else 'no'}")


def _print_offline_summary(run: ReasoningRun) -> None:
    _print_reasoning_summary(run)
    for line in (
        "Observation count: 0",
        "Action execution count: 0",
        "Clipboard read: no",
        "Application activation: no",
        "Paste action: no",
        "Screenshot creation: no",
    ):
        print(line)


def _print_failures(failures: Sequence[str]) -> None:
    for failure in failures:
        print(f"Failed condition: {failure}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 04 Experiment 10 deterministic cross-application workflow."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute live computer Actions. Default is disabled.",
    )
    parser.add_argument(
        "--wait-seconds",
        type=wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait before live execution, from 0 through 30.",
    )
    return parser.parse_args(argv)


def _reason(
    *,
    client: DeterministicFakeLLMClient | None,
    run_builder: Callable[[], object] | None,
) -> object:
    return run_builder() if run_builder is not None else run_deterministic_reasoning(client)


def _execute(
    *,
    run: ReasoningRun,
    plan: StructuredPlan,
    wait: int,
    runner: Callable[[StructuredPlan], object],
    prerequisite_checker: Callable[[str | Path], bool],
    sleeper: Callable[[float], None],
) -> int:
    print("Execution mode: enabled; live computer Actions may execute.")
    print("Preparation:")
    print("1. Open a blank TextEdit document manually.")
    print("2. Ensure the document body is empty.")
    print("3. Ensure the insertion caret/focus is in that blank document.")
    print("4. Leave TextEdit open.")
    print("5. Open the Experiment 10 fixture manually in Google Chrome.")
    print("6. Keep the Chrome fixture visible and focused.")
    print("7. Do not manually copy or paste the transfer token.")
    print(f"Authorized app activations: {sorted(ALLOWED_APP_NAMES)}")
    print(f"Final screenshot path: {CAPTURE_PATH}")
    if not prerequisite_checker(FIXTURE_PATH):
        return 1
    wait_for_focus(wait, sleeper=sleeper)
    result = runner(plan)
    print()
    _print_reasoning_summary(run)
    return print_agent_loop_acceptance_result(result, plan)


def main(
    argv: Sequence[str] | None = None,
    *,
    client: DeterministicFakeLLMClient | None = None,
    run_builder: Callable[[], object] | None = None,
    runner: Callable[[StructuredPlan], object] = run_live_agent_loop,
    prerequisite_checker: Callable[[str | Path], bool] = live_prerequisites_available,
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = _parse_args(argv)
    print(TITLE)
    print(f"Fixture: {FIXTURE_PATH}")

    try:
        run = _reason(client=client, run_builder=run_builder)
    except Exception as error:
        print("Reasoning acceptance: failed")
        print(f"Failed condition: reasoning run raised {type(error).__name__}: {error}")
        print("Experiment acceptance: failed")
        return 1

    if isinstance(run, ReasoningRun) and isinstance(run.result.plan, StructuredPlan):
        _print_plan(run.result.plan)
    failures = reasoning_acceptance_failures(run)
    if failures:
        print("Reasoning acceptance: failed")
        _print_failures(failures)
        print("Experiment acceptance: failed")
        return 1

    print("Reasoning acceptance: passed")
    plan = run.result.plan
    if plan is None:
        raise RuntimeError("READY reasoning result did not contain a plan")
    if args.execute:
        return _execute(
            run=run,
            plan=plan,
            wait=args.wait_seconds,
            runner=runner,
            prerequisite_checker=prerequisite_checker,
            sleeper=sleeper,
        )

    print("Execution mode: disabled")
    print("Run with --execute to perform the live deterministic workflow.")
    _print_offline_summary(run)
    print("Dry-run acceptance: passed")
    print("Experiment acceptance: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
