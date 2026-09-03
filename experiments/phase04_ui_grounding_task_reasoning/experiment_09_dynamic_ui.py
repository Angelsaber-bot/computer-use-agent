"""Phase 04 experiment for dynamic UI adaptation through the agent stack."""

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
from computer_agent.planning import PlanOperation, StructuredPlan
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
    / "experiment_09_dynamic_ui.html"
)
CAPTURE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase04_ui_grounding_task_reasoning"
    / "experiment_09_dynamic_ui.png"
)

TITLE = "Phase 04 Experiment 09: Dynamic Environment Adaptation"
TASK = "Complete the deterministic dynamic UI workflow"
EXPECTED_TASK_GOAL = TASK
EXPECTED_STEPS: tuple[tuple[str, str, str], ...] = (
    ("Start the dynamic workflow", "DYNAMIC_START_09", "DYNAMIC_CONTINUE_09"),
    (
        "Continue after the layout changes",
        "DYNAMIC_CONTINUE_09",
        "DYNAMIC_CONFIRM_09",
    ),
    (
        "Confirm the foreground workflow step",
        "DYNAMIC_CONFIRM_09",
        "DYNAMIC_COMPLETE_09",
    ),
)
EXPECTED_OPERATION = PlanOperation.CLICK_TARGET
EXPECTED_ELEMENT_TYPES = ("button",)
STEP_MAX_ATTEMPTS = 1
EXPECTED_PLAN_STEPS = 3
EXPECTED_ACTION_EXECUTIONS = 3
CLICK_TOOL_NAME = "click_mouse"
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
        self._response = response if response is not None else (
            build_fake_response_json()
        )
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
        "steps": [
            {
                "goal": goal,
                "operation": EXPECTED_OPERATION.value,
                "action_target": _target_payload(action),
                "verification_target": _target_payload(verification),
                "max_attempts": STEP_MAX_ATTEMPTS,
            }
            for goal, action, verification in EXPECTED_STEPS
        ],
    }


def build_fake_response_json() -> str:
    return json.dumps(build_fake_response_payload())


def run_deterministic_reasoning(
    client: DeterministicFakeLLMClient | None = None,
) -> ReasoningRun:
    if client is None:
        client = DeterministicFakeLLMClient()
    result = LLMReasoner(client=client).reason(TASK)
    return ReasoningRun(result=result, client_call_count=client.call_count)


def run_live_agent_loop(
    plan: StructuredPlan,
    *,
    capture_path: str | Path = CAPTURE_PATH,
    perception_engine_builder=build_live_perception_engine,
    executor_builder=build_live_tool_executor,
) -> AgentLoopResult:
    capture = Path(capture_path)
    capture.parent.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        perception_engine=perception_engine_builder(capture),
        executor=executor_builder(),
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
    if result.plan is not plan:
        failures.append("AgentLoopResult plan was not the reasoned plan object")
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
    if len(records) != EXPECTED_ACTION_EXECUTIONS:
        failures.append(
            "Action execution count was "
            f"{len(records)}, not {EXPECTED_ACTION_EXECUTIONS}"
        )
    for index, record in enumerate(records, start=1):
        if record.action.tool_name != CLICK_TOOL_NAME:
            failures.append(f"attempt {index} tool was {record.action.tool_name}")
        if not record.result.success:
            failures.append(
                f"attempt {index} ToolResult failed: {record.result.error}"
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

    failures = agent_loop_acceptance_failures(result, plan)
    if failures:
        print("AgentLoop acceptance: failed")
        _print_failures(failures)
        print("Experiment acceptance: failed")
        return 1

    print("AgentLoop acceptance: passed")
    print("Recovery retry demonstrated: no")
    print("Experiment acceptance: passed")
    return 0


def run_acceptance(
    *,
    client: DeterministicFakeLLMClient | None = None,
    run_builder: Callable[[], object] | None = None,
) -> int:
    return main([], client=client, run_builder=run_builder)


def _target_payload(text: str) -> dict[str, object]:
    return {"text": text, "element_types": list(EXPECTED_ELEMENT_TYPES)}


def _plan_failures(plan: StructuredPlan) -> tuple[str, ...]:
    failures: list[str] = []
    _append_mismatch(failures, "task goal", plan.task_goal, EXPECTED_TASK_GOAL)
    if len(plan.steps) != len(EXPECTED_STEPS):
        failures.append(
            f"step count was {len(plan.steps)}, expected {len(EXPECTED_STEPS)}"
        )

    for index, expected in enumerate(EXPECTED_STEPS, start=1):
        if len(plan.steps) < index:
            failures.append(f"step {index} was missing")
            continue
        goal, action, verification = expected
        step = plan.steps[index - 1]
        for label, actual, expected_value in (
            ("goal", step.goal, goal),
            ("operation", step.operation, EXPECTED_OPERATION),
            ("action target text", step.action_target.text, action),
            ("verification target text", step.verification_target.text, verification),
            ("max_attempts", step.max_attempts, STEP_MAX_ATTEMPTS),
            ("action element_types", step.action_target.element_types, EXPECTED_ELEMENT_TYPES),
            (
                "verification element_types",
                step.verification_target.element_types,
                EXPECTED_ELEMENT_TYPES,
            ),
        ):
            _append_mismatch(failures, f"step {index} {label}", actual, expected_value)

    for index, step in enumerate(plan.steps, start=1):
        if any(
            isinstance(value, Action)
            for value in (
                step.goal,
                step.operation,
                step.action_target,
                step.verification_target,
                step.max_attempts,
            )
        ):
            failures.append(f"step {index} unexpectedly contained an Action")
        if any(field in step.__dataclass_fields__ for field in ("x", "y", "coordinates")):
            failures.append(f"step {index} unexpectedly exposed coordinates")
        for label, target in (
            ("action target", step.action_target),
            ("verification target", step.verification_target),
        ):
            if target.identifier is not None:
                failures.append(f"step {index} {label} unexpectedly used identifier")
            if target.reference_point is not None:
                failures.append(
                    f"step {index} {label} unexpectedly exposed coordinates"
                )
    return tuple(failures)


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
        print()
        print(f"Step {index}")
        for label, value in (
            ("Goal", step.goal),
            ("Operation", step.operation.value),
            ("Action target", step.action_target.text),
            ("Action element types", step.action_target.element_types),
            ("Verification target", step.verification_target.text),
            ("Verification element types", step.verification_target.element_types),
            ("Max attempts", step.max_attempts),
        ):
            print(f"{label}: {value}")
    print()


def _print_reasoning_summary(run: ReasoningRun) -> None:
    print(f"LLM client call count: {run.client_call_count}")
    print(f"LLM provider: {run.provider}")
    print(f"Live API request: {'yes' if run.live_api_request else 'no'}")


def _print_offline_summary(run: ReasoningRun) -> None:
    _print_reasoning_summary(run)
    print("Observation count: 0")
    print("Action execution count: 0")


def _print_failures(failures: Sequence[str]) -> None:
    for failure in failures:
        print(f"Failed condition: {failure}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 04 Experiment 09 reasoner-to-AgentLoop "
            "dynamic UI workflow."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute live click_mouse Actions. Default is disabled.",
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
    return run_builder() if run_builder is not None else (
        run_deterministic_reasoning(client)
    )


def _execute(
    *,
    run: ReasoningRun,
    plan: StructuredPlan,
    wait: int,
    runner: Callable[[StructuredPlan], object],
    prerequisite_checker: Callable[[str | Path], bool],
    sleeper: Callable[[float], None],
) -> int:
    print("Execution mode: enabled; live click_mouse Actions may execute.")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before execution.")
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
        print(
            "Failed condition: reasoning run raised "
            f"{type(error).__name__}: {error}"
        )
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
    print("Run with --execute to perform the live deterministic UI workflow.")
    _print_offline_summary(run)
    print("Experiment acceptance: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
