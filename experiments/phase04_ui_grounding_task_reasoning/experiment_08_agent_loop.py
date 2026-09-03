"""Phase 04 experiment for deterministic AgentLoop orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Callable, Sequence
import time

from computer_agent.agent import AgentLoop, AgentLoopStatus, AgentStatus
from computer_agent.planning import PlanOperation, PlanStep, StructuredPlan
from computer_agent.grounding import TargetSpec

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
    / "experiment_08_agent_loop.html"
)
CAPTURE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase04_ui_grounding_task_reasoning"
    / "experiment_08_agent_loop.png"
)

TITLE = "Phase 04 Experiment 08: Agent Loop"
TASK_GOAL = "Complete the deterministic Agent Loop workflow"
STEP_1_GOAL = "Recover and complete the first UI target"
STEP_2_GOAL = "Complete the final UI target"
STEP_1_ACTION_TEXT = "STEP_1_TARGET_08"
STEP_1_VERIFICATION_TEXT = "STEP_1_COMPLETE_08"
STEP_2_ACTION_TEXT = "STEP_2_TARGET_08"
STEP_2_VERIFICATION_TEXT = "TASK_COMPLETE_08"
CLICK_TOOL_NAME = "click_mouse"
EXPECTED_PLAN_STEPS = 2
EXPECTED_ACTION_EXECUTIONS = 3
DEFAULT_WAIT_SECONDS = 8


def build_structured_plan() -> StructuredPlan:
    """Build the deterministic two-step plan for this experiment."""

    return StructuredPlan(
        task_goal=TASK_GOAL,
        steps=(
            PlanStep(
                goal=STEP_1_GOAL,
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(text=STEP_1_ACTION_TEXT),
                verification_target=TargetSpec(
                    text=STEP_1_VERIFICATION_TEXT,
                ),
                max_attempts=2,
            ),
            PlanStep(
                goal=STEP_2_GOAL,
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(text=STEP_2_ACTION_TEXT),
                verification_target=TargetSpec(
                    text=STEP_2_VERIFICATION_TEXT,
                ),
                max_attempts=1,
            ),
        ),
    )


def run_live_agent_loop(
    plan: StructuredPlan,
    *,
    capture_path: str | Path = CAPTURE_PATH,
    perception_engine_builder=build_live_perception_engine,
    executor_builder=build_live_tool_executor,
):
    """Run the production AgentLoop against real live components."""

    capture = Path(capture_path)
    capture.parent.mkdir(parents=True, exist_ok=True)
    loop = AgentLoop(
        perception_engine=perception_engine_builder(capture),
        executor=executor_builder(),
    )
    return loop.run(plan)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic Phase 04 Experiment 08 AgentLoop "
            "workflow against an already-focused local fixture."
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


def _print_dry_run(plan: StructuredPlan) -> None:
    print(TITLE)
    print(f"Fixture: {FIXTURE_PATH}")
    print("Execution mode: disabled")
    print("Run with --execute to perform the live deterministic UI workflow.")
    print(f"Task goal: {plan.task_goal}")
    print(f"Plan steps: {len(plan.steps)}")
    for number, step in enumerate(plan.steps, start=1):
        print(
            f"Step {number}: {step.goal} "
            f"({step.operation.value}, max_attempts={step.max_attempts})"
        )


def _print_live_instructions() -> None:
    print(TITLE)
    print(f"Fixture: {FIXTURE_PATH}")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before execution.")
    print("Execution mode: enabled; live click_mouse Actions may run.")


def _action_records(result) -> tuple[object, ...]:
    return tuple(result.state.steps)


def acceptance_failures(result, plan: StructuredPlan) -> list[str]:
    """Return formal live-acceptance failures for an AgentLoopResult."""

    records = _action_records(result)
    failures = []

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
            failures.append(
                f"attempt {index} tool was {record.action.tool_name}"
            )
        if not record.result.success:
            failures.append(
                f"attempt {index} ToolResult failed: {record.result.error}"
            )

    if len(records) >= 2 and (
        records[0].action.arguments == records[1].action.arguments
    ):
        failures.append(
            "first and second click arguments were identical"
        )

    return failures


def _print_attempts(result) -> None:
    for index, record in enumerate(_action_records(result), start=1):
        print(f"Attempt {index} tool: {record.action.tool_name}")
        print(f"Attempt {index} arguments: {record.action.arguments}")
        print(f"Attempt {index} ToolResult success: {record.result.success}")
        print(f"Attempt {index} ToolResult error: {record.result.error}")


def print_acceptance_result(result, plan: StructuredPlan) -> int:
    print(f"AgentLoopResult reason: {result.reason}")
    print(f"Agent loop status: {result.status.value}")
    print(f"Agent state: {result.state.status.value}")
    print(
        "Completed plan steps: "
        f"{result.completed_plan_steps} / {len(plan.steps)}"
    )
    print(f"Action executions: {len(_action_records(result))}")
    _print_attempts(result)

    failures = acceptance_failures(result, plan)
    if failures:
        print("Experiment acceptance: failed")
        for failure in failures:
            print(f"  {failure}")
        print("Recovery retry demonstrated: no")
        return 1

    print("Experiment acceptance: passed")
    print("Recovery retry demonstrated: yes")
    return 0


def _run_execute(
    plan: StructuredPlan,
    *,
    wait: int,
    runner: Callable[[StructuredPlan], object],
    prerequisite_checker: Callable[[str | Path], bool],
    sleeper: Callable[[float], None],
) -> int:
    _print_live_instructions()
    if not prerequisite_checker(FIXTURE_PATH):
        return 1

    wait_for_focus(wait, sleeper=sleeper)
    result = runner(plan)
    return print_acceptance_result(result, plan)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[StructuredPlan], object] = run_live_agent_loop,
    prerequisite_checker: Callable[[str | Path], bool] = (
        live_prerequisites_available
    ),
    sleeper: Callable[[float], None] = time.sleep,
) -> int:
    args = _parse_args(argv)
    plan = build_structured_plan()

    if not args.execute:
        _print_dry_run(plan)
        return 0

    return _run_execute(
        plan,
        wait=args.wait_seconds,
        runner=runner,
        prerequisite_checker=prerequisite_checker,
        sleeper=sleeper,
    )


if __name__ == "__main__":
    raise SystemExit(main())
