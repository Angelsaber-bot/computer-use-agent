"""Phase 04 experiment for deterministic structured planning."""

from __future__ import annotations

import argparse
from typing import Callable

from computer_agent.grounding import TargetSpec
from computer_agent.planning import PlanOperation, PlanStep
from computer_agent.planning import StructuredPlan, StructuredPlanner


TITLE = "Phase 04 Experiment 06: Deterministic Structured Planning"
TASK_GOAL = "Complete the deterministic two-step workflow"

STEP_1_GOAL = "Activate the first target"
STEP_1_ACTION_TARGET = TargetSpec(
    text="STEP_1_TARGET_06",
    element_types=("button",),
)
STEP_1_VERIFICATION_TARGET = TargetSpec(
    text="STEP_1_COMPLETE_06",
    element_types=("button",),
)

STEP_2_GOAL = "Activate the second target"
STEP_2_ACTION_TARGET = TargetSpec(
    text="STEP_2_TARGET_06",
    element_types=("button",),
)
STEP_2_VERIFICATION_TARGET = TargetSpec(
    text="TASK_COMPLETE_06",
    element_types=("button",),
)

STEP_MAX_ATTEMPTS = 2
OBSERVATION_COUNT = 0
ACTION_EXECUTION_COUNT = 0


ExpectedStep = tuple[str, TargetSpec, TargetSpec]
PlanBuilder = Callable[[], object]


EXPECTED_STEPS: tuple[ExpectedStep, ...] = (
    (
        STEP_1_GOAL,
        STEP_1_ACTION_TARGET,
        STEP_1_VERIFICATION_TARGET,
    ),
    (
        STEP_2_GOAL,
        STEP_2_ACTION_TARGET,
        STEP_2_VERIFICATION_TARGET,
    ),
)


def build_acceptance_steps() -> tuple[PlanStep, ...]:
    """Return the explicit semantic steps used by Experiment 06."""

    return tuple(
        PlanStep(
            goal=goal,
            operation=PlanOperation.CLICK_TARGET,
            action_target=action_target,
            verification_target=verification_target,
            max_attempts=STEP_MAX_ATTEMPTS,
        )
        for goal, action_target, verification_target in EXPECTED_STEPS
    )


def build_acceptance_plan(
    planner: StructuredPlanner | None = None,
) -> StructuredPlan:
    """Build the acceptance plan through the production planner boundary."""

    if planner is None:
        planner = StructuredPlanner()

    return planner.build_plan(
        task_goal=TASK_GOAL,
        steps=build_acceptance_steps(),
    )


def acceptance_failures(plan: object) -> tuple[str, ...]:
    """Return deterministic acceptance failures for a candidate plan."""

    if not isinstance(plan, StructuredPlan):
        return (
            "returned object was not a StructuredPlan: "
            f"{type(plan).__name__}",
        )

    failures: list[str] = []

    if plan.task_goal != TASK_GOAL:
        failures.append(
            f"task goal was {plan.task_goal!r}, expected {TASK_GOAL!r}"
        )

    if len(plan.steps) != len(EXPECTED_STEPS):
        failures.append(
            f"step count was {len(plan.steps)}, expected {len(EXPECTED_STEPS)}"
        )

    for index, expected in enumerate(EXPECTED_STEPS, start=1):
        if len(plan.steps) < index:
            failures.append(f"step {index} was missing")
            continue

        step = plan.steps[index - 1]
        expected_goal, expected_action, expected_verification = expected

        if step.operation is not PlanOperation.CLICK_TARGET:
            failures.append(
                f"step {index} operation was {step.operation.value}, "
                f"expected {PlanOperation.CLICK_TARGET.value}"
            )

        if step.goal != expected_goal:
            failures.append(
                f"step {index} goal was {step.goal!r}, "
                f"expected {expected_goal!r}"
            )

        if step.action_target != expected_action:
            failures.append(
                f"step {index} action target was {step.action_target!r}, "
                f"expected {expected_action!r}"
            )

        if step.verification_target != expected_verification:
            failures.append(
                "step "
                f"{index} verification target was "
                f"{step.verification_target!r}, "
                f"expected {expected_verification!r}"
            )

        if step.max_attempts != STEP_MAX_ATTEMPTS:
            failures.append(
                f"step {index} max_attempts was {step.max_attempts}, "
                f"expected {STEP_MAX_ATTEMPTS}"
            )

    return tuple(failures)


def run_acceptance(
    *,
    plan_builder: PlanBuilder = build_acceptance_plan,
) -> int:
    """Build, print, and validate the deterministic acceptance plan."""

    print(TITLE)

    try:
        plan = plan_builder()
    except (TypeError, ValueError) as error:
        print("Experiment acceptance: failed")
        print(
            "Failed condition: plan construction raised "
            f"{type(error).__name__}: {error}"
        )
        _print_execution_summary()
        return 1

    if isinstance(plan, StructuredPlan):
        _print_plan_summary(plan)

    failures = acceptance_failures(plan)
    if failures:
        print("Experiment acceptance: failed")
        for failure in failures:
            print(f"Failed condition: {failure}")
        _print_execution_summary()
        return 1

    print("Experiment acceptance: passed")
    _print_execution_summary()
    return 0


def _print_plan_summary(plan: StructuredPlan) -> None:
    print(f"Task goal: {plan.task_goal}")
    print(f"Plan step count: {len(plan.steps)}")

    for index, step in enumerate(plan.steps, start=1):
        print()
        print(f"Step {index}")
        print(f"Goal: {step.goal}")
        print(f"Operation: {step.operation.value}")
        print(f"Action target: {step.action_target.text}")
        print(f"Verification target: {step.verification_target.text}")
        print(f"Max attempts: {step.max_attempts}")

    print()


def _print_execution_summary() -> None:
    print("Execution: not applicable")
    print(f"Observation count: {OBSERVATION_COUNT}")
    print(f"Action execution count: {ACTION_EXECUTION_COUNT}")


def _parse_args(argv: tuple[str, ...] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic Experiment 06 structured plan.",
    )
    return parser.parse_args(argv)


def main(argv: tuple[str, ...] | None = None) -> int:
    _parse_args(argv)
    return run_acceptance()


if __name__ == "__main__":
    raise SystemExit(main())
