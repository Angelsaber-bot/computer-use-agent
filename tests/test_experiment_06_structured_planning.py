import importlib
import os
from pathlib import Path
import subprocess
import sys

from computer_agent.grounding import TargetSpec
from computer_agent.planning import PlanOperation, PlanStep
from computer_agent.planning import StructuredPlan, StructuredPlanner
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_06_structured_planning as experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments/phase04_ui_grounding_task_reasoning/"
    "experiment_06_structured_planning.py"
)


def _step(
    goal: str,
    *,
    action_target: TargetSpec,
    verification_target: TargetSpec,
    max_attempts: int = experiment.STEP_MAX_ATTEMPTS,
) -> PlanStep:
    return PlanStep(
        goal=goal,
        operation=PlanOperation.CLICK_TARGET,
        action_target=action_target,
        verification_target=verification_target,
        max_attempts=max_attempts,
    )


def _plan(
    *,
    task_goal: str = experiment.TASK_GOAL,
    steps: tuple[PlanStep, ...] | None = None,
) -> StructuredPlan:
    if steps is None:
        steps = experiment.build_acceptance_steps()

    return StructuredPlanner().build_plan(
        task_goal=task_goal,
        steps=steps,
    )


def _run_with_plan(plan, capsys):
    code = experiment.run_acceptance(plan_builder=lambda: plan)
    return code, capsys.readouterr().out


def test_exact_successful_two_step_plan_construction():
    plan = experiment.build_acceptance_plan()

    assert isinstance(plan, StructuredPlan)
    assert plan.task_goal == experiment.TASK_GOAL
    assert len(plan.steps) == 2
    assert plan.steps[0].goal == experiment.STEP_1_GOAL
    assert plan.steps[0].operation is PlanOperation.CLICK_TARGET
    assert plan.steps[0].action_target == experiment.STEP_1_ACTION_TARGET
    assert plan.steps[0].verification_target == (
        experiment.STEP_1_VERIFICATION_TARGET
    )
    assert plan.steps[0].max_attempts == experiment.STEP_MAX_ATTEMPTS
    assert plan.steps[1].goal == experiment.STEP_2_GOAL
    assert plan.steps[1].operation is PlanOperation.CLICK_TARGET
    assert plan.steps[1].action_target == experiment.STEP_2_ACTION_TARGET
    assert plan.steps[1].verification_target == (
        experiment.STEP_2_VERIFICATION_TARGET
    )
    assert plan.steps[1].max_attempts == experiment.STEP_MAX_ATTEMPTS
    assert experiment.acceptance_failures(plan) == ()


def test_exact_order_preservation():
    plan = experiment.build_acceptance_plan()

    assert tuple(step.goal for step in plan.steps) == (
        experiment.STEP_1_GOAL,
        experiment.STEP_2_GOAL,
    )
    assert tuple(step.action_target.text for step in plan.steps) == (
        "STEP_1_TARGET_06",
        "STEP_2_TARGET_06",
    )


def test_run_acceptance_success(capsys):
    code = experiment.run_acceptance()
    output = capsys.readouterr().out

    assert code == 0
    assert experiment.TITLE in output
    assert f"Task goal: {experiment.TASK_GOAL}" in output
    assert "Plan step count: 2" in output
    assert "Step 1" in output
    assert f"Goal: {experiment.STEP_1_GOAL}" in output
    assert "Action target: STEP_1_TARGET_06" in output
    assert "Verification target: STEP_1_COMPLETE_06" in output
    assert "Step 2" in output
    assert f"Goal: {experiment.STEP_2_GOAL}" in output
    assert "Action target: STEP_2_TARGET_06" in output
    assert "Verification target: TASK_COMPLETE_06" in output
    assert "Operation: click_target" in output
    assert "Max attempts: 2" in output
    assert "Experiment acceptance: passed" in output
    assert "Execution: not applicable" in output
    assert "Observation count: 0" in output
    assert "Action execution count: 0" in output


def test_wrong_task_goal_fails_closed(capsys):
    code, output = _run_with_plan(
        _plan(task_goal="Complete another workflow"),
        capsys,
    )

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "task goal was" in output


def test_wrong_number_of_steps_fails_closed(capsys):
    steps = experiment.build_acceptance_steps()
    code, output = _run_with_plan(_plan(steps=(steps[0],)), capsys)

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "step count was 1, expected 2" in output
    assert "step 2 was missing" in output


def test_swapped_step_order_fails_closed(capsys):
    first, second = experiment.build_acceptance_steps()
    code, output = _run_with_plan(_plan(steps=(second, first)), capsys)

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "step 1 goal was" in output
    assert "step 2 goal was" in output


def test_wrong_action_target_fails_closed(capsys):
    first, second = experiment.build_acceptance_steps()
    altered_first = _step(
        experiment.STEP_1_GOAL,
        action_target=TargetSpec(
            text="WRONG_TARGET_06",
            element_types=("button",),
        ),
        verification_target=experiment.STEP_1_VERIFICATION_TARGET,
    )
    code, output = _run_with_plan(
        _plan(steps=(altered_first, second)),
        capsys,
    )

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "step 1 action target was" in output


def test_wrong_verification_target_fails_closed(capsys):
    first, second = experiment.build_acceptance_steps()
    altered_second = _step(
        experiment.STEP_2_GOAL,
        action_target=experiment.STEP_2_ACTION_TARGET,
        verification_target=TargetSpec(
            text="WRONG_COMPLETE_06",
            element_types=("button",),
        ),
    )
    code, output = _run_with_plan(
        _plan(steps=(first, altered_second)),
        capsys,
    )

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "step 2 verification target was" in output


def test_wrong_max_attempts_fails_closed(capsys):
    first, second = experiment.build_acceptance_steps()
    altered_second = _step(
        experiment.STEP_2_GOAL,
        action_target=experiment.STEP_2_ACTION_TARGET,
        verification_target=experiment.STEP_2_VERIFICATION_TARGET,
        max_attempts=1,
    )
    code, output = _run_with_plan(
        _plan(steps=(first, altered_second)),
        capsys,
    )

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "step 2 max_attempts was 1, expected 2" in output


def test_non_structured_plan_builder_result_fails_closed(capsys):
    code = experiment.run_acceptance(plan_builder=lambda: object())
    output = capsys.readouterr().out

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "returned object was not a StructuredPlan" in output
    assert "Execution: not applicable" in output
    assert "Observation count: 0" in output
    assert "Action execution count: 0" in output


def test_direct_script_help_exits_zero_without_running_experiment():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "Experiment acceptance:" not in result.stdout
    assert result.stderr == ""


def test_importing_experiment_performs_no_execution_or_output():
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT / "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "importlib.import_module("
                "'experiments.phase04_ui_grounding_task_reasoning."
                "experiment_06_structured_planning'"
                ")"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_imported_module_identity_is_stable():
    module = importlib.import_module(
        "experiments.phase04_ui_grounding_task_reasoning."
        "experiment_06_structured_planning"
    )

    assert module is experiment
