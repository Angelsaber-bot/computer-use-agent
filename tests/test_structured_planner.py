from dataclasses import FrozenInstanceError
import importlib
import inspect

import pytest

from computer_agent.core.models import Action
from computer_agent.grounding import TargetSpec
from computer_agent.planning import (
    MAX_PLAN_STEP_ATTEMPTS,
    MAX_STRUCTURED_PLAN_STEPS,
    PlanOperation,
    PlanStep,
    StructuredPlan,
    StructuredPlanner,
)
import computer_agent.planning.models as planning_models
import computer_agent.planning.structured_planner as structured_planner_module


def _target(text: str) -> TargetSpec:
    return TargetSpec(
        text=text,
        element_types=("button",),
    )


def _step(
    goal: str = "Open settings",
    *,
    action_target: TargetSpec | None = None,
    verification_target: TargetSpec | None = None,
    max_attempts: int = 1,
) -> PlanStep:
    return PlanStep(
        goal=goal,
        operation=PlanOperation.CLICK_TARGET,
        action_target=action_target or _target("Settings"),
        verification_target=verification_target or _target("Settings panel"),
        max_attempts=max_attempts,
    )


def test_single_step_plan_is_ready_semantic_plan():
    step = _step()

    plan = StructuredPlanner().build_plan(
        task_goal="Open the settings panel",
        steps=(step,),
    )

    assert plan == StructuredPlan(
        task_goal="Open the settings panel",
        steps=(step,),
    )
    assert plan.steps[0].operation is PlanOperation.CLICK_TARGET


def test_valid_multi_step_plan_preserves_exact_order():
    first = _step("Open settings")
    second = _step("Open advanced settings")
    third = _step("Open keyboard settings")

    plan = StructuredPlanner().build_plan(
        task_goal="Navigate to keyboard settings",
        steps=(
            first,
            second,
            third,
        ),
    )

    assert plan.steps == (
        first,
        second,
        third,
    )
    assert list(plan.steps) == [
        first,
        second,
        third,
    ]


def test_target_spec_objects_are_preserved_by_identity():
    action_target = _target("Settings")
    verification_target = _target("Settings panel")
    step = _step(
        action_target=action_target,
        verification_target=verification_target,
    )

    plan = StructuredPlanner().build_plan(
        task_goal="Open settings",
        steps=(step,),
    )

    assert plan.steps[0] is step
    assert plan.steps[0].action_target is action_target
    assert plan.steps[0].verification_target is verification_target


def test_plan_models_are_immutable_and_slotted():
    step = _step()
    plan = StructuredPlan(
        task_goal="Open settings",
        steps=(step,),
    )

    assert not hasattr(step, "__dict__")
    assert not hasattr(plan, "__dict__")

    with pytest.raises(FrozenInstanceError):
        step.goal = "changed"

    with pytest.raises(FrozenInstanceError):
        plan.task_goal = "changed"


def test_empty_task_goal_is_rejected():
    with pytest.raises(
        ValueError,
        match="task_goal must be a non-empty string",
    ):
        StructuredPlanner().build_plan(
            task_goal=" ",
            steps=(_step(),),
        )


def test_empty_step_goal_is_rejected():
    with pytest.raises(
        ValueError,
        match="goal must be a non-empty string",
    ):
        _step(goal="")


def test_zero_steps_are_rejected():
    with pytest.raises(
        ValueError,
        match="steps must contain at least one PlanStep",
    ):
        StructuredPlanner().build_plan(
            task_goal="Open settings",
            steps=(),
        )


def test_non_tuple_steps_are_rejected():
    with pytest.raises(
        ValueError,
        match="steps must be a tuple of PlanStep objects",
    ):
        StructuredPlanner().build_plan(
            task_goal="Open settings",
            steps=[_step()],
        )


def test_invalid_step_object_is_rejected():
    with pytest.raises(
        ValueError,
        match="steps must contain PlanStep objects",
    ):
        StructuredPlanner().build_plan(
            task_goal="Open settings",
            steps=(object(),),
        )


@pytest.mark.parametrize(
    "max_attempts",
    [
        True,
        0,
        -1,
        MAX_PLAN_STEP_ATTEMPTS + 1,
    ],
)
def test_invalid_max_attempts_are_rejected(max_attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        _step(max_attempts=max_attempts)


def test_invalid_operation_value_is_rejected():
    with pytest.raises(ValueError, match="operation must be a PlanOperation"):
        PlanStep(
            goal="Open settings",
            operation="type_text",
            action_target=_target("Settings"),
            verification_target=_target("Settings panel"),
        )


def test_too_many_steps_are_rejected():
    steps = tuple(
        _step(f"Step {number}")
        for number in range(MAX_STRUCTURED_PLAN_STEPS + 1)
    )

    with pytest.raises(ValueError, match="steps must contain no more than"):
        StructuredPlanner().build_plan(
            task_goal="Open settings",
            steps=steps,
        )


def test_no_executable_action_or_coordinates_are_introduced():
    plan = StructuredPlanner().build_plan(
        task_goal="Open settings",
        steps=(_step(),),
    )

    assert not any(
        isinstance(value, Action)
        for step in plan.steps
        for value in (
            step.goal,
            step.operation,
            step.action_target,
            step.verification_target,
            step.max_attempts,
        )
    )
    assert "x" not in PlanStep.__dataclass_fields__
    assert "y" not in PlanStep.__dataclass_fields__
    assert "coordinates" not in PlanStep.__dataclass_fields__
    assert "x" not in StructuredPlan.__dataclass_fields__
    assert "y" not in StructuredPlan.__dataclass_fields__
    assert "coordinates" not in StructuredPlan.__dataclass_fields__


def test_planner_has_no_pipeline_calls():
    source = inspect.getsource(structured_planner_module)

    forbidden_terms = (
        "PerceptionEngine",
        "UIGrounder",
        "ActionGrounder",
        "ToolExecutor",
        "ActionVerifier",
        "ActionRecovery",
        ".observe(",
        ".ground(",
        ".ground_click(",
        ".execute(",
        ".verify_",
        ".prepare_retry(",
        "openai",
        "llm",
    )

    assert all(term not in source for term in forbidden_terms)


def test_planning_imports_are_safe():
    module = importlib.import_module("computer_agent.planning")

    assert module.PlanOperation is PlanOperation
    assert module.PlanStep is PlanStep
    assert module.StructuredPlan is StructuredPlan
    assert module.StructuredPlanner is StructuredPlanner


def test_production_source_has_no_forbidden_dependencies():
    source = (
        inspect.getsource(planning_models)
        + inspect.getsource(structured_planner_module)
    )

    forbidden_terms = (
        "computer_agent.control",
        "computer_agent.tools",
        "controller",
        "executor",
        "pyautogui",
        "experiments",
    )

    assert all(term not in source for term in forbidden_terms)
