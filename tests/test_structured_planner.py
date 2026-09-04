from dataclasses import FrozenInstanceError
import importlib
import inspect

import pytest

from computer_agent.core.models import Action
from computer_agent.grounding import TargetSpec
from computer_agent.planning import (
    ActivateAppStep,
    InsertTextStep,
    MAX_PLAN_STEP_ATTEMPTS,
    MAX_STRUCTURED_PLAN_STEPS,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    SemanticPlanStep,
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


def _read_clipboard_step(
    *,
    goal: str = "Read copied transfer value",
    value_key: str = "transfer_value",
    expected_text: str = "CROSS_APP_TRANSFER_10",
    max_attempts: int = 1,
) -> ReadClipboardStep:
    return ReadClipboardStep(
        goal=goal,
        value_key=value_key,
        expected_text=expected_text,
        max_attempts=max_attempts,
    )


def _activate_app_step(
    *,
    goal: str = "Activate TextEdit",
    app_name: str = "TextEdit",
    max_attempts: int = 1,
) -> ActivateAppStep:
    return ActivateAppStep(
        goal=goal,
        app_name=app_name,
        max_attempts=max_attempts,
    )


def _insert_text_step(
    *,
    goal: str = "Insert transfer value",
    value_key: str = "transfer_value",
    max_attempts: int = 1,
) -> InsertTextStep:
    return InsertTextStep(
        goal=goal,
        value_key=value_key,
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


def test_plan_operation_contains_bounded_non_click_operations():
    assert PlanOperation.CLICK_TARGET.value == "click_target"
    assert PlanOperation.READ_CLIPBOARD.value == "read_clipboard"
    assert PlanOperation.ACTIVATE_APP.value == "activate_app"
    assert PlanOperation.INSERT_TEXT.value == "insert_text"


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


def test_existing_plan_step_constructor_remains_click_target_compatible():
    action_target = TargetSpec(
        text="Settings",
        element_types=("button",),
    )
    verification_target = TargetSpec(
        text="Settings panel",
        element_types=("text",),
    )

    step = PlanStep(
        goal="Open settings",
        operation=PlanOperation.CLICK_TARGET,
        action_target=action_target,
        verification_target=verification_target,
        max_attempts=2,
    )

    assert step.goal == "Open settings"
    assert step.operation is PlanOperation.CLICK_TARGET
    assert step.action_target is action_target
    assert step.verification_target is verification_target
    assert step.max_attempts == 2


@pytest.mark.parametrize(
    "operation",
    [
        PlanOperation.READ_CLIPBOARD,
        PlanOperation.ACTIVATE_APP,
        PlanOperation.INSERT_TEXT,
    ],
)
def test_plan_step_remains_click_target_only(operation):
    with pytest.raises(ValueError, match="PlanStep operation must be click_target"):
        PlanStep(
            goal="Invalid click step",
            operation=operation,
            action_target=_target("Settings"),
            verification_target=_target("Settings panel"),
        )


def test_read_clipboard_step_construction_sets_fixed_operation():
    step = _read_clipboard_step(max_attempts=2)

    assert step.goal == "Read copied transfer value"
    assert step.value_key == "transfer_value"
    assert step.expected_text == "CROSS_APP_TRANSFER_10"
    assert step.max_attempts == 2
    assert step.operation is PlanOperation.READ_CLIPBOARD
    assert not hasattr(step, "__dict__")

    with pytest.raises(FrozenInstanceError):
        step.value_key = "other_value"


@pytest.mark.parametrize(
    "value_key",
    [
        "",
        "   ",
        "1transfer_value",
        "_transfer_value",
        "transfer-value",
        "transfer.value",
        "transfer/value",
        "transfer value",
        "transfer$value",
        "transfer[0]",
        "value" * 17,
    ],
)
def test_read_clipboard_step_rejects_invalid_value_key(value_key):
    with pytest.raises(ValueError, match="value_key"):
        _read_clipboard_step(value_key=value_key)


@pytest.mark.parametrize(
    "expected_text",
    [
        "",
        "   ",
        123,
        None,
    ],
)
def test_read_clipboard_step_rejects_empty_expected_text(expected_text):
    with pytest.raises(ValueError, match="expected_text"):
        _read_clipboard_step(expected_text=expected_text)


def test_activate_app_step_construction_sets_fixed_operation():
    step = _activate_app_step(max_attempts=2)

    assert step.goal == "Activate TextEdit"
    assert step.app_name == "TextEdit"
    assert step.max_attempts == 2
    assert step.operation is PlanOperation.ACTIVATE_APP
    assert not hasattr(step, "__dict__")

    with pytest.raises(FrozenInstanceError):
        step.app_name = "Safari"


@pytest.mark.parametrize(
    "app_name",
    [
        "",
        "   ",
        123,
        None,
    ],
)
def test_activate_app_step_rejects_empty_app_name(app_name):
    with pytest.raises(ValueError, match="app_name"):
        _activate_app_step(app_name=app_name)


def test_insert_text_step_construction_sets_fixed_operation():
    step = _insert_text_step(max_attempts=2)

    assert step.goal == "Insert transfer value"
    assert step.value_key == "transfer_value"
    assert step.max_attempts == 2
    assert step.operation is PlanOperation.INSERT_TEXT
    assert not hasattr(step, "__dict__")

    with pytest.raises(FrozenInstanceError):
        step.value_key = "other_value"


@pytest.mark.parametrize(
    "value_key",
    [
        "",
        "   ",
        "1transfer_value",
        "_transfer_value",
        "transfer-value",
        "transfer.value",
        "transfer/value",
        "transfer value",
        "transfer$value",
        "transfer[0]",
        "value" * 17,
    ],
)
def test_insert_text_step_rejects_invalid_value_key(value_key):
    with pytest.raises(ValueError, match="value_key"):
        _insert_text_step(value_key=value_key)


def test_insert_text_step_has_no_literal_text_payload_field():
    fields = InsertTextStep.__dataclass_fields__

    assert set(fields) == {
        "goal",
        "value_key",
        "max_attempts",
        "operation",
    }
    assert "text" not in fields
    assert "text_to_type" not in fields
    assert "expected_text" not in fields


@pytest.mark.parametrize(
    ("step_factory", "kwargs"),
    [
        (ReadClipboardStep, {"goal": "Read", "value_key": "v", "expected_text": "x"}),
        (ActivateAppStep, {"goal": "Activate", "app_name": "TextEdit"}),
        (InsertTextStep, {"goal": "Insert", "value_key": "v"}),
    ],
)
def test_non_click_step_operation_cannot_be_overridden(step_factory, kwargs):
    with pytest.raises(TypeError, match="operation"):
        step_factory(
            **kwargs,
            operation=PlanOperation.CLICK_TARGET,
        )


@pytest.mark.parametrize(
    "step_factory",
    [
        _read_clipboard_step,
        _activate_app_step,
        _insert_text_step,
    ],
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
def test_new_semantic_steps_reject_invalid_max_attempts(
    step_factory,
    max_attempts,
):
    with pytest.raises(ValueError, match="max_attempts"):
        step_factory(max_attempts=max_attempts)


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


def test_structured_plan_accepts_mixed_semantic_step_types():
    click_step = _step()
    read_step = _read_clipboard_step()
    activate_step = _activate_app_step()
    insert_step = _insert_text_step()

    plan = StructuredPlan(
        task_goal="Transfer text across apps",
        steps=(
            click_step,
            read_step,
            activate_step,
            insert_step,
        ),
    )

    assert plan.steps == (
        click_step,
        read_step,
        activate_step,
        insert_step,
    )
    assert tuple(step.operation for step in plan.steps) == (
        PlanOperation.CLICK_TARGET,
        PlanOperation.READ_CLIPBOARD,
        PlanOperation.ACTIVATE_APP,
        PlanOperation.INSERT_TEXT,
    )


def test_structured_planner_accepts_mixed_semantic_step_types():
    steps: tuple[SemanticPlanStep, ...] = (
        _step(),
        _read_clipboard_step(),
        _activate_app_step(),
        _insert_text_step(),
    )

    plan = StructuredPlanner().build_plan(
        task_goal="Transfer text across apps",
        steps=steps,
    )

    assert plan.steps is steps


def test_structured_planner_click_plan_behavior_remains_unchanged():
    step = _step(max_attempts=2)

    plan = StructuredPlanner().build_plan(
        task_goal="Open settings",
        steps=(step,),
    )

    assert plan == StructuredPlan(
        task_goal="Open settings",
        steps=(step,),
    )
    assert isinstance(plan.steps[0], PlanStep)
    assert plan.steps[0].operation is PlanOperation.CLICK_TARGET


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


def test_new_semantic_steps_do_not_contain_actions_or_coordinates():
    for step in (
        _read_clipboard_step(),
        _activate_app_step(),
        _insert_text_step(),
    ):
        assert not any(
            isinstance(value, Action)
            for value in (
                getattr(step, field_name)
                for field_name in step.__dataclass_fields__
            )
        )
        assert "x" not in step.__dataclass_fields__
        assert "y" not in step.__dataclass_fields__
        assert "coordinates" not in step.__dataclass_fields__
        assert "tool_name" not in step.__dataclass_fields__
        assert "arguments" not in step.__dataclass_fields__


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
    assert module.ReadClipboardStep is ReadClipboardStep
    assert module.ActivateAppStep is ActivateAppStep
    assert module.InsertTextStep is InsertTextStep
    assert module.SemanticPlanStep is SemanticPlanStep
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
