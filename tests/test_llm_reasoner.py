from dataclasses import FrozenInstanceError
import importlib
import json
from typing import Any

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
from computer_agent.reasoning import (
    LLMReasoner,
    ReasoningResult,
    ReasoningStatus,
    SUPPORTED_REASONING_ELEMENT_TYPES,
)


class FakeLLMClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> object:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response

        return self.response


class RecordingPlanner(StructuredPlanner):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.returned_plan: StructuredPlan | None = None

    def build_plan(
        self,
        *,
        task_goal: str,
        steps: tuple[PlanStep, ...],
    ) -> StructuredPlan:
        self.calls.append(
            {
                "task_goal": task_goal,
                "steps": steps,
            }
        )
        self.returned_plan = super().build_plan(
            task_goal=task_goal,
            steps=steps,
        )
        return self.returned_plan


class FailingPlanner(StructuredPlanner):
    def build_plan(
        self,
        *,
        task_goal: str,
        steps: tuple[PlanStep, ...],
    ) -> StructuredPlan:
        raise ValueError("planner validation failed")


class InternalErrorPlanner(StructuredPlanner):
    def build_plan(
        self,
        *,
        task_goal: str,
        steps: tuple[PlanStep, ...],
    ) -> StructuredPlan:
        raise RuntimeError("planner implementation failed")


def _target_json(
    text: str = "Settings",
    element_types: tuple[str, ...] = ("button",),
) -> dict[str, object]:
    return {
        "text": text,
        "element_types": list(element_types),
    }


def _step_json(
    *,
    goal: object = "Open settings",
    operation: object = "click_target",
    action_target: object | None = None,
    verification_target: object | None = None,
    max_attempts: object = 1,
) -> dict[str, object]:
    if action_target is None:
        action_target = _target_json("Settings")

    if verification_target is None:
        verification_target = _target_json("Settings panel")

    return {
        "goal": goal,
        "operation": operation,
        "action_target": action_target,
        "verification_target": verification_target,
        "max_attempts": max_attempts,
    }


def _response(
    *,
    task_goal: object = "Open settings",
    steps: object | None = None,
    extra: dict[str, object] | None = None,
) -> str:
    if steps is None:
        steps = [_step_json()]

    payload = {
        "task_goal": task_goal,
        "steps": steps,
    }
    if extra:
        payload.update(extra)

    return json.dumps(payload)


def _duplicate_top_level_key_response() -> str:
    steps = json.dumps([_step_json()])
    return (
        '{"task_goal": "Open settings", '
        '"task_goal": "Duplicate", '
        f'"steps": {steps}}}'
    )


def _duplicate_nested_target_key_response() -> str:
    return (
        '{"task_goal": "Open settings", "steps": ['
        '{"goal": "Open settings", '
        '"operation": "click_target", '
        '"action_target": {'
        '"text": "Settings", '
        '"text": "Duplicate", '
        '"element_types": ["button"]'
        '}, '
        '"verification_target": {'
        '"text": "Settings panel", '
        '"element_types": ["window"]'
        '}, '
        '"max_attempts": 1}'
        ']}'
    )


def _reason(
    response: object,
    *,
    task: str = "Open the settings panel",
    planner: StructuredPlanner | None = None,
) -> tuple[ReasoningResult, FakeLLMClient, LLMReasoner]:
    client = FakeLLMClient(response)
    reasoner = LLMReasoner(
        client=client,
        planner=planner,
    )

    return reasoner.reason(task), client, reasoner


def _assert_blocked(result: ReasoningResult) -> None:
    assert result.status is ReasoningStatus.BLOCKED
    assert result.plan is None
    assert result.reason.strip()


def test_valid_one_step_response_returns_ready_plan():
    result, client, _reasoner = _reason(_response())

    assert result.status is ReasoningStatus.READY
    assert isinstance(result.plan, StructuredPlan)
    assert result.plan.task_goal == "Open settings"
    assert len(result.plan.steps) == 1
    assert result.reason.strip()
    assert len(client.calls) == 1


def test_valid_ordered_two_step_response_returns_ready_plan():
    result, _client, _reasoner = _reason(
        _response(
            task_goal="Complete the workflow",
            steps=[
                _step_json(
                    goal="Open settings",
                    action_target=_target_json("Settings"),
                    verification_target=_target_json("Settings panel"),
                ),
                _step_json(
                    goal="Open advanced settings",
                    action_target=_target_json("Advanced"),
                    verification_target=_target_json("Advanced panel"),
                    max_attempts=2,
                ),
            ],
        )
    )

    assert result.status is ReasoningStatus.READY
    assert len(result.plan.steps) == 2


def test_step_order_is_preserved_exactly():
    result, _client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(goal="First", action_target=_target_json("One")),
                _step_json(goal="Second", action_target=_target_json("Two")),
            ],
        )
    )

    assert tuple(step.goal for step in result.plan.steps) == (
        "First",
        "Second",
    )
    assert tuple(
        step.action_target.text
        for step in result.plan.steps
    ) == (
        "One",
        "Two",
    )


def test_target_spec_values_are_decoded_strictly():
    result, _client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(
                    action_target=_target_json(
                        "Settings",
                        ("button", "text"),
                    ),
                    verification_target=_target_json(
                        "Preferences",
                        ("text",),
                    ),
                    max_attempts=3,
                ),
            ],
        )
    )

    step = result.plan.steps[0]
    assert step.operation is PlanOperation.CLICK_TARGET
    assert step.max_attempts == 3
    assert step.action_target == TargetSpec(
        text="Settings",
        element_types=("button", "text"),
    )
    assert step.verification_target == TargetSpec(
        text="Preferences",
        element_types=("text",),
    )
    assert step.action_target.identifier is None
    assert step.action_target.reference_point is None


@pytest.mark.parametrize(
    "element_type",
    SUPPORTED_REASONING_ELEMENT_TYPES,
)
def test_each_supported_reasoning_element_type_can_reach_ready(
    element_type: str,
):
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(
                    action_target=_target_json(
                        "Target",
                        (element_type,),
                    ),
                    verification_target=_target_json(
                        "Done",
                        (element_type,),
                    ),
                )
            ]
        )
    )

    assert result.status is ReasoningStatus.READY
    assert result.plan.steps[0].action_target.element_types == (
        element_type,
    )
    assert result.plan.steps[0].verification_target.element_types == (
        element_type,
    )
    assert len(client.calls) == 1


def test_empty_reasoning_element_types_reach_ready_as_empty_tuple():
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(
                    action_target=_target_json("Target", ()),
                    verification_target=_target_json("Done", ()),
                )
            ]
        )
    )

    assert result.status is ReasoningStatus.READY
    assert result.plan.steps[0].action_target.element_types == ()
    assert result.plan.steps[0].verification_target.element_types == ()
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "element_types",
    [
        ("heading",),
        ("page title",),
        ("navigation item",),
        ("link",),
        ("menuitem",),
        ("button", "menuitem"),
    ],
)
def test_unsupported_reasoning_element_types_fail_closed(
    element_types: tuple[str, ...],
):
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(
                    action_target=_target_json(
                        "Target",
                        element_types,
                    ),
                )
            ]
        )
    )

    _assert_blocked(result)
    assert len(client.calls) == 1


def test_system_prompt_communicates_supported_element_type_policy():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    for element_type in SUPPORTED_REASONING_ELEMENT_TYPES:
        assert element_type in prompt

    assert "If the UI role is uncertain" in prompt
    assert "empty element_types array" in prompt
    assert "Never invent a role" in prompt


def test_final_plan_is_constructed_through_injected_planner_seam():
    planner = RecordingPlanner()

    result, _client, _reasoner = _reason(
        _response(),
        planner=planner,
    )

    assert result.status is ReasoningStatus.READY
    assert len(planner.calls) == 1
    assert planner.calls[0]["task_goal"] == "Open settings"
    assert isinstance(planner.calls[0]["steps"], tuple)
    assert all(
        isinstance(step, PlanStep)
        for step in planner.calls[0]["steps"]
    )
    assert result.plan is planner.returned_plan


def test_exactly_one_client_call_per_reason_invocation():
    result, client, _reasoner = _reason("{")

    _assert_blocked(result)
    assert len(client.calls) == 1


def test_system_prompt_is_deterministic():
    client = FakeLLMClient(_response())
    reasoner = LLMReasoner(client=client)

    first = reasoner.reason("Open settings")
    second = reasoner.reason("Open settings")

    assert first.status is ReasoningStatus.READY
    assert second.status is ReasoningStatus.READY
    assert len(client.calls) == 2
    assert client.calls[0]["system_prompt"] == reasoner.system_prompt
    assert client.calls[1]["system_prompt"] == reasoner.system_prompt


def test_user_task_is_passed_to_client_as_task_intent():
    task = "Open the keyboard settings"

    result, client, _reasoner = _reason(
        _response(),
        task=task,
    )

    assert result.status is ReasoningStatus.READY
    assert client.calls[0]["user_prompt"] == f"Task intent:\n{task}"


@pytest.mark.parametrize("task", ["", "   ", "\t\n"])
def test_empty_caller_task_is_rejected(task: str):
    client = FakeLLMClient(_response())
    reasoner = LLMReasoner(client=client)

    with pytest.raises(ValueError, match="task must be a non-empty string"):
        reasoner.reason(task)

    assert client.calls == []


@pytest.mark.parametrize(
    ("response", "case_name"),
    [
        (RuntimeError("provider failed"), "provider exception"),
        (123, "non-string response"),
        ("", "empty response"),
        ("{", "malformed JSON"),
        ("[]", "top-level list"),
        (
            json.dumps({"steps": [_step_json()]}),
            "missing top-level field",
        ),
        (
            _response(task_goal=123),
            "task_goal not string",
        ),
        (
            _response(task_goal=""),
            "empty task_goal",
        ),
        (
            _response(extra={"note": "extra"}),
            "extra top-level field",
        ),
        (
            _duplicate_top_level_key_response(),
            "duplicate top-level key",
        ),
        (
            '{"task_goal": NaN, "steps": []}',
            "JSON NaN constant",
        ),
        (
            '{"task_goal": Infinity, "steps": []}',
            "JSON Infinity constant",
        ),
        (
            _response(steps={"goal": "not a list"}),
            "steps not list",
        ),
        (
            _response(steps=[]),
            "empty steps",
        ),
        (
            _response(
                steps=[
                    _step_json(goal=f"Step {index}")
                    for index in range(MAX_STRUCTURED_PLAN_STEPS + 1)
                ]
            ),
            "too many steps",
        ),
        (
            _response(
                steps=[
                    {
                        key: value
                        for key, value in _step_json().items()
                        if key != "goal"
                    }
                ]
            ),
            "missing step field",
        ),
        (
            _response(
                steps=[
                    {
                        **_step_json(),
                        "note": "extra",
                    }
                ]
            ),
            "extra step field",
        ),
        (
            _response(steps=[_step_json(goal=123)]),
            "step goal not string",
        ),
        (
            _response(steps=[_step_json(goal="")]),
            "empty step goal",
        ),
        (
            _response(steps=[_step_json(operation="CLICK_TARGET")]),
            "unsupported operation",
        ),
        (
            _response(steps=[_step_json(action_target="Settings")]),
            "target not object",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        action_target={"text": "Settings"},
                    )
                ]
            ),
            "missing target field",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        action_target={
                            **_target_json("Settings"),
                            "x": 10,
                        },
                    )
                ]
            ),
            "extra target field",
        ),
        (
            _duplicate_nested_target_key_response(),
            "duplicate nested target key",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        action_target=_target_json(""),
                    )
                ]
            ),
            "empty target text",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        action_target={
                            "text": "Settings",
                            "element_types": "button",
                        },
                    )
                ]
            ),
            "element_types not list",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        action_target=_target_json(
                            "Settings",
                            ("button", ""),
                        ),
                    )
                ]
            ),
            "invalid element_type",
        ),
        (
            _response(steps=[_step_json(max_attempts=True)]),
            "bool max_attempts",
        ),
        (
            _response(steps=[_step_json(max_attempts=1.5)]),
            "float max_attempts",
        ),
        (
            _response(steps=[_step_json(max_attempts=0)]),
            "zero max_attempts",
        ),
        (
            _response(
                steps=[
                    _step_json(
                        max_attempts=MAX_PLAN_STEP_ATTEMPTS + 1,
                    )
                ]
            ),
            "max_attempts over bound",
        ),
    ],
)
def test_unsafe_runtime_responses_fail_closed(
    response: object,
    case_name: str,
):
    result, client, _reasoner = _reason(response)

    assert case_name
    _assert_blocked(result)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "coordinates",
        "bounding_box",
        "tool_name",
        "arguments",
        "identifier",
        "minimum_confidence",
        "reference_point",
    ],
)
def test_forbidden_target_fields_fail_closed(forbidden_field: str):
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _step_json(
                    action_target={
                        **_target_json("Settings"),
                        forbidden_field: "not allowed",
                    },
                )
            ]
        )
    )

    _assert_blocked(result)
    assert len(client.calls) == 1


def test_injected_planner_failure_fails_closed():
    result, client, _reasoner = _reason(
        _response(),
        planner=FailingPlanner(),
    )

    _assert_blocked(result)
    assert len(client.calls) == 1


def test_unexpected_internal_planner_error_is_visible():
    client = FakeLLMClient(_response())
    reasoner = LLMReasoner(
        client=client,
        planner=InternalErrorPlanner(),
    )

    with pytest.raises(RuntimeError, match="planner implementation failed"):
        reasoner.reason("Open settings")

    assert len(client.calls) == 1


def test_reasoning_result_invariants_are_enforced():
    plan = StructuredPlanner().build_plan(
        task_goal="Open settings",
        steps=(
            PlanStep(
                goal="Open settings",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text="Settings",
                    element_types=("button",),
                ),
                verification_target=TargetSpec(
                    text="Settings panel",
                    element_types=("window",),
                ),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="READY results require a StructuredPlan",
    ):
        ReasoningResult(
            status=ReasoningStatus.READY,
            plan=None,
            reason="ready",
        )

    with pytest.raises(
        ValueError,
        match="BLOCKED results must not contain a plan",
    ):
        ReasoningResult(
            status=ReasoningStatus.BLOCKED,
            plan=plan,
            reason="blocked",
        )

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        ReasoningResult(
            status=ReasoningStatus.BLOCKED,
            plan=None,
            reason="",
        )


def test_reasoning_result_is_frozen_and_slotted():
    result = ReasoningResult(
        status=ReasoningStatus.BLOCKED,
        plan=None,
        reason="blocked",
    )

    assert not hasattr(result, "__dict__")

    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"


def test_successful_plan_contains_no_action_or_coordinates():
    result, _client, _reasoner = _reason(_response())

    assert result.status is ReasoningStatus.READY
    for step in result.plan.steps:
        values: tuple[Any, ...] = (
            step.goal,
            step.operation,
            step.action_target,
            step.verification_target,
            step.max_attempts,
        )
        assert not any(isinstance(value, Action) for value in values)
        assert step.action_target.identifier is None
        assert step.verification_target.identifier is None
        assert step.action_target.reference_point is None
        assert step.verification_target.reference_point is None

    assert "x" not in PlanStep.__dataclass_fields__
    assert "y" not in PlanStep.__dataclass_fields__
    assert "coordinates" not in PlanStep.__dataclass_fields__
    assert "x" not in StructuredPlan.__dataclass_fields__
    assert "y" not in StructuredPlan.__dataclass_fields__
    assert "coordinates" not in StructuredPlan.__dataclass_fields__


def test_reasoning_public_imports_are_safe():
    module = importlib.import_module("computer_agent.reasoning")

    assert module.LLMReasoner is LLMReasoner
    assert module.ReasoningResult is ReasoningResult
    assert module.ReasoningStatus is ReasoningStatus
