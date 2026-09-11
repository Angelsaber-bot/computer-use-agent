from dataclasses import FrozenInstanceError
import importlib
import json
from typing import Any

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
    WebTextInputStep,
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
        steps: tuple[SemanticPlanStep, ...],
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
        steps: tuple[SemanticPlanStep, ...],
    ) -> StructuredPlan:
        raise ValueError("planner validation failed")


class InternalErrorPlanner(StructuredPlanner):
    def build_plan(
        self,
        *,
        task_goal: str,
        steps: tuple[SemanticPlanStep, ...],
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


def _read_clipboard_step_json(
    *,
    goal: object = "Read copied transfer value",
    value_key: object = "transfer_value",
    expected_text: object = "CROSS_APP_TRANSFER_10",
    max_attempts: object = 1,
) -> dict[str, object]:
    return {
        "goal": goal,
        "operation": "read_clipboard",
        "value_key": value_key,
        "expected_text": expected_text,
        "max_attempts": max_attempts,
    }


def _activate_app_step_json(
    *,
    goal: object = "Switch to TextEdit",
    app_name: object = "TextEdit",
    max_attempts: object = 1,
) -> dict[str, object]:
    return {
        "goal": goal,
        "operation": "activate_app",
        "app_name": app_name,
        "max_attempts": max_attempts,
    }


def _insert_text_step_json(
    *,
    goal: object = "Insert the verified transfer value",
    value_key: object = "transfer_value",
    max_attempts: object = 1,
) -> dict[str, object]:
    return {
        "goal": goal,
        "operation": "insert_text",
        "value_key": value_key,
        "max_attempts": max_attempts,
    }


def _type_into_target_step_json(
    *,
    goal: object = "Enter the search query",
    target: object | None = None,
    input_text: object = "typing",
    max_attempts: object = 1,
) -> dict[str, object]:
    if target is None:
        target = _target_json("Search This Site", ("text_field",))

    return {
        "goal": goal,
        "operation": "type_into_target",
        "target": target,
        "input_text": input_text,
        "max_attempts": max_attempts,
    }


def _cross_app_steps_json() -> list[dict[str, object]]:
    return [
        _step_json(
            goal="Copy the transfer value from the browser fixture",
            action_target=_target_json(
                "COPY_TRANSFER_VALUE_10",
                ("button",),
            ),
            verification_target=_target_json(
                "TRANSFER_COPIED_10",
                ("button",),
            ),
        ),
        _read_clipboard_step_json(
            goal="Read and verify the copied transfer value",
        ),
        _activate_app_step_json(),
        _insert_text_step_json(),
    ]


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


def _duplicate_runtime_step_key_response() -> str:
    return (
        '{"task_goal": "Read clipboard", "steps": ['
        '{"goal": "Read", '
        '"operation": "read_clipboard", '
        '"value_key": "transfer_value", '
        '"value_key": "duplicate_value", '
        '"expected_text": "CROSS_APP_TRANSFER_10", '
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


def test_exact_returned_task_goal_is_parsed_normally():
    task = (
        'On Wikipedia, search for "computer vision" and verify that the '
        '"Computer vision" article heading appears.'
    )

    result, client, _reasoner = _reason(
        _response(task_goal=task),
        task=task,
    )

    assert result.status is ReasoningStatus.READY
    assert result.plan.task_goal == task
    assert client.calls[0]["user_prompt"].endswith(task)


def test_valid_existing_click_target_response_remains_ready_plan_step():
    result, client, _reasoner = _reason(_response())

    assert result.status is ReasoningStatus.READY
    assert isinstance(result.plan.steps[0], PlanStep)
    assert result.plan.steps[0].operation is PlanOperation.CLICK_TARGET
    assert result.plan.steps[0].action_target.text == "Settings"
    assert result.plan.steps[0].verification_target.text == "Settings panel"
    assert len(client.calls) == 1


def test_valid_read_clipboard_response_returns_typed_step():
    result, client, _reasoner = _reason(
        _response(steps=[_read_clipboard_step_json(max_attempts=2)])
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, ReadClipboardStep)
    assert step.operation is PlanOperation.READ_CLIPBOARD
    assert step.goal == "Read copied transfer value"
    assert step.value_key == "transfer_value"
    assert step.expected_text == "CROSS_APP_TRANSFER_10"
    assert step.max_attempts == 2
    assert len(client.calls) == 1


def test_valid_activate_app_response_returns_typed_step():
    result, client, _reasoner = _reason(
        _response(steps=[_activate_app_step_json(max_attempts=2)])
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, ActivateAppStep)
    assert step.operation is PlanOperation.ACTIVATE_APP
    assert step.goal == "Switch to TextEdit"
    assert step.app_name == "TextEdit"
    assert step.max_attempts == 2
    assert len(client.calls) == 1


def test_activate_app_remains_valid_when_explicitly_requested():
    result, client, _reasoner = _reason(
        _response(
            task_goal="Switch to TextEdit",
            steps=[_activate_app_step_json()],
        ),
        task="Switch to TextEdit",
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, ActivateAppStep)
    assert step.operation is PlanOperation.ACTIVATE_APP
    assert step.app_name == "TextEdit"
    assert step.max_attempts == 1
    assert len(client.calls) == 1


def test_valid_insert_text_response_returns_typed_step():
    result, client, _reasoner = _reason(
        _response(steps=[_insert_text_step_json(max_attempts=2)])
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, InsertTextStep)
    assert step.operation is PlanOperation.INSERT_TEXT
    assert step.goal == "Insert the verified transfer value"
    assert step.value_key == "transfer_value"
    assert step.max_attempts == 2
    assert len(client.calls) == 1


def test_valid_type_into_target_response_returns_web_text_input_step():
    result, client, _reasoner = _reason(
        _response(steps=[_type_into_target_step_json()])
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, WebTextInputStep)
    assert step.operation is PlanOperation.TYPE_INTO_TARGET
    assert step.goal == "Enter the search query"
    assert step.target == TargetSpec(
        text="Search This Site",
        element_types=("text_field",),
    )
    assert step.target.identifier is None
    assert step.target.reference_point is None
    assert step.input_text == "typing"
    assert step.max_attempts == 1
    assert len(client.calls) == 1


def test_type_into_target_preserves_caller_visible_input_text_exactly():
    input_text = "  Python release schedule 3.14  "

    result, _client, _reasoner = _reason(
        _response(
            steps=[
                _type_into_target_step_json(
                    input_text=input_text,
                )
            ]
        )
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, WebTextInputStep)
    assert step.input_text == input_text


def test_mixed_type_into_target_then_click_target_preserves_order_and_types():
    result, _client, _reasoner = _reason(
        _response(
            task_goal="Search python.org",
            steps=[
                _type_into_target_step_json(),
                _step_json(
                    goal="Submit the search",
                    action_target=_target_json("GO", ("button",)),
                    verification_target=_target_json(
                        "Search Results",
                        ("text",),
                    ),
                ),
            ],
        )
    )

    assert result.status is ReasoningStatus.READY
    first, second = result.plan.steps
    assert isinstance(first, WebTextInputStep)
    assert isinstance(second, PlanStep)
    assert tuple(step.operation for step in result.plan.steps) == (
        PlanOperation.TYPE_INTO_TARGET,
        PlanOperation.CLICK_TARGET,
    )
    assert first.target.text == "Search This Site"
    assert first.input_text == "typing"
    assert second.action_target.text == "GO"


def test_valid_mixed_four_step_cross_app_response_preserves_exact_order():
    result, client, _reasoner = _reason(
        _response(
            task_goal=(
                "Transfer the deterministic browser fixture value "
                "CROSS_APP_TRANSFER_10 into the blank TextEdit document."
            ),
            steps=_cross_app_steps_json(),
        )
    )

    assert result.status is ReasoningStatus.READY
    assert result.plan.task_goal == (
        "Transfer the deterministic browser fixture value "
        "CROSS_APP_TRANSFER_10 into the blank TextEdit document."
    )
    first, second, third, fourth = result.plan.steps
    assert isinstance(first, PlanStep)
    assert isinstance(second, ReadClipboardStep)
    assert isinstance(third, ActivateAppStep)
    assert isinstance(fourth, InsertTextStep)
    assert tuple(step.operation for step in result.plan.steps) == (
        PlanOperation.CLICK_TARGET,
        PlanOperation.READ_CLIPBOARD,
        PlanOperation.ACTIVATE_APP,
        PlanOperation.INSERT_TEXT,
    )
    assert first.action_target.text == "COPY_TRANSFER_VALUE_10"
    assert first.verification_target.text == "TRANSFER_COPIED_10"
    assert second.expected_text == "CROSS_APP_TRANSFER_10"
    assert third.app_name == "TextEdit"
    assert len(client.calls) == 1


def test_runtime_value_key_is_preserved_between_read_and_insert_steps():
    result, _client, _reasoner = _reason(
        _response(steps=_cross_app_steps_json())
    )

    read_step = result.plan.steps[1]
    insert_step = result.plan.steps[3]
    assert isinstance(read_step, ReadClipboardStep)
    assert isinstance(insert_step, InsertTextStep)
    assert read_step.value_key == "transfer_value"
    assert insert_step.value_key == "transfer_value"


def test_insert_text_step_does_not_contain_literal_payload_from_plan():
    result, _client, _reasoner = _reason(
        _response(steps=_cross_app_steps_json())
    )

    step = result.plan.steps[3]
    assert isinstance(step, InsertTextStep)
    assert "text" not in step.__dataclass_fields__
    assert "text_to_type" not in step.__dataclass_fields__
    assert "expected_text" not in step.__dataclass_fields__
    assert "CROSS_APP_TRANSFER_10" not in {
        getattr(step, field_name)
        for field_name in step.__dataclass_fields__
    }


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


def test_type_into_target_uses_same_reasoning_element_type_policy():
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _type_into_target_step_json(
                    target=_target_json("Search", ("text_field", "text")),
                )
            ]
        )
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, WebTextInputStep)
    assert step.target.element_types == ("text_field", "text")
    assert len(client.calls) == 1


def test_system_prompt_communicates_supported_element_type_policy():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    for element_type in SUPPORTED_REASONING_ELEMENT_TYPES:
        assert element_type in prompt

    assert "semantic UI roles" in prompt
    assert "not descriptions of visible content" in prompt
    assert 'Do not choose "text" merely because a string is visible' in prompt
    assert "If the role is not known from the task intent" in prompt
    assert "empty element_types array" in prompt
    assert "Never invent a semantic role" in prompt


def test_system_prompt_documents_appearance_verification_role_policy():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert "verify that 'Results' appears" in prompt
    assert '"element_types": []' in prompt
    assert "unless the role was explicitly known" in prompt


def test_system_prompt_documents_heading_role_policy():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert "heading is a semantic role for a page, section, or article heading" in prompt
    assert "verify or target a heading or title represented semantically" in prompt
    assert "Do not use heading merely because text is visually prominent" in prompt
    assert "If the role is not known from the task intent" in prompt
    assert "empty element_types array" in prompt


def test_system_prompt_documents_type_into_target_policy():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert "type_into_target" in prompt
    assert '"target": target' in prompt
    assert '"input_text": string' in prompt
    assert '"max_attempts": 1' in prompt
    assert "literal caller-visible text" in prompt
    assert "not a command or tool call" in prompt
    assert "max_attempts must be exactly 1" in prompt
    assert "never coordinates" in prompt
    assert "tool names" in prompt


def test_system_prompt_requires_verbatim_task_goal_preservation():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert "task_goal must copy the caller's task intent verbatim" in prompt
    assert "Do not summarize, paraphrase, shorten, or rewrite task_goal" in prompt


def test_system_prompt_says_web_tasks_do_not_imply_activate_app():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert "Use activate_app only when the task explicitly asks" in prompt
    assert "activate, switch to, focus" in prompt
    assert "website, browser-based task, or webpage does not imply activate_app" in prompt
    assert "Do not add activate_app as preparatory housekeeping" in prompt


def test_system_prompt_requires_single_attempt_generated_click_target():
    reasoner = LLMReasoner(client=FakeLLMClient(_response()))
    prompt = reasoner.system_prompt

    assert (
        'click_target: {"goal": string, "operation": "click_target", '
        '"action_target": target, "verification_target": target, '
        '"max_attempts": 1}'
    ) in prompt
    assert "click_target max_attempts must be exactly 1" in prompt


def test_type_into_target_single_attempt_behavior_remains_unchanged():
    result, _client, _reasoner = _reason(
        _response(steps=[_type_into_target_step_json(max_attempts=1)])
    )

    assert result.status is ReasoningStatus.READY
    step = result.plan.steps[0]
    assert isinstance(step, WebTextInputStep)
    assert step.max_attempts == 1


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
            _response(steps=[_step_json(operation="future_operation")]),
            "unknown operation",
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
        (
            _response(
                steps=[
                    {
                        key: value
                        for key, value in _read_clipboard_step_json().items()
                        if key != "value_key"
                    }
                ]
            ),
            "read_clipboard missing value_key",
        ),
        (
            _response(
                steps=[
                    {
                        **_read_clipboard_step_json(),
                        "note": "extra",
                    }
                ]
            ),
            "read_clipboard extra key",
        ),
        (
            _response(
                steps=[
                    {
                        **_step_json(operation="read_clipboard"),
                    }
                ]
            ),
            "click-only keys on read_clipboard",
        ),
        (
            _response(
                steps=[
                    _read_clipboard_step_json(
                        value_key="transfer-value",
                    )
                ]
            ),
            "malformed read_clipboard value_key",
        ),
        (
            _response(
                steps=[
                    _insert_text_step_json(
                        value_key="transfer.value",
                    )
                ]
            ),
            "malformed insert_text value_key",
        ),
        (
            _response(
                steps=[
                    _read_clipboard_step_json(
                        expected_text="",
                    )
                ]
            ),
            "empty expected_text",
        ),
        (
            _response(
                steps=[
                    _activate_app_step_json(
                        app_name="",
                    )
                ]
            ),
            "empty app_name",
        ),
        (
            _response(
                steps=[
                    {
                        key: value
                        for key, value in _activate_app_step_json().items()
                        if key != "app_name"
                    }
                ]
            ),
            "activate_app missing app_name",
        ),
        (
            _response(
                steps=[
                    {
                        **_activate_app_step_json(),
                        "bundle_id": "com.apple.TextEdit",
                    }
                ]
            ),
            "activate_app executable detail",
        ),
        (
            _response(
                steps=[
                    {
                        key: value
                        for key, value in _insert_text_step_json().items()
                        if key != "value_key"
                    }
                ]
            ),
            "insert_text missing value_key",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "expected_text": "CROSS_APP_TRANSFER_10",
                    }
                ]
            ),
            "expected_text supplied to insert_text",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "text": "CROSS_APP_TRANSFER_10",
                    }
                ]
            ),
            "text supplied to insert_text",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "text_to_type": "CROSS_APP_TRANSFER_10",
                    }
                ]
            ),
            "text_to_type supplied to insert_text",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "tool_name": "paste_text",
                    }
                ]
            ),
            "raw tool_name",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "arguments": {"text": "CROSS_APP_TRANSFER_10"},
                    }
                ]
            ),
            "raw arguments",
        ),
        (
            _response(
                steps=[
                    {
                        **_insert_text_step_json(),
                        "hotkey": ["command", "v"],
                    }
                ]
            ),
            "raw hotkey",
        ),
        (
            _response(steps=[_insert_text_step_json(max_attempts=True)]),
            "insert_text bool max_attempts",
        ),
        (
            _response(steps=[_activate_app_step_json(max_attempts=0)]),
            "activate_app zero max_attempts",
        ),
        (
            _duplicate_runtime_step_key_response(),
            "duplicate runtime step key",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        max_attempts=0,
                    )
                ]
            ),
            "type_into_target zero max_attempts",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        max_attempts=2,
                    )
                ]
            ),
            "type_into_target two max_attempts",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        max_attempts=True,
                    )
                ]
            ),
            "type_into_target bool max_attempts",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        max_attempts=1.0,
                    )
                ]
            ),
            "type_into_target float max_attempts",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        input_text="",
                    )
                ]
            ),
            "type_into_target empty input_text",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        input_text="   ",
                    )
                ]
            ),
            "type_into_target whitespace input_text",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        input_text=123,
                    )
                ]
            ),
            "type_into_target non-string input_text",
        ),
        (
            _response(
                steps=[
                    {
                        key: value
                        for key, value in _type_into_target_step_json().items()
                        if key != "input_text"
                    }
                ]
            ),
            "type_into_target missing input_text",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "note": "extra",
                    }
                ]
            ),
            "type_into_target extra key",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "action_target": _target_json("Search"),
                    }
                ]
            ),
            "click-only action_target on type_into_target",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "verification_target": _target_json("Results"),
                    }
                ]
            ),
            "click-only verification_target on type_into_target",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "value_key": "transfer_value",
                    }
                ]
            ),
            "runtime field on type_into_target",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "tool_name": "type_text",
                    }
                ]
            ),
            "tool name on type_into_target",
        ),
        (
            _response(
                steps=[
                    {
                        **_type_into_target_step_json(),
                        "arguments": {"text": "typing"},
                    }
                ]
            ),
            "raw arguments on type_into_target",
        ),
        (
            _response(
                steps=[
                    _type_into_target_step_json(
                        target={
                            **_target_json("Search"),
                            "coordinates": [1, 2],
                        },
                    )
                ]
            ),
            "coordinates in type_into_target target",
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


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "coordinates",
        "x",
        "y",
        "bounding_box",
        "tool_name",
        "arguments",
        "identifier",
        "minimum_confidence",
        "reference_point",
    ],
)
def test_forbidden_type_into_target_target_fields_fail_closed(
    forbidden_field: str,
):
    result, client, _reasoner = _reason(
        _response(
            steps=[
                _type_into_target_step_json(
                    target={
                        **_target_json("Search"),
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
