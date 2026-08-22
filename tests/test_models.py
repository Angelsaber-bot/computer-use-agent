import pytest

from computer_agent.core.models import (
    Action,
    Observation,
    StepRecord,
    ToolResult,
)


def test_action_stores_tool_information():
    action = Action(
        tool_name="click_mouse",
        arguments={"x": 500, "y": 300},
        reason="Click the submit button",
    )

    assert action.tool_name == "click_mouse"
    assert action.arguments == {"x": 500, "y": 300}
    assert action.reason == "Click the submit button"
    assert action.action_id
    assert action.created_at.tzinfo is not None


def test_action_rejects_empty_tool_name():
    with pytest.raises(ValueError, match="tool_name cannot be empty"):
        Action(tool_name="   ")


def test_successful_tool_result():
    action = Action(tool_name="open_url")

    result = ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=True,
        output="https://example.com",
    )

    assert result.success is True
    assert result.output == "https://example.com"
    assert result.error is None


def test_failed_result_requires_error_message():
    with pytest.raises(
        ValueError,
        match="a failed result must contain an error",
    ):
        ToolResult(
            action_id="action-1",
            tool_name="click_mouse",
            success=False,
        )


def test_successful_result_rejects_error_message():
    with pytest.raises(
        ValueError,
        match="a successful result cannot contain an error",
    ):
        ToolResult(
            action_id="action-1",
            tool_name="click_mouse",
            success=True,
            error="unexpected error",
        )


def test_observation_requires_source():
    with pytest.raises(
        ValueError,
        match="observation source cannot be empty",
    ):
        Observation(source="   ", data={})


def test_step_record_requires_matching_action_and_result():
    action = Action(tool_name="click_mouse")

    result = ToolResult(
        action_id="different-action-id",
        tool_name=action.tool_name,
        success=True,
    )

    with pytest.raises(
        ValueError,
        match="action and result IDs must match",
    ):
        StepRecord(
            step_number=1,
            action=action,
            result=result,
        )