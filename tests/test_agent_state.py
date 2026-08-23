import pytest

from computer_agent.agent.state import AgentState, AgentStatus
from computer_agent.core.models import (
    Action,
    Observation,
    ToolResult,
)


def test_new_task_starts_pending():
    state = AgentState(user_task="Open a webpage")

    assert state.status is AgentStatus.PENDING
    assert state.steps == []
    assert state.next_step_number == 1
    assert state.last_error is None
    assert state.task_id


def test_task_rejects_empty_description():
    with pytest.raises(ValueError, match="user_task cannot be empty"):
        AgentState(user_task="   ")


def test_task_can_start():
    state = AgentState(user_task="Open a webpage")

    state.start()

    assert state.status is AgentStatus.RUNNING


def test_running_task_records_successful_step():
    state = AgentState(user_task="Open a webpage")
    state.start()

    action = Action(
        tool_name="open_url",
        arguments={"url": "https://example.com"},
    )

    result = ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=True,
        output="https://example.com",
    )

    observation = Observation(
        source="browser",
        data={"loaded": True},
    )

    record = state.record_step(
        action=action,
        result=result,
        observation=observation,
    )

    assert record.step_number == 1
    assert state.steps == [record]
    assert state.next_step_number == 2
    assert state.last_error is None


def test_task_records_failure():
    state = AgentState(user_task="Click a button")
    state.start()

    action = Action(
        tool_name="click_mouse",
        arguments={"x": 500, "y": 300},
    )

    result = ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=False,
        error="button was not found",
    )

    state.record_step(action=action, result=result)
    state.fail(result.error)

    assert state.status is AgentStatus.FAILED
    assert state.last_error == "button was not found"
    assert len(state.steps) == 1


def test_completed_task_rejects_more_changes():
    state = AgentState(user_task="Complete a task")
    state.start()
    state.succeed()

    assert state.status is AgentStatus.SUCCEEDED

    with pytest.raises(ValueError, match="cannot start"):
        state.start()

    with pytest.raises(ValueError, match="completed task cannot fail"):
        state.fail("late failure")