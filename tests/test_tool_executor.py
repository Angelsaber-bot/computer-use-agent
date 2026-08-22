from computer_agent.core.models import Action
from computer_agent.tools.base import (
    BaseTool,
    ToolParameter,
)
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


class AddTool(BaseTool):
    """Safe tool used to test successful execution."""

    name = "add"
    description = "Add two integers."

    parameters = {
        "a": ToolParameter(
            int,
            "First integer.",
        ),
        "b": ToolParameter(
            int,
            "Second integer.",
        ),
    }

    def run(self, **arguments):
        return arguments["a"] + arguments["b"]


class FailingTool(BaseTool):
    """Tool that intentionally raises an error."""

    name = "fail"
    description = "Raise a planned error."

    def run(self, **arguments):
        raise RuntimeError("planned failure")


class UnavailableTool(BaseTool):
    """Tool that supports no real operating system."""

    name = "unavailable"
    description = "A tool unavailable on this computer."

    supported_platforms = (
        "unsupported-test-platform",
    )

    def run(self, **arguments):
        return "should not run"


def test_executor_runs_valid_action():
    registry = ToolRegistry([AddTool()])
    executor = ToolExecutor(registry)

    action = Action(
        tool_name="add",
        arguments={"a": 2, "b": 3},
    )

    result = executor.execute(action)

    assert result.success is True
    assert result.output == 5
    assert result.error is None
    assert result.action_id == action.action_id
    assert result.tool_name == "add"
    assert result.duration_ms >= 0


def test_executor_returns_failure_for_missing_tool():
    executor = ToolExecutor(ToolRegistry())

    action = Action(
        tool_name="missing",
    )

    result = executor.execute(action)

    assert result.success is False
    assert result.output is None
    assert "ToolNotFoundError" in result.error
    assert "tool not found: missing" in result.error


def test_executor_returns_failure_for_invalid_arguments():
    registry = ToolRegistry([AddTool()])
    executor = ToolExecutor(registry)

    action = Action(
        tool_name="add",
        arguments={"a": 2},
    )

    result = executor.execute(action)

    assert result.success is False
    assert "ToolValidationError" in result.error
    assert "missing required argument: b" in result.error


def test_executor_catches_tool_runtime_error():
    registry = ToolRegistry([FailingTool()])
    executor = ToolExecutor(registry)

    action = Action(tool_name="fail")

    result = executor.execute(action)

    assert result.success is False
    assert result.output is None
    assert result.error == (
        "RuntimeError: planned failure"
    )


def test_executor_rejects_unavailable_tool():
    registry = ToolRegistry(
        [UnavailableTool()]
    )

    executor = ToolExecutor(registry)

    action = Action(
        tool_name="unavailable"
    )

    result = executor.execute(action)

    assert result.success is False
    assert "ToolUnavailableError" in result.error
    assert "is not available" in result.error


def test_executor_exposes_its_registry():
    registry = ToolRegistry([AddTool()])
    executor = ToolExecutor(registry)

    assert executor.registry is registry