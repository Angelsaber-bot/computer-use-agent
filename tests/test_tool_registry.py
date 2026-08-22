import pytest

from computer_agent.tools.base import (
    BaseTool,
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)


class EchoTool(BaseTool):
    """Safe test tool that returns a message."""

    name = "echo"
    description = "Return a message unchanged."

    parameters = {
        "message": ToolParameter(
            str,
            "Message to return.",
        ),
        "uppercase": ToolParameter(
            bool,
            "Whether to convert the message to uppercase.",
            required=False,
            default=False,
        ),
    }

    def run(self, **arguments):
        message = arguments["message"]

        if arguments["uppercase"]:
            return message.upper()

        return message


class MacOnlyTool(BaseTool):
    """Test tool that supports only macOS."""

    name = "mac_only"
    description = "A tool available only on macOS."
    supported_platforms = ("darwin",)

    def run(self, **arguments):
        return "macOS"


def test_tool_validates_arguments_and_adds_default():
    tool = EchoTool()

    arguments = tool.validate_arguments(
        {"message": "hello"}
    )

    assert arguments == {
        "message": "hello",
        "uppercase": False,
    }

    assert tool.run(**arguments) == "hello"


@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            {},
            "missing required argument: message",
        ),
        (
            {"message": 123},
            "must be str, not int",
        ),
        (
            {
                "message": "hello",
                "extra": True,
            },
            "unknown argument",
        ),
    ],
)
def test_tool_rejects_invalid_arguments(
    arguments,
    error_message,
):
    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        EchoTool().validate_arguments(arguments)


def test_tool_schema_contains_planner_metadata():
    schema = EchoTool().schema()

    assert schema["name"] == "echo"
    assert schema["description"] == (
        "Return a message unchanged."
    )

    properties = schema["parameters"]["properties"]

    assert properties["message"]["type"] == "string"
    assert properties["uppercase"]["type"] == "boolean"
    assert properties["uppercase"]["default"] is False
    assert schema["parameters"]["required"] == ["message"]


def test_platform_availability():
    tool = MacOnlyTool()

    assert tool.is_available("darwin") is True
    assert tool.is_available("win32") is False
    assert tool.is_available("linux") is False


def test_registry_registers_and_finds_tools():
    echo = EchoTool()
    mac_only = MacOnlyTool()

    registry = ToolRegistry(
        [mac_only, echo]
    )

    assert len(registry) == 2
    assert "echo" in registry
    assert registry.get("echo") is echo

    names = [
        tool.name
        for tool in registry.list_tools()
    ]

    assert names == ["echo", "mac_only"]


def test_registry_unregisters_tool():
    echo = EchoTool()
    registry = ToolRegistry([echo])

    removed = registry.unregister("echo")

    assert removed is echo
    assert len(registry) == 0
    assert "echo" not in registry


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([EchoTool()])

    with pytest.raises(
        DuplicateToolError,
        match="tool already registered: echo",
    ):
        registry.register(EchoTool())


def test_registry_reports_missing_tool():
    registry = ToolRegistry()

    with pytest.raises(
        ToolNotFoundError,
        match="tool not found: missing",
    ):
        registry.get("missing")