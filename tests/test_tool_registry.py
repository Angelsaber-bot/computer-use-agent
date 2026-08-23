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


class TypeValidationTool(BaseTool):
    """Test tool for primitive argument type validation."""

    name = "type_validation"
    description = "Validate primitive values."

    parameters = {
        "integer": ToolParameter(
            int,
            "Integer value.",
            required=False,
            default=0,
        ),
        "number": ToolParameter(
            (int, float),
            "Numeric value.",
            required=False,
            default=0,
        ),
        "flag": ToolParameter(
            bool,
            "Boolean value.",
            required=False,
            default=False,
        ),
        "bool_or_int": ToolParameter(
            (bool, int),
            "Boolean or integer value.",
            required=False,
            default=False,
        ),
    }

    def run(self, **arguments):
        return arguments


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


@pytest.mark.parametrize("value", [True, False])
def test_tool_rejects_bool_for_int_parameter(value):
    with pytest.raises(
        ToolValidationError,
        match="argument 'integer' must be int, not bool",
    ):
        TypeValidationTool().validate_arguments(
            {"integer": value}
        )


@pytest.mark.parametrize("value", [True, False])
def test_tool_rejects_bool_for_numeric_tuple_parameter(value):
    with pytest.raises(
        ToolValidationError,
        match="argument 'number' must be int or float, not bool",
    ):
        TypeValidationTool().validate_arguments(
            {"number": value}
        )


@pytest.mark.parametrize("value", [True, False])
def test_tool_accepts_bool_for_bool_parameter(value):
    arguments = TypeValidationTool().validate_arguments(
        {"flag": value}
    )

    assert arguments["flag"] is value


@pytest.mark.parametrize("value", [True, False])
def test_tool_accepts_bool_when_explicitly_in_type_tuple(value):
    arguments = TypeValidationTool().validate_arguments(
        {"bool_or_int": value}
    )

    assert arguments["bool_or_int"] is value


def test_tool_accepts_ordinary_numeric_values():
    arguments = TypeValidationTool().validate_arguments(
        {
            "integer": 7,
            "number": 1.5,
        }
    )

    assert arguments["integer"] == 7
    assert arguments["number"] == 1.5


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
