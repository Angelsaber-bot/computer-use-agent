from unittest.mock import Mock

import pytest

from computer_agent.tools.base import (
    ToolValidationError,
)
from computer_agent.tools.computer.application import (
    ActivateAppTool,
    OpenURLTool,
)


def test_activate_app_tool():
    controller = Mock()
    tool = ActivateAppTool(controller)

    arguments = tool.validate_arguments(
        {
            "app_name": "TextEdit",
        }
    )

    output = tool.run(**arguments)

    controller.activate_app.assert_called_once_with(
        "TextEdit"
    )

    assert output == {
        "app_name": "TextEdit",
    }


def test_open_url_tool_uses_default_browser():
    controller = Mock()
    tool = OpenURLTool(controller)

    arguments = tool.validate_arguments(
        {
            "url": "https://example.com",
        }
    )

    output = tool.run(**arguments)

    controller.open_url.assert_called_once_with(
        "https://example.com",
        browser="Google Chrome",
    )

    assert output == {
        "url": "https://example.com",
        "browser": "Google Chrome",
    }


def test_activate_app_rejects_empty_name():
    controller = Mock()
    tool = ActivateAppTool(controller)

    arguments = tool.validate_arguments(
        {
            "app_name": "   ",
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'app_name' cannot be empty",
    ):
        tool.run(**arguments)

    controller.activate_app.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "error_message"),
    [
        (
            {
                "url": "   ",
            },
            "argument 'url' cannot be empty",
        ),
        (
            {
                "url": "https://example.com",
                "browser": "   ",
            },
            "argument 'browser' cannot be empty",
        ),
    ],
)
def test_open_url_rejects_empty_values(
    arguments,
    error_message,
):
    controller = Mock()
    tool = OpenURLTool(controller)

    validated = tool.validate_arguments(arguments)

    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        tool.run(**validated)

    controller.open_url.assert_not_called()