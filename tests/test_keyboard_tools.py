from unittest.mock import Mock

import pytest

from computer_agent.tools.base import (
    ToolValidationError,
)
from computer_agent.tools.computer.keyboard import (
    HotkeyTool,
    PressKeyTool,
    TypeTextTool,
)


def test_type_text_tool():
    controller = Mock()
    tool = TypeTextTool(controller)

    arguments = tool.validate_arguments(
        {
            "text": "Hello",
        }
    )

    output = tool.run(**arguments)

    controller.type_text.assert_called_once_with(
        "Hello",
        interval=0.05,
    )

    assert output == {
        "text": "Hello",
        "interval": 0.05,
    }


def test_press_key_tool():
    controller = Mock()
    tool = PressKeyTool(controller)

    arguments = tool.validate_arguments(
        {
            "key": "enter",
        }
    )

    output = tool.run(**arguments)

    controller.press_key.assert_called_once_with(
        "enter"
    )

    assert output == {
        "key": "enter",
    }


def test_hotkey_tool():
    controller = Mock()
    tool = HotkeyTool(controller)

    arguments = tool.validate_arguments(
        {
            "keys": [
                "command",
                "v",
            ],
        }
    )

    output = tool.run(**arguments)

    controller.hotkey.assert_called_once_with(
        "command",
        "v",
        interval=0.1,
    )

    assert output == {
        "keys": [
            "command",
            "v",
        ],
        "interval": 0.1,
    }


@pytest.mark.parametrize(
    "keys",
    [
        [],
        ["command", 5],
        ["command", "   "],
    ],
)
def test_hotkey_rejects_invalid_key_list(keys):
    controller = Mock()
    tool = HotkeyTool(controller)

    arguments = tool.validate_arguments(
        {
            "keys": keys,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=(
            "argument 'keys' must be "
            "a non-empty list of strings"
        ),
    ):
        tool.run(**arguments)

    controller.hotkey.assert_not_called()