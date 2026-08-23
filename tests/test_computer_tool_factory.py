from unittest.mock import Mock

from computer_agent.core.models import Action
from computer_agent.tools.computer import (
    create_computer_tools,
)
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


EXPECTED_TOOL_NAMES = {
    "activate_app",
    "capture_screenshot",
    "click_mouse",
    "copy_to_clipboard",
    "get_mouse_position",
    "get_screen_size",
    "hotkey",
    "move_mouse",
    "open_url",
    "paste_text",
    "press_key",
    "read_from_clipboard",
    "scroll",
    "type_text",
}


def test_factory_creates_all_computer_tools():
    controller = Mock()

    tools = create_computer_tools(controller)

    assert len(tools) == 14

    assert {
        tool.name
        for tool in tools
    } == EXPECTED_TOOL_NAMES

    assert all(
        tool.controller is controller
        for tool in tools
    )


def test_registered_computer_tool_executes_action(
    monkeypatch,
):
    monkeypatch.setattr(
        "computer_agent.tools.base.sys.platform",
        "darwin",
    )

    controller = Mock()

    registry = ToolRegistry(
        create_computer_tools(controller)
    )

    executor = ToolExecutor(registry)

    action = Action(
        tool_name="open_url",
        arguments={
            "url": "https://example.com",
        },
    )

    result = executor.execute(action)

    assert len(registry) == 14
    assert result.success is True

    assert result.output == {
        "url": "https://example.com",
        "browser": "Google Chrome",
    }

    controller.open_url.assert_called_once_with(
        "https://example.com",
        browser="Google Chrome",
    )