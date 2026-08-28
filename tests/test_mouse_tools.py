from unittest.mock import Mock

import pytest

from computer_agent.tools.base import ToolValidationError
from computer_agent.tools.computer.mouse import (
    ClickMouseTool,
    GetMousePositionTool,
    MoveMouseTool,
    ScrollTool,
)


def test_get_mouse_position_tool():
    controller = Mock()
    controller.get_mouse_position.return_value = (
        100,
        200,
    )

    tool = GetMousePositionTool(controller)
    output = tool.run()

    controller.get_mouse_position.assert_called_once()
    assert output == {
        "x": 100,
        "y": 200,
    }


def test_move_mouse_tool():
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = MoveMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 300,
            "y": 400,
        }
    )

    output = tool.run(**arguments)

    controller.get_screen_size.assert_called_once()
    controller.move_mouse.assert_called_once_with(
        300,
        400,
        duration=0.5,
    )

    assert output == {
        "x": 300,
        "y": 400,
        "duration": 0.5,
    }


def test_move_mouse_tool_accepts_explicit_duration():
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = MoveMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 300,
            "y": 400,
            "duration": 2.25,
        }
    )

    output = tool.run(**arguments)

    controller.move_mouse.assert_called_once_with(
        300,
        400,
        duration=2.25,
    )
    assert output["duration"] == 2.25


@pytest.mark.parametrize(
    "duration",
    [
        0.0,
        5.0,
    ],
)
def test_move_mouse_tool_accepts_duration_boundaries(duration):
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = MoveMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 300,
            "y": 400,
            "duration": duration,
        }
    )

    output = tool.run(**arguments)

    controller.move_mouse.assert_called_once_with(
        300,
        400,
        duration=duration,
    )
    assert output["duration"] == duration


def test_click_mouse_tool():
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = ClickMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 500,
            "y": 500,
        }
    )

    output = tool.run(**arguments)

    controller.get_screen_size.assert_called_once()
    controller.click_mouse.assert_called_once_with(
        500,
        500,
    )

    assert output == {
        "x": 500,
        "y": 500,
    }


@pytest.mark.parametrize(
    ("tool_class", "action_method"),
    [
        (MoveMouseTool, "move_mouse"),
        (ClickMouseTool, "click_mouse"),
    ],
)
def test_mouse_tools_accept_boundary_coordinates(
    tool_class,
    action_method,
):
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = tool_class(controller)

    first = tool.validate_arguments(
        {
            "x": 0,
            "y": 0,
        }
    )
    second = tool.validate_arguments(
        {
            "x": 799,
            "y": 599,
        }
    )

    tool.run(**first)
    tool.run(**second)

    assert getattr(controller, action_method).call_count == 2


@pytest.mark.parametrize(
    ("x", "y", "error_message"),
    [
        (-1, 10, "argument 'x' must be inside the screen"),
        (10, -1, "argument 'y' must be inside the screen"),
        (800, 10, "argument 'x' must be inside the screen"),
        (801, 10, "argument 'x' must be inside the screen"),
        (10, 600, "argument 'y' must be inside the screen"),
        (10, 601, "argument 'y' must be inside the screen"),
    ],
)
@pytest.mark.parametrize(
    ("tool_class", "action_method"),
    [
        (MoveMouseTool, "move_mouse"),
        (ClickMouseTool, "click_mouse"),
    ],
)
def test_mouse_tools_reject_out_of_bounds_coordinates(
    tool_class,
    action_method,
    x,
    y,
    error_message,
):
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = tool_class(controller)
    arguments = tool.validate_arguments(
        {
            "x": x,
            "y": y,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        tool.run(**arguments)

    getattr(controller, action_method).assert_not_called()


@pytest.mark.parametrize(
    ("duration", "error_message"),
    [
        (-0.01, "argument 'duration' must be between 0.0 and 5.0"),
        (5.01, "argument 'duration' must be between 0.0 and 5.0"),
        (float("nan"), "argument 'duration' must be finite"),
        (float("inf"), "argument 'duration' must be finite"),
        (float("-inf"), "argument 'duration' must be finite"),
    ],
)
def test_move_mouse_tool_rejects_invalid_duration(
    duration,
    error_message,
):
    controller = Mock()
    controller.get_screen_size.return_value = (
        800,
        600,
    )
    tool = MoveMouseTool(controller)
    arguments = tool.validate_arguments(
        {
            "x": 300,
            "y": 400,
            "duration": duration,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        tool.run(**arguments)

    controller.get_screen_size.assert_not_called()
    controller.move_mouse.assert_not_called()


def test_scroll_tool():
    controller = Mock()
    tool = ScrollTool(controller)

    arguments = tool.validate_arguments(
        {
            "amount": -5,
        }
    )

    output = tool.run(**arguments)

    controller.scroll.assert_called_once_with(-5)

    assert output == {
        "amount": -5,
    }
