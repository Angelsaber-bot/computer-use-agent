from unittest.mock import Mock

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
    tool = MoveMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 300,
            "y": 400,
        }
    )

    output = tool.run(**arguments)

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


def test_click_mouse_tool():
    controller = Mock()
    tool = ClickMouseTool(controller)

    arguments = tool.validate_arguments(
        {
            "x": 500,
            "y": 600,
        }
    )

    output = tool.run(**arguments)

    controller.click_mouse.assert_called_once_with(
        500,
        600,
    )

    assert output == {
        "x": 500,
        "y": 600,
    }


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