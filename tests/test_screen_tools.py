from unittest.mock import Mock

from computer_agent.tools.computer.screen import (
    CaptureScreenshotTool,
    GetScreenSizeTool,
)


def test_get_screen_size_tool():
    controller = Mock()
    controller.get_screen_size.return_value = (
        1470,
        956,
    )

    tool = GetScreenSizeTool(controller)
    output = tool.run()

    controller.get_screen_size.assert_called_once()

    assert output == {
        "width": 1470,
        "height": 956,
    }


def test_capture_screenshot_tool():
    controller = Mock()
    controller.capture_screenshot.return_value = (
        "screen.png"
    )

    tool = CaptureScreenshotTool(controller)

    arguments = tool.validate_arguments(
        {
            "output_path": "screen.png",
        }
    )

    output = tool.run(**arguments)

    controller.capture_screenshot.assert_called_once_with(
        "screen.png"
    )

    assert output == {
        "output_path": "screen.png",
    }