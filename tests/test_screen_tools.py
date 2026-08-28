from unittest.mock import Mock

import pytest

from computer_agent.tools.base import ToolValidationError
from computer_agent.tools.computer import screen as screen_module
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
    expected_path = (
        screen_module._SCREENSHOT_ROOT
        / "phase01_computer_control/screen.png"
    )
    controller.capture_screenshot.return_value = expected_path

    tool = CaptureScreenshotTool(controller)

    arguments = tool.validate_arguments(
        {
            "output_path": "phase01_computer_control/screen.png",
        }
    )

    output = tool.run(**arguments)

    controller.capture_screenshot.assert_called_once_with(
        str(expected_path)
    )

    assert output == {
        "output_path": str(expected_path),
    }


def test_capture_screenshot_accepts_project_relative_path():
    controller = Mock()
    expected_path = (
        screen_module._SCREENSHOT_ROOT
        / "phase01_computer_control/project-relative.png"
    )
    controller.capture_screenshot.return_value = expected_path
    tool = CaptureScreenshotTool(controller)

    arguments = tool.validate_arguments(
        {
            "output_path": (
                "assets/screenshots/phase01_computer_control/"
                "project-relative.png"
            ),
        }
    )

    output = tool.run(**arguments)

    controller.capture_screenshot.assert_called_once_with(
        str(expected_path)
    )
    assert output == {
        "output_path": str(expected_path),
    }


def test_capture_screenshot_accepts_absolute_path_inside_allowed_directory():
    controller = Mock()
    expected_path = (
        screen_module._SCREENSHOT_ROOT
        / "phase01_computer_control/absolute.png"
    )
    controller.capture_screenshot.return_value = expected_path
    tool = CaptureScreenshotTool(controller)

    arguments = tool.validate_arguments(
        {
            "output_path": str(expected_path),
        }
    )

    output = tool.run(**arguments)

    controller.capture_screenshot.assert_called_once_with(
        str(expected_path)
    )
    assert output == {
        "output_path": str(expected_path),
    }


@pytest.mark.parametrize(
    ("output_path", "error_message"),
    [
        ("", "argument 'output_path' cannot be empty"),
        ("   ", "argument 'output_path' cannot be empty"),
        (
            "assets/screenshots",
            "argument 'output_path' must be a file path",
        ),
        (".", "argument 'output_path' must be a file path"),
        (
            "phase01_computer_control",
            "argument 'output_path' must be a file path",
        ),
    ],
)
def test_capture_screenshot_rejects_non_file_output_paths(
    output_path,
    error_message,
):
    controller = Mock()
    tool = CaptureScreenshotTool(controller)
    arguments = tool.validate_arguments(
        {
            "output_path": output_path,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match=error_message,
    ):
        tool.run(**arguments)

    controller.capture_screenshot.assert_not_called()


def test_capture_screenshot_rejects_screenshot_root_path():
    controller = Mock()
    tool = CaptureScreenshotTool(controller)
    arguments = tool.validate_arguments(
        {
            "output_path": str(screen_module._SCREENSHOT_ROOT),
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'output_path' must be a file path",
    ):
        tool.run(**arguments)

    controller.capture_screenshot.assert_not_called()


def test_capture_screenshot_converts_path_resolution_errors():
    controller = Mock()
    tool = CaptureScreenshotTool(controller)
    arguments = tool.validate_arguments(
        {
            "output_path": "bad\x00path.png",
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'output_path' is not a valid path",
    ):
        tool.run(**arguments)

    controller.capture_screenshot.assert_not_called()


@pytest.mark.parametrize(
    "output_path",
    [
        "phase01_computer_control/../../fixtures/escape.png",
        "assets/screenshots/../../README.md",
    ],
)
def test_capture_screenshot_rejects_path_traversal(output_path):
    controller = Mock()
    tool = CaptureScreenshotTool(controller)
    arguments = tool.validate_arguments(
        {
            "output_path": output_path,
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'output_path' must be inside assets/screenshots",
    ):
        tool.run(**arguments)

    controller.capture_screenshot.assert_not_called()


def test_capture_screenshot_rejects_absolute_path_outside_allowed_directory(
    tmp_path,
):
    controller = Mock()
    tool = CaptureScreenshotTool(controller)
    arguments = tool.validate_arguments(
        {
            "output_path": str(tmp_path / "outside.png"),
        }
    )

    with pytest.raises(
        ToolValidationError,
        match="argument 'output_path' must be inside assets/screenshots",
    ):
        tool.run(**arguments)

    controller.capture_screenshot.assert_not_called()
