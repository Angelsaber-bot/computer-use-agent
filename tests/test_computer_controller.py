import subprocess

import pytest
from unittest.mock import patch

from computer_agent.control.computer_controller import ComputerController


@patch("computer_agent.control.computer_controller.pyautogui.position")
def test_get_mouse_position(mock_position):
    mock_position.return_value = (100, 200)

    controller = ComputerController()
    result = controller.get_mouse_position()

    assert result == (100, 200)
    mock_position.assert_called_once()

@patch("computer_agent.control.computer_controller.pyautogui.moveTo")
def test_move_mouse(mock_move):
    controller = ComputerController()
    controller.move_mouse(100, 200, duration=0.25)

    mock_move.assert_called_once_with(100, 200, duration=0.25)

@patch("computer_agent.control.computer_controller.pyautogui.click")
def test_click_mouse(mock_click):
    controller = ComputerController()
    controller.click_mouse(100, 200)

    mock_click.assert_called_once_with(100, 200)


@patch("computer_agent.control.computer_controller.pyautogui.scroll")
def test_scroll(mock_scroll):
    controller = ComputerController()
    controller.scroll(-5)

    mock_scroll.assert_called_once_with(-5)

@patch("computer_agent.control.computer_controller.pyautogui.write")
def test_type_text(mock_write):
    controller = ComputerController()
    controller.type_text("Hello", interval=0.1)

    mock_write.assert_called_once_with("Hello", interval=0.1)


@patch("computer_agent.control.computer_controller.pyautogui.press")
def test_press_key(mock_press):
    controller = ComputerController()
    controller.press_key("enter")

    mock_press.assert_called_once_with("enter")


@patch("computer_agent.control.computer_controller.pyautogui.hotkey")
def test_hotkey(mock_hotkey):
    controller = ComputerController()
    controller.hotkey("command", "c")

    mock_hotkey.assert_called_once_with(
        "command",
        "c",
        interval=0.1
    )

@patch("computer_agent.control.computer_controller.pyperclip.copy")
def test_copy_to_clipboard(mock_copy):
    controller = ComputerController()
    controller.copy_to_clipboard("Hello")

    mock_copy.assert_called_once_with("Hello")


@patch("computer_agent.control.computer_controller.pyperclip.paste")
def test_read_from_clipboard(mock_paste):
    mock_paste.return_value = "Hello"

    controller = ComputerController()
    result = controller.read_from_clipboard()

    assert result == "Hello"


@patch("computer_agent.control.computer_controller.pyautogui.hotkey")
@patch("computer_agent.control.computer_controller.pyperclip.copy")
def test_paste_text(mock_copy, mock_hotkey):
    controller = ComputerController()
    controller.paste_text("Hello")

    mock_copy.assert_called_once_with("Hello")
    mock_hotkey.assert_called_once_with(
        "command",
        "v",
        interval=0.1
    )

@patch("computer_agent.control.computer_controller.pyautogui.size")
def test_get_screen_size(mock_size):
    mock_size.return_value = (1470, 956)

    controller = ComputerController()
    result = controller.get_screen_size()

    assert result == (1470, 956)


@patch("computer_agent.control.computer_controller.pyautogui.screenshot")
def test_capture_screenshot(mock_screenshot):
    controller = ComputerController()
    result = controller.capture_screenshot("screen.png")

    mock_screenshot.return_value.save.assert_called_once_with("screen.png")
    assert result == "screen.png"

@patch("computer_agent.control.computer_controller.subprocess.run")
def test_activate_app(mock_run):
    controller = ComputerController()
    controller.activate_app("TextEdit")

    mock_run.assert_called_once_with(
        ["open", "-a", "TextEdit"],
        check=True
    )


@patch("computer_agent.control.computer_controller.subprocess.run")
def test_open_url(mock_run):
    controller = ComputerController()
    controller.open_url("https://example.com")

    mock_run.assert_called_once_with(
        ["open", "-a", "Google Chrome", "https://example.com"],
        check=True
    )

@patch("computer_agent.control.computer_controller.subprocess.run")
def test_open_url_in_new_chrome_window(mock_run):
    controller = ComputerController()

    controller.open_url(
        "https://example.com",
        browser="Google Chrome",
        new_window=True,
    )

    mock_run.assert_called_once()

    command = mock_run.call_args.args[0]

    assert command[0] == "osascript"
    assert command[1] == "-e"
    assert "make new window" in command[2]
    assert (
        "set URL of active tab of taskWindow"
        in command[2]
    )
    assert command[-1] == "https://example.com"
    assert (
        mock_run.call_args.kwargs["check"]
        is True
    )


@patch(
    "computer_agent.control.computer_controller.subprocess.run"
)
def test_open_task_chrome_window_creates_marker_tab(mock_run):
    controller = ComputerController()

    marker_url = controller.open_task_chrome_window(
        "https://www.python.org/",
        "task-123",
    )

    assert marker_url == (
        "about:blank#computer-agent-task=task-123"
    )
    mock_run.assert_called_once()

    command = mock_run.call_args.args[0]

    assert command[0] == "osascript"
    assert command[1] == "-e"
    assert "make new window" in command[2]
    assert "make new tab at end of tabs" in command[2]
    assert "set active tab index of taskWindow to 1" in command[2]
    assert command[-2] == "https://www.python.org/"
    assert command[-1] == marker_url


@patch(
    "computer_agent.control.computer_controller.subprocess.run"
)
def test_activate_task_chrome_window_selects_working_tab(mock_run):
    controller = ComputerController()

    controller.activate_task_chrome_window(
        "about:blank#computer-agent-task=task-123",
        "https://www.python.org/",
    )

    mock_run.assert_called_once()

    command = mock_run.call_args.args[0]

    assert command[0] == "osascript"
    assert command[1] == "-e"
    assert "markerFoundInWindow" in command[2]
    assert "workingTabCount is not 1" in command[2]
    assert "foundWindowCount is greater than 1" in command[2]
    assert (
        "set active tab index of selectedWindow to selectedTabIndex"
        in command[2]
    )
    assert command[-2] == "about:blank#computer-agent-task=task-123"
    assert command[-1] == "https://www.python.org/"


def test_activate_task_chrome_window_rejects_invalid_marker():
    controller = ComputerController()

    with pytest.raises(
        ValueError,
        match="marker_url must be a valid",
    ):
        controller.activate_task_chrome_window(
            "chrome-window-id:48291",
            "https://www.python.org/",
        )


def test_activate_task_chrome_window_rejects_empty_working_prefix():
    controller = ComputerController()

    with pytest.raises(
        ValueError,
        match="working_url_prefix must be a non-empty string",
    ):
        controller.activate_task_chrome_window(
            "about:blank#computer-agent-task=task-123",
            " ",
        )


@patch(
    "computer_agent.control.computer_controller.subprocess.run"
)
def test_activate_task_chrome_window_fails_closed_when_chrome_rejects(
    mock_run,
):
    mock_run.side_effect = subprocess.CalledProcessError(
        1,
        ["osascript"],
    )
    controller = ComputerController()

    with pytest.raises(
        RuntimeError,
        match="could not be safely resumed",
    ):
        controller.activate_task_chrome_window(
            "about:blank#computer-agent-task=task-123",
            "https://www.python.org/",
        )
