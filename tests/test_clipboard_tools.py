from unittest.mock import Mock

from computer_agent.tools.computer.clipboard import (
    CopyToClipboardTool,
    PasteTextTool,
    ReadFromClipboardTool,
)


def test_copy_to_clipboard_tool():
    controller = Mock()
    tool = CopyToClipboardTool(controller)

    arguments = tool.validate_arguments(
        {
            "text": "Hello",
        }
    )

    output = tool.run(**arguments)

    controller.copy_to_clipboard.assert_called_once_with(
        "Hello"
    )

    assert output == {
        "text": "Hello",
    }


def test_read_from_clipboard_tool():
    controller = Mock()
    controller.read_from_clipboard.return_value = (
        "Copied text"
    )

    tool = ReadFromClipboardTool(controller)
    output = tool.run()

    controller.read_from_clipboard.assert_called_once()

    assert output == {
        "text": "Copied text",
    }


def test_paste_text_tool():
    controller = Mock()
    tool = PasteTextTool(controller)

    arguments = tool.validate_arguments(
        {
            "text": "Hello",
        }
    )

    output = tool.run(**arguments)

    controller.paste_text.assert_called_once_with(
        "Hello"
    )

    assert output == {
        "text": "Hello",
    }