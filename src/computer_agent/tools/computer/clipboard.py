"""Clipboard tools backed by ComputerController."""

from computer_agent.tools.base import ToolParameter
from computer_agent.tools.computer.base import ComputerTool


class CopyToClipboardTool(ComputerTool):
    """Write text to the system clipboard."""

    name = "copy_to_clipboard"
    description = "Copy text to the system clipboard."

    parameters = {
        "text": ToolParameter(
            str,
            "Text to copy.",
        ),
    }

    def run(self, **arguments):
        self.controller.copy_to_clipboard(
            arguments["text"]
        )

        return {
            "text": arguments["text"],
        }


class ReadFromClipboardTool(ComputerTool):
    """Read text from the system clipboard."""

    name = "read_from_clipboard"
    description = "Read text from the system clipboard."

    def run(self, **arguments):
        text = self.controller.read_from_clipboard()

        return {
            "text": text,
        }


class PasteTextTool(ComputerTool):
    """Copy text and paste it into the active application."""

    name = "paste_text"
    description = (
        "Copy text and paste it into "
        "the active application."
    )

    parameters = {
        "text": ToolParameter(
            str,
            "Text to paste.",
        ),
    }

    def run(self, **arguments):
        self.controller.paste_text(
            arguments["text"]
        )

        return {
            "text": arguments["text"],
        }