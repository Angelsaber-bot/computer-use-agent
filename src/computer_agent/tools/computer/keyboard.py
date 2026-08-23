"""Keyboard-control tools backed by ComputerController."""

from computer_agent.tools.base import (
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.computer.base import ComputerTool


class TypeTextTool(ComputerTool):
    """Type text using the keyboard."""

    name = "type_text"
    description = "Type text into the active application."

    parameters = {
        "text": ToolParameter(
            str,
            "Text to type.",
        ),
        "interval": ToolParameter(
            (int, float),
            "Delay between characters in seconds.",
            required=False,
            default=0.05,
        ),
    }

    def run(self, **arguments):
        self.controller.type_text(
            arguments["text"],
            interval=arguments["interval"],
        )

        return {
            "text": arguments["text"],
            "interval": arguments["interval"],
        }


class PressKeyTool(ComputerTool):
    """Press one keyboard key."""

    name = "press_key"
    description = "Press one keyboard key."

    parameters = {
        "key": ToolParameter(
            str,
            "Name of the key to press.",
        ),
    }

    def run(self, **arguments):
        self.controller.press_key(
            arguments["key"]
        )

        return {
            "key": arguments["key"],
        }


class HotkeyTool(ComputerTool):
    """Press a keyboard shortcut."""

    name = "hotkey"
    description = "Press multiple keys as one shortcut."

    parameters = {
        "keys": ToolParameter(
            list,
            "Ordered list of keys in the shortcut.",
        ),
        "interval": ToolParameter(
            (int, float),
            "Delay between shortcut keys in seconds.",
            required=False,
            default=0.1,
        ),
    }

    def run(self, **arguments):
        keys = arguments["keys"]

        if (
            not keys
            or not all(
                isinstance(key, str) and key.strip()
                for key in keys
            )
        ):
            raise ToolValidationError(
                "argument 'keys' must be "
                "a non-empty list of strings"
            )

        self.controller.hotkey(
            *keys,
            interval=arguments["interval"],
        )

        return {
            "keys": keys,
            "interval": arguments["interval"],
        }