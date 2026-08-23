"""Mouse-control tools backed by ComputerController."""

from computer_agent.tools.base import ToolParameter
from computer_agent.tools.computer.base import ComputerTool


class GetMousePositionTool(ComputerTool):
    """Return the current mouse position."""

    name = "get_mouse_position"
    description = "Return the current mouse coordinates."

    def run(self, **arguments):
        x, y = self.controller.get_mouse_position()

        return {
            "x": x,
            "y": y,
        }


class MoveMouseTool(ComputerTool):
    """Move the mouse to a screen coordinate."""

    name = "move_mouse"
    description = "Move the mouse to an x and y coordinate."

    parameters = {
        "x": ToolParameter(
            int,
            "Horizontal screen coordinate.",
        ),
        "y": ToolParameter(
            int,
            "Vertical screen coordinate.",
        ),
        "duration": ToolParameter(
            (int, float),
            "Movement duration in seconds.",
            required=False,
            default=0.5,
        ),
    }

    def run(self, **arguments):
        self.controller.move_mouse(
            arguments["x"],
            arguments["y"],
            duration=arguments["duration"],
        )

        return {
            "x": arguments["x"],
            "y": arguments["y"],
            "duration": arguments["duration"],
        }


class ClickMouseTool(ComputerTool):
    """Click the mouse at a screen coordinate."""

    name = "click_mouse"
    description = "Click the mouse at an x and y coordinate."

    parameters = {
        "x": ToolParameter(
            int,
            "Horizontal screen coordinate.",
        ),
        "y": ToolParameter(
            int,
            "Vertical screen coordinate.",
        ),
    }

    def run(self, **arguments):
        self.controller.click_mouse(
            arguments["x"],
            arguments["y"],
        )

        return {
            "x": arguments["x"],
            "y": arguments["y"],
        }


class ScrollTool(ComputerTool):
    """Scroll the active page or window."""

    name = "scroll"
    description = "Scroll by a positive or negative amount."

    parameters = {
        "amount": ToolParameter(
            int,
            "Positive scrolls up and negative scrolls down.",
        ),
    }

    def run(self, **arguments):
        self.controller.scroll(
            arguments["amount"]
        )

        return {
            "amount": arguments["amount"],
        }