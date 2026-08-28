"""Mouse-control tools backed by ComputerController."""

import math
from numbers import Real

from computer_agent.tools.base import (
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.computer.base import ComputerTool


def _validate_duration(duration):
    if isinstance(duration, bool) or not isinstance(duration, Real):
        raise ToolValidationError(
            "argument 'duration' must be numeric"
        )

    if not math.isfinite(duration):
        raise ToolValidationError(
            "argument 'duration' must be finite"
        )

    if not 0.0 <= duration <= 5.0:
        raise ToolValidationError(
            "argument 'duration' must be between 0.0 and 5.0"
        )


def _validate_screen_coordinates(controller, x, y):
    screen_width, screen_height = controller.get_screen_size()

    if x < 0 or x >= screen_width:
        raise ToolValidationError(
            "argument 'x' must be inside the screen"
        )

    if y < 0 or y >= screen_height:
        raise ToolValidationError(
            "argument 'y' must be inside the screen"
        )


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
            (
                "Horizontal coordinate; must be within the "
                "current logical screen bounds."
            ),
        ),
        "y": ToolParameter(
            int,
            (
                "Vertical coordinate; must be within the "
                "current logical screen bounds."
            ),
        ),
        "duration": ToolParameter(
            (int, float),
            "Movement duration in seconds, from 0.0 through 5.0.",
            required=False,
            default=0.5,
        ),
    }

    def run(self, **arguments):
        _validate_duration(arguments["duration"])
        _validate_screen_coordinates(
            self.controller,
            arguments["x"],
            arguments["y"],
        )

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
            (
                "Horizontal coordinate; must be within the "
                "current logical screen bounds."
            ),
        ),
        "y": ToolParameter(
            int,
            (
                "Vertical coordinate; must be within the "
                "current logical screen bounds."
            ),
        ),
    }

    def run(self, **arguments):
        _validate_screen_coordinates(
            self.controller,
            arguments["x"],
            arguments["y"],
        )

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
