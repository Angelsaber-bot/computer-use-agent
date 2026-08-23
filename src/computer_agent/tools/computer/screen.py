"""Screen tools backed by ComputerController."""

from computer_agent.tools.base import ToolParameter
from computer_agent.tools.computer.base import ComputerTool


class GetScreenSizeTool(ComputerTool):
    """Return the current screen size."""

    name = "get_screen_size"
    description = "Return the screen width and height."

    def run(self, **arguments):
        width, height = (
            self.controller.get_screen_size()
        )

        return {
            "width": width,
            "height": height,
        }


class CaptureScreenshotTool(ComputerTool):
    """Capture the full screen and save it to a file."""

    name = "capture_screenshot"
    description = (
        "Capture the full screen and save it "
        "to an image file."
    )

    parameters = {
        "output_path": ToolParameter(
            str,
            "Path where the screenshot will be saved.",
        ),
    }

    def run(self, **arguments):
        saved_path = (
            self.controller.capture_screenshot(
                arguments["output_path"]
            )
        )

        return {
            "output_path": str(saved_path),
        }