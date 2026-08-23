"""Screenshot capture converted into perception frames."""

from pathlib import Path
from typing import Protocol

from PIL import Image

from computer_agent.perception.models import ScreenFrame


class ScreenController(Protocol):
    """Controller behavior needed by screen perception."""

    def capture_screenshot(self, output_path: str) -> str | Path:
        """Capture a screenshot to the provided path."""

    def get_screen_size(self) -> tuple[int, int]:
        """Return the logical screen size."""


class ScreenCapture:
    """Capture screenshots and attach coordinate metadata."""

    def __init__(
        self,
        controller: ScreenController,
    ) -> None:
        self.controller = controller

    def capture(self, output_path: str | Path) -> ScreenFrame:
        """Capture the screen and return a structured frame."""

        requested_path = Path(output_path)
        saved_path = Path(
            self.controller.capture_screenshot(str(requested_path))
        )

        if not saved_path.exists():
            raise FileNotFoundError(
                f"Screenshot file was not created: {saved_path}"
            )

        with Image.open(saved_path) as image:
            pixel_width, pixel_height = image.size

        screen_width, screen_height = self.controller.get_screen_size()

        return ScreenFrame(
            image_path=saved_path.resolve(),
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            screen_width=screen_width,
            screen_height=screen_height,
        )
