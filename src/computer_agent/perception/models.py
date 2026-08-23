"""Data models used by the screen perception system."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from computer_agent.core.models import utc_now


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    """A captured screenshot and its coordinate metadata."""

    image_path: Path
    pixel_width: int
    pixel_height: int
    screen_width: int
    screen_height: int
    captured_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        dimensions = (
            self.pixel_width,
            self.pixel_height,
            self.screen_width,
            self.screen_height,
        )

        if any(value <= 0 for value in dimensions):
            raise ValueError(
                "screen frame dimensions must be positive"
            )

        if (
            self.captured_at.tzinfo is None
            or self.captured_at.utcoffset() is None
        ):
            raise ValueError(
                "captured_at must be timezone-aware"
            )

    @property
    def pixel_size(self) -> tuple[int, int]:
        """Return the screenshot image size in pixels."""

        return self.pixel_width, self.pixel_height

    @property
    def screen_size(self) -> tuple[int, int]:
        """Return the logical screen size used for input."""

        return self.screen_width, self.screen_height

    @property
    def scale_x(self) -> float:
        """Return the horizontal pixel-to-screen scale."""

        return self.pixel_width / self.screen_width

    @property
    def scale_y(self) -> float:
        """Return the vertical pixel-to-screen scale."""

        return self.pixel_height / self.screen_height
