"""Data models used by the screen perception system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from computer_agent.core.models import utc_now


def _validate_int_field(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer"
        )


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A rectangular region using exclusive right and bottom edges."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            _validate_int_field(
                name,
                getattr(self, name),
            )

        if self.x < 0:
            raise ValueError("x must be non-negative")

        if self.y < 0:
            raise ValueError("y must be non-negative")

        if self.width <= 0:
            raise ValueError("width must be positive")

        if self.height <= 0:
            raise ValueError("height must be positive")

    @property
    def left(self) -> int:
        """Return the inclusive left edge."""

        return self.x

    @property
    def top(self) -> int:
        """Return the inclusive top edge."""

        return self.y

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""

        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""

        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return the geometric center point."""

        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def area(self) -> int:
        """Return the box area."""

        return self.width * self.height

    def contains_point(self, x: int, y: int) -> bool:
        """Return whether a point is inside the box."""

        return (
            self.left <= x < self.right
            and self.top <= y < self.bottom
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Return whether this box overlaps another box."""

        if not isinstance(other, BoundingBox):
            raise ValueError(
                "other must be a BoundingBox"
            )

        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def intersection(
        self,
        other: BoundingBox,
    ) -> BoundingBox | None:
        """Return the overlapping box, or None when boxes do not overlap."""

        if not self.intersects(other):
            return None

        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )


@dataclass(frozen=True, slots=True)
class UIElement:
    """A perceived user-interface element."""

    element_type: str
    bounding_box: BoundingBox
    confidence: float = 1.0
    text: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.element_type, str)
            or not self.element_type.strip()
        ):
            raise ValueError(
                "element_type must be a non-empty string"
            )

        if not isinstance(self.bounding_box, BoundingBox):
            raise ValueError(
                "bounding_box must be a BoundingBox"
            )

        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
        ):
            raise ValueError(
                "confidence must be numeric"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0.0 and 1.0"
            )

        if self.text is not None and not isinstance(self.text, str):
            raise ValueError(
                "text must be a string or None"
            )

    @property
    def center(self) -> tuple[float, float]:
        """Return the element center point."""

        return self.bounding_box.center


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
