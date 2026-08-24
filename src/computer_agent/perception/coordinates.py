"""Coordinate mapping between screenshot pixels and logical screen space."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from computer_agent.perception.models import (
    BoundingBox,
    ScreenFrame,
    UIElement,
)


class ScreenCoordinateMapper:
    """Map perception results from pixel coordinates to logical coordinates."""

    def __init__(
        self,
        frame: ScreenFrame,
    ) -> None:
        if not isinstance(frame, ScreenFrame):
            raise ValueError(
                "frame must be a ScreenFrame"
            )

        self.frame = frame

    def pixel_point_to_logical(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Return a pixel point converted to logical screen coordinates."""

        pixel_x = self._validate_coordinate(
            "x",
            x,
        )
        pixel_y = self._validate_coordinate(
            "y",
            y,
        )

        if not 0 <= pixel_x < self.frame.pixel_width:
            raise ValueError(
                "x must be inside the pixel frame"
            )

        if not 0 <= pixel_y < self.frame.pixel_height:
            raise ValueError(
                "y must be inside the pixel frame"
            )

        return (
            pixel_x / self.frame.scale_x,
            pixel_y / self.frame.scale_y,
        )

    def pixel_box_to_logical(
        self,
        bounding_box: BoundingBox,
    ) -> BoundingBox:
        """Return a pixel box converted to a containing logical box."""

        self._validate_bounding_box(bounding_box)

        logical_left = math.floor(
            bounding_box.left / self.frame.scale_x
        )
        logical_top = math.floor(
            bounding_box.top / self.frame.scale_y
        )
        logical_right = math.ceil(
            bounding_box.right / self.frame.scale_x
        )
        logical_bottom = math.ceil(
            bounding_box.bottom / self.frame.scale_y
        )

        return BoundingBox(
            x=logical_left,
            y=logical_top,
            width=logical_right - logical_left,
            height=logical_bottom - logical_top,
        )

    def pixel_element_to_logical(
        self,
        element: UIElement,
    ) -> UIElement:
        """Return a UI element with its box mapped into logical coordinates."""

        if not isinstance(element, UIElement):
            raise ValueError(
                "element must be a UIElement"
            )

        return UIElement(
            element_type=element.element_type,
            bounding_box=self.pixel_box_to_logical(
                element.bounding_box
            ),
            confidence=element.confidence,
            text=element.text,
        )

    def _validate_bounding_box(
        self,
        bounding_box: BoundingBox,
    ) -> None:
        if not isinstance(bounding_box, BoundingBox):
            raise ValueError(
                "bounding_box must be a BoundingBox"
            )

        if (
            bounding_box.right > self.frame.pixel_width
            or bounding_box.bottom > self.frame.pixel_height
        ):
            raise ValueError(
                "bounding_box must be inside the pixel frame"
            )

    @staticmethod
    def _validate_coordinate(
        name: str,
        value: Any,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"{name} must be numeric"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return float(value)
