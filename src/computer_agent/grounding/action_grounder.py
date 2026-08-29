"""Deterministic conversion from UI grounding to click actions."""

from __future__ import annotations

from computer_agent.core.models import Action
from computer_agent.grounding.action_models import (
    ActionGroundingResult,
    ActionGroundingStatus,
)
from computer_agent.grounding.models import GroundingResult, GroundingStatus
from computer_agent.perception.models import BoundingBox


_CLICK_REASON = "Click the UI element resolved by deterministic grounding."


class ActionGrounder:
    """Construct click actions from resolved UI grounding results."""

    def __init__(self, safe_edge_margin: int = 1) -> None:
        if (
            isinstance(safe_edge_margin, bool)
            or not isinstance(safe_edge_margin, int)
        ):
            raise ValueError(
                "safe_edge_margin must be a non-boolean integer"
            )

        if safe_edge_margin < 0:
            raise ValueError(
                "safe_edge_margin must be greater than or equal to zero"
            )

        self._safe_edge_margin = safe_edge_margin

    @property
    def safe_edge_margin(self) -> int:
        """Return the configured unsafe screen-edge margin."""

        return self._safe_edge_margin

    def ground_click(
        self,
        grounding_result: GroundingResult,
        screen_size: tuple[int, int],
    ) -> ActionGroundingResult:
        """Return a click action for a resolved element, or a blocked result."""

        if not isinstance(grounding_result, GroundingResult):
            raise ValueError(
                "grounding_result must be a GroundingResult"
            )

        screen_width, screen_height = _validate_screen_size(screen_size)

        if grounding_result.status is not GroundingStatus.RESOLVED:
            return ActionGroundingResult(
                status=ActionGroundingStatus.BLOCKED,
                action=None,
                reason=(
                    "grounding result was "
                    f"{grounding_result.status.value}: "
                    f"{grounding_result.reason}"
                ),
            )

        box = grounding_result.element.bounding_box
        x, y = _floor_center(box)

        if _screen_has_no_usable_interior(
            screen_width,
            screen_height,
            self._safe_edge_margin,
        ):
            return ActionGroundingResult(
                status=ActionGroundingStatus.BLOCKED,
                action=None,
                reason=(
                    "safe edge margin leaves no usable screen interior: "
                    f"screen_size=({screen_width}, {screen_height}), "
                    f"margin={self._safe_edge_margin}"
                ),
            )

        if not _point_is_inside_safe_bounds(
            x,
            y,
            screen_width,
            screen_height,
            self._safe_edge_margin,
        ):
            return ActionGroundingResult(
                status=ActionGroundingStatus.BLOCKED,
                action=None,
                reason=(
                    f"click point ({x}, {y}) violates safe screen bounds: "
                    f"screen_size=({screen_width}, {screen_height}), "
                    f"margin={self._safe_edge_margin}"
                ),
            )

        return ActionGroundingResult(
            status=ActionGroundingStatus.READY,
            action=Action(
                tool_name="click_mouse",
                arguments={
                    "x": x,
                    "y": y,
                },
                reason=_CLICK_REASON,
            ),
            reason="click action ready",
        )


def _validate_screen_size(screen_size: object) -> tuple[int, int]:
    if not isinstance(screen_size, tuple) or len(screen_size) != 2:
        raise ValueError(
            "screen_size must be a two-item tuple of positive integers"
        )

    width, height = screen_size
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                "screen_size must be a two-item tuple of positive integers"
            )

        if value <= 0:
            raise ValueError(
                "screen_size must be a two-item tuple of positive integers"
            )

    return width, height


def _floor_center(box: BoundingBox) -> tuple[int, int]:
    return (
        box.x + box.width // 2,
        box.y + box.height // 2,
    )


def _screen_has_no_usable_interior(
    screen_width: int,
    screen_height: int,
    margin: int,
) -> bool:
    return screen_width <= margin * 2 or screen_height <= margin * 2


def _point_is_inside_safe_bounds(
    x: int,
    y: int,
    screen_width: int,
    screen_height: int,
    margin: int,
) -> bool:
    return (
        margin <= x < screen_width - margin
        and margin <= y < screen_height - margin
    )
