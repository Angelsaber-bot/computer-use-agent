"""Screen perception components."""

from computer_agent.perception.models import (
    BoundingBox,
    ScreenFrame,
    UIElement,
)
from computer_agent.perception.screen_capture import (
    ScreenCapture,
)

__all__ = [
    "BoundingBox",
    "ScreenCapture",
    "ScreenFrame",
    "UIElement",
]
