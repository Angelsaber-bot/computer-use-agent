"""Screen perception components."""

from computer_agent.perception.accessibility import (
    MacOSAccessibility,
)
from computer_agent.perception.coordinates import (
    ScreenCoordinateMapper,
)
from computer_agent.perception.models import (
    BoundingBox,
    ScreenFrame,
    UIElement,
)
from computer_agent.perception.ocr import (
    TesseractOCR,
)
from computer_agent.perception.preprocessing import (
    ImagePreprocessor,
)
from computer_agent.perception.screen_capture import (
    ScreenCapture,
)
from computer_agent.perception.text_locator import (
    TextTargetLocator,
)

__all__ = [
    "BoundingBox",
    "ImagePreprocessor",
    "MacOSAccessibility",
    "ScreenCapture",
    "ScreenCoordinateMapper",
    "ScreenFrame",
    "TesseractOCR",
    "UIElement",
    "TextTargetLocator",
]
