"""Screen perception components."""

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

__all__ = [
    "BoundingBox",
    "ImagePreprocessor",
    "ScreenCapture",
    "ScreenCoordinateMapper",
    "ScreenFrame",
    "TesseractOCR",
    "UIElement",
]
