"""Reusable hybrid perception observation engine."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from computer_agent.perception.coordinates import ScreenCoordinateMapper
from computer_agent.perception.models import (
    ScreenFrame,
    UIElement,
)


class _ScreenCapture(Protocol):
    def capture(self, output_path: str | Path) -> ScreenFrame:
        """Capture the screen and return a frame."""


class _AccessibilityReader(Protocol):
    def read_frontmost_controls(self) -> Iterable[UIElement]:
        """Return current Accessibility elements."""


class _OCRRecognizer(Protocol):
    def recognize(self, image: Image.Image) -> Iterable[UIElement]:
        """Return OCR elements in pixel coordinates."""


class _FusionComponent(Protocol):
    def fuse(
        self,
        accessibility_elements: Iterable[UIElement],
        ocr_elements: Iterable[UIElement],
    ) -> Iterable[UIElement]:
        """Return fused UI elements."""


@dataclass(frozen=True, slots=True)
class PerceptionSnapshot:
    """A single hybrid perception observation."""

    frame: ScreenFrame
    image: Image.Image
    accessibility_elements: tuple[UIElement, ...]
    ocr_elements: tuple[UIElement, ...]
    fused_elements: tuple[UIElement, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ScreenFrame):
            raise ValueError("frame must be a ScreenFrame")

        if not isinstance(self.image, Image.Image):
            raise ValueError("image must be a PIL Image")

        if self.image.mode != "RGB":
            raise ValueError("image must be RGB")

        object.__setattr__(
            self,
            "accessibility_elements",
            tuple(self.accessibility_elements),
        )
        object.__setattr__(
            self,
            "ocr_elements",
            tuple(self.ocr_elements),
        )
        object.__setattr__(
            self,
            "fused_elements",
            tuple(self.fused_elements),
        )
        object.__setattr__(
            self,
            "warnings",
            tuple(self.warnings),
        )

    @property
    def source_counts(self) -> dict[str, int]:
        """Return current source element counts."""

        return {
            "accessibility": len(self.accessibility_elements),
            "ocr": len(self.ocr_elements),
            "fused": len(self.fused_elements),
        }


class PerceptionEngine:
    """Observe the current screen through Accessibility, OCR, and fusion."""

    def __init__(
        self,
        *,
        screen_capture: _ScreenCapture,
        accessibility_reader: _AccessibilityReader,
        ocr: _OCRRecognizer,
        fusion: _FusionComponent,
        capture_path: str | Path,
    ) -> None:
        self.screen_capture = screen_capture
        self.accessibility_reader = accessibility_reader
        self.ocr = ocr
        self.fusion = fusion
        self.capture_path = Path(capture_path)

    def observe(self) -> PerceptionSnapshot:
        """Capture and return a fresh perception snapshot."""

        frame = self.screen_capture.capture(self.capture_path)
        image = self._load_rgb_image(frame)

        if image.size != frame.pixel_size:
            raise RuntimeError(
                "captured image size mismatch: "
                f"frame pixel size {frame.pixel_size}, "
                f"loaded image size {image.size}"
            )

        warnings: list[str] = []

        try:
            accessibility_elements = tuple(
                self.accessibility_reader.read_frontmost_controls()
            )
        except Exception as error:
            accessibility_elements = ()
            warnings.append(
                _source_warning(
                    "Accessibility observation failed:",
                    error,
                )
            )

        try:
            pixel_ocr_elements = tuple(
                self.ocr.recognize(image)
            )
            mapper = ScreenCoordinateMapper(frame)
            ocr_elements = tuple(
                mapper.pixel_element_to_logical(element)
                for element in pixel_ocr_elements
            )
        except Exception as error:
            ocr_elements = ()
            warnings.append(
                _source_warning(
                    "OCR observation failed:",
                    error,
                )
            )

        fused_elements = tuple(
            self.fusion.fuse(
                accessibility_elements,
                ocr_elements,
            )
        )

        return PerceptionSnapshot(
            frame=frame,
            image=image,
            accessibility_elements=accessibility_elements,
            ocr_elements=ocr_elements,
            fused_elements=fused_elements,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _load_rgb_image(frame: ScreenFrame) -> Image.Image:
        with Image.open(frame.image_path) as source:
            image = source.convert("RGB")
            image.load()

        return image


def _source_warning(prefix: str, error: Exception) -> str:
    return f"{prefix} {type(error).__name__}: {error}"
