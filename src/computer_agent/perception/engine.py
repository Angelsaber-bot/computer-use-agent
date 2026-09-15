"""Reusable hybrid perception observation engine."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Protocol

from PIL import Image

from computer_agent.perception.coordinates import ScreenCoordinateMapper
from computer_agent.perception.accessibility import AccessibilitySnapshot
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


class _ConsolidatedAccessibilityReader(_AccessibilityReader, Protocol):
    def read_frontmost_snapshot(self) -> AccessibilitySnapshot:
        """Return current consolidated Accessibility state."""


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
    timings: dict[str, float] = field(default_factory=dict)
    accessibility_snapshot: AccessibilitySnapshot | None = None

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
        if not isinstance(self.timings, dict):
            raise ValueError("timings must be a dict")
        timings = {}
        for key, value in self.timings.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("timing keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("timing values must be numeric")
            if value < 0:
                raise ValueError("timing values must be non-negative")
            timings[key] = float(value)
        object.__setattr__(self, "timings", timings)
        if self.accessibility_snapshot is not None and not isinstance(
            self.accessibility_snapshot,
            AccessibilitySnapshot,
        ):
            raise ValueError(
                "accessibility_snapshot must be an AccessibilitySnapshot "
                "or None"
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
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.screen_capture = screen_capture
        self.accessibility_reader = accessibility_reader
        self.ocr = ocr
        self.fusion = fusion
        self.capture_path = Path(capture_path)
        self.clock = clock

    def observe(
        self,
        *,
        include_ocr: bool = True,
    ) -> PerceptionSnapshot:
        """Capture and return a fresh perception snapshot."""
        if type(include_ocr) is not bool:
            raise ValueError("include_ocr must be a boolean")

        total_started = self.clock()
        timings: dict[str, float] = {}

        capture_started = self.clock()
        frame = self.screen_capture.capture(self.capture_path)
        timings["screen_capture"] = self.clock() - capture_started

        load_started = self.clock()
        image = self._load_rgb_image(frame)
        timings["image_load"] = self.clock() - load_started

        if image.size != frame.pixel_size:
            raise RuntimeError(
                "captured image size mismatch: "
                f"frame pixel size {frame.pixel_size}, "
                f"loaded image size {image.size}"
            )

        warnings: list[str] = []

        accessibility_elements, accessibility_snapshot = (
            self._read_accessibility(
                warnings,
                timings,
            )
        )

        if include_ocr:
            ocr_elements = self._read_ocr(
                image,
                frame,
                warnings,
                timings,
            )
        else:
            ocr_elements = ()
            timings["ocr"] = 0.0
            timings["ocr_executed"] = 0.0

        fusion_started = self.clock()
        fused_elements = tuple(
            self.fusion.fuse(
                accessibility_elements,
                ocr_elements,
            )
        )
        timings["fusion"] = self.clock() - fusion_started
        timings["perception_total"] = self.clock() - total_started

        return PerceptionSnapshot(
            frame=frame,
            image=image,
            accessibility_elements=accessibility_elements,
            ocr_elements=ocr_elements,
            fused_elements=fused_elements,
            warnings=tuple(warnings),
            timings=timings,
            accessibility_snapshot=accessibility_snapshot,
        )

    def with_ocr(
        self,
        snapshot: PerceptionSnapshot,
    ) -> PerceptionSnapshot:
        """Upgrade one same-capture snapshot by running OCR once."""
        if not isinstance(snapshot, PerceptionSnapshot):
            raise ValueError("snapshot must be a PerceptionSnapshot")

        if snapshot.timings.get("ocr_executed") == 1.0:
            return snapshot

        upgrade_started = self.clock()
        warnings = list(snapshot.warnings)
        timings = dict(snapshot.timings)
        ocr_elements = self._read_ocr(
            snapshot.image,
            snapshot.frame,
            warnings,
            timings,
        )

        fusion_started = self.clock()
        fused_elements = tuple(
            self.fusion.fuse(
                snapshot.accessibility_elements,
                ocr_elements,
            )
        )
        fusion_duration = self.clock() - fusion_started
        timings["fusion"] = (
            timings.get("fusion", 0.0) + fusion_duration
        )
        timings["perception_total"] = timings.get(
            "perception_total",
            0.0,
        ) + timings["ocr"] + fusion_duration
        timings["ocr_upgrade_total"] = self.clock() - upgrade_started

        return PerceptionSnapshot(
            frame=snapshot.frame,
            image=snapshot.image,
            accessibility_elements=snapshot.accessibility_elements,
            ocr_elements=ocr_elements,
            fused_elements=fused_elements,
            warnings=tuple(warnings),
            timings=timings,
            accessibility_snapshot=snapshot.accessibility_snapshot,
        )

    def _read_accessibility(
        self,
        warnings: list[str],
        timings: dict[str, float],
    ) -> tuple[tuple[UIElement, ...], AccessibilitySnapshot | None]:
        accessibility_started = self.clock()
        try:
            snapshot_reader = getattr(
                self.accessibility_reader,
                "read_frontmost_snapshot",
                None,
            )
            if callable(snapshot_reader):
                accessibility_snapshot = snapshot_reader()
                accessibility_elements = tuple(
                    accessibility_snapshot.controls
                )
                timings["accessibility_tree_traversals"] = float(
                    accessibility_snapshot.tree_traversals
                )
                timings["accessibility_nodes"] = float(
                    accessibility_snapshot.traversed_nodes
                )
                timings["accessibility_truncated"] = (
                    1.0
                    if accessibility_snapshot.traversal_truncated
                    else 0.0
                )
                timings["accessibility_maximum_depth_reached"] = (
                    1.0
                    if accessibility_snapshot.maximum_depth_reached
                    else 0.0
                )
                timings["accessibility_maximum_nodes_reached"] = (
                    1.0
                    if accessibility_snapshot.maximum_nodes_reached
                    else 0.0
                )
                timings["accessibility_maximum_controls_reached"] = (
                    1.0
                    if accessibility_snapshot.maximum_controls_reached
                    else 0.0
                )
            else:
                accessibility_snapshot = None
                accessibility_elements = tuple(
                    self.accessibility_reader.read_frontmost_controls()
                )
                timings["accessibility_tree_traversals"] = 1.0
                timings["accessibility_nodes"] = 0.0
                timings["accessibility_truncated"] = 0.0
                timings["accessibility_maximum_depth_reached"] = 0.0
                timings["accessibility_maximum_nodes_reached"] = 0.0
                timings["accessibility_maximum_controls_reached"] = 0.0
        except Exception as error:
            accessibility_snapshot = None
            accessibility_elements = ()
            timings["accessibility_tree_traversals"] = 0.0
            timings["accessibility_nodes"] = 0.0
            timings["accessibility_truncated"] = 0.0
            timings["accessibility_maximum_depth_reached"] = 0.0
            timings["accessibility_maximum_nodes_reached"] = 0.0
            timings["accessibility_maximum_controls_reached"] = 0.0
            warnings.append(
                _source_warning(
                    "Accessibility observation failed:",
                    error,
                )
            )
        timings["accessibility_controls"] = (
            self.clock() - accessibility_started
        )
        return accessibility_elements, accessibility_snapshot

    def _read_ocr(
        self,
        image: Image.Image,
        frame: ScreenFrame,
        warnings: list[str],
        timings: dict[str, float],
    ) -> tuple[UIElement, ...]:
        ocr_started = self.clock()
        try:
            pixel_ocr_elements = tuple(
                self.ocr.recognize(image)
            )
            mapper = ScreenCoordinateMapper(frame)
            ocr_elements = tuple(
                mapper.pixel_element_to_logical(element)
                for element in pixel_ocr_elements
            )
            timings["ocr_executed"] = 1.0
        except Exception as error:
            ocr_elements = ()
            timings["ocr_executed"] = 1.0
            warnings.append(
                _source_warning(
                    "OCR observation failed:",
                    error,
                )
            )
        timings["ocr"] = self.clock() - ocr_started
        return ocr_elements

    @staticmethod
    def _load_rgb_image(frame: ScreenFrame) -> Image.Image:
        with Image.open(frame.image_path) as source:
            image = source.convert("RGB")
            image.load()

        return image


def _source_warning(prefix: str, error: Exception) -> str:
    return f"{prefix} {type(error).__name__}: {error}"
