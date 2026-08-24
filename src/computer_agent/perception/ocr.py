"""OCR text recognition using Tesseract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from PIL import Image
import pytesseract

from computer_agent.perception.models import BoundingBox, UIElement


class TesseractOCR:
    """Recognize sparse screen text with Tesseract OCR."""

    def __init__(
        self,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.minimum_confidence = self._validate_minimum_confidence(
            minimum_confidence
        )

    @staticmethod
    def is_available() -> bool:
        """Return whether the Tesseract executable can be used."""

        try:
            pytesseract.get_tesseract_version()
        except Exception:
            return False

        return True

    def recognize(
        self,
        image: Image.Image,
    ) -> tuple[UIElement, ...]:
        """Return recognized word-level text elements."""

        self._validate_image(image)

        data = pytesseract.image_to_data(
            image.copy(),
            lang="eng",
            config="--psm 11",
            output_type=pytesseract.Output.DICT,
        )

        return self._elements_from_data(
            data,
            image.size,
        )

    def _elements_from_data(
        self,
        data: Mapping[str, Sequence[Any]],
        image_size: tuple[int, int],
    ) -> tuple[UIElement, ...]:
        elements = []
        columns = (
            data.get("text", ()),
            data.get("conf", ()),
            data.get("left", ()),
            data.get("top", ()),
            data.get("width", ()),
            data.get("height", ()),
        )

        for (
            raw_text,
            raw_confidence,
            raw_x,
            raw_y,
            raw_width,
            raw_height,
        ) in zip(*columns):
            text = "" if raw_text is None else str(raw_text).strip()
            if not text:
                continue

            confidence = self._normalize_confidence(raw_confidence)
            if (
                confidence is None
                or confidence < self.minimum_confidence
            ):
                continue

            bounding_box = self._build_bounding_box(
                raw_x,
                raw_y,
                raw_width,
                raw_height,
                image_size,
            )
            if bounding_box is None:
                continue

            elements.append(
                UIElement(
                    element_type="text",
                    bounding_box=bounding_box,
                    confidence=confidence,
                    text=text,
                )
            )

        return tuple(elements)

    @staticmethod
    def _build_bounding_box(
        raw_x: Any,
        raw_y: Any,
        raw_width: Any,
        raw_height: Any,
        image_size: tuple[int, int],
    ) -> BoundingBox | None:
        x = TesseractOCR._parse_int(raw_x)
        y = TesseractOCR._parse_int(raw_y)
        width = TesseractOCR._parse_int(raw_width)
        height = TesseractOCR._parse_int(raw_height)

        if None in (x, y, width, height):
            return None

        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return None

        if x + width > image_size[0] or y + height > image_size[1]:
            return None

        return BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        )

    @staticmethod
    def _normalize_confidence(value: Any) -> float | None:
        if isinstance(value, bool):
            return None

        try:
            raw_confidence = float(value)
        except (TypeError, ValueError):
            return None

        if (
            not math.isfinite(raw_confidence)
            or not 0.0 <= raw_confidence <= 100.0
        ):
            return None

        return raw_confidence / 100.0

    @staticmethod
    def _parse_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None

            return int(value)

        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _validate_image(image: Any) -> None:
        if not isinstance(image, Image.Image):
            raise ValueError(
                "image must be a PIL Image"
            )

    @staticmethod
    def _validate_minimum_confidence(
        minimum_confidence: Any,
    ) -> float:
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, Real)
        ):
            raise ValueError(
                "minimum_confidence must be numeric"
            )

        if not math.isfinite(minimum_confidence):
            raise ValueError(
                "minimum_confidence must be finite"
            )

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0.0 and 1.0"
            )

        return float(minimum_confidence)
