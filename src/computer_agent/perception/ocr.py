"""OCR text recognition using Tesseract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Integral, Real
from typing import Any

from PIL import Image
import pytesseract

from computer_agent.perception.models import BoundingBox, UIElement


class TesseractOCR:
    """Recognize sparse screen text with Tesseract OCR."""

    def __init__(
        self,
        minimum_confidence: float = 0.0,
        page_segmentation_mode: int = 11,
        group_words_by_line: bool = False,
    ) -> None:
        self.minimum_confidence = self._validate_minimum_confidence(
            minimum_confidence
        )
        self.page_segmentation_mode = self._validate_page_segmentation_mode(
            page_segmentation_mode
        )
        self.group_words_by_line = self._validate_group_words_by_line(
            group_words_by_line
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
        """Return recognized OCR text elements."""

        self._validate_image(image)

        data = pytesseract.image_to_data(
            image.copy(),
            lang="eng",
            config=f"--psm {self.page_segmentation_mode}",
            output_type=pytesseract.Output.DICT,
        )

        if self.group_words_by_line:
            return self._line_elements_from_data(
                data,
                image.size,
            )

        return self._elements_from_data(
            data,
            image.size,
        )

    def recognize_region(
        self,
        image: Image.Image,
        region: BoundingBox,
    ) -> tuple[UIElement, ...]:
        """Return OCR elements from a pixel region in full-image coordinates."""

        self._validate_image(image)
        self._validate_region(
            region,
            image.size,
        )

        cropped = image.crop(
            (
                region.left,
                region.top,
                region.right,
                region.bottom,
            )
        )
        elements = self.recognize(cropped)

        return tuple(
            replace(
                element,
                bounding_box=BoundingBox(
                    x=element.bounding_box.x + region.x,
                    y=element.bounding_box.y + region.y,
                    width=element.bounding_box.width,
                    height=element.bounding_box.height,
                ),
            )
            for element in elements
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

    def _line_elements_from_data(
        self,
        data: Mapping[str, Sequence[Any]],
        image_size: tuple[int, int],
    ) -> tuple[UIElement, ...]:
        lines: dict[
            tuple[int, int, int, int],
            list[tuple[str, float, BoundingBox]],
        ] = {}
        columns = (
            data.get("page_num", ()),
            data.get("block_num", ()),
            data.get("par_num", ()),
            data.get("line_num", ()),
            data.get("text", ()),
            data.get("conf", ()),
            data.get("left", ()),
            data.get("top", ()),
            data.get("width", ()),
            data.get("height", ()),
        )

        for (
            raw_page_num,
            raw_block_num,
            raw_par_num,
            raw_line_num,
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

            line_key_parts = (
                self._parse_int(raw_page_num),
                self._parse_int(raw_block_num),
                self._parse_int(raw_par_num),
                self._parse_int(raw_line_num),
            )
            if None in line_key_parts:
                continue

            confidence = self._normalize_confidence(raw_confidence)
            if confidence is None:
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

            line_key = (
                line_key_parts[0],
                line_key_parts[1],
                line_key_parts[2],
                line_key_parts[3],
            )
            lines.setdefault(
                line_key,
                [],
            ).append(
                (
                    text,
                    confidence,
                    bounding_box,
                )
            )

        elements = []
        for words in lines.values():
            confidence = min(
                word_confidence for _, word_confidence, _ in words
            )
            if confidence < self.minimum_confidence:
                continue

            bounding_box = self._union_bounding_boxes(
                [box for _, _, box in words]
            )
            elements.append(
                UIElement(
                    element_type="text",
                    bounding_box=bounding_box,
                    confidence=confidence,
                    text=" ".join(text for text, _, _ in words),
                )
            )

        return tuple(elements)

    @staticmethod
    def _union_bounding_boxes(
        boxes: Sequence[BoundingBox],
    ) -> BoundingBox:
        left = min(box.left for box in boxes)
        top = min(box.top for box in boxes)
        right = max(box.right for box in boxes)
        bottom = max(box.bottom for box in boxes)

        return BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        )

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
    def _validate_region(
        region: Any,
        image_size: tuple[int, int],
    ) -> None:
        if not isinstance(region, BoundingBox):
            raise ValueError(
                "region must be a BoundingBox"
            )

        image_width, image_height = image_size
        if region.right > image_width or region.bottom > image_height:
            raise ValueError(
                "region must be inside the image"
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

    @staticmethod
    def _validate_page_segmentation_mode(
        page_segmentation_mode: Any,
    ) -> int:
        if (
            isinstance(page_segmentation_mode, bool)
            or not isinstance(page_segmentation_mode, Integral)
        ):
            raise ValueError(
                "page_segmentation_mode must be an integer"
            )

        if not 0 <= page_segmentation_mode <= 13:
            raise ValueError(
                "page_segmentation_mode must be between 0 and 13"
            )

        return int(page_segmentation_mode)

    @staticmethod
    def _validate_group_words_by_line(
        group_words_by_line: Any,
    ) -> bool:
        if not isinstance(group_words_by_line, bool):
            raise ValueError(
                "group_words_by_line must be a bool"
            )

        return group_words_by_line
