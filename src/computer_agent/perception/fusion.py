"""Pure UI element fusion helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import replace
from numbers import Real

from computer_agent.perception.models import BoundingBox, UIElement


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_ui_text(text: str | None) -> str:
    """Return normalized UI text for cross-source matching."""

    if text is None:
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    )
    normalized = normalized.replace(
        "_",
        " ",
    )
    normalized = _WHITESPACE_RE.sub(
        " ",
        normalized,
    ).strip()

    return normalized.casefold()


def smaller_area_overlap_ratio(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    """Return intersection area divided by the smaller box area."""

    intersection = first.intersection(second)
    if intersection is None:
        return 0.0

    return intersection.area / min(
        first.area,
        second.area,
    )


class UIElementFusion:
    """Fuse Accessibility and OCR UI elements in one coordinate system."""

    def __init__(
        self,
        minimum_overlap_ratio: float = 0.5,
    ) -> None:
        self.minimum_overlap_ratio = self._validate_minimum_overlap_ratio(
            minimum_overlap_ratio
        )

    def fuse(
        self,
        accessibility_elements: Iterable[UIElement],
        ocr_elements: Iterable[UIElement],
    ) -> tuple[UIElement, ...]:
        """Return fused UI elements without coordinate conversion."""

        accessibility_elements = tuple(accessibility_elements)
        ocr_elements = tuple(ocr_elements)
        consumed_ocr_indexes: set[int] = set()
        fused_elements: list[UIElement] = []

        for accessibility_element in accessibility_elements:
            matches = self._matching_ocr_elements(
                accessibility_element,
                ocr_elements,
                consumed_ocr_indexes,
            )

            if matches:
                for ocr_index, _ in matches:
                    consumed_ocr_indexes.add(ocr_index)

                best_ocr_element = max(
                    matches,
                    key=lambda match: match[1].confidence,
                )[1]
                fused_elements.append(
                    self._fused_element(
                        accessibility_element,
                        best_ocr_element,
                    )
                )
            else:
                fused_elements.append(
                    self._with_default_source(
                        accessibility_element,
                        "accessibility",
                    )
                )

        ocr_only_elements = self._deduplicated_ocr_elements(
            (
                (ocr_index, ocr_element)
                for ocr_index, ocr_element in enumerate(ocr_elements)
                if ocr_index not in consumed_ocr_indexes
            )
        )

        return (
            *fused_elements,
            *ocr_only_elements,
        )

    def _matching_ocr_elements(
        self,
        accessibility_element: UIElement,
        ocr_elements: tuple[UIElement, ...],
        consumed_ocr_indexes: set[int],
    ) -> list[tuple[int, UIElement]]:
        normalized_text = normalize_ui_text(accessibility_element.text)
        if not normalized_text:
            return []

        matches = []
        for ocr_index, ocr_element in enumerate(ocr_elements):
            if ocr_index in consumed_ocr_indexes:
                continue

            if normalize_ui_text(ocr_element.text) != normalized_text:
                continue

            if self._boxes_overlap_sufficiently(
                accessibility_element.bounding_box,
                ocr_element.bounding_box,
            ):
                matches.append(
                    (
                        ocr_index,
                        ocr_element,
                    )
                )

        return matches

    def _deduplicated_ocr_elements(
        self,
        indexed_elements: Iterable[tuple[int, UIElement]],
    ) -> tuple[UIElement, ...]:
        groups: list[list[tuple[int, UIElement]]] = []

        for ocr_index, ocr_element in indexed_elements:
            matching_groups = self._find_ocr_duplicate_groups(
                ocr_element,
                groups,
            )
            if not matching_groups:
                groups.append(
                    [
                        (
                            ocr_index,
                            ocr_element,
                        )
                    ]
                )
            else:
                primary_group = matching_groups[0]
                primary_group.append(
                    (
                        ocr_index,
                        ocr_element,
                    )
                )
                for duplicate_group in matching_groups[1:]:
                    primary_group.extend(duplicate_group)
                    groups.remove(duplicate_group)

        retained_elements = []
        for group in groups:
            retained_elements.append(
                max(
                    group,
                    key=lambda indexed_element: indexed_element[
                        1
                    ].confidence,
                )
            )

        return tuple(
            self._with_default_source(
                ocr_element,
                "ocr",
            )
            for _, ocr_element in sorted(
                retained_elements,
                key=lambda indexed_element: indexed_element[0],
            )
        )

    def _find_ocr_duplicate_groups(
        self,
        ocr_element: UIElement,
        groups: list[list[tuple[int, UIElement]]],
    ) -> list[list[tuple[int, UIElement]]]:
        normalized_text = normalize_ui_text(ocr_element.text)
        if not normalized_text:
            return []

        matching_groups = []
        for group in groups:
            group_text = normalize_ui_text(
                group[0][1].text
            )
            if group_text != normalized_text:
                continue

            if any(
                self._boxes_overlap_sufficiently(
                    ocr_element.bounding_box,
                    grouped_element.bounding_box,
                )
                for _, grouped_element in group
            ):
                matching_groups.append(group)

        return matching_groups

    def _boxes_overlap_sufficiently(
        self,
        first: BoundingBox,
        second: BoundingBox,
    ) -> bool:
        overlap_ratio = smaller_area_overlap_ratio(
            first,
            second,
        )
        return (
            overlap_ratio > 0.0
            and overlap_ratio >= self.minimum_overlap_ratio
        )

    @staticmethod
    def _fused_element(
        accessibility_element: UIElement,
        ocr_element: UIElement,
    ) -> UIElement:
        return UIElement(
            element_type=accessibility_element.element_type,
            bounding_box=accessibility_element.bounding_box,
            confidence=max(
                accessibility_element.confidence,
                ocr_element.confidence,
            ),
            text=accessibility_element.text,
            identifier=accessibility_element.identifier,
            value=accessibility_element.value,
            enabled=accessibility_element.enabled,
            focused=accessibility_element.focused,
            selected=accessibility_element.selected,
            source="hybrid",
        )

    @staticmethod
    def _with_default_source(
        element: UIElement,
        source: str,
    ) -> UIElement:
        if element.source is not None:
            return element

        return replace(
            element,
            source=source,
        )

    @staticmethod
    def _validate_minimum_overlap_ratio(
        minimum_overlap_ratio: object,
    ) -> float:
        if (
            isinstance(minimum_overlap_ratio, bool)
            or not isinstance(minimum_overlap_ratio, Real)
        ):
            raise ValueError(
                "minimum_overlap_ratio must be numeric"
            )

        if not math.isfinite(minimum_overlap_ratio):
            raise ValueError(
                "minimum_overlap_ratio must be finite"
            )

        if not 0.0 <= minimum_overlap_ratio <= 1.0:
            raise ValueError(
                "minimum_overlap_ratio must be between 0.0 and 1.0"
            )

        return float(minimum_overlap_ratio)
