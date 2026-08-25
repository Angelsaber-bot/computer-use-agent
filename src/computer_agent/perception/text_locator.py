"""Utilities for locating OCR text elements."""

from collections.abc import Iterable
import math

from computer_agent.perception.models import BoundingBox, UIElement


class TextTargetLocator:
    """Locate OCR UI elements by recognized text."""

    @staticmethod
    def find_all(
        elements: Iterable[UIElement],
        target_text: str,
        *,
        case_sensitive: bool = False,
        partial_match: bool = False,
    ) -> tuple[UIElement, ...]:
        """Return all elements whose text matches the target."""

        target = TextTargetLocator._validate_target_text(target_text)

        if not case_sensitive:
            target = target.casefold()

        matches = []

        for element in elements:
            if element.text is None:
                continue

            text = element.text.strip()

            if not case_sensitive:
                text = text.casefold()

            if (
                text == target
                or partial_match
                and target in text
            ):
                matches.append(element)

        return tuple(matches)

    @staticmethod
    def extract_target(
        element: UIElement,
        target_text: str,
        *,
        case_sensitive: bool = False,
    ) -> UIElement | None:
        """Return an estimated UI element for a matched target substring.

        Substring boxes are character-proportion estimates within the source
        OCR box and are most accurate for monospaced text.
        """

        if not isinstance(element, UIElement):
            raise ValueError(
                "element must be a UIElement"
            )

        target = TextTargetLocator._validate_target_text(target_text)

        if element.text is None:
            return None

        source_text = element.text.strip()
        if not source_text:
            return None

        match_source = source_text
        match_target = target

        if not case_sensitive:
            match_source = match_source.casefold()
            match_target = match_target.casefold()

        start = match_source.find(match_target)
        if start < 0:
            return None

        if match_source == match_target:
            return UIElement(
                element_type=element.element_type,
                bounding_box=element.bounding_box,
                confidence=element.confidence,
                text=source_text,
            )

        end = start + len(match_target)
        box = element.bounding_box
        character_width = box.width / len(source_text)
        left = box.left + math.floor(
            start * character_width
        )
        right = box.left + math.ceil(
            end * character_width
        )
        left = max(
            box.left,
            min(
                left,
                box.right - 1,
            ),
        )
        right = max(
            left + 1,
            min(
                right,
                box.right,
            ),
        )

        return UIElement(
            element_type=element.element_type,
            bounding_box=BoundingBox(
                x=left,
                y=box.y,
                width=right - left,
                height=box.height,
            ),
            confidence=element.confidence,
            text=source_text[start:end],
        )

    @classmethod
    def find_first(
        cls,
        elements: Iterable[UIElement],
        target_text: str,
        *,
        case_sensitive: bool = False,
        partial_match: bool = False,
    ) -> UIElement | None:
        """Return the first matching element."""

        matches = cls.find_all(
            elements,
            target_text,
            case_sensitive=case_sensitive,
            partial_match=partial_match,
        )

        if not matches:
            return None

        return matches[0]

    @staticmethod
    def _validate_target_text(
        target_text: str,
    ) -> str:
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError(
                "target_text must be a non-empty string"
            )

        return target_text.strip()
