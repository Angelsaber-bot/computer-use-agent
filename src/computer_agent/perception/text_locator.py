"""Utilities for locating OCR text elements."""

from collections.abc import Iterable

from computer_agent.perception.models import UIElement


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

        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError(
                "target_text must be a non-empty string"
            )

        target = target_text.strip()

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
