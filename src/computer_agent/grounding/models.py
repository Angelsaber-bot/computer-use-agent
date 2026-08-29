"""Data models for deterministic UI grounding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any

from computer_agent.perception.models import UIElement


class GroundingStatus(str, Enum):
    """Possible outcomes of one UI grounding attempt."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    UNSAFE = "unsafe"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Semantic description of a UI target."""

    text: str | None = None
    identifier: str | None = None
    element_types: tuple[str, ...] = ()
    minimum_confidence: float = 0.70
    reference_point: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.text is not None:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("text must be a non-empty string or None")

        if self.identifier is not None:
            if not isinstance(self.identifier, str) or not self.identifier.strip():
                raise ValueError(
                    "identifier must be a non-empty string or None"
                )

        if self.text is None and self.identifier is None:
            raise ValueError("TargetSpec requires text or identifier")

        if not isinstance(self.element_types, tuple):
            raise ValueError("element_types must be a tuple of strings")

        for element_type in self.element_types:
            if not isinstance(element_type, str) or not element_type.strip():
                raise ValueError(
                    "element_types must contain non-empty strings"
                )

        object.__setattr__(
            self,
            "minimum_confidence",
            _validate_confidence(self.minimum_confidence),
        )

        if self.reference_point is not None:
            object.__setattr__(
                self,
                "reference_point",
                _validate_reference_point(self.reference_point),
            )


@dataclass(frozen=True, slots=True)
class GroundingCandidate:
    """One semantic grounding candidate and its evaluation."""

    element: UIElement
    match_basis: str
    rejection_reasons: tuple[str, ...] = ()
    distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, UIElement):
            raise ValueError("element must be a UIElement")

        if not isinstance(self.match_basis, str) or not self.match_basis.strip():
            raise ValueError(
                "match_basis must be a non-empty string"
            )

        if not isinstance(self.rejection_reasons, tuple):
            raise ValueError(
                "rejection_reasons must be a tuple of strings"
            )

        for reason in self.rejection_reasons:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "rejection_reasons must contain non-empty strings"
                )

        if self.distance is not None:
            if isinstance(self.distance, bool) or not isinstance(
                self.distance,
                Real,
            ):
                raise ValueError("distance must be numeric or None")

            if not math.isfinite(self.distance) or self.distance < 0:
                raise ValueError(
                    "distance must be finite and non-negative"
                )

            object.__setattr__(
                self,
                "distance",
                float(self.distance),
            )

    @property
    def eligible(self) -> bool:
        """Return whether this candidate passed all hard filters."""

        return not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Explicit result of deterministic UI grounding."""

    status: GroundingStatus
    element: UIElement | None
    candidates: tuple[GroundingCandidate, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, GroundingStatus):
            raise ValueError(
                "status must be a GroundingStatus"
            )

        if self.element is not None and not isinstance(
            self.element,
            UIElement,
        ):
            raise ValueError(
                "element must be a UIElement or None"
            )

        if not isinstance(self.candidates, tuple):
            raise ValueError(
                "candidates must be a tuple of GroundingCandidate objects"
            )

        for candidate in self.candidates:
            if not isinstance(candidate, GroundingCandidate):
                raise ValueError(
                    "candidates must contain GroundingCandidate objects"
                )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(
                "reason must be a non-empty string"
            )

        if (
            self.status is GroundingStatus.RESOLVED
            and self.element is None
        ):
            raise ValueError(
                "RESOLVED results require an element"
            )

        if (
            self.status is not GroundingStatus.RESOLVED
            and self.element is not None
        ):
            raise ValueError(
                "non-RESOLVED results must not contain an actionable element"
            )


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "minimum_confidence must be numeric"
        )

    if not math.isfinite(value):
        raise ValueError(
            "minimum_confidence must be finite"
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "minimum_confidence must be between 0.0 and 1.0"
        )

    return float(value)


def _validate_reference_point(
    point: object,
) -> tuple[float, float]:
    if not isinstance(point, tuple) or len(point) != 2:
        raise ValueError(
            "reference_point must be a two-value tuple or None"
        )

    coordinates: list[float] = []

    for coordinate in point:
        if isinstance(coordinate, bool) or not isinstance(
            coordinate,
            Real,
        ):
            raise ValueError(
                "reference_point coordinates must be numeric"
            )

        if not math.isfinite(coordinate):
            raise ValueError(
                "reference_point coordinates must be finite"
            )

        coordinates.append(float(coordinate))

    return coordinates[0], coordinates[1]
