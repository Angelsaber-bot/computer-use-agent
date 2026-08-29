"""Deterministic UI element grounding."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real

from computer_agent.grounding.models import (
    GroundingCandidate,
    GroundingResult,
    GroundingStatus,
    TargetSpec,
)
from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.models import UIElement


_SOURCE_PRIORITY = {
    "hybrid": 0,
    "accessibility": 1,
    "ocr": 2,
}


class UIGrounder:
    """Resolve semantic target specifications to perceived UI elements."""

    def ground(
        self,
        target_spec: TargetSpec,
        elements: Iterable[UIElement],
    ) -> GroundingResult:
        """Return an explicit deterministic grounding result."""

        if not isinstance(target_spec, TargetSpec):
            raise ValueError("target_spec must be a TargetSpec")

        element_tuple = tuple(elements)
        for element in element_tuple:
            if not isinstance(element, UIElement):
                raise ValueError("elements must contain UIElement objects")

        tier_name, tier_elements = self._select_semantic_tier(
            target_spec,
            element_tuple,
        )

        if tier_name is None:
            return GroundingResult(
                status=GroundingStatus.NOT_FOUND,
                element=None,
                candidates=(),
                reason="no exact identifier or normalized text match",
            )

        candidates = tuple(
            sorted(
                (
                    self._candidate_for(
                        element,
                        tier_name,
                        target_spec,
                    )
                    for element in tier_elements
                ),
                key=_candidate_sort_key,
            )
        )
        eligible = tuple(
            candidate
            for candidate in candidates
            if candidate.eligible
        )

        if not eligible:
            return GroundingResult(
                status=GroundingStatus.UNSAFE,
                element=None,
                candidates=candidates,
                reason=f"{tier_name} candidates were unsafe",
            )

        best = self._best_candidates(eligible)
        if len(best) == 1:
            return GroundingResult(
                status=GroundingStatus.RESOLVED,
                element=best[0].element,
                candidates=candidates,
                reason=f"resolved by {tier_name}",
            )

        return GroundingResult(
            status=GroundingStatus.AMBIGUOUS,
            element=None,
            candidates=candidates,
            reason=f"ambiguous {tier_name} candidates",
        )

    def _select_semantic_tier(
        self,
        target_spec: TargetSpec,
        elements: tuple[UIElement, ...],
    ) -> tuple[str | None, tuple[UIElement, ...]]:
        if target_spec.identifier is not None:
            identifier_matches = tuple(
                element
                for element in elements
                if element.identifier is not None
                and element.identifier.strip() == target_spec.identifier.strip()
            )

            if identifier_matches:
                return "identifier", identifier_matches

        if target_spec.text is not None:
            target_text = normalize_ui_text(target_spec.text)
            text_matches = tuple(
                element
                for element in elements
                if normalize_ui_text(element.text) == target_text
            )

            if text_matches:
                return "text", text_matches

        return None, ()

    def _candidate_for(
        self,
        element: UIElement,
        match_basis: str,
        target_spec: TargetSpec,
    ) -> GroundingCandidate:
        rejection_reasons = []
        box_usable = _box_is_usable(element)

        if not box_usable:
            rejection_reasons.append("invalid_bounding_box")

        if element.enabled is False:
            rejection_reasons.append("disabled")

        if not _element_type_is_compatible(element, target_spec):
            rejection_reasons.append("incompatible_element_type")

        if not _confidence_is_usable(element.confidence):
            rejection_reasons.append("invalid_confidence")
        elif element.confidence < target_spec.minimum_confidence:
            rejection_reasons.append("low_confidence")

        if _has_identifier_text_conflict(element, target_spec):
            rejection_reasons.append("identifier_text_conflict")

        return GroundingCandidate(
            element=element,
            match_basis=match_basis,
            rejection_reasons=tuple(rejection_reasons),
            distance=_distance_from_reference(element, target_spec)
            if box_usable
            else None,
        )

    def _best_candidates(
        self,
        candidates: tuple[GroundingCandidate, ...],
    ) -> tuple[GroundingCandidate, ...]:
        source_priority = min(
            _source_priority(candidate.element.source)
            for candidate in candidates
        )
        candidates = tuple(
            candidate
            for candidate in candidates
            if _source_priority(candidate.element.source) == source_priority
        )

        distances = tuple(
            candidate.distance
            for candidate in candidates
            if candidate.distance is not None
        )
        if distances:
            nearest = min(distances)
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.distance == nearest
            )

        confidence = max(
            candidate.element.confidence
            for candidate in candidates
        )
        return tuple(
            candidate
            for candidate in candidates
            if candidate.element.confidence == confidence
        )


def _element_type_is_compatible(
    element: UIElement,
    target_spec: TargetSpec,
) -> bool:
    if not target_spec.element_types:
        return True

    element_type = normalize_ui_text(element.element_type)
    return element_type in {
        normalize_ui_text(expected_type)
        for expected_type in target_spec.element_types
    }


def _has_identifier_text_conflict(
    element: UIElement,
    target_spec: TargetSpec,
) -> bool:
    if target_spec.identifier is None or target_spec.text is None:
        return False

    if element.identifier is None:
        return False

    if element.identifier.strip() != target_spec.identifier.strip():
        return False

    return normalize_ui_text(element.text) != normalize_ui_text(target_spec.text)


def _box_is_usable(element: UIElement) -> bool:
    box = getattr(element, "bounding_box", None)
    if box is None:
        return False

    try:
        left = box.left
        top = box.top
        right = box.right
        bottom = box.bottom
        width = box.width
        height = box.height
    except Exception:
        return False

    values = (
        left,
        top,
        right,
        bottom,
        width,
        height,
    )
    if any(not _finite_number(value) for value in values):
        return False

    return (
        width > 0
        and height > 0
        and 0 <= left < right
        and 0 <= top < bottom
    )


def _confidence_is_usable(value: object) -> bool:
    return _finite_number(value) and 0.0 <= value <= 1.0


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
    )


def _distance_from_reference(
    element: UIElement,
    target_spec: TargetSpec,
) -> float | None:
    if target_spec.reference_point is None:
        return None

    center_x, center_y = element.center
    return math.hypot(
        center_x - target_spec.reference_point[0],
        center_y - target_spec.reference_point[1],
    )


def _source_priority(source: str | None) -> int:
    source_name = _normalized_source(source)
    if source_name is None:
        return 4

    return _SOURCE_PRIORITY.get(source_name, 3)


def _candidate_sort_key(
    candidate: GroundingCandidate,
) -> tuple[object, ...]:
    element = candidate.element
    return (
        0 if candidate.match_basis == "identifier" else 1,
        _source_priority(element.source),
        math.inf if candidate.distance is None else candidate.distance,
        -element.confidence
        if _confidence_is_usable(element.confidence)
        else math.inf,
        *_ui_element_sort_key(element),
        candidate.rejection_reasons,
    )


def _ui_element_sort_key(element: UIElement) -> tuple[object, ...]:
    return (
        normalize_ui_text(element.element_type),
        element.element_type,
        *_box_sort_key(element),
        _number_sort_key(element.confidence),
        *_optional_string_sort_key(element.text),
        *_optional_string_sort_key(element.identifier),
        *_optional_scalar_sort_key(element.value),
        *_optional_bool_sort_key(element.enabled),
        *_optional_bool_sort_key(element.focused),
        *_optional_bool_sort_key(element.selected),
        *_optional_string_sort_key(element.source),
    )


def _box_sort_key(element: UIElement) -> tuple[float, float, float, float]:
    box = getattr(element, "bounding_box", None)
    if box is None:
        return (math.inf, math.inf, math.inf, math.inf)

    values = []
    for name in ("left", "top", "right", "bottom"):
        try:
            value = getattr(box, name)
        except Exception:
            return (math.inf, math.inf, math.inf, math.inf)

        if not _finite_number(value):
            return (math.inf, math.inf, math.inf, math.inf)

        values.append(float(value))

    return values[0], values[1], values[2], values[3]


def _number_sort_key(value: object) -> tuple[int, float]:
    if not _finite_number(value):
        return 1, math.inf

    return 0, float(value)


def _optional_string_sort_key(value: object) -> tuple[object, ...]:
    if value is None:
        return (0, "")

    if isinstance(value, str):
        return (
            1,
            normalize_ui_text(value),
            value,
        )

    return (
        2,
        type(value).__module__,
        type(value).__qualname__,
    )


def _optional_scalar_sort_key(value: object) -> tuple[object, ...]:
    if value is None:
        return (0, "")

    if isinstance(value, bool):
        return 1, int(value)

    if isinstance(value, int):
        return 2, value

    if isinstance(value, float):
        return 3, *_float_sort_key(value)

    if isinstance(value, str):
        return 4, value

    return (
        5,
        type(value).__module__,
        type(value).__qualname__,
    )


def _float_sort_key(value: float) -> tuple[int, float]:
    if math.isnan(value):
        return 2, 0.0

    if value == -math.inf:
        return 0, 0.0

    if value == math.inf:
        return 3, 0.0

    return 1, value


def _optional_bool_sort_key(value: object) -> tuple[object, ...]:
    if value is None:
        return (0, 0)

    if isinstance(value, bool):
        return 1, int(value)

    return (
        2,
        type(value).__module__,
        type(value).__qualname__,
    )


def _normalized_source(source: str | None) -> str | None:
    if source is None:
        return None

    source_name = source.strip()
    if not source_name:
        return None

    return source_name.casefold()
