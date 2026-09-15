"""Bounded semantic target resolution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from typing import TypeAlias

from computer_agent.grounding.models import (
    GroundingCandidate,
    GroundingResult,
    GroundingStatus,
    TargetSpec,
)
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.models import BoundingBox, UIElement


DestinationIdentity: TypeAlias = tuple[str, str]
DestinationNormalizer: TypeAlias = Callable[
    [object],
    DestinationIdentity | None,
]


@dataclass(frozen=True, slots=True)
class EquivalentLinkResolution:
    """Evidence for collapsing same-text link ambiguity."""

    result: GroundingResult
    canonical_destination: str
    candidate_destinations: tuple[str, ...]


class SemanticTargetResolver:
    """Resolve exact, equivalent-link, and wrapped semantic targets."""

    def __init__(
        self,
        *,
        destination_normalizer: DestinationNormalizer | None = None,
    ) -> None:
        if destination_normalizer is not None and not callable(
            destination_normalizer
        ):
            raise ValueError("destination_normalizer must be callable or None")

        self._grounder = UIGrounder()
        self._destination_normalizer = destination_normalizer

    def ground(
        self,
        target_spec: TargetSpec,
        elements: Iterable[UIElement],
        *,
        viewport: BoundingBox | None = None,
    ) -> GroundingResult:
        """Return a fail-closed semantic grounding result."""

        element_tuple = tuple(elements)
        exact = self._grounder.ground(
            target_spec,
            element_tuple,
            viewport=viewport,
        )
        if exact.status is GroundingStatus.RESOLVED:
            return exact

        equivalent = self.resolve_equivalent_links(exact)
        if equivalent is not None:
            return equivalent.result

        composite = self._resolve_composite_text(
            target_spec,
            element_tuple,
            viewport=viewport,
        )
        if composite is not None:
            equivalent = self.resolve_equivalent_links(composite)
            if equivalent is not None:
                return equivalent.result
            return composite

        return exact

    def resolve_equivalent_links(
        self,
        grounding: GroundingResult,
    ) -> EquivalentLinkResolution | None:
        """Collapse ambiguous link candidates only when AXURL proves identity."""

        if (
            grounding.status is not GroundingStatus.AMBIGUOUS
            or self._destination_normalizer is None
        ):
            return None

        eligible = tuple(
            candidate
            for candidate in grounding.candidates
            if candidate.eligible
        )
        if len(eligible) < 2:
            return None

        identities: list[DestinationIdentity] = []
        destinations: list[str] = []
        for candidate in eligible:
            element = candidate.element
            if normalize_ui_text(element.element_type) != "link":
                return None

            identity = self._destination_normalizer(element.value)
            if identity is None:
                return None

            identities.append(identity)
            destinations.append(identity[0])

        canonical = {identity[0] for identity in identities}
        if len(canonical) != 1:
            return None

        chosen = min(
            (candidate.element for candidate in eligible),
            key=_physical_sort_key,
        )
        canonical_destination = next(iter(canonical))
        return EquivalentLinkResolution(
            result=GroundingResult(
                status=GroundingStatus.RESOLVED,
                element=chosen,
                candidates=grounding.candidates,
                reason=(
                    "resolved equivalent links by accessibility URL"
                ),
            ),
            canonical_destination=canonical_destination,
            candidate_destinations=tuple(destinations),
        )

    def _resolve_composite_text(
        self,
        target_spec: TargetSpec,
        elements: tuple[UIElement, ...],
        *,
        viewport: BoundingBox | None,
    ) -> GroundingResult | None:
        if target_spec.text is None:
            return None

        target_text = normalize_ui_text(target_spec.text)
        fragments = tuple(
            sorted(
                (
                    element
                    for element in elements
                    if _element_can_participate(
                        element,
                        target_spec,
                        viewport,
                    )
                ),
                key=_physical_sort_key,
            )
        )
        matches: list[GroundingCandidate] = []
        max_fragments = min(6, len(fragments))

        for start in range(len(fragments)):
            span: list[UIElement] = []
            for end in range(start, min(len(fragments), start + max_fragments)):
                element = fragments[end]
                if span and not _fragments_are_adjacent(span[-1], element):
                    break
                span.append(element)

                if len(span) < 2:
                    continue

                if target_text not in _normalized_span_forms(span):
                    continue

                matches.extend(
                    _candidates_for_span(
                        span,
                        target_spec,
                    )
                )

        if not matches:
            return None

        candidates = tuple(sorted(matches, key=_candidate_sort_key))
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
                reason="composite text candidates were unsafe",
            )

        if len(eligible) == 1:
            return GroundingResult(
                status=GroundingStatus.RESOLVED,
                element=eligible[0].element,
                candidates=candidates,
                reason="resolved by composite text",
            )

        return GroundingResult(
            status=GroundingStatus.AMBIGUOUS,
            element=None,
            candidates=candidates,
            reason="ambiguous composite text candidates",
        )


def _element_can_participate(
    element: UIElement,
    target_spec: TargetSpec,
    viewport: BoundingBox | None,
) -> bool:
    if not _box_is_usable(element):
        return False
    if viewport is not None and not element.bounding_box.intersects(viewport):
        return False
    if element.enabled is False:
        return False
    if element.confidence < target_spec.minimum_confidence:
        return False
    if not _element_type_is_compatible(element, target_spec):
        return False
    if not normalize_ui_text(element.text):
        return False

    return True


def _candidates_for_span(
    span: list[UIElement],
    target_spec: TargetSpec,
) -> tuple[GroundingCandidate, ...]:
    rejection_reasons: list[str] = []
    element_type = normalize_ui_text(span[0].element_type)
    source = normalize_ui_text(span[0].source)

    if any(normalize_ui_text(element.element_type) != element_type for element in span):
        rejection_reasons.append("mixed_element_type")

    if any(normalize_ui_text(element.source) != source for element in span):
        rejection_reasons.append("mixed_source")

    if "link" in {
        normalize_ui_text(kind)
        for kind in target_spec.element_types
    }:
        if element_type != "link":
            rejection_reasons.append("not_actionable_link")
        if source == "ocr":
            rejection_reasons.append("ocr_text_not_actionable")

        all_values = [
            element.value
            for element in span
        ]
        values = {
            value
            for value in all_values
            if value is not None
        }
        if len(values) != len(set(all_values)):
            rejection_reasons.append("unproven_link_destination")

        if not rejection_reasons and len(values) > 1:
            return tuple(
                GroundingCandidate(
                    element=element,
                    match_basis="composite_text",
                    distance=_distance_from_reference(
                        element,
                        target_spec,
                    ),
                )
                for element in sorted(span, key=_physical_sort_key)
            )

        chosen = min(span, key=_physical_sort_key)
        return (
            GroundingCandidate(
                element=chosen,
                match_basis="composite_text",
                rejection_reasons=tuple(rejection_reasons),
                distance=_distance_from_reference(chosen, target_spec),
            ),
        )

    combined = _combined_element(span, target_spec)
    return (
        GroundingCandidate(
            element=combined,
            match_basis="composite_text",
            rejection_reasons=tuple(rejection_reasons),
            distance=_distance_from_reference(combined, target_spec),
        ),
    )


def _combined_element(
    span: list[UIElement],
    target_spec: TargetSpec,
) -> UIElement:
    left = min(element.bounding_box.left for element in span)
    top = min(element.bounding_box.top for element in span)
    right = max(element.bounding_box.right for element in span)
    bottom = max(element.bounding_box.bottom for element in span)
    confidence = min(element.confidence for element in span)
    return UIElement(
        element_type=span[0].element_type,
        bounding_box=BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        ),
        confidence=confidence,
        text=target_spec.text,
        identifier=None,
        value=span[0].value,
        enabled=span[0].enabled,
        focused=None,
        selected=None,
        source=span[0].source,
    )


def _normalized_span_forms(
    span: list[UIElement],
) -> set[str]:
    texts = [
        element.text or ""
        for element in span
    ]
    joined = " ".join(texts)
    forms = {normalize_ui_text(joined)}

    compact_parts: list[str] = []
    preserved_parts: list[str] = []
    consumed_next = False
    preserved_consumed_next = False
    for index, text in enumerate(texts):
        if consumed_next:
            consumed_next = False
        elif index + 1 < len(texts) and text.rstrip().endswith("-"):
            compact_parts.append(
                text.rstrip()[:-1] + texts[index + 1].lstrip()
            )
            consumed_next = True
        else:
            compact_parts.append(text)

        if preserved_consumed_next:
            preserved_consumed_next = False
        elif index + 1 < len(texts) and text.rstrip().endswith("-"):
            preserved_parts.append(
                text.rstrip() + texts[index + 1].lstrip()
            )
            preserved_consumed_next = True
        else:
            preserved_parts.append(text)
    forms.add(normalize_ui_text(" ".join(compact_parts)))
    forms.add(normalize_ui_text(" ".join(preserved_parts)))

    return forms


def _fragments_are_adjacent(
    first: UIElement,
    second: UIElement,
) -> bool:
    first_box = first.bounding_box
    second_box = second.bounding_box
    height = max(first_box.height, second_box.height)
    if height <= 0:
        return False

    same_line = (
        abs(first_box.center[1] - second_box.center[1])
        <= height * 0.45
    )
    if same_line:
        if second_box.left < first_box.left:
            return False
        return (second_box.left - first_box.right) <= height * 4

    vertical_gap = second_box.top - first_box.bottom
    if vertical_gap < 0 or vertical_gap > height * 1.5:
        return False

    left_delta = abs(second_box.left - first_box.left)
    if left_delta > height * 4:
        return False

    return True


def _element_type_is_compatible(
    element: UIElement,
    target_spec: TargetSpec,
) -> bool:
    if not target_spec.element_types:
        return True

    element_type = normalize_ui_text(element.element_type)
    return element_type in {
        normalize_ui_text(expected)
        for expected in target_spec.element_types
    }


def _box_is_usable(element: UIElement) -> bool:
    box = element.bounding_box
    values = (
        box.left,
        box.top,
        box.right,
        box.bottom,
        box.width,
        box.height,
    )
    return (
        all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in values
        )
        and box.width > 0
        and box.height > 0
        and 0 <= box.left < box.right
        and 0 <= box.top < box.bottom
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


def _candidate_sort_key(candidate: GroundingCandidate) -> tuple[object, ...]:
    return (
        _physical_sort_key(candidate.element),
        candidate.rejection_reasons,
    )


def _physical_sort_key(element: UIElement) -> tuple[object, ...]:
    return (
        element.bounding_box.top,
        element.bounding_box.left,
        element.bounding_box.bottom,
        element.bounding_box.right,
        normalize_ui_text(element.text),
        normalize_ui_text(element.element_type),
    )
