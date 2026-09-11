"""Predicate-oriented observation of UI state from perception snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Iterable

from computer_agent.grounding.models import TargetSpec
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.models import UIElement


class StateObservationStatus(str, Enum):
    """Possible outcomes for one observed UI state predicate."""

    PRESENT = "present"
    ABSENT = "absent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class StateObservationCandidate:
    """One element considered while evaluating a state predicate."""

    element: UIElement
    match_basis: str
    predicate_match: bool
    mismatch_reasons: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.element, UIElement):
            raise ValueError("element must be a UIElement")

        if not isinstance(self.match_basis, str) or not self.match_basis.strip():
            raise ValueError("match_basis must be a non-empty string")

        if not isinstance(self.predicate_match, bool):
            raise ValueError("predicate_match must be a bool")

        _validate_string_tuple(self.mismatch_reasons, "mismatch_reasons")
        _validate_string_tuple(self.uncertainty_reasons, "uncertainty_reasons")

        if self.distance is not None:
            if (
                isinstance(self.distance, bool)
                or not isinstance(self.distance, Real)
                or not math.isfinite(self.distance)
                or self.distance < 0
            ):
                raise ValueError(
                    "distance must be finite non-negative numeric or None"
                )

            object.__setattr__(self, "distance", float(self.distance))

    @property
    def reliable_match(self) -> bool:
        """Return whether this candidate proves the full predicate present."""

        return (
            self.predicate_match
            and not self.mismatch_reasons
            and not self.uncertainty_reasons
        )


@dataclass(frozen=True, slots=True)
class StateObservationResult:
    """Result of evaluating one TargetSpec as observed UI state."""

    status: StateObservationStatus
    target: TargetSpec
    candidates: tuple[StateObservationCandidate, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, StateObservationStatus):
            raise ValueError("status must be a StateObservationStatus")

        if not isinstance(self.target, TargetSpec):
            raise ValueError("target must be a TargetSpec")

        if not isinstance(self.candidates, tuple):
            raise ValueError("candidates must be a tuple")

        for candidate in self.candidates:
            if not isinstance(candidate, StateObservationCandidate):
                raise ValueError(
                    "candidates must contain StateObservationCandidate objects"
                )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


class StateObserver:
    """Evaluate TargetSpec predicates as observed state, not action targets."""

    def observe(
        self,
        *,
        snapshot: PerceptionSnapshot,
        target: TargetSpec,
    ) -> StateObservationResult:
        """Return whether the snapshot supports the requested state predicate.

        ABSENT means absent from the supplied snapshot's fused elements only.
        Whole-document absence is outside this v1 observer's scope.
        """

        if not isinstance(snapshot, PerceptionSnapshot):
            raise ValueError("snapshot must be a PerceptionSnapshot")

        result = self.observe_elements(
            target=target,
            elements=snapshot.fused_elements,
        )
        if (
            result.status is StateObservationStatus.ABSENT
            and snapshot.warnings
        ):
            return StateObservationResult(
                status=StateObservationStatus.INCONCLUSIVE,
                target=result.target,
                candidates=result.candidates,
                reason=(
                    "state observation inconclusive: absence could not be "
                    "established because snapshot observation contained "
                    f"warnings {snapshot.warnings!r}"
                ),
            )

        return result

    def observe_elements(
        self,
        *,
        target: TargetSpec,
        elements: Iterable[UIElement],
    ) -> StateObservationResult:
        """Evaluate a target against already selected fused UI elements."""

        if not isinstance(target, TargetSpec):
            raise ValueError("target must be a TargetSpec")

        if target.reference_point is not None:
            raise ValueError(
                "StateObserver v1 does not support reference_point"
            )

        element_tuple = tuple(elements)
        for element in element_tuple:
            if not isinstance(element, UIElement):
                raise ValueError("elements must contain UIElement objects")

        candidates = tuple(
            candidate
            for candidate in (
                _candidate_for(element, target) for element in element_tuple
            )
            if candidate is not None
        )
        status = _reduce_status(candidates)
        return StateObservationResult(
            status=status,
            target=target,
            candidates=candidates,
            reason=_result_reason(status, candidates),
        )


def _candidate_for(
    element: UIElement,
    target: TargetSpec,
) -> StateObservationCandidate | None:
    text_matches = (
        target.text is not None
        and normalize_ui_text(element.text) == normalize_ui_text(target.text)
    )
    identifier_matches = (
        target.identifier is not None
        and element.identifier is not None
        and element.identifier.strip() == target.identifier.strip()
    )

    if not text_matches and not identifier_matches:
        return None

    mismatch_reasons: list[str] = []
    uncertainty_reasons: list[str] = []

    if target.text is not None and not text_matches:
        mismatch_reasons.append("text_mismatch")

    if target.identifier is not None and not identifier_matches:
        mismatch_reasons.append("identifier_mismatch")

    if not _element_type_is_compatible(element, target):
        mismatch_reasons.append("incompatible_element_type")

    if not _confidence_is_usable(element.confidence):
        uncertainty_reasons.append("invalid_confidence")
    elif element.confidence < target.minimum_confidence:
        uncertainty_reasons.append("low_confidence")

    return StateObservationCandidate(
        element=element,
        match_basis=_match_basis(text_matches, identifier_matches),
        predicate_match=not mismatch_reasons,
        mismatch_reasons=tuple(mismatch_reasons),
        uncertainty_reasons=tuple(uncertainty_reasons),
        distance=None,
    )


def _element_type_is_compatible(
    element: UIElement,
    target: TargetSpec,
) -> bool:
    if not target.element_types:
        return True

    element_type = normalize_ui_text(element.element_type)
    return element_type in {
        normalize_ui_text(expected_type)
        for expected_type in target.element_types
    }


def _confidence_is_usable(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _match_basis(
    text_matches: bool,
    identifier_matches: bool,
) -> str:
    if text_matches and identifier_matches:
        return "text+identifier"
    if identifier_matches:
        return "identifier"
    return "text"


def _reduce_status(
    candidates: tuple[StateObservationCandidate, ...],
) -> StateObservationStatus:
    if any(candidate.reliable_match for candidate in candidates):
        return StateObservationStatus.PRESENT

    if any(
        candidate.predicate_match and candidate.uncertainty_reasons
        for candidate in candidates
    ):
        return StateObservationStatus.INCONCLUSIVE

    return StateObservationStatus.ABSENT


def _result_reason(
    status: StateObservationStatus,
    candidates: tuple[StateObservationCandidate, ...],
) -> str:
    reliable = sum(1 for candidate in candidates if candidate.reliable_match)
    uncertain = sum(
        1
        for candidate in candidates
        if candidate.predicate_match and candidate.uncertainty_reasons
    )
    mismatched = sum(1 for candidate in candidates if candidate.mismatch_reasons)
    return (
        f"state observation {status.value}: {reliable} reliable matches, "
        f"{uncertain} uncertain possible matches, "
        f"{mismatched} predicate mismatches, "
        f"{len(candidates)} candidates in supplied element collection"
    )


def _validate_string_tuple(value: object, field_name: str) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
