"""Viewport models and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.models import BoundingBox


class VisibilityStatus(str, Enum):
    """Possible visibility states for a semantic target."""

    VISIBLE = "visible"
    OUTSIDE_VIEWPORT = "outside_viewport"
    GEOMETRY_UNAVAILABLE = "geometry_unavailable"


class DiagnosisStatus(str, Enum):
    """Read-only viewport target diagnosis outcomes."""

    VISIBLE = "visible"
    NEEDS_SEARCH = "needs_search"
    BLOCKED = "blocked"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class Viewport:
    """The currently visible webpage region."""

    bounds: BoundingBox

    def __post_init__(self) -> None:
        if not isinstance(self.bounds, BoundingBox):
            raise ValueError("bounds must be a BoundingBox")


def classify_visibility(
    target_bounds: BoundingBox | None,
    viewport: Viewport,
) -> VisibilityStatus:
    """Classify target visibility against the current viewport."""

    if not isinstance(viewport, Viewport):
        raise ValueError("viewport must be a Viewport")

    if target_bounds is None:
        return VisibilityStatus.GEOMETRY_UNAVAILABLE

    if not isinstance(target_bounds, BoundingBox):
        raise ValueError(
            "target_bounds must be a BoundingBox or None"
        )

    if target_bounds.intersects(viewport.bounds):
        return VisibilityStatus.VISIBLE

    return VisibilityStatus.OUTSIDE_VIEWPORT


@dataclass(frozen=True, slots=True)
class SemanticAXElement:
    """A read-only Accessibility semantic element with optional geometry."""

    role: str
    text: str | None
    bounds: BoundingBox | None
    value: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ValueError("role must be a non-empty string")

        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("text must be a string or None")

        if self.value is not None and not isinstance(
            self.value,
            (str, int, float, bool),
        ):
            raise ValueError(
                "value must be a string, number, boolean, or None"
            )

        if self.bounds is not None and not isinstance(
            self.bounds,
            BoundingBox,
        ):
            raise ValueError(
                "bounds must be a BoundingBox or None"
            )


@dataclass(frozen=True, slots=True)
class ViewportTargetDiagnosis:
    """Deterministic read-only target diagnosis for a viewport observation."""

    status: DiagnosisStatus
    matches: tuple[SemanticAXElement, ...]
    visibility: VisibilityStatus | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, DiagnosisStatus):
            raise ValueError("status must be a DiagnosisStatus")

        if not isinstance(self.matches, tuple) or any(
            not isinstance(match, SemanticAXElement)
            for match in self.matches
        ):
            raise ValueError(
                "matches must be a tuple of SemanticAXElement objects"
            )

        if self.visibility is not None and not isinstance(
            self.visibility,
            VisibilityStatus,
        ):
            raise ValueError(
                "visibility must be a VisibilityStatus or None"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def diagnose_semantic_target_visibility(
    *,
    application_name: str | None,
    viewport: Viewport | None,
    semantic_elements: tuple[SemanticAXElement, ...],
    target_role: str,
    target_text: str,
    expected_application_name: str = "Google Chrome",
) -> ViewportTargetDiagnosis:
    """Return a deterministic read-only diagnosis for one semantic target."""

    _validate_non_empty_string(
        "target_role",
        target_role,
    )
    _validate_non_empty_string(
        "target_text",
        target_text,
    )
    _validate_non_empty_string(
        "expected_application_name",
        expected_application_name,
    )

    if application_name is not None and not isinstance(
        application_name,
        str,
    ):
        raise ValueError(
            "application_name must be a string or None"
        )

    if viewport is not None and not isinstance(viewport, Viewport):
        raise ValueError("viewport must be a Viewport or None")

    if not isinstance(semantic_elements, tuple) or any(
        not isinstance(element, SemanticAXElement)
        for element in semantic_elements
    ):
        raise ValueError(
            "semantic_elements must be a tuple of SemanticAXElement objects"
        )

    if (
        application_name is None
        or application_name.strip() != expected_application_name
    ):
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.BLOCKED,
            matches=(),
            visibility=None,
            reason=(
                "frontmost application is not "
                f"{expected_application_name}"
            ),
        )

    if viewport is None:
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.BLOCKED,
            matches=(),
            visibility=None,
            reason="no reliable unique viewport is available",
        )

    matches = _matching_semantic_elements(
        semantic_elements,
        target_role=target_role,
        target_text=target_text,
    )

    if not matches:
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.NOT_FOUND,
            matches=(),
            visibility=None,
            reason="no matching semantic target in this observation",
        )

    if len(matches) > 1:
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.BLOCKED,
            matches=matches,
            visibility=None,
            reason="multiple matching semantic targets in this observation",
        )

    visibility = classify_visibility(
        matches[0].bounds,
        viewport,
    )

    if visibility is VisibilityStatus.VISIBLE:
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.VISIBLE,
            matches=matches,
            visibility=visibility,
            reason="unique semantic target is visible",
        )

    if visibility in (
        VisibilityStatus.GEOMETRY_UNAVAILABLE,
        VisibilityStatus.OUTSIDE_VIEWPORT,
    ):
        return ViewportTargetDiagnosis(
            status=DiagnosisStatus.NEEDS_SEARCH,
            matches=matches,
            visibility=visibility,
            reason=(
                "unique semantic target may require viewport search"
            ),
        )

    return ViewportTargetDiagnosis(
        status=DiagnosisStatus.BLOCKED,
        matches=matches,
        visibility=visibility,
        reason="target visibility is ambiguous",
    )


def _matching_semantic_elements(
    semantic_elements: tuple[SemanticAXElement, ...],
    *,
    target_role: str,
    target_text: str,
) -> tuple[SemanticAXElement, ...]:
    normalized_target_text = normalize_ui_text(target_text)

    return tuple(
        element
        for element in semantic_elements
        if element.role == target_role
        and normalize_ui_text(element.text) == normalized_target_text
    )


def _validate_non_empty_string(
    name: str,
    value: object,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
