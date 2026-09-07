"""Tests for viewport visibility classification."""

from computer_agent.perception.models import BoundingBox
from computer_agent.perception.viewport import (
    DiagnosisStatus,
    SemanticAXElement,
    Viewport,
    VisibilityStatus,
    classify_visibility,
    diagnose_semantic_target_visibility,
)


def test_geometry_unavailable():
    viewport = Viewport(
        BoundingBox(
            x=0,
            y=100,
            width=1000,
            height=600,
        )
    )

    result = classify_visibility(
        None,
        viewport,
    )

    assert result is VisibilityStatus.GEOMETRY_UNAVAILABLE


def test_visible_when_target_intersects_viewport():
    viewport = Viewport(
        BoundingBox(
            x=0,
            y=100,
            width=1000,
            height=600,
        )
    )
    target = BoundingBox(
        x=100,
        y=500,
        width=120,
        height=24,
    )

    result = classify_visibility(
        target,
        viewport,
    )

    assert result is VisibilityStatus.VISIBLE


def test_outside_viewport_when_target_does_not_intersect():
    viewport = Viewport(
        BoundingBox(
            x=0,
            y=100,
            width=1000,
            height=600,
        )
    )
    target = BoundingBox(
        x=100,
        y=900,
        width=120,
        height=24,
    )

    result = classify_visibility(
        target,
        viewport,
    )

    assert result is VisibilityStatus.OUTSIDE_VIEWPORT


def test_semantic_ax_element_allows_missing_geometry():
    element = SemanticAXElement(
        role="AXLink",
        text="Privacy Notice",
        bounds=None,
    )

    assert element.role == "AXLink"
    assert element.text == "Privacy Notice"
    assert element.bounds is None


def _viewport():
    return Viewport(
        BoundingBox(
            x=0,
            y=100,
            width=1000,
            height=600,
        )
    )


def _semantic_target(
    *,
    bounds=None,
    text="Privacy Notice",
    role="AXLink",
):
    return SemanticAXElement(
        role=role,
        text=text,
        bounds=bounds,
    )


def _diagnose(
    *,
    application_name="Google Chrome",
    viewport=None,
    elements=(),
):
    return diagnose_semantic_target_visibility(
        application_name=application_name,
        viewport=_viewport() if viewport is None else viewport,
        semantic_elements=tuple(elements),
        target_role="AXLink",
        target_text="Privacy Notice",
    )


def test_unique_visible_target_diagnosis_is_visible():
    result = _diagnose(
        elements=(
            _semantic_target(
                bounds=BoundingBox(
                    x=100,
                    y=500,
                    width=120,
                    height=24,
                ),
            ),
        ),
    )

    assert result.status is DiagnosisStatus.VISIBLE
    assert result.visibility is VisibilityStatus.VISIBLE
    assert len(result.matches) == 1


def test_unique_geometry_unavailable_target_diagnosis_needs_search():
    result = _diagnose(
        elements=(
            _semantic_target(
                bounds=None,
            ),
        ),
    )

    assert result.status is DiagnosisStatus.NEEDS_SEARCH
    assert result.visibility is VisibilityStatus.GEOMETRY_UNAVAILABLE
    assert len(result.matches) == 1


def test_unique_outside_viewport_target_diagnosis_needs_search():
    result = _diagnose(
        elements=(
            _semantic_target(
                bounds=BoundingBox(
                    x=100,
                    y=900,
                    width=120,
                    height=24,
                ),
            ),
        ),
    )

    assert result.status is DiagnosisStatus.NEEDS_SEARCH
    assert result.visibility is VisibilityStatus.OUTSIDE_VIEWPORT
    assert len(result.matches) == 1


def test_zero_target_matches_diagnosis_is_not_found():
    result = _diagnose(
        elements=(
            _semantic_target(
                text="Docs",
                bounds=BoundingBox(
                    x=100,
                    y=500,
                    width=120,
                    height=24,
                ),
            ),
        ),
    )

    assert result.status is DiagnosisStatus.NOT_FOUND
    assert result.visibility is None
    assert result.matches == ()


def test_duplicate_target_matches_diagnosis_is_blocked():
    result = _diagnose(
        elements=(
            _semantic_target(
                bounds=BoundingBox(
                    x=100,
                    y=500,
                    width=120,
                    height=24,
                ),
            ),
            _semantic_target(
                bounds=None,
            ),
        ),
    )

    assert result.status is DiagnosisStatus.BLOCKED
    assert result.visibility is None
    assert len(result.matches) == 2


def test_non_chrome_frontmost_app_diagnosis_is_blocked():
    result = _diagnose(
        application_name="Safari",
        elements=(
            _semantic_target(
                bounds=BoundingBox(
                    x=100,
                    y=500,
                    width=120,
                    height=24,
                ),
            ),
        ),
    )

    assert result.status is DiagnosisStatus.BLOCKED
    assert result.matches == ()
    assert result.visibility is None


def test_missing_or_ambiguous_viewport_diagnosis_is_blocked():
    result = diagnose_semantic_target_visibility(
        application_name="Google Chrome",
        viewport=None,
        semantic_elements=(
            _semantic_target(
                bounds=BoundingBox(
                    x=100,
                    y=500,
                    width=120,
                    height=24,
                ),
            ),
        ),
        target_role="AXLink",
        target_text="Privacy Notice",
    )

    assert result.status is DiagnosisStatus.BLOCKED
    assert result.matches == ()
    assert result.visibility is None
