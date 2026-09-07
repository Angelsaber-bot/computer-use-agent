"""Focused tests for Experiment 05.04 viewport-aware grounding."""

from computer_agent.grounding.models import GroundingStatus, TargetSpec
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.perception.models import BoundingBox, UIElement


def test_viewport_outside_ax_target_is_not_resolved():
    viewport = BoundingBox(
        x=0,
        y=100,
        width=1000,
        height=600,
    )

    target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=900,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        enabled=True,
        source="accessibility",
    )

    result = UIGrounder().ground(
        TargetSpec(
            text="Privacy Notice",
            element_types=("link",),
        ),
        (target,),
        viewport=viewport,
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == (
        "outside_viewport",
    )


def test_viewport_visible_target_still_resolves():
    viewport = BoundingBox(
        x=0,
        y=100,
        width=1000,
        height=600,
    )

    target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=500,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        enabled=True,
        source="accessibility",
    )

    result = UIGrounder().ground(
        TargetSpec(
            text="Privacy Notice",
            element_types=("link",),
        ),
        (target,),
        viewport=viewport,
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_viewport_partial_intersection_counts_as_visible():
    viewport = BoundingBox(
        x=0,
        y=100,
        width=1000,
        height=600,
    )

    target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=690,
            width=120,
            height=30,
        ),
        confidence=1.0,
        text="Privacy Notice",
        enabled=True,
        source="accessibility",
    )

    result = UIGrounder().ground(
        TargetSpec(
            text="Privacy Notice",
            element_types=("link",),
        ),
        (target,),
        viewport=viewport,
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_no_viewport_preserves_existing_grounding_behavior():
    target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=900,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        enabled=True,
        source="accessibility",
    )

    result = UIGrounder().ground(
        TargetSpec(
            text="Privacy Notice",
            element_types=("link",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_outside_identifier_match_does_not_fall_back_to_visible_text_decoy():
    viewport = BoundingBox(
        x=0,
        y=100,
        width=1000,
        height=600,
    )

    identifier_target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=900,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        identifier="privacy-link",
        enabled=True,
        source="accessibility",
    )

    visible_text_decoy = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=500,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        identifier="other-link",
        enabled=True,
        source="accessibility",
    )

    result = UIGrounder().ground(
        TargetSpec(
            text="Privacy Notice",
            identifier="privacy-link",
            element_types=("link",),
        ),
        (
            visible_text_decoy,
            identifier_target,
        ),
        viewport=viewport,
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert len(result.candidates) == 1
    assert result.candidates[0].element is identifier_target
    assert result.candidates[0].rejection_reasons == (
        "outside_viewport",
    )


def test_invalid_viewport_type_is_rejected():
    target = UIElement(
        element_type="link",
        bounding_box=BoundingBox(
            x=100,
            y=500,
            width=120,
            height=24,
        ),
        confidence=1.0,
        text="Privacy Notice",
        enabled=True,
        source="accessibility",
    )

    try:
        UIGrounder().ground(
            TargetSpec(
                text="Privacy Notice",
                element_types=("link",),
            ),
            (target,),
            viewport=(0, 100, 1000, 600),
        )
    except ValueError as exc:
        assert str(exc) == "viewport must be a BoundingBox or None"
    else:
        raise AssertionError("expected invalid viewport to be rejected")
