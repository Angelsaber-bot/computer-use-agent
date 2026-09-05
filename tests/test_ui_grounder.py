import inspect
import math

import pytest

from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.grounding.models import GroundingCandidate
import computer_agent.grounding.ui_grounder as ui_grounder_module
from computer_agent.perception.models import BoundingBox, UIElement


def _box(
    x=10,
    y=20,
    width=100,
    height=30,
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    text=None,
    *,
    identifier=None,
    element_type="button",
    confidence=0.95,
    value=None,
    enabled=True,
    focused=None,
    selected=None,
    source="accessibility",
    x=10,
    y=20,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(
            x=x,
            y=y,
        ),
        confidence=confidence,
        text=text,
        identifier=identifier,
        value=value,
        enabled=enabled,
        focused=focused,
        selected=selected,
        source=source,
    )


def _ground(
    target_spec,
    elements,
):
    return UIGrounder().ground(
        target_spec,
        elements,
    )


def test_unique_normalized_text_target_resolves():
    target = _element("Submit")

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
        ),
        (
            _element("Cancel"),
            target,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target
    assert result.reason == "resolved by text"


def test_matching_is_case_insensitive_and_whitespace_underscore_normalized():
    target = _element(" TARGET   INPUT_12 ")

    result = _ground(
        TargetSpec(
            text="target input 12",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_identifier_match_outranks_text_only_match():
    identifier_target = _element(
        "Submit",
        identifier="submit-button",
        x=100,
    )
    text_decoy = _element(
        "Submit",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            identifier="submit-button",
            element_types=("button",),
        ),
        (
            text_decoy,
            identifier_target,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is identifier_target
    assert [candidate.match_basis for candidate in result.candidates] == [
        "identifier"
    ]


def test_identifier_only_target_spec_resolves_exact_identifier():
    target = _element(
        "Different Text",
        identifier="submit-button",
    )

    result = _ground(
        TargetSpec(
            identifier="submit-button",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target
    assert result.reason == "resolved by identifier"


def test_text_match_is_fallback_when_no_identifier_matches():
    target = _element(
        "Submit",
        identifier="actual-button",
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            identifier="missing-button",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target
    assert result.candidates[0].match_basis == "text"


def test_unsafe_identifier_match_prevents_fallback_to_text_decoy():
    unsafe_identifier = _element(
        "Submit",
        identifier="submit-button",
        enabled=False,
    )
    text_decoy = _element(
        "Submit",
        identifier="other-button",
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            identifier="submit-button",
            element_types=("button",),
        ),
        (
            text_decoy,
            unsafe_identifier,
        ),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates == (
        GroundingCandidate(
            element=unsafe_identifier,
            match_basis="identifier",
            rejection_reasons=("disabled",),
        ),
    )


def test_identifier_text_conflict_returns_unsafe():
    target = _element(
        "Cancel",
        identifier="submit-button",
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            identifier="submit-button",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == (
        "identifier_text_conflict",
    )


def test_same_text_wrong_role_decoy_is_rejected_while_correct_role_resolves():
    wrong_role = _element(
        "Save",
        element_type="text",
        source="ocr",
        x=100,
    )
    target = _element(
        "Save",
        element_type="button",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Save",
            element_types=("button",),
        ),
        (
            wrong_role,
            target,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target
    assert result.candidates[1].element is wrong_role
    assert result.candidates[1].rejection_reasons == (
        "incompatible_element_type",
    )


def test_wrong_role_only_semantic_match_returns_unsafe():
    target = _element(
        "Save",
        element_type="text",
        source="ocr",
        enabled=None,
    )

    result = _ground(
        TargetSpec(
            text="Save",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == (
        "incompatible_element_type",
    )


def test_disabled_only_exact_target_returns_unsafe():
    target = _element(
        "Submit",
        enabled=False,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
        ),
        (target,),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == ("disabled",)


def test_disabled_decoy_does_not_block_one_safe_target_in_same_tier():
    disabled_decoy = _element(
        "Submit",
        enabled=False,
        x=100,
    )
    target = _element(
        "Submit",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
        ),
        (
            disabled_decoy,
            target,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target
    assert any(
        candidate.rejection_reasons == ("disabled",)
        for candidate in result.candidates
    )


def test_high_confidence_ocr_only_target_resolves_with_enabled_none():
    target = _element(
        "CANVAS_ACTION_12",
        element_type="text",
        confidence=0.8,
        enabled=None,
        source="ocr",
    )

    result = _ground(
        TargetSpec(
            text="canvas action 12",
            element_types=("text",),
            minimum_confidence=0.7,
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_low_confidence_ocr_only_target_returns_unsafe():
    target = _element(
        "CANVAS_ACTION_12",
        element_type="text",
        confidence=0.69,
        enabled=None,
        source="ocr",
    )

    result = _ground(
        TargetSpec(
            text="CANVAS_ACTION_12",
            element_types=("text",),
            minimum_confidence=0.7,
        ),
        (target,),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == ("low_confidence",)


def test_confidence_equal_to_minimum_threshold_remains_eligible():
    target = _element(
        "Submit",
        confidence=0.7,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            minimum_confidence=0.7,
        ),
        (target,),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is target


def test_uniquely_higher_confidence_resolves_within_same_semantic_and_source_tier():
    low_confidence = _element(
        "Submit",
        confidence=0.7,
        x=100,
    )
    high_confidence = _element(
        "Submit",
        confidence=0.9,
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            minimum_confidence=0.7,
        ),
        (
            low_confidence,
            high_confidence,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is high_confidence


def test_two_equally_valid_candidates_return_ambiguous():
    first = _element(
        "Submit",
        x=100,
    )
    second = _element(
        "Submit",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
        ),
        (
            second,
            first,
        ),
    )

    assert result.status is GroundingStatus.AMBIGUOUS
    assert result.element is None
    assert [candidate.element for candidate in result.candidates] == [
        first,
        second,
    ]


def test_reversed_ambiguous_candidates_differing_only_optional_state_are_identical():
    first = _element(
        "Submit",
        value="alpha",
        focused=False,
        selected=True,
    )
    second = _element(
        "Submit",
        value="beta",
        focused=True,
        selected=False,
    )
    target_spec = TargetSpec(
        text="Submit",
        element_types=("button",),
    )

    forward = _ground(
        target_spec,
        (
            first,
            second,
        ),
    )
    reversed_result = _ground(
        target_spec,
        (
            second,
            first,
        ),
    )

    assert forward.status is GroundingStatus.AMBIGUOUS
    assert reversed_result == forward


def test_higher_priority_source_resolves_within_same_semantic_tier():
    ocr = _element(
        "Submit",
        confidence=1.0,
        enabled=None,
        source="ocr",
        x=100,
    )
    hybrid = _element(
        "Submit",
        confidence=0.7,
        source="hybrid",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            minimum_confidence=0.7,
        ),
        (
            ocr,
            hybrid,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is hybrid


def test_complete_source_priority_order_is_deterministic():
    hybrid = _element(
        "Submit",
        source="hybrid",
        x=10,
    )
    accessibility = _element(
        "Submit",
        source="accessibility",
        x=20,
    )
    ocr = _element(
        "Submit",
        enabled=None,
        source="ocr",
        x=30,
    )
    other = _element(
        "Submit",
        source="vision",
        x=40,
    )
    missing_source = _element(
        "Submit",
        source=None,
        x=50,
    )
    blank_source = _element(
        "Submit",
        source=None,
        x=60,
    )
    object.__setattr__(blank_source, "source", "   ")
    target_spec = TargetSpec(
        text="Submit",
        element_types=("button",),
    )

    result = _ground(
        target_spec,
        (
            blank_source,
            missing_source,
            other,
            ocr,
            accessibility,
            hybrid,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is hybrid
    assert [candidate.element.source for candidate in result.candidates[:4]] == [
        "hybrid",
        "accessibility",
        "ocr",
        "vision",
    ]
    assert {
        candidate.element.source
        for candidate in result.candidates[4:]
    } == {
        None,
        "   ",
    }

    for expected, remaining in (
        (
            accessibility,
            (
                blank_source,
                missing_source,
                other,
                ocr,
                accessibility,
            ),
        ),
        (
            ocr,
            (
                blank_source,
                missing_source,
                other,
                ocr,
            ),
        ),
        (
            other,
            (
                blank_source,
                missing_source,
                other,
            ),
        ),
    ):
        priority_result = _ground(
            target_spec,
            tuple(reversed(remaining)),
        )

        assert priority_result.status is GroundingStatus.RESOLVED
        assert priority_result.element is expected

    absent_only = _ground(
        target_spec,
        (
            blank_source,
            missing_source,
        ),
    )

    assert absent_only.status is GroundingStatus.AMBIGUOUS
    assert absent_only.element is None


def test_unique_nearest_candidate_resolves_when_reference_point_is_provided():
    near = _element(
        "Submit",
        x=100,
    )
    far = _element(
        "Submit",
        x=400,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            reference_point=(150, 35),
        ),
        (
            far,
            near,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is near
    assert result.candidates[0].distance == 0.0


def test_equal_distance_candidates_remain_ambiguous():
    left = _element(
        "Submit",
        x=0,
    )
    right = _element(
        "Submit",
        x=100,
    )

    result = _ground(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            reference_point=(100, 35),
        ),
        (
            right,
            left,
        ),
    )

    assert result.status is GroundingStatus.AMBIGUOUS
    assert result.element is None
    assert result.candidates[0].distance == 50.0
    assert result.candidates[1].distance == 50.0


def test_no_semantic_match_returns_not_found():
    result = _ground(
        TargetSpec(
            text="Submit",
        ),
        (
            _element("Cancel"),
        ),
    )

    assert result.status is GroundingStatus.NOT_FOUND
    assert result.element is None
    assert result.candidates == ()


def test_reversing_input_order_produces_same_result():
    first = _element(
        "Submit",
        element_type="button",
        source="ocr",
        enabled=None,
        x=100,
    )
    second = _element(
        "Submit",
        element_type="button",
        source="hybrid",
        x=200,
    )
    target_spec = TargetSpec(
        text="Submit",
        element_types=("button",),
    )

    forward = _ground(
        target_spec,
        (
            first,
            second,
        ),
    )
    reversed_result = _ground(
        target_spec,
        (
            second,
            first,
        ),
    )

    assert reversed_result == forward


def test_input_elements_are_not_mutated():
    elements = [
        _element(
            "Submit",
            source=None,
            enabled=None,
        )
    ]
    before = tuple(elements)

    _ground(
        TargetSpec(
            text="Submit",
        ),
        elements,
    )

    assert tuple(elements) == before


def test_non_finite_bounding_box_data_cannot_become_resolved():
    target = _element("Submit")
    broken_box = object.__new__(BoundingBox)
    object.__setattr__(broken_box, "x", math.nan)
    object.__setattr__(broken_box, "y", 20)
    object.__setattr__(broken_box, "width", 100)
    object.__setattr__(broken_box, "height", 30)
    object.__setattr__(target, "bounding_box", broken_box)

    result = _ground(
        TargetSpec(
            text="Submit",
        ),
        (target,),
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None
    assert result.candidates[0].rejection_reasons == (
        "invalid_bounding_box",
    )


def test_module_has_no_control_tool_application_capture_or_llm_dependency():
    source = inspect.getsource(ui_grounder_module)

    forbidden_terms = (
        "computer_agent.control",
        "computer_agent.tools",
        "application",
        "ScreenCapture",
        "PerceptionEngine",
        "openai",
        "llm",
    )

    assert all(term not in source for term in forbidden_terms)


def test_target_spec_requires_text_or_identifier():
    with pytest.raises(
        ValueError,
        match="TargetSpec requires text or identifier",
    ):
        TargetSpec()


@pytest.mark.parametrize(
    "minimum_confidence",
    [
        True,
        "0.7",
        float("nan"),
        float("inf"),
        -0.01,
        1.01,
    ],
)
def test_target_spec_rejects_invalid_minimum_confidence(
    minimum_confidence,
):
    with pytest.raises(ValueError, match="minimum_confidence"):
        TargetSpec(
            text="Submit",
            minimum_confidence=minimum_confidence,
        )


@pytest.mark.parametrize(
    "reference_point",
    [
        (1,),
        [1, 2],
        (True, 2),
        ("1", 2),
        (math.nan, 2),
        (1, math.inf),
    ],
)
def test_target_spec_rejects_invalid_reference_point(
    reference_point,
):
    with pytest.raises(ValueError, match="reference_point"):
        TargetSpec(
            text="Submit",
            reference_point=reference_point,
        )


def test_non_resolved_result_cannot_contain_actionable_element():
    with pytest.raises(
        ValueError,
        match="non-RESOLVED results must not contain an actionable element",
    ):
        GroundingResult(
            status=GroundingStatus.UNSAFE,
            element=_element("Submit"),
            candidates=(),
            reason="unsafe",
        )


def test_real_web_same_text_resolves_link_role():
    link = _element(
        "Docs",
        element_type="link",
        x=100,
    )
    heading = _element(
        "Docs",
        element_type="heading",
        x=200,
    )
    text = _element(
        "Docs",
        element_type="text",
        x=300,
    )

    result = _ground(
        TargetSpec(
            text="Docs",
            element_types=("link",),
        ),
        (
            heading,
            link,
            text,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is link

    eligible = tuple(
        candidate
        for candidate in result.candidates
        if candidate.eligible
    )

    assert len(eligible) == 1
    assert eligible[0].element is link


def test_real_web_same_text_resolves_heading_role():
    link = _element(
        "Docs",
        element_type="link",
        x=100,
    )
    heading = _element(
        "Docs",
        element_type="heading",
        x=200,
    )
    text = _element(
        "Docs",
        element_type="text",
        x=300,
    )

    result = _ground(
        TargetSpec(
            text="Docs",
            element_types=("heading",),
        ),
        (
            link,
            text,
            heading,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is heading

    eligible = tuple(
        candidate
        for candidate in result.candidates
        if candidate.eligible
    )

    assert len(eligible) == 1
    assert eligible[0].element is heading


def test_real_web_text_field_beats_same_text_static_text():
    text_field = _element(
        "Search This Site",
        element_type="text_field",
        x=100,
    )
    text = _element(
        "Search This Site",
        element_type="text",
        x=200,
    )

    result = _ground(
        TargetSpec(
            text="Search This Site",
            element_types=("text_field",),
        ),
        (
            text,
            text_field,
        ),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is text_field

    eligible = tuple(
        candidate
        for candidate in result.candidates
        if candidate.eligible
    )

    assert len(eligible) == 1
    assert eligible[0].element is text_field
