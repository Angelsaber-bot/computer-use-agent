import pytest

from computer_agent.perception import (
    BoundingBox,
    UIElement,
    UIElementFusion,
    normalize_ui_text,
    smaller_area_overlap_ratio,
)


def _box(
    x=10,
    y=20,
    width=40,
    height=20,
):
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    text,
    *,
    x=10,
    y=20,
    width=40,
    height=20,
    element_type="text",
    confidence=0.75,
    identifier=None,
    value=None,
    enabled=None,
    focused=None,
    selected=None,
    source=None,
):
    return UIElement(
        element_type=element_type,
        bounding_box=_box(
            x=x,
            y=y,
            width=width,
            height=height,
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


@pytest.mark.parametrize(
    "text",
    [
        "NATIVE_BUTTON_12",
        "native button 12",
        " NATIVE   BUTTON_12 ",
        "ＮＡＴＩＶＥ＿ＢＵＴＴＯＮ＿１２",
    ],
)
def test_normalize_ui_text_matches_case_underscores_and_spaces(text):
    assert normalize_ui_text(text) == "native button 12"


def test_normalize_ui_text_preserves_meaningful_punctuation():
    assert normalize_ui_text(" Save-Now! ") == "save-now!"


def test_smaller_area_overlap_ratio_uses_smaller_box_area():
    large = BoundingBox(
        x=0,
        y=0,
        width=100,
        height=100,
    )
    small = BoundingBox(
        x=50,
        y=50,
        width=20,
        height=20,
    )

    assert smaller_area_overlap_ratio(
        large,
        small,
    ) == 1.0


def test_fusion_merges_native_control_duplicate():
    accessibility_element = _element(
        "NATIVE_BUTTON_12",
        element_type="button",
        confidence=0.4,
    )
    ocr_element = _element(
        "native button 12",
        x=14,
        y=22,
        width=32,
        height=16,
        confidence=0.9,
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [ocr_element],
    )

    assert fused == (
        UIElement(
            element_type="button",
            bounding_box=accessibility_element.bounding_box,
            confidence=0.9,
            text="NATIVE_BUTTON_12",
            source="hybrid",
        ),
    )


def test_fusion_preserves_accessibility_only_element_with_default_source():
    accessibility_element = _element(
        "TARGET_INPUT_12",
        element_type="textbox",
        confidence=0.8,
        identifier="hybrid-target-input",
        enabled=True,
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [],
    )

    assert fused == (
        UIElement(
            element_type="textbox",
            bounding_box=accessibility_element.bounding_box,
            confidence=0.8,
            text="TARGET_INPUT_12",
            identifier="hybrid-target-input",
            enabled=True,
            source="accessibility",
        ),
    )
    assert accessibility_element.source is None


def test_fusion_preserves_ocr_only_element_with_default_source():
    ocr_element = _element(
        "CANVAS_ACTION_12",
        confidence=0.68,
    )

    fused = UIElementFusion().fuse(
        [],
        [ocr_element],
    )

    assert fused == (
        UIElement(
            element_type="text",
            bounding_box=ocr_element.bounding_box,
            confidence=0.68,
            text="CANVAS_ACTION_12",
            source="ocr",
        ),
    )
    assert ocr_element.source is None


def test_fusion_preserves_accessibility_semantic_metadata():
    accessibility_element = _element(
        "Submit",
        element_type="button",
        confidence=0.95,
        identifier="submit-button",
        value="Submit",
        enabled=True,
        focused=False,
        selected=True,
        source="accessibility",
    )
    ocr_element = _element(
        "submit",
        confidence=0.6,
        source="ocr",
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [ocr_element],
    )

    assert fused == (
        UIElement(
            element_type="button",
            bounding_box=accessibility_element.bounding_box,
            confidence=0.95,
            text="Submit",
            identifier="submit-button",
            value="Submit",
            enabled=True,
            focused=False,
            selected=True,
            source="hybrid",
        ),
    )


def test_fusion_consumes_multiple_ocr_matches_and_uses_best_confidence():
    accessibility_element = _element(
        "Submit",
        element_type="button",
        confidence=0.45,
    )
    low_confidence_ocr = _element(
        "submit",
        x=12,
        y=20,
        confidence=0.55,
    )
    high_confidence_ocr = _element(
        "SUBMIT",
        x=14,
        y=22,
        confidence=0.87,
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [
            low_confidence_ocr,
            high_confidence_ocr,
        ],
    )

    assert len(fused) == 1
    assert fused[0].confidence == 0.87
    assert fused[0].source == "hybrid"


def test_fusion_deduplicates_overlapping_ocr_only_elements():
    low_confidence_ocr = _element(
        "Canvas Action",
        x=50,
        y=50,
        confidence=0.42,
    )
    high_confidence_ocr = _element(
        "CANVAS_ACTION",
        x=54,
        y=52,
        confidence=0.91,
    )

    fused = UIElementFusion().fuse(
        [],
        [
            low_confidence_ocr,
            high_confidence_ocr,
        ],
    )

    assert fused == (
        UIElement(
            element_type="text",
            bounding_box=high_confidence_ocr.bounding_box,
            confidence=0.91,
            text="CANVAS_ACTION",
            source="ocr",
        ),
    )


def test_fusion_does_not_merge_distant_identical_text():
    first_ocr = _element(
        "Same",
        x=0,
        y=0,
    )
    second_ocr = _element(
        "same",
        x=200,
        y=200,
    )

    fused = UIElementFusion().fuse(
        [],
        [
            first_ocr,
            second_ocr,
        ],
    )

    assert [
        element.text for element in fused
    ] == ["Same", "same"]
    assert [
        element.source for element in fused
    ] == ["ocr", "ocr"]


def test_fusion_does_not_merge_different_overlapping_text():
    accessibility_element = _element(
        "Submit",
        element_type="button",
    )
    ocr_element = _element(
        "Cancel",
        x=12,
        y=22,
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [ocr_element],
    )

    assert [
        element.text for element in fused
    ] == ["Submit", "Cancel"]
    assert [
        element.source for element in fused
    ] == ["accessibility", "ocr"]


def test_fusion_does_not_merge_empty_text_elements():
    accessibility_element = _element(
        "",
        element_type="button",
    )
    first_ocr = _element(
        "",
        x=12,
        y=22,
    )
    second_ocr = _element(
        "   ",
        x=14,
        y=24,
    )

    fused = UIElementFusion().fuse(
        [accessibility_element],
        [
            first_ocr,
            second_ocr,
        ],
    )

    assert [
        element.text for element in fused
    ] == ["", "", "   "]
    assert [
        element.source for element in fused
    ] == ["accessibility", "ocr", "ocr"]


def test_fusion_preserves_deterministic_ordering():
    matched_accessibility = _element(
        "Primary",
        x=0,
        y=0,
    )
    accessibility_only = _element(
        "Accessibility Only",
        x=300,
        y=0,
    )
    matching_ocr = _element(
        "primary",
        x=2,
        y=2,
        confidence=0.7,
    )
    duplicate_low = _element(
        "Duplicate OCR",
        x=100,
        y=0,
        confidence=0.2,
    )
    middle_ocr = _element(
        "Middle OCR",
        x=200,
        y=0,
        confidence=0.5,
    )
    duplicate_high = _element(
        "duplicate ocr",
        x=102,
        y=2,
        confidence=0.9,
    )
    last_ocr = _element(
        "Last OCR",
        x=400,
        y=0,
        confidence=0.6,
    )

    fused = UIElementFusion().fuse(
        [
            matched_accessibility,
            accessibility_only,
        ],
        [
            matching_ocr,
            duplicate_low,
            middle_ocr,
            duplicate_high,
            last_ocr,
        ],
    )

    assert [
        element.text for element in fused
    ] == [
        "Primary",
        "Accessibility Only",
        "Middle OCR",
        "duplicate ocr",
        "Last OCR",
    ]
    assert [
        element.source for element in fused
    ] == [
        "hybrid",
        "accessibility",
        "ocr",
        "ocr",
        "ocr",
    ]


@pytest.mark.parametrize(
    ("minimum_overlap_ratio", "error_message"),
    [
        (True, "minimum_overlap_ratio must be numeric"),
        (False, "minimum_overlap_ratio must be numeric"),
        ("0.5", "minimum_overlap_ratio must be numeric"),
        (float("nan"), "minimum_overlap_ratio must be finite"),
        (float("inf"), "minimum_overlap_ratio must be finite"),
        (float("-inf"), "minimum_overlap_ratio must be finite"),
        (-0.01, "minimum_overlap_ratio must be between 0.0 and 1.0"),
        (1.01, "minimum_overlap_ratio must be between 0.0 and 1.0"),
    ],
)
def test_fusion_rejects_invalid_overlap_thresholds(
    minimum_overlap_ratio,
    error_message,
):
    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        UIElementFusion(
            minimum_overlap_ratio=minimum_overlap_ratio,
        )
