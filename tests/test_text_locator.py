import pytest

from computer_agent.perception.models import (
    BoundingBox,
    UIElement,
)
from computer_agent.perception.text_locator import (
    TextTargetLocator,
)


def make_element(text: str) -> UIElement:
    return UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=100,
            y=200,
            width=80,
            height=30,
        ),
        confidence=0.95,
        text=text,
    )


def make_custom_element(
    text: str | None,
    box: BoundingBox,
) -> UIElement:
    return UIElement(
        element_type="text",
        bounding_box=box,
        confidence=0.95,
        text=text,
    )


def test_find_all_exact_match():
    elements = (
        make_element("Next"),
        make_element("Back"),
        make_element("Next"),
    )

    matches = TextTargetLocator.find_all(
        elements,
        "Next",
    )

    assert len(matches) == 2


def test_find_all_is_case_insensitive_by_default():
    elements = (
        make_element("NEXT"),
        make_element("next"),
    )

    matches = TextTargetLocator.find_all(
        elements,
        "Next",
    )

    assert len(matches) == 2


def test_find_all_can_be_case_sensitive():
    elements = (
        make_element("Next"),
        make_element("next"),
    )

    matches = TextTargetLocator.find_all(
        elements,
        "Next",
        case_sensitive=True,
    )

    assert len(matches) == 1
    assert matches[0].text == "Next"


def test_find_first_returns_first_match():
    first = make_element("Next")
    second = make_element("Next")

    result = TextTargetLocator.find_first(
        (first, second),
        "Next",
    )

    assert result is first


def test_find_first_returns_none_when_missing():
    result = TextTargetLocator.find_first(
        (make_element("Back"),),
        "Next",
    )

    assert result is None


def test_partial_matching_is_disabled_by_default():
    elements = (
        make_element("text='computer_agent',"),
    )

    matches = TextTargetLocator.find_all(
        elements,
        "computer_agent",
    )

    assert matches == ()


def test_partial_matching_finds_quoted_diagnostic_text():
    element = make_element("text='computer_agent',")

    matches = TextTargetLocator.find_all(
        (element,),
        "computer_agent",
        partial_match=True,
    )

    assert matches == (element,)


def test_partial_matching_finds_parenthesized_text():
    element = make_element("(computer_agent)")

    matches = TextTargetLocator.find_all(
        (element,),
        "computer_agent",
        partial_match=True,
    )

    assert matches == (element,)


def test_partial_matching_is_case_insensitive_by_default():
    element = make_element("(COMPUTER_AGENT)")

    matches = TextTargetLocator.find_all(
        (element,),
        "computer_agent",
        partial_match=True,
    )

    assert matches == (element,)


def test_find_first_supports_partial_matching():
    first = make_element("Ocomputer_agent")
    second = make_element("(computer_agent)")

    result = TextTargetLocator.find_first(
        (first, second),
        "computer_agent",
        partial_match=True,
    )

    assert result is first


def test_extract_target_exact_match_preserves_original_box():
    element = make_element("computer_agent")

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted == element
    assert extracted is not element
    assert extracted.bounding_box is element.bounding_box


def test_extract_target_excludes_parentheses():
    element = make_element("(computer_agent)")

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted == UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=105,
            y=200,
            width=70,
            height=30,
        ),
        confidence=0.95,
        text="computer_agent",
    )


def test_extract_target_excludes_prefix():
    element = make_element("Ocomputer_agent")

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted == UIElement(
        element_type="text",
        bounding_box=BoundingBox(
            x=105,
            y=200,
            width=75,
            height=30,
        ),
        confidence=0.95,
        text="computer_agent",
    )


def test_extract_target_from_long_path_uses_substring_center():
    box = BoundingBox(
        x=10,
        y=20,
        width=240,
        height=30,
    )
    element = make_custom_element(
        "longprefix/computer_agent/x",
        box,
    )

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted.bounding_box == BoundingBox(
        x=107,
        y=20,
        width=126,
        height=30,
    )
    assert extracted.bounding_box.width < box.width
    assert extracted.center[0] > box.center[0]


def test_extract_target_is_case_insensitive_by_default():
    element = make_element("(COMPUTER_AGENT)")

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted.text == "COMPUTER_AGENT"


def test_extract_target_supports_case_sensitive_matching():
    element = make_element("(COMPUTER_AGENT)")

    extracted = TextTargetLocator.extract_target(
        element,
        "COMPUTER_AGENT",
        case_sensitive=True,
    )
    missing = TextTargetLocator.extract_target(
        element,
        "computer_agent",
        case_sensitive=True,
    )

    assert extracted.text == "COMPUTER_AGENT"
    assert missing is None


def test_extract_target_returns_none_when_missing():
    result = TextTargetLocator.extract_target(
        make_element("other_text"),
        "computer_agent",
    )

    assert result is None


def test_extract_target_returns_none_when_element_text_is_none():
    element = make_custom_element(
        None,
        BoundingBox(
            x=100,
            y=200,
            width=80,
            height=30,
        ),
    )

    result = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert result is None


@pytest.mark.parametrize(
    "target_text",
    [
        "",
        "   ",
        123,
    ],
)
def test_extract_target_rejects_invalid_target_text(target_text):
    with pytest.raises(
        ValueError,
        match="target_text must be a non-empty string",
    ):
        TextTargetLocator.extract_target(
            make_element("computer_agent"),
            target_text,
        )


def test_extract_target_rejects_invalid_element_type():
    with pytest.raises(
        ValueError,
        match="element must be a UIElement",
    ):
        TextTargetLocator.extract_target(
            "not an element",
            "computer_agent",
        )


@pytest.mark.parametrize(
    "text",
    [
        "(computer_agent)",
        "Ocomputer_agent",
        "longprefix/computer_agent/x",
        "text='computer_agent',",
    ],
)
def test_extract_target_box_stays_inside_original_box(text):
    element = make_element(text)

    extracted = TextTargetLocator.extract_target(
        element,
        "computer_agent",
    )

    assert extracted.bounding_box.width > 0
    assert element.bounding_box.left <= extracted.bounding_box.left
    assert extracted.bounding_box.right <= element.bounding_box.right
    assert extracted.bounding_box.y == element.bounding_box.y
    assert extracted.bounding_box.height == element.bounding_box.height
