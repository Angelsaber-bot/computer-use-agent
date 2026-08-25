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
