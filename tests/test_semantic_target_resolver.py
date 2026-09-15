from computer_agent.grounding import (
    GroundingStatus,
    SemanticTargetResolver,
    TargetSpec,
)
from computer_agent.perception.models import BoundingBox, UIElement


def _box(
    x: int,
    y: int,
    width: int = 80,
    height: int = 18,
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    text: str,
    *,
    x: int = 10,
    y: int = 10,
    element_type: str = "static_text",
    source: str = "accessibility",
    value=None,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x, y),
        confidence=0.95,
        text=text,
        value=value,
        enabled=True,
        source=source,
    )


def _identity(value):
    if not isinstance(value, str) or not value.startswith("https://example.test/"):
        return None
    return value.removeprefix("https://example.test"), value


def _resolve(
    target: str,
    *elements: UIElement,
    element_type: str = "static_text",
):
    return SemanticTargetResolver(
        destination_normalizer=_identity,
    ).ground(
        TargetSpec(
            text=target,
            element_types=(element_type,),
        ),
        elements,
    )


def test_adjacent_wrapped_text_resolves_exact_composite():
    result = _resolve(
        "computer science",
        _element("computer", y=10),
        _element("science", y=31),
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element.text == "computer science"


def test_fragments_far_apart_do_not_form_composite():
    result = _resolve(
        "computer science",
        _element("computer", y=10),
        _element("science", y=120),
    )

    assert result.status is GroundingStatus.NOT_FOUND


def test_wrong_reading_order_does_not_match_target():
    result = _resolve(
        "computer science",
        _element("science", y=10),
        _element("computer", y=31),
    )

    assert result.status is GroundingStatus.NOT_FOUND


def test_two_matching_composite_spans_are_ambiguous():
    result = _resolve(
        "computer science",
        _element("computer", y=10),
        _element("science", y=31),
        _element("computer", y=70),
        _element("science", y=91),
    )

    assert result.status is GroundingStatus.AMBIGUOUS


def test_line_break_hyphen_can_be_removed_only_for_exact_target():
    result = _resolve(
        "reinforcement",
        _element("reinforce-", y=10),
        _element("ment", y=31),
    )

    assert result.status is GroundingStatus.RESOLVED


def test_hyphen_removal_that_does_not_match_target_fails():
    result = _resolve(
        "reinforce ment",
        _element("reinforce-", y=10),
        _element("ment", y=31),
    )

    assert result.status is GroundingStatus.NOT_FOUND


def test_meaningful_hyphen_is_preserved_across_wrap():
    result = _resolve(
        "state-of-the-art",
        _element("state-of-", y=10),
        _element("the-art", y=31),
    )

    assert result.status is GroundingStatus.RESOLVED


def test_ocr_only_split_text_is_not_actionable_link():
    result = _resolve(
        "computer science",
        _element(
            "computer",
            y=10,
            element_type="link",
            source="ocr",
            value="https://example.test/computer-science",
        ),
        _element(
            "science",
            y=31,
            element_type="link",
            source="ocr",
            value="https://example.test/computer-science",
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.UNSAFE
    assert result.element is None


def test_multiple_axlink_fragments_with_same_url_resolve_actionably():
    first = _element(
        "computer",
        y=10,
        element_type="link",
        value="https://example.test/computer-science",
    )
    result = _resolve(
        "computer science",
        first,
        _element(
            "science",
            y=31,
            element_type="link",
            value="https://example.test/computer-science",
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is first


def test_multiple_axlink_fragments_with_different_urls_are_ambiguous():
    result = _resolve(
        "computer science",
        _element(
            "computer",
            y=10,
            element_type="link",
            value="https://example.test/computer",
        ),
        _element(
            "science",
            y=31,
            element_type="link",
            value="https://example.test/science",
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.AMBIGUOUS


def test_single_exact_link_uses_normal_grounding():
    link = _element(
        "perception",
        element_type="link",
        value="https://example.test/perception",
    )

    result = _resolve(
        "perception",
        link,
        element_type="link",
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is link
    assert result.reason == "resolved by text"


def test_duplicate_same_text_links_with_same_url_collapse_to_topmost_leftmost():
    lower = _element(
        "perception",
        x=20,
        y=80,
        element_type="link",
        value="https://example.test/perception",
    )
    chosen = _element(
        "perception",
        x=10,
        y=20,
        element_type="link",
        value="https://example.test/perception",
    )

    result = _resolve(
        "perception",
        lower,
        chosen,
        element_type="link",
    )

    assert result.status is GroundingStatus.RESOLVED
    assert result.element is chosen
    assert "accessibility URL" in result.reason


def test_duplicate_same_text_links_with_different_urls_stay_ambiguous():
    result = _resolve(
        "perception",
        _element(
            "perception",
            y=20,
            element_type="link",
            value="https://example.test/perception",
        ),
        _element(
            "perception",
            y=50,
            element_type="link",
            value="https://example.test/perception-journal",
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.AMBIGUOUS


def test_duplicate_same_text_link_missing_url_stays_ambiguous():
    result = _resolve(
        "perception",
        _element(
            "perception",
            y=20,
            element_type="link",
            value="https://example.test/perception",
        ),
        _element(
            "perception",
            y=50,
            element_type="link",
            value=None,
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.AMBIGUOUS


def test_duplicate_same_text_wrong_domain_stays_ambiguous():
    result = _resolve(
        "perception",
        _element(
            "perception",
            y=20,
            element_type="link",
            value="https://example.test/perception",
        ),
        _element(
            "perception",
            y=50,
            element_type="link",
            value="https://wrong.example/perception",
        ),
        element_type="link",
    )

    assert result.status is GroundingStatus.AMBIGUOUS
