import pytest

from computer_agent.agent.web_recovery import (
    WebRecoveryDecision,
    decide_failed_grounding_recovery,
)
from computer_agent.grounding import (
    GroundingCandidate,
    GroundingResult,
    GroundingStatus,
)
from computer_agent.perception import BoundingBox, UIElement


def _element():
    return UIElement(
        element_type="text_field",
        bounding_box=BoundingBox(
            x=10,
            y=10,
            width=100,
            height=30,
        ),
        confidence=1.0,
        text="Search This Site",
        enabled=True,
        source="accessibility",
    )


def _candidate(
    *rejection_reasons,
):
    return GroundingCandidate(
        element=_element(),
        match_basis="text",
        rejection_reasons=tuple(rejection_reasons),
    )


def _grounding(
    status,
    *,
    candidates=(),
):
    return GroundingResult(
        status=status,
        element=_element() if status is GroundingStatus.RESOLVED else None,
        candidates=tuple(candidates),
        reason=f"{status.value} test result",
    )


def test_not_found_allows_viewport_search():
    result = decide_failed_grounding_recovery(
        _grounding(GroundingStatus.NOT_FOUND)
    )

    assert result.decision is WebRecoveryDecision.VIEWPORT_SEARCH
    assert "not found" in result.reason
    assert "viewport search is eligible" in result.reason


def test_unsafe_only_outside_viewport_allows_viewport_search():
    result = decide_failed_grounding_recovery(
        _grounding(
            GroundingStatus.UNSAFE,
            candidates=(
                _candidate("outside_viewport"),
                _candidate("outside_viewport"),
            ),
        )
    )

    assert result.decision is WebRecoveryDecision.VIEWPORT_SEARCH
    assert "outside the viewport" in result.reason


def test_unsafe_outside_viewport_combined_with_disabled_blocks():
    result = decide_failed_grounding_recovery(
        _grounding(
            GroundingStatus.UNSAFE,
            candidates=(
                _candidate("outside_viewport", "disabled"),
            ),
        )
    )

    assert result.decision is WebRecoveryDecision.BLOCK


def test_unsafe_with_no_candidates_blocks():
    result = decide_failed_grounding_recovery(
        _grounding(
            GroundingStatus.UNSAFE,
            candidates=(),
        )
    )

    assert result.decision is WebRecoveryDecision.BLOCK


def test_unsafe_mixed_candidate_rejections_block():
    result = decide_failed_grounding_recovery(
        _grounding(
            GroundingStatus.UNSAFE,
            candidates=(
                _candidate("outside_viewport"),
                _candidate("disabled"),
            ),
        )
    )

    assert result.decision is WebRecoveryDecision.BLOCK


@pytest.mark.parametrize(
    "rejection_reason",
    [
        "disabled",
        "incompatible_element_type",
        "low_confidence",
        "invalid_confidence",
        "identifier_text_conflict",
        "invalid_bounding_box",
    ],
)
def test_unsafe_non_viewport_rejection_blocks(rejection_reason):
    result = decide_failed_grounding_recovery(
        _grounding(
            GroundingStatus.UNSAFE,
            candidates=(
                _candidate(rejection_reason),
            ),
        )
    )

    assert result.decision is WebRecoveryDecision.BLOCK


def test_ambiguous_blocks():
    result = decide_failed_grounding_recovery(
        _grounding(GroundingStatus.AMBIGUOUS)
    )

    assert result.decision is WebRecoveryDecision.BLOCK


def test_resolved_blocks():
    result = decide_failed_grounding_recovery(
        _grounding(GroundingStatus.RESOLVED)
    )

    assert result.decision is WebRecoveryDecision.BLOCK


def test_invalid_input_rejected():
    with pytest.raises(ValueError, match="grounding_result"):
        decide_failed_grounding_recovery(object())
