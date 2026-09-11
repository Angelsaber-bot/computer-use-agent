from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import inspect
import math
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.grounding import GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.verification import (
    StateObservationCandidate,
    StateObservationResult,
    StateObservationStatus,
    StateObserver,
)
import computer_agent.verification.state_observation as state_observation_module


def _box(
    *,
    x: int = 10,
    y: int = 20,
    width: int = 100,
    height: int = 30,
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    text: str | None = "Target",
    *,
    identifier: str | None = None,
    element_type: str = "button",
    confidence: float = 0.95,
    enabled: bool | None = True,
    source: str | None = "accessibility",
    x: int = 10,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x=x),
        confidence=confidence,
        text=text,
        identifier=identifier,
        enabled=enabled,
        source=source,
    )


def _snapshot(
    *elements: UIElement,
    warnings: tuple[str, ...] = (),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic-state-observer.png"),
            pixel_width=300,
            pixel_height=180,
            screen_width=300,
            screen_height=180,
            captured_at=datetime(
                2026,
                9,
                10,
                12,
                0,
                tzinfo=timezone.utc,
            ),
        ),
        image=Image.new("RGB", (300, 180)),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=warnings,
    )


def _observe(target: TargetSpec, *elements: UIElement) -> StateObservationResult:
    return StateObserver().observe(
        snapshot=_snapshot(*elements),
        target=target,
    )


def _observe_with_warnings(
    target: TargetSpec,
    *elements: UIElement,
    warnings: tuple[str, ...] = ("ocr unavailable",),
) -> StateObservationResult:
    return StateObserver().observe(
        snapshot=_snapshot(*elements, warnings=warnings),
        target=target,
    )


def test_models_are_frozen_slotted_and_validate_fields():
    element = _element()
    target = TargetSpec(text="Target")
    candidate = StateObservationCandidate(
        element=element,
        match_basis="text",
        predicate_match=True,
        distance=1,
    )
    result = StateObservationResult(
        status=StateObservationStatus.PRESENT,
        target=target,
        candidates=(candidate,),
        reason="present",
    )

    assert not hasattr(candidate, "__dict__")
    assert not hasattr(result, "__dict__")
    assert candidate.distance == 1.0
    assert candidate.reliable_match is True
    with pytest.raises(FrozenInstanceError):
        candidate.match_basis = "identifier"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"element": object()}, "element"),
        ({"match_basis": ""}, "match_basis"),
        ({"predicate_match": "yes"}, "predicate_match"),
        ({"mismatch_reasons": ["wrong"]}, "mismatch_reasons"),
        ({"mismatch_reasons": ("",)}, "mismatch_reasons"),
        ({"uncertainty_reasons": ["low"]}, "uncertainty_reasons"),
        ({"distance": math.nan}, "distance"),
    ],
)
def test_candidate_rejects_invalid_fields(kwargs, message):
    arguments = {
        "element": _element(),
        "match_basis": "text",
        "predicate_match": True,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        StateObservationCandidate(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "present"}, "status"),
        ({"target": object()}, "target"),
        ({"candidates": []}, "candidates"),
        ({"candidates": (object(),)}, "candidates"),
        ({"reason": ""}, "reason"),
    ],
)
def test_result_rejects_invalid_fields(kwargs, message):
    arguments = {
        "status": StateObservationStatus.PRESENT,
        "target": TargetSpec(text="Target"),
        "candidates": (),
        "reason": "present",
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        StateObservationResult(**arguments)


def test_observer_rejects_invalid_inputs_and_unsupported_reference_point():
    observer = StateObserver()
    with pytest.raises(ValueError, match="snapshot"):
        observer.observe(snapshot=object(), target=TargetSpec(text="Target"))

    with pytest.raises(ValueError, match="target"):
        observer.observe(snapshot=_snapshot(), target=object())

    with pytest.raises(ValueError, match="reference_point"):
        observer.observe(
            snapshot=_snapshot(_element()),
            target=TargetSpec(text="Target", reference_point=(10, 20)),
        )

    with pytest.raises(ValueError, match="elements"):
        observer.observe_elements(
            target=TargetSpec(text="Target"),
            elements=(object(),),
        )


def test_text_matching_is_normalized_exactly():
    result = _observe(
        TargetSpec(text="target input", element_types=("button",)),
        _element(" TARGET   INPUT "),
    )

    assert result.status is StateObservationStatus.PRESENT
    assert result.candidates[0].match_basis == "text"


def test_identifier_only_target_matches_identifier():
    result = _observe(
        TargetSpec(identifier="primary", element_types=("button",)),
        _element("Other", identifier="primary"),
    )

    assert result.status is StateObservationStatus.PRESENT
    assert result.candidates[0].match_basis == "identifier"


def test_text_and_identifier_are_conjunctive_on_the_same_element():
    result = _observe(
        TargetSpec(
            text="Submit",
            identifier="primary-submit",
            element_types=("button",),
        ),
        _element("Submit", identifier="other"),
        _element("Cancel", identifier="primary-submit"),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert [candidate.match_basis for candidate in result.candidates] == [
        "text",
        "identifier",
    ]
    assert result.candidates[0].mismatch_reasons == ("identifier_mismatch",)
    assert result.candidates[1].mismatch_reasons == ("text_mismatch",)


def test_same_text_wrong_element_type_is_absent_for_state_observation():
    result = _observe(
        TargetSpec(text="Wikipedia The Free Encyclopedia", element_types=("heading",)),
        _element(
            "Wikipedia The Free Encyclopedia",
            element_type="link",
            confidence=1.0,
        ),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert result.candidates[0].predicate_match is False
    assert result.candidates[0].mismatch_reasons == (
        "incompatible_element_type",
    )


def test_disabled_matching_element_still_proves_present():
    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Submit", element_type="button", enabled=False, confidence=1.0),
    )

    assert result.status is StateObservationStatus.PRESENT
    assert result.candidates[0].reliable_match is True


def test_multiple_valid_elements_prove_present_without_uniqueness():
    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Submit", element_type="button", x=10),
        _element("Submit", element_type="button", x=140),
    )

    assert result.status is StateObservationStatus.PRESENT
    assert len(result.candidates) == 2
    assert all(candidate.reliable_match for candidate in result.candidates)


def test_low_confidence_full_match_is_inconclusive():
    result = _observe(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            minimum_confidence=0.7,
        ),
        _element("Submit", element_type="button", confidence=0.69),
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates[0].predicate_match is True
    assert result.candidates[0].uncertainty_reasons == ("low_confidence",)
    assert "uncertain possible matches" in result.reason


def test_invalid_confidence_full_match_is_inconclusive():
    element = _element("Submit", element_type="button")
    object.__setattr__(element, "confidence", math.nan)

    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        element,
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates[0].predicate_match is True
    assert result.candidates[0].uncertainty_reasons == ("invalid_confidence",)


def test_no_text_or_identifier_match_is_absent_within_snapshot_scope():
    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Cancel", element_type="button"),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert result.candidates == ()
    assert "supplied element collection" in result.reason


def test_warning_free_snapshot_no_match_proves_absent_within_snapshot_scope():
    result = StateObserver().observe(
        snapshot=_snapshot(_element("Cancel", element_type="button")),
        target=TargetSpec(text="Submit", element_types=("button",)),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert result.candidates == ()


def test_snapshot_warnings_make_no_match_inconclusive_not_absent():
    result = _observe_with_warnings(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Cancel", element_type="button"),
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates == ()
    assert "absence could not be established" in result.reason
    assert "warnings" in result.reason
    assert "ocr unavailable" in result.reason


def test_snapshot_warnings_make_wrong_role_absence_inconclusive():
    result = _observe_with_warnings(
        TargetSpec(text="Submit", element_types=("heading",)),
        _element("Submit", element_type="link"),
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates[0].mismatch_reasons == (
        "incompatible_element_type",
    )
    assert "absence could not be established" in result.reason


def test_snapshot_warnings_do_not_block_reliable_present_evidence():
    result = _observe_with_warnings(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Submit", element_type="button", confidence=1.0),
    )

    assert result.status is StateObservationStatus.PRESENT
    assert result.candidates[0].reliable_match is True


def test_snapshot_warnings_keep_low_confidence_match_inconclusive():
    result = _observe_with_warnings(
        TargetSpec(
            text="Submit",
            element_types=("button",),
            minimum_confidence=0.7,
        ),
        _element("Submit", element_type="button", confidence=0.2),
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates[0].uncertainty_reasons == ("low_confidence",)
    assert "uncertain possible matches" in result.reason


def test_observe_elements_no_match_remains_absent_without_snapshot_metadata():
    result = StateObserver().observe_elements(
        target=TargetSpec(text="Submit", element_types=("button",)),
        elements=(_element("Cancel", element_type="button"),),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert result.candidates == ()
    assert "supplied element collection" in result.reason


def test_identifier_match_with_required_text_conflict_is_absent():
    result = _observe(
        TargetSpec(
            text="Submit",
            identifier="primary-submit",
            element_types=("button",),
        ),
        _element("Cancel", identifier="primary-submit", element_type="button"),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert result.candidates[0].mismatch_reasons == ("text_mismatch",)


def test_invalid_or_missing_geometry_does_not_block_semantic_presence():
    invalid_box = object.__new__(BoundingBox)
    object.__setattr__(invalid_box, "x", math.nan)
    object.__setattr__(invalid_box, "y", 20)
    object.__setattr__(invalid_box, "width", 100)
    object.__setattr__(invalid_box, "height", 30)
    invalid_geometry = _element("Submit", element_type="button", x=10)
    missing_geometry = _element("Submit", element_type="button", x=140)
    object.__setattr__(invalid_geometry, "bounding_box", invalid_box)
    object.__setattr__(missing_geometry, "bounding_box", None)

    invalid_result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        invalid_geometry,
    )
    missing_result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        missing_geometry,
    )

    assert invalid_result.status is StateObservationStatus.PRESENT
    assert missing_result.status is StateObservationStatus.PRESENT


def test_multiple_wrong_role_same_text_candidates_are_absent_not_ambiguous():
    result = _observe(
        TargetSpec(text="Submit", element_types=("heading",)),
        _element("Submit", element_type="link", x=10),
        _element("Submit", element_type="button", x=140),
    )

    assert result.status is StateObservationStatus.ABSENT
    assert len(result.candidates) == 2
    assert all(
        candidate.mismatch_reasons == ("incompatible_element_type",)
        for candidate in result.candidates
    )


def test_uncertain_possible_match_prevents_absent_result():
    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        _element("Submit", element_type="link", x=10),
        _element("Submit", element_type="button", confidence=0.2, x=140),
    )

    assert result.status is StateObservationStatus.INCONCLUSIVE
    assert result.candidates[0].mismatch_reasons == (
        "incompatible_element_type",
    )
    assert result.candidates[1].uncertainty_reasons == ("low_confidence",)


def test_candidate_diagnostics_are_deterministic():
    first = _element("Submit", element_type="link", source="accessibility", x=10)
    second = _element(
        "Submit",
        element_type="button",
        confidence=0.2,
        source="ocr",
        x=140,
    )

    result = _observe(
        TargetSpec(text="Submit", element_types=("button",)),
        first,
        second,
    )

    assert [
        (
            candidate.element,
            candidate.match_basis,
            candidate.predicate_match,
            candidate.mismatch_reasons,
            candidate.uncertainty_reasons,
            candidate.distance,
        )
        for candidate in result.candidates
    ] == [
        (
            first,
            "text",
            False,
            ("incompatible_element_type",),
            (),
            None,
        ),
        (
            second,
            "text",
            True,
            (),
            ("low_confidence",),
            None,
        ),
    ]


def test_wrong_role_differs_intentionally_from_action_grounding():
    element = _element(
        "Wikipedia The Free Encyclopedia",
        element_type="link",
        confidence=1.0,
    )
    target = TargetSpec(
        text="Wikipedia The Free Encyclopedia",
        element_types=("heading",),
    )

    action_grounding = UIGrounder().ground(target, (element,))
    state_observation = _observe(target, element)

    assert action_grounding.status is GroundingStatus.UNSAFE
    assert state_observation.status is StateObservationStatus.ABSENT


def test_state_observer_is_exported_from_verification_package():
    import computer_agent.verification as verification

    assert verification.StateObserver is StateObserver
    assert verification.StateObservationStatus is StateObservationStatus
    assert verification.StateObservationCandidate is StateObservationCandidate
    assert verification.StateObservationResult is StateObservationResult


def test_module_does_not_depend_on_action_grounding_or_runtime_execution():
    source = inspect.getsource(state_observation_module)

    forbidden_terms = (
        "UIGrounder",
        "ActionGrounder",
        "ActionVerifier",
        "ActionRecovery",
        "AgentLoop",
        "computer_agent.control",
        "computer_agent.tools",
        "openai",
    )

    assert all(term not in source for term in forbidden_terms)
