from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.verification import (
    ActionVerificationStatus,
    ActionVerifier,
    PresenceExpectation,
    StateObservationStatus,
    StateObserver,
    StateTransitionVerificationResult,
    StateTransitionVerifier,
    StateVerificationStatus,
    UIStateCondition,
    UIStateConditionEvaluation,
    VerificationSpec,
)


def _time(seconds: int = 0):
    return datetime(
        2026,
        9,
        9,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _box(x: int = 10) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=20,
        width=100,
        height=30,
    )


def _element(
    *,
    text: str = "DIALOG",
    identifier: str | None = None,
    element_type: str = "button",
    enabled: bool = True,
    confidence: float = 0.95,
    x: int = 10,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x),
        confidence=confidence,
        text=text,
        identifier=identifier,
        enabled=enabled,
        source="accessibility",
    )


def _snapshot(*elements, seconds: int = 0) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic-state-transition.png"),
            pixel_width=300,
            pixel_height=180,
            screen_width=300,
            screen_height=180,
            captured_at=_time(seconds),
        ),
        image=Image.new("RGB", (300, 180)),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=(),
    )


def _warning_snapshot(*elements, seconds: int = 0) -> PerceptionSnapshot:
    snapshot = _snapshot(*elements, seconds=seconds)
    object.__setattr__(snapshot, "warnings", ("ocr degraded",))
    return snapshot


def _snapshot_for_status(
    status: GroundingStatus,
    *,
    text: str = "DIALOG",
    seconds: int = 0,
) -> PerceptionSnapshot:
    if status is GroundingStatus.NOT_FOUND:
        return _snapshot(seconds=seconds)

    if status is GroundingStatus.RESOLVED:
        return _snapshot(_element(text=text), seconds=seconds)

    if status is GroundingStatus.AMBIGUOUS:
        return _snapshot(
            _element(text=text, x=10),
            _element(text=text, x=140),
            seconds=seconds,
        )

    if status is GroundingStatus.UNSAFE:
        return _snapshot(_element(text=text, enabled=False), seconds=seconds)

    raise AssertionError(f"unknown status: {status}")


def _target(text: str = "DIALOG") -> TargetSpec:
    return TargetSpec(text=text, element_types=("button",))


def _identifier_target(identifier: str = "primary-submit") -> TargetSpec:
    return TargetSpec(identifier=identifier, element_types=("button",))


def _condition(
    expectation: PresenceExpectation,
    *,
    text: str = "DIALOG",
) -> UIStateCondition:
    return UIStateCondition(
        target=_target(text),
        expectation=expectation,
    )


def _spec(
    *,
    before=(),
    after=(),
) -> VerificationSpec:
    return VerificationSpec(
        before_conditions=tuple(before),
        after_conditions=tuple(after),
    )


def _verify(
    *,
    before_snapshot=None,
    after_snapshot=None,
    verification_spec=None,
) -> StateTransitionVerificationResult:
    return StateTransitionVerifier().verify(
        before_snapshot=before_snapshot or _snapshot(seconds=0),
        after_snapshot=after_snapshot or _snapshot(seconds=1),
        verification_spec=verification_spec
        or _spec(after=(_condition(PresenceExpectation.PRESENT),)),
    )


def _verify_with_state_observer(
    *,
    before_snapshot=None,
    after_snapshot=None,
    verification_spec=None,
) -> StateTransitionVerificationResult:
    return StateTransitionVerifier(state_observer=StateObserver()).verify(
        before_snapshot=before_snapshot or _snapshot(seconds=0),
        after_snapshot=after_snapshot or _snapshot(seconds=1),
        verification_spec=verification_spec
        or _spec(after=(_condition(PresenceExpectation.PRESENT),)),
    )


class RecordingStateObserver:
    def __init__(self) -> None:
        self.observer = StateObserver()
        self.observe_calls = []
        self.observe_elements_calls = []

    def observe(self, **kwargs):
        self.observe_calls.append(kwargs)
        return self.observer.observe(**kwargs)

    def observe_elements(self, **kwargs):
        self.observe_elements_calls.append(kwargs)
        raise AssertionError("StateTransitionVerifier must call observe()")


@pytest.mark.parametrize(
    ("grounding_status", "expected_status"),
    [
        (GroundingStatus.RESOLVED, StateVerificationStatus.VERIFIED),
        (GroundingStatus.NOT_FOUND, StateVerificationStatus.FAILED),
        (GroundingStatus.AMBIGUOUS, StateVerificationStatus.INCONCLUSIVE),
        (GroundingStatus.UNSAFE, StateVerificationStatus.INCONCLUSIVE),
    ],
)
def test_present_condition_semantics(grounding_status, expected_status):
    result = _verify(
        after_snapshot=_snapshot_for_status(grounding_status, seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert evaluation.status is expected_status
    assert evaluation.grounding.status is grounding_status


@pytest.mark.parametrize(
    ("grounding_status", "expected_status"),
    [
        (GroundingStatus.NOT_FOUND, StateVerificationStatus.VERIFIED),
        (GroundingStatus.RESOLVED, StateVerificationStatus.FAILED),
        (GroundingStatus.AMBIGUOUS, StateVerificationStatus.INCONCLUSIVE),
        (GroundingStatus.UNSAFE, StateVerificationStatus.INCONCLUSIVE),
    ],
)
def test_absent_condition_semantics(grounding_status, expected_status):
    result = _verify(
        after_snapshot=_snapshot_for_status(grounding_status, seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.ABSENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert evaluation.status is expected_status
    assert evaluation.grounding.status is grounding_status


def test_default_state_transition_verifier_remains_legacy_grounder_backed():
    verifier = StateTransitionVerifier()

    assert isinstance(verifier.grounder, UIGrounder)
    assert verifier.state_observer is None


def test_state_transition_verifier_accepts_opt_in_state_observer_backend():
    observer = StateObserver()
    verifier = StateTransitionVerifier(state_observer=observer)

    assert verifier.state_observer is observer
    assert verifier.grounder is None


def test_state_transition_verifier_rejects_ambiguous_backend_configuration():
    with pytest.raises(ValueError, match="either grounder or state_observer"):
        StateTransitionVerifier(
            grounder=UIGrounder(),
            state_observer=StateObserver(),
        )


def test_state_transition_verifier_rejects_invalid_state_observer():
    with pytest.raises(ValueError, match="state_observer"):
        StateTransitionVerifier(state_observer=object())


def test_state_observer_backend_uses_full_snapshot_observe_method():
    observer = RecordingStateObserver()
    before = _snapshot(seconds=0)
    after = _warning_snapshot(seconds=1)
    target = _target("DIALOG")

    result = StateTransitionVerifier(state_observer=observer).verify(
        before_snapshot=before,
        after_snapshot=after,
        verification_spec=_spec(
            after=(
                UIStateCondition(
                    target=target,
                    expectation=PresenceExpectation.ABSENT,
                ),
            )
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert observer.observe_elements_calls == []
    assert len(observer.observe_calls) == 1
    assert observer.observe_calls[0]["snapshot"] is after
    assert observer.observe_calls[0]["target"] is target
    assert "warnings" in result.after_evaluations[0].observation.reason


@pytest.mark.parametrize(
    (
        "expectation",
        "elements",
        "expected_observation_status",
        "expected_condition_status",
    ),
    [
        (
            PresenceExpectation.PRESENT,
            (_element(text="DIALOG"),),
            StateObservationStatus.PRESENT,
            StateVerificationStatus.VERIFIED,
        ),
        (
            PresenceExpectation.PRESENT,
            (),
            StateObservationStatus.ABSENT,
            StateVerificationStatus.FAILED,
        ),
        (
            PresenceExpectation.PRESENT,
            (_element(text="DIALOG", confidence=0.2),),
            StateObservationStatus.INCONCLUSIVE,
            StateVerificationStatus.INCONCLUSIVE,
        ),
        (
            PresenceExpectation.ABSENT,
            (),
            StateObservationStatus.ABSENT,
            StateVerificationStatus.VERIFIED,
        ),
        (
            PresenceExpectation.ABSENT,
            (_element(text="DIALOG"),),
            StateObservationStatus.PRESENT,
            StateVerificationStatus.FAILED,
        ),
        (
            PresenceExpectation.ABSENT,
            (_element(text="DIALOG", confidence=0.2),),
            StateObservationStatus.INCONCLUSIVE,
            StateVerificationStatus.INCONCLUSIVE,
        ),
    ],
)
def test_state_observer_condition_truth_table(
    expectation,
    elements,
    expected_observation_status,
    expected_condition_status,
):
    result = _verify_with_state_observer(
        after_snapshot=_snapshot(*elements, seconds=1),
        verification_spec=_spec(after=(_condition(expectation),)),
    )

    evaluation = result.after_evaluations[0]
    assert evaluation.status is expected_condition_status
    assert evaluation.grounding is None
    assert evaluation.observation.status is expected_observation_status
    assert "state observation was" in evaluation.reason


def test_state_observer_wrong_role_absence_verifies_while_legacy_is_inconclusive():
    target = TargetSpec(
        text="LANDING_IDENTITY",
        element_types=("heading",),
    )
    condition = UIStateCondition(
        target=target,
        expectation=PresenceExpectation.ABSENT,
    )
    after = _snapshot(
        _element(
            text="LANDING_IDENTITY",
            element_type="link",
            confidence=1.0,
        ),
        seconds=1,
    )

    legacy = _verify(
        after_snapshot=after,
        verification_spec=_spec(after=(condition,)),
    )
    observer = _verify_with_state_observer(
        after_snapshot=after,
        verification_spec=_spec(after=(condition,)),
    )

    assert legacy.status is StateVerificationStatus.INCONCLUSIVE
    assert legacy.after_evaluations[0].grounding.status is GroundingStatus.UNSAFE
    assert observer.status is StateVerificationStatus.VERIFIED
    assert observer.after_evaluations[0].observation.status is (
        StateObservationStatus.ABSENT
    )
    assert observer.after_evaluations[0].status is (
        StateVerificationStatus.VERIFIED
    )


def test_state_observer_disabled_match_is_present_and_absent_expectation_fails():
    result = _verify_with_state_observer(
        after_snapshot=_snapshot(
            _element(text="DIALOG", enabled=False, confidence=1.0),
            seconds=1,
        ),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.ABSENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert result.status is StateVerificationStatus.FAILED
    assert evaluation.observation.status is StateObservationStatus.PRESENT
    assert evaluation.status is StateVerificationStatus.FAILED


def test_state_observer_multiple_valid_matches_are_present_not_ambiguous():
    result = _verify_with_state_observer(
        after_snapshot=_snapshot(
            _element(text="DIALOG", x=10),
            _element(text="DIALOG", x=140),
            seconds=1,
        ),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert result.status is StateVerificationStatus.VERIFIED
    assert evaluation.observation.status is StateObservationStatus.PRESENT
    assert len(evaluation.observation.candidates) == 2


def test_state_observer_low_confidence_match_is_inconclusive():
    result = _verify_with_state_observer(
        after_snapshot=_snapshot(
            _element(text="DIALOG", confidence=0.2),
            seconds=1,
        ),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert evaluation.observation.status is StateObservationStatus.INCONCLUSIVE
    assert evaluation.observation.candidates[0].uncertainty_reasons == (
        "low_confidence",
    )


def test_state_observer_warning_snapshot_without_match_is_inconclusive():
    result = _verify_with_state_observer(
        after_snapshot=_warning_snapshot(seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.ABSENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert evaluation.observation.status is StateObservationStatus.INCONCLUSIVE
    assert "warnings" in evaluation.observation.reason


def test_state_observer_warning_free_snapshot_without_match_is_absent():
    result = _verify_with_state_observer(
        after_snapshot=_snapshot(seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.ABSENT),)
        ),
    )

    evaluation = result.after_evaluations[0]
    assert result.status is StateVerificationStatus.VERIFIED
    assert evaluation.observation.status is StateObservationStatus.ABSENT


def test_state_observer_before_only_verification_spec_works():
    result = _verify_with_state_observer(
        before_snapshot=_snapshot(_element(text="OLD_STATE"), seconds=0),
        verification_spec=VerificationSpec(
            before_conditions=(
                UIStateCondition(
                    target=_target("OLD_STATE"),
                    expectation=PresenceExpectation.PRESENT,
                ),
            ),
            after_conditions=(),
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.before_evaluations[0].observation.status is (
        StateObservationStatus.PRESENT
    )
    assert result.after_evaluations == ()


@pytest.mark.parametrize(
    ("before_seconds", "after_seconds"),
    [(1, 1), (2, 1)],
)
def test_state_observer_mode_preserves_temporal_freshness_override(
    before_seconds,
    after_seconds,
):
    result = _verify_with_state_observer(
        before_snapshot=_snapshot(seconds=before_seconds),
        after_snapshot=_snapshot(_element(text="DIALOG"), seconds=after_seconds),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.after_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert "after snapshot was not newer than before snapshot" in result.reason


def test_state_observer_mode_retains_all_evaluations_on_mixed_outcomes():
    result = _verify_with_state_observer(
        before_snapshot=_snapshot(_element(text="OLD_STATE"), seconds=0),
        after_snapshot=_snapshot(
            _element(text="PRESENT_TARGET"),
            _element(text="LOW_CONFIDENCE", confidence=0.2),
            seconds=1,
        ),
        verification_spec=_spec(
            before=(
                UIStateCondition(
                    target=_target("OLD_STATE"),
                    expectation=PresenceExpectation.PRESENT,
                ),
            ),
            after=(
                UIStateCondition(
                    target=_target("MISSING"),
                    expectation=PresenceExpectation.PRESENT,
                ),
                UIStateCondition(
                    target=_target("PRESENT_TARGET"),
                    expectation=PresenceExpectation.PRESENT,
                ),
                UIStateCondition(
                    target=_target("LOW_CONFIDENCE"),
                    expectation=PresenceExpectation.PRESENT,
                ),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert len(result.before_evaluations) == 1
    assert len(result.after_evaluations) == 3
    assert [evaluation.status for evaluation in result.after_evaluations] == [
        StateVerificationStatus.FAILED,
        StateVerificationStatus.VERIFIED,
        StateVerificationStatus.INCONCLUSIVE,
    ]
    assert all(
        evaluation.observation is not None
        for evaluation in (
            *result.before_evaluations,
            *result.after_evaluations,
        )
    )


def test_generic_transition_with_before_present_after_present_and_absent():
    result = _verify(
        before_snapshot=_snapshot(
            _element(text="OLD_STATE"),
            seconds=0,
        ),
        after_snapshot=_snapshot(
            _element(text="NEW_STATE"),
            seconds=1,
        ),
        verification_spec=_spec(
            before=(
                _condition(PresenceExpectation.PRESENT, text="OLD_STATE"),
            ),
            after=(
                _condition(PresenceExpectation.PRESENT, text="NEW_STATE"),
                _condition(PresenceExpectation.ABSENT, text="OLD_STATE"),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert [evaluation.status for evaluation in result.before_evaluations] == [
        StateVerificationStatus.VERIFIED
    ]
    assert [evaluation.status for evaluation in result.after_evaluations] == [
        StateVerificationStatus.VERIFIED,
        StateVerificationStatus.VERIFIED,
    ]


def test_definite_failed_condition_without_uncertainty_fails_overall():
    result = _verify(
        after_snapshot=_snapshot(seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT, text="MENU"),)
        ),
    )

    assert result.status is StateVerificationStatus.FAILED
    assert result.after_evaluations[0].status is StateVerificationStatus.FAILED


def test_inconclusive_takes_precedence_over_failed_condition():
    result = _verify(
        after_snapshot=_snapshot(
            _element(text="DIALOG", x=10),
            _element(text="DIALOG", x=140),
            seconds=1,
        ),
        verification_spec=_spec(
            after=(
                _condition(PresenceExpectation.PRESENT, text="MENU"),
                _condition(PresenceExpectation.PRESENT, text="DIALOG"),
            )
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert [evaluation.status for evaluation in result.after_evaluations] == [
        StateVerificationStatus.FAILED,
        StateVerificationStatus.INCONCLUSIVE,
    ]


def test_empty_before_conditions_with_valid_after_condition_is_valid():
    result = _verify(
        after_snapshot=_snapshot(_element(text="DIALOG"), seconds=1),
        verification_spec=VerificationSpec(
            before_conditions=(),
            after_conditions=(
                _condition(PresenceExpectation.PRESENT, text="DIALOG"),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.before_evaluations == ()
    assert len(result.after_evaluations) == 1


def test_empty_after_conditions_with_valid_before_condition_is_valid():
    result = _verify(
        before_snapshot=_snapshot(_element(text="OLD_STATE"), seconds=0),
        verification_spec=VerificationSpec(
            before_conditions=(
                _condition(PresenceExpectation.PRESENT, text="OLD_STATE"),
            ),
            after_conditions=(),
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert len(result.before_evaluations) == 1
    assert result.after_evaluations == ()


def test_after_conditions_with_equal_snapshot_time_are_inconclusive():
    result = _verify(
        before_snapshot=_snapshot(seconds=1),
        after_snapshot=_snapshot(_element(text="DIALOG"), seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.after_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert "after snapshot was not newer than before snapshot" in result.reason


def test_after_conditions_with_stale_snapshot_are_inconclusive():
    result = _verify(
        before_snapshot=_snapshot(seconds=2),
        after_snapshot=_snapshot(_element(text="DIALOG"), seconds=1),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.after_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert "after snapshot was not newer than before snapshot" in result.reason


def test_after_conditions_with_newer_snapshot_and_satisfied_conditions_verify():
    result = _verify(
        before_snapshot=_snapshot(seconds=1),
        after_snapshot=_snapshot(_element(text="DIALOG"), seconds=2),
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED


@pytest.mark.parametrize("after_seconds", [1, 2])
def test_before_only_spec_does_not_require_after_snapshot_freshness(
    after_seconds,
):
    result = _verify(
        before_snapshot=_snapshot(_element(text="OLD_STATE"), seconds=2),
        after_snapshot=_snapshot(seconds=after_seconds),
        verification_spec=_spec(
            before=(
                _condition(PresenceExpectation.PRESENT, text="OLD_STATE"),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.before_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert result.after_evaluations == ()
    assert "after snapshot was not newer than before snapshot" not in result.reason


def test_all_conditions_are_evaluated_when_temporal_ordering_is_invalid():
    result = _verify(
        before_snapshot=_snapshot(_element(text="OLD_STATE"), seconds=2),
        after_snapshot=_snapshot(
            _element(text="NEW_STATE"),
            _element(text="DIALOG", x=10),
            _element(text="DIALOG", x=140),
            seconds=1,
        ),
        verification_spec=_spec(
            before=(
                _condition(PresenceExpectation.PRESENT, text="OLD_STATE"),
                _condition(PresenceExpectation.ABSENT, text="MENU"),
            ),
            after=(
                _condition(PresenceExpectation.PRESENT, text="NEW_STATE"),
                _condition(PresenceExpectation.PRESENT, text="MISSING"),
                _condition(PresenceExpectation.PRESENT, text="DIALOG"),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert [evaluation.status for evaluation in result.before_evaluations] == [
        StateVerificationStatus.VERIFIED,
        StateVerificationStatus.VERIFIED,
    ]
    assert [evaluation.status for evaluation in result.after_evaluations] == [
        StateVerificationStatus.VERIFIED,
        StateVerificationStatus.FAILED,
        StateVerificationStatus.INCONCLUSIVE,
    ]
    assert "after snapshot was not newer than before snapshot" in result.reason


def test_identifier_only_target_reason_uses_identifier_description():
    result = _verify(
        after_snapshot=_snapshot(seconds=1),
        verification_spec=_spec(
            after=(
                UIStateCondition(
                    target=_identifier_target("primary-submit"),
                    expectation=PresenceExpectation.PRESENT,
                ),
            ),
        ),
    )

    reason = result.after_evaluations[0].reason
    assert "identifier 'primary-submit'" in reason
    assert "target None" not in reason


def test_completely_empty_verification_spec_is_rejected():
    with pytest.raises(ValueError, match="at least one UIStateCondition"):
        VerificationSpec()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "target": object(),
                "expectation": PresenceExpectation.PRESENT,
            },
            "target",
        ),
        (
            {
                "target": _target(),
                "expectation": "present",
            },
            "expectation",
        ),
    ],
)
def test_ui_state_condition_rejects_invalid_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        UIStateCondition(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"before_conditions": []}, "before_conditions"),
        ({"after_conditions": []}, "after_conditions"),
        ({"before_conditions": (object(),)}, "before_conditions"),
        ({"after_conditions": (object(),)}, "after_conditions"),
    ],
)
def test_verification_spec_rejects_invalid_fields(kwargs, message):
    with pytest.raises(ValueError, match=message):
        VerificationSpec(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"condition": object()}, "condition"),
        ({"grounding": object()}, "grounding"),
        ({"grounding": None}, "grounding or observation"),
        ({"grounding": None, "observation": object()}, "observation"),
        ({"status": "verified"}, "status"),
        ({"reason": ""}, "reason"),
    ],
)
def test_ui_state_condition_evaluation_rejects_invalid_fields(kwargs, message):
    condition = _condition(PresenceExpectation.PRESENT)
    grounding = GroundingResult(
        status=GroundingStatus.NOT_FOUND,
        element=None,
        candidates=(),
        reason="not found",
    )
    fields = {
        "condition": condition,
        "grounding": grounding,
        "status": StateVerificationStatus.FAILED,
        "reason": "failed",
    }
    fields.update(kwargs)

    with pytest.raises(ValueError, match=message):
        UIStateConditionEvaluation(**fields)


def test_ui_state_condition_evaluation_rejects_both_evidence_sources():
    condition = _condition(PresenceExpectation.PRESENT)
    grounding = GroundingResult(
        status=GroundingStatus.NOT_FOUND,
        element=None,
        candidates=(),
        reason="not found",
    )
    observation = StateObserver().observe(
        snapshot=_snapshot(seconds=1),
        target=condition.target,
    )

    with pytest.raises(ValueError, match="grounding or observation"):
        UIStateConditionEvaluation(
            condition=condition,
            grounding=grounding,
            observation=observation,
            status=StateVerificationStatus.FAILED,
            reason="failed",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "verified"}, "status"),
        ({"before_evaluations": []}, "before_evaluations"),
        ({"after_evaluations": []}, "after_evaluations"),
        ({"before_evaluations": (object(),)}, "before_evaluations"),
        ({"after_evaluations": (object(),)}, "after_evaluations"),
        ({"reason": ""}, "reason"),
    ],
)
def test_state_transition_result_rejects_invalid_fields(kwargs, message):
    evaluation = UIStateConditionEvaluation(
        condition=_condition(PresenceExpectation.PRESENT),
        grounding=GroundingResult(
            status=GroundingStatus.NOT_FOUND,
            element=None,
            candidates=(),
            reason="not found",
        ),
        status=StateVerificationStatus.FAILED,
        reason="failed",
    )
    fields = {
        "status": StateVerificationStatus.FAILED,
        "before_evaluations": (),
        "after_evaluations": (evaluation,),
        "reason": "failed",
    }
    fields.update(kwargs)

    with pytest.raises(ValueError, match=message):
        StateTransitionVerificationResult(**fields)


def test_state_transition_result_rejects_no_evaluations():
    with pytest.raises(ValueError, match="at least one evaluation"):
        StateTransitionVerificationResult(
            status=StateVerificationStatus.VERIFIED,
            before_evaluations=(),
            after_evaluations=(),
            reason="empty",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"before_snapshot": object()}, "before_snapshot"),
        ({"after_snapshot": object()}, "after_snapshot"),
        ({"verification_spec": object()}, "verification_spec"),
    ],
)
def test_state_transition_verifier_rejects_invalid_inputs(kwargs, message):
    fields = {
        "before_snapshot": _snapshot(seconds=0),
        "after_snapshot": _snapshot(seconds=1),
        "verification_spec": _spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    }
    fields.update(kwargs)

    with pytest.raises(ValueError, match=message):
        StateTransitionVerifier().verify(**fields)


def test_state_transition_verifier_rejects_invalid_grounder():
    with pytest.raises(ValueError, match="grounder"):
        StateTransitionVerifier(grounder=object())


def test_state_transition_verifier_accepts_compatible_grounder_dependency():
    class CompatibleGrounder:
        def __init__(self) -> None:
            self.calls = []

        def ground(self, target_spec, elements):
            self.calls.append(
                {
                    "target_spec": target_spec,
                    "elements": elements,
                }
            )
            return GroundingResult(
                status=GroundingStatus.RESOLVED,
                element=elements[0],
                candidates=(),
                reason="resolved by fake grounder",
            )

    grounder = CompatibleGrounder()
    snapshot = _snapshot(_element(text="DIALOG"), seconds=1)
    verifier = StateTransitionVerifier(grounder=grounder)

    result = verifier.verify(
        before_snapshot=_snapshot(seconds=0),
        after_snapshot=snapshot,
        verification_spec=_spec(
            after=(_condition(PresenceExpectation.PRESENT),)
        ),
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.after_evaluations[0].grounding.reason == (
        "resolved by fake grounder"
    )
    assert verifier.grounder is grounder
    assert grounder.calls[0]["elements"] == snapshot.fused_elements


def test_multiple_conditions_are_all_evaluated():
    result = _verify(
        before_snapshot=_snapshot(
            _element(text="OLD_STATE"),
            seconds=0,
        ),
        after_snapshot=_snapshot(
            _element(text="NEW_STATE"),
            _element(text="DIALOG", x=10),
            _element(text="DIALOG", x=140),
            seconds=1,
        ),
        verification_spec=_spec(
            before=(
                _condition(PresenceExpectation.PRESENT, text="OLD_STATE"),
                _condition(PresenceExpectation.ABSENT, text="MENU"),
            ),
            after=(
                _condition(PresenceExpectation.PRESENT, text="NEW_STATE"),
                _condition(PresenceExpectation.ABSENT, text="OLD_STATE"),
                _condition(PresenceExpectation.PRESENT, text="DIALOG"),
            ),
        ),
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert len(result.before_evaluations) == 2
    assert len(result.after_evaluations) == 3
    assert [
        evaluation.condition.target.text
        for evaluation in result.after_evaluations
    ] == [
        "NEW_STATE",
        "OLD_STATE",
        "DIALOG",
    ]


def _action() -> Action:
    return Action(
        tool_name="click_mouse",
        arguments={"x": 10, "y": 20},
        reason="synthetic click",
    )


def _tool_result(action: Action) -> ToolResult:
    return ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=True,
    )


def _appearance_verification_spec(target: TargetSpec) -> VerificationSpec:
    return VerificationSpec(
        before_conditions=(
            UIStateCondition(
                target=target,
                expectation=PresenceExpectation.ABSENT,
            ),
        ),
        after_conditions=(
            UIStateCondition(
                target=target,
                expectation=PresenceExpectation.PRESENT,
            ),
        ),
    )


def _verify_legacy_and_generic_appearance_pattern(
    *,
    before_status: GroundingStatus,
    after_status: GroundingStatus,
    before_seconds: int = 0,
    after_seconds: int = 1,
):
    target = _target()
    action = _action()
    before_snapshot = _snapshot_for_status(
        before_status,
        seconds=before_seconds,
    )
    after_snapshot = _snapshot_for_status(after_status, seconds=after_seconds)

    legacy = ActionVerifier().verify_target_appeared(
        action=action,
        tool_result=_tool_result(action),
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        target_spec=target,
    )
    generic = StateTransitionVerifier().verify(
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        verification_spec=_appearance_verification_spec(target),
    )

    return legacy, generic


def test_legacy_and_generic_verify_when_target_was_absent_then_resolved():
    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.NOT_FOUND,
        after_status=GroundingStatus.RESOLVED,
    )

    assert legacy.status is ActionVerificationStatus.VERIFIED
    assert generic.status is StateVerificationStatus.VERIFIED
    assert generic.before_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert generic.after_evaluations[0].status is StateVerificationStatus.VERIFIED


def test_legacy_and_generic_fail_when_target_remains_absent():
    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.NOT_FOUND,
        after_status=GroundingStatus.NOT_FOUND,
    )

    assert legacy.status is ActionVerificationStatus.FAILED
    assert generic.status is StateVerificationStatus.FAILED
    assert generic.before_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert generic.after_evaluations[0].status is StateVerificationStatus.FAILED


def test_preexisting_target_semantic_difference_is_locked_down():
    """Pre-existing targets differ between legacy action and generic state checks."""

    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.RESOLVED,
        after_status=GroundingStatus.RESOLVED,
    )

    assert legacy.status is ActionVerificationStatus.INCONCLUSIVE
    assert generic.status is StateVerificationStatus.FAILED
    assert generic.before_evaluations[0].status is StateVerificationStatus.FAILED
    assert generic.after_evaluations[0].status is StateVerificationStatus.VERIFIED


def test_legacy_and_generic_are_inconclusive_when_before_absence_is_ambiguous():
    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.AMBIGUOUS,
        after_status=GroundingStatus.RESOLVED,
    )

    assert legacy.status is ActionVerificationStatus.INCONCLUSIVE
    assert generic.status is StateVerificationStatus.INCONCLUSIVE
    assert (
        generic.before_evaluations[0].status
        is StateVerificationStatus.INCONCLUSIVE
    )
    assert generic.after_evaluations[0].status is StateVerificationStatus.VERIFIED


def test_legacy_and_generic_are_inconclusive_when_after_presence_is_ambiguous():
    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.NOT_FOUND,
        after_status=GroundingStatus.AMBIGUOUS,
    )

    assert legacy.status is ActionVerificationStatus.INCONCLUSIVE
    assert generic.status is StateVerificationStatus.INCONCLUSIVE
    assert generic.before_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert (
        generic.after_evaluations[0].status
        is StateVerificationStatus.INCONCLUSIVE
    )


@pytest.mark.parametrize(
    ("before_seconds", "after_seconds"),
    [(0, 0), (2, 1)],
)
def test_legacy_and_generic_are_inconclusive_with_temporally_invalid_after(
    before_seconds,
    after_seconds,
):
    legacy, generic = _verify_legacy_and_generic_appearance_pattern(
        before_status=GroundingStatus.NOT_FOUND,
        after_status=GroundingStatus.RESOLVED,
        before_seconds=before_seconds,
        after_seconds=after_seconds,
    )

    assert legacy.status is ActionVerificationStatus.INCONCLUSIVE
    assert generic.status is StateVerificationStatus.INCONCLUSIVE
    assert generic.before_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert generic.after_evaluations[0].status is StateVerificationStatus.VERIFIED
    assert "after snapshot was not newer than before snapshot" in legacy.reason
    assert "after snapshot was not newer than before snapshot" in generic.reason


def test_models_are_frozen_and_slotted():
    condition = _condition(PresenceExpectation.PRESENT)

    assert not hasattr(condition, "__dict__")
    with pytest.raises(FrozenInstanceError):
        condition.expectation = PresenceExpectation.ABSENT
