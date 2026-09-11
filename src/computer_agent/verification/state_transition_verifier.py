"""Generic deterministic verification for before/after UI state contracts."""

from __future__ import annotations

from computer_agent.grounding.models import (
    GroundingResult,
    GroundingStatus,
    TargetSpec,
)
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.verification.models import (
    PresenceExpectation,
    StateTransitionVerificationResult,
    StateVerificationStatus,
    UIStateCondition,
    UIStateConditionEvaluation,
    VerificationSpec,
)
from computer_agent.verification.state_observation import (
    StateObservationResult,
    StateObservationStatus,
    StateObserver,
)


class StateTransitionVerifier:
    """Evaluate generic target presence conditions across two snapshots."""

    def __init__(
        self,
        grounder: object | None = None,
        *,
        state_observer: object | None = None,
    ) -> None:
        if grounder is not None and state_observer is not None:
            raise ValueError(
                "StateTransitionVerifier accepts either grounder or "
                "state_observer, not both"
            )

        if state_observer is not None:
            method = getattr(state_observer, "observe", None)
            if not callable(method):
                raise ValueError("state_observer must provide observe()")

            self._grounder = None
            self._state_observer = state_observer
            return

        if grounder is None:
            grounder = UIGrounder()
        else:
            method = getattr(grounder, "ground", None)
            if not callable(method):
                raise ValueError("grounder must provide ground()")

        self._grounder = grounder
        self._state_observer = None

    @property
    def grounder(self) -> object | None:
        """Return the configured UI grounder."""

        return self._grounder

    @property
    def state_observer(self) -> object | None:
        """Return the configured state observer, when using observer mode."""

        return self._state_observer

    def verify(
        self,
        *,
        before_snapshot: PerceptionSnapshot,
        after_snapshot: PerceptionSnapshot,
        verification_spec: VerificationSpec,
    ) -> StateTransitionVerificationResult:
        """Evaluate every condition in a generic verification spec."""

        if not isinstance(before_snapshot, PerceptionSnapshot):
            raise ValueError("before_snapshot must be a PerceptionSnapshot")

        if not isinstance(after_snapshot, PerceptionSnapshot):
            raise ValueError("after_snapshot must be a PerceptionSnapshot")

        if not isinstance(verification_spec, VerificationSpec):
            raise ValueError("verification_spec must be a VerificationSpec")

        before_evaluations = tuple(
            self._evaluate_condition(condition, before_snapshot)
            for condition in verification_spec.before_conditions
        )
        after_evaluations = tuple(
            self._evaluate_condition(condition, after_snapshot)
            for condition in verification_spec.after_conditions
        )
        evaluations = (*before_evaluations, *after_evaluations)
        status = _aggregate_status(evaluations)
        temporal_reason: str | None = None

        if (
            verification_spec.after_conditions
            and after_snapshot.frame.captured_at
            <= before_snapshot.frame.captured_at
        ):
            status = StateVerificationStatus.INCONCLUSIVE
            temporal_reason = (
                "after snapshot was not newer than before snapshot"
            )

        return StateTransitionVerificationResult(
            status=status,
            before_evaluations=before_evaluations,
            after_evaluations=after_evaluations,
            reason=_result_reason(status, evaluations, temporal_reason),
        )

    def _evaluate_condition(
        self,
        condition: UIStateCondition,
        snapshot: PerceptionSnapshot,
    ) -> UIStateConditionEvaluation:
        if self._state_observer is not None:
            return self._evaluate_observed_condition(condition, snapshot)

        grounding = self._grounder.ground(
            condition.target,
            snapshot.fused_elements,
        )
        status = _condition_status(condition.expectation, grounding)
        return UIStateConditionEvaluation(
            condition=condition,
            grounding=grounding,
            status=status,
            reason=_condition_reason(condition, grounding, status),
        )

    def _evaluate_observed_condition(
        self,
        condition: UIStateCondition,
        snapshot: PerceptionSnapshot,
    ) -> UIStateConditionEvaluation:
        observation = self._state_observer.observe(
            snapshot=snapshot,
            target=condition.target,
        )
        if not isinstance(observation, StateObservationResult):
            raise ValueError(
                "state_observer must return StateObservationResult objects"
            )

        status = _observed_condition_status(
            condition.expectation,
            observation.status,
        )
        return UIStateConditionEvaluation(
            condition=condition,
            grounding=None,
            observation=observation,
            status=status,
            reason=_observed_condition_reason(
                condition,
                observation,
                status,
            ),
        )


def _condition_status(
    expectation: PresenceExpectation,
    grounding: GroundingResult,
) -> StateVerificationStatus:
    if expectation is PresenceExpectation.PRESENT:
        if grounding.status is GroundingStatus.RESOLVED:
            return StateVerificationStatus.VERIFIED
        if grounding.status is GroundingStatus.NOT_FOUND:
            return StateVerificationStatus.FAILED
        return StateVerificationStatus.INCONCLUSIVE

    if expectation is PresenceExpectation.ABSENT:
        if grounding.status is GroundingStatus.NOT_FOUND:
            return StateVerificationStatus.VERIFIED
        if grounding.status is GroundingStatus.RESOLVED:
            return StateVerificationStatus.FAILED
        return StateVerificationStatus.INCONCLUSIVE

    raise RuntimeError(f"unsupported presence expectation: {expectation}")


def _observed_condition_status(
    expectation: PresenceExpectation,
    observation_status: StateObservationStatus,
) -> StateVerificationStatus:
    if expectation is PresenceExpectation.PRESENT:
        if observation_status is StateObservationStatus.PRESENT:
            return StateVerificationStatus.VERIFIED
        if observation_status is StateObservationStatus.ABSENT:
            return StateVerificationStatus.FAILED
        if observation_status is StateObservationStatus.INCONCLUSIVE:
            return StateVerificationStatus.INCONCLUSIVE

    if expectation is PresenceExpectation.ABSENT:
        if observation_status is StateObservationStatus.ABSENT:
            return StateVerificationStatus.VERIFIED
        if observation_status is StateObservationStatus.PRESENT:
            return StateVerificationStatus.FAILED
        if observation_status is StateObservationStatus.INCONCLUSIVE:
            return StateVerificationStatus.INCONCLUSIVE

    raise RuntimeError(
        "unsupported state observation condition: "
        f"{expectation}, {observation_status}"
    )


def _aggregate_status(
    evaluations: tuple[UIStateConditionEvaluation, ...],
) -> StateVerificationStatus:
    if any(
        evaluation.status is StateVerificationStatus.INCONCLUSIVE
        for evaluation in evaluations
    ):
        return StateVerificationStatus.INCONCLUSIVE

    if any(
        evaluation.status is StateVerificationStatus.FAILED
        for evaluation in evaluations
    ):
        return StateVerificationStatus.FAILED

    return StateVerificationStatus.VERIFIED


def _condition_reason(
    condition: UIStateCondition,
    grounding: GroundingResult,
    status: StateVerificationStatus,
) -> str:
    return (
        f"target {_target_description(condition.target)} expected "
        f"{condition.expectation.value}; grounding was "
        f"{grounding.status.value}; condition {status.value}"
    )


def _observed_condition_reason(
    condition: UIStateCondition,
    observation: StateObservationResult,
    status: StateVerificationStatus,
) -> str:
    return (
        f"target {_target_description(condition.target)} expected "
        f"{condition.expectation.value}; state observation was "
        f"{observation.status.value}; condition {status.value}"
    )


def _result_reason(
    status: StateVerificationStatus,
    evaluations: tuple[UIStateConditionEvaluation, ...],
    temporal_reason: str | None = None,
) -> str:
    counts = {
        StateVerificationStatus.VERIFIED: 0,
        StateVerificationStatus.FAILED: 0,
        StateVerificationStatus.INCONCLUSIVE: 0,
    }
    for evaluation in evaluations:
        counts[evaluation.status] += 1

    reason = (
        f"state transition verification {status.value}: "
        f"{counts[StateVerificationStatus.VERIFIED]} verified, "
        f"{counts[StateVerificationStatus.FAILED]} failed, "
        f"{counts[StateVerificationStatus.INCONCLUSIVE]} inconclusive"
    )

    if temporal_reason is not None:
        return f"{reason}; {temporal_reason}"

    return reason


def _target_description(target: TargetSpec) -> str:
    if target.text is not None:
        return f"text {target.text!r}"

    return f"identifier {target.identifier!r}"
