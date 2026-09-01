"""Deterministic recovery policy for preparing grounded retry actions."""

from __future__ import annotations

from computer_agent.core.models import ToolResult
from computer_agent.grounding.action_grounder import ActionGrounder
from computer_agent.grounding.action_models import ActionGroundingStatus
from computer_agent.grounding.models import GroundingStatus, TargetSpec
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.recovery.models import RecoveryResult, RecoveryStatus
from computer_agent.verification.models import (
    ActionVerificationResult,
    ActionVerificationStatus,
)


class ActionRecovery:
    """Prepare deterministic retries from verification evidence."""

    def __init__(
        self,
        *,
        grounder: UIGrounder | None = None,
        action_grounder: ActionGrounder | None = None,
    ) -> None:
        if grounder is None:
            grounder = UIGrounder()
        elif not isinstance(grounder, UIGrounder):
            raise ValueError("grounder must be a UIGrounder or None")

        if action_grounder is None:
            action_grounder = ActionGrounder()
        elif not isinstance(action_grounder, ActionGrounder):
            raise ValueError(
                "action_grounder must be an ActionGrounder or None"
            )

        self._grounder = grounder
        self._action_grounder = action_grounder

    @property
    def grounder(self) -> UIGrounder:
        """Return the configured UI grounder."""

        return self._grounder

    @property
    def action_grounder(self) -> ActionGrounder:
        """Return the configured action grounder."""

        return self._action_grounder

    def prepare_retry(
        self,
        *,
        verification_result: ActionVerificationResult,
        tool_result: ToolResult,
        target_spec: TargetSpec,
        latest_snapshot: PerceptionSnapshot,
        completed_attempts: int,
        max_attempts: int,
    ) -> RecoveryResult:
        """Prepare one retry using the caller-supplied latest snapshot.

        ``completed_attempts > max_attempts`` is accepted and treated as
        exhausted, matching the same terminal path as ``>= max_attempts``.
        """

        _validate_inputs(
            verification_result,
            tool_result,
            target_spec,
            latest_snapshot,
            completed_attempts,
            max_attempts,
        )

        if (
            verification_result.status
            is ActionVerificationStatus.VERIFIED
        ):
            return RecoveryResult(
                status=RecoveryStatus.NOT_NEEDED,
                grounding_result=None,
                action_grounding_result=None,
                reason="verification already succeeded; retry not needed",
            )

        if (
            verification_result.status
            is ActionVerificationStatus.INCONCLUSIVE
        ):
            return RecoveryResult(
                status=RecoveryStatus.BLOCKED,
                grounding_result=None,
                action_grounding_result=None,
                reason=(
                    "verification was inconclusive; deterministic recovery "
                    "will not retry an uncertain state"
                ),
            )

        if not tool_result.success:
            return RecoveryResult(
                status=RecoveryStatus.BLOCKED,
                grounding_result=None,
                action_grounding_result=None,
                reason=(
                    "tool execution failed; deterministic UI recovery "
                    "cannot retry execution failures"
                ),
            )

        if completed_attempts >= max_attempts:
            return RecoveryResult(
                status=RecoveryStatus.EXHAUSTED,
                grounding_result=None,
                action_grounding_result=None,
                reason=(
                    "retry attempts exhausted: "
                    f"completed_attempts={completed_attempts}, "
                    f"max_attempts={max_attempts}"
                ),
            )

        grounding_result = self._grounder.ground(
            target_spec,
            latest_snapshot.fused_elements,
        )
        if grounding_result.status is not GroundingStatus.RESOLVED:
            return RecoveryResult(
                status=RecoveryStatus.BLOCKED,
                grounding_result=grounding_result,
                action_grounding_result=None,
                reason=(
                    "fresh UI grounding was "
                    f"{grounding_result.status.value}: "
                    f"{grounding_result.reason}"
                ),
            )

        action_grounding_result = self._action_grounder.ground_click(
            grounding_result,
            latest_snapshot.frame.screen_size,
        )
        if (
            action_grounding_result.status
            is not ActionGroundingStatus.READY
        ):
            return RecoveryResult(
                status=RecoveryStatus.BLOCKED,
                grounding_result=grounding_result,
                action_grounding_result=action_grounding_result,
                reason=(
                    "fresh action grounding was "
                    f"{action_grounding_result.status.value}: "
                    f"{action_grounding_result.reason}"
                ),
            )

        return RecoveryResult(
            status=RecoveryStatus.RETRY_READY,
            grounding_result=grounding_result,
            action_grounding_result=action_grounding_result,
            reason="retry action ready from latest snapshot",
        )


def _validate_inputs(
    verification_result: object,
    tool_result: object,
    target_spec: object,
    latest_snapshot: object,
    completed_attempts: object,
    max_attempts: object,
) -> None:
    if not isinstance(verification_result, ActionVerificationResult):
        raise ValueError(
            "verification_result must be an ActionVerificationResult"
        )

    if not isinstance(tool_result, ToolResult):
        raise ValueError("tool_result must be a ToolResult")

    if not isinstance(target_spec, TargetSpec):
        raise ValueError("target_spec must be a TargetSpec")

    if not isinstance(latest_snapshot, PerceptionSnapshot):
        raise ValueError("latest_snapshot must be a PerceptionSnapshot")

    if (
        isinstance(completed_attempts, bool)
        or not isinstance(completed_attempts, int)
    ):
        raise ValueError(
            "completed_attempts must be a non-boolean integer"
        )

    if completed_attempts < 0:
        raise ValueError(
            "completed_attempts must be greater than or equal to zero"
        )

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise ValueError("max_attempts must be a non-boolean integer")

    if max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
