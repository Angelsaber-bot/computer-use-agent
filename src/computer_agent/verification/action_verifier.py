"""Deterministic verification for target-appearance postconditions."""

from __future__ import annotations

from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding.models import GroundingResult, GroundingStatus, TargetSpec
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.verification.models import (
    ActionVerificationResult,
    ActionVerificationStatus,
)


class ActionVerifier:
    """Verify whether a target appeared after a successful action."""

    def __init__(self, grounder: UIGrounder | None = None) -> None:
        if grounder is None:
            grounder = UIGrounder()
        elif not isinstance(grounder, UIGrounder):
            raise ValueError(
                "grounder must be a UIGrounder or None"
            )

        self._grounder = grounder

    @property
    def grounder(self) -> UIGrounder:
        """Return the configured UI grounder."""

        return self._grounder

    def verify_target_appeared(
        self,
        *,
        action: Action,
        tool_result: ToolResult,
        before_snapshot: PerceptionSnapshot,
        after_snapshot: PerceptionSnapshot,
        target_spec: TargetSpec,
    ) -> ActionVerificationResult:
        """Verify that a target was absent before and resolved afterward."""

        _validate_inputs(
            action,
            tool_result,
            before_snapshot,
            after_snapshot,
            target_spec,
        )
        _validate_action_result_pair(action, tool_result)

        before_grounding = self._grounder.ground(
            target_spec,
            before_snapshot.fused_elements,
        )
        after_grounding = self._grounder.ground(
            target_spec,
            after_snapshot.fused_elements,
        )

        return _verification_result(
            tool_result,
            before_snapshot,
            after_snapshot,
            before_grounding,
            after_grounding,
        )


def _validate_inputs(
    action: object,
    tool_result: object,
    before_snapshot: object,
    after_snapshot: object,
    target_spec: object,
) -> None:
    if not isinstance(action, Action):
        raise ValueError("action must be an Action")

    if not isinstance(tool_result, ToolResult):
        raise ValueError("tool_result must be a ToolResult")

    if not isinstance(before_snapshot, PerceptionSnapshot):
        raise ValueError("before_snapshot must be a PerceptionSnapshot")

    if not isinstance(after_snapshot, PerceptionSnapshot):
        raise ValueError("after_snapshot must be a PerceptionSnapshot")

    if not isinstance(target_spec, TargetSpec):
        raise ValueError("target_spec must be a TargetSpec")


def _validate_action_result_pair(
    action: Action,
    tool_result: ToolResult,
) -> None:
    if tool_result.action_id != action.action_id:
        raise ValueError(
            "tool_result.action_id must match action.action_id"
        )

    if tool_result.tool_name != action.tool_name:
        raise ValueError(
            "tool_result.tool_name must match action.tool_name"
        )


def _verification_result(
    tool_result: ToolResult,
    before_snapshot: PerceptionSnapshot,
    after_snapshot: PerceptionSnapshot,
    before_grounding: GroundingResult,
    after_grounding: GroundingResult,
) -> ActionVerificationResult:
    if not tool_result.success:
        return ActionVerificationResult(
            status=ActionVerificationStatus.FAILED,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason=f"tool_result failed: {tool_result.error}",
        )

    if after_snapshot.frame.captured_at <= before_snapshot.frame.captured_at:
        return ActionVerificationResult(
            status=ActionVerificationStatus.INCONCLUSIVE,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason="after snapshot was not newer than before snapshot",
        )

    if before_grounding.status is GroundingStatus.RESOLVED:
        return ActionVerificationResult(
            status=ActionVerificationStatus.INCONCLUSIVE,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason="before grounding was resolved; target already existed",
        )

    if before_grounding.status is not GroundingStatus.NOT_FOUND:
        return ActionVerificationResult(
            status=ActionVerificationStatus.INCONCLUSIVE,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason=(
                f"before grounding was {before_grounding.status.value}; "
                "absence was not established"
            ),
        )

    if after_grounding.status is GroundingStatus.RESOLVED:
        return ActionVerificationResult(
            status=ActionVerificationStatus.VERIFIED,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason="target verified: before=not_found, after=resolved",
        )

    if after_grounding.status is GroundingStatus.NOT_FOUND:
        return ActionVerificationResult(
            status=ActionVerificationStatus.FAILED,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason="after grounding was not_found; target did not appear",
        )

    return ActionVerificationResult(
        status=ActionVerificationStatus.INCONCLUSIVE,
        before_grounding=before_grounding,
        after_grounding=after_grounding,
        reason=(
            f"after grounding was {after_grounding.status.value}; "
            "target resolution was inconclusive"
        ),
    )
