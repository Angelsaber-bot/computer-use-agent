from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import GroundingResult, GroundingStatus, TargetSpec
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.verification import (
    ActionVerificationResult,
    ActionVerificationStatus,
    ActionVerifier,
)


TARGET_TEXT = "VERIFICATION_TARGET"


def _time(seconds: int = 0):
    return datetime(
        2026,
        8,
        29,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _element(
    *,
    text=TARGET_TEXT,
    enabled=True,
    confidence=0.95,
    source="accessibility",
    x=10,
) -> UIElement:
    return UIElement(
        element_type="button",
        bounding_box=BoundingBox(
            x=x,
            y=20,
            width=100,
            height=30,
        ),
        confidence=confidence,
        text=text,
        enabled=enabled,
        source=source,
    )


def _snapshot(
    elements=(),
    *,
    captured_at=None,
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic.png"),
            pixel_width=200,
            pixel_height=100,
            screen_width=200,
            screen_height=100,
            captured_at=captured_at or _time(),
        ),
        image=Image.new(
            "RGB",
            (200, 100),
        ),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=(),
    )


def _snapshot_for_status(status: GroundingStatus, *, captured_at=None):
    if status is GroundingStatus.NOT_FOUND:
        return _snapshot(
            (),
            captured_at=captured_at,
        )

    if status is GroundingStatus.RESOLVED:
        return _snapshot(
            (_element(),),
            captured_at=captured_at,
        )

    if status is GroundingStatus.AMBIGUOUS:
        return _snapshot(
            (
                _element(x=10),
                _element(x=40),
            ),
            captured_at=captured_at,
        )

    if status is GroundingStatus.UNSAFE:
        return _snapshot(
            (_element(enabled=False),),
            captured_at=captured_at,
        )

    raise AssertionError(f"unknown status: {status}")


def _target_spec() -> TargetSpec:
    return TargetSpec(
        text=TARGET_TEXT,
        element_types=("button",),
    )


def _action() -> Action:
    return Action(
        tool_name="click_mouse",
        arguments={
            "x": 60,
            "y": 35,
        },
        reason="synthetic click",
    )


def _tool_result(
    action,
    *,
    success=True,
    error=None,
) -> ToolResult:
    return ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=success,
        error=error,
    )


def _verify(
    *,
    action=None,
    tool_result=None,
    before_snapshot=None,
    after_snapshot=None,
    target_spec=None,
):
    action = action or _action()
    return ActionVerifier().verify_target_appeared(
        action=action,
        tool_result=tool_result or _tool_result(action),
        before_snapshot=before_snapshot
        or _snapshot_for_status(
            GroundingStatus.NOT_FOUND,
            captured_at=_time(0),
        ),
        after_snapshot=after_snapshot
        or _snapshot_for_status(
            GroundingStatus.RESOLVED,
            captured_at=_time(1),
        ),
        target_spec=target_spec or _target_spec(),
    )


def test_target_absent_before_and_resolved_after_verifies():
    result = _verify()

    assert result.status is ActionVerificationStatus.VERIFIED
    assert result.before_grounding.status is GroundingStatus.NOT_FOUND
    assert result.after_grounding.status is GroundingStatus.RESOLVED
    assert "not_found" in result.reason
    assert "resolved" in result.reason


def test_verified_result_includes_real_grounding_evidence():
    result = _verify()

    assert isinstance(result.before_grounding, GroundingResult)
    assert isinstance(result.after_grounding, GroundingResult)
    assert result.before_grounding.reason == (
        "no exact identifier or normalized text match"
    )
    assert result.after_grounding.reason == "resolved by text"


def test_failed_tool_result_fails_even_if_target_appears():
    action = _action()

    result = _verify(
        action=action,
        tool_result=_tool_result(
            action,
            success=False,
            error="click failed",
        ),
    )

    assert result.status is ActionVerificationStatus.FAILED
    assert "click failed" in result.reason


def test_successful_action_but_target_still_missing_fails():
    result = _verify(
        after_snapshot=_snapshot_for_status(
            GroundingStatus.NOT_FOUND,
            captured_at=_time(1),
        ),
    )

    assert result.status is ActionVerificationStatus.FAILED
    assert "not_found" in result.reason


def test_target_already_resolved_before_action_is_inconclusive():
    result = _verify(
        before_snapshot=_snapshot_for_status(
            GroundingStatus.RESOLVED,
            captured_at=_time(0),
        ),
    )

    assert result.status is ActionVerificationStatus.INCONCLUSIVE
    assert "resolved" in result.reason


@pytest.mark.parametrize(
    "before_status",
    [
        GroundingStatus.AMBIGUOUS,
        GroundingStatus.UNSAFE,
    ],
)
def test_ambiguous_or_unsafe_before_state_is_inconclusive(before_status):
    result = _verify(
        before_snapshot=_snapshot_for_status(
            before_status,
            captured_at=_time(0),
        ),
    )

    assert result.status is ActionVerificationStatus.INCONCLUSIVE
    assert before_status.value in result.reason


@pytest.mark.parametrize(
    "after_status",
    [
        GroundingStatus.AMBIGUOUS,
        GroundingStatus.UNSAFE,
    ],
)
def test_ambiguous_or_unsafe_after_state_is_inconclusive(after_status):
    result = _verify(
        after_snapshot=_snapshot_for_status(
            after_status,
            captured_at=_time(1),
        ),
    )

    assert result.status is ActionVerificationStatus.INCONCLUSIVE
    assert after_status.value in result.reason


@pytest.mark.parametrize(
    "after_time",
    [
        _time(0),
        _time(0) - timedelta(seconds=1),
    ],
)
def test_equal_or_older_after_snapshot_is_inconclusive(after_time):
    result = _verify(
        after_snapshot=_snapshot_for_status(
            GroundingStatus.RESOLVED,
            captured_at=after_time,
        ),
    )

    assert result.status is ActionVerificationStatus.INCONCLUSIVE
    assert "not newer" in result.reason


def test_tool_failure_takes_precedence_over_stale_snapshot_chronology():
    action = _action()

    result = _verify(
        action=action,
        tool_result=_tool_result(
            action,
            success=False,
            error="known execution failure",
        ),
        after_snapshot=_snapshot_for_status(
            GroundingStatus.RESOLVED,
            captured_at=_time(0),
        ),
    )

    assert result.status is ActionVerificationStatus.FAILED
    assert "known execution failure" in result.reason


@pytest.mark.parametrize(
    "tool_result",
    [
        ToolResult(
            action_id="different",
            tool_name="click_mouse",
            success=True,
        ),
        ToolResult(
            action_id="placeholder",
            tool_name="move_mouse",
            success=True,
        ),
    ],
)
def test_action_result_identity_mismatch_is_rejected(tool_result):
    action = _action()
    if tool_result.tool_name == action.tool_name:
        mismatched = tool_result
    else:
        mismatched = ToolResult(
            action_id=action.action_id,
            tool_name=tool_result.tool_name,
            success=True,
        )

    with pytest.raises(ValueError, match="tool_result"):
        _verify(
            action=action,
            tool_result=mismatched,
        )


@pytest.mark.parametrize(
    ("argument_name", "value"),
    [
        ("action", object()),
        ("tool_result", object()),
        ("before_snapshot", object()),
        ("after_snapshot", object()),
        ("target_spec", object()),
    ],
)
def test_invalid_public_method_arguments_are_rejected(argument_name, value):
    arguments = {
        "action": _action(),
        "before_snapshot": _snapshot_for_status(
            GroundingStatus.NOT_FOUND,
            captured_at=_time(0),
        ),
        "after_snapshot": _snapshot_for_status(
            GroundingStatus.RESOLVED,
            captured_at=_time(1),
        ),
        "target_spec": _target_spec(),
    }
    arguments["tool_result"] = _tool_result(arguments["action"])
    arguments[argument_name] = value

    with pytest.raises(ValueError, match=argument_name):
        ActionVerifier().verify_target_appeared(**arguments)


@pytest.mark.parametrize(
    "grounder",
    [
        object(),
        "grounder",
    ],
)
def test_invalid_injected_grounder_is_rejected(grounder):
    with pytest.raises(ValueError, match="grounder"):
        ActionVerifier(grounder=grounder)


@pytest.mark.parametrize(
    ("status", "before_grounding", "after_grounding", "reason", "message"),
    [
        (
            "verified",
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            GroundingResult(
                GroundingStatus.RESOLVED,
                _element(),
                (),
                "resolved",
            ),
            "verified",
            "status",
        ),
        (
            ActionVerificationStatus.FAILED,
            object(),
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            "failed",
            "before_grounding",
        ),
        (
            ActionVerificationStatus.FAILED,
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            object(),
            "failed",
            "after_grounding",
        ),
        (
            ActionVerificationStatus.FAILED,
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            "",
            "reason",
        ),
        (
            ActionVerificationStatus.VERIFIED,
            GroundingResult(
                GroundingStatus.RESOLVED,
                _element(),
                (),
                "resolved",
            ),
            GroundingResult(
                GroundingStatus.RESOLVED,
                _element(),
                (),
                "resolved",
            ),
            "verified",
            "before_grounding",
        ),
        (
            ActionVerificationStatus.VERIFIED,
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            GroundingResult(
                GroundingStatus.NOT_FOUND,
                None,
                (),
                "missing",
            ),
            "verified",
            "after_grounding",
        ),
    ],
)
def test_action_verification_result_invariants(
    status,
    before_grounding,
    after_grounding,
    reason,
    message,
):
    with pytest.raises(ValueError, match=message):
        ActionVerificationResult(
            status=status,
            before_grounding=before_grounding,
            after_grounding=after_grounding,
            reason=reason,
        )


def test_action_verification_result_is_immutable_and_slotted():
    result = ActionVerificationResult(
        status=ActionVerificationStatus.INCONCLUSIVE,
        before_grounding=GroundingResult(
            GroundingStatus.NOT_FOUND,
            None,
            (),
            "missing",
        ),
        after_grounding=GroundingResult(
            GroundingStatus.NOT_FOUND,
            None,
            (),
            "missing",
        ),
        reason="inconclusive",
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"
