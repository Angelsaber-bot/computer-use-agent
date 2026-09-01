from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import importlib
import inspect
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import (
    ActionGrounder,
    ActionGroundingResult,
    ActionGroundingStatus,
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.perception.models import BoundingBox, ScreenFrame, UIElement
from computer_agent.recovery import (
    ActionRecovery,
    RecoveryResult,
    RecoveryStatus,
)
import computer_agent.recovery.action_recovery as action_recovery_module
from computer_agent.verification import (
    ActionVerificationResult,
    ActionVerificationStatus,
)


TARGET_TEXT = "RECOVERY_TARGET"


class FailingGrounder(UIGrounder):
    def ground(self, target_spec, elements):
        raise AssertionError("UI grounding should not run")


class FailingActionGrounder(ActionGrounder):
    def ground_click(self, grounding_result, screen_size):
        raise AssertionError("action grounding should not run")


class RecordingActionGrounder(ActionGrounder):
    def __init__(self) -> None:
        super().__init__()
        self.screen_sizes = []

    def ground_click(self, grounding_result, screen_size):
        self.screen_sizes.append(screen_size)
        return super().ground_click(
            grounding_result,
            screen_size,
        )


def _time() -> datetime:
    return datetime(
        2026,
        8,
        31,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )


def _box(
    x=10,
    y=20,
    width=100,
    height=30,
) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _element(
    *,
    text=TARGET_TEXT,
    element_type="button",
    enabled=True,
    confidence=0.95,
    source="accessibility",
    x=10,
    y=20,
    width=100,
    height=30,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
        confidence=confidence,
        text=text,
        enabled=enabled,
        source=source,
    )


def _snapshot(
    elements=(),
    *,
    screen_size=(200, 100),
    pixel_size=None,
) -> PerceptionSnapshot:
    pixel_size = pixel_size or screen_size
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic.png"),
            pixel_width=pixel_size[0],
            pixel_height=pixel_size[1],
            screen_width=screen_size[0],
            screen_height=screen_size[1],
            captured_at=_time(),
        ),
        image=Image.new(
            "RGB",
            pixel_size,
        ),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=(),
    )


def _target_spec() -> TargetSpec:
    return TargetSpec(
        text=TARGET_TEXT,
        element_types=("button",),
    )


def _grounding(
    status: GroundingStatus,
    *,
    element=None,
    reason=None,
) -> GroundingResult:
    if status is GroundingStatus.RESOLVED:
        element = element or _element()
    else:
        element = None

    return GroundingResult(
        status=status,
        element=element,
        candidates=(),
        reason=reason or status.value,
    )


def _verification(
    status: ActionVerificationStatus,
) -> ActionVerificationResult:
    if status is ActionVerificationStatus.VERIFIED:
        before_grounding = _grounding(GroundingStatus.NOT_FOUND)
        after_grounding = _grounding(GroundingStatus.RESOLVED)
    else:
        before_grounding = _grounding(GroundingStatus.NOT_FOUND)
        after_grounding = _grounding(GroundingStatus.NOT_FOUND)

    return ActionVerificationResult(
        status=status,
        before_grounding=before_grounding,
        after_grounding=after_grounding,
        reason=status.value,
    )


def _tool_result(
    *,
    success=True,
    error=None,
) -> ToolResult:
    if not success and error is None:
        error = "execution failed"

    return ToolResult(
        action_id="attempt-action",
        tool_name="click_mouse",
        success=success,
        error=error,
    )


def _failed_verification_with_after(
    after_grounding: GroundingResult,
) -> ActionVerificationResult:
    return ActionVerificationResult(
        status=ActionVerificationStatus.FAILED,
        before_grounding=_grounding(GroundingStatus.NOT_FOUND),
        after_grounding=after_grounding,
        reason="failed",
    )


def _prepare_retry(
    *,
    verification_result=None,
    tool_result=None,
    target_spec=None,
    latest_snapshot=None,
    completed_attempts=0,
    max_attempts=2,
    recovery=None,
) -> RecoveryResult:
    recovery = recovery or ActionRecovery()
    return recovery.prepare_retry(
        verification_result=verification_result
        or _verification(ActionVerificationStatus.FAILED),
        tool_result=tool_result or _tool_result(),
        target_spec=target_spec or _target_spec(),
        latest_snapshot=latest_snapshot or _snapshot((_element(),)),
        completed_attempts=completed_attempts,
        max_attempts=max_attempts,
    )


def _ready_action_grounding() -> ActionGroundingResult:
    return ActionGroundingResult(
        status=ActionGroundingStatus.READY,
        action=Action(
            tool_name="click_mouse",
            arguments={
                "x": 60,
                "y": 35,
            },
            reason="ready",
        ),
        reason="ready",
    )


def _blocked_action_grounding() -> ActionGroundingResult:
    return ActionGroundingResult(
        status=ActionGroundingStatus.BLOCKED,
        action=None,
        reason="blocked",
    )


def test_verified_result_does_not_need_recovery_or_grounding():
    result = _prepare_retry(
        verification_result=_verification(ActionVerificationStatus.VERIFIED),
        recovery=ActionRecovery(
            grounder=FailingGrounder(),
            action_grounder=FailingActionGrounder(),
        ),
    )

    assert result.status is RecoveryStatus.NOT_NEEDED
    assert result.grounding_result is None
    assert result.action_grounding_result is None
    assert "not needed" in result.reason


def test_inconclusive_result_blocks_without_retrying_uncertain_state():
    result = _prepare_retry(
        verification_result=_verification(
            ActionVerificationStatus.INCONCLUSIVE
        ),
        recovery=ActionRecovery(
            grounder=FailingGrounder(),
            action_grounder=FailingActionGrounder(),
        ),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result is None
    assert result.action_grounding_result is None
    assert "inconclusive" in result.reason


def test_failed_execution_blocks_without_ui_recovery():
    result = _prepare_retry(
        tool_result=_tool_result(
            success=False,
            error="click failed",
        ),
        recovery=ActionRecovery(
            grounder=FailingGrounder(),
            action_grounder=FailingActionGrounder(),
        ),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result is None
    assert result.action_grounding_result is None
    assert "tool execution failed" in result.reason
    assert "deterministic UI recovery" in result.reason


def test_failed_execution_takes_precedence_over_exhausted_attempt_count():
    result = _prepare_retry(
        tool_result=_tool_result(
            success=False,
            error="click failed",
        ),
        completed_attempts=3,
        max_attempts=2,
        recovery=ActionRecovery(
            grounder=FailingGrounder(),
            action_grounder=FailingActionGrounder(),
        ),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result is None
    assert result.action_grounding_result is None
    assert "tool execution failed" in result.reason


@pytest.mark.parametrize(
    "completed_attempts",
    [
        2,
        3,
    ],
)
def test_successful_execution_failed_postcondition_exhausts_on_attempt_limit(
    completed_attempts,
):
    result = _prepare_retry(
        completed_attempts=completed_attempts,
        max_attempts=2,
        recovery=ActionRecovery(
            grounder=FailingGrounder(),
            action_grounder=FailingActionGrounder(),
        ),
    )

    assert result.status is RecoveryStatus.EXHAUSTED
    assert result.grounding_result is None
    assert result.action_grounding_result is None
    assert f"completed_attempts={completed_attempts}" in result.reason


def test_successful_execution_failed_postcondition_prepares_retry_action():
    result = _prepare_retry(
        latest_snapshot=_snapshot(
            (
                _element(
                    x=30,
                    y=40,
                    width=20,
                    height=10,
                ),
            )
        ),
    )

    assert result.status is RecoveryStatus.RETRY_READY
    assert result.grounding_result.status is GroundingStatus.RESOLVED
    assert result.action_grounding_result.status is (
        ActionGroundingStatus.READY
    )
    assert result.action_grounding_result.action.arguments == {
        "x": 40,
        "y": 45,
    }


def test_failed_result_with_missing_fresh_target_blocks_without_action():
    result = _prepare_retry(
        latest_snapshot=_snapshot(()),
        recovery=ActionRecovery(action_grounder=FailingActionGrounder()),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result.status is GroundingStatus.NOT_FOUND
    assert result.action_grounding_result is None


def test_failed_result_with_ambiguous_fresh_target_blocks_without_action():
    result = _prepare_retry(
        latest_snapshot=_snapshot(
            (
                _element(x=10),
                _element(x=40),
            )
        ),
        recovery=ActionRecovery(action_grounder=FailingActionGrounder()),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result.status is GroundingStatus.AMBIGUOUS
    assert result.action_grounding_result is None


def test_failed_result_with_unsafe_fresh_target_blocks_without_action():
    result = _prepare_retry(
        latest_snapshot=_snapshot((_element(enabled=False),)),
        recovery=ActionRecovery(action_grounder=FailingActionGrounder()),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result.status is GroundingStatus.UNSAFE
    assert result.action_grounding_result is None


def test_failed_result_with_resolved_target_but_unsafe_click_blocks():
    result = _prepare_retry(
        latest_snapshot=_snapshot(
            (
                _element(
                    x=0,
                    y=10,
                    width=1,
                    height=1,
                ),
            ),
            screen_size=(100, 80),
        ),
    )

    assert result.status is RecoveryStatus.BLOCKED
    assert result.grounding_result.status is GroundingStatus.RESOLVED
    assert result.action_grounding_result.status is (
        ActionGroundingStatus.BLOCKED
    )
    assert "violates safe screen bounds" in (
        result.action_grounding_result.reason
    )


def test_latest_snapshot_is_used_for_regrounding():
    old_element = _element(
        x=10,
        y=20,
        width=20,
        height=10,
    )
    fresh_element = _element(
        x=90,
        y=50,
        width=20,
        height=10,
    )

    result = _prepare_retry(
        verification_result=_failed_verification_with_after(
            _grounding(
                GroundingStatus.RESOLVED,
                element=old_element,
            )
        ),
        latest_snapshot=_snapshot((fresh_element,)),
    )

    assert result.status is RecoveryStatus.RETRY_READY
    assert result.grounding_result.element is fresh_element
    assert result.action_grounding_result.action.arguments == {
        "x": 100,
        "y": 55,
    }


def test_retry_ready_generates_fresh_action_each_time():
    snapshot = _snapshot(
        (
            _element(
                x=30,
                y=40,
                width=20,
                height=10,
            ),
        )
    )
    recovery = ActionRecovery()

    first = _prepare_retry(
        latest_snapshot=snapshot,
        recovery=recovery,
    )
    second = _prepare_retry(
        latest_snapshot=snapshot,
        recovery=recovery,
    )

    assert first.status is RecoveryStatus.RETRY_READY
    assert second.status is RecoveryStatus.RETRY_READY
    first_action = first.action_grounding_result.action
    second_action = second.action_grounding_result.action
    assert first_action is not second_action
    assert first_action.action_id != second_action.action_id
    assert first_action.arguments == second_action.arguments == {
        "x": 40,
        "y": 45,
    }


def test_latest_screen_size_is_passed_to_action_grounder():
    action_grounder = RecordingActionGrounder()

    result = _prepare_retry(
        latest_snapshot=_snapshot(
            (
                _element(
                    x=10,
                    y=10,
                    width=20,
                    height=10,
                ),
            ),
            screen_size=(123, 45),
            pixel_size=(200, 100),
        ),
        recovery=ActionRecovery(action_grounder=action_grounder),
    )

    assert result.status is RecoveryStatus.RETRY_READY
    assert action_grounder.screen_sizes == [(123, 45)]


@pytest.mark.parametrize(
    ("argument_name", "value", "message"),
    [
        ("verification_result", object(), "verification_result"),
        ("tool_result", object(), "tool_result"),
        ("target_spec", object(), "target_spec"),
        ("latest_snapshot", object(), "latest_snapshot"),
        ("completed_attempts", True, "completed_attempts"),
        ("completed_attempts", False, "completed_attempts"),
        ("completed_attempts", -1, "completed_attempts"),
        ("completed_attempts", 0.0, "completed_attempts"),
        ("completed_attempts", "0", "completed_attempts"),
        ("completed_attempts", None, "completed_attempts"),
        ("max_attempts", True, "max_attempts"),
        ("max_attempts", False, "max_attempts"),
        ("max_attempts", 0, "max_attempts"),
        ("max_attempts", -1, "max_attempts"),
        ("max_attempts", 1.0, "max_attempts"),
        ("max_attempts", "1", "max_attempts"),
        ("max_attempts", None, "max_attempts"),
    ],
)
def test_invalid_prepare_retry_arguments_are_rejected(
    argument_name,
    value,
    message,
):
    arguments = {
        "verification_result": _verification(ActionVerificationStatus.FAILED),
        "tool_result": _tool_result(),
        "target_spec": _target_spec(),
        "latest_snapshot": _snapshot((_element(),)),
        "completed_attempts": 0,
        "max_attempts": 2,
    }
    arguments[argument_name] = value

    with pytest.raises(ValueError, match=message):
        ActionRecovery().prepare_retry(**arguments)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("grounder", object(), "grounder"),
        ("action_grounder", object(), "action_grounder"),
    ],
)
def test_invalid_injected_dependencies_are_rejected(
    keyword,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        ActionRecovery(**{keyword: value})


@pytest.mark.parametrize(
    ("status", "grounding_result", "action_grounding_result", "reason"),
    [
        (
            "retry_ready",
            _grounding(GroundingStatus.RESOLVED),
            _ready_action_grounding(),
            "status",
        ),
        (
            RecoveryStatus.RETRY_READY,
            None,
            _ready_action_grounding(),
            "grounding_result",
        ),
        (
            RecoveryStatus.RETRY_READY,
            _grounding(GroundingStatus.NOT_FOUND),
            _ready_action_grounding(),
            "resolved UI grounding",
        ),
        (
            RecoveryStatus.RETRY_READY,
            _grounding(GroundingStatus.RESOLVED),
            None,
            "action_grounding_result",
        ),
        (
            RecoveryStatus.RETRY_READY,
            _grounding(GroundingStatus.RESOLVED),
            _blocked_action_grounding(),
            "ready action grounding",
        ),
        (
            RecoveryStatus.NOT_NEEDED,
            _grounding(GroundingStatus.NOT_FOUND),
            None,
            "NOT_NEEDED",
        ),
        (
            RecoveryStatus.NOT_NEEDED,
            None,
            _blocked_action_grounding(),
            "NOT_NEEDED",
        ),
        (
            RecoveryStatus.EXHAUSTED,
            _grounding(GroundingStatus.NOT_FOUND),
            None,
            "EXHAUSTED",
        ),
        (
            RecoveryStatus.BLOCKED,
            None,
            _blocked_action_grounding(),
            "grounding_result",
        ),
        (
            RecoveryStatus.BLOCKED,
            _grounding(GroundingStatus.NOT_FOUND),
            _blocked_action_grounding(),
            "unresolved UI grounding",
        ),
        (
            RecoveryStatus.BLOCKED,
            _grounding(GroundingStatus.RESOLVED),
            _ready_action_grounding(),
            "ready action grounding",
        ),
        (
            RecoveryStatus.BLOCKED,
            _grounding(GroundingStatus.RESOLVED),
            None,
            "blocked action grounding",
        ),
        (
            RecoveryStatus.BLOCKED,
            None,
            None,
            "reason",
        ),
    ],
)
def test_recovery_result_invariants(
    status,
    grounding_result,
    action_grounding_result,
    reason,
):
    with pytest.raises(ValueError, match=reason):
        RecoveryResult(
            status=status,
            grounding_result=grounding_result,
            action_grounding_result=action_grounding_result,
            reason="" if reason == "reason" else "invalid",
        )


def test_blocked_recovery_allows_expected_evidence_shapes():
    no_retry = RecoveryResult(
        status=RecoveryStatus.BLOCKED,
        grounding_result=None,
        action_grounding_result=None,
        reason="inconclusive",
    )
    unresolved = RecoveryResult(
        status=RecoveryStatus.BLOCKED,
        grounding_result=_grounding(GroundingStatus.NOT_FOUND),
        action_grounding_result=None,
        reason="not found",
    )
    unsafe_action = RecoveryResult(
        status=RecoveryStatus.BLOCKED,
        grounding_result=_grounding(GroundingStatus.RESOLVED),
        action_grounding_result=_blocked_action_grounding(),
        reason="unsafe click",
    )

    assert no_retry.status is RecoveryStatus.BLOCKED
    assert unresolved.grounding_result.status is GroundingStatus.NOT_FOUND
    assert unsafe_action.action_grounding_result.status is (
        ActionGroundingStatus.BLOCKED
    )


def test_recovery_result_is_immutable_and_slotted():
    result = RecoveryResult(
        status=RecoveryStatus.EXHAUSTED,
        grounding_result=None,
        action_grounding_result=None,
        reason="exhausted",
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"


def test_recovery_imports_are_safe_and_have_no_execution_side_effects():
    module = importlib.import_module("computer_agent.recovery")
    source = inspect.getsource(action_recovery_module)

    forbidden_terms = (
        "computer_agent.control",
        "computer_agent.tools",
        "computer_agent.perception.ocr",
        "PerceptionEngine",
        ".observe(",
        "controller",
        "executor",
        "openai",
        "llm",
        "pyautogui",
        "pytesseract",
        "experiments",
    )

    assert module.ActionRecovery is ActionRecovery
    assert all(term not in source for term in forbidden_terms)
