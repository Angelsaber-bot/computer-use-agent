from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.verification import (
    StateVerificationResult,
    StateVerificationStatus,
    StateVerifier,
)


class FakeApplicationObserver:
    def __init__(self, response=None, *, error=None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    def read_frontmost_application_name(self):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return self.response


def _frame() -> ScreenFrame:
    return ScreenFrame(
        image_path=Path("synthetic.png"),
        pixel_width=200,
        pixel_height=100,
        screen_width=200,
        screen_height=100,
        captured_at=datetime(
            2026,
            9,
            3,
            12,
            0,
            tzinfo=timezone.utc,
        ),
    )


def _box(x=10) -> BoundingBox:
    return BoundingBox(
        x=x,
        y=20,
        width=100,
        height=30,
    )


def _element(
    *,
    element_type="text_area",
    value="CROSS_APP_TRANSFER_10",
    focused=True,
    source="accessibility",
    text="Editor",
    x=10,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x),
        confidence=1.0,
        text=text,
        value=value,
        focused=focused,
        source=source,
    )


def _snapshot(
    *,
    accessibility_elements=(),
    ocr_elements=(),
    fused_elements=(),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=_frame(),
        image=Image.new("RGB", (200, 100)),
        accessibility_elements=tuple(accessibility_elements),
        ocr_elements=tuple(ocr_elements),
        fused_elements=tuple(fused_elements),
        warnings=(),
    )


def _verifier(observer=None) -> StateVerifier:
    if observer is None:
        observer = FakeApplicationObserver()

    return StateVerifier(application_observer=observer)


def test_frontmost_application_exact_expected_verifies():
    observer = FakeApplicationObserver("TextEdit")

    result = _verifier(observer).verify_frontmost_application("TextEdit")

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.reason.strip()
    assert observer.calls == 1


def test_frontmost_application_different_app_fails():
    observer = FakeApplicationObserver("Safari")

    result = _verifier(observer).verify_frontmost_application("TextEdit")

    assert result.status is StateVerificationStatus.FAILED
    assert "Safari" in result.reason
    assert "TextEdit" in result.reason
    assert observer.calls == 1


def test_frontmost_application_none_is_inconclusive():
    observer = FakeApplicationObserver(None)

    result = _verifier(observer).verify_frontmost_application("TextEdit")

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()
    assert observer.calls == 1


def test_frontmost_application_observer_exception_is_inconclusive():
    observer = FakeApplicationObserver(error=RuntimeError("AX failed"))

    result = _verifier(observer).verify_frontmost_application("TextEdit")

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert "RuntimeError" in result.reason
    assert "AX failed" in result.reason
    assert observer.calls == 1


@pytest.mark.parametrize("expected_app_name", ["", "   ", 123, None])
def test_empty_expected_frontmost_application_is_rejected(expected_app_name):
    with pytest.raises(
        ValueError,
        match="expected_app_name must be a non-empty string",
    ):
        _verifier().verify_frontmost_application(expected_app_name)


def test_focused_text_area_with_exact_value_verifies():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_area",
                value="CROSS_APP_TRANSFER_10",
                focused=True,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.reason.strip()


def test_focused_text_field_with_exact_value_verifies():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_field",
                value="CROSS_APP_TRANSFER_10",
                focused=True,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.VERIFIED
    assert result.reason.strip()


def test_focused_editable_with_different_value_fails():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_area",
                value="different",
                focused=True,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.FAILED
    assert result.reason.strip()


def test_no_focused_editable_is_inconclusive():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="button",
                value="CROSS_APP_TRANSFER_10",
                focused=True,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()


def test_multiple_focused_editables_are_inconclusive():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_area",
                value="CROSS_APP_TRANSFER_10",
                focused=True,
                x=10,
            ),
            _element(
                element_type="text_field",
                value="CROSS_APP_TRANSFER_10",
                focused=True,
                x=120,
            ),
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()


@pytest.mark.parametrize("value", [None, 123, 1.5, True])
def test_focused_editable_without_string_value_is_inconclusive(value):
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_area",
                value=value,
                focused=True,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()


def test_unfocused_matching_text_does_not_verify():
    snapshot = _snapshot(
        accessibility_elements=[
            _element(
                element_type="text_area",
                value="CROSS_APP_TRANSFER_10",
                focused=False,
            )
        ]
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()


def test_ocr_only_matching_text_does_not_verify():
    matching_ocr = _element(
        element_type="text",
        value=None,
        focused=None,
        source="ocr",
        text="CROSS_APP_TRANSFER_10",
    )
    snapshot = _snapshot(
        ocr_elements=[matching_ocr],
        fused_elements=[matching_ocr],
    )

    result = _verifier().verify_focused_editable_value(
        snapshot,
        "CROSS_APP_TRANSFER_10",
    )

    assert result.status is StateVerificationStatus.INCONCLUSIVE
    assert result.reason.strip()


@pytest.mark.parametrize("expected_value", ["", "   ", 123, None])
def test_empty_expected_focused_editable_value_is_rejected(expected_value):
    snapshot = _snapshot()

    with pytest.raises(
        ValueError,
        match="expected_value must be a non-empty string",
    ):
        _verifier().verify_focused_editable_value(snapshot, expected_value)


def test_invalid_snapshot_is_rejected():
    with pytest.raises(ValueError, match="snapshot must be a PerceptionSnapshot"):
        _verifier().verify_focused_editable_value(
            object(),
            "CROSS_APP_TRANSFER_10",
        )


def test_invalid_application_observer_is_rejected():
    with pytest.raises(ValueError, match="application_observer"):
        StateVerifier(application_observer=object())


def test_state_verification_result_invariants_are_enforced():
    with pytest.raises(ValueError, match="status"):
        StateVerificationResult(
            status="verified",
            reason="verified",
        )

    with pytest.raises(ValueError, match="reason"):
        StateVerificationResult(
            status=StateVerificationStatus.INCONCLUSIVE,
            reason="",
        )


def test_state_verification_result_is_immutable_and_slotted():
    result = StateVerificationResult(
        status=StateVerificationStatus.INCONCLUSIVE,
        reason="inconclusive",
    )

    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"


def test_state_verification_imports_are_exported():
    import computer_agent.verification as verification

    assert verification.StateVerifier is StateVerifier
    assert verification.StateVerificationResult is StateVerificationResult
    assert verification.StateVerificationStatus is StateVerificationStatus
