from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from computer_agent.perception import (
    ScreenCapture,
    ScreenFrame,
)


def test_capture_builds_screen_frame(tmp_path):
    output_path = tmp_path / "screen.png"

    Image.new(
        "RGB",
        (2940, 1912),
    ).save(output_path)

    controller = Mock()
    controller.capture_screenshot.return_value = output_path
    controller.get_screen_size.return_value = (
        1470,
        956,
    )

    capture = ScreenCapture(controller)
    frame = capture.capture(output_path)

    controller.capture_screenshot.assert_called_once_with(
        str(output_path)
    )
    controller.get_screen_size.assert_called_once()

    assert frame.image_path == output_path.resolve()
    assert frame.pixel_size == (2940, 1912)
    assert frame.screen_size == (1470, 956)
    assert frame.scale_x == 2.0
    assert frame.scale_y == 2.0
    assert frame.captured_at.tzinfo is not None


def test_capture_rejects_missing_file(tmp_path):
    output_path = tmp_path / "missing.png"

    controller = Mock()
    controller.capture_screenshot.return_value = output_path

    capture = ScreenCapture(controller)

    with pytest.raises(
        FileNotFoundError,
        match="Screenshot file was not created",
    ):
        capture.capture(output_path)


def test_screen_frame_rejects_invalid_dimensions():
    with pytest.raises(
        ValueError,
        match="dimensions must be positive",
    ):
        ScreenFrame(
            image_path=Path("screen.png"),
            pixel_width=0,
            pixel_height=1080,
            screen_width=1920,
            screen_height=1080,
        )


def test_screen_frame_rejects_naive_timestamp():
    with pytest.raises(
        ValueError,
        match="captured_at must be timezone-aware",
    ):
        ScreenFrame(
            image_path=Path("screen.png"),
            pixel_width=1920,
            pixel_height=1080,
            screen_width=1920,
            screen_height=1080,
            captured_at=datetime(2026, 8, 23, 12, 0, 0),
        )


def test_screen_frame_accepts_timezone_aware_timestamp():
    captured_at = datetime(
        2026,
        8,
        23,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )

    frame = ScreenFrame(
        image_path=Path("screen.png"),
        pixel_width=1920,
        pixel_height=1080,
        screen_width=1920,
        screen_height=1080,
        captured_at=captured_at,
    )

    assert frame.captured_at is captured_at
