from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

import computer_agent.perception.engine as engine_module
from computer_agent.perception import (
    BoundingBox,
    PerceptionEngine,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)


def _captured_at(second=0):
    return datetime(
        2026,
        8,
        28,
        12,
        0,
        second,
        tzinfo=timezone.utc,
    )


def _frame(
    image_path,
    *,
    pixel_width=100,
    pixel_height=80,
    screen_width=50,
    screen_height=40,
    second=0,
):
    return ScreenFrame(
        image_path=Path(image_path),
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        screen_width=screen_width,
        screen_height=screen_height,
        captured_at=_captured_at(second),
    )


def _element(
    text,
    *,
    x=10,
    y=20,
    width=20,
    height=10,
    element_type="text",
    confidence=0.75,
    source=None,
):
    return UIElement(
        element_type=element_type,
        bounding_box=BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
        confidence=confidence,
        text=text,
        source=source,
    )


def _save_image(
    path,
    *,
    size=(100, 80),
    mode="RGBA",
    color=(10, 20, 30, 255),
):
    Image.new(
        mode,
        size,
        color,
    ).save(path)


class FakeScreenCapture:
    def __init__(
        self,
        frames=(),
        *,
        error=None,
    ):
        self.frames = list(frames)
        self.error = error
        self.calls = []

    def capture(self, output_path):
        self.calls.append(output_path)

        if self.error is not None:
            raise self.error

        return self.frames.pop(0)


class FakeAccessibilityReader:
    def __init__(
        self,
        responses=(),
        *,
        error=None,
    ):
        self.responses = list(responses)
        self.error = error
        self.calls = 0

    def read_frontmost_controls(self):
        self.calls += 1

        if self.error is not None:
            raise self.error

        return self.responses.pop(0)


class FakeOCR:
    def __init__(
        self,
        responses=(),
        *,
        error=None,
    ):
        self.responses = list(responses)
        self.error = error
        self.calls = 0
        self.images = []

    def recognize(self, image):
        self.calls += 1
        self.images.append(image)

        if self.error is not None:
            raise self.error

        return self.responses.pop(0)


class RecordingFusion:
    def __init__(
        self,
        responses=(),
        *,
        error=None,
    ):
        self.responses = list(responses)
        self.error = error
        self.calls = []

    def fuse(
        self,
        accessibility_elements,
        ocr_elements,
    ):
        self.calls.append(
            (
                accessibility_elements,
                ocr_elements,
            )
        )

        if self.error is not None:
            raise self.error

        return self.responses.pop(0)


def _engine(
    *,
    screen_capture,
    accessibility_reader,
    ocr,
    fusion,
    capture_path,
):
    return PerceptionEngine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=capture_path,
    )


def test_complete_observation_loads_rgb_maps_ocr_and_fuses(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    capture_path = tmp_path / "capture.png"
    frame = _frame(image_path)
    accessibility_element = _element(
        "Native",
        element_type="button",
        source="accessibility",
    )
    pixel_ocr_element = _element(
        "OCR",
        x=20,
        y=10,
        width=40,
        height=20,
        confidence=0.8,
    )
    logical_ocr_element = _element(
        "OCR",
        x=10,
        y=5,
        width=20,
        height=10,
        confidence=0.8,
    )
    fused_element = _element(
        "Fused",
        source="hybrid",
    )
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            [
                accessibility_element,
            ]
        ]
    )
    ocr = FakeOCR(
        [
            [
                pixel_ocr_element,
            ]
        ]
    )
    fusion = RecordingFusion(
        [
            [
                fused_element,
            ]
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=str(capture_path),
    )

    snapshot = engine.observe()
    image_path.unlink()

    assert screen_capture.calls == [capture_path]
    assert accessibility_reader.calls == 1
    assert ocr.calls == 1
    assert ocr.images[0].mode == "RGB"
    assert ocr.images[0].size == frame.pixel_size
    assert fusion.calls == [
        (
            (accessibility_element,),
            (logical_ocr_element,),
        )
    ]
    assert snapshot.frame is frame
    assert snapshot.image.mode == "RGB"
    assert snapshot.image.size == frame.pixel_size
    assert snapshot.image.getpixel((0, 0)) == (10, 20, 30)
    assert snapshot.accessibility_elements == (accessibility_element,)
    assert snapshot.ocr_elements == (logical_ocr_element,)
    assert snapshot.fused_elements == (fused_element,)
    assert snapshot.warnings == ()
    assert snapshot.source_counts == {
        "accessibility": 1,
        "ocr": 1,
        "fused": 1,
    }


def test_accessibility_only_partial_success_when_ocr_fails(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    frame = _frame(image_path)
    accessibility_element = _element(
        "Native",
        source="accessibility",
    )
    fused_element = _element(
        "Native",
        source="accessibility",
    )
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            [
                accessibility_element,
            ]
        ]
    )
    ocr = FakeOCR(
        error=RuntimeError("tesseract unavailable")
    )
    fusion = RecordingFusion(
        [
            [
                fused_element,
            ]
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    snapshot = engine.observe()

    assert snapshot.accessibility_elements == (accessibility_element,)
    assert snapshot.ocr_elements == ()
    assert snapshot.fused_elements == (fused_element,)
    assert fusion.calls == [
        (
            (accessibility_element,),
            (),
        )
    ]
    assert len(snapshot.warnings) == 1
    assert snapshot.warnings[0].startswith("OCR observation failed:")
    assert "RuntimeError: tesseract unavailable" in snapshot.warnings[0]


def test_ocr_mapping_failure_is_ocr_partial_failure(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    frame = _frame(image_path)
    accessibility_element = _element(
        "Native",
        source="accessibility",
    )
    pixel_ocr_element = _element(
        "OCR",
        x=20,
        y=10,
        width=40,
        height=20,
    )
    fused_element = _element(
        "Native",
        source="accessibility",
    )

    class FailingMapper:
        def __init__(
            self,
            frame,
        ):
            self.frame = frame

        def pixel_element_to_logical(
            self,
            element,
        ):
            raise ValueError("mapping failed")

    monkeypatch.setattr(
        engine_module,
        "ScreenCoordinateMapper",
        FailingMapper,
    )
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            [
                accessibility_element,
            ]
        ]
    )
    ocr = FakeOCR(
        [
            [
                pixel_ocr_element,
            ]
        ]
    )
    fusion = RecordingFusion(
        [
            [
                fused_element,
            ]
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    snapshot = engine.observe()

    assert snapshot.accessibility_elements == (accessibility_element,)
    assert snapshot.ocr_elements == ()
    assert snapshot.fused_elements == (fused_element,)
    assert fusion.calls == [
        (
            (accessibility_element,),
            (),
        )
    ]
    assert len(snapshot.warnings) == 1
    assert snapshot.warnings[0].startswith("OCR observation failed:")
    assert "ValueError: mapping failed" in snapshot.warnings[0]


def test_ocr_only_partial_success_when_accessibility_fails(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    frame = _frame(image_path)
    pixel_ocr_element = _element(
        "OCR",
        x=20,
        y=10,
        width=40,
        height=20,
    )
    logical_ocr_element = _element(
        "OCR",
        x=10,
        y=5,
        width=20,
        height=10,
    )
    fused_element = _element(
        "OCR",
        source="ocr",
    )
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        error=RuntimeError("accessibility unavailable")
    )
    ocr = FakeOCR(
        [
            [
                pixel_ocr_element,
            ]
        ]
    )
    fusion = RecordingFusion(
        [
            [
                fused_element,
            ]
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    snapshot = engine.observe()

    assert snapshot.accessibility_elements == ()
    assert snapshot.ocr_elements == (logical_ocr_element,)
    assert snapshot.fused_elements == (fused_element,)
    assert fusion.calls == [
        (
            (),
            (logical_ocr_element,),
        )
    ]
    assert len(snapshot.warnings) == 1
    assert snapshot.warnings[0].startswith(
        "Accessibility observation failed:"
    )
    assert "RuntimeError: accessibility unavailable" in snapshot.warnings[0]


def test_both_sources_fail_with_deterministic_warning_order(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    frame = _frame(image_path)
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        error=RuntimeError("permission denied")
    )
    ocr = FakeOCR(
        error=ValueError("bad OCR")
    )
    fusion = RecordingFusion(
        [
            []
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    snapshot = engine.observe()

    assert snapshot.accessibility_elements == ()
    assert snapshot.ocr_elements == ()
    assert snapshot.fused_elements == ()
    assert fusion.calls == [
        (
            (),
            (),
        )
    ]
    assert len(snapshot.warnings) == 2
    assert snapshot.warnings[0].startswith(
        "Accessibility observation failed:"
    )
    assert snapshot.warnings[1].startswith("OCR observation failed:")


def test_image_size_mismatch_fails_before_source_calls(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(
        image_path,
        size=(101, 80),
    )
    frame = _frame(
        image_path,
        pixel_width=100,
        pixel_height=80,
    )
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            []
        ]
    )
    ocr = FakeOCR(
        [
            []
        ]
    )
    fusion = RecordingFusion(
        [
            []
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"frame pixel size \(100, 80\), "
            r"loaded image size \(101, 80\)"
        ),
    ):
        engine.observe()

    assert accessibility_reader.calls == 0
    assert ocr.calls == 0
    assert fusion.calls == []


def test_capture_failure_propagates_before_later_dependencies(tmp_path):
    error = RuntimeError("capture failed")
    screen_capture = FakeScreenCapture(
        error=error
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            []
        ]
    )
    ocr = FakeOCR(
        [
            []
        ]
    )
    fusion = RecordingFusion(
        [
            []
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    with pytest.raises(RuntimeError) as raised:
        engine.observe()

    assert raised.value is error
    assert accessibility_reader.calls == 0
    assert ocr.calls == 0
    assert fusion.calls == []


def test_fusion_failure_propagates_without_returning_snapshot(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    frame = _frame(image_path)
    error = RuntimeError("fusion failed")
    screen_capture = FakeScreenCapture(
        [frame]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            []
        ]
    )
    ocr = FakeOCR(
        [
            []
        ]
    )
    fusion = RecordingFusion(
        error=error
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=tmp_path / "capture.png",
    )

    with pytest.raises(RuntimeError) as raised:
        engine.observe()

    assert raised.value is error
    assert accessibility_reader.calls == 1
    assert ocr.calls == 1
    assert fusion.calls == [
        (
            (),
            (),
        )
    ]


def test_repeated_observations_are_fresh(tmp_path):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _save_image(
        first_path,
        color=(10, 20, 30, 255),
    )
    _save_image(
        second_path,
        color=(40, 50, 60, 255),
    )
    first_frame = _frame(
        first_path,
        second=1,
    )
    second_frame = _frame(
        second_path,
        second=2,
    )
    first_accessibility = _element(
        "First Accessibility",
        source="accessibility",
    )
    second_accessibility = _element(
        "Second Accessibility",
        source="accessibility",
    )
    first_pixel_ocr = _element(
        "First OCR",
        x=20,
        y=10,
        width=40,
        height=20,
    )
    second_pixel_ocr = _element(
        "Second OCR",
        x=40,
        y=20,
        width=20,
        height=20,
    )
    first_logical_ocr = _element(
        "First OCR",
        x=10,
        y=5,
        width=20,
        height=10,
    )
    second_logical_ocr = _element(
        "Second OCR",
        x=20,
        y=10,
        width=10,
        height=10,
    )
    first_fused = _element(
        "First Fused",
        source="hybrid",
    )
    second_fused = _element(
        "Second Fused",
        source="hybrid",
    )
    capture_path = tmp_path / "capture.png"
    screen_capture = FakeScreenCapture(
        [
            first_frame,
            second_frame,
        ]
    )
    accessibility_reader = FakeAccessibilityReader(
        [
            [
                first_accessibility,
            ],
            [
                second_accessibility,
            ],
        ]
    )
    ocr = FakeOCR(
        [
            [
                first_pixel_ocr,
            ],
            [
                second_pixel_ocr,
            ],
        ]
    )
    fusion = RecordingFusion(
        [
            [
                first_fused,
            ],
            [
                second_fused,
            ],
        ]
    )
    engine = _engine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=capture_path,
    )

    first_snapshot = engine.observe()
    second_snapshot = engine.observe()

    assert screen_capture.calls == [
        capture_path,
        capture_path,
    ]
    assert accessibility_reader.calls == 2
    assert ocr.calls == 2
    assert len(fusion.calls) == 2
    assert first_snapshot is not second_snapshot
    assert first_snapshot.frame is first_frame
    assert second_snapshot.frame is second_frame
    assert first_snapshot.image is not second_snapshot.image
    assert first_snapshot.image.getpixel((0, 0)) == (10, 20, 30)
    assert second_snapshot.image.getpixel((0, 0)) == (40, 50, 60)
    assert first_snapshot.accessibility_elements is not (
        second_snapshot.accessibility_elements
    )
    assert first_snapshot.ocr_elements is not second_snapshot.ocr_elements
    assert first_snapshot.fused_elements is not second_snapshot.fused_elements
    assert first_snapshot.accessibility_elements == (first_accessibility,)
    assert second_snapshot.accessibility_elements == (second_accessibility,)
    assert first_snapshot.ocr_elements == (first_logical_ocr,)
    assert second_snapshot.ocr_elements == (second_logical_ocr,)
    assert first_snapshot.fused_elements == (first_fused,)
    assert second_snapshot.fused_elements == (second_fused,)


def test_source_counts_are_computed_without_mutable_stored_state():
    snapshot = PerceptionSnapshot(
        frame=_frame("screen.png"),
        image=Image.new(
            "RGB",
            (100, 80),
        ),
        accessibility_elements=(
            _element("A1"),
            _element("A2"),
        ),
        ocr_elements=(
            _element("OCR"),
        ),
        fused_elements=(
            _element("F1"),
            _element("F2"),
            _element("F3"),
        ),
        warnings=(),
    )

    counts = snapshot.source_counts
    counts["accessibility"] = 99

    assert snapshot.source_counts == {
        "accessibility": 2,
        "ocr": 1,
        "fused": 3,
    }
    assert snapshot.source_counts is not snapshot.source_counts


def test_engine_requires_only_observation_dependencies(tmp_path):
    image_path = tmp_path / "screen.png"
    _save_image(image_path)
    capture_path = tmp_path / "capture.png"

    class CaptureOnly:
        def capture(self, output_path):
            assert output_path == capture_path

            return _frame(image_path)

    class AccessibilityOnly:
        def read_frontmost_controls(self):
            return []

    class OCROnly:
        def recognize(self, image):
            assert image.mode == "RGB"

            return []

    class FusionOnly:
        def fuse(
            self,
            accessibility_elements,
            ocr_elements,
        ):
            assert accessibility_elements == ()
            assert ocr_elements == ()

            return []

    snapshot = PerceptionEngine(
        screen_capture=CaptureOnly(),
        accessibility_reader=AccessibilityOnly(),
        ocr=OCROnly(),
        fusion=FusionOnly(),
        capture_path=capture_path,
    ).observe()

    assert snapshot.source_counts == {
        "accessibility": 0,
        "ocr": 0,
        "fused": 0,
    }
