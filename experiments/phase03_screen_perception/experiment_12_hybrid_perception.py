"""Phase 03 experiment for hybrid Accessibility and OCR perception."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import time
from typing import Any

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import ComputerController
from computer_agent.core.models import Action
from computer_agent.perception import (
    BoundingBox,
    MacOSAccessibility,
    ScreenCapture,
    ScreenCoordinateMapper,
    ScreenFrame,
    TesseractOCR,
    UIElement,
    UIElementFusion,
    normalize_ui_text,
    smaller_area_overlap_ratio,
)
from computer_agent.tools.computer import create_computer_tools
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase03_screen_perception"
    / "experiment_12_hybrid_perception.html"
)
BEFORE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_12_hybrid_perception_before.png"
)
PLAN_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_12_hybrid_perception_plan.png"
)
AFTER_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_12_hybrid_perception_after.png"
)

TARGET_INPUT_TEXT = "TARGET_INPUT_12"
TARGET_INPUT_IDENTIFIER = "hybrid-target-input"
NATIVE_BUTTON_TEXT = "NATIVE_BUTTON_12"
CANVAS_ACTION_TEXT = "CANVAS_ACTION_12"
TYPED_VALUE = "HYBRID_INPUT_VALUE_12"
VERIFICATION_TEXT = "FUSION_VERIFIED_12"
ERROR_TEXT = "INPUT_REQUIRED_12"

OCR_MINIMUM_CONFIDENCE = 0.05
ACTION_CONFIDENCE = 0.70
COMPLETION_OCR_CONFIDENCE_FLOOR = 0.60
REGIONAL_EXPANSION_LOGICAL_PIXELS = 40
REGIONAL_OVERLAP_RATIO = 0.80
COMPLETION_SAMPLE_EXPANSION_LOGICAL_PIXELS = 10
COMPLETED_CANVAS_RGB = (197, 243, 186)
COMPLETED_CANVAS_RGB_TOLERANCE = 15
COMPLETION_SAMPLE_DARK_MEAN_THRESHOLD = 120
MIN_COMPLETION_SAMPLE_PIXELS = 100
CAPTURE_COUNTDOWN_SECONDS = 5
COUNTDOWN_SECONDS = 3
SAFE_EDGE_MARGIN = 10
POSITION_TOLERANCE = 1
MOVE_DURATION = 1.0
TYPE_INTERVAL = 0.05
FOCUS_SETTLE_SECONDS = 0.4
TYPE_SETTLE_SECONDS = 0.5
CANVAS_SETTLE_SECONDS = 0.6
MAX_OBSERVATION_ATTEMPTS = 5
OBSERVATION_RETRY_DELAY_SECONDS = 0.25

INPUT_COLOR = (0, 96, 255)
NATIVE_BUTTON_COLOR = (220, 0, 0)
CANVAS_COLOR = (0, 150, 70)
POINT_COLOR = (0, 0, 0)


@dataclass(frozen=True, slots=True)
class HybridObservation:
    """A screenshot frame, image, and fused perception results."""

    frame: ScreenFrame
    image: Image.Image
    accessibility_elements: tuple[UIElement, ...]
    ocr_elements: tuple[UIElement, ...]
    fused_elements: tuple[UIElement, ...]


@dataclass(frozen=True, slots=True)
class HybridTargets:
    """The three required fixture targets after fusion validation."""

    target_input: UIElement
    native_button: UIElement
    canvas_action: UIElement


@dataclass(frozen=True, slots=True)
class ValidatedHybridState:
    """A fresh hybrid observation and its safe planned action points."""

    observation: HybridObservation
    targets: HybridTargets
    input_click_point: tuple[int, int]
    canvas_click_point: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RegionalOCRSearch:
    """A dynamic regional OCR pass and its logical OCR elements."""

    region: BoundingBox
    elements: tuple[UIElement, ...]


@dataclass(frozen=True, slots=True)
class RegionalOCRMatch:
    """A recovered OCR target from a dynamic regional crop."""

    search: RegionalOCRSearch
    target: UIElement
    overlap_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class CanvasBackgroundSample:
    """Median sampled Canvas background state."""

    logical_box: BoundingBox
    pixel_box: tuple[int, int, int, int]
    usable_pixel_count: int
    median_rgb: tuple[int, int, int]
    classification: str


def _count_down_to_capture() -> None:
    print(
        "Switch to the already-open and freshly reloaded Experiment 12 "
        "Chrome fixture before the initial screenshot is captured."
    )

    for remaining in range(CAPTURE_COUNTDOWN_SECONDS, 0, -1):
        print(f"Capturing in {remaining}...")
        time.sleep(1)


def _count_down_to_movement() -> None:
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Moving in {remaining}...")
        time.sleep(1)


def _normalized(text: str) -> str:
    return normalize_ui_text(text)


def _target_point(element: UIElement) -> tuple[int, int]:
    box = element.bounding_box
    x = min(
        max(round(element.center[0]), box.left),
        box.right - 1,
    )
    y = min(
        max(round(element.center[1]), box.top),
        box.bottom - 1,
    )

    return x, y


def _box_inside_screen(
    element: UIElement,
    screen_size: tuple[int, int],
) -> bool:
    box = element.bounding_box
    width, height = screen_size

    return 0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height


def _box_in_safe_area(
    element: UIElement,
    screen_size: tuple[int, int],
) -> bool:
    box = element.bounding_box
    width, height = screen_size

    return (
        SAFE_EDGE_MARGIN <= box.left
        and SAFE_EDGE_MARGIN <= box.top
        and box.right <= width - SAFE_EDGE_MARGIN
        and box.bottom <= height - SAFE_EDGE_MARGIN
    )


def _point_inside_screen(
    point: tuple[int, int],
    screen_size: tuple[int, int],
) -> bool:
    x, y = point
    width, height = screen_size

    return 0 <= x < width and 0 <= y < height


def _point_in_safe_area(
    point: tuple[int, int],
    screen_size: tuple[int, int],
) -> bool:
    x, y = point
    width, height = screen_size

    return (
        SAFE_EDGE_MARGIN <= x <= width - SAFE_EDGE_MARGIN
        and SAFE_EDGE_MARGIN <= y <= height - SAFE_EDGE_MARGIN
    )


def _point_in_fail_safe_corner(
    point: tuple[int, int],
    screen_size: tuple[int, int],
) -> bool:
    x, y = point
    width, height = screen_size

    return (
        (x <= 0 and y <= 0)
        or (x <= 0 and y >= height - 1)
        or (x >= width - 1 and y <= 0)
        or (x >= width - 1 and y >= height - 1)
    )


def _within_tolerance(
    actual: tuple[int, int],
    expected: tuple[int, int],
) -> bool:
    return (
        abs(actual[0] - expected[0]) <= POSITION_TOLERANCE
        and abs(actual[1] - expected[1]) <= POSITION_TOLERANCE
    )


def _require_positive_contained_safe_box(
    element: UIElement,
    screen_size: tuple[int, int],
    label: str,
) -> None:
    box = element.bounding_box

    if box.width <= 0 or box.height <= 0:
        raise RuntimeError(f"{label} bounding box had zero size")

    if not _box_inside_screen(element, screen_size):
        raise RuntimeError(f"{label} bounding box was outside the screen")

    if not _box_in_safe_area(element, screen_size):
        raise RuntimeError(
            f"{label} bounding box was inside the {SAFE_EDGE_MARGIN}-pixel "
            "screen-edge margin"
        )


def _require_safe_click_point(
    point: tuple[int, int],
    element: UIElement,
    screen_size: tuple[int, int],
    label: str,
) -> None:
    if not element.bounding_box.contains_point(*point):
        raise RuntimeError(f"{label} click point was outside its target box")

    if not _point_inside_screen(point, screen_size):
        raise RuntimeError(f"{label} click point was outside the screen")

    if not _point_in_safe_area(point, screen_size):
        raise RuntimeError(
            f"{label} click point was inside the {SAFE_EDGE_MARGIN}-pixel "
            "screen-edge margin"
        )


def _capture_hybrid_observation(
    controller: ComputerController,
    reader: MacOSAccessibility,
    output_path: Path,
) -> HybridObservation:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = ScreenCapture(controller).capture(output_path)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    if image.size != frame.pixel_size:
        raise RuntimeError(
            "screenshot size did not match its ScreenFrame pixel metadata"
        )

    accessibility_elements = tuple(reader.read_frontmost_controls())
    ocr_pixel_elements = TesseractOCR(
        minimum_confidence=OCR_MINIMUM_CONFIDENCE,
        page_segmentation_mode=6,
        group_words_by_line=True,
    ).recognize(image)

    mapper = ScreenCoordinateMapper(frame)
    ocr_elements = tuple(
        mapper.pixel_element_to_logical(element)
        for element in ocr_pixel_elements
    )
    fused_elements = UIElementFusion().fuse(
        accessibility_elements,
        ocr_elements,
    )

    return HybridObservation(
        frame=frame,
        image=image,
        accessibility_elements=accessibility_elements,
        ocr_elements=ocr_elements,
        fused_elements=fused_elements,
    )


def _matches_text(
    element: UIElement,
    text: str,
) -> bool:
    return normalize_ui_text(element.text) == _normalized(text)


def _find_by_text(
    elements: tuple[UIElement, ...],
    text: str,
) -> list[UIElement]:
    return [
        element
        for element in elements
        if _matches_text(element, text)
    ]


def _find_by_normalized_text(
    elements: tuple[UIElement, ...],
    normalized_text: str,
) -> list[UIElement]:
    return [
        element
        for element in elements
        if normalize_ui_text(element.text) == normalized_text
    ]


def _find_ocr_by_text(
    elements: tuple[UIElement, ...],
    text: str,
) -> list[UIElement]:
    return [
        element
        for element in elements
        if _matches_text(element, text) and element.source == "ocr"
    ]


def _require_one_by_text(
    elements: tuple[UIElement, ...],
    text: str,
) -> UIElement:
    matches = _find_by_text(elements, text)

    if len(matches) != 1:
        raise RuntimeError(
            f"{text!r} expected exactly one result, found {len(matches)}"
        )

    return matches[0]


def _with_ocr_source(
    element: UIElement,
) -> UIElement:
    if element.source == "ocr":
        return element

    return UIElement(
        element_type=element.element_type,
        bounding_box=element.bounding_box,
        confidence=element.confidence,
        text=element.text,
        identifier=element.identifier,
        value=element.value,
        enabled=element.enabled,
        focused=element.focused,
        selected=element.selected,
        source="ocr",
    )


def _assert_text_absent(
    elements: tuple[UIElement, ...],
    text: str,
    stage_name: str,
) -> None:
    matches = _find_by_text(elements, text)

    if matches:
        raise RuntimeError(
            f"{stage_name}: unexpected {text!r} result was present"
        )


def _full_screen_status(
    matches: list[UIElement],
    text: str,
) -> str:
    if not matches:
        return "full-screen missing"

    if len(matches) == 1:
        return f"full-screen confidence {matches[0].confidence:.2f}"

    return f"full-screen {len(matches)} matches for {text!r}"


def _dynamic_pixel_region(
    frame: ScreenFrame,
    previous_target: UIElement,
) -> BoundingBox:
    box = previous_target.bounding_box
    left = math.floor(box.left * frame.scale_x)
    top = math.floor(box.top * frame.scale_y)
    right = math.ceil(box.right * frame.scale_x)
    bottom = math.ceil(box.bottom * frame.scale_y)
    expand_x = math.ceil(REGIONAL_EXPANSION_LOGICAL_PIXELS * frame.scale_x)
    expand_y = math.ceil(REGIONAL_EXPANSION_LOGICAL_PIXELS * frame.scale_y)

    left = max(0, left - expand_x)
    top = max(0, top - expand_y)
    right = min(frame.pixel_width, right + expand_x)
    bottom = min(frame.pixel_height, bottom + expand_y)

    if left >= right or top >= bottom:
        raise RuntimeError("dynamic regional OCR crop was empty")

    return BoundingBox(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )


def _recognize_dynamic_region(
    observation: HybridObservation,
    previous_target: UIElement,
) -> RegionalOCRSearch:
    region = _dynamic_pixel_region(
        observation.frame,
        previous_target,
    )
    ocr = TesseractOCR(
        minimum_confidence=OCR_MINIMUM_CONFIDENCE,
        page_segmentation_mode=7,
        group_words_by_line=True,
    )
    pixel_elements = ocr.recognize_region(
        observation.image,
        region,
    )
    mapper = ScreenCoordinateMapper(observation.frame)
    logical_elements = tuple(
        _with_ocr_source(
            mapper.pixel_element_to_logical(element)
        )
        for element in pixel_elements
    )

    return RegionalOCRSearch(
        region=region,
        elements=logical_elements,
    )


def _require_regional_target(
    search: RegionalOCRSearch,
    expected_normalized_text: str,
    expected_label: str,
    minimum_confidence: float = ACTION_CONFIDENCE,
) -> UIElement:
    matches = _find_by_normalized_text(
        search.elements,
        expected_normalized_text,
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"{expected_label!r} regional OCR expected exactly one result, "
            f"found {len(matches)}"
        )

    target = matches[0]
    if target.confidence < minimum_confidence:
        raise RuntimeError(
            f"{expected_label!r} regional OCR confidence was "
            f"{target.confidence:.2f}, expected at least "
            f"{minimum_confidence:.2f}"
        )

    return target


def _recover_regional_target(
    observation: HybridObservation,
    previous_target: UIElement,
    expected_normalized_text: str,
    expected_label: str,
    full_screen_status: str,
    *,
    require_overlap: bool = False,
    search: RegionalOCRSearch | None = None,
    minimum_confidence: float = ACTION_CONFIDENCE,
) -> RegionalOCRMatch:
    if search is None:
        search = _recognize_dynamic_region(
            observation,
            previous_target,
        )

    target = _require_regional_target(
        search,
        expected_normalized_text,
        expected_label,
        minimum_confidence=minimum_confidence,
    )
    overlap_ratio = None

    if require_overlap:
        overlap_ratio = smaller_area_overlap_ratio(
            previous_target.bounding_box,
            target.bounding_box,
        )
        if overlap_ratio < REGIONAL_OVERLAP_RATIO:
            raise RuntimeError(
                f"{expected_label!r} regional OCR overlap ratio was "
                f"{overlap_ratio:.2f}, expected at least "
                f"{REGIONAL_OVERLAP_RATIO:.2f}"
            )

    print(f"Regional OCR recovery for {expected_label}:")
    print(f"  full-screen status: {full_screen_status}")
    print(f"  dynamic pixel crop: {search.region}")
    print(f"  recovered text: {target.text!r}")
    print(f"  recovered confidence: {target.confidence:.2f}")
    print(f"  recovered logical bounding box: {target.bounding_box}")

    if overlap_ratio is not None:
        print(f"  overlap ratio: {overlap_ratio:.2f}")

    return RegionalOCRMatch(
        search=search,
        target=target,
        overlap_ratio=overlap_ratio,
    )


def _resolve_ocr_target(
    observation: HybridObservation,
    previous_target: UIElement,
    expected_text: str,
    *,
    require_overlap: bool = False,
) -> RegionalOCRMatch:
    matches = _find_ocr_by_text(
        observation.fused_elements,
        expected_text,
    )

    if len(matches) > 1:
        raise RuntimeError(
            f"{expected_text!r} expected at most one full-screen OCR result, "
            f"found {len(matches)}"
        )

    if len(matches) == 1 and matches[0].confidence >= ACTION_CONFIDENCE:
        return RegionalOCRMatch(
            search=RegionalOCRSearch(
                region=_dynamic_pixel_region(
                    observation.frame,
                    matches[0],
                ),
                elements=(),
            ),
            target=matches[0],
        )

    return _recover_regional_target(
        observation,
        previous_target,
        _normalized(expected_text),
        expected_text,
        _full_screen_status(
            matches,
            expected_text,
        ),
        require_overlap=require_overlap,
    )


def _completion_sample_logical_box(
    frame: ScreenFrame,
    last_canvas_action: UIElement,
) -> BoundingBox:
    box = last_canvas_action.bounding_box
    left = max(
        0,
        box.left - COMPLETION_SAMPLE_EXPANSION_LOGICAL_PIXELS,
    )
    top = max(
        0,
        box.top - COMPLETION_SAMPLE_EXPANSION_LOGICAL_PIXELS,
    )
    right = min(
        frame.screen_width,
        box.right + COMPLETION_SAMPLE_EXPANSION_LOGICAL_PIXELS,
    )
    bottom = min(
        frame.screen_height,
        box.bottom + COMPLETION_SAMPLE_EXPANSION_LOGICAL_PIXELS,
    )

    if left >= right or top >= bottom:
        raise RuntimeError("completion Canvas logical sample box was empty")

    return BoundingBox(
        x=left,
        y=top,
        width=right - left,
        height=bottom - top,
    )


def _logical_box_to_pixel_bounds(
    frame: ScreenFrame,
    box: BoundingBox,
) -> tuple[int, int, int, int]:
    left = max(
        0,
        math.floor(box.left * frame.scale_x),
    )
    top = max(
        0,
        math.floor(box.top * frame.scale_y),
    )
    right = min(
        frame.pixel_width,
        math.ceil(box.right * frame.scale_x),
    )
    bottom = min(
        frame.pixel_height,
        math.ceil(box.bottom * frame.scale_y),
    )

    if left >= right or top >= bottom:
        raise RuntimeError("completion Canvas pixel sample box was empty")

    return left, top, right, bottom


def _rgb_delta(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        abs(first[0] - second[0]),
        abs(first[1] - second[1]),
        abs(first[2] - second[2]),
    )


def _classify_canvas_background(
    median_rgb: tuple[int, int, int],
) -> str:
    channel_delta = _rgb_delta(
        median_rgb,
        COMPLETED_CANVAS_RGB,
    )

    if max(channel_delta) <= COMPLETED_CANVAS_RGB_TOLERANCE:
        return "completed"

    return "not_completed"


def _sample_canvas_background(
    observation: HybridObservation,
    last_canvas_action: UIElement,
) -> CanvasBackgroundSample:
    logical_box = _completion_sample_logical_box(
        observation.frame,
        last_canvas_action,
    )
    pixel_box = _logical_box_to_pixel_bounds(
        observation.frame,
        logical_box,
    )
    crop = observation.image.crop(pixel_box)
    usable_pixels = [
        pixel[:3]
        for pixel in crop.get_flattened_data()
        if sum(pixel[:3]) / 3 >= COMPLETION_SAMPLE_DARK_MEAN_THRESHOLD
    ]

    if len(usable_pixels) < MIN_COMPLETION_SAMPLE_PIXELS:
        raise RuntimeError(
            "not enough non-dark Canvas pixels were available for "
            "completion background classification"
        )

    median_rgb = (
        round(median(pixel[0] for pixel in usable_pixels)),
        round(median(pixel[1] for pixel in usable_pixels)),
        round(median(pixel[2] for pixel in usable_pixels)),
    )

    return CanvasBackgroundSample(
        logical_box=logical_box,
        pixel_box=pixel_box,
        usable_pixel_count=len(usable_pixels),
        median_rgb=median_rgb,
        classification=_classify_canvas_background(median_rgb),
    )


def _print_composite_completion_result(
    verification: UIElement,
    sample: CanvasBackgroundSample,
) -> None:
    print(f"Completion OCR text: {verification.text!r}")
    print(f"Completion OCR confidence: {verification.confidence:.2f}")
    print(f"Dynamic logical sample box: {sample.logical_box}")
    print(f"Dynamic pixel sample box: {sample.pixel_box}")
    print(f"Usable pixel count: {sample.usable_pixel_count}")
    print(f"Median RGB: {sample.median_rgb}")
    print(f"Expected completed RGB: {COMPLETED_CANVAS_RGB}")
    print(f"Color classification: {sample.classification}")

    if verification.confidence < ACTION_CONFIDENCE:
        print(
            "Completion OCR is accepted only through composite verification, "
            "not as action-confidence OCR."
        )

    print("Composite verification result: passed")


def _validate_initial_targets(
    observation: HybridObservation,
    *,
    expected_input_value: str,
    require_canvas_action: bool = True,
) -> HybridTargets:
    frame = observation.frame
    fused = observation.fused_elements

    target_input = _require_one_by_text(
        fused,
        TARGET_INPUT_TEXT,
    )
    native_button = _require_one_by_text(
        fused,
        NATIVE_BUTTON_TEXT,
    )

    if require_canvas_action:
        canvas_action = _require_one_by_text(
            fused,
            CANVAS_ACTION_TEXT,
        )
    else:
        canvas_action = UIElement(
            element_type="text",
            bounding_box=BoundingBox(
                x=SAFE_EDGE_MARGIN,
                y=SAFE_EDGE_MARGIN,
                width=1,
                height=1,
            ),
        )

    _validate_target_input(
        target_input,
        expected_input_value,
        frame,
    )
    _validate_native_button(
        native_button,
        frame,
    )

    if require_canvas_action:
        _validate_canvas_action(
            canvas_action,
            CANVAS_ACTION_TEXT,
            frame,
        )

    return HybridTargets(
        target_input=target_input,
        native_button=native_button,
        canvas_action=canvas_action,
    )


def _validate_target_input(
    target_input: UIElement,
    expected_value: str,
    frame: ScreenFrame,
) -> None:
    if target_input.element_type != "text_field":
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} element_type was "
            f"{target_input.element_type!r}, expected 'text_field'"
        )

    if target_input.identifier != TARGET_INPUT_IDENTIFIER:
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} identifier was "
            f"{target_input.identifier!r}, expected "
            f"{TARGET_INPUT_IDENTIFIER!r}"
        )

    if target_input.value != expected_value:
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} value was {target_input.value!r}, "
            f"expected {expected_value!r}"
        )

    if target_input.enabled is not True:
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} enabled was {target_input.enabled!r}, "
            "expected True"
        )

    if target_input.source != "accessibility":
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} source was {target_input.source!r}, "
            "expected 'accessibility'"
        )

    _require_positive_contained_safe_box(
        target_input,
        frame.screen_size,
        TARGET_INPUT_TEXT,
    )


def _validate_native_button(
    native_button: UIElement,
    frame: ScreenFrame,
) -> None:
    if native_button.element_type != "button":
        raise RuntimeError(
            f"{NATIVE_BUTTON_TEXT!r} element_type was "
            f"{native_button.element_type!r}, expected 'button'"
        )

    if native_button.identifier != "native-button":
        raise RuntimeError(
            f"{NATIVE_BUTTON_TEXT!r} identifier was "
            f"{native_button.identifier!r}, expected 'native-button'"
        )

    if native_button.enabled is not True:
        raise RuntimeError(
            f"{NATIVE_BUTTON_TEXT!r} enabled was {native_button.enabled!r}, "
            "expected True"
        )

    if native_button.source != "hybrid":
        raise RuntimeError(
            f"{NATIVE_BUTTON_TEXT!r} source was {native_button.source!r}, "
            "expected 'hybrid'"
        )

    _require_positive_contained_safe_box(
        native_button,
        frame.screen_size,
        NATIVE_BUTTON_TEXT,
    )


def _validate_canvas_action(
    canvas_action: UIElement,
    expected_text: str,
    frame: ScreenFrame,
) -> None:
    if canvas_action.source != "ocr":
        raise RuntimeError(
            f"{expected_text!r} source was {canvas_action.source!r}, "
            "expected 'ocr'"
        )

    if canvas_action.confidence < ACTION_CONFIDENCE:
        raise RuntimeError(
            f"{expected_text!r} OCR confidence was "
            f"{canvas_action.confidence:.2f}, expected at least "
            f"{ACTION_CONFIDENCE:.2f}"
        )

    _require_positive_contained_safe_box(
        canvas_action,
        frame.screen_size,
        expected_text,
    )


def _validate_action_points(
    targets: HybridTargets,
    screen_size: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    input_point = _target_point(targets.target_input)
    canvas_point = _target_point(targets.canvas_action)

    _require_safe_click_point(
        input_point,
        targets.target_input,
        screen_size,
        TARGET_INPUT_TEXT,
    )
    _require_safe_click_point(
        canvas_point,
        targets.canvas_action,
        screen_size,
        CANVAS_ACTION_TEXT,
    )

    return input_point, canvas_point


def _validate_hybrid_state(
    observation: HybridObservation,
    *,
    stage_name: str,
    expected_input_value: str,
    require_initial_canvas_action: bool = True,
    reject_completion_text: bool = False,
) -> ValidatedHybridState:
    if reject_completion_text:
        _assert_text_absent(
            observation.fused_elements,
            VERIFICATION_TEXT,
            stage_name,
        )
        _assert_text_absent(
            observation.fused_elements,
            ERROR_TEXT,
            stage_name,
        )

    targets = _validate_initial_targets(
        observation,
        expected_input_value=expected_input_value,
        require_canvas_action=require_initial_canvas_action,
    )
    input_point, canvas_point = _validate_action_points(
        targets,
        observation.frame.screen_size,
    )

    return ValidatedHybridState(
        observation=observation,
        targets=targets,
        input_click_point=input_point,
        canvas_click_point=canvas_point,
    )


def _is_accessibility_permission_failure(error: RuntimeError) -> bool:
    message = str(error)

    return (
        "macOS Accessibility frameworks are unavailable" in message
        or "macOS Accessibility permission is not trusted" in message
    )


def _copy_observation_image(
    observation: HybridObservation,
    output_path: Path,
) -> HybridObservation:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    observation.image.save(output_path)

    frame = ScreenFrame(
        image_path=output_path.resolve(),
        pixel_width=observation.frame.pixel_width,
        pixel_height=observation.frame.pixel_height,
        screen_width=observation.frame.screen_width,
        screen_height=observation.frame.screen_height,
        captured_at=observation.frame.captured_at,
    )

    return HybridObservation(
        frame=frame,
        image=observation.image,
        accessibility_elements=observation.accessibility_elements,
        ocr_elements=observation.ocr_elements,
        fused_elements=observation.fused_elements,
    )


def _observe_valid_hybrid_state(
    controller: ComputerController,
    reader: MacOSAccessibility,
    *,
    stage_name: str,
    expected_input_value: str,
    reject_completion_text: bool = False,
    output_path: Path | None = None,
    require_focus: bool = False,
) -> ValidatedHybridState:
    final_failure_reason = "unknown validation failure"

    with TemporaryDirectory(prefix="experiment_12_hybrid_observation_") as temp_dir:
        temp_root = Path(temp_dir)

        for attempt_number in range(1, MAX_OBSERVATION_ATTEMPTS + 1):
            capture_path = temp_root / f"{stage_name.lower()}_{attempt_number}.png"

            try:
                observation = _capture_hybrid_observation(
                    controller,
                    reader,
                    capture_path,
                )
                state = _validate_hybrid_state(
                    observation,
                    stage_name=stage_name,
                    expected_input_value=expected_input_value,
                    reject_completion_text=reject_completion_text,
                )

                if require_focus:
                    _require_target_focus(state.observation)

            except RuntimeError as error:
                if _is_accessibility_permission_failure(error):
                    raise

                final_failure_reason = str(error)
                print(
                    f"{stage_name} observation attempt "
                    f"{attempt_number}/{MAX_OBSERVATION_ATTEMPTS} failed: "
                    f"{final_failure_reason}"
                )

                if attempt_number < MAX_OBSERVATION_ATTEMPTS:
                    time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)

                continue

            if attempt_number > 1:
                print(
                    f"{stage_name} observation recovered on attempt "
                    f"{attempt_number}."
                )

            if output_path is not None:
                copied_observation = _copy_observation_image(
                    state.observation,
                    output_path,
                )
                state = ValidatedHybridState(
                    observation=copied_observation,
                    targets=state.targets,
                    input_click_point=state.input_click_point,
                    canvas_click_point=state.canvas_click_point,
                )

            return state

    raise RuntimeError(
        f"{stage_name} observation failed after "
        f"{MAX_OBSERVATION_ATTEMPTS} attempts; final failure: "
        f"{final_failure_reason}"
    )


def _validate_pre_canvas_observation(
    observation: HybridObservation,
    previous_canvas_action: UIElement,
) -> ValidatedHybridState:
    _assert_text_absent(
        observation.fused_elements,
        VERIFICATION_TEXT,
        "Post-typing",
    )
    _assert_text_absent(
        observation.fused_elements,
        ERROR_TEXT,
        "Post-typing",
    )

    target_input = _require_one_by_text(
        observation.fused_elements,
        TARGET_INPUT_TEXT,
    )
    native_button = _require_one_by_text(
        observation.fused_elements,
        NATIVE_BUTTON_TEXT,
    )
    _validate_target_input(
        target_input,
        TYPED_VALUE,
        observation.frame,
    )
    _validate_native_button(
        native_button,
        observation.frame,
    )

    canvas_match = _resolve_ocr_target(
        observation,
        previous_canvas_action,
        CANVAS_ACTION_TEXT,
        require_overlap=True,
    )
    _validate_canvas_action(
        canvas_match.target,
        CANVAS_ACTION_TEXT,
        observation.frame,
    )

    targets = HybridTargets(
        target_input=target_input,
        native_button=native_button,
        canvas_action=canvas_match.target,
    )
    input_point, canvas_point = _validate_action_points(
        targets,
        observation.frame.screen_size,
    )

    return ValidatedHybridState(
        observation=observation,
        targets=targets,
        input_click_point=input_point,
        canvas_click_point=canvas_point,
    )


def _observe_pre_canvas_state(
    controller: ComputerController,
    reader: MacOSAccessibility,
    previous_canvas_action: UIElement,
) -> ValidatedHybridState:
    final_failure_reason = "unknown validation failure"

    with TemporaryDirectory(prefix="experiment_12_pre_canvas_observation_") as temp_dir:
        temp_root = Path(temp_dir)

        for attempt_number in range(1, MAX_OBSERVATION_ATTEMPTS + 1):
            capture_path = temp_root / f"pre_canvas_{attempt_number}.png"

            try:
                observation = _capture_hybrid_observation(
                    controller,
                    reader,
                    capture_path,
                )
                state = _validate_pre_canvas_observation(
                    observation,
                    previous_canvas_action,
                )

            except RuntimeError as error:
                if _is_accessibility_permission_failure(error):
                    raise

                final_failure_reason = str(error)
                print(
                    "Post-typing observation attempt "
                    f"{attempt_number}/{MAX_OBSERVATION_ATTEMPTS} failed: "
                    f"{final_failure_reason}"
                )

                if attempt_number < MAX_OBSERVATION_ATTEMPTS:
                    time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)

                continue

            if attempt_number > 1:
                print(
                    "Post-typing observation recovered on attempt "
                    f"{attempt_number}."
                )

            return state

    raise RuntimeError(
        "Post-typing observation failed after "
        f"{MAX_OBSERVATION_ATTEMPTS} attempts; final failure: "
        f"{final_failure_reason}"
    )


def _require_target_focus(
    observation: HybridObservation,
) -> None:
    focused = [
        element
        for element in observation.accessibility_elements
        if element.focused is True
    ]

    if len(focused) != 1:
        raise RuntimeError(
            f"expected exactly one focused control, found {len(focused)}"
        )

    focused_control = focused[0]
    if (
        not _matches_text(focused_control, TARGET_INPUT_TEXT)
        or focused_control.identifier != TARGET_INPUT_IDENTIFIER
    ):
        raise RuntimeError(
            "the unique focused control was not the target input"
        )


def _validate_pre_canvas_state(
    state: ValidatedHybridState,
) -> None:
    if state.targets.target_input.value != TYPED_VALUE:
        raise RuntimeError(
            f"{TARGET_INPUT_TEXT!r} value was "
            f"{state.targets.target_input.value!r}, expected {TYPED_VALUE!r}"
        )

    if state.targets.target_input.enabled is not True:
        raise RuntimeError(f"{TARGET_INPUT_TEXT!r} was not enabled")

    if state.targets.native_button.source != "hybrid":
        raise RuntimeError(f"{NATIVE_BUTTON_TEXT!r} was not hybrid")

    if state.targets.canvas_action.source != "ocr":
        raise RuntimeError(f"{CANVAS_ACTION_TEXT!r} was not OCR-only")


def _observe_verified_completion(
    controller: ComputerController,
    reader: MacOSAccessibility,
    last_canvas_action: UIElement,
) -> ValidatedHybridState:
    final_failure_reason = "unknown validation failure"

    with TemporaryDirectory(prefix="experiment_12_completion_observation_") as temp_dir:
        temp_root = Path(temp_dir)

        for attempt_number in range(1, MAX_OBSERVATION_ATTEMPTS + 1):
            capture_path = temp_root / f"completion_{attempt_number}.png"

            try:
                observation = _capture_hybrid_observation(
                    controller,
                    reader,
                    capture_path,
                )
                state = _validate_completion_state(
                    observation,
                    last_canvas_action,
                )

            except RuntimeError as error:
                if _is_accessibility_permission_failure(error):
                    raise

                final_failure_reason = str(error)
                print(
                    "Completion observation attempt "
                    f"{attempt_number}/{MAX_OBSERVATION_ATTEMPTS} failed: "
                    f"{final_failure_reason}"
                )

                if attempt_number < MAX_OBSERVATION_ATTEMPTS:
                    time.sleep(OBSERVATION_RETRY_DELAY_SECONDS)

                continue

            if attempt_number > 1:
                print(
                    "Completion observation recovered on attempt "
                    f"{attempt_number}."
                )

            copied_observation = _copy_observation_image(
                state.observation,
                AFTER_PATH,
            )

            return ValidatedHybridState(
                observation=copied_observation,
                targets=state.targets,
                input_click_point=state.input_click_point,
                canvas_click_point=state.canvas_click_point,
            )

    raise RuntimeError(
        "Completion observation failed after "
        f"{MAX_OBSERVATION_ATTEMPTS} attempts; final failure: "
        f"{final_failure_reason}"
    )


def _validate_completion_state(
    observation: HybridObservation,
    last_canvas_action: UIElement,
) -> ValidatedHybridState:
    target_input = _require_one_by_text(
        observation.fused_elements,
        TARGET_INPUT_TEXT,
    )
    native_button = _require_one_by_text(
        observation.fused_elements,
        NATIVE_BUTTON_TEXT,
    )
    _assert_text_absent(
        observation.ocr_elements,
        ERROR_TEXT,
        "Completion full-screen OCR",
    )

    regional_search = _recognize_dynamic_region(
        observation,
        last_canvas_action,
    )
    _assert_text_absent(
        regional_search.elements,
        ERROR_TEXT,
        "Completion regional OCR",
    )
    regional_verification = _require_regional_target(
        regional_search,
        _normalized(VERIFICATION_TEXT),
        VERIFICATION_TEXT,
        minimum_confidence=COMPLETION_OCR_CONFIDENCE_FLOOR,
    )
    background_sample = _sample_canvas_background(
        observation,
        last_canvas_action,
    )

    _validate_target_input(
        target_input,
        TYPED_VALUE,
        observation.frame,
    )
    _validate_native_button(
        native_button,
        observation.frame,
    )

    full_matches = _find_ocr_by_text(
        observation.fused_elements,
        VERIFICATION_TEXT,
    )
    if len(full_matches) > 1:
        raise RuntimeError(
            f"{VERIFICATION_TEXT!r} expected at most one full-screen OCR "
            f"result, found {len(full_matches)}"
        )

    if not (
        len(full_matches) == 1
        and full_matches[0].confidence >= ACTION_CONFIDENCE
    ):
        regional_verification = _recover_regional_target(
            observation,
            last_canvas_action,
            _normalized(VERIFICATION_TEXT),
            VERIFICATION_TEXT,
            _full_screen_status(
                full_matches,
                VERIFICATION_TEXT,
            ),
            search=regional_search,
            minimum_confidence=COMPLETION_OCR_CONFIDENCE_FLOOR,
        ).target

    if regional_verification.source != "ocr":
        raise RuntimeError(
            f"{VERIFICATION_TEXT!r} source was "
            f"{regional_verification.source!r}, expected 'ocr'"
        )

    if regional_verification.confidence < COMPLETION_OCR_CONFIDENCE_FLOOR:
        raise RuntimeError(
            f"{VERIFICATION_TEXT!r} regional OCR confidence was "
            f"{regional_verification.confidence:.2f}, expected at least "
            f"{COMPLETION_OCR_CONFIDENCE_FLOOR:.2f}"
        )

    _require_positive_contained_safe_box(
        regional_verification,
        observation.frame.screen_size,
        VERIFICATION_TEXT,
    )

    if background_sample.classification != "completed":
        raise RuntimeError(
            "completion Canvas background did not classify as completed "
            f"green; median RGB was {background_sample.median_rgb}"
        )

    _print_composite_completion_result(
        regional_verification,
        background_sample,
    )

    targets = HybridTargets(
        target_input=target_input,
        native_button=native_button,
        canvas_action=regional_verification,
    )
    input_point, canvas_point = _validate_action_points(
        targets,
        observation.frame.screen_size,
    )

    return ValidatedHybridState(
        observation=observation,
        targets=targets,
        input_click_point=input_point,
        canvas_click_point=canvas_point,
    )


def _pixel_box(
    frame: ScreenFrame,
    element: UIElement,
) -> tuple[int, int, int, int]:
    box = element.bounding_box
    left = max(
        0,
        math.floor(box.left * frame.scale_x),
    )
    top = max(
        0,
        math.floor(box.top * frame.scale_y),
    )
    right = min(
        frame.pixel_width,
        math.ceil(box.right * frame.scale_x),
    )
    bottom = min(
        frame.pixel_height,
        math.ceil(box.bottom * frame.scale_y),
    )

    if left >= right or top >= bottom:
        raise RuntimeError("pixel annotation box was empty")

    return left, top, right, bottom


def _pixel_point(
    frame: ScreenFrame,
    point: tuple[int, int],
) -> tuple[int, int]:
    pixel_x = round(point[0] * frame.scale_x)
    pixel_y = round(point[1] * frame.scale_y)

    return (
        min(max(pixel_x, 0), frame.pixel_width - 1),
        min(max(pixel_y, 0), frame.pixel_height - 1),
    )


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    frame: ScreenFrame,
    element: UIElement,
    label: str,
    color: tuple[int, int, int],
    image_size: tuple[int, int],
) -> None:
    left, top, right, bottom = _pixel_box(
        frame,
        element,
    )

    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        outline=color,
        width=6,
    )

    text_box = draw.textbbox(
        (0, 0),
        label,
    )
    label_width = text_box[2] - text_box[0] + 14
    label_height = text_box[3] - text_box[1] + 10
    label_x = min(
        max(left, 0),
        max(0, image_size[0] - label_width),
    )
    label_y = top - label_height - 6

    if label_y < 0:
        label_y = min(
            bottom + 6,
            max(0, image_size[1] - label_height),
        )

    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height,
        ),
        fill=color,
    )
    draw.text(
        (label_x + 7, label_y + 5),
        label,
        fill=(255, 255, 255),
    )


def _draw_numbered_point(
    draw: ImageDraw.ImageDraw,
    frame: ScreenFrame,
    point: tuple[int, int],
    number: int,
) -> None:
    x, y = _pixel_point(
        frame,
        point,
    )
    radius = 22
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=POINT_COLOR,
        width=6,
    )
    draw.line(
        (x - radius - 12, y, x + radius + 12, y),
        fill=POINT_COLOR,
        width=4,
    )
    draw.line(
        (x, y - radius - 12, x, y + radius + 12),
        fill=POINT_COLOR,
        width=4,
    )
    draw.text(
        (x - 5, y - 7),
        str(number),
        fill=POINT_COLOR,
    )


def _save_plan_screenshot(
    state: ValidatedHybridState,
) -> None:
    plan = state.observation.image.copy()
    draw = ImageDraw.Draw(plan)
    frame = state.observation.frame

    _draw_labeled_box(
        draw,
        frame,
        state.targets.target_input,
        "1 ACCESSIBILITY INPUT",
        INPUT_COLOR,
        plan.size,
    )
    _draw_labeled_box(
        draw,
        frame,
        state.targets.native_button,
        "HYBRID NATIVE BUTTON",
        NATIVE_BUTTON_COLOR,
        plan.size,
    )
    _draw_labeled_box(
        draw,
        frame,
        state.targets.canvas_action,
        "2 OCR CANVAS ACTION",
        CANVAS_COLOR,
        plan.size,
    )
    _draw_numbered_point(
        draw,
        frame,
        state.input_click_point,
        1,
    )
    _draw_numbered_point(
        draw,
        frame,
        state.canvas_click_point,
        2,
    )

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.save(PLAN_PATH)

    with Image.open(PLAN_PATH) as saved:
        if saved.mode != "RGB":
            raise RuntimeError("plan screenshot mode was not RGB")

        if saved.size != state.observation.frame.pixel_size:
            raise RuntimeError(
                "plan screenshot size did not match the before screenshot"
            )


def _print_element(
    label: str,
    element: UIElement,
) -> None:
    print(f"{label}:")
    print(f"  text: {element.text!r}")
    print(f"  normalized_text: {normalize_ui_text(element.text)!r}")
    print(f"  element_type: {element.element_type!r}")
    print(f"  identifier: {element.identifier!r}")
    print(f"  value: {element.value!r}")
    print(f"  enabled: {element.enabled!r}")
    print(f"  focused: {element.focused!r}")
    print(f"  selected: {element.selected!r}")
    print(f"  source: {element.source!r}")
    print(f"  confidence: {element.confidence:.2f}")
    print(f"  bounding_box: {element.bounding_box}")


def _print_state_summary(
    state: ValidatedHybridState,
) -> None:
    observation = state.observation
    frame = observation.frame

    print("Phase 03 Experiment 12: Hybrid Accessibility and OCR Perception")
    print(f"Fixture path: {FIXTURE_PATH}")
    print(f"Before screenshot path: {BEFORE_PATH}")
    print(f"Plan screenshot path: {PLAN_PATH}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"Accessibility controls: {len(observation.accessibility_elements)}")
    print(f"OCR lines: {len(observation.ocr_elements)}")
    print(f"Fused elements: {len(observation.fused_elements)}")
    print("Fused target metadata:")
    _print_element(
        TARGET_INPUT_TEXT,
        state.targets.target_input,
    )
    _print_element(
        NATIVE_BUTTON_TEXT,
        state.targets.native_button,
    )
    _print_element(
        CANVAS_ACTION_TEXT,
        state.targets.canvas_action,
    )
    print(f"Planned input click point: {state.input_click_point}")
    print(f"Planned Canvas click point: {state.canvas_click_point}")


def _execute_action(
    executor: ToolExecutor,
    action: Action,
) -> Any:
    result = executor.execute(action)

    if not result.success:
        raise RuntimeError(f"Structured action failed: {result.error}")

    return result.output


def _get_mouse_position(
    executor: ToolExecutor,
) -> tuple[int, int]:
    output = _execute_action(
        executor,
        Action(
            tool_name="get_mouse_position",
            reason="Read the current cursor position.",
        ),
    )

    return int(output["x"]), int(output["y"])


def _move_mouse(
    executor: ToolExecutor,
    point: tuple[int, int],
    reason: str,
) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="move_mouse",
            arguments={
                "x": point[0],
                "y": point[1],
                "duration": MOVE_DURATION,
            },
            reason=reason,
        ),
    )


def _click_mouse(
    executor: ToolExecutor,
    point: tuple[int, int],
    reason: str,
) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="click_mouse",
            arguments={
                "x": point[0],
                "y": point[1],
            },
            reason=reason,
        ),
    )


def _type_text(
    executor: ToolExecutor,
) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="type_text",
            arguments={
                "text": TYPED_VALUE,
                "interval": TYPE_INTERVAL,
            },
            reason="Type the exact hybrid fixture input value.",
        ),
    )


def _run_dry_run(
    state: ValidatedHybridState,
) -> int:
    script_path = Path(__file__).relative_to(PROJECT_ROOT)

    print("Dry run: no mouse-control or keyboard-control Action was created or executed.")
    print(f"Dry-run planned input click point: {state.input_click_point}")
    print(f"Dry-run planned Canvas click point: {state.canvas_click_point}")
    print(
        "Run this command to execute hybrid perception: "
        f".venv/bin/python {script_path} --execute"
    )

    return 0


def _run_execute_mode(
    controller: ComputerController,
    reader: MacOSAccessibility,
    initial_state: ValidatedHybridState,
) -> int:
    executor = ToolExecutor(
        ToolRegistry(
            create_computer_tools(controller)
        )
    )

    try:
        original = _get_mouse_position(executor)
    except RuntimeError as error:
        print(f"Safety abort: failed to read initial mouse position: {error}")
        return 1

    if _point_in_fail_safe_corner(
        original,
        initial_state.observation.frame.screen_size,
    ):
        print("Safety abort: the cursor is at a PyAutoGUI fail-safe corner.")
        return 1

    print("Execute mode enabled. PyAutoGUI fail-safe remains active.")
    print(f"Original cursor position: {original}")
    _count_down_to_movement()

    mouse_movement_started = False
    succeeded = False

    try:
        pre_input_state = _observe_valid_hybrid_state(
            controller,
            reader,
            stage_name="Pre-input",
            expected_input_value="",
            reject_completion_text=True,
        )
        input_point = pre_input_state.input_click_point

        print(f"Fresh input logical bounding box: {pre_input_state.targets.target_input.bounding_box}")
        print(f"Fresh planned input click point: {input_point}")

        mouse_movement_started = True
        _move_mouse(
            executor,
            input_point,
            "Move the cursor to the Accessibility-grounded target input.",
        )

        reached = _get_mouse_position(executor)
        if not _within_tolerance(reached, input_point):
            raise RuntimeError(
                "reached input position was outside the allowed tolerance"
            )

        print(f"Reached input cursor position: {reached}")

        _click_mouse(
            executor,
            input_point,
            "Click the Accessibility-grounded target input.",
        )
        print(f"Clicked target input position: {input_point}")
        time.sleep(FOCUS_SETTLE_SECONDS)

        focused_state = _observe_valid_hybrid_state(
            controller,
            reader,
            stage_name="Post-input-click",
            expected_input_value="",
            reject_completion_text=True,
            require_focus=True,
        )
        print(
            "Verified target focus before typing: "
            f"{focused_state.targets.target_input.focused!r}"
        )

        _type_text(executor)
        print(f"Typed value: {TYPED_VALUE!r}")
        time.sleep(TYPE_SETTLE_SECONDS)

        pre_canvas_state = _observe_pre_canvas_state(
            controller,
            reader,
            focused_state.targets.canvas_action,
        )
        _validate_pre_canvas_state(pre_canvas_state)
        canvas_point = pre_canvas_state.canvas_click_point

        print(f"Fresh Canvas OCR bounding box: {pre_canvas_state.targets.canvas_action.bounding_box}")
        print(f"Fresh planned Canvas click point: {canvas_point}")

        _move_mouse(
            executor,
            canvas_point,
            "Move the cursor to the OCR-grounded Canvas action.",
        )
        reached = _get_mouse_position(executor)
        if not _within_tolerance(reached, canvas_point):
            raise RuntimeError(
                "reached Canvas position was outside the allowed tolerance"
            )

        print(f"Reached Canvas cursor position: {reached}")

        _click_mouse(
            executor,
            canvas_point,
            "Click the OCR-grounded Canvas action once.",
        )
        print(f"Clicked Canvas action position: {canvas_point}")
        time.sleep(CANVAS_SETTLE_SECONDS)

        completed_state = _observe_verified_completion(
            controller,
            reader,
            pre_canvas_state.targets.canvas_action,
        )
        print(f"After screenshot path: {AFTER_PATH}")
        print(
            "Verified completion OCR confidence: "
            f"{completed_state.targets.canvas_action.confidence:.2f}"
        )
        print(
            "Final input value: "
            f"{completed_state.targets.target_input.value!r}"
        )
        print("Perception source summary:")
        print("  input: accessibility")
        print("  native button: hybrid")
        print("  Canvas action/verification: ocr")
        succeeded = True

    except Exception as error:
        print(f"Safety abort: {type(error).__name__}: {error}")

    finally:
        if mouse_movement_started:
            try:
                _move_mouse(
                    executor,
                    original,
                    "Restore the cursor to its original position.",
                )
                restored = _get_mouse_position(executor)

                if _within_tolerance(restored, original):
                    print(f"Restored cursor position: {restored}")
                else:
                    print(
                        "Safety abort: restored cursor position was outside "
                        "the allowed tolerance."
                    )
                    succeeded = False

            except RuntimeError as error:
                print(f"Safety abort: failed to restore cursor: {error}")
                succeeded = False

    if not succeeded:
        return 1

    print(
        "Phase 03 Experiment 12 hybrid Accessibility and OCR perception "
        "completed successfully."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse Accessibility and OCR perception for Phase 03 Experiment "
            "12. The default mode is dry-run and performs no mouse-control "
            "or keyboard-control action."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Click the Accessibility-grounded input, type the fixture value, "
            "click the OCR-grounded Canvas action once, verify completion, "
            "and restore the cursor."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not FIXTURE_PATH.is_file():
        print(f"Safety abort: fixture file was not found: {FIXTURE_PATH}")
        return 1

    if not MacOSAccessibility.is_available():
        print("Safety abort: macOS Accessibility frameworks are unavailable.")
        return 1

    if not MacOSAccessibility.is_trusted():
        print("Safety abort: macOS Accessibility permission is not trusted.")
        return 1

    controller = ComputerController()
    reader = MacOSAccessibility()

    _count_down_to_capture()

    try:
        initial_state = _observe_valid_hybrid_state(
            controller,
            reader,
            stage_name="Initial",
            expected_input_value="",
            reject_completion_text=True,
            output_path=BEFORE_PATH,
        )
        _save_plan_screenshot(initial_state)
        _print_state_summary(initial_state)

        if not args.execute:
            return _run_dry_run(initial_state)

        return _run_execute_mode(
            controller,
            reader,
            initial_state,
        )

    except RuntimeError as error:
        print(f"Safety abort: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
