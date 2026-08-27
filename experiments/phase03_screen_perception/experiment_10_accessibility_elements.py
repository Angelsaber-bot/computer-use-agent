"""Phase 03 experiment for read-only Accessibility element detection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import ComputerController
from computer_agent.perception import (
    MacOSAccessibility,
    ScreenCapture,
    ScreenFrame,
    UIElement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase03_screen_perception"
    / "experiment_10_accessibility_elements.html"
)
BEFORE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_10_accessibility_elements_before.png"
)
ANNOTATED_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_10_accessibility_elements_annotated.png"
)
CAPTURE_COUNTDOWN_SECONDS = 5

ELEMENT_COLORS = {
    "text_field": (0, 96, 255),
    "button": (220, 0, 0),
    "checkbox": (128, 0, 180),
    "popup_button": (230, 120, 0),
    "radio_button": (0, 150, 70),
}


@dataclass(frozen=True, slots=True)
class ExpectedControl:
    """Semantic metadata expected for one fixture control."""

    text: str
    element_type: str
    identifier: str
    enabled: bool
    value: str | int | float | bool | None = None
    validate_value: bool = False
    require_selected_none: bool = False


EXPECTED_CONTROLS = (
    ExpectedControl(
        text="EMPTY_TEXT_FIELD_10",
        element_type="text_field",
        identifier="empty-text-field",
        value="",
        enabled=True,
        validate_value=True,
    ),
    ExpectedControl(
        text="PLACEHOLDER_TEXT_FIELD_10",
        element_type="text_field",
        identifier="placeholder-text-field",
        value="",
        enabled=True,
        validate_value=True,
    ),
    ExpectedControl(
        text="ACTIVE_BUTTON_10",
        element_type="button",
        identifier="active-button",
        enabled=True,
    ),
    ExpectedControl(
        text="DISABLED_BUTTON_10",
        element_type="button",
        identifier="disabled-button",
        enabled=False,
    ),
    ExpectedControl(
        text="CHECKBOX_OPTION_10",
        element_type="checkbox",
        identifier="checkbox-option",
        enabled=True,
        require_selected_none=True,
    ),
    ExpectedControl(
        text="MODE_SELECTOR_10",
        element_type="popup_button",
        identifier="mode-selector",
        value="MODE_ALPHA_10",
        enabled=True,
        validate_value=True,
    ),
    ExpectedControl(
        text="RADIO_ALPHA_10",
        element_type="radio_button",
        identifier="radio-alpha",
        enabled=True,
        require_selected_none=True,
    ),
    ExpectedControl(
        text="RADIO_BETA_10",
        element_type="radio_button",
        identifier="radio-beta",
        enabled=True,
        require_selected_none=True,
    ),
)


def _count_down_to_capture() -> None:
    print(
        "Switch to the already-open Experiment 10 Chrome fixture before "
        "the screenshot is captured."
    )

    for remaining in range(CAPTURE_COUNTDOWN_SECONDS, 0, -1):
        print(f"Capturing in {remaining}...")
        time.sleep(1)


def _box_inside_screen(
    element: UIElement,
    screen_size: tuple[int, int],
) -> bool:
    box = element.bounding_box
    width, height = screen_size

    return 0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height


def _find_exact_control(
    controls: list[UIElement],
    expected: ExpectedControl,
) -> UIElement:
    matches = [
        control
        for control in controls
        if control.text == expected.text
    ]

    if not matches:
        raise RuntimeError(
            f"expected fixture control {expected.text!r} was not found"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"expected fixture control {expected.text!r} was duplicated"
        )

    return matches[0]


def _validate_control(
    control: UIElement,
    expected: ExpectedControl,
    frame: ScreenFrame,
) -> None:
    if control.element_type != expected.element_type:
        raise RuntimeError(
            f"{expected.text!r} element_type was {control.element_type!r}, "
            f"expected {expected.element_type!r}"
        )

    if control.identifier != expected.identifier:
        raise RuntimeError(
            f"{expected.text!r} identifier was {control.identifier!r}, "
            f"expected {expected.identifier!r}"
        )

    if control.enabled is not expected.enabled:
        raise RuntimeError(
            f"{expected.text!r} enabled was {control.enabled!r}, "
            f"expected {expected.enabled!r}"
        )

    if expected.validate_value and control.value != expected.value:
        raise RuntimeError(
            f"{expected.text!r} value was {control.value!r}, "
            f"expected {expected.value!r}"
        )

    if control.source != "accessibility":
        raise RuntimeError(
            f"{expected.text!r} source was {control.source!r}, "
            "expected 'accessibility'"
        )

    if control.confidence != 1.0:
        raise RuntimeError(
            f"{expected.text!r} confidence was {control.confidence!r}, "
            "expected 1.0"
        )

    if not _box_inside_screen(control, frame.screen_size):
        raise RuntimeError(
            f"{expected.text!r} bounding box was outside the logical screen"
        )

    # Chrome does not reliably expose native checked state through AXValue,
    # AXSelected, or related attributes for checkbox and radio controls.
    if expected.require_selected_none and control.selected is not None:
        raise RuntimeError(
            f"{expected.text!r} selected was {control.selected!r}, "
            "expected None"
        )


def _match_fixture_controls(
    controls: list[UIElement],
    frame: ScreenFrame,
) -> list[UIElement]:
    matched = []

    for expected in EXPECTED_CONTROLS:
        control = _find_exact_control(controls, expected)
        _validate_control(control, expected, frame)
        matched.append(control)

    return matched


def _pixel_box(
    frame: ScreenFrame,
    element: UIElement,
) -> tuple[int, int, int, int]:
    box = element.bounding_box
    left = math.floor(box.left * frame.scale_x)
    top = math.floor(box.top * frame.scale_y)
    right = math.ceil(box.right * frame.scale_x)
    bottom = math.ceil(box.bottom * frame.scale_y)

    return left, top, right, bottom


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    frame: ScreenFrame,
    element: UIElement,
    image_size: tuple[int, int],
) -> None:
    left, top, right, bottom = _pixel_box(frame, element)
    color = ELEMENT_COLORS[element.element_type]

    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        outline=color,
        width=6,
    )

    label = f"{element.element_type}: {element.text}"
    text_box = draw.textbbox((0, 0), label)
    label_width = text_box[2] - text_box[0] + 12
    label_height = text_box[3] - text_box[1] + 8
    label_x = min(
        max(left, 0),
        max(0, image_size[0] - label_width),
    )
    label_y = top - label_height - 4

    if label_y < 0:
        label_y = min(bottom + 4, max(0, image_size[1] - label_height))

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
        (label_x + 6, label_y + 4),
        label,
        fill=(255, 255, 255),
    )


def _save_annotated_screenshot(
    frame: ScreenFrame,
    controls: list[UIElement],
) -> None:
    with Image.open(frame.image_path) as source:
        annotated = source.convert("RGB")

    draw = ImageDraw.Draw(annotated)

    for control in controls:
        _draw_labeled_box(
            draw,
            frame,
            control,
            annotated.size,
        )

    ANNOTATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(ANNOTATED_PATH)

    with Image.open(ANNOTATED_PATH) as saved:
        assert saved.mode == "RGB"
        assert saved.size == frame.pixel_size


def _print_control(control: UIElement) -> None:
    print(f"- text: {control.text!r}")
    print(f"  element_type: {control.element_type!r}")
    print(f"  identifier: {control.identifier!r}")
    print(f"  value: {control.value!r}")
    print(f"  enabled: {control.enabled!r}")
    print(f"  focused: {control.focused!r}")
    print(f"  selected: {control.selected!r}")
    print(f"  source: {control.source!r}")
    print(f"  confidence: {control.confidence:.1f}")
    print(f"  bounding_box: {control.bounding_box}")


def main() -> int:
    print("Phase 03 Experiment 10: Accessibility Elements")

    if not MacOSAccessibility.is_available():
        print("Safety abort: macOS Accessibility frameworks are unavailable.")
        return 1

    if not MacOSAccessibility.is_trusted():
        print("Safety abort: macOS Accessibility permission is not trusted.")
        return 1

    if not FIXTURE_PATH.is_file():
        print(f"Safety abort: fixture file was not found: {FIXTURE_PATH}")
        return 1

    BEFORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    controller = ComputerController()
    reader = MacOSAccessibility()

    _count_down_to_capture()

    frame = ScreenCapture(controller).capture(BEFORE_PATH)
    controls = reader.read_frontmost_controls()

    try:
        matched_controls = _match_fixture_controls(
            controls,
            frame,
        )
    except RuntimeError as error:
        print(f"Safety abort: {error}")
        return 1

    _save_annotated_screenshot(
        frame,
        matched_controls,
    )

    print(f"Fixture path: {FIXTURE_PATH}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"Total frontmost-window controls: {len(controls)}")
    print(f"Matched fixture-control count: {len(matched_controls)}")
    print("Matched fixture controls:")

    for control in matched_controls:
        _print_control(control)

    print(f"Annotated screenshot path: {ANNOTATED_PATH}")
    print(
        "Phase 03 Experiment 10 Accessibility element detection "
        "completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
