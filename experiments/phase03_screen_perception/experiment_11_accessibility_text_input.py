"""Phase 03 experiment for Accessibility-grounded text input."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import ComputerController
from computer_agent.core.models import Action
from computer_agent.perception import (
    MacOSAccessibility,
    ScreenCapture,
    ScreenFrame,
    UIElement,
)
from computer_agent.tools.computer import create_computer_tools
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase03_screen_perception"
    / "experiment_11_accessibility_text_input.html"
)
BEFORE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_11_accessibility_text_input_before.png"
)
PLAN_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_11_accessibility_text_input_plan.png"
)
AFTER_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_11_accessibility_text_input_after.png"
)

TARGET_TEXT = "TARGET_TEXT_FIELD_11"
TARGET_IDENTIFIER = "target-text-field"
TYPED_VALUE = "ACCESSIBILITY_TYPED_VALUE_11"
DECOY_TEXT = "DECOY_TEXT_FIELD_11"
DECOY_IDENTIFIER = "decoy-text-field"
DECOY_VALUE = "DECOY_VALUE_11"
DISABLED_TEXT = "DISABLED_TEXT_FIELD_11"
DISABLED_IDENTIFIER = "disabled-text-field"
DISABLED_VALUE = "LOCKED_VALUE_11"

CAPTURE_COUNTDOWN_SECONDS = 5
COUNTDOWN_SECONDS = 3
SAFE_EDGE_MARGIN = 10
POSITION_TOLERANCE = 1
MOVE_DURATION = 1.0
TYPE_INTERVAL = 0.05
FOCUS_SETTLE_SECONDS = 0.4
TYPE_SETTLE_SECONDS = 0.5
MAX_OBSERVATION_ATTEMPTS = 5
OBSERVATION_RETRY_DELAY_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class AccessibilityObservation:
    """A screenshot frame, optional image copy, and Accessibility controls."""

    frame: ScreenFrame
    controls: tuple[UIElement, ...]
    image: Image.Image | None = None


@dataclass(frozen=True, slots=True)
class ExpectedField:
    """Semantic metadata expected for one text-field control."""

    name: str
    text: str
    identifier: str
    value: str
    enabled: bool
    element_type: str = "text_field"


@dataclass(frozen=True, slots=True)
class MatchedFields:
    """The three fixture fields after exact semantic matching."""

    target: UIElement
    decoy: UIElement
    disabled: UIElement


@dataclass(frozen=True, slots=True)
class ValidatedFixtureState:
    """A fresh Accessibility observation that passed all fixture checks."""

    observation: AccessibilityObservation
    fields: MatchedFields
    click_point: tuple[int, int]


TARGET_INITIAL_FIELD = ExpectedField(
    name="target",
    text=TARGET_TEXT,
    identifier=TARGET_IDENTIFIER,
    value="",
    enabled=True,
)
DECOY_FIELD = ExpectedField(
    name="decoy",
    text=DECOY_TEXT,
    identifier=DECOY_IDENTIFIER,
    value=DECOY_VALUE,
    enabled=True,
)
DISABLED_FIELD = ExpectedField(
    name="disabled",
    text=DISABLED_TEXT,
    identifier=DISABLED_IDENTIFIER,
    value=DISABLED_VALUE,
    enabled=False,
)


def _expected_fields(target_value: str) -> tuple[ExpectedField, ...]:
    return (
        ExpectedField(
            name=TARGET_INITIAL_FIELD.name,
            text=TARGET_INITIAL_FIELD.text,
            identifier=TARGET_INITIAL_FIELD.identifier,
            value=target_value,
            enabled=TARGET_INITIAL_FIELD.enabled,
        ),
        DECOY_FIELD,
        DISABLED_FIELD,
    )


def _count_down_to_capture() -> None:
    print(
        "Switch to the already-open and freshly reloaded Experiment 11 "
        "Chrome fixture before the initial screenshot is captured."
    )

    for remaining in range(CAPTURE_COUNTDOWN_SECONDS, 0, -1):
        print(f"Capturing in {remaining}...")
        time.sleep(1)


def _count_down_to_movement() -> None:
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Moving in {remaining}...")
        time.sleep(1)


def _target_point(element: UIElement) -> tuple[int, int]:
    center_x, center_y = element.bounding_box.center

    return math.floor(center_x), math.floor(center_y)


def _box_inside_screen(
    element: UIElement,
    screen_size: tuple[int, int],
) -> bool:
    box = element.bounding_box
    width, height = screen_size

    return 0 <= box.left < box.right <= width and 0 <= box.top < box.bottom <= height


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


def _capture_before_observation(
    controller: ComputerController,
) -> AccessibilityObservation:
    BEFORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = ScreenCapture(controller).capture(BEFORE_PATH)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    if image.size != frame.pixel_size:
        raise RuntimeError(
            "before screenshot size did not match its screen frame metadata"
        )

    return AccessibilityObservation(
        frame=frame,
        image=image,
        controls=(),
    )


def _read_accessibility_observation(
    reader: MacOSAccessibility,
    frame: ScreenFrame,
    image: Image.Image | None = None,
) -> AccessibilityObservation:
    return AccessibilityObservation(
        frame=frame,
        image=image,
        controls=tuple(reader.read_frontmost_controls()),
    )


def _capture_after_screenshot(
    controller: ComputerController,
) -> ScreenFrame:
    AFTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = ScreenCapture(controller).capture(AFTER_PATH)

    with Image.open(frame.image_path) as source:
        image_size = source.size

    if image_size != frame.pixel_size:
        raise RuntimeError(
            "after screenshot size did not match its screen frame metadata"
        )

    return frame


def _find_exact_control(
    controls: tuple[UIElement, ...],
    expected: ExpectedField,
) -> UIElement:
    matches = [
        control
        for control in controls
        if control.text == expected.text
    ]

    if not matches:
        raise RuntimeError(
            f"expected fixture field {expected.text!r} was not found"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"expected fixture field {expected.text!r} was duplicated"
        )

    return matches[0]


def _reject_unexpected_identifier_matches(
    controls: tuple[UIElement, ...],
    expected_fields: tuple[ExpectedField, ...],
) -> None:
    expected_by_identifier = {
        expected.identifier: expected
        for expected in expected_fields
    }

    for control in controls:
        expected = expected_by_identifier.get(control.identifier)
        if expected is None:
            continue

        if control.text != expected.text:
            raise RuntimeError(
                f"identifier {control.identifier!r} belonged to "
                f"unexpected control text {control.text!r}"
            )


def _validate_control(
    control: UIElement,
    expected: ExpectedField,
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

    if control.value != expected.value:
        raise RuntimeError(
            f"{expected.text!r} value was {control.value!r}, "
            f"expected {expected.value!r}"
        )

    if control.enabled is not expected.enabled:
        raise RuntimeError(
            f"{expected.text!r} enabled was {control.enabled!r}, "
            f"expected {expected.enabled!r}"
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

    box = control.bounding_box
    if box.width <= 0 or box.height <= 0:
        raise RuntimeError(
            f"{expected.text!r} bounding box had zero size"
        )

    if not _box_inside_screen(control, frame.screen_size):
        raise RuntimeError(
            f"{expected.text!r} bounding box was outside the logical screen"
        )


def _match_and_validate_fields(
    observation: AccessibilityObservation,
    target_value: str,
) -> MatchedFields:
    expected_fields = _expected_fields(target_value)
    _reject_unexpected_identifier_matches(
        observation.controls,
        expected_fields,
    )

    matched = {
        expected.name: _find_exact_control(observation.controls, expected)
        for expected in expected_fields
    }

    for expected in expected_fields:
        _validate_control(
            matched[expected.name],
            expected,
            observation.frame,
        )

    return MatchedFields(
        target=matched["target"],
        decoy=matched["decoy"],
        disabled=matched["disabled"],
    )


def _is_accessibility_permission_failure(error: RuntimeError) -> bool:
    message = str(error)

    return (
        "macOS Accessibility frameworks are unavailable" in message
        or "macOS Accessibility permission is not trusted" in message
    )


def _validate_target_plan(
    fields: MatchedFields,
    screen_size: tuple[int, int],
) -> tuple[int, int]:
    point = _target_point(fields.target)
    target_box = fields.target.bounding_box

    if not target_box.contains_point(*point):
        raise RuntimeError("planned click point was outside the target field")

    if not _point_inside_screen(point, screen_size):
        raise RuntimeError("planned click point was outside the logical screen")

    if not _point_in_safe_area(point, screen_size):
        raise RuntimeError(
            "planned click point was too close to a logical screen edge"
        )

    if fields.decoy.bounding_box.contains_point(*point):
        raise RuntimeError("planned click point overlapped the decoy field")

    if fields.disabled.bounding_box.contains_point(*point):
        raise RuntimeError("planned click point overlapped the disabled field")

    return point


def _pixel_box(
    frame: ScreenFrame,
    element: UIElement,
) -> tuple[int, int, int, int]:
    box = element.bounding_box
    left = max(0, math.floor(box.left * frame.scale_x))
    top = max(0, math.floor(box.top * frame.scale_y))
    right = min(frame.pixel_width, math.ceil(box.right * frame.scale_x))
    bottom = min(frame.pixel_height, math.ceil(box.bottom * frame.scale_y))

    if left >= right or top >= bottom:
        raise RuntimeError("target pixel annotation box was empty")

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


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
) -> None:
    x, y = point
    radius = 18
    draw.line((x - radius, y, x + radius, y), fill=(255, 0, 0), width=5)
    draw.line((x, y - radius, x, y + radius), fill=(255, 0, 0), width=5)
    draw.ellipse(
        (x - 8, y - 8, x + 8, y + 8),
        outline=(0, 96, 255),
        width=4,
    )


def _draw_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    left: int,
    top: int,
    bottom: int,
    image_size: tuple[int, int],
) -> None:
    text_box = draw.textbbox((0, 0), label)
    label_width = text_box[2] - text_box[0] + 14
    label_height = text_box[3] - text_box[1] + 10
    label_x = min(max(left, 0), max(0, image_size[0] - label_width))
    label_y = top - label_height - 6

    if label_y < 0:
        label_y = min(bottom + 6, max(0, image_size[1] - label_height))

    draw.rectangle(
        (
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height,
        ),
        fill=(255, 0, 0),
    )
    draw.text(
        (label_x + 7, label_y + 5),
        label,
        fill=(255, 255, 255),
    )


def _save_plan_screenshot(
    observation: AccessibilityObservation,
    target: UIElement,
    point: tuple[int, int],
) -> None:
    if observation.image is None:
        raise RuntimeError("before screenshot image was unavailable")

    plan = observation.image.copy()
    draw = ImageDraw.Draw(plan)
    left, top, right, bottom = _pixel_box(observation.frame, target)
    pixel_point = _pixel_point(observation.frame, point)

    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        outline=(255, 0, 0),
        width=6,
    )
    _draw_crosshair(draw, pixel_point)
    _draw_label(draw, TARGET_TEXT, left, top, bottom, plan.size)

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.save(PLAN_PATH)

    with Image.open(PLAN_PATH) as saved:
        if saved.mode != "RGB":
            raise RuntimeError("plan screenshot mode was not RGB")

        if saved.size != observation.frame.pixel_size:
            raise RuntimeError(
                "plan screenshot size did not match the before screenshot"
            )


def _focused_controls(
    controls: tuple[UIElement, ...],
) -> list[UIElement]:
    return [
        control
        for control in controls
        if control.focused is True
    ]


def _require_target_focus(
    observation: AccessibilityObservation,
    fields: MatchedFields,
) -> None:
    focused = _focused_controls(observation.controls)

    if len(focused) != 1:
        raise RuntimeError(
            f"expected exactly one focused control, found {len(focused)}"
        )

    focused_control = focused[0]
    if (
        focused_control.text != TARGET_TEXT
        or focused_control.identifier != TARGET_IDENTIFIER
    ):
        raise RuntimeError(
            "the unique focused control was not the target text field"
        )

    if fields.target.focused is not True:
        raise RuntimeError("target text field was not focused")


def _observe_valid_fixture_state(
    reader: MacOSAccessibility,
    frame: ScreenFrame,
    *,
    stage_name: str,
    target_value: str,
    require_target_focus: bool = False,
    image: Image.Image | None = None,
) -> ValidatedFixtureState:
    final_failure_reason = "unknown validation failure"

    for attempt_number in range(1, MAX_OBSERVATION_ATTEMPTS + 1):
        try:
            observation = _read_accessibility_observation(
                reader,
                frame,
                image=image,
            )
            fields = _match_and_validate_fields(
                observation,
                target_value=target_value,
            )
            click_point = _validate_target_plan(
                fields,
                frame.screen_size,
            )

            if require_target_focus:
                _require_target_focus(observation, fields)

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

        return ValidatedFixtureState(
            observation=observation,
            fields=fields,
            click_point=click_point,
        )

    raise RuntimeError(
        f"{stage_name} observation failed after "
        f"{MAX_OBSERVATION_ATTEMPTS} attempts; final failure: "
        f"{final_failure_reason}"
    )


def _execute_action(executor: ToolExecutor, action: Action) -> Any:
    result = executor.execute(action)

    if not result.success:
        raise RuntimeError(f"Structured action failed: {result.error}")

    return result.output


def _get_mouse_position(executor: ToolExecutor) -> tuple[int, int]:
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
) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="click_mouse",
            arguments={
                "x": point[0],
                "y": point[1],
            },
            reason="Click the Accessibility-grounded target text field.",
        ),
    )


def _type_text(executor: ToolExecutor) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="type_text",
            arguments={
                "text": TYPED_VALUE,
                "interval": TYPE_INTERVAL,
            },
            reason="Type the verified fixture value into the focused field.",
        ),
    )


def _print_initial_summary(
    observation: AccessibilityObservation,
    fields: MatchedFields,
    click_point: tuple[int, int],
) -> None:
    frame = observation.frame

    print("Phase 03 Experiment 11: Accessibility-Grounded Text Input")
    print(f"Fixture path: {FIXTURE_PATH}")
    print(f"Before screenshot path: {BEFORE_PATH}")
    print(f"Plan screenshot path: {PLAN_PATH}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"Total frontmost-window controls: {len(observation.controls)}")
    print("Target semantic metadata:")
    _print_control(fields.target)
    print(f"Target logical bounding box: {fields.target.bounding_box}")
    print(f"Planned logical click point: {click_point}")


def _print_control(control: UIElement) -> None:
    print(f"- text: {control.text!r}")
    print(f"  element_type: {control.element_type!r}")
    print(f"  identifier: {control.identifier!r}")
    print(f"  value: {control.value!r}")
    print(f"  enabled: {control.enabled!r}")
    print(f"  focused: {control.focused!r}")
    print(f"  source: {control.source!r}")
    print(f"  confidence: {control.confidence:.1f}")
    print(f"  bounding_box: {control.bounding_box}")


def _run_dry_run(
    fields: MatchedFields,
    click_point: tuple[int, int],
) -> int:
    script_path = Path(__file__).relative_to(PROJECT_ROOT)

    print("Dry-run target semantic metadata:")
    _print_control(fields.target)
    print(f"Dry-run target logical bounding box: {fields.target.bounding_box}")
    print(f"Dry-run planned logical click point: {click_point}")
    print("Dry run: no mouse-control or keyboard-control Action was created or executed.")
    print(
        "Run this command to execute Accessibility-grounded text input: "
        f".venv/bin/python {script_path} --execute"
    )

    return 0


def _verify_ready_to_type(
    reader: MacOSAccessibility,
    frame: ScreenFrame,
) -> None:
    _observe_valid_fixture_state(
        reader,
        frame,
        stage_name="Post-click focus",
        target_value="",
        require_target_focus=True,
    )


def _verify_completed(
    reader: MacOSAccessibility,
    frame: ScreenFrame,
) -> MatchedFields:
    final_state = _observe_valid_fixture_state(
        reader,
        frame,
        stage_name="Post-typing value",
        target_value=TYPED_VALUE,
        require_target_focus=True,
    )

    return final_state.fields


def _run_execute_mode(
    controller: ComputerController,
    reader: MacOSAccessibility,
    frame: ScreenFrame,
) -> int:
    executor = ToolExecutor(ToolRegistry(create_computer_tools(controller)))

    try:
        original = _get_mouse_position(executor)
    except RuntimeError as error:
        print(f"Safety abort: failed to read initial mouse position: {error}")
        return 1

    if _point_in_fail_safe_corner(original, frame.screen_size):
        print("Safety abort: the cursor is at a PyAutoGUI fail-safe corner.")
        return 1

    print("Execute mode enabled. PyAutoGUI fail-safe remains active.")
    print(f"Original cursor position: {original}")
    _count_down_to_movement()

    mouse_movement_started = False
    succeeded = False

    try:
        fresh_state = _observe_valid_fixture_state(
            reader,
            frame,
            stage_name="Pre-movement",
            target_value="",
        )
        fresh_fields = fresh_state.fields
        click_point = fresh_state.click_point

        print(f"Fresh target logical bounding box: {fresh_fields.target.bounding_box}")
        print(f"Fresh planned logical click point: {click_point}")

        mouse_movement_started = True
        _move_mouse(
            executor,
            click_point,
            "Move the cursor to the Accessibility-grounded target field.",
        )

        reached = _get_mouse_position(executor)
        if not _within_tolerance(reached, click_point):
            raise RuntimeError(
                "reached cursor position was outside the allowed tolerance"
            )

        print(f"Reached cursor position: {reached}")

        _click_mouse(executor, click_point)
        print(f"Clicked target field position: {click_point}")
        time.sleep(FOCUS_SETTLE_SECONDS)

        _verify_ready_to_type(reader, frame)
        print("Verified target focus before typing.")

        _type_text(executor)
        print(f"Typed value: {TYPED_VALUE!r}")
        time.sleep(TYPE_SETTLE_SECONDS)

        after_frame = _capture_after_screenshot(controller)
        final_fields = _verify_completed(reader, after_frame)

        print(f"After screenshot path: {AFTER_PATH}")
        print(f"Final target value: {final_fields.target.value!r}")
        print(f"Final decoy value: {final_fields.decoy.value!r}")
        print(f"Final disabled value: {final_fields.disabled.value!r}")
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
        "Phase 03 Experiment 11 Accessibility-grounded text input "
        "completed successfully."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Click and type into a semantically located Phase 03 Experiment "
            "11 text field. The default mode is dry-run and performs no "
            "mouse-control or keyboard-control action."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Click the validated target field, verify focus, type the fixture "
            "value, verify values, and restore the cursor."
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
        captured = _capture_before_observation(controller)
        initial_state = _observe_valid_fixture_state(
            reader,
            captured.frame,
            stage_name="Initial",
            target_value="",
            image=captured.image,
        )
        _save_plan_screenshot(
            initial_state.observation,
            initial_state.fields.target,
            initial_state.click_point,
        )
        _print_initial_summary(
            initial_state.observation,
            initial_state.fields,
            initial_state.click_point,
        )

        if not args.execute:
            return _run_dry_run(
                initial_state.fields,
                initial_state.click_point,
            )

        return _run_execute_mode(
            controller,
            reader,
            initial_state.observation.frame,
        )

    except RuntimeError as error:
        print(f"Safety abort: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
