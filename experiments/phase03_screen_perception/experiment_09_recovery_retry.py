"""Phase 03 experiment for recovery retry on a localized text target."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
import time

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import ComputerController
from computer_agent.core.models import Action
from computer_agent.perception import (
    ScreenCapture,
    ScreenCoordinateMapper,
    ScreenFrame,
    TesseractOCR,
    TextTargetLocator,
    UIElement,
)
from computer_agent.tools.computer import create_computer_tools
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase03_screen_perception"
    / "experiment_09_recovery_retry.html"
)
BEFORE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_before.png"
)
ATTEMPT_PLAN_PATHS = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_1_plan.png",
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_2_plan.png",
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_3_plan.png",
)
ATTEMPT_AFTER_PATHS = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_1_after.png",
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_2_after.png",
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_09_recovery_retry_attempt_3_after.png",
)
TARGET_TEXT = "RECOVERY_TARGET_09"
OCR_CANDIDATE_CONFIDENCE = 0.05
ACTION_CONFIDENCE = 0.70
MAX_ATTEMPTS = 3
MIN_RELOCATION_DISTANCE = 100
SAFE_EDGE_MARGIN = 10
POSITION_TOLERANCE = 1
CAPTURE_COUNTDOWN_SECONDS = 5
COUNTDOWN_SECONDS = 3
CLICK_SETTLE_SECONDS = 1.0
MOVE_DURATION = 1.0
INCOMPLETE_TARGET_COLOR = (255, 244, 184)
COMPLETED_TARGET_COLOR = (200, 230, 201)
COLOR_TOLERANCE = 35
COLOR_SAMPLE_PADDING_X = 36
COLOR_SAMPLE_PADDING_Y = 36
DARK_PIXEL_THRESHOLD = 80
MIN_COLOR_SAMPLE_PIXELS = 100


@dataclass(frozen=True, slots=True)
class ScreenObservation:
    """A screenshot, its frame metadata, and mapped OCR elements."""

    frame: ScreenFrame
    image: Image.Image
    elements: tuple[UIElement, ...]


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """A localized target selected from one fresh screen observation."""

    source: UIElement
    target: UIElement
    match_type: str
    click_point: tuple[int, int]


@dataclass(frozen=True, slots=True)
class TargetBackground:
    """The classified target background color."""

    state: str
    median_rgb: tuple[int, int, int]


def _attempt_plan_path(attempt_number: int) -> Path:
    return ATTEMPT_PLAN_PATHS[attempt_number - 1]


def _attempt_after_path(attempt_number: int) -> Path:
    return ATTEMPT_AFTER_PATHS[attempt_number - 1]


def _draw_box(
    draw: ImageDraw.ImageDraw,
    element: UIElement,
    color: tuple[int, int, int],
    width: int,
) -> None:
    box = element.bounding_box
    draw.rectangle(
        (box.left, box.top, box.right - 1, box.bottom - 1),
        outline=color,
        width=width,
    )


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
) -> None:
    x, y = point
    radius = 12
    draw.line((x - radius, y, x + radius, y), fill=(255, 0, 0), width=3)
    draw.line((x, y - radius, x, y + radius), fill=(255, 0, 0), width=3)


def _count_down_to_capture() -> None:
    print(
        "Switch to the already-open and freshly reloaded Chrome fixture "
        "before the initial screenshot is captured."
    )

    for remaining in range(CAPTURE_COUNTDOWN_SECONDS, 0, -1):
        print(f"Capturing in {remaining}...")
        time.sleep(1)


def _count_down_to_movement() -> None:
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Moving in {remaining}...")
        time.sleep(1)


def _target_point(element: UIElement) -> tuple[int, int]:
    box = element.bounding_box
    x = min(max(round(element.center[0]), box.left), box.right - 1)
    y = min(max(round(element.center[1]), box.top), box.bottom - 1)

    return x, y


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
        SAFE_EDGE_MARGIN <= x < width - SAFE_EDGE_MARGIN
        and SAFE_EDGE_MARGIN <= y < height - SAFE_EDGE_MARGIN
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


def _distance(
    first: tuple[int, int],
    second: tuple[int, int],
) -> float:
    return math.hypot(
        second[0] - first[0],
        second[1] - first[1],
    )


def _select_target(
    elements: tuple[UIElement, ...],
) -> tuple[UIElement | None, UIElement | None, str | None]:
    source = TextTargetLocator.find_best(
        elements,
        TARGET_TEXT,
        minimum_confidence=ACTION_CONFIDENCE,
    )
    match_type = "exact"

    if source is None:
        source = TextTargetLocator.find_best(
            elements,
            TARGET_TEXT,
            partial_match=True,
            minimum_confidence=ACTION_CONFIDENCE,
        )
        match_type = "partial"

    if source is None:
        return None, None, None

    return source, TextTargetLocator.extract_target(source, TARGET_TEXT), match_type


def _validate_target(
    target: UIElement,
    click_point: tuple[int, int],
    screen_size: tuple[int, int],
) -> None:
    if not _box_inside_screen(target, screen_size):
        raise RuntimeError("extracted target box is outside the screen")

    if not _point_inside_screen(click_point, screen_size):
        raise RuntimeError("planned click point is outside the screen")

    if not _point_in_safe_area(click_point, screen_size):
        raise RuntimeError("planned click point is too close to a screen edge")


def _save_plan(
    image: Image.Image,
    elements: tuple[UIElement, ...],
    target: UIElement,
    point: tuple[int, int],
    logical_size: tuple[int, int],
    output_path: Path,
) -> None:
    plan = image.resize(logical_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(plan)

    for element in elements:
        _draw_box(draw, element, (0, 180, 0), 1)

    _draw_box(draw, target, (255, 0, 0), 4)
    _draw_crosshair(draw, point)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.save(output_path)

    with Image.open(output_path) as saved:
        assert saved.mode == "RGB"
        assert saved.size == logical_size


def _capture_observation(
    controller: ComputerController,
    output_path: Path,
) -> ScreenObservation:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = ScreenCapture(controller).capture(output_path)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    assert image.size == frame.pixel_size

    ocr = TesseractOCR(minimum_confidence=OCR_CANDIDATE_CONFIDENCE)
    mapper = ScreenCoordinateMapper(frame)
    elements = tuple(
        mapper.pixel_element_to_logical(element)
        for element in ocr.recognize(image)
    )

    return ScreenObservation(
        frame=frame,
        image=image,
        elements=elements,
    )


def _pixel_sample_bounds(
    frame: ScreenFrame,
    target: UIElement,
) -> tuple[int, int, int, int]:
    box = target.bounding_box
    logical_left = max(0, box.left - COLOR_SAMPLE_PADDING_X)
    logical_top = max(0, box.top - COLOR_SAMPLE_PADDING_Y)
    logical_right = min(frame.screen_width, box.right + COLOR_SAMPLE_PADDING_X)
    logical_bottom = min(frame.screen_height, box.bottom + COLOR_SAMPLE_PADDING_Y)

    pixel_left = max(0, math.floor(logical_left * frame.scale_x))
    pixel_top = max(0, math.floor(logical_top * frame.scale_y))
    pixel_right = min(frame.pixel_width, math.ceil(logical_right * frame.scale_x))
    pixel_bottom = min(frame.pixel_height, math.ceil(logical_bottom * frame.scale_y))

    if pixel_left >= pixel_right or pixel_top >= pixel_bottom:
        raise RuntimeError("target color sample region was empty")

    return pixel_left, pixel_top, pixel_right, pixel_bottom


def _median_background_color(
    observation: ScreenObservation,
    target: UIElement,
) -> tuple[int, int, int]:
    bounds = _pixel_sample_bounds(observation.frame, target)
    crop = observation.image.crop(bounds)
    samples = [
        pixel[:3]
        for pixel in crop.get_flattened_data()
        if max(pixel[:3]) >= DARK_PIXEL_THRESHOLD
    ]

    if len(samples) < MIN_COLOR_SAMPLE_PIXELS:
        raise RuntimeError(
            "not enough non-dark pixels were available for target color "
            "classification"
        )

    return (
        round(median(pixel[0] for pixel in samples)),
        round(median(pixel[1] for pixel in samples)),
        round(median(pixel[2] for pixel in samples)),
    )


def _color_delta(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> int:
    return max(
        abs(first[0] - second[0]),
        abs(first[1] - second[1]),
        abs(first[2] - second[2]),
    )


def _classify_target_background(
    observation: ScreenObservation,
    target: UIElement,
) -> TargetBackground:
    sample_color = _median_background_color(observation, target)
    incomplete_delta = _color_delta(sample_color, INCOMPLETE_TARGET_COLOR)
    completed_delta = _color_delta(sample_color, COMPLETED_TARGET_COLOR)

    if (
        completed_delta <= COLOR_TOLERANCE
        and completed_delta <= incomplete_delta
    ):
        return TargetBackground(
            state="completed",
            median_rgb=sample_color,
        )

    if (
        incomplete_delta <= COLOR_TOLERANCE
        and incomplete_delta < completed_delta
    ):
        return TargetBackground(
            state="incomplete",
            median_rgb=sample_color,
        )

    raise RuntimeError(
        f"target background median RGB {sample_color} was neither "
        "recognizably incomplete yellow nor completed green"
    )


def _execute_action(executor: ToolExecutor, action: Action):
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
) -> None:
    _execute_action(
        executor,
        Action(
            tool_name="move_mouse",
            arguments={"x": point[0], "y": point[1], "duration": MOVE_DURATION},
            reason="Move the cursor to the localized recovery target.",
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
            reason="Click the localized recovery target.",
        ),
    )


def _within_tolerance(
    actual: tuple[int, int],
    expected: tuple[int, int],
) -> bool:
    return (
        abs(actual[0] - expected[0]) <= POSITION_TOLERANCE
        and abs(actual[1] - expected[1]) <= POSITION_TOLERANCE
    )


def _print_initial_summary(observation: ScreenObservation) -> None:
    frame = observation.frame

    print("Phase 03 Experiment 09: Recovery Retry")
    print(f"Fixture path: {FIXTURE_PATH}")
    print(f"Target text: {TARGET_TEXT}")
    print(f"Initial screenshot path: {BEFORE_PATH}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"OCR candidate confidence: {OCR_CANDIDATE_CONFIDENCE:.2f}")
    print(f"Action confidence: {ACTION_CONFIDENCE:.2f}")
    print(f"Accepted initial OCR elements: {len(observation.elements)}")


def _prepare_attempt(
    observation: ScreenObservation,
    attempt_number: int,
) -> TargetPlan:
    source, target, match_type = _select_target(observation.elements)

    if source is None or target is None:
        raise RuntimeError(
            "no trusted exact or partial recovery target met the "
            "action confidence threshold"
        )

    click_point = _target_point(target)
    _validate_target(target, click_point, observation.frame.screen_size)

    plan_path = _attempt_plan_path(attempt_number)
    _save_plan(
        observation.image,
        observation.elements,
        target,
        click_point,
        observation.frame.screen_size,
        plan_path,
    )

    print(f"Attempt {attempt_number} match type: {match_type}")
    print(f"Attempt {attempt_number} source text: {source.text!r}")
    print(f"Attempt {attempt_number} confidence: {source.confidence:.2f}")
    print(f"Attempt {attempt_number} source logical box: {source.bounding_box}")
    print(
        f"Attempt {attempt_number} extracted target logical box: "
        f"{target.bounding_box}"
    )
    print(f"Attempt {attempt_number} planned click point: {click_point}")
    print(f"Attempt {attempt_number} plan path: {plan_path}")

    return TargetPlan(
        source=source,
        target=target,
        match_type=match_type,
        click_point=click_point,
    )


def _run_initial_preflight(observation: ScreenObservation) -> None:
    source, target, match_type = _select_target(observation.elements)

    if source is None or target is None:
        raise RuntimeError(
            "no trusted exact or partial recovery target met the "
            "action confidence threshold"
        )

    click_point = _target_point(target)
    _validate_target(target, click_point, observation.frame.screen_size)
    background = _classify_target_background(
        observation,
        target,
    )

    if background.state != "incomplete":
        raise RuntimeError(
            "initial target background was not the incomplete yellow state; "
            "reload the fixture first"
        )

    print(
        "Initial target background: incomplete "
        f"(median RGB {background.median_rgb})"
    )
    print(f"Initial target match type: {match_type}")
    print(f"Initial target confidence: {source.confidence:.2f}")


def _evaluate_after_click(
    observation: ScreenObservation,
    attempt_number: int,
    previous_click_point: tuple[int, int],
) -> bool:
    source, target, match_type = _select_target(observation.elements)

    if source is None or target is None:
        raise RuntimeError(
            "target was not detected in the after-click observation"
        )

    next_click_point = _target_point(target)
    _validate_target(target, next_click_point, observation.frame.screen_size)
    background = _classify_target_background(observation, target)

    print(f"After-click match type: {match_type}")
    print(f"After-click confidence: {source.confidence:.2f}")
    print(
        f"Detected target background after attempt {attempt_number}: "
        f"{background.state} (median RGB {background.median_rgb})"
    )

    if background.state == "completed":
        print("Recovery required: no")
        return True

    relocation_distance = _distance(previous_click_point, next_click_point)
    print(f"Previous target point: {previous_click_point}")
    print(f"New target point: {next_click_point}")
    print(f"Relocation distance: {relocation_distance:.2f}")

    if relocation_distance < MIN_RELOCATION_DISTANCE:
        raise RuntimeError(
            "re-localized target did not move far enough for recovery"
        )

    print("Recovery required: yes")
    return False


def _run_dry_run(
    observation: ScreenObservation,
) -> int:
    plan = _prepare_attempt(
        observation,
        attempt_number=1,
    )
    script_path = Path(__file__).relative_to(PROJECT_ROOT)

    print("Dry run: no mouse-control Action was created or executed.")
    print(f"Dry-run planned click point: {plan.click_point}")
    print(
        "Run this command to execute recovery retry: "
        f".venv/bin/python {script_path} --execute"
    )

    return 0


def _run_execute_mode(
    controller: ComputerController,
    initial_observation: ScreenObservation,
) -> int:
    executor = ToolExecutor(ToolRegistry(create_computer_tools(controller)))

    try:
        original = _get_mouse_position(executor)
    except RuntimeError as error:
        print(f"Safety abort: failed to read initial mouse position: {error}")
        return 1

    if _point_in_fail_safe_corner(
        original,
        initial_observation.frame.screen_size,
    ):
        print("Safety abort: the cursor is at a PyAutoGUI fail-safe corner.")
        return 1

    print(
        "Execute mode enabled. Moving the cursor to a screen corner "
        "triggers the PyAutoGUI fail-safe."
    )
    print(f"Original cursor position: {original}")
    _count_down_to_movement()

    current_observation = initial_observation
    mouse_movement_started = False
    succeeded = False

    try:
        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            print(f"Starting attempt {attempt_number} of {MAX_ATTEMPTS}.")
            plan = _prepare_attempt(
                current_observation,
                attempt_number,
            )

            mouse_movement_started = True
            _move_mouse(executor, plan.click_point)

            reached = _get_mouse_position(executor)
            if not _within_tolerance(reached, plan.click_point):
                raise RuntimeError(
                    "reached cursor position was outside the allowed tolerance"
                )

            print(f"Reached cursor position: {reached}")

            _click_mouse(executor, plan.click_point)
            print(f"Clicked position: {plan.click_point}")

            time.sleep(CLICK_SETTLE_SECONDS)

            after_path = _attempt_after_path(attempt_number)
            after_observation = _capture_observation(controller, after_path)
            print(
                f"Attempt {attempt_number} after-click OCR elements: "
                f"{len(after_observation.elements)}"
            )
            print(f"Attempt {attempt_number} after-click screenshot: {after_path}")

            if _evaluate_after_click(
                after_observation,
                attempt_number,
                plan.click_point,
            ):
                print(f"Successful attempt number: {attempt_number}")
                succeeded = True
                break

            if attempt_number == MAX_ATTEMPTS:
                raise RuntimeError(
                    "maximum attempts exhausted without completed target color"
                )

            current_observation = after_observation

    except Exception as error:
        print(f"Safety abort: {type(error).__name__}: {error}")

    finally:
        if mouse_movement_started:
            try:
                _move_mouse(executor, original)
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

    print("Phase 03 Experiment 09 recovery retry completed successfully.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry clicks on a localized Phase 03 Experiment 09 fixture "
            "target until the target background shows completion. The default "
            "mode is dry-run and performs no input action."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Move to the localized target, click it, observe the target "
            "background, retry when required, and restore the cursor."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not FIXTURE_PATH.is_file():
        print(f"Safety abort: fixture file was not found: {FIXTURE_PATH}")
        return 1

    controller = ComputerController()
    _count_down_to_capture()

    try:
        observation = _capture_observation(controller, BEFORE_PATH)
        _print_initial_summary(observation)
        _run_initial_preflight(observation)

        if not args.execute:
            return _run_dry_run(observation)

    except RuntimeError as error:
        print(f"Safety abort: {error}")
        return 1

    return _run_execute_mode(controller, observation)


if __name__ == "__main__":
    raise SystemExit(main())
