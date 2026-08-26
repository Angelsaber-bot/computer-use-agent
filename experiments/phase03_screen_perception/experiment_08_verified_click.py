"""Phase 03 experiment for verified clicking on a localized text target."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import ComputerController
from computer_agent.core.models import Action
from computer_agent.perception import (
    ScreenCapture,
    ScreenCoordinateMapper,
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
    / "experiment_08_verified_click.html"
)
BEFORE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_08_verified_click_before.png"
)
AFTER_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_08_verified_click_after.png"
)
PLAN_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_08_verified_click_plan.png"
)
TARGET_TEXT = "SAFE_CLICK_TARGET_08"
VERIFICATION_TEXT = "CLICK_VERIFIED"
OCR_CANDIDATE_CONFIDENCE = 0.05
ACTION_CONFIDENCE = 0.70
MOVE_DURATION = 1.0
CAPTURE_COUNTDOWN_SECONDS = 5
COUNTDOWN_SECONDS = 3
CLICK_SETTLE_SECONDS = 1.0
POSITION_TOLERANCE = 1
SAFE_EDGE_MARGIN = 10


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
        "Switch to the already-open Chrome fixture before the initial "
        "screenshot is captured."
    )

    for remaining in range(CAPTURE_COUNTDOWN_SECONDS, 0, -1):
        print(f"Capturing in {remaining}...")
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


def _save_plan(
    image: Image.Image,
    elements: tuple[UIElement, ...],
    target: UIElement,
    point: tuple[int, int],
    logical_size: tuple[int, int],
) -> None:
    plan = image.resize(logical_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(plan)

    for element in elements:
        _draw_box(draw, element, (0, 180, 0), 1)

    _draw_box(draw, target, (255, 0, 0), 4)
    _draw_crosshair(draw, point)

    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.save(PLAN_PATH)

    with Image.open(PLAN_PATH) as saved:
        assert saved.mode == "RGB"
        assert saved.size == logical_size


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
            reason="Move the cursor to the localized text target.",
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
            reason="Click the localized verified-click target.",
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


def _find_verification_text(
    elements: tuple[UIElement, ...],
) -> UIElement | None:
    verification = TextTargetLocator.find_best(
        elements,
        VERIFICATION_TEXT,
        minimum_confidence=ACTION_CONFIDENCE,
    )

    if verification is None:
        verification = TextTargetLocator.find_best(
            elements,
            VERIFICATION_TEXT,
            partial_match=True,
            minimum_confidence=ACTION_CONFIDENCE,
        )

    return verification


def _capture_after_click_verification(
    controller: ComputerController,
) -> tuple[UIElement | None, int]:
    frame = ScreenCapture(controller).capture(AFTER_PATH)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    elements = TesseractOCR(
        minimum_confidence=OCR_CANDIDATE_CONFIDENCE
    ).recognize(image)

    return _find_verification_text(elements), len(elements)


def _run_execute_mode(
    controller: ComputerController,
    screen_size: tuple[int, int],
    click_point: tuple[int, int],
) -> int:
    executor = ToolExecutor(ToolRegistry(create_computer_tools(controller)))

    try:
        original = _get_mouse_position(executor)
    except RuntimeError as error:
        print(f"Safety abort: failed to read initial mouse position: {error}")
        return 1

    if _point_in_fail_safe_corner(original, screen_size):
        print("Safety abort: the cursor is at a PyAutoGUI fail-safe corner.")
        return 1

    print(
        "Execute mode enabled. Moving the cursor to a screen corner "
        "triggers the PyAutoGUI fail-safe."
    )
    print(f"Original cursor position: {original}")

    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"Moving in {remaining}...")
        time.sleep(1)

    moved_to_target = False
    succeeded = False

    try:
        _move_mouse(executor, click_point)
        moved_to_target = True

        reached = _get_mouse_position(executor)
        if not _within_tolerance(reached, click_point):
            raise RuntimeError(
                "reached cursor position was outside the allowed tolerance"
            )

        print(f"Reached cursor position: {reached}")

        _click_mouse(executor, click_point)
        print(f"Clicked target position: {click_point}")

        time.sleep(CLICK_SETTLE_SECONDS)

        verification, after_element_count = _capture_after_click_verification(
            controller
        )
        print(f"After-click OCR elements: {after_element_count}")
        print(f"After-click screenshot: {AFTER_PATH}")

        if verification is None:
            raise RuntimeError(
                f"verification text {VERIFICATION_TEXT!r} was not detected"
            )

        print(f"Verification text: {verification.text!r}")
        print(f"Verification confidence: {verification.confidence:.2f}")
        succeeded = True

    except RuntimeError as error:
        print(f"Safety abort: {error}")

    finally:
        if moved_to_target:
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

    print("Phase 03 Experiment 08 verified click completed successfully.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a click on a localized Phase 03 Experiment 08 fixture "
            "target. The default mode is dry-run and performs no input action."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Move to the localized target, click it, verify the fixture state, "
            "and restore the cursor."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not FIXTURE_PATH.is_file():
        print(f"Safety abort: fixture file was not found: {FIXTURE_PATH}")
        return 1

    BEFORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    controller = ComputerController()
    _count_down_to_capture()
    frame = ScreenCapture(controller).capture(BEFORE_PATH)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    assert image.size == frame.pixel_size

    ocr = TesseractOCR(minimum_confidence=OCR_CANDIDATE_CONFIDENCE)
    mapper = ScreenCoordinateMapper(frame)
    logical_elements = tuple(
        mapper.pixel_element_to_logical(element)
        for element in ocr.recognize(image)
    )
    existing_verification = _find_verification_text(logical_elements)

    if existing_verification is not None:
        print(
            "Safety abort: the fixture is already in its verified state. "
            "Reload the webpage before running the experiment."
        )
        return 1
    source, target, match_type = _select_target(logical_elements)

    print("Phase 03 Experiment 08: Verified Click")
    print(f"Fixture path: {FIXTURE_PATH}")
    print(f"Target text: {TARGET_TEXT}")
    print(f"Verification text: {VERIFICATION_TEXT}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"OCR candidate confidence: {OCR_CANDIDATE_CONFIDENCE:.2f}")
    print(f"Action confidence: {ACTION_CONFIDENCE:.2f}")
    print(f"Accepted OCR source elements: {len(logical_elements)}")

    if source is None or target is None:
        print(
            "Safety abort: no trusted exact or partial text target "
            "met the action confidence threshold."
        )
        return 1

    click_point = _target_point(target)

    if not _box_inside_screen(target, frame.screen_size):
        print("Safety abort: extracted target box is outside the screen.")
        return 1

    if not _point_inside_screen(click_point, frame.screen_size):
        print("Safety abort: planned click point is outside the screen.")
        return 1

    if not _point_in_safe_area(click_point, frame.screen_size):
        print("Safety abort: planned click point is too close to a screen edge.")
        return 1

    _save_plan(image, logical_elements, target, click_point, frame.screen_size)

    print(f"Match type: {match_type}")
    print(f"Source text: {source.text!r}")
    print(f"Confidence: {source.confidence:.2f}")
    print(f"Source logical box: {source.bounding_box}")
    print(f"Extracted target logical box: {target.bounding_box}")
    print(f"Planned click point: {click_point}")
    print(f"Click plan path: {PLAN_PATH}")

    if not args.execute:
        script_path = Path(__file__).relative_to(PROJECT_ROOT)
        print("Dry run: no mouse-control Action was created or executed.")
        print(
            "Run this command to execute verified clicking: "
            f".venv/bin/python {script_path} --execute"
        )
        return 0

    return _run_execute_mode(controller, frame.screen_size, click_point)


if __name__ == "__main__":
    raise SystemExit(main())
