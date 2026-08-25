"""Phase 03 experiment for UI text localization."""

from pathlib import Path

from PIL import Image, ImageDraw

from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.perception import (
    ScreenCapture,
    ScreenCoordinateMapper,
    TesseractOCR,
    TextTargetLocator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_06_ui_text_localization_input.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_06_ui_text_localization.png"
)
TARGET_TEXT = "computer_agent"
MINIMUM_CONFIDENCE = 0.05


def _preview_text(text: str) -> str:
    preview = text.encode(
        "ascii",
        "backslashreplace",
    ).decode("ascii")
    preview = preview.replace(
        "\n",
        "\\n",
    ).replace(
        "\r",
        "\\r",
    )

    if len(preview) > 40:
        return f"{preview[:37]}..."

    return preview


def _format_box(box) -> str:
    return (
        "("
        f"x={box.x}, "
        f"y={box.y}, "
        f"width={box.width}, "
        f"height={box.height}"
        ")"
    )


def _format_point(point: tuple[float, float]) -> str:
    return (
        "("
        f"x={point[0]:.2f}, "
        f"y={point[1]:.2f}"
        ")"
    )


def _assert_box_inside_screen(
    box,
    screen_width: int,
    screen_height: int,
) -> None:
    assert 0 <= box.left < box.right <= screen_width
    assert 0 <= box.top < box.bottom <= screen_height


def _assert_point_inside_screen(
    point: tuple[float, float],
    screen_width: int,
    screen_height: int,
) -> None:
    x, y = point

    assert 0 <= x < screen_width
    assert 0 <= y < screen_height


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
) -> None:
    x = round(center[0])
    y = round(center[1])
    radius = 12

    draw.line(
        (
            x - radius,
            y,
            x + radius,
            y,
        ),
        fill=(255, 0, 0),
        width=3,
    )
    draw.line(
        (
            x,
            y - radius,
            x,
            y + radius,
        ),
        fill=(255, 0, 0),
        width=3,
    )


def _is_exact_match(
    element,
    exact_matches,
) -> bool:
    return any(
        element is exact_match
        for exact_match in exact_matches
    )


def main() -> int:
    INPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller = ComputerController()
    capture = ScreenCapture(controller)
    frame = capture.capture(INPUT_PATH)

    with Image.open(frame.image_path) as source:
        image = source.convert("RGB")

    assert image.size == frame.pixel_size

    ocr = TesseractOCR(
        minimum_confidence=MINIMUM_CONFIDENCE,
    )
    pixel_elements = ocr.recognize(image)

    mapper = ScreenCoordinateMapper(frame)
    logical_elements = tuple(
        mapper.pixel_element_to_logical(element)
        for element in pixel_elements
    )
    exact_matches = TextTargetLocator.find_all(
        logical_elements,
        TARGET_TEXT,
    )
    candidate_matches = TextTargetLocator.find_all(
        logical_elements,
        TARGET_TEXT,
        partial_match=True,
    )
    candidate_targets = []

    for candidate in candidate_matches:
        extracted = TextTargetLocator.extract_target(
            candidate,
            TARGET_TEXT,
        )
        assert extracted is not None
        candidate_targets.append(
            (
                candidate,
                extracted,
                "exact"
                if _is_exact_match(
                    candidate,
                    exact_matches,
                )
                else "partial",
            )
        )

    assert candidate_matches, (
        f"Expected at least one target candidate: {TARGET_TEXT}"
    )

    selected_source = max(
        exact_matches or candidate_matches,
        key=lambda element: element.confidence,
    )
    selected_target = TextTargetLocator.extract_target(
        selected_source,
        TARGET_TEXT,
    )
    assert selected_target is not None
    selected_center = selected_target.center

    for _, target, _ in candidate_targets:
        _assert_box_inside_screen(
            target.bounding_box,
            frame.screen_width,
            frame.screen_height,
        )
        _assert_point_inside_screen(
            target.center,
            frame.screen_width,
            frame.screen_height,
        )

    _assert_point_inside_screen(
        selected_center,
        frame.screen_width,
        frame.screen_height,
    )

    visualization = image.resize(
        frame.screen_size,
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(visualization)

    for element in logical_elements:
        box = element.bounding_box
        draw.rectangle(
            (
                box.left,
                box.top,
                box.right - 1,
                box.bottom - 1,
            ),
            outline=(0, 180, 0),
            width=1,
        )

    for _, target, _ in candidate_targets:
        box = target.bounding_box
        draw.rectangle(
            (
                box.left,
                box.top,
                box.right - 1,
                box.bottom - 1,
            ),
            outline=(255, 0, 0),
            width=4,
        )

    _draw_crosshair(
        draw,
        selected_center,
    )

    visualization.save(OUTPUT_PATH)

    with Image.open(OUTPUT_PATH) as saved_image:
        assert saved_image.mode == "RGB"
        assert saved_image.size == frame.screen_size

    print("Phase 03 Experiment 06: UI Text Localization")
    print(f"Target text: {TARGET_TEXT}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(
        "Coordinate scale: "
        f"x={frame.scale_x:.2f}, "
        f"y={frame.scale_y:.2f}"
    )
    print(f"Total recognized OCR elements: {len(logical_elements)}")
    print(f"Exact matches: {len(exact_matches)}")
    print(f"Candidate matches: {len(candidate_matches)}")

    for index, match in enumerate(
        candidate_targets,
        start=1,
    ):
        source, target, match_type = match
        print(
            f"{index:02d}. "
            f"source_text='{_preview_text(source.text or '')}', "
            f"match_type={match_type}, "
            f"confidence={source.confidence:.2f}, "
            f"source_logical_bbox={_format_box(source.bounding_box)}, "
            f"target_logical_bbox={_format_box(target.bounding_box)}, "
            f"target_logical_center={_format_point(target.center)}"
        )

    print(
        "Selected extracted target center: "
        f"{_format_point(selected_center)}"
    )
    print(f"Visualization path: {OUTPUT_PATH}")
    print(
        "This experiment verifies localization only and does not "
        "move or click the mouse."
    )
    print("Phase 03 Experiment 06 completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
