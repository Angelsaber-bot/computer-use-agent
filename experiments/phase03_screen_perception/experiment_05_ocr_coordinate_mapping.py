"""Phase 03 experiment for OCR coordinate mapping."""

from pathlib import Path

from PIL import Image, ImageDraw
import pytesseract

from computer_agent.perception import (
    ScreenCoordinateMapper,
    ScreenFrame,
    TesseractOCR,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_01_screen_capture.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_05_ocr_coordinate_mapping.png"
)
PIXEL_SIZE = (2940, 1912)
LOGICAL_SCREEN_SIZE = (1470, 956)
MINIMUM_CONFIDENCE = 0.70
PREVIEW_LIMIT = 12


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


def main() -> int:
    if not TesseractOCR.is_available():
        raise RuntimeError(
            "Tesseract executable is not available"
        )

    with Image.open(INPUT_PATH) as source:
        image = source.convert("RGB")

    frame = ScreenFrame(
        image_path=INPUT_PATH.resolve(),
        pixel_width=PIXEL_SIZE[0],
        pixel_height=PIXEL_SIZE[1],
        screen_width=LOGICAL_SCREEN_SIZE[0],
        screen_height=LOGICAL_SCREEN_SIZE[1],
    )

    assert image.mode == "RGB"
    assert image.size == frame.pixel_size
    assert frame.screen_size == LOGICAL_SCREEN_SIZE

    ocr = TesseractOCR(
        minimum_confidence=MINIMUM_CONFIDENCE,
    )
    pixel_elements = ocr.recognize(image)
    mapper = ScreenCoordinateMapper(frame)
    logical_elements = tuple(
        mapper.pixel_element_to_logical(element)
        for element in pixel_elements
    )

    assert len(logical_elements) == len(pixel_elements)
    assert logical_elements, "Expected at least one mapped OCR element"

    for element in logical_elements:
        box = element.bounding_box

        assert 0 <= box.left < box.right <= frame.screen_width
        assert 0 <= box.top < box.bottom <= frame.screen_height

    logical_image = image.resize(
        frame.screen_size,
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(logical_image)

    for element in logical_elements:
        box = element.bounding_box
        draw.rectangle(
            (
                box.left,
                box.top,
                box.right - 1,
                box.bottom - 1,
            ),
            outline=(0, 200, 0),
            width=2,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    logical_image.save(OUTPUT_PATH)

    with Image.open(OUTPUT_PATH) as saved_image:
        assert saved_image.mode == "RGB"
        assert saved_image.size == frame.screen_size

    print("Phase 03 Experiment 05: OCR Coordinate Mapping")
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    print(f"Input image mode: {image.mode}")
    print(f"Input pixel dimensions: {image.size}")
    print(f"Logical screen dimensions: {frame.screen_size}")
    print(
        "Coordinate scale: "
        f"x={frame.scale_x:.2f}, "
        f"y={frame.scale_y:.2f}"
    )
    print(f"Minimum confidence: {MINIMUM_CONFIDENCE:.2f}")
    print(f"OCR pixel elements: {len(pixel_elements)}")
    print(f"Mapped logical elements: {len(logical_elements)}")
    print(
        "Mapped boxes use PyAutoGUI logical coordinates. "
        "This experiment performs no mouse, keyboard, "
        "or other computer-control action."
    )
    print(
        "Rounding policy: floor logical left/top edges "
        "and ceil logical right/bottom edges so each "
        "logical box contains the mapped pixel region."
    )
    print("Preview:")

    for index, (pixel_element, logical_element) in enumerate(
        zip(
            pixel_elements[:PREVIEW_LIMIT],
            logical_elements[:PREVIEW_LIMIT],
        ),
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"text='{_preview_text(pixel_element.text or '')}', "
            f"confidence={pixel_element.confidence:.2f}, "
            f"pixel_bbox={_format_box(pixel_element.bounding_box)}, "
            f"logical_bbox={_format_box(logical_element.bounding_box)}"
        )

    print(f"Saved logical-coordinate visualization: {OUTPUT_PATH}")
    print("Phase 03 Experiment 05 completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
