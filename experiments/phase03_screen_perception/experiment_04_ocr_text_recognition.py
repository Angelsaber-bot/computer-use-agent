"""Phase 03 experiment for OCR text recognition."""

from pathlib import Path

from PIL import Image, ImageDraw
import pytesseract

from computer_agent.perception import TesseractOCR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_01_screen_capture.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_04_ocr_text_recognition.png"
)
MINIMUM_CONFIDENCE = 0.70
PREVIEW_LIMIT = 20
EXPECTED_PIXEL_SIZE = (2940, 1912)
LOGICAL_SCREEN_SIZE = (1470, 956)
SCALE_X = EXPECTED_PIXEL_SIZE[0] / LOGICAL_SCREEN_SIZE[0]
SCALE_Y = EXPECTED_PIXEL_SIZE[1] / LOGICAL_SCREEN_SIZE[1]
EXPECTED_ANCHORS = (
    "computer_agent",
    "Project",
    "experiments",
    "experiment_01_screen_capture",
    "Screenshot",
    "Pixel",
    "Logical",
    "Coordinate",
    "Captured",
    "print",
    "return",
    "main",
)
MINIMUM_MATCHED_ANCHORS = 8


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


def _matched_anchors(elements) -> tuple[str, ...]:
    normalized_text = " ".join(
        (element.text or "").casefold()
        for element in elements
    )

    return tuple(
        anchor
        for anchor in EXPECTED_ANCHORS
        if anchor.casefold() in normalized_text
    )


def main() -> int:
    if not TesseractOCR.is_available():
        raise RuntimeError(
            "Tesseract executable is not available"
        )

    with Image.open(INPUT_PATH) as source:
        image = source.convert("RGB")

    assert image.mode == "RGB"
    assert image.size == EXPECTED_PIXEL_SIZE

    ocr = TesseractOCR(
        minimum_confidence=MINIMUM_CONFIDENCE,
    )
    elements = ocr.recognize(image)

    assert elements, "Expected at least one recognized text element"
    matched_anchors = _matched_anchors(elements)

    assert len(matched_anchors) >= MINIMUM_MATCHED_ANCHORS, (
        "Expected at least "
        f"{MINIMUM_MATCHED_ANCHORS} matched OCR anchors, "
        f"got {len(matched_anchors)}"
    )

    for element in elements:
        box = element.bounding_box

        assert element.text is not None
        assert element.text.strip()
        assert 0.0 <= element.confidence <= 1.0
        assert 0 <= box.left < box.right <= image.width
        assert 0 <= box.top < box.bottom <= image.height

    annotated_image = image.convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    for element in elements:
        box = element.bounding_box
        draw.rectangle(
            (
                box.left,
                box.top,
                box.right - 1,
                box.bottom - 1,
            ),
            outline=(255, 0, 0),
            width=2,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    annotated_image.save(OUTPUT_PATH)

    print("Phase 03 Experiment 04: OCR Text Recognition")
    print(f"Tesseract version: {pytesseract.get_tesseract_version()}")
    print(f"Input image mode: {image.mode}")
    print(f"Input pixel dimensions: {image.size}")
    print(f"Logical screen dimensions: {LOGICAL_SCREEN_SIZE}")
    print(
        "Coordinate scale: "
        f"x={SCALE_X:.2f}, "
        f"y={SCALE_Y:.2f}"
    )
    print(f"Minimum confidence: {MINIMUM_CONFIDENCE:.2f}")
    print(f"Recognized text elements: {len(elements)}")
    print(
        "Matched anchors: "
        f"{len(matched_anchors)}/{len(EXPECTED_ANCHORS)} "
        f"{', '.join(matched_anchors)}"
    )
    print(
        "OCR bounding boxes are high-resolution pixel coordinates, "
        "not PyAutoGUI logical coordinates."
    )
    print(
        "Future screen parsing will convert coordinates with "
        "logical_x = pixel_x / scale_x and "
        "logical_y = pixel_y / scale_y."
    )
    print("Preview:")

    for index, element in enumerate(
        elements[:PREVIEW_LIMIT],
        start=1,
    ):
        box = element.bounding_box
        print(
            f"{index:02d}. "
            f"text='{_preview_text(element.text or '')}', "
            f"confidence={element.confidence:.2f}, "
            "pixel_bbox=("
            f"x={box.x}, "
            f"y={box.y}, "
            f"width={box.width}, "
            f"height={box.height}"
            ")"
        )

    print(f"Saved annotated image: {OUTPUT_PATH}")
    print("Phase 03 Experiment 04 completed successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
