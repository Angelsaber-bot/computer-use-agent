"""Phase 03 experiment for deterministic image preprocessing."""

from pathlib import Path

from PIL import Image

from computer_agent.perception import ImagePreprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_01_screen_capture.png"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_03_image_preprocessing.png"
)
SCALE_FACTOR = 0.5


def main() -> int:
    with Image.open(SOURCE_PATH) as image:
        source_image = image.copy()

    preprocessor = ImagePreprocessor()
    grayscale_image = preprocessor.convert_to_grayscale(source_image)
    resized_image = preprocessor.resize(
        grayscale_image,
        SCALE_FACTOR,
    )
    enhanced_image = preprocessor.enhance_contrast(resized_image)
    prepared_image = preprocessor.prepare_for_ocr(
        source_image,
        scale_factor=SCALE_FACTOR,
    )

    expected_dimensions = (
        max(
            1,
            int(source_image.width * SCALE_FACTOR),
        ),
        max(
            1,
            int(source_image.height * SCALE_FACTOR),
        ),
    )

    assert grayscale_image.mode == "L"
    assert resized_image.size == expected_dimensions
    assert enhanced_image.mode == "L"
    assert prepared_image.mode == "L"
    assert prepared_image.size == expected_dimensions
    assert prepared_image.tobytes() == enhanced_image.tobytes()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    prepared_image.save(OUTPUT_PATH)

    print("Phase 03 Experiment 03: Image Preprocessing")
    print(f"Source mode: {source_image.mode}")
    print(f"Output mode: {prepared_image.mode}")
    print(f"Source dimensions: {source_image.size}")
    print(f"Output dimensions: {prepared_image.size}")
    print(f"Saved output path: {OUTPUT_PATH}")
    print(
        "Phase 03 Experiment 03 completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
