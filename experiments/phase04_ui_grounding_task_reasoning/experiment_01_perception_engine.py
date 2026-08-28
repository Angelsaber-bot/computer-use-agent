"""Phase 04 experiment for the reusable perception engine."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase03_screen_perception"
    / "experiment_12_hybrid_perception.html"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase04_ui_grounding_task_reasoning"
    / "experiment_01_perception_engine.png"
)

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6
SENTINELS = (
    "TARGET_INPUT_12",
    "NATIVE_BUTTON_12",
    "CANVAS_ACTION_12",
)


def _wait_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be an integer"
        ) from error

    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError(
            "--wait-seconds must be from 0 through 30"
        )

    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Observe the already-focused Phase 03 Experiment 12 fixture "
            "with the reusable PerceptionEngine."
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=5,
        help=(
            "Seconds to wait before observation so the fixture can be "
            "focused. Must be from 0 through 30. Defaults to 5."
        ),
    )

    return parser.parse_args()


def _print_manual_focus_instructions() -> None:
    print("Phase 04 Experiment 01: Reusable Perception Engine")
    print(f"Fixture path: {FIXTURE_PATH}")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before observation.")
    print(
        "This experiment only observes the screen and will not interact "
        "with the computer."
    )


def _wait_for_focus(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        time.sleep(1)


def _build_engine():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import (
        MacOSAccessibility,
        PerceptionEngine,
        ScreenCapture,
        TesseractOCR,
        UIElementFusion,
    )

    controller = ComputerController()
    screen_capture = ScreenCapture(controller)
    accessibility_reader = MacOSAccessibility()
    ocr = TesseractOCR(
        minimum_confidence=OCR_MINIMUM_CONFIDENCE,
        page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
        group_words_by_line=True,
    )
    fusion = UIElementFusion()

    return PerceptionEngine(
        screen_capture=screen_capture,
        accessibility_reader=accessibility_reader,
        ocr=ocr,
        fusion=fusion,
        capture_path=EVIDENCE_PATH,
    )


def _normalize_text(text: str | None) -> str:
    from computer_agent.perception import normalize_ui_text

    return normalize_ui_text(text)


def _observed_texts(snapshot) -> tuple[str, ...]:
    texts = []

    for collection in (
        snapshot.accessibility_elements,
        snapshot.ocr_elements,
        snapshot.fused_elements,
    ):
        for element in collection:
            if element.text is not None and element.text.strip():
                texts.append(element.text)

    return tuple(texts)


def _sentinel_observed(snapshot, sentinel: str) -> bool:
    expected = _normalize_text(sentinel)

    return any(
        expected in _normalize_text(text)
        for text in _observed_texts(snapshot)
    )


def _source_distribution(snapshot) -> dict[str, int]:
    counts = Counter(
        element.source or "unspecified"
        for element in snapshot.fused_elements
    )

    return dict(sorted(counts.items()))


def _sentinel_results(snapshot) -> dict[str, bool]:
    return {
        sentinel: _sentinel_observed(
            snapshot,
            sentinel,
        )
        for sentinel in SENTINELS
    }


def _print_evidence(snapshot) -> None:
    frame = snapshot.frame
    counts = snapshot.source_counts

    print(f"Screenshot path: {EVIDENCE_PATH}")
    print(f"Screenshot pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(f"Coordinate scale: x={frame.scale_x:.2f}, y={frame.scale_y:.2f}")
    print(f"Capture timestamp: {frame.captured_at.isoformat()}")
    print(f"Accessibility element count: {counts['accessibility']}")
    print(f"Logical OCR element count: {counts['ocr']}")
    print(f"Fused element count: {counts['fused']}")

    if snapshot.warnings:
        print("Warnings:")
        for warning in snapshot.warnings:
            print(f"  {warning}")
    else:
        print("Warnings: none")

    print(f"Fused element source distribution: {_source_distribution(snapshot)}")
    print("Fixture sentinel observations:")
    for sentinel, observed in _sentinel_results(snapshot).items():
        print(f"  {sentinel}: {observed}")


def _acceptance_failures(snapshot) -> list[str]:
    failures = []
    counts = snapshot.source_counts
    sentinel_results = _sentinel_results(snapshot)

    if not EVIDENCE_PATH.is_file():
        failures.append(f"screenshot was not created at {EVIDENCE_PATH}")
    else:
        with Image.open(EVIDENCE_PATH) as saved:
            if saved.size != snapshot.frame.pixel_size:
                failures.append(
                    "loaded screenshot size did not match frame pixel size: "
                    f"{saved.size} != {snapshot.frame.pixel_size}"
                )

    if snapshot.image.mode != "RGB":
        failures.append(f"snapshot image mode was {snapshot.image.mode}, not RGB")

    if counts["accessibility"] <= 0:
        failures.append("Accessibility element count was not greater than zero")

    if counts["ocr"] <= 0:
        failures.append("logical OCR element count was not greater than zero")

    if counts["fused"] <= 0:
        failures.append("fused element count was not greater than zero")

    if snapshot.warnings:
        failures.append("source warning occurred")

    for sentinel, observed in sentinel_results.items():
        if not observed:
            failures.append(f"fixture sentinel was not observed: {sentinel}")

    if not any(element.source == "hybrid" for element in snapshot.fused_elements):
        failures.append("no fused element had hybrid source metadata")

    return failures


def _print_acceptance_result(snapshot) -> int:
    failures = _acceptance_failures(snapshot)

    if failures:
        print("Live acceptance failed:")
        for failure in failures:
            print(f"  {failure}")

        return 1

    print("Live acceptance result: passed")
    return 0


def main() -> int:
    args = _parse_args()

    if sys.platform != "darwin":
        print("Live acceptance failed: this experiment runs only on macOS.")
        return 1

    if not FIXTURE_PATH.is_file():
        print(f"Live acceptance failed: fixture file was not found: {FIXTURE_PATH}")
        return 1

    _print_manual_focus_instructions()
    _wait_for_focus(args.wait_seconds)

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    engine = _build_engine()

    try:
        snapshot = engine.observe()
    except RuntimeError as error:
        if str(error).startswith("captured image size mismatch:"):
            print(f"Observation failed: RuntimeError: {error}")
            return 1

        raise
    except (FileNotFoundError, OSError) as error:
        print(f"Observation failed: {type(error).__name__}: {error}")
        return 1

    _print_evidence(snapshot)
    return _print_acceptance_result(snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
