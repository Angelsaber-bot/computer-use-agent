"""Phase 05 Experiment 01: real-web semantic perception acceptance."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase05_real_web_autonomy"
    / "experiment_01_real_web_accessibility.png"
)

TARGET_URL = "https://www.python.org/"

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6

REQUIRED_WEB_TYPES = (
    "link",
    "heading",
    "text",
    "text_field",
)

PAGE_MARKERS = (
    "Python",
    "Search This Site",
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
            "Observe python.org through the production hybrid "
            "PerceptionEngine without interacting with the computer."
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=8,
        help=(
            "Seconds to wait before observation. "
            "Must be from 0 through 30."
        ),
    )

    return parser.parse_args()


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
    accessibility = MacOSAccessibility()

    engine = PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=EVIDENCE_PATH,
    )

    return engine, accessibility


def _wait_for_focus(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        time.sleep(1)


def _normalized_text(value: str | None) -> str:
    from computer_agent.perception import normalize_ui_text

    return normalize_ui_text(value)


def _type_counts(elements) -> Counter[str]:
    return Counter(
        element.element_type
        for element in elements
    )


def _text_samples(elements) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = defaultdict(list)

    for element in elements:
        if element.text is None or not element.text.strip():
            continue

        if len(samples[element.element_type]) >= 5:
            continue

        samples[element.element_type].append(
            element.text
        )

    return dict(samples)


def _visible_on_screen(element, screen_size: tuple[int, int]) -> bool:
    width, height = screen_size
    box = element.bounding_box

    return (
        box.x < width
        and box.y < height
        and box.x + box.width > 0
        and box.y + box.height > 0
    )


def _marker_observed(elements, marker: str) -> bool:
    expected = _normalized_text(marker)

    return any(
        element.text is not None
        and expected in _normalized_text(element.text)
        for element in elements
    )


def _print_evidence(
    snapshot,
    frontmost_app: str | None,
) -> None:
    accessibility_counts = _type_counts(
        snapshot.accessibility_elements
    )
    samples = _text_samples(
        snapshot.accessibility_elements
    )

    print(f"Frontmost application: {frontmost_app}")
    print(f"Screenshot path: {EVIDENCE_PATH}")
    print(f"Screenshot pixel size: {snapshot.frame.pixel_size}")
    print(f"Logical screen size: {snapshot.frame.screen_size}")
    print(
        "Coordinate scale: "
        f"x={snapshot.frame.scale_x:.2f}, "
        f"y={snapshot.frame.scale_y:.2f}"
    )
    print(
        "Capture timestamp: "
        f"{snapshot.frame.captured_at.isoformat()}"
    )

    print(
        "Accessibility element count: "
        f"{len(snapshot.accessibility_elements)}"
    )
    print(
        "OCR element count: "
        f"{len(snapshot.ocr_elements)}"
    )
    print(
        "Fused element count: "
        f"{len(snapshot.fused_elements)}"
    )

    print("Accessibility semantic type counts:")

    for element_type, count in sorted(
        accessibility_counts.items()
    ):
        print(f"  {element_type}: {count}")

        for sample in samples.get(element_type, ()):
            print(f"    text: {sample!r}")

    if snapshot.warnings:
        print("Warnings:")
        for warning in snapshot.warnings:
            print(f"  {warning}")
    else:
        print("Warnings: none")


def _acceptance_failures(
    snapshot,
    frontmost_app: str | None,
) -> list[str]:
    failures: list[str] = []

    if frontmost_app != "Google Chrome":
        failures.append(
            "frontmost application was not Google Chrome"
        )

    if not EVIDENCE_PATH.is_file():
        failures.append(
            "evidence screenshot was not created"
        )
    else:
        with Image.open(EVIDENCE_PATH) as saved:
            if saved.size != snapshot.frame.pixel_size:
                failures.append(
                    "saved screenshot size did not match "
                    "captured frame size"
                )

    if snapshot.warnings:
        failures.append(
            "perception source warning occurred"
        )

    if not snapshot.accessibility_elements:
        failures.append(
            "Accessibility returned no elements"
        )

    if not snapshot.ocr_elements:
        failures.append(
            "OCR returned no elements"
        )

    if not snapshot.fused_elements:
        failures.append(
            "fusion returned no elements"
        )

    screen_size = snapshot.frame.screen_size

    for element_type in REQUIRED_WEB_TYPES:
        typed_elements = [
            element
            for element in snapshot.accessibility_elements
            if element.element_type == element_type
        ]

        if not typed_elements:
            failures.append(
                f"required web type was not observed: {element_type}"
            )
            continue

        meaningful = [
            element
            for element in typed_elements
            if element.text is not None
            and element.text.strip()
        ]

        if not meaningful:
            failures.append(
                "required web type had no meaningful text: "
                f"{element_type}"
            )
            continue

        if not any(
            _visible_on_screen(element, screen_size)
            for element in meaningful
        ):
            failures.append(
                "required web type had no visible element: "
                f"{element_type}"
            )

    for marker in PAGE_MARKERS:
        if not _marker_observed(
            snapshot.accessibility_elements,
            marker,
        ):
            failures.append(
                f"python.org page marker was not observed: {marker}"
            )

    return failures


def main() -> int:
    args = _parse_args()

    print(
        "Phase 05 Experiment 01: "
        "Real Web Accessibility Perception"
    )
    print(f"Target URL: {TARGET_URL}")
    print(
        "Open the target URL manually in Google Chrome "
        "and keep it visible and focused."
    )
    print(
        "This experiment is read-only and will not move, click, "
        "type, scroll, navigate, or execute computer actions."
    )

    if sys.platform != "darwin":
        print(
            "Live acceptance failed: "
            "this experiment requires macOS."
        )
        return 1

    engine, accessibility = _build_engine()

    if not accessibility.is_available():
        print(
            "Live acceptance failed: "
            "macOS Accessibility is unavailable."
        )
        return 1

    if not accessibility.is_trusted():
        print(
            "Live acceptance failed: "
            "Accessibility permission is not trusted."
        )
        return 1

    _wait_for_focus(args.wait_seconds)

    frontmost_app = (
        accessibility.read_frontmost_application_name()
    )

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = engine.observe()

    _print_evidence(
        snapshot,
        frontmost_app,
    )

    failures = _acceptance_failures(
        snapshot,
        frontmost_app,
    )

    if failures:
        print("Live acceptance failed:")

        for failure in failures:
            print(f"  {failure}")

        return 1

    print("Live acceptance result: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
