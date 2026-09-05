"""Phase 05 Experiment 02: inspect semantic grounding on a real webpage."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVIDENCE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase05_real_web_autonomy"
    / "experiment_02_real_web_semantic_grounding.png"
)

TARGET_URL = "https://www.python.org/"

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6
GROUNDING_MINIMUM_CONFIDENCE = 0.70


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
            "Observe python.org and inspect deterministic semantic "
            "grounding without interacting with the computer."
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=8,
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

    return (
        PerceptionEngine(
            screen_capture=ScreenCapture(controller),
            accessibility_reader=MacOSAccessibility(),
            ocr=TesseractOCR(
                minimum_confidence=OCR_MINIMUM_CONFIDENCE,
                page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
                group_words_by_line=True,
            ),
            fusion=UIElementFusion(),
            capture_path=EVIDENCE_PATH,
        ),
        MacOSAccessibility(),
    )


def _build_targets():
    from computer_agent.grounding import TargetSpec

    return (
        (
            "Docs link",
            TargetSpec(
                text="Docs",
                element_types=("link",),
                minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
            ),
        ),
        (
            "Docs heading",
            TargetSpec(
                text="Docs",
                element_types=("heading",),
                minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
            ),
        ),
        (
            "Search field",
            TargetSpec(
                text="Search This Site",
                element_types=("text_field",),
                minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
            ),
        ),
        (
            "Missing link",
            TargetSpec(
                text="PHASE05_MISSING_LINK_02",
                element_types=("link",),
                minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
            ),
        ),
    )


def _format_box(element) -> str:
    box = element.bounding_box

    return (
        f"x={box.x}, y={box.y}, "
        f"width={box.width}, height={box.height}"
    )


def _print_result(name, target, result) -> None:
    print()
    print(f"Case: {name}")
    print(f"TargetSpec: {target!r}")
    print(f"Status: {result.status.value}")
    print(f"Reason: {result.reason}")
    print(f"Candidate count: {len(result.candidates)}")

    if result.element is None:
        print("Resolved element: None")
    else:
        element = result.element
        print("Resolved element:")
        print(f"  type: {element.element_type!r}")
        print(f"  text: {element.text!r}")
        print(f"  source: {element.source!r}")
        print(f"  confidence: {element.confidence:.2f}")
        print(f"  enabled: {element.enabled!r}")
        print(f"  box: {_format_box(element)}")

    for index, candidate in enumerate(
        result.candidates,
        start=1,
    ):
        element = candidate.element

        print(f"Candidate {index}:")
        print(f"  type: {element.element_type!r}")
        print(f"  text: {element.text!r}")
        print(f"  source: {element.source!r}")
        print(f"  confidence: {element.confidence:.2f}")
        print(f"  enabled: {element.enabled!r}")
        print(f"  box: {_format_box(element)}")
        print(f"  eligible: {candidate.eligible}")
        print(
            "  rejection reasons: "
            f"{candidate.rejection_reasons!r}"
        )


def main() -> int:
    args = _parse_args()

    print(
        "Phase 05 Experiment 02: "
        "Real Web Semantic Grounding"
    )
    print(f"Target URL: {TARGET_URL}")
    print(
        "Open python.org manually in Google Chrome and keep it "
        "visible and focused."
    )
    print(
        "This experiment is read-only and will not move, click, "
        "type, scroll, navigate, or execute computer actions."
    )

    if sys.platform != "darwin":
        print("Live acceptance failed: macOS is required.")
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

    for remaining in range(
        args.wait_seconds,
        0,
        -1,
    ):
        print(f"Observing in {remaining}...")
        time.sleep(1)

    print(
        "Frontmost application:",
        accessibility.read_frontmost_application_name(),
    )

    EVIDENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot = engine.observe()

    print(
        "Fused element count:",
        len(snapshot.fused_elements),
    )

    if snapshot.warnings:
        print("Warnings:")
        for warning in snapshot.warnings:
            print(f"  {warning}")
    else:
        print("Warnings: none")

    from computer_agent.grounding import UIGrounder

    grounder = UIGrounder()
    results = {}

    for name, target in _build_targets():
        result = grounder.ground(
            target,
            snapshot.fused_elements,
        )
        results[name] = result

        _print_result(
            name,
            target,
            result,
        )

    from computer_agent.grounding import GroundingStatus

    failures = []

    expected_resolved_types = {
        "Docs link": "link",
        "Docs heading": "heading",
        "Search field": "text_field",
    }

    for name, expected_type in expected_resolved_types.items():
        result = results[name]

        if result.status is not GroundingStatus.RESOLVED:
            failures.append(
                f"{name} was not resolved: {result.status.value}"
            )
            continue

        if result.element is None:
            failures.append(
                f"{name} resolved without an element"
            )
            continue

        if result.element.element_type != expected_type:
            failures.append(
                f"{name} resolved to {result.element.element_type!r} "
                f"instead of {expected_type!r}"
            )

        eligible = tuple(
            candidate
            for candidate in result.candidates
            if candidate.eligible
        )

        if len(eligible) != 1:
            failures.append(
                f"{name} did not have exactly one eligible candidate"
            )

    docs_link_roles = {
        candidate.element.element_type
        for candidate in results["Docs link"].candidates
    }

    if not {"link", "heading", "text"}.issubset(docs_link_roles):
        failures.append(
            "Docs collision did not include link, heading, and text roles"
        )

    search_roles = {
        candidate.element.element_type
        for candidate in results["Search field"].candidates
    }

    if not {"text_field", "text"}.issubset(search_roles):
        failures.append(
            "Search field collision did not include text_field and text roles"
        )

    missing_result = results["Missing link"]

    if missing_result.status is not GroundingStatus.NOT_FOUND:
        failures.append(
            "missing target did not return not_found"
        )

    if missing_result.element is not None:
        failures.append(
            "missing target returned an element"
        )

    if missing_result.candidates:
        failures.append(
            "missing target returned unexpected candidates"
        )

    frontmost_app = (
        accessibility.read_frontmost_application_name()
    )

    if frontmost_app != "Google Chrome":
        failures.append(
            "frontmost application was not Google Chrome"
        )

    if snapshot.warnings:
        failures.append(
            "perception source warning occurred"
        )

    if failures:
        print()
        print("Live acceptance failed:")

        for failure in failures:
            print(f"  {failure}")

        return 1

    print()
    print("Live acceptance result: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
