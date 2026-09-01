"""Shared live harness utilities for Phase 04 experiments."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

from computer_agent.perception.fusion import normalize_ui_text


DEFAULT_OCR_MINIMUM_CONFIDENCE = 0.05
DEFAULT_OCR_PAGE_SEGMENTATION_MODE = 11
DEFAULT_DRY_RUN_MESSAGE = (
    "Dry-run mode: one observation only; tool execution is skipped."
)


def wait_seconds(value: str) -> int:
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


def post_action_wait_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--post-action-wait-seconds must be numeric"
        ) from error
    if not math.isfinite(seconds) or not 0.0 <= seconds <= 5.0:
        raise argparse.ArgumentTypeError(
            "--post-action-wait-seconds must be from 0.0 through 5.0"
        )
    return seconds


def parse_execute_harness_args(
    *,
    description: str,
    execute_help: str,
    post_action_wait_default: float,
    post_action_wait_help: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=execute_help,
    )
    parser.add_argument(
        "--wait-seconds",
        type=wait_seconds,
        default=5,
        help="Seconds to wait before observation, from 0 through 30.",
    )
    parser.add_argument(
        "--post-action-wait-seconds",
        type=post_action_wait_seconds,
        default=post_action_wait_default,
        help=post_action_wait_help,
    )
    return parser.parse_args()


def print_manual_focus_instructions(
    *,
    title: str,
    fixture_path: str | Path,
    execute: bool,
    execute_message: str,
    dry_run_message: str = DEFAULT_DRY_RUN_MESSAGE,
) -> None:
    print(title)
    print(f"Fixture path: {fixture_path}")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before observation.")
    print(execute_message if execute else dry_run_message)


def wait_for_focus(seconds: int, *, sleeper=time.sleep) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        sleeper(1)


def build_live_perception_engine(
    capture_path: str | Path,
    *,
    ocr_minimum_confidence: float = DEFAULT_OCR_MINIMUM_CONFIDENCE,
    ocr_page_segmentation_mode: int = DEFAULT_OCR_PAGE_SEGMENTATION_MODE,
):
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import MacOSAccessibility, PerceptionEngine
    from computer_agent.perception import ScreenCapture, TesseractOCR
    from computer_agent.perception import UIElementFusion

    controller = ComputerController()
    return PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=MacOSAccessibility(),
        ocr=TesseractOCR(
            minimum_confidence=ocr_minimum_confidence,
            page_segmentation_mode=ocr_page_segmentation_mode,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )


def build_live_tool_executor():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.tools.computer import create_computer_tools
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    controller = ComputerController()
    return ToolExecutor(ToolRegistry(create_computer_tools(controller)))


def live_prerequisites_available(fixture_path: str | Path) -> bool:
    path = Path(fixture_path)
    if sys.platform != "darwin":
        print("Live acceptance failed: this experiment runs only on macOS.")
        return False
    if not path.is_file():
        print(f"Live acceptance failed: fixture file was not found: {path}")
        return False
    return True


def raw_observation_texts(snapshot) -> tuple[str, ...]:
    return tuple(
        element.text
        for collection in (snapshot.accessibility_elements, snapshot.ocr_elements)
        for element in collection
        if element.text is not None and element.text.strip()
    )


def fixture_identity_observed(snapshot, marker: str) -> bool:
    expected_marker = normalize_ui_text(marker)
    if not expected_marker:
        return False
    return any(
        expected_marker in normalize_ui_text(text)
        for text in raw_observation_texts(snapshot)
    )


def print_snapshot_summary(
    label: str,
    snapshot,
    capture_path: str | Path,
    *,
    fixture_marker: str,
) -> None:
    counts = snapshot.source_counts
    print(f"{label} screenshot path: {capture_path}")
    print(f"{label} pixel size: {snapshot.frame.pixel_size}")
    print(f"{label} screen size: {snapshot.frame.screen_size}")
    print(
        f"{label} counts: accessibility={counts['accessibility']}, "
        f"ocr={counts['ocr']}, fused={counts['fused']}"
    )
    print(
        f"{label} fixture marker observed: "
        f"{fixture_identity_observed(snapshot, fixture_marker)}"
    )
    print(f"{label} warnings: {snapshot.warnings or 'none'}")


def print_verification_summary(label: str, result) -> None:
    print(f"{label} verification status: {result.status.value}")
    print(f"{label} verification reason: {result.reason}")
    print(
        f"{label} verification before status: "
        f"{result.before_grounding.status.value}"
    )
    print(
        f"{label} verification after status: "
        f"{result.after_grounding.status.value}"
    )


def print_tool_result(label: str, result) -> None:
    print(f"{label} ToolResult success: {result.success}")
    print(f"{label} ToolResult error: {result.error}")


def print_recovery_summary(result) -> None:
    print(f"Recovery status: {result.status.value}")
    print(f"Recovery reason: {result.reason}")
    if result.grounding_result is not None:
        print(f"Recovery grounding status: {result.grounding_result.status.value}")
    if result.action_grounding_result is not None:
        print(
            "Recovery action grounding status: "
            f"{result.action_grounding_result.status.value}"
        )


def failed_condition_messages(
    *conditions: tuple[bool, str],
) -> list[str]:
    return [
        message
        for passed, message in conditions
        if not passed
    ]


def print_failures(failures: list[str], retained_paths: tuple[Path, ...]) -> None:
    print("Live acceptance failed:")
    for failure in failures:
        print(f"  {failure}")
    for path in retained_paths:
        if path.exists():
            print(f"Candidate screenshot retained for debugging: {path}")


def promote_candidate_evidence(
    *,
    candidate_evidence_path: str | Path,
    formal_evidence_path: str | Path,
) -> None:
    formal_path = Path(formal_evidence_path)
    formal_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_evidence_path, formal_path)


def observe_with_capture_path(observer_builder, capture_path: str | Path):
    path = Path(capture_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return observer_builder(path).observe()


def snapshot_path_matches(snapshot, expected_path: str | Path) -> bool:
    return Path(snapshot.frame.image_path).resolve() == Path(expected_path).resolve()


def snapshot_path_failure(
    label: str,
    snapshot,
    expected_path: str | Path,
) -> str:
    return (
        f"{label} snapshot path was {Path(snapshot.frame.image_path).resolve()}; "
        f"expected {Path(expected_path).resolve()}"
    )


def snapshot_path_failures(
    *observations: tuple[str, object, str | Path],
) -> list[str]:
    return [
        snapshot_path_failure(label, snapshot, expected_path)
        for label, snapshot, expected_path in observations
        if not snapshot_path_matches(snapshot, expected_path)
    ]


def snapshot_path_mismatched(
    label: str,
    snapshot,
    expected_path: str | Path,
    retained_paths: tuple[Path, ...],
) -> bool:
    failures = snapshot_path_failures((label, snapshot, expected_path))
    if not failures:
        return False
    print_failures(failures, retained_paths)
    return True
