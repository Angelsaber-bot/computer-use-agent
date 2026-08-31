"""Phase 04 experiment for deterministic action verification."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

from computer_agent.grounding import ActionGrounder, ActionGroundingStatus
from computer_agent.grounding import GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.verification import ActionVerificationStatus, ActionVerifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "assets/fixtures/phase04_ui_grounding_task_reasoning"
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase04_ui_grounding_task_reasoning"
FIXTURE_PATH = FIXTURE_DIR / "experiment_04_action_verification.html"
BEFORE_CAPTURE_PATH = SCREENSHOT_DIR / "experiment_04_action_verification_before.png"
CANDIDATE_EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_04_action_verification_candidate.png"
EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_04_action_verification.png"

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 11
ACTION_CONFIDENCE = 0.70
DEFAULT_POST_ACTION_WAIT_SECONDS = 0.25

FIXTURE_MARKER = "PHASE04_ACTION_VERIFICATION_FIXTURE_04"
ACTION_TARGET_TEXT = "ACTION_TARGET_04"
ACTION_TARGET_IDENTIFIER = "verification-action-target-04"
VERIFICATION_TARGET_TEXT = "VERIFICATION_TARGET_04"
CLICK_TOOL_NAME = "click_mouse"
ACTION_TARGET_SPEC = TargetSpec(
    text=ACTION_TARGET_TEXT, identifier=ACTION_TARGET_IDENTIFIER,
    element_types=("button",), minimum_confidence=ACTION_CONFIDENCE)
VERIFICATION_TARGET_SPEC = TargetSpec(text=VERIFICATION_TARGET_TEXT, minimum_confidence=ACTION_CONFIDENCE)


def _wait_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--wait-seconds must be an integer") from error

    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError("--wait-seconds must be from 0 through 30")

    return seconds


def _post_action_wait_seconds(value: str) -> float:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Observe the focused Experiment 04 fixture and verify one click action."
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute the generated click action. Default is dry-run.",
    )
    parser.add_argument(
        "--wait-seconds", type=_wait_seconds, default=5,
        help="Seconds to wait before observation, from 0 through 30.",
    )
    parser.add_argument(
        "--post-action-wait-seconds", type=_post_action_wait_seconds,
        default=DEFAULT_POST_ACTION_WAIT_SECONDS,
        help="Seconds to wait after execution, from 0.0 through 5.0.",
    )
    return parser.parse_args()


def _print_manual_focus_instructions(execute: bool) -> None:
    print("Phase 04 Experiment 04: Deterministic Action Verification")
    print(f"Fixture path: {FIXTURE_PATH}")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before observation.")
    if execute:
        print("Execute mode: one generated click_mouse Action will be executed.")
    else:
        print("Dry-run mode: one observation only; tool execution is skipped.")


def _wait_for_focus(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        time.sleep(1)


def _build_engine(capture_path: str | Path = BEFORE_CAPTURE_PATH):
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import MacOSAccessibility, PerceptionEngine
    from computer_agent.perception import ScreenCapture, TesseractOCR, UIElementFusion

    controller = ComputerController()
    return PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=MacOSAccessibility(),
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )


def _build_executor():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.tools.computer import create_computer_tools
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    controller = ComputerController()
    registry = ToolRegistry(create_computer_tools(controller))
    return ToolExecutor(registry)


def _raw_observation_texts(snapshot) -> tuple[str, ...]:
    return tuple(
        element.text
        for collection in (snapshot.accessibility_elements, snapshot.ocr_elements)
        for element in collection
        if element.text is not None and element.text.strip()
    )


def _fixture_identity_observed(snapshot, marker: str = FIXTURE_MARKER) -> bool:
    expected_marker = normalize_ui_text(marker)
    return any(
        expected_marker in normalize_ui_text(text)
        for text in _raw_observation_texts(snapshot)
    ) if expected_marker else False


def _ground_before_snapshot(snapshot):
    ui_grounder = UIGrounder()
    grounding_result = ui_grounder.ground(ACTION_TARGET_SPEC, snapshot.fused_elements)
    action_result = ActionGrounder().ground_click(
        grounding_result, snapshot.frame.screen_size
    )
    before_target = ui_grounder.ground(VERIFICATION_TARGET_SPEC, snapshot.fused_elements)
    return grounding_result, action_result, before_target


def _print_snapshot(label: str, snapshot, capture_path: Path) -> None:
    counts = snapshot.source_counts
    print(f"{label} screenshot path: {capture_path}")
    print(f"{label} pixel size: {snapshot.frame.pixel_size}")
    print(f"{label} screen size: {snapshot.frame.screen_size}")
    print(
        f"{label} counts: accessibility={counts['accessibility']}, "
        f"ocr={counts['ocr']}, fused={counts['fused']}"
    )
    print(f"{label} fixture marker observed: {_fixture_identity_observed(snapshot)}")
    print(f"{label} warnings: {snapshot.warnings or 'none'}")


def _print_grounding(grounding_result, action_result, before_target) -> None:
    print(f"Action target grounding status: {grounding_result.status.value}")
    print(f"Action target grounding reason: {grounding_result.reason}")
    print(f"Action grounding status: {action_result.status.value}")
    print(f"Action grounding reason: {action_result.reason}")
    print(f"Before verification target status: {before_target.status.value}")
    print(f"Before verification target reason: {before_target.reason}")
    if action_result.action is not None:
        print(f"Generated Action tool name: {action_result.action.tool_name}")
        print(f"Generated Action arguments: {action_result.action.arguments}")


def _before_failures(grounding_result, action_result, before_target) -> list[str]:
    failures = []
    if grounding_result.status is not GroundingStatus.RESOLVED:
        failures.append(
            f"action target grounding was {grounding_result.status.value}"
        )
    elif grounding_result.element.identifier != ACTION_TARGET_IDENTIFIER:
        failures.append("action target did not resolve to the expected identifier")
    elif not all(c.match_basis == "identifier" for c in grounding_result.candidates):
        failures.append("action target did not resolve through identifier evidence")

    if action_result.status is not ActionGroundingStatus.READY:
        failures.append(f"action grounding was {action_result.status.value}")
    elif action_result.action.tool_name != CLICK_TOOL_NAME:
        failures.append(f"generated Action tool was {action_result.action.tool_name}")
    elif set(action_result.action.arguments) != {"x", "y"}:
        failures.append("generated Action arguments were not x/y coordinates")

    if before_target.status is not GroundingStatus.NOT_FOUND:
        failures.append(f"before verification target was {before_target.status.value}")

    return failures


def _execute_failures(
    *,
    before_snapshot,
    after_snapshot,
    before_capture_path: Path,
    candidate_evidence_path: Path,
    tool_result,
    verification_result,
) -> list[str]:
    return [
        message
        for passed, message in (
            (
                Path(before_snapshot.frame.image_path).resolve()
                == before_capture_path.resolve(),
                "before snapshot did not use the before capture path",
            ),
            (
                Path(after_snapshot.frame.image_path).resolve()
                == candidate_evidence_path.resolve(),
                "after snapshot did not use the candidate evidence path",
            ),
            (candidate_evidence_path.is_file(), "after candidate screenshot was not created"),
            (tool_result.success, f"tool result failed: {tool_result.error}"),
            (_fixture_identity_observed(after_snapshot), "after marker was not observed"),
            (
                verification_result.status is ActionVerificationStatus.VERIFIED,
                f"verification status was {verification_result.status.value}",
            ),
        )
        if not passed
    ]


def _print_failures(failures: list[str], candidate_evidence_path: Path | None) -> None:
    print("Live acceptance failed:")
    for failure in failures:
        print(f"  {failure}")
    if candidate_evidence_path is not None:
        print(f"Candidate screenshot retained for debugging: {candidate_evidence_path}")


def _promote_candidate_evidence(
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
) -> None:
    formal_path = Path(formal_evidence_path)
    formal_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_evidence_path, formal_path)


def _run_acceptance(
    *,
    execute: bool,
    post_action_wait_seconds: float = DEFAULT_POST_ACTION_WAIT_SECONDS,
    before_capture_path: str | Path = BEFORE_CAPTURE_PATH,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
    observer_builder=_build_engine,
    executor_builder=_build_executor,
    sleeper=time.sleep,
) -> int:
    before_path = Path(before_capture_path)
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)

    try:
        before_path.parent.mkdir(parents=True, exist_ok=True)
        before_snapshot = observer_builder(before_path).observe()
    except (OSError, RuntimeError) as error:
        print(f"Before observation failed: {type(error).__name__}: {error}")
        return 1

    _print_snapshot("Before", before_snapshot, before_path)
    if not _fixture_identity_observed(before_snapshot):
        _print_failures(["before fixture marker was not observed"], None)
        print("Execution skipped: fixture identity was not established.")
        return 1

    grounding_result, action_result, before_target = _ground_before_snapshot(before_snapshot)
    _print_grounding(grounding_result, action_result, before_target)
    failures = _before_failures(grounding_result, action_result, before_target)
    if failures:
        _print_failures(failures, None)
        print("Execution skipped: action preconditions were not satisfied.")
        return 1

    if not execute:
        print("Live acceptance result: passed")
        print("Execution skipped: dry-run mode.")
        print("Observation count: 1")
        print("Action execution count: 0")
        print("Evidence promotion: skipped")
        return 0

    action = action_result.action
    try:
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        executor = executor_builder()
        tool_result = executor.execute(action)
        print("Execution performed: one Action executed.")
        if post_action_wait_seconds:
            sleeper(post_action_wait_seconds)
    except (OSError, RuntimeError) as error:
        print(f"Execution failed: {type(error).__name__}: {error}")
        return 1

    print(f"Tool result success: {tool_result.success}")
    print(f"Tool result tool name: {tool_result.tool_name}")
    print(f"Tool result error: {tool_result.error}")

    try:
        after_snapshot = observer_builder(candidate_path).observe()
    except (OSError, RuntimeError) as error:
        print(f"After observation failed: {type(error).__name__}: {error}")
        print(f"Candidate screenshot retained when available: {candidate_path}")
        return 1

    _print_snapshot("After", after_snapshot, candidate_path)
    if not _fixture_identity_observed(after_snapshot):
        _print_failures(["after fixture marker was not observed"], candidate_path)
        return 1

    verification_result = ActionVerifier().verify_target_appeared(
        action=action,
        tool_result=tool_result,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        target_spec=VERIFICATION_TARGET_SPEC,
    )
    print(f"Verification status: {verification_result.status.value}")
    print(f"Verification reason: {verification_result.reason}")
    print(f"Verification before status: {verification_result.before_grounding.status.value}")
    print(f"Verification after status: {verification_result.after_grounding.status.value}")

    failures = _execute_failures(
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        before_capture_path=before_path,
        candidate_evidence_path=candidate_path,
        tool_result=tool_result,
        verification_result=verification_result,
    )
    if failures:
        _print_failures(failures, candidate_path)
        return 1

    try:
        _promote_candidate_evidence(
            candidate_evidence_path=candidate_path,
            formal_evidence_path=formal_path,
        )
    except (OSError, RuntimeError) as error:
        print(f"Evidence promotion failed: {type(error).__name__}: {error}")
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        return 1

    print(f"Formal evidence updated from candidate: {formal_path}")
    print("Live acceptance result: passed")
    print("Observation count: 2")
    print("Action execution count: 1")
    print("Evidence promotion: completed")
    return 0


def main() -> int:
    args = _parse_args()
    if sys.platform != "darwin":
        print("Live acceptance failed: this experiment runs only on macOS.")
        return 1
    if not FIXTURE_PATH.is_file():
        print(f"Live acceptance failed: fixture file was not found: {FIXTURE_PATH}")
        return 1

    _print_manual_focus_instructions(args.execute)
    _wait_for_focus(args.wait_seconds)
    return _run_acceptance(execute=args.execute, post_action_wait_seconds=args.post_action_wait_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
