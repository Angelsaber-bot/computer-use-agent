"""Phase 04 experiment for observation-only click action grounding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time

from computer_agent.grounding import ActionGrounder, ActionGroundingStatus
from computer_agent.grounding import GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception.fusion import normalize_ui_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "assets/fixtures/phase04_ui_grounding_task_reasoning"
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase04_ui_grounding_task_reasoning"
FIXTURE_PATH = FIXTURE_DIR / "experiment_02_ui_grounding.html"
EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_03_action_grounding.png"
CANDIDATE_EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_03_action_grounding_candidate.png"

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 11
ACTION_CONFIDENCE = 0.70

FIXTURE_MARKER = "PHASE04_UI_GROUNDING_FIXTURE_02"
IDENTIFIER_TEXT = "IDENTIFIER_TARGET_02"
IDENTIFIER_TARGET_ID = "identifier-target-02"
DISABLED_ONLY_TEXT = "DISABLED_ONLY_02"


@dataclass(frozen=True, slots=True)
class ActionGroundingCase:
    """One live click-action grounding case."""

    name: str
    target_spec: TargetSpec
    expected_grounding_status: GroundingStatus
    expected_action_status: ActionGroundingStatus
    expected_identifier: str | None = None


def _wait_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--wait-seconds must be an integer") from error

    if not 0 <= seconds <= 30:
        raise argparse.ArgumentTypeError("--wait-seconds must be from 0 through 30")
    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Observe the already-focused Phase 04 Experiment 02 fixture "
            "and construct deterministic click actions without executing them."
        )
    )
    parser.add_argument("--wait-seconds", type=_wait_seconds, default=5)
    return parser.parse_args()


def _print_manual_focus_instructions() -> None:
    print("Phase 04 Experiment 03: Deterministic Click Action Grounding")
    print(f"Fixture path: {FIXTURE_PATH}")
    print("Open the reused Experiment 02 fixture in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before observation.")
    print(
        "This experiment observes once and will not click, move, type, "
        "switch applications, or execute tools."
    )


def _wait_for_focus(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        time.sleep(1)


def _fixture_identity_observed(
    snapshot,
    marker: str = FIXTURE_MARKER,
) -> bool:
    expected_marker = normalize_ui_text(marker)
    for elements in (snapshot.accessibility_elements, snapshot.ocr_elements):
        for element in elements:
            if element.text and expected_marker in normalize_ui_text(element.text):
                return True
    return False


def _build_engine(capture_path: str | Path = CANDIDATE_EVIDENCE_PATH):
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import MacOSAccessibility, PerceptionEngine
    from computer_agent.perception import ScreenCapture, TesseractOCR, UIElementFusion

    return PerceptionEngine(
        screen_capture=ScreenCapture(ComputerController()),
        accessibility_reader=MacOSAccessibility(),
        ocr=TesseractOCR(
            minimum_confidence=OCR_MINIMUM_CONFIDENCE,
            page_segmentation_mode=OCR_PAGE_SEGMENTATION_MODE,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )


def _build_cases() -> tuple[ActionGroundingCase, ...]:
    return (
        ActionGroundingCase(
            name=IDENTIFIER_TEXT,
            target_spec=TargetSpec(
                text=IDENTIFIER_TEXT,
                identifier=IDENTIFIER_TARGET_ID,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
            expected_grounding_status=GroundingStatus.RESOLVED,
            expected_action_status=ActionGroundingStatus.READY,
            expected_identifier=IDENTIFIER_TARGET_ID,
        ),
        ActionGroundingCase(
            name=DISABLED_ONLY_TEXT,
            target_spec=TargetSpec(
                text=DISABLED_ONLY_TEXT,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
            expected_grounding_status=GroundingStatus.UNSAFE,
            expected_action_status=ActionGroundingStatus.BLOCKED,
            expected_identifier=None,
        ),
    )


def _run_action_cases(snapshot) -> tuple[tuple[ActionGroundingCase, object, object], ...]:
    ui_grounder = UIGrounder()
    action_grounder = ActionGrounder()
    results = []
    for case in _build_cases():
        grounding_result = ui_grounder.ground(case.target_spec, snapshot.fused_elements)
        action_result = action_grounder.ground_click(
            grounding_result, snapshot.frame.screen_size
        )
        _print_case_result(case, grounding_result, action_result)
        results.append((case, grounding_result, action_result))
    return tuple(results)


def _print_snapshot_evidence(
    snapshot,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
) -> None:
    counts = snapshot.source_counts
    print(f"Candidate screenshot path: {candidate_evidence_path}")
    print(f"Formal evidence path: {formal_evidence_path}")
    print(f"Screenshot pixel size: {snapshot.frame.pixel_size}")
    print(f"Logical screen size: {snapshot.frame.screen_size}")
    print(f"Accessibility element count: {counts['accessibility']}")
    print(f"Logical OCR element count: {counts['ocr']}")
    print(f"Fused element count: {counts['fused']}")
    print(f"Warnings: {snapshot.warnings or 'none'}")
    print(f"Fixture marker observed: {_fixture_identity_observed(snapshot)}")


def _print_case_result(case, grounding_result, action_result) -> None:
    print(f"Case: {case.name}")
    print(
        "  grounding status: "
        f"expected={case.expected_grounding_status.value}, "
        f"actual={grounding_result.status.value}"
    )
    print(f"  grounding reason: {grounding_result.reason}")
    print(
        "  action-grounding status: "
        f"expected={case.expected_action_status.value}, "
        f"actual={action_result.status.value}"
    )
    print(f"  action-grounding reason: {action_result.reason}")
    if grounding_result.element is not None:
        box = grounding_result.element.bounding_box
        print(f"  resolved box: x={box.x}, y={box.y}, width={box.width}, height={box.height}")
    if action_result.action is not None:
        action = action_result.action
        print(f"  generated tool name: {action.tool_name}")
        print(f"  generated arguments: {action.arguments}")
        print(f"  generated Action reason: {action.reason}")


def _acceptance_failures(
    snapshot,
    results,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
) -> list[str]:
    failures = []
    candidate_path = Path(candidate_evidence_path)
    if Path(snapshot.frame.image_path).resolve() != candidate_path.resolve():
        failures.append("snapshot did not use the candidate evidence path")
    if not candidate_path.is_file():
        failures.append("candidate screenshot was not created")
    if len(results) != 2:
        failures.append(f"expected 2 cases, got {len(results)}")

    for case, grounding_result, action_result in results:
        if grounding_result.status is not case.expected_grounding_status:
            failures.append(f"{case.name}: unexpected grounding status")
        if action_result.status is not case.expected_action_status:
            failures.append(f"{case.name}: unexpected action-grounding status")
        if case.name == IDENTIFIER_TEXT:
            failures.extend(
                _resolved_action_failures(case, grounding_result, action_result)
            )
        elif action_result.action is not None:
            failures.append(f"{case.name}: blocked result returned an Action")
    return failures


def _resolved_action_failures(case, grounding_result, action_result) -> list[str]:
    if grounding_result.element is None:
        return [f"{case.name}: missing resolved element"]
    if grounding_result.element.identifier != case.expected_identifier:
        return [f"{case.name}: unexpected identifier"]
    action = action_result.action
    if action is None:
        return [f"{case.name}: missing Action"]
    if action.tool_name != "click_mouse" or set(action.arguments) != {"x", "y"}:
        return [f"{case.name}: invalid click Action"]

    x = action.arguments["x"]
    y = action.arguments["y"]
    if type(x) is not int or type(y) is not int:
        return [f"{case.name}: click coordinates were not integers"]
    if not grounding_result.element.bounding_box.contains_point(x, y):
        return [f"{case.name}: click point was outside the resolved box"]
    return []


def _promote_candidate_evidence(
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
) -> None:
    formal_path = Path(formal_evidence_path)
    formal_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(candidate_evidence_path, formal_path)


def _evaluate_observed_snapshot(
    snapshot,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
    acceptance_checker=None,
) -> int:
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)
    _print_snapshot_evidence(
        snapshot, candidate_evidence_path=candidate_path, formal_evidence_path=formal_path
    )
    if not _fixture_identity_observed(snapshot):
        print("Environment error: expected fixture marker was not observed.")
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        print("Grounding and action-grounding cases were not run.")
        return 1

    results = _run_action_cases(snapshot)
    failures = (acceptance_checker or _acceptance_failures)(
        snapshot, results, candidate_evidence_path=candidate_path
    )
    if failures:
        print("Live acceptance failed:")
        for failure in failures:
            print(f"  {failure}")
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        return 1

    try:
        _promote_candidate_evidence(
            candidate_evidence_path=candidate_path,
            formal_evidence_path=formal_path,
        )
    except OSError as error:
        print(f"Evidence promotion failed: {type(error).__name__}: {error}")
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        return 1

    print(f"Formal evidence updated from candidate: {formal_path}")
    print("Live acceptance result: passed")
    print("Execution skipped: no generated Action was executed.")
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
    try:
        CANDIDATE_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot = _build_engine().observe()
    except (OSError, RuntimeError) as error:
        print(f"Observation failed: {type(error).__name__}: {error}")
        return 1

    return _evaluate_observed_snapshot(snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
