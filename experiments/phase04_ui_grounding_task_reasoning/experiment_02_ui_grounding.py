"""Phase 04 experiment for observation-only deterministic UI grounding."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT
    / "assets/fixtures/phase04_ui_grounding_task_reasoning"
    / "experiment_02_ui_grounding.html"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase04_ui_grounding_task_reasoning"
    / "experiment_02_ui_grounding.png"
)
CANDIDATE_EVIDENCE_PATH = EVIDENCE_PATH.with_name(
    "experiment_02_ui_grounding_candidate.png"
)

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 11
ACTION_CONFIDENCE = 0.70

FIXTURE_MARKER = "PHASE04_UI_GROUNDING_FIXTURE_02"
IDENTIFIER_TEXT = "IDENTIFIER_TARGET_02"
IDENTIFIER_TARGET_ID = "identifier-target-02"
ROLE_TEXT = "ROLE_TARGET_02"
ROLE_TARGET_ID = "role-button-target-02"
DISABLED_ONLY_TEXT = "DISABLED_ONLY_02"
BLOCKED_IDENTIFIER_TEXT = "BLOCKED_IDENTIFIER_02"
BLOCKED_IDENTIFIER_ID = "blocked-identifier-target-02"
AMBIGUOUS_TEXT = "AMBIGUOUS_TARGET_02"
OCR_ONLY_TEXT = "OCR_ONLY_TARGET_02"
MISSING_TEXT = "MISSING_TARGET_02"

SOURCE_PRIORITIES = {
    "hybrid": 0,
    "accessibility": 1,
    "ocr": 2,
}


@dataclass(frozen=True, slots=True)
class GroundingCase:
    """One live grounding acceptance case."""

    name: str
    description: str
    expected_status: Any
    target_spec: Any
    expected_identifier: str | None = None


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
            "Observe the already-focused Phase 04 Experiment 02 fixture "
            "and ground deterministic UI targets without interacting with "
            "the computer."
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
    print("Phase 04 Experiment 02: Deterministic UI Grounding")
    print(f"Fixture path: {FIXTURE_PATH}")
    print("Open the fixture manually in Google Chrome or Safari.")
    print("Keep the fixture window visible and focused before observation.")
    print(
        "This experiment observes once and will not move, click, type, "
        "switch applications, or execute tools."
    )


def _wait_for_focus(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"Observing in {remaining}...")
        time.sleep(1)


def _normalized_text(text: str | None) -> str:
    from computer_agent.perception.fusion import normalize_ui_text

    return normalize_ui_text(text)


def _raw_observation_texts(snapshot) -> tuple[str, ...]:
    texts = []

    for collection in (
        snapshot.accessibility_elements,
        snapshot.ocr_elements,
    ):
        for element in collection:
            if element.text is not None and element.text.strip():
                texts.append(element.text)

    return tuple(texts)


def _fixture_identity_observed(
    snapshot,
    marker: str = FIXTURE_MARKER,
) -> bool:
    expected_marker = _normalized_text(marker)
    if not expected_marker:
        return False

    return any(
        expected_marker in _normalized_text(text)
        for text in _raw_observation_texts(snapshot)
    )


def _print_fixture_identity_error(
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
) -> None:
    print(
        "Environment error: expected Phase 04 Experiment 02 fixture "
        "was not observed."
    )
    print(f"Expected visible fixture marker: {FIXTURE_MARKER}")
    print(
        "The wrong window may be focused; open the fixture manually and "
        "focus that window before rerunning."
    )
    print(f"Candidate screenshot retained for debugging: {candidate_evidence_path}")
    print("Grounding cases were not run.")


def _build_engine(
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
):
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
        capture_path=capture_path,
    )


def _build_cases() -> tuple[GroundingCase, ...]:
    from computer_agent.grounding import GroundingStatus, TargetSpec

    return (
        GroundingCase(
            name=IDENTIFIER_TEXT,
            description="exact identifier target",
            expected_status=GroundingStatus.RESOLVED,
            target_spec=TargetSpec(
                text=IDENTIFIER_TEXT,
                identifier=IDENTIFIER_TARGET_ID,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
            expected_identifier=IDENTIFIER_TARGET_ID,
        ),
        GroundingCase(
            name=ROLE_TEXT,
            description="matching text appears on different roles",
            expected_status=GroundingStatus.RESOLVED,
            target_spec=TargetSpec(
                text=ROLE_TEXT,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
            expected_identifier=ROLE_TARGET_ID,
        ),
        GroundingCase(
            name=DISABLED_ONLY_TEXT,
            description="only matching target is disabled",
            expected_status=GroundingStatus.UNSAFE,
            target_spec=TargetSpec(
                text=DISABLED_ONLY_TEXT,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
        ),
        GroundingCase(
            name=BLOCKED_IDENTIFIER_TEXT,
            description=(
                "requested identifier belongs to a disabled control; "
                "text fallback must not occur"
            ),
            expected_status=GroundingStatus.UNSAFE,
            target_spec=TargetSpec(
                text=BLOCKED_IDENTIFIER_TEXT,
                identifier=BLOCKED_IDENTIFIER_ID,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
        ),
        GroundingCase(
            name=AMBIGUOUS_TEXT,
            description="two equally eligible Accessibility controls",
            expected_status=GroundingStatus.AMBIGUOUS,
            target_spec=TargetSpec(
                text=AMBIGUOUS_TEXT,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
        ),
        GroundingCase(
            name=OCR_ONLY_TEXT,
            description="large high-contrast canvas text exposed only by OCR",
            expected_status=GroundingStatus.RESOLVED,
            target_spec=TargetSpec(
                text=OCR_ONLY_TEXT,
                element_types=("text",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
        ),
        GroundingCase(
            name=MISSING_TEXT,
            description="target absent from the fixture",
            expected_status=GroundingStatus.NOT_FOUND,
            target_spec=TargetSpec(
                text=MISSING_TEXT,
                element_types=("button",),
                minimum_confidence=ACTION_CONFIDENCE,
            ),
        ),
    )


def _source_distribution(snapshot) -> dict[str, int]:
    counts = Counter(
        element.source or "unspecified"
        for element in snapshot.fused_elements
    )

    return dict(sorted(counts.items()))


def _status_value(status: Any) -> str:
    return getattr(status, "value", str(status))


def _format_distance(distance: float | None) -> str:
    if distance is None:
        return "None"

    return f"{distance:.2f}"


def _format_box(element) -> str:
    box = element.bounding_box
    return (
        "("
        f"x={box.x}, "
        f"y={box.y}, "
        f"width={box.width}, "
        f"height={box.height}"
        ")"
    )


def _print_element(prefix: str, element) -> None:
    print(f"{prefix} role: {element.element_type!r}")
    print(f"{prefix} source: {element.source!r}")
    print(f"{prefix} text: {element.text!r}")
    print(f"{prefix} identifier: {element.identifier!r}")
    print(f"{prefix} value: {element.value!r}")
    print(f"{prefix} enabled: {element.enabled!r}")
    print(f"{prefix} focused: {element.focused!r}")
    print(f"{prefix} selected: {element.selected!r}")
    print(f"{prefix} confidence: {element.confidence:.2f}")
    print(f"{prefix} bounding box: {_format_box(element)}")


def _print_case_result(case: GroundingCase, result) -> None:
    print(f"Case: {case.name}")
    print(f"  description: {case.description}")
    print(f"  TargetSpec: {case.target_spec!r}")
    print(f"  expected status: {_status_value(case.expected_status)}")
    print(f"  actual status: {_status_value(result.status)}")
    print(f"  reason: {result.reason}")

    if result.element is None:
        print("  resolved element: None")
    else:
        _print_element("  resolved element", result.element)

    print(f"  candidate count: {len(result.candidates)}")
    for index, candidate in enumerate(result.candidates, start=1):
        element = candidate.element
        print(f"  candidate {index}:")
        print(f"    match basis: {candidate.match_basis!r}")
        print(f"    source: {element.source!r}")
        print(f"    role: {element.element_type!r}")
        print(f"    text: {element.text!r}")
        print(f"    identifier: {element.identifier!r}")
        print(f"    value: {element.value!r}")
        print(f"    enabled: {element.enabled!r}")
        print(f"    focused: {element.focused!r}")
        print(f"    selected: {element.selected!r}")
        print(f"    confidence: {element.confidence:.2f}")
        print(f"    distance: {_format_distance(candidate.distance)}")
        print(f"    bounding box: {_format_box(element)}")
        print(f"    eligible: {candidate.eligible}")
        print(f"    rejection reasons: {candidate.rejection_reasons!r}")


def _run_grounding_cases(snapshot) -> tuple[tuple[GroundingCase, Any], ...]:
    from computer_agent.grounding import UIGrounder

    grounder = UIGrounder()
    results = []

    for case in _build_cases():
        result = grounder.ground(
            case.target_spec,
            snapshot.fused_elements,
        )
        _print_case_result(case, result)
        results.append((case, result))

    return tuple(results)


def _normalized_source(source: str | None) -> str | None:
    if source is None:
        return None

    source_name = source.strip()
    if not source_name:
        return None

    return source_name.casefold()


def _source_priority(source: str | None) -> int:
    source_name = _normalized_source(source)
    if source_name is None:
        return 4

    return SOURCE_PRIORITIES.get(source_name, 3)


def _best_eligible_candidates(result) -> tuple[Any, ...]:
    candidates = tuple(
        candidate
        for candidate in result.candidates
        if candidate.eligible
    )
    if not candidates:
        return ()

    source_priority = min(
        _source_priority(candidate.element.source)
        for candidate in candidates
    )
    candidates = tuple(
        candidate
        for candidate in candidates
        if _source_priority(candidate.element.source) == source_priority
    )

    distances = tuple(
        candidate.distance
        for candidate in candidates
        if candidate.distance is not None
    )
    if distances:
        nearest = min(distances)
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.distance == nearest
        )

    confidence = max(
        candidate.element.confidence
        for candidate in candidates
    )
    return tuple(
        candidate
        for candidate in candidates
        if candidate.element.confidence == confidence
    )


def _case_result(results, name: str):
    for case, result in results:
        if case.name == name:
            return case, result

    raise RuntimeError(f"missing case result: {name}")


def _print_snapshot_evidence(
    snapshot,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
) -> None:
    frame = snapshot.frame
    counts = snapshot.source_counts

    print(f"Candidate screenshot path: {candidate_evidence_path}")
    print(f"Formal evidence path: {formal_evidence_path}")
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
    print(
        "Fixture marker observed: "
        f"{_fixture_identity_observed(snapshot)}"
    )
    print("Raw logical OCR evidence:")
    if not snapshot.ocr_elements:
        print("  none")

    for index, element in enumerate(snapshot.ocr_elements, start=1):
        print(
            f"  {index}: "
            f"text={element.text!r}; "
            f"confidence={element.confidence!r}; "
            f"element type={element.element_type!r}; "
            f"logical bounding box={_format_box(element)}"
        )


def _acceptance_failures(
    snapshot,
    results,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
) -> list[str]:
    from PIL import Image
    from computer_agent.grounding import GroundingStatus

    failures = []
    counts = snapshot.source_counts
    candidate_path = Path(candidate_evidence_path)

    if Path(snapshot.frame.image_path).resolve() != candidate_path.resolve():
        failures.append(
            "snapshot frame image path was not the candidate evidence path: "
            f"{snapshot.frame.image_path} != {candidate_path}"
        )

    if not candidate_path.is_file():
        failures.append(
            f"candidate screenshot was not created at {candidate_path}"
        )
    else:
        with Image.open(candidate_path) as saved:
            if saved.size != snapshot.frame.pixel_size:
                failures.append(
                    "loaded screenshot size did not match frame pixel size: "
                    f"{saved.size} != {snapshot.frame.pixel_size}"
                )

    if counts["accessibility"] <= 0:
        failures.append("Accessibility element count was not greater than zero")

    if counts["ocr"] <= 0:
        failures.append("logical OCR element count was not greater than zero")

    if counts["fused"] <= 0:
        failures.append("fused element count was not greater than zero")

    if snapshot.warnings:
        failures.append("perception warning occurred")

    for case, result in results:
        if result.status is not case.expected_status:
            failures.append(
                f"{case.name}: status was {_status_value(result.status)}, "
                f"expected {_status_value(case.expected_status)}"
            )

    identifier_case, identifier_result = _case_result(results, IDENTIFIER_TEXT)
    if identifier_result.status is GroundingStatus.RESOLVED:
        if identifier_result.element.identifier != identifier_case.expected_identifier:
            failures.append(
                f"{IDENTIFIER_TEXT}: resolved identifier was "
                f"{identifier_result.element.identifier!r}, expected "
                f"{identifier_case.expected_identifier!r}"
            )

        if not all(
            candidate.match_basis == "identifier"
            for candidate in identifier_result.candidates
        ):
            failures.append(
                f"{IDENTIFIER_TEXT}: did not resolve through identifier tier"
            )

    role_case, role_result = _case_result(results, ROLE_TEXT)
    if role_result.status is GroundingStatus.RESOLVED:
        if role_result.element.identifier != role_case.expected_identifier:
            failures.append(
                f"{ROLE_TEXT}: resolved identifier was "
                f"{role_result.element.identifier!r}, expected "
                f"{role_case.expected_identifier!r}"
            )

        if role_result.element.element_type != "button":
            failures.append(
                f"{ROLE_TEXT}: resolved role was "
                f"{role_result.element.element_type!r}, expected 'button'"
            )

        if not any(
            "incompatible_element_type" in candidate.rejection_reasons
            for candidate in role_result.candidates
        ):
            failures.append(
                f"{ROLE_TEXT}: no same-text different-role candidate was rejected"
            )

    for name in (DISABLED_ONLY_TEXT, BLOCKED_IDENTIFIER_TEXT):
        _, result = _case_result(results, name)
        if not any(
            "disabled" in candidate.rejection_reasons
            for candidate in result.candidates
        ):
            failures.append(f"{name}: no candidate had disabled rejection")

    _, disabled_result = _case_result(results, DISABLED_ONLY_TEXT)
    if disabled_result.candidates and not all(
        candidate.element.enabled is False
        for candidate in disabled_result.candidates
    ):
        failures.append(
            f"{DISABLED_ONLY_TEXT}: a candidate was not disabled"
        )

    _, blocked_result = _case_result(results, BLOCKED_IDENTIFIER_TEXT)
    if not blocked_result.candidates:
        failures.append(f"{BLOCKED_IDENTIFIER_TEXT}: no identifier candidate")
    elif any(
        candidate.match_basis != "identifier"
        for candidate in blocked_result.candidates
    ):
        failures.append(f"{BLOCKED_IDENTIFIER_TEXT}: fell back to text tier")

    if any(
        candidate.element.identifier != BLOCKED_IDENTIFIER_ID
        for candidate in blocked_result.candidates
    ):
        failures.append(
            f"{BLOCKED_IDENTIFIER_TEXT}: non-target identifier appeared "
            "in candidates"
        )

    _, ambiguous_result = _case_result(results, AMBIGUOUS_TEXT)
    ambiguous_eligible = tuple(
        candidate
        for candidate in ambiguous_result.candidates
        if candidate.eligible
    )
    best_ambiguous = _best_eligible_candidates(ambiguous_result)
    if len(best_ambiguous) < 2:
        failures.append(
            f"{AMBIGUOUS_TEXT}: fewer than two eligible best candidates"
        )

    if len(ambiguous_eligible) != 2:
        failures.append(
            f"{AMBIGUOUS_TEXT}: expected exactly two eligible candidates, "
            f"got {len(ambiguous_eligible)}"
        )

    if any(
        candidate.element.element_type != "button"
        for candidate in ambiguous_eligible
    ):
        failures.append(f"{AMBIGUOUS_TEXT}: eligible candidate role was not button")

    if any(
        _normalized_source(candidate.element.source) != "accessibility"
        for candidate in ambiguous_eligible
    ):
        failures.append(
            f"{AMBIGUOUS_TEXT}: eligible candidates were not Accessibility-only"
        )

    ambiguous_source_priorities = {
        _source_priority(candidate.element.source)
        for candidate in ambiguous_eligible
    }
    if len(ambiguous_source_priorities) > 1:
        failures.append(
            f"{AMBIGUOUS_TEXT}: candidates had different source tiers"
        )

    ambiguous_confidences = {
        candidate.element.confidence
        for candidate in ambiguous_eligible
    }
    if len(ambiguous_confidences) > 1:
        failures.append(
            f"{AMBIGUOUS_TEXT}: candidates had different confidences"
        )

    _, ocr_result = _case_result(results, OCR_ONLY_TEXT)
    if ocr_result.status is GroundingStatus.RESOLVED:
        if ocr_result.element.source != "ocr":
            failures.append(
                f"{OCR_ONLY_TEXT}: source was "
                f"{ocr_result.element.source!r}, expected 'ocr'"
            )

        if ocr_result.element.element_type != "text":
            failures.append(
                f"{OCR_ONLY_TEXT}: role was "
                f"{ocr_result.element.element_type!r}, expected 'text'"
            )

        if ocr_result.element.enabled is not None:
            failures.append(
                f"{OCR_ONLY_TEXT}: enabled was "
                f"{ocr_result.element.enabled!r}, expected None"
            )

        if _normalized_text(ocr_result.element.text) != _normalized_text(OCR_ONLY_TEXT):
            failures.append(
                f"{OCR_ONLY_TEXT}: resolved text did not match "
                f"{OCR_ONLY_TEXT!r}"
            )

    if any(
        _normalized_source(candidate.element.source) != "ocr"
        for candidate in ocr_result.candidates
    ):
        failures.append(f"{OCR_ONLY_TEXT}: non-OCR candidate appeared")

    _, missing_result = _case_result(results, MISSING_TEXT)
    if missing_result.candidates:
        failures.append(f"{MISSING_TEXT}: expected empty candidates")

    return failures


def _print_acceptance_failures(failures: list[str]) -> None:
    print("Live acceptance failed:")
    for failure in failures:
        print(f"  {failure}")


def _promote_candidate_evidence(
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
) -> None:
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)
    formal_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    os.replace(
        candidate_path,
        formal_path,
    )


def _evaluate_observed_snapshot(
    snapshot,
    *,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
    grounding_runner=None,
    acceptance_checker=None,
) -> int:
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)
    if grounding_runner is None:
        grounding_runner = _run_grounding_cases

    if acceptance_checker is None:
        acceptance_checker = _acceptance_failures

    _print_snapshot_evidence(
        snapshot,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
    )

    if not _fixture_identity_observed(snapshot):
        _print_fixture_identity_error(candidate_path)
        return 1

    results = grounding_runner(snapshot)
    failures = acceptance_checker(
        snapshot,
        results,
        candidate_evidence_path=candidate_path,
    )

    if failures:
        _print_acceptance_failures(failures)
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        return 1

    try:
        _promote_candidate_evidence(
            candidate_evidence_path=candidate_path,
            formal_evidence_path=formal_path,
        )
    except (FileNotFoundError, OSError) as error:
        print(
            "Evidence promotion failed: "
            f"{type(error).__name__}: {error}"
        )
        print(f"Candidate screenshot retained for debugging: {candidate_path}")
        return 1

    print(f"Formal evidence updated from candidate: {formal_path}")
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

    CANDIDATE_EVIDENCE_PATH.parent.mkdir(
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

    return _evaluate_observed_snapshot(snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
