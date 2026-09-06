"""Phase 05 Experiment 03: verified real-web navigation."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

from computer_agent.grounding import (
    ActionGrounder,
    ActionGroundingStatus,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.verification import (
    ActionVerificationStatus,
    ActionVerifier,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCREENSHOT_DIR = (
    PROJECT_ROOT
    / "assets/screenshots/phase05_real_web_autonomy"
)

BEFORE_EVIDENCE_PATH = (
    SCREENSHOT_DIR
    / "experiment_03_verified_web_navigation_before.png"
)

AFTER_CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR
    / "experiment_03_verified_web_navigation_candidate.png"
)

EVIDENCE_PATH = (
    SCREENSHOT_DIR
    / "experiment_03_verified_web_navigation.png"
)

TARGET_URL = "https://www.python.org/"

OCR_MINIMUM_CONFIDENCE = 0.05
OCR_PAGE_SEGMENTATION_MODE = 6
GROUNDING_MINIMUM_CONFIDENCE = 0.70
DEFAULT_POST_ACTION_WAIT_SECONDS = 2.0

START_PAGE_MARKER_SPEC = TargetSpec(
    text="Search This Site",
    element_types=("text_field",),
    minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
)

ACTION_TARGET_SPEC = TargetSpec(
    text="Docs",
    element_types=("link",),
    minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
)

VERIFICATION_TARGET_SPEC = TargetSpec(
    text="Library reference",
    element_types=("link",),
    minimum_confidence=GROUNDING_MINIMUM_CONFIDENCE,
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


def _post_action_wait_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--post-action-wait-seconds must be numeric"
        ) from error

    if (
        not math.isfinite(seconds)
        or not 0.0 <= seconds <= 10.0
    ):
        raise argparse.ArgumentTypeError(
            "--post-action-wait-seconds must be "
            "from 0.0 through 10.0"
        )

    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground and optionally execute one verified navigation "
            "from python.org to the Python documentation."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the generated click action. "
            "Default is dry-run."
        ),
    )

    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=8,
        help=(
            "Seconds to wait before the initial observation. "
            "Must be from 0 through 30."
        ),
    )

    parser.add_argument(
        "--post-action-wait-seconds",
        type=_post_action_wait_seconds,
        default=DEFAULT_POST_ACTION_WAIT_SECONDS,
        help=(
            "Seconds to wait after the live click before "
            "verification."
        ),
    )

    return parser.parse_args()


def _build_engine(capture_path: str | Path):
    from computer_agent.control.computer_controller import (
        ComputerController,
    )
    from computer_agent.perception import (
        MacOSAccessibility,
        PerceptionEngine,
        ScreenCapture,
        TesseractOCR,
        UIElementFusion,
    )

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
    from computer_agent.control.computer_controller import (
        ComputerController,
    )
    from computer_agent.tools.computer import (
        create_computer_tools,
    )
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    controller = ComputerController()

    return ToolExecutor(
        ToolRegistry(
            create_computer_tools(controller)
        )
    )


def _read_frontmost_application_name() -> str | None:
    from computer_agent.perception import MacOSAccessibility

    return (
        MacOSAccessibility()
        .read_frontmost_application_name()
    )


def _format_box(element) -> str:
    box = element.bounding_box

    return (
        f"x={box.x}, y={box.y}, "
        f"width={box.width}, height={box.height}"
    )


def _print_grounding(label: str, result) -> None:
    print(f"{label} status: {result.status.value}")
    print(f"{label} reason: {result.reason}")

    if result.element is None:
        print(f"{label} element: None")
        return

    element = result.element

    print(f"{label} type: {element.element_type!r}")
    print(f"{label} text: {element.text!r}")
    print(f"{label} source: {element.source!r}")
    print(
        f"{label} confidence: "
        f"{element.confidence:.2f}"
    )
    print(f"{label} enabled: {element.enabled!r}")
    print(f"{label} box: {_format_box(element)}")


def _ground_before(snapshot):
    grounder = UIGrounder()

    start_marker = grounder.ground(
        START_PAGE_MARKER_SPEC,
        snapshot.fused_elements,
    )

    action_target = grounder.ground(
        ACTION_TARGET_SPEC,
        snapshot.fused_elements,
    )

    verification_before = grounder.ground(
        VERIFICATION_TARGET_SPEC,
        snapshot.fused_elements,
    )

    action_grounding = ActionGrounder().ground_click(
        action_target,
        snapshot.frame.screen_size,
    )

    return (
        start_marker,
        action_target,
        verification_before,
        action_grounding,
    )


def _before_failures(
    *,
    frontmost_app,
    snapshot,
    start_marker,
    action_target,
    verification_before,
    action_grounding,
) -> list[str]:
    failures = []

    if frontmost_app != "Google Chrome":
        failures.append(
            "frontmost application was not Google Chrome"
        )

    if snapshot.warnings:
        failures.append(
            "before observation contained perception warnings"
        )

    if (
        start_marker.status
        is not GroundingStatus.RESOLVED
    ):
        failures.append(
            "python.org start-page marker did not resolve"
        )
    elif (
        start_marker.element is None
        or start_marker.element.element_type != "text_field"
    ):
        failures.append(
            "start-page marker was not a text field"
        )

    if (
        action_target.status
        is not GroundingStatus.RESOLVED
    ):
        failures.append(
            "Docs link did not resolve"
        )
    elif (
        action_target.element is None
        or action_target.element.element_type != "link"
    ):
        failures.append(
            "Docs target did not resolve to a link"
        )

    if (
        verification_before.status
        is not GroundingStatus.NOT_FOUND
    ):
        failures.append(
            "verification target already existed before navigation"
        )

    if (
        action_grounding.status
        is not ActionGroundingStatus.READY
    ):
        failures.append(
            "click action was not safely grounded"
        )
    elif action_grounding.action is None:
        failures.append(
            "READY action grounding contained no Action"
        )
    else:
        action = action_grounding.action

        if action.tool_name != "click_mouse":
            failures.append(
                "generated action was not click_mouse"
            )

        if set(action.arguments) != {"x", "y"}:
            failures.append(
                "generated click did not contain x/y arguments"
            )

    return failures


def _promote_candidate_evidence(
    *,
    candidate_evidence_path: str | Path,
    formal_evidence_path: str | Path,
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


def _run_acceptance(
    *,
    execute: bool,
    post_action_wait_seconds: float = (
        DEFAULT_POST_ACTION_WAIT_SECONDS
    ),
    before_capture_path: str | Path = BEFORE_EVIDENCE_PATH,
    candidate_evidence_path: str | Path = (
        AFTER_CANDIDATE_EVIDENCE_PATH
    ),
    formal_evidence_path: str | Path = EVIDENCE_PATH,
    observer_builder=_build_engine,
    executor_builder=_build_executor,
    frontmost_reader=_read_frontmost_application_name,
    sleeper=time.sleep,
) -> int:
    before_path = Path(before_capture_path)
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)

    before_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        before_snapshot = (
            observer_builder(before_path).observe()
        )
    except (OSError, RuntimeError) as error:
        print(
            "Before observation failed: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    before_frontmost_app = frontmost_reader()

    print(
        f"Frontmost application: "
        f"{before_frontmost_app}"
    )
    print(
        "Before fused element count:",
        len(before_snapshot.fused_elements),
    )
    print(
        "Before warnings:",
        before_snapshot.warnings or "none",
    )

    (
        start_marker,
        action_target,
        verification_before,
        action_grounding,
    ) = _ground_before(before_snapshot)

    _print_grounding(
        "Start-page marker",
        start_marker,
    )
    _print_grounding(
        "Action target grounding",
        action_target,
    )
    _print_grounding(
        "Before verification target",
        verification_before,
    )

    print(
        "Action grounding status:",
        action_grounding.status.value,
    )
    print(
        "Action grounding reason:",
        action_grounding.reason,
    )

    if action_grounding.action is not None:
        print(
            "Generated Action tool:",
            action_grounding.action.tool_name,
        )
        print(
            "Generated Action arguments:",
            action_grounding.action.arguments,
        )

    failures = _before_failures(
        frontmost_app=before_frontmost_app,
        snapshot=before_snapshot,
        start_marker=start_marker,
        action_target=action_target,
        verification_before=verification_before,
        action_grounding=action_grounding,
    )

    if failures:
        print()
        print("Precondition acceptance failed:")

        for failure in failures:
            print(f"  {failure}")

        print("Execution skipped.")
        return 1

    print()
    print("Precondition acceptance result: passed")

    if not execute:
        print("Execution skipped: dry-run mode.")
        print("Action execution count: 0")
        print("Evidence promotion: skipped")
        return 0

    action = action_grounding.action

    if action is None:
        raise RuntimeError(
            "READY action grounding contained no Action"
        )

    try:
        executor = executor_builder()
        tool_result = executor.execute(action)
    except (OSError, RuntimeError) as error:
        print(
            "Execution failed: "
            f"{type(error).__name__}: {error}"
        )
        return 1

    print()
    print("Execution performed: one Action executed.")
    print(
        f"Tool result success: {tool_result.success}"
    )
    print(
        f"Tool result tool name: {tool_result.tool_name}"
    )
    print(
        f"Tool result error: {tool_result.error}"
    )

    if post_action_wait_seconds:
        sleeper(post_action_wait_seconds)

    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        after_snapshot = (
            observer_builder(candidate_path).observe()
        )
    except (OSError, RuntimeError) as error:
        print(
            "After observation failed: "
            f"{type(error).__name__}: {error}"
        )
        print(
            "Candidate screenshot retained when available:",
            candidate_path,
        )
        return 1

    after_frontmost_app = frontmost_reader()

    print()
    print(
        f"After frontmost application: "
        f"{after_frontmost_app}"
    )
    print(
        "After fused element count:",
        len(after_snapshot.fused_elements),
    )
    print(
        "After warnings:",
        after_snapshot.warnings or "none",
    )

    verification_result = (
        ActionVerifier().verify_target_appeared(
            action=action,
            tool_result=tool_result,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            target_spec=VERIFICATION_TARGET_SPEC,
        )
    )

    print(
        "Verification status:",
        verification_result.status.value,
    )
    print(
        "Verification reason:",
        verification_result.reason,
    )

    _print_grounding(
        "Verification before",
        verification_result.before_grounding,
    )
    _print_grounding(
        "Verification after",
        verification_result.after_grounding,
    )

    failures = []

    if not tool_result.success:
        failures.append(
            f"click tool failed: {tool_result.error}"
        )

    if after_frontmost_app != "Google Chrome":
        failures.append(
            "Google Chrome was not frontmost after navigation"
        )

    if after_snapshot.warnings:
        failures.append(
            "after observation contained perception warnings"
        )

    if (
        Path(before_snapshot.frame.image_path).resolve()
        != before_path.resolve()
    ):
        failures.append(
            "before snapshot used the wrong evidence path"
        )

    if (
        Path(after_snapshot.frame.image_path).resolve()
        != candidate_path.resolve()
    ):
        failures.append(
            "after snapshot used the wrong candidate path"
        )

    if not candidate_path.is_file():
        failures.append(
            "after-action candidate screenshot was not created"
        )

    if (
        verification_result.status
        is not ActionVerificationStatus.VERIFIED
    ):
        failures.append(
            "navigation verification was "
            f"{verification_result.status.value}"
        )

    after_grounding = (
        verification_result.after_grounding
    )

    if (
        after_grounding.status
        is not GroundingStatus.RESOLVED
    ):
        failures.append(
            "Library reference did not resolve after navigation"
        )
    elif (
        after_grounding.element is None
        or after_grounding.element.element_type != "link"
    ):
        failures.append(
            "verification target was not a link"
        )

    if failures:
        print()
        print("Live acceptance failed:")

        for failure in failures:
            print(f"  {failure}")

        print(
            "Candidate screenshot retained for debugging:",
            candidate_path,
        )
        return 1

    try:
        _promote_candidate_evidence(
            candidate_evidence_path=candidate_path,
            formal_evidence_path=formal_path,
        )
    except (OSError, RuntimeError) as error:
        print(
            "Evidence promotion failed: "
            f"{type(error).__name__}: {error}"
        )
        print(
            "Candidate screenshot retained for debugging:",
            candidate_path,
        )
        return 1

    print()
    print(f"Formal evidence path: {formal_path}")
    print("Action execution count: 1")
    print("Evidence promotion: completed")
    print("Live acceptance result: passed")
    return 0


def main() -> int:
    args = _parse_args()

    print(
        "Phase 05 Experiment 03: "
        "Verified Web Navigation"
    )
    print(f"Start URL: {TARGET_URL}")
    print(
        "Start-page marker: "
        "Search This Site [text_field]"
    )
    print("Action target: Docs [link]")
    print(
        "Verification target: "
        "Library reference [link]"
    )

    if args.execute:
        print(
            "Execute mode: one generated click_mouse Action "
            "may be executed after all preconditions pass."
        )
    else:
        print(
            "Dry-run mode: no mouse action will be executed."
        )

    if sys.platform != "darwin":
        print(
            "Live acceptance failed: "
            "this experiment requires macOS."
        )
        return 1

    from computer_agent.perception import MacOSAccessibility

    accessibility = MacOSAccessibility()

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

    return _run_acceptance(
        execute=args.execute,
        post_action_wait_seconds=(
            args.post_action_wait_seconds
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
