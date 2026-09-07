"""Phase 05 Experiment 04: scroll and viewport search."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from computer_agent.core.models import ToolResult
from computer_agent.perception import (
    DiagnosisStatus,
    MacOSAccessibility,
    ViewportSearchController,
    ViewportSearchObservation,
    ViewportSearchPolicy,
    ViewportSearchResult,
    ViewportSearchStatus,
    classify_visibility,
    diagnose_semantic_target_visibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_TEXT = "Privacy Notice"
TARGET_ROLE = "AXLink"
DEFAULT_MAX_SCROLL_ATTEMPTS = 6
DEFAULT_SCROLL_AMOUNT = 4
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.3


def _bounded_int(
    value: str,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be an integer"
        ) from error

    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be from {minimum} through {maximum}"
        )

    return parsed


def _wait_seconds(value: str) -> int:
    return _bounded_int(
        value,
        name="wait-seconds",
        minimum=0,
        maximum=30,
    )


def _max_scroll_attempts(value: str) -> int:
    return _bounded_int(
        value,
        name="max-scroll-attempts",
        minimum=0,
        maximum=30,
    )


def _scroll_amount(value: str) -> int:
    return _bounded_int(
        value,
        name="scroll-amount",
        minimum=1,
        maximum=20,
    )


def _stabilization_wait_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "stabilization-wait-seconds must be numeric"
        ) from error

    if not 0.0 <= parsed <= 10.0:
        raise argparse.ArgumentTypeError(
            "stabilization-wait-seconds must be from 0.0 through 10.0"
        )

    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose real-web viewport and target visibility "
            "without interacting with the computer."
        )
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=5,
        help="Seconds to wait before the first observation.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute bounded downward scroll actions when search is needed."
        ),
    )
    parser.add_argument(
        "--max-scroll-attempts",
        type=_max_scroll_attempts,
        default=DEFAULT_MAX_SCROLL_ATTEMPTS,
        help="Maximum downward scroll attempts in execute mode.",
    )
    parser.add_argument(
        "--scroll-amount",
        type=_scroll_amount,
        default=DEFAULT_SCROLL_AMOUNT,
        help="Positive scroll units per downward attempt.",
    )
    parser.add_argument(
        "--stabilization-wait-seconds",
        type=_stabilization_wait_seconds,
        default=DEFAULT_STABILIZATION_WAIT_SECONDS,
        help="Seconds to wait after each scroll before observing again.",
    )
    return parser.parse_args()


def _observe(
    accessibility: MacOSAccessibility,
) -> ViewportSearchObservation:
    return ViewportSearchObservation(
        application_name=(
            accessibility.read_frontmost_application_name()
        ),
        viewport=accessibility.read_frontmost_viewport(),
        semantic_elements=tuple(
            accessibility.read_frontmost_semantic_elements()
        ),
    )


def _run_diagnosis(
    *,
    accessibility: MacOSAccessibility,
) -> DiagnosisStatus:
    observation = _observe(
        accessibility,
    )

    diagnosis = diagnose_semantic_target_visibility(
        application_name=observation.application_name,
        viewport=observation.viewport,
        semantic_elements=observation.semantic_elements,
        target_role=TARGET_ROLE,
        target_text=TARGET_TEXT,
    )

    _print_observation(
        observation,
        diagnosis,
        index=0,
    )

    return diagnosis.status


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


def _run_search(
    *,
    execute: bool,
    accessibility: MacOSAccessibility,
    policy: ViewportSearchPolicy,
    executor=None,
    sleeper=time.sleep,
) -> ViewportSearchResult:
    controller = ViewportSearchController(
        observer=lambda: _observe(accessibility),
        target_role=TARGET_ROLE,
        target_text=TARGET_TEXT,
        policy=policy,
        sleeper=sleeper,
    )

    result = controller.search(
        execute=execute,
        executor=executor,
    )

    _print_search_result(
        result,
        execute=execute,
        policy=policy,
    )

    return result


def _print_observation(
    observation: ViewportSearchObservation,
    diagnosis,
    *,
    index: int,
) -> None:
    print()
    print(f"Observation {index}:")
    print("Application:", observation.application_name)
    print("Viewport:", observation.viewport)
    print(
        "Raw semantic element count:",
        len(observation.semantic_elements),
    )
    print("Target matches:", len(diagnosis.matches))
    print("Diagnosis status:", diagnosis.status.value)
    print("Diagnosis reason:", diagnosis.reason)

    for match_index, element in enumerate(diagnosis.matches):
        visibility = (
            classify_visibility(
                element.bounds,
                observation.viewport,
            )
            if observation.viewport is not None
            else None
        )

        print(
            f"[{match_index}] "
            f"text={element.text!r} "
            f"bounds={element.bounds} "
            f"visibility={visibility}"
        )


def _print_search_result(
    result: ViewportSearchResult,
    *,
    execute: bool,
    policy: ViewportSearchPolicy,
) -> None:
    print()
    print("Search mode:", "execute" if execute else "dry-run")
    print("Search direction: down")
    print("Max scroll attempts:", policy.max_scroll_attempts)
    print("Scroll amount:", policy.scroll_amount)
    print("Scroll attempts performed:", result.scroll_attempts)
    print("Search status:", result.status.value)
    print("Search reason:", result.reason)

    for index, observation in enumerate(result.observations):
        diagnosis = diagnose_semantic_target_visibility(
            application_name=observation.application_name,
            viewport=observation.viewport,
            semantic_elements=observation.semantic_elements,
            target_role=TARGET_ROLE,
            target_text=TARGET_TEXT,
        )
        _print_observation(
            observation,
            diagnosis,
            index=index,
        )

    for index, tool_result in enumerate(result.tool_results):
        _print_tool_result(
            index,
            tool_result,
        )


def _print_tool_result(
    index: int,
    tool_result: ToolResult,
) -> None:
    print(
        f"Tool result {index}: "
        f"tool={tool_result.tool_name} "
        f"success={tool_result.success} "
        f"error={tool_result.error}"
    )


def _exit_code(status: ViewportSearchStatus) -> int:
    if status in (
        ViewportSearchStatus.FOUND,
        ViewportSearchStatus.NEEDS_SCROLL,
    ):
        return 0

    return 1


def main() -> int:
    args = _parse_args()

    print("Experiment 05.04: Scroll and Viewport Search")
    print(f"Target: {TARGET_TEXT!r} ({TARGET_ROLE})")

    if args.execute:
        print(
            "Execute mode: bounded downward scroll actions may run "
            "only while viewport search is needed."
        )
    else:
        print(
            "Dry-run mode: one read-only observation; no scroll action "
            "will be executed."
        )

    if sys.platform != "darwin":
        print(
            "Search status:",
            ViewportSearchStatus.BLOCKED.value,
        )
        print("Search reason: this experiment requires macOS.")
        return 1

    accessibility = MacOSAccessibility()

    if not accessibility.is_available():
        print(
            "Search status:",
            ViewportSearchStatus.BLOCKED.value,
        )
        print(
            "Search reason: "
            "macOS Accessibility is unavailable."
        )
        return 1

    if not accessibility.is_trusted():
        print(
            "Search status:",
            ViewportSearchStatus.BLOCKED.value,
        )
        print(
            "Search reason: "
            "Accessibility permission is not trusted."
        )
        return 1

    if args.wait_seconds > 0:
        print(
            f"Switch to Chrome with python.org open. "
            f"Observing in {args.wait_seconds} seconds..."
        )
        time.sleep(args.wait_seconds)

    policy = ViewportSearchPolicy(
        max_scroll_attempts=args.max_scroll_attempts,
        scroll_amount=args.scroll_amount,
        stabilization_wait_seconds=args.stabilization_wait_seconds,
    )

    executor = _build_executor() if args.execute else None

    result = _run_search(
        execute=args.execute,
        accessibility=accessibility,
        policy=policy,
        executor=executor,
    )

    return _exit_code(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
