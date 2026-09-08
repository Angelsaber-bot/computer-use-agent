"""Phase 05 Experiment 07: adaptive web recovery."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
import time

from computer_agent.agent.web_recovery import WebRecoveryDecision
from computer_agent.agent.web_recovery import WebRecoveryDecisionResult
from computer_agent.agent.web_recovery import decide_failed_grounding_recovery
from computer_agent.grounding import GroundingResult, GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception import (
    MacOSAccessibility,
    ViewportSearchController, ViewportSearchObservation, ViewportSearchPolicy,
    ViewportSearchResult, ViewportSearchStatus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_EVIDENCE_PATH = PROJECT_ROOT / (
    "assets/screenshots/phase05_real_web_autonomy/"
    "experiment_07_adaptive_web_recovery_candidate.png"
)

TARGET_URL = "https://www.python.org/"
EXPECTED_APPLICATION_NAME = "Google Chrome"
TARGET_TEXT = "Privacy Notice"
TARGET_SPEC = TargetSpec(text=TARGET_TEXT, element_types=("link",))
DEFAULT_WAIT_SECONDS = 8
DEFAULT_MAX_SCROLL_ATTEMPTS = 6
DEFAULT_SCROLL_AMOUNT = 12
DEFAULT_STABILIZATION_WAIT_SECONDS = 0.3


class AdaptiveWebRecoveryStatus(str, Enum):
    """Experiment-level outcomes for adaptive web recovery."""

    PASSED = "passed"
    NEEDS_SCROLL = "needs_scroll"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class AdaptiveWebRecoveryObservation:
    application_name: str | None
    viewport: object | None
    snapshot: object
    semantic_elements: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveWebRecoveryResult:
    status: AdaptiveWebRecoveryStatus
    reason: str
    execute: bool
    initial_observation: AdaptiveWebRecoveryObservation
    initial_grounding: GroundingResult
    recovery_decision: WebRecoveryDecisionResult | None
    search_result: ViewportSearchResult | None
    final_observation: AdaptiveWebRecoveryObservation | None = None
    final_grounding: GroundingResult | None = None

    @property
    def action_execution_count(self) -> int:
        if self.search_result is None:
            return 0
        return len(self.search_result.tool_results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(
        "Demonstrate cause-aware bounded viewport-search recovery "
        "for python.org's Privacy Notice link."
    ))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--max-scroll-attempts", type=int, default=6)
    return parser.parse_args()


def _countdown(seconds: int, *, sleeper=time.sleep) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"{remaining}...")
        sleeper(1)


def _build_engine(capture_path: str | Path):
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.perception import PerceptionEngine, ScreenCapture
    from computer_agent.perception import TesseractOCR, UIElementFusion

    controller = ComputerController()
    accessibility = MacOSAccessibility()
    ocr = TesseractOCR(
        minimum_confidence=0.05,
        page_segmentation_mode=6,
        group_words_by_line=True,
    )
    engine = PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=ocr,
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )
    return engine, accessibility


class _LiveObserver:
    def __init__(self, *, capture_path: str | Path) -> None:
        self.capture_path = Path(capture_path)
        self.engine, self.accessibility = _build_engine(self.capture_path)

    def observe(self) -> AdaptiveWebRecoveryObservation:
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.engine.observe()
        return AdaptiveWebRecoveryObservation(
            application_name=self.accessibility.read_frontmost_application_name(),
            viewport=self.accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(self.accessibility.read_frontmost_semantic_elements()),
        )


def _build_executor():
    from computer_agent.control.computer_controller import ComputerController
    from computer_agent.tools.computer import create_computer_tools
    from computer_agent.tools.executor import ToolExecutor
    from computer_agent.tools.registry import ToolRegistry

    return ToolExecutor(ToolRegistry(create_computer_tools(ComputerController())))


def _run_acceptance(
    *,
    execute: bool,
    observer_builder=_LiveObserver,
    executor_builder=_build_executor,
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    policy: ViewportSearchPolicy | None = None,
    sleeper=time.sleep,
    recovery_decider=decide_failed_grounding_recovery,
) -> AdaptiveWebRecoveryResult:
    observer = observer_builder(capture_path=capture_path)
    executor = executor_builder() if execute else None
    result = _run_recovery(
        execute=execute,
        observer=observer.observe,
        executor=executor,
        policy=policy or ViewportSearchPolicy(
            max_scroll_attempts=DEFAULT_MAX_SCROLL_ATTEMPTS,
            scroll_amount=DEFAULT_SCROLL_AMOUNT,
            stabilization_wait_seconds=DEFAULT_STABILIZATION_WAIT_SECONDS,
        ),
        sleeper=sleeper,
        recovery_decider=recovery_decider,
    )
    _print_result(result)
    return result


def _run_recovery(
    *,
    execute: bool,
    observer,
    executor,
    policy: ViewportSearchPolicy,
    sleeper,
    recovery_decider,
) -> AdaptiveWebRecoveryResult:
    initial = observer()
    grounder = UIGrounder()
    initial_grounding = _ground(grounder, initial)

    def finish(
        reason,
        *,
        status=AdaptiveWebRecoveryStatus.BLOCKED,
        decision=None,
        search=None,
        final=None,
        final_grounding=None,
    ):
        return AdaptiveWebRecoveryResult(
            status=status,
            reason=reason,
            execute=execute,
            initial_observation=initial,
            initial_grounding=initial_grounding,
            recovery_decision=decision,
            search_result=search,
            final_observation=final,
            final_grounding=final_grounding,
        )

    block_reason = _unsafe_initial_reason(initial)
    if block_reason is not None:
        return finish(block_reason)

    if initial_grounding.status is not GroundingStatus.NOT_FOUND:
        return finish(
            "initial grounding was not the expected not_found status",
        )

    decision = recovery_decider(initial_grounding)
    if not isinstance(decision, WebRecoveryDecisionResult):
        raise ValueError("recovery_decider must return WebRecoveryDecisionResult")
    if decision.decision is not WebRecoveryDecision.VIEWPORT_SEARCH:
        return finish(
            "recovery decision blocked viewport search",
            decision=decision,
        )

    search = _run_viewport_search(
        execute=execute,
        initial_observation=initial,
        observer=observer,
        executor=executor,
        policy=policy,
        sleeper=sleeper,
    )
    if not execute:
        status = AdaptiveWebRecoveryStatus.NEEDS_SCROLL
        reason = "bounded viewport search would be required"
        if search.status is not ViewportSearchStatus.NEEDS_SCROLL:
            status = AdaptiveWebRecoveryStatus.BLOCKED
            reason = "dry-run viewport search did not report needs_scroll"
        return finish(reason, status=status, decision=decision, search=search)

    if search.status is not ViewportSearchStatus.FOUND:
        return finish(
            f"viewport search did not find target: {search.status.value}",
            decision=decision,
            search=search,
        )

    if _has_unexpected_tool_activity(search, executor):
        return finish(
            "viewport search used an unexpected non-scroll action",
            decision=decision,
            search=search,
        )

    final = observer()
    final_grounding = _ground(grounder, final)
    if final_grounding.status is not GroundingStatus.RESOLVED:
        return finish(
            "final grounding was not resolved",
            decision=decision,
            search=search,
            final=final,
            final_grounding=final_grounding,
        )

    if final_grounding.element.text != TARGET_TEXT:
        return finish(
            "final resolved target text did not match expected text",
            decision=decision,
            search=search,
            final=final,
            final_grounding=final_grounding,
        )

    return finish(
        "cause-aware viewport-search recovery passed",
        status=AdaptiveWebRecoveryStatus.PASSED,
        decision=decision,
        search=search,
        final=final,
        final_grounding=final_grounding,
    )


def _ground(grounder: UIGrounder, observation: AdaptiveWebRecoveryObservation):
    viewport = observation.viewport.bounds if observation.viewport is not None else None
    return grounder.ground(TARGET_SPEC, observation.snapshot.fused_elements, viewport=viewport)


def _unsafe_initial_reason(observation: AdaptiveWebRecoveryObservation) -> str | None:
    if observation.application_name != EXPECTED_APPLICATION_NAME:
        return f"frontmost application is not {EXPECTED_APPLICATION_NAME}"
    if observation.viewport is None:
        return "no trusted viewport is available"
    if observation.snapshot.warnings:
        return "perception warnings make observation unsafe"
    return None


def _run_viewport_search(
    *,
    execute: bool,
    initial_observation: AdaptiveWebRecoveryObservation,
    observer,
    executor,
    policy: ViewportSearchPolicy,
    sleeper,
) -> ViewportSearchResult:
    use_initial = True

    def observe_for_search() -> ViewportSearchObservation:
        nonlocal use_initial
        if use_initial:
            use_initial = False
            return _viewport_search_observation(initial_observation)
        return _viewport_search_observation(observer())

    controller = ViewportSearchController(
        observer=observe_for_search,
        target_role="AXLink",
        target_text=TARGET_TEXT,
        policy=policy,
        sleeper=sleeper,
    )
    return controller.search(execute=execute, executor=executor)


def _viewport_search_observation(
    observation: AdaptiveWebRecoveryObservation,
) -> ViewportSearchObservation:
    return ViewportSearchObservation(
        application_name=observation.application_name,
        viewport=observation.viewport,
        semantic_elements=observation.semantic_elements,
    )


def _has_unexpected_tool_activity(search: ViewportSearchResult, executor) -> bool:
    if any(result.tool_name != "scroll" for result in search.tool_results):
        return True
    actions = getattr(executor, "actions", None)
    return actions is not None and any(
        getattr(action, "tool_name", None) != "scroll" for action in actions
    )


def _print_result(result: AdaptiveWebRecoveryResult) -> None:
    initial = result.initial_observation
    grounding = result.initial_grounding
    print("Adaptive recovery status:", result.status.value)
    print("Adaptive recovery reason:", result.reason)
    print("Initial app:", initial.application_name)
    print("Initial viewport:", initial.viewport)
    print("Initial grounding status:", grounding.status.value)
    print("Initial grounding reason:", grounding.reason)
    print("Initial candidate count:", len(grounding.candidates))
    decision = result.recovery_decision
    print(
        "Recovery decision:",
        "None" if decision is None else decision.decision.value,
    )
    if decision is not None:
        print("Recovery reason:", decision.reason)
    _print_search(result.search_result)
    _print_final_grounding(result.final_grounding)
    print("Total action execution count:", result.action_execution_count)


def _print_search(search: ViewportSearchResult | None) -> None:
    if search is None:
        print("Viewport search status: None")
        print("Scroll attempt count: 0")
        return

    print("Viewport search status:", search.status.value)
    print("Viewport search reason:", search.reason)
    print("Scroll attempt count:", search.scroll_attempts)
    for index, tool_result in enumerate(search.tool_results):
        print(
            f"Tool result {index}: tool={tool_result.tool_name} "
            f"success={tool_result.success} error={tool_result.error}"
        )


def _print_final_grounding(grounding: GroundingResult | None) -> None:
    if grounding is None:
        print("Final grounding status: None")
        return

    print("Final grounding status:", grounding.status.value)
    print("Final grounding reason:", grounding.reason)
    if grounding.element is None:
        return

    print("Final resolved element text:", repr(grounding.element.text))
    print("Final resolved element bounds:", grounding.element.bounding_box)


def _exit_code(status: AdaptiveWebRecoveryStatus) -> int:
    return 0 if status in {
        AdaptiveWebRecoveryStatus.PASSED,
        AdaptiveWebRecoveryStatus.NEEDS_SCROLL,
    } else 1


def main() -> int:
    args = _parse_args()
    print("Phase 05 Experiment 07: Adaptive Web Recovery")
    print(f"Target URL: {TARGET_URL}")
    print(f"Browser: {EXPECTED_APPLICATION_NAME}")
    print(f"Target: {TARGET_TEXT!r} [link]")
    if args.execute:
        print("Execute mode: bounded downward scroll actions may run.")
        print("The experiment will not click the recovered link.")
    else:
        print("Dry-run mode: no scroll action will be executed.")

    if sys.platform != "darwin":
        print("Adaptive recovery status:", AdaptiveWebRecoveryStatus.BLOCKED.value)
        print("Adaptive recovery reason: this experiment requires macOS.")
        return 1
    accessibility = MacOSAccessibility()
    if not accessibility.is_available():
        print("Adaptive recovery status:", AdaptiveWebRecoveryStatus.BLOCKED.value)
        print("Adaptive recovery reason: macOS Accessibility is unavailable.")
        return 1
    if not accessibility.is_trusted():
        print("Adaptive recovery status:", AdaptiveWebRecoveryStatus.BLOCKED.value)
        print("Adaptive recovery reason: Accessibility permission is not trusted.")
        return 1
    if args.wait_seconds > 0:
        print(
            "Switch to Chrome with python.org at the top. "
            "Observation starts after countdown:"
        )
        _countdown(args.wait_seconds)

    result = _run_acceptance(
        execute=args.execute,
        policy=ViewportSearchPolicy(
            max_scroll_attempts=args.max_scroll_attempts,
            scroll_amount=DEFAULT_SCROLL_AMOUNT,
            stabilization_wait_seconds=DEFAULT_STABILIZATION_WAIT_SECONDS,
        ),
    )
    return _exit_code(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
