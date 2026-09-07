"""Phase 05 Experiment 05: real-web text input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from computer_agent.agent import (
    TextInputController,
    TextInputObservation,
    TextInputResult,
    TextInputStatus,
)
from computer_agent.grounding import TargetSpec
from computer_agent.perception import (
    MacOSAccessibility,
    ViewportSearchPolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCREENSHOT_DIR = (
    PROJECT_ROOT
    / "assets/screenshots/phase05_real_web_autonomy"
)
CANDIDATE_EVIDENCE_PATH = (
    SCREENSHOT_DIR
    / "experiment_05_real_web_text_input_candidate.png"
)

TARGET_URL = "https://www.python.org/"
TARGET_TEXT = "Search This Site"
TARGET_SPEC = TargetSpec(
    text=TARGET_TEXT,
    element_types=("text_field",),
    minimum_confidence=0.70,
)
DEFAULT_INPUT_TEXT = "accessibility test"
DEFAULT_WAIT_SECONDS = 8
DEFAULT_POST_ACTION_WAIT_SECONDS = 0.5


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

    if not 0.0 <= seconds <= 10.0:
        raise argparse.ArgumentTypeError(
            "--post-action-wait-seconds must be from 0.0 through 10.0"
        )

    return seconds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ground and optionally type into python.org's search field "
            "without submitting the form."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute the generated focus and type actions. "
            "Default is dry-run."
        ),
    )
    parser.add_argument(
        "--wait-seconds",
        type=_wait_seconds,
        default=DEFAULT_WAIT_SECONDS,
        help="Seconds to wait before the first observation.",
    )
    parser.add_argument(
        "--input-text",
        default=DEFAULT_INPUT_TEXT,
        help="Deterministic text to type into the search field.",
    )
    parser.add_argument(
        "--post-action-wait-seconds",
        type=_post_action_wait_seconds,
        default=DEFAULT_POST_ACTION_WAIT_SECONDS,
        help="Seconds to wait after typing before verification.",
    )
    return parser.parse_args()


def _build_engine(capture_path: str | Path):
    from computer_agent.control.computer_controller import (
        ComputerController,
    )
    from computer_agent.perception import (
        PerceptionEngine,
        ScreenCapture,
        TesseractOCR,
        UIElementFusion,
    )

    controller = ComputerController()
    accessibility = MacOSAccessibility()

    return PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=0.05,
            page_segmentation_mode=6,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    ), accessibility


class _LiveObserver:
    def __init__(
        self,
        *,
        capture_path: str | Path,
    ) -> None:
        self.capture_path = Path(capture_path)
        self.engine, self.accessibility = _build_engine(
            self.capture_path
        )

    def observe(self) -> TextInputObservation:
        self.capture_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        snapshot = self.engine.observe()

        return TextInputObservation(
            application_name=(
                self.accessibility.read_frontmost_application_name()
            ),
            viewport=self.accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(
                self.accessibility.read_frontmost_semantic_elements()
            ),
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


def _run_acceptance(
    *,
    execute: bool,
    input_text: str = DEFAULT_INPUT_TEXT,
    post_action_wait_seconds: float = DEFAULT_POST_ACTION_WAIT_SECONDS,
    observer_builder=_LiveObserver,
    executor_builder=_build_executor,
    capture_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    sleeper=time.sleep,
) -> TextInputResult:
    observer = observer_builder(
        capture_path=capture_path,
    )
    executor = executor_builder() if execute else None

    controller = TextInputController(
        observer=observer.observe,
        target_spec=TARGET_SPEC,
        input_text=input_text,
        viewport_search_policy=ViewportSearchPolicy(
            scroll_amount=12,
        ),
        sleeper=sleeper,
        stabilization_wait_seconds=post_action_wait_seconds,
    )

    result = controller.run(
        execute=execute,
        executor=executor,
    )
    _print_result(result)

    return result


def _print_result(result: TextInputResult) -> None:
    print("Text input status:", result.status.value)
    print("Text input reason:", result.reason)
    _print_observation(
        "Before",
        result.before_observation,
        result.before_grounding,
    )
    print(
        "Focus action status:",
        result.action_grounding.status.value,
    )
    print(
        "Focus action reason:",
        result.action_grounding.reason,
    )
    if result.action_grounding.action is not None:
        print(
            "Focus action tool:",
            result.action_grounding.action.tool_name,
        )
        print(
            "Focus action arguments:",
            result.action_grounding.action.arguments,
        )

    for index, tool_result in enumerate(result.tool_results):
        print(
            f"Tool result {index}: "
            f"tool={tool_result.tool_name} "
            f"success={tool_result.success} "
            f"error={tool_result.error}"
        )

    if result.after_observation is not None:
        _print_observation(
            "After",
            result.after_observation,
            result.after_grounding,
        )

    print(
        "Action execution count:",
        len(result.tool_results),
    )


def _print_observation(
    label: str,
    observation: TextInputObservation,
    grounding,
) -> None:
    print(f"{label} frontmost application:", observation.application_name)
    print(f"{label} viewport:", observation.viewport)
    print(
        f"{label} fused element count:",
        len(observation.snapshot.fused_elements),
    )
    print(
        f"{label} warnings:",
        observation.snapshot.warnings or "none",
    )

    if grounding is None:
        print(f"{label} grounding: None")
        return

    print(f"{label} grounding status:", grounding.status.value)
    print(f"{label} grounding reason:", grounding.reason)

    if grounding.element is None:
        print(f"{label} field: None")
        return

    element = grounding.element
    print(f"{label} field type:", element.element_type)
    print(f"{label} field text:", repr(element.text))
    print(f"{label} field value:", repr(element.value))
    print(f"{label} field box:", element.bounding_box)


def _exit_code(status: TextInputStatus) -> int:
    if status in (
        TextInputStatus.VERIFIED,
        TextInputStatus.NEEDS_ACTION,
    ):
        return 0

    return 1


def main() -> int:
    args = _parse_args()

    print("Phase 05 Experiment 05: Real Web Text Input")
    print(f"Target URL: {TARGET_URL}")
    print(f"Target field: {TARGET_TEXT!r} [text_field]")
    print(f"Input text: {args.input_text!r}")

    if args.execute:
        print(
            "Execute mode: one focus click and one type_text action may run."
        )
        print("The experiment will not submit the search form.")
    else:
        print(
            "Dry-run mode: observe and ground only; no click or typing."
        )

    if sys.platform != "darwin":
        print("Text input status:", TextInputStatus.BLOCKED.value)
        print("Text input reason: this experiment requires macOS.")
        return 1

    accessibility = MacOSAccessibility()

    if not accessibility.is_available():
        print("Text input status:", TextInputStatus.BLOCKED.value)
        print(
            "Text input reason: "
            "macOS Accessibility is unavailable."
        )
        return 1

    if not accessibility.is_trusted():
        print("Text input status:", TextInputStatus.BLOCKED.value)
        print(
            "Text input reason: "
            "Accessibility permission is not trusted."
        )
        return 1

    if args.wait_seconds > 0:
        print(
            f"Switch to Chrome with python.org open. "
            f"Observing in {args.wait_seconds} seconds..."
        )
        time.sleep(args.wait_seconds)

    result = _run_acceptance(
        execute=args.execute,
        input_text=args.input_text,
        post_action_wait_seconds=args.post_action_wait_seconds,
    )

    return _exit_code(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
