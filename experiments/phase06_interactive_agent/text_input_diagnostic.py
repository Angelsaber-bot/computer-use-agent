"""Phase 06 diagnostic: deterministic text-input mechanism comparison.

Dry-run is the default. Execute mode mutates the currently focused text field
after the operator has prepared a safe target. The diagnostic does not submit
forms or navigate after entry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time

import pyautogui
import pyperclip

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from computer_agent.agent import TextInputObservation
from computer_agent.control.computer_controller import ComputerController
from computer_agent.grounding import TargetSpec, UIGrounder
from computer_agent.perception import (
    MacOSAccessibility,
    PerceptionEngine,
    ScreenCapture,
    TesseractOCR,
    UIElementFusion,
)
from computer_agent.perception.fusion import normalize_ui_text


DIAGNOSTIC_STRINGS = (
    "Alan Turing",
    "computer agent",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "test@example.com",
    "spaces, punctuation: hello world!",
)

MECHANISMS = (
    "pyautogui_write",
    "per_key_press",
    "clipboard_paste",
)

DEFAULT_TARGET = TargetSpec(
    text="Text input diagnostic",
    element_types=("text_field",),
    minimum_confidence=0.70,
)


@dataclass(frozen=True, slots=True)
class DiagnosticAttempt:
    intended: str
    mechanism: str
    focused_app: str | None
    target_identity: str | None
    pre_value: object
    post_value: object
    exact_equal: bool | None
    elapsed_ms: float
    input_source_summary: str
    ime_or_composition_active: bool | None
    modifier_flags: int | None
    modifier_keys_stuck: bool | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current macOS text-input mechanisms against a "
            "focused AX text field."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually focus, clear, and write into the target. "
            "Default is dry-run."
        ),
    )
    parser.add_argument(
        "--target-text",
        default="Text input diagnostic",
        help="Accessible name of the text field to test.",
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        default=(
            Path.home()
            / "Library"
            / "Application Support"
            / "Computer Agent"
            / "live"
            / "text_input_diagnostic.png"
        ),
    )
    parser.add_argument(
        "--post-entry-wait",
        type=float,
        default=0.1,
        help="Short observation settling interval after each mechanism.",
    )
    return parser.parse_args()


def _build_observer(capture_path: Path):
    controller = ComputerController()
    accessibility = MacOSAccessibility()
    engine = PerceptionEngine(
        screen_capture=ScreenCapture(controller),
        accessibility_reader=accessibility,
        ocr=TesseractOCR(
            minimum_confidence=0.05,
            page_segmentation_mode=6,
            group_words_by_line=True,
        ),
        fusion=UIElementFusion(),
        capture_path=capture_path,
    )

    def observe() -> TextInputObservation:
        capture_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        snapshot = engine.observe(
            include_ocr=False,
        )
        accessibility_snapshot = snapshot.accessibility_snapshot
        if accessibility_snapshot is not None:
            return TextInputObservation(
                application_name=accessibility_snapshot.application_name,
                viewport=accessibility_snapshot.viewport,
                snapshot=snapshot,
                semantic_elements=accessibility_snapshot.semantic_elements,
            )

        return TextInputObservation(
            application_name=accessibility.read_frontmost_application_name(),
            viewport=accessibility.read_frontmost_viewport(),
            snapshot=snapshot,
            semantic_elements=tuple(
                accessibility.read_frontmost_semantic_elements()
            ),
        )

    return observe


def _clear_active_field() -> None:
    ComputerController.release_modifier_keys()
    ComputerController.hotkey(
        "command",
        "a",
        interval=0.05,
    )
    ComputerController.press_key("delete")


def _enter_text(mechanism: str, text: str) -> None:
    if mechanism == "pyautogui_write":
        ComputerController.type_text(text)
        return

    if mechanism == "per_key_press":
        for character in text:
            if character == " ":
                pyautogui.press("space")
            else:
                pyautogui.press(character)
        return

    if mechanism == "clipboard_paste":
        prior = pyperclip.paste()
        pyperclip.copy(text)
        ComputerController.hotkey(
            "command",
            "v",
            interval=0.05,
        )
        pyperclip.copy(prior)
        return

    raise RuntimeError(f"unsupported mechanism: {mechanism}")


def _ground_target(
    observation: TextInputObservation,
    target: TargetSpec,
):
    viewport_bounds = (
        observation.viewport.bounds
        if observation.viewport is not None
        else None
    )
    return UIGrounder().ground(
        target,
        observation.snapshot.fused_elements,
        viewport=viewport_bounds,
    )


def _input_source_summary() -> str:
    try:
        result = subprocess.run(
            [
                "plutil",
                "-p",
                str(
                    Path.home()
                    / "Library/Preferences/com.apple.HIToolbox.plist"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"

    interesting = []
    for line in result.stdout.splitlines():
        if any(
            key in line
            for key in (
                "AppleCurrentKeyboardLayoutInputSourceID",
                "AppleSelectedInputSources",
                "Input Mode",
                "InputSourceKind",
                "KeyboardLayout Name",
            )
        ):
            interesting.append(line.strip())
    return " | ".join(interesting) if interesting else "unavailable"


def _modifier_flags() -> int | None:
    try:
        import Quartz
    except Exception:
        return None

    try:
        return int(Quartz.CGEventSourceFlagsState(1))
    except Exception:
        return None


def _modifiers_stuck(flags: int | None) -> bool | None:
    if flags is None:
        return None
    mask = 0
    for name in (
        "kCGEventFlagMaskShift",
        "kCGEventFlagMaskControl",
        "kCGEventFlagMaskAlternate",
        "kCGEventFlagMaskCommand",
    ):
        mask |= int(getattr(__import__("Quartz"), name, 0))
    return bool(flags & mask)


def _composition_active(summary: str) -> bool | None:
    if not summary or summary.startswith("unavailable"):
        return None
    normalized = summary.lower()
    return any(
        marker in normalized
        for marker in (
            "input mode",
            "input method",
            "pinyin",
            "scim",
            "itabc",
        )
    )


def _attempt(
    *,
    observe,
    target: TargetSpec,
    mechanism: str,
    text: str,
    execute: bool,
    post_entry_wait: float,
) -> DiagnosticAttempt:
    source_summary = _input_source_summary()
    flags = _modifier_flags()
    observation = observe()
    grounding = _ground_target(observation, target)
    element = grounding.element
    pre_value = element.value if element is not None else None
    identity = _identity(element)

    if execute and element is not None:
        ComputerController.click_mouse(
            int(element.center[0]),
            int(element.center[1]),
        )
        _clear_active_field()
        _enter_text(mechanism, text)
        if post_entry_wait > 0:
            time.sleep(post_entry_wait)

    started = time.perf_counter()
    post_observation = observe()
    elapsed_ms = (time.perf_counter() - started) * 1000
    post_grounding = _ground_target(post_observation, target)
    post_element = post_grounding.element
    post_value = post_element.value if post_element is not None else None
    exact = post_value == text if isinstance(post_value, str) else None

    return DiagnosticAttempt(
        intended=text,
        mechanism=mechanism,
        focused_app=post_observation.application_name,
        target_identity=identity,
        pre_value=pre_value,
        post_value=post_value,
        exact_equal=exact,
        elapsed_ms=elapsed_ms,
        input_source_summary=source_summary,
        ime_or_composition_active=_composition_active(source_summary),
        modifier_flags=flags,
        modifier_keys_stuck=_modifiers_stuck(flags),
    )


def _identity(element) -> str | None:
    if element is None:
        return None
    text = normalize_ui_text(element.text or "")
    if element.identifier:
        return f"id:{element.identifier}"
    if text:
        return f"{element.element_type}:{text}"
    return element.element_type


def _print_attempt(attempt: DiagnosticAttempt) -> None:
    print("attempt")
    print(f"  intended={attempt.intended!r}")
    print(f"  mechanism={attempt.mechanism}")
    print(f"  focused_app={attempt.focused_app!r}")
    print(f"  target_identity={attempt.target_identity!r}")
    print(f"  pre_value={attempt.pre_value!r}")
    print(f"  post_value={attempt.post_value!r}")
    print(f"  exact_equal={attempt.exact_equal}")
    print(f"  elapsed_ms={attempt.elapsed_ms:.2f}")
    print(f"  input_source={attempt.input_source_summary}")
    print(f"  ime_or_composition_active={attempt.ime_or_composition_active}")
    print(f"  modifier_flags={attempt.modifier_flags}")
    print(f"  modifier_keys_stuck={attempt.modifier_keys_stuck}")


def main() -> int:
    args = _parse_args()
    target = TargetSpec(
        text=args.target_text,
        element_types=DEFAULT_TARGET.element_types,
        minimum_confidence=DEFAULT_TARGET.minimum_confidence,
    )

    print("Phase 06 Text Input Diagnostic")
    print(f"execute={args.execute}")
    print(f"target_text={args.target_text!r}")
    if not args.execute:
        print("dry-run: no text will be entered")

    if sys.platform != "darwin":
        print("status=blocked reason=requires macOS")
        return 1

    accessibility = MacOSAccessibility()
    if not accessibility.is_available() or not accessibility.is_trusted():
        print("status=blocked reason=macOS Accessibility unavailable")
        return 1

    observe = _build_observer(args.capture_path)
    for mechanism in MECHANISMS:
        for text in DIAGNOSTIC_STRINGS:
            attempt = _attempt(
                observe=observe,
                target=target,
                mechanism=mechanism,
                text=text,
                execute=args.execute,
                post_entry_wait=args.post_entry_wait,
            )
            _print_attempt(attempt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
