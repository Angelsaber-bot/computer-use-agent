"""Phase 04 experiment for deterministic recovery and re-grounding."""

from __future__ import annotations

from pathlib import Path
import time

from computer_agent.grounding import ActionGrounder, ActionGroundingStatus
from computer_agent.grounding import GroundingStatus, TargetSpec, UIGrounder
from computer_agent.recovery import ActionRecovery, RecoveryStatus
from computer_agent.verification import ActionVerificationStatus, ActionVerifier

if __package__:
    from .live_harness_utils import (
        build_live_perception_engine,
        build_live_tool_executor,
        failed_condition_messages,
        fixture_identity_observed,
        live_prerequisites_available,
        observe_with_capture_path,
        parse_execute_harness_args,
        print_failures,
        print_manual_focus_instructions,
        print_recovery_summary,
        print_snapshot_summary,
        print_tool_result,
        print_verification_summary,
        promote_candidate_evidence,
        snapshot_path_mismatched,
        wait_for_focus,
    )
else:
    from live_harness_utils import (
        build_live_perception_engine,
        build_live_tool_executor,
        failed_condition_messages,
        fixture_identity_observed,
        live_prerequisites_available,
        observe_with_capture_path,
        parse_execute_harness_args,
        print_failures,
        print_manual_focus_instructions,
        print_recovery_summary,
        print_snapshot_summary,
        print_tool_result,
        print_verification_summary,
        promote_candidate_evidence,
        snapshot_path_mismatched,
        wait_for_focus,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = PROJECT_ROOT / "assets/fixtures/phase04_ui_grounding_task_reasoning"
SCREENSHOT_DIR = PROJECT_ROOT / "assets/screenshots/phase04_ui_grounding_task_reasoning"
FIXTURE_PATH = FIXTURE_DIR / "experiment_05_recovery_regrounding.html"
BEFORE_CAPTURE_PATH = SCREENSHOT_DIR / "experiment_05_recovery_before.png"
AFTER_FIRST_CAPTURE_PATH = SCREENSHOT_DIR / "experiment_05_recovery_after_first.png"
CANDIDATE_EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_05_recovery_candidate.png"
EVIDENCE_PATH = SCREENSHOT_DIR / "experiment_05_recovery_regrounding.png"

ACTION_CONFIDENCE = 0.70
DEFAULT_POST_ACTION_WAIT_SECONDS = 0.25
MAX_ATTEMPTS = 2

FIXTURE_MARKER = "PHASE04_RECOVERY_REGROUNDING_FIXTURE_05"
ACTION_TARGET_TEXT = "ACTION_TARGET_05"
ACTION_TARGET_IDENTIFIER = "recovery-action-target-05"
VERIFICATION_TARGET_TEXT = "VERIFICATION_TARGET_05"
CLICK_TOOL_NAME = "click_mouse"
ACTION_TARGET_SPEC = TargetSpec(
    text=ACTION_TARGET_TEXT,
    identifier=ACTION_TARGET_IDENTIFIER,
    element_types=("button",),
    minimum_confidence=ACTION_CONFIDENCE,
)
VERIFICATION_TARGET_SPEC = TargetSpec(
    text=VERIFICATION_TARGET_TEXT,
    element_types=("button",),
    minimum_confidence=ACTION_CONFIDENCE,
)


def _ground_initial_snapshot(snapshot):
    ui_grounder = UIGrounder()
    action_grounding = ui_grounder.ground(ACTION_TARGET_SPEC, snapshot.fused_elements)
    click_grounding = ActionGrounder().ground_click(
        action_grounding, snapshot.frame.screen_size
    )
    verification_grounding = ui_grounder.ground(
        VERIFICATION_TARGET_SPEC, snapshot.fused_elements
    )
    return action_grounding, click_grounding, verification_grounding


def _action_coordinates(action) -> tuple[int, int] | None:
    if action is None or not isinstance(action.arguments, dict):
        return None
    x = action.arguments.get("x")
    y = action.arguments.get("y")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (x, y)):
        return None
    return x, y


def _click_action_failures(label: str, action) -> list[str]:
    if action is None:
        return [f"{label} Action was not generated"]

    failures = []
    if action.tool_name != CLICK_TOOL_NAME:
        failures.append(f"{label} Action tool was {action.tool_name}")

    if not isinstance(action.arguments, dict) or set(action.arguments) != {"x", "y"}:
        failures.append(f"{label} Action arguments were not exactly x/y coordinates")
    elif _action_coordinates(action) is None:
        failures.append(f"{label} Action arguments were not integer x/y coordinates")

    return failures


def _print_snapshot(label: str, snapshot, capture_path: Path) -> None:
    print_snapshot_summary(
        label,
        snapshot,
        capture_path,
        fixture_marker=FIXTURE_MARKER,
    )


def _initial_failures(action_grounding, click_grounding, verification_grounding):
    failures = []
    if action_grounding.status is not GroundingStatus.RESOLVED:
        failures.append(f"action target grounding was {action_grounding.status.value}")
    elif action_grounding.element.identifier != ACTION_TARGET_IDENTIFIER:
        failures.append("action target did not resolve to the expected identifier")
    elif not action_grounding.candidates:
        failures.append("action target did not include candidate evidence")
    elif not all(c.match_basis == "identifier" for c in action_grounding.candidates):
        failures.append("action target did not resolve through identifier evidence")

    if click_grounding.status is not ActionGroundingStatus.READY:
        failures.append(f"action grounding was {click_grounding.status.value}")
    else:
        failures.extend(_click_action_failures("initial", click_grounding.action))

    if verification_grounding.status is not GroundingStatus.NOT_FOUND:
        failures.append(f"initial verification target was {verification_grounding.status.value}")
    return failures


def _recovery_failures(result, first_action) -> list[str]:
    if result.status is not RecoveryStatus.RETRY_READY:
        return [f"recovery status was {result.status.value}"]

    failures = []
    if result.grounding_result.status is not GroundingStatus.RESOLVED:
        failures.append("recovery grounding was not resolved")
    if result.action_grounding_result.status is not ActionGroundingStatus.READY:
        failures.append("recovery action grounding was not ready")

    retry_action = result.action_grounding_result.action
    failures.extend(_click_action_failures("retry", retry_action))
    if retry_action.action_id == first_action.action_id:
        failures.append("retry Action reused the first Action id")
    if _action_coordinates(retry_action) == _action_coordinates(first_action):
        failures.append("retry Action coordinates did not change")
    return failures


def _run_acceptance(
    *,
    execute: bool,
    post_action_wait_seconds: float = DEFAULT_POST_ACTION_WAIT_SECONDS,
    before_capture_path: str | Path = BEFORE_CAPTURE_PATH,
    after_first_capture_path: str | Path = AFTER_FIRST_CAPTURE_PATH,
    candidate_evidence_path: str | Path = CANDIDATE_EVIDENCE_PATH,
    formal_evidence_path: str | Path = EVIDENCE_PATH,
    observer_builder=build_live_perception_engine,
    executor_builder=build_live_tool_executor,
    recovery_builder=ActionRecovery,
    sleeper=time.sleep,
) -> int:
    before_path = Path(before_capture_path)
    after_first_path = Path(after_first_capture_path)
    candidate_path = Path(candidate_evidence_path)
    formal_path = Path(formal_evidence_path)
    observation_count = 0
    execution_count = 0

    try:
        before_snapshot = observe_with_capture_path(observer_builder, before_path)
        observation_count += 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Initial observation failed: {type(error).__name__}: {error}")
        return 1

    _print_snapshot("Observation 1", before_snapshot, before_path)
    if snapshot_path_mismatched("observation 1", before_snapshot, before_path, ()):
        print("Execution skipped: observation 1 capture path was not established.")
        return 1

    if not fixture_identity_observed(before_snapshot, FIXTURE_MARKER):
        print_failures(["observation 1 fixture marker was not observed"], ())
        print("Execution skipped: fixture identity was not established.")
        return 1

    action_grounding, click_grounding, initial_verification = _ground_initial_snapshot(before_snapshot)
    print(f"Action target grounding status: {action_grounding.status.value}")
    print(f"Action grounding status: {click_grounding.status.value}")
    print(f"Initial verification target status: {initial_verification.status.value}")
    failures = _initial_failures(action_grounding, click_grounding, initial_verification)
    if failures:
        print_failures(failures, ())
        print("Execution skipped: action preconditions were not satisfied.")
        return 1

    if not execute:
        print("Live acceptance result: passed")
        print("Execution skipped: dry-run mode.")
        print("Observation count: 1")
        print("Action execution count: 0")
        print("Evidence promotion: skipped")
        return 0

    first_action = click_grounding.action
    try:
        executor = executor_builder()
        first_tool_result = executor.execute(first_action)
        execution_count += 1
        if post_action_wait_seconds:
            sleeper(post_action_wait_seconds)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"First execution failed: {type(error).__name__}: {error}")
        return 1

    print_tool_result("First", first_tool_result)

    try:
        after_first_snapshot = observe_with_capture_path(
            observer_builder,
            after_first_path,
        )
        observation_count += 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Observation 2 failed: {type(error).__name__}: {error}")
        print_failures(["observation 2 was unavailable"], (after_first_path,))
        return 1

    _print_snapshot("Observation 2", after_first_snapshot, after_first_path)
    if snapshot_path_mismatched(
        "observation 2",
        after_first_snapshot,
        after_first_path,
        (after_first_path,),
    ):
        return 1

    if not fixture_identity_observed(after_first_snapshot, FIXTURE_MARKER):
        print_failures(
            ["observation 2 fixture marker was not observed"],
            (after_first_path,),
        )
        return 1

    verifier = ActionVerifier()
    first_verification = verifier.verify_target_appeared(
        action=first_action,
        tool_result=first_tool_result,
        before_snapshot=before_snapshot,
        after_snapshot=after_first_snapshot,
        target_spec=VERIFICATION_TARGET_SPEC,
    )
    print_verification_summary("First", first_verification)
    failures = failed_condition_messages(
        (
            first_tool_result.success,
            f"first ToolResult failed: {first_tool_result.error}",
        ),
        (
            first_verification.status is ActionVerificationStatus.FAILED,
            f"first verification status was {first_verification.status.value}",
        ),
        (
            first_verification.after_grounding.status is GroundingStatus.NOT_FOUND,
            "first verification after grounding was "
            f"{first_verification.after_grounding.status.value}",
        ),
    )
    if failures:
        print_failures(failures, (after_first_path,))
        return 1

    recovery_result = recovery_builder().prepare_retry(
        verification_result=first_verification,
        tool_result=first_tool_result,
        target_spec=ACTION_TARGET_SPEC,
        latest_snapshot=after_first_snapshot,
        completed_attempts=1,
        max_attempts=MAX_ATTEMPTS,
    )
    print_recovery_summary(recovery_result)
    failures = _recovery_failures(recovery_result, first_action)
    if failures:
        print_failures(failures, (after_first_path,))
        return 1

    retry_action = recovery_result.action_grounding_result.action
    try:
        second_tool_result = executor.execute(retry_action)
        execution_count += 1
        if post_action_wait_seconds:
            sleeper(post_action_wait_seconds)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Retry execution failed: {type(error).__name__}: {error}")
        print_failures(["retry execution was unavailable"], (after_first_path,))
        return 1

    print_tool_result("Second", second_tool_result)

    try:
        final_snapshot = observe_with_capture_path(observer_builder, candidate_path)
        observation_count += 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Observation 3 failed: {type(error).__name__}: {error}")
        print_failures(
            ["observation 3 was unavailable"],
            (after_first_path, candidate_path),
        )
        return 1

    _print_snapshot("Observation 3", final_snapshot, candidate_path)
    if snapshot_path_mismatched(
        "observation 3",
        final_snapshot,
        candidate_path,
        (after_first_path, candidate_path),
    ):
        return 1

    if not fixture_identity_observed(final_snapshot, FIXTURE_MARKER):
        print_failures(
            ["observation 3 fixture marker was not observed"],
            (after_first_path, candidate_path),
        )
        return 1

    final_verification = verifier.verify_target_appeared(
        action=retry_action,
        tool_result=second_tool_result,
        before_snapshot=after_first_snapshot,
        after_snapshot=final_snapshot,
        target_spec=VERIFICATION_TARGET_SPEC,
    )
    print_verification_summary("Final", final_verification)
    failures = failed_condition_messages(
        (
            second_tool_result.success,
            f"second ToolResult failed: {second_tool_result.error}",
        ),
        (
            final_verification.status is ActionVerificationStatus.VERIFIED,
            f"final verification status was {final_verification.status.value}",
        ),
        (candidate_path.is_file(), "final candidate screenshot was not created"),
        (observation_count == 3, f"observation count was {observation_count}"),
        (execution_count == 2, f"action execution count was {execution_count}"),
    )
    if failures:
        print_failures(failures, (after_first_path, candidate_path))
        return 1

    try:
        promote_candidate_evidence(
            candidate_evidence_path=candidate_path,
            formal_evidence_path=formal_path,
        )
    except (OSError, RuntimeError) as error:
        print(f"Evidence promotion failed: {type(error).__name__}: {error}")
        print_failures(
            ["evidence promotion failed"],
            (after_first_path, candidate_path),
        )
        return 1

    print(f"Formal evidence updated from candidate: {formal_path}")
    print("Live acceptance result: passed")
    print("Observation count: 3")
    print("Action execution count: 2")
    print("Evidence promotion: completed")
    return 0


def main() -> int:
    args = parse_execute_harness_args(
        description="Observe the Experiment 05 fixture and prove one recovery retry.",
        execute_help="Execute two bounded click_mouse Actions. Default is dry-run.",
        post_action_wait_default=DEFAULT_POST_ACTION_WAIT_SECONDS,
        post_action_wait_help=(
            "Seconds to wait after each execution, from 0.0 through 5.0."
        ),
    )
    if not live_prerequisites_available(FIXTURE_PATH):
        return 1

    print_manual_focus_instructions(
        title="Phase 04 Experiment 05: Deterministic Recovery and Re-grounding",
        fixture_path=FIXTURE_PATH,
        execute=args.execute,
        execute_message="Execute mode: exactly two bounded click_mouse Actions may run.",
    )
    wait_for_focus(args.wait_seconds)
    return _run_acceptance(
        execute=args.execute,
        post_action_wait_seconds=args.post_action_wait_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
