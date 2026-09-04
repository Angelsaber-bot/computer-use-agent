"""Deterministic orchestration for executing structured plans."""

from __future__ import annotations

from collections.abc import Callable, Collection
import math
import time

from computer_agent.agent.loop_models import AgentLoopResult, AgentLoopStatus
from computer_agent.agent.state import AgentState
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding.action_grounder import ActionGrounder
from computer_agent.grounding.action_models import ActionGroundingStatus
from computer_agent.grounding.models import GroundingStatus
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.planning.models import (
    ActivateAppStep,
    InsertTextStep,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    StructuredPlan,
)
from computer_agent.recovery.action_recovery import ActionRecovery
from computer_agent.recovery.models import RecoveryStatus
from computer_agent.verification.action_verifier import ActionVerifier
from computer_agent.verification.models import (
    ActionVerificationStatus,
    StateVerificationStatus,
)


_DEFAULT_FRONTMOST_APP_SETTLE_TIMEOUT_SECONDS = 1.0
_DEFAULT_FRONTMOST_APP_SETTLE_POLL_SECONDS = 0.1


class AgentLoop:
    """Execute a structured plan through deterministic production layers."""

    def __init__(
        self,
        *,
        perception_engine: object,
        grounder: object | None = None,
        action_grounder: object | None = None,
        executor: object,
        verifier: object | None = None,
        recovery: object | None = None,
        state_verifier: object | None = None,
        allowed_app_names: Collection[str] | None = None,
        frontmost_app_settle_timeout_seconds: float = (
            _DEFAULT_FRONTMOST_APP_SETTLE_TIMEOUT_SECONDS
        ),
        frontmost_app_settle_poll_seconds: float = (
            _DEFAULT_FRONTMOST_APP_SETTLE_POLL_SECONDS
        ),
        settling_sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _require_method(
            perception_engine,
            "observe",
            "perception_engine",
        )

        _require_method(
            executor,
            "execute",
            "executor",
        )

        if grounder is None:
            grounder = UIGrounder()
        _require_method(
            grounder,
            "ground",
            "grounder",
        )

        if action_grounder is None:
            action_grounder = ActionGrounder()
        _require_method(
            action_grounder,
            "ground_click",
            "action_grounder",
        )

        if verifier is None:
            verifier = _default_verifier(grounder)
        _require_method(
            verifier,
            "verify_target_appeared",
            "verifier",
        )

        if recovery is None:
            recovery = _default_recovery(
                grounder,
                action_grounder,
            )
        _require_method(
            recovery,
            "prepare_retry",
            "recovery",
        )

        if state_verifier is not None:
            _require_method(
                state_verifier,
                "verify_frontmost_application",
                "state_verifier",
            )
            _require_method(
                state_verifier,
                "verify_focused_editable_value",
                "state_verifier",
            )

        frontmost_app_settle_timeout_seconds = _validate_seconds(
            frontmost_app_settle_timeout_seconds,
            "frontmost_app_settle_timeout_seconds",
            allow_zero=True,
        )
        frontmost_app_settle_poll_seconds = _validate_seconds(
            frontmost_app_settle_poll_seconds,
            "frontmost_app_settle_poll_seconds",
            allow_zero=False,
        )
        if not callable(settling_sleep):
            raise ValueError("settling_sleep must be callable")

        self._perception_engine = perception_engine
        self._grounder = grounder
        self._action_grounder = action_grounder
        self._executor = executor
        self._verifier = verifier
        self._recovery = recovery
        self._state_verifier = state_verifier
        self._allowed_app_names = _normalize_allowed_app_names(
            allowed_app_names
        )
        self._frontmost_app_settle_timeout_seconds = (
            frontmost_app_settle_timeout_seconds
        )
        self._frontmost_app_settle_poll_seconds = (
            frontmost_app_settle_poll_seconds
        )
        self._settling_sleep = settling_sleep

    @property
    def perception_engine(self) -> object:
        """Return the perception engine used by this loop."""

        return self._perception_engine

    @property
    def grounder(self) -> object:
        """Return the UI grounder used for initial step grounding."""

        return self._grounder

    @property
    def action_grounder(self) -> object:
        """Return the action grounder used for initial click grounding."""

        return self._action_grounder

    @property
    def executor(self) -> object:
        """Return the executor used for prepared actions."""

        return self._executor

    @property
    def verifier(self) -> object:
        """Return the action verifier used by this loop."""

        return self._verifier

    @property
    def recovery(self) -> object:
        """Return the recovery policy used by this loop."""

        return self._recovery

    @property
    def state_verifier(self) -> object | None:
        """Return the state verifier used for direct semantic operations."""

        return self._state_verifier

    @property
    def allowed_app_names(self) -> frozenset[str]:
        """Return exact application names authorized for activation."""

        return self._allowed_app_names

    def run(self, plan: StructuredPlan) -> AgentLoopResult:
        """Run all plan steps until completion or a typed terminal outcome."""

        if not isinstance(plan, StructuredPlan):
            raise ValueError("plan must be a StructuredPlan")

        state = AgentState(user_task=plan.task_goal)
        state.start()
        completed_plan_steps = 0

        for step in plan.steps:
            if (
                isinstance(step, PlanStep)
                and step.operation is PlanOperation.CLICK_TARGET
            ):
                terminal_result = self._run_click_step(
                    plan=plan,
                    state=state,
                    step=step,
                    completed_plan_steps=completed_plan_steps,
                )
            elif (
                isinstance(step, ReadClipboardStep)
                and step.operation is PlanOperation.READ_CLIPBOARD
            ):
                terminal_result = self._run_read_clipboard_step(
                    plan=plan,
                    state=state,
                    step=step,
                    completed_plan_steps=completed_plan_steps,
                )
            elif (
                isinstance(step, ActivateAppStep)
                and step.operation is PlanOperation.ACTIVATE_APP
            ):
                terminal_result = self._run_activate_app_step(
                    plan=plan,
                    state=state,
                    step=step,
                    completed_plan_steps=completed_plan_steps,
                )
            elif (
                isinstance(step, InsertTextStep)
                and step.operation is PlanOperation.INSERT_TEXT
            ):
                terminal_result = self._run_insert_text_step(
                    plan=plan,
                    state=state,
                    step=step,
                    completed_plan_steps=completed_plan_steps,
                )
            else:
                raise RuntimeError(
                    f"unsupported plan operation: {step.operation}"
                )

            if terminal_result is not None:
                return terminal_result

            completed_plan_steps += 1

        state.succeed()
        return AgentLoopResult(
            status=AgentLoopStatus.COMPLETED,
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            reason="all plan steps completed",
        )

    def _run_click_step(
        self,
        *,
        plan: StructuredPlan,
        state: AgentState,
        step: PlanStep,
        completed_plan_steps: int,
    ) -> AgentLoopResult | None:
        before_snapshot = self._perception_engine.observe()

        grounding_result = self._grounder.ground(
            step.action_target,
            before_snapshot.fused_elements,
        )
        if grounding_result.status is not GroundingStatus.RESOLVED:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "initial grounding was "
                    f"{grounding_result.status.value}: "
                    f"{grounding_result.reason}"
                ),
            )

        action_grounding_result = self._action_grounder.ground_click(
            grounding_result,
            before_snapshot.frame.screen_size,
        )
        if (
            action_grounding_result.status
            is not ActionGroundingStatus.READY
        ):
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "initial action grounding was "
                    f"{action_grounding_result.status.value}: "
                    f"{action_grounding_result.reason}"
                ),
            )

        action = action_grounding_result.action
        if action is None:
            raise RuntimeError(
                "READY action grounding did not contain an action"
            )

        completed_attempts = 0

        while True:
            completed_attempts += 1
            tool_result = self._executor.execute(action)
            state.record_step(action, tool_result)

            after_snapshot = self._perception_engine.observe()
            verification_result = self._verifier.verify_target_appeared(
                action=action,
                tool_result=tool_result,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                target_spec=step.verification_target,
            )

            if (
                verification_result.status
                is ActionVerificationStatus.VERIFIED
            ):
                return None

            recovery_result = self._recovery.prepare_retry(
                verification_result=verification_result,
                tool_result=tool_result,
                target_spec=step.action_target,
                latest_snapshot=after_snapshot,
                completed_attempts=completed_attempts,
                max_attempts=step.max_attempts,
            )

            if recovery_result.status is RecoveryStatus.RETRY_READY:
                action_grounding_result = (
                    recovery_result.action_grounding_result
                )
                if action_grounding_result is None:
                    raise RuntimeError(
                        "RETRY_READY recovery did not contain "
                        "action grounding"
                    )

                action = action_grounding_result.action
                if action is None:
                    raise RuntimeError(
                        "RETRY_READY recovery did not contain an action"
                    )

                before_snapshot = after_snapshot
                continue

            if recovery_result.status is RecoveryStatus.BLOCKED:
                return _terminal_failure(
                    plan=plan,
                    state=state,
                    completed_plan_steps=completed_plan_steps,
                    status=AgentLoopStatus.BLOCKED,
                    reason=(
                        "recovery blocked retry: "
                        f"{recovery_result.reason}"
                    ),
                )

            if recovery_result.status is RecoveryStatus.EXHAUSTED:
                return _terminal_failure(
                    plan=plan,
                    state=state,
                    completed_plan_steps=completed_plan_steps,
                    status=AgentLoopStatus.EXHAUSTED,
                    reason=(
                        "recovery exhausted attempts: "
                        f"{recovery_result.reason}"
                    ),
                )

            if recovery_result.status is RecoveryStatus.NOT_NEEDED:
                raise RuntimeError(
                    "recovery returned NOT_NEEDED after failed verification"
                )

            raise RuntimeError(
                f"unsupported recovery status: {recovery_result.status}"
            )

    def _run_read_clipboard_step(
        self,
        *,
        plan: StructuredPlan,
        state: AgentState,
        step: ReadClipboardStep,
        completed_plan_steps: int,
    ) -> AgentLoopResult | None:
        values = state.context.get("values")
        if values is not None and not isinstance(values, dict):
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason="runtime values context is not a mapping",
            )

        last_failure_reason = "clipboard content was not verified"
        for _attempt_number in range(1, step.max_attempts + 1):
            action = Action(
                tool_name="read_from_clipboard",
                arguments={},
                reason="Read and verify clipboard text for runtime context",
            )
            tool_result = self._executor.execute(action)
            state.record_step(action, tool_result)

            verified_text, failure_reason = _verified_clipboard_text(
                tool_result,
                step.expected_text,
            )
            if verified_text is not None:
                values = _runtime_values_mapping(
                    state,
                    create=True,
                )
                if values is None:
                    return _terminal_failure(
                        plan=plan,
                        state=state,
                        completed_plan_steps=completed_plan_steps,
                        status=AgentLoopStatus.BLOCKED,
                        reason="runtime values context is not a mapping",
                    )

                values[step.value_key] = verified_text
                return None

            last_failure_reason = failure_reason

        return _terminal_failure(
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            status=AgentLoopStatus.EXHAUSTED,
            reason=(
                "clipboard read verification exhausted attempts: "
                f"{last_failure_reason}"
            ),
        )

    def _run_activate_app_step(
        self,
        *,
        plan: StructuredPlan,
        state: AgentState,
        step: ActivateAppStep,
        completed_plan_steps: int,
    ) -> AgentLoopResult | None:
        if step.app_name not in self._allowed_app_names:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "application activation is not authorized for "
                    f"{step.app_name}"
                ),
            )

        if self._state_verifier is None:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason="state verifier is required for application activation",
            )

        last_failure_reason = "application activation was not verified"
        for _attempt_number in range(1, step.max_attempts + 1):
            action = Action(
                tool_name="activate_app",
                arguments={"app_name": step.app_name},
                reason="Activate allowlisted application for semantic step",
            )
            tool_result = self._executor.execute(action)
            state.record_step(action, tool_result)

            if not tool_result.success:
                last_failure_reason = (
                    "application activation tool failed: "
                    f"{tool_result.error}"
                )
                continue

            verification_result = (
                self._verify_frontmost_application_with_settling(
                    step.app_name
                )
            )
            if (
                verification_result.status
                is StateVerificationStatus.VERIFIED
            ):
                return None

            last_failure_reason = (
                "frontmost application verification was "
                f"{verification_result.status.value}: "
                f"{verification_result.reason}"
            )

        return _terminal_failure(
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            status=AgentLoopStatus.EXHAUSTED,
            reason=(
                "application activation exhausted attempts: "
                f"{last_failure_reason}"
            ),
        )

    def _verify_frontmost_application_with_settling(
        self,
        app_name: str,
    ):
        if self._state_verifier is None:
            raise RuntimeError(
                "frontmost application settling requires state verifier"
            )

        verification_result = (
            self._state_verifier.verify_frontmost_application(app_name)
        )
        if verification_result.status is StateVerificationStatus.VERIFIED:
            return verification_result

        remaining_seconds = self._frontmost_app_settle_timeout_seconds
        while remaining_seconds > 0.0:
            sleep_seconds = min(
                self._frontmost_app_settle_poll_seconds,
                remaining_seconds,
            )
            self._settling_sleep(sleep_seconds)
            remaining_seconds = max(
                0.0,
                remaining_seconds - sleep_seconds,
            )

            verification_result = (
                self._state_verifier.verify_frontmost_application(app_name)
            )
            if (
                verification_result.status
                is StateVerificationStatus.VERIFIED
            ):
                return verification_result

        return verification_result

    def _run_insert_text_step(
        self,
        *,
        plan: StructuredPlan,
        state: AgentState,
        step: InsertTextStep,
        completed_plan_steps: int,
    ) -> AgentLoopResult | None:
        if step.max_attempts != 1:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason="insert_text requires max_attempts of exactly 1",
            )

        if self._state_verifier is None:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason="state verifier is required for text insertion",
            )

        values = _runtime_values_mapping(
            state,
            create=False,
        )
        if values is None:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason="runtime values context is missing or not a mapping",
            )

        value = values.get(step.value_key)
        if not isinstance(value, str) or not value.strip():
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "runtime value is missing or not a non-empty string "
                    f"for key {step.value_key}"
                ),
            )

        action = Action(
            tool_name="paste_text",
            arguments={"text": value},
            reason="Insert verified runtime text into focused application",
        )
        tool_result = self._executor.execute(action)
        state.record_step(action, tool_result)

        if not tool_result.success:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.EXHAUSTED,
                reason=(
                    "text insertion tool failed: "
                    f"{tool_result.error}"
                ),
            )

        after_snapshot = self._perception_engine.observe()
        verification_result = (
            self._state_verifier.verify_focused_editable_value(
                after_snapshot,
                value,
            )
        )
        if verification_result.status is StateVerificationStatus.VERIFIED:
            return None

        return _terminal_failure(
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            status=AgentLoopStatus.EXHAUSTED,
            reason=(
                "focused editable verification was "
                f"{verification_result.status.value}: "
                f"{verification_result.reason}"
            ),
        )


def _terminal_failure(
    *,
    plan: StructuredPlan,
    state: AgentState,
    completed_plan_steps: int,
    status: AgentLoopStatus,
    reason: str,
) -> AgentLoopResult:
    state.fail(reason)
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=state,
        completed_plan_steps=completed_plan_steps,
        reason=reason,
    )


def _verified_clipboard_text(
    tool_result: ToolResult,
    expected_text: str,
) -> tuple[str | None, str]:
    if not tool_result.success:
        return (
            None,
            f"clipboard read tool failed: {tool_result.error}",
        )

    if not isinstance(tool_result.output, dict):
        return (
            None,
            "clipboard output was not a dict",
        )

    text = tool_result.output.get("text")
    if not isinstance(text, str):
        return (
            None,
            "clipboard output text was not a string",
        )

    if text != expected_text:
        return (
            None,
            "clipboard text did not match expected value",
        )

    return text, "clipboard text matched expected value"


def _runtime_values_mapping(
    state: AgentState,
    *,
    create: bool,
) -> dict | None:
    values = state.context.get("values")
    if values is None:
        if not create:
            return None

        values = {}
        state.context["values"] = values
        return values

    if not isinstance(values, dict):
        return None

    return values


def _normalize_allowed_app_names(
    allowed_app_names: Collection[str] | None,
) -> frozenset[str]:
    if allowed_app_names is None:
        return frozenset()

    if isinstance(allowed_app_names, (str, bytes)):
        raise ValueError(
            "allowed_app_names must be a collection of non-empty strings"
        )

    names = []
    try:
        iterator = iter(allowed_app_names)
    except TypeError as error:
        raise ValueError(
            "allowed_app_names must be a collection of non-empty strings"
        ) from error

    for name in iterator:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "allowed_app_names must contain only non-empty strings"
            )

        names.append(name)

    return frozenset(names)


def _validate_seconds(
    value: object,
    field_name: str,
    *,
    allow_zero: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")

    seconds = float(value)
    if not math.isfinite(seconds):
        raise ValueError(f"{field_name} must be finite")

    if allow_zero:
        if seconds < 0.0:
            raise ValueError(f"{field_name} must be non-negative")
    elif seconds <= 0.0:
        raise ValueError(f"{field_name} must be positive")

    return seconds


def _require_method(
    dependency: object,
    method_name: str,
    dependency_name: str,
) -> None:
    if dependency is None:
        raise ValueError(f"{dependency_name} is required")

    method = getattr(
        dependency,
        method_name,
        None,
    )
    if not callable(method):
        raise ValueError(
            f"{dependency_name} must provide {method_name}()"
        )


def _default_verifier(grounder: object) -> ActionVerifier:
    if not isinstance(grounder, UIGrounder):
        raise ValueError(
            "custom grounder requires explicit verifier"
        )

    return ActionVerifier(grounder=grounder)


def _default_recovery(
    grounder: object,
    action_grounder: object,
) -> ActionRecovery:
    if not isinstance(grounder, UIGrounder):
        raise ValueError(
            "custom grounder requires explicit recovery"
        )

    if not isinstance(action_grounder, ActionGrounder):
        raise ValueError(
            "custom action_grounder requires explicit recovery"
        )

    return ActionRecovery(
        grounder=grounder,
        action_grounder=action_grounder,
    )
