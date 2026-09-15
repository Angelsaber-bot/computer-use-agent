"""Deterministic single-field text input workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from time import perf_counter

from computer_agent.core.models import Action, ToolResult
from computer_agent.agent.web_recovery import (
    WebRecoveryDecision,
    decide_failed_grounding_recovery,
)
from computer_agent.grounding import (
    ActionGrounder,
    ActionGroundingResult,
    ActionGroundingStatus,
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    PerceptionSnapshot,
    SemanticAXElement,
    Viewport,
    ViewportSearchController,
    ViewportSearchObservation,
    ViewportSearchPolicy,
    ViewportSearchStatus,
)
from computer_agent.perception.fusion import normalize_ui_text


class TextInputStatus(str, Enum):
    """Outcomes for one deterministic text-input attempt."""

    VERIFIED = "verified"
    NEEDS_ACTION = "needs_action"
    BLOCKED = "blocked"
    ACTION_FAILED = "action_failed"
    VERIFICATION_FAILED = "verification_failed"
    FOCUS_FAILED = "focus_failed"
    CLEAR_FAILED = "clear_failed"
    INPUT_MISMATCH = "input_mismatch"
    TARGET_DISAPPEARED = "target_disappeared"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"
    RETRY_EXHAUSTED = "retry_exhausted"


class TextInputMechanism(str, Enum):
    """Supported concrete text delivery mechanisms."""

    CLIPBOARD_PASTE = "clipboard_paste"
    KEYBOARD_TYPING = "keyboard_typing"


@dataclass(frozen=True, slots=True)
class TextInputAttempt:
    """Structured evidence for one bounded text-entry attempt."""

    attempt_number: int
    mechanism: TextInputMechanism
    status: TextInputStatus
    reason: str
    intended_text: str
    observed_text: str | None
    exact_match: bool | None
    pre_entry_value: str | int | float | bool | None
    post_entry_value: str | int | float | bool | None
    focused: bool | None
    clear_verified: bool | None
    target_identity: str | None
    elapsed_ms: float
    tool_results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt_number, bool)
            or not isinstance(self.attempt_number, int)
            or self.attempt_number < 1
        ):
            raise ValueError("attempt_number must be a positive integer")

        if not isinstance(self.mechanism, TextInputMechanism):
            raise ValueError("mechanism must be a TextInputMechanism")

        if not isinstance(self.status, TextInputStatus):
            raise ValueError("status must be a TextInputStatus")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if not isinstance(self.intended_text, str):
            raise ValueError("intended_text must be a string")

        if self.observed_text is not None and not isinstance(
            self.observed_text,
            str,
        ):
            raise ValueError("observed_text must be a string or None")

        if self.exact_match is not None and type(self.exact_match) is not bool:
            raise ValueError("exact_match must be a bool or None")

        if self.focused is not None and type(self.focused) is not bool:
            raise ValueError("focused must be a bool or None")

        if self.clear_verified is not None and type(self.clear_verified) is not bool:
            raise ValueError("clear_verified must be a bool or None")

        if self.target_identity is not None and not isinstance(
            self.target_identity,
            str,
        ):
            raise ValueError("target_identity must be a string or None")

        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, (int, float))
            or self.elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms must be a non-negative number")

        if not isinstance(self.tool_results, tuple) or any(
            not isinstance(result, ToolResult)
            for result in self.tool_results
        ):
            raise ValueError(
                "tool_results must be a tuple of ToolResult objects"
            )


@dataclass(frozen=True, slots=True)
class TextInputObservation:
    """One fresh observation used by text-input workflow decisions."""

    application_name: str | None
    viewport: Viewport | None
    snapshot: PerceptionSnapshot
    semantic_elements: tuple[SemanticAXElement, ...] = ()

    def __post_init__(self) -> None:
        if self.application_name is not None and not isinstance(
            self.application_name,
            str,
        ):
            raise ValueError(
                "application_name must be a string or None"
            )

        if self.viewport is not None and not isinstance(
            self.viewport,
            Viewport,
        ):
            raise ValueError("viewport must be a Viewport or None")

        if not isinstance(self.snapshot, PerceptionSnapshot):
            raise ValueError("snapshot must be a PerceptionSnapshot")

        if not isinstance(self.semantic_elements, tuple) or any(
            not isinstance(element, SemanticAXElement)
            for element in self.semantic_elements
        ):
            raise ValueError(
                "semantic_elements must be a tuple of SemanticAXElement "
                "objects"
            )


@dataclass(frozen=True, slots=True)
class TextInputResult:
    """Result of one deterministic text-input workflow."""

    status: TextInputStatus
    reason: str
    before_observation: TextInputObservation
    before_grounding: GroundingResult
    action_grounding: ActionGroundingResult
    tool_results: tuple[ToolResult, ...] = ()
    after_observation: TextInputObservation | None = None
    after_grounding: GroundingResult | None = None
    attempts: tuple[TextInputAttempt, ...] = ()
    clipboard_restore_result: ToolResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, TextInputStatus):
            raise ValueError("status must be a TextInputStatus")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if not isinstance(
            self.before_observation,
            TextInputObservation,
        ):
            raise ValueError(
                "before_observation must be a TextInputObservation"
            )

        if not isinstance(self.before_grounding, GroundingResult):
            raise ValueError(
                "before_grounding must be a GroundingResult"
            )

        if not isinstance(
            self.action_grounding,
            ActionGroundingResult,
        ):
            raise ValueError(
                "action_grounding must be an ActionGroundingResult"
            )

        if not isinstance(self.tool_results, tuple) or any(
            not isinstance(result, ToolResult)
            for result in self.tool_results
        ):
            raise ValueError(
                "tool_results must be a tuple of ToolResult objects"
            )

        if self.after_observation is not None and not isinstance(
            self.after_observation,
            TextInputObservation,
        ):
            raise ValueError(
                "after_observation must be a TextInputObservation or None"
            )

        if self.after_grounding is not None and not isinstance(
            self.after_grounding,
            GroundingResult,
        ):
            raise ValueError(
                "after_grounding must be a GroundingResult or None"
            )

        if not isinstance(self.attempts, tuple) or any(
            not isinstance(attempt, TextInputAttempt)
            for attempt in self.attempts
        ):
            raise ValueError(
                "attempts must be a tuple of TextInputAttempt objects"
            )

        if self.clipboard_restore_result is not None and not isinstance(
            self.clipboard_restore_result,
            ToolResult,
        ):
            raise ValueError(
                "clipboard_restore_result must be a ToolResult or None"
            )


class TextInputController:
    """Safely focus, type into, and verify one real text field."""

    def __init__(
        self,
        *,
        observer: Callable[[], TextInputObservation],
        target_spec: TargetSpec,
        input_text: str,
        expected_application_name: str = "Google Chrome",
        viewport_search_policy: ViewportSearchPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
        stabilization_wait_seconds: float = 0.5,
        max_attempts: int = 2,
        primary_mechanism: TextInputMechanism = (
            TextInputMechanism.CLIPBOARD_PASTE
        ),
    ) -> None:
        if not callable(observer):
            raise ValueError("observer must be callable")

        if not isinstance(target_spec, TargetSpec):
            raise ValueError("target_spec must be a TargetSpec")

        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("input_text must be a non-empty string")

        if (
            not isinstance(expected_application_name, str)
            or not expected_application_name.strip()
        ):
            raise ValueError(
                "expected_application_name must be a non-empty string"
            )

        if viewport_search_policy is not None and not isinstance(
            viewport_search_policy,
            ViewportSearchPolicy,
        ):
            raise ValueError(
                "viewport_search_policy must be a ViewportSearchPolicy "
                "or None"
            )

        if sleeper is not None and not callable(sleeper):
            raise ValueError("sleeper must be callable or None")

        if not isinstance(stabilization_wait_seconds, (int, float)):
            raise ValueError(
                "stabilization_wait_seconds must be numeric"
            )

        if stabilization_wait_seconds < 0:
            raise ValueError(
                "stabilization_wait_seconds must be non-negative"
            )

        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 3
        ):
            raise ValueError("max_attempts must be an integer from 1 through 3")

        if not isinstance(primary_mechanism, TextInputMechanism):
            raise ValueError(
                "primary_mechanism must be a TextInputMechanism"
            )

        self._observer = observer
        self._target_spec = target_spec
        self._input_text = input_text
        self._expected_application_name = expected_application_name
        self._viewport_search_policy = (
            viewport_search_policy or ViewportSearchPolicy()
        )
        self._sleeper = sleeper or (lambda _seconds: None)
        self._stabilization_wait_seconds = float(
            stabilization_wait_seconds
        )
        self._max_attempts = max_attempts
        self._primary_mechanism = primary_mechanism
        self._ui_grounder = UIGrounder()
        self._action_grounder = ActionGrounder()

    def run(
        self,
        *,
        execute: bool,
        executor: object | None = None,
    ) -> TextInputResult:
        """Run dry-run safety checks or execute one verified text input."""

        if not isinstance(execute, bool):
            raise ValueError("execute must be a bool")

        before = self._observe()
        before_grounding = self._ground(before)
        action_grounding = self._ground_focus_action(
            before,
            before_grounding,
        )

        precondition_failure = self._precondition_failure(
            before,
            before_grounding,
            action_grounding,
        )

        if (
            precondition_failure is not None
            and execute
            and self._trusted_observation(before)
            and decide_failed_grounding_recovery(
                before_grounding
            ).decision is WebRecoveryDecision.VIEWPORT_SEARCH
        ):
            search_result = self._search_if_appropriate(
                before,
                executor,
            )
            if search_result is not None:
                if search_result.status is ViewportSearchStatus.FOUND:
                    before = self._observe()
                    before_grounding = self._ground(before)
                    action_grounding = self._ground_focus_action(
                        before,
                        before_grounding,
                    )
                    precondition_failure = self._precondition_failure(
                        before,
                        before_grounding,
                        action_grounding,
                    )
                elif search_result.status in (
                    ViewportSearchStatus.BLOCKED,
                    ViewportSearchStatus.STALLED,
                    ViewportSearchStatus.EXHAUSTED,
                ):
                    return self._result(
                        TextInputStatus.BLOCKED,
                        "viewport search did not find visible target: "
                        f"{search_result.status.value}",
                        before,
                        before_grounding,
                        action_grounding,
                    )

        if precondition_failure is not None:
            return self._result(
                TextInputStatus.BLOCKED,
                precondition_failure,
                before,
                before_grounding,
                action_grounding,
            )

        if not execute:
            return self._result(
                TextInputStatus.NEEDS_ACTION,
                "text input could execute safely",
                before,
                before_grounding,
                action_grounding,
            )

        if executor is None or not hasattr(executor, "execute"):
            raise ValueError(
                "executor with an execute method is required in execute mode"
            )

        saved_clipboard, clipboard_read_result = (
            self._read_clipboard_if_possible(executor)
        )
        attempts: list[TextInputAttempt] = []
        all_tool_results: list[ToolResult] = []
        if clipboard_read_result is not None:
            all_tool_results.append(clipboard_read_result)
        after: TextInputObservation | None = None
        after_grounding: GroundingResult | None = None

        for attempt_number in range(1, self._max_attempts + 1):
            attempt_result = self._run_entry_attempt(
                attempt_number=attempt_number,
                executor=executor,
                initial_observation=before
                if attempt_number == 1
                else None,
                initial_grounding=before_grounding
                if attempt_number == 1
                else None,
                initial_action_grounding=action_grounding
                if attempt_number == 1
                else None,
            )
            attempt = attempt_result.attempt
            attempts.append(attempt)
            all_tool_results.extend(attempt.tool_results)
            after = attempt_result.after_observation
            after_grounding = attempt_result.after_grounding

            if attempt.status is TextInputStatus.VERIFIED:
                restore_result = self._restore_clipboard_if_possible(
                    executor,
                    saved_clipboard,
                )
                if restore_result is not None:
                    all_tool_results.append(restore_result)
                return self._result(
                    TextInputStatus.VERIFIED,
                    "post-action field value exactly matched requested input",
                    before,
                    before_grounding,
                    action_grounding,
                    tool_results=tuple(all_tool_results),
                    after_observation=after,
                    after_grounding=after_grounding,
                    attempts=tuple(attempts),
                    clipboard_restore_result=restore_result,
                )

            if attempt.status in (
                TextInputStatus.BLOCKED,
                TextInputStatus.ACTION_FAILED,
                TextInputStatus.FOCUS_FAILED,
                TextInputStatus.VERIFICATION_UNAVAILABLE,
            ):
                break

        restore_result = self._restore_clipboard_if_possible(
            executor,
            saved_clipboard,
        )
        if restore_result is not None:
            all_tool_results.append(restore_result)

        final_status = (
            attempts[-1].status
            if attempts
            else TextInputStatus.RETRY_EXHAUSTED
        )
        if final_status in (
            TextInputStatus.INPUT_MISMATCH,
            TextInputStatus.CLEAR_FAILED,
            TextInputStatus.TARGET_DISAPPEARED,
        ):
            final_status = TextInputStatus.RETRY_EXHAUSTED

        return self._result(
            final_status,
            (
                "text entry exhausted bounded recovery attempts: "
                f"{attempts[-1].reason if attempts else 'no attempts ran'}"
            ),
            before,
            before_grounding,
            action_grounding,
            tool_results=tuple(all_tool_results),
            after_observation=after,
            after_grounding=after_grounding,
            attempts=tuple(attempts),
            clipboard_restore_result=restore_result,
        )

    def _run_entry_attempt(
        self,
        *,
        attempt_number: int,
        executor: object,
        initial_observation: TextInputObservation | None = None,
        initial_grounding: GroundingResult | None = None,
        initial_action_grounding: ActionGroundingResult | None = None,
    ) -> "_AttemptResult":
        started = perf_counter()
        tool_results: list[ToolResult] = []
        context = _AttemptContext()

        before = initial_observation or self._observe()
        context.after_observation = before
        before_grounding = initial_grounding or self._ground(before)
        context.after_grounding = before_grounding
        if before_grounding.element is not None:
            context.pre_entry_value = before_grounding.element.value
        action_grounding = (
            initial_action_grounding
            or self._ground_focus_action(
                before,
                before_grounding,
            )
        )
        failure = self._precondition_failure(
            before,
            before_grounding,
            action_grounding,
        )
        if failure is not None:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.BLOCKED,
                failure,
                started,
                context,
                tool_results,
                focused=None,
                clear_verified=None,
            )

        release_result = _execute_action(
            executor,
            Action(
                tool_name="release_modifier_keys",
                arguments={},
                reason="Release modifier keys before verified text entry",
            ),
        )
        tool_results.append(release_result)
        if not release_result.success:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.ACTION_FAILED,
                "modifier-key release failed",
                started,
                context,
                tool_results,
                focused=None,
                clear_verified=None,
            )

        focus_action = action_grounding.action
        if focus_action is None:
            raise RuntimeError("READY focus grounding contained no Action")

        focus_result = _execute_action(executor, focus_action)
        tool_results.append(focus_result)
        if not focus_result.success:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.FOCUS_FAILED,
                "focus action failed",
                started,
                context,
                tool_results,
                focused=False,
                clear_verified=None,
            )

        focused_observation = self._observe()
        context.after_observation = focused_observation
        focused_grounding = self._ground(focused_observation)
        context.after_grounding = focused_grounding
        if focused_grounding.status is not GroundingStatus.RESOLVED:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.TARGET_DISAPPEARED,
                "target did not re-ground after focus",
                started,
                context,
                tool_results,
                focused=None,
                clear_verified=None,
            )

        focused = focused_grounding.element.focused
        if focused is False:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.FOCUS_FAILED,
                "target did not report focus after focus action",
                started,
                context,
                tool_results,
                focused=False,
                clear_verified=None,
            )

        for action in (
            Action(
                tool_name="hotkey",
                arguments={"keys": ["command", "a"]},
                reason="Select existing text before verified text entry",
            ),
            Action(
                tool_name="press_key",
                arguments={"key": "delete"},
                reason="Clear selected text before verified text entry",
            ),
        ):
            result = _execute_action(executor, action)
            tool_results.append(result)
            if not result.success:
                return self._attempt_result(
                    attempt_number,
                    TextInputStatus.CLEAR_FAILED,
                    f"{action.tool_name} clear action failed",
                    started,
                    context,
                    tool_results,
                    focused=focused,
                    clear_verified=False,
                )

        cleared_observation = self._observe()
        context.after_observation = cleared_observation
        cleared_grounding = self._ground(cleared_observation)
        context.after_grounding = cleared_grounding
        if cleared_grounding.status is not GroundingStatus.RESOLVED:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.TARGET_DISAPPEARED,
                "target did not re-ground after clear",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=None,
            )

        clear_verified = _element_value_matches(
            cleared_grounding.element.value,
            "",
        )
        if not clear_verified:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.CLEAR_FAILED,
                "target value was not empty after clear",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=False,
            )

        input_result = _execute_action(
            executor,
            self._input_action(),
        )
        tool_results.append(input_result)
        if not input_result.success:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.ACTION_FAILED,
                "text entry action failed",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )

        self._sleeper(self._stabilization_wait_seconds)

        after = self._observe()
        context.after_observation = after
        if not self._trusted_observation(after):
            return self._attempt_result(
                attempt_number,
                TextInputStatus.BLOCKED,
                "post-action observation cannot be trusted",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )

        after_grounding = self._ground(after)
        context.after_grounding = after_grounding
        if after_grounding.status is GroundingStatus.AMBIGUOUS:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.BLOCKED,
                "post-action field grounding was ambiguous",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )
        if after_grounding.status is not GroundingStatus.RESOLVED:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.TARGET_DISAPPEARED,
                "post-action field did not resolve",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )

        value = after_grounding.element.value
        if value is None:
            return self._attempt_result(
                attempt_number,
                TextInputStatus.VERIFICATION_UNAVAILABLE,
                "post-action AXValue was unavailable",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )

        if _element_value_matches(value, self._input_text):
            return self._attempt_result(
                attempt_number,
                TextInputStatus.VERIFIED,
                "post-action field value exactly matched requested input",
                started,
                context,
                tool_results,
                focused=focused,
                clear_verified=True,
            )

        return self._attempt_result(
            attempt_number,
            TextInputStatus.INPUT_MISMATCH,
            "post-action field value did not exactly match requested input",
            started,
            context,
            tool_results,
            focused=focused,
            clear_verified=True,
        )

    def _input_action(self) -> Action:
        if self._primary_mechanism is TextInputMechanism.CLIPBOARD_PASTE:
            return Action(
                tool_name="paste_text",
                arguments={"text": self._input_text},
                reason="Paste intended text into focused verified field",
            )

        return Action(
            tool_name="type_text",
            arguments={"text": self._input_text},
            reason="Type intended text into focused verified field",
        )

    def _read_clipboard_if_possible(
        self,
        executor: object,
    ) -> tuple[str | None, ToolResult | None]:
        if self._primary_mechanism is not TextInputMechanism.CLIPBOARD_PASTE:
            return None, None

        result = _execute_action(
            executor,
            Action(
                tool_name="read_from_clipboard",
                arguments={},
                reason="Preserve clipboard before verified text paste",
            ),
        )
        if not result.success or not isinstance(result.output, dict):
            return None, result

        text = result.output.get("text")
        return text if isinstance(text, str) else None, result

    def _restore_clipboard_if_possible(
        self,
        executor: object,
        saved_clipboard: str | None,
    ) -> ToolResult | None:
        if (
            self._primary_mechanism is not TextInputMechanism.CLIPBOARD_PASTE
            or saved_clipboard is None
        ):
            return None

        return _execute_action(
            executor,
            Action(
                tool_name="copy_to_clipboard",
                arguments={"text": saved_clipboard},
                reason="Restore clipboard after verified text paste",
            ),
        )

    def _attempt_result(
        self,
        attempt_number: int,
        status: TextInputStatus,
        reason: str,
        started: float,
        context: "_AttemptContext",
        tool_results: list[ToolResult],
        *,
        focused: bool | None,
        clear_verified: bool | None,
    ) -> "_AttemptResult":
        grounding = context.after_grounding
        element = grounding.element if grounding is not None else None
        attempt = TextInputAttempt(
            attempt_number=attempt_number,
            mechanism=self._primary_mechanism,
            status=status,
            reason=reason,
            intended_text=self._input_text,
            observed_text=_observed_text(element.value if element else None),
            exact_match=_exact_match_if_observable(
                element.value if element else None,
                self._input_text,
            ),
            pre_entry_value=context.pre_entry_value,
            post_entry_value=element.value if element else None,
            focused=focused,
            clear_verified=clear_verified,
            target_identity=_target_identity(element),
            elapsed_ms=(perf_counter() - started) * 1000,
            tool_results=tuple(tool_results),
        )
        return _AttemptResult(
            attempt=attempt,
            after_observation=context.after_observation,
            after_grounding=context.after_grounding,
        )

    def _observe(self) -> TextInputObservation:
        observation = self._observer()
        if not isinstance(observation, TextInputObservation):
            raise ValueError(
                "observer must return TextInputObservation objects"
            )

        return observation

    def _ground(
        self,
        observation: TextInputObservation,
    ) -> GroundingResult:
        return self._ui_grounder.ground(
            self._target_spec,
            observation.snapshot.fused_elements,
            viewport=observation.viewport.bounds
            if observation.viewport is not None
            else None,
        )

    def _ground_focus_action(
        self,
        observation: TextInputObservation,
        grounding: GroundingResult,
    ) -> ActionGroundingResult:
        return self._action_grounder.ground_click(
            grounding,
            observation.snapshot.frame.screen_size,
        )

    def _precondition_failure(
        self,
        observation: TextInputObservation,
        grounding: GroundingResult,
        action_grounding: ActionGroundingResult,
    ) -> str | None:
        if not self._trusted_observation(observation):
            return "observation cannot be trusted"

        if grounding.status is not GroundingStatus.RESOLVED:
            return f"target grounding was {grounding.status.value}"

        element = grounding.element
        if element is None:
            return "resolved grounding had no element"

        if normalize_ui_text(element.element_type) not in {
            "text field",
            "text area",
        }:
            return "target role is not compatible with text input"

        if action_grounding.status is not ActionGroundingStatus.READY:
            return f"focus action was blocked: {action_grounding.reason}"

        return None

    def _trusted_observation(
        self,
        observation: TextInputObservation,
    ) -> bool:
        return (
            observation.application_name == self._expected_application_name
            and observation.viewport is not None
            and not observation.snapshot.warnings
        )

    def _search_if_appropriate(
        self,
        observation: TextInputObservation,
        executor: object | None,
    ):
        if executor is None:
            return None

        search_controller = ViewportSearchController(
            observer=lambda: _viewport_search_observation(
                self._observe()
            ),
            target_role=_raw_role_from_target(self._target_spec),
            target_text=self._target_spec.text or "",
            policy=self._viewport_search_policy,
            sleeper=self._sleeper,
        )

        search_diagnosis = search_controller.search(
            execute=True,
            executor=executor,
        )

        if len(search_diagnosis.observations) <= 1:
            return None

        return search_diagnosis

    @staticmethod
    def _result(
        status: TextInputStatus,
        reason: str,
        before_observation: TextInputObservation,
        before_grounding: GroundingResult,
        action_grounding: ActionGroundingResult,
        *,
        tool_results: tuple[ToolResult, ...] = (),
        after_observation: TextInputObservation | None = None,
        after_grounding: GroundingResult | None = None,
        attempts: tuple[TextInputAttempt, ...] = (),
        clipboard_restore_result: ToolResult | None = None,
    ) -> TextInputResult:
        return TextInputResult(
            status=status,
            reason=reason,
            before_observation=before_observation,
            before_grounding=before_grounding,
            action_grounding=action_grounding,
            tool_results=tool_results,
            after_observation=after_observation,
            after_grounding=after_grounding,
            attempts=attempts,
            clipboard_restore_result=clipboard_restore_result,
        )


@dataclass(slots=True)
class _AttemptContext:
    after_observation: TextInputObservation | None = None
    after_grounding: GroundingResult | None = None
    pre_entry_value: str | int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class _AttemptResult:
    attempt: TextInputAttempt
    after_observation: TextInputObservation | None
    after_grounding: GroundingResult | None


def _viewport_search_observation(
    observation: TextInputObservation,
) -> ViewportSearchObservation:
    return ViewportSearchObservation(
        application_name=observation.application_name,
        viewport=observation.viewport,
        semantic_elements=observation.semantic_elements,
    )


def _raw_role_from_target(target_spec: TargetSpec) -> str:
    element_types = {
        normalize_ui_text(element_type)
        for element_type in target_spec.element_types
    }
    if "text area" in element_types:
        return "AXTextArea"

    return "AXTextField"


def _value_is_non_empty(value: object) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    return True


def _element_value_matches(
    value: object,
    expected: object,
) -> bool:
    return isinstance(value, str) and value == str(expected)


def _execute_action(
    executor: object,
    action: Action,
) -> ToolResult:
    result = executor.execute(action)
    if not isinstance(result, ToolResult):
        raise ValueError("executor must return ToolResult objects")
    return result


def _observed_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _exact_match_if_observable(
    value: object,
    expected: str,
) -> bool | None:
    if not isinstance(value, str):
        return None
    return value == expected


def _target_identity(element: object) -> str | None:
    if element is None:
        return None

    identifier = getattr(element, "identifier", None)
    if isinstance(identifier, str) and identifier.strip():
        return f"id:{identifier.strip()}"

    text = getattr(element, "text", None)
    element_type = getattr(element, "element_type", None)
    if isinstance(text, str) and text.strip() and isinstance(
        element_type,
        str,
    ):
        return f"{element_type}:{text.strip()}"

    if isinstance(element_type, str):
        return element_type

    return None
