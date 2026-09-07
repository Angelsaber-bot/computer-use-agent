"""Deterministic single-field text input workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from computer_agent.core.models import Action, ToolResult
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

        if precondition_failure is not None and execute:
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

        focus_action = action_grounding.action
        if focus_action is None:
            raise RuntimeError("READY focus grounding contained no Action")

        focus_result = executor.execute(focus_action)
        if not isinstance(focus_result, ToolResult):
            raise ValueError("executor must return ToolResult objects")

        if not focus_result.success:
            return self._result(
                TextInputStatus.ACTION_FAILED,
                "focus action failed",
                before,
                before_grounding,
                action_grounding,
                tool_results=(focus_result,),
            )

        type_action = Action(
            tool_name="type_text",
            arguments={
                "text": self._input_text,
            },
            reason="type deterministic test text into resolved field",
        )
        type_result = executor.execute(type_action)
        if not isinstance(type_result, ToolResult):
            raise ValueError("executor must return ToolResult objects")

        tool_results = (
            focus_result,
            type_result,
        )

        if not type_result.success:
            return self._result(
                TextInputStatus.ACTION_FAILED,
                "type action failed",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
            )

        self._sleeper(self._stabilization_wait_seconds)

        after = self._observe()
        if not self._trusted_observation(after):
            return self._result(
                TextInputStatus.BLOCKED,
                "post-action observation cannot be trusted",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
                after_observation=after,
            )

        after_grounding = self._ground(after)

        if after_grounding.status is GroundingStatus.AMBIGUOUS:
            return self._result(
                TextInputStatus.BLOCKED,
                "post-action field grounding was ambiguous",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
                after_observation=after,
                after_grounding=after_grounding,
            )

        if after_grounding.status is not GroundingStatus.RESOLVED:
            return self._result(
                TextInputStatus.VERIFICATION_FAILED,
                "post-action field did not resolve",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
                after_observation=after,
                after_grounding=after_grounding,
            )

        if not _element_value_matches(
            after_grounding.element.value,
            self._input_text,
        ):
            return self._result(
                TextInputStatus.VERIFICATION_FAILED,
                "post-action field value did not match requested input",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
                after_observation=after,
                after_grounding=after_grounding,
            )

        if _element_value_matches(
            before_grounding.element.value,
            after_grounding.element.value,
        ):
            return self._result(
                TextInputStatus.VERIFICATION_FAILED,
                "post-action field value did not change",
                before,
                before_grounding,
                action_grounding,
                tool_results=tool_results,
                after_observation=after,
                after_grounding=after_grounding,
            )

        return self._result(
            TextInputStatus.VERIFIED,
            "post-action field value matched requested input",
            before,
            before_grounding,
            action_grounding,
            tool_results=tool_results,
            after_observation=after,
            after_grounding=after_grounding,
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

        if _value_is_non_empty(element.value):
            return "target field is not empty before typing"

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
        )


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
    if value is None:
        return False

    return normalize_ui_text(str(value)) == normalize_ui_text(str(expected))
