"""Provider-neutral LLM reasoner for semantic structured plans."""

from __future__ import annotations

import json
from typing import Any

from computer_agent.grounding.models import TargetSpec
from computer_agent.planning.models import (
    ActivateAppStep,
    InsertTextStep,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    SemanticPlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from computer_agent.planning.structured_planner import StructuredPlanner
from computer_agent.reasoning.llm_client import LLMClient
from computer_agent.reasoning.models import (
    SUPPORTED_REASONING_ELEMENT_TYPES,
    ReasoningResult,
    ReasoningStatus,
)


SYSTEM_PROMPT = f"""
Convert the user's task intent into a semantic UI plan.
Return JSON only, with no markdown, comments, or prose.
The top-level object must contain exactly:
{{"task_goal": string, "steps": array}}
The top-level task_goal must copy the caller's task intent verbatim.
Do not summarize, paraphrase, shorten, or rewrite task_goal.
Only supported semantic operations are allowed: click_target, read_clipboard, activate_app, insert_text, type_into_target.
Each step must use exactly one operation shape:
click_target: {{"goal": string, "operation": "click_target", "action_target": target, "verification_target": target, "max_attempts": 1}}
read_clipboard: {{"goal": string, "operation": "read_clipboard", "value_key": string, "expected_text": string, "max_attempts": integer}}
activate_app: {{"goal": string, "operation": "activate_app", "app_name": string, "max_attempts": integer}}
insert_text: {{"goal": string, "operation": "insert_text", "value_key": string, "max_attempts": integer}}
type_into_target: {{"goal": string, "operation": "type_into_target", "target": target, "input_text": string, "max_attempts": 1}}
Each target object must contain exactly:
{{"text": string, "element_types": array}}
element_types may contain only: {", ".join(SUPPORTED_REASONING_ELEMENT_TYPES)}.
element_types are semantic UI roles, not descriptions of visible content.
Do not choose "text" merely because a string is visible on screen.
heading is a semantic role for a page, section, or article heading.
Use heading when the caller explicitly asks to verify or target a heading or title represented semantically as a heading.
Do not use heading merely because text is visually prominent.
Only specify an element role when the task intent explicitly states it or strongly semantically implies it.
If the role is not known from the task intent, use an empty element_types array.
Appearance verification such as "verify that 'Results' appears" should use "element_types": [] unless the role was explicitly known from the task intent.
Never invent a semantic role in order to make a target more specific.
click_target identifies semantic UI targets, never coordinates.
click_target max_attempts must be exactly 1 unless the caller explicitly requests different retry behavior.
read_clipboard stores the clipboard value into value_key.
Use activate_app only when the task explicitly asks to activate, switch to, focus, or otherwise requires application activation.
Merely mentioning a website, browser-based task, or webpage does not imply activate_app.
Do not add activate_app as preparatory housekeeping.
activate_app identifies only the application name.
insert_text consumes a previously stored runtime value by value_key and must never contain literal text to type.
Use type_into_target when the task requires entering literal caller-visible text into a semantic UI target.
type_into_target input_text is literal text only, not a command or tool call.
type_into_target max_attempts must be exactly 1.
Never emit executable Action objects, tool names, raw tool arguments, coordinates, hotkeys, shell commands, AppleScript, or bundle IDs.
Do not claim to observe the screen, operate the computer, or execute the task.
""".strip()

_USER_PROMPT_PREFIX = "Task intent:"
_TOP_LEVEL_KEYS = frozenset(("task_goal", "steps"))
_SUPPORTED_ELEMENT_TYPES = frozenset(SUPPORTED_REASONING_ELEMENT_TYPES)
_CLICK_TARGET_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "action_target",
        "verification_target",
        "max_attempts",
    )
)
_READ_CLIPBOARD_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "value_key",
        "expected_text",
        "max_attempts",
    )
)
_ACTIVATE_APP_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "app_name",
        "max_attempts",
    )
)
_INSERT_TEXT_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "value_key",
        "max_attempts",
    )
)
_TYPE_INTO_TARGET_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "target",
        "input_text",
        "max_attempts",
    )
)
_TARGET_KEYS = frozenset(("text", "element_types"))


class LLMReasoner:
    """Convert natural-language task intent into a trusted structured plan."""

    def __init__(
        self,
        *,
        client: LLMClient,
        planner: StructuredPlanner | None = None,
    ) -> None:
        self._client = client
        self._planner = planner if planner is not None else StructuredPlanner()

    @property
    def system_prompt(self) -> str:
        """Return the deterministic system prompt owned by this reasoner."""

        return SYSTEM_PROMPT

    @property
    def planner(self) -> StructuredPlanner:
        """Return the configured structured planner."""

        return self._planner

    def reason(self, task: str) -> ReasoningResult:
        """Return a structured plan, or fail closed when reasoning is unsafe."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        try:
            response = self._client.generate(
                system_prompt=self._system_prompt,
                user_prompt=_build_user_prompt(task),
            )
        except Exception:
            return _blocked("LLM generation failed")

        try:
            plan = self._build_plan_from_response(response)
            return ReasoningResult(
                status=ReasoningStatus.READY,
                plan=plan,
                reason="structured plan ready",
            )
        except ValueError:
            return _blocked(
                "LLM response could not be converted into a structured plan"
            )

    @property
    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def _build_plan_from_response(self, response: object) -> StructuredPlan:
        if not isinstance(response, str):
            raise _ReasoningResponseError("response must be a string")

        if not response.strip():
            raise _ReasoningResponseError("response must be non-empty")

        payload = json.loads(
            response,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        _require_object(payload, "top-level response")
        _require_exact_keys(payload, _TOP_LEVEL_KEYS, "top-level response")

        steps_data = payload["steps"]
        if not isinstance(steps_data, list):
            raise _ReasoningResponseError("steps must be a JSON array")

        steps = tuple(
            _parse_step(step_data)
            for step_data in steps_data
        )
        return self._planner.build_plan(
            task_goal=payload["task_goal"],
            steps=steps,
        )


class _ReasoningResponseError(ValueError):
    """Internal marker for invalid or unsafe model output."""


def _build_user_prompt(task: str) -> str:
    return f"{_USER_PROMPT_PREFIX}\n{task}"


def _parse_step(step_data: object) -> SemanticPlanStep:
    _require_object(step_data, "step")

    operation = _parse_operation(step_data.get("operation"))
    if operation is PlanOperation.CLICK_TARGET:
        return _parse_click_target_step(step_data)

    if operation is PlanOperation.READ_CLIPBOARD:
        return _parse_read_clipboard_step(step_data)

    if operation is PlanOperation.ACTIVATE_APP:
        return _parse_activate_app_step(step_data)

    if operation is PlanOperation.INSERT_TEXT:
        return _parse_insert_text_step(step_data)

    if operation is PlanOperation.TYPE_INTO_TARGET:
        return _parse_type_into_target_step(step_data)

    raise _ReasoningResponseError("unsupported operation")


def _parse_operation(value: object) -> PlanOperation:
    if not isinstance(value, str) or not value.strip():
        raise _ReasoningResponseError("operation must be a non-empty string")

    try:
        return PlanOperation(value)
    except ValueError as error:
        raise _ReasoningResponseError("unsupported operation") from error


def _parse_click_target_step(step_data: dict[str, Any]) -> PlanStep:
    _require_exact_keys(step_data, _CLICK_TARGET_STEP_KEYS, "click_target step")
    return PlanStep(
        goal=step_data["goal"],
        operation=PlanOperation.CLICK_TARGET,
        action_target=_parse_target(step_data["action_target"]),
        verification_target=_parse_target(step_data["verification_target"]),
        max_attempts=_parse_max_attempts(step_data["max_attempts"]),
    )


def _parse_read_clipboard_step(step_data: dict[str, Any]) -> ReadClipboardStep:
    _require_exact_keys(
        step_data,
        _READ_CLIPBOARD_STEP_KEYS,
        "read_clipboard step",
    )
    return ReadClipboardStep(
        goal=step_data["goal"],
        value_key=step_data["value_key"],
        expected_text=step_data["expected_text"],
        max_attempts=_parse_max_attempts(step_data["max_attempts"]),
    )


def _parse_activate_app_step(step_data: dict[str, Any]) -> ActivateAppStep:
    _require_exact_keys(step_data, _ACTIVATE_APP_STEP_KEYS, "activate_app step")
    return ActivateAppStep(
        goal=step_data["goal"],
        app_name=step_data["app_name"],
        max_attempts=_parse_max_attempts(step_data["max_attempts"]),
    )


def _parse_insert_text_step(step_data: dict[str, Any]) -> InsertTextStep:
    _require_exact_keys(step_data, _INSERT_TEXT_STEP_KEYS, "insert_text step")
    return InsertTextStep(
        goal=step_data["goal"],
        value_key=step_data["value_key"],
        max_attempts=_parse_max_attempts(step_data["max_attempts"]),
    )


def _parse_type_into_target_step(
    step_data: dict[str, Any],
) -> WebTextInputStep:
    _require_exact_keys(
        step_data,
        _TYPE_INTO_TARGET_STEP_KEYS,
        "type_into_target step",
    )
    return WebTextInputStep(
        goal=step_data["goal"],
        target=_parse_target(step_data["target"]),
        input_text=_parse_input_text(step_data["input_text"]),
        max_attempts=_parse_type_into_target_max_attempts(
            step_data["max_attempts"]
        ),
    )


def _parse_max_attempts(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ReasoningResponseError(
            "max_attempts must be a JSON integer"
        )

    return value


def _parse_type_into_target_max_attempts(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ReasoningResponseError(
            "type_into_target max_attempts must be the JSON integer 1"
        )

    if value != 1:
        raise _ReasoningResponseError(
            "type_into_target max_attempts must be exactly 1"
        )

    return value


def _parse_input_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _ReasoningResponseError(
            "input_text must be a non-empty string"
        )

    return value


def _parse_target(target_data: object) -> TargetSpec:
    _require_object(target_data, "target")
    _require_exact_keys(target_data, _TARGET_KEYS, "target")

    text = target_data["text"]
    if not isinstance(text, str) or not text.strip():
        raise _ReasoningResponseError(
            "target text must be a non-empty string"
        )

    element_types = target_data["element_types"]
    if not isinstance(element_types, list):
        raise _ReasoningResponseError(
            "element_types must be a JSON array"
        )

    for element_type in element_types:
        if not isinstance(element_type, str) or not element_type.strip():
            raise _ReasoningResponseError(
                "element_types must contain non-empty strings"
            )

        if element_type not in _SUPPORTED_ELEMENT_TYPES:
            raise _ReasoningResponseError(
                "element_types must contain only supported reasoning types"
            )

    return TargetSpec(
        text=text,
        element_types=tuple(element_types),
    )


def _require_object(value: object, field_name: str) -> None:
    if not isinstance(value, dict):
        raise _ReasoningResponseError(f"{field_name} must be a JSON object")


def _require_exact_keys(
    value: dict[str, Any],
    expected_keys: frozenset[str],
    field_name: str,
) -> None:
    if frozenset(value) != expected_keys:
        raise _ReasoningResponseError(
            f"{field_name} did not match the required schema"
        )


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _ReasoningResponseError("duplicate JSON keys are rejected")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _ReasoningResponseError(f"unsupported JSON constant: {value}")


def _blocked(reason: str) -> ReasoningResult:
    return ReasoningResult(
        status=ReasoningStatus.BLOCKED,
        plan=None,
        reason=reason,
    )
