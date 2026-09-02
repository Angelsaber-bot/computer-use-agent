"""Provider-neutral LLM reasoner for semantic structured plans."""

from __future__ import annotations

import json
from typing import Any

from computer_agent.grounding.models import TargetSpec
from computer_agent.planning.models import PlanOperation, PlanStep, StructuredPlan
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
Each step object must contain exactly:
{{"goal": string, "operation": "click_target", "action_target": target, "verification_target": target, "max_attempts": integer}}
Each target object must contain exactly:
{{"text": string, "element_types": array}}
element_types may contain only: {", ".join(SUPPORTED_REASONING_ELEMENT_TYPES)}.
If the UI role is uncertain, use an empty element_types array.
Never invent a role.
Use only the "click_target" operation.
Do not claim to observe the screen, operate the computer, or execute the task.
""".strip()

_USER_PROMPT_PREFIX = "Task intent:"
_TOP_LEVEL_KEYS = frozenset(("task_goal", "steps"))
_SUPPORTED_ELEMENT_TYPES = frozenset(SUPPORTED_REASONING_ELEMENT_TYPES)
_STEP_KEYS = frozenset(
    (
        "goal",
        "operation",
        "action_target",
        "verification_target",
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


def _parse_step(step_data: object) -> PlanStep:
    _require_object(step_data, "step")
    _require_exact_keys(step_data, _STEP_KEYS, "step")

    operation_value = step_data["operation"]
    if operation_value != PlanOperation.CLICK_TARGET.value:
        raise _ReasoningResponseError("unsupported operation")

    max_attempts = step_data["max_attempts"]
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise _ReasoningResponseError(
            "max_attempts must be a JSON integer"
        )

    return PlanStep(
        goal=step_data["goal"],
        operation=PlanOperation.CLICK_TARGET,
        action_target=_parse_target(step_data["action_target"]),
        verification_target=_parse_target(step_data["verification_target"]),
        max_attempts=max_attempts,
    )


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
