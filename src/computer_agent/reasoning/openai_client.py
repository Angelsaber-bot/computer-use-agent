"""OpenAI Responses API adapter for provider-neutral LLM reasoning."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from computer_agent.planning import (
    MAX_PLAN_STEP_ATTEMPTS,
    MAX_STRUCTURED_PLAN_STEPS,
    PlanOperation,
)
from computer_agent.reasoning.models import SUPPORTED_REASONING_ELEMENT_TYPES


DEFAULT_OPENAI_REASONING_MODEL = "gpt-5.6-terra"

_REASONING_TARGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "minLength": 1,
        },
        "element_types": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 1,
                "enum": list(SUPPORTED_REASONING_ELEMENT_TYPES),
            },
        },
    },
    "required": [
        "text",
        "element_types",
    ],
    "additionalProperties": False,
}

_REASONING_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "minLength": 1,
        },
        "operation": {
            "type": "string",
            "enum": [
                PlanOperation.CLICK_TARGET.value,
            ],
        },
        "action_target": _REASONING_TARGET_SCHEMA,
        "verification_target": _REASONING_TARGET_SCHEMA,
        "max_attempts": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_PLAN_STEP_ATTEMPTS,
        },
    },
    "required": [
        "goal",
        "operation",
        "action_target",
        "verification_target",
        "max_attempts",
    ],
    "additionalProperties": False,
}

REASONING_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_goal": {
            "type": "string",
            "minLength": 1,
        },
        "steps": {
            "type": "array",
            "items": _REASONING_STEP_SCHEMA,
            "minItems": 1,
            "maxItems": MAX_STRUCTURED_PLAN_STEPS,
        },
    },
    "required": [
        "task_goal",
        "steps",
    ],
    "additionalProperties": False,
}

_REASONING_PLAN_TEXT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "computer_agent_reasoning_plan",
    "strict": True,
    "schema": REASONING_PLAN_JSON_SCHEMA,
}


class OpenAILLMClient:
    """LLMClient implementation backed by the OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_REASONING_MODEL,
        client: object | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")

        self._model = model
        if client is None:
            self._client = _build_default_openai_client()
        else:
            self._client = client

    @property
    def model(self) -> str:
        """Return the configured OpenAI model."""

        return self._model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return raw output text from one OpenAI Responses API call."""

        _validate_prompt(system_prompt, "system_prompt")
        _validate_prompt(user_prompt, "user_prompt")

        response = self._client.responses.create(
            model=self._model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text={
                "format": deepcopy(_REASONING_PLAN_TEXT_FORMAT),
            },
            store=False,
        )
        return _extract_output_text(response)


def _build_default_openai_client() -> object:
    from openai import OpenAI

    return OpenAI()


def _validate_prompt(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _extract_output_text(response: object) -> str:
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str):
        raise ValueError("OpenAI response output_text must be a string")

    if not output_text.strip():
        raise ValueError("OpenAI response output_text must be non-empty")

    return output_text
