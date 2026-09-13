"""OpenAI adapter for adaptive next-step reasoning."""

from __future__ import annotations

from computer_agent.reasoning.openai_client import (
    DEFAULT_OPENAI_REASONING_MODEL,
)


class OpenAIAdaptiveLLMClient:
    """OpenAI Responses API client for one adaptive JSON decision."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENAI_REASONING_MODEL,
        client: object | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                "model must be a non-empty string"
            )

        self._model = model

        if client is None:
            self._client = _build_default_openai_client()
        else:
            self._client = client

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Return one raw JSON decision response."""

        _validate_prompt(
            system_prompt,
            "system_prompt",
        )
        _validate_prompt(
            user_prompt,
            "user_prompt",
        )

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
                "format": {
                    "type": "json_object",
                },
            },
            store=False,
        )

        return _extract_output_text(response)


def _build_default_openai_client() -> object:
    from openai import OpenAI

    return OpenAI()


def _validate_prompt(
    value: object,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )


def _extract_output_text(
    response: object,
) -> str:
    output_text = getattr(
        response,
        "output_text",
        None,
    )

    if not isinstance(output_text, str):
        raise ValueError(
            "OpenAI response output_text "
            "must be a string"
        )

    if not output_text.strip():
        raise ValueError(
            "OpenAI response output_text "
            "must be non-empty"
        )

    return output_text
