"""Tests for the adaptive OpenAI Responses API adapter."""

from __future__ import annotations

import pytest

from computer_agent.reasoning.adaptive_openai_client import (
    OpenAIAdaptiveLLMClient,
)
from computer_agent.reasoning.openai_client import (
    DEFAULT_OPENAI_REASONING_MODEL,
)


class FakeResponse:
    def __init__(
        self,
        output_text: object,
    ) -> None:
        self.output_text = output_text


class FakeResponses:
    def __init__(
        self,
        output_text: object,
    ) -> None:
        self.output_text = output_text
        self.calls: list[
            dict[str, object]
        ] = []

    def create(
        self,
        **kwargs: object,
    ) -> object:
        self.calls.append(kwargs)

        return FakeResponse(
            self.output_text
        )


class FakeOpenAI:
    def __init__(
        self,
        output_text: object,
    ) -> None:
        self.responses = FakeResponses(
            output_text
        )


def test_adaptive_client_uses_json_object_mode() -> None:
    sdk = FakeOpenAI(
        '{"decision":"ask_user","question":"Continue?"}'
    )

    client = OpenAIAdaptiveLLMClient(
        client=sdk
    )

    result = client.generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert result == (
        '{"decision":"ask_user","question":"Continue?"}'
    )

    assert len(
        sdk.responses.calls
    ) == 1

    call = sdk.responses.calls[0]

    assert call["model"] == (
        DEFAULT_OPENAI_REASONING_MODEL
    )

    assert call["text"] == {
        "format": {
            "type": "json_object",
        },
    }

    assert call["store"] is False


def test_adaptive_client_forwards_prompts() -> None:
    sdk = FakeOpenAI(
        '{"decision":"complete","summary":"Done"}'
    )

    client = OpenAIAdaptiveLLMClient(
        client=sdk
    )

    client.generate(
        system_prompt="adaptive system",
        user_prompt="adaptive user",
    )

    call = sdk.responses.calls[0]

    assert call["input"] == [
        {
            "role": "system",
            "content": "adaptive system",
        },
        {
            "role": "user",
            "content": "adaptive user",
        },
    ]


def test_adaptive_client_uses_no_tools() -> None:
    sdk = FakeOpenAI(
        '{"decision":"complete","summary":"Done"}'
    )

    client = OpenAIAdaptiveLLMClient(
        client=sdk
    )

    client.generate(
        system_prompt="system",
        user_prompt="user",
    )

    call = sdk.responses.calls[0]

    assert "tools" not in call
    assert "tool_choice" not in call


@pytest.mark.parametrize(
    "output_text",
    (
        "",
        "   ",
        None,
        123,
    ),
)
def test_invalid_output_fails_explicitly(
    output_text: object,
) -> None:
    client = OpenAIAdaptiveLLMClient(
        client=FakeOpenAI(
            output_text
        )
    )

    with pytest.raises(
        ValueError,
    ):
        client.generate(
            system_prompt="system",
            user_prompt="user",
        )
