import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from computer_agent.planning import (
    MAX_PLAN_STEP_ATTEMPTS,
    MAX_STRUCTURED_PLAN_STEPS,
    PlanOperation,
)
import computer_agent.reasoning.openai_client as openai_client_module
from computer_agent.reasoning import SUPPORTED_REASONING_ELEMENT_TYPES
from computer_agent.reasoning.openai_client import (
    DEFAULT_OPENAI_REASONING_MODEL,
    OpenAILLMClient,
)


class FakeOpenAIResponse:
    def __init__(self, output_text: object) -> None:
        self.output_text = output_text


class FakeResponses:
    def __init__(
        self,
        response: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        return self.response


class FakeOpenAIClient:
    def __init__(
        self,
        output_text: object = '{"task_goal": "Open settings", "steps": []}',
        error: Exception | None = None,
    ) -> None:
        self.responses = FakeResponses(
            response=FakeOpenAIResponse(output_text),
            error=error,
        )


class FalseyFakeOpenAIClient(FakeOpenAIClient):
    def __bool__(self) -> bool:
        return False


def _generate(
    *,
    model: str = DEFAULT_OPENAI_REASONING_MODEL,
    sdk_client: FakeOpenAIClient | None = None,
    system_prompt: str = "system prompt",
    user_prompt: str = "user prompt",
) -> tuple[str, FakeOpenAIClient, dict[str, object]]:
    if sdk_client is None:
        sdk_client = FakeOpenAIClient()

    client = OpenAILLMClient(
        model=model,
        client=sdk_client,
    )
    result = client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return result, sdk_client, sdk_client.responses.calls[0]


def _text_format(call: dict[str, object]) -> dict[str, object]:
    text = call["text"]
    assert isinstance(text, dict)

    text_format = text["format"]
    assert isinstance(text_format, dict)
    return text_format


def _schema(call: dict[str, object]) -> dict[str, object]:
    schema = _text_format(call)["schema"]
    assert isinstance(schema, dict)
    return schema


def _step_variants(call: dict[str, object]) -> tuple[dict[str, object], ...]:
    step_schema = _schema(call)["properties"]["steps"]["items"]
    variants = step_schema["anyOf"]
    assert isinstance(variants, list)
    return tuple(variants)


def _step_variants_by_operation(
    call: dict[str, object],
) -> dict[str, dict[str, object]]:
    variants_by_operation = {}
    for variant in _step_variants(call):
        assert isinstance(variant, dict)
        operation_schema = variant["properties"]["operation"]
        operation_values = operation_schema["enum"]
        assert isinstance(operation_values, list)
        assert len(operation_values) == 1
        variants_by_operation[operation_values[0]] = variant
    return variants_by_operation


def _step_variant(
    call: dict[str, object],
    operation: PlanOperation,
) -> dict[str, object]:
    return _step_variants_by_operation(call)[operation.value]


def _click_target_schemas(
    call: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    step_schema = _step_variant(call, PlanOperation.CLICK_TARGET)
    action_target = step_schema["properties"]["action_target"]
    verification_target = step_schema["properties"]["verification_target"]
    assert isinstance(action_target, dict)
    assert isinstance(verification_target, dict)
    return action_target, verification_target


def test_default_model_is_experiment_model():
    _result, _sdk_client, call = _generate()

    assert call["model"] == DEFAULT_OPENAI_REASONING_MODEL
    assert OpenAILLMClient(client=FakeOpenAIClient()).model == (
        "gpt-5.6-terra"
    )


def test_custom_model_is_preserved():
    _result, _sdk_client, call = _generate(model="custom-model")

    assert call["model"] == "custom-model"
    assert OpenAILLMClient(
        model="custom-model",
        client=FakeOpenAIClient(),
    ).model == "custom-model"


def test_falsey_injected_sdk_client_is_preserved(monkeypatch):
    def fail_default_client() -> object:
        raise AssertionError("default OpenAI client should not be built")

    monkeypatch.setattr(
        openai_client_module,
        "_build_default_openai_client",
        fail_default_client,
    )
    sdk_client = FalseyFakeOpenAIClient("provider text")

    result, _sdk_client, call = _generate(sdk_client=sdk_client)

    assert result == "provider text"
    assert call["model"] == DEFAULT_OPENAI_REASONING_MODEL
    assert len(sdk_client.responses.calls) == 1


def test_generate_makes_exactly_one_responses_create_call():
    _result, sdk_client, _call = _generate()

    assert len(sdk_client.responses.calls) == 1


def test_system_and_user_prompts_are_forwarded():
    _result, _sdk_client, call = _generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert call["input"] == [
        {
            "role": "system",
            "content": "system",
        },
        {
            "role": "user",
            "content": "user",
        },
    ]


def test_no_tools_are_supplied_to_responses_create():
    _result, _sdk_client, call = _generate()

    assert "tools" not in call
    assert "tool_choice" not in call
    assert "parallel_tool_calls" not in call


def test_response_is_not_stored():
    _result, _sdk_client, call = _generate()

    assert call["store"] is False


def test_strict_structured_output_schema_is_supplied():
    _result, _sdk_client, call = _generate()

    text_format = _text_format(call)
    assert text_format["type"] == "json_schema"
    assert text_format["name"] == "computer_agent_reasoning_plan"
    assert text_format["strict"] is True
    assert isinstance(text_format["schema"], dict)


def test_schema_forbids_extra_top_level_keys():
    _result, _sdk_client, call = _generate()

    schema = _schema(call)
    assert schema["required"] == [
        "task_goal",
        "steps",
    ]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["steps"]["maxItems"] == (
        MAX_STRUCTURED_PLAN_STEPS
    )


def test_schema_contains_four_operation_specific_step_variants():
    _result, _sdk_client, call = _generate()

    variants_by_operation = _step_variants_by_operation(call)
    assert set(variants_by_operation) == {
        PlanOperation.CLICK_TARGET.value,
        PlanOperation.READ_CLIPBOARD.value,
        PlanOperation.ACTIVATE_APP.value,
        PlanOperation.INSERT_TEXT.value,
    }
    assert len(_step_variants(call)) == 4


@pytest.mark.parametrize(
    ("operation", "required_keys"),
    [
        (
            PlanOperation.CLICK_TARGET,
            [
                "goal",
                "operation",
                "action_target",
                "verification_target",
                "max_attempts",
            ],
        ),
        (
            PlanOperation.READ_CLIPBOARD,
            [
                "goal",
                "operation",
                "value_key",
                "expected_text",
                "max_attempts",
            ],
        ),
        (
            PlanOperation.ACTIVATE_APP,
            [
                "goal",
                "operation",
                "app_name",
                "max_attempts",
            ],
        ),
        (
            PlanOperation.INSERT_TEXT,
            [
                "goal",
                "operation",
                "value_key",
                "max_attempts",
            ],
        ),
    ],
)
def test_schema_forbids_extra_step_keys(
    operation: PlanOperation,
    required_keys: list[str],
):
    _result, _sdk_client, call = _generate()

    step_schema = _step_variant(call, operation)
    assert step_schema["required"] == required_keys
    assert set(step_schema["properties"]) == set(required_keys)
    assert step_schema["additionalProperties"] is False


def test_schema_forbids_extra_target_keys():
    _result, _sdk_client, call = _generate()

    for target_schema in _click_target_schemas(call):
        assert target_schema["required"] == [
            "text",
            "element_types",
        ]
        assert target_schema["additionalProperties"] is False


def test_schema_element_types_item_enum_matches_reasoning_policy():
    _result, _sdk_client, call = _generate()

    for target_schema in _click_target_schemas(call):
        element_types_schema = target_schema["properties"]["element_types"]
        assert tuple(element_types_schema["items"]["enum"]) == (
            SUPPORTED_REASONING_ELEMENT_TYPES
        )


def test_schema_allows_empty_element_types_list():
    _result, _sdk_client, call = _generate()

    for target_schema in _click_target_schemas(call):
        element_types_schema = target_schema["properties"]["element_types"]
        assert "minItems" not in element_types_schema


def test_action_and_verification_targets_use_same_constrained_schema():
    _result, _sdk_client, call = _generate()

    action_target, verification_target = _click_target_schemas(call)
    assert action_target == verification_target
    assert tuple(
        action_target["properties"]["element_types"]["items"]["enum"]
    ) == SUPPORTED_REASONING_ELEMENT_TYPES


def test_schema_constrains_operation_per_variant():
    _result, _sdk_client, call = _generate()

    for operation, step_schema in _step_variants_by_operation(call).items():
        assert step_schema["properties"]["operation"] == {
            "type": "string",
            "enum": [
                operation,
            ],
        }


def test_schema_uses_planning_attempt_bound():
    _result, _sdk_client, call = _generate()

    for step_schema in _step_variants(call):
        max_attempts_schema = step_schema["properties"]["max_attempts"]
        assert max_attempts_schema["maximum"] == MAX_PLAN_STEP_ATTEMPTS


def test_insert_text_schema_contains_only_runtime_value_reference():
    _result, _sdk_client, call = _generate()

    step_schema = _step_variant(call, PlanOperation.INSERT_TEXT)
    assert step_schema["required"] == [
        "goal",
        "operation",
        "value_key",
        "max_attempts",
    ]
    assert set(step_schema["properties"]) == {
        "goal",
        "operation",
        "value_key",
        "max_attempts",
    }
    assert not {
        "text",
        "text_to_type",
        "expected_text",
        "tool_name",
        "arguments",
        "hotkey",
    } & set(step_schema["properties"])


def test_response_output_text_is_returned_exactly():
    output_text = '  {"task_goal": "Open settings", "steps": []}\n'

    result, _sdk_client, _call = _generate(
        sdk_client=FakeOpenAIClient(output_text),
    )

    assert result == output_text


@pytest.mark.parametrize("output_text", ["", "   ", "\n\t"])
def test_empty_output_text_fails_explicitly(output_text: str):
    client = OpenAILLMClient(
        client=FakeOpenAIClient(output_text),
    )

    with pytest.raises(
        ValueError,
        match="OpenAI response output_text must be non-empty",
    ):
        client.generate(
            system_prompt="system",
            user_prompt="user",
        )


@pytest.mark.parametrize("output_text", [None, 123, ["text"]])
def test_non_string_output_text_fails_explicitly(output_text: object):
    client = OpenAILLMClient(
        client=FakeOpenAIClient(output_text),
    )

    with pytest.raises(
        ValueError,
        match="OpenAI response output_text must be a string",
    ):
        client.generate(
            system_prompt="system",
            user_prompt="user",
        )


def test_sdk_exception_propagates_from_openai_client():
    sdk_client = FakeOpenAIClient(
        error=RuntimeError("sdk failure"),
    )
    client = OpenAILLMClient(client=sdk_client)

    with pytest.raises(RuntimeError, match="sdk failure"):
        client.generate(
            system_prompt="system",
            user_prompt="user",
        )

    assert len(sdk_client.responses.calls) == 1


def test_importing_openai_client_module_has_no_api_call_or_output():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import computer_agent.reasoning.openai_client",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_no_api_key_is_embedded_in_source_or_fixtures():
    source = (
        inspect.getsource(openai_client_module)
        + Path(__file__).read_text()
    )
    secret_prefix = "s" + "k-"
    api_key_argument = "api" + "_key="

    assert secret_prefix not in source
    assert api_key_argument not in source
