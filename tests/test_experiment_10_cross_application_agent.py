from html.parser import HTMLParser
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from computer_agent.agent import (
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    AgentStatus,
)
from computer_agent.core.models import Action, ToolResult
from computer_agent.planning import (
    ActivateAppStep,
    InsertTextStep,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    StructuredPlan,
)
from computer_agent.reasoning import LLMReasoner, ReasoningResult
from computer_agent.reasoning import ReasoningStatus
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_10_cross_application_agent as experiment,
)


SCRIPT_PATH = Path(experiment.__file__).resolve()
_MISSING = object()


class FixtureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._button_depth = 0
        self._button_parts = []
        self.buttons = []
        self.button_attrs = []
        self.visible_texts = []
        self.visible_attrs = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        attr_dict = dict(attrs)
        self.visible_attrs.extend(
            str(value)
            for value in attr_dict.values()
            if value is not None
        )

        if tag == "button":
            self._button_depth += 1
            self._button_parts = []
            self.button_attrs.append(attr_dict)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._ignored_depth -= 1
            return

        if self._ignored_depth:
            return

        if tag == "button" and self._button_depth:
            self._button_depth -= 1
            self.buttons.append("".join(self._button_parts).strip())
            self._button_parts = []

    def handle_data(self, data):
        if self._ignored_depth:
            return

        if self._button_depth:
            self._button_parts.append(data)

        text = data.strip()
        if text:
            self.visible_texts.append(text)


class RecordingRunner:
    def __init__(self, result=None) -> None:
        self.result = result
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        return self.result if self.result is not None else _loop_result(plan)


class FailingRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        pytest.fail("runner should not be called")


class RecordingPrerequisites:
    def __init__(self, result=True) -> None:
        self.result = result
        self.calls = []

    def __call__(self, fixture_path):
        self.calls.append(Path(fixture_path))
        return self.result


class FailingPrerequisites:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, fixture_path):
        self.calls.append(Path(fixture_path))
        pytest.fail("prerequisites should not be called")


def _path_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(experiment.PROJECT_ROOT / "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return env


def _fixture_source() -> str:
    return experiment.FIXTURE_PATH.read_text(encoding="utf-8")


def _fixture_parser() -> FixtureParser:
    parser = FixtureParser()
    parser.feed(_fixture_source())
    return parser


def _function_body(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"function body not found: {name}")


def _click_handler_source(source: str) -> str:
    start = source.index('.addEventListener("click"')
    return source[start : source.index("</script>", start)]


def _payload() -> dict[str, object]:
    return copy.deepcopy(experiment.build_fake_response_payload())


def _client_for_payload(payload) -> experiment.DeterministicFakeLLMClient:
    return experiment.DeterministicFakeLLMClient(json.dumps(payload))


def _run() -> experiment.ReasoningRun:
    return experiment.run_deterministic_reasoning()


def _plan() -> StructuredPlan:
    run = _run()
    assert run.result.status is ReasoningStatus.READY
    assert isinstance(run.result.plan, StructuredPlan)
    return run.result.plan


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def _action(tool_name, arguments=None) -> Action:
    if arguments is None:
        arguments = _arguments_for_tool(tool_name)
    return Action(
        tool_name=tool_name,
        arguments=arguments,
        reason="synthetic Experiment 10 action",
    )


def _arguments_for_tool(tool_name, *, paste_value=_MISSING):
    if tool_name == "click_mouse":
        return {"x": 10, "y": 20}
    if tool_name == "activate_app":
        return {"app_name": experiment.TEXTEDIT_APP_NAME}
    if tool_name == "paste_text":
        text = (
            experiment.TRANSFER_VALUE
            if paste_value is _MISSING
            else paste_value
        )
        return {"text": text}
    return {}


def _state_with_actions(
    plan,
    tool_names,
    *,
    failed_attempts=(),
    context_value=experiment.TRANSFER_VALUE,
    paste_value=_MISSING,
    final_status=AgentStatus.SUCCEEDED,
) -> AgentState:
    state = AgentState(user_task=plan.task_goal)
    state.start()
    if context_value is not _MISSING:
        state.context["values"] = {
            experiment.VALUE_KEY: context_value,
        }

    for index, tool_name in enumerate(tool_names, start=1):
        action = _action(
            tool_name,
            _arguments_for_tool(tool_name, paste_value=paste_value),
        )
        success = index not in failed_attempts
        result = ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            error=None if success else "synthetic failure",
        )
        state.record_step(action, result)

    if final_status is AgentStatus.SUCCEEDED:
        state.succeed()
    elif final_status is AgentStatus.FAILED:
        state.fail("terminal failure")
    else:
        raise AssertionError(f"unsupported final status: {final_status}")
    return state


def _loop_result(
    plan,
    *,
    result_plan=None,
    status=AgentLoopStatus.COMPLETED,
    completed_plan_steps=experiment.EXPECTED_PLAN_STEPS,
    tool_names=experiment.EXPECTED_ACTION_TOOLS,
    failed_attempts=(),
    context_value=experiment.TRANSFER_VALUE,
    paste_value=_MISSING,
) -> AgentLoopResult:
    if result_plan is None:
        result_plan = plan
    final_status = (
        AgentStatus.SUCCEEDED
        if status is AgentLoopStatus.COMPLETED
        else AgentStatus.FAILED
    )
    return AgentLoopResult(
        status=status,
        plan=result_plan,
        state=_state_with_actions(
            result_plan,
            tool_names,
            failed_attempts=failed_attempts,
            context_value=context_value,
            paste_value=paste_value,
            final_status=final_status,
        ),
        completed_plan_steps=completed_plan_steps,
        reason=f"{status.value} synthetic result",
    )


def _with_completed_steps(result, completed_steps):
    object.__setattr__(result, "completed_plan_steps", completed_steps)
    return result


def test_formal_task_and_constants_are_exact():
    assert experiment.TITLE == "Phase 04 Experiment 10: Cross-Application Agent"
    assert experiment.TASK == (
        "Transfer the deterministic browser fixture value "
        "CROSS_APP_TRANSFER_10 into the blank TextEdit document."
    )
    assert experiment.EXPECTED_TASK_GOAL == experiment.TASK
    assert experiment.DEFAULT_WAIT_SECONDS == 8
    assert experiment.ALLOWED_APP_NAMES == frozenset({"TextEdit"})
    assert experiment.CAPTURE_PATH.name == (
        "experiment_10_cross_application_agent.png"
    )


def test_fake_response_payload_matches_required_semantic_plan():
    assert experiment.build_fake_response_payload() == {
        "task_goal": experiment.TASK,
        "steps": [
            {
                "goal": "Copy the transfer value from the browser fixture",
                "operation": "click_target",
                "action_target": {
                    "text": "COPY_TRANSFER_VALUE_10",
                    "element_types": ["button"],
                },
                "verification_target": {
                    "text": "TRANSFER_COPIED_10",
                    "element_types": ["button"],
                },
                "max_attempts": 1,
            },
            {
                "goal": "Read and verify the copied transfer value",
                "operation": "read_clipboard",
                "value_key": "transfer_value",
                "expected_text": "CROSS_APP_TRANSFER_10",
                "max_attempts": 1,
            },
            {
                "goal": "Switch to TextEdit",
                "operation": "activate_app",
                "app_name": "TextEdit",
                "max_attempts": 1,
            },
            {
                "goal": "Insert the verified transfer value",
                "operation": "insert_text",
                "value_key": "transfer_value",
                "max_attempts": 1,
            },
        ],
    }


def test_fake_provider_called_once_and_production_reasoner_parses_plan():
    client = experiment.DeterministicFakeLLMClient()
    run = experiment.run_deterministic_reasoning(client)

    assert experiment.LLMReasoner is LLMReasoner
    assert run.result.status is ReasoningStatus.READY
    assert isinstance(run.result.plan, StructuredPlan)
    assert client.call_count == 1
    assert run.client_call_count == 1
    assert client.calls[0]["user_prompt"].endswith(experiment.TASK)
    assert experiment.reasoning_acceptance_failures(run) == ()


def test_fake_client_returns_provider_style_json_and_records_prompts():
    client = experiment.DeterministicFakeLLMClient()
    response = client.generate(system_prompt="system", user_prompt="user")

    assert json.loads(response) == experiment.build_fake_response_payload()
    assert client.call_count == 1
    assert client.calls == [{"system_prompt": "system", "user_prompt": "user"}]


def test_exact_plan_order_types_and_step_fields():
    plan = _plan()

    assert plan.task_goal == experiment.TASK
    assert len(plan.steps) == 4
    assert [step.operation for step in plan.steps] == [
        PlanOperation.CLICK_TARGET,
        PlanOperation.READ_CLIPBOARD,
        PlanOperation.ACTIVATE_APP,
        PlanOperation.INSERT_TEXT,
    ]
    assert isinstance(plan.steps[0], PlanStep)
    assert isinstance(plan.steps[1], ReadClipboardStep)
    assert isinstance(plan.steps[2], ActivateAppStep)
    assert isinstance(plan.steps[3], InsertTextStep)

    click_step, read_step, activate_step, insert_step = plan.steps
    assert click_step.action_target.text == "COPY_TRANSFER_VALUE_10"
    assert click_step.action_target.element_types == ("button",)
    assert click_step.verification_target.text == "TRANSFER_COPIED_10"
    assert click_step.verification_target.element_types == ("button",)
    assert read_step.value_key == "transfer_value"
    assert read_step.expected_text == "CROSS_APP_TRANSFER_10"
    assert activate_step.app_name == "TextEdit"
    assert insert_step.value_key == "transfer_value"
    assert read_step.value_key == insert_step.value_key
    assert not hasattr(insert_step, "text")
    assert not hasattr(insert_step, "expected_text")
    assert not hasattr(insert_step, "arguments")


def test_plan_contains_no_actions_coordinates_or_raw_argument_authority():
    plan = _plan()

    for step in plan.steps:
        fields = step.__dataclass_fields__
        assert not any(field in fields for field in ("x", "y", "coordinates"))
        assert not any(
            isinstance(getattr(step, field), Action)
            for field in fields
        )
    assert plan.steps[0].action_target.reference_point is None
    assert plan.steps[0].verification_target.reference_point is None


def test_provider_plan_contains_no_raw_tool_names_or_arguments():
    payload = experiment.build_fake_response_payload()
    forbidden_keys = {"tool_name", "arguments", "x", "y", "coordinates"}
    forbidden_values = {"click_mouse", "read_from_clipboard", "paste_text"}

    for item in _walk(payload):
        if isinstance(item, dict):
            assert forbidden_keys.isdisjoint(item)
        elif isinstance(item, str):
            assert item not in forbidden_values


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(task_goal="Wrong task"), "task goal was"),
        (lambda payload: payload["steps"].pop(), "step count was 3, expected 4"),
        (
            lambda payload: payload["steps"][0]["action_target"].update(
                text="WRONG_TARGET_10"
            ),
            "step 1 action target text was",
        ),
        (
            lambda payload: payload["steps"][1].update(value_key="other_value"),
            "step 2 value_key was",
        ),
        (
            lambda payload: payload["steps"][2].update(app_name="Safari"),
            "step 3 app_name was",
        ),
        (
            lambda payload: payload["steps"][3].update(value_key="other_value"),
            "step 4 value_key was",
        ),
    ],
)
def test_reasoning_acceptance_rejects_wrong_plan_details(
    mutator,
    message,
    capsys,
):
    payload = _payload()
    mutator(payload)

    code = experiment.main([], client=_client_for_payload(payload))
    output = capsys.readouterr().out

    assert code == 1
    assert "Reasoning acceptance: failed" in output
    assert message in output


def test_insert_step_literal_text_payload_fails_closed(capsys):
    payload = _payload()
    payload["steps"][3]["text"] = experiment.TRANSFER_VALUE

    code = experiment.main([], client=_client_for_payload(payload))
    output = capsys.readouterr().out

    assert code == 1
    assert "Reasoning acceptance: failed" in output


def test_default_dry_run_is_offline_and_non_actuating(capsys, tmp_path, monkeypatch):
    runner = FailingRunner()
    prerequisites = FailingPrerequisites()
    capture_path = tmp_path / "experiment_10_cross_application_agent.png"
    monkeypatch.setattr(experiment, "CAPTURE_PATH", capture_path)

    code = experiment.main(
        [],
        runner=runner,
        prerequisite_checker=prerequisites,
        sleeper=lambda _seconds: pytest.fail("sleep should not run"),
    )
    output = capsys.readouterr().out

    assert code == 0
    assert runner.calls == []
    assert prerequisites.calls == []
    assert not capture_path.exists()
    assert "Reasoning acceptance: passed" in output
    assert "Execution mode: disabled" in output
    assert "Observation count: 0" in output
    assert "Action execution count: 0" in output
    assert "Clipboard read: no" in output
    assert "Application activation: no" in output
    assert "Paste action: no" in output
    assert "Screenshot creation: no" in output
    assert "Live API request: no" in output
    assert "Dry-run acceptance: passed" in output


def test_execute_mode_is_explicit_and_uses_injected_runner_only(capsys):
    runner = RecordingRunner()
    prerequisites = RecordingPrerequisites(True)
    sleeps = []

    code = experiment.main(
        ["--execute", "--wait-seconds", "0"],
        runner=runner,
        prerequisite_checker=prerequisites,
        sleeper=sleeps.append,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert len(runner.calls) == 1
    assert isinstance(runner.calls[0], StructuredPlan)
    assert prerequisites.calls == [experiment.FIXTURE_PATH]
    assert sleeps == []
    assert "Execution mode: enabled" in output
    assert "Open a blank TextEdit document manually." in output
    assert "Open the Experiment 10 fixture manually in Google Chrome." in output
    assert "AgentLoop acceptance: passed" in output


def test_direct_safe_dry_run_script_succeeds_offline():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert experiment.TITLE in result.stdout
    assert "Reasoning acceptance: passed" in result.stdout
    assert "Execution mode: disabled" in result.stdout
    assert "Live API request: no" in result.stdout
    assert "Observation count: 0" in result.stdout
    assert "Action execution count: 0" in result.stdout
    assert "Dry-run acceptance: passed" in result.stdout
    assert result.stderr == ""


def test_help_and_invalid_wait_are_non_actuating():
    help_result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    invalid_wait = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--wait-seconds", "31"],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "--execute" in help_result.stdout
    assert "--wait-seconds" in help_result.stdout
    assert experiment.TITLE not in help_result.stdout
    assert invalid_wait.returncode == 2
    assert "--wait-seconds must be from 0 through 30" in invalid_wait.stderr


def test_live_agent_loop_wiring_sets_textedit_allowlist(tmp_path, monkeypatch):
    created = {}

    class RecordingAgentLoop:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

        def run(self, plan):
            created["plan"] = plan
            return _loop_result(plan)

    monkeypatch.setattr(experiment, "AgentLoop", RecordingAgentLoop)
    capture_path = tmp_path / "nested" / "evidence.png"
    plan = _plan()

    result = experiment.run_live_agent_loop(
        plan,
        capture_path=capture_path,
        perception_engine_builder=lambda path: ("perception", path),
        executor_builder=lambda: "executor",
        state_verifier_builder=lambda: "state verifier",
    )

    assert isinstance(result, AgentLoopResult)
    assert capture_path.parent.is_dir()
    assert not capture_path.exists()
    assert created["plan"] is plan
    assert created["kwargs"] == {
        "perception_engine": ("perception", capture_path),
        "executor": "executor",
        "state_verifier": "state verifier",
        "allowed_app_names": frozenset({"TextEdit"}),
    }


def test_fixture_initial_state_contract():
    parser = _fixture_parser()
    visible = "\n".join(parser.visible_texts)
    attrs = "\n".join(parser.visible_attrs)

    assert "Phase 04 Experiment 10: Cross-Application Agent" in visible
    assert "Transfer value:" in visible
    assert "CROSS_APP_TRANSFER_10" in visible
    assert parser.buttons == ["COPY_TRANSFER_VALUE_10"]
    assert parser.button_attrs == [
        {
            "id": "copy-transfer-value-10",
            "type": "button",
            "aria-label": "COPY_TRANSFER_VALUE_10",
        }
    ]
    assert "TRANSFER_COPIED_10" not in visible
    assert "TRANSFER_COPIED_10" not in attrs


def test_fixture_clipboard_success_marker_is_conditional():
    source = _fixture_source()
    handler = _click_handler_source(source)
    failure_body = _function_body(source, "renderCopyFailure")

    assert 'document.execCommand("copy")' in source
    assert "if (copyToClipboard(TRANSFER_VALUE))" in handler
    assert "renderCopiedState();" in handler
    assert "renderCopyFailure();" in handler
    assert handler.index("renderCopiedState();") < handler.index(
        "renderCopyFailure();"
    )
    assert "TRANSFER_COPIED_10" not in failure_body


def test_fixture_completion_marker_uses_accessible_native_button_pattern():
    body = _function_body(_fixture_source(), "renderCopiedState")

    assert 'document.createElement("button")' in body
    assert 'marker.type = "button";' in body
    assert 'marker.tabIndex = -1;' in body
    assert 'marker.setAttribute("aria-label", COPIED_MARKER);' in body
    assert "marker.textContent = COPIED_MARKER;" in body
    assert "addEventListener" not in body
    assert "pointer-events: none;" in _fixture_source()
    assert "panel.replaceChildren();" in body


def test_fixture_has_no_network_timers_randomness_or_textedit_automation():
    source = _fixture_source().lower()
    forbidden_terms = (
        "http://",
        "https://",
        "fetch(",
        "xmlhttprequest",
        "websocket",
        "settimeout",
        "setinterval",
        "requestanimationframe",
        "math.random",
        "navigator.clipboard",
        "textedit",
        "open -a",
        "applescript",
        "pyautogui",
        "click_mouse",
        "screenx",
        "screeny",
        "clientx",
        "clienty",
        "getboundingclientrect",
    )

    assert all(term not in source for term in forbidden_terms)


def test_fixture_does_not_auto_copy_on_page_load():
    source = _fixture_source()
    handler = _click_handler_source(source)

    assert source.count("copyToClipboard(") == 2
    assert "copyToClipboard(TRANSFER_VALUE)" in handler
    assert source.count('document.execCommand("copy")') == 1


def test_agent_loop_acceptance_passes_exact_four_action_result():
    plan = _plan()
    result = _loop_result(plan)

    assert experiment.agent_loop_acceptance_failures(result, plan) == ()


@pytest.mark.parametrize(
    ("result_builder", "message"),
    [
        (
            lambda plan: _loop_result(
                plan,
                tool_names=experiment.EXPECTED_ACTION_TOOLS[:3],
            ),
            "Action execution count was 3, not 4",
        ),
        (
            lambda plan: _loop_result(
                plan,
                tool_names=experiment.EXPECTED_ACTION_TOOLS
                + ("paste_text",),
            ),
            "Action execution count was 5, not 4",
        ),
        (
            lambda plan: _loop_result(
                plan,
                tool_names=(
                    "click_mouse",
                    "activate_app",
                    "read_from_clipboard",
                    "paste_text",
                ),
            ),
            "action tool order was",
        ),
        (
            lambda plan: _loop_result(plan, failed_attempts=(2,)),
            "attempt 2 ToolResult failed",
        ),
        (
            lambda plan: _loop_result(plan, context_value=_MISSING),
            "runtime transfer value was None",
        ),
        (
            lambda plan: _loop_result(plan, context_value="wrong"),
            "runtime transfer value was 'wrong'",
        ),
        (
            lambda plan: _loop_result(
                plan,
                context_value=experiment.TRANSFER_VALUE,
                paste_value="literal mismatch",
            ),
            "paste_text value was 'literal mismatch'",
        ),
        (
            lambda plan: _with_completed_steps(_loop_result(plan), 3),
            "completed plan steps were 3, not 4",
        ),
        (
            lambda plan: _loop_result(plan, result_plan=_plan()),
            "AgentLoopResult plan was not the reasoned plan object",
        ),
    ],
)
def test_agent_loop_acceptance_rejects_invalid_results(
    result_builder,
    message,
):
    plan = _plan()

    failures = experiment.agent_loop_acceptance_failures(
        result_builder(plan),
        plan,
    )

    assert any(message in failure for failure in failures)


def test_print_agent_loop_acceptance_result_reports_pass(capsys):
    plan = _plan()
    result = _loop_result(plan)

    code = experiment.print_agent_loop_acceptance_result(result, plan)
    output = capsys.readouterr().out

    assert code == 0
    assert "AgentLoop acceptance: passed" in output
    assert "Action executions: 4" in output
    assert "Runtime transfer value: CROSS_APP_TRANSFER_10" in output


def test_production_does_not_import_experiment_module():
    production_root = experiment.PROJECT_ROOT / "src/computer_agent"
    for path in production_root.rglob("*.py"):
        assert "experiment_10_cross_application_agent" not in path.read_text(
            encoding="utf-8"
        )


def test_harness_composes_production_capabilities_without_reimplementing_them():
    source = inspect.getsource(experiment)

    assert "LLMReasoner(client=client).reason(TASK)" in source
    assert "AgentLoop(" in source
    assert "build_live_perception_engine" in source
    assert "build_live_tool_executor" in source
    assert "StateVerifier" in source
    assert "OpenAIClient" not in source
    assert "StructuredPlanner" not in source
    assert "GroundingResult" not in source
    assert "RecoveryResult" not in source
    assert "BoundingBox" not in source


def test_importing_experiment_performs_no_execution_or_output():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "importlib.import_module("
                "'experiments.phase04_ui_grounding_task_reasoning."
                "experiment_10_cross_application_agent'"
                ")"
            ),
        ],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_imported_module_identity_is_stable():
    module = importlib.import_module(
        "experiments.phase04_ui_grounding_task_reasoning."
        "experiment_10_cross_application_agent"
    )

    assert module is experiment
