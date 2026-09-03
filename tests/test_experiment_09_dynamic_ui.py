import copy
from html.parser import HTMLParser
import importlib
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
from computer_agent.planning import PlanOperation, StructuredPlan
from computer_agent.reasoning import ReasoningResult, ReasoningStatus
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_09_dynamic_ui as experiment,
)


SCRIPT_PATH = Path(experiment.__file__).resolve()


class InitialFixtureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._button_depth = 0
        self._button_parts = []
        self.buttons = []
        self.button_attrs = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "button":
            self._button_depth += 1
            self._button_parts = []
            self.button_attrs.append(dict(attrs))

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
        if not self._ignored_depth and self._button_depth:
            self._button_parts.append(data)


class RecordingRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        return _loop_result(plan)


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


def _fixture_parser() -> InitialFixtureParser:
    parser = InitialFixtureParser()
    parser.feed(_fixture_source())
    return parser


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


def _action(tool_name=experiment.CLICK_TOOL_NAME, arguments=None) -> Action:
    return Action(
        tool_name=tool_name,
        arguments=arguments or {"x": 10, "y": 20},
        reason="synthetic Experiment 09 action",
    )


def _state_with_actions(plan, action_specs, final_status) -> AgentState:
    state = AgentState(user_task=plan.task_goal)
    state.start()
    for spec in action_specs:
        action = _action(
            tool_name=spec.get("tool_name", experiment.CLICK_TOOL_NAME),
            arguments=spec.get("arguments"),
        )
        success = spec.get("success", True)
        result = ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            error=None if success else "synthetic failure",
        )
        state.record_step(action, result)
    if final_status is AgentStatus.SUCCEEDED:
        state.succeed()
    else:
        state.fail("terminal failure")
    return state


def _loop_result(
    plan,
    *,
    status=AgentLoopStatus.COMPLETED,
    completed_plan_steps=experiment.EXPECTED_PLAN_STEPS,
    action_specs=None,
) -> AgentLoopResult:
    if action_specs is None:
        action_specs = (
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 110, "y": 220}},
            {"arguments": {"x": 210, "y": 320}},
        )
    final_status = (
        AgentStatus.SUCCEEDED
        if status is AgentLoopStatus.COMPLETED
        else AgentStatus.FAILED
    )
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=_state_with_actions(plan, action_specs, final_status),
        completed_plan_steps=completed_plan_steps,
        reason=f"{status.value} synthetic result",
    )


def _run_main_with_payload(payload, capsys):
    code = experiment.main([], client=_client_for_payload(payload))
    return code, capsys.readouterr().out


def test_formal_task_constant():
    assert experiment.TASK == "Complete the deterministic dynamic UI workflow"
    assert experiment.EXPECTED_TASK_GOAL == experiment.TASK


def test_fake_response_payload_matches_required_semantic_plan():
    assert experiment.build_fake_response_payload() == {
        "task_goal": "Complete the deterministic dynamic UI workflow",
        "steps": [
            {
                "goal": "Start the dynamic workflow",
                "operation": "click_target",
                "action_target": {
                    "text": "DYNAMIC_START_09",
                    "element_types": ["button"],
                },
                "verification_target": {
                    "text": "DYNAMIC_CONTINUE_09",
                    "element_types": ["button"],
                },
                "max_attempts": 1,
            },
            {
                "goal": "Continue after the layout changes",
                "operation": "click_target",
                "action_target": {
                    "text": "DYNAMIC_CONTINUE_09",
                    "element_types": ["button"],
                },
                "verification_target": {
                    "text": "DYNAMIC_CONFIRM_09",
                    "element_types": ["button"],
                },
                "max_attempts": 1,
            },
            {
                "goal": "Confirm the foreground workflow step",
                "operation": "click_target",
                "action_target": {
                    "text": "DYNAMIC_CONFIRM_09",
                    "element_types": ["button"],
                },
                "verification_target": {
                    "text": "DYNAMIC_COMPLETE_09",
                    "element_types": ["button"],
                },
                "max_attempts": 1,
            },
        ],
    }


def test_fake_client_returns_provider_style_json_and_records_prompts():
    client = experiment.DeterministicFakeLLMClient()
    response = client.generate(system_prompt="system", user_prompt="user")

    assert json.loads(response) == experiment.build_fake_response_payload()
    assert client.call_count == 1
    assert client.calls == [{"system_prompt": "system", "user_prompt": "user"}]


def test_successful_reasoning_uses_llm_reasoner_and_accepts_plan():
    client = experiment.DeterministicFakeLLMClient()
    run = experiment.run_deterministic_reasoning(client)

    assert run.result.status is ReasoningStatus.READY
    assert isinstance(run.result.plan, StructuredPlan)
    assert run.client_call_count == 1
    assert client.call_count == 1
    assert experiment.reasoning_acceptance_failures(run) == ()


def test_exact_three_step_plan_order_and_targets():
    plan = _plan()

    assert plan.task_goal == experiment.EXPECTED_TASK_GOAL
    assert len(plan.steps) == 3
    assert tuple(step.goal for step in plan.steps) == tuple(
        step[0] for step in experiment.EXPECTED_STEPS
    )
    assert tuple(step.action_target.text for step in plan.steps) == (
        "DYNAMIC_START_09",
        "DYNAMIC_CONTINUE_09",
        "DYNAMIC_CONFIRM_09",
    )
    assert tuple(step.verification_target.text for step in plan.steps) == (
        "DYNAMIC_CONTINUE_09",
        "DYNAMIC_CONFIRM_09",
        "DYNAMIC_COMPLETE_09",
    )


def test_exact_operations_element_types_and_attempt_limits():
    plan = _plan()

    assert all(step.operation is PlanOperation.CLICK_TARGET for step in plan.steps)
    assert all(step.max_attempts == 1 for step in plan.steps)
    assert all(
        step.action_target.element_types == ("button",)
        for step in plan.steps
    )
    assert all(
        step.verification_target.element_types == ("button",)
        for step in plan.steps
    )


def test_reasoned_plan_contains_no_actions_or_coordinate_authority():
    plan = _plan()

    for step in plan.steps:
        values = (
            step.goal,
            step.operation,
            step.action_target,
            step.verification_target,
            step.max_attempts,
        )
        assert not any(isinstance(value, Action) for value in values)
        assert step.action_target.identifier is None
        assert step.verification_target.identifier is None
        assert step.action_target.reference_point is None
        assert step.verification_target.reference_point is None
    assert "x" not in plan.__dataclass_fields__
    assert "y" not in plan.__dataclass_fields__
    assert "coordinates" not in plan.__dataclass_fields__


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(task_goal="Wrong task"), "task goal was"),
        (lambda payload: payload["steps"].pop(), "step count was 2, expected 3"),
        (lambda payload: payload["steps"].reverse(), "step 1 goal was"),
        (
            lambda payload: payload["steps"][0]["action_target"].update(
                text="WRONG_TARGET_09"
            ),
            "step 1 action target text was",
        ),
        (
            lambda payload: payload["steps"][1]["verification_target"].update(
                text="WRONG_COMPLETE_09"
            ),
            "step 2 verification target text was",
        ),
        (
            lambda payload: payload["steps"][2].update(max_attempts=2),
            "step 3 max_attempts was 2, expected 1",
        ),
        (
            lambda payload: payload["steps"][0]["action_target"].update(
                element_types=[]
            ),
            "step 1 action element_types was (), expected ('button',)",
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

    code, output = _run_main_with_payload(payload, capsys)

    assert code == 1
    assert "Reasoning acceptance: failed" in output
    assert message in output


def test_malformed_provider_output_fails_closed(capsys):
    client = experiment.DeterministicFakeLLMClient("{")

    code = experiment.main([], client=client)
    output = capsys.readouterr().out

    assert code == 1
    assert "Reasoning acceptance: failed" in output
    assert "reasoning status was blocked" in output
    assert client.call_count == 1


def test_non_ready_reasoning_result_fails_acceptance(capsys):
    run = experiment.ReasoningRun(
        result=ReasoningResult(
            status=ReasoningStatus.BLOCKED,
            plan=None,
            reason="blocked for test",
        ),
        client_call_count=1,
    )

    code = experiment.main([], run_builder=lambda: run)
    output = capsys.readouterr().out

    assert code == 1
    assert "Reasoning acceptance: failed" in output
    assert "reasoning status was blocked" in output


def test_default_cli_is_dry_safe_and_does_not_call_live_runner(capsys):
    runner = FailingRunner()
    prerequisites = FailingPrerequisites()

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
    assert "Execution mode: disabled" in output
    assert "Action execution count: 0" in output
    assert "Live API request: no" in output


def test_execute_mode_passes_exact_reasoned_plan_to_runner(capsys):
    client = experiment.DeterministicFakeLLMClient()
    runner = RecordingRunner()
    prerequisites = RecordingPrerequisites(True)
    sleeps = []

    code = experiment.main(
        ["--execute", "--wait-seconds", "0"],
        client=client,
        runner=runner,
        prerequisite_checker=prerequisites,
        sleeper=sleeps.append,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert client.call_count == 1
    assert len(runner.calls) == 1
    assert isinstance(runner.calls[0], StructuredPlan)
    assert experiment.reasoning_acceptance_failures(
        experiment.ReasoningRun(
            result=ReasoningResult(
                status=ReasoningStatus.READY,
                plan=runner.calls[0],
                reason="ready",
            ),
            client_call_count=1,
        )
    ) == ()
    assert prerequisites.calls == [experiment.FIXTURE_PATH]
    assert sleeps == []
    assert "Execution mode: enabled" in output
    assert "AgentLoop acceptance: passed" in output
    assert "Recovery retry demonstrated: no" in output


def test_execute_mode_stops_before_runner_when_prerequisites_fail(capsys):
    runner = FailingRunner()
    prerequisites = RecordingPrerequisites(False)

    code = experiment.main(
        ["--execute", "--wait-seconds", "0"],
        runner=runner,
        prerequisite_checker=prerequisites,
        sleeper=lambda _seconds: pytest.fail("sleep should not run"),
    )

    assert code == 1
    assert runner.calls == []
    assert prerequisites.calls == [experiment.FIXTURE_PATH]


def test_successful_agent_loop_result_passes_acceptance(capsys):
    plan = _plan()
    result = _loop_result(plan)

    code = experiment.print_agent_loop_acceptance_result(result, plan)
    output = capsys.readouterr().out

    assert code == 0
    assert "AgentLoop acceptance: passed" in output
    assert "Completed plan steps: 3 / 3" in output
    assert "Action executions: 3" in output


@pytest.mark.parametrize(
    ("result_builder", "message"),
    [
        (
            lambda plan: _loop_result(plan, status=AgentLoopStatus.BLOCKED),
            "Agent loop status was blocked",
        ),
        (
            lambda plan: _with_state_status(_loop_result(plan), AgentStatus.FAILED),
            "Agent state was failed",
        ),
        (
            lambda plan: _with_completed_steps(_loop_result(plan), 2),
            "completed plan steps were 2, not 3",
        ),
        (
            lambda plan: _loop_result(
                plan,
                action_specs=(
                    {"arguments": {"x": 1, "y": 1}},
                    {"arguments": {"x": 2, "y": 2}},
                    {"arguments": {"x": 3, "y": 3}},
                    {"arguments": {"x": 4, "y": 4}},
                ),
            ),
            "Action execution count was 4, not 3",
        ),
        (
            lambda plan: _loop_result(
                plan,
                action_specs=(
                    {"arguments": {"x": 1, "y": 1}},
                    {"arguments": {"x": 2, "y": 2}, "success": False},
                    {"arguments": {"x": 3, "y": 3}},
                ),
            ),
            "attempt 2 ToolResult failed",
        ),
        (
            lambda plan: _loop_result(
                plan,
                action_specs=(
                    {"arguments": {"x": 1, "y": 1}},
                    {"tool_name": "type_text", "arguments": {"text": "bad"}},
                    {"arguments": {"x": 3, "y": 3}},
                ),
            ),
            "attempt 2 tool was type_text",
        ),
    ],
)
def test_agent_loop_acceptance_rejects_invalid_live_results(
    result_builder,
    message,
):
    plan = _plan()

    failures = experiment.agent_loop_acceptance_failures(
        result_builder(plan),
        plan,
    )

    assert any(message in failure for failure in failures)


def _with_state_status(result, status):
    result.state.status = status
    return result


def _with_completed_steps(result, completed_steps):
    object.__setattr__(result, "completed_plan_steps", completed_steps)
    return result


def test_result_for_different_plan_object_fails_acceptance():
    plan = _plan()
    other_plan = _plan()
    result = _loop_result(other_plan)

    failures = experiment.agent_loop_acceptance_failures(result, plan)

    assert "AgentLoopResult plan was not the reasoned plan object" in failures


def test_fixture_paths_follow_phase04_conventions():
    assert experiment.FIXTURE_PATH.name == "experiment_09_dynamic_ui.html"
    assert experiment.CAPTURE_PATH.name == "experiment_09_dynamic_ui.png"
    assert experiment.FIXTURE_PATH.is_file()


def test_fixture_initial_state_contains_only_start_button():
    parser = _fixture_parser()

    assert parser.buttons == ["DYNAMIC_START_09"]
    assert len(parser.button_attrs) == 1
    assert parser.button_attrs[0].get("aria-label") == "DYNAMIC_START_09"
    assert "disabled" not in parser.button_attrs[0]


def test_fixture_contains_required_dynamic_targets_and_no_races():
    source = _fixture_source()

    for target in (
        "DYNAMIC_START_09",
        "DYNAMIC_CONTINUE_09",
        "DYNAMIC_CANCEL_09",
        "DYNAMIC_CONFIRM_09",
        "DYNAMIC_COMPLETE_09",
    ):
        assert target in source

    assert "stage.replaceChildren(panel)" in source
    assert "setTimeout" not in source
    assert "setInterval" not in source
    assert "requestAnimationFrame" not in source
    assert "Math.random" not in source


def test_fixture_modal_decoy_and_confirm_are_distinct():
    source = _fixture_source()

    assert 'makeButton("dynamic-cancel-09", "DYNAMIC_CANCEL_09")' in source
    assert 'makeButton("dynamic-confirm-09", "DYNAMIC_CONFIRM_09")' in source
    assert 'confirm.addEventListener("click", renderCompleteState)' in source
    assert "Decoy selected; foreground state remains active." in source


def test_fixture_completion_marker_uses_accessible_noninteractive_button():
    source = _fixture_source()
    completion_section = source[source.index("function renderCompleteState") :]

    assert '"DYNAMIC_COMPLETE_09"' in completion_section
    assert 'completion.className = "completion-marker";' in completion_section
    assert "completion.tabIndex = -1;" in completion_section
    assert "completion.addEventListener" not in completion_section
    assert "pointer-events: none;" in source
    assert "disabled" not in source


def test_direct_script_help_exits_zero_without_running_experiment():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--execute" in result.stdout
    assert "Experiment acceptance:" not in result.stdout
    assert result.stderr == ""


def test_direct_dry_run_script_succeeds_offline():
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
    assert "LLM provider: deterministic fake" in result.stdout
    assert "Live API request: no" in result.stdout
    assert "Observation count: 0" in result.stdout
    assert "Action execution count: 0" in result.stdout
    assert result.stderr == ""


def test_importing_experiment_performs_no_execution_or_output():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "importlib.import_module("
                "'experiments.phase04_ui_grounding_task_reasoning."
                "experiment_09_dynamic_ui'"
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
        "experiment_09_dynamic_ui"
    )

    assert module is experiment
