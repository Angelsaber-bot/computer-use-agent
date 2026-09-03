from html.parser import HTMLParser
import importlib
import inspect
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
from computer_agent.planning import PlanOperation, PlanStep, StructuredPlan
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_08_agent_loop as experiment,
)


SCRIPT_PATH = Path(experiment.__file__).resolve()
FUTURE_MARKERS = (
    experiment.STEP_1_VERIFICATION_TEXT,
    experiment.STEP_2_ACTION_TEXT,
    experiment.STEP_2_VERIFICATION_TEXT,
)


class InitialFixtureParser(HTMLParser):
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
    def __init__(self, result) -> None:
        self.result = result
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        return self.result


class FailingRunner:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, plan):
        self.calls.append(plan)
        pytest.fail("runner should not be called without --execute")


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
        pytest.fail("prerequisites should not run without --execute")


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


def _step_one_handler_source(source: str) -> str:
    start = source.index('getElementById("step-one-target-08")')
    end = source.index("</script>", start)
    return source[start:end]


def _fixture_parser() -> InitialFixtureParser:
    parser = InitialFixtureParser()
    parser.feed(_fixture_source())
    return parser


def _action(
    *,
    tool_name=experiment.CLICK_TOOL_NAME,
    arguments=None,
) -> Action:
    return Action(
        tool_name=tool_name,
        arguments=arguments or {"x": 10, "y": 20},
        reason="synthetic test action",
    )


def _state_with_actions(
    plan,
    action_specs,
    *,
    final_status=AgentStatus.SUCCEEDED,
) -> AgentState:
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
    elif final_status is AgentStatus.FAILED:
        state.fail("terminal failure")
    else:
        raise AssertionError(f"unsupported final status: {final_status}")

    return state


def _loop_result(
    *,
    status=AgentLoopStatus.COMPLETED,
    completed_plan_steps=2,
    action_specs=None,
    reason="synthetic result",
) -> AgentLoopResult:
    plan = experiment.build_structured_plan()
    if action_specs is None:
        action_specs = [
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 110, "y": 120}},
            {"arguments": {"x": 210, "y": 220}},
        ]

    final_status = (
        AgentStatus.SUCCEEDED
        if status is AgentLoopStatus.COMPLETED
        else AgentStatus.FAILED
    )
    state = _state_with_actions(
        plan,
        action_specs,
        final_status=final_status,
    )
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=state,
        completed_plan_steps=completed_plan_steps,
        reason=reason,
    )


def test_deterministic_plan_has_exactly_two_steps():
    plan = experiment.build_structured_plan()

    assert isinstance(plan, StructuredPlan)
    assert len(plan.steps) == 2


def test_deterministic_plan_has_exact_task_goal_and_step_goals():
    plan = experiment.build_structured_plan()

    assert plan.task_goal == (
        "Complete the deterministic Agent Loop workflow"
    )
    assert [step.goal for step in plan.steps] == [
        "Recover and complete the first UI target",
        "Complete the final UI target",
    ]


def test_deterministic_plan_has_exact_operations_and_targets():
    plan = experiment.build_structured_plan()

    assert [step.operation for step in plan.steps] == [
        PlanOperation.CLICK_TARGET,
        PlanOperation.CLICK_TARGET,
    ]
    assert [step.action_target.text for step in plan.steps] == [
        "STEP_1_TARGET_08",
        "STEP_2_TARGET_08",
    ]
    assert [step.verification_target.text for step in plan.steps] == [
        "STEP_1_COMPLETE_08",
        "TASK_COMPLETE_08",
    ]


def test_deterministic_plan_has_exact_attempt_limits():
    first, second = experiment.build_structured_plan().steps

    assert first.max_attempts == 2
    assert second.max_attempts == 1


def test_plan_contains_no_coordinates_or_actions():
    plan = experiment.build_structured_plan()

    assert all(isinstance(step, PlanStep) for step in plan.steps)
    assert not any(
        isinstance(value, Action)
        for step in plan.steps
        for value in (
            step.goal,
            step.operation,
            step.action_target,
            step.verification_target,
            step.max_attempts,
        )
    )
    assert all(
        step.action_target.reference_point is None
        and step.verification_target.reference_point is None
        for step in plan.steps
    )
    assert "x" not in PlanStep.__dataclass_fields__
    assert "y" not in PlanStep.__dataclass_fields__
    assert "coordinates" not in PlanStep.__dataclass_fields__


def test_default_cli_exits_zero_and_reports_execution_disabled():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Execution mode: disabled" in result.stdout
    assert "Run with --execute" in result.stdout
    assert "click_mouse Actions may run" not in result.stdout
    assert result.stderr == ""


def test_default_cli_performs_no_live_execution(capsys):
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
    assert "Observing in" not in result.stdout
    assert result.stderr == ""


def test_importing_experiment_produces_no_output_or_execution():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "importlib.import_module("
                "'experiments.phase04_ui_grounding_task_reasoning."
                "experiment_08_agent_loop'"
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


def test_execute_routing_uses_injected_runner_only(capsys):
    plan = experiment.build_structured_plan()
    runner = RecordingRunner(_loop_result())
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
    assert runner.calls[0] == plan
    assert prerequisites.calls == [experiment.FIXTURE_PATH]
    assert sleeps == []
    assert "Execution mode: enabled" in output
    assert "Experiment acceptance: passed" in output


def test_completed_succeeded_three_executions_passes_acceptance(capsys):
    plan = experiment.build_structured_plan()
    result = _loop_result()

    code = experiment.print_acceptance_result(result, plan)
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment acceptance: passed" in output
    assert "Agent loop status: completed" in output
    assert "Agent state: succeeded" in output
    assert "Completed plan steps: 2 / 2" in output
    assert "Action executions: 3" in output
    assert "Recovery retry demonstrated: yes" in output


@pytest.mark.parametrize(
    ("action_specs", "message"),
    [
        (
            [
                {"arguments": {"x": 10, "y": 20}},
                {"arguments": {"x": 110, "y": 120}},
            ],
            "not 3",
        ),
        (
            [
                {"arguments": {"x": 10, "y": 20}},
                {"arguments": {"x": 110, "y": 120}},
                {"arguments": {"x": 210, "y": 220}},
                {"arguments": {"x": 310, "y": 320}},
            ],
            "not 3",
        ),
    ],
)
def test_wrong_execution_counts_fail_acceptance(action_specs, message):
    plan = experiment.build_structured_plan()
    result = _loop_result(action_specs=action_specs)

    failures = experiment.acceptance_failures(result, plan)

    assert any(message in failure for failure in failures)


def test_first_second_identical_click_arguments_fail_recovery_evidence():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        action_specs=[
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 210, "y": 220}},
        ]
    )

    failures = experiment.acceptance_failures(result, plan)

    assert "first and second click arguments were identical" in failures


def test_first_second_different_click_arguments_pass_recovery_evidence():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        action_specs=[
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 110, "y": 120}},
            {"arguments": {"x": 210, "y": 220}},
        ]
    )

    failures = experiment.acceptance_failures(result, plan)

    assert failures == []


def test_failed_tool_result_fails_acceptance():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        action_specs=[
            {"arguments": {"x": 10, "y": 20}},
            {"arguments": {"x": 110, "y": 120}, "success": False},
            {"arguments": {"x": 210, "y": 220}},
        ]
    )

    failures = experiment.acceptance_failures(result, plan)

    assert any("ToolResult failed" in failure for failure in failures)


def test_non_click_mouse_action_fails_acceptance():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        action_specs=[
            {"arguments": {"x": 10, "y": 20}},
            {
                "tool_name": "type_text",
                "arguments": {"text": "not a click"},
            },
            {"arguments": {"x": 210, "y": 220}},
        ]
    )

    failures = experiment.acceptance_failures(result, plan)

    assert any("tool was type_text" in failure for failure in failures)


@pytest.mark.parametrize(
    "status",
    [
        AgentLoopStatus.BLOCKED,
        AgentLoopStatus.EXHAUSTED,
    ],
)
def test_terminal_non_completed_results_fail_acceptance(capsys, status):
    plan = experiment.build_structured_plan()
    result = _loop_result(
        status=status,
        completed_plan_steps=0,
        action_specs=[
            {"arguments": {"x": 10, "y": 20}},
        ],
        reason=f"{status.value} reason",
    )

    code = experiment.print_acceptance_result(result, plan)
    output = capsys.readouterr().out

    assert code == 1
    assert f"AgentLoopResult reason: {status.value} reason" in output
    assert f"Agent loop status: {status.value}" in output
    assert "Experiment acceptance: failed" in output


def test_fixture_contains_required_initial_target():
    parser = _fixture_parser()

    assert parser.buttons == ["STEP_1_TARGET_08"]
    assert len(parser.button_attrs) == 1
    assert parser.button_attrs[0].get("aria-label") == "STEP_1_TARGET_08"
    assert "disabled" not in parser.button_attrs[0]


def test_fixture_source_has_deterministic_retry_and_completion_behavior():
    source = _fixture_source()

    assert "stepOneClicks === 1" in source
    assert "retry-position" in source
    assert "button.classList.add(\"retry-position\")" in source
    assert "left: 606px;" in source
    assert "top: 358px;" in source
    assert "button.remove();" in source
    assert "STEP_1_COMPLETE_08" in source
    assert "STEP_2_TARGET_08" in source
    assert "TASK_COMPLETE_08" in source
    assert "Math.random" not in source


def test_step_one_completion_marker_is_created_only_after_second_click():
    handler = _step_one_handler_source(_fixture_source())
    first_click_return = handler.index("return;")
    completion_index = handler.index('"STEP_1_COMPLETE_08"')

    assert completion_index > first_click_return
    assert 'addCompletionMarker(\n          "step-one-complete-08",' in handler
    assert "STEP_1_COMPLETE_08" not in handler[:first_click_return]
    assert "addStepTwoTarget();" in handler[completion_index:]


def test_task_completion_marker_is_created_only_after_step_two_click():
    body = _function_body(_fixture_source(), "addStepTwoTarget")
    click_handler = body[body.index('button.addEventListener("click"') :]

    assert '"TASK_COMPLETE_08"' in click_handler
    assert click_handler.index("button.remove();") < click_handler.index(
        '"TASK_COMPLETE_08"'
    )
    assert 'addCompletionMarker(\n          "task-complete-08",' in (
        click_handler
    )


def test_completion_markers_use_accessible_enabled_button_representation():
    body = _function_body(_fixture_source(), "addCompletionMarker")

    assert 'document.createElement("button")' in body
    assert 'marker.type = "button";' in body
    assert 'marker.tabIndex = -1;' in body
    assert 'marker.setAttribute("aria-label", text);' in body
    assert "marker.textContent = text;" in body
    assert "disabled" not in body
    assert ".disabled" not in body
    assert "setAttribute(\"disabled\"" not in body
    assert "setAttribute(\"role\", \"status\")" not in body


def test_completion_markers_have_no_click_handlers():
    body = _function_body(_fixture_source(), "addCompletionMarker")

    assert "addEventListener" not in body
    assert "onclick" not in body


def test_completion_markers_are_non_interactive_through_css():
    source = _fixture_source()
    rule = source[
        source.index(".completion-marker {") : source.index(
            "#step-one-complete-08"
        )
    ]

    assert "pointer-events: none;" in rule
    assert "cursor: default;" in rule


def test_step_two_target_remains_normal_actionable_button():
    body = _function_body(_fixture_source(), "addStepTwoTarget")

    assert 'document.createElement("button")' in body
    assert 'button.type = "button";' in body
    assert 'button.setAttribute("aria-label", "STEP_2_TARGET_08");' in body
    assert 'button.textContent = "STEP_2_TARGET_08";' in body
    assert 'button.addEventListener("click"' in body
    assert "button.disabled" not in body
    assert 'button.setAttribute("disabled"' not in body
    assert "button.tabIndex = -1" not in body
    assert "button.style.pointerEvents" not in body


def test_fixture_uses_supported_marker_role_without_production_changes():
    source = _fixture_source()

    assert 'document.createElement("button")' in source
    assert 'marker.setAttribute("role", "status")' not in source
    assert "computer_agent.perception" not in source
    assert "AccessibilityReader" not in source


def test_fixture_does_not_initially_render_future_markers():
    parser = _fixture_parser()
    initial_visible_text = "\n".join(parser.visible_texts)
    initial_attrs = "\n".join(parser.visible_attrs)

    for marker in FUTURE_MARKERS:
        assert marker not in initial_visible_text
        assert marker not in initial_attrs

    assert "display: none" not in _fixture_source()
    assert "hidden" not in _fixture_source()


def test_experiment_has_no_llm_openai_or_network_dependency():
    source = inspect.getsource(experiment)
    fixture_source = _fixture_source()
    combined = f"{source}\n{fixture_source}".lower()

    forbidden_terms = (
        "openai",
        "llm",
        "http://",
        "https://",
        "fetch(",
        "xmlhttprequest",
        "websocket",
    )

    assert all(term not in combined for term in forbidden_terms)


def test_experiment_constructs_no_actions_or_coordinates():
    source = inspect.getsource(experiment)

    forbidden_terms = (
        "Action(",
        "BoundingBox",
        ".center",
        "contains_point",
        "arguments.get(",
        "arguments[",
        "reference_point",
    )

    assert all(term not in source for term in forbidden_terms)


def test_live_execution_is_impossible_without_explicit_execute(capsys):
    runner = FailingRunner()

    code = experiment.main(
        ["--wait-seconds", "0"],
        runner=runner,
        prerequisite_checker=lambda _path: pytest.fail(
            "prerequisite check should not run in disabled mode"
        ),
        sleeper=lambda _seconds: pytest.fail(
            "countdown should not run in disabled mode"
        ),
    )
    output = capsys.readouterr().out

    assert code == 0
    assert runner.calls == []
    assert "Execution mode: disabled" in output


def test_module_identity_is_stable():
    module = importlib.import_module(
        "experiments.phase04_ui_grounding_task_reasoning."
        "experiment_08_agent_loop"
    )

    assert module is experiment
