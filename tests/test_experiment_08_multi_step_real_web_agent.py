from pathlib import Path
import os
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
    PlanOperation,
    PlanStep,
    StructuredPlan,
    WebTextInputStep,
)
from experiments.phase05_real_web_autonomy import (
    experiment_08_multi_step_real_web_agent as experiment,
)


SCRIPT_PATH = Path(experiment.__file__).resolve()


class FakeAccessibility:
    available = True
    trusted = True

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def is_trusted(cls):
        return cls.trusted


class FakeExecutorBuilder:
    def __init__(self, executor=None) -> None:
        self.executor = executor if executor is not None else experiment.SyntheticExecutor()
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.executor


class FakeLiveEnvironment:
    def __init__(
        self,
        *,
        capture_path,
        initial_observation=None,
        final_observation=None,
        text_input_observations=None,
        submit_snapshots=None,
        write_candidate=False,
        sleeper=None,
        stabilization_wait_seconds=0.0,
    ) -> None:
        self.capture_path = Path(capture_path)
        self.initial_observation = initial_observation or _initial_observation()
        self.final_observation = final_observation or _final_observation()
        self.text_input_observations = list(
            text_input_observations
            if text_input_observations is not None
            else experiment._successful_text_input_observations()
        )
        self.perception_engine = experiment.SyntheticPerceptionEngine(
            submit_snapshots
            if submit_snapshots is not None
            else experiment._successful_submit_snapshots()
        )
        self.write_candidate = write_candidate
        self.sleeper = sleeper
        self.stabilization_wait_seconds = stabilization_wait_seconds
        self.observe_calls = 0

    def observe(self):
        self.observe_calls += 1
        if self.sleeper is not None:
            self.sleeper(self.stabilization_wait_seconds)
        if self.write_candidate:
            self.capture_path.parent.mkdir(parents=True, exist_ok=True)
            self.capture_path.write_text("candidate evidence", encoding="utf-8")
        if self.observe_calls == 1:
            return self.initial_observation
        return self.final_observation

    def text_input_observe(self):
        if not self.text_input_observations:
            raise AssertionError("unexpected text-input observation")
        observation = self.text_input_observations.pop(0)
        return observation


class FakeEnvironmentBuilder:
    def __init__(self, environment: FakeLiveEnvironment | None = None, **kwargs):
        self.environment = environment
        self.kwargs = kwargs
        self.capture_paths = []

    def __call__(
        self,
        *,
        capture_path,
        sleeper=None,
        stabilization_wait_seconds=0.0,
    ):
        self.capture_paths.append(Path(capture_path))
        if self.environment is not None:
            return self.environment
        return FakeLiveEnvironment(
            capture_path=capture_path,
            sleeper=sleeper,
            stabilization_wait_seconds=stabilization_wait_seconds,
            **self.kwargs,
        )


def _path_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(experiment.PROJECT_ROOT / "src")
        + os.pathsep
        + str(experiment.PROJECT_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return env


def _action(tool_name: str, arguments=None) -> Action:
    return Action(
        tool_name=tool_name,
        arguments=arguments or {},
        reason="synthetic acceptance action",
    )


def _state_with_tools(
    plan: StructuredPlan,
    tools: tuple[str, ...],
    *,
    final_status: AgentStatus = AgentStatus.SUCCEEDED,
) -> AgentState:
    state = AgentState(user_task=plan.task_goal)
    state.start()

    for tool_name in tools:
        arguments = {"text": experiment.SEARCH_INPUT_TEXT}
        if tool_name != "type_text":
            arguments = {"x": 10, "y": 20}
        action = _action(tool_name, arguments)
        state.record_step(
            action,
            ToolResult(
                action_id=action.action_id,
                tool_name=action.tool_name,
                success=True,
            ),
        )

    if final_status is AgentStatus.SUCCEEDED:
        state.succeed()
    else:
        state.fail("synthetic failure")

    return state


def _loop_result(
    plan: StructuredPlan,
    *,
    status: AgentLoopStatus = AgentLoopStatus.COMPLETED,
    completed_plan_steps: int = experiment.EXPECTED_PLAN_STEPS,
    tools: tuple[str, ...] = experiment.EXPECTED_ACTION_ORDER,
) -> AgentLoopResult:
    state_status = (
        AgentStatus.SUCCEEDED
        if status is AgentLoopStatus.COMPLETED
        else AgentStatus.FAILED
    )
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=_state_with_tools(
            plan,
            tools,
            final_status=state_status,
        ),
        completed_plan_steps=completed_plan_steps,
        reason="synthetic loop result",
    )


def _initial_observation(**kwargs):
    elements = kwargs.pop(
        "elements",
        (
            experiment._search_field_element(value=None),
            experiment._submit_element(),
        ),
    )
    return experiment._web_observation(elements=elements, seconds=0, **kwargs)


def _final_observation(**kwargs):
    elements = kwargs.pop(
        "elements",
        (
            experiment._submit_element(),
            experiment._result_marker_element(),
        ),
    )
    return experiment._web_observation(elements=elements, seconds=4, **kwargs)


def _run_live(
    tmp_path,
    *,
    environment_builder=None,
    executor_builder=None,
    platform_name="darwin",
    accessibility_cls=FakeAccessibility,
    sleeper=None,
):
    if environment_builder is None:
        environment_builder = FakeEnvironmentBuilder(write_candidate=True)
    if executor_builder is None:
        executor_builder = FakeExecutorBuilder()
    sleeps = [] if sleeper is None else sleeper
    code = experiment.run_live_acceptance(
        environment_builder=environment_builder,
        executor_builder=executor_builder,
        platform_name=platform_name,
        accessibility_cls=accessibility_cls,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=sleeps.append if sleeper is None else sleeper,
        wait_seconds=0,
    )
    return code, environment_builder, executor_builder, sleeps


def test_plan_shape_has_exactly_two_semantic_steps():
    plan = experiment.build_structured_plan()

    assert isinstance(plan, StructuredPlan)
    assert len(plan.steps) == 2
    assert plan.task_goal == experiment.TASK_GOAL


def test_plan_step_types_and_order_are_exact():
    first, second = experiment.build_structured_plan().steps

    assert isinstance(first, WebTextInputStep)
    assert first.operation is PlanOperation.TYPE_INTO_TARGET
    assert isinstance(second, PlanStep)
    assert second.operation is PlanOperation.CLICK_TARGET


def test_plan_uses_exact_live_semantic_targets_and_constants():
    first, second = experiment.build_structured_plan().steps

    assert first.goal == experiment.WEB_TEXT_GOAL
    assert first.target == experiment.SEARCH_FIELD_TARGET
    assert first.target.text == "Search This Site"
    assert first.target.element_types == ("text_field",)
    assert first.target.minimum_confidence == 0.70
    assert first.input_text == "typing"
    assert first.max_attempts == 1

    assert second.goal == experiment.SUBMIT_GOAL
    assert second.action_target == experiment.SUBMIT_TARGET_SPEC
    assert second.action_target.text == "GO"
    assert second.action_target.element_types == ("button",)
    assert second.action_target.minimum_confidence == 0.70
    assert second.verification_target == experiment.RESULT_MARKER_SPEC
    assert second.verification_target.text == "Results"
    assert second.verification_target.element_types == ("heading",)
    assert second.verification_target.minimum_confidence == 0.70
    assert second.max_attempts == 1


def test_successful_full_synthetic_workflow_completes():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(plan)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.completed_plan_steps == 2
    assert experiment.acceptance_failures(result, plan) == ()


def test_successful_synthetic_workflow_records_exact_three_actions():
    result = experiment.run_synthetic_agent_loop(
        experiment.build_structured_plan()
    )

    assert len(result.state.steps) == 3
    assert len(result.state.steps) == experiment.EXPECTED_ACTION_EXECUTIONS
    assert len(result.state.steps) == len(
        {record.action.action_id for record in result.state.steps}
    )


def test_successful_synthetic_workflow_records_exact_tool_order():
    result = experiment.run_synthetic_agent_loop(
        experiment.build_structured_plan()
    )

    assert tuple(record.action.tool_name for record in result.state.steps) == (
        "click_mouse",
        "type_text",
        "click_mouse",
    )


def test_text_input_actions_are_not_duplicate_recorded():
    result = experiment.run_synthetic_agent_loop(
        experiment.build_structured_plan()
    )
    tools = tuple(record.action.tool_name for record in result.state.steps)

    assert tools[:2] == ("click_mouse", "type_text")
    assert tools.count("type_text") == 1
    assert result.state.steps[1].action.arguments == {
        "text": experiment.SEARCH_INPUT_TEXT
    }


def test_submit_click_is_not_duplicate_recorded():
    result = experiment.run_synthetic_agent_loop(
        experiment.build_structured_plan()
    )
    tools = tuple(record.action.tool_name for record in result.state.steps)

    assert tools.count("click_mouse") == 2
    assert result.state.steps[2].action.tool_name == "click_mouse"
    assert result.state.steps[2].action is not result.state.steps[0].action


def test_no_scroll_actions_in_deterministic_happy_path():
    result = experiment.run_synthetic_agent_loop(
        experiment.build_structured_plan()
    )

    assert "scroll" not in tuple(
        record.action.tool_name for record in result.state.steps
    )


def test_plan_contains_no_coordinates_actions_or_tool_results():
    plan = experiment.build_structured_plan()

    assert experiment.plan_acceptance_failures(plan) == ()
    for step in plan.steps:
        assert "x" not in step.__dataclass_fields__
        assert "y" not in step.__dataclass_fields__
        assert "coordinates" not in step.__dataclass_fields__


def test_text_input_blocked_fails_synthetic_workflow():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(
        plan,
        text_input_observations=(
            experiment._text_input_observation(value="already filled"),
        ),
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert tuple(result.state.steps) == ()
    assert experiment.acceptance_failures(result, plan)


def test_text_input_verification_failure_fails_synthetic_workflow():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(
        plan,
        text_input_observations=(
            experiment._text_input_observation(value=None, seconds=0),
            experiment._text_input_observation(value="wrong", seconds=1),
        ),
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert tuple(record.action.tool_name for record in result.state.steps) == (
        "click_mouse",
        "type_text",
    )
    assert experiment.acceptance_failures(result, plan)


def test_submit_grounding_failure_fails_synthetic_workflow():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(
        plan,
        submit_snapshots=(
            experiment._snapshot(elements=(), seconds=2),
        ),
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert tuple(record.action.tool_name for record in result.state.steps) == (
        "click_mouse",
        "type_text",
    )
    assert experiment.acceptance_failures(result, plan)


def test_submit_tool_result_failure_fails_synthetic_workflow():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(
        plan,
        executor=experiment.SyntheticExecutor(fail_tool_call=3),
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert len(result.state.steps) == 3
    assert result.state.steps[2].result.success is False
    assert experiment.acceptance_failures(result, plan)


def test_submit_verification_failure_fails_synthetic_workflow():
    plan = experiment.build_structured_plan()
    result = experiment.run_synthetic_agent_loop(
        plan,
        submit_snapshots=(
            experiment._snapshot(
                elements=(experiment._submit_element(),),
                seconds=2,
            ),
            experiment._snapshot(
                elements=(experiment._submit_element(),),
                seconds=3,
            ),
        ),
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert len(result.state.steps) == 3
    assert experiment.acceptance_failures(result, plan)


def test_default_direct_run_remains_offline_synthetic():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=experiment.PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Experiment mode: synthetic" in result.stdout
    assert "Live browser actions: no" in result.stdout
    assert "OpenAI request: no" in result.stdout
    assert "Synthetic execution: yes" in result.stdout
    assert "Experiment acceptance: passed" in result.stdout
    assert result.stderr == ""


def test_help_does_not_run_acceptance():
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


def test_execute_path_counts_down_before_live_setup(capsys, tmp_path):
    sleeps = []

    code = experiment.run_live_acceptance(
        environment_builder=FakeEnvironmentBuilder(write_candidate=True),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=sleeps.append,
        wait_seconds=2,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "2..." in output
    assert "1..." in output
    assert sleeps == [1, 1, 0.5, 0.5]


def test_live_environment_rejects_invalid_stabilization_delay(tmp_path):
    class Access:
        pass

    for value in (-0.1, "0.5", True):
        with pytest.raises(ValueError, match="stabilization_wait_seconds"):
            experiment.LiveWebEnvironment(
                capture_path=tmp_path / "candidate.png",
                accessibility=Access(),
                perception_engine=experiment.SyntheticPerceptionEngine(
                    (
                        experiment._snapshot(
                            elements=(),
                            seconds=0,
                        ),
                    )
                ),
                sleeper=lambda _seconds: None,
                stabilization_wait_seconds=value,
            )


def test_live_observe_uses_injected_stabilization_sleeper(tmp_path):
    sleeps = []

    class Access:
        def read_frontmost_application_name(self):
            return experiment.EXPECTED_APPLICATION_NAME

        def read_frontmost_viewport(self):
            return experiment._viewport()

        def read_frontmost_semantic_elements(self):
            return []

    environment = experiment.LiveWebEnvironment(
        capture_path=tmp_path / "candidate.png",
        accessibility=Access(),
        perception_engine=experiment.SyntheticPerceptionEngine(
            (
                experiment._snapshot(
                    elements=(),
                    seconds=0,
                ),
            )
        ),
        sleeper=sleeps.append,
        stabilization_wait_seconds=0.25,
    )

    environment.observe()

    assert sleeps == [0.25]


@pytest.mark.parametrize(
    ("platform_name", "available", "trusted", "message"),
    [
        ("linux", True, True, "platform is not macOS"),
        ("darwin", False, True, "Accessibility is unavailable"),
        ("darwin", True, False, "Accessibility is not trusted"),
    ],
)
def test_execute_path_requires_macos_and_trusted_accessibility(
    capsys,
    tmp_path,
    platform_name,
    available,
    trusted,
    message,
):
    class Access:
        @staticmethod
        def is_available():
            return available

        @staticmethod
        def is_trusted():
            return trusted

    executor_builder = FakeExecutorBuilder()
    code, builder, executor_builder, _sleeps = _run_live(
        tmp_path,
        environment_builder=FakeEnvironmentBuilder(),
        executor_builder=executor_builder,
        platform_name=platform_name,
        accessibility_cls=Access,
    )
    output = capsys.readouterr().out

    assert code == 1
    assert message in output
    assert builder.capture_paths == []
    assert executor_builder.calls == 0


@pytest.mark.parametrize(
    ("initial_observation", "message"),
    [
        (_initial_observation(app="Safari"), "frontmost app"),
        (_initial_observation(viewport=None), "viewport was unavailable"),
        (_initial_observation(warnings=("ocr warning",)), "perception warnings"),
        (
            _initial_observation(elements=(experiment._submit_element(),)),
            "Search This Site grounding",
        ),
        (
            _initial_observation(
                elements=(
                    experiment._search_field_element(value="typing"),
                    experiment._submit_element(),
                )
            ),
            "field was not empty",
        ),
        (
            _initial_observation(
                elements=(experiment._search_field_element(value=None),)
            ),
            "GO grounding",
        ),
        (
            _initial_observation(
                elements=(
                    experiment._search_field_element(value=None),
                    experiment._submit_element(),
                    experiment._result_marker_element(),
                )
            ),
            "Results grounding",
        ),
    ],
)
def test_live_precondition_failures_block_before_actions(
    capsys,
    tmp_path,
    initial_observation,
    message,
):
    executor_builder = FakeExecutorBuilder()
    code, _builder, executor_builder, _sleeps = _run_live(
        tmp_path,
        environment_builder=FakeEnvironmentBuilder(
            initial_observation=initial_observation
        ),
        executor_builder=executor_builder,
    )
    output = capsys.readouterr().out

    assert code == 1
    assert message in output
    assert executor_builder.calls == 0
    assert executor_builder.executor.actions == []


def test_live_agent_loop_completed_is_accepted(capsys, tmp_path):
    code, _builder, executor_builder, _sleeps = _run_live(tmp_path)
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment mode: live" in output
    assert "Agent loop status: completed" in output
    assert "Agent state: succeeded" in output
    assert "Completed plan steps: 2 / 2" in output
    assert "Action tool order: ('click_mouse', 'type_text', 'click_mouse')" in output
    assert "Final Results grounding status: resolved" in output
    assert executor_builder.calls == 1


def test_acceptance_rejects_wrong_completed_step_count():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        plan,
        status=AgentLoopStatus.EXHAUSTED,
        completed_plan_steps=1,
        tools=("click_mouse", "type_text", "click_mouse"),
    )

    failures = experiment.acceptance_failures(result, plan)

    assert any("completed plan steps" in failure for failure in failures)


def test_acceptance_rejects_unexpected_scroll():
    plan = experiment.build_structured_plan()
    result = _loop_result(
        plan,
        tools=("click_mouse", "type_text", "scroll"),
    )

    failures = experiment.acceptance_failures(result, plan)

    assert any("scroll" in failure for failure in failures)


@pytest.mark.parametrize(
    "tools",
    [
        ("click_mouse", "type_text", "type_text"),
        ("click_mouse", "click_mouse", "click_mouse"),
        ("type_text", "type_text", "click_mouse"),
    ],
)
def test_acceptance_rejects_duplicate_or_missing_click_type_records(tools):
    plan = experiment.build_structured_plan()
    result = _loop_result(plan, tools=tools)

    failures = experiment.acceptance_failures(result, plan)

    assert failures


def test_acceptance_rejects_final_results_unresolved():
    plan = experiment.build_structured_plan()
    result = _loop_result(plan)
    final_report = experiment._final_observation_report(
        _final_observation(elements=(experiment._submit_element(),))
    )

    failures = experiment.acceptance_failures(
        result,
        plan,
        final_report=final_report,
        require_final_report=True,
    )

    assert any("final Results grounding" in failure for failure in failures)


def test_acceptance_rejects_final_app_not_chrome():
    plan = experiment.build_structured_plan()
    result = _loop_result(plan)
    final_report = experiment._final_observation_report(
        _final_observation(app="Safari")
    )

    failures = experiment.acceptance_failures(
        result,
        plan,
        final_report=final_report,
        require_final_report=True,
    )

    assert any("final frontmost app" in failure for failure in failures)


def test_acceptance_rejects_final_warnings():
    plan = experiment.build_structured_plan()
    result = _loop_result(plan)
    final_report = experiment._final_observation_report(
        _final_observation(warnings=("late warning",))
    )

    failures = experiment.acceptance_failures(
        result,
        plan,
        final_report=final_report,
        require_final_report=True,
    )

    assert any("final perception warnings" in failure for failure in failures)


def test_failed_live_acceptance_does_not_promote_formal_evidence(tmp_path):
    candidate = tmp_path / "candidate.png"
    formal = tmp_path / "formal.png"
    formal.write_text("protected formal evidence", encoding="utf-8")
    env_builder = FakeEnvironmentBuilder(
        final_observation=_final_observation(
            elements=(experiment._submit_element(),)
        ),
        write_candidate=True,
    )

    code = experiment.run_live_acceptance(
        environment_builder=env_builder,
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=candidate,
        formal_evidence_path=formal,
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert code == 1
    assert candidate.read_text(encoding="utf-8") == "candidate evidence"
    assert formal.read_text(encoding="utf-8") == "protected formal evidence"


def test_successful_live_acceptance_fails_when_candidate_missing(
    capsys,
    tmp_path,
):
    candidate = tmp_path / "candidate.png"
    formal = tmp_path / "formal.png"

    code = experiment.run_live_acceptance(
        environment_builder=FakeEnvironmentBuilder(write_candidate=False),
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=candidate,
        formal_evidence_path=formal,
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )
    output = capsys.readouterr().out

    assert code == 1
    assert "candidate evidence file was missing" in output
    assert "Experiment acceptance: passed" not in output
    assert not formal.exists()


def test_successful_live_acceptance_promotes_candidate_evidence(tmp_path):
    candidate = tmp_path / "candidate.png"
    formal = tmp_path / "formal.png"
    env_builder = FakeEnvironmentBuilder(write_candidate=True)

    code = experiment.run_live_acceptance(
        environment_builder=env_builder,
        executor_builder=FakeExecutorBuilder(),
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=candidate,
        formal_evidence_path=formal,
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert code == 0
    assert candidate.read_text(encoding="utf-8") == "candidate evidence"
    assert formal.read_text(encoding="utf-8") == "candidate evidence"
