import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from computer_agent.core.models import Action
from computer_agent.planning import PlanOperation, StructuredPlan
from computer_agent.reasoning import ReasoningResult, ReasoningStatus
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_07_llm_reasoner as experiment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments/phase04_ui_grounding_task_reasoning/"
    "experiment_07_llm_reasoner.py"
)


def _run() -> experiment.ReasoningRun:
    return experiment.run_deterministic_reasoning()


def _plan() -> StructuredPlan:
    run = _run()
    assert run.result.status is ReasoningStatus.READY
    assert isinstance(run.result.plan, StructuredPlan)
    return run.result.plan


def _payload() -> dict[str, object]:
    return copy.deepcopy(experiment.build_fake_response_payload())


def _client_for_payload(
    payload: dict[str, object],
) -> experiment.DeterministicFakeLLMClient:
    return experiment.DeterministicFakeLLMClient(json.dumps(payload))


def _run_acceptance_with_payload(payload: dict[str, object], capsys):
    code = experiment.run_acceptance(client=_client_for_payload(payload))
    return code, capsys.readouterr().out


def _path_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT / "src")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    return env


def test_deterministic_acceptance_returns_zero(capsys):
    code = experiment.run_acceptance()
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment acceptance: passed" in output


def test_exact_expected_two_step_plan():
    plan = _plan()

    assert isinstance(plan, StructuredPlan)
    assert plan.task_goal == experiment.EXPECTED_TASK_GOAL
    assert len(plan.steps) == 2
    assert experiment.acceptance_failures(_run()) == ()


def test_exact_step_order():
    plan = _plan()

    assert tuple(step.goal for step in plan.steps) == (
        experiment.STEP_1_GOAL,
        experiment.STEP_2_GOAL,
    )


def test_exact_target_texts():
    plan = _plan()

    assert tuple(step.action_target.text for step in plan.steps) == (
        experiment.STEP_1_ACTION_TARGET_TEXT,
        experiment.STEP_2_ACTION_TARGET_TEXT,
    )
    assert tuple(step.verification_target.text for step in plan.steps) == (
        experiment.STEP_1_VERIFICATION_TARGET_TEXT,
        experiment.STEP_2_VERIFICATION_TARGET_TEXT,
    )


def test_exact_operations():
    plan = _plan()

    assert tuple(step.operation for step in plan.steps) == (
        PlanOperation.CLICK_TARGET,
        PlanOperation.CLICK_TARGET,
    )


def test_exact_max_attempts():
    plan = _plan()

    assert tuple(step.max_attempts for step in plan.steps) == (
        experiment.STEP_MAX_ATTEMPTS,
        experiment.STEP_MAX_ATTEMPTS,
    )


def test_exact_element_types():
    plan = _plan()

    assert tuple(step.action_target.element_types for step in plan.steps) == (
        experiment.EXPECTED_ELEMENT_TYPES,
        experiment.EXPECTED_ELEMENT_TYPES,
    )
    assert tuple(
        step.verification_target.element_types for step in plan.steps
    ) == (
        experiment.EXPECTED_ELEMENT_TYPES,
        experiment.EXPECTED_ELEMENT_TYPES,
    )


def test_fake_client_called_once():
    client = experiment.DeterministicFakeLLMClient()
    run = experiment.run_deterministic_reasoning(client=client)

    assert run.client_call_count == 1
    assert client.call_count == 1


def test_malformed_fake_response_fails_acceptance(capsys):
    client = experiment.DeterministicFakeLLMClient("{")
    code = experiment.run_acceptance(client=client)
    output = capsys.readouterr().out

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "reasoning status was blocked" in output
    assert client.call_count == 1


def test_blocked_result_fails_acceptance(capsys):
    run = experiment.ReasoningRun(
        result=ReasoningResult(
            status=ReasoningStatus.BLOCKED,
            plan=None,
            reason="blocked for test",
        ),
        client_call_count=1,
    )

    code = experiment.run_acceptance(run_builder=lambda: run)
    output = capsys.readouterr().out

    assert code == 1
    assert "Experiment acceptance: failed" in output
    assert "reasoning status was blocked" in output


def test_wrong_task_goal_fails(capsys):
    payload = _payload()
    payload["task_goal"] = "Complete another workflow"

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "task goal was" in output


def test_wrong_step_count_fails(capsys):
    payload = _payload()
    payload["steps"] = payload["steps"][:1]

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "step count was 1, expected 2" in output
    assert "step 2 was missing" in output


def test_wrong_order_fails(capsys):
    payload = _payload()
    first, second = payload["steps"]
    payload["steps"] = [second, first]

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "step 1 goal was" in output
    assert "step 2 goal was" in output


def test_wrong_action_target_fails(capsys):
    payload = _payload()
    payload["steps"][0]["action_target"]["text"] = "WRONG_TARGET_07"

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "step 1 action target text was" in output


def test_wrong_verification_target_fails(capsys):
    payload = _payload()
    payload["steps"][1]["verification_target"]["text"] = (
        "WRONG_COMPLETE_07"
    )

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "step 2 verification target text was" in output


def test_wrong_max_attempts_fails(capsys):
    payload = _payload()
    payload["steps"][1]["max_attempts"] = 2

    code, output = _run_acceptance_with_payload(payload, capsys)

    assert code == 1
    assert "step 2 max_attempts was 2, expected 3" in output


def test_unexpected_action_or_coordinate_authority_absent():
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
        assert "x" not in step.__dataclass_fields__
        assert "y" not in step.__dataclass_fields__
        assert "coordinates" not in step.__dataclass_fields__
        assert step.action_target.reference_point is None
        assert step.verification_target.reference_point is None


def test_direct_script_help_exits_zero_without_running_experiment():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "Experiment acceptance:" not in result.stdout
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
                "experiment_07_llm_reasoner'"
                ")"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_direct_deterministic_script_execution_succeeds():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
        ],
        cwd=PROJECT_ROOT,
        env=_path_env(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert experiment.TITLE in result.stdout
    assert "Experiment acceptance: passed" in result.stdout
    assert "LLM provider: deterministic fake" in result.stdout
    assert "Live API request: no" in result.stdout
    assert "Observation count: 0" in result.stdout
    assert "Action execution count: 0" in result.stdout
    assert result.stderr == ""


def test_imported_module_identity_is_stable():
    module = importlib.import_module(
        "experiments.phase04_ui_grounding_task_reasoning."
        "experiment_07_llm_reasoner"
    )

    assert module is experiment
