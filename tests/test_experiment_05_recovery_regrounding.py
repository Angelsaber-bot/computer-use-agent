from datetime import datetime, timezone
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import (
    ActionGroundingResult,
    ActionGroundingStatus,
    GroundingResult,
    GroundingStatus,
)
from computer_agent.perception.engine import PerceptionSnapshot
from computer_agent.perception.models import BoundingBox, ScreenFrame, UIElement
from computer_agent.recovery import RecoveryResult, RecoveryStatus
from experiments.phase04_ui_grounding_task_reasoning import (
    experiment_05_recovery_regrounding as experiment,
)
from experiments.phase04_ui_grounding_task_reasoning import live_harness_utils


FORMAL_BYTES = b"existing formal evidence"
AFTER_FIRST_BYTES = b"after first evidence"
FINAL_CANDIDATE_BYTES = b"final candidate evidence"


def _time(seconds=0):
    return datetime(
        2026,
        9,
        1,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _element(
    text,
    *,
    identifier=None,
    element_type="button",
    enabled=True,
    source="accessibility",
    x=40,
    y=40,
    width=120,
    height=32,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
        confidence=0.95,
        text=text,
        identifier=identifier,
        enabled=enabled,
        source=source,
    )


def _snapshot(
    path,
    *,
    accessibility=(),
    ocr=(),
    fused=(),
    seconds=0,
    screen_size=(400, 240),
):
    return PerceptionSnapshot(
        frame=ScreenFrame(
            Path(path),
            screen_size[0],
            screen_size[1],
            screen_size[0],
            screen_size[1],
            _time(seconds),
        ),
        image=Image.new("RGB", screen_size),
        accessibility_elements=tuple(accessibility),
        ocr_elements=tuple(ocr),
        fused_elements=tuple(fused),
        warnings=(),
    )


def _marker(source="accessibility"):
    return _element(
        f"prefix {experiment.FIXTURE_MARKER} suffix",
        element_type="text",
        source=source,
    )


def _action_target(*, position="a", state="resolved"):
    if state == "missing":
        return ()

    x, y = {
        "a": (40, 40),
        "b": (220, 120),
    }[position]
    return (
        _element(
            experiment.ACTION_TARGET_TEXT,
            identifier=None
            if state == "text_only"
            else experiment.ACTION_TARGET_IDENTIFIER,
            enabled=state != "unsafe",
            x=x,
            y=y,
        ),
    )


def _verification_target(*, position="b"):
    x, y = {
        "a": (40, 40),
        "b": (220, 120),
    }[position]
    return (
        _element(
            experiment.VERIFICATION_TARGET_TEXT,
            identifier=experiment.ACTION_TARGET_IDENTIFIER,
            x=x,
            y=y,
        ),
    )


def _observation_1(
    path,
    *,
    frame_path=None,
    marker=True,
    target_state="resolved",
    verify=False,
):
    return _snapshot(
        frame_path or path,
        accessibility=(_marker(),) if marker else (),
        fused=(
            *_action_target(position="a", state=target_state),
            *(_verification_target(position="a") if verify else ()),
        ),
        seconds=0,
    )


def _observation_2(
    path,
    *,
    frame_path=None,
    marker=True,
    target_state="resolved",
    seconds=1,
    screen_size=(400, 240),
):
    return _snapshot(
        frame_path or path,
        accessibility=(_marker(),) if marker else (),
        fused=_action_target(position="b", state=target_state),
        seconds=seconds,
        screen_size=screen_size,
    )


def _observation_2_verified(path):
    return _snapshot(
        path,
        accessibility=(_marker(),),
        fused=_verification_target(position="b"),
        seconds=1,
    )


def _observation_3(
    path,
    *,
    frame_path=None,
    marker=True,
    verified=True,
    seconds=2,
):
    return _snapshot(
        frame_path or path,
        accessibility=(_marker(),) if marker else (),
        fused=_verification_target(position="b") if verified else (),
        seconds=seconds,
    )


class ObserverBuilder:
    def __init__(self, snapshots, capture_writes=None):
        self.snapshots = list(snapshots)
        self.capture_writes = dict(capture_writes or {})
        self.paths = []
        self.snapshot_paths = []
        self.observation_count = 0

    def __call__(self, path):
        self.paths.append(Path(path))
        return self

    def observe(self):
        self.observation_count += 1
        path = self.paths[-1]
        if path in self.capture_writes:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.capture_writes[path])

        item = self.snapshots.pop(0)
        if isinstance(item, (OSError, RuntimeError, ValueError)):
            raise item
        snapshot = item(path) if callable(item) else item
        self.snapshot_paths.append(Path(snapshot.frame.image_path))
        return snapshot


class ExecutorBuilder:
    def __init__(self, results=None):
        self.results = list(results or [True, True])
        self.construction_count = 0
        self.execution_count = 0
        self.actions = []

    def __call__(self):
        self.construction_count += 1
        return self

    def execute(self, action):
        self.execution_count += 1
        self.actions.append(action)
        item = self.results.pop(0)
        if isinstance(item, (OSError, RuntimeError, ValueError)):
            raise item

        success = bool(item)
        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            error=None if success else "synthetic tool failure",
        )


class FailingRecovery:
    def prepare_retry(self, **_kwargs):
        raise AssertionError("recovery should not be called")


class StaticRecovery:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def prepare_retry(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class RecordingRecovery:
    def __init__(self):
        self.calls = []
        self.delegate = experiment.ActionRecovery()

    def prepare_retry(self, **kwargs):
        self.calls.append(kwargs)
        return self.delegate.prepare_retry(**kwargs)


class RetryActionFromToolResultRecovery:
    def __init__(self, *, tool_name=None, arguments=None, reuse_action_id=False):
        self.tool_name = tool_name or experiment.CLICK_TOOL_NAME
        self.arguments = arguments if arguments is not None else {"x": 280, "y": 136}
        self.reuse_action_id = reuse_action_id
        self.calls = []

    def prepare_retry(self, **kwargs):
        self.calls.append(kwargs)
        action_kwargs = {}
        if self.reuse_action_id:
            action_kwargs["action_id"] = kwargs["tool_result"].action_id

        return _retry_ready_result_with_action(
            Action(
                tool_name=self.tool_name,
                arguments=self.arguments,
                reason="synthetic retry",
                **action_kwargs,
            )
        )


def _paths(tmp_path):
    before_path = tmp_path / "before.png"
    after_first_path = tmp_path / "after_first.png"
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"
    formal_path.write_bytes(FORMAL_BYTES)
    return before_path, after_first_path, candidate_path, formal_path


def _run_case(
    tmp_path,
    capsys,
    *,
    execute=True,
    snapshots,
    executor=None,
    recovery=None,
    capture_final=True,
    capture_after_first=True,
    sleeper=lambda _seconds: None,
):
    before_path, after_first_path, candidate_path, formal_path = _paths(tmp_path)
    writes = {}
    if capture_after_first:
        writes[after_first_path] = AFTER_FIRST_BYTES
    if capture_final:
        writes[candidate_path] = FINAL_CANDIDATE_BYTES

    observer = ObserverBuilder(snapshots, writes)
    executor = executor or ExecutorBuilder()
    recovery_builder = (lambda: recovery) if recovery is not None else experiment.ActionRecovery

    code = experiment._run_acceptance(
        execute=execute,
        post_action_wait_seconds=0.1,
        before_capture_path=before_path,
        after_first_capture_path=after_first_path,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        observer_builder=observer,
        executor_builder=executor,
        recovery_builder=recovery_builder,
        sleeper=sleeper,
    )
    return SimpleNamespace(
        code=code,
        output=capsys.readouterr().out,
        before_path=before_path,
        after_first_path=after_first_path,
        candidate_path=candidate_path,
        formal_path=formal_path,
        observer=observer,
        executor=executor,
        recovery=recovery,
    )


def _retry_ready_result_with_action(action):
    return RecoveryResult(
        status=RecoveryStatus.RETRY_READY,
        grounding_result=GroundingResult(
            status=GroundingStatus.RESOLVED,
            element=_action_target(position="b")[0],
            candidates=(),
            reason="resolved",
        ),
        action_grounding_result=ActionGroundingResult(
            status=ActionGroundingStatus.READY,
            action=action,
            reason="ready",
        ),
        reason="ready",
    )


def _retry_ready_result_with_same_coordinates():
    first_action = Action(
        tool_name=experiment.CLICK_TOOL_NAME,
        arguments={
            "x": 100,
            "y": 56,
        },
        reason="same coordinates",
    )
    return _retry_ready_result_with_action(first_action)


def _terminal_recovery_result(status):
    return RecoveryResult(
        status=status,
        grounding_result=None,
        action_grounding_result=None,
        reason=status.value,
    )


def test_dry_run_observes_once_and_skips_execution(tmp_path, capsys):
    result = _run_case(
        tmp_path,
        capsys,
        execute=False,
        snapshots=[_observation_1],
        recovery=FailingRecovery(),
        sleeper=lambda _seconds: pytest.fail("post-action wait should not run"),
    )

    assert result.code == 0
    assert result.observer.observation_count == 1
    assert result.observer.paths == [result.before_path]
    assert result.observer.snapshot_paths == [result.before_path]
    assert result.executor.construction_count == 0
    assert result.executor.execution_count == 0
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "Execution skipped: dry-run mode." in result.output
    assert "Observation count: 1" in result.output
    assert "Action execution count: 0" in result.output


def test_dry_run_requires_observation_1_capture_path(tmp_path, capsys):
    wrong_path = tmp_path / "wrong_before.png"
    result = _run_case(
        tmp_path,
        capsys,
        execute=False,
        snapshots=[lambda path: _observation_1(path, frame_path=wrong_path)],
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 1
    assert result.executor.execution_count == 0
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "observation 1 snapshot path was" in result.output
    assert "Execution skipped: observation 1 capture path" in result.output


@pytest.mark.parametrize(
    "observation",
    [
        lambda path: _observation_1(path, marker=False),
        lambda path: _observation_1(path, target_state="missing"),
        lambda path: _observation_1(path, target_state="unsafe"),
        lambda path: _observation_1(path, target_state="text_only"),
        lambda path: _observation_1(path, verify=True),
    ],
)
def test_dry_run_requires_correct_initial_grounding_state(
    tmp_path,
    capsys,
    observation,
):
    result = _run_case(
        tmp_path,
        capsys,
        execute=False,
        snapshots=[observation],
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 1
    assert result.executor.execution_count == 0
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert "Execution skipped:" in result.output


def test_execute_success_path_uses_three_observations_and_two_executions(
    tmp_path,
    capsys,
):
    sleeps = []
    recovery = RecordingRecovery()
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2, _observation_3],
        recovery=recovery,
        sleeper=sleeps.append,
    )

    assert result.code == 0
    assert result.observer.observation_count == 3
    assert result.observer.paths == [
        result.before_path,
        result.after_first_path,
        result.candidate_path,
    ]
    assert result.observer.snapshot_paths == [
        result.before_path,
        result.after_first_path,
        result.candidate_path,
    ]
    assert result.executor.construction_count == 1
    assert result.executor.execution_count == 2
    assert len(result.executor.actions) == 2
    first_action, retry_action = result.executor.actions
    assert first_action.tool_name == experiment.CLICK_TOOL_NAME
    assert set(first_action.arguments) == {"x", "y"}
    assert first_action.action_id != retry_action.action_id
    assert first_action.arguments == {
        "x": 100,
        "y": 56,
    }
    assert retry_action.tool_name == experiment.CLICK_TOOL_NAME
    assert set(retry_action.arguments) == {"x", "y"}
    assert retry_action.arguments == {
        "x": 280,
        "y": 136,
    }
    assert first_action.arguments != retry_action.arguments
    assert len(recovery.calls) == 1
    recovery_call = recovery.calls[0]
    assert recovery_call["verification_result"].status is (
        experiment.ActionVerificationStatus.FAILED
    )
    assert recovery_call["verification_result"].after_grounding.status is (
        GroundingStatus.NOT_FOUND
    )
    assert recovery_call["tool_result"].action_id == first_action.action_id
    assert recovery_call["target_spec"] is experiment.ACTION_TARGET_SPEC
    assert recovery_call["latest_snapshot"].frame.image_path == result.after_first_path
    assert recovery_call["completed_attempts"] == 1
    assert recovery_call["max_attempts"] == experiment.MAX_ATTEMPTS
    assert result.formal_path.read_bytes() == FINAL_CANDIDATE_BYTES
    assert not result.candidate_path.exists()
    assert result.after_first_path.read_bytes() == AFTER_FIRST_BYTES
    assert sleeps == [0.1, 0.1]
    assert "First verification status: failed" in result.output
    assert "First verification after status: not_found" in result.output
    assert "Recovery status: retry_ready" in result.output
    assert "Final verification status: verified" in result.output
    assert "Observation count: 3" in result.output
    assert "Action execution count: 2" in result.output
    assert "Evidence promotion: completed" in result.output


def test_execute_requires_observation_2_capture_path_before_recovery(
    tmp_path,
    capsys,
):
    wrong_path = tmp_path / "wrong_after_first.png"
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            _observation_1,
            lambda path: _observation_2(path, frame_path=wrong_path),
        ],
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.observer.snapshot_paths == [result.before_path, wrong_path]
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "observation 2 snapshot path was" in result.output


def test_execute_requires_observation_3_capture_path_before_promotion(
    tmp_path,
    capsys,
):
    wrong_path = tmp_path / "wrong_candidate.png"
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            _observation_1,
            _observation_2,
            lambda path: _observation_3(path, frame_path=wrong_path),
        ],
    )

    assert result.code == 1
    assert result.observer.observation_count == 3
    assert result.observer.snapshot_paths == [
        result.before_path,
        result.after_first_path,
        wrong_path,
    ]
    assert result.executor.execution_count == 2
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == FINAL_CANDIDATE_BYTES
    assert "observation 3 snapshot path was" in result.output
    assert "Candidate screenshot retained" in result.output


def test_first_verification_must_be_failed_before_recovery_is_accepted(
    tmp_path,
    capsys,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2_verified],
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "First verification status: verified" in result.output


@pytest.mark.parametrize(
    "status",
    [
        RecoveryStatus.BLOCKED,
        RecoveryStatus.EXHAUSTED,
        RecoveryStatus.NOT_NEEDED,
    ],
)
def test_recovery_terminal_statuses_cannot_be_accepted(tmp_path, capsys, status):
    recovery = StaticRecovery(_terminal_recovery_result(status))
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        recovery=recovery,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert len(recovery.calls) == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert f"recovery status was {status.value}" in result.output


def test_fixture_acceptance_requires_retry_coordinates_to_change(
    tmp_path,
    capsys,
):
    recovery = StaticRecovery(_retry_ready_result_with_same_coordinates())
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        recovery=recovery,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "retry Action coordinates did not change" in result.output


def test_retry_action_id_must_change_even_when_coordinates_change(
    tmp_path,
    capsys,
):
    recovery = RetryActionFromToolResultRecovery(reuse_action_id=True)
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        recovery=recovery,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "retry Action reused the first Action id" in result.output


def test_retry_action_must_be_click_mouse(tmp_path, capsys):
    recovery = RetryActionFromToolResultRecovery(tool_name="type_text")
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        recovery=recovery,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "retry Action tool was type_text" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        {"x": 280, "y": 136, "button": "left"},
        {"x": "280", "y": 136},
    ],
)
def test_retry_action_arguments_must_be_exact_integer_coordinates(
    tmp_path,
    capsys,
    arguments,
):
    recovery = RetryActionFromToolResultRecovery(arguments=arguments)
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        recovery=recovery,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "retry Action arguments were not" in result.output


def test_recovery_uses_observation_2_screen_size_for_retry_safety(
    tmp_path,
    capsys,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            _observation_1,
            lambda path: _observation_2(path, screen_size=(260, 150)),
        ],
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "Recovery status: blocked" in result.output
    assert "Recovery action grounding status: blocked" in result.output


def test_final_verification_must_be_verified_and_candidate_is_retained(
    tmp_path,
    capsys,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            _observation_1,
            _observation_2,
            lambda path: _observation_3(path, verified=False),
        ],
    )

    assert result.code == 1
    assert result.observer.observation_count == 3
    assert result.executor.execution_count == 2
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == FINAL_CANDIDATE_BYTES
    assert "Final verification status: failed" in result.output
    assert "Candidate screenshot retained" in result.output


def test_tool_failure_blocks_evidence_promotion(tmp_path, capsys):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2],
        executor=ExecutorBuilder(results=[False]),
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "first ToolResult failed" in result.output


@pytest.mark.parametrize(
    "observation_2",
    [
        lambda path: _observation_2(path, marker=False),
        lambda path: _observation_2(path, seconds=0),
    ],
)
def test_observation_2_identity_or_chronology_failure_closes(
    tmp_path,
    capsys,
    observation_2,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, observation_2],
        recovery=FailingRecovery(),
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()


@pytest.mark.parametrize(
    "observation_3",
    [
        lambda path: _observation_3(path, marker=False),
        lambda path: _observation_3(path, seconds=1),
    ],
)
def test_observation_3_identity_or_chronology_failure_closes(
    tmp_path,
    capsys,
    observation_3,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2, observation_3],
    )

    assert result.code == 1
    assert result.observer.observation_count == 3
    assert result.executor.execution_count == 2
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == FINAL_CANDIDATE_BYTES


def test_no_extra_execution_beyond_two_actions(tmp_path, capsys):
    executor = ExecutorBuilder(results=[True, True, RuntimeError("extra action")])
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2, _observation_3],
        executor=executor,
    )

    assert result.code == 0
    assert result.executor.execution_count == 2
    assert len(result.executor.results) == 1


def test_final_candidate_must_exist_before_promotion(tmp_path, capsys):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2, _observation_3],
        capture_final=False,
    )

    assert result.code == 1
    assert result.observer.observation_count == 3
    assert result.executor.execution_count == 2
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "final candidate screenshot was not created" in result.output


def test_promotion_failure_retains_candidate_evidence(
    tmp_path,
    capsys,
    monkeypatch,
):
    def fail_promotion(**_kwargs):
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(experiment, "promote_candidate_evidence", fail_promotion)
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[_observation_1, _observation_2, _observation_3],
    )

    assert result.code == 1
    assert result.observer.observation_count == 3
    assert result.executor.execution_count == 2
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == FINAL_CANDIDATE_BYTES
    assert "Evidence promotion failed: RuntimeError: promotion failed" in result.output
    assert "Candidate screenshot retained" in result.output


def test_fixture_identity_accepts_raw_marker_from_accessibility_or_ocr(tmp_path):
    accessibility_marker = _marker("accessibility")
    ocr_marker = _marker("ocr")

    accessibility_snapshot = _snapshot(
        tmp_path / "accessibility.png",
        accessibility=(accessibility_marker,),
    )
    ocr_snapshot = _snapshot(
        tmp_path / "ocr.png",
        ocr=(ocr_marker,),
    )

    assert live_harness_utils.fixture_identity_observed(
        accessibility_snapshot,
        experiment.FIXTURE_MARKER,
    )
    assert live_harness_utils.fixture_identity_observed(
        ocr_snapshot,
        experiment.FIXTURE_MARKER,
    )


def test_fused_only_marker_does_not_establish_fixture_identity(tmp_path):
    snapshot = _snapshot(
        tmp_path / "fused.png",
        fused=(
            _element(
                experiment.FIXTURE_MARKER,
                element_type="text",
                source="hybrid",
            ),
        ),
    )

    assert not live_harness_utils.fixture_identity_observed(
        snapshot,
        experiment.FIXTURE_MARKER,
    )


def test_direct_script_help_supports_local_harness_import():
    experiment_path = Path(experiment.__file__).resolve()
    result = subprocess.run(
        [sys.executable, str(experiment_path), "--help"],
        cwd=experiment.PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    combined_output = result.stdout + result.stderr

    assert result.returncode == 0
    assert (
        "Observe the Experiment 05 fixture and prove one recovery retry."
        in result.stdout
    )
    assert "ModuleNotFoundError" not in combined_output


def test_import_safety_keeps_live_dependencies_lazy():
    source = inspect.getsource(experiment)
    top_level_source = source.split("def _build_engine", maxsplit=1)[0]
    helper_source = inspect.getsource(live_harness_utils)
    helper_top_level_source = helper_source.split(
        "def build_live_perception_engine",
        maxsplit=1,
    )[0]

    forbidden_top_level_terms = (
        "computer_agent.control",
        "computer_agent.tools",
        "pyautogui",
        "openai",
        "llm",
        "planning",
        "phase03",
    )

    assert all(term not in top_level_source for term in forbidden_top_level_terms)
    assert all(
        term not in helper_top_level_source
        for term in forbidden_top_level_terms
    )
    assert "while " not in source
