from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from computer_agent.core.models import ToolResult
from computer_agent.perception import BoundingBox, PerceptionSnapshot, ScreenFrame, UIElement
from experiments.phase04_ui_grounding_task_reasoning import experiment_04_action_verification as experiment


FORMAL_BYTES = b"existing formal evidence"
CANDIDATE_BYTES = b"after candidate evidence"


def _time(seconds=0):
    return datetime(2026, 8, 31, 12, 0, seconds, tzinfo=timezone.utc)


def _element(
    text,
    *,
    identifier=None,
    element_type="button",
    enabled=True,
    source="accessibility",
    x=10,
    y=10,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        bounding_box=BoundingBox(x=x, y=y, width=120, height=32),
        confidence=0.95,
        text=text,
        identifier=identifier,
        enabled=enabled,
        source=source,
    )


def _snapshot(path, *, accessibility=(), ocr=(), fused=(), seconds=0):
    return PerceptionSnapshot(
        frame=ScreenFrame(Path(path), 300, 180, 300, 180, _time(seconds)),
        image=Image.new("RGB", (300, 180)),
        accessibility_elements=tuple(accessibility),
        ocr_elements=tuple(ocr),
        fused_elements=tuple(fused),
        warnings=(),
    )


def _marker(source="accessibility"):
    return _element(f"prefix {experiment.FIXTURE_MARKER} suffix", element_type="text", source=source)


def _action_target(state="resolved"):
    if state == "missing":
        return ()
    return (
        _element(
            experiment.ACTION_TARGET_TEXT,
            identifier=None if state == "text_only" else experiment.ACTION_TARGET_IDENTIFIER,
            enabled=state != "unsafe",
        ),
    )


def _verification_targets(state="resolved"):
    if state == "missing":
        return ()
    first = _element(experiment.VERIFICATION_TARGET_TEXT, element_type="text", x=20, y=80)
    if state == "ambiguous":
        return first, _element(
            experiment.VERIFICATION_TARGET_TEXT,
            element_type="text",
            x=160,
            y=80,
        )
    return (first,)


def _before(path, *, marker=True, target_state="resolved"):
    return _snapshot(
        path,
        accessibility=(_marker(),) if marker else (),
        fused=_action_target(target_state),
        seconds=0,
    )


def _after(path, *, marker=True, verification_state="resolved"):
    return _snapshot(
        path,
        accessibility=(_marker(),) if marker else (),
        fused=(*_action_target(), *_verification_targets(verification_state)),
        seconds=1,
    )


class ObserverBuilder:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.paths = []
        self.observation_count = 0

    def __call__(self, path):
        self.paths.append(Path(path))
        return self

    def observe(self):
        self.observation_count += 1
        item = self.snapshots.pop(0)
        if isinstance(item, (OSError, RuntimeError)):
            raise item
        return item


class ExecutorBuilder:
    def __init__(self, *, success=True, error=None, execute_error=None):
        self.success = success
        self.error = error
        self.execute_error = execute_error
        self.construction_count = 0
        self.execution_count = 0
        self.actions = []

    def __call__(self):
        self.construction_count += 1
        return self

    def execute(self, action):
        self.execution_count += 1
        self.actions.append(action)
        if self.execute_error is not None:
            raise self.execute_error
        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=self.success,
            error=None if self.success else self.error or "synthetic tool failure",
        )


def _paths(tmp_path):
    before_path, candidate_path, formal_path = (
        tmp_path / "before.png", tmp_path / "candidate.png", tmp_path / "formal.png"
    )
    formal_path.write_bytes(FORMAL_BYTES)
    return before_path, candidate_path, formal_path


def _run_case(
    tmp_path,
    capsys,
    *,
    execute=True,
    snapshots,
    executor=None,
    candidate=False,
    sleeper=lambda _seconds: None,
):
    before_path, candidate_path, formal_path = _paths(tmp_path)
    if candidate:
        candidate_path.write_bytes(CANDIDATE_BYTES)
    observer = ObserverBuilder(
        [item(before_path, candidate_path) if callable(item) else item for item in snapshots]
    )
    executor = executor or ExecutorBuilder()
    code = experiment._run_acceptance(
        execute=execute,
        post_action_wait_seconds=0.1,
        before_capture_path=before_path,
        candidate_evidence_path=candidate_path,
        formal_evidence_path=formal_path,
        observer_builder=observer,
        executor_builder=executor,
        sleeper=sleeper,
    )
    return SimpleNamespace(
        code=code,
        output=capsys.readouterr().out,
        before_path=before_path,
        candidate_path=candidate_path,
        formal_path=formal_path,
        observer=observer,
        executor=executor,
    )


@pytest.mark.parametrize("source", ["accessibility", "ocr"])
def test_fixture_identity_accepts_raw_marker(tmp_path, source):
    marker = _marker(source)
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        accessibility=(marker,) if source == "accessibility" else (),
        ocr=(marker,) if source == "ocr" else (),
    )

    assert experiment._fixture_identity_observed(snapshot)


def test_fused_only_marker_evidence_is_rejected(tmp_path):
    snapshot = _snapshot(
        tmp_path / "candidate.png",
        fused=(_element(experiment.FIXTURE_MARKER, element_type="text", source="hybrid"),),
    )

    assert not experiment._fixture_identity_observed(snapshot)


def test_dry_run_observes_once_and_skips_execution(tmp_path, capsys):
    result = _run_case(
        tmp_path,
        capsys,
        execute=False,
        snapshots=[lambda before, _candidate: _before(before)],
        sleeper=lambda _seconds: pytest.fail("post-action wait should not run"),
    )

    assert result.code == 0
    assert result.observer.observation_count == 1
    assert result.observer.paths == [result.before_path]
    assert result.executor.construction_count == 0
    assert result.executor.execution_count == 0
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "Execution skipped: dry-run mode." in result.output
    assert "Execution performed:" not in result.output
    assert "Evidence promotion: skipped" in result.output


@pytest.mark.parametrize(
    "before_item",
    [
        lambda before, _candidate: _before(before, marker=False),
        lambda before, _candidate: _before(before, target_state="missing"),
        lambda before, _candidate: _before(before, target_state="unsafe"),
        lambda before, _candidate: _before(before, target_state="text_only"),
    ],
)
def test_before_gates_block_execution(tmp_path, capsys, before_item):
    result = _run_case(tmp_path, capsys, snapshots=[before_item])

    assert result.code == 1
    assert result.observer.observation_count == 1
    assert result.executor.construction_count == 0
    assert result.executor.execution_count == 0
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert not result.candidate_path.exists()
    assert "Execution skipped:" in result.output


def test_verified_path_observes_executes_and_promotes(tmp_path, capsys):
    sleeps = []
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: _before(before),
            lambda _before, candidate: _after(candidate),
        ],
        candidate=True,
        sleeper=sleeps.append,
    )

    assert result.code == 0
    assert result.observer.observation_count == 2
    assert result.observer.paths == [result.before_path, result.candidate_path]
    assert result.executor.construction_count == 1
    assert result.executor.execution_count == 1
    assert result.executor.actions[0].tool_name == experiment.CLICK_TOOL_NAME
    assert result.formal_path.read_bytes() == CANDIDATE_BYTES
    assert not result.candidate_path.exists()
    assert sleeps == [0.1]
    assert "Execution performed: one Action executed." in result.output
    assert "Execution skipped:" not in result.output
    assert "Verification status: verified" in result.output
    assert "Evidence promotion: completed" in result.output


@pytest.mark.parametrize(
    ("after_item", "executor", "expected_text"),
    [
        (
            lambda _before, candidate: _after(candidate),
            ExecutorBuilder(success=False, error="click failed"),
            "Tool result success: False",
        ),
        (
            lambda _before, candidate: _after(candidate, marker=False),
            ExecutorBuilder(),
            "after fixture marker was not observed",
        ),
        (
            lambda _before, candidate: _after(candidate, verification_state="missing"),
            ExecutorBuilder(),
            "Verification status: failed",
        ),
        (
            lambda _before, candidate: _after(candidate, verification_state="ambiguous"),
            ExecutorBuilder(),
            "Verification status: inconclusive",
        ),
    ],
)
def test_execute_failures_preserve_formal_evidence(
    tmp_path,
    capsys,
    after_item,
    executor,
    expected_text,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[lambda before, _candidate: _before(before), after_item],
        executor=executor,
        candidate=True,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == CANDIDATE_BYTES
    assert "Execution performed: one Action executed." in result.output
    assert expected_text in result.output


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
@pytest.mark.parametrize(
    ("point", "expected_observations", "expected_executions", "candidate"),
    [
        ("before", 1, 0, False),
        ("execution", 1, 1, False),
        ("after", 2, 1, True),
    ],
)
def test_runtime_failures_return_nonzero_with_single_path(
    tmp_path,
    capsys,
    point,
    expected_observations,
    expected_executions,
    candidate,
    error_type,
):
    error = error_type(f"{point} failed")
    snapshots = [lambda before, _candidate: _before(before)]
    executor = ExecutorBuilder(execute_error=error if point == "execution" else None)
    if point == "before":
        snapshots = [error]
    elif point == "after":
        snapshots.append(error)

    result = _run_case(
        tmp_path,
        capsys,
        snapshots=snapshots,
        executor=executor,
        candidate=candidate,
        sleeper=lambda _seconds: None,
    )

    assert result.code == 1
    assert result.observer.observation_count == expected_observations
    assert result.executor.execution_count == expected_executions
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.exists() is candidate


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_promotion_failures_return_nonzero_with_candidate_retained(
    tmp_path,
    capsys,
    monkeypatch,
    error_type,
):
    def fail_promotion(**_kwargs):
        raise error_type("promotion failed")

    monkeypatch.setattr(experiment, "_promote_candidate_evidence", fail_promotion)
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: _before(before),
            lambda _before, candidate: _after(candidate),
        ],
        candidate=True,
    )

    assert result.code == 1
    assert result.observer.observation_count == 2
    assert result.executor.execution_count == 1
    assert result.formal_path.read_bytes() == FORMAL_BYTES
    assert result.candidate_path.read_bytes() == CANDIDATE_BYTES
