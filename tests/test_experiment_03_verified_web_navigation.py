from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from computer_agent.core.models import ToolResult
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from experiments.phase05_real_web_autonomy import (
    experiment_03_verified_web_navigation as experiment,
)


FORMAL_BYTES = b"existing formal evidence"
CANDIDATE_BYTES = b"verified navigation candidate"


def _time(seconds=0):
    return datetime(
        2026,
        9,
        6,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _element(
    text,
    *,
    element_type,
    x=20,
    y=20,
    enabled=True,
):
    return UIElement(
        element_type=element_type,
        bounding_box=BoundingBox(
            x=x,
            y=y,
            width=120,
            height=32,
        ),
        confidence=0.95,
        text=text,
        enabled=enabled,
        source="accessibility",
    )


def _snapshot(
    path,
    *,
    fused=(),
    seconds=0,
    warnings=(),
):
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(path),
            pixel_width=300,
            pixel_height=180,
            screen_width=300,
            screen_height=180,
            captured_at=_time(seconds),
        ),
        image=Image.new(
            "RGB",
            (300, 180),
        ),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(fused),
        warnings=tuple(warnings),
    )


def _before(
    path,
    *,
    start_marker=True,
    action_target=True,
    verification_target=False,
):
    elements = []

    if start_marker:
        elements.append(
            _element(
                "Search This Site",
                element_type="text_field",
                x=20,
            )
        )

    if action_target:
        elements.append(
            _element(
                "Docs",
                element_type="link",
                x=150,
            )
        )

    if verification_target:
        elements.append(
            _element(
                "Library reference",
                element_type="link",
                x=40,
                y=100,
            )
        )

    return _snapshot(
        path,
        fused=elements,
        seconds=0,
    )


def _after(
    path,
    *,
    verification_target=True,
):
    elements = []

    if verification_target:
        elements.append(
            _element(
                "Library reference",
                element_type="link",
                x=60,
                y=80,
            )
        )

    return _snapshot(
        path,
        fused=elements,
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
    def __init__(
        self,
        *,
        success=True,
        error=None,
    ):
        self.success = success
        self.error = error
        self.construction_count = 0
        self.execution_count = 0
        self.actions = []

    def __call__(self):
        self.construction_count += 1
        return self

    def execute(self, action):
        self.execution_count += 1
        self.actions.append(action)

        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=self.success,
            error=(
                None
                if self.success
                else self.error or "synthetic click failure"
            ),
        )


def _paths(tmp_path):
    before_path = tmp_path / "before.png"
    candidate_path = tmp_path / "candidate.png"
    formal_path = tmp_path / "formal.png"

    formal_path.write_bytes(FORMAL_BYTES)

    return (
        before_path,
        candidate_path,
        formal_path,
    )


def _run_case(
    tmp_path,
    capsys,
    *,
    execute=True,
    snapshots,
    executor=None,
    candidate=False,
    frontmost="Google Chrome",
    sleeper=lambda _seconds: None,
):
    (
        before_path,
        candidate_path,
        formal_path,
    ) = _paths(tmp_path)

    if candidate:
        candidate_path.write_bytes(
            CANDIDATE_BYTES
        )

    materialized = []

    for item in snapshots:
        if callable(item):
            materialized.append(
                item(
                    before_path,
                    candidate_path,
                )
            )
        else:
            materialized.append(item)

    observer = ObserverBuilder(
        materialized
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
        frontmost_reader=lambda: frontmost,
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


def test_dry_run_observes_once_and_never_executes(
    tmp_path,
    capsys,
):
    result = _run_case(
        tmp_path,
        capsys,
        execute=False,
        snapshots=[
            lambda before, _candidate: _before(
                before
            )
        ],
        sleeper=lambda _seconds: pytest.fail(
            "post-action wait must not run"
        ),
    )

    assert result.code == 0

    assert (
        result.observer.observation_count
        == 1
    )
    assert result.observer.paths == [
        result.before_path
    ]

    assert (
        result.executor.construction_count
        == 0
    )
    assert (
        result.executor.execution_count
        == 0
    )

    assert (
        result.formal_path.read_bytes()
        == FORMAL_BYTES
    )
    assert not result.candidate_path.exists()

    assert (
        "Precondition acceptance result: passed"
        in result.output
    )
    assert (
        "Execution skipped: dry-run mode."
        in result.output
    )
    assert (
        "Action execution count: 0"
        in result.output
    )
    assert (
        "Evidence promotion: skipped"
        in result.output
    )


@pytest.mark.parametrize(
    "before_factory",
    [
        lambda path: _before(
            path,
            start_marker=False,
        ),
        lambda path: _before(
            path,
            action_target=False,
        ),
        lambda path: _before(
            path,
            verification_target=True,
        ),
    ],
)
def test_failed_preconditions_never_execute_click(
    tmp_path,
    capsys,
    before_factory,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: (
                before_factory(before)
            )
        ],
    )

    assert result.code == 1

    assert (
        result.observer.observation_count
        == 1
    )
    assert (
        result.executor.construction_count
        == 0
    )
    assert (
        result.executor.execution_count
        == 0
    )

    assert (
        result.formal_path.read_bytes()
        == FORMAL_BYTES
    )
    assert not result.candidate_path.exists()

    assert (
        "Precondition acceptance failed:"
        in result.output
    )
    assert (
        "Execution skipped."
        in result.output
    )


def test_verified_navigation_executes_once_and_promotes_evidence(
    tmp_path,
    capsys,
):
    sleeps = []

    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: _before(
                before
            ),
            lambda _before, candidate: _after(
                candidate
            ),
        ],
        candidate=True,
        sleeper=sleeps.append,
    )

    assert result.code == 0

    assert (
        result.observer.observation_count
        == 2
    )
    assert result.observer.paths == [
        result.before_path,
        result.candidate_path,
    ]

    assert (
        result.executor.construction_count
        == 1
    )
    assert (
        result.executor.execution_count
        == 1
    )

    assert len(result.executor.actions) == 1

    action = result.executor.actions[0]

    assert action.tool_name == "click_mouse"
    assert set(action.arguments) == {
        "x",
        "y",
    }

    assert sleeps == [0.1]

    assert (
        result.formal_path.read_bytes()
        == CANDIDATE_BYTES
    )
    assert not result.candidate_path.exists()

    assert (
        "Verification status: verified"
        in result.output
    )
    assert (
        "Action execution count: 1"
        in result.output
    )
    assert (
        "Evidence promotion: completed"
        in result.output
    )
    assert (
        "Live acceptance result: passed"
        in result.output
    )


def test_failed_navigation_verification_preserves_formal_evidence(
    tmp_path,
    capsys,
):
    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: _before(
                before
            ),
            lambda _before, candidate: _after(
                candidate,
                verification_target=False,
            ),
        ],
        candidate=True,
    )

    assert result.code == 1

    assert (
        result.executor.execution_count
        == 1
    )

    assert (
        result.formal_path.read_bytes()
        == FORMAL_BYTES
    )
    assert (
        result.candidate_path.read_bytes()
        == CANDIDATE_BYTES
    )

    assert (
        "Verification status: failed"
        in result.output
    )
    assert (
        "Live acceptance failed:"
        in result.output
    )


def test_failed_click_cannot_produce_verified_navigation(
    tmp_path,
    capsys,
):
    executor = ExecutorBuilder(
        success=False,
        error="click failed",
    )

    result = _run_case(
        tmp_path,
        capsys,
        snapshots=[
            lambda before, _candidate: _before(
                before
            ),
            lambda _before, candidate: _after(
                candidate
            ),
        ],
        executor=executor,
        candidate=True,
    )

    assert result.code == 1

    assert executor.execution_count == 1

    assert (
        result.formal_path.read_bytes()
        == FORMAL_BYTES
    )
    assert (
        result.candidate_path.read_bytes()
        == CANDIDATE_BYTES
    )

    assert (
        "Tool result success: False"
        in result.output
    )
    assert (
        "Verification status: failed"
        in result.output
    )
    assert (
        "Live acceptance result: passed"
        not in result.output
    )
