from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from computer_agent.core.models import ToolResult
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    SemanticAXElement,
    UIElement,
    Viewport,
)
from experiments.phase05_real_web_autonomy import (
    experiment_05_real_web_text_input as experiment,
)


class FakeObserver:
    def __init__(
        self,
        observations,
    ):
        self.observations = list(observations)
        self.calls = 0

    def observe(self):
        self.calls += 1

        if len(self.observations) > 1:
            return self.observations.pop(0)

        return self.observations[0]


class FakeObserverBuilder:
    def __init__(
        self,
        observations,
    ):
        self.observer = FakeObserver(observations)
        self.capture_paths = []

    def __call__(
        self,
        *,
        capture_path,
    ):
        self.capture_paths.append(Path(capture_path))

        return self.observer


class FakeExecutor:
    def __init__(
        self,
        *,
        fail_tool=None,
    ):
        self.fail_tool = fail_tool
        self.actions = []

    def execute(self, action):
        self.actions.append(action)

        success = action.tool_name != self.fail_tool

        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            output=dict(action.arguments) if success else None,
            error=None if success else f"{action.tool_name} failed",
        )


class FakeExecutorBuilder:
    def __init__(
        self,
        executor,
    ):
        self.executor = executor
        self.calls = 0

    def __call__(self):
        self.calls += 1

        return self.executor


def _time(seconds=0):
    return datetime(
        2026,
        9,
        7,
        12,
        0,
        seconds,
        tzinfo=timezone.utc,
    )


def _box(
    x=100,
    y=500,
    width=180,
    height=30,
):
    return BoundingBox(
        x=x,
        y=y,
        width=width,
        height=height,
    )


def _viewport():
    return Viewport(
        BoundingBox(
            x=0,
            y=100,
            width=1000,
            height=600,
        )
    )


def _field(
    *,
    text="Search This Site",
    value=None,
    element_type="text_field",
    box=None,
    x=100,
):
    return UIElement(
        element_type=element_type,
        bounding_box=box
        if box is not None
        else _box(x=x),
        confidence=1.0,
        text=text,
        value=value,
        enabled=True,
        source="accessibility",
    )


def _semantic_field(
    *,
    text="Search This Site",
    value=None,
    bounds=None,
):
    return SemanticAXElement(
        role="AXTextField",
        text=text,
        value=value,
        bounds=bounds if bounds is not None else _box(),
    )


def _snapshot(
    *,
    elements=(),
    seconds=0,
    warnings=(),
):
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"capture-{seconds}.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=_time(seconds),
        ),
        image=Image.new(
            "RGB",
            (1000, 700),
        ),
        accessibility_elements=tuple(elements),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=tuple(warnings),
    )


def _observation(
    *,
    app="Google Chrome",
    viewport=None,
    elements=(),
    semantics=(),
    seconds=0,
    warnings=(),
):
    return experiment.TextInputObservation(
        application_name=app,
        viewport=_viewport() if viewport is None else viewport,
        snapshot=_snapshot(
            elements=elements,
            seconds=seconds,
            warnings=warnings,
        ),
        semantic_elements=tuple(semantics),
    )


def _run(
    tmp_path,
    *,
    observations,
    execute=False,
    executor=None,
    input_text="accessibility test",
):
    observer_builder = FakeObserverBuilder(observations)
    executor = executor or FakeExecutor()
    executor_builder = FakeExecutorBuilder(executor)
    sleeps = []

    result = experiment._run_acceptance(
        execute=execute,
        input_text=input_text,
        post_action_wait_seconds=0.25,
        observer_builder=observer_builder,
        executor_builder=executor_builder,
        capture_path=tmp_path / "candidate.png",
        sleeper=sleeps.append,
    )

    return (
        result,
        observer_builder,
        executor_builder,
        executor,
        sleeps,
    )


def _assert_no_actions(executor):
    assert executor.actions == []


def test_dry_run_unique_visible_text_field_needs_action_with_zero_actions(
    tmp_path,
):
    (
        result,
        observer_builder,
        executor_builder,
        executor,
        sleeps,
    ) = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(
                    _field(),
                ),
                semantics=(
                    _semantic_field(),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.NEEDS_ACTION
    assert observer_builder.observer.calls == 1
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_dry_run_missing_target_blocks_with_zero_actions(tmp_path):
    result, _builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(),
                semantics=(),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_dry_run_duplicate_target_blocks_with_zero_actions(tmp_path):
    result, _builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(
                    _field(x=100),
                    _field(x=250),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_dry_run_wrong_role_blocks_with_zero_actions(tmp_path):
    result, _builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(
                    _field(element_type="button"),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_dry_run_geometry_unavailable_blocks_with_zero_actions(tmp_path):
    result, _builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(),
                semantics=(
                    _semantic_field(bounds=None),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_dry_run_non_chrome_blocks_with_zero_actions(tmp_path):
    result, _builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                app="Safari",
                elements=(
                    _field(),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_execute_focuses_types_and_verifies_changed_value(tmp_path):
    executor = FakeExecutor()

    result, _builder, executor_builder, _executor, sleeps = _run(
        tmp_path,
        execute=True,
        executor=executor,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=0,
            ),
            _observation(
                elements=(
                    _field(value="accessibility test"),
                ),
                seconds=1,
            ),
        ],
    )

    assert result.status is experiment.TextInputStatus.VERIFIED
    assert executor_builder.calls == 1
    assert [
        action.tool_name
        for action in executor.actions
    ] == [
        "click_mouse",
        "type_text",
    ]
    assert executor.actions[1].arguments == {
        "text": "accessibility test",
    }
    assert sleeps == [0.25]


def test_execute_click_failure_returns_action_failed(tmp_path):
    executor = FakeExecutor(
        fail_tool="click_mouse",
    )

    result, _builder, _executor_builder, _executor, sleeps = _run(
        tmp_path,
        execute=True,
        executor=executor,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.ACTION_FAILED
    assert [
        action.tool_name
        for action in executor.actions
    ] == ["click_mouse"]
    assert sleeps == []


def test_execute_type_failure_returns_action_failed(tmp_path):
    executor = FakeExecutor(
        fail_tool="type_text",
    )

    result, _builder, _executor_builder, _executor, sleeps = _run(
        tmp_path,
        execute=True,
        executor=executor,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
            )
        ],
    )

    assert result.status is experiment.TextInputStatus.ACTION_FAILED
    assert [
        action.tool_name
        for action in executor.actions
    ] == [
        "click_mouse",
        "type_text",
    ]
    assert sleeps == []


def test_execute_post_action_field_missing_fails_verification(tmp_path):
    result, _builder, _executor_builder, executor, sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=0,
            ),
            _observation(
                elements=(),
                seconds=1,
            ),
        ],
    )

    assert result.status is experiment.TextInputStatus.VERIFICATION_FAILED
    assert len(executor.actions) == 2
    assert sleeps == [0.25]


def test_execute_post_action_value_unchanged_fails_verification(tmp_path):
    result, _builder, _executor_builder, executor, sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=0,
            ),
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=1,
            ),
        ],
    )

    assert result.status is experiment.TextInputStatus.VERIFICATION_FAILED
    assert len(executor.actions) == 2
    assert sleeps == [0.25]


def test_execute_post_action_ambiguous_field_blocks(tmp_path):
    result, _builder, _executor_builder, executor, sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=0,
            ),
            _observation(
                elements=(
                    _field(value="accessibility test", x=100),
                    _field(value="accessibility test", x=250),
                ),
                seconds=1,
            ),
        ],
    )

    assert result.status is experiment.TextInputStatus.BLOCKED
    assert len(executor.actions) == 2
    assert sleeps == [0.25]


def test_execute_issues_no_enter_submit_navigation_or_clipboard_actions(
    tmp_path,
):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(
                    _field(value=""),
                ),
                seconds=0,
            ),
            _observation(
                elements=(
                    _field(value="accessibility test"),
                ),
                seconds=1,
            ),
        ],
    )

    forbidden_tools = {
        "press_key",
        "hotkey",
        "open_url",
        "copy_to_clipboard",
        "paste_text",
        "read_from_clipboard",
    }

    assert result.status is experiment.TextInputStatus.VERIFIED
    assert not {
        action.tool_name
        for action in executor.actions
    } & forbidden_tools
    assert len(executor.actions) == 2
