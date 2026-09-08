from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from computer_agent.agent.web_recovery import (
    WebRecoveryDecision,
    WebRecoveryDecisionResult,
)
from computer_agent.core.models import ToolResult
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    SemanticAXElement,
    UIElement,
    Viewport,
    ViewportSearchPolicy,
)
from experiments.phase05_real_web_autonomy import (
    experiment_07_adaptive_web_recovery as experiment,
)


DEFAULT_VIEWPORT = object()


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
        tool_name=None,
    ):
        self.tool_name = tool_name
        self.actions = []

    def execute(self, action):
        self.actions.append(action)
        return ToolResult(
            action_id=action.action_id,
            tool_name=self.tool_name or action.tool_name,
            success=True,
            output=dict(action.arguments),
            error=None,
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
    width=140,
    height=24,
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


def _link_element(
    *,
    text="Privacy Notice",
    box=None,
    x=100,
):
    return UIElement(
        element_type="link",
        bounding_box=box
        if box is not None
        else _box(x=x),
        confidence=1.0,
        text=text,
        enabled=True,
        source="accessibility",
    )


def _semantic_link(
    *,
    text="Privacy Notice",
    bounds=None,
):
    return SemanticAXElement(
        role="AXLink",
        text=text,
        bounds=bounds if bounds is not None else _box(),
    )


def _semantic_marker(
    *,
    text="Marker",
    y=500,
):
    return SemanticAXElement(
        role="AXStaticText",
        text=text,
        bounds=_box(y=y),
    )


def _snapshot(
    *,
    elements=(),
    warnings=(),
    seconds=0,
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
    viewport=DEFAULT_VIEWPORT,
    elements=(),
    semantics=(),
    warnings=(),
    seconds=0,
):
    return experiment.AdaptiveWebRecoveryObservation(
        application_name=app,
        viewport=_viewport()
        if viewport is DEFAULT_VIEWPORT
        else viewport,
        snapshot=_snapshot(
            elements=elements,
            warnings=warnings,
            seconds=seconds,
        ),
        semantic_elements=tuple(semantics),
    )


def _policy(
    *,
    max_scroll_attempts=6,
):
    return ViewportSearchPolicy(
        max_scroll_attempts=max_scroll_attempts,
        scroll_amount=12,
        stabilization_wait_seconds=0.0,
    )


def _run(
    tmp_path,
    *,
    observations,
    execute=False,
    executor=None,
    policy=None,
    recovery_decider=experiment.decide_failed_grounding_recovery,
):
    observer_builder = FakeObserverBuilder(observations)
    executor = executor or FakeExecutor()
    executor_builder = FakeExecutorBuilder(executor)
    sleeps = []

    result = experiment._run_acceptance(
        execute=execute,
        observer_builder=observer_builder,
        executor_builder=executor_builder,
        capture_path=tmp_path / "candidate.png",
        policy=policy or _policy(),
        sleeper=sleeps.append,
        recovery_decider=recovery_decider,
    )

    return (
        result,
        observer_builder,
        executor_builder,
        executor,
        sleeps,
    )


def _action_names(executor):
    return [
        action.tool_name
        for action in executor.actions
    ]


def _assert_no_actions(executor):
    assert executor.actions == []


def _blocking_decider(_grounding):
    return WebRecoveryDecisionResult(
        decision=WebRecoveryDecision.BLOCK,
        reason="synthetic block",
    )


def test_dry_run_not_found_recovery_reports_needs_scroll_with_zero_actions(
    tmp_path,
):
    result, builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        observations=[
            _observation(
                elements=(),
                semantics=(),
            )
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.NEEDS_SCROLL
    assert result.recovery_decision.decision is (
        WebRecoveryDecision.VIEWPORT_SEARCH
    )
    assert result.search_result.status.value == "needs_scroll"
    assert result.action_execution_count == 0
    assert builder.observer.calls == 1
    assert executor_builder.calls == 0
    _assert_no_actions(executor)
    assert sleeps == []


def test_execute_not_found_searches_then_final_resolved_passes(tmp_path):
    result, builder, executor_builder, executor, sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(),
                seconds=0,
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_link(bounds=_box()),
                ),
                seconds=1,
            ),
            _observation(
                elements=(
                    _link_element(),
                ),
                semantics=(
                    _semantic_link(),
                ),
                seconds=2,
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.PASSED
    assert result.search_result.status.value == "found"
    assert result.final_grounding.status.value == "resolved"
    assert result.final_grounding.element.text == "Privacy Notice"
    assert result.action_execution_count == 1
    assert _action_names(executor) == ["scroll"]
    assert executor_builder.calls == 1
    assert builder.observer.calls == 3
    assert sleeps == [0.0]


def test_recovery_decision_block_blocks_with_zero_actions(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(),
            )
        ],
        recovery_decider=_blocking_decider,
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.recovery_decision.decision is WebRecoveryDecision.BLOCK
    assert result.search_result is None
    _assert_no_actions(executor)


def test_non_chrome_blocks_before_recovery(tmp_path):
    result, _builder, executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                app="Safari",
                elements=(),
                semantics=(),
            )
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.recovery_decision is None
    assert executor_builder.calls == 1
    _assert_no_actions(executor)


def test_missing_viewport_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                viewport=None,
                elements=(),
                semantics=(),
            )
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.recovery_decision is None
    _assert_no_actions(executor)


def test_perception_warning_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(),
                warnings=("synthetic warning",),
            )
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.recovery_decision is None
    _assert_no_actions(executor)


def test_unexpected_initial_resolved_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(
                    _link_element(),
                ),
                semantics=(
                    _semantic_link(),
                ),
            )
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.initial_grounding.status.value == "resolved"
    assert result.search_result is None
    _assert_no_actions(executor)


def test_search_blocked_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(
                    _semantic_marker(y=500),
                ),
            ),
            _observation(
                app="Safari",
                elements=(),
                semantics=(
                    _semantic_marker(y=450),
                ),
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.search_result.status.value == "blocked"
    assert _action_names(executor) == ["scroll"]


def test_search_stalled_blocks(tmp_path):
    repeated = _observation(
        elements=(),
        semantics=(
            _semantic_marker(y=500),
        ),
    )

    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            repeated,
            repeated,
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.search_result.status.value == "stalled"
    assert _action_names(executor) == ["scroll"]


def test_search_exhausted_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        policy=_policy(max_scroll_attempts=1),
        observations=[
            _observation(
                elements=(),
                semantics=(
                    _semantic_marker(y=500),
                ),
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_marker(y=450),
                ),
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.search_result.status.value == "exhausted"
    assert _action_names(executor) == ["scroll"]


def test_final_grounding_not_resolved_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(),
                seconds=0,
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_link(bounds=_box()),
                ),
                seconds=1,
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_link(),
                ),
                seconds=2,
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.search_result.status.value == "found"
    assert result.final_grounding.status.value == "not_found"
    assert _action_names(executor) == ["scroll"]


def test_wrong_final_target_text_blocks(tmp_path):
    result, _builder, _executor_builder, executor, _sleeps = _run(
        tmp_path,
        execute=True,
        observations=[
            _observation(
                elements=(),
                semantics=(),
                seconds=0,
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_link(bounds=_box()),
                ),
                seconds=1,
            ),
            _observation(
                elements=(
                    _link_element(text="privacy notice"),
                ),
                semantics=(
                    _semantic_link(text="privacy notice"),
                ),
                seconds=2,
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.final_grounding.status.value == "resolved"
    assert result.final_grounding.element.text == "privacy notice"
    assert _action_names(executor) == ["scroll"]


def test_unexpected_non_scroll_tool_result_blocks(tmp_path):
    executor = FakeExecutor(
        tool_name="click_mouse",
    )

    result, _builder, _executor_builder, _executor, _sleeps = _run(
        tmp_path,
        execute=True,
        executor=executor,
        observations=[
            _observation(
                elements=(),
                semantics=(),
                seconds=0,
            ),
            _observation(
                elements=(),
                semantics=(
                    _semantic_link(bounds=_box()),
                ),
                seconds=1,
            ),
            _observation(
                elements=(
                    _link_element(),
                ),
                semantics=(
                    _semantic_link(),
                ),
                seconds=2,
            ),
        ],
    )

    assert result.status is experiment.AdaptiveWebRecoveryStatus.BLOCKED
    assert result.search_result.status.value == "found"
    assert result.search_result.tool_results[0].tool_name == "click_mouse"
    assert _action_names(executor) == ["scroll"]


def test_countdown_prints_each_visible_second(capsys):
    sleeps = []

    experiment._countdown(
        3,
        sleeper=sleeps.append,
    )

    assert capsys.readouterr().out.splitlines() == [
        "3...",
        "2...",
        "1...",
    ]
    assert sleeps == [
        1,
        1,
        1,
    ]
