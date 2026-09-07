from computer_agent.core.models import ToolResult
from computer_agent.perception import (
    BoundingBox,
    DiagnosisStatus,
    SemanticAXElement,
    Viewport,
    ViewportSearchObservation,
    ViewportSearchPolicy,
    ViewportSearchStatus,
)
from experiments.phase05_real_web_autonomy import (
    experiment_04_scroll_viewport_search as experiment,
)


DEFAULT_VIEWPORT = object()


class FakeAccessibility:
    def __init__(
        self,
        observations,
    ):
        self.observations = list(observations)
        self.index = 0
        self.calls = []

    @property
    def current(self):
        return self.observations[self.index]

    def read_frontmost_application_name(self):
        self.calls.append("read_frontmost_application_name")

        return self.current.application_name

    def read_frontmost_viewport(self):
        self.calls.append("read_frontmost_viewport")

        return self.current.viewport

    def read_frontmost_semantic_elements(self):
        self.calls.append("read_frontmost_semantic_elements")
        semantic_elements = self.current.semantic_elements
        if self.index < len(self.observations) - 1:
            self.index += 1

        return semantic_elements


class FakeExecutor:
    def __init__(
        self,
        *,
        success=True,
    ):
        self.success = success
        self.actions = []

    def execute(self, action):
        self.actions.append(action)

        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=self.success,
            output={"amount": action.arguments["amount"]}
            if self.success
            else None,
            error=None if self.success else "synthetic scroll failure",
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


def _box(
    *,
    y,
):
    return BoundingBox(
        x=100,
        y=y,
        width=120,
        height=24,
    )


def _target(
    *,
    bounds,
    text="Privacy Notice",
):
    return SemanticAXElement(
        role="AXLink",
        text=text,
        bounds=bounds,
    )


def _marker(
    *,
    y,
    text="Marker",
):
    return SemanticAXElement(
        role="AXStaticText",
        text=text,
        bounds=_box(y=y),
    )


def _observation(
    *,
    application_name="Google Chrome",
    viewport=DEFAULT_VIEWPORT,
    elements=(),
):
    return ViewportSearchObservation(
        application_name=application_name,
        viewport=_viewport()
        if viewport is DEFAULT_VIEWPORT
        else viewport,
        semantic_elements=tuple(elements),
    )


def _policy(
    *,
    max_scroll_attempts=6,
    scroll_amount=4,
):
    return ViewportSearchPolicy(
        max_scroll_attempts=max_scroll_attempts,
        scroll_amount=scroll_amount,
        stabilization_wait_seconds=0.0,
    )


def _run(
    *,
    observations,
    execute=False,
    policy=None,
    executor=None,
):
    accessibility = FakeAccessibility(observations)
    executor = executor or FakeExecutor()
    sleeps = []

    result = experiment._run_search(
        execute=execute,
        accessibility=accessibility,
        policy=policy or _policy(),
        executor=executor if execute else None,
        sleeper=sleeps.append,
    )

    return result, accessibility, executor, sleeps


def _assert_only_scroll_actions(
    executor,
    *,
    count,
    amount=-4,
):
    assert len(executor.actions) == count
    assert all(action.tool_name == "scroll" for action in executor.actions)
    assert all(
        action.arguments == {"amount": amount}
        for action in executor.actions
    )


def test_read_only_diagnosis_path_performs_no_control_actions(capsys):
    accessibility = FakeAccessibility(
        [
            _observation(
                elements=(
                    _target(
                        bounds=None,
                    ),
                ),
            )
        ]
    )

    status = experiment._run_diagnosis(
        accessibility=accessibility,
    )

    output = capsys.readouterr().out

    assert status is DiagnosisStatus.NEEDS_SEARCH
    assert accessibility.calls == [
        "read_frontmost_application_name",
        "read_frontmost_viewport",
        "read_frontmost_semantic_elements",
    ]
    assert "Diagnosis status: needs_search" in output
    assert "visibility=VisibilityStatus.GEOMETRY_UNAVAILABLE" in output


def test_dry_run_needs_search_reports_needs_scroll_without_execution():
    result, accessibility, executor, sleeps = _run(
        observations=[
            _observation(
                elements=(
                    _target(
                        bounds=None,
                    ),
                ),
            )
        ],
    )

    assert result.status is ViewportSearchStatus.NEEDS_SCROLL
    assert result.scroll_attempts == 0
    assert len(result.observations) == 1
    assert len(accessibility.calls) == 3
    _assert_only_scroll_actions(executor, count=0)
    assert sleeps == []


def test_dry_run_visible_target_is_found_without_execution():
    result, _accessibility, executor, sleeps = _run(
        observations=[
            _observation(
                elements=(
                    _target(
                        bounds=_box(y=500),
                    ),
                ),
            )
        ],
    )

    assert result.status is ViewportSearchStatus.FOUND
    assert result.scroll_attempts == 0
    _assert_only_scroll_actions(executor, count=0)
    assert sleeps == []


def test_dry_run_blocked_diagnosis_blocks_without_execution():
    result, _accessibility, executor, sleeps = _run(
        observations=[
            _observation(
                application_name="Safari",
                elements=(
                    _target(
                        bounds=_box(y=500),
                    ),
                ),
            )
        ],
    )

    assert result.status is ViewportSearchStatus.BLOCKED
    assert result.scroll_attempts == 0
    _assert_only_scroll_actions(executor, count=0)
    assert sleeps == []


def test_execute_visible_target_is_found_without_scroll():
    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(
                        bounds=_box(y=500),
                    ),
                ),
            )
        ],
    )

    assert result.status is ViewportSearchStatus.FOUND
    assert result.scroll_attempts == 0
    _assert_only_scroll_actions(executor, count=0)
    assert sleeps == []


def test_execute_scrolls_once_then_finds_visible_target():
    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(
                        bounds=None,
                    ),
                ),
            ),
            _observation(
                elements=(
                    _target(
                        bounds=_box(y=500),
                    ),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.FOUND
    assert result.scroll_attempts == 1
    assert len(result.observations) == 2
    _assert_only_scroll_actions(executor, count=1)
    assert sleeps == [0.0]


def test_execute_repeated_needs_search_exhausts_exact_budget():
    policy = _policy(max_scroll_attempts=2)

    result, _accessibility, executor, sleeps = _run(
        execute=True,
        policy=policy,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=450),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=400),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.EXHAUSTED
    assert result.scroll_attempts == 2
    assert len(result.observations) == 3
    _assert_only_scroll_actions(executor, count=2)
    assert sleeps == [
        0.0,
        0.0,
    ]


def test_execute_successful_scroll_with_unchanged_observation_stalls():
    repeated = _observation(
        elements=(
            _target(bounds=None),
            _marker(y=500),
        ),
    )

    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            repeated,
            repeated,
        ],
    )

    assert result.status is ViewportSearchStatus.STALLED
    assert result.scroll_attempts == 1
    _assert_only_scroll_actions(executor, count=1)
    assert sleeps == [0.0]


def test_execute_blocks_when_app_changes_from_chrome():
    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                application_name="Safari",
                elements=(
                    _target(bounds=None),
                    _marker(y=450),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.BLOCKED
    assert result.scroll_attempts == 1
    _assert_only_scroll_actions(executor, count=1)
    assert sleeps == [0.0]


def test_execute_blocks_when_viewport_becomes_unavailable():
    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                viewport=None,
                elements=(
                    _target(bounds=None),
                    _marker(y=450),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.BLOCKED
    assert result.scroll_attempts == 1
    _assert_only_scroll_actions(executor, count=1)
    assert sleeps == [0.0]


def test_execute_blocks_when_target_becomes_ambiguous_after_scroll():
    result, _accessibility, executor, sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _target(bounds=_box(y=500)),
                    _marker(y=450),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.BLOCKED
    assert result.scroll_attempts == 1
    _assert_only_scroll_actions(executor, count=1)
    assert sleeps == [0.0]


def test_execute_no_scroll_count_exceeds_configured_maximum():
    policy = _policy(
        max_scroll_attempts=3,
        scroll_amount=5,
    )

    result, _accessibility, executor, _sleeps = _run(
        execute=True,
        policy=policy,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=450),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=400),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=350),
                ),
            ),
        ],
    )

    assert result.status is ViewportSearchStatus.EXHAUSTED
    assert result.scroll_attempts == policy.max_scroll_attempts
    _assert_only_scroll_actions(
        executor,
        count=3,
        amount=-5,
    )


def test_execute_search_uses_no_click_type_navigation_clipboard_or_llm_calls():
    result, _accessibility, executor, _sleeps = _run(
        execute=True,
        observations=[
            _observation(
                elements=(
                    _target(bounds=None),
                    _marker(y=500),
                ),
            ),
            _observation(
                elements=(
                    _target(bounds=_box(y=500)),
                    _marker(y=450),
                ),
            ),
        ],
    )

    forbidden_tools = {
        "click_mouse",
        "type_text",
        "open_url",
        "copy_to_clipboard",
        "paste_text",
        "read_from_clipboard",
    }

    assert result.status is ViewportSearchStatus.FOUND
    assert not {
        action.tool_name
        for action in executor.actions
    } & forbidden_tools
    _assert_only_scroll_actions(executor, count=1)
