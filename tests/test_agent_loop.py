from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import importlib
import inspect
from pathlib import Path

from PIL import Image
import pytest

from computer_agent.agent import (
    AgentLoop,
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    AgentStatus,
)
import computer_agent.agent.agent_loop as agent_loop_module
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import (
    ActionGrounder,
    ActionGroundingResult,
    ActionGroundingStatus,
    GroundingResult,
    GroundingStatus,
    TargetSpec,
    UIGrounder,
)
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.planning import (
    ActivateAppStep,
    InsertTextStep,
    PlanOperation,
    PlanStep,
    ReadClipboardStep,
    StructuredPlan,
)
from computer_agent.recovery import (
    ActionRecovery,
    RecoveryResult,
    RecoveryStatus,
)
from computer_agent.verification import (
    ActionVerificationResult,
    ActionVerificationStatus,
    ActionVerifier,
    StateVerificationResult,
    StateVerificationStatus,
)


TARGET_TEXT = "AGENT_LOOP_TARGET"


def _time(second: int = 0) -> datetime:
    return datetime(
        2026,
        9,
        1,
        12,
        0,
        second,
        tzinfo=timezone.utc,
    )


def _element(
    *,
    text: str = TARGET_TEXT,
    x: int = 10,
    y: int = 20,
    width: int = 100,
    height: int = 30,
) -> UIElement:
    return UIElement(
        element_type="button",
        bounding_box=BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
        ),
        confidence=0.95,
        text=text,
        enabled=True,
        source="accessibility",
    )


def _snapshot(
    elements=(),
    *,
    second: int = 0,
    screen_size: tuple[int, int] = (200, 100),
) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("synthetic.png"),
            pixel_width=screen_size[0],
            pixel_height=screen_size[1],
            screen_width=screen_size[0],
            screen_height=screen_size[1],
            captured_at=_time(second),
        ),
        image=Image.new(
            "RGB",
            screen_size,
        ),
        accessibility_elements=(),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=(),
    )


def _target(text: str = TARGET_TEXT) -> TargetSpec:
    return TargetSpec(
        text=text,
        element_types=("button",),
    )


def _step(
    goal: str = "Click target",
    *,
    action_text: str = "Action target",
    verification_text: str = "Verification target",
    max_attempts: int = 1,
) -> PlanStep:
    return PlanStep(
        goal=goal,
        operation=PlanOperation.CLICK_TARGET,
        action_target=_target(action_text),
        verification_target=_target(verification_text),
        max_attempts=max_attempts,
    )


def _read_step(
    *,
    value_key: str = "clipboard_value",
    expected_text: str = "CROSS_APP_TRANSFER_10",
    max_attempts: int = 1,
) -> ReadClipboardStep:
    return ReadClipboardStep(
        goal="Read clipboard value",
        value_key=value_key,
        expected_text=expected_text,
        max_attempts=max_attempts,
    )


def _activate_step(
    *,
    app_name: str = "TextEdit",
    max_attempts: int = 1,
) -> ActivateAppStep:
    return ActivateAppStep(
        goal="Activate application",
        app_name=app_name,
        max_attempts=max_attempts,
    )


def _insert_step(
    *,
    value_key: str = "clipboard_value",
    max_attempts: int = 1,
) -> InsertTextStep:
    return InsertTextStep(
        goal="Insert runtime value",
        value_key=value_key,
        max_attempts=max_attempts,
    )


def _plan(*steps) -> StructuredPlan:
    return StructuredPlan(
        task_goal="Complete deterministic task",
        steps=steps,
    )


def _action(
    *,
    x: int = 60,
    y: int = 35,
    reason: str = "prepared by fake action grounder",
) -> Action:
    return Action(
        tool_name="click_mouse",
        arguments={
            "x": x,
            "y": y,
        },
        reason=reason,
    )


def _tool_result(
    action: Action,
    *,
    success: bool = True,
    output=None,
    error: str | None = None,
) -> ToolResult:
    if not success and error is None:
        error = "execution failed"

    return ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=success,
        output=output,
        error=error,
    )


def _successful_output(output):
    return lambda action: _tool_result(
        action,
        output=output,
    )


def _failed_result(error: str = "execution failed"):
    return lambda action: _tool_result(
        action,
        success=False,
        error=error,
    )


def _grounding(status: GroundingStatus) -> GroundingResult:
    return GroundingResult(
        status=status,
        element=_element() if status is GroundingStatus.RESOLVED else None,
        candidates=(),
        reason=status.value,
    )


def _ready_action_grounding(action: Action) -> ActionGroundingResult:
    return ActionGroundingResult(
        status=ActionGroundingStatus.READY,
        action=action,
        reason="ready",
    )


def _blocked_action_grounding() -> ActionGroundingResult:
    return ActionGroundingResult(
        status=ActionGroundingStatus.BLOCKED,
        action=None,
        reason="blocked",
    )


def _verification(
    status: ActionVerificationStatus,
) -> ActionVerificationResult:
    if status is ActionVerificationStatus.VERIFIED:
        before_grounding = _grounding(GroundingStatus.NOT_FOUND)
        after_grounding = _grounding(GroundingStatus.RESOLVED)
    else:
        before_grounding = _grounding(GroundingStatus.NOT_FOUND)
        after_grounding = _grounding(GroundingStatus.NOT_FOUND)

    return ActionVerificationResult(
        status=status,
        before_grounding=before_grounding,
        after_grounding=after_grounding,
        reason=status.value,
    )


def _state_verification(
    status: StateVerificationStatus,
) -> StateVerificationResult:
    return StateVerificationResult(
        status=status,
        reason=status.value,
    )


def _retry_ready(action: Action) -> RecoveryResult:
    return RecoveryResult(
        status=RecoveryStatus.RETRY_READY,
        grounding_result=_grounding(GroundingStatus.RESOLVED),
        action_grounding_result=_ready_action_grounding(action),
        reason="retry ready",
    )


def _recovery_blocked(reason: str = "blocked") -> RecoveryResult:
    return RecoveryResult(
        status=RecoveryStatus.BLOCKED,
        grounding_result=None,
        action_grounding_result=None,
        reason=reason,
    )


def _recovery_exhausted(reason: str = "exhausted") -> RecoveryResult:
    return RecoveryResult(
        status=RecoveryStatus.EXHAUSTED,
        grounding_result=None,
        action_grounding_result=None,
        reason=reason,
    )


def _recovery_not_needed() -> RecoveryResult:
    return RecoveryResult(
        status=RecoveryStatus.NOT_NEEDED,
        grounding_result=None,
        action_grounding_result=None,
        reason="not needed",
    )


class SequencePerception:
    def __init__(
        self,
        snapshots=(),
        *,
        error_on_call: int | None = None,
    ) -> None:
        self.snapshots = list(snapshots)
        self.error_on_call = error_on_call
        self.calls = 0

    def observe(self):
        self.calls += 1

        if self.error_on_call == self.calls:
            raise RuntimeError("perception failed")

        if not self.snapshots:
            raise AssertionError("unexpected perception observe")

        return self.snapshots.pop(0)


class RecordingGrounder:
    def __init__(
        self,
        results=(),
        *,
        error: RuntimeError | None = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls = []

    def ground(self, target_spec, elements):
        self.calls.append(
            {
                "target_spec": target_spec,
                "elements": elements,
            }
        )

        if self.error is not None:
            raise self.error

        if not self.results:
            raise AssertionError("unexpected UI grounding")

        return self.results.pop(0)


class RecordingActionGrounder:
    def __init__(
        self,
        results=(),
    ) -> None:
        self.results = list(results)
        self.calls = []

    def ground_click(self, grounding_result, screen_size):
        self.calls.append(
            {
                "grounding_result": grounding_result,
                "screen_size": screen_size,
            }
        )

        if not self.results:
            raise AssertionError("unexpected action grounding")

        return self.results.pop(0)


class RecordingExecutor:
    def __init__(self, outcomes=()) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def execute(self, action):
        self.calls.append(action)

        if self.outcomes:
            outcome = self.outcomes.pop(0)
        else:
            outcome = True

        if isinstance(outcome, ToolResult):
            return outcome

        if callable(outcome):
            return outcome(action)

        return _tool_result(
            action,
            success=bool(outcome),
        )


class RecordingVerifier:
    def __init__(
        self,
        results=(),
        *,
        error: RuntimeError | None = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls = []

    def verify_target_appeared(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        if not self.results:
            raise AssertionError("unexpected verification")

        result = self.results.pop(0)
        if isinstance(result, ActionVerificationStatus):
            return _verification(result)

        return result


class RecordingStateVerifier:
    def __init__(
        self,
        *,
        frontmost_results=(),
        focused_results=(),
    ) -> None:
        self.frontmost_results = list(frontmost_results)
        self.focused_results = list(focused_results)
        self.frontmost_calls = []
        self.focused_calls = []

    def verify_frontmost_application(self, expected_app_name):
        self.frontmost_calls.append(expected_app_name)

        if not self.frontmost_results:
            raise AssertionError("unexpected frontmost verification")

        result = self.frontmost_results.pop(0)
        if isinstance(result, StateVerificationStatus):
            return _state_verification(result)

        return result

    def verify_focused_editable_value(self, snapshot, expected_value):
        self.focused_calls.append(
            {
                "snapshot": snapshot,
                "expected_value": expected_value,
            }
        )

        if not self.focused_results:
            raise AssertionError("unexpected focused editable verification")

        result = self.focused_results.pop(0)
        if isinstance(result, StateVerificationStatus):
            return _state_verification(result)

        return result


class RecordingSleeper:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class RecordingRecovery:
    def __init__(
        self,
        results=(),
        *,
        error: RuntimeError | None = None,
    ) -> None:
        self.results = list(results)
        self.error = error
        self.calls = []

    def prepare_retry(self, **kwargs):
        self.calls.append(kwargs)

        if self.error is not None:
            raise self.error

        if not self.results:
            raise AssertionError("unexpected recovery")

        return self.results.pop(0)


def _agent_loop(
    *,
    perception_engine,
    grounder,
    action_grounder,
    executor,
    verifier,
    recovery,
    state_verifier=None,
    allowed_app_names=None,
    frontmost_app_settle_timeout_seconds=0.0,
    frontmost_app_settle_poll_seconds=0.1,
    settling_sleep=None,
) -> AgentLoop:
    if settling_sleep is None:
        settling_sleep = RecordingSleeper()

    return AgentLoop(
        perception_engine=perception_engine,
        grounder=grounder,
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
        state_verifier=state_verifier,
        allowed_app_names=allowed_app_names,
        frontmost_app_settle_timeout_seconds=(
            frontmost_app_settle_timeout_seconds
        ),
        frontmost_app_settle_poll_seconds=(
            frontmost_app_settle_poll_seconds
        ),
        settling_sleep=settling_sleep,
    )


def _run_with_initial_context(
    monkeypatch,
    loop: AgentLoop,
    plan: StructuredPlan,
    context,
):
    state = AgentState(user_task=plan.task_goal)
    state.context.update(context)

    def make_state(user_task):
        assert user_task == plan.task_goal
        return state

    monkeypatch.setattr(agent_loop_module, "AgentState", make_state)

    return loop.run(plan)


def test_one_step_successful_plan_completes_and_records_attempt():
    step = _step()
    plan = _plan(step)
    before = _snapshot(second=0)
    after = _snapshot(second=1)
    action = _action(x=913, y=411)
    executor = RecordingExecutor()

    result = _agent_loop(
        perception_engine=SequencePerception((before, after)),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(action),)
        ),
        executor=executor,
        verifier=RecordingVerifier((ActionVerificationStatus.VERIFIED,)),
        recovery=RecordingRecovery(()),
    ).run(plan)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.state.user_task == plan.task_goal
    assert len(result.state.steps) == 1
    assert result.state.steps[0].action is action
    assert result.state.steps[0].observation is None
    assert executor.calls == [action]
    assert result.state.steps[0].action.arguments == {
        "x": 913,
        "y": 411,
    }


def test_two_step_successful_plan_completes_with_two_records():
    first = _step("Click first", action_text="First")
    second = _step("Click second", action_text="Second")
    plan = _plan(first, second)
    snapshots = (
        _snapshot(second=0),
        _snapshot(second=1),
        _snapshot(second=2),
        _snapshot(second=3),
    )
    first_action = _action(x=10, y=20)
    second_action = _action(x=30, y=40)
    grounder = RecordingGrounder(
        (
            _grounding(GroundingStatus.RESOLVED),
            _grounding(GroundingStatus.RESOLVED),
        )
    )
    action_grounder = RecordingActionGrounder(
        (
            _ready_action_grounding(first_action),
            _ready_action_grounding(second_action),
        )
    )
    executor = RecordingExecutor()
    verifier = RecordingVerifier(
        (
            ActionVerificationStatus.VERIFIED,
            ActionVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(snapshots),
        grounder=grounder,
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=RecordingRecovery(()),
    ).run(plan)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 2
    assert result.state.status is AgentStatus.SUCCEEDED
    assert len(result.state.steps) == 2
    assert [record.action for record in result.state.steps] == [
        first_action,
        second_action,
    ]
    assert executor.calls == [
        first_action,
        second_action,
    ]
    assert len(grounder.calls) == 2
    assert len(action_grounder.calls) == 2
    assert verifier.calls[0]["before_snapshot"] is snapshots[0]
    assert verifier.calls[0]["after_snapshot"] is snapshots[1]
    assert verifier.calls[1]["before_snapshot"] is snapshots[2]
    assert verifier.calls[1]["after_snapshot"] is snapshots[3]


@pytest.mark.parametrize(
    "status",
    [
        GroundingStatus.NOT_FOUND,
        GroundingStatus.AMBIGUOUS,
        GroundingStatus.UNSAFE,
    ],
)
def test_initial_grounding_failure_blocks_without_execution_or_recovery(
    status,
):
    executor = RecordingExecutor()
    recovery = RecordingRecovery(())
    verifier = RecordingVerifier(())
    action_grounder = RecordingActionGrounder(())

    result = _agent_loop(
        perception_engine=SequencePerception((_snapshot(),)),
        grounder=RecordingGrounder((_grounding(status),)),
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
    ).run(_plan(_step()))

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert result.state.status is AgentStatus.FAILED
    assert status.value in result.reason
    assert executor.calls == []
    assert action_grounder.calls == []
    assert verifier.calls == []
    assert recovery.calls == []


def test_action_grounding_blocked_blocks_without_execution_or_recovery():
    executor = RecordingExecutor()
    recovery = RecordingRecovery(())
    verifier = RecordingVerifier(())

    result = _agent_loop(
        perception_engine=SequencePerception((_snapshot(),)),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_blocked_action_grounding(),)
        ),
        executor=executor,
        verifier=verifier,
        recovery=recovery,
    ).run(_plan(_step()))

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert result.state.status is AgentStatus.FAILED
    assert "initial action grounding" in result.reason
    assert executor.calls == []
    assert verifier.calls == []
    assert recovery.calls == []


def test_verified_step_advances_and_failure_keeps_partial_count():
    first = _step("Click first")
    second = _step("Click second")
    plan = _plan(first, second)
    first_action = _action(x=10, y=20)
    executor = RecordingExecutor()

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
                _snapshot(second=2),
            )
        ),
        grounder=RecordingGrounder(
            (
                _grounding(GroundingStatus.RESOLVED),
                _grounding(GroundingStatus.NOT_FOUND),
            )
        ),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(first_action),)
        ),
        executor=executor,
        verifier=RecordingVerifier((ActionVerificationStatus.VERIFIED,)),
        recovery=RecordingRecovery(()),
    ).run(plan)

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert result.state.status is AgentStatus.FAILED
    assert len(result.state.steps) == 1
    assert executor.calls == [first_action]


def test_failed_verification_retry_ready_executes_recovery_action():
    step = _step(max_attempts=2)
    plan = _plan(step)
    before = _snapshot(second=0)
    after_first = _snapshot(second=1)
    after_retry = _snapshot(second=2)
    initial_action = _action(x=11, y=22)
    retry_action = _action(x=77, y=88, reason="prepared by recovery")
    grounder = RecordingGrounder((_grounding(GroundingStatus.RESOLVED),))
    action_grounder = RecordingActionGrounder(
        (_ready_action_grounding(initial_action),)
    )
    executor = RecordingExecutor()
    verifier = RecordingVerifier(
        (
            ActionVerificationStatus.FAILED,
            ActionVerificationStatus.VERIFIED,
        )
    )
    recovery = RecordingRecovery((_retry_ready(retry_action),))

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                before,
                after_first,
                after_retry,
            )
        ),
        grounder=grounder,
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
    ).run(plan)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert len(result.state.steps) == 2
    assert executor.calls == [
        initial_action,
        retry_action,
    ]
    assert result.state.steps[1].action is retry_action
    assert len(grounder.calls) == 1
    assert len(action_grounder.calls) == 1
    assert verifier.calls[0]["before_snapshot"] is before
    assert verifier.calls[0]["after_snapshot"] is after_first
    assert verifier.calls[1]["before_snapshot"] is after_first
    assert verifier.calls[1]["after_snapshot"] is after_retry
    assert recovery.calls[0]["latest_snapshot"] is after_first
    assert recovery.calls[0]["completed_attempts"] == 1
    assert recovery.calls[0]["max_attempts"] == 2
    assert recovery.calls[0]["target_spec"] is step.action_target


def test_multiple_retries_pass_updated_attempt_counts_until_exhausted():
    step = _step(max_attempts=3)
    first_action = _action(x=1, y=1)
    second_action = _action(x=2, y=2)
    third_action = _action(x=3, y=3)
    recovery = RecordingRecovery(
        (
            _retry_ready(second_action),
            _retry_ready(third_action),
            _recovery_exhausted("limit reached"),
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
                _snapshot(second=2),
                _snapshot(second=3),
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(first_action),)
        ),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier(
            (
                ActionVerificationStatus.FAILED,
                ActionVerificationStatus.FAILED,
                ActionVerificationStatus.FAILED,
            )
        ),
        recovery=recovery,
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert result.state.status is AgentStatus.FAILED
    assert len(result.state.steps) == 3
    assert [
        call["completed_attempts"]
        for call in recovery.calls
    ] == [1, 2, 3]


def test_recovery_blocked_returns_blocked_terminal_result():
    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(_action()),)
        ),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier((ActionVerificationStatus.FAILED,)),
        recovery=RecordingRecovery((_recovery_blocked("cannot reground"),)),
    ).run(_plan(_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert "cannot reground" in result.reason


def test_inconclusive_verification_blocks_through_recovery():
    recovery = RecordingRecovery((_recovery_blocked("inconclusive"),))

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(_action()),)
        ),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier((ActionVerificationStatus.INCONCLUSIVE,)),
        recovery=recovery,
    ).run(_plan(_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.BLOCKED
    assert recovery.calls[0]["verification_result"].status is (
        ActionVerificationStatus.INCONCLUSIVE
    )


def test_failed_tool_result_blocks_through_recovery():
    action = _action()
    recovery = RecordingRecovery((_recovery_blocked("execution failed"),))

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(action),)
        ),
        executor=RecordingExecutor((False,)),
        verifier=RecordingVerifier((ActionVerificationStatus.FAILED,)),
        recovery=recovery,
    ).run(_plan(_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.BLOCKED
    assert len(result.state.steps) == 1
    assert result.state.steps[0].result.success is False
    assert recovery.calls[0]["tool_result"].success is False
    assert result.state.last_error == result.reason


def test_unexpected_recovery_not_needed_raises_runtime_error():
    loop = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(_action()),)
        ),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier((ActionVerificationStatus.FAILED,)),
        recovery=RecordingRecovery((_recovery_not_needed(),)),
    )

    with pytest.raises(RuntimeError, match="NOT_NEEDED"):
        loop.run(_plan(_step(max_attempts=2)))


def test_unsupported_future_operation_raises_runtime_error():
    step = _step()
    object.__setattr__(
        step,
        "operation",
        "future_operation",
    )
    loop = _agent_loop(
        perception_engine=SequencePerception((_snapshot(),)),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    )

    with pytest.raises(RuntimeError, match="unsupported plan operation"):
        loop.run(_plan(step))


def test_invalid_typed_semantic_step_cannot_execute():
    step = _read_step()
    object.__setattr__(
        step,
        "operation",
        PlanOperation.CLICK_TARGET,
    )
    executor = RecordingExecutor()

    loop = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    )

    with pytest.raises(RuntimeError, match="unsupported plan operation"):
        loop.run(_plan(step))

    assert executor.calls == []


def test_read_clipboard_constructs_exact_action_and_stores_verified_value():
    step = _read_step()
    executor = RecordingExecutor(
        (
            _successful_output(
                {
                    "text": "CROSS_APP_TRANSFER_10",
                }
            ),
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.completed_plan_steps == 1
    assert result.state.context["values"][step.value_key] == (
        "CROSS_APP_TRANSFER_10"
    )
    assert len(result.state.steps) == 1
    action = result.state.steps[0].action
    assert action.tool_name == "read_from_clipboard"
    assert action.arguments == {}
    assert action.reason
    assert executor.calls == [action]


def test_read_clipboard_preserves_existing_context_and_values(monkeypatch):
    step = _read_step()
    plan = _plan(step)
    loop = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _successful_output(
                    {
                        "text": "CROSS_APP_TRANSFER_10",
                    }
                ),
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    )

    result = _run_with_initial_context(
        monkeypatch,
        loop,
        plan,
        {
            "other": "preserved",
            "values": {
                "existing": "kept",
            },
        },
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.context["other"] == "preserved"
    assert result.state.context["values"] == {
        "existing": "kept",
        step.value_key: "CROSS_APP_TRANSFER_10",
    }


def test_read_clipboard_malformed_values_context_blocks_before_execution(
    monkeypatch,
):
    step = _read_step()
    plan = _plan(step)
    executor = RecordingExecutor()
    loop = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    )

    result = _run_with_initial_context(
        monkeypatch,
        loop,
        plan,
        {
            "values": "not a mapping",
        },
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert executor.calls == []


@pytest.mark.parametrize(
    "outcome",
    [
        _successful_output({"text": "different"}),
        _successful_output(None),
        _successful_output({}),
        _successful_output({"text": 123}),
        _failed_result("clipboard unavailable"),
    ],
)
def test_unverified_clipboard_content_is_never_stored(outcome):
    step = _read_step()

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor((outcome,)),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert result.state.context == {}
    assert len(result.state.steps) == 1
    assert result.state.steps[0].action.tool_name == "read_from_clipboard"


def test_read_clipboard_retry_succeeds_and_records_separate_attempts():
    step = _read_step(max_attempts=3)

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _successful_output({"text": "not yet"}),
                _successful_output({"text": "CROSS_APP_TRANSFER_10"}),
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert result.state.context["values"][step.value_key] == (
        "CROSS_APP_TRANSFER_10"
    )
    assert len(result.state.steps) == 2
    assert result.state.steps[0] is not result.state.steps[1]
    assert result.state.steps[0].action is not result.state.steps[1].action
    assert result.state.steps[0].action.action_id != (
        result.state.steps[1].action.action_id
    )


def test_read_clipboard_exhaustion_returns_exhausted_without_value():
    step = _read_step(max_attempts=2)

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _successful_output({"text": "wrong"}),
                _successful_output({"text": "still wrong"}),
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert result.state.context == {}
    assert len(result.state.steps) == 2


def test_activate_app_default_allowlist_blocks_before_execution():
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(_plan(_activate_step()))

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert executor.calls == []
    assert state_verifier.frontmost_calls == []


@pytest.mark.parametrize(
    "allowed_app_names",
    [
        {"Safari"},
        {"Text"},
        {"TextEdit Pro"},
    ],
)
def test_activate_app_disallowed_names_block_exactly(allowed_app_names):
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names=allowed_app_names,
    ).run(_plan(_activate_step(app_name="TextEdit")))

    assert result.status is AgentLoopStatus.BLOCKED
    assert executor.calls == []
    assert state_verifier.frontmost_calls == []


def test_activate_app_missing_state_verifier_blocks_before_execution():
    executor = RecordingExecutor()

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        allowed_app_names={"TextEdit"},
    ).run(_plan(_activate_step()))

    assert result.status is AgentLoopStatus.BLOCKED
    assert "state verifier" in result.reason
    assert executor.calls == []


def test_activate_app_allowlisted_action_is_exact_and_verified():
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    ).run(_plan(_activate_step()))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert state_verifier.frontmost_calls == ["TextEdit"]
    assert len(result.state.steps) == 1
    action = result.state.steps[0].action
    assert action.tool_name == "activate_app"
    assert action.arguments == {"app_name": "TextEdit"}
    assert set(action.arguments) == {"app_name"}
    assert executor.calls == [action]


def test_activate_app_settling_can_verify_after_single_action():
    sleeper = RecordingSleeper()
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.FAILED,
            StateVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=sleeper,
    ).run(_plan(_activate_step(max_attempts=1)))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert len(executor.calls) == 1
    assert len(result.state.steps) == 1
    assert result.state.steps[0].action.tool_name == "activate_app"
    assert state_verifier.frontmost_calls == ["TextEdit", "TextEdit"]
    assert sleeper.calls == [0.1]


def test_activate_app_settling_allows_multiple_failed_polls_then_verified():
    sleeper = RecordingSleeper()
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.FAILED,
            StateVerificationStatus.FAILED,
            StateVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=sleeper,
    ).run(_plan(_activate_step(max_attempts=1)))

    assert result.status is AgentLoopStatus.COMPLETED
    assert len(executor.calls) == 1
    assert len(result.state.steps) == 1
    assert state_verifier.frontmost_calls == [
        "TextEdit",
        "TextEdit",
        "TextEdit",
    ]
    assert sleeper.calls == [0.1, 0.1]


def test_activate_app_settling_allows_inconclusive_then_verified():
    sleeper = RecordingSleeper()
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.INCONCLUSIVE,
            StateVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=sleeper,
    ).run(_plan(_activate_step(max_attempts=1)))

    assert result.status is AgentLoopStatus.COMPLETED
    assert len(executor.calls) == 1
    assert len(result.state.steps) == 1
    assert state_verifier.frontmost_calls == ["TextEdit", "TextEdit"]
    assert sleeper.calls == [0.1]


def test_activate_app_settling_exhausts_before_attempt_fails():
    sleeper = RecordingSleeper()
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.FAILED,
            StateVerificationStatus.FAILED,
            StateVerificationStatus.FAILED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=sleeper,
    ).run(_plan(_activate_step(max_attempts=1)))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1
    assert len(result.state.steps) == 1
    assert state_verifier.frontmost_calls == [
        "TextEdit",
        "TextEdit",
        "TextEdit",
    ]
    assert sleeper.calls == [0.1, 0.1]


def test_activate_app_semantic_retry_starts_after_settling_exhausts():
    sleeper = RecordingSleeper()
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.FAILED,
            StateVerificationStatus.FAILED,
            StateVerificationStatus.FAILED,
            StateVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=sleeper,
    ).run(_plan(_activate_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert len(executor.calls) == 2
    assert len(result.state.steps) == 2
    assert result.state.steps[0].action is not result.state.steps[1].action
    assert [record.action.tool_name for record in result.state.steps] == [
        "activate_app",
        "activate_app",
    ]
    assert state_verifier.frontmost_calls == [
        "TextEdit",
        "TextEdit",
        "TextEdit",
        "TextEdit",
    ]
    assert sleeper.calls == [0.1, 0.1]


@pytest.mark.parametrize(
    "first_status",
    [
        StateVerificationStatus.FAILED,
        StateVerificationStatus.INCONCLUSIVE,
    ],
)
def test_activate_app_retries_failed_or_inconclusive_verification(
    first_status,
):
    step = _activate_step(max_attempts=2)
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            first_status,
            StateVerificationStatus.VERIFIED,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    ).run(_plan(step))

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 1
    assert state_verifier.frontmost_calls == ["TextEdit", "TextEdit"]
    assert len(result.state.steps) == 2
    assert result.state.steps[0].action is not result.state.steps[1].action


def test_activate_app_tool_failure_can_retry_until_verified():
    state_verifier = RecordingStateVerifier(
        frontmost_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _failed_result("activation failed"),
                True,
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    ).run(_plan(_activate_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.COMPLETED
    assert len(result.state.steps) == 2
    assert result.state.steps[0].result.success is False
    assert result.state.steps[1].result.success is True
    assert state_verifier.frontmost_calls == ["TextEdit"]


def test_activate_app_exhaustion_returns_exhausted():
    state_verifier = RecordingStateVerifier(
        frontmost_results=(
            StateVerificationStatus.FAILED,
            StateVerificationStatus.INCONCLUSIVE,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    ).run(_plan(_activate_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.state.status is AgentStatus.FAILED
    assert result.completed_plan_steps == 0
    assert len(result.state.steps) == 2


def test_insert_text_missing_value_key_blocks_before_execution():
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(_plan(_insert_step()))

    assert result.status is AgentLoopStatus.BLOCKED
    assert executor.calls == []
    assert state_verifier.focused_calls == []


@pytest.mark.parametrize(
    "context",
    [
        {"values": "not a mapping"},
        {"values": {"clipboard_value": 123}},
        {"values": {"clipboard_value": ""}},
        {"values": {"clipboard_value": "   "}},
    ],
)
def test_insert_text_malformed_runtime_value_blocks(
    monkeypatch,
    context,
):
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )
    plan = _plan(_insert_step())
    loop = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    )

    result = _run_with_initial_context(
        monkeypatch,
        loop,
        plan,
        context,
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert executor.calls == []
    assert state_verifier.focused_calls == []


def test_insert_text_missing_state_verifier_blocks_before_execution():
    step = _insert_step()
    read_step = _read_step(value_key=step.value_key)
    executor = RecordingExecutor(
        (
            _successful_output({"text": "CROSS_APP_TRANSFER_10"}),
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    ).run(
        _plan(
            read_step,
            step,
        )
    )

    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == [
        "read_from_clipboard"
    ]


def test_insert_text_max_attempts_above_one_blocks_before_execution():
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(_plan(_insert_step(max_attempts=2)))

    assert result.status is AgentLoopStatus.BLOCKED
    assert "max_attempts" in result.reason
    assert executor.calls == []
    assert state_verifier.focused_calls == []


def test_insert_text_uses_verified_runtime_value_from_context():
    class RuntimeText(str):
        pass

    runtime_text = RuntimeText("CROSS_APP_TRANSFER_10")
    read_step = _read_step(expected_text=str(runtime_text))
    insert_step = _insert_step(value_key=read_step.value_key)
    after_insert = _snapshot(second=2)
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception((after_insert,)),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _successful_output({"text": runtime_text}),
                True,
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(
        _plan(
            read_step,
            insert_step,
        )
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.completed_plan_steps == 2
    assert result.state.context["values"][read_step.value_key] is runtime_text
    paste_action = result.state.steps[1].action
    assert paste_action.tool_name == "paste_text"
    assert paste_action.arguments == {"text": runtime_text}
    assert paste_action.arguments["text"] is runtime_text
    assert len(state_verifier.focused_calls) == 1
    assert state_verifier.focused_calls[0]["snapshot"] is after_insert
    assert state_verifier.focused_calls[0]["expected_value"] is runtime_text


def test_insert_text_failed_paste_returns_exhausted_without_retry():
    read_step = _read_step()
    insert_step = _insert_step(value_key=read_step.value_key)
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(
            (
                _successful_output({"text": "CROSS_APP_TRANSFER_10"}),
                _failed_result("paste failed"),
            )
        ),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(
        _plan(
            read_step,
            insert_step,
        )
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert len(result.state.steps) == 2
    assert result.state.steps[1].action.tool_name == "paste_text"
    assert state_verifier.focused_calls == []


@pytest.mark.parametrize(
    "status",
    [
        StateVerificationStatus.FAILED,
        StateVerificationStatus.INCONCLUSIVE,
    ],
)
def test_insert_text_unverified_postcondition_is_exhausted_without_retry(
    status,
):
    read_step = _read_step()
    insert_step = _insert_step(value_key=read_step.value_key)
    after_insert = _snapshot(second=2)
    state_verifier = RecordingStateVerifier(focused_results=(status,))
    executor = RecordingExecutor(
        (
            _successful_output({"text": "CROSS_APP_TRANSFER_10"}),
            True,
            True,
        )
    )

    result = _agent_loop(
        perception_engine=SequencePerception((after_insert,)),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=executor,
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
    ).run(
        _plan(
            read_step,
            insert_step,
        )
    )

    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 1
    assert len(result.state.steps) == 2
    assert [record.action.tool_name for record in result.state.steps] == [
        "read_from_clipboard",
        "paste_text",
    ]
    assert len(executor.calls) == 2
    assert state_verifier.focused_calls[0]["snapshot"] is after_insert


def test_full_mixed_plan_completes_with_expected_action_order():
    click_step = _step()
    read_step = _read_step()
    activate_step = _activate_step()
    insert_step = _insert_step(value_key=read_step.value_key)
    plan = _plan(
        click_step,
        read_step,
        activate_step,
        insert_step,
    )
    click_action = _action(x=15, y=25)
    clipboard_text = "CROSS_APP_TRANSFER_10"
    after_insert = _snapshot(second=3)
    executor = RecordingExecutor(
        (
            True,
            _successful_output({"text": clipboard_text}),
            True,
            True,
        )
    )
    state_verifier = RecordingStateVerifier(
        frontmost_results=(StateVerificationStatus.VERIFIED,),
        focused_results=(StateVerificationStatus.VERIFIED,),
    )

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
                after_insert,
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(click_action),)
        ),
        executor=executor,
        verifier=RecordingVerifier((ActionVerificationStatus.VERIFIED,)),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    ).run(plan)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.completed_plan_steps == 4
    assert result.plan is plan
    assert len(result.state.steps) == 4
    assert all(record.result.success for record in result.state.steps)
    assert [record.action.tool_name for record in result.state.steps] == [
        "click_mouse",
        "read_from_clipboard",
        "activate_app",
        "paste_text",
    ]
    assert result.state.context["values"][read_step.value_key] == (
        clipboard_text
    )
    assert result.state.steps[3].action.arguments == {
        "text": clipboard_text,
    }
    assert executor.calls == [
        record.action for record in result.state.steps
    ]
    assert state_verifier.frontmost_calls == ["TextEdit"]
    assert len(state_verifier.focused_calls) == 1
    assert state_verifier.focused_calls[0]["snapshot"] is after_insert
    assert state_verifier.focused_calls[0]["expected_value"] == clipboard_text


def test_click_read_and_insert_paths_do_not_use_activation_settling():
    click_step = _step()
    read_step = _read_step()
    insert_step = _insert_step(value_key=read_step.value_key)
    click_action = _action()
    clipboard_text = "CROSS_APP_TRANSFER_10"
    after_insert = _snapshot(second=3)
    state_verifier = RecordingStateVerifier(
        focused_results=(StateVerificationStatus.VERIFIED,)
    )

    result = _agent_loop(
        perception_engine=SequencePerception(
            (
                _snapshot(second=0),
                _snapshot(second=1),
                after_insert,
            )
        ),
        grounder=RecordingGrounder((_grounding(GroundingStatus.RESOLVED),)),
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(click_action),)
        ),
        executor=RecordingExecutor(
            (
                True,
                _successful_output({"text": clipboard_text}),
                True,
            )
        ),
        verifier=RecordingVerifier((ActionVerificationStatus.VERIFIED,)),
        recovery=RecordingRecovery(()),
        state_verifier=state_verifier,
        frontmost_app_settle_timeout_seconds=0.2,
        frontmost_app_settle_poll_seconds=0.1,
        settling_sleep=lambda _seconds: pytest.fail(
            "activation settling should not run"
        ),
    ).run(
        _plan(
            click_step,
            read_step,
            insert_step,
        )
    )

    assert result.status is AgentLoopStatus.COMPLETED
    assert [record.action.tool_name for record in result.state.steps] == [
        "click_mouse",
        "read_from_clipboard",
        "paste_text",
    ]
    assert state_verifier.frontmost_calls == []


def test_agent_loop_result_accepts_valid_terminal_shapes():
    step = _step()
    plan = _plan(step)
    succeeded = AgentState(user_task=plan.task_goal)
    succeeded.start()
    succeeded.succeed()
    failed = AgentState(user_task=plan.task_goal)
    failed.fail("blocked")

    completed = AgentLoopResult(
        status=AgentLoopStatus.COMPLETED,
        plan=plan,
        state=succeeded,
        completed_plan_steps=1,
        reason="complete",
    )
    blocked = AgentLoopResult(
        status=AgentLoopStatus.BLOCKED,
        plan=plan,
        state=failed,
        completed_plan_steps=0,
        reason="blocked",
    )

    assert completed.status is AgentLoopStatus.COMPLETED
    assert blocked.status is AgentLoopStatus.BLOCKED
    assert not hasattr(completed, "__dict__")
    with pytest.raises(FrozenInstanceError):
        completed.reason = "changed"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"status": "completed"},
            "status",
        ),
        (
            {"plan": object()},
            "plan",
        ),
        (
            {"state": object()},
            "state",
        ),
        (
            {"completed_plan_steps": True},
            "completed_plan_steps",
        ),
        (
            {"completed_plan_steps": -1},
            "completed_plan_steps",
        ),
        (
            {"completed_plan_steps": 2},
            "completed_plan_steps",
        ),
        (
            {"reason": ""},
            "reason",
        ),
    ],
)
def test_agent_loop_result_rejects_invalid_fields(kwargs, message):
    plan = _plan(_step())
    state = AgentState(user_task=plan.task_goal)
    state.start()
    state.succeed()
    arguments = {
        "status": AgentLoopStatus.COMPLETED,
        "plan": plan,
        "state": state,
        "completed_plan_steps": 1,
        "reason": "complete",
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        AgentLoopResult(**arguments)


@pytest.mark.parametrize(
    ("status", "completed_plan_steps", "state_status", "message"),
    [
        (
            AgentLoopStatus.COMPLETED,
            0,
            AgentStatus.SUCCEEDED,
            "every plan step",
        ),
        (
            AgentLoopStatus.COMPLETED,
            1,
            AgentStatus.FAILED,
            "SUCCEEDED",
        ),
        (
            AgentLoopStatus.BLOCKED,
            0,
            AgentStatus.RUNNING,
            "FAILED",
        ),
        (
            AgentLoopStatus.EXHAUSTED,
            0,
            AgentStatus.SUCCEEDED,
            "FAILED",
        ),
    ],
)
def test_agent_loop_result_rejects_invalid_lifecycle_invariants(
    status,
    completed_plan_steps,
    state_status,
    message,
):
    plan = _plan(_step())
    state = AgentState(user_task=plan.task_goal)
    if state_status is AgentStatus.RUNNING:
        state.start()
    elif state_status is AgentStatus.SUCCEEDED:
        state.start()
        state.succeed()
    elif state_status is AgentStatus.FAILED:
        state.fail("failed")

    with pytest.raises(ValueError, match=message):
        AgentLoopResult(
            status=status,
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            reason="terminal",
        )


def test_dependency_injection_exposes_custom_and_default_components():
    perception_engine = SequencePerception(())
    executor = RecordingExecutor()

    loop = AgentLoop(
        perception_engine=perception_engine,
        executor=executor,
    )

    assert loop.perception_engine is perception_engine
    assert loop.executor is executor
    assert isinstance(loop.grounder, UIGrounder)
    assert isinstance(loop.action_grounder, ActionGrounder)
    assert isinstance(loop.verifier, ActionVerifier)
    assert isinstance(loop.recovery, ActionRecovery)
    assert loop.verifier.grounder is loop.grounder
    assert loop.recovery.grounder is loop.grounder
    assert loop.recovery.action_grounder is loop.action_grounder
    assert loop.state_verifier is None
    assert loop.allowed_app_names == frozenset()


def test_dependency_injection_accepts_state_verifier_and_app_allowlist():
    perception_engine = SequencePerception(())
    executor = RecordingExecutor()
    state_verifier = RecordingStateVerifier()

    loop = AgentLoop(
        perception_engine=perception_engine,
        executor=executor,
        state_verifier=state_verifier,
        allowed_app_names={"TextEdit"},
    )

    assert loop.state_verifier is state_verifier
    assert loop.allowed_app_names == frozenset({"TextEdit"})


def test_explicit_concrete_grounder_is_reused_by_default_components():
    perception_engine = SequencePerception(())
    executor = RecordingExecutor()
    grounder = UIGrounder()

    loop = AgentLoop(
        perception_engine=perception_engine,
        grounder=grounder,
        executor=executor,
    )

    assert loop.grounder is grounder
    assert loop.verifier.grounder is grounder
    assert loop.recovery.grounder is grounder


def test_explicit_concrete_action_grounder_is_reused_by_default_recovery():
    perception_engine = SequencePerception(())
    executor = RecordingExecutor()
    action_grounder = ActionGrounder()

    loop = AgentLoop(
        perception_engine=perception_engine,
        action_grounder=action_grounder,
        executor=executor,
    )

    assert loop.action_grounder is action_grounder
    assert loop.recovery.action_grounder is action_grounder


def test_custom_grounder_with_omitted_verifier_fails_clearly():
    with pytest.raises(ValueError, match="explicit verifier"):
        AgentLoop(
            perception_engine=SequencePerception(()),
            grounder=RecordingGrounder(()),
            executor=RecordingExecutor(),
            recovery=RecordingRecovery(()),
        )


def test_custom_grounder_with_omitted_recovery_fails_clearly():
    with pytest.raises(ValueError, match="explicit recovery"):
        AgentLoop(
            perception_engine=SequencePerception(()),
            grounder=RecordingGrounder(()),
            executor=RecordingExecutor(),
            verifier=RecordingVerifier(()),
        )


def test_custom_action_grounder_with_omitted_recovery_fails_clearly():
    with pytest.raises(ValueError, match="explicit recovery"):
        AgentLoop(
            perception_engine=SequencePerception(()),
            action_grounder=RecordingActionGrounder(()),
            executor=RecordingExecutor(),
        )


def test_fully_explicit_custom_dependency_injection_remains_valid():
    perception_engine = SequencePerception(())
    grounder = RecordingGrounder(())
    action_grounder = RecordingActionGrounder(())
    executor = RecordingExecutor()
    verifier = RecordingVerifier(())
    recovery = RecordingRecovery(())

    loop = AgentLoop(
        perception_engine=perception_engine,
        grounder=grounder,
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
    )

    assert loop.perception_engine is perception_engine
    assert loop.grounder is grounder
    assert loop.action_grounder is action_grounder
    assert loop.executor is executor
    assert loop.verifier is verifier
    assert loop.recovery is recovery


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "perception_engine",
            None,
            "perception_engine",
        ),
        (
            "executor",
            None,
            "executor",
        ),
        (
            "grounder",
            object(),
            "grounder",
        ),
        (
            "action_grounder",
            object(),
            "action_grounder",
        ),
        (
            "verifier",
            object(),
            "verifier",
        ),
        (
            "recovery",
            object(),
            "recovery",
        ),
        (
            "state_verifier",
            object(),
            "state_verifier",
        ),
        (
            "settling_sleep",
            object(),
            "settling_sleep",
        ),
    ],
)
def test_invalid_dependencies_are_rejected(keyword, value, message):
    kwargs = {
        "perception_engine": SequencePerception(()),
        "executor": RecordingExecutor(),
    }
    kwargs[keyword] = value

    with pytest.raises(ValueError, match=message):
        AgentLoop(**kwargs)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "frontmost_app_settle_timeout_seconds",
            -0.1,
            "non-negative",
        ),
        (
            "frontmost_app_settle_timeout_seconds",
            float("nan"),
            "finite",
        ),
        (
            "frontmost_app_settle_poll_seconds",
            0,
            "positive",
        ),
        (
            "frontmost_app_settle_poll_seconds",
            "0.1",
            "numeric",
        ),
    ],
)
def test_invalid_frontmost_app_settling_configuration_is_rejected(
    keyword,
    value,
    message,
):
    kwargs = {
        "perception_engine": SequencePerception(()),
        "executor": RecordingExecutor(),
        keyword: value,
    }

    with pytest.raises(ValueError, match=message):
        AgentLoop(**kwargs)


@pytest.mark.parametrize(
    "allowed_app_names",
    [
        "TextEdit",
        b"TextEdit",
        {""},
        {"   "},
        {None},
        {123},
    ],
)
def test_invalid_allowed_app_names_are_rejected(allowed_app_names):
    with pytest.raises(ValueError, match="allowed_app_names"):
        AgentLoop(
            perception_engine=SequencePerception(()),
            executor=RecordingExecutor(),
            allowed_app_names=allowed_app_names,
        )


@pytest.mark.parametrize(
    "component",
    [
        "perception",
        "grounder",
        "verifier",
        "recovery",
    ],
)
def test_unexpected_internal_runtime_errors_propagate(component):
    perception_engine = SequencePerception(
        (
            _snapshot(second=0),
            _snapshot(second=1),
        ),
        error_on_call=1 if component == "perception" else None,
    )
    grounder = RecordingGrounder(
        (_grounding(GroundingStatus.RESOLVED),),
        error=RuntimeError("grounder failed")
        if component == "grounder"
        else None,
    )
    verifier = RecordingVerifier(
        (ActionVerificationStatus.FAILED,),
        error=RuntimeError("verifier failed")
        if component == "verifier"
        else None,
    )
    recovery = RecordingRecovery(
        (_recovery_blocked(),),
        error=RuntimeError("recovery failed")
        if component == "recovery"
        else None,
    )

    loop = _agent_loop(
        perception_engine=perception_engine,
        grounder=grounder,
        action_grounder=RecordingActionGrounder(
            (_ready_action_grounding(_action()),)
        ),
        executor=RecordingExecutor(),
        verifier=verifier,
        recovery=recovery,
    )

    with pytest.raises(RuntimeError, match="failed"):
        loop.run(_plan(_step(max_attempts=2)))


def test_run_rejects_non_structured_plan():
    loop = _agent_loop(
        perception_engine=SequencePerception(()),
        grounder=RecordingGrounder(()),
        action_grounder=RecordingActionGrounder(()),
        executor=RecordingExecutor(),
        verifier=RecordingVerifier(()),
        recovery=RecordingRecovery(()),
    )

    with pytest.raises(ValueError, match="StructuredPlan"):
        loop.run(object())


def test_agent_loop_imports_are_safe_and_do_not_create_coordinates():
    module = importlib.import_module("computer_agent.agent")
    source = inspect.getsource(agent_loop_module)

    forbidden_terms = (
        "computer_agent.reasoning",
        "openai",
        "llm",
        "pyautogui",
        '"x"',
        '"y"',
    )

    assert module.AgentLoop is AgentLoop
    assert module.AgentLoopResult is AgentLoopResult
    assert module.AgentLoopStatus is AgentLoopStatus
    assert all(term not in source for term in forbidden_terms)
