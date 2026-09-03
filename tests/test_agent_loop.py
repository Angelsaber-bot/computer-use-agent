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
from computer_agent.planning import PlanOperation, PlanStep, StructuredPlan
from computer_agent.recovery import (
    ActionRecovery,
    RecoveryResult,
    RecoveryStatus,
)
from computer_agent.verification import (
    ActionVerificationResult,
    ActionVerificationStatus,
    ActionVerifier,
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


def _plan(*steps: PlanStep) -> StructuredPlan:
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
    error: str | None = None,
) -> ToolResult:
    if not success and error is None:
        error = "execution failed"

    return ToolResult(
        action_id=action.action_id,
        tool_name=action.tool_name,
        success=success,
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
) -> AgentLoop:
    return AgentLoop(
        perception_engine=perception_engine,
        grounder=grounder,
        action_grounder=action_grounder,
        executor=executor,
        verifier=verifier,
        recovery=recovery,
    )


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
        "Action(",
        '"x"',
        '"y"',
    )

    assert module.AgentLoop is AgentLoop
    assert module.AgentLoopResult is AgentLoopResult
    assert module.AgentLoopStatus is AgentLoopStatus
    assert all(term not in source for term in forbidden_terms)
