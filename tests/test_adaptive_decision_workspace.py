"""Tests for the integrated adaptive-decision workspace UI."""

from __future__ import annotations

import os
import time
from threading import Event, get_ident

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen",
)

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPlainTextEdit,
    QTabWidget,
)

from computer_agent.app.adaptive_decision_bridge import (
    AdaptiveDecisionBridge,
)
from computer_agent.app.adaptive_decision_panel import (
    AdaptiveDecisionPanel,
)
from computer_agent.app.evidence_demo import (
    create_evidence_demo_worker,
)
from computer_agent.app.main_window import MainWindow
from computer_agent.app.workspace_controller import (
    WorkspaceController,
)
from computer_agent.grounding import TargetSpec
from computer_agent.planning import (
    PlanOperation,
    PlanStep,
)
from computer_agent.reasoning import (
    AdaptiveDecisionOutcome,
    AdaptiveDecisionSnapshot,
    AdaptiveReasoningContext,
    NextStepDecision,
    NextStepDecisionType,
    NextStepReasoningResult,
    NextStepReasoningStatus,
    ObservationContext,
)
from computer_agent.runtime import (
    RuntimeControl,
    RuntimeStatus,
    RuntimeTask,
)
from computer_agent.task import TaskState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    yield app


def _wait_until(
    app: QApplication,
    predicate,
    *,
    timeout: float = 1.0,
) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        app.processEvents()

        if predicate():
            return True

        time.sleep(0.005)

    app.processEvents()
    return bool(predicate())


def _context() -> AdaptiveReasoningContext:
    return AdaptiveReasoningContext(
        goal="Submit registration exactly once.",
        constraints=(
            "Do not submit twice.",
        ),
        task_status="running",
        verified_subgoals=(),
        unresolved_subgoals=(
            "pending: Submit registration.",
        ),
        current_evidence=(),
        stale_or_unknown_evidence=(),
        unresolved_side_effects=(),
        pending_questions=(),
        completion_allowed=False,
        completion_blockers=(
            "subgoal is not verified: Submit registration.",
        ),
        observation=ObservationContext(
            application_name="Google Chrome",
            window_title="Registration",
            visible_text=(
                "Submit",
                "Check status",
                "Submission status",
            ),
        ),
        blocked_action_keys=(
            "click_target:submit",
        ),
    )


def _action_decision(
    target: str = "Submit",
) -> NextStepDecision:
    return NextStepDecision(
        decision_type=NextStepDecisionType.ACTION,
        action=PlanStep(
            goal=f"Click {target}.",
            operation=PlanOperation.CLICK_TARGET,
            action_target=TargetSpec(
                text=target,
                element_types=("button",),
            ),
            verification_target=TargetSpec(
                text="Submission status",
            ),
            max_attempts=1,
        ),
        expected_effect=(
            "The submission state should change."
        ),
    )


def _snapshot_for(
    result: NextStepReasoningResult,
    *,
    attempts: tuple[
        NextStepReasoningResult,
        ...,
    ]
    | None = None,
    safety_replan_used: bool = False,
) -> AdaptiveDecisionSnapshot:
    attempt_results = attempts or (result,)

    return AdaptiveDecisionSnapshot.from_outcome(
        context=_context(),
        outcome=AdaptiveDecisionOutcome(
            result=result,
            attempts=len(attempt_results),
            safety_replan_used=safety_replan_used,
            attempt_results=attempt_results,
        ),
    )


def test_adaptive_decision_snapshot_represents_decision_types() -> None:
    action = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=_action_decision(),
            reason="validated next-step decision ready",
        )
    )
    assert action.final_decision_type == "ACTION"
    assert action.target_text == "Submit"

    ask_user = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=NextStepDecision(
                decision_type=(
                    NextStepDecisionType.ASK_USER
                ),
                question="Should I continue?",
            ),
            reason="validated next-step decision ready",
        )
    )
    assert ask_user.final_decision_type == "ASK_USER"
    assert ask_user.question == "Should I continue?"

    complete = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=NextStepDecision(
                decision_type=(
                    NextStepDecisionType.COMPLETE
                ),
                completion_summary="Confirmed once.",
            ),
            reason="validated next-step decision ready",
        )
    )
    assert complete.final_decision_type == "COMPLETE"
    assert (
        complete.completion_summary
        == "Confirmed once."
    )

    blocked = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.BLOCKED,
            decision=None,
            reason="validation blocked the decision",
        )
    )
    assert blocked.final_decision_type == "BLOCKED"
    assert blocked.final_status == "BLOCKED"
    assert (
        blocked.blocked_reason
        == "validation blocked the decision"
    )


def test_adaptive_decision_bridge_delivers_snapshot(
    qapp,
) -> None:
    snapshot = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=_action_decision(),
            reason="validated next-step decision ready",
        )
    )

    bridge = AdaptiveDecisionBridge()
    received = []

    bridge.snapshot_received.connect(
        received.append
    )

    bridge.publish(snapshot)
    qapp.processEvents()

    assert received == [snapshot]


def test_adaptive_decision_panel_renders_reasoning_snapshot(
    qapp,
) -> None:
    rejected = NextStepReasoningResult(
        status=NextStepReasoningStatus.BLOCKED,
        decision=None,
        reason=(
            "model proposed an unsafe retry of an "
            "unresolved non-idempotent side effect"
        ),
        rejected_decision=_action_decision("Submit"),
    )
    accepted = NextStepReasoningResult(
        status=NextStepReasoningStatus.READY,
        decision=_action_decision("Check status"),
        reason="validated next-step decision ready",
    )
    snapshot = _snapshot_for(
        accepted,
        attempts=(rejected, accepted),
        safety_replan_used=True,
    )

    panel = AdaptiveDecisionPanel()
    panel.render(snapshot)

    observation = panel.findChild(
        QPlainTextEdit,
        "observationText",
    )
    attempts = panel.findChild(
        QPlainTextEdit,
        "decisionAttempts",
    )
    attempt_count = panel.findChild(
        QLabel,
        "attemptCount",
    )
    replan = panel.findChild(
        QLabel,
        "safetyReplan",
    )
    blocked = panel.findChild(
        QPlainTextEdit,
        "blockedActionKeys",
    )
    final_target = panel.findChild(
        QLabel,
        "finalTarget",
    )

    assert observation is not None
    assert attempts is not None
    assert attempt_count is not None
    assert replan is not None
    assert blocked is not None
    assert final_target is not None

    assert "Submit" in observation.toPlainText()
    assert attempt_count.text() == "2"
    assert replan.text() == "YES"
    assert (
        "click_target:submit"
        in blocked.toPlainText()
    )
    assert (
        "ACTION -> Submit  REJECTED"
        in attempts.toPlainText()
    )
    assert (
        "ACTION -> Check status  ACCEPTED"
        in attempts.toPlainText()
    )
    assert final_target.text() == "Check status"


def test_main_window_contains_adaptive_decision_tab(
    qapp,
) -> None:
    window = MainWindow(
        worker_factory=lambda: (
            lambda task, control, progress: None
        )
    )

    tabs = window.findChild(
        QTabWidget,
        "workspaceTabs",
    )

    assert tabs is not None
    assert "Adaptive Decision" in {
        tabs.tabText(index)
        for index in range(tabs.count())
    }

    window.close()


def test_workspace_controller_publishes_task_and_decision_updates() -> None:
    task_snapshots = []
    decision_snapshots = []

    expected = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=_action_decision(),
            reason="validated next-step decision ready",
        )
    )

    def semantic_factory(
        state,
        publish_state,
        publish_decision,
    ):
        def worker(task, control, progress) -> None:
            publish_state()
            publish_decision(expected)

        return worker

    controller = WorkspaceController(
        semantic_worker_factory=semantic_factory,
        event_listener=lambda event: None,
        task_state_listener=task_snapshots.append,
        adaptive_decision_listener=(
            decision_snapshots.append
        ),
    )

    controller.start("Run semantic worker")

    assert controller.wait(timeout=1.0)
    assert len(task_snapshots) >= 1
    assert decision_snapshots == [expected]


def test_main_window_uses_queued_delivery_for_worker_snapshots(
    qapp,
) -> None:
    expected = _snapshot_for(
        NextStepReasoningResult(
            status=NextStepReasoningStatus.READY,
            decision=_action_decision(),
            reason="validated next-step decision ready",
        )
    )
    worker_started = Event()
    worker_thread_ids: list[int] = []
    render_threads: list[QThread] = []

    def semantic_factory(
        state,
        publish_state,
        publish_decision,
    ):
        def worker(task, control, progress) -> None:
            worker_thread_ids.append(get_ident())
            publish_decision(expected)
            worker_started.set()

        return worker

    window = MainWindow(
        semantic_worker_factory=semantic_factory
    )

    original_render = (
        window._adaptive_decision_panel.render
    )

    def record_render(snapshot) -> None:
        render_threads.append(
            QThread.currentThread()
        )
        original_render(snapshot)

    window._adaptive_decision_panel.render = (
        record_render
    )

    task_input = window.findChild(
        QPlainTextEdit,
        "taskInput",
    )

    assert task_input is not None
    task_input.setPlainText("Run queued delivery task")

    window._start_task()

    assert worker_started.wait(timeout=1.0)
    assert _wait_until(
        qapp,
        lambda: bool(render_threads),
    )

    assert worker_thread_ids[0] != get_ident()
    assert render_threads[0] == qapp.thread()

    window.close()


def test_integrated_demo_runs_three_stage_adaptive_flow() -> None:
    state = TaskState(
        goal=(
            "Submit registration exactly once "
            "and confirm result."
        )
    )
    task = RuntimeTask(goal=state.goal)
    control = RuntimeControl()
    progress_messages: list[str] = []
    snapshots: list[AdaptiveDecisionSnapshot] = []

    worker = create_evidence_demo_worker(
        state,
        lambda: None,
        snapshots.append,
        stage_delay=0.0,
    )

    worker(
        task,
        control,
        progress_messages.append,
    )

    assert len(snapshots) == 3

    first, second, third = snapshots

    assert first.target_text == "Submit"
    assert first.final_status == "ACCEPTED"
    assert first.model_attempts == 1

    assert (
        second.observation_text
        == first.observation_text
    )
    assert second.model_attempts == 2
    assert second.safety_replan_used is True
    assert second.attempts[0].target_text == "Submit"
    assert second.attempts[0].result == "REJECTED"
    assert (
        second.attempts[1].target_text
        == "Check status"
    )
    assert second.attempts[1].result == "ACCEPTED"
    assert (
        second.blocked_action_keys
        == ("click_target:submit",)
    )
    assert second.target_text == "Check status"

    accepted_stage_two_submits = [
        attempt
        for attempt in second.attempts
        if attempt.target_text == "Submit"
        and attempt.result == "ACCEPTED"
    ]
    assert accepted_stage_two_submits == []

    assert third.final_decision_type == "COMPLETE"
    assert third.final_status == "ACCEPTED"
    assert third.model_attempts == 1

    assert max(
        snapshot.model_attempts
        for snapshot in snapshots
    ) <= 2
    assert (
        "Duplicate Submit executed = 0."
        in progress_messages
    )


def test_integrated_demo_pause_resume_and_stop() -> None:
    def semantic_factory(
        state,
        publish_state,
        publish_decision,
    ):
        return create_evidence_demo_worker(
            state,
            publish_state,
            publish_decision,
            stage_delay=0.2,
        )

    controller = WorkspaceController(
        semantic_worker_factory=semantic_factory,
        event_listener=lambda event: None,
    )

    task = controller.start("Run integrated pause demo")
    time.sleep(0.05)

    assert controller.pause() is True
    assert task.status is RuntimeStatus.PAUSED
    assert controller.resume() is True
    assert controller.wait(timeout=3.0)
    assert task.status is RuntimeStatus.COMPLETED

    stop_controller = WorkspaceController(
        semantic_worker_factory=semantic_factory,
        event_listener=lambda event: None,
    )

    stop_task = stop_controller.start(
        "Run integrated stop demo"
    )
    time.sleep(0.05)

    assert stop_controller.stop() is True
    assert stop_controller.wait(timeout=3.0)
    assert stop_task.status is RuntimeStatus.STOPPED
