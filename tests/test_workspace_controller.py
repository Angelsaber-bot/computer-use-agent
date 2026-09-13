"""Tests for the Agent Workspace runtime controller."""

from __future__ import annotations

from threading import Event

import pytest

from computer_agent.app.workspace_controller import WorkspaceController
from computer_agent.runtime import (
    RuntimeControl,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
)


def test_workspace_controller_starts_runtime_and_forwards_events() -> None:
    events: list[RuntimeEvent] = []

    def worker_factory():
        def worker(task, control, progress) -> None:
            progress("Workspace worker executed.")

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=events.append,
    )

    task = controller.start("Complete workspace task")

    assert controller.wait(timeout=1.0)
    assert task.status is RuntimeStatus.COMPLETED

    assert tuple(
        event.event_type
        for event in events
    ) == (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    )


def test_workspace_controller_pause_resume_controls_real_runtime() -> None:
    events: list[RuntimeEvent] = []

    ready = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker_factory():
        def worker(
            task: RuntimeTask,
            control: RuntimeControl,
            progress,
        ) -> None:
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError("checkpoint release timeout")

            control.checkpoint()

            passed_checkpoint.set()

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=events.append,
    )

    task = controller.start("Pause workspace task")

    assert ready.wait(timeout=1.0)

    assert controller.pause() is True
    assert task.status is RuntimeStatus.PAUSED

    allow_checkpoint.set()

    assert passed_checkpoint.wait(timeout=0.05) is False

    assert controller.resume() is True

    assert passed_checkpoint.wait(timeout=1.0)
    assert controller.wait(timeout=1.0)

    assert task.status is RuntimeStatus.COMPLETED

    assert RuntimeEventType.TASK_PAUSED in tuple(
        event.event_type
        for event in events
    )

    assert RuntimeEventType.TASK_RESUMED in tuple(
        event.event_type
        for event in events
    )


def test_workspace_controller_stop_controls_real_runtime() -> None:
    ready = Event()
    allow_checkpoint = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            ready.set()

            if not allow_checkpoint.wait(timeout=1.0):
                raise RuntimeError("checkpoint release timeout")

            control.checkpoint()

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=lambda event: None,
    )

    task = controller.start("Stop workspace task")

    assert ready.wait(timeout=1.0)

    assert controller.pause() is True

    allow_checkpoint.set()

    assert controller.stop() is True

    assert controller.wait(timeout=1.0)
    assert task.status is RuntimeStatus.STOPPED


def test_workspace_controller_rejects_empty_goal() -> None:
    controller = WorkspaceController(
        worker_factory=lambda: lambda task, control, progress: None,
        event_listener=lambda event: None,
    )

    with pytest.raises(
        ValueError,
        match="goal must be a non-empty string",
    ):
        controller.start("   ")


def test_workspace_controller_rejects_second_active_task() -> None:
    release = Event()

    def worker_factory():
        def worker(task, control, progress) -> None:
            while not release.is_set():
                control.checkpoint()
                release.wait(timeout=0.01)

        return worker

    controller = WorkspaceController(
        worker_factory=worker_factory,
        event_listener=lambda event: None,
    )

    controller.start("First task")

    with pytest.raises(
        RuntimeError,
        match="cannot start a new task",
    ):
        controller.start("Second task")

    release.set()

    assert controller.wait(timeout=1.0)


def test_workspace_controller_allows_new_task_after_completion() -> None:
    controller = WorkspaceController(
        worker_factory=lambda: lambda task, control, progress: None,
        event_listener=lambda event: None,
    )

    first = controller.start("First task")
    assert controller.wait(timeout=1.0)
    assert first.status is RuntimeStatus.COMPLETED

    second = controller.start("Second task")
    assert controller.wait(timeout=1.0)
    assert second.status is RuntimeStatus.COMPLETED

    assert first.task_id != second.task_id


def test_workspace_controller_persists_semantic_state(
    tmp_path,
) -> None:
    from computer_agent.task import (
        TaskStateStatus,
        TaskStateStore,
    )

    store = TaskStateStore(
        tmp_path
    )

    def semantic_factory(
        state,
        publish_state,
        publish_decision,
    ):
        def worker(
            task,
            control,
            progress,
        ) -> None:
            state.status = (
                TaskStateStatus.RUNNING
            )
            state.touch()
            publish_state()

        return worker

    controller = WorkspaceController(
        semantic_worker_factory=(
            semantic_factory
        ),
        event_listener=lambda event: None,
        task_store=store,
    )

    task = controller.start(
        "Persist workspace task"
    )

    assert controller.wait(
        timeout=1.0
    )

    assert store.exists(
        task.task_id
    )

    restored = store.load(
        task.task_id
    )

    assert (
        restored.task_id
        == task.task_id
    )
    assert (
        restored.goal
        == "Persist workspace task"
    )
    assert (
        restored.status
        is TaskStateStatus.RUNNING
    )


def test_workspace_controller_restores_semantic_task(
    tmp_path,
) -> None:
    from threading import Event

    from computer_agent.task import (
        ClaimRecord,
        ClaimStatus,
        EvidenceFreshness,
        EvidenceKind,
        EvidenceRecord,
        SubgoalRecord,
        SubgoalStatus,
        TaskState,
        TaskStateStatus,
        TaskStateStore,
    )

    store = TaskStateStore(
        tmp_path
    )

    evidence = EvidenceRecord(
        evidence_id="evidence-live-browser",
        summary=(
            "The current browser contains "
            "the intended search query."
        ),
        source="Live browser observation",
        kind=EvidenceKind.VERIFICATION,
    )

    claim = ClaimRecord(
        claim_id="claim-query-entered",
        statement=(
            "The intended search query "
            "has been entered."
        ),
        status=ClaimStatus.VERIFIED,
        evidence_ids=(
            evidence.evidence_id,
        ),
    )

    subgoal = SubgoalRecord(
        subgoal_id="subgoal-enter-query",
        description=(
            "Enter the intended search query."
        ),
        status=SubgoalStatus.VERIFIED,
        claim_ids=(
            claim.claim_id,
        ),
    )

    persisted = TaskState(
        goal=(
            "Search python.org and continue "
            "after restart."
        ),
        task_id="workspace-persisted-task",
        status=TaskStateStatus.PAUSED,
        evidence={
            evidence.evidence_id: evidence,
        },
        claims={
            claim.claim_id: claim,
        },
        subgoals={
            subgoal.subgoal_id: subgoal,
        },
    )

    store.save(
        persisted
    )

    release = Event()

    def semantic_factory(
        state,
        publish_state,
        publish_decision,
    ):
        def worker(
            task,
            control,
            progress,
        ) -> None:
            if not release.wait(
                timeout=1.0
            ):
                raise RuntimeError(
                    "restore worker release timeout"
                )

        return worker

    controller = WorkspaceController(
        semantic_worker_factory=(
            semantic_factory
        ),
        event_listener=lambda event: None,
        task_store=store,
    )

    task = controller.restore_task(
        persisted.task_id
    )

    restored = controller.task_state

    assert restored is not None
    assert restored is not persisted

    assert (
        task.task_id
        == persisted.task_id
    )
    assert (
        restored.task_id
        == persisted.task_id
    )
    assert (
        restored.goal
        == persisted.goal
    )
    assert (
        restored.status
        is TaskStateStatus.PAUSED
    )

    assert (
        restored.evidence[
            evidence.evidence_id
        ].freshness
        is EvidenceFreshness.STALE
    )

    assert (
        restored.claims[
            claim.claim_id
        ].status
        is ClaimStatus.UNKNOWN
    )

    assert (
        restored.subgoals[
            subgoal.subgoal_id
        ].status
        is SubgoalStatus.UNKNOWN
    )

    release.set()

    assert controller.wait(
        timeout=1.0
    )

    checkpoint = store.load(
        persisted.task_id
    )

    assert (
        checkpoint.evidence[
            evidence.evidence_id
        ].freshness
        is EvidenceFreshness.STALE
    )


def test_workspace_controller_restore_requires_persistence() -> None:
    controller = WorkspaceController(
        semantic_worker_factory=(
            lambda state, publish_state, publish_decision:
            lambda task, control, progress: None
        ),
        event_listener=lambda event: None,
    )

    with pytest.raises(
        RuntimeError,
        match="task persistence is not configured",
    ):
        controller.restore_task(
            "missing-task"
        )
