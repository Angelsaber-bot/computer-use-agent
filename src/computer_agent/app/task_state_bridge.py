"""Qt bridge for immutable semantic task-state snapshots."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from computer_agent.task import TaskStateSnapshot


class TaskStateBridge(QObject):
    """Marshal TaskState snapshots into the Qt event loop."""

    snapshot_received = Signal(object)

    def publish(
        self,
        snapshot: TaskStateSnapshot,
    ) -> None:
        if not isinstance(snapshot, TaskStateSnapshot):
            raise ValueError(
                "snapshot must be a TaskStateSnapshot"
            )

        self.snapshot_received.emit(snapshot)
