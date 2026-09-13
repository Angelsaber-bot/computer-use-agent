"""Qt bridge for immutable adaptive-decision snapshots."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from computer_agent.reasoning import AdaptiveDecisionSnapshot


class AdaptiveDecisionBridge(QObject):
    """Marshal adaptive-decision snapshots into the Qt event loop."""

    snapshot_received = Signal(object)

    def publish(
        self,
        snapshot: AdaptiveDecisionSnapshot,
    ) -> None:
        if not isinstance(
            snapshot,
            AdaptiveDecisionSnapshot,
        ):
            raise ValueError(
                "snapshot must be an AdaptiveDecisionSnapshot"
            )

        self.snapshot_received.emit(snapshot)
