"""Thread-safe Qt bridge for interactive runtime events."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from computer_agent.runtime import RuntimeEvent


class RuntimeEventBridge(QObject):
    """Forward runtime events safely into the Qt main thread."""

    event_received = Signal(object)

    def publish(self, event: RuntimeEvent) -> None:
        """Receive a runtime event from any thread and emit a Qt signal."""
        if not isinstance(event, RuntimeEvent):
            raise ValueError("event must be a RuntimeEvent")

        self.event_received.emit(event)
