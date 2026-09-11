"""Event publishing helpers for the interactive task runtime."""

from __future__ import annotations

from collections.abc import Callable

from computer_agent.runtime.models import RuntimeEvent


RuntimeEventListener = Callable[[RuntimeEvent], None]


class RuntimeEventBus:
    """Small synchronous event publisher for runtime observations."""

    def __init__(self) -> None:
        self._listeners: list[RuntimeEventListener] = []

    def subscribe(self, listener: RuntimeEventListener) -> None:
        if not callable(listener):
            raise ValueError("listener must be callable")

        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: RuntimeEventListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def publish(self, event: RuntimeEvent) -> None:
        if not isinstance(event, RuntimeEvent):
            raise ValueError("event must be a RuntimeEvent")

        for listener in tuple(self._listeners):
            listener(event)
