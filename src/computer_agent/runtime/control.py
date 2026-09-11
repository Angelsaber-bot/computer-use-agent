"""Cooperative pause, resume, and stop control for task execution."""

from __future__ import annotations

from threading import Condition


class RuntimeStopRequested(RuntimeError):
    """Raised at a safe checkpoint after a stop request."""


class RuntimeControl:
    """Thread-safe cooperative control for one running task."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._paused = False
        self._stop_requested = False

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def stop_requested(self) -> bool:
        with self._condition:
            return self._stop_requested

    def pause(self) -> bool:
        """Pause future progress.

        Return True only when this call changed the state.
        """
        with self._condition:
            if self._stop_requested or self._paused:
                return False

            self._paused = True
            return True

    def resume(self) -> bool:
        """Resume progress and wake blocked checkpoints."""
        with self._condition:
            if self._stop_requested or not self._paused:
                return False

            self._paused = False
            self._condition.notify_all()
            return True

    def request_stop(self) -> bool:
        """Request clean termination and wake paused checkpoints."""
        with self._condition:
            if self._stop_requested:
                return False

            self._stop_requested = True
            self._paused = False
            self._condition.notify_all()
            return True

    def checkpoint(self) -> None:
        """Block while paused and raise when stop has been requested."""
        with self._condition:
            while self._paused and not self._stop_requested:
                self._condition.wait()

            if self._stop_requested:
                raise RuntimeStopRequested("runtime stop requested")
