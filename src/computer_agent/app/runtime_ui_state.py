"""Shared runtime UI state derived from public runtime events."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import time

from computer_agent.runtime import RuntimeEvent, RuntimeEventType


@dataclass(frozen=True, slots=True)
class RuntimeUIState:
    status: str = "Idle"
    current: str = "No active task."
    step_current: int | None = None
    step_total: int | None = None
    verified_steps: int = 0
    started_at: float | None = None
    finished: bool = False
    latest_timing: str | None = None

    def elapsed(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        if now is None:
            now = time.monotonic()
        return max(0.0, now - self.started_at)


class RuntimeUIStateTracker:
    """Parse runtime events into display state for Qt surfaces."""

    def __init__(
        self,
        *,
        clock=time.monotonic,
    ) -> None:
        self._clock = clock
        self._state = RuntimeUIState()

    @property
    def state(self) -> RuntimeUIState:
        return self._state

    def handle_event(
        self,
        event: RuntimeEvent,
    ) -> RuntimeUIState:
        if not isinstance(event, RuntimeEvent):
            raise ValueError("event must be a RuntimeEvent")

        state = self._state
        if event.event_type is RuntimeEventType.TASK_STARTED:
            state = RuntimeUIState(
                status="Running",
                current="Task runtime started.",
                started_at=self._clock(),
            )
        elif event.event_type is RuntimeEventType.PROGRESS:
            state = self._handle_progress(state, event.message)
        elif event.event_type is RuntimeEventType.TASK_PAUSED:
            state = replace(
                state,
                status="Paused",
                current="Execution paused at a safe checkpoint.",
            )
        elif event.event_type is RuntimeEventType.TASK_RESUMED:
            state = replace(
                state,
                status="Running",
                current="Execution resumed.",
            )
        elif event.event_type is RuntimeEventType.STOP_REQUESTED:
            state = replace(
                state,
                status="Stopping",
                current="Waiting for the next safe checkpoint.",
            )
        elif event.event_type is RuntimeEventType.TASK_STOPPED:
            state = replace(
                state,
                status="Stopped",
                current="Task stopped safely.",
                finished=True,
            )
        elif event.event_type is RuntimeEventType.TASK_COMPLETED:
            state = replace(
                state,
                status="Completed",
                current="Task runtime completed.",
                verified_steps=state.step_total or state.verified_steps,
                finished=True,
            )
        elif event.event_type is RuntimeEventType.TASK_FAILED:
            error = event.data.get("error")
            state = replace(
                state,
                status="Failed",
                current=error
                if isinstance(error, str) and error.strip()
                else "Task runtime failed.",
                finished=True,
            )

        self._state = state
        return state

    def _handle_progress(
        self,
        state: RuntimeUIState,
        message: str,
    ) -> RuntimeUIState:
        if message.startswith("Timing: "):
            return replace(
                state,
                latest_timing=message.removeprefix("Timing: ").strip(),
            )

        plan_match = re.search(
            r"Plan ready:\s+(\d+) durable step",
            message,
            re.IGNORECASE,
        )
        if plan_match is not None:
            total = int(plan_match.group(1))
            return replace(
                state,
                current=message,
                step_current=None,
                step_total=total if total > 0 else None,
                verified_steps=0,
            )

        step_match = re.search(
            r"Step\s+(\d+)\s*/\s*(\d+)",
            message,
            re.IGNORECASE,
        )
        if step_match is None:
            return replace(
                state,
                current=message,
            )

        current = int(step_match.group(1))
        total = int(step_match.group(2))
        verified = "VERIFIED" in message.upper()
        return replace(
            state,
            current=message,
            step_current=current,
            step_total=total,
            verified_steps=min(
                current if verified else current - 1,
                total,
            ),
        )
