"""Interactive lifecycle wrapper for long-running agent tasks."""

from __future__ import annotations

from collections.abc import Callable

from computer_agent.runtime.control import (
    RuntimeControl,
    RuntimeStopRequested,
)
from computer_agent.runtime.events import RuntimeEventBus
from computer_agent.runtime.models import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
)


RuntimeWorker = Callable[
    [RuntimeTask, RuntimeControl, Callable[[str], None]],
    None,
]


class TaskRuntime:
    """Own lifecycle, control, and observable events for one task."""

    def __init__(
        self,
        *,
        task: RuntimeTask,
        worker: RuntimeWorker,
        event_bus: RuntimeEventBus | None = None,
    ) -> None:
        if not isinstance(task, RuntimeTask):
            raise ValueError("task must be a RuntimeTask")

        if not callable(worker):
            raise ValueError("worker must be callable")

        self._task = task
        self._worker = worker
        self._control = RuntimeControl()
        self._event_bus = event_bus or RuntimeEventBus()

    @property
    def task(self) -> RuntimeTask:
        return self._task

    @property
    def control(self) -> RuntimeControl:
        return self._control

    @property
    def event_bus(self) -> RuntimeEventBus:
        return self._event_bus

    def run(self) -> None:
        """Run the worker synchronously until a terminal outcome."""
        if self._task.status is not RuntimeStatus.CREATED:
            raise RuntimeError(
                f"cannot run task with status {self._task.status.value}"
            )

        self._task.set_status(RuntimeStatus.RUNNING)
        self._emit(
            RuntimeEventType.TASK_STARTED,
            "Task runtime started.",
        )

        try:
            self._worker(
                self._task,
                self._control,
                self._publish_progress,
            )
        except RuntimeStopRequested:
            self._task.set_status(RuntimeStatus.STOPPED)
            self._emit(
                RuntimeEventType.TASK_STOPPED,
                "Task stopped at a safe runtime checkpoint.",
            )
            return
        except Exception as error:
            self._task.last_error = str(error)
            self._task.set_status(RuntimeStatus.FAILED)
            self._emit(
                RuntimeEventType.TASK_FAILED,
                "Task runtime failed.",
                error=str(error),
            )
            return

        self._task.set_status(RuntimeStatus.COMPLETED)
        self._emit(
            RuntimeEventType.TASK_COMPLETED,
            "Task runtime completed.",
        )

    def pause(self) -> bool:
        """Request cooperative pause."""
        if self._task.status is not RuntimeStatus.RUNNING:
            return False

        if not self._control.pause():
            return False

        self._task.set_status(RuntimeStatus.PAUSED)
        self._emit(
            RuntimeEventType.TASK_PAUSED,
            "Task runtime paused.",
        )
        return True

    def resume(self) -> bool:
        """Resume cooperative execution."""
        if self._task.status is not RuntimeStatus.PAUSED:
            return False

        if not self._control.resume():
            return False

        self._task.set_status(RuntimeStatus.RUNNING)
        self._emit(
            RuntimeEventType.TASK_RESUMED,
            "Task runtime resumed.",
        )
        return True

    def stop(self) -> bool:
        """Request clean termination at the next safe checkpoint."""
        if self._task.status not in (
            RuntimeStatus.RUNNING,
            RuntimeStatus.PAUSED,
        ):
            return False

        if not self._control.request_stop():
            return False

        self._emit(
            RuntimeEventType.STOP_REQUESTED,
            "Task stop requested.",
        )
        return True

    def _publish_progress(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("progress message must be a non-empty string")

        self._emit(
            RuntimeEventType.PROGRESS,
            message,
        )

    def _emit(
        self,
        event_type: RuntimeEventType,
        message: str,
        **data: object,
    ) -> None:
        self._event_bus.publish(
            RuntimeEvent(
                event_type=event_type,
                task_id=self._task.task_id,
                message=message,
                data=dict(data),
            )
        )
