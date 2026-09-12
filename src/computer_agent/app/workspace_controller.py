"""Controller connecting the desktop workspace to TaskRuntime."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread

from computer_agent.runtime import (
    RuntimeControl,
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeStatus,
    RuntimeTask,
    RuntimeWorker,
    TaskRuntime,
)


class WorkspaceController:
    """Own one interactive runtime session for the desktop workspace."""

    def __init__(
        self,
        *,
        worker_factory: Callable[[], RuntimeWorker],
        event_listener: Callable[[RuntimeEvent], None],
    ) -> None:
        if not callable(worker_factory):
            raise ValueError("worker_factory must be callable")

        if not callable(event_listener):
            raise ValueError("event_listener must be callable")

        self._worker_factory = worker_factory
        self._event_listener = event_listener

        self._runtime: TaskRuntime | None = None
        self._thread: Thread | None = None
        self._lock = Lock()

    @property
    def runtime(self) -> TaskRuntime | None:
        with self._lock:
            return self._runtime

    @property
    def task(self) -> RuntimeTask | None:
        runtime = self.runtime
        return None if runtime is None else runtime.task

    @property
    def running(self) -> bool:
        task = self.task

        return (
            task is not None
            and task.status
            in (
                RuntimeStatus.RUNNING,
                RuntimeStatus.PAUSED,
            )
        )

    def start(self, goal: str) -> RuntimeTask:
        """Create and start one interactive task."""
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        with self._lock:
            if self._runtime is not None:
                status = self._runtime.task.status

                if status in (
                    RuntimeStatus.RUNNING,
                    RuntimeStatus.PAUSED,
                ):
                    raise RuntimeError(
                        "cannot start a new task while another task is active"
                    )

            task = RuntimeTask(goal=goal)

            event_bus = RuntimeEventBus()
            event_bus.subscribe(self._event_listener)

            runtime = TaskRuntime(
                task=task,
                worker=self._worker_factory(),
                event_bus=event_bus,
            )

            thread = Thread(
                target=runtime.run,
                name=f"computer-agent-task-{task.task_id}",
                daemon=True,
            )

            self._runtime = runtime
            self._thread = thread

            thread.start()

            return task

    def pause(self) -> bool:
        runtime = self.runtime
        return False if runtime is None else runtime.pause()

    def resume(self) -> bool:
        runtime = self.runtime
        return False if runtime is None else runtime.resume()

    def stop(self) -> bool:
        runtime = self.runtime
        return False if runtime is None else runtime.stop()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current worker thread to finish."""
        with self._lock:
            thread = self._thread

        if thread is None:
            return True

        thread.join(timeout=timeout)
        return not thread.is_alive()


def create_workspace_demo_worker() -> RuntimeWorker:
    """Return a bounded deterministic worker for Workspace integration."""

    def worker(
        task: RuntimeTask,
        control: RuntimeControl,
        progress: Callable[[str], None],
    ) -> None:
        stages = (
            "Preparing task.",
            "Observing environment.",
            "Processing current state.",
            "Preparing next action.",
            "Verifying progress.",
            "Finishing task.",
        )

        import time

        for stage in stages:
            control.checkpoint()
            progress(stage)
            time.sleep(0.6)

            control.checkpoint()

    return worker
