"""Controller connecting the desktop workspace to TaskRuntime."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread

from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeStatus,
    RuntimeTask,
    RuntimeWorker,
    TaskRuntime,
)
from computer_agent.task import (
    TaskState,
    TaskStateSnapshot,
)


LegacyWorkerFactory = Callable[[], RuntimeWorker]
SemanticWorkerFactory = Callable[
    [TaskState, Callable[[], None]],
    RuntimeWorker,
]
TaskStateListener = Callable[
    [TaskStateSnapshot],
    None,
]


class WorkspaceController:
    """Own one interactive runtime session for the desktop workspace."""

    def __init__(
        self,
        *,
        event_listener: Callable[[RuntimeEvent], None],
        worker_factory: LegacyWorkerFactory | None = None,
        semantic_worker_factory: SemanticWorkerFactory | None = None,
        task_state_listener: TaskStateListener | None = None,
    ) -> None:
        if not callable(event_listener):
            raise ValueError(
                "event_listener must be callable"
            )

        if worker_factory is None and semantic_worker_factory is None:
            raise ValueError(
                "a worker factory is required"
            )

        if (
            worker_factory is not None
            and semantic_worker_factory is not None
        ):
            raise ValueError(
                "provide only one worker factory"
            )

        if (
            worker_factory is not None
            and not callable(worker_factory)
        ):
            raise ValueError(
                "worker_factory must be callable"
            )

        if (
            semantic_worker_factory is not None
            and not callable(semantic_worker_factory)
        ):
            raise ValueError(
                "semantic_worker_factory must be callable"
            )

        if (
            task_state_listener is not None
            and not callable(task_state_listener)
        ):
            raise ValueError(
                "task_state_listener must be callable"
            )

        self._worker_factory = worker_factory
        self._semantic_worker_factory = (
            semantic_worker_factory
        )
        self._event_listener = event_listener
        self._task_state_listener = (
            task_state_listener
        )

        self._runtime: TaskRuntime | None = None
        self._task_state: TaskState | None = None
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
    def task_state(self) -> TaskState | None:
        with self._lock:
            return self._task_state

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
            raise ValueError(
                "goal must be a non-empty string"
            )

        with self._lock:
            if self._runtime is not None:
                status = self._runtime.task.status

                if status in (
                    RuntimeStatus.CREATED,
                    RuntimeStatus.RUNNING,
                    RuntimeStatus.PAUSED,
                ):
                    raise RuntimeError(
                        "cannot start a new task while "
                        "another task is active"
                    )

            task = RuntimeTask(goal=goal)
            task_state = TaskState(
                goal=goal,
                task_id=task.task_id,
            )

            event_bus = RuntimeEventBus()
            event_bus.subscribe(
                self._event_listener
            )

            if (
                self._semantic_worker_factory
                is not None
            ):
                worker = (
                    self._semantic_worker_factory(
                        task_state,
                        lambda: self._publish_task_state(
                            task_state
                        ),
                    )
                )
            else:
                assert self._worker_factory is not None
                worker = self._worker_factory()

            runtime = TaskRuntime(
                task=task,
                worker=worker,
                event_bus=event_bus,
            )

            thread = Thread(
                target=runtime.run,
                name=(
                    "computer-agent-task-"
                    f"{task.task_id}"
                ),
                daemon=True,
            )

            self._runtime = runtime
            self._task_state = task_state
            self._thread = thread

            self._publish_task_state(
                task_state
            )

            thread.start()

            return task

    def pause(self) -> bool:
        runtime = self.runtime
        return (
            False
            if runtime is None
            else runtime.pause()
        )

    def resume(self) -> bool:
        runtime = self.runtime
        return (
            False
            if runtime is None
            else runtime.resume()
        )

    def stop(self) -> bool:
        runtime = self.runtime
        return (
            False
            if runtime is None
            else runtime.stop()
        )

    def wait(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait for the current worker thread to finish."""

        with self._lock:
            thread = self._thread

        if thread is None:
            return True

        thread.join(timeout=timeout)
        return not thread.is_alive()

    def _publish_task_state(
        self,
        state: TaskState,
    ) -> None:
        listener = self._task_state_listener

        if listener is None:
            return

        listener(
            TaskStateSnapshot.from_state(state)
        )


def create_workspace_demo_worker() -> RuntimeWorker:
    """Return the original bounded Phase 06.02 demo worker."""

    def worker(
        task,
        control,
        progress,
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
