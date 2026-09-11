"""Phase 06 Experiment 01: deterministic interactive task runtime."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread

from computer_agent.runtime import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeEventType,
    RuntimeStatus,
    RuntimeTask,
    TaskRuntime,
)


TITLE = "Phase 06 Experiment 01: Interactive Task Runtime"

WAIT_TIMEOUT_SECONDS = 1.0
PAUSE_PROBE_SECONDS = 0.05

RESUME_TASK_GOAL = "Demonstrate deterministic pause and resume."
STOP_TASK_GOAL = "Demonstrate deterministic stop while paused."

RESUME_FIRST_PROGRESS = "Worker reached the resume checkpoint."
RESUME_SECOND_PROGRESS = "Worker continued after resume."

STOP_FIRST_PROGRESS = "Worker reached the stop checkpoint."
STOP_FORBIDDEN_PROGRESS = "Worker continued after stop."


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    """Acceptance evidence for one deterministic runtime scenario."""

    name: str
    task: RuntimeTask
    events: tuple[RuntimeEvent, ...]
    pause_accepted: bool
    resume_accepted: bool | None
    stop_accepted: bool | None
    blocked_while_paused: bool
    worker_thread_stopped: bool
    failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Combined result for Experiment 06.01."""

    resume_scenario: ScenarioReport
    stop_scenario: ScenarioReport

    @property
    def failures(self) -> tuple[str, ...]:
        return (
            *self.resume_scenario.failures,
            *self.stop_scenario.failures,
        )


def run_pause_resume_scenario() -> ScenarioReport:
    """Prove pause blocks progress and resume allows completion."""

    task = RuntimeTask(goal=RESUME_TASK_GOAL)

    event_bus = RuntimeEventBus()
    events: list[RuntimeEvent] = []
    event_bus.subscribe(events.append)

    ready_for_checkpoint = Event()
    allow_checkpoint = Event()
    passed_checkpoint = Event()

    def worker(task, control, progress) -> None:
        progress(RESUME_FIRST_PROGRESS)
        ready_for_checkpoint.set()

        if not allow_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
            raise RuntimeError(
                "resume scenario did not receive checkpoint release"
            )

        control.checkpoint()

        passed_checkpoint.set()
        progress(RESUME_SECOND_PROGRESS)

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    thread = Thread(
        target=runtime.run,
        name="phase06-exp01-resume",
    )
    thread.start()

    failures: list[str] = []

    if not ready_for_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
        failures.append(
            "resume worker did not reach the pre-checkpoint stage"
        )

    pause_accepted = runtime.pause()
    if not pause_accepted:
        failures.append("resume scenario pause request was rejected")

    if task.status is not RuntimeStatus.PAUSED:
        failures.append(
            "resume scenario task did not enter PAUSED status"
        )

    allow_checkpoint.set()

    blocked_while_paused = not passed_checkpoint.wait(
        timeout=PAUSE_PROBE_SECONDS
    )
    if not blocked_while_paused:
        failures.append(
            "resume worker passed checkpoint while runtime was paused"
        )

    resume_accepted = runtime.resume()
    if not resume_accepted:
        failures.append("resume scenario resume request was rejected")

    if not passed_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
        failures.append(
            "resume worker did not continue after resume"
        )

    thread.join(timeout=WAIT_TIMEOUT_SECONDS)
    worker_thread_stopped = not thread.is_alive()

    if not worker_thread_stopped:
        failures.append(
            "resume worker thread did not terminate"
        )

    if task.status is not RuntimeStatus.COMPLETED:
        failures.append(
            "resume scenario final task status was "
            f"{task.status.value}, not completed"
        )

    expected_event_types = (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.TASK_RESUMED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    )
    actual_event_types = tuple(
        event.event_type
        for event in events
    )

    if actual_event_types != expected_event_types:
        failures.append(
            "resume scenario event order was "
            f"{tuple(event.value for event in actual_event_types)}, "
            "not "
            f"{tuple(event.value for event in expected_event_types)}"
        )

    progress_messages = tuple(
        event.message
        for event in events
        if event.event_type is RuntimeEventType.PROGRESS
    )

    expected_progress_messages = (
        RESUME_FIRST_PROGRESS,
        RESUME_SECOND_PROGRESS,
    )

    if progress_messages != expected_progress_messages:
        failures.append(
            "resume scenario progress messages were "
            f"{progress_messages}, not {expected_progress_messages}"
        )

    if any(
        event.task_id != task.task_id
        for event in events
    ):
        failures.append(
            "resume scenario emitted an event with the wrong task id"
        )

    if runtime.control.paused:
        failures.append(
            "resume scenario control remained paused after completion"
        )

    if runtime.control.stop_requested:
        failures.append(
            "resume scenario unexpectedly recorded a stop request"
        )

    return ScenarioReport(
        name="pause-resume-complete",
        task=task,
        events=tuple(events),
        pause_accepted=pause_accepted,
        resume_accepted=resume_accepted,
        stop_accepted=None,
        blocked_while_paused=blocked_while_paused,
        worker_thread_stopped=worker_thread_stopped,
        failures=tuple(failures),
    )


def run_pause_stop_scenario() -> ScenarioReport:
    """Prove stop wakes a paused checkpoint and terminates cleanly."""

    task = RuntimeTask(goal=STOP_TASK_GOAL)

    event_bus = RuntimeEventBus()
    events: list[RuntimeEvent] = []
    event_bus.subscribe(events.append)

    ready_for_checkpoint = Event()
    allow_checkpoint = Event()
    forbidden_progress = Event()

    def worker(task, control, progress) -> None:
        progress(STOP_FIRST_PROGRESS)
        ready_for_checkpoint.set()

        if not allow_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
            raise RuntimeError(
                "stop scenario did not receive checkpoint release"
            )

        control.checkpoint()

        forbidden_progress.set()
        progress(STOP_FORBIDDEN_PROGRESS)

    runtime = TaskRuntime(
        task=task,
        worker=worker,
        event_bus=event_bus,
    )

    thread = Thread(
        target=runtime.run,
        name="phase06-exp01-stop",
    )
    thread.start()

    failures: list[str] = []

    if not ready_for_checkpoint.wait(timeout=WAIT_TIMEOUT_SECONDS):
        failures.append(
            "stop worker did not reach the pre-checkpoint stage"
        )

    pause_accepted = runtime.pause()
    if not pause_accepted:
        failures.append("stop scenario pause request was rejected")

    if task.status is not RuntimeStatus.PAUSED:
        failures.append(
            "stop scenario task did not enter PAUSED status"
        )

    allow_checkpoint.set()

    stop_accepted = runtime.stop()
    if not stop_accepted:
        failures.append("stop scenario stop request was rejected")

    thread.join(timeout=WAIT_TIMEOUT_SECONDS)
    worker_thread_stopped = not thread.is_alive()

    if not worker_thread_stopped:
        failures.append(
            "stop worker thread did not terminate"
        )

    blocked_while_paused = not forbidden_progress.is_set()
    if not blocked_while_paused:
        failures.append(
            "stop worker executed progress after stop"
        )

    if task.status is not RuntimeStatus.STOPPED:
        failures.append(
            "stop scenario final task status was "
            f"{task.status.value}, not stopped"
        )

    expected_event_types = (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.STOP_REQUESTED,
        RuntimeEventType.TASK_STOPPED,
    )
    actual_event_types = tuple(
        event.event_type
        for event in events
    )

    if actual_event_types != expected_event_types:
        failures.append(
            "stop scenario event order was "
            f"{tuple(event.value for event in actual_event_types)}, "
            "not "
            f"{tuple(event.value for event in expected_event_types)}"
        )

    progress_messages = tuple(
        event.message
        for event in events
        if event.event_type is RuntimeEventType.PROGRESS
    )

    if progress_messages != (STOP_FIRST_PROGRESS,):
        failures.append(
            "stop scenario progress messages were "
            f"{progress_messages}, not {(STOP_FIRST_PROGRESS,)}"
        )

    if any(
        event.task_id != task.task_id
        for event in events
    ):
        failures.append(
            "stop scenario emitted an event with the wrong task id"
        )

    if not runtime.control.stop_requested:
        failures.append(
            "stop scenario did not preserve stop_requested"
        )

    if runtime.control.paused:
        failures.append(
            "stop scenario control remained paused after stop"
        )

    return ScenarioReport(
        name="pause-stop",
        task=task,
        events=tuple(events),
        pause_accepted=pause_accepted,
        resume_accepted=None,
        stop_accepted=stop_accepted,
        blocked_while_paused=blocked_while_paused,
        worker_thread_stopped=worker_thread_stopped,
        failures=tuple(failures),
    )


def run_experiment() -> ExperimentReport:
    """Run both deterministic runtime acceptance scenarios."""

    return ExperimentReport(
        resume_scenario=run_pause_resume_scenario(),
        stop_scenario=run_pause_stop_scenario(),
    )


def print_report(report: ExperimentReport) -> None:
    """Print stable Experiment 06.01 acceptance evidence."""

    print(TITLE)
    print()

    _print_scenario(report.resume_scenario)
    print()
    _print_scenario(report.stop_scenario)
    print()

    if report.failures:
        print("Experiment acceptance: failed")

        for failure in report.failures:
            print(f"  {failure}")

        return

    print("Experiment acceptance: passed")


def _print_scenario(report: ScenarioReport) -> None:
    print(f"Scenario: {report.name}")
    print(f"Task goal: {report.task.goal}")
    print(f"Final status: {report.task.status.value}")
    print(
        "Pause accepted: "
        f"{_yes_no(report.pause_accepted)}"
    )

    if report.resume_accepted is not None:
        print(
            "Resume accepted: "
            f"{_yes_no(report.resume_accepted)}"
        )

    if report.stop_accepted is not None:
        print(
            "Stop accepted: "
            f"{_yes_no(report.stop_accepted)}"
        )

    print(
        "Blocked at cooperative checkpoint: "
        f"{_yes_no(report.blocked_while_paused)}"
    )
    print(
        "Worker thread stopped: "
        f"{_yes_no(report.worker_thread_stopped)}"
    )

    print(
        "Event order: "
        + " -> ".join(
            event.event_type.value
            for event in report.events
        )
    )

    progress_messages = tuple(
        event.message
        for event in report.events
        if event.event_type is RuntimeEventType.PROGRESS
    )
    print(f"Progress messages: {progress_messages}")

    if report.failures:
        print("Scenario acceptance: failed")

        for failure in report.failures:
            print(f"  {failure}")
    else:
        print("Scenario acceptance: passed")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
