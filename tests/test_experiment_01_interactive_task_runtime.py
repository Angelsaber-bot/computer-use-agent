"""Tests for Phase 06 Experiment 01."""

from __future__ import annotations

from computer_agent.runtime import (
    RuntimeEventType,
    RuntimeStatus,
)

from experiments.phase06_interactive_agent import (
    experiment_01_interactive_task_runtime as experiment,
)


def test_pause_resume_scenario_passes() -> None:
    report = experiment.run_pause_resume_scenario()

    assert report.failures == ()
    assert report.task.status is RuntimeStatus.COMPLETED
    assert report.pause_accepted is True
    assert report.resume_accepted is True
    assert report.stop_accepted is None
    assert report.blocked_while_paused is True
    assert report.worker_thread_stopped is True

    assert tuple(
        event.event_type
        for event in report.events
    ) == (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.TASK_RESUMED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_COMPLETED,
    )


def test_pause_stop_scenario_passes() -> None:
    report = experiment.run_pause_stop_scenario()

    assert report.failures == ()
    assert report.task.status is RuntimeStatus.STOPPED
    assert report.pause_accepted is True
    assert report.resume_accepted is None
    assert report.stop_accepted is True
    assert report.blocked_while_paused is True
    assert report.worker_thread_stopped is True

    assert tuple(
        event.event_type
        for event in report.events
    ) == (
        RuntimeEventType.TASK_STARTED,
        RuntimeEventType.PROGRESS,
        RuntimeEventType.TASK_PAUSED,
        RuntimeEventType.STOP_REQUESTED,
        RuntimeEventType.TASK_STOPPED,
    )


def test_combined_experiment_passes() -> None:
    report = experiment.run_experiment()

    assert report.failures == ()


def test_print_report_contains_stable_acceptance_output(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)

    output = capsys.readouterr().out

    assert experiment.TITLE in output
    assert "Scenario: pause-resume-complete" in output
    assert "Final status: completed" in output
    assert "Resume accepted: yes" in output
    assert "Scenario: pause-stop" in output
    assert "Final status: stopped" in output
    assert "Stop accepted: yes" in output
    assert "Experiment acceptance: passed" in output


def test_main_returns_success(capsys) -> None:
    assert experiment.main() == 0

    output = capsys.readouterr().out
    assert "Experiment acceptance: passed" in output
