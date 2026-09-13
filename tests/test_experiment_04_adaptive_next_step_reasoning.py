"""Tests for Phase 06 Experiment 04."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_04_adaptive_next_step_reasoning
    as experiment,
)


def test_experiment_04_acceptance() -> None:
    report = experiment.run_experiment()

    assert (
        report.initial_action_passed
        is True
    )
    assert (
        report.state_adaptation_passed
        is True
    )
    assert (
        report.no_duplicate_submit_passed
        is True
    )
    assert (
        report.reconciliation_passed
        is True
    )
    assert (
        report.completion_gate_passed
        is True
    )
    assert (
        report.same_goal_passed
        is True
    )
    assert report.failures == ()


def test_experiment_04_print_report(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(
        report
    )

    output = capsys.readouterr().out

    assert (
        "Initial next-step action: passed"
        in output
    )
    assert (
        "Task-state adaptation: passed"
        in output
    )
    assert (
        "Duplicate-submit avoidance: passed"
        in output
    )
    assert (
        "Completion-gate reasoning: passed"
        in output
    )
    assert (
        "Experiment acceptance: passed"
        in output
    )


def test_experiment_04_main(
    monkeypatch,
    capsys,
) -> None:
    report = experiment.run_experiment()

    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0

    output = capsys.readouterr().out
    assert (
        "Experiment acceptance: passed"
        in output
    )
