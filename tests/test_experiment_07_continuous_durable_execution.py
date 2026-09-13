"""Tests for Phase 06 Experiment 07."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_07_continuous_durable_execution
    as experiment,
)


def test_experiment_07_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.uninterrupted_passed is True
    assert (
        report.restart_after_query_passed
        is True
    )
    assert (
        report.restart_after_submit_passed
        is True
    )
    assert report.failures == ()


def test_experiment_07_print_report(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(
        report
    )

    output = capsys.readouterr().out

    assert (
        "Uninterrupted continuous run: passed"
        in output
    )
    assert (
        "Restart after query checkpoint: passed"
        in output
    )
    assert (
        "Restart after GO before Results checkpoint: passed"
        in output
    )
    assert (
        "Experiment acceptance: passed"
        in output
    )


def test_experiment_07_main(
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
