"""Tests for Phase 06 Experiment 07.02."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_08_configurable_durable_web_workflow
    as experiment,
)


def test_experiment_08_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.python_uninterrupted_passed is True
    assert report.wikipedia_uninterrupted_passed is True
    assert (
        report.wikipedia_restart_after_query_passed
        is True
    )
    assert (
        report.wikipedia_restart_after_submit_passed
        is True
    )
    assert report.workflow_mismatch_passed is True
    assert report.failures == ()


def test_experiment_08_print_report(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(
        report
    )

    output = capsys.readouterr().out

    assert "Python uninterrupted: passed" in output
    assert "Wikipedia uninterrupted: passed" in output
    assert (
        "Wikipedia restart after query checkpoint: passed"
        in output
    )
    assert (
        "Wikipedia restart after submit before result checkpoint: passed"
        in output
    )
    assert (
        "Workflow identity mismatch: passed"
        in output
    )
    assert (
        "Experiment acceptance: passed"
        in output
    )


def test_experiment_08_main(
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
