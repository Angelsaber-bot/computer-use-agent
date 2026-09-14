"""Tests for Phase 06 Experiment 08.02."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_10_durable_followup_navigation
    as experiment,
)


def test_experiment_10_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.claude_uninterrupted is True
    assert report.alan_query_restart is True
    assert report.claude_submit_restart is True
    assert report.claude_followup_restart is True
    assert report.missing_link_failed_closed is True
    assert (
        report.followup_identity_mismatch_failed_closed
        is True
    )
    assert report.search_only_regression is True
    assert report.failures == ()


def test_experiment_10_print_report(capsys) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)
    output = capsys.readouterr().out

    assert (
        "Claude Shannon to Information theory uninterrupted: passed"
        in output
    )
    assert (
        "Alan Turing to Turing machine query restart: passed"
        in output
    )
    assert (
        "Claude Shannon follow-up restart: passed"
        in output
    )
    assert "Requested link missing fails closed: passed" in output
    assert "Experiment acceptance: passed" in output


def test_experiment_10_main(monkeypatch, capsys) -> None:
    report = experiment.run_experiment()
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0
    assert "Experiment acceptance: passed" in capsys.readouterr().out
