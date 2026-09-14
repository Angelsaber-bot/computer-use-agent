"""Tests for Phase 06 Experiment 08.03."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_11_persisted_durable_plan as experiment,
)


def test_experiment_11_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.wikipedia_search_plan_two_steps is True
    assert report.wikipedia_followup_plan_three_steps is True
    assert report.python_plan_two_steps is True
    assert report.uninterrupted_three_step_execution is True
    assert report.resume_from_step_1 is True
    assert report.resume_from_step_2 is True
    assert report.resume_at_step_3_destination is True
    assert report.persisted_plan_tampering_failed_closed is True
    assert report.search_only_regression is True
    assert report.failures == ()


def test_experiment_11_print_report(capsys) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)
    output = capsys.readouterr().out

    assert "Wikipedia search-only plan has 2 steps: passed" in output
    assert "Wikipedia follow-up plan has 3 steps: passed" in output
    assert "Resume already at Step 3 destination: passed" in output
    assert "Persisted plan tampering fails closed: passed" in output
    assert "Experiment acceptance: passed" in output


def test_experiment_11_main(monkeypatch, capsys) -> None:
    report = experiment.run_experiment()
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0
    assert "Experiment acceptance: passed" in capsys.readouterr().out
