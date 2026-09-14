"""Tests for Phase 06 Experiment 08.04."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_12_llm_durable_planning as experiment,
)


def test_experiment_12_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.wikipedia_two_step_plan is True
    assert report.wikipedia_three_step_plan is True
    assert report.python_plan is True
    assert report.mismatch_rejected_before_browser_action is True
    assert report.invalid_ordering_rejected is True
    assert report.persisted_plan_resumes_without_planner_call is True
    assert report.three_step_execution_uses_reconciler is True
    assert report.crash_recovery_does_not_replan is True
    assert report.planner_call_count == 2
    assert report.failures == ()


def test_experiment_12_print_report(capsys) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)
    output = capsys.readouterr().out

    assert "Wikipedia 2-step LLM durable plan: passed" in output
    assert "Crash recovery does not re-plan: passed" in output
    assert "Planner calls observed: 2" in output
    assert "Experiment acceptance: passed" in output


def test_experiment_12_main(monkeypatch, capsys) -> None:
    report = experiment.run_experiment()
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0
    assert "Experiment acceptance: passed" in capsys.readouterr().out
