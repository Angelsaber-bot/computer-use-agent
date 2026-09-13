"""Tests for Phase 06 Experiment 03."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_03_evidence_grounded_task_state as experiment,
)


def test_evidence_grounded_task_state_experiment_passes() -> None:
    report = experiment.run_experiment()

    assert report.failures == ()
    assert report.initial_verification_passed is True
    assert report.stale_invalidation_passed is True
    assert report.explicit_reverification_passed is True
    assert report.unknown_side_effect_passed is True
    assert report.reconciliation_passed is True
    assert report.artifact_passed is True


def test_experiment_report_contains_acceptance_evidence(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)

    output = capsys.readouterr().out

    assert experiment.TITLE in output
    assert "Initial evidence verification: passed" in output
    assert "Stale evidence invalidation: passed" in output
    assert "Explicit re-verification: passed" in output
    assert "Unknown side-effect preservation: passed" in output
    assert "Side-effect reconciliation: passed" in output
    assert "Artifact recording: passed" in output
    assert "Experiment acceptance: passed" in output


def test_main_returns_success(
    capsys,
    monkeypatch,
) -> None:
    report = experiment.ExperimentReport(
        initial_verification_passed=True,
        stale_invalidation_passed=True,
        explicit_reverification_passed=True,
        unknown_side_effect_passed=True,
        reconciliation_passed=True,
        artifact_passed=True,
        failures=(),
    )

    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0

    output = capsys.readouterr().out
    assert "Experiment acceptance: passed" in output
