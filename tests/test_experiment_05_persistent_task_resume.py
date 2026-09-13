"""Tests for Phase 06 Experiment 05."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_05_persistent_task_resume
    as experiment,
)


def test_experiment_05_acceptance() -> None:
    report = experiment.run_experiment()

    assert (
        report.checkpoint_round_trip_passed
        is True
    )
    assert (
        report.restart_invalidated_old_evidence_passed
        is True
    )
    assert (
        report.unknown_side_effect_survived_passed
        is True
    )
    assert (
        report.duplicate_submit_blocked_passed
        is True
    )
    assert (
        report.bounded_replan_passed
        is True
    )
    assert (
        report.fresh_reverification_passed
        is True
    )
    assert (
        report.completion_gate_passed
        is True
    )
    assert (
        report.final_checkpoint_passed
        is True
    )
    assert (
        report.same_goal_passed
        is True
    )
    assert report.failures == ()


def test_experiment_05_print_report(
    capsys,
) -> None:
    report = experiment.run_experiment()

    experiment.print_report(
        report
    )

    output = capsys.readouterr().out

    assert (
        "Checkpoint round trip: passed"
        in output
    )
    assert (
        "Restart evidence invalidation: passed"
        in output
    )
    assert (
        "UNKNOWN side-effect persistence: passed"
        in output
    )
    assert (
        "Duplicate-submit safety block: passed"
        in output
    )
    assert (
        "Bounded reconciliation replan: passed"
        in output
    )
    assert (
        "Fresh re-verification: passed"
        in output
    )
    assert (
        "Completion-gate recovery: passed"
        in output
    )
    assert (
        "Final checkpoint recovery: passed"
        in output
    )
    assert (
        "Experiment acceptance: passed"
        in output
    )


def test_experiment_05_main(
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
