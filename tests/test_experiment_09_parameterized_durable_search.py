"""Tests for Phase 06 Experiment 08.01."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_09_parameterized_durable_search
    as experiment,
)


def test_experiment_09_acceptance() -> None:
    report = experiment.run_experiment()

    assert report.wikipedia_claude_uninterrupted is True
    assert report.wikipedia_reinforcement_restart is True
    assert report.wikipedia_alan_submit_restart is True
    assert report.python_asyncio_uninterrupted is True
    assert report.python_dataclasses_restart is True
    assert report.malformed_unsupported_rejected is True
    assert report.failures == ()


def test_experiment_09_print_report(capsys) -> None:
    report = experiment.run_experiment()

    experiment.print_report(report)
    output = capsys.readouterr().out

    assert "Wikipedia Claude Shannon uninterrupted: passed" in output
    assert (
        "Wikipedia reinforcement learning query restart: passed"
        in output
    )
    assert "Wikipedia Alan Turing submit restart: passed" in output
    assert "python.org asyncio uninterrupted: passed" in output
    assert "python.org dataclasses restart: passed" in output
    assert "Experiment acceptance: passed" in output


def test_experiment_09_main(monkeypatch, capsys) -> None:
    report = experiment.run_experiment()
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: report,
    )

    assert experiment.main() == 0
    assert "Experiment acceptance: passed" in capsys.readouterr().out
