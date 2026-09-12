"""Tests for Phase 06 Experiment 02."""

from __future__ import annotations

from experiments.phase06_interactive_agent import (
    experiment_02_agent_workspace as experiment,
)


def test_agent_workspace_experiment_passes() -> None:
    passed, failures = experiment.run_experiment()

    assert passed is True
    assert failures == ()


def test_agent_workspace_main_returns_success(
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda: (True, ()),
    )

    result = experiment.main()

    output = capsys.readouterr().out

    assert result == 0
    assert experiment.TITLE in output
    assert "Workspace creation: passed" in output
    assert "Runtime event delivery: passed" in output
    assert "Pause checkpoint blocking: passed" in output
    assert "Resume continuation: passed" in output
    assert "Terminal UI controls: passed" in output
    assert "Experiment acceptance: passed" in output
