"""Tests for cooperative interactive-runtime control."""

from __future__ import annotations

from threading import Event, Thread

import pytest

from computer_agent.runtime import (
    RuntimeControl,
    RuntimeStopRequested,
)


def test_runtime_control_defaults_to_active() -> None:
    control = RuntimeControl()

    assert control.paused is False
    assert control.stop_requested is False


def test_pause_and_resume_are_idempotent() -> None:
    control = RuntimeControl()

    assert control.pause() is True
    assert control.paused is True

    assert control.pause() is False
    assert control.paused is True

    assert control.resume() is True
    assert control.paused is False

    assert control.resume() is False
    assert control.paused is False


def test_checkpoint_returns_immediately_when_active() -> None:
    control = RuntimeControl()

    control.checkpoint()


def test_checkpoint_blocks_while_paused_and_continues_after_resume() -> None:
    control = RuntimeControl()

    checkpoint_entered = Event()
    checkpoint_returned = Event()

    assert control.pause() is True

    def worker() -> None:
        checkpoint_entered.set()
        control.checkpoint()
        checkpoint_returned.set()

    thread = Thread(target=worker)
    thread.start()

    assert checkpoint_entered.wait(timeout=1.0)
    assert checkpoint_returned.wait(timeout=0.05) is False

    assert control.resume() is True

    assert checkpoint_returned.wait(timeout=1.0)

    thread.join(timeout=1.0)
    assert thread.is_alive() is False


def test_stop_request_causes_checkpoint_to_raise() -> None:
    control = RuntimeControl()

    assert control.request_stop() is True
    assert control.stop_requested is True

    with pytest.raises(
        RuntimeStopRequested,
        match="runtime stop requested",
    ):
        control.checkpoint()


def test_duplicate_stop_request_is_rejected() -> None:
    control = RuntimeControl()

    assert control.request_stop() is True
    assert control.request_stop() is False


def test_stop_wakes_a_paused_checkpoint() -> None:
    control = RuntimeControl()

    checkpoint_entered = Event()
    stop_observed = Event()

    assert control.pause() is True

    def worker() -> None:
        checkpoint_entered.set()

        try:
            control.checkpoint()
        except RuntimeStopRequested:
            stop_observed.set()

    thread = Thread(target=worker)
    thread.start()

    assert checkpoint_entered.wait(timeout=1.0)
    assert stop_observed.wait(timeout=0.05) is False

    assert control.request_stop() is True

    assert stop_observed.wait(timeout=1.0)

    thread.join(timeout=1.0)
    assert thread.is_alive() is False

    assert control.paused is False
    assert control.stop_requested is True


def test_pause_and_resume_are_rejected_after_stop_request() -> None:
    control = RuntimeControl()

    assert control.request_stop() is True

    assert control.pause() is False
    assert control.resume() is False
