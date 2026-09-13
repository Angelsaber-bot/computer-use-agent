"""Persistent storage locations for the desktop Agent Workspace."""

from __future__ import annotations

from pathlib import Path

from computer_agent.task import (
    TaskStateStore,
)


def default_task_checkpoint_directory() -> Path:
    """Return the macOS workspace checkpoint directory."""
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Computer Agent"
        / "checkpoints"
    )


def create_default_task_store() -> TaskStateStore:
    """Create the production workspace task store."""
    return TaskStateStore(
        default_task_checkpoint_directory()
    )
