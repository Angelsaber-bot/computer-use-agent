"""Immutable presentation snapshots for semantic task state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from computer_agent.task.models import TaskState
from computer_agent.task.transitions import TaskStateTransitions


@dataclass(frozen=True, slots=True)
class SubgoalSnapshot:
    description: str
    status: str


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    summary: str
    source: str
    freshness: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class SideEffectSnapshot:
    description: str
    state: str
    idempotent: bool | None
    action_key: str | None


@dataclass(frozen=True, slots=True)
class TaskStateSnapshot:
    """Immutable UI-safe view of one TaskState."""

    task_id: str
    goal: str
    status: str
    subgoals: tuple[SubgoalSnapshot, ...]
    evidence: tuple[EvidenceSnapshot, ...]
    side_effects: tuple[SideEffectSnapshot, ...]
    pending_questions: tuple[str, ...]
    completion_allowed: bool
    completion_blockers: tuple[str, ...]

    @classmethod
    def from_state(
        cls,
        state: TaskState,
    ) -> "TaskStateSnapshot":
        if not isinstance(state, TaskState):
            raise ValueError("state must be a TaskState")

        transitions = TaskStateTransitions(state)
        blockers = transitions.completion_blockers()

        return cls(
            task_id=state.task_id,
            goal=state.goal,
            status=state.status.value,
            subgoals=tuple(
                SubgoalSnapshot(
                    description=item.description,
                    status=item.status.value,
                )
                for item in state.subgoals.values()
            ),
            evidence=tuple(
                EvidenceSnapshot(
                    summary=item.summary,
                    source=item.source,
                    freshness=item.freshness.value,
                    observed_at=item.observed_at,
                )
                for item in state.evidence.values()
            ),
            side_effects=tuple(
                SideEffectSnapshot(
                    description=item.description,
                    state=item.state.value,
                    idempotent=item.idempotent,
                    action_key=item.action_key,
                )
                for item in state.side_effects.values()
            ),
            pending_questions=state.pending_questions,
            completion_allowed=not blockers,
            completion_blockers=blockers,
        )
