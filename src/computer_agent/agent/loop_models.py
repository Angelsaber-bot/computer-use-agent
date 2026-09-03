"""Result models for deterministic agent loop orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.agent.state import AgentState, AgentStatus
from computer_agent.planning.models import StructuredPlan


class AgentLoopStatus(str, Enum):
    """Terminal outcomes for one deterministic agent loop run."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Terminal result for running a structured plan."""

    status: AgentLoopStatus
    plan: StructuredPlan
    state: AgentState
    completed_plan_steps: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, AgentLoopStatus):
            raise ValueError("status must be an AgentLoopStatus")

        if not isinstance(self.plan, StructuredPlan):
            raise ValueError("plan must be a StructuredPlan")

        if not isinstance(self.state, AgentState):
            raise ValueError("state must be an AgentState")

        if isinstance(self.completed_plan_steps, bool) or not isinstance(
            self.completed_plan_steps,
            int,
        ):
            raise ValueError(
                "completed_plan_steps must be a non-boolean integer"
            )

        if not 0 <= self.completed_plan_steps <= len(self.plan.steps):
            raise ValueError(
                "completed_plan_steps must be between zero and plan length"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if self.status is AgentLoopStatus.COMPLETED:
            if self.completed_plan_steps != len(self.plan.steps):
                raise ValueError(
                    "COMPLETED results require every plan step complete"
                )

            if self.state.status is not AgentStatus.SUCCEEDED:
                raise ValueError(
                    "COMPLETED results require a SUCCEEDED AgentState"
                )

        if self.status in (
            AgentLoopStatus.BLOCKED,
            AgentLoopStatus.EXHAUSTED,
        ) and self.state.status is not AgentStatus.FAILED:
            raise ValueError(
                "BLOCKED and EXHAUSTED results require a FAILED AgentState"
            )
