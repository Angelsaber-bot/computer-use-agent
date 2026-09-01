"""Deterministic builder for structured semantic plans."""

from __future__ import annotations

from computer_agent.planning.models import PlanStep, StructuredPlan


class StructuredPlanner:
    """Build structured plans from explicit semantic steps."""

    def build_plan(
        self,
        *,
        task_goal: str,
        steps: tuple[PlanStep, ...],
    ) -> StructuredPlan:
        """Return an immutable structured plan after validation."""

        return StructuredPlan(
            task_goal=task_goal,
            steps=steps,
        )
