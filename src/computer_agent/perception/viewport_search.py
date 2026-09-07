"""Bounded deterministic viewport search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from numbers import Real

from computer_agent.core.models import Action, ToolResult
from computer_agent.perception.fusion import normalize_ui_text
from computer_agent.perception.viewport import (
    DiagnosisStatus,
    SemanticAXElement,
    Viewport,
    ViewportTargetDiagnosis,
    diagnose_semantic_target_visibility,
)


class ViewportSearchStatus(str, Enum):
    """Outcomes for bounded viewport search."""

    FOUND = "found"
    NEEDS_SCROLL = "needs_scroll"
    BLOCKED = "blocked"
    STALLED = "stalled"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True, slots=True)
class ViewportSearchPolicy:
    """Configuration for one bounded viewport search."""

    max_scroll_attempts: int = 6
    scroll_amount: int = 4
    stabilization_wait_seconds: float = 0.3

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_scroll_attempts, bool)
            or not isinstance(self.max_scroll_attempts, int)
        ):
            raise ValueError("max_scroll_attempts must be an integer")

        if self.max_scroll_attempts < 0:
            raise ValueError("max_scroll_attempts must be non-negative")

        if (
            isinstance(self.scroll_amount, bool)
            or not isinstance(self.scroll_amount, int)
        ):
            raise ValueError("scroll_amount must be an integer")

        if self.scroll_amount <= 0:
            raise ValueError("scroll_amount must be positive")

        if (
            isinstance(self.stabilization_wait_seconds, bool)
            or not isinstance(self.stabilization_wait_seconds, Real)
        ):
            raise ValueError(
                "stabilization_wait_seconds must be numeric"
            )

        if self.stabilization_wait_seconds < 0:
            raise ValueError(
                "stabilization_wait_seconds must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class ViewportSearchObservation:
    """One read-only browser viewport observation."""

    application_name: str | None
    viewport: Viewport | None
    semantic_elements: tuple[SemanticAXElement, ...]

    def __post_init__(self) -> None:
        if self.application_name is not None and not isinstance(
            self.application_name,
            str,
        ):
            raise ValueError(
                "application_name must be a string or None"
            )

        if self.viewport is not None and not isinstance(
            self.viewport,
            Viewport,
        ):
            raise ValueError("viewport must be a Viewport or None")

        if not isinstance(self.semantic_elements, tuple) or any(
            not isinstance(element, SemanticAXElement)
            for element in self.semantic_elements
        ):
            raise ValueError(
                "semantic_elements must be a tuple of SemanticAXElement "
                "objects"
            )


@dataclass(frozen=True, slots=True)
class ViewportSearchResult:
    """Result of a bounded deterministic viewport search."""

    status: ViewportSearchStatus
    diagnosis: ViewportTargetDiagnosis
    scroll_attempts: int
    observations: tuple[ViewportSearchObservation, ...]
    tool_results: tuple[ToolResult, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ViewportSearchStatus):
            raise ValueError("status must be a ViewportSearchStatus")

        if not isinstance(self.diagnosis, ViewportTargetDiagnosis):
            raise ValueError(
                "diagnosis must be a ViewportTargetDiagnosis"
            )

        if (
            isinstance(self.scroll_attempts, bool)
            or not isinstance(self.scroll_attempts, int)
        ):
            raise ValueError("scroll_attempts must be an integer")

        if self.scroll_attempts < 0:
            raise ValueError("scroll_attempts must be non-negative")

        if not isinstance(self.observations, tuple) or any(
            not isinstance(observation, ViewportSearchObservation)
            for observation in self.observations
        ):
            raise ValueError(
                "observations must be a tuple of "
                "ViewportSearchObservation objects"
            )

        if not isinstance(self.tool_results, tuple) or any(
            not isinstance(result, ToolResult)
            for result in self.tool_results
        ):
            raise ValueError(
                "tool_results must be a tuple of ToolResult objects"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


class ViewportSearchController:
    """Run bounded downward viewport search using fresh observations."""

    def __init__(
        self,
        *,
        observer: Callable[[], ViewportSearchObservation],
        target_role: str,
        target_text: str,
        policy: ViewportSearchPolicy | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not callable(observer):
            raise ValueError("observer must be callable")

        if not isinstance(target_role, str) or not target_role.strip():
            raise ValueError("target_role must be a non-empty string")

        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("target_text must be a non-empty string")

        if policy is not None and not isinstance(
            policy,
            ViewportSearchPolicy,
        ):
            raise ValueError(
                "policy must be a ViewportSearchPolicy or None"
            )

        if sleeper is not None and not callable(sleeper):
            raise ValueError("sleeper must be callable or None")

        self._observer = observer
        self._target_role = target_role
        self._target_text = target_text
        self._policy = policy or ViewportSearchPolicy()
        self._sleeper = sleeper or (lambda _seconds: None)

    def search(
        self,
        *,
        execute: bool,
        executor: object | None = None,
    ) -> ViewportSearchResult:
        """Search downward until the target is visible or policy stops."""

        if not isinstance(execute, bool):
            raise ValueError("execute must be a bool")

        observations: list[ViewportSearchObservation] = []
        tool_results: list[ToolResult] = []

        observation = self._observe()
        observations.append(observation)
        diagnosis = self._diagnose(observation)

        if diagnosis.status is DiagnosisStatus.VISIBLE:
            return self._result(
                ViewportSearchStatus.FOUND,
                diagnosis,
                observations,
                tool_results,
                "target is visible",
            )

        if diagnosis.status is DiagnosisStatus.BLOCKED:
            return self._result(
                ViewportSearchStatus.BLOCKED,
                diagnosis,
                observations,
                tool_results,
                diagnosis.reason,
            )

        if self._policy.max_scroll_attempts == 0:
            return self._result(
                ViewportSearchStatus.EXHAUSTED,
                diagnosis,
                observations,
                tool_results,
                "scroll-attempt budget exhausted",
            )

        if not execute:
            return self._result(
                ViewportSearchStatus.NEEDS_SCROLL,
                diagnosis,
                observations,
                tool_results,
                "downward scrolling would be required",
            )

        if executor is None or not hasattr(executor, "execute"):
            raise ValueError(
                "executor with an execute method is required in execute mode"
            )

        previous_signature = _semantic_viewport_signature(observation)

        for _attempt in range(self._policy.max_scroll_attempts):
            action = Action(
                tool_name="scroll",
                arguments={
                    "amount": -self._policy.scroll_amount,
                },
                reason="bounded downward viewport search",
            )

            tool_result = executor.execute(action)
            tool_results.append(tool_result)

            if not isinstance(tool_result, ToolResult):
                raise ValueError("executor must return ToolResult objects")

            if not tool_result.success:
                return self._result(
                    ViewportSearchStatus.BLOCKED,
                    diagnosis,
                    observations,
                    tool_results,
                    "scroll command failed",
                )

            self._sleeper(self._policy.stabilization_wait_seconds)

            observation = self._observe()
            observations.append(observation)
            diagnosis = self._diagnose(observation)

            if diagnosis.status is DiagnosisStatus.VISIBLE:
                return self._result(
                    ViewportSearchStatus.FOUND,
                    diagnosis,
                    observations,
                    tool_results,
                    "target became visible",
                )

            if diagnosis.status is DiagnosisStatus.BLOCKED:
                return self._result(
                    ViewportSearchStatus.BLOCKED,
                    diagnosis,
                    observations,
                    tool_results,
                    diagnosis.reason,
                )

            current_signature = _semantic_viewport_signature(observation)
            if current_signature == previous_signature:
                return self._result(
                    ViewportSearchStatus.STALLED,
                    diagnosis,
                    observations,
                    tool_results,
                    "semantic viewport signature did not change",
                )

            previous_signature = current_signature

        return self._result(
            ViewportSearchStatus.EXHAUSTED,
            diagnosis,
            observations,
            tool_results,
            "scroll-attempt budget exhausted",
        )

    def _observe(self) -> ViewportSearchObservation:
        observation = self._observer()
        if not isinstance(observation, ViewportSearchObservation):
            raise ValueError(
                "observer must return ViewportSearchObservation objects"
            )

        return observation

    def _diagnose(
        self,
        observation: ViewportSearchObservation,
    ) -> ViewportTargetDiagnosis:
        return diagnose_semantic_target_visibility(
            application_name=observation.application_name,
            viewport=observation.viewport,
            semantic_elements=observation.semantic_elements,
            target_role=self._target_role,
            target_text=self._target_text,
        )

    @staticmethod
    def _result(
        status: ViewportSearchStatus,
        diagnosis: ViewportTargetDiagnosis,
        observations: list[ViewportSearchObservation],
        tool_results: list[ToolResult],
        reason: str,
    ) -> ViewportSearchResult:
        return ViewportSearchResult(
            status=status,
            diagnosis=diagnosis,
            scroll_attempts=len(tool_results),
            observations=tuple(observations),
            tool_results=tuple(tool_results),
            reason=reason,
        )


def _semantic_viewport_signature(
    observation: ViewportSearchObservation,
) -> tuple[object, ...]:
    viewport_bounds = None
    if observation.viewport is not None:
        bounds = observation.viewport.bounds
        viewport_bounds = (
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
        )

    return (
        observation.application_name,
        viewport_bounds,
        tuple(
            (
                element.role,
                normalize_ui_text(element.text),
                element.value,
                _bounds_signature(element),
            )
            for element in observation.semantic_elements
        ),
    )


def _bounds_signature(
    element: SemanticAXElement,
) -> tuple[int, int, int, int] | None:
    if element.bounds is None:
        return None

    return (
        element.bounds.x,
        element.bounds.y,
        element.bounds.width,
        element.bounds.height,
    )
