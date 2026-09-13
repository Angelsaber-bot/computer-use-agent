"""Models for observation-conditioned adaptive next-step reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from computer_agent.planning.models import SemanticPlanStep


def _require_text(
    value: object,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string"
        )


def _require_text_tuple(
    values: object,
    field_name: str,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(
            f"{field_name} must be a tuple"
        )

    for value in values:
        _require_text(
            value,
            field_name,
        )


class NextStepDecisionType(str, Enum):
    """Kinds of decisions produced by adaptive reasoning."""

    ACTION = "action"
    ASK_USER = "ask_user"
    COMPLETE = "complete"


class NextStepReasoningStatus(str, Enum):
    """Outcome of one adaptive reasoning attempt."""

    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ObservedElement:
    """One compact semantic UI element exposed to reasoning."""

    text: str
    element_type: str | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        _require_text(self.text, "text")

        if self.element_type is not None:
            _require_text(
                self.element_type,
                "element_type",
            )

        if (
            self.enabled is not None
            and not isinstance(self.enabled, bool)
        ):
            raise ValueError(
                "enabled must be bool or None"
            )


@dataclass(frozen=True, slots=True)
class ObservationContext:
    """Compact current-environment context supplied to reasoning."""

    application_name: str | None = None
    window_title: str | None = None
    visible_text: tuple[str, ...] = ()
    elements: tuple[ObservedElement, ...] = ()

    def __post_init__(self) -> None:
        if self.application_name is not None:
            _require_text(
                self.application_name,
                "application_name",
            )

        if self.window_title is not None:
            _require_text(
                self.window_title,
                "window_title",
            )

        for text in self.visible_text:
            _require_text(
                text,
                "visible_text item",
            )

        for element in self.elements:
            if not isinstance(
                element,
                ObservedElement,
            ):
                raise ValueError(
                    "elements must contain "
                    "ObservedElement objects"
                )


@dataclass(frozen=True, slots=True)
class AdaptiveReasoningContext:
    """Complete bounded input for one next-step decision."""

    goal: str
    constraints: tuple[str, ...]
    task_status: str

    verified_subgoals: tuple[str, ...]
    unresolved_subgoals: tuple[str, ...]

    current_evidence: tuple[str, ...]
    stale_or_unknown_evidence: tuple[str, ...]

    unresolved_side_effects: tuple[str, ...]
    pending_questions: tuple[str, ...]

    completion_allowed: bool
    completion_blockers: tuple[str, ...]

    observation: ObservationContext
    blocked_action_keys: tuple[str, ...] = ()
    decision_feedback: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(
            self.goal,
            "goal",
        )
        _require_text(
            self.task_status,
            "task_status",
        )

        for collection_name, values in (
            ("constraint", self.constraints),
            (
                "verified_subgoal",
                self.verified_subgoals,
            ),
            (
                "unresolved_subgoal",
                self.unresolved_subgoals,
            ),
            (
                "current_evidence",
                self.current_evidence,
            ),
            (
                "stale_or_unknown_evidence",
                self.stale_or_unknown_evidence,
            ),
            (
                "unresolved_side_effect",
                self.unresolved_side_effects,
            ),
            (
                "pending_question",
                self.pending_questions,
            ),
            (
                "completion_blocker",
                self.completion_blockers,
            ),
            (
                "blocked_action_key",
                self.blocked_action_keys,
            ),
            (
                "decision_feedback",
                self.decision_feedback,
            ),
        ):
            _require_text_tuple(
                values,
                collection_name,
            )

        if not isinstance(
            self.completion_allowed,
            bool,
        ):
            raise ValueError(
                "completion_allowed must be a bool"
            )

        if not isinstance(
            self.observation,
            ObservationContext,
        ):
            raise ValueError(
                "observation must be an ObservationContext"
            )


@dataclass(frozen=True, slots=True)
class NextStepDecision:
    """One bounded adaptive decision."""

    decision_type: NextStepDecisionType

    action: SemanticPlanStep | None = None
    expected_effect: str | None = None

    question: str | None = None
    completion_summary: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision_type,
            NextStepDecisionType,
        ):
            raise ValueError(
                "decision_type must be a "
                "NextStepDecisionType"
            )

        if (
            self.decision_type
            is NextStepDecisionType.ACTION
        ):
            if self.action is None:
                raise ValueError(
                    "ACTION decisions require an action"
                )

            _require_text(
                self.expected_effect,
                "expected_effect",
            )

            if (
                self.question is not None
                or self.completion_summary is not None
            ):
                raise ValueError(
                    "ACTION decisions cannot contain "
                    "question or completion_summary"
                )

            return

        if (
            self.decision_type
            is NextStepDecisionType.ASK_USER
        ):
            _require_text(
                self.question,
                "question",
            )

            if (
                self.action is not None
                or self.expected_effect is not None
                or self.completion_summary is not None
            ):
                raise ValueError(
                    "ASK_USER decisions may contain "
                    "only a question"
                )

            return

        if (
            self.decision_type
            is NextStepDecisionType.COMPLETE
        ):
            _require_text(
                self.completion_summary,
                "completion_summary",
            )

            if (
                self.action is not None
                or self.expected_effect is not None
                or self.question is not None
            ):
                raise ValueError(
                    "COMPLETE decisions may contain "
                    "only a completion_summary"
                )


@dataclass(frozen=True, slots=True)
class NextStepReasoningResult:
    """Explicit result of one adaptive reasoning attempt."""

    status: NextStepReasoningStatus
    decision: NextStepDecision | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.status,
            NextStepReasoningStatus,
        ):
            raise ValueError(
                "status must be a "
                "NextStepReasoningStatus"
            )

        _require_text(
            self.reason,
            "reason",
        )

        if (
            self.status
            is NextStepReasoningStatus.READY
        ):
            if not isinstance(
                self.decision,
                NextStepDecision,
            ):
                raise ValueError(
                    "READY results require "
                    "a NextStepDecision"
                )

        if (
            self.status
            is NextStepReasoningStatus.BLOCKED
            and self.decision is not None
        ):
            raise ValueError(
                "BLOCKED results must not "
                "contain a decision"
            )
