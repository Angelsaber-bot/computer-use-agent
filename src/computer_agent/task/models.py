"""Evidence-grounded semantic task-state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_aware_datetime(
    value: datetime,
    field_name: str,
) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{field_name} must be timezone-aware"
        )


class TaskStateStatus(str, Enum):
    """Semantic lifecycle of a user task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubgoalStatus(str, Enum):
    """Evidence-grounded progress state of one task subgoal."""

    PENDING = "pending"
    ACTIVE = "active"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    """Current truth status of one semantic claim."""

    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class EvidenceFreshness(str, Enum):
    """Whether evidence may still support a current-state decision."""

    CURRENT = "current"
    STALE = "stale"
    INVALIDATED = "invalidated"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    """Origin category for one evidence record."""

    OBSERVATION = "observation"
    VERIFICATION = "verification"
    ARTIFACT = "artifact"
    USER_CONFIRMATION = "user_confirmation"
    EXTERNAL_STATE = "external_state"


class SideEffectState(str, Enum):
    """State of a potentially consequential external side effect."""

    INTENDED = "intended"
    EXECUTED = "executed"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One source-backed observation or verification fact."""

    summary: str
    source: str
    kind: EvidenceKind
    freshness: EvidenceFreshness = EvidenceFreshness.CURRENT
    observed_at: datetime = field(default_factory=utc_now)
    evidence_id: str = field(default_factory=_new_id)
    uncertainty: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.summary, "summary")
        _require_text(self.source, "source")
        _require_text(self.evidence_id, "evidence_id")

        if not isinstance(self.kind, EvidenceKind):
            raise ValueError("kind must be an EvidenceKind")

        if not isinstance(
            self.freshness,
            EvidenceFreshness,
        ):
            raise ValueError(
                "freshness must be an EvidenceFreshness"
            )

        _require_aware_datetime(
            self.observed_at,
            "observed_at",
        )

        if self.uncertainty is not None:
            _require_text(
                self.uncertainty,
                "uncertainty",
            )


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """One semantic proposition the task may rely on."""

    statement: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    evidence_ids: tuple[str, ...] = ()
    claim_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        _require_text(self.statement, "statement")
        _require_text(self.claim_id, "claim_id")

        if not isinstance(self.status, ClaimStatus):
            raise ValueError("status must be a ClaimStatus")

        for evidence_id in self.evidence_ids:
            _require_text(evidence_id, "evidence_id")


@dataclass(frozen=True, slots=True)
class SubgoalRecord:
    """One bounded semantic unit of task progress."""

    description: str
    status: SubgoalStatus = SubgoalStatus.PENDING
    claim_ids: tuple[str, ...] = ()
    subgoal_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        _require_text(self.description, "description")
        _require_text(self.subgoal_id, "subgoal_id")

        if not isinstance(self.status, SubgoalStatus):
            raise ValueError(
                "status must be a SubgoalStatus"
            )

        for claim_id in self.claim_ids:
            _require_text(claim_id, "claim_id")


@dataclass(frozen=True, slots=True)
class SideEffectRecord:
    """Track an external action whose outcome may require reconciliation."""

    description: str
    state: SideEffectState = SideEffectState.INTENDED
    side_effect_id: str = field(default_factory=_new_id)
    external_reference: str | None = None
    evidence_ids: tuple[str, ...] = ()
    idempotent: bool | None = None
    action_key: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.description, "description")
        _require_text(
            self.side_effect_id,
            "side_effect_id",
        )

        if not isinstance(
            self.state,
            SideEffectState,
        ):
            raise ValueError(
                "state must be a SideEffectState"
            )

        if self.external_reference is not None:
            _require_text(
                self.external_reference,
                "external_reference",
            )

        for evidence_id in self.evidence_ids:
            _require_text(evidence_id, "evidence_id")

        if (
            self.idempotent is not None
            and not isinstance(self.idempotent, bool)
        ):
            raise ValueError(
                "idempotent must be bool or None"
            )

        if self.action_key is not None:
            _require_text(
                self.action_key,
                "action_key",
            )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One task-relevant external artifact or deliverable."""

    description: str
    location: str
    artifact_id: str = field(default_factory=_new_id)
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.description, "description")
        _require_text(self.location, "location")
        _require_text(self.artifact_id, "artifact_id")

        for evidence_id in self.evidence_ids:
            _require_text(evidence_id, "evidence_id")


@dataclass(slots=True)
class TaskState:
    """Semantic state of what the agent currently believes about a task."""

    goal: str
    constraints: tuple[str, ...] = ()
    task_id: str = field(default_factory=_new_id)
    status: TaskStateStatus = TaskStateStatus.PENDING

    subgoals: dict[str, SubgoalRecord] = field(
        default_factory=dict
    )
    claims: dict[str, ClaimRecord] = field(
        default_factory=dict
    )
    evidence: dict[str, EvidenceRecord] = field(
        default_factory=dict
    )
    side_effects: dict[str, SideEffectRecord] = field(
        default_factory=dict
    )
    artifacts: dict[str, ArtifactRecord] = field(
        default_factory=dict
    )

    pending_questions: tuple[str, ...] = ()

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        _require_text(self.goal, "goal")
        _require_text(self.task_id, "task_id")

        if not isinstance(
            self.status,
            TaskStateStatus,
        ):
            raise ValueError(
                "status must be a TaskStateStatus"
            )

        for constraint in self.constraints:
            _require_text(constraint, "constraint")

        for question in self.pending_questions:
            _require_text(question, "question")

        _require_aware_datetime(
            self.created_at,
            "created_at",
        )
        _require_aware_datetime(
            self.updated_at,
            "updated_at",
        )

    def touch(self) -> None:
        """Update the semantic-state modification timestamp."""
        self.updated_at = utc_now()
