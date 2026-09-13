"""Versioned persistence codec for evidence-grounded task state."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from computer_agent.task.models import (
    ArtifactRecord,
    ClaimRecord,
    ClaimStatus,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
)


TASK_STATE_SCHEMA_VERSION = 1


class TaskStatePersistenceError(ValueError):
    """Raised when persisted task-state data is invalid or unsupported."""


def task_state_to_payload(
    state: TaskState,
) -> dict[str, Any]:
    """Convert one TaskState into a versioned JSON-compatible payload."""
    if not isinstance(state, TaskState):
        raise ValueError("state must be a TaskState")

    return {
        "schema_version": TASK_STATE_SCHEMA_VERSION,
        "task_state": {
            "task_id": state.task_id,
            "goal": state.goal,
            "constraints": list(state.constraints),
            "status": state.status.value,
            "subgoals": [
                {
                    "subgoal_id": record.subgoal_id,
                    "description": record.description,
                    "status": record.status.value,
                    "claim_ids": list(record.claim_ids),
                }
                for record in state.subgoals.values()
            ],
            "claims": [
                {
                    "claim_id": record.claim_id,
                    "statement": record.statement,
                    "status": record.status.value,
                    "evidence_ids": list(record.evidence_ids),
                }
                for record in state.claims.values()
            ],
            "evidence": [
                {
                    "evidence_id": record.evidence_id,
                    "summary": record.summary,
                    "source": record.source,
                    "kind": record.kind.value,
                    "freshness": record.freshness.value,
                    "observed_at": record.observed_at.isoformat(),
                    "uncertainty": record.uncertainty,
                }
                for record in state.evidence.values()
            ],
            "side_effects": [
                {
                    "side_effect_id": record.side_effect_id,
                    "description": record.description,
                    "state": record.state.value,
                    "external_reference": record.external_reference,
                    "evidence_ids": list(record.evidence_ids),
                    "idempotent": record.idempotent,
                    "action_key": record.action_key,
                }
                for record in state.side_effects.values()
            ],
            "artifacts": [
                {
                    "artifact_id": record.artifact_id,
                    "description": record.description,
                    "location": record.location,
                    "evidence_ids": list(record.evidence_ids),
                }
                for record in state.artifacts.values()
            ],
            "pending_questions": list(
                state.pending_questions
            ),
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        },
    }


def task_state_from_payload(
    payload: dict[str, Any],
) -> TaskState:
    """Restore one TaskState from a versioned persistence payload."""
    if not isinstance(payload, dict):
        raise TaskStatePersistenceError(
            "payload must be a dictionary"
        )

    version = payload.get("schema_version")

    if version != TASK_STATE_SCHEMA_VERSION:
        raise TaskStatePersistenceError(
            "unsupported task-state schema version: "
            f"{version!r}"
        )

    raw_state = payload.get("task_state")

    if not isinstance(raw_state, dict):
        raise TaskStatePersistenceError(
            "task_state must be a dictionary"
        )

    try:
        evidence_records = [
            EvidenceRecord(
                evidence_id=_required_text(
                    item,
                    "evidence_id",
                ),
                summary=_required_text(
                    item,
                    "summary",
                ),
                source=_required_text(
                    item,
                    "source",
                ),
                kind=EvidenceKind(
                    item["kind"]
                ),
                freshness=EvidenceFreshness(
                    item["freshness"]
                ),
                observed_at=_parse_datetime(
                    item["observed_at"],
                    "observed_at",
                ),
                uncertainty=_optional_text(
                    item.get("uncertainty"),
                    "uncertainty",
                ),
            )
            for item in _record_list(
                raw_state,
                "evidence",
            )
        ]

        claim_records = [
            ClaimRecord(
                claim_id=_required_text(
                    item,
                    "claim_id",
                ),
                statement=_required_text(
                    item,
                    "statement",
                ),
                status=ClaimStatus(
                    item["status"]
                ),
                evidence_ids=_text_tuple(
                    item.get(
                        "evidence_ids",
                        [],
                    ),
                    "evidence_ids",
                ),
            )
            for item in _record_list(
                raw_state,
                "claims",
            )
        ]

        subgoal_records = [
            SubgoalRecord(
                subgoal_id=_required_text(
                    item,
                    "subgoal_id",
                ),
                description=_required_text(
                    item,
                    "description",
                ),
                status=SubgoalStatus(
                    item["status"]
                ),
                claim_ids=_text_tuple(
                    item.get(
                        "claim_ids",
                        [],
                    ),
                    "claim_ids",
                ),
            )
            for item in _record_list(
                raw_state,
                "subgoals",
            )
        ]

        side_effect_records = [
            SideEffectRecord(
                side_effect_id=_required_text(
                    item,
                    "side_effect_id",
                ),
                description=_required_text(
                    item,
                    "description",
                ),
                state=SideEffectState(
                    item["state"]
                ),
                external_reference=_optional_text(
                    item.get(
                        "external_reference"
                    ),
                    "external_reference",
                ),
                evidence_ids=_text_tuple(
                    item.get(
                        "evidence_ids",
                        [],
                    ),
                    "evidence_ids",
                ),
                idempotent=_optional_bool(
                    item.get("idempotent"),
                    "idempotent",
                ),
                action_key=_optional_text(
                    item.get("action_key"),
                    "action_key",
                ),
            )
            for item in _record_list(
                raw_state,
                "side_effects",
            )
        ]

        artifact_records = [
            ArtifactRecord(
                artifact_id=_required_text(
                    item,
                    "artifact_id",
                ),
                description=_required_text(
                    item,
                    "description",
                ),
                location=_required_text(
                    item,
                    "location",
                ),
                evidence_ids=_text_tuple(
                    item.get(
                        "evidence_ids",
                        [],
                    ),
                    "evidence_ids",
                ),
            )
            for item in _record_list(
                raw_state,
                "artifacts",
            )
        ]

        state = TaskState(
            task_id=_required_text(
                raw_state,
                "task_id",
            ),
            goal=_required_text(
                raw_state,
                "goal",
            ),
            constraints=_text_tuple(
                raw_state.get(
                    "constraints",
                    [],
                ),
                "constraints",
            ),
            status=TaskStateStatus(
                raw_state["status"]
            ),
            subgoals=_index_unique(
                subgoal_records,
                "subgoal_id",
            ),
            claims=_index_unique(
                claim_records,
                "claim_id",
            ),
            evidence=_index_unique(
                evidence_records,
                "evidence_id",
            ),
            side_effects=_index_unique(
                side_effect_records,
                "side_effect_id",
            ),
            artifacts=_index_unique(
                artifact_records,
                "artifact_id",
            ),
            pending_questions=_text_tuple(
                raw_state.get(
                    "pending_questions",
                    [],
                ),
                "pending_questions",
            ),
            created_at=_parse_datetime(
                raw_state["created_at"],
                "created_at",
            ),
            updated_at=_parse_datetime(
                raw_state["updated_at"],
                "updated_at",
            ),
        )
    except TaskStatePersistenceError:
        raise
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise TaskStatePersistenceError(
            "invalid persisted task state: "
            f"{error}"
        ) from error

    _validate_references(state)

    return state


def _record_list(
    mapping: dict[str, Any],
    key: str,
) -> list[dict[str, Any]]:
    value = mapping.get(key, [])

    if not isinstance(value, list):
        raise TaskStatePersistenceError(
            f"{key} must be a list"
        )

    for item in value:
        if not isinstance(item, dict):
            raise TaskStatePersistenceError(
                f"{key} entries must be dictionaries"
            )

    return value


def _required_text(
    mapping: dict[str, Any],
    key: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise TaskStatePersistenceError(
            f"{key} must be a non-empty string"
        )

    return value


def _optional_text(
    value: Any,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise TaskStatePersistenceError(
            f"{field_name} must be a non-empty string or null"
        )

    return value


def _optional_bool(
    value: Any,
    field_name: str,
) -> bool | None:
    if value is None:
        return None

    if not isinstance(value, bool):
        raise TaskStatePersistenceError(
            f"{field_name} must be a boolean or null"
        )

    return value


def _text_tuple(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskStatePersistenceError(
            f"{field_name} must be a list"
        )

    result: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise TaskStatePersistenceError(
                f"{field_name} entries must be non-empty strings"
            )
        result.append(item)

    return tuple(result)


def _parse_datetime(
    value: Any,
    field_name: str,
) -> datetime:
    if not isinstance(value, str):
        raise TaskStatePersistenceError(
            f"{field_name} must be an ISO datetime string"
        )

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise TaskStatePersistenceError(
            f"{field_name} is not a valid ISO datetime"
        ) from error

    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
    ):
        raise TaskStatePersistenceError(
            f"{field_name} must be timezone-aware"
        )

    return parsed


def _index_unique(
    records: list[Any],
    id_attribute: str,
) -> dict[str, Any]:
    indexed: dict[str, Any] = {}

    for record in records:
        record_id = getattr(
            record,
            id_attribute,
        )

        if record_id in indexed:
            raise TaskStatePersistenceError(
                f"duplicate persisted ID: {record_id}"
            )

        indexed[record_id] = record

    return indexed


def _validate_references(
    state: TaskState,
) -> None:
    evidence_ids = set(state.evidence)
    claim_ids = set(state.claims)

    for claim in state.claims.values():
        missing = (
            set(claim.evidence_ids)
            - evidence_ids
        )
        if missing:
            raise TaskStatePersistenceError(
                "claim references missing evidence: "
                + ", ".join(sorted(missing))
            )

    for subgoal in state.subgoals.values():
        missing = (
            set(subgoal.claim_ids)
            - claim_ids
        )
        if missing:
            raise TaskStatePersistenceError(
                "subgoal references missing claims: "
                + ", ".join(sorted(missing))
            )

    for effect in state.side_effects.values():
        missing = (
            set(effect.evidence_ids)
            - evidence_ids
        )
        if missing:
            raise TaskStatePersistenceError(
                "side effect references missing evidence: "
                + ", ".join(sorted(missing))
            )

    for artifact in state.artifacts.values():
        missing = (
            set(artifact.evidence_ids)
            - evidence_ids
        )
        if missing:
            raise TaskStatePersistenceError(
                "artifact references missing evidence: "
                + ", ".join(sorted(missing))
            )


class TaskStateStore:
    """Persist TaskState checkpoints as atomic versioned JSON files."""

    def __init__(
        self,
        directory: str | Path,
    ) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        """Return the checkpoint directory."""
        return self._directory

    def checkpoint_path(
        self,
        task_id: str,
    ) -> Path:
        """Return the stable checkpoint path for one task ID."""
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(
                "task_id must be a non-empty string"
            )

        digest = sha256(
            task_id.encode("utf-8")
        ).hexdigest()

        return self._directory / f"{digest}.json"

    def save(
        self,
        state: TaskState,
    ) -> Path:
        """Atomically save one task-state checkpoint."""
        if not isinstance(state, TaskState):
            raise ValueError(
                "state must be a TaskState"
            )

        try:
            self._directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as error:
            raise TaskStatePersistenceError(
                "failed to create checkpoint directory: "
                f"{error}"
            ) from error

        destination = self.checkpoint_path(
            state.task_id
        )
        payload = task_state_to_payload(state)

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=".task-state-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)

                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(
                temporary_path,
                destination,
            )
        except (
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise TaskStatePersistenceError(
                "failed to save task checkpoint: "
                f"{error}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(
                        missing_ok=True
                    )
                except OSError:
                    pass

        return destination

    def load(
        self,
        task_id: str,
    ) -> TaskState:
        """Load and validate one persisted task-state checkpoint."""
        path = self.checkpoint_path(task_id)

        try:
            raw_text = path.read_text(
                encoding="utf-8"
            )
        except FileNotFoundError:
            raise
        except OSError as error:
            raise TaskStatePersistenceError(
                "failed to read task checkpoint: "
                f"{error}"
            ) from error

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise TaskStatePersistenceError(
                "task checkpoint contains invalid JSON"
            ) from error

        if not isinstance(payload, dict):
            raise TaskStatePersistenceError(
                "task checkpoint root must be a dictionary"
            )

        state = task_state_from_payload(payload)

        if state.task_id != task_id:
            raise TaskStatePersistenceError(
                "checkpoint task_id does not match "
                "the requested task"
            )

        return state

    def latest_resumable_task_id(
        self,
    ) -> str | None:
        """Return the newest non-terminal persisted task ID."""
        if not self._directory.is_dir():
            return None

        candidates: list[TaskState] = []

        try:
            paths = tuple(
                self._directory.glob("*.json")
            )
        except OSError as error:
            raise TaskStatePersistenceError(
                "failed to inspect checkpoint directory: "
                f"{error}"
            ) from error

        for path in paths:
            try:
                raw_text = path.read_text(
                    encoding="utf-8"
                )
            except OSError as error:
                raise TaskStatePersistenceError(
                    "failed to inspect task checkpoint: "
                    f"{error}"
                ) from error

            try:
                payload = json.loads(
                    raw_text
                )
            except json.JSONDecodeError as error:
                raise TaskStatePersistenceError(
                    "task checkpoint contains invalid JSON"
                ) from error

            if not isinstance(
                payload,
                dict,
            ):
                raise TaskStatePersistenceError(
                    "task checkpoint root must be a dictionary"
                )

            state = task_state_from_payload(
                payload
            )

            expected_path = self.checkpoint_path(
                state.task_id
            )

            if path.name != expected_path.name:
                raise TaskStatePersistenceError(
                    "checkpoint filename does not match "
                    "its persisted task_id"
                )

            if state.status in (
                TaskStateStatus.COMPLETED,
                TaskStateStatus.CANCELLED,
            ):
                continue

            candidates.append(
                state
            )

        if not candidates:
            return None

        latest = max(
            candidates,
            key=lambda state: (
                state.updated_at,
                state.task_id,
            ),
        )

        return latest.task_id

    def exists(
        self,
        task_id: str,
    ) -> bool:
        """Return whether a checkpoint exists for one task."""
        return self.checkpoint_path(
            task_id
        ).is_file()

