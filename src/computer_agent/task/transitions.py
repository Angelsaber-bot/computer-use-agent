"""Deterministic transitions for evidence-grounded task state."""

from __future__ import annotations

from dataclasses import replace

from computer_agent.task.models import (
    ArtifactRecord,
    ClaimRecord,
    ClaimStatus,
    EvidenceFreshness,
    EvidenceRecord,
    SideEffectRecord,
    SideEffectState,
    SubgoalRecord,
    SubgoalStatus,
    TaskState,
    TaskStateStatus,
)


class TaskStateTransitions:
    """Apply validated semantic transitions to one TaskState."""

    def __init__(self, state: TaskState) -> None:
        if not isinstance(state, TaskState):
            raise ValueError("state must be a TaskState")

        self._state = state

    @property
    def state(self) -> TaskState:
        return self._state

    def add_evidence(
        self,
        record: EvidenceRecord,
    ) -> EvidenceRecord:
        """Add one evidence record."""

        if not isinstance(record, EvidenceRecord):
            raise ValueError(
                "record must be an EvidenceRecord"
            )

        if record.evidence_id in self._state.evidence:
            raise ValueError(
                f"duplicate evidence_id: {record.evidence_id}"
            )

        self._state.evidence[record.evidence_id] = record
        self._state.touch()
        return record

    def set_evidence_freshness(
        self,
        evidence_id: str,
        freshness: EvidenceFreshness,
    ) -> EvidenceRecord:
        """Change evidence freshness and invalidate unsafe dependents."""

        record = self._get_evidence(evidence_id)

        if not isinstance(
            freshness,
            EvidenceFreshness,
        ):
            raise ValueError(
                "freshness must be an EvidenceFreshness"
            )

        if (
            record.freshness
            is not EvidenceFreshness.CURRENT
            and freshness is EvidenceFreshness.CURRENT
        ):
            raise ValueError(
                "stale or invalidated evidence cannot become current; "
                "create a new EvidenceRecord from a fresh observation"
            )

        updated = replace(
            record,
            freshness=freshness,
        )
        self._state.evidence[evidence_id] = updated

        if freshness is not EvidenceFreshness.CURRENT:
            self._invalidate_unsupported_claims()

        self._state.touch()
        return updated

    def add_claim(
        self,
        record: ClaimRecord,
    ) -> ClaimRecord:
        """Add one semantic claim."""

        if not isinstance(record, ClaimRecord):
            raise ValueError(
                "record must be a ClaimRecord"
            )

        if record.claim_id in self._state.claims:
            raise ValueError(
                f"duplicate claim_id: {record.claim_id}"
            )

        for evidence_id in record.evidence_ids:
            self._get_evidence(evidence_id)

        if record.status is ClaimStatus.VERIFIED:
            self._require_current_evidence(
                record.evidence_ids
            )

        self._state.claims[record.claim_id] = record
        self._state.touch()
        return record

    def verify_claim(
        self,
        claim_id: str,
        evidence_ids: tuple[str, ...],
    ) -> ClaimRecord:
        """Verify a claim using explicit current evidence."""

        claim = self._get_claim(claim_id)

        self._require_current_evidence(
            evidence_ids
        )

        updated = replace(
            claim,
            status=ClaimStatus.VERIFIED,
            evidence_ids=evidence_ids,
        )

        self._state.claims[claim_id] = updated
        self._state.touch()
        return updated

    def mark_claim_unknown(
        self,
        claim_id: str,
    ) -> ClaimRecord:
        """Mark a claim unresolved without deleting its history."""

        claim = self._get_claim(claim_id)

        updated = replace(
            claim,
            status=ClaimStatus.UNKNOWN,
        )

        self._state.claims[claim_id] = updated
        self._invalidate_verified_subgoals()
        self._state.touch()
        return updated

    def add_subgoal(
        self,
        record: SubgoalRecord,
    ) -> SubgoalRecord:
        """Add one semantic subgoal."""

        if not isinstance(record, SubgoalRecord):
            raise ValueError(
                "record must be a SubgoalRecord"
            )

        if record.subgoal_id in self._state.subgoals:
            raise ValueError(
                f"duplicate subgoal_id: {record.subgoal_id}"
            )

        for claim_id in record.claim_ids:
            self._get_claim(claim_id)

        if record.status is SubgoalStatus.VERIFIED:
            self._require_verified_claims(
                record.claim_ids
            )

        self._state.subgoals[
            record.subgoal_id
        ] = record
        self._state.touch()
        return record

    def verify_subgoal(
        self,
        subgoal_id: str,
    ) -> SubgoalRecord:
        """Verify a subgoal only from currently verified claims."""

        subgoal = self._get_subgoal(subgoal_id)

        self._require_verified_claims(
            subgoal.claim_ids
        )

        updated = replace(
            subgoal,
            status=SubgoalStatus.VERIFIED,
        )

        self._state.subgoals[
            subgoal_id
        ] = updated
        self._state.touch()
        return updated

    def add_side_effect(
        self,
        record: SideEffectRecord,
    ) -> SideEffectRecord:
        """Add one externally consequential side effect."""

        if not isinstance(
            record,
            SideEffectRecord,
        ):
            raise ValueError(
                "record must be a SideEffectRecord"
            )

        if (
            record.side_effect_id
            in self._state.side_effects
        ):
            raise ValueError(
                "duplicate side_effect_id: "
                f"{record.side_effect_id}"
            )

        for evidence_id in record.evidence_ids:
            self._get_evidence(evidence_id)

        self._state.side_effects[
            record.side_effect_id
        ] = record
        self._state.touch()
        return record

    def mark_side_effect_executed(
        self,
        side_effect_id: str,
    ) -> SideEffectRecord:
        """Record that execution occurred but is not yet confirmed."""

        record = self._get_side_effect(
            side_effect_id
        )

        if record.state is not SideEffectState.INTENDED:
            raise RuntimeError(
                "side effect must be intended before execution"
            )

        updated = replace(
            record,
            state=SideEffectState.EXECUTED,
        )

        self._state.side_effects[
            side_effect_id
        ] = updated
        self._state.touch()
        return updated

    def mark_side_effect_unknown(
        self,
        side_effect_id: str,
    ) -> SideEffectRecord:
        """Record uncertain outcome after execution."""

        record = self._get_side_effect(
            side_effect_id
        )

        if record.state is not SideEffectState.EXECUTED:
            raise RuntimeError(
                "side effect must be executed before becoming unknown"
            )

        updated = replace(
            record,
            state=SideEffectState.UNKNOWN,
        )

        self._state.side_effects[
            side_effect_id
        ] = updated
        self._state.touch()
        return updated

    def confirm_side_effect(
        self,
        side_effect_id: str,
        evidence_ids: tuple[str, ...],
    ) -> SideEffectRecord:
        """Confirm an executed or unknown side effect from evidence."""

        record = self._get_side_effect(
            side_effect_id
        )

        if record.state not in (
            SideEffectState.EXECUTED,
            SideEffectState.UNKNOWN,
        ):
            raise RuntimeError(
                "side effect must be executed or unknown "
                "before confirmation"
            )

        self._require_current_evidence(
            evidence_ids
        )

        updated = replace(
            record,
            state=SideEffectState.CONFIRMED,
            evidence_ids=evidence_ids,
        )

        self._state.side_effects[
            side_effect_id
        ] = updated
        self._state.touch()
        return updated

    def fail_side_effect(
        self,
        side_effect_id: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> SideEffectRecord:
        """Mark an executed or unknown side effect as failed."""

        record = self._get_side_effect(
            side_effect_id
        )

        if record.state not in (
            SideEffectState.EXECUTED,
            SideEffectState.UNKNOWN,
        ):
            raise RuntimeError(
                "side effect must be executed or unknown "
                "before failure"
            )

        if evidence_ids:
            self._require_current_evidence(
                evidence_ids
            )

        updated = replace(
            record,
            state=SideEffectState.FAILED,
            evidence_ids=evidence_ids,
        )

        self._state.side_effects[
            side_effect_id
        ] = updated
        self._state.touch()
        return updated

    def add_artifact(
        self,
        record: ArtifactRecord,
    ) -> ArtifactRecord:
        """Add one task-relevant external artifact."""

        if not isinstance(record, ArtifactRecord):
            raise ValueError(
                "record must be an ArtifactRecord"
            )

        if record.artifact_id in self._state.artifacts:
            raise ValueError(
                f"duplicate artifact_id: {record.artifact_id}"
            )

        if record.evidence_ids:
            self._require_current_evidence(
                record.evidence_ids
            )

        self._state.artifacts[
            record.artifact_id
        ] = record
        self._state.touch()
        return record

    def completion_blockers(self) -> tuple[str, ...]:
        """Return deterministic reasons the task cannot be completed."""

        blockers: list[str] = []

        if not self._state.subgoals:
            blockers.append("task has no subgoals")

        for subgoal in self._state.subgoals.values():
            if subgoal.status is not SubgoalStatus.VERIFIED:
                blockers.append(
                    "subgoal is not verified: "
                    f"{subgoal.description}"
                )

        for effect in self._state.side_effects.values():
            if effect.state in (
                SideEffectState.INTENDED,
                SideEffectState.EXECUTED,
                SideEffectState.UNKNOWN,
            ):
                blockers.append(
                    "side effect is unresolved: "
                    f"{effect.description}"
                )

            if effect.state is SideEffectState.FAILED:
                blockers.append(
                    "side effect failed: "
                    f"{effect.description}"
                )

        for question in self._state.pending_questions:
            blockers.append(
                f"pending user question: {question}"
            )

        return tuple(blockers)

    def can_complete(self) -> bool:
        """Return whether semantic evidence permits completion."""

        return not self.completion_blockers()

    def complete_task(self) -> None:
        """Complete the task only when no semantic blocker remains."""

        blockers = self.completion_blockers()

        if blockers:
            raise RuntimeError(
                "task cannot complete: "
                + "; ".join(blockers)
            )

        self._state.status = TaskStateStatus.COMPLETED
        self._state.touch()

    def add_pending_question(
        self,
        question: str,
    ) -> None:
        """Add a user-facing unresolved question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "question must be a non-empty string"
            )

        normalized = question.strip()

        if normalized in self._state.pending_questions:
            return

        self._state.pending_questions = (
            *self._state.pending_questions,
            normalized,
        )
        self._state.touch()

    def resolve_pending_question(
        self,
        question: str,
    ) -> bool:
        """Remove one previously unresolved question."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "question must be a non-empty string"
            )

        normalized = question.strip()

        if normalized not in self._state.pending_questions:
            return False

        self._state.pending_questions = tuple(
            item
            for item in self._state.pending_questions
            if item != normalized
        )
        self._state.touch()
        return True

    def _require_current_evidence(
        self,
        evidence_ids: tuple[str, ...],
    ) -> None:
        if not evidence_ids:
            raise ValueError(
                "at least one evidence_id is required"
            )

        for evidence_id in evidence_ids:
            record = self._get_evidence(
                evidence_id
            )

            if (
                record.freshness
                is not EvidenceFreshness.CURRENT
            ):
                raise ValueError(
                    "evidence is not current: "
                    f"{evidence_id}"
                )

    def _require_verified_claims(
        self,
        claim_ids: tuple[str, ...],
    ) -> None:
        if not claim_ids:
            raise ValueError(
                "at least one claim_id is required"
            )

        for claim_id in claim_ids:
            claim = self._get_claim(
                claim_id
            )

            if claim.status is not ClaimStatus.VERIFIED:
                raise ValueError(
                    "claim is not verified: "
                    f"{claim_id}"
                )

            self._require_current_evidence(
                claim.evidence_ids
            )

    def _invalidate_unsupported_claims(
        self,
    ) -> None:
        changed = False

        for claim_id, claim in tuple(
            self._state.claims.items()
        ):
            if claim.status is not ClaimStatus.VERIFIED:
                continue

            if self._claim_is_currently_supported(
                claim
            ):
                continue

            self._state.claims[claim_id] = replace(
                claim,
                status=ClaimStatus.UNKNOWN,
            )
            changed = True

        if changed:
            self._invalidate_verified_subgoals()

    def _invalidate_verified_subgoals(
        self,
    ) -> None:
        for subgoal_id, subgoal in tuple(
            self._state.subgoals.items()
        ):
            if (
                subgoal.status
                is not SubgoalStatus.VERIFIED
            ):
                continue

            if self._subgoal_is_currently_supported(
                subgoal
            ):
                continue

            self._state.subgoals[
                subgoal_id
            ] = replace(
                subgoal,
                status=SubgoalStatus.UNKNOWN,
            )

    def _claim_is_currently_supported(
        self,
        claim: ClaimRecord,
    ) -> bool:
        if not claim.evidence_ids:
            return False

        for evidence_id in claim.evidence_ids:
            evidence = self._state.evidence.get(
                evidence_id
            )

            if evidence is None:
                return False

            if (
                evidence.freshness
                is not EvidenceFreshness.CURRENT
            ):
                return False

        return True

    def _subgoal_is_currently_supported(
        self,
        subgoal: SubgoalRecord,
    ) -> bool:
        if not subgoal.claim_ids:
            return False

        for claim_id in subgoal.claim_ids:
            claim = self._state.claims.get(
                claim_id
            )

            if claim is None:
                return False

            if claim.status is not ClaimStatus.VERIFIED:
                return False

            if not self._claim_is_currently_supported(
                claim
            ):
                return False

        return True

    def _get_evidence(
        self,
        evidence_id: str,
    ) -> EvidenceRecord:
        try:
            return self._state.evidence[
                evidence_id
            ]
        except KeyError as error:
            raise KeyError(
                f"unknown evidence_id: {evidence_id}"
            ) from error

    def _get_claim(
        self,
        claim_id: str,
    ) -> ClaimRecord:
        try:
            return self._state.claims[
                claim_id
            ]
        except KeyError as error:
            raise KeyError(
                f"unknown claim_id: {claim_id}"
            ) from error

    def _get_subgoal(
        self,
        subgoal_id: str,
    ) -> SubgoalRecord:
        try:
            return self._state.subgoals[
                subgoal_id
            ]
        except KeyError as error:
            raise KeyError(
                f"unknown subgoal_id: {subgoal_id}"
            ) from error

    def _get_side_effect(
        self,
        side_effect_id: str,
    ) -> SideEffectRecord:
        try:
            return self._state.side_effects[
                side_effect_id
            ]
        except KeyError as error:
            raise KeyError(
                "unknown side_effect_id: "
                f"{side_effect_id}"
            ) from error
