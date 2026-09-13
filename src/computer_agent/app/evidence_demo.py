"""Deterministic visible demo of evidence-grounded task state."""

from __future__ import annotations

import time
from collections.abc import Callable

from computer_agent.runtime import (
    RuntimeControl,
    RuntimeTask,
    RuntimeWorker,
)
from computer_agent.task import (
    ArtifactRecord,
    ClaimRecord,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceRecord,
    SideEffectRecord,
    SubgoalRecord,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
)


TaskStatePublisher = Callable[[], None]


def create_evidence_demo_worker(
    state: TaskState,
    publish_state: TaskStatePublisher,
) -> RuntimeWorker:
    """Create the visible Experiment 06.03 semantic demo worker."""

    if not isinstance(state, TaskState):
        raise ValueError("state must be a TaskState")

    if not callable(publish_state):
        raise ValueError(
            "publish_state must be callable"
        )

    def worker(
        task: RuntimeTask,
        control: RuntimeControl,
        progress: Callable[[str], None],
    ) -> None:
        transitions = TaskStateTransitions(state)

        state.status = TaskStateStatus.RUNNING
        state.touch()
        publish_state()

        def stage(
            message: str,
            delay: float = 1.25,
        ) -> None:
            progress(message)
            publish_state()
            time.sleep(delay)
            control.checkpoint()

        control.checkpoint()

        initial = EvidenceRecord(
            summary=(
                "Current task inputs are present and "
                "ready for the external action."
            ),
            source="Deterministic demo observation",
            kind=EvidenceKind.OBSERVATION,
        )
        transitions.add_evidence(initial)

        claim = ClaimRecord(
            statement=(
                "The task is ready for the external action."
            )
        )
        transitions.add_claim(claim)
        transitions.verify_claim(
            claim.claim_id,
            (initial.evidence_id,),
        )

        subgoal = SubgoalRecord(
            description="Verify task preconditions.",
            claim_ids=(claim.claim_id,),
        )
        transitions.add_subgoal(subgoal)
        transitions.verify_subgoal(
            subgoal.subgoal_id
        )

        stage(
            "Verified the first subgoal from current evidence."
        )

        transitions.set_evidence_freshness(
            initial.evidence_id,
            EvidenceFreshness.STALE,
        )

        stage(
            "The previous observation became stale; "
            "verified progress was invalidated."
        )

        fresh = EvidenceRecord(
            summary=(
                "Fresh observation confirms the task "
                "preconditions still hold."
            ),
            source="Fresh deterministic demo observation",
            kind=EvidenceKind.VERIFICATION,
        )
        transitions.add_evidence(fresh)
        transitions.verify_claim(
            claim.claim_id,
            (fresh.evidence_id,),
        )
        transitions.verify_subgoal(
            subgoal.subgoal_id
        )

        stage(
            "Fresh evidence explicitly re-verified "
            "the subgoal."
        )

        effect = SideEffectRecord(
            description="Commit the external task result.",
            idempotent=False,
        )
        transitions.add_side_effect(effect)
        transitions.mark_side_effect_executed(
            effect.side_effect_id
        )
        transitions.mark_side_effect_unknown(
            effect.side_effect_id
        )

        stage(
            "The external action was executed, but "
            "its outcome is still UNKNOWN.",
            delay=1.8,
        )

        confirmation = EvidenceRecord(
            summary=(
                "Reconciliation observes exactly one "
                "committed external result."
            ),
            source="Deterministic reconciliation source",
            kind=EvidenceKind.VERIFICATION,
        )
        transitions.add_evidence(confirmation)
        transitions.confirm_side_effect(
            effect.side_effect_id,
            (confirmation.evidence_id,),
        )

        artifact = ArtifactRecord(
            description="Confirmed task result",
            location="demo://confirmed-result",
            evidence_ids=(
                confirmation.evidence_id,
            ),
        )
        transitions.add_artifact(artifact)

        transitions.complete_task()

        stage(
            "Reconciliation confirmed the side effect; "
            "semantic completion is now allowed.",
            delay=1.0,
        )

    return worker
