"""Phase 06 Experiment 03: evidence-grounded semantic task state."""

from __future__ import annotations

from dataclasses import dataclass

from computer_agent.task import (
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
    TaskStateTransitions,
)


TITLE = "Phase 06 Experiment 03: Evidence-Grounded Task State"


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Stable acceptance evidence for Experiment 06.03."""

    initial_verification_passed: bool
    stale_invalidation_passed: bool
    explicit_reverification_passed: bool
    unknown_side_effect_passed: bool
    reconciliation_passed: bool
    artifact_passed: bool
    failures: tuple[str, ...]


def run_experiment() -> ExperimentReport:
    """Run one deterministic evidence-grounded task-state workflow."""

    failures: list[str] = []

    state = TaskState(
        goal=(
            "Prepare and submit the registration packet "
            "without creating a duplicate submission."
        ),
        constraints=(
            "Do not submit more than once.",
            "Do not declare completion without evidence.",
        ),
    )
    transitions = TaskStateTransitions(state)

    initial_evidence = EvidenceRecord(
        summary=(
            "All required registration fields contain "
            "the intended values."
        ),
        source="Current registration form observation",
        kind=EvidenceKind.OBSERVATION,
    )
    transitions.add_evidence(initial_evidence)

    ready_claim = ClaimRecord(
        statement="The registration form is ready to submit."
    )
    transitions.add_claim(ready_claim)

    transitions.verify_claim(
        ready_claim.claim_id,
        (initial_evidence.evidence_id,),
    )

    preparation_subgoal = SubgoalRecord(
        description="Prepare the registration form.",
        claim_ids=(ready_claim.claim_id,),
    )
    transitions.add_subgoal(preparation_subgoal)
    transitions.verify_subgoal(
        preparation_subgoal.subgoal_id
    )

    initial_verification_passed = (
        state.claims[ready_claim.claim_id].status
        is ClaimStatus.VERIFIED
        and state.subgoals[
            preparation_subgoal.subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )

    if not initial_verification_passed:
        failures.append(
            "initial evidence did not verify claim and subgoal"
        )

    transitions.set_evidence_freshness(
        initial_evidence.evidence_id,
        EvidenceFreshness.STALE,
    )

    stale_invalidation_passed = (
        state.claims[ready_claim.claim_id].status
        is ClaimStatus.UNKNOWN
        and state.subgoals[
            preparation_subgoal.subgoal_id
        ].status
        is SubgoalStatus.UNKNOWN
    )

    if not stale_invalidation_passed:
        failures.append(
            "stale evidence did not invalidate dependent progress"
        )

    fresh_evidence = EvidenceRecord(
        summary=(
            "Fresh observation confirms all required fields "
            "still contain the intended values."
        ),
        source="Fresh registration form observation",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(fresh_evidence)

    claim_still_unknown_before_reverification = (
        state.claims[ready_claim.claim_id].status
        is ClaimStatus.UNKNOWN
    )

    transitions.verify_claim(
        ready_claim.claim_id,
        (fresh_evidence.evidence_id,),
    )
    transitions.verify_subgoal(
        preparation_subgoal.subgoal_id
    )

    explicit_reverification_passed = (
        claim_still_unknown_before_reverification
        and state.claims[
            ready_claim.claim_id
        ].status
        is ClaimStatus.VERIFIED
        and state.subgoals[
            preparation_subgoal.subgoal_id
        ].status
        is SubgoalStatus.VERIFIED
    )

    if not explicit_reverification_passed:
        failures.append(
            "fresh evidence bypassed or failed explicit re-verification"
        )

    submit_effect = SideEffectRecord(
        description="Submit the registration packet.",
        idempotent=False,
    )
    transitions.add_side_effect(submit_effect)

    transitions.mark_side_effect_executed(
        submit_effect.side_effect_id
    )
    transitions.mark_side_effect_unknown(
        submit_effect.side_effect_id
    )

    unknown_side_effect_passed = (
        state.side_effects[
            submit_effect.side_effect_id
        ].state
        is SideEffectState.UNKNOWN
    )

    if not unknown_side_effect_passed:
        failures.append(
            "uncertain submission outcome was not preserved as UNKNOWN"
        )

    confirmation_evidence = EvidenceRecord(
        summary=(
            "Server-visible submission history shows exactly "
            "one accepted submission with ID REG-42."
        ),
        source="Registration submission history",
        kind=EvidenceKind.VERIFICATION,
    )
    transitions.add_evidence(
        confirmation_evidence
    )

    transitions.confirm_side_effect(
        submit_effect.side_effect_id,
        (confirmation_evidence.evidence_id,),
    )

    reconciliation_passed = (
        state.side_effects[
            submit_effect.side_effect_id
        ].state
        is SideEffectState.CONFIRMED
        and state.side_effects[
            submit_effect.side_effect_id
        ].evidence_ids
        == (confirmation_evidence.evidence_id,)
    )

    if not reconciliation_passed:
        failures.append(
            "reconciliation evidence did not confirm side effect"
        )

    receipt = ArtifactRecord(
        description="Registration confirmation receipt.",
        location="/tmp/REG-42-confirmation.pdf",
        evidence_ids=(
            confirmation_evidence.evidence_id,
        ),
    )
    transitions.add_artifact(receipt)

    artifact_passed = (
        state.artifacts[receipt.artifact_id]
        == receipt
    )

    if not artifact_passed:
        failures.append(
            "confirmed artifact was not recorded"
        )

    return ExperimentReport(
        initial_verification_passed=initial_verification_passed,
        stale_invalidation_passed=stale_invalidation_passed,
        explicit_reverification_passed=(
            explicit_reverification_passed
        ),
        unknown_side_effect_passed=(
            unknown_side_effect_passed
        ),
        reconciliation_passed=(
            reconciliation_passed
        ),
        artifact_passed=artifact_passed,
        failures=tuple(failures),
    )


def print_report(
    report: ExperimentReport,
) -> None:
    """Print stable acceptance evidence."""

    print(TITLE)
    print(
        "Initial evidence verification: "
        f"{_passed(report.initial_verification_passed)}"
    )
    print(
        "Stale evidence invalidation: "
        f"{_passed(report.stale_invalidation_passed)}"
    )
    print(
        "Explicit re-verification: "
        f"{_passed(report.explicit_reverification_passed)}"
    )
    print(
        "Unknown side-effect preservation: "
        f"{_passed(report.unknown_side_effect_passed)}"
    )
    print(
        "Side-effect reconciliation: "
        f"{_passed(report.reconciliation_passed)}"
    )
    print(
        "Artifact recording: "
        f"{_passed(report.artifact_passed)}"
    )

    if report.failures:
        print("Experiment acceptance: failed")

        for failure in report.failures:
            print(f"- {failure}")

        return

    print("Experiment acceptance: passed")


def _passed(value: bool) -> str:
    return "passed" if value else "failed"


def main() -> int:
    report = run_experiment()
    print_report(report)
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
