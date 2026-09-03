"""Deterministic orchestration for executing structured plans."""

from __future__ import annotations

from computer_agent.agent.loop_models import AgentLoopResult, AgentLoopStatus
from computer_agent.agent.state import AgentState
from computer_agent.grounding.action_grounder import ActionGrounder
from computer_agent.grounding.action_models import ActionGroundingStatus
from computer_agent.grounding.models import GroundingStatus
from computer_agent.grounding.ui_grounder import UIGrounder
from computer_agent.planning.models import PlanOperation, PlanStep, StructuredPlan
from computer_agent.recovery.action_recovery import ActionRecovery
from computer_agent.recovery.models import RecoveryStatus
from computer_agent.verification.action_verifier import ActionVerifier
from computer_agent.verification.models import ActionVerificationStatus


class AgentLoop:
    """Execute a structured plan through deterministic production layers."""

    def __init__(
        self,
        *,
        perception_engine: object,
        grounder: object | None = None,
        action_grounder: object | None = None,
        executor: object,
        verifier: object | None = None,
        recovery: object | None = None,
    ) -> None:
        _require_method(
            perception_engine,
            "observe",
            "perception_engine",
        )

        _require_method(
            executor,
            "execute",
            "executor",
        )

        if grounder is None:
            grounder = UIGrounder()
        _require_method(
            grounder,
            "ground",
            "grounder",
        )

        if action_grounder is None:
            action_grounder = ActionGrounder()
        _require_method(
            action_grounder,
            "ground_click",
            "action_grounder",
        )

        if verifier is None:
            verifier = _default_verifier(grounder)
        _require_method(
            verifier,
            "verify_target_appeared",
            "verifier",
        )

        if recovery is None:
            recovery = _default_recovery(
                grounder,
                action_grounder,
            )
        _require_method(
            recovery,
            "prepare_retry",
            "recovery",
        )

        self._perception_engine = perception_engine
        self._grounder = grounder
        self._action_grounder = action_grounder
        self._executor = executor
        self._verifier = verifier
        self._recovery = recovery

    @property
    def perception_engine(self) -> object:
        """Return the perception engine used by this loop."""

        return self._perception_engine

    @property
    def grounder(self) -> object:
        """Return the UI grounder used for initial step grounding."""

        return self._grounder

    @property
    def action_grounder(self) -> object:
        """Return the action grounder used for initial click grounding."""

        return self._action_grounder

    @property
    def executor(self) -> object:
        """Return the executor used for prepared actions."""

        return self._executor

    @property
    def verifier(self) -> object:
        """Return the action verifier used by this loop."""

        return self._verifier

    @property
    def recovery(self) -> object:
        """Return the recovery policy used by this loop."""

        return self._recovery

    def run(self, plan: StructuredPlan) -> AgentLoopResult:
        """Run all plan steps until completion or a typed terminal outcome."""

        if not isinstance(plan, StructuredPlan):
            raise ValueError("plan must be a StructuredPlan")

        state = AgentState(user_task=plan.task_goal)
        state.start()
        completed_plan_steps = 0

        for step in plan.steps:
            if step.operation is not PlanOperation.CLICK_TARGET:
                raise RuntimeError(
                    f"unsupported plan operation: {step.operation}"
                )

            terminal_result = self._run_click_step(
                plan=plan,
                state=state,
                step=step,
                completed_plan_steps=completed_plan_steps,
            )

            if terminal_result is not None:
                return terminal_result

            completed_plan_steps += 1

        state.succeed()
        return AgentLoopResult(
            status=AgentLoopStatus.COMPLETED,
            plan=plan,
            state=state,
            completed_plan_steps=completed_plan_steps,
            reason="all plan steps completed",
        )

    def _run_click_step(
        self,
        *,
        plan: StructuredPlan,
        state: AgentState,
        step: PlanStep,
        completed_plan_steps: int,
    ) -> AgentLoopResult | None:
        before_snapshot = self._perception_engine.observe()

        grounding_result = self._grounder.ground(
            step.action_target,
            before_snapshot.fused_elements,
        )
        if grounding_result.status is not GroundingStatus.RESOLVED:
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "initial grounding was "
                    f"{grounding_result.status.value}: "
                    f"{grounding_result.reason}"
                ),
            )

        action_grounding_result = self._action_grounder.ground_click(
            grounding_result,
            before_snapshot.frame.screen_size,
        )
        if (
            action_grounding_result.status
            is not ActionGroundingStatus.READY
        ):
            return _terminal_failure(
                plan=plan,
                state=state,
                completed_plan_steps=completed_plan_steps,
                status=AgentLoopStatus.BLOCKED,
                reason=(
                    "initial action grounding was "
                    f"{action_grounding_result.status.value}: "
                    f"{action_grounding_result.reason}"
                ),
            )

        action = action_grounding_result.action
        if action is None:
            raise RuntimeError(
                "READY action grounding did not contain an action"
            )

        completed_attempts = 0

        while True:
            completed_attempts += 1
            tool_result = self._executor.execute(action)
            state.record_step(action, tool_result)

            after_snapshot = self._perception_engine.observe()
            verification_result = self._verifier.verify_target_appeared(
                action=action,
                tool_result=tool_result,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                target_spec=step.verification_target,
            )

            if (
                verification_result.status
                is ActionVerificationStatus.VERIFIED
            ):
                return None

            recovery_result = self._recovery.prepare_retry(
                verification_result=verification_result,
                tool_result=tool_result,
                target_spec=step.action_target,
                latest_snapshot=after_snapshot,
                completed_attempts=completed_attempts,
                max_attempts=step.max_attempts,
            )

            if recovery_result.status is RecoveryStatus.RETRY_READY:
                action_grounding_result = (
                    recovery_result.action_grounding_result
                )
                if action_grounding_result is None:
                    raise RuntimeError(
                        "RETRY_READY recovery did not contain "
                        "action grounding"
                    )

                action = action_grounding_result.action
                if action is None:
                    raise RuntimeError(
                        "RETRY_READY recovery did not contain an action"
                    )

                before_snapshot = after_snapshot
                continue

            if recovery_result.status is RecoveryStatus.BLOCKED:
                return _terminal_failure(
                    plan=plan,
                    state=state,
                    completed_plan_steps=completed_plan_steps,
                    status=AgentLoopStatus.BLOCKED,
                    reason=(
                        "recovery blocked retry: "
                        f"{recovery_result.reason}"
                    ),
                )

            if recovery_result.status is RecoveryStatus.EXHAUSTED:
                return _terminal_failure(
                    plan=plan,
                    state=state,
                    completed_plan_steps=completed_plan_steps,
                    status=AgentLoopStatus.EXHAUSTED,
                    reason=(
                        "recovery exhausted attempts: "
                        f"{recovery_result.reason}"
                    ),
                )

            if recovery_result.status is RecoveryStatus.NOT_NEEDED:
                raise RuntimeError(
                    "recovery returned NOT_NEEDED after failed verification"
                )

            raise RuntimeError(
                f"unsupported recovery status: {recovery_result.status}"
            )


def _terminal_failure(
    *,
    plan: StructuredPlan,
    state: AgentState,
    completed_plan_steps: int,
    status: AgentLoopStatus,
    reason: str,
) -> AgentLoopResult:
    state.fail(reason)
    return AgentLoopResult(
        status=status,
        plan=plan,
        state=state,
        completed_plan_steps=completed_plan_steps,
        reason=reason,
    )


def _require_method(
    dependency: object,
    method_name: str,
    dependency_name: str,
) -> None:
    if dependency is None:
        raise ValueError(f"{dependency_name} is required")

    method = getattr(
        dependency,
        method_name,
        None,
    )
    if not callable(method):
        raise ValueError(
            f"{dependency_name} must provide {method_name}()"
        )


def _default_verifier(grounder: object) -> ActionVerifier:
    if not isinstance(grounder, UIGrounder):
        raise ValueError(
            "custom grounder requires explicit verifier"
        )

    return ActionVerifier(grounder=grounder)


def _default_recovery(
    grounder: object,
    action_grounder: object,
) -> ActionRecovery:
    if not isinstance(grounder, UIGrounder):
        raise ValueError(
            "custom grounder requires explicit recovery"
        )

    if not isinstance(action_grounder, ActionGrounder):
        raise ValueError(
            "custom action_grounder requires explicit recovery"
        )

    return ActionRecovery(
        grounder=grounder,
        action_grounder=action_grounder,
    )
