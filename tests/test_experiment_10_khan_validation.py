from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from computer_agent.agent import (
    AgentLoop,
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    AgentStatus,
)
from computer_agent.core.models import Action, ToolResult
from computer_agent.grounding import GroundingStatus, TargetSpec, UIGrounder
from computer_agent.perception import BoundingBox, PerceptionSnapshot, ScreenFrame, UIElement
from computer_agent.planning import PlanOperation, PlanStep, StructuredPlan
from computer_agent.recovery.models import RecoveryStatus
from computer_agent.reasoning import ReasoningResult, ReasoningStatus
from computer_agent.verification import (
    StateObservationStatus,
    StateObserver,
    StateTransitionVerifier,
    StateVerificationStatus,
)
from experiments.phase05_real_web_autonomy import experiment_10_khan_validation as experiment


def _box(x=100, y=140, width=260, height=36):
    return BoundingBox(x=x, y=y, width=width, height=height)


def _element(
    *,
    text,
    element_type,
    x=100,
    confidence=1.0,
    source="accessibility",
):
    return UIElement(
        element_type=element_type,
        bounding_box=_box(x=x),
        confidence=confidence,
        text=text,
        enabled=True,
        focused=False,
        source=source,
    )


def _snapshot(second, *, elements=(), warnings=()):
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path(f"synthetic-khan-e52-{second}.png"),
            pixel_width=1000,
            pixel_height=700,
            screen_width=1000,
            screen_height=700,
            captured_at=datetime(
                2026,
                9,
                10,
                16,
                30,
                second,
                tzinfo=timezone.utc,
            ),
        ),
        image=Image.new("RGB", (1000, 700)),
        accessibility_elements=tuple(elements),
        ocr_elements=(),
        fused_elements=tuple(elements),
        warnings=tuple(warnings),
    )


def _course_heading():
    return _element(text="SAT Math", element_type="heading", x=460)


def _course_link_after_navigation():
    return _element(text="SAT Math", element_type="link", x=570)


def _unit_link():
    return _element(
        text="UNIT 2 Foundations: Algebra",
        element_type="link",
        x=25,
    )


def _destination_heading(*, confidence=1.0):
    return _element(
        text="Unit 2: Foundations: Algebra",
        element_type="heading",
        x=455,
        confidence=confidence,
    )


class SequencePerception:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = 0

    def observe(self):
        self.calls += 1
        if not self.snapshots:
            raise AssertionError("unexpected perception observation")
        return self.snapshots.pop(0)


class RecordingExecutor:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def execute(self, action):
        self.calls.append(action)
        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=not self.fail,
            error="synthetic click failure" if self.fail else None,
        )


class ExplodingLegacyVerifier:
    def verify_target_appeared(self, **kwargs):
        raise AssertionError("legacy ActionVerifier must not be used")


class ExplodingRecovery:
    def prepare_retry(self, **kwargs):
        raise AssertionError("ActionRecovery must not be used")


class RecordingStateTransitionVerifier:
    def __init__(self):
        self.inner = StateTransitionVerifier(state_observer=StateObserver())
        self.calls = []

    def verify(self, **kwargs):
        result = self.inner.verify(**kwargs)
        self.calls.append({**kwargs, "result": result})
        return result


def _run_agent(*, before, after, fail_tool=False, settle_timeout=0.0):
    perception = SequencePerception((before, after))
    executor = RecordingExecutor(fail=fail_tool)
    transition = RecordingStateTransitionVerifier()
    result = AgentLoop(
        perception_engine=perception,
        grounder=UIGrounder(),
        executor=executor,
        verifier=ExplodingLegacyVerifier(),
        recovery=ExplodingRecovery(),
        state_transition_verifier=transition,
        post_action_settle_timeout_seconds=settle_timeout,
    ).run(experiment.build_khan_generic_execution_plan(experiment.khan_offline_plan()))
    return result, executor, transition


def test_khan_offline_plan_is_exact_and_accepted():
    plan = experiment.khan_offline_plan()
    assert plan.task_goal == experiment.KHAN_TASK_INTENT
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert isinstance(step, PlanStep)
    assert step.operation is PlanOperation.CLICK_TARGET
    assert step.action_target == TargetSpec(
        text="UNIT 2 Foundations: Algebra",
        element_types=(),
    )
    assert step.verification_target == TargetSpec(
        text="Unit 2: Foundations: Algebra",
        element_types=("heading",),
    )
    assert step.max_attempts == 1
    assert experiment.khan_plan_acceptance_failures(plan) == ()


def test_khan_offline_report_has_no_live_side_effects(capsys):
    report = experiment.run_khan_offline_planning()
    assert report.live_openai_request is False
    assert report.browser_actions is False
    assert report.failures == ()
    assert report.execution_plan is not None
    experiment.print_khan_offline_planning_report(report)
    output = capsys.readouterr().out
    assert "deterministic offline Khan plan" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output
    assert "Offline acceptance: passed" in output


def test_khan_trusted_conversion_is_additive_and_exact():
    planning = experiment.khan_offline_plan()
    execution = experiment.build_khan_generic_execution_plan(planning)
    assert execution is not planning
    assert execution.steps[0] is not planning.steps[0]
    assert planning.steps[0].verification_target is not None
    assert planning.steps[0].verification_spec is None

    step = execution.steps[0]
    assert step.verification_target is None
    assert step.verification_spec == experiment.khan_execution_verification_spec()
    assert step.action_target == planning.steps[0].action_target
    assert experiment.khan_execution_plan_acceptance_failures(execution) == ()


def test_khan_verification_contract_is_exact():
    spec = experiment.khan_execution_verification_spec()
    assert tuple(
        (c.target.text, c.target.element_types, c.expectation.value)
        for c in spec.before_conditions
    ) == (
        ("SAT Math", ("heading",), "present"),
        ("Unit 2: Foundations: Algebra", ("heading",), "absent"),
    )
    assert tuple(
        (c.target.text, c.target.element_types, c.expectation.value)
        for c in spec.after_conditions
    ) == (
        ("Unit 2: Foundations: Algebra", ("heading",), "present"),
        ("SAT Math", ("heading",), "absent"),
    )


def test_khan_planning_acceptance_rejects_wrong_task_goal():
    plan = experiment.khan_offline_plan()
    altered = StructuredPlan(task_goal="Open algebra.", steps=plan.steps)
    assert experiment.khan_plan_acceptance_failures(altered)


def test_khan_planning_acceptance_rejects_wrong_action_target():
    plan = StructuredPlan(
        task_goal=experiment.KHAN_TASK_INTENT,
        steps=(
            PlanStep(
                goal="Open algebra",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(text="Foundations: Algebra", element_types=()),
                verification_target=TargetSpec(
                    text=experiment.KHAN_DESTINATION_HEADING_TEXT,
                    element_types=("heading",),
                ),
                max_attempts=1,
            ),
        ),
    )
    failures = experiment.khan_plan_acceptance_failures(plan)
    assert any("action target text" in failure for failure in failures)


def test_khan_planning_acceptance_rejects_action_role_drift():
    plan = StructuredPlan(
        task_goal=experiment.KHAN_TASK_INTENT,
        steps=(
            PlanStep(
                goal="Open algebra",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=experiment.KHAN_ACTION_TARGET_TEXT,
                    element_types=("button",),
                ),
                verification_target=TargetSpec(
                    text=experiment.KHAN_DESTINATION_HEADING_TEXT,
                    element_types=("heading",),
                ),
                max_attempts=1,
            ),
        ),
    )
    failures = experiment.khan_plan_acceptance_failures(plan)
    assert any("action target element_types" in failure for failure in failures)


def test_khan_planning_acceptance_rejects_wrong_destination_heading():
    plan = StructuredPlan(
        task_goal=experiment.KHAN_TASK_INTENT,
        steps=(
            PlanStep(
                goal="Open algebra",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text=experiment.KHAN_ACTION_TARGET_TEXT,
                    element_types=(),
                ),
                verification_target=TargetSpec(
                    text="Foundations: Algebra",
                    element_types=("heading",),
                ),
                max_attempts=1,
            ),
        ),
    )
    failures = experiment.khan_plan_acceptance_failures(plan)
    assert any("verification target text" in failure for failure in failures)


def test_khan_exact_synthetic_course_to_unit_transition_completes():
    before = _snapshot(
        0,
        elements=(_course_heading(), _unit_link()),
    )
    after = _snapshot(
        1,
        elements=(_destination_heading(), _course_link_after_navigation()),
    )

    result, executor, transition = _run_agent(before=before, after=after)

    assert result.status is AgentLoopStatus.COMPLETED
    assert result.state.status is AgentStatus.SUCCEEDED
    assert result.completed_plan_steps == 1
    assert [action.tool_name for action in executor.calls] == ["click_mouse"]
    assert len(transition.calls) == 2
    final = transition.calls[-1]["result"]
    assert final.status.value == "verified"
    assert [e.observation.status for e in final.before_evaluations] == [
        StateObservationStatus.PRESENT,
        StateObservationStatus.ABSENT,
    ]
    assert [e.observation.status for e in final.after_evaluations] == [
        StateObservationStatus.PRESENT,
        StateObservationStatus.ABSENT,
    ]
    old_course_after = final.after_evaluations[1].observation
    assert old_course_after.candidates[0].element.element_type == "link"
    assert old_course_after.candidates[0].predicate_match is False
    assert "incompatible_element_type" in old_course_after.candidates[0].mismatch_reasons


def test_khan_missing_destination_fails_closed_exhausted():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(1, elements=(_course_link_after_navigation(),))
    result, executor, _ = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1


def test_khan_low_confidence_destination_blocks():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(
        1,
        elements=(
            _destination_heading(confidence=0.2),
            _course_link_after_navigation(),
        ),
    )
    result, executor, _ = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1


def test_khan_warning_bearing_missing_destination_blocks():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(
        1,
        elements=(_course_link_after_navigation(),),
        warnings=("synthetic partial perception",),
    )
    result, executor, _ = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1


def test_khan_old_course_heading_still_present_fails():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(
        1,
        elements=(_destination_heading(), _course_heading()),
    )
    result, executor, _ = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1


def test_khan_multiple_destination_headings_still_prove_present():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(
        1,
        elements=(
            _destination_heading(),
            _element(
                text="Unit 2: Foundations: Algebra",
                element_type="heading",
                x=760,
            ),
            _course_link_after_navigation(),
        ),
    )
    result, executor, transition = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.COMPLETED
    assert len(executor.calls) == 1
    observation = transition.calls[-1]["result"].after_evaluations[0].observation
    assert observation.status is StateObservationStatus.PRESENT
    assert sum(c.reliable_match for c in observation.candidates) == 2


def test_khan_action_target_missing_blocks_before_execution():
    before = _snapshot(0, elements=(_course_heading(),))
    after = _snapshot(1, elements=(_destination_heading(),))
    result, executor, _ = _run_agent(before=before, after=after)
    assert result.status is AgentLoopStatus.BLOCKED
    assert result.completed_plan_steps == 0
    assert executor.calls == []


def test_khan_failed_tool_result_exhausts_without_retry():
    before = _snapshot(0, elements=(_course_heading(), _unit_link()))
    after = _snapshot(1, elements=(_course_heading(),))
    result, executor, _ = _run_agent(
        before=before,
        after=after,
        fail_tool=True,
    )
    assert result.status is AgentLoopStatus.EXHAUSTED
    assert result.completed_plan_steps == 0
    assert len(executor.calls) == 1
    assert "generic click tool failed" in result.reason

class RecordingReasoner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def reason(self, task):
        self.calls.append(task)
        return self.result


class RecordingReasonerBuilder:
    def __init__(self, reasoner):
        self.reasoner = reasoner
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.reasoner


def _ready_reasoning_result(plan=None):
    if plan is None:
        plan = experiment.khan_offline_plan()
    return ReasoningResult(
        status=ReasoningStatus.READY,
        plan=plan,
        reason="structured plan ready",
    )


def _blocked_reasoning_result():
    return ReasoningResult(
        status=ReasoningStatus.BLOCKED,
        plan=None,
        reason="synthetic reasoning failure",
    )


def test_khan_live_openai_makes_exactly_one_reasoner_call():
    reasoner = RecordingReasoner(_ready_reasoning_result())
    builder = RecordingReasonerBuilder(reasoner)

    report = experiment.run_khan_live_openai_planning(
        reasoner_builder=builder,
    )

    assert report.live_openai_request is True
    assert report.browser_actions is False
    assert report.agent_loop_execution is False
    assert report.failures == ()
    assert builder.calls == 1
    assert reasoner.calls == [experiment.KHAN_TASK_INTENT]


def test_khan_live_openai_valid_plan_passes(capsys):
    reasoner = RecordingReasoner(_ready_reasoning_result())

    code = experiment.run_khan_live_openai_acceptance(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Experiment increment: live OpenAI Khan planning" in output
    assert "Live OpenAI request: yes" in output
    assert "Browser actions: no" in output
    assert "AgentLoop execution: no" in output
    assert "ReasoningStatus: ready" in output
    assert f"task_goal: {experiment.KHAN_TASK_INTENT}" in output
    assert "Plan steps: 1" in output
    assert "Step 1 class: PlanStep" in output
    assert "Step 1 operation: click_target" in output
    assert "Step 1 action target element_types: ()" in output
    assert "Step 1 verification target element_types: ('heading',)" in output
    assert "Planning acceptance: passed" in output


def test_khan_live_openai_invalid_plan_fails_closed(capsys):
    plan = StructuredPlan(
        task_goal=experiment.KHAN_TASK_INTENT,
        steps=(
            PlanStep(
                goal="Open algebra",
                operation=PlanOperation.CLICK_TARGET,
                action_target=TargetSpec(
                    text="Foundations: Algebra",
                    element_types=(),
                ),
                verification_target=TargetSpec(
                    text=experiment.KHAN_DESTINATION_HEADING_TEXT,
                    element_types=("heading",),
                ),
                max_attempts=1,
            ),
        ),
    )
    reasoner = RecordingReasoner(_ready_reasoning_result(plan))

    code = experiment.run_khan_live_openai_acceptance(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
    )
    output = capsys.readouterr().out

    assert code == 1
    assert "Planning acceptance: failed" in output
    assert "action target text" in output
    assert "Browser actions: no" in output
    assert "AgentLoop execution: no" in output


def test_khan_live_openai_reasoning_failure_fails_closed(capsys):
    reasoner = RecordingReasoner(_blocked_reasoning_result())

    code = experiment.run_khan_live_openai_acceptance(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
    )
    output = capsys.readouterr().out

    assert code == 1
    assert reasoner.calls == [experiment.KHAN_TASK_INTENT]
    assert "ReasoningStatus: blocked" in output
    assert "synthetic reasoning failure" in output
    assert "Planning acceptance: failed" in output
    assert "Browser actions: no" in output
    assert "AgentLoop execution: no" in output


def test_khan_live_openai_main_routes_to_planning_only(capsys):
    reasoner = RecordingReasoner(_ready_reasoning_result())
    builder = RecordingReasonerBuilder(reasoner)

    code = experiment.main(
        ["--live-openai"],
        reasoner_builder=builder,
    )
    output = capsys.readouterr().out

    assert code == 0
    assert builder.calls == 1
    assert reasoner.calls == [experiment.KHAN_TASK_INTENT]
    assert "live OpenAI Khan planning" in output
    assert "Browser actions: no" in output
    assert "AgentLoop execution: no" in output


def test_khan_default_main_remains_offline(capsys):
    code = experiment.main([])
    output = capsys.readouterr().out

    assert code == 0
    assert "deterministic offline Khan plan" in output
    assert "Live OpenAI request: no" in output
    assert "Browser actions: no" in output


class FakeAccessibility:
    available = True
    trusted = True

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def is_trusted(cls):
        return cls.trusted


class RecordingExecutorBuilder:
    def __init__(self):
        self.calls = 0
        self.executor = RecordingExecutor()

    def __call__(self):
        self.calls += 1
        return self.executor


class FakeKhanEnvironment:
    app_name = "Google Chrome"
    warning_texts = ()
    precondition_elements = (_course_heading(), _unit_link())
    final_elements = (_destination_heading(), _course_link_after_navigation())

    def __init__(
        self,
        *,
        capture_path,
        sleeper,
        stabilization_wait_seconds,
    ):
        self.capture_path = Path(capture_path)
        self.sleeper = sleeper
        self.stabilization_wait_seconds = stabilization_wait_seconds
        self.observe_calls = 0
        self.before_snapshot = _snapshot(
            11,
            elements=(_course_heading(), _unit_link()),
        )
        self.after_snapshot = _snapshot(
            12,
            elements=(
                _destination_heading(),
                _course_link_after_navigation(),
            ),
        )
        self.perception_engine = SequencePerception(
            (self.before_snapshot, self.after_snapshot)
        )

    def observe(self):
        self.observe_calls += 1
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        self.capture_path.write_text("candidate evidence")
        if self.observe_calls == 1:
            snapshot = _snapshot(
                10,
                elements=self.precondition_elements,
                warnings=self.warning_texts,
            )
        else:
            snapshot = _snapshot(13, elements=self.final_elements)
        return experiment.KhanLiveObservation(
            frontmost_app=self.app_name,
            viewport=None,
            snapshot=snapshot,
            semantic_elements=(),
        )


class RecordingEnvironmentBuilder:
    def __init__(self, env_cls=FakeKhanEnvironment):
        self.env_cls = env_cls
        self.calls = 0
        self.instances = []

    def __call__(self, **kwargs):
        self.calls += 1
        env = self.env_cls(**kwargs)
        self.instances.append(env)
        return env


class FakeAgentLoop:
    instances = []
    fail_tool = False

    def __init__(
        self,
        *,
        perception_engine,
        grounder,
        executor,
        state_transition_verifier,
        **kwargs,
    ):
        self.perception_engine = perception_engine
        self.grounder = grounder
        self.executor = executor
        self.state_transition_verifier = state_transition_verifier
        self.run_calls = []
        type(self).instances.append(self)

    def run(self, plan):
        self.run_calls.append(plan)
        before = self.perception_engine.observe()
        after = self.perception_engine.observe()
        self.state_transition_verifier.verify(
            before_snapshot=before,
            after_snapshot=after,
            verification_spec=plan.steps[0].verification_spec,
        )
        action = Action(tool_name="click_mouse", arguments={"x": 1, "y": 2})
        result = self.executor.execute(action)
        state = AgentState(user_task=plan.task_goal)
        state.start()
        state.record_step(action, result)
        if result.success and not self.fail_tool:
            state.succeed()
            return AgentLoopResult(
                status=AgentLoopStatus.COMPLETED,
                plan=plan,
                state=state,
                completed_plan_steps=1,
                reason="synthetic completion",
            )
        state.fail("synthetic failure")
        return AgentLoopResult(
            status=AgentLoopStatus.EXHAUSTED,
            plan=plan,
            state=state,
            completed_plan_steps=0,
            reason="synthetic failure",
        )


def _successful_live_report(tmp_path, **overrides):
    FakeAgentLoop.instances = []
    reasoner = RecordingReasoner(_ready_reasoning_result())
    executor_builder = RecordingExecutorBuilder()
    env_builder = RecordingEnvironmentBuilder()
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
        environment_builder=env_builder,
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
        **overrides,
    )
    return report, reasoner, env_builder, executor_builder


def test_khan_execute_requires_live_openai_and_observes_nothing(capsys):
    def exploding_runner(**kwargs):
        raise AssertionError("execution runner must not be called")

    code = experiment.main(["--execute"], live_execution_runner=exploding_runner)
    output = capsys.readouterr().out

    assert code == 2
    assert "--execute requires --live-openai" in output
    assert "Browser actions: no" in output


def test_khan_planning_failure_prevents_observation_and_execution(tmp_path):
    reasoner = RecordingReasoner(_blocked_reasoning_result())

    def exploding_environment(**kwargs):
        raise AssertionError("environment must not be constructed")

    def exploding_executor():
        raise AssertionError("executor must not be constructed")

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
        environment_builder=exploding_environment,
        executor_builder=exploding_executor,
        agent_loop_cls=FakeAgentLoop,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert reasoner.calls == [experiment.KHAN_TASK_INTENT]
    assert report.preconditions is None
    assert report.agent_result is None
    assert report.evidence_promoted is False
    assert not (tmp_path / "formal.png").exists()


def test_khan_invalid_converted_plan_prevents_execution(monkeypatch, tmp_path):
    reasoner = RecordingReasoner(_ready_reasoning_result())

    def invalid_conversion(plan):
        return StructuredPlan(task_goal="wrong task", steps=plan.steps)

    monkeypatch.setattr(
        experiment,
        "build_khan_generic_execution_plan",
        invalid_conversion,
    )

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(reasoner),
        environment_builder=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment must not be constructed")
        ),
        executor_builder=lambda: (_ for _ in ()).throw(
            AssertionError("executor must not be constructed")
        ),
        agent_loop_cls=FakeAgentLoop,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.agent_result is None
    assert any("execution plan task_goal" in f for f in report.execution_failures)
    assert report.evidence_promoted is False


def test_khan_unsupported_platform_prevents_execution(tmp_path):
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment must not be constructed")
        ),
        executor_builder=lambda: (_ for _ in ()).throw(
            AssertionError("executor must not be constructed")
        ),
        agent_loop_cls=FakeAgentLoop,
        platform_name="linux",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert report.preconditions.observation is None
    assert "platform is not macOS" in report.execution_failures
    assert report.agent_result is None


def test_khan_unavailable_accessibility_prevents_execution(tmp_path):
    class UnavailableAccessibility(FakeAccessibility):
        available = False

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment must not be constructed")
        ),
        executor_builder=lambda: (_ for _ in ()).throw(
            AssertionError("executor must not be constructed")
        ),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=UnavailableAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert "macOS Accessibility is unavailable" in report.execution_failures
    assert report.agent_result is None


def test_khan_untrusted_accessibility_prevents_execution(tmp_path):
    class UntrustedAccessibility(FakeAccessibility):
        trusted = False

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("environment must not be constructed")
        ),
        executor_builder=lambda: (_ for _ in ()).throw(
            AssertionError("executor must not be constructed")
        ),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=UntrustedAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
    )

    assert "macOS Accessibility is not trusted" in report.execution_failures
    assert report.agent_result is None


def test_khan_wrong_frontmost_app_prevents_executor_construction(tmp_path):
    class WrongAppEnvironment(FakeKhanEnvironment):
        app_name = "Safari"

    env_builder = RecordingEnvironmentBuilder(WrongAppEnvironment)
    executor_builder = RecordingExecutorBuilder()
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=env_builder,
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert env_builder.calls == 1
    assert executor_builder.calls == 0
    assert report.agent_result is None
    assert any("frontmost app" in f for f in report.execution_failures)
    assert report.evidence_promoted is False


def test_khan_before_state_contract_failure_prevents_execution(tmp_path):
    class AlreadyAtDestinationEnvironment(FakeKhanEnvironment):
        precondition_elements = (
            _course_heading(),
            _unit_link(),
            _destination_heading(),
        )

    executor_builder = RecordingExecutorBuilder()
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(
            AlreadyAtDestinationEnvironment
        ),
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert executor_builder.calls == 0
    assert report.agent_result is None
    assert any("not absent" in f for f in report.execution_failures)
    assert report.evidence_promoted is False


def test_khan_unresolved_action_target_prevents_execution(tmp_path):
    class MissingActionEnvironment(FakeKhanEnvironment):
        precondition_elements = (_course_heading(),)

    executor_builder = RecordingExecutorBuilder()
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(MissingActionEnvironment),
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert executor_builder.calls == 0
    assert report.agent_result is None
    assert any("action target grounding" in f for f in report.execution_failures)


def test_khan_ambiguous_action_target_prevents_execution(tmp_path):
    class AmbiguousActionEnvironment(FakeKhanEnvironment):
        precondition_elements = (
            _course_heading(),
            _unit_link(),
            _element(
                text="UNIT 2 Foundations: Algebra",
                element_type="link",
                x=80,
            ),
        )

    executor_builder = RecordingExecutorBuilder()
    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(
            AmbiguousActionEnvironment
        ),
        executor_builder=executor_builder,
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert executor_builder.calls == 0
    assert report.agent_result is None
    assert any("ambiguous" in f for f in report.execution_failures)


def test_khan_live_execution_success_calls_agent_loop_once(tmp_path):
    report, reasoner, env_builder, executor_builder = _successful_live_report(
        tmp_path
    )

    assert report.execution_failures == ()
    assert reasoner.calls == [experiment.KHAN_TASK_INTENT]
    assert env_builder.calls == 1
    assert executor_builder.calls == 1
    assert len(FakeAgentLoop.instances) == 1
    assert len(FakeAgentLoop.instances[0].run_calls) == 1
    assert report.agent_result.completed_plan_steps == 1
    assert [a.tool_name for a in executor_builder.executor.calls] == [
        "click_mouse"
    ]
    assert report.generic_verification_records[0].result.status is (
        StateVerificationStatus.VERIFIED
    )
    assert report.final_report.destination_heading_grounding.status is (
        GroundingStatus.RESOLVED
    )
    assert report.evidence_promoted is True
    assert (tmp_path / "formal.png").exists()


def test_khan_live_execution_records_one_click_and_one_generic_call(tmp_path):
    report, _, _, executor_builder = _successful_live_report(tmp_path)

    action_order = tuple(
        record.action.tool_name for record in report.agent_result.state.steps
    )
    assert action_order == ("click_mouse",)
    assert len(executor_builder.executor.calls) == 1
    assert len(report.generic_verification_records) == 1
    assert report.generic_verification_records[0].verification_spec == (
        experiment.khan_execution_verification_spec()
    )


def test_khan_failed_generic_verification_rejects_execution(tmp_path):
    class FailedTransitionEnvironment(FakeKhanEnvironment):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.after_snapshot = _snapshot(
                12,
                elements=(_course_heading(),),
            )
            self.perception_engine = SequencePerception(
                (self.before_snapshot, self.after_snapshot)
            )

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(
            FailedTransitionEnvironment
        ),
        executor_builder=RecordingExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert len(report.generic_verification_records) == 1
    assert any(
        "generic StateTransitionVerifier result" in f
        for f in report.execution_failures
    )
    assert report.evidence_promoted is False
    assert not (tmp_path / "formal.png").exists()


def test_khan_failed_final_destination_grounding_rejects_execution(tmp_path):
    class MissingFinalDestinationEnvironment(FakeKhanEnvironment):
        final_elements = (_course_link_after_navigation(),)

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(
            MissingFinalDestinationEnvironment
        ),
        executor_builder=RecordingExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert any(
        "final Unit 2 heading grounding" in f
        for f in report.execution_failures
    )
    assert report.evidence_promoted is False
    assert not (tmp_path / "formal.png").exists()


def test_khan_evidence_not_promoted_when_candidate_missing(tmp_path):
    class MissingCandidateEnvironment(FakeKhanEnvironment):
        def observe(self):
            observation = super().observe()
            self.capture_path.unlink(missing_ok=True)
            return observation

    report = experiment.run_khan_live_execution(
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        environment_builder=RecordingEnvironmentBuilder(
            MissingCandidateEnvironment
        ),
        executor_builder=RecordingExecutorBuilder(),
        agent_loop_cls=FakeAgentLoop,
        platform_name="darwin",
        accessibility_cls=FakeAccessibility,
        capture_path=tmp_path / "candidate.png",
        formal_evidence_path=tmp_path / "formal.png",
        sleeper=lambda _seconds: None,
        wait_seconds=0,
        stabilization_wait_seconds=0,
    )

    assert "candidate evidence file was missing" in report.execution_failures
    assert report.evidence_promoted is False
    assert not (tmp_path / "formal.png").exists()


def test_khan_cli_execute_success_and_failure_codes(tmp_path, capsys):
    def success_runner(**kwargs):
        report, *_ = _successful_live_report(tmp_path)
        return report

    success_code = experiment.main(
        ["--live-openai", "--execute", "--wait-seconds", "0"],
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        live_execution_runner=success_runner,
    )
    success_output = capsys.readouterr().out

    def failure_runner(**kwargs):
        report, *_ = _successful_live_report(tmp_path)
        return experiment.KhanExecutionReport(
            planning=report.planning,
            execution_plan=report.execution_plan,
            preconditions=report.preconditions,
            agent_result=report.agent_result,
            final_report=report.final_report,
            generic_verification_records=report.generic_verification_records,
            execution_failures=("synthetic failure",),
            evidence_promoted=False,
        )

    failure_code = experiment.main(
        ["--live-openai", "--execute", "--wait-seconds", "0"],
        reasoner_builder=RecordingReasonerBuilder(
            RecordingReasoner(_ready_reasoning_result())
        ),
        live_execution_runner=failure_runner,
    )
    failure_output = capsys.readouterr().out

    assert success_code == 0
    assert "Execution acceptance: passed" in success_output
    assert failure_code == 1
    assert "Execution acceptance: failed" in failure_output
