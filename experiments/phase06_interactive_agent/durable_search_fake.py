"""Deterministic fake browser harness for durable search experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from computer_agent.agent import (
    AgentLoopResult,
    AgentLoopStatus,
    AgentState,
    TextInputObservation,
)
from computer_agent.app.live_web_worker import (
    BROWSER_WINDOW_MARKER_PREFIX,
    FOLLOWUP_TARGET_ARTIFACT_ID,
    QUERY_ARTIFACT_ID,
    QueryVerificationMode,
    create_live_web_worker,
    resolve_live_web_task,
    _ensure_followup_identity,
    _ensure_live_task_structure,
    _ensure_query_identity,
    _ensure_workflow_identity,
)
from computer_agent.perception import (
    BoundingBox,
    PerceptionSnapshot,
    ScreenFrame,
    UIElement,
)
from computer_agent.planning import (
    PlanOperation,
    WebTextInputStep,
)
from computer_agent.runtime import RuntimeTask
from computer_agent.task import (
    ArtifactRecord,
    TaskState,
    TaskStateStatus,
    TaskStateTransitions,
)


@dataclass(frozen=True, slots=True)
class FakeRunResult:
    completed: bool
    opened: int
    activated: int
    typed: int
    clicked: int
    followup_clicked: int
    query_identity: str | None
    followup_identity: str | None


class FakeControl:
    def checkpoint(self) -> None:
        return


class FakeSearchEnvironment:
    def __init__(
        self,
        *,
        capture_path,
        spec,
        query_text: str,
        followup_target_text: str | None,
        mode: str,
    ) -> None:
        del capture_path
        self.spec = spec
        self.query_text = query_text
        self.followup_target_text = followup_target_text
        self.mode = mode
        self.open_count = 0
        self.activate_count = 0
        self.type_count = 0
        self.click_count = 0
        self.followup_click_count = 0

    def open_task_site(self, start_url: str, task_marker: str) -> str:
        if start_url != self.spec.start_url:
            raise RuntimeError("unexpected start URL")
        self.open_count += 1
        return BROWSER_WINDOW_MARKER_PREFIX + task_marker

    def activate_task_chrome_window(
        self,
        marker_url: str,
        working_url_prefix: str,
    ) -> None:
        if not marker_url.startswith(BROWSER_WINDOW_MARKER_PREFIX):
            raise RuntimeError("invalid task marker")
        if working_url_prefix != self.spec.working_url_prefix:
            raise RuntimeError("unexpected URL prefix")
        self.activate_count += 1

    def observe(self) -> TextInputObservation:
        return TextInputObservation(
            application_name=self.spec.expected_application,
            viewport=None,
            snapshot=_snapshot(self._elements()),
            semantic_elements=(),
        )

    def execute_plan(self, plan) -> AgentLoopResult:
        step = plan.steps[0]
        if isinstance(step, WebTextInputStep):
            if step.input_text != self.query_text:
                raise RuntimeError("wrong runtime query typed")
            self.type_count += 1
            self.mode = "query"
        elif step.operation is PlanOperation.CLICK_TARGET:
            target_text = getattr(
                step.action_target,
                "text",
                None,
            )
            if target_text == self._followup_target_text:
                self.followup_click_count += 1
                self.click_count += 1
                self.mode = "destination"
            else:
                self.click_count += 1
                self.mode = "results"
        else:
            raise RuntimeError(f"unexpected step: {step!r}")

        state = AgentState(user_task=plan.task_goal)
        state.start()
        state.succeed()
        return AgentLoopResult(
            status=AgentLoopStatus.COMPLETED,
            plan=plan,
            state=state,
            completed_plan_steps=1,
            reason="deterministic fake execution",
        )

    def _elements(self) -> tuple[UIElement, ...]:
        submit = _button(
            self.spec.submit_target.text or "",
            BoundingBox(430, 10, 70, 24),
        )
        if self.mode == "empty":
            return (
                _field(self.spec.search_field.text or "", ""),
                submit,
            )
        if self.mode == "query":
            if (
                self.spec.query_verification_mode
                is QueryVerificationMode.VISIBLE_SEARCH_UI
            ):
                return (
                    _element(
                        self.query_text,
                        self.query_text,
                        "text",
                        BoundingBox(120, 12, 230, 18),
                    ),
                    submit,
                )
            return (
                _field(
                    self.spec.search_field.text or "",
                    self.query_text,
                ),
                submit,
            )
        if self.mode == "results":
            elements = [
                _field(
                    self.spec.search_field.text or "",
                    self.query_text,
                ),
                submit,
                _element(
                    self.spec.result_target.text or "",
                    None,
                    "heading",
                    BoundingBox(10, 60, 240, 28),
                ),
            ]
            if self._followup_target_text is not None:
                elements.append(
                    _element(
                        self._followup_target_text,
                        None,
                        "link",
                        BoundingBox(20, 100, 220, 20),
                    )
                )
            return tuple(elements)
        if self.mode == "destination":
            if self._followup_target_text is None:
                raise RuntimeError("missing follow-up target")
            return (
                _element(
                    self._followup_target_text,
                    None,
                    "heading",
                    BoundingBox(10, 60, 260, 32),
                ),
            )
        if self.mode == "missing_link":
            return (
                _field(
                    self.spec.search_field.text or "",
                    self.query_text,
                ),
                submit,
                _element(
                    self.spec.result_target.text or "",
                    None,
                    "heading",
                    BoundingBox(10, 60, 240, 28),
                ),
            )
        raise RuntimeError(f"unknown mode: {self.mode}")

    @property
    def _followup_target_text(self) -> str | None:
        return self.followup_target_text


def run_fake_worker_state(
    state: TaskState,
    mode: str,
) -> FakeRunResult:
    resolved = resolve_live_web_task(state.goal)
    assert resolved is not None
    environment = FakeSearchEnvironment(
        capture_path=None,
        spec=resolved.spec,
        query_text=resolved.query_text,
        followup_target_text=(
            resolved.followup_target_text
        ),
        mode=mode,
    )
    worker = create_live_web_worker(
        state,
        lambda: None,
        lambda decision: None,
        environment_factory=lambda *, capture_path: environment,
    )
    worker(
        RuntimeTask(goal=state.goal, task_id=state.task_id),
        FakeControl(),
        lambda message: None,
    )
    query = state.artifacts.get(QUERY_ARTIFACT_ID)
    followup = state.artifacts.get(
        FOLLOWUP_TARGET_ARTIFACT_ID
    )
    return FakeRunResult(
        completed=state.status is TaskStateStatus.COMPLETED,
        opened=environment.open_count,
        activated=environment.activate_count,
        typed=environment.type_count,
        clicked=environment.click_count,
        followup_clicked=(
            environment.followup_click_count
        ),
        query_identity=None if query is None else query.location,
        followup_identity=(
            None
            if followup is None
            else followup.location
        ),
    )


def state_with_workspace(goal: str) -> TaskState:
    resolved = resolve_live_web_task(goal)
    assert resolved is not None
    state = TaskState(
        goal=goal,
        task_id=resolved.query_text.replace(" ", "-"),
        status=TaskStateStatus.PAUSED,
    )
    transitions = TaskStateTransitions(state)
    _ensure_workflow_identity(transitions, resolved.spec)
    _ensure_query_identity(transitions, resolved)
    _ensure_followup_identity(transitions, resolved)
    _ensure_live_task_structure(transitions, resolved)
    transitions.add_artifact(
        ArtifactRecord(
            artifact_id=resolved.spec.workspace_artifact_id,
            description=resolved.spec.workspace_description,
            location=BROWSER_WINDOW_MARKER_PREFIX + state.task_id,
        )
    )
    return state


def _snapshot(elements: tuple[UIElement, ...]) -> PerceptionSnapshot:
    return PerceptionSnapshot(
        frame=ScreenFrame(
            image_path=Path("fake.png"),
            pixel_width=10,
            pixel_height=10,
            screen_width=10,
            screen_height=10,
        ),
        image=Image.new("RGB", (10, 10)),
        accessibility_elements=elements,
        ocr_elements=(),
        fused_elements=elements,
        warnings=(),
    )


def _field(text: str, value: str) -> UIElement:
    return _element(text, value, "text_field", BoundingBox(10, 10, 180, 24))


def _button(text: str, box: BoundingBox) -> UIElement:
    return _element(text, None, "button", box)


def _element(
    text: str,
    value: str | None,
    element_type: str,
    box: BoundingBox,
) -> UIElement:
    return UIElement(
        element_type=element_type,
        text=text,
        value=value,
        confidence=0.95,
        enabled=True,
        bounding_box=box,
        source="accessibility",
    )
