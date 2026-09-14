"""Tests for LLM-generated durable plan proposals."""

from __future__ import annotations

import json

import pytest

from computer_agent.app.durable_planning import (
    MAX_DURABLE_PROPOSAL_STEPS,
    DurablePlanProposal,
    DurableStepProposal,
    LLMDurablePlanner,
    compile_plan_proposal,
    default_durable_planning_context,
    parse_plan_proposal_json,
)
from computer_agent.app.live_web_worker import (
    CRASH_ENV_VAR,
    DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID,
    DURABLE_PLAN_ARTIFACT_ID,
    DurableStepKind,
    LiveCrashPoint,
    TaskState,
    durable_plan_canonical_json,
)
import computer_agent.app.live_web_worker as live_web_worker
from experiments.phase06_interactive_agent.durable_search_fake import (
    FakeControl,
    FakeSearchEnvironment,
    run_fake_worker_state,
)
from computer_agent.runtime import RuntimeTask


WIKI_GOAL = "Search Wikipedia for Claude Shannon."
WIKI_FOLLOWUP_GOAL = (
    "Search Wikipedia for Claude Shannon and open Information theory."
)
PYTHON_GOAL = "Search python.org for asyncio."


class FakeStructuredLLMClient:
    def __init__(
        self,
        response: str,
        *,
        model: str = "fake-durable-planner",
    ) -> None:
        self.response = response
        self.model = model
        self.calls: list[dict[str, object]] = []

    def generate_json(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return self.response

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return self.response


class FailingPlanner:
    planner_type = "llm"
    model_identifier = "failing"

    def __init__(self) -> None:
        self.calls = 0

    def plan(self, goal, context):
        del goal, context
        self.calls += 1
        raise AssertionError("planner should not be called")


def _proposal(
    workflow_id: str,
    steps: list[dict[str, object]],
) -> str:
    return json.dumps(
        {
            "version": 1,
            "workflow_id": workflow_id,
            "steps": steps,
        }
    )


def _llm_planner(
    response: str,
) -> LLMDurablePlanner:
    return LLMDurablePlanner(
        client=FakeStructuredLLMClient(
            response
        )
    )


def test_valid_wikipedia_search_proposal() -> None:
    proposal = parse_plan_proposal_json(
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
            ],
        )
    )

    assert proposal.workflow_id == "wikipedia-search"
    assert [step.kind for step in proposal.steps] == [
        "enter_text",
        "submit_search",
    ]


def test_valid_wikipedia_search_and_open_proposal() -> None:
    plan = compile_plan_proposal(
        WIKI_FOLLOWUP_GOAL,
        parse_plan_proposal_json(
            _proposal(
                "wikipedia-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "Claude Shannon",
                    },
                    {
                        "kind": "submit_search",
                    },
                    {
                        "kind": "open_link",
                        "target_text": "Information theory",
                    },
                ],
            )
        ),
    )

    assert [step.kind for step in plan.steps] == [
        DurableStepKind.ENTER_TEXT,
        DurableStepKind.ACTIVATE_CONTROL,
        DurableStepKind.OPEN_LINK,
    ]


def test_valid_python_search_proposal() -> None:
    plan = compile_plan_proposal(
        PYTHON_GOAL,
        parse_plan_proposal_json(
            _proposal(
                "python-org-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "asyncio",
                    },
                    {
                        "kind": "submit_search",
                    },
                ],
            )
        ),
    )

    assert plan.workflow_id == "python-org-search"
    assert len(plan.steps) == 2


@pytest.mark.parametrize(
    "response",
    [
        "{",
        "not json",
        "",
    ],
)
def test_malformed_json_rejected(response: str) -> None:
    with pytest.raises(ValueError):
        parse_plan_proposal_json(
            response
        )


@pytest.mark.parametrize(
    "response",
    [
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "press_button",
                },
            ],
        ),
        _proposal(
            "google-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
            ],
        ),
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "",
                },
            ],
        ),
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "submit_search",
                }
            ]
            * (MAX_DURABLE_PROPOSAL_STEPS + 1),
        ),
    ],
)
def test_invalid_proposal_schema_rejected(response: str) -> None:
    with pytest.raises(ValueError):
        parse_plan_proposal_json(
            response
        )


def test_correct_goal_and_matching_proposal_is_accepted() -> None:
    plan = compile_plan_proposal(
        WIKI_GOAL,
        DurablePlanProposal(
            version=1,
            workflow_id="wikipedia-search",
            steps=(
                DurableStepProposal(
                    kind="enter_text",
                    text="Claude Shannon",
                ),
                DurableStepProposal(
                    kind="submit_search",
                ),
            ),
        ),
    )

    assert len(plan.steps) == 2


@pytest.mark.parametrize(
    "goal,proposal",
    [
        (
            WIKI_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="enter_text",
                        text="Alan Turing",
                    ),
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                ),
            ),
        ),
        (
            WIKI_FOLLOWUP_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="enter_text",
                        text="Claude Shannon",
                    ),
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                    DurableStepProposal(
                        kind="open_link",
                        target_text="Turing machine",
                    ),
                ),
            ),
        ),
        (
            PYTHON_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="enter_text",
                        text="asyncio",
                    ),
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                ),
            ),
        ),
        (
            WIKI_FOLLOWUP_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="enter_text",
                        text="Claude Shannon",
                    ),
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                ),
            ),
        ),
        (
            WIKI_GOAL,
            DurablePlanProposal(
                version=1,
                workflow_id="wikipedia-search",
                steps=(
                    DurableStepProposal(
                        kind="submit_search",
                    ),
                    DurableStepProposal(
                        kind="enter_text",
                        text="Claude Shannon",
                    ),
                ),
            ),
        ),
    ],
)
def test_proposal_acceptance_rejects_mismatch(
    goal: str,
    proposal: DurablePlanProposal,
) -> None:
    with pytest.raises(ValueError):
        compile_plan_proposal(
            goal,
            proposal,
        )


def test_llm_generated_plan_executes_wikipedia_search() -> None:
    state = TaskState(
        goal=WIKI_GOAL
    )
    result = run_fake_worker_state(
        state,
        "empty",
        durable_planner=_llm_planner(
            _proposal(
                "wikipedia-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "Claude Shannon",
                    },
                    {
                        "kind": "submit_search",
                    },
                ],
            )
        ),
    )

    assert result.completed is True
    assert result.typed == 1
    assert result.clicked == 1
    assert DURABLE_PLAN_ARTIFACT_ID in state.artifacts


def test_llm_generated_plan_executes_wikipedia_followup() -> None:
    state = TaskState(
        goal=WIKI_FOLLOWUP_GOAL
    )
    result = run_fake_worker_state(
        state,
        "empty",
        durable_planner=_llm_planner(
            _proposal(
                "wikipedia-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "Claude Shannon",
                    },
                    {
                        "kind": "submit_search",
                    },
                    {
                        "kind": "open_link",
                        "target_text": "Information theory",
                    },
                ],
            )
        ),
    )

    assert result.completed is True
    assert result.typed == 1
    assert result.clicked == 2
    assert result.followup_clicked == 1


def test_llm_generated_plan_executes_python_search() -> None:
    state = TaskState(
        goal=PYTHON_GOAL
    )
    result = run_fake_worker_state(
        state,
        "empty",
        durable_planner=_llm_planner(
            _proposal(
                "python-org-search",
                [
                    {
                        "kind": "enter_text",
                        "text": "asyncio",
                    },
                    {
                        "kind": "submit_search",
                    },
                ],
            )
        ),
    )

    assert result.completed is True
    assert result.typed == 1
    assert result.clicked == 1


def test_mismatch_rejected_before_browser_action() -> None:
    state = TaskState(
        goal=WIKI_GOAL
    )
    factory_calls = 0

    def factory(*, capture_path):
        nonlocal factory_calls
        del capture_path
        factory_calls += 1
        raise AssertionError("environment should not be created")

    with pytest.raises(
        RuntimeError,
        match="Durable planning failed before browser action",
    ):
        live_web_worker.create_live_web_worker(
            state,
            lambda: None,
            lambda decision: None,
            environment_factory=factory,
            durable_planner=_llm_planner(
                _proposal(
                    "wikipedia-search",
                    [
                        {
                            "kind": "enter_text",
                            "text": "Alan Turing",
                        },
                        {
                            "kind": "submit_search",
                        },
                    ],
                )
            ),
        )

    assert factory_calls == 0


def test_accepted_llm_plan_persists_canonical_plan_and_provenance() -> None:
    client = FakeStructuredLLMClient(
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
            ],
        ),
        model="fake-model-1",
    )
    state = TaskState(
        goal=WIKI_GOAL
    )

    run_fake_worker_state(
        state,
        "empty",
        durable_planner=LLMDurablePlanner(
            client=client
        ),
    )

    assert len(client.calls) == 1
    plan_artifact = state.artifacts[
        DURABLE_PLAN_ARTIFACT_ID
    ]
    assert plan_artifact.location == durable_plan_canonical_json(
        live_web_worker.compile_durable_task_plan(
            live_web_worker.resolve_live_web_task(
                WIKI_GOAL
            )
        )
    )
    provenance = json.loads(
        state.artifacts[
            DURABLE_PLANNER_PROVENANCE_ARTIFACT_ID
        ].location
    )
    assert provenance["planner_type"] == "llm"
    assert provenance["model_identifier"] == "fake-model-1"


def test_restart_uses_persisted_plan_without_planner_call(
    monkeypatch,
) -> None:
    class TestCrash(RuntimeError):
        pass

    monkeypatch.setenv(
        CRASH_ENV_VAR,
        LiveCrashPoint.QUERY_CHECKPOINT.value,
    )
    monkeypatch.setattr(
        live_web_worker,
        "_terminate_process_for_test",
        lambda: (_ for _ in ()).throw(TestCrash()),
    )
    client = FakeStructuredLLMClient(
        _proposal(
            "wikipedia-search",
            [
                {
                    "kind": "enter_text",
                    "text": "Claude Shannon",
                },
                {
                    "kind": "submit_search",
                },
                {
                    "kind": "open_link",
                    "target_text": "Information theory",
                },
            ],
        )
    )
    state = TaskState(
        goal=WIKI_FOLLOWUP_GOAL
    )
    resolved = live_web_worker.resolve_live_web_task(
        WIKI_FOLLOWUP_GOAL
    )
    first_environment = FakeSearchEnvironment(
        capture_path=None,
        spec=resolved.spec,
        query_text=resolved.query_text,
        followup_target_text=resolved.followup_target_text,
        mode="empty",
    )
    first_worker = live_web_worker.create_live_web_worker(
        state,
        lambda: None,
        lambda decision: None,
        environment_factory=lambda *, capture_path: first_environment,
        durable_planner=LLMDurablePlanner(
            client=client
        ),
    )
    with pytest.raises(TestCrash):
        first_worker(
            RuntimeTask(
                goal=state.goal,
                task_id=state.task_id,
            ),
            FakeControl(),
            lambda message: None,
        )

    assert len(client.calls) == 1
    monkeypatch.delenv(
        CRASH_ENV_VAR,
        raising=False,
    )
    failing_planner = FailingPlanner()
    restart_environment = FakeSearchEnvironment(
        capture_path=None,
        spec=resolved.spec,
        query_text=resolved.query_text,
        followup_target_text=resolved.followup_target_text,
        mode="query",
    )
    restart_worker = live_web_worker.create_live_web_worker(
        state,
        lambda: None,
        lambda decision: None,
        environment_factory=lambda *, capture_path: restart_environment,
        durable_planner=failing_planner,
    )
    restart_worker(
        RuntimeTask(
            goal=state.goal,
            task_id=state.task_id,
        ),
        FakeControl(),
        lambda message: None,
    )

    assert failing_planner.calls == 0
    assert restart_environment.type_count == 0
    assert restart_environment.click_count == 2
