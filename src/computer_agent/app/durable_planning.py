"""Bounded durable task planning for live web workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol, TYPE_CHECKING

from computer_agent.reasoning.llm_client import LLMClient

if TYPE_CHECKING:
    from computer_agent.app.live_web_worker import DurableTaskPlan


MAX_DURABLE_PROPOSAL_STEPS = 6
DURABLE_PLAN_PROPOSAL_SCHEMA_NAME = (
    "computer_agent_durable_plan_proposal"
)

_TOP_LEVEL_KEYS = frozenset(("version", "workflow_id", "steps"))
_ENTER_TEXT_KEYS = frozenset(("kind", "text"))
_SUBMIT_SEARCH_KEYS = frozenset(("kind",))
_OPEN_LINK_KEYS = frozenset(("kind", "target_text"))
_SUPPORTED_STEP_KINDS = frozenset(
    (
        "enter_text",
        "submit_search",
        "activate_control",
        "open_link",
    )
)
_SUPPORTED_WORKFLOW_IDS = frozenset(
    (
        "wikipedia-search",
        "python-org-search",
    )
)

DURABLE_PLAN_PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "version": {
            "type": "integer",
            "enum": [
                1,
            ],
        },
        "workflow_id": {
            "type": "string",
            "enum": [
                "wikipedia-search",
                "python-org-search",
            ],
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_DURABLE_PROPOSAL_STEPS,
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "enter_text",
                                ],
                            },
                            "text": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": [
                            "kind",
                            "text",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "submit_search",
                                    "activate_control",
                                ],
                            },
                        },
                        "required": [
                            "kind",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "open_link",
                                ],
                            },
                            "target_text": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": [
                            "kind",
                            "target_text",
                        ],
                        "additionalProperties": False,
                    },
                ],
            },
        },
    },
    "required": [
        "version",
        "workflow_id",
        "steps",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class DurableStepProposal:
    """One LLM-proposed durable semantic step."""

    kind: str
    text: str | None = None
    target_text: str | None = None


@dataclass(frozen=True, slots=True)
class DurablePlanProposal:
    """Strict external proposal compiled into a DurableTaskPlan."""

    version: int
    workflow_id: str
    steps: tuple[DurableStepProposal, ...]


@dataclass(frozen=True, slots=True)
class DurableWorkflowCapability:
    """Machine-readable capability exposed to the durable planner."""

    workflow_id: str
    site: str
    allowed_step_kinds: tuple[str, ...]
    required_order: tuple[str, ...]
    verification: tuple[str, ...]
    constraints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DurablePlanningContext:
    """Bounded planning context for durable live-web tasks."""

    workflows: tuple[DurableWorkflowCapability, ...]
    max_steps: int = MAX_DURABLE_PROPOSAL_STEPS

    def to_payload(self) -> dict[str, object]:
        return {
            "max_steps": self.max_steps,
            "workflows": [
                {
                    "workflow_id": workflow.workflow_id,
                    "site": workflow.site,
                    "allowed_step_kinds": list(
                        workflow.allowed_step_kinds
                    ),
                    "required_order": list(
                        workflow.required_order
                    ),
                    "verification": list(workflow.verification),
                    "constraints": list(workflow.constraints),
                }
                for workflow in self.workflows
            ],
        }


class DurablePlanner(Protocol):
    """Interface for durable semantic planners."""

    def plan(
        self,
        goal: str,
        context: DurablePlanningContext,
    ) -> "DurableTaskPlan":
        """Return a validated durable plan for the goal."""


@dataclass(frozen=True, slots=True)
class DeterministicDurablePlanner:
    """Durable planner backed by the deterministic bounded grammar."""

    planner_type: str = "deterministic"
    model_identifier: str | None = None

    def plan(
        self,
        goal: str,
        context: DurablePlanningContext,
    ) -> "DurableTaskPlan":
        del context
        from computer_agent.app.live_web_worker import (
            compile_durable_task_plan,
            resolve_live_web_task,
        )

        resolved_task = resolve_live_web_task(goal)
        assert resolved_task is not None
        return compile_durable_task_plan(resolved_task)


class LLMDurablePlanner:
    """LLM proposal planner with deterministic plan acceptance."""

    planner_type = "llm"

    def __init__(
        self,
        *,
        client: LLMClient,
    ) -> None:
        self._client = client

    @property
    def model_identifier(self) -> str | None:
        model = getattr(self._client, "model", None)
        return model if isinstance(model, str) and model.strip() else None

    def plan(
        self,
        goal: str,
        context: DurablePlanningContext,
    ) -> "DurableTaskPlan":
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")

        response = self._generate_proposal_text(
            goal,
            context,
        )
        proposal = parse_plan_proposal_json(
            response
        )
        return compile_plan_proposal(
            goal,
            proposal,
            context,
        )

    def _generate_proposal_text(
        self,
        goal: str,
        context: DurablePlanningContext,
    ) -> str:
        generate_json = getattr(
            self._client,
            "generate_json",
            None,
        )
        if callable(generate_json):
            return generate_json(
                system_prompt=LLM_DURABLE_PLANNER_SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(
                    goal,
                    context,
                ),
                schema_name=DURABLE_PLAN_PROPOSAL_SCHEMA_NAME,
                schema=DURABLE_PLAN_PROPOSAL_JSON_SCHEMA,
            )

        return self._client.generate(
            system_prompt=LLM_DURABLE_PLANNER_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(
                goal,
                context,
            ),
        )


LLM_DURABLE_PLANNER_SYSTEM_PROMPT = """
Convert a supported user goal into a durable semantic plan proposal.
Return JSON only, with no markdown, comments, or prose.
The proposal is intent only. It must not contain coordinates, action keys,
claim IDs, subgoal IDs, side-effect IDs, confidence values, evidence,
tool calls, or completion decisions.
The deterministic runtime will validate, compile, execute, and verify.
The top-level object must contain exactly:
{"version": 1, "workflow_id": string, "steps": array}
Supported workflows are bounded to Wikipedia search and python.org search.
Supported step kinds are enter_text, submit_search, and open_link.
Use activate_control only as an alias for submit_search when necessary.
enter_text steps must contain exactly {"kind": "enter_text", "text": string}.
submit_search steps must contain exactly {"kind": "submit_search"}.
open_link steps must contain exactly {"kind": "open_link", "target_text": string}.
Do not invent websites, controls, hidden state, retries, or follow-up targets.
Do not decide that an action succeeded.
""".strip()


def default_durable_planning_context() -> DurablePlanningContext:
    """Return the bounded planning context exposed to LLM planners."""

    return DurablePlanningContext(
        workflows=(
            DurableWorkflowCapability(
                workflow_id="wikipedia-search",
                site="Wikipedia",
                allowed_step_kinds=(
                    "enter_text",
                    "submit_search",
                    "open_link",
                ),
                required_order=(
                    "enter_text",
                    "submit_search",
                    "open_link_optional",
                ),
                verification=(
                    "enter_text verifies the intended query is visible in the search UI",
                    "submit_search verifies the Wikipedia search outcome is visible",
                    "open_link verifies the requested destination heading is visible",
                ),
                constraints=(
                    "open_link is allowed only after submit_search",
                    "open_link target_text must match the user's requested link exactly",
                ),
            ),
            DurableWorkflowCapability(
                workflow_id="python-org-search",
                site="python.org",
                allowed_step_kinds=(
                    "enter_text",
                    "submit_search",
                ),
                required_order=(
                    "enter_text",
                    "submit_search",
                ),
                verification=(
                    "enter_text verifies the intended query is in the site search field",
                    "submit_search verifies the Results page is visible",
                ),
                constraints=(
                    "open_link is not supported for python.org in this phase",
                ),
            ),
        )
    )


def parse_plan_proposal_json(
    value: object,
) -> DurablePlanProposal:
    """Parse strict proposal JSON into immutable proposal objects."""

    if not isinstance(value, str):
        raise ValueError("proposal response must be a string")
    if not value.strip():
        raise ValueError("proposal response must be non-empty")

    try:
        payload = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("proposal response is not valid JSON") from exc

    _require_object(payload, "proposal")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "proposal")

    if payload["version"] != 1:
        raise ValueError("proposal version is unsupported")
    workflow_id = _require_non_empty_string(
        payload["workflow_id"],
        "workflow_id",
    )
    if workflow_id not in _SUPPORTED_WORKFLOW_IDS:
        raise ValueError("unsupported workflow")
    steps_payload = payload["steps"]
    if not isinstance(steps_payload, list):
        raise ValueError("proposal steps must be an array")
    if not steps_payload:
        raise ValueError("proposal must contain steps")
    if len(steps_payload) > MAX_DURABLE_PROPOSAL_STEPS:
        raise ValueError("proposal contains too many steps")

    return DurablePlanProposal(
        version=1,
        workflow_id=workflow_id,
        steps=tuple(
            _parse_step_proposal(step_payload)
            for step_payload in steps_payload
        ),
    )


def compile_plan_proposal(
    goal: str,
    proposal: DurablePlanProposal,
    context: DurablePlanningContext | None = None,
) -> "DurableTaskPlan":
    """Validate and compile a proposal into a canonical DurableTaskPlan."""

    if context is None:
        context = default_durable_planning_context()

    workflow_ids = {
        workflow.workflow_id
        for workflow in context.workflows
    }
    if proposal.workflow_id not in workflow_ids:
        raise ValueError("unsupported workflow")

    if len(proposal.steps) > context.max_steps:
        raise ValueError("proposal contains too many steps")

    from computer_agent.app.live_web_worker import (
        compile_durable_task_plan,
        resolve_live_web_task,
    )

    try:
        resolved_task = resolve_live_web_task(goal)
    except RuntimeError as exc:
        raise ValueError("goal cannot be represented by supported workflows") from exc

    assert resolved_task is not None

    if proposal.workflow_id != resolved_task.spec.workflow_id:
        raise ValueError("proposal workflow does not match goal")

    kinds = tuple(
        _canonical_step_kind(step.kind)
        for step in proposal.steps
    )
    expected_kinds = (
        (
            "enter_text",
            "submit_search",
            "open_link",
        )
        if resolved_task.followup_target_text is not None
        else (
            "enter_text",
            "submit_search",
        )
    )
    if kinds != expected_kinds:
        raise ValueError("proposal step ordering does not match goal")

    enter_step = proposal.steps[0]
    if enter_step.text != resolved_task.query_text:
        raise ValueError("proposal query does not match goal")

    if resolved_task.followup_target_text is not None:
        open_step = proposal.steps[2]
        if open_step.target_text != resolved_task.followup_target_text:
            raise ValueError("proposal follow-up target does not match goal")
    elif any(
        _canonical_step_kind(step.kind) == "open_link"
        for step in proposal.steps
    ):
        raise ValueError("proposal follow-up target does not match goal")

    return compile_durable_task_plan(
        resolved_task
    )


def _build_user_prompt(
    goal: str,
    context: DurablePlanningContext,
) -> str:
    payload = {
        "goal": goal,
        "capabilities": context.to_payload(),
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_step_proposal(
    value: object,
) -> DurableStepProposal:
    _require_object(value, "proposal step")
    kind = _require_non_empty_string(
        value.get("kind"),
        "step.kind",
    )
    if kind not in _SUPPORTED_STEP_KINDS:
        raise ValueError("unsupported step kind")

    canonical_kind = _canonical_step_kind(kind)
    if canonical_kind == "enter_text":
        _require_exact_keys(value, _ENTER_TEXT_KEYS, "enter_text step")
        return DurableStepProposal(
            kind=kind,
            text=_require_non_empty_string(
                value["text"],
                "enter_text.text",
            ),
        )

    if canonical_kind == "submit_search":
        _require_exact_keys(value, _SUBMIT_SEARCH_KEYS, "submit_search step")
        return DurableStepProposal(kind=kind)

    if canonical_kind == "open_link":
        _require_exact_keys(value, _OPEN_LINK_KEYS, "open_link step")
        return DurableStepProposal(
            kind=kind,
            target_text=_require_non_empty_string(
                value["target_text"],
                "open_link.target_text",
            ),
        )

    raise ValueError("unsupported step kind")


def _canonical_step_kind(
    kind: str,
) -> str:
    if kind == "activate_control":
        return "submit_search"
    return kind


def _require_object(
    value: object,
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    keys = frozenset(value)
    if keys != expected:
        raise ValueError(
            f"{label} must contain exactly keys: "
            + ", ".join(sorted(expected))
        )


def _require_non_empty_string(
    value: object,
    label: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(
    value: str,
) -> None:
    raise ValueError(f"unsupported JSON constant: {value}")
