"""Development-only live acceptance harness for durable web workflows.

This script invokes the production live web worker with real Chrome,
macOS Accessibility, and the normal durable TaskState checkpoint store.
It intentionally does not duplicate workflow reconciliation logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import gettempdir
from uuid import uuid4

from computer_agent.app.live_web_worker import (
    CRASH_ENV_VAR,
    LiveCrashPoint,
    create_live_web_worker,
)
from computer_agent.app.durable_planning import (
    DeterministicDurablePlanner,
    LLMDurablePlanner,
)
from computer_agent.reasoning.openai_client import (
    OpenAILLMClient,
)
from computer_agent.runtime import RuntimeControl, RuntimeTask
from computer_agent.task import (
    SideEffectState,
    TaskState,
    TaskStateStatus,
    TaskStateStore,
    prepare_state_for_resume,
)


GOALS = {
    "python": "Search python.org for typing.",
    "wikipedia": "Search Wikipedia for computer use agent.",
}


def main() -> int:
    args = _parse_args()

    if args.child is not None:
        return _run_child(args)

    return _run_path(args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live durable web workflow acceptance paths."
        )
    )
    parser.add_argument(
        "--workflow",
        choices=tuple(GOALS),
        default="wikipedia",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help=(
            "Exact deterministic goal, e.g. "
            "'Search Wikipedia for Claude Shannon.'"
        ),
    )
    parser.add_argument(
        "--path",
        choices=(
            "normal",
            "query-crash",
            "submit-crash",
            "followup-crash",
        ),
        default="normal",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--task-id",
        default=None,
    )
    parser.add_argument(
        "--child",
        choices=("start", "resume"),
        default=None,
    )
    parser.add_argument(
        "--planner",
        choices=("configured", "deterministic", "llm"),
        default="configured",
        help=(
            "Durable planner to inject; 'configured' uses "
            "COMPUTER_AGENT_DURABLE_PLANNER."
        ),
    )
    parser.add_argument(
        "--count-planner-calls",
        action="store_true",
        help="Print planner call counts for live validation.",
    )
    return parser.parse_args()


def _run_path(args: argparse.Namespace) -> int:
    store_dir = args.store_dir or (
        Path(gettempdir())
        / "computer-agent-live-acceptance"
        / f"{args.workflow}-{args.path}-{uuid4()}"
    )
    task_id = args.task_id or (
        f"live-{args.workflow}-{args.path}-{uuid4()}"
    )

    if args.path == "normal":
        return _run_child(
            argparse.Namespace(
                **{
                    **vars(args),
                    "store_dir": store_dir,
                    "task_id": task_id,
                    "child": "start",
                }
            )
        )

    crash_points = {
        "query-crash": LiveCrashPoint.QUERY_CHECKPOINT,
        "submit-crash": LiveCrashPoint.SUBMIT_EXECUTION,
        "followup-crash": LiveCrashPoint.FOLLOWUP_EXECUTION,
    }
    crash_point = crash_points[args.path]
    env = {
        **os.environ,
        CRASH_ENV_VAR: crash_point.value,
    }

    start = subprocess.run(
        [
            sys.executable,
            __file__,
            "--goal",
            _goal_from_args(args),
            "--path",
            args.path,
            "--store-dir",
            str(store_dir),
            "--task-id",
            task_id,
            "--child",
            "start",
            "--planner",
            args.planner,
            *(
                ["--count-planner-calls"]
                if args.count_planner_calls
                else []
            ),
        ],
        env=env,
        check=False,
    )

    print(
        f"crash child exit={start.returncode}"
    )
    if start.returncode != 86:
        return start.returncode or 1

    resume = subprocess.run(
        [
            sys.executable,
            __file__,
            "--goal",
            _goal_from_args(args),
            "--path",
            args.path,
            "--store-dir",
            str(store_dir),
            "--task-id",
            task_id,
            "--child",
            "resume",
            "--planner",
            args.planner,
            *(
                ["--count-planner-calls"]
                if args.count_planner_calls
                else []
            ),
        ],
        check=False,
    )

    return resume.returncode


def _run_child(args: argparse.Namespace) -> int:
    assert args.store_dir is not None
    assert args.task_id is not None

    store = TaskStateStore(
        args.store_dir
    )

    if args.child == "resume":
        state = store.load(
            args.task_id
        )
        prepare_state_for_resume(
            state
        )
        store.save(
            state
        )
    else:
        state = TaskState(
            goal=_goal_from_args(args),
            task_id=args.task_id,
        )

    decisions = []

    def publish_state() -> None:
        store.save(
            state
        )
        print(
            _state_line(state),
            flush=True,
        )

    planner = _planner_from_args(
        args
    )

    worker = create_live_web_worker(
        state,
        publish_state,
        decisions.append,
        durable_planner=planner,
    )

    worker(
        RuntimeTask(
            goal=state.goal,
            task_id=state.task_id,
        ),
        RuntimeControl(),
        lambda message: print(
            f"progress: {message}",
            flush=True,
        ),
    )

    store.save(
        state
    )
    print(
        _state_line(state),
        flush=True,
    )
    if (
        args.count_planner_calls
        and planner is not None
        and hasattr(planner, "calls")
    ):
        print(
            f"planner_calls={planner.calls}",
            flush=True,
        )

    return (
        0
        if state.status is TaskStateStatus.COMPLETED
        else 1
    )


def _goal_from_args(
    args: argparse.Namespace,
) -> str:
    if args.goal is not None and args.goal.strip():
        return args.goal.strip()
    return GOALS[args.workflow]


def _planner_from_args(
    args: argparse.Namespace,
):
    if args.planner == "configured":
        return None
    if args.planner == "deterministic":
        planner = DeterministicDurablePlanner()
    elif args.planner == "llm":
        planner = LLMDurablePlanner(
            client=OpenAILLMClient()
        )
    else:
        raise RuntimeError(
            f"Unsupported planner: {args.planner}"
        )

    if not args.count_planner_calls:
        return planner
    return _CountingPlanner(
        planner
    )


class _CountingPlanner:
    def __init__(
        self,
        planner,
    ) -> None:
        self._planner = planner
        self.calls = 0
        self.planner_type = getattr(
            planner,
            "planner_type",
            "unknown",
        )
        self.model_identifier = getattr(
            planner,
            "model_identifier",
            None,
        )

    def plan(
        self,
        goal,
        context,
    ):
        self.calls += 1
        print(
            f"planner_call={self.calls} "
            f"planner_type={self.planner_type} "
            f"model={self.model_identifier}",
            flush=True,
        )
        return self._planner.plan(
            goal,
            context,
        )


def _state_line(state: TaskState) -> str:
    subgoals = ", ".join(
        f"{subgoal.description}={subgoal.status.value}"
        for subgoal in state.subgoals.values()
    )
    effects = ", ".join(
        f"{effect.action_key}:{effect.state.value}"
        for effect in state.side_effects.values()
        if effect.state
        in {
            SideEffectState.INTENDED,
            SideEffectState.EXECUTED,
            SideEffectState.UNKNOWN,
            SideEffectState.CONFIRMED,
        }
    )
    return (
        f"task={state.task_id} status={state.status.value} "
        f"subgoals=[{subgoals}] side_effects=[{effects}]"
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
