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
    PYTHON_WORKFLOW,
    WIKIPEDIA_WORKFLOW,
    create_live_web_worker,
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
    "python": PYTHON_WORKFLOW.supported_goal,
    "wikipedia": WIKIPEDIA_WORKFLOW.supported_goal,
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
        "--path",
        choices=("normal", "query-crash", "submit-crash"),
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

    crash_point = (
        LiveCrashPoint.QUERY_CHECKPOINT
        if args.path == "query-crash"
        else LiveCrashPoint.SUBMIT_EXECUTION
    )
    env = {
        **os.environ,
        CRASH_ENV_VAR: crash_point.value,
    }

    start = subprocess.run(
        [
            sys.executable,
            __file__,
            "--workflow",
            args.workflow,
            "--path",
            args.path,
            "--store-dir",
            str(store_dir),
            "--task-id",
            task_id,
            "--child",
            "start",
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
            "--workflow",
            args.workflow,
            "--path",
            args.path,
            "--store-dir",
            str(store_dir),
            "--task-id",
            task_id,
            "--child",
            "resume",
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
            goal=GOALS[args.workflow],
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

    worker = create_live_web_worker(
        state,
        publish_state,
        decisions.append,
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

    return (
        0
        if state.status is TaskStateStatus.COMPLETED
        else 1
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
