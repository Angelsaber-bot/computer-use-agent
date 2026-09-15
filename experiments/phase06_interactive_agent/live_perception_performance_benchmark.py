"""Live performance benchmark for bounded web workflows.

The benchmark records timings and counters; it does not enforce latency
thresholds because network, Chrome, and macOS Accessibility timing vary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import statistics
import subprocess
import sys
import time


TASKS = (
    (
        "alan-turing",
        "Search Wikipedia for Alan Turing and open Turing machine.",
    ),
    (
        "data-structures",
        "Search Wikipedia for computer science and open data structures.",
    ),
    (
        "ai-perception",
        "Search Wikipedia for Artificial intelligence and open perception",
    ),
    (
        "python-asyncio",
        "Search python.org for asyncio.",
    ),
)

PERFORMANCE_RE = re.compile(
    r"performance "
    r"observations=(?P<observations>\d+) "
    r"ax_traversals=(?P<ax_traversals>\d+) "
    r"ocr_calls=(?P<ocr_calls>\d+) "
    r"actions=(?P<actions>\d+) "
    r"readiness_reobservations=(?P<readiness>\d+) "
    r"ax_seconds=(?P<ax_seconds>\d+(?:\.\d+)?) "
    r"ocr_seconds=(?P<ocr_seconds>\d+(?:\.\d+)?) "
    r"stabilization_seconds=(?P<stabilization_seconds>\d+(?:\.\d+)?) "
    r"observation_seconds=(?P<observation_seconds>\d+(?:\.\d+)?)"
)


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    task: str
    run: str
    result: str
    wall_seconds: float
    observations: int | None
    ax_traversals: int | None
    ocr_calls: int | None
    actions: int | None
    readiness_reobservations: int | None
    ax_seconds: float | None
    ocr_seconds: float | None
    stabilization_seconds: float | None
    observation_seconds: float | None
    final_url: str | None


def main() -> int:
    args = _parse_args()
    runs: list[BenchmarkRun] = []

    selected_tasks = [
        item for item in TASKS if not args.task or item[0] in args.task
    ]

    if args.warmup:
        name, goal = selected_tasks[0]
        runs.append(
            _run_one(
                name,
                "warmup",
                goal,
                emit_output=args.verbose,
            )
        )

    for name, goal in selected_tasks:
        for index in range(1, args.runs + 1):
            runs.append(
                _run_one(
                    name,
                    str(index),
                    goal,
                    emit_output=args.verbose,
                )
            )

    _print_table(runs)
    _print_medians(runs)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record live perception performance counters."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Measured fresh runs per selected task.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warm-up task before measured runs.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[name for name, _goal in TASKS],
        help="Limit benchmark to one or more task names.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print child harness output.",
    )
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")
    return args


def _run_one(
    task_name: str,
    run_label: str,
    goal: str,
    *,
    emit_output: bool,
) -> BenchmarkRun:
    started = time.perf_counter()
    result = subprocess.run(
        [
            sys.executable,
            "experiments/phase06_interactive_agent/"
            "live_durable_web_acceptance.py",
            "--goal",
            goal,
            "--planner",
            "deterministic",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    if emit_output:
        print(result.stdout, end="")

    metrics = _parse_performance(result.stdout)
    final_url = _read_chrome_url()
    return BenchmarkRun(
        task=task_name,
        run=run_label,
        result="completed" if result.returncode == 0 else "failed",
        wall_seconds=wall_seconds,
        final_url=final_url,
        **metrics,
    )


def _parse_performance(output: str) -> dict[str, object]:
    match = None
    for candidate in PERFORMANCE_RE.finditer(output):
        match = candidate
    if match is None:
        return {
            "observations": None,
            "ax_traversals": None,
            "ocr_calls": None,
            "actions": None,
            "readiness_reobservations": None,
            "ax_seconds": None,
            "ocr_seconds": None,
            "stabilization_seconds": None,
            "observation_seconds": None,
        }

    groups = match.groupdict()
    return {
        "observations": int(groups["observations"]),
        "ax_traversals": int(groups["ax_traversals"]),
        "ocr_calls": int(groups["ocr_calls"]),
        "actions": int(groups["actions"]),
        "readiness_reobservations": int(groups["readiness"]),
        "ax_seconds": float(groups["ax_seconds"]),
        "ocr_seconds": float(groups["ocr_seconds"]),
        "stabilization_seconds": float(groups["stabilization_seconds"]),
        "observation_seconds": float(groups["observation_seconds"]),
    }


def _read_chrome_url() -> str | None:
    result = subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "Google Chrome" to get URL of active tab '
            "of front window",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _print_table(runs: list[BenchmarkRun]) -> None:
    print(
        "task,run,result,wall_seconds,observations,ax_traversals,"
        "ocr_calls,actions,readiness_reobservations,ax_seconds,"
        "ocr_seconds,stabilization_seconds,observation_seconds,final_url"
    )
    for run in runs:
        print(
            f"{run.task},{run.run},{run.result},{run.wall_seconds:.2f},"
            f"{_field(run.observations)},{_field(run.ax_traversals)},"
            f"{_field(run.ocr_calls)},{_field(run.actions)},"
            f"{_field(run.readiness_reobservations)},"
            f"{_field(run.ax_seconds)},{_field(run.ocr_seconds)},"
            f"{_field(run.stabilization_seconds)},"
            f"{_field(run.observation_seconds)},"
            f"{run.final_url or ''}"
        )


def _print_medians(runs: list[BenchmarkRun]) -> None:
    measured = [run for run in runs if run.run != "warmup"]
    for task_name, _goal in TASKS:
        task_runs = [
            run for run in measured if run.task == task_name
        ]
        if not task_runs:
            continue
        walls = [run.wall_seconds for run in task_runs]
        print(
            f"median {task_name} wall_seconds="
            f"{statistics.median(walls):.2f}"
        )


def _field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
