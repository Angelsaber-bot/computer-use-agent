"""Phase 02 integration experiment for the structured tool workflow."""

from pathlib import Path
from time import sleep

from computer_agent.agent.state import AgentState, AgentStatus
from computer_agent.control.computer_controller import ComputerController
from computer_agent.core.models import Action
from computer_agent.tools.computer import create_computer_tools
from computer_agent.tools.executor import ToolExecutor
from computer_agent.tools.registry import ToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase02_tool_system"
    / "experiment_01_tool_workflow.png"
)

MESSAGE = (
    "Phase 02 Integration Experiment\n"
    "Structured tool workflow completed successfully."
)


def main() -> int:
    SCREENSHOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller = ComputerController()
    registry = ToolRegistry(
        create_computer_tools(controller)
    )
    executor = ToolExecutor(registry)

    state = AgentState(
        user_task=(
            "Open a webpage, create a TextEdit document, "
            "paste a message, and capture the result."
        )
    )

    actions = [
        (
            Action(
                tool_name="open_url",
                arguments={
                    "url": "https://example.com",
                    "browser": "Google Chrome",
                },
                reason="Open a webpage in Chrome.",
            ),
            2.0,
        ),
        (
            Action(
                tool_name="activate_app",
                arguments={"app_name": "TextEdit"},
                reason="Bring TextEdit to the foreground.",
            ),
            1.5,
        ),
        (
            Action(
                tool_name="hotkey",
                arguments={
                    "keys": ["command", "n"],
                    "interval": 0.1,
                },
                reason="Create a new TextEdit document.",
            ),
            1.0,
        ),
        (
            Action(
                tool_name="paste_text",
                arguments={"text": MESSAGE},
                reason="Paste the experiment result.",
            ),
            1.0,
        ),
        (
            Action(
                tool_name="capture_screenshot",
                arguments={
                    "output_path": str(SCREENSHOT_PATH),
                },
                reason="Save evidence of the completed workflow.",
            ),
            0.0,
        ),
    ]

    print("Phase 02 Integration Experiment")
    print(f"Registered tools: {len(registry)}")
    print()

    state.start()

    for action, wait_seconds in actions:
        result = executor.execute(action)
        record = state.record_step(action, result)

        result_label = (
            "SUCCESS" if result.success else "FAILED"
        )

        print(
            f"[{record.step_number}] "
            f"{action.tool_name}: {result_label}"
        )

        if result.output is not None:
            print(f"    Output: {result.output}")

        if not result.success:
            state.fail(
                result.error or "Unknown tool execution error"
            )
            print(f"    Error: {result.error}")
            break

        sleep(wait_seconds)

    if state.status is AgentStatus.RUNNING:
        state.succeed()

    print()
    print(f"Final status: {state.status.value.upper()}")
    print(f"Recorded steps: {len(state.steps)}")
    print(f"Screenshot: {SCREENSHOT_PATH}")

    return (
        0
        if state.status is AgentStatus.SUCCEEDED
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())