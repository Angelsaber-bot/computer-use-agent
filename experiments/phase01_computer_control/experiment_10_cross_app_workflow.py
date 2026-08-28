from pathlib import Path
import time
import subprocess
from computer_agent.control.computer_controller import ComputerController


def main() -> None:
    controller = ComputerController()
    url = "https://example.com"
    output = Path(__file__).parents[2] / "assets/screenshots/phase01_computer_control/experiment_10_workflow.png"

    controller.open_url(url)
    time.sleep(2)

    controller.activate_app("TextEdit")

    subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "TextEdit" to make new document'
        ],
        check=True
    )

    time.sleep(1)

    controller.paste_text(url)
    controller.capture_screenshot(output)

    print("Cross-app workflow completed.")


if __name__ == "__main__":
    main()
