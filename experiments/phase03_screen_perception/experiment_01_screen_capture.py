"""Phase 03 experiment for normalized screen capture."""

from pathlib import Path


from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.perception import ScreenCapture


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_PATH = (
    PROJECT_ROOT
    / "assets/screenshots/phase03_screen_perception"
    / "experiment_01_screen_capture.png"
)


def main() -> int:
    SCREENSHOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    controller = ComputerController()
    capture = ScreenCapture(controller)

    frame = capture.capture(SCREENSHOT_PATH)

    print("Phase 03 Experiment 01: Screen Capture")
    print(f"Screenshot: {frame.image_path}")
    print(f"Pixel size: {frame.pixel_size}")
    print(f"Logical screen size: {frame.screen_size}")
    print(
        "Coordinate scale: "
        f"x={frame.scale_x:.2f}, "
        f"y={frame.scale_y:.2f}"
    )
    print(f"Captured at: {frame.captured_at.isoformat()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
