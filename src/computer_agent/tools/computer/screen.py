"""Screen tools backed by ComputerController."""

from pathlib import Path

from computer_agent.tools.base import (
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.computer.base import ComputerTool


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SCREENSHOT_ROOT = (
    _PROJECT_ROOT / "assets/screenshots"
).resolve()


def _is_relative_to(
    path: Path,
    root: Path,
) -> bool:
    return path == root or root in path.parents


def _candidate_screenshot_paths(
    output_path: str,
) -> tuple[Path, ...]:
    if not output_path.strip():
        raise ToolValidationError(
            "argument 'output_path' cannot be empty"
        )

    try:
        path = Path(output_path)
    except (ValueError, OSError, RuntimeError) as error:
        raise ToolValidationError(
            "argument 'output_path' is not a valid path"
        ) from error

    if path.is_absolute():
        return (path,)

    if path.parts[:1] == ("assets",):
        return (_PROJECT_ROOT / path,)

    return (
        _PROJECT_ROOT / path,
        _SCREENSHOT_ROOT / path,
    )


def _validated_screenshot_path(
    output_path: str,
) -> Path:
    try:
        for candidate in _candidate_screenshot_paths(output_path):
            resolved = candidate.resolve(strict=False)

            if _is_relative_to(
                resolved,
                _SCREENSHOT_ROOT,
            ):
                if resolved == _SCREENSHOT_ROOT:
                    raise ToolValidationError(
                        "argument 'output_path' must be a file path"
                    )

                if resolved.is_dir():
                    raise ToolValidationError(
                        "argument 'output_path' must be a file path"
                    )

                return resolved
    except ToolValidationError:
        raise
    except (ValueError, OSError, RuntimeError) as error:
        raise ToolValidationError(
            "argument 'output_path' is not a valid path"
        ) from error

    raise ToolValidationError(
        "argument 'output_path' must be inside assets/screenshots"
    )


class GetScreenSizeTool(ComputerTool):
    """Return the current screen size."""

    name = "get_screen_size"
    description = "Return the screen width and height."

    def run(self, **arguments):
        width, height = (
            self.controller.get_screen_size()
        )

        return {
            "width": width,
            "height": height,
        }


class CaptureScreenshotTool(ComputerTool):
    """Capture the full screen and save it to a file."""

    name = "capture_screenshot"
    description = (
        "Capture the full screen and save it "
        "to an image file."
    )

    parameters = {
        "output_path": ToolParameter(
            str,
            (
                "Screenshot file path; must resolve inside "
                "assets/screenshots/. Relative paths resolve "
                "under that directory."
            ),
        ),
    }

    def run(self, **arguments):
        output_path = _validated_screenshot_path(
            arguments["output_path"]
        )
        saved_path = (
            self.controller.capture_screenshot(
                str(output_path)
            )
        )

        return {
            "output_path": str(saved_path),
        }
