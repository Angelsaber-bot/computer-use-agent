"""Application tools backed by ComputerController."""

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from computer_agent.tools.base import (
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.computer.base import ComputerTool


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_ROOT = (
    _PROJECT_ROOT / "assets/fixtures"
).resolve()
_ALLOWED_BROWSERS = {
    "Google Chrome",
    "Safari",
}


def _is_relative_to(
    path: Path,
    root: Path,
) -> bool:
    return path == root or root in path.parents


def _validate_browser(browser: str) -> str:
    if not browser.strip():
        raise ToolValidationError(
            "argument 'browser' cannot be empty"
        )

    if browser not in _ALLOWED_BROWSERS:
        raise ToolValidationError(
            "argument 'browser' must be Google Chrome or Safari"
        )

    return browser


def _validate_file_url(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc not in ("", "localhost"):
        raise ToolValidationError(
            "argument 'url' file path must be inside assets/fixtures"
        )

    path = Path(
        url2pathname(parsed.path)
    )
    if not path.is_absolute():
        raise ToolValidationError(
            "argument 'url' file path must be absolute"
        )

    resolved = path.resolve(strict=False)
    if not _is_relative_to(
        resolved,
        _FIXTURE_ROOT,
    ):
        raise ToolValidationError(
            "argument 'url' file path must be inside assets/fixtures"
        )

    return resolved.as_uri()


def _has_unescaped_whitespace_or_ascii_control(value: str) -> bool:
    return any(
        character.isspace()
        or ord(character) <= 0x1F
        or ord(character) == 0x7F
        for character in value
    )


def _validate_http_url(
    raw_url: str,
    normalized_url: str,
    parsed,
) -> str:
    if _has_unescaped_whitespace_or_ascii_control(raw_url):
        raise ToolValidationError(
            "argument 'url' cannot contain whitespace or control characters"
        )

    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ToolValidationError(
            "argument 'url' must include a valid host and port"
        ) from error

    if hostname is None or not hostname.strip():
        raise ToolValidationError(
            "argument 'url' must include a host"
        )

    return normalized_url


def _validate_url(url: str) -> str:
    normalized_url = url.strip()

    if not normalized_url:
        raise ToolValidationError(
            "argument 'url' cannot be empty"
        )

    try:
        parsed = urlparse(normalized_url)
    except ValueError as error:
        raise ToolValidationError(
            "argument 'url' must include a valid host and port"
        ) from error

    if parsed.scheme in ("http", "https"):
        return _validate_http_url(
            url,
            normalized_url,
            parsed,
        )

    if parsed.scheme == "file":
        return _validate_file_url(normalized_url)

    raise ToolValidationError(
        "argument 'url' must use http, https, or file scheme"
    )


class ActivateAppTool(ComputerTool):
    """Open or activate a macOS application."""

    name = "activate_app"
    description = "Open or activate a macOS application."

    parameters = {
        "app_name": ToolParameter(
            str,
            "Name of the application.",
        ),
    }

    def run(self, **arguments):
        app_name = arguments["app_name"]

        if not app_name.strip():
            raise ToolValidationError(
                "argument 'app_name' cannot be empty"
            )

        self.controller.activate_app(app_name)

        return {
            "app_name": app_name,
        }


class OpenURLTool(ComputerTool):
    """Open a URL in a selected browser."""

    name = "open_url"
    description = "Open a URL in a selected browser."

    parameters = {
        "url": ToolParameter(
            str,
            (
                "URL to open; accepts HTTP, HTTPS, or a "
                "local file URL inside assets/fixtures/."
            ),
        ),
        "browser": ToolParameter(
            str,
            "Browser application name; must be Google Chrome or Safari.",
            required=False,
            default="Google Chrome",
        ),
    }

    def run(self, **arguments):
        url = _validate_url(arguments["url"])
        browser = _validate_browser(arguments["browser"])

        self.controller.open_url(
            url,
            browser=browser,
        )

        return {
            "url": url,
            "browser": browser,
        }
