"""Application tools backed by ComputerController."""

from computer_agent.tools.base import (
    ToolParameter,
    ToolValidationError,
)
from computer_agent.tools.computer.base import ComputerTool


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
            "URL to open.",
        ),
        "browser": ToolParameter(
            str,
            "Browser application name.",
            required=False,
            default="Google Chrome",
        ),
    }

    def run(self, **arguments):
        url = arguments["url"]
        browser = arguments["browser"]

        if not url.strip():
            raise ToolValidationError(
                "argument 'url' cannot be empty"
            )

        if not browser.strip():
            raise ToolValidationError(
                "argument 'browser' cannot be empty"
            )

        self.controller.open_url(
            url,
            browser=browser,
        )

        return {
            "url": url,
            "browser": browser,
        }