"""Safe execution of structured agent actions."""

from __future__ import annotations

import sys
from time import perf_counter
from typing import Any

from computer_agent.core.models import (
    Action,
    ToolResult,
    utc_now,
)
from computer_agent.tools.registry import ToolRegistry


class ToolUnavailableError(RuntimeError):
    """Raised when a tool does not support the current platform."""


class ToolExecutor:
    """Validate and execute actions through a tool registry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        """Return the registry used by this executor."""

        return self._registry

    def execute(self, action: Action) -> ToolResult:
        """Execute one action and always return a ToolResult."""

        started_at = utc_now()
        start_counter = perf_counter()

        success = False
        output: Any = None
        error_message: str | None = None

        try:
            tool = self._registry.get(
                action.tool_name
            )

            if not tool.is_available():
                raise ToolUnavailableError(
                    f"tool '{tool.name}' is not available "
                    f"on platform '{sys.platform}'"
                )

            arguments = tool.validate_arguments(
                action.arguments
            )

            output = tool.run(**arguments)
            success = True

        except Exception as error:
            error_message = (
                f"{type(error).__name__}: {error}"
            )

        finished_at = utc_now()

        duration_ms = (
            perf_counter() - start_counter
        ) * 1000

        return ToolResult(
            action_id=action.action_id,
            tool_name=action.tool_name,
            success=success,
            output=output,
            error=error_message,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
        )