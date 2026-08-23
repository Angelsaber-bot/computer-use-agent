"""Registration and discovery of tools available to the agent."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from computer_agent.tools.base import BaseTool


class DuplicateToolError(ValueError):
    """Raised when a tool name is already registered."""


class ToolNotFoundError(LookupError):
    """Raised when a requested tool is not registered."""


class ToolRegistry:
    """Store tools by unique name and expose their schemas."""

    def __init__(
        self,
        tools: Iterable[BaseTool] | None = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}

        for tool in tools or ():
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register one tool using its unique name."""

        if tool.name in self._tools:
            raise DuplicateToolError(
                f"tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> BaseTool:
        """Remove and return a registered tool."""

        try:
            return self._tools.pop(name)
        except KeyError as error:
            raise ToolNotFoundError(
                f"tool not found: {name}"
            ) from error

    def get(self, name: str) -> BaseTool:
        """Return a registered tool by name."""

        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(
                f"tool not found: {name}"
            ) from error

    def list_tools(
        self,
        available_only: bool = False,
    ) -> list[BaseTool]:
        """Return registered tools sorted by name."""

        tools = sorted(
            self._tools.values(),
            key=lambda tool: tool.name,
        )

        if available_only:
            return [
                tool
                for tool in tools
                if tool.is_available()
            ]

        return tools

    def schemas(
        self,
        available_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return descriptions for registered tools."""

        return [
            tool.schema()
            for tool in self.list_tools(
                available_only=available_only
            )
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)