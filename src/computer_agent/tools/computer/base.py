"""Shared base class for computer-control tools."""

from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.tools.base import BaseTool


class ComputerTool(BaseTool):
    """Base class for tools backed by ComputerController."""

    supported_platforms = ("darwin",)

    def __init__(
        self,
        controller: ComputerController,
    ) -> None:
        self._controller = controller

    @property
    def controller(self) -> ComputerController:
        """Return the wrapped computer controller."""

        return self._controller