"""Computer-control tools and their default tool set."""

from computer_agent.control.computer_controller import (
    ComputerController,
)
from computer_agent.tools.computer.application import (
    ActivateAppTool,
    OpenURLTool,
)
from computer_agent.tools.computer.base import ComputerTool
from computer_agent.tools.computer.clipboard import (
    CopyToClipboardTool,
    PasteTextTool,
    ReadFromClipboardTool,
)
from computer_agent.tools.computer.keyboard import (
    HotkeyTool,
    PressKeyTool,
    TypeTextTool,
)
from computer_agent.tools.computer.mouse import (
    ClickMouseTool,
    GetMousePositionTool,
    MoveMouseTool,
    ScrollTool,
)
from computer_agent.tools.computer.screen import (
    CaptureScreenshotTool,
    GetScreenSizeTool,
)


def create_computer_tools(
    controller: ComputerController,
) -> list[ComputerTool]:
    """Create all computer tools using one controller."""

    return [
        GetMousePositionTool(controller),
        MoveMouseTool(controller),
        ClickMouseTool(controller),
        ScrollTool(controller),
        TypeTextTool(controller),
        PressKeyTool(controller),
        HotkeyTool(controller),
        CopyToClipboardTool(controller),
        ReadFromClipboardTool(controller),
        PasteTextTool(controller),
        GetScreenSizeTool(controller),
        CaptureScreenshotTool(controller),
        ActivateAppTool(controller),
        OpenURLTool(controller),
    ]


__all__ = [
    "ActivateAppTool",
    "CaptureScreenshotTool",
    "ClickMouseTool",
    "ComputerTool",
    "CopyToClipboardTool",
    "GetMousePositionTool",
    "GetScreenSizeTool",
    "HotkeyTool",
    "MoveMouseTool",
    "OpenURLTool",
    "PasteTextTool",
    "PressKeyTool",
    "ReadFromClipboardTool",
    "ScrollTool",
    "TypeTextTool",
    "create_computer_tools",
]