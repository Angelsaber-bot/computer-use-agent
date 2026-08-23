# Development Log

## Phase 01: Computer Control

### Experiment 01: Mouse Position Reader

**Date:** August 21, 2026

**Objective:**  
Read and display the current mouse position.

**File:**  
`experiments/phase01_computer_control/experiment_01_mouse_position_test.py`

**Result:**  
Success. The program returned the mouse position `x=723, y=436`.

### Experiment 02: Mouse Movement

**Date:** August 21, 2026

**Objective:**  
Move the mouse to the screen center and return it to its starting position.

**File:**  
`experiments/phase01_computer_control/experiment_02_mouse_movement_test.py`

**Result:**  
Success. The mouse moved to `(735, 478)` and returned to its starting position.

### Experiment 03: Mouse Click

**Date:** August 21, 2026

**Objective:**  
Perform a mouse click at a selected safe location.

**File:**  
`experiments/phase01_computer_control/experiment_03_mouse_click_test.py`

**Result:**  
Success. The program clicked at `(987, 245)`.

### Experiment 04: Keyboard Typing

**Date:** August 21, 2026

**Objective:**  
Open a new TextEdit document and type text automatically.

**File:**  
`experiments/phase01_computer_control/experiment_04_keyboard_typing_test.py`

**Result:**  
Success. The program created a TextEdit document and typed the test message correctly.

### Experiment 05: Clipboard

**Date:** August 21, 2026

**Objective:**  
Write text to the system clipboard and read it back.

**File:**  
`experiments/phase01_computer_control/experiment_05_clipboard_test.py`

**Result:**  
Success. The copied and retrieved text matched.

### Experiment 06: Scrolling

**Date:** August 21, 2026

**Objective:**  
Scroll down and up inside a scrollable page.

**File:**  
`experiments/phase01_computer_control/experiment_06_scroll_test.py`

**Result:**  
Success. The page scrolled down and returned upward correctly.

**Future Improvement:**  
Detect the end of a page by comparing screenshots before and after scrolling.

### Experiment 07: Screenshot Capture

**Date:** August 21, 2026

**Objective:**  
Capture the computer screen and save it as an image file.

**File:**  
`experiments/phase01_computer_control/experiment_07_screenshot_test.py`

**Output:**  
`assets/screenshots/phase01_computer_control/experiment_07_screen.png`

**Result:**  
Success. The program captured the full screen and saved the image correctly.

### Experiment 08: Browser Navigation

**Date:** August 21, 2026

**Objective:**  
Open Google Chrome and navigate to a specified URL.

**File:**  
`experiments/phase01_computer_control/experiment_08_browser_navigation_test.py`

**Result:**  
Success. Google Chrome opened `https://example.com`.

### Experiment 09: App Switching

**Date:** August 21, 2026

**Objective:**  
Switch between multiple macOS applications automatically.

**File:**  
`experiments/phase01_computer_control/experiment_09_app_switching_test.py`

**Result:**  
Success. The program activated TextEdit and then switched to Google Chrome.

### Experiment 10: Cross-App Workflow

**Date:** August 21, 2026

**Objective:**  
Combine browser navigation, clipboard access, app switching, keyboard input, and screenshot capture.

**File:**  
`experiments/phase01_computer_control/experiment_10_cross_app_workflow_test.py`

**Output:**  
`assets/screenshots/phase01_computer_control/experiment_10_workflow.png`

**Result:**  
Success. The program opened a webpage, switched to TextEdit, entered the copied URL, and saved a screenshot.

### Formal Module 01: Computer Controller

**Date:** August 22, 2026

**Objective:**  
Convert the mouse position experiment into a reusable computer-control class.

**Source File:**  
`src/computer_agent/control/computer_controller.py`

**Test File:**  
`tests/test_computer_controller.py`

**Result:**  
Success. Pytest completed with `14 passed`.

**Integration Verification:**  
Experiment 10 completed successfully using the formal `ComputerController` class.

**Reliability Fix:**  
An intermittent macOS focus issue caused shortcut letters such as `n` and `v` to be typed as normal text. Hotkey timing was improved, clipboard paste now reuses the reliable hotkey method, and Experiment 10 explicitly creates a new TextEdit document. The integration workflow was repeated twice successfully with no extra characters.

## Phase 02: Tool System and Agent State

All earlier `prework` code is excluded from the formal project.

### Formal Module 02: Structured Agent Data Models

**Date:** August 22, 2026

**Objective:**

Create standard data formats for agent actions, tool results, observations, and execution history.

**Source File:**

`src/computer_agent/core/models.py`

**Test File:**

`tests/test_models.py`

**Implemented:**

- Added `Action` for structured tool requests.
- Added `ToolResult` for successful and failed execution results.
- Added `Observation` for information collected from the environment.
- Added `StepRecord` for connecting one action with its result and observation.
- Added unique IDs and timestamps for execution tracking.

**Result:**

Success. All 7 new data-model tests passed. The complete test suite finished with `21 passed`.

**Commit:**

`8f374fc` — `feat: add phase 02 agent data models`


### Formal Module 03: Agent State Management

**Date:** August 22, 2026

**Objective:**

Track the complete status and execution history of one user task.

**Source File:**

`src/computer_agent/agent/state.py`

**Test File:**

`tests/test_agent_state.py`

**Implemented:**

- Added pending, running, succeeded, and failed task states.
- Added validated state transitions.
- Added step-history recording.
- Added task context and latest-error storage.
- Prevented completed tasks from being restarted or changed incorrectly.

**Result:**

Success. All 6 new Agent-state tests passed. The complete test suite finished with `27 passed`.

**Commit:**

`666972f` — `feat: add agent task state management`


### Formal Module 04: Base Tool and Tool Registry

**Date:** August 22, 2026

**Objective:**

Create a common tool interface and a central system for tool registration and discovery.

**Source Files:**

- `src/computer_agent/tools/base.py`
- `src/computer_agent/tools/registry.py`

**Test File:**

`tests/test_tool_registry.py`

**Implemented:**

- Added a common `BaseTool` interface.
- Added tool names, descriptions, parameters, and platform support.
- Added argument validation and optional default values.
- Added planner-readable tool schemas.
- Added tool registration, lookup, listing, and removal.
- Added duplicate-tool and missing-tool error handling.

**Result:**

Success. All 10 new tool-system tests passed. The complete test suite finished with `37 passed`.

**Commit:**

`9e0012f` — `feat: add base tool interface and registry`

### Formal Module 05: Tool Executor

**Date:** August 22, 2026

**Objective:**

Safely execute structured agent actions through the tool registry and always return a standardized result.

**Source File:**

`src/computer_agent/tools/executor.py`

**Test File:**

`tests/test_tool_executor.py`

**Implemented:**

- Added `ToolExecutor` for executing structured `Action` objects.
- Connected `Action`, `ToolRegistry`, `BaseTool`, and `ToolResult`.
- Added platform-availability checks before tool execution.
- Added automatic tool-argument validation.
- Converted missing-tool errors into failed `ToolResult` objects.
- Converted unavailable-tool errors into failed `ToolResult` objects.
- Converted invalid arguments into failed `ToolResult` objects.
- Converted runtime exceptions into failed `ToolResult` objects.
- Added execution timestamps and duration measurement.
- Kept execution separate from `AgentState` to preserve clear module responsibilities.

**Result:**

Success. All 6 new ToolExecutor tests passed. The complete test suite finished with `43 passed`.

The executor can now safely run registered tools and report failures without crashing the agent. Real computer-control tools will be connected in the next module.

**Commit:**

`2e5bc54` — `feat: add structured tool executor`

### Formal Module 06: Computer Control Tool Adapters

**Date:** August 22, 2026

**Objective:**

Expose all Phase 1 computer-control functions through the standardized Phase 2 tool system.

**Source Files:**

- `src/computer_agent/tools/computer/base.py`
- `src/computer_agent/tools/computer/mouse.py`
- `src/computer_agent/tools/computer/keyboard.py`
- `src/computer_agent/tools/computer/clipboard.py`
- `src/computer_agent/tools/computer/screen.py`
- `src/computer_agent/tools/computer/application.py`
- `src/computer_agent/tools/computer/__init__.py`

**Test Files:**

- `tests/test_mouse_tools.py`
- `tests/test_keyboard_tools.py`
- `tests/test_clipboard_tools.py`
- `tests/test_screen_tools.py`
- `tests/test_application_tools.py`
- `tests/test_computer_tool_factory.py`

**Implemented:**

- Added a shared `ComputerTool` adapter base class.
- Added four mouse tools for position, movement, clicking, and scrolling.
- Added three keyboard tools for typing, single keys, and shortcuts.
- Added three clipboard tools for copying, reading, and pasting text.
- Added two screen tools for screen size and screenshot capture.
- Added two application tools for activating apps and opening URLs.
- Added `create_computer_tools()` to create all 14 tools with one controller.
- Declared the current computer adapters as macOS tools.
- Returned structured output from every computer tool.
- Added validation for keyboard shortcuts, application names, URLs, and browser names.
- Connected `Action`, `ToolExecutor`, `ToolRegistry`, computer tools, and `ToolResult` in one safe mocked execution test.

**Result:**

Success. All 22 new computer-tool tests passed. The complete test suite finished with `65 passed`.

All 14 Phase 1 `ComputerController` functions are now available through the structured Phase 2 tool system. Unit tests used mock controllers, so no real mouse, keyboard, browser, or application actions occurred during automated testing.

The next step is the Phase 2 integration experiment using the real macOS `ComputerController`.

**Commit:**

`964e9c5` — `feat: wrap computer controller as tools`


### Formal Module 07: Phase 02 Integration Experiment

**Date:** August 22, 2026

**Objective:**

Verify that structured actions can pass through the complete Phase 2 tool system and perform real macOS operations.

**Experiment File:**

`experiments/phase02_tool_system/experiment_01_tool_workflow.py`

**Screenshot:**

`assets/screenshots/phase02_tool_system/experiment_01_tool_workflow.png`

**Workflow:**

1. Created the real macOS `ComputerController`.
2. Created and registered all 14 computer-control tools.
3. Created a `ToolExecutor` using the tool registry.
4. Created and started an `AgentState`.
5. Executed `open_url` to open `example.com` in Google Chrome.
6. Executed `activate_app` to activate TextEdit.
7. Executed `hotkey` to create a new TextEdit document.
8. Executed `paste_text` to insert the experiment message.
9. Executed `capture_screenshot` to save visual evidence.
10. Recorded every `Action` and `ToolResult` in the Agent-state history.

**Observed Result:**

- All five structured actions returned successful `ToolResult` objects.
- The Agent state recorded all five execution steps.
- The final Agent status was `SUCCEEDED`.
- Google Chrome successfully opened `example.com`.
- TextEdit successfully created a new document.
- The experiment message appeared correctly with no extra characters.
- The screenshot was saved as a valid `2940 x 1912` PNG image.
- The experiment process finished with exit code `0`.
- The complete automated test suite still finished with `65 passed`.

Screen-content correctness was verified manually because automatic screen understanding belongs to the later Screen Perception and Verification phases.

**Result:**

Success. The complete Phase 2 execution path worked with the real macOS computer controller:

`Action -> ToolRegistry -> ToolExecutor -> ComputerTool -> ComputerController -> macOS -> ToolResult -> AgentState`

**Commit:**

`7fe0900` — `test: add phase 02 integration experiment`

## Phase 03: Screen Perception

### Experiment 01: Normalized Screen Capture

**Date:** August 23, 2026

**Objective:**

Create the input boundary for screen perception by converting a raw screenshot into a structured `ScreenFrame`.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_01_screen_capture.py`

**Source Files:**

- `src/computer_agent/perception/__init__.py`
- `src/computer_agent/perception/models.py`
- `src/computer_agent/perception/screen_capture.py`

**Test File:**

`tests/test_screen_capture.py`

**Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_01_screen_capture.png`

**Implemented:**

- Added the `computer_agent.perception` package.
- Added `ScreenFrame` for screenshot path, pixel dimensions, logical screen dimensions, capture time, and coordinate scaling.
- Added `ScreenController` Protocol for dependency injection.
- Added `ScreenCapture`, which reuses `ComputerController` and reads image dimensions with Pillow.
- Added three unit tests.
- Added a live integration experiment and screenshot evidence.
- Configured the repository as an editable Python `src`-layout package.

**Problems Encountered:**

- `screen_capture.py` was initially missing because code was placed in the wrong file.
- The `src`-layout package was not installed, causing `ModuleNotFoundError` outside pytest.
- A temporary `sys.path` workaround was removed after configuring editable installation.

**Resolution:**

- Separated `ScreenFrame` and `ScreenCapture` into the correct modules.
- Added standard `pyproject.toml` packaging metadata.
- Installed the project using editable mode.
- Used protocol-based dependency injection instead of environment-specific imports.

**Validation:**

- Import resolved to `/Users/lejiazhang/Desktop/computer_agent/src/computer_agent/__init__.py`.
- The complete automated test suite finished with `68 passed`.
- Screenshot pixel size was `2940 x 1912`.
- Logical screen size was `1470 x 956`.
- Retina coordinate scale was `x=2.00, y=2.00`.
- Screenshot evidence was saved at `assets/screenshots/phase03_screen_perception/experiment_01_screen_capture.png`.

**Result:**

Success. The perception system can now capture a screen image while preserving the metadata needed to convert image coordinates into logical mouse coordinates.

**Next Step:**

Define reusable perception data models such as `BoundingBox` and `UIElement`.

### Repository Audit Remediation

**Date:** August 23, 2026

**Summary:**

- A complete read-only audit was performed at commit `e19e61f`.
- Audit result: 0 P0, 0 P1, 4 P2, and 3 P3 findings.
- Fixed bool values being accepted as integer and numeric tool arguments.
- Added timezone-awareness validation for `ScreenFrame.captured_at`.
- Updated README Phase 03 progress.
- Added regression tests for numeric boolean validation and timestamps.
- Full test result after remediation: `79 passed`.
- `pip check` reported no broken requirements.

**Deferred Before Phase 04:**

- Mouse coordinate and duration bounds.
- URL scheme and browser validation.
- Import-safe historical experiment scripts.

**Deferred Design Decision:**

Screenshot output should eventually use an approved artifact-root policy rather than simply rejecting every absolute path.

**Next Step:**

Phase 03 Experiment 02 — `BoundingBox` and `UIElement` perception models.

### Experiment 02: Reusable Perception Models

**Date:** August 23, 2026

**Objective:**

Create reusable immutable data models for detected screen regions and user-interface elements.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_02_perception_models.py`

**Source File:**

`src/computer_agent/perception/models.py`

**Test File:**

`tests/test_perception_models.py`

**Implemented:**

- Added immutable `BoundingBox` coordinates with inclusive left and top edges and exclusive right and bottom edges.
- Added geometry helpers for edges, center, area, containment, intersection checks, and intersection boxes.
- Added validation for non-negative coordinates, positive dimensions, and boolean rejection for integer fields.
- Added immutable `UIElement` with element type, bounding box, confidence, optional text, and delegated center access.
- Added validation for element types, bounding boxes, confidence values, and text values.
- Exported the new perception models from `computer_agent.perception`.

**Validation:**

- The complete automated test suite finished with `108 passed`.
- The Experiment 02 script completed successfully.

**Result:**

Success. The perception package now has reusable data contracts for future screen element detection without adding OCR, image processing, planner integration, or automation behavior.

**Next Step:**

Phase 03 Experiment 03.
