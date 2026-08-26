# Development Log

## Phase 01: Computer Control

### Experiment 01: Mouse Position Reader

**Date:** August 21, 2026

**Objective:**  
Read and display the current mouse position.

**File:**  
`experiments/phase01_computer_control/experiment_01_mouse_position.py`

**Result:**  
Success. The program returned the mouse position `x=723, y=436`.

### Experiment 02: Mouse Movement

**Date:** August 21, 2026

**Objective:**  
Move the mouse to the screen center and return it to its starting position.

**File:**  
`experiments/phase01_computer_control/experiment_02_mouse_movement.py`

**Result:**  
Success. The mouse moved to `(735, 478)` and returned to its starting position.

### Experiment 03: Mouse Click

**Date:** August 21, 2026

**Objective:**  
Perform a mouse click at a selected safe location.

**File:**  
`experiments/phase01_computer_control/experiment_03_mouse_click.py`

**Result:**  
Success. The program clicked at `(987, 245)`.

### Experiment 04: Keyboard Typing

**Date:** August 21, 2026

**Objective:**  
Open a new TextEdit document and type text automatically.

**File:**  
`experiments/phase01_computer_control/experiment_04_keyboard_typing.py`

**Result:**  
Success. The program created a TextEdit document and typed the test message correctly.

### Experiment 05: Clipboard

**Date:** August 21, 2026

**Objective:**  
Write text to the system clipboard and read it back.

**File:**  
`experiments/phase01_computer_control/experiment_05_clipboard.py`

**Result:**  
Success. The copied and retrieved text matched.

### Experiment 06: Scrolling

**Date:** August 21, 2026

**Objective:**  
Scroll down and up inside a scrollable page.

**File:**  
`experiments/phase01_computer_control/experiment_06_scroll.py`

**Result:**  
Success. The page scrolled down and returned upward correctly.

**Future Improvement:**  
Detect the end of a page by comparing screenshots before and after scrolling.

### Experiment 07: Screenshot Capture

**Date:** August 21, 2026

**Objective:**  
Capture the computer screen and save it as an image file.

**File:**  
`experiments/phase01_computer_control/experiment_07_screenshot.py`

**Output:**  
`assets/screenshots/phase01_computer_control/experiment_07_screen.png`

**Result:**  
Success. The program captured the full screen and saved the image correctly.

### Experiment 08: Browser Navigation

**Date:** August 21, 2026

**Objective:**  
Open Google Chrome and navigate to a specified URL.

**File:**  
`experiments/phase01_computer_control/experiment_08_browser_navigation.py`

**Result:**  
Success. Google Chrome opened `https://example.com`.

### Experiment 09: App Switching

**Date:** August 21, 2026

**Objective:**  
Switch between multiple macOS applications automatically.

**File:**  
`experiments/phase01_computer_control/experiment_09_app_switching.py`

**Result:**  
Success. The program activated TextEdit and then switched to Google Chrome.

### Experiment 10: Cross-App Workflow

**Date:** August 21, 2026

**Objective:**  
Combine browser navigation, clipboard access, app switching, keyboard input, and screenshot capture.

**File:**  
`experiments/phase01_computer_control/experiment_10_cross_app_workflow.py`

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

### Experiment 03: Image Preprocessing

**Date:** August 23, 2026

**Objective:**

Create deterministic, reusable PIL image preprocessing for later screen perception work.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_03_image_preprocessing.py`

**Source File:**

`src/computer_agent/perception/preprocessing.py`

**Test File:**

`tests/test_image_preprocessing.py`

**Output:**

`assets/screenshots/phase03_screen_perception/experiment_03_image_preprocessing.png`

**Implemented:**

- Added `ImagePreprocessor` for grayscale conversion, deterministic resizing, automatic contrast enhancement, and OCR-oriented preparation.
- Used Pillow grayscale mode `L`.
- Used `Image.Resampling.LANCZOS` for resized images.
- Added validation for PIL image inputs and numeric, finite, positive scale factors.
- Rejected boolean scale factors explicitly.
- Preserved source images by returning processed copies.
- Exported `ImagePreprocessor` from `computer_agent.perception`.

**Validation:**

- Focused preprocessing tests finished with `27 passed`.
- The complete automated test suite finished with `135 passed`.
- Experiment 03 completed successfully.
- The saved processed image was mode `L` with dimensions `1470 x 956`.

**Result:**

Success. The perception package now has deterministic image preprocessing that prepares screenshots for future OCR-oriented work without implementing OCR, template matching, UI detection, planner integration, or automation behavior.

**Next Step:**

Phase 03 Experiment 04.

### Experiment 04: OCR Text Recognition

**Date:** August 23, 2026

**Objective:**

Recognize word-level text from the original high-resolution Phase 03 screenshot using Tesseract OCR and convert accepted OCR words into immutable `UIElement` results.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_04_ocr_text_recognition.py`

**Source File:**

`src/computer_agent/perception/ocr.py`

**Test File:**

`tests/test_ocr.py`

**Input:**

`assets/screenshots/phase03_screen_perception/experiment_01_screen_capture.png`

**Output:**

`assets/screenshots/phase03_screen_perception/experiment_04_ocr_text_recognition.png`

**OCR Engine and Versions:**

- Python package: `pytesseract==0.3.13`
- External executable: Tesseract `5.5.1`
- The external `tesseract` executable must be installed as a system prerequisite and available on `PATH`; it is not installed through pip.

**Implemented:**

- Added `TesseractOCR` for sparse English screen text using Tesseract page segmentation mode 11.
- Used `pytesseract.image_to_data` with `pytesseract.Output.DICT`.
- Added configurable minimum confidence validation from `0.0` to `1.0`.
- Converted Tesseract confidence values from `0-100` into normalized `0.0-1.0` values.
- Filtered empty text, low confidence text, invalid confidence values, invalid boxes, zero-sized boxes, and boxes outside the image dimensions.
- Returned immutable `UIElement` objects with `element_type="text"`.
- Preserved source images by passing a copy to the OCR backend.
- Added Tesseract executable availability reporting.
- Exported `TesseractOCR` from `computer_agent.perception`.
- Updated the live experiment to convert the original screenshot to RGB and preserve the Retina pixel dimensions for OCR.

**Experiment Settings:**

- Confidence threshold: `0.70`
- Input image mode: `RGB`
- Input pixel dimensions: `2940 x 1912`
- Logical screen dimensions: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Recognized text elements: `93`
- Matched expected anchors: `12/12`
- Annotated output mode: `RGB`
- Annotated output dimensions: `2940 x 1912`

**OCR Benchmark Results:**

A read-only benchmark compared four inputs using Tesseract page segmentation mode 11:

- The existing `1470 x 956` processed image.
- The original `2940 x 1912` screenshot converted to RGB.
- The original screenshot converted to grayscale with automatic contrast.
- The grayscale automatic-contrast image inverted.

The original high-resolution RGB screenshot was selected because it preserved Retina pixel detail and produced the best confidence profile among variants that recognized all expected anchors. At confidence `0.50`, it recognized `12/12` expected anchors with mean confidence `0.8411` and median confidence `0.9100`. At the final confidence threshold `0.70`, it produced `93` accepted word-level OCR results. Grayscale conversion, automatic contrast, and inversion did not improve recognition. The Experiment 03 preprocessing functions remain valid, but the `0.5x` demonstration image is not the preferred input for small-text OCR.

A separate read-only Apple Vision benchmark compared the current Tesseract baseline with Apple Vision accurate mode using language correction both enabled and disabled. Apple Vision with language correction disabled matched `12/12` expected anchors and was faster than Tesseract on this screenshot, but it returned only `46` larger text observations instead of Tesseract's `93` accepted word-level elements. Range-level Apple Vision boxes were smaller than their parent text-observation boxes for `10/12` anchors, but they were often phrase or chunk boxes rather than precise word boxes. Confidence values from different OCR engines are not directly comparable. The current decision is to retain Tesseract for word-level OCR and defer Apple Vision as a possible coarse-text or fallback backend.

**Validation:**

- Focused OCR tests finished with `23 passed`.
- The complete automated test suite finished with `158 passed`.
- Experiment 04 completed successfully with the real Tesseract executable.
- The saved annotated PNG was verified with Pillow as mode `RGB` and dimensions `2940 x 1912`.

**Limitations:**

- This experiment performs word-level text recognition only.
- Recognized words are not merged into lines or paragraphs.
- Buttons and other controls are not classified.
- OCR bounding boxes are currently high-resolution pixel coordinates, not PyAutoGUI logical coordinates.
- Future screen parsing must convert OCR coordinates with `logical_x = pixel_x / scale_x` and `logical_y = pixel_y / scale_y` before using them as logical screen coordinates.
- OCR results can contain errors, especially on small, low-contrast, icon-like, stylized, or partially occluded text.
- The OCR output is not connected to the planner, locator, mouse, or keyboard systems.

**Result:**

Success. The perception package can now recognize sparse screen text from a high-resolution RGB screenshot and represent accepted words as bounded `UIElement` objects without adding control classification, template matching, screen parsing, coordinate conversion, planner integration, or automation behavior.

### Experiment 05: OCR Coordinate Mapping

**Date:** August 23, 2026

**Objective:**

Convert high-resolution OCR pixel coordinates into PyAutoGUI logical screen coordinates using `ScreenFrame` scale metadata, without performing any mouse, keyboard, or other computer-control action.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_05_ocr_coordinate_mapping.py`

**Source File:**

`src/computer_agent/perception/coordinates.py`

**Test File:**

`tests/test_screen_coordinate_mapper.py`

**Input:**

`assets/screenshots/phase03_screen_perception/experiment_01_screen_capture.png`

**Output:**

`assets/screenshots/phase03_screen_perception/experiment_05_ocr_coordinate_mapping.png`

**Implemented:**

- Added `ScreenCoordinateMapper` for converting screenshot pixel coordinates into logical screen coordinates.
- Converted pixel points with `logical_x = pixel_x / frame.scale_x` and `logical_y = pixel_y / frame.scale_y`.
- Converted pixel `BoundingBox` objects into integer logical `BoundingBox` objects.
- Preserved exclusive right and bottom edge semantics.
- Used floor for logical left and top edges.
- Used ceil for logical right and bottom edges.
- Calculated logical width and height from the converted logical edges so each logical box contains the full mapped pixel region.
- Converted pixel-coordinate `UIElement` objects into new logical-coordinate `UIElement` objects while preserving element type, text, and confidence.
- Rejected invalid object types, boolean point values, non-finite point values, out-of-frame points, and boxes outside the `ScreenFrame` pixel dimensions.
- Kept coordinate mapping independent from Tesseract and PyAutoGUI.
- Exported `ScreenCoordinateMapper` from `computer_agent.perception`.

**Experiment Settings:**

- OCR backend: Tesseract `5.5.1` through `pytesseract==0.3.13`
- Tesseract page segmentation mode: `11`
- Confidence threshold: `0.70`
- Input image mode: `RGB`
- Input pixel dimensions: `2940 x 1912`
- Logical screen dimensions: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- OCR pixel elements: `93`
- Mapped logical elements: `93`
- Visualization mode: `RGB`
- Visualization dimensions: `1470 x 956`

**Validation:**

- Focused coordinate-mapping tests finished with `27 passed`.
- The complete automated test suite finished with `185 passed`.
- Experiment 05 completed successfully with the real Tesseract executable.
- The saved logical-coordinate visualization was verified with Pillow as mode `RGB` and dimensions `1470 x 956`.
- `git diff --check` reported no whitespace errors.

**Limitations:**

- This experiment only maps OCR coordinates; it does not classify controls, parse the screen, locate targets, or execute any input action.
- Logical boxes are suitable for representing PyAutoGUI coordinate space, but click targeting still requires later validation and locator work.
- OCR errors and false positives from Experiment 04 are preserved because this module only maps coordinates.

**Result:**

Success. OCR word boxes can now be mapped from high-resolution screenshot pixels into logical screen coordinates using `ScreenFrame` scale metadata while preserving immutable perception models and avoiding any computer-control action.

### Experiment 06: UI Text Localization

**Date:** August 25, 2026

**Objective:**

Demonstrate UI text localization by capturing a live screen, recognizing text with OCR, converting OCR pixel boxes into PyAutoGUI logical coordinates, locating a target text candidate, and visualizing the selected target without executing a computer-control action.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_06_ui_text_localization.py`

**Source File:**

`src/computer_agent/perception/text_locator.py`

**Test File:**

`tests/test_text_locator.py`

**Input:**

`assets/screenshots/phase03_screen_perception/experiment_06_ui_text_localization_input.png`

**Output:**

`assets/screenshots/phase03_screen_perception/experiment_06_ui_text_localization.png`

**Implemented:**

- Added `TextTargetLocator` for locating OCR `UIElement` objects by text.
- `TextTargetLocator` returns exact matches by default.
- Matching trims whitespace and is case-insensitive by default.
- Added `partial_match=True` to enable substring candidate matching for OCR strings with extra punctuation, prefixes, paths, or surrounding text.
- Added `TextTargetLocator.extract_target()`.
- `extract_target()` preserves exact-match boxes.
- For partial matches, `extract_target()` finds the first matching substring and estimates its horizontal position from its character start and end positions within the source OCR text.
- `extract_target()` floors the left edge, ceils the right edge, keeps the result inside the source box, and guarantees positive width.
- `extract_target()` preserves element type, confidence, vertical position, and height.
- Experiment 06 captures a live screenshot with `ComputerController` and `ScreenCapture`.
- Experiment 06 runs Tesseract OCR at minimum confidence `0.05`.
- Experiment 06 maps high-resolution OCR pixel boxes into PyAutoGUI logical coordinates with `ScreenCoordinateMapper`.
- Experiment 06 keeps original OCR boxes in green.
- Only extracted target substring boxes are drawn in red.
- The highest-confidence exact match is preferred when selecting a target.
- Partial source matches are fallback candidates.
- The crosshair uses the selected extracted target center.
- The experiment performs no mouse or keyboard action.

**Experiment Settings and Results:**

- Target text: `computer_agent`
- OCR backend: Tesseract
- Minimum confidence: `0.05`
- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Recognized OCR elements: `184`
- Exact matches: `1`
- Candidate matches: `6`
- Selected extracted target center: `x=169.00`, `y=58.00`

**Observed Examples:**

- `~/Desktop/computer_agent`: source box `x=203`, `y=131`, `width=168`, `height=13`; target box `x=273`, `y=131`, `width=98`, `height=13`.
- Long `/Users/.../computer_agent/...` path: source box `x=93`, `y=708`, `width=443`, `height=14`; target box `x=295`, `y=708`, `width=109`, `height=14`.

**Validation:**

- Focused text-locator tests finished with `26 passed in 0.10s`.
- The complete automated test suite finished with `211 passed in 0.57s`.
- The live localization experiment completed successfully.
- The visualization was saved as `assets/screenshots/phase03_screen_perception/experiment_06_ui_text_localization.png`.

**Limitations:**

- OCR can miss small or low-confidence text.
- A `0.05` threshold improves text coverage but admits more low-confidence OCR candidates.
- A green OCR box does not guarantee that the recognized text is correct.
- OCR may return extra punctuation, prefixes, paths, or incomplete words.
- Substring boxes are character-proportion estimates.
- Substring boxes are most accurate for monospaced text and may be less precise with proportional fonts.
- Low-confidence targets will require verification before future mouse actions.
- Duplicate text still requires contextual target selection.
- No mouse movement or clicking occurs yet.

**Result:**

Experiment 06 now demonstrates high-recall OCR candidate collection and target-only substring localization inside longer OCR strings, while remaining visualization-only.

### Experiment 07: Safe Mouse Movement to a Localized Text Target

**Date:** August 25, 2026

**Objective:**

Demonstrate a guarded observe-to-act path that captures a live screen, localizes a target text candidate, plans a safe mouse movement through structured tool actions, verifies the reached cursor position, restores the original cursor position, and never clicks.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_07_safe_mouse_movement.py`

**Source File:**

`src/computer_agent/perception/text_locator.py`

**Test File:**

`tests/test_text_locator.py`

**Input:**

`assets/screenshots/phase03_screen_perception/experiment_07_safe_mouse_movement_input.png`

**Output:**

`assets/screenshots/phase03_screen_perception/experiment_07_safe_mouse_movement_plan.png`

**Implemented:**

- Added `TextTargetLocator.find_best()`.
- `find_best()` reuses existing matching behavior.
- `find_best()` validates `minimum_confidence` between `0.0` and `1.0`.
- `find_best()` returns the highest-confidence eligible match.
- Equal-confidence ties preserve the original element order.
- Experiment 07 uses exact matching first and partial matching only as fallback.
- Partial matches use `extract_target()` to estimate the target-only box.
- OCR candidate collection uses minimum confidence `0.05`.
- Mouse movement requires action confidence of at least `0.70`.
- The default mode is dry-run and creates no input-control `Action`.
- Real movement requires the explicit `--execute` flag.
- Execution uses `Action`, `ToolRegistry`, `ToolExecutor`, and `create_computer_tools()`.
- Only `get_mouse_position` and `move_mouse` actions are used.
- No click, keyboard, clipboard, scrolling, application, or browser action occurs.
- The script verifies the reached cursor position.
- The original cursor position is restored in a `finally` block after successful target movement.
- Restoration is also verified.
- The plan image draws OCR boxes in green and the selected target box and crosshair in red.
- Screen dimensions and coordinate scale come from the live `ScreenFrame`.

**Experiment Settings and Results:**

- Target text: `computer_agent`
- OCR candidate confidence: `0.05`
- Action confidence: `0.70`
- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Movement duration: `1.0` seconds
- Countdown: `3` seconds
- Target hold: `2.0` seconds
- Position tolerance: `1` logical pixel
- Safe edge margin: `10` logical pixels

Dry-run results:

- Accepted OCR source elements: `287`
- Match type: `exact`
- Confidence: `0.91`
- Source box: `x=119`, `y=52`, `width=100`, `height=12`
- Extracted target box: `x=119`, `y=52`, `width=100`, `height=12`
- Planned movement point: `x=169`, `y=58`
- No input-control `Action` was created or executed.

Execute-mode results:

- Accepted OCR source elements: `229`
- Match type: `exact`
- Confidence: `0.91`
- Planned movement point: `x=169`, `y=58`
- Original cursor position: `x=668`, `y=955`
- Reached cursor position: `x=169`, `y=58`
- Restored cursor position: `x=668`, `y=955`
- Execution completed successfully.
- No click occurred.

**Validation:**

- Experiment file compiled successfully with `py_compile`.
- Focused text-locator tests finished with `42 passed in 0.09s`.
- The complete automated test suite finished with `227 passed in 0.59s`.
- `git diff --check` passed.
- Dry-run completed successfully.
- Execute mode completed successfully.
- Refactored Experiment 07 file length: `382` lines.

**Safety Measures:**

- Explicit `--execute` gate.
- Three-second countdown.
- PyAutoGUI fail-safe remains enabled.
- Execution aborts when the initial cursor is in a fail-safe corner.
- Target point must be inside the logical screen and at least `10` pixels from its edges.
- Target confidence must be at least `0.70`.
- Reached and restored positions must be within one logical pixel.
- Restoration is attempted in `finally` after successful movement to the target.
- The experiment never clicks.

**Limitations:**

- Duplicate exact text is currently selected by confidence rather than UI context.
- OCR and the live interface may change between separate runs.
- Partial target boxes remain character-proportion estimates.
- The screenshot is not recaptured immediately before movement for target re-verification.
- The experiment verifies cursor position but not whether a UI element remained unchanged.
- It moves only and does not click yet.
- The `1.0`-second movement duration is intentionally visible for experimentation and may later become configurable for production use.

**Result:**

Experiment 07 successfully demonstrated a guarded observe-to-act path from live screenshot capture and OCR localization to structured mouse movement, reached-position verification, and cursor restoration without clicking.

### Experiment 08: Verified Click on a Localized Text Target

**Date:** August 26, 2026

**Objective:**

Demonstrate a guarded verified-click path that captures a live fixture screen, localizes a target text button, moves and clicks through structured tool actions, captures the screen again, verifies the clicked state through OCR, and restores the original cursor position.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_08_verified_click.py`

**Fixture File:**

`assets/fixtures/phase03_screen_perception/experiment_08_verified_click.html`

**Before Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_08_verified_click_before.png`

**Plan Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_08_verified_click_plan.png`

**After Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_08_verified_click_after.png`

**Implemented:**

- Target text is `SAFE_CLICK_TARGET_08`.
- Verification text is `CLICK_VERIFIED`.
- OCR candidate collection uses minimum confidence `0.05`.
- Mouse control requires action confidence of at least `0.70`.
- The default dry-run mode creates and executes no mouse-control `Action`.
- Real control requires the explicit `--execute` flag.
- A five-second pre-capture countdown lets the user switch to the already-open Chrome fixture before the initial screenshot.
- A three-second pre-movement countdown runs before execute-mode mouse movement.
- The browser fixture is opened or reloaded manually, not automatically.
- Target localization uses exact matching first and partial matching only as fallback.
- The experiment aborts if `CLICK_VERIFIED` is already visible, because the fixture is already in its verified state.
- The target box and click point are validated against the logical screen bounds and safe edge margin.
- PyAutoGUI fail-safe corner protection remains enabled.
- Execution uses structured `get_mouse_position`, `move_mouse`, and `click_mouse` `Action` objects.
- The script verifies the reached cursor position before clicking.
- After clicking, the script waits briefly, captures a new screenshot, and performs a second OCR pass.
- Success requires `CLICK_VERIFIED` to meet the action-confidence threshold.
- Cursor restoration and restoration verification run in a `finally` block.
- The fixture button colors were changed from white-on-blue to black-on-light-yellow because the first full-screen PSM 11 OCR dry-run omitted the target entirely.

**Experiment Settings and Live Results:**

- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Target text: `SAFE_CLICK_TARGET_08`
- Verification text: `CLICK_VERIFIED`
- OCR candidate confidence: `0.05`
- Action confidence: `0.70`

Successful dry-run:

- Accepted OCR elements: `59`
- Match type: `exact`
- Source text: `SAFE_CLICK_TARGET_08`
- Confidence: `0.88`
- Source logical box: `x=454`, `y=541`, `width=562`, `height=45`
- Planned click point: `x=735`, `y=564`
- No mouse-control `Action` was created or executed.

Successful execute mode:

- Accepted initial OCR elements: `59`
- Original cursor position: `x=460`, `y=859`
- Reached cursor position: `x=735`, `y=564`
- Clicked position: `x=735`, `y=564`
- After-click OCR elements: `58`
- Verification text: `CLICK_VERIFIED`
- Verification confidence: `0.90`
- Restored cursor position: `x=460`, `y=859`
- Verified click completed successfully.

**Validation:**

- Experiment file compiled successfully with `py_compile`.
- The complete automated test suite finished with `227 passed in 0.61s`.
- `git diff --check` passed.
- Dry-run completed successfully.
- Real execute-mode test completed successfully.

**Safety Measures:**

- Explicit `--execute` gate.
- Five-second pre-capture countdown.
- Three-second pre-movement countdown.
- PyAutoGUI fail-safe remains enabled.
- Execution aborts when the initial cursor is in a fail-safe corner.
- Target point must be inside the logical screen and at least `10` pixels from its edges.
- Target confidence must be at least `0.70`.
- The fixture must not already show `CLICK_VERIFIED` before clicking.
- Reached and restored positions must be within one logical pixel.
- Restoration is attempted in `finally` after movement to the target.
- The experiment uses one structured `click_mouse` action only after successful target localization and reached-position verification.

**Limitations:**

- The target text is predetermined rather than selected by task reasoning.
- Duplicate text still lacks contextual disambiguation.
- OCR remains sensitive to visual contrast and layout.
- The fixture must be manually opened or reloaded and kept in the foreground.
- Verification currently depends on visible OCR text.
- This experiment does not yet perform recovery retries after a failed click or failed verification.

**Result:**

Experiment 08 successfully demonstrated a guarded verified-click path from live screenshot capture and OCR localization to structured mouse movement, click execution, after-click OCR verification, and cursor restoration.

**Next Step:** Phase 03 Experiment 09
