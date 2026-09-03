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

### Experiment 09: Recovery Retry with Visual State Verification

**Date:** August 27, 2026

**Objective:**

Demonstrate a guarded recovery loop against a fixture that secretly requires one through three clicks, using fresh OCR localization for each attempt and target-background color verification to decide whether to stop or retry.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_09_recovery_retry.py`

**Fixture File:**

`assets/fixtures/phase03_screen_perception/experiment_09_recovery_retry.html`

**Before Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_09_recovery_retry_before.png`

**Attempt 1 Plan Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_09_recovery_retry_attempt_1_plan.png`

**Attempt 1 After Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_09_recovery_retry_attempt_1_after.png`

**Implemented:**

- The fixture displays only `RECOVERY_TARGET_09`.
- The fixture randomly requires one through three clicks.
- An unsuccessful click keeps the button light yellow and moves it to a different predefined position.
- A successful click disables the button and changes it to light green.
- The secret required click count is never displayed.
- OCR is used to locate the target in each fresh screenshot.
- OCR candidate collection uses minimum confidence `0.05`.
- Mouse control requires action confidence of at least `0.70`.
- Exact matching is preferred; partial matching is fallback.
- The default mode is dry-run.
- Real control requires the explicit `--execute` flag.
- Mouse movement and clicking use structured `Action` objects.
- After every click, the script captures a fresh screenshot and locates the target again.
- Background verification uses the median of multiple sampled button-interior pixels.
- Very dark pixels are excluded so black text and borders do not affect classification.
- Expected colors are `#fff4b8` for incomplete and `#c8e6c9` for completed, with RGB tolerance.
- An incomplete attempt requires the new target center to be at least `100` logical pixels from the previous center.
- The maximum is three attempts.
- Cursor restoration runs in `finally`.
- `crop.get_flattened_data()` is used instead of the deprecated `getdata()` API.

**Experiment Settings and Live Results:**

Screen:

- Pixel size: `2940 x 1912`
- Logical size: `1470 x 956`
- Scale: `x=2.00`, `y=2.00`

Successful dry-run:

- Accepted OCR elements: `72`
- Initial background: `incomplete`
- Initial median RGB: `(253, 245, 191)`
- Match type: `exact`
- Confidence: `0.88`
- No mouse-control `Action` was created or executed.

Successful three-attempt recovery run:

- Attempt 1 background: `incomplete`
- Attempt 1 relocation distance: `650.00` logical pixels
- Attempt 2 background: `incomplete`
- Attempt 2 relocation distance: `392.46` logical pixels
- Attempt 3 background: `completed`
- Completed successfully on attempt `3`.
- Original cursor position was restored.

Final post-maintenance execute validation:

- Accepted initial OCR elements: `22`
- Initial background: `incomplete`
- Initial median RGB: `(253, 245, 191)`
- Match type: `exact`
- Confidence: `0.88`
- Click point: `(735, 772)`
- Completed successfully on attempt `1`.
- Completed median RGB: `(205, 228, 202)`
- Cursor restored to `(624, 899)`.
- No `DeprecationWarning` occurred.

**Validation:**

- Experiment file compiled successfully with `py_compile`.
- `git diff --check` passed.
- The complete automated test suite finished with `227 passed in 0.76s`.
- Dry-run completed successfully.
- One-attempt and three-attempt execute paths both completed successfully.

**Safety Measures:**

- Explicit `--execute` gate.
- Five-second pre-capture countdown.
- Three-second pre-movement countdown.
- The browser fixture is opened or reloaded manually, not automatically.
- PyAutoGUI fail-safe remains enabled.
- Execution aborts when the initial cursor is in a fail-safe corner.
- Target point must be inside the logical screen and at least `10` pixels from its edges.
- Target confidence must be at least `0.70`.
- Each retry uses a fresh screenshot, OCR pass, coordinate mapping, and target localization.
- Incomplete attempts must relocate by at least `100` logical pixels before retrying.
- The experiment never exceeds three attempts.
- Reached and restored positions must be within one logical pixel.
- Restoration is attempted in `finally` after mouse movement starts.

**Limitations:**

- The target text is predetermined rather than selected by task reasoning.
- Duplicate text still lacks contextual disambiguation.
- OCR remains sensitive to visual contrast and layout.
- The fixture must be manually opened or reloaded and kept in the foreground.
- Visual verification depends on the target button background staying close to the expected yellow and green colors.
- The color sample is tuned for this fixture rather than a general UI-state classifier.
- This experiment does not yet perform recovery beyond the three-attempt fixture limit.

**Result:**

Experiment 09 demonstrated observe -> locate -> act -> capture fresh observation -> visually verify -> relocate -> retry -> stop on success -> restore cursor.

### Experiment 10: macOS Accessibility Element Detection

**Date:** August 27, 2026

**Objective:**

Demonstrate read-only semantic control detection from the focused macOS Accessibility tree, validating native controls in the Experiment 10 Chrome fixture and creating an annotated screenshot of their logical actionable bounds.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_10_accessibility_elements.py`

**Fixture File:**

`assets/fixtures/phase03_screen_perception/experiment_10_accessibility_elements.html`

**Before Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_10_accessibility_elements_before.png`

**Annotated Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_10_accessibility_elements_annotated.png`

**Implemented:**

- Added the Darwin-only dependency `pyobjc-framework-ApplicationServices==12.2.2`.
- Added optional semantic metadata to `UIElement` in `src/computer_agent/perception/models.py`: `identifier`, `value`, `enabled`, `focused`, `selected`, and `source`.
- Existing `UIElement` construction remains backward compatible.
- Added the read-only `MacOSAccessibility` adapter in `src/computer_agent/perception/accessibility.py`.
- Framework imports remain safe on unsupported platforms.
- The reader exposes `is_available()`, `is_trusted()`, and `read_frontmost_controls()`.
- Frontmost application discovery uses `NSWorkspace` to obtain the PID, `AXUIElementCreateApplication` for the application element, and the focused-window Accessibility tree.
- Traversal has maximum-element and maximum-depth limits.
- Supported mappings are `AXTextField -> text_field`, `AXButton -> button`, `AXCheckBox -> checkbox`, `AXPopUpButton -> popup_button`, and `AXRadioButton -> radio_button`.
- Accessible title is preferred with description fallback.
- `AXDOMIdentifier` is preserved as `identifier`.
- Position and size are converted into logical `BoundingBox` values.
- Left and top use floor; right and bottom use ceil.
- Accessibility-derived controls use source `"accessibility"` and confidence `1.0`.
- Invalid or unavailable geometry is skipped.
- Experiment 10 performs no mouse, keyboard, focus, value-change, or structured `Action` operation.
- The experiment validates exactly eight fixture controls and draws their actionable bounds on a copied screenshot.
- Focused tests cover the Accessibility adapter in `tests/test_accessibility.py`.
- Focused model tests cover semantic `UIElement` metadata in `tests/test_perception_models.py`.

**Experiment Settings and Live Results:**

- Accessibility available: `True`
- Accessibility trusted: `True`
- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Total frontmost-window controls: `40`
- Matched fixture controls: `8`
- All eight controls had exact expected semantic types and identifiers.
- `EMPTY_TEXT_FIELD_10` was detected as a `text_field` even though its value was empty.
- `ACTIVE_BUTTON_10` was enabled.
- `DISABLED_BUTTON_10` was disabled.
- `MODE_SELECTOR_10` exposed `MODE_ALPHA_10` as its current value.
- All fixture elements used source `accessibility` and confidence `1.0`.
- All bounding boxes were inside the logical screen.
- Visual inspection confirmed that every annotation matched its actionable control area.

Checked-state limitation:

- Chrome returned unreliable `AXValue` and `AXSelected` information for native checkbox and radio checked state.
- The adapter therefore reports `selected=None` instead of making a false inference.
- Checked-state recovery is deferred to later OCR/visual and Accessibility fusion work.

**Validation:**

- `tests/test_perception_models.py` finished with `55 passed`.
- `tests/test_accessibility.py` finished with `31 passed`.
- The complete automated test suite finished with `284 passed in 0.63s`.
- Experiment script compiled successfully with `py_compile`.
- `git diff --check` passed.
- Live integration experiment completed successfully.
- Annotated screenshot was visually verified.

**Safety Measures:**

- Read-only Accessibility traversal.
- No `--execute` mode.
- No computer-control `Action`.
- No mouse movement or clicking.
- No keyboard input.
- No focus or value changes.
- Accessibility availability and trust are checked before capture.
- Exact fixture-control validation.
- Positive geometry and logical-screen bounds validation.
- Bounded tree traversal.

**Limitations:**

- The adapter is macOS-specific.
- Accessibility permission is required.
- Accessibility metadata quality depends on the application.
- Browser interface controls are returned alongside webpage controls and require semantic filtering.
- Chrome does not reliably expose checkbox/radio checked state through the tested AX attributes.
- The current experiment reads only the focused window.
- This experiment does not yet fuse Accessibility results with OCR.
- This experiment detects controls but does not interact with them.

**Result:**

Experiment 10 demonstrated that the agent can discover actionable UI semantics and logical control bounds directly from the focused macOS Accessibility tree, including an empty text field that OCR alone cannot localize reliably.

### Experiment 11: Accessibility-Grounded Text Input

**Date:** August 27, 2026

**Objective:**

Demonstrate a guarded observe -> semantically locate -> click -> verify focus -> type -> verify value -> restore cursor workflow using macOS Accessibility grounding and structured computer-control actions.

**Experiment File:**

`experiments/phase03_screen_perception/experiment_11_accessibility_text_input.py`

**Fixture File:**

`assets/fixtures/phase03_screen_perception/experiment_11_accessibility_text_input.html`

The fixture contains three native labeled text inputs:

- `DECOY_TEXT_FIELD_11` appears before the target and starts with `DECOY_VALUE_11`.
- `TARGET_TEXT_FIELD_11` is enabled and initially empty.
- `DISABLED_TEXT_FIELD_11` starts with `LOCKED_VALUE_11` and is disabled.
- Native labels provide exact accessible names.
- `load` and `pageshow` reset handlers prevent Chrome from restoring stale input values.
- The fixture contains no success text, submit button, autofocus, or automatic interaction.

**Before Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_11_accessibility_text_input_before.png`

**Plan Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_11_accessibility_text_input_plan.png`

**After Screenshot:**

`assets/screenshots/phase03_screen_perception/experiment_11_accessibility_text_input_after.png`

**Implemented:**

- The target value typed by the experiment is `ACCESSIBILITY_TYPED_VALUE_11`.
- Exact semantic text, identifier, type, enabled state, value, source, confidence, and geometry are validated.
- Accessibility logical bounds determine the click point without OCR or fixed coordinates.
- The default mode is dry-run.
- Real control requires the explicit `--execute` flag.
- Dry-run creates and executes no mouse or keyboard control `Action`.
- Execute mode uses structured `get_mouse_position`, `move_mouse`, `click_mouse`, and `type_text` actions.
- A fresh Accessibility observation and fresh target center are required immediately before movement.
- Reached mouse position is verified before clicking.
- After clicking, the target must be the unique focused control before typing is permitted.
- After typing, completion requires the exact target value.
- The decoy and disabled-field values must remain unchanged.
- The disabled field must remain disabled.
- Cursor restoration and verification run in `finally` after movement begins.
- The typed value intentionally remains in the target after success.
- The macOS Accessibility adapter used by this workflow is `src/computer_agent/perception/accessibility.py`.
- Focused fake-framework coverage for the adapter is in `tests/test_accessibility.py`.

Bounded observation recovery:

- Each semantic stage supports up to five fresh Accessibility observations.
- Retry delay is `0.25` seconds.
- Retries apply to initial, pre-movement, post-click focus, and post-typing value observations.
- No `Action` is performed from an incomplete observation.
- Exact safety validation is never weakened.

Chrome Accessibility activation:

- An initial execute attempt safely aborted because Chrome exposed browser controls but temporarily omitted the webpage Accessibility subtree.
- The captured screenshot proved that the correct fixture was fully rendered.
- Five repeated reads still omitted the webpage fields, so this was not ordinary rendering delay.
- Before activation, the reader returned `36` controls and no fixture fields.
- `AXEnhancedUserInterface` mutation was rejected with error `-25208` and is not used by the implementation.
- Reading `kAXRoleAttribute` from the `AXApplication` activated Chrome's native webpage Accessibility support.
- After the application-role request, the reader returned `39` controls and all three fixture fields.
- `MacOSAccessibility` now performs this generic read-only application-role request before reading focused UI element, focused window, and window descendants.
- The reader contains no Chrome-name branch, Accessibility setter, or internal sleep.
- A failed best-effort application-role read does not prevent otherwise valid traversal.

**Experiment Settings and Live Results:**

Live dry-run results:

- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Total frontmost-window controls: `39`
- Target box: `BoundingBox(x=197, y=469, width=1076, height=68)`
- Planned click point: `(735, 503)`
- Target was empty, enabled, unfocused, source `accessibility`, and confidence `1.0`.
- No mouse-control or keyboard-control `Action` was created or executed.

Successful execute result:

- Total initial controls: `39`
- Original cursor position: `(507, 763)`
- Fresh target box: `BoundingBox(x=197, y=469, width=1076, height=68)`
- Fresh click point: `(735, 503)`
- Reached cursor position: `(735, 503)`
- Target focus was verified before typing.
- Final target value: `ACCESSIBILITY_TYPED_VALUE_11`
- Final decoy value: `DECOY_VALUE_11`
- Final disabled value: `LOCKED_VALUE_11`
- Cursor restored to: `(507, 763)`
- No successful-stage observation retry was needed.
- The execute workflow completed successfully.

**Validation:**

- `tests/test_accessibility.py`: `40 passed`
- Complete automated test suite: `293 passed in 0.65s`
- Experiment script compiled successfully with `py_compile`.
- `git diff --check` passed.
- Semantic fixture validation passed.
- Dry-run completed successfully.
- Execute mode completed successfully.
- Before, plan, and after screenshots were visually verified.

**Safety Measures:**

- Explicit `--execute` gate.
- Five-second initial countdown.
- Three-second pre-movement countdown.
- PyAutoGUI fail-safe remains enabled.
- Exact semantic target and identifier validation.
- Initial target must be empty and enabled.
- Decoy and disabled-field invariant checks.
- Positive geometry and logical-screen edge validation.
- Fresh observation and coordinate recalculation before movement.
- Reached-position verification.
- Focus verification before typing.
- Final value verification after typing.
- Bounded observation retries without weakened validation.
- Structured `Action` objects only.
- Cursor restoration in `finally`.

**Limitations:**

- macOS Accessibility permission is required.
- Accessibility metadata depends on application support.
- Chrome may initialize its webpage Accessibility tree on demand.
- The fixture must be manually opened, freshly reloaded, and kept in front.
- The target is predetermined rather than chosen by task reasoning.
- The typed test value is ASCII.
- This experiment covers text input but not checkbox, radio, popup, or submit workflows.
- It does not yet fuse Accessibility elements with OCR.
- Checkbox/radio checked state remains unknown in the current Chrome AX mapping.

**Result:**

Experiment 11 demonstrated a guarded semantic input workflow from Accessibility-based field discovery through structured clicking, verified focus, structured typing, exact value verification, invariant protection, and cursor restoration.

### Experiment 12: Hybrid Accessibility and OCR Perception

**Date:** August 27, 2026

**Objective:**

Demonstrate a complete hybrid perception/action loop where different controls require Accessibility, fused Accessibility+OCR, or OCR-only grounding.

**Files:**

- Experiment script: `experiments/phase03_screen_perception/experiment_12_hybrid_perception.py`
- Fixture: `assets/fixtures/phase03_screen_perception/experiment_12_hybrid_perception.html`
- Before screenshot: `assets/screenshots/phase03_screen_perception/experiment_12_hybrid_perception_before.png`
- Plan screenshot: `assets/screenshots/phase03_screen_perception/experiment_12_hybrid_perception_plan.png`
- After screenshot: `assets/screenshots/phase03_screen_perception/experiment_12_hybrid_perception_after.png`

**Reusable Implementation:**

- Configurable Tesseract PSM values `0..13`, with default PSM `11`.
- Optional word grouping by OCR line.
- Conservative minimum line confidence.
- `recognize_region()` with full-image coordinate restoration.
- `normalize_ui_text()`.
- `smaller_area_overlap_ratio()`.
- `UIElementFusion`.
- Semantic metadata preservation.
- Accessibility-first actionable bounds.
- OCR-only preservation.
- Overlapping duplicate removal.

**Live Target Results:**

`TARGET_INPUT_12`:

- Source: `accessibility`
- Type: `text_field`
- Identifier: `hybrid-target-input`
- Initial value: empty
- Box: `x=196, y=275, width=760, height=59`

`NATIVE_BUTTON_12`:

- Source: `hybrid`
- Type: `button`
- Identifier: `native-button`
- Returned exactly once.
- Box: `x=475, y=389, width=520, height=71`

`CANVAS_ACTION_12`:

- Source: `ocr`
- Initial confidence: `0.95`
- Box: `x=434, y=629, width=340, height=45`

**Execute Results:**

- Original cursor: `(127, 532)`
- Initial observation attempt 1 saw Canvas confidence `0.50`.
- Retry recovered to `0.95` on attempt 2.
- Input click point: `(576, 304)`
- Focus verified through Accessibility.
- Typed `HYBRID_INPUT_VALUE_12`.
- Post-typing full-screen Canvas confidence dropped to `0.34`.
- Dynamic regional PSM 7 recovered it to `0.95`.
- Recovered box: `x=434, y=637, width=340, height=37`
- Overlap ratio: `1.00`
- Canvas click point: `(604, 656)`
- Canvas was clicked exactly once.
- Completion OCR: `FUSION VERIFIED 12`
- Completion confidence: `0.68`
- Completed median RGB: `(197, 242, 186)`
- Expected RGB: `(197, 243, 186)`
- Composite verification passed.
- Input value remained `HYBRID_INPUT_VALUE_12`.
- Native button remained exactly one hybrid element.
- Cursor restored to `(127, 532)`.

**Validation:**

- `tests/test_ocr.py`: `47 passed`
- `tests/test_perception_fusion.py`: `24 passed`
- Complete suite: `341 passed in 0.65s`
- Experiment script compiled successfully.
- `git diff --check` passed.
- Dry-run passed.
- Live execute passed.
- Before, plan, and after screenshots visually verified.

**Safety:**

- Dry-run default.
- Explicit `--execute`.
- Structured `Action` objects only.
- No direct pyautogui calls.
- Fail-safe enabled.
- Fresh observation before actions.
- No hardcoded target coordinates, crop coordinates, or Retina scale.
- Action OCR threshold stayed `0.70`.
- Regional recovery derived from the last accepted target.
- Canvas clicked only once.
- Completion required OCR + green visual state + Accessibility input value.
- Color alone could not declare success.
- Cursor restored in `finally`.

**Limitations:**

- macOS Accessibility is platform-specific.
- Chrome Accessibility may require the read-only application-role probe.
- Full-screen Tesseract confidence can change when focus styling changes.
- Canvas action semantics still come from OCR text/task context.
- Completed color classification is fixture-specific.
- Foreground fixture switching is manual.
- The experiment uses predetermined target names rather than task reasoning.

**Result:**

Experiment 12 demonstrated the complete flow: observe → Accessibility/OCR → coordinate mapping → fusion → semantic input action → fresh observation → regional OCR recovery → OCR-only Canvas action → fresh observation → composite verification → cursor restoration.

Phase 03 is complete. Phase 04 will turn these perception results into task-dependent target selection and action decisions.

**Next Step:** Phase 04 — UI Grounding and Task Reasoning

## Phase 04: UI Grounding and Task Reasoning

### Experiment 01: Reusable Perception Engine

**Date:** August 28, 2026

**Objective:**

Extract the reusable hybrid observation pipeline from Phase 03 Experiment 12 into a production `PerceptionEngine` without carrying forward target selection, planning, action execution, recovery, or verification logic.

**Extraction Boundary:**

- Moved only the reusable observation sequence: screen capture, RGB image loading, Accessibility collection, OCR recognition, pixel-to-logical OCR coordinate mapping, Accessibility/OCR fusion, warnings, and evidence metadata.
- Kept fixture text, target validation, click-point calculation, retry/recovery, regional OCR, Canvas color sampling, CLI action flow, mouse movement, keyboard typing, and post-action verification in experiment-only code.
- The engine observes only and never moves or clicks the mouse, types, pastes, presses keys, opens URLs, switches applications, creates actions, selects task targets, plans, or verifies task completion.

**Production Files:**

- Added `src/computer_agent/perception/engine.py`.
- Updated `src/computer_agent/perception/__init__.py`.
- Added `tests/test_perception_engine.py`.

**Experiment Script:**

`experiments/phase04_ui_grounding_task_reasoning/experiment_01_perception_engine.py`

The script reuses the Phase 03 Experiment 12 fixture:

`assets/fixtures/phase03_screen_perception/experiment_12_hybrid_perception.html`

It is import-safe, `--help` safe, macOS-only, manually focused, and observation-only. It does not open the fixture, switch applications, use the tool system, move or click the mouse, type, paste, press keys, run recovery, or execute any action.

**Public API:**

`snapshot = engine.observe()`

`PerceptionEngine` requires injected capture, Accessibility, OCR, fusion, and capture-path dependencies. It does not instantiate `ComputerController`, `MacOSAccessibility`, `TesseractOCR`, or `UIElementFusion` internally.

`PerceptionSnapshot` stores the `ScreenFrame`, a detached RGB image, logical Accessibility elements, logical OCR elements, fused elements, warnings, and computed source counts. It does not duplicate the frame timestamp or screen metadata.

**Partial Failure Semantics:**

- Accessibility failure returns an empty Accessibility tuple, records a warning beginning with `Accessibility observation failed:`, and continues OCR and fusion.
- OCR recognition or OCR coordinate-mapping failure returns an empty OCR tuple, records a warning beginning with `OCR observation failed:`, and still fuses available Accessibility elements.
- If both sources fail, fusion is called with two empty tuples and both warnings are returned in deterministic Accessibility-then-OCR order.
- Screen capture failure, image-open failure, image-size mismatch, and fusion failure remain fail-fast.

**Deterministic Unit Coverage:**

- Complete Accessibility + OCR observation.
- Accessibility-only partial success.
- OCR-only partial success.
- Both sources fail.
- OCR coordinate-mapping failure as OCR partial-source failure.
- Image-size mismatch before source calls.
- Capture failure propagation.
- Fusion failure propagation.
- Repeated fresh observations.
- Computed source counts without mutable stored count state.
- Constructor independence from controllers, tool registries, tool executors, mouse tools, keyboard tools, browser tools, and application tools.

**Live Result:**

- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Coordinate scale: `x=2.00`, `y=2.00`
- Capture timestamp: `2026-08-28T17:22:37.208806+00:00`
- Accessibility element count: `30`
- Logical OCR element count: `6`
- Fused element count: `35`
- Warnings: none
- Fused element source distribution: `{'accessibility': 29, 'hybrid': 1, 'ocr': 5}`
- `TARGET_INPUT_12`: observed
- `NATIVE_BUTTON_12`: observed
- `CANVAS_ACTION_12`: observed
- Live acceptance result: passed

**Evidence Screenshot:**

`assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_01_perception_engine.png`

Pillow inspection confirmed the evidence file exists as a `PNG` image with pixel size `2940 x 1912` and mode `RGBA`.

**Validation:**

- `tests/test_perception_engine.py`: `11 passed`
- Complete automated test suite: `408 passed`
- `pip check`: no broken requirements
- `git diff --check` passed
- Import safety passed.
- CLI `--help` safety passed.

**Safety:**

- The live script performed observation only.
- No action was executed.
- No Phase 03 experiment was rerun.
- No fixture was added or modified.
- The evidence screenshot was generated once by the successful live run and was not regenerated during documentation closeout.

**Result:**

Experiment 04.01 completed the reusable perception engine extraction and validated it with deterministic unit tests plus one successful live observation of the Phase 03 Experiment 12 fixture.

Phase 04 is not complete.

**Next Step:** Experiment 04.02 — UI Grounding

### Experiment 02: UI Grounding

**Date:** August 29, 2026

**Objective:**

Add deterministic UI grounding that selects task targets from reusable perception snapshots without executing actions.

**Delivered Behavior:**

- Exact identifier matching.
- Normalized text matching.
- Optional role filtering.
- Enabled-state and confidence eligibility checks.
- Identifier-tier safety that prevents unsafe text fallback.
- Deterministic source-priority, distance, and confidence tie-breaking.
- Explicit `resolved`, `ambiguous`, `unsafe`, and `not_found` statuses.

**Live Fixture Cases:**

- `IDENTIFIER_TARGET_02`: `resolved`
- `ROLE_TARGET_02`: `resolved`, with the text-field candidate rejected
- `DISABLED_ONLY_02`: `unsafe`
- `BLOCKED_IDENTIFIER_02`: `unsafe`, with no fallback to text matching
- `AMBIGUOUS_TARGET_02`: `ambiguous`
- `OCR_ONLY_TARGET_02`: `resolved` from OCR with confidence `0.95`
- `MISSING_TARGET_02`: `not_found`

All seven live acceptance cases passed.

**Harness Hardening:**

- The visible fixture marker is verified from raw Accessibility/OCR evidence.
- Observation first writes a candidate screenshot.
- The candidate is promoted to the formal evidence path only after fixture identity and every acceptance check pass.
- A wrong foreground window cannot overwrite the formal evidence image.

**Live Observation:**

- Screenshot pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Scale factor: `2`
- Accessibility elements: `45`
- OCR elements: `14`
- Fused elements: `59`
- Warnings: none

**Validation:**

- Experiment harness tests: `6 passed`
- Grounding plus harness tests: `46 passed`
- Complete suite: `454 passed`
- Live acceptance result: passed
- Implemented by commit `43cc488` (`feat: add deterministic UI grounding`).

**Safety:**

- The experiment remained observation-only.
- No action was executed.
- Evidence promotion is protected by fixture identity and acceptance checks.
- Runtime target selection uses the production grounder; the experiment harness verifies expected outcomes and evidence handling.

**Result:**

Experiment 04.02 completed deterministic UI grounding with protected live evidence and explicit target-resolution statuses.

Phase 04 is not complete.

**Next Step:** Phase 04 Experiment 03 — Action Grounding

### Experiment 03: Action Grounding

**Date:** August 29, 2026

**Objective:**

Convert deterministic UI grounding results into safe, structured, unexecuted click Actions.

**Production Contract:**

- Added `ActionGroundingStatus` with `ready` and `blocked`.
- Added immutable `ActionGroundingResult`.
- Added `ActionGrounder.ground_click(...)`.
- Reused the existing `computer_agent.core.models.Action`.
- A resolved target can produce:

```python
Action(
    tool_name="click_mouse",
    arguments={"x": x, "y": y},
    reason="Click the UI element resolved by deterministic grounding.",
)
```

- `ambiguous`, `unsafe`, and `not_found` grounding results cannot produce an Action.
- Coordinates use exact integer floor-center conversion.
- Logical screen size is validated.
- The default one-pixel safe edge margin blocks outer-edge coordinates.
- Screens without a usable safe interior are blocked.
- No executor, controller, desktop operation, verification, retry, recovery, planner, or LLM was added.

**Live Experiment:**

- Reused the existing Experiment 02 fixture unchanged.
- Performed exactly one observation.
- Fixture marker was verified from raw Accessibility/OCR evidence.
- Candidate evidence was promoted only after all acceptance checks passed.
- Formal evidence: `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_03_action_grounding.png`

**Live Observation:**

- Pixel size: `2940 x 1912`
- Logical screen size: `1470 x 956`
- Accessibility elements: `45`
- OCR elements: `14`
- Fused elements: `59`
- Warnings: none
- Fixture marker observed: true

**Live Cases:**

Resolved case:

- Target: `IDENTIFIER_TARGET_02`
- Identifier: `identifier-target-02`
- Grounding: `resolved`
- Action grounding: `ready`
- Resolved box: `x=397, y=236, width=297, height=48`
- Generated tool: `click_mouse`
- Generated arguments: `{"x": 545, "y": 260}`
- Action was not executed.

Blocked case:

- Target: `DISABLED_ONLY_02`
- Grounding: `unsafe`
- Action grounding: `blocked`
- Generated Action: none

**Validation:**

- Affected grounding/action/harness tests: `100 passed`
- Action Grounder focused tests: `50 passed`
- Experiment 03 harness tests: `10 passed`
- Complete suite: `514 passed`
- `pip check`: no broken requirements
- Import check: passed
- Live acceptance: passed
- Execution: skipped

**Safety:**

- The live experiment constructed Actions only.
- No generated Action was executed.
- Evidence promotion is protected by fixture identity and acceptance checks.
- Promotion failures retain candidate evidence and preserve existing formal evidence.

**Result:**

Experiment 04.03 completed Action Grounding, while Phase 04 remains in progress.

**Next Step:** Phase 04 Experiment 04 — Verification

### Experiment 04: Verification

**Date:** August 31, 2026

**Objective:**

Add deterministic verification for target-appearance postconditions after an executed structured Action, while keeping task success separate from tool execution success.

**Production Files:**

- `src/computer_agent/verification/__init__.py`
- `src/computer_agent/verification/models.py`
- `src/computer_agent/verification/action_verifier.py`

**Verification Contract:**

- Public verifier: `ActionVerifier.verify_target_appeared(...)`
- Inputs: before `PerceptionSnapshot`, `Action`, `ToolResult`, after `PerceptionSnapshot`, and `TargetSpec`.
- Statuses: `verified`, `failed`, and `inconclusive`.
- `verified` requires before grounding `not_found` and after grounding `resolved`.
- A successful `ToolResult` alone is not task success.
- A failed `ToolResult` returns `failed`.
- Equal or older after snapshots close as `inconclusive`.
- If the target already existed before the action, or either grounding state is ambiguous or unsafe, verification is `inconclusive`.
- If the target was absent before and remains absent after a successful action, verification is `failed`.

**Live Fixture and Harness:**

The Experiment 04 fixture exposed `ACTION_TARGET_04` as the click target and `VERIFICATION_TARGET_04` as the postcondition. The live harness performed a before observation, grounded one click Action, optionally executed it only under `--execute`, performed an after observation, then verified the target-appearance postcondition.

During live testing, visible `VERIFICATION_TARGET_04` initially appeared visually but was not observed by the configured perception path. The issue was fixture exposure to the existing perception path, not a verifier defect. The focused fixture correction made the button update both visible text and `aria-label` to `VERIFICATION_TARGET_04` after the click, so the dynamic state was available to Accessibility.

**Final Live Result:**

- Live fixture required exactly one click.
- First/before verification target: `not_found`
- `ToolResult.success`: `True`
- Final verification: `verified`
- Formal evidence: `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_04_action_verification.png`

**Validation:**

- Focused verifier and Experiment 04 harness tests passed.
- Complete automated test suite: `563 passed`
- Commit: `f8b1f79` (`feat: add deterministic action verification`)

**Safety:**

- Dry-run remained the default.
- Execution required explicit `--execute`.
- Verification used production grounding behavior instead of experiment-local reimplementation.
- Candidate evidence was promoted only after fixture identity and verification acceptance passed.

**Result:**

Experiment 04 completed deterministic post-action verification while keeping Phase 04 deterministic and separating execution success from task success.

**Next Step:** Experiment 05 — Recovery and Re-grounding

### Experiment 05: Recovery and Re-grounding

**Date:** September 1, 2026

**Objective:**

Add deterministic recovery that can prepare one safe retry Action from a fresh caller-supplied snapshot after successful tool execution but failed UI verification.

**Production Files:**

- `src/computer_agent/recovery/__init__.py`
- `src/computer_agent/recovery/models.py`
- `src/computer_agent/recovery/action_recovery.py`

**Public API:**

`ActionRecovery.prepare_retry(...)`

**Deterministic Recovery Decision Table:**

- `verified` verification -> `not_needed`
- `inconclusive` verification -> `blocked`
- Tool execution failure -> `blocked`, with no UI re-grounding
- Successful execution with failed UI postcondition and exhausted attempts -> `exhausted`
- Successful execution with failed UI postcondition, fresh safe UI grounding, and ready action grounding -> `retry_ready`
- Fresh grounding that is `not_found`, `ambiguous`, or `unsafe` -> `blocked`
- Fresh action grounding that is not `ready` -> `blocked`

Recovery consumes the caller-supplied latest `PerceptionSnapshot`; it does not observe the screen itself. Production recovery contains no LLM, planning, observation, tool execution, controller, or retry loop. The distinction is explicit: a tool execution failure is terminal for deterministic recovery, while a successful execution whose UI postcondition failed may re-ground and prepare a retry.

**Live Fixture and Harness:**

The Experiment 05 fixture is a deterministic state machine: initial position A, first click moves the target to position B without completing the task, and second click exposes `VERIFICATION_TARGET_05`.

Live execute flow:

`Obs1 -> Action1 -> Obs2 -> FAILED -> ActionRecovery using Obs2 -> RETRY_READY -> Action2 -> Obs3 -> VERIFIED`

The harness binds each observation to its expected capture path before using it. Candidate-first formal evidence protection keeps existing formal evidence intact until fixture identity, verification, recovery, observation-count, execution-count, and promotion checks pass.

The experiment-local `live_harness_utils.py` contains non-domain Phase 04 live-harness plumbing. Recovery acceptance logic remains local to Experiment 05, and production recovery remains under `src/computer_agent/recovery/`.

**Import Safety:**

A direct-script import bug was found before the first dry-run: the absolute `experiments...` helper import failed under direct execution. The fix uses a relative helper import in package mode and a sibling helper import in direct script mode, with no `sys.path` manipulation. A safe direct `--help` regression test covers this path.

**Live Results:**

Dry-run:

- Observation count: `1`
- Execution count: `0`
- Result: passed

Execute:

- Observation count: `3`
- Execution count: `2`
- First `ToolResult.success`: `True`
- First verification: `failed` because the target remained `not_found`
- Recovery status: `retry_ready`
- Recovery grounding: `resolved`
- Recovery action grounding: `ready`
- Retry Action used a new `action_id` and different coordinates
- Second `ToolResult.success`: `True`
- Final verification: `verified`
- Formal evidence promoted successfully
- Formal evidence: `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_05_recovery_regrounding.png`

**Validation:**

- Recovery plus Experiment 05 focused tests: `82 passed`
- Complete automated test suite: `645 passed`
- `pip check`: no broken requirements
- `git diff --check`: passed
- Live Experiment 05 acceptance: passed

**Result:**

Experiment 05 completed deterministic recovery and fresh re-grounding through the formal Phase 04 architecture.

Phase 04 remains in progress.

**Next Step:**

Experiment 06 — Structured Planning

### Experiment 06: Structured Planning

**Date:** September 1, 2026

**Objective:**

Add deterministic structured semantic planning for ordered multi-step tasks.
The goal was to create a production planning boundary that can represent
explicit task intent without observing the screen, grounding UI elements,
generating executable Actions, executing tools, verifying outcomes, recovering,
retrying, using controllers, or invoking an LLM.

**Architecture / Production Boundary:**

Structured Planning owns:

- Ordered semantic task structure.
- Semantic action target specification.
- Semantic verification target specification.
- Bounded attempt policy.

Structured Planning does not own:

- Screen observation.
- Accessibility or OCR.
- UI grounding.
- Screen coordinates.
- Action generation.
- Tool execution.
- Verification execution.
- Recovery execution.
- Retry loops.
- Controllers.
- `pyautogui`.
- LLM reasoning.

`StructuredPlanner` is intentionally thin. It provides a stable production
construction seam, so Experiment 07 can later feed validated semantic steps
through the same seam. Complexity was not added merely to make the planner
appear substantial.

**Production Files:**

- `src/computer_agent/planning/__init__.py`
- `src/computer_agent/planning/models.py`
- `src/computer_agent/planning/structured_planner.py`

**Experiment:**

`experiments/phase04_ui_grounding_task_reasoning/experiment_06_structured_planning.py`

**Tests:**

- `tests/test_structured_planner.py`
- `tests/test_experiment_06_structured_planning.py`

**Data Contracts:**

- `PlanOperation` currently contains only `CLICK_TARGET`, whose value is
  `click_target`.
- No redundant supported-operation registry is maintained.
- `PlanStep` contains a human-readable goal, a `PlanOperation`, a semantic
  action `TargetSpec`, a semantic verification `TargetSpec`, and
  `max_attempts`.
- `StructuredPlan` contains a human-readable task goal and an ordered
  non-empty tuple of `PlanStep` objects.
- Plans are immutable and slotted.
- `MAX_PLAN_STEP_ATTEMPTS = 3`
- `MAX_STRUCTURED_PLAN_STEPS = 20`

**StructuredPlanner Public API:**

`StructuredPlanner.build_plan(task_goal=..., steps=(...))`

The builder constructs the final `StructuredPlan` from explicit semantic
`PlanStep` objects supplied by code. It does not infer targets from an
observation and does not create executable Actions.

**Validation / Invariants:**

- `PlanStep.goal` must be a non-empty string.
- `PlanStep.operation` must be a `PlanOperation`; invalid raw operation values
  are rejected.
- `PlanStep.action_target` must be a `TargetSpec`.
- `PlanStep.verification_target` must be a `TargetSpec`.
- `PlanStep.max_attempts` must be a non-boolean integer from `1` through `3`.
- `StructuredPlan.task_goal` must be a non-empty string.
- `StructuredPlan.steps` must be a tuple.
- `StructuredPlan.steps` must contain at least one `PlanStep`.
- `StructuredPlan.steps` must contain no more than `20` steps.
- Invalid explicit construction inputs raise validation errors directly.
- No `PlanningStatus` or `PlanningResult` wrapper was added because this layer
  has no runtime planning attempt outcome.

Plans do not contain executable `Action` objects, screen coordinates,
`PerceptionSnapshot`, `GroundingResult`, `ToolResult`, verification results, or
recovery results.

**Formal Headless Experiment:**

Experiment 06 is intentionally headless. It is not a live UI experiment and it
does not have screenshot evidence. It has no browser fixture, no observation,
no action execution, and no LLM. Formal evidence consists of deterministic
direct-run acceptance output plus automated test validation.

The final plan is constructed through `StructuredPlanner.build_plan(...)`, not
by directly substituting a `StructuredPlan` for the production planner seam.

**Deterministic Formal Plan:**

Task goal:

`Complete the deterministic two-step workflow`

Step 1:

- Goal: `Activate the first target`
- Operation: `click_target`
- Action target: `STEP_1_TARGET_06`
- Verification target: `STEP_1_COMPLETE_06`
- Max attempts: `2`

Step 2:

- Goal: `Activate the second target`
- Operation: `click_target`
- Action target: `STEP_2_TARGET_06`
- Verification target: `TASK_COMPLETE_06`
- Max attempts: `2`

**Acceptance Conditions:**

- Returned object must be `StructuredPlan`.
- Exact task goal.
- Exactly two steps.
- Caller order preserved.
- Both operations are `click_target`.
- Exact step goals.
- Exact action `TargetSpec` objects.
- Exact verification `TargetSpec` objects.
- Both `max_attempts == 2`.

Acceptance fails closed and prints the failed conditions. A non-`StructuredPlan`
injected result fails closed. The observation count remains `0`, and the action
execution count remains `0`.

**Direct-Run Result:**

- Direct experiment execution: passed.
- Experiment acceptance: passed.
- Execution: not applicable.
- Observation count: `0`.
- Action execution count: `0`.
- Direct script runs without `sys.path` mutation.
- `--help` exits `0` without running acceptance.
- Importing the experiment produces no execution or output.

**Validation:**

- Focused planning plus Experiment 06 tests: `32 passed`
- Complete automated test suite: `677 passed`
- `pip check`: no broken requirements
- `py_compile`: passed
- `git diff --check`: passed
- Direct experiment execution: passed
- Direct `--help`: passed

**Safety:**

- No files are written by the experiment.
- No browser or application is controlled.
- No screen observation occurs.
- No executable Action is created by the plan.
- No tool execution occurs.
- No recovery or retry loop occurs.
- No LLM is used.

**Result:**

Experiment 06 completed deterministic structured semantic planning.

Phase 04 remains in progress.

**Next Step:**

Experiment 07 — LLM Reasoner

### Experiment 07: LLM Reasoner

**Date:** September 2, 2026

**Objective:**

Add the Phase 04 LLM reasoning boundary and close it out with a formal,
headless harness. The experiment demonstrates the narrow semantic path from
natural-language task intent to an LLM client boundary, `LLMReasoner`, strict
semantic validation, `StructuredPlanner`, and final `StructuredPlan`.

**Architecture / Production Boundary:**

Experiment 07 added provider-neutral reasoning under
`src/computer_agent/reasoning/`.

Production reasoning owns:

- `LLMClient`, a provider-neutral text-generation protocol.
- `LLMReasoner`, which builds deterministic prompts and treats provider output
  as untrusted.
- `ReasoningStatus` and `ReasoningResult`.
- Strict JSON parsing, duplicate-key rejection, exact top-level, step, and
  target-key validation.
- Canonical element-type validation.
- Conversion into semantic `PlanStep` objects.
- Final construction through `StructuredPlanner`.

Production reasoning does not own:

- Screen observation.
- Perception.
- UI grounding.
- Action grounding.
- Executable `Action` objects.
- Screen coordinates.
- Mouse or keyboard execution.
- Verification.
- Recovery.
- Retry behavior.
- Agent loops.

**OpenAI Adapter:**

`OpenAILLMClient` adapts the provider-neutral boundary to the OpenAI Responses
API. It uses `openai==3.6.0`, strict Structured Outputs through a JSON schema,
and `store=False`. Importing the module does not construct `OpenAI()`; the SDK
client is constructed only when `OpenAILLMClient` is instantiated without an
injected client. Tests use injected fake SDK clients and never call the network.

Model output remains untrusted even with strict Structured Outputs. The
provider adapter supplies a schema, but code still owns JSON parsing, exact-key
validation, canonical element-type validation, bounded `max_attempts`, and
`StructuredPlanner` construction.

**Canonical Element Types:**

The supported LLM element-type vocabulary is:

- `button`
- `checkbox`
- `popup_button`
- `radio_button`
- `text_field`
- `text`

Empty `element_types` is valid and means no element-type grounding restriction.
This provides a safe fallback when the model cannot identify a UI role that the
current perception layer can actually produce.

**Live Provider Evidence:**

Two real `gpt-5.6-terra` calls were performed manually before this formal
closeout task. No live API request was made during the formal harness closeout.

The first live call returned `ready`, but generated unsupported speculative UI
roles including `link`, `menuitem`, `navigation item`, `heading`, and
`page title`. That exposed a real integration issue because `UIGrounder` treats
`TargetSpec.element_types` as a hard compatibility filter, while current
perception cannot produce those roles.

The production reasoning policy was corrected by adding the canonical
element-type vocabulary and telling the model to use empty `element_types` when
the UI role is uncertain.

The second live call used the task `Open Settings.` and succeeded:

- Status: `ready`
- Reason: `structured plan ready`
- Supported element types: `button`, `checkbox`, `popup_button`,
  `radio_button`, `text_field`, `text`
- Task goal: `Open Settings.`
- Step count: `1`
- Step goal: `Open the Settings interface.`
- Operation: `click_target`
- Action target: `Settings`
- Action element types: empty
- Verification target: `Settings`
- Verification element types: empty
- Max attempts: `3`

The successful second live call showed that the model used the empty role
fallback rather than inventing an unsupported role. No UI execution occurred in
Experiment 07.

**Production Files:**

- `src/computer_agent/reasoning/__init__.py`
- `src/computer_agent/reasoning/llm_client.py`
- `src/computer_agent/reasoning/llm_reasoner.py`
- `src/computer_agent/reasoning/models.py`
- `src/computer_agent/reasoning/openai_client.py`

**Experiment:**

`experiments/phase04_ui_grounding_task_reasoning/experiment_07_llm_reasoner.py`

The formal harness is intentionally headless:

- No fixture.
- No screenshot.
- No perception.
- No UI grounding.
- No action grounding.
- No action execution.
- No mouse or keyboard operation.
- No verification.
- No recovery.
- No agent loop.

The default mode is deterministic and offline. It injects a small experiment
owned fake LLM client that returns JSON text for a simple two-step semantic
plan. The fake does not parse, validate, or build plans; production
`LLMReasoner` and `StructuredPlanner` do that work.

Optional `--live` mode was not included. Live provider validation was performed
separately and is documented above.

**Deterministic Formal Task:**

`Complete the deterministic LLM reasoning workflow`

Step 1:

- Goal: `Activate the first reasoning target`
- Operation: `click_target`
- Action target: `STEP_1_TARGET_07`
- Action element types: empty
- Verification target: `STEP_1_COMPLETE_07`
- Verification element types: empty
- Max attempts: `3`

Step 2:

- Goal: `Activate the second reasoning target`
- Operation: `click_target`
- Action target: `STEP_2_TARGET_07`
- Action element types: empty
- Verification target: `TASK_COMPLETE_07`
- Verification element types: empty
- Max attempts: `3`

**Acceptance Conditions:**

- `ReasoningStatus.READY`.
- Result plan is a `StructuredPlan`.
- Exact task goal.
- Exactly two steps.
- Exact step order.
- Exact goals.
- Exact operations.
- Exact action target text.
- Exact verification target text.
- Exact `max_attempts`.
- Exact element types.
- No executable `Action` objects.
- No x, y, coordinate, or reference-point authority.
- Fake client called exactly once.
- LLM provider is deterministic fake.
- Live API request is `no`.
- Observation count is `0`.
- Action execution count is `0`.

Failures return non-zero and print the failed conditions.

**Tests:**

`tests/test_experiment_07_llm_reasoner.py`

The tests cover deterministic success, exact plan content, exact step order,
exact target texts, exact operations, exact max attempts, exact element types,
fake-client call count, malformed provider output, blocked reasoning results,
wrong task goal, wrong step count, wrong order, wrong action target, wrong
verification target, wrong max attempts, absence of executable Actions and
coordinate authority, `--help`, import safety, and direct deterministic script
execution.

**Direct-Run Result:**

- Direct experiment execution: passed.
- Experiment acceptance: passed.
- LLM provider: deterministic fake.
- Live API request: `no`.
- Observation count: `0`.
- Action execution count: `0`.
- Direct `--help`: passed.
- Importing the experiment produces no execution or output.

**Validation:**

- Focused Experiment 07 reasoning tests: `135 passed`
- Complete automated test suite: `793 passed`
- `pip check`: no broken requirements
- `py_compile`: passed
- `git diff --check`: passed
- Direct experiment execution: passed
- Direct `--help`: passed

**Safety:**

- No API key, credential, billing, or payment details are stored or printed.
- No test performs a network request.
- No live provider call was made during this closeout task.
- No UI execution occurred.
- LLM reasoning produces only semantic `StructuredPlan` data.
- Experiment 08 will consume `StructuredPlan` and execute the deterministic
  `observe -> ground -> act -> verify -> recover` behavior.

**Result:**

Experiment 07 completed provider-neutral LLM reasoning and the formal headless
acceptance harness.

Phase 04 remains in progress.

### Experiment 08: Agent Loop

**Date:** September 2, 2026

**Objective:**

Add the deterministic production agent-loop orchestration layer and close it
out with a live UI harness that consumes an existing `StructuredPlan`.

Experiment 08 does not accept arbitrary natural-language tasks directly. The
division remains explicit: Experiment 07 produces a `StructuredPlan` from LLM
reasoning, and Experiment 08 consumes a `StructuredPlan` for deterministic
execution.

**Production Architecture:**

Added deterministic agent orchestration under `src/computer_agent/agent/`:

- `AgentLoopStatus`
- `AgentLoopResult`
- `AgentLoop`

`AgentLoop` consumes a `StructuredPlan` and creates an `AgentState` from
`plan.task_goal`. It supports only `PlanOperation.CLICK_TARGET` and fails
visibly for unsupported future operations.

The loop itself does not call an LLM, OpenAI, natural-language planning, or any
provider API. It does not generate coordinates. Coordinates come from current UI
grounding through `ActionGrounder` on the initial attempt and from
`ActionRecovery` on retry attempts.

**Execution Flow:**

For each semantic `PlanStep`, `AgentLoop` orchestrates:

`PerceptionEngine.observe() -> UIGrounder.ground(...) -> ActionGrounder.ground_click(...) -> ToolExecutor.execute(...) -> AgentState.record_step(...) -> PerceptionEngine.observe() -> ActionVerifier.verify_target_appeared(...)`

If verification succeeds, the current semantic plan step is complete and
`completed_plan_steps` is incremented.

If verification fails or is inconclusive, the loop calls
`ActionRecovery.prepare_retry(...)` with the latest snapshot, completed attempt
count, and the step's `max_attempts`.

Initial UI grounding outcomes other than `resolved` fail closed as
`AgentLoopStatus.BLOCKED`. Initial action-grounding outcomes other than `ready`
also fail closed as `blocked`. Existing recovery policy owns
`inconclusive`, failed `ToolResult`, `blocked`, `exhausted`, and
`retry_ready` semantics; the loop does not duplicate that policy.

On `retry_ready`, the loop uses `ActionRecovery`'s supplied Action directly. It
does not call `UIGrounder` again and does not call `ActionGrounder` again. The
retry `before_snapshot` is the previous attempt's `after_snapshot`, preserving
the correct verification pairing.

`AgentState.steps` records actual Action executions. `completed_plan_steps`
records verified semantic `PlanStep` completion. In the final live run,
`AgentState.steps` contained three click attempts while
`completed_plan_steps == 2`.

**Result Model:**

`AgentLoopResult` is immutable and slotted. It validates:

- Status is `completed`, `blocked`, or `exhausted`.
- `plan` is a `StructuredPlan`.
- `state` is an `AgentState`.
- `completed_plan_steps` is a non-boolean integer within the plan length.
- `reason` is non-empty.
- `completed` requires all plan steps complete and `AgentState.succeeded`.
- `blocked` and `exhausted` require `AgentState.failed`.

**Dependency Composition Correction:**

Review found a partial-dependency-injection issue before closeout. A custom
grounder could be injected while default verifier/recovery silently created
separate production grounders. The constructor was corrected so the all-default
case shares the exact same `UIGrounder` across initial grounding, verifier, and
recovery, and shares the exact same `ActionGrounder` across initial action
grounding and recovery. Custom grounders or action grounders now require
explicit verifier/recovery dependencies rather than silent replacement.

**Formal Live Harness:**

`experiments/phase04_ui_grounding_task_reasoning/experiment_08_agent_loop.py`

Default mode is safe and offline:

- Builds the deterministic plan.
- Prints fixture and plan details.
- Performs no observation.
- Takes no screenshot.
- Constructs no live executor path.
- Executes no mouse action.

Live execution requires explicit `--execute`. The harness tells the user to
open the local fixture manually in Chrome or Safari, keep it visible and
focused, and waits before invoking the production loop. It reuses existing
Phase 04 live wiring through `live_harness_utils.py`:

- `build_live_perception_engine`
- `build_live_tool_executor`

The live `AgentLoop` is created as:

```python
AgentLoop(
    perception_engine=real_perception_engine,
    executor=real_tool_executor,
)
```

This exercises the production default shared `UIGrounder`,
`ActionGrounder`, `ActionVerifier`, and `ActionRecovery`.

**Deterministic Formal Plan:**

Task goal:

`Complete the deterministic Agent Loop workflow`

Step 1:

- Goal: `Recover and complete the first UI target`
- Operation: `click_target`
- Action target: `STEP_1_TARGET_08`
- Verification target: `STEP_1_COMPLETE_08`
- Max attempts: `2`

Step 2:

- Goal: `Complete the final UI target`
- Operation: `click_target`
- Action target: `STEP_2_TARGET_08`
- Verification target: `TASK_COMPLETE_08`
- Max attempts: `1`

The plan contains no executable `Action` objects and no coordinates.

**Fixture:**

`assets/fixtures/phase04_ui_grounding_task_reasoning/experiment_08_agent_loop.html`

Initial state contains exactly one actionable button with visible text and
`aria-label`:

`STEP_1_TARGET_08`

The first click succeeds at the tool level but does not create
`STEP_1_COMPLETE_08`. It keeps exactly one `STEP_1_TARGET_08` target visible
and moves it to a deterministic, clearly different location. This forces
verification to return `failed`, causing recovery to re-ground the moved target
and prepare a retry Action.

The second click removes the step-one action target, creates
`STEP_1_COMPLETE_08`, and creates the actionable `STEP_2_TARGET_08` button.
The third click removes the final target and creates `TASK_COMPLETE_08`.

**Live Failure, Diagnosis, and Fixture Correction:**

The first real live execution intentionally exercised recovery:

- Attempt 1: `click_mouse {'x': 499, 'y': 438}`, `ToolResult.success=True`
- First click moved `STEP_1_TARGET_08`.
- Recovery re-grounded the moved target.
- Attempt 2: `click_mouse {'x': 1033, 'y': 704}`,
  `ToolResult.success=True`

The first and second click coordinates were observed runtime values and were
not hardcoded. Their difference proved real re-grounding and execution of the
recovery-supplied retry Action.

After attempt 2, the browser visibly transitioned to:

- `STEP_1_COMPLETE_08`
- `STEP_2_TARGET_08`

However the loop ended `exhausted` because verification still returned
`failed`. A diagnostic observation on the already-transitioned page showed:

`STEP_1_COMPLETE_08`:

- Accessibility: absent
- OCR: absent
- Fused: absent
- Grounding: `not_found`

`STEP_2_TARGET_08`:

- Accessibility: button
- Fused: present
- Grounding: `resolved`

Root cause: the fixture created completion markers as `<div role="status">`,
which current `MacOSAccessibility` did not expose, and OCR also failed to
recognize the marker. This was a fixture/perception contract mismatch, not an
AgentLoop, recovery, or timing bug.

No production perception, grounding, verifier, recovery, or AgentLoop behavior
was changed for this issue. The fixture-only correction changed completion
markers to enabled native buttons with:

- `type="button"`
- `tabindex="-1"`
- Exact `aria-label`
- Exact visible text
- `pointer-events: none`
- `cursor: default`
- No click handlers
- Not disabled

Follow-up observation diagnostics proved:

`STEP_1_COMPLETE_08`:

- Accessibility: `1`
- Type: `button`
- Confidence: `1.0`
- Enabled: `True`
- Fused: `1`
- Grounding: `resolved`

`STEP_2_TARGET_08`:

- Accessibility: `1`
- Type: `button`
- Confidence: `1.0`
- Enabled: `True`
- Grounding: `resolved`

**Final Live Success:**

The second full live execution succeeded:

- `AgentLoopResult.reason`: `all plan steps completed`
- Agent loop status: `completed`
- Agent state: `succeeded`
- Completed plan steps: `2 / 2`
- Action executions: `3`

Observed runtime actions:

- Attempt 1: `click_mouse {'x': 499, 'y': 438}`,
  `ToolResult.success=True`
- Attempt 2: `click_mouse {'x': 1033, 'y': 704}`,
  `ToolResult.success=True`
- Attempt 3: `click_mouse {'x': 749, 'y': 526}`,
  `ToolResult.success=True`

Experiment acceptance passed, and recovery retry was demonstrated. The
coordinates above are live evidence from this run, not hardcoded coordinates.

**Evidence Screenshot:**

`assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_08_agent_loop.png`

The screenshot shows the final successful fixture state with
`STEP_1_COMPLETE_08`, `TASK_COMPLETE_08`, and the workflow-complete status. It
matches the repository's Phase 04 screenshot evidence convention and is kept as
formal Experiment 08 evidence.

**Tests:**

- `tests/test_agent_loop.py`
- `tests/test_experiment_08_agent_loop.py`

The tests cover loop success, multi-step behavior, typed terminal outcomes,
retry attempt counting, recovery-supplied Actions, no duplicate grounding or
action grounding on retry, before/after snapshot pairing, result invariants,
dependency-injection sharing, import safety, dry/default CLI safety, `--help`
safety, offline execute routing with injected fakes, formal acceptance
conditions, fixture dynamic state transitions, and the corrected accessible
completion-marker representation.

No test performs live mouse actions, opens a browser, takes screenshots, or
calls the network.

**Validation:**

- Experiment 08 targeted tests after fixture correction: `78 passed`
- Earlier adjacent deterministic stack: `223 passed`
- Final focused closeout stack: `230 passed`
- Complete automated test suite: `871 passed`
- `pip check`: no broken requirements
- `py_compile`: passed
- `git diff --check`: passed
- Dry direct run: passed
- Direct `--help`: passed

**Safety:**

- `--execute` is required for real UI execution.
- Default direct execution cannot move or click the mouse.
- `--help` has no observation or action side effects.
- Importing the experiment produces no output or execution.
- No API keys, secrets, billing data, or unrelated machine-sensitive
  information were stored.
- No LLM/OpenAI dependency exists in `AgentLoop`.

**Result:**

Experiment 08 completed deterministic Agent Loop production orchestration and
the formal live UI harness.

Phase 04 remains in progress.

**Next Step:**

Experiment 09 — Dynamic UI
