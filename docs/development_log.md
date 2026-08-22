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