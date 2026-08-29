# Computer-Use AI Agent

A Python-based AI agent that can observe a computer screen, make decisions, and perform actions using the mouse and keyboard.

## Setup

- Install Python dependencies with `python -m pip install -r requirements.txt`.
- OCR support uses `pytesseract==0.3.13` and requires the external `tesseract` executable on `PATH`. The Tesseract executable is a system prerequisite and is not installed through pip.

## Current Progress

### Phase 01: Basic Computer Control

- [x] Read the mouse position
- [x] Move and click the mouse
- [x] Type text automatically
- [x] Read and write clipboard content
- [x] Scroll a page
- [x] Capture screenshots
- [x] Open webpages
- [x] Switch between applications
- [x] Complete a cross-app workflow

### Phase 02: Tool System and Agent State

- [x] Define structured actions, results, observations, and step records
- [x] Track task status and execution history
- [x] Create a common tool interface and tool registry
- [x] Execute structured actions through a safe tool executor
- [x] Wrap all 14 computer-control functions as registered tools
- [x] Complete a real macOS integration experiment

### Phase 03: Screen Perception

- [x] Experiment 01: Normalized Screen Capture
- [x] Experiment 02: Reusable `BoundingBox` and `UIElement` perception models
- [x] Experiment 03: Image Preprocessing
- [x] Experiment 04: OCR Text Recognition on high-resolution RGB screenshots
- [x] Experiment 05: OCR Coordinate Mapping
- [x] Experiment 06: UI Text Localization
- [x] Experiment 07: Safe Mouse Movement to a Localized Text Target
- [x] Experiment 08: Verified Click on a Localized Text Target
- [x] Experiment 09: Recovery Retry with Visual State Verification
- [x] Experiment 10: macOS Accessibility Element Detection
- [x] Experiment 11: Accessibility-Grounded Text Input
- [x] Experiment 12: Hybrid Accessibility and OCR Perception

Experiment 04 uses the original Retina screenshot at `2940 x 1912` with minimum confidence `0.70`, producing `93` accepted word-level OCR elements. OCR bounding boxes are high-resolution pixel coordinates, not PyAutoGUI logical coordinates; Experiment 05 now converts them through `ScreenCoordinateMapper`.

Experiment 05 maps OCR pixel boxes into PyAutoGUI logical coordinates using `ScreenFrame` scale metadata. It floors logical left/top edges and ceils logical right/bottom edges so each mapped logical box contains the full pixel OCR region.

Experiment 06 captures a live screen, runs Tesseract OCR at minimum confidence `0.05` to retain low-confidence text candidates, converts pixel boxes to PyAutoGUI logical coordinates, supports exact and partial text matching, extracts the matching target substring from longer OCR strings, estimates a target-only logical bounding box from character position, draws the extracted target box and selected center, and performs no mouse or keyboard action.

Experiment 07 captures and localizes target text using the existing perception pipeline, collects OCR candidates with a `0.05` confidence threshold, and applies a separate `0.70` action-confidence threshold before movement. It selects the highest-confidence exact match first, uses partial matching only as fallback, defaults to dry-run, requires `--execute` for real movement, moves through structured tool `Action` objects, verifies the reached position, restores the original cursor position, and never clicks.

Experiment 08 captures and OCR-localizes a target, defaults to dry-run, and requires `--execute` for real control. In execute mode it moves and clicks through structured tool `Action` objects, captures the screen again, succeeds only after detecting `CLICK_VERIFIED`, and restores and verifies the original cursor position.

Experiment 09 defaults to dry-run and requires `--execute` for real control. It OCR-localizes `RECOVERY_TARGET_09` from a fresh screenshot on every attempt, clicks through structured mouse `Action` objects, verifies completion from the target background color rather than OCR status text, treats light yellow as incomplete and light green as completed, relocalizes a moved target and retries up to three attempts, and restores the original cursor position.

Experiment 10 reads the focused macOS window through the Accessibility API and detects semantic controls including empty text fields, buttons, checkboxes, popup buttons, and radio buttons. It returns roles, accessible names, identifiers, values, enabled/focused state, logical bounding boxes, and source metadata; distinguishes enabled and disabled controls; detects an empty input field without relying on visible OCR text; remains completely read-only; and produces an annotated screenshot.

Experiment 11 locates a specific empty text field through macOS Accessibility semantics, selects it among an enabled decoy and a disabled field, defaults to dry-run, and requires `--execute` for real control. It uses structured mouse and keyboard `Action` objects, verifies focus before typing, verifies the exact final Accessibility value, confirms the decoy and disabled fields remain unchanged, and restores the original cursor position.

Experiment 12 completes the hybrid perception/action loop. It reads an Accessibility-only empty input, runs full-screen PSM 6 line-level OCR, maps OCR pixel boxes to logical coordinates, fuses Accessibility and OCR elements, deduplicates a native button visible to both sources, types through Accessibility-grounded structured actions, clicks an OCR-only Canvas action, dynamically recovers low-confidence full-screen OCR with regional PSM 7, verifies completion using exact OCR, Canvas color, and Accessibility value, and restores the cursor.

Phase 03 Screen Perception is complete.

### Phase 04: UI Grounding and Task Reasoning

- [x] Experiment 01: Reusable Perception Engine
- [x] Experiment 02: UI Grounding
- [ ] Experiment 03: Action Grounding

Phase 04 is in progress. Experiment 04.01 extracted the reusable hybrid observation pipeline from Phase 03 Experiment 12 into `PerceptionEngine`, whose public API is `snapshot = engine.observe()`.

The engine uses injected screen capture, Accessibility, OCR, and fusion dependencies. `PerceptionSnapshot` contains the `ScreenFrame` metadata, a detached RGB image, logical Accessibility/OCR/fused element tuples, warnings, and computed source counts. Accessibility and OCR failures are partial-source failures with warnings, while capture failures, image-size mismatches, and fusion failures remain fail-fast. The engine only observes; it has no action, target-selection, planning, verification, application-switching, mouse, or keyboard behavior.

Experiment 04.01 reused the Phase 03 Experiment 12 fixture and saved evidence at `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_01_perception_engine.png`. The successful live observation reported pixel size `2940 x 1912`, logical screen size `1470 x 956`, scale `x=2.00`, `y=2.00`, timestamp `2026-08-28T17:22:37.208806+00:00`, `30` Accessibility elements, `6` logical OCR elements, `35` fused elements, no warnings, source distribution `{'accessibility': 29, 'hybrid': 1, 'ocr': 5}`, and observed `TARGET_INPUT_12`, `NATIVE_BUTTON_12`, and `CANVAS_ACTION_12`.

Experiment 04.02 added deterministic UI grounding over observed UI elements. It resolves targets through exact identifier matching, normalized text matching, optional role filtering, enabled-state and confidence eligibility, identifier-tier safety that prevents unsafe text fallback, and deterministic source-priority, distance, and confidence tie-breaking. Grounding returns explicit `resolved`, `ambiguous`, `unsafe`, and `not_found` results. The live harness validates fixture identity from raw Accessibility/OCR evidence and writes a candidate screenshot before promoting it to protected formal evidence only after fixture identity and all acceptance checks pass.

The complete automated test suite now finishes with `454 passed`.

**Next Step:** Phase 04 Experiment 03 — Action Grounding
