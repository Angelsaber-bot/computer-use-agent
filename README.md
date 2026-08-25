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

Experiment 04 uses the original Retina screenshot at `2940 x 1912` with minimum confidence `0.70`, producing `93` accepted word-level OCR elements. OCR bounding boxes are high-resolution pixel coordinates, not PyAutoGUI logical coordinates; Experiment 05 now converts them through `ScreenCoordinateMapper`.

Experiment 05 maps OCR pixel boxes into PyAutoGUI logical coordinates using `ScreenFrame` scale metadata. It floors logical left/top edges and ceils logical right/bottom edges so each mapped logical box contains the full pixel OCR region.

Experiment 06 captures a live screen, runs Tesseract OCR, converts pixel boxes to PyAutoGUI logical coordinates, supports exact and opt-in partial text matching, prefers exact matches when selecting a target, and creates a visualization without moving or clicking the mouse.

**Next Step:** Phase 03 Experiment 07 — Safe Mouse Movement to a Localized Text Target
