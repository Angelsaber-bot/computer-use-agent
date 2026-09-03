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
- [x] Experiment 03: Action Grounding
- [x] Experiment 04: Verification
- [x] Experiment 05: Recovery and Re-grounding
- [x] Experiment 06: Structured Planning
- [x] Experiment 07: LLM Reasoner
- [x] Experiment 08: Agent Loop

Phase 04 is in progress. Experiment 04.01 extracted the reusable hybrid observation pipeline from Phase 03 Experiment 12 into `PerceptionEngine`, whose public API is `snapshot = engine.observe()`.

The engine uses injected screen capture, Accessibility, OCR, and fusion dependencies. `PerceptionSnapshot` contains the `ScreenFrame` metadata, a detached RGB image, logical Accessibility/OCR/fused element tuples, warnings, and computed source counts. Accessibility and OCR failures are partial-source failures with warnings, while capture failures, image-size mismatches, and fusion failures remain fail-fast. The engine only observes; it has no action, target-selection, planning, verification, application-switching, mouse, or keyboard behavior.

Experiment 04.01 reused the Phase 03 Experiment 12 fixture and saved evidence at `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_01_perception_engine.png`. The successful live observation reported pixel size `2940 x 1912`, logical screen size `1470 x 956`, scale `x=2.00`, `y=2.00`, timestamp `2026-08-28T17:22:37.208806+00:00`, `30` Accessibility elements, `6` logical OCR elements, `35` fused elements, no warnings, source distribution `{'accessibility': 29, 'hybrid': 1, 'ocr': 5}`, and observed `TARGET_INPUT_12`, `NATIVE_BUTTON_12`, and `CANVAS_ACTION_12`.

Experiment 04.02 added deterministic UI grounding over observed UI elements. It resolves targets through exact identifier matching, normalized text matching, optional role filtering, enabled-state and confidence eligibility, identifier-tier safety that prevents unsafe text fallback, and deterministic source-priority, distance, and confidence tie-breaking. Grounding returns explicit `resolved`, `ambiguous`, `unsafe`, and `not_found` results. The live harness validates fixture identity from raw Accessibility/OCR evidence and writes a candidate screenshot before promoting it to protected formal evidence only after fixture identity and all acceptance checks pass.

Experiment 04.03 converts a resolved `GroundingResult` into the existing structured `Action` model, creating `click_mouse` with integer logical-screen coordinates. It applies deterministic floor-center conversion and configurable safe screen-edge margins, returns explicit `ready` or `blocked` results, and never executes the generated Action. The live harness reuses the Experiment 02 fixture and protects formal evidence through candidate-first promotion.

Experiment 04 added the reusable deterministic `ActionVerifier` for target-appearance postconditions. It consumes a before `PerceptionSnapshot`, an `Action`, a `ToolResult`, an after `PerceptionSnapshot`, and a `TargetSpec`, and returns explicit `verified`, `failed`, or `inconclusive` status. Successful verification requires the before target to be `not_found` and the after target to be `resolved`; tool execution success alone is not task success. Stale or non-new after snapshots fail closed as `inconclusive`. The live fixture required exactly one click: first/before verification target `not_found`, `ToolResult.success == True`, and final verification `verified`. Formal evidence is `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_04_action_verification.png`. Commit `f8b1f79` (`feat: add deterministic action verification`) completed Experiment 04 with the full suite at `563 passed`.

Experiment 05 added the reusable deterministic `ActionRecovery` with explicit `retry_ready`, `not_needed`, `blocked`, and `exhausted` outcomes. Recovery does not execute tools or observe the screen; it consumes the caller-supplied latest `PerceptionSnapshot`. `verified` maps to `not_needed`, `inconclusive` maps to `blocked`, tool execution failure maps to `blocked` without UI re-grounding, and exhausted attempts map to `exhausted`. A successful execution with a failed UI postcondition may re-ground; safe fresh UI grounding plus action grounding returns `retry_ready`, and the retry `Action` is newly generated from the latest snapshot. The experiment-local `live_harness_utils.py` contains non-domain Phase 04 live-harness plumbing; recovery acceptance logic remains local to Experiment 05, and production recovery remains under `src/computer_agent/recovery/`.

Live Experiment 05 passed in dry-run with `1` observation and `0` executions. Execute mode passed with exactly `3` observations and exactly `2` click executions: the first `ToolResult` succeeded, first verification failed because the target remained `not_found`, recovery returned `retry_ready`, recovery grounding was `resolved`, recovery action grounding was `ready`, the retry `Action` had a new `action_id` and different coordinates, the second `ToolResult` succeeded, final verification was `verified`, and formal evidence was promoted successfully to `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_05_recovery_regrounding.png`. Current final validation: recovery plus Experiment 05 focused tests `82 passed`, complete suite `645 passed`, `pip check` reported no broken requirements, `git diff --check` passed, and live Experiment 05 acceptance passed.

Experiment 06 added deterministic semantic planning under `src/computer_agent/planning/`. The public production models are `PlanOperation`, `PlanStep`, and `StructuredPlan`, and the public builder is `StructuredPlanner.build_plan(...)`. The initial supported operation is `click_target`. Each `PlanStep` contains a human-readable goal, a semantic action `TargetSpec`, a semantic verification `TargetSpec`, and bounded `max_attempts` in `1..3`. `StructuredPlan` contains a human-readable task goal and an ordered non-empty tuple of `PlanStep` objects, with at most `20` steps. Plans are immutable/slotted, invalid explicit construction inputs raise validation errors, and no `PlanningStatus` or `PlanningResult` wrapper was added because this layer has no runtime planning attempt outcome. Plans do not contain executable `Action` objects, screen coordinates, `PerceptionSnapshot`, `GroundingResult`, `ToolResult`, verification results, or recovery results.

The formal Experiment 06 harness is intentionally headless: no browser fixture, screenshot, observation, action execution, or LLM. It constructs the formal final plan through `StructuredPlanner.build_plan(...)`. The deterministic task goal is `Complete the deterministic two-step workflow`. Step 1 is `Activate the first target`, operation `click_target`, action target `STEP_1_TARGET_06`, verification target `STEP_1_COMPLETE_06`, and `max_attempts=2`. Step 2 is `Activate the second target`, operation `click_target`, action target `STEP_2_TARGET_06`, verification target `TASK_COMPLETE_06`, and `max_attempts=2`. Formal direct-run acceptance passed with execution not applicable, observation count `0`, and action execution count `0`. Current final validation: focused planning plus Experiment 06 tests `32 passed`, complete suite `677 passed`, `pip check` reported no broken requirements, `py_compile` passed, `git diff --check` passed, direct experiment execution passed, and direct `--help` passed.

Experiment 07 added provider-neutral LLM reasoning under `src/computer_agent/reasoning/`. `LLMClient` is the provider boundary, `LLMReasoner` owns strict JSON parsing and exact-key semantic validation, and the final plan is constructed through `StructuredPlanner`. The OpenAI adapter uses the Responses API with strict Structured Outputs and `store=False` on `openai==3.6.0`. Model output remains untrusted: code owns parsing, schema validation, canonical element-type validation, and planner construction. The LLM layer cannot produce executable `Action` objects, screen coordinates, observations, UI grounding, tool execution, verification, recovery, or an agent loop.

The canonical reasoning element-type vocabulary is `button`, `checkbox`, `popup_button`, `radio_button`, `text_field`, and `text`. Empty `element_types` is valid and means no element-type grounding restriction. Separate live validation with `gpt-5.6-terra` first exposed unsupported-role hallucination (`link`, `menuitem`, `navigation item`, `heading`, and `page title`) that current perception cannot produce; after adding the canonical vocabulary and empty fallback policy, the second live call for `Open Settings.` returned `ready` with an empty element-type fallback. No UI execution occurred in Experiment 07.

The formal Experiment 07 harness is headless and offline by default. It uses a deterministic fake LLM client, makes no live API request, observes no screen state, creates no fixture or screenshot, performs no UI grounding, creates no executable actions, and performs no verification or recovery. Its formal task is `Complete the deterministic LLM reasoning workflow`, producing two `click_target` semantic steps with `max_attempts=3`, empty element types, and targets `STEP_1_TARGET_07`, `STEP_1_COMPLETE_07`, `STEP_2_TARGET_07`, and `TASK_COMPLETE_07`. Current final validation: focused Experiment 07 reasoning tests `135 passed`, complete suite `793 passed`, `pip check` reported no broken requirements, `py_compile` passed, `git diff --check` passed, direct experiment execution passed, and direct `--help` passed.

Experiment 08 added the deterministic production `AgentLoop` orchestration layer under `src/computer_agent/agent/`. The loop consumes an existing `StructuredPlan` and does not call an LLM or OpenAI. It orchestrates `observe -> ground -> action ground -> execute -> observe -> verify -> recover/re-ground/retry`, fails closed on initial grounding or action-grounding failure, and relies on the existing recovery policy for verification `inconclusive`, failed `ToolResult`, `retry_ready`, `blocked`, and `exhausted` outcomes. `AgentState.steps` records actual executed Actions, while `completed_plan_steps` records verified semantic `PlanStep` completion. On `retry_ready`, the loop executes the `ActionRecovery`-supplied Action without duplicate UI grounding or action grounding, and the retry `before_snapshot` is the previous `after_snapshot`.

The formal Experiment 08 harness uses a deterministic pre-built two-step `StructuredPlan`; it does not accept arbitrary natural-language tasks directly. Experiment 07 produces `StructuredPlan` from LLM reasoning, while Experiment 08 consumes a `StructuredPlan` for deterministic execution. Coordinates are not hardcoded in the plan or harness; they come from current UI grounding through `ActionGrounder` and `ActionRecovery`. The live fixture intentionally required recovery: the first click moved `STEP_1_TARGET_08`, recovery re-grounded it, and the retry clicked different observed runtime coordinates. A first live run exposed a fixture/perception contract mismatch because `<div role="status">` completion markers were not exposed by current perception; the fixture was minimally corrected to create enabled native button completion markers with exact labels/text, `tabindex=-1`, `pointer-events:none`, and no click handlers. No production perception expansion was needed. The second live run completed with status `completed`, state `succeeded`, `2 / 2` completed plan steps, `3` successful `click_mouse` executions, and recovery retry demonstrated. Formal evidence is `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_08_agent_loop.png`.

**Next Step:** Phase 04 Experiment 09 — Dynamic UI.
