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
- [x] Experiment 09: Dynamic UI
- [x] Experiment 10: Cross-Application Agent

Phase 04 is complete. Experiment 04.01 extracted the reusable hybrid observation pipeline from Phase 03 Experiment 12 into `PerceptionEngine`, whose public API is `snapshot = engine.observe()`.

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

The canonical reasoning element-type vocabulary at the time of Experiment 07 was `button`, `checkbox`, `popup_button`, `radio_button`, `text_field`, and `text`. Empty `element_types` is valid and means no element-type grounding restriction. Separate live validation with `gpt-5.6-terra` first exposed unsupported-role hallucination (`link`, `menuitem`, `navigation item`, `heading`, and `page title`) that Phase 04 perception could not yet produce; after adding the canonical vocabulary and empty fallback policy, the second live call for `Open Settings.` returned `ready` with an empty element-type fallback. No UI execution occurred in Experiment 07.

The formal Experiment 07 harness is headless and offline by default. It uses a deterministic fake LLM client, makes no live API request, observes no screen state, creates no fixture or screenshot, performs no UI grounding, creates no executable actions, and performs no verification or recovery. Its formal task is `Complete the deterministic LLM reasoning workflow`, producing two `click_target` semantic steps with `max_attempts=3`, empty element types, and targets `STEP_1_TARGET_07`, `STEP_1_COMPLETE_07`, `STEP_2_TARGET_07`, and `TASK_COMPLETE_07`. Current final validation: focused Experiment 07 reasoning tests `135 passed`, complete suite `793 passed`, `pip check` reported no broken requirements, `py_compile` passed, `git diff --check` passed, direct experiment execution passed, and direct `--help` passed.

Experiment 08 added the deterministic production `AgentLoop` orchestration layer under `src/computer_agent/agent/`. The loop consumes an existing `StructuredPlan` and does not call an LLM or OpenAI. It orchestrates `observe -> ground -> action ground -> execute -> observe -> verify -> recover/re-ground/retry`, fails closed on initial grounding or action-grounding failure, and relies on the existing recovery policy for verification `inconclusive`, failed `ToolResult`, `retry_ready`, `blocked`, and `exhausted` outcomes. `AgentState.steps` records actual executed Actions, while `completed_plan_steps` records verified semantic `PlanStep` completion. On `retry_ready`, the loop executes the `ActionRecovery`-supplied Action without duplicate UI grounding or action grounding, and the retry `before_snapshot` is the previous `after_snapshot`.

The formal Experiment 08 harness uses a deterministic pre-built two-step `StructuredPlan`; it does not accept arbitrary natural-language tasks directly. Experiment 07 produces `StructuredPlan` from LLM reasoning, while Experiment 08 consumes a `StructuredPlan` for deterministic execution. Coordinates are not hardcoded in the plan or harness; they come from current UI grounding through `ActionGrounder` and `ActionRecovery`. The live fixture intentionally required recovery: the first click moved `STEP_1_TARGET_08`, recovery re-grounded it, and the retry clicked different observed runtime coordinates. A first live run exposed a fixture/perception contract mismatch because `<div role="status">` completion markers were not exposed by current perception; the fixture was minimally corrected to create enabled native button completion markers with exact labels/text, `tabindex=-1`, `pointer-events:none`, and no click handlers. No production perception expansion was needed. The second live run completed with status `completed`, state `succeeded`, `2 / 2` completed plan steps, `3` successful `click_mouse` executions, and recovery retry demonstrated. Formal evidence is `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_08_agent_loop.png`.

Experiment 09 completed the first formal Reasoner -> StructuredPlan -> AgentLoop integration. A natural-language task is passed to a deterministic fake LLM provider, production `LLMReasoner` validates the provider output into the exact `StructuredPlan`, and production `AgentLoop` executes that plan against a bounded three-state dynamic fixture. Formal acceptance remains deterministic and offline: it uses the fake provider once, makes no live API request, and does not claim arbitrary natural-language execution beyond the bounded semantic planning contract. The live fixture changes from an initial control, to a reflowed second state, to a foreground modal state containing a semantic decoy. The final live run completed `3 / 3` semantic steps with exactly `3` successful `click_mouse` actions, no recovery retry, and runtime-grounded click coordinates rather than plan constants. No production-code changes were required. Formal evidence is `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_09_dynamic_ui.png`.

Final Experiment 09 validation: focused Experiment 09 tests `36 passed`; Experiment 07 regression `90 passed`; Experiment 08 regression `78 passed`; complete suite `907 passed`; `pip check` reported no broken requirements in the project virtual environment; `py_compile` passed; `git diff --check` passed; dry-run passed; direct `--help` passed; live Experiment 09 acceptance passed.

Experiment 10 completed the Cross-Application Agent milestone. The formal natural-language task is `Transfer the deterministic browser fixture value CROSS_APP_TRANSFER_10 into the blank TextEdit document.` Acceptance uses a deterministic offline fake LLM provider; no live OpenAI request is required. The semantic sequence is `click_target -> read_clipboard -> activate_app -> insert_text`, and production execution is `click_mouse -> read_from_clipboard -> activate_app -> paste_text`.

Experiment 10 proves a bounded autonomous agent can transfer runtime state from a browser UI into another macOS application through a structured semantic plan: `Natural-language task -> deterministic fake LLM -> LLMReasoner -> StructuredPlan -> AgentLoop -> UI grounding / tool execution / verification -> runtime AgentState value flow -> application switching -> final editable-value verification`. Runtime value flow is `browser fixture -> system clipboard -> read_from_clipboard ToolResult -> AgentState.context["values"]["transfer_value"] -> paste_text`; `InsertTextStep` stores only the runtime value key and does not contain the literal transfer text.

The successful live acceptance result was `PASS`: `4 / 4` plan steps completed, `4` action executions, zero semantic retries, no duplicate paste, and final runtime value `CROSS_APP_TRANSFER_10`. The final TextEdit content verification uses macOS Accessibility focused editable state with exact value equality, not OCR. Formal evidence is `assets/screenshots/phase04_ui_grounding_task_reasoning/experiment_10_cross_application_agent.png`, which shows the final TextEdit document containing the runtime value.

Experiment 10 also produced an important reusable macOS finding: `NSWorkspace.frontmostApplication` can remain stale in a command-line Python process unless the Cocoa run loop is serviced. `MacOSAccessibility` now refreshes AppKit state at the shared frontmost-application lookup boundary before reading `NSWorkspace.frontmostApplication`, so both `read_frontmost_application_name()` and `read_frontmost_controls()` observe current post-activation state. `ComputerController.activate_app("TextEdit")` was not the root bug and remains the production activation primitive.

Final Experiment 10 validation: focused closeout tests `284 passed`; complete suite `1127 passed`; `pip check` reported no broken requirements in the project virtual environment; `py_compile` passed for changed Python files; `git diff --check` passed; dry-run remains the default; live Experiment 10 acceptance passed.

Phase 04 UI Grounding and Task Reasoning is complete, including the original Task Reasoning plus core Autonomous Agent milestone.

### Phase 05: Real-World Web Autonomy

- [x] Experiment 01: Real Web Accessibility Perception
- [x] Experiment 02: Real Web Semantic Grounding
- [x] Experiment 03: Verified Web Navigation
- [x] Experiment 04: Scroll and Viewport Search
- [x] Experiment 05: Real Web Text Input
- [x] Experiment 06: Web Information Extraction
- [x] Experiment 07: Adaptive Web Recovery
- [x] Experiment 08: Multi-Step Real Web Agent
- [x] Experiment 09: Live OpenAI Web Agent
- [x] Experiment 10: Cross-Site Generalization

Experiment 05.10 completed the cross-site generalization milestone on Khan
Academy SAT Math. Khan read-only perception/grounding qualification,
deterministic offline planning, live OpenAI planning-only qualification, and
gated live production `AgentLoop` execution all passed. The accepted live run
completed exactly one semantic `click_mouse` action and one plan step, produced
one generic state-transition verification with `4 verified, 0 failed, 0
inconclusive`, resolved the final `Unit 2: Foundations: Algebra` heading,
promoted formal evidence, and ended with `Execution acceptance: passed`.

An earlier live attempt was correctly rejected by the fail-closed precondition
gate before any browser action or evidence promotion. Subsequent read-only
qualification and the successful live rerun confirmed the Khan semantic
contract without weakening grounding or verification safety.

Experiment 05.01 moved the production perception pipeline from controlled HTML fixtures to a real public website. A raw macOS Accessibility audit of `https://www.python.org/` in Google Chrome visited `682` Accessibility nodes and confirmed that real webpages expose semantic roles including `AXLink`, `AXHeading`, `AXStaticText`, `AXTextField`, `AXButton`, and `AXWebArea`.

Production `MacOSAccessibility` now maps `AXLink -> link`, `AXHeading -> heading`, and `AXStaticText -> text`. Live inspection also established that Chrome exposes visible `AXStaticText` content primarily through `AXValue` while `AXTitle` and `AXDescription` may be empty. The production reader therefore uses `AXValue` as a text fallback only for the normalized `text` role, preserving existing control-name semantics for buttons, text fields, and other controls.

The formal read-only live acceptance used the production `PerceptionEngine` against python.org. It reported `152` Accessibility elements, `18` OCR elements, `167` fused elements, and no warnings. The real webpage produced semantic `link`, `heading`, `text`, `text_field`, `button`, `popup_button`, and `radio_button` elements with valid logical-screen geometry and meaningful web text. The experiment performed no mouse movement, clicking, typing, scrolling, navigation, or OpenAI request.

Formal evidence is `assets/screenshots/phase05_real_web_autonomy/experiment_01_real_web_accessibility.png`.

Experiment 05.01 validation: Accessibility regression `53 passed`; complete suite `1129 passed` using `python -m pytest -q`; `pip check` reported no broken requirements; `py_compile` passed; `git diff --check` passed; and live real-web acceptance passed.

Experiment 05.02 validated the existing deterministic `UIGrounder` against real `python.org` perception output without changing production grounding logic. The experiment grounds directly against `PerceptionSnapshot.fused_elements`, matching the same input used by the autonomous `AgentLoop`.

Real-web semantic collisions were handled correctly. The visible text `Docs` appeared simultaneously as `link`, `heading`, and `text` elements, yet `TargetSpec(text="Docs", element_types=("link",))` resolved uniquely to the link while the same target constrained to `heading` resolved uniquely to the heading. `Search This Site` similarly resolved to the `text_field` while the same-text static `text` candidate was rejected as an incompatible element type. A deliberately missing link returned `not_found`.

The formal Experiment 05.02 acceptance remained completely read-only: no mouse movement, click, typing, scrolling, navigation, computer-action execution, or OpenAI request occurred. Final live acceptance passed with no perception warnings. Evidence is `assets/screenshots/phase05_real_web_autonomy/experiment_02_real_web_semantic_grounding.png`.

Experiment 05.02 added regression coverage for real-web role collisions in `tests/test_ui_grounder.py`. The complete UI-grounder module passed `43 tests`; the complete repository suite passed `1132 tests`; `pip check` reported no broken requirements; `py_compile` and `git diff --check` passed.

Experiment 05.03 completed the first verified real-world web navigation loop. Starting from `https://www.python.org/`, the production perception pipeline resolved the `Search This Site` text field as a start-page identity marker and resolved `Docs` uniquely as a `link`. `ActionGrounder` converted that semantic target into one safe `click_mouse` action at logical screen coordinates `(635, 148)`.

In live execute mode, `ToolExecutor` executed exactly one grounded click. After navigation, a fresh production observation found `Library reference` as a `link` on the Python documentation page. `ActionVerifier` verified the transition because the target was `not_found` before the click and `resolved` afterward. The live result was `verified` with no perception warnings.

The experiment remains dry-run by default and requires explicit `--execute` for real interaction. A start-page identity gate, semantic action grounding, safe coordinate grounding, before/after observations, target-appearance verification, and candidate-evidence promotion prevent a successful tool call from being treated as task success without semantic postcondition evidence. Formal evidence is preserved as both `experiment_03_verified_web_navigation_before.png` and `experiment_03_verified_web_navigation.png`.

Experiment 05.03 added seven harness regression tests covering dry-run safety, failed preconditions, exactly-once execution, successful verification and evidence promotion, failed verification, and failed tool execution. The complete repository suite passed `1139 tests`; `pip check` reported no broken requirements; `py_compile` and `git diff --check` passed; and the real-web execute acceptance passed.

Experiment 05.04 has a first read-only diagnostic increment for scroll and viewport search. Production perception now has optional-geometry raw semantic targets, viewport visibility classification, and deterministic target diagnosis statuses: `visible`, `needs_search`, `not_found`, and `blocked`. The experiment reads the frontmost Chrome app name, unique `AXWebArea` viewport, and raw Accessibility semantics only; it performs no mouse movement, clicking, typing, scrolling, navigation, clipboard modification, or LLM request. On the validated python.org observation, the unique `Privacy Notice` `AXLink` existed with unavailable geometry, so the correct diagnosis is `needs_search` rather than `not_found`.

Experiment 05.04 validation for this increment: focused Experiment 05.04 harness test `1 passed`; focused viewport plus experiment tests `18 passed`; requested viewport, viewport-grounding, UI-grounder, and Accessibility regressions `116 passed`; complete suite passed with `PYTHONPATH=src` as `1160 passed`; plain `python -m pytest -q` still exposed three pre-existing subprocess import-path failures in direct-import/help tests; `python -m pip check` still reported the environment `pynacl`/`cffi` mismatch; `py_compile` for changed Python files and `git diff --check` passed.

Experiment 05.04 then added bounded downward viewport search behind an explicit `--execute` flag. Dry-run remains the default and performs exactly one observation; if the target is currently visible it reports `found`, and if the read-only diagnosis is `needs_search` it reports `needs_scroll` with zero scrolls. Execute mode uses the existing structured `scroll` tool through `Action` and `ToolExecutor`, never clicks the found target, re-observes after every scroll, stops on `found`, `blocked`, `stalled`, or `exhausted`, and caps downward scroll attempts by policy.

Bounded-search validation: focused Phase 05.04 tests `13 passed`; viewport plus Phase 05.04 tests `24 passed`; requested viewport, viewport-grounding, UI-grounder, and Accessibility tests `116 passed`; existing scroll/tool tests `48 passed`; `PYTHONPATH=src python -m pytest -q` passed with `1172 passed`; `python -m pip check` still reports the documented `pynacl`/`cffi` environment mismatch.

Live python.org validation closed out the bounded viewport-search increment. At the top of the page, the unique `AXLink` target `Privacy Notice` had `bounds=None`, `VisibilityStatus.GEOMETRY_UNAVAILABLE`, and diagnosis `needs_search`. Execute mode with default policy (`max_scroll_attempts=6`, `scroll_amount=4`) dispatched six successful structured scroll tool calls but exhausted the budget while the target remained geometry-unavailable, showing that dispatch and re-observation worked but that this calibration was insufficient for this page. Execute mode with `scroll_amount=12` found the target after four of six allowed downward scroll attempts; the target became `BoundingBox(x=878, y=922, width=72, height=15)`, visibility became `visible`, final search status was `found`, and no click was performed. With the page left at the footer, dry-run performed one observation, zero scrolls, diagnosed the same target as `visible`, and returned `found`, confirming the controller stops before scrolling when the target is already visible and uses fresh state for each run.

This is successful python.org calibration evidence for this environment, not a universal scroll amount or a claim that arbitrary webpage scrolling is solved. The current implementation remains downward-only, hard-budget bounded by `max_scroll_attempts`, limited to one primary viewport, and does not handle nested scroll containers, infinite-scroll pages, clicking, typing, or LLM recovery.

Experiment 05.05 starts real-web text input on python.org's `Search This Site` text field. The first increment adds a deterministic `TextInputController` that observes Chrome, requires a reliable viewport, grounds the visible `text_field` target through existing `UIGrounder` and `ActionGrounder`, dry-runs to `needs_action` without executing, and in execute mode performs only a focus `click_mouse` action followed by `type_text`. Verification is based on a fresh post-action observation where the field's semantic `value` must equal the requested text and differ from the pre-action value; successful click/type `ToolResult`s alone are not accepted as success.

This increment also extends `SemanticAXElement` with an explicit optional `value` field so raw Accessibility semantics can preserve entered values without overloading the field label/name.

Experiment 05.05 validation: focused tests `13 passed`; relevant grounding and viewport-search tests `112 passed`; relevant Accessibility, viewport, and 05.05 tests `87 passed`; click/type/tool executor tests `54 passed`; complete suite passed with `PYTHONPATH=src python -m pytest -q` as `1186 passed`; `python -m pip check` still reports the documented `pynacl`/`cffi` environment mismatch.

Live python.org validation closed out Experiment 05.05 for the minimal single-visible-field text-input scope. Dry-run on `Google Chrome` at `python.org` resolved the `Search This Site` `text_field` with empty value, bounds `BoundingBox(x=937, y=213, width=224, height=38)`, no warnings, ready focus action `click_mouse {"x": 1049, "y": 232}`, status `needs_action`, and action execution count `0`. Execute mode typed `computer agent` after one successful focus click and one successful `type_text` action, then a fresh post-action observation resolved the same field with value `computer agent`, bounds `BoundingBox(x=873, y=213, width=288, height=38)`, no warnings, final status `verified`, and action execution count `2`.

Experiment 05.05 did not press Enter, submit the search form, navigate, use clipboard paste, or call an LLM. This completes only the current minimal scope: one deterministic visible field, one focus click, one type action, fresh semantic value verification, and a pre-action empty-field requirement. Arbitrary form filling, arbitrary websites, offscreen text-field search validation, multi-field forms, submission, Enter-key behavior, and LLM recovery remain out of scope.

Experiment 05.06 completed the current deterministic python.org Latest News semantic-extraction scope. The production `semantic_extraction` module locates one `AXHeading` section by normalized text, slices semantic elements until the next heading, and parses ordered news records from date text plus authoritative `AXLink` titles. It accepts `bounds=None`, combines split date fragments such as `2026-` plus `09-01`, ignores duplicate static-text title copies and `>>>More`, and reports explicit malformed-input statuses instead of inventing missing data.

Live read-only validation used `PYTHONPATH=src python experiments/phase05_real_web_autonomy/experiment_06_web_information_extraction.py` with Google Chrome frontmost on `https://www.python.org/`. The result was `passed`: `490` raw semantic elements, `0` actions, `1` matching `Latest News` heading, `24` section elements, following heading `Upcoming Events`, news status `extracted`, `5` records, and `0` issues. Extracted records were `2026-09-01 — The 2026 PSF Board Election is Open!`, `2026-09-01 — Inaugural Python Packaging Council Election: Voting is now open!`, `2026-09-01 — Python 3.15.0 candidate 2 is here!`, `2026-08-31 — Kojo Idrissa: 2026 PSF Board Election Candidate Interview`, and `2026-08-25 — Ramya Ravi: 2026 PSF Board Election Candidate Interview`.

Experiment 05.06 validates structure and extraction success from live Accessibility semantics and structured parsing; no pre-known live titles are used as PASS conditions. It does not claim arbitrary websites, arbitrary webpage structures, or generic news extraction are solved. Limitations remain: one known semantic section pattern, heading-delimited section extraction, deterministic date/title parsing, no scrolling, no clicking, no typing, no navigation, no OCR, no screenshots, no clipboard mutation, and no LLM.

Experiment 05.06 final automated validation: focused Phase 05.06 tests `25 passed`; complete suite passed with `PYTHONPATH=src python -m pytest -q` as `1211 passed`; `py_compile` passed; `git diff --check` passed; `python -m pip check` still reports the documented `pynacl`/`cffi` environment mismatch.

Experiment 05.07 completed the implemented cause-aware adaptive web recovery scope for web interaction, currently integrated into text-input recovery eligibility and live-validated with bounded viewport search. The new pure web recovery decision layer maps `not_found` grounding to `viewport_search`; maps `unsafe` to `viewport_search` only when every candidate rejection reason is exactly `("outside_viewport",)`; and maps `ambiguous`, `resolved`, disabled, incompatible element type, low or invalid confidence, identifier/text conflict, invalid bounding box, and mixed rejection reasons to `block`. `TextInputController` now attempts viewport search only for trusted observations and a structured `viewport_search` decision. Existing `ViewportSearchController`, `ActionRecovery`, and retry-loop behavior were reused rather than replaced.

Live validation on `https://www.python.org/` targeted `"Privacy Notice"` as a `link`. Initial state was Google Chrome with viewport `BoundingBox(x=0, y=124, width=1470, height=832)`, initial grounding `not_found`, reason `no exact identifier or normalized text match`, and candidate count `0`. Recovery decision was `viewport_search` with reason `semantic target was not found; bounded viewport search is eligible`. Bounded downward viewport search returned `found` with reason `target became visible` after `4` successful scroll actions. Final grounding was `resolved`, reason `resolved by text`, final text `"Privacy Notice"`, final bounds `BoundingBox(x=878, y=922, width=72, height=15)`, and total action execution count `4`. No click, typing, or navigation occurred.

Experiment 05.07 final automated validation: focused Phase 05.07-related tests `52 passed`; complete suite passed with `PYTHONPATH=src python -m pytest -q` as `1250 passed in 4.74s`; `git diff --check` passed. This is intentionally not a generic arbitrary web recovery planner. Current limitations remain: narrow recovery decision space, downward bounded viewport search only, no bidirectional search, no LLM-based recovery choice, and no automatic click after recovery.

Experiment 05.08 completed the current bounded multi-step real-web AgentLoop workflow on `https://www.python.org/`. Increment 1 added `PlanOperation.TYPE_INTO_TARGET` and frozen/slotted `WebTextInputStep` support to `StructuredPlan`; `AgentLoop` dispatches that step through the existing production `TextInputController`. The submit/navigation step reuses the existing `click_target` `PlanStep`. The final live workflow used one `StructuredPlan` and one `AgentLoop.run(plan)` call, with no manual sequencing outside AgentLoop and no hardcoded coordinates in the plan.

The formal live task resolved the `"Search This Site"` `text_field`, typed `typing`, resolved and clicked the `"GO"` `button`, and verified the result page by the appearance of the `"Results"` `heading`. Initial `"Results"` grounding was `not_found`; final `"Results"` grounding was `resolved`. Live acceptance passed with `AgentLoopResult` status `completed`, `AgentState` status `succeeded`, `2 / 2` completed plan steps, `3` action executions, exact tool order `click_mouse -> type_text -> click_mouse`, all `ToolResult`s successful, no scroll action in the validated happy path, final app `Google Chrome`, and empty final perception warnings.

Formal evidence is `assets/screenshots/phase05_real_web_autonomy/experiment_08_multi_step_real_web_agent.png`. Candidate evidence was promoted only after acceptance passed. Validation: focused tests `245 passed`; full suite `1321 passed in 5.21s`; `git diff --check` passed; live Experiment 05.08 acceptance passed.

Experiment 05.08 remains bounded. Verification still uses the predefined postcondition marker `"Results"`, `StructuredPlan` has no semantic extraction step yet, and cross-site generalization is not complete.

Experiment 05.09 completed the bounded Live OpenAI Web Agent workflow for the same python.org task. The formal architecture is: `Natural-language task -> real OpenAI Responses API -> OpenAILLMClient -> LLMReasoner -> trusted StructuredPlan -> deterministic semantic plan acceptance -> strict live precondition gate -> one production AgentLoop.run(plan) -> production TextInputController / UI grounding / ToolExecutor -> real browser execution -> final semantic verification`.

Increment 1 added `LLMReasoner` support for `PlanOperation.TYPE_INTO_TARGET`. The reasoner can now construct the existing production `WebTextInputStep` from the provider-neutral `type_into_target` JSON shape with `goal`, `operation`, `target`, `input_text`, and `max_attempts`. `max_attempts` must be exactly integer `1`; existing safe `TargetSpec` policy is reused; the OpenAI strict Structured Outputs schema includes `type_into_target`; and existing operations remained unchanged.

Increment 2 added the planning-only Experiment 05.09 harness. Default mode remains deterministic/offline, while `--live-openai` performs exactly one real OpenAI planning request and no browser actions. The first real OpenAI planning run was safely blocked before browser execution because the model inferred `Results` as `element_types=("text",)`. That was not accepted because Phase 05.08 live Accessibility evidence established that `Results` is exposed as a heading, while the current reasoning vocabulary intentionally does not support `heading`. Experiment acceptance was not weakened; instead, the generic reasoning prompt was strengthened so semantic roles are not treated as visible-content descriptions, `"text"` is not used merely because text appears, and unknown roles use empty `element_types`. The second real OpenAI planning run passed acceptance.

The accepted real OpenAI plan contained two steps. Step 1 was `WebTextInputStep`, operation `type_into_target`, target text `"Search This Site"`, target element types `("text_field",)`, input text `"typing"`, and `max_attempts=1`. Step 2 was `PlanStep`, operation `click_target`, action target text `"GO"`, action target element types `("button",)`, verification target text `"Results"`, verification target element types `()`, and `max_attempts=1`.

Increment 3 added explicit gated execution mode: `--live-openai --execute`. `--live-openai` alone remains planning-only, and `--execute` without `--live-openai` is rejected. Executor construction occurs only after plan acceptance and strict read-only live preconditions pass: macOS, Accessibility available/trusted, Google Chrome frontmost, viewport available, empty perception warnings, `"Search This Site"` resolved as an empty `text_field`, `"GO"` resolved as a `button`, and `"Results"` initially `not_found`. Core execution is exactly one `AgentLoop(...).run(trusted_plan)` call, with no manual sequencing outside `AgentLoop` and no hardcoded coordinates in the LLM plan.

The formal live execution passed. Planning acceptance passed; initial state was Google Chrome with `"Search This Site"` resolved, `"GO"` resolved, and `"Results"` `not_found`; `AgentLoopResult.reason` was `all plan steps completed`; status was `completed`; `AgentState` was `succeeded`; completed plan steps were `2 / 2`; action executions were `3`; exact tool order was `click_mouse -> type_text -> click_mouse`; all `ToolResult`s succeeded; and no scroll occurred in the validated happy path. Final `"Results"` grounding was `resolved`, final app remained `Google Chrome`, final warnings were `()`, evidence was promoted, and execution acceptance passed.

Formal evidence is `assets/screenshots/phase05_real_web_autonomy/experiment_09_live_openai_web_agent.png`. Candidate evidence was promoted only after full execution acceptance passed. Validation: Increment 3 focused suite `414 passed`; complete repository suite `1405 passed in 5.74s`; `git diff --check` passed; real OpenAI planning-only acceptance passed; and real OpenAI plus real AgentLoop execution acceptance passed.

Experiment 05.09 remains bounded to the predefined python.org workflow. `"Results"` remains a predefined postcondition marker. There is no dynamic postcondition discovery yet, no semantic extraction step emitted by the LLM plan yet, and cross-site generalization is not complete. This does not claim arbitrary web autonomy.

### Phase 06: Evidence-Grounded Interactive Agent

- [x] Experiment 01: Interactive Task Runtime
- [ ] Experiment 02: Agent Workspace
- [ ] Experiment 03: Evidence-Grounded Task State
- [ ] Experiment 04: Adaptive Next-Step Reasoning
- [ ] Experiment 05: Task-Level Verification and Reconciliation
- [ ] Experiment 06: Persistence and Resume
- [ ] Experiment 07: Controlled Reliability Benchmark
- [ ] Experiment 08: DeltaMath External Transfer
- [ ] Experiment 09: Useful Cross-Application Workflow
- [ ] Experiment 10: Integrated Phase 06 Demonstration

Experiment 06.01 introduced the reusable interactive task runtime under `src/computer_agent/runtime/`. It adds a runtime-owned task lifecycle, cooperative pause/resume/stop control, structured runtime events, and a synchronous event bus without changing the existing Phase 04-05 `AgentLoop`.

`RuntimeControl` uses explicit safe checkpoints. A pause request blocks future progress only when execution reaches `control.checkpoint()`; it does not interrupt an action halfway through. Resume wakes a paused checkpoint, while stop wakes a paused checkpoint and raises `RuntimeStopRequested` so the runtime can terminate cleanly.

The deterministic Experiment 06.01 acceptance harness validates two independent scenarios. The first demonstrates `started -> paused -> resumed -> completed` and proves that worker progress does not cross the checkpoint while paused. The second demonstrates `started -> paused -> stop_requested -> stopped` and proves that no worker progress occurs after the stop request.

Final Experiment 06.01 validation: focused runtime and experiment tests `30 passed`; complete repository suite `1738 passed in 7.11s`; `pip check` reported no broken requirements; `git diff --check` passed; and direct Experiment 06.01 acceptance passed.
