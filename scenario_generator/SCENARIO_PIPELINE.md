# Scenario Generator Pipeline (CoLMDriver)

This document onboards new contributors to the LLM-driven scenario pipeline used by CoLMDriver/MDriveBench. It combines a step-by-step flow with a file-by-file map of the codebase and shared assets.

---

## Pipeline Steps at a Glance (each in one sentence)
1) **Crop selection** – Find a map crop in the chosen town that matches the requested topology/geometry for the scenario.  
2) **Legal path enumeration** – Build lane connectivity inside that crop and enumerate boundary-crossing legal paths with signatures for every candidate ego route.  
3) **Path picking** – Assign one legal path per ego using schema-driven CSP or LLM-extracted constraints from the description.  
4) **Path refinement** – (Skipped for roundabouts) Trim/extend starts, inject lane-change macros, and set speeds via a soft CSP guided by LLM constraints.  
5) **Object placement** – Two-stage LLM + CSP to place static props and moving NPCs with guardrails and spacing checks, producing the final scene.  
6) **Route export** – Convert the scene to CARLA route XMLs plus manifests/behaviors and optionally align them with CARLA’s GlobalRoutePlanner.  
7) **Validation & retries** – (Audit runner only) Validate scenes against specs, hash outputs, and retry/repair failed generations before CARLA/video.

---

## High-Level Data Flow
Natural-language description **or** JSON `ScenarioSpec` → Geometry spec → Crop assignment → Legal paths JSON/prompt → Picked (and refined) paths JSON → Scene objects JSON/PNG → Routes XML/manifest (optional alignment) → CARLA evaluation/reporting.

---

## Primary Entry Points
- `tools/run_audit_benchmark.py` — automation harness: samples/repairs schema specs, reruns failed stages, validates scenes, and can push through CARLA + video.  
- `scenario_generator/run_scenario_pipeline.py` — single/batch runner from text; uses one shared HF model across all stages.

### Choosing the right entry point
- **Rapid, single-scenario debugging (NL or JSON):** use `run_scenario_pipeline.py` with `--description` (or a schema file) plus `--crop` and `--town` for tight control over the region.  
- **Batch schema-first generation with automated retries/validation:** use `run_audit_benchmark.py`; it handles spec generation, repair loops, validation, and optional CARLA/video.  
- **Generate specs only (no CARLA), then inspect/run later:** use `scenario_generator/generate_schema_scenarios.py` to emit `scenario_spec.json` and `scenario_description.txt` per case, then pass those into `PipelineRunner` or `run_scenario_pipeline.py`.  
- **Crop/paths debugging without LLM variability:** invoke step wrappers directly (`run_crop_region_picker.py`, `generate_legal_paths.py`, `run_path_picker.py`) to isolate geometric issues.  
- **Route-only conversion/alignment:** if you already have `scene_objects.json`, call `convert_scene_to_routes.py` (and optionally step_07 alignment) without re-running earlier stages.

### Minimal Run Recipes
- Single description end-to-end:  
  `python scenario_generator/run_scenario_pipeline.py --description "Vehicle 1 ..." --town Town05 --crop XMIN XMAX YMIN YMAX --out-dir log`
- Benchmark loop (schema-first with retries/validation/CARLA):  
  `python tools/run_audit_benchmark.py --run-id demo --variants-per-category 1 --categories "Unprotected Left Turn" --model Qwen/Qwen2.5-32B-Instruct-AWQ`

---

## File-by-File Outline (what it is, where it fits)
- **Audit/automation**
  - `tools/run_audit_benchmark.py`: top-level entry for automated benchmarking; drives schema generation, retries/repairs, runs full pipeline via `PipelineRunner`, validates scenes, optionally runs CARLA/video, and logs to CSV.
- **Direct pipeline runners**
  - `scenario_generator/run_scenario_pipeline.py`: CLI orchestrator for crop → legal paths → path picker → optional refiner → object placer → optional routes, reusing one model.
  - `scenario_generator/run_crop_region_picker.py`: wrapper exposing step 01 crop selection helpers (LLM geometry extraction, CSP assignment).
  - `scenario_generator/generate_legal_paths.py`: wrapper for step 02 to enumerate legal paths and write prompt/JSON/viz.
  - `scenario_generator/run_path_picker.py`: wrapper for step 03 to pick ego paths via constraints/CSP/LLM.
  - `scenario_generator/run_path_refiner.py`: wrapper for step 04 to refine spawns, lane changes, speeds.
  - `scenario_generator/run_object_placer.py`: wrapper for step 05 two-stage placement with guardrails/viz.
  - `scenario_generator/convert_scene_to_routes.py`: thin wrapper around step 06 route export utilities/CLI.
  - `scenario_generator/generate_schema_scenarios.py`: CLI to auto-generate JSON specs per category and run the pipeline.
- **Core package (schema-first layer) — `scenario_generator/scenario_generator/`**
  - `README.md`: quick run instructions and stage list.
  - `__init__.py`: exports package API (capabilities, constraints, schema generator, pipeline runner, validator).
  - `capabilities.py`: ground-truth capability enums, map requirements, and category definitions (must/avoid/vary).
  - `constraints.py`: typed IR for `ScenarioSpec`, ego vehicles, inter-vehicle constraints, non-ego actors.
  - `schema_utils.py`: bridges `ScenarioSpec` ↔ `GeometrySpec`, plus description generation.
  - `schema_generator.py`: LLM/template spec generator with validation/repair loops and variation handling.
  - `schema_generation_loop.py`: orchestrates repeated schema generation → pipeline run → validation.
  - `scene_validator.py`: validates `scene_objects` vs description/spec (counts, maneuvers, constraints, intersections, actors).
  - `pipeline_runner.py`: in-process orchestrator for full pipeline with shared model/tokenizer and logging.
- **Step implementations — `scenario_generator/pipeline/`**
  - **step_01_crop**: `main.py` (CLI), `candidates.py` (crop generation), `csp.py` (assignment), `llm_extractor.py`, `models.py`, `features.py`, `scenario_io.py`, `scoring.py`, `viz.py`.
  - **step_02_legal_paths**: `main.py`, `segments.py` (lane graph & crop), `connectivity.py`, `signatures.py`, `prompt.py`, `geometry.py`, `viz.py`.
  - **step_03_path_picker**: `main.py`, `constraints.py` (prompting/guardrails), `csp.py`, `parsing.py`, `fuzzy.py`, `candidates.py`, `viz.py`.
  - **step_04_path_refiner**: `main.py`, `csp.py` (soft spawn/speed), `geometry.py`, `lane_change.py`, `llm.py`, `viz.py`.
  - **step_05_object_placer**: `main.py`, `assets.py`, `csp.py`, `filters.py`, `guardrails.py`, `model.py`, `nodes.py`, `parsing.py`, `path_extension.py`, `prompts.py`, `spawn.py`, `geometry.py`, `utils.py`, `viz.py`.
  - **step_06_scene_to_routes**: `main.py` (scene→routes XML/manifest/behavior, optional GRP align).
  - **step_07_route_alignment**: `main.py` (CARLA GRP snapping/compression/reheading).
- **Utilities, tests, and artifacts**
  - `scenario_generator/debug_path_picker_pipeline.py`, `visualize_roundabout.py`, `dump_carla_assets.py`, `dump_town_nodes.py`: ad-hoc debugging/inspection scripts.
  - `scenario_generator/test_legal_paths_pipeline.py`, `test_csp_crop_filter.py`: standalone test harnesses.
  - `scenario_generator/tests/test_t_junction_spawn.py` + `tests/artifacts/*`: regression for T-junction spawn/lane-change macro handling.
  - `scenario_generator/debug_output/*.png`, `legal_paths_test/*`: visual baselines for regressions.
  - Data assets: `scenario_generator/town_nodes/Town*.json` (map graphs), `scenario_generator/carla_assets.json` (asset catalog), `scenario_generator/crop_map*.json` (cached crop assignments), `scenario_generator/scenarios.json` / `scenarios.txt` (seed descriptions).

---

## Big-Picture Notes
- One shared HF model instance is reused across stages for efficiency; `PipelineRunner` and `run_scenario_pipeline.py` manage this.  
- Roundabout topology uses a hardcoded Town03 crop and skips the refiner; two-lane corridor has corridor-specific crop generation and path splitting.  
- On-ramp/merge requirements propagate from category/spec into picker/refiner/object placement to enforce merge conflicts.  
- Scene validation (`scene_validator.py`) is used inside the audit runner to gate retries and repairs before CARLA execution.

---

## What Each Step Produces (key artifacts)
- Step 01: `crop_map.json`, `crop_map_detailed.json`, optional crop PNGs.  
- Step 02: `legal_paths_prompt.txt`, `legal_paths_detailed.json`, optional `legal_paths*.png`.  
- Step 03: `picked_paths_detailed.json`, optional `picked_paths_viz.png`.  
- Step 04: `picked_paths_refined.json`, optional `picked_paths_refined_viz.png`.  
- Step 05: `scene_objects.json`, `scene_objects.png`.  
- Step 06/07: route XMLs, `actors_manifest.json`, optional `actors_behavior.json`, aligned variants when enabled.  
- Audit runner extras: CSV log, hashes, CARLA/video outputs per run.

---

## Shared Assets (keep in sync)
- Town graphs: `scenario_generator/town_nodes/Town*.json`  
- Asset catalog: `scenario_generator/carla_assets.json`  
- Optional cached crops: `scenario_generator/crop_map*.json`

---

## Quick Start Checklist for New Contributors
- Confirm CARLA town node files are present and match your target towns.  
- Load or start a single shared HF model (default `Qwen/Qwen2.5-32B-Instruct-AWQ`) before batch runs.  
- For schema-first experiments, start with `generate_schema_scenarios.py` or `run_audit_benchmark.py`; for free-text, use `run_scenario_pipeline.py`.  
- Keep `carla_assets.json` aligned with the CARLA build you’re using so bboxes and models remain valid.  
- Use the saved artifacts at each stage to debug failures (prompts/JSON/PNGs are written per scenario).
