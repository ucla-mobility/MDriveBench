"""
Generation Loop

Robust end-to-end scenario generation with deep validation.
Generates N valid scenarios with retry logic, validation, and diversity tracking.
"""

import json
import os
import re
import sys
import time
import io
import contextlib
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from datetime import datetime

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .capabilities import (
    CATEGORY_FEASIBILITY,
    get_feasible_categories,
)
from .generator import (
    ScenarioGenerator,
    GenerationConfig,
    select_variation_values,
    build_generation_prompt,
    build_generation_prompt_simple,
    build_system_prompt,
    build_repair_prompt,
    build_repair_prompt_with_ir,
)
from .validator import ScenarioValidator
from .scene_validator import SceneValidator, SceneValidationResult
from .scenario_ir import (
    ScenarioIR,
    extract_ir_with_llm,
    extract_ir_with_regex,
    merge_ir_extractions,
)
from .geometric_checker import (
    validate_geometric_consistency,
    build_geometric_feedback,
)


# =============================================================================
# LOGGING WITH CLEAN OUTPUT
# =============================================================================

class GenerationLogger:
    """Clean logging for generation loop, muting internal pipeline output."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.log_buffer: List[str] = []
        self.start_time = time.time()
    
    def _timestamp(self) -> str:
        elapsed = time.time() - self.start_time
        return f"[{elapsed:6.1f}s]"
    
    def info(self, msg: str):
        line = f"{self._timestamp()} [INFO] {msg}"
        print(line)
        self.log_buffer.append(line)
    
    def success(self, msg: str):
        line = f"{self._timestamp()} [OK] {msg}"
        print(line)
        self.log_buffer.append(line)
    
    def warning(self, msg: str):
        line = f"{self._timestamp()} [WARN] {msg}"
        print(line)
        self.log_buffer.append(line)
    
    def error(self, msg: str):
        line = f"{self._timestamp()} [ERROR] {msg}"
        print(line)
        self.log_buffer.append(line)
    
    def progress(self, current: int, total: int, msg: str):
        line = f"{self._timestamp()} [{current}/{total}] {msg}"
        print(line)
        self.log_buffer.append(line)
    
    def debug(self, msg: str):
        if self.verbose:
            line = f"{self._timestamp()} [DEBUG] {msg}"
            print(line)
            self.log_buffer.append(line)
    
    def divider(self):
        print("─" * 60)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            f.write("\n".join(self.log_buffer))


@contextlib.contextmanager
def mute_stdout():
    """Context manager to mute stdout (for suppressing pipeline output)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# =============================================================================
# PIPELINE RUNNER (wraps run_scenario_pipeline)
# =============================================================================

class PipelineRunner:
    """
    Wrapper around the scenario pipeline with muted output and structured results.
    
    Uses the actual run_scenario_pipeline.py architecture to run scenarios.
    Supports running from specific stages with custom hyperparameters for repair.
    """
    
    def __init__(
        self,
        output_base_dir: str = "log",
        model=None,
        tokenizer=None,
        viz_objects: bool = True,
        routes_out_dir: str = "routes",
        routes_ego_num: Optional[int] = None,
        refiner_max_new_tokens: int = 2048,
        placer_max_new_tokens: int = 2048,
        placer_do_sample: bool = False,
        placer_temperature: float = 0.2,
        placer_top_p: float = 0.95,
    ):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.viz_objects = viz_objects
        self.routes_out_dir = routes_out_dir
        self.routes_ego_num = routes_ego_num
        self.refiner_max_new_tokens = refiner_max_new_tokens
        self.placer_max_new_tokens = placer_max_new_tokens
        self.placer_do_sample = placer_do_sample
        self.placer_temperature = placer_temperature
        self.placer_top_p = placer_top_p
        
        # Shared model/tokenizer (if provided, reuse; otherwise load on demand)
        self._model = model
        self._tokenizer = tokenizer
        self._model_loaded = model is not None
        
        # Cache for intermediate results (for repair strategies)
        self._cached_legal_paths: Dict[str, Any] = {}
        self._cached_candidates: Dict[str, List[Dict]] = {}
    
    def _ensure_model(self, model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"):
        """Load the shared model if not already loaded."""
        if self._model_loaded:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        self._model_loaded = True
    
    def run_full_pipeline(
        self,
        scenario_text: str,
        category: str,
        difficulty: int,
        scenario_id: str,
        town: str = "Town05",
        mute: bool = True,
        model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run the full pipeline for a scenario using the shared model.
        
        Returns:
            (success, scene_objects_path, error_message)
        """
        # Ensure model is loaded
        self._ensure_model(model_id)
        
        # Create output directory
        safe_name = scenario_id.replace(" ", "_").replace("/", "_")
        out_dir = self.output_base_dir / safe_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the scenario text for reference
        with open(out_dir / "scenario_input.txt", 'w') as f:
            f.write(f"Category: {category}\n")
            f.write(f"Difficulty: {difficulty}\n")
            f.write(f"Town: {town}\n")
            f.write(f"Text: {scenario_text}\n")
        
        try:
            if mute:
                with mute_stdout():
                    return self._run_pipeline_direct(out_dir, scenario_text, town, scenario_id, category=category)
            else:
                return self._run_pipeline_direct(out_dir, scenario_text, town, scenario_id, category=category)
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg
    
    def _run_pipeline_direct(
        self,
        out_dir: Path,
        scenario_text: str,
        town: str,
        scenario_id: str,
        category: str = "",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run the scenario pipeline directly using the shared model (no subprocess).
        """
        # Check if category requires CORRIDOR topology (straight paths only)
        from .capabilities import CATEGORY_FEASIBILITY, TopologyType
        require_straight = False
        cat_info = CATEGORY_FEASIBILITY.get(category)
        if cat_info and cat_info.required_topology == TopologyType.CORRIDOR:
            require_straight = True
            print(f"[INFO] Category '{category}' requires CORRIDOR topology, filtering to straight paths")
        
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from convert_scene_to_routes import convert_scene_to_routes
        from generate_legal_paths import (
            CropBox,
            build_connectivity,
            build_path_signature,
            build_segments,
            crop_segments,
            generate_legal_paths,
            load_nodes,
            make_path_name,
            save_aggregated_signatures_json,
            save_prompt_file,
            build_segments_detailed_for_path,
        )
        from run_path_picker import pick_paths_with_model
        from run_path_refiner import refine_picked_paths_with_model
        from run_object_placer import run_object_placer
        import run_crop_region_picker as crop_picker
        from pipeline.step_01_crop.scoring import crop_satisfies_spec
        
        # Setup paths
        nodes_path = parent_dir / "town_nodes" / f"{town}.json"
        if not nodes_path.exists():
            return False, None, f"Nodes file not found: {nodes_path}"
        
        legal_prompt_path = out_dir / "legal_paths_prompt.txt"
        legal_json_path = out_dir / "legal_paths_detailed.json"
        picker_prompt_path = out_dir / "path_picker_prompt.txt"
        picked_paths_path = out_dir / "picked_paths_detailed.json"
        refined_paths_path = out_dir / "picked_paths_refined.json"
        scene_json_path = out_dir / "scene_objects.json"
        scene_png_path = out_dir / "scene_objects.png"
        
        # Step 1: Get a crop region for this scenario
        # Use the LLM extractor to determine geometric requirements
        extractor = crop_picker.LLMGeometryExtractor(model_name="", device="cuda")
        extractor._tok = self._tokenizer
        extractor._mdl = self._model
        
        spec = extractor.extract(scenario_text)
        
        # Build candidate crops for the town
        radii = [45.0, 55.0, 65.0]
        crops = crop_picker.build_candidate_crops_for_town(
            town_name=town,
            town_json_path=str(nodes_path),
            radii=radii,
            min_path_len=22.0,
            max_paths=80,
            max_depth=8,
        )
        
        if not crops:
            return False, None, f"No candidate crops found for {town}"

        # Hard gate: if required features don't exist in any crop, surface a crop logic issue.
        count_on_ramp = sum(1 for c in crops if getattr(c, "has_on_ramp", False))
        count_merge = sum(1 for c in crops if getattr(c, "has_merge_onto_same_road", False))
        count_multi_lane = sum(1 for c in crops if getattr(c, "has_multi_lane", False))
        count_oncoming = sum(1 for c in crops if getattr(c, "has_oncoming_pair", False))

        if spec.needs_on_ramp and count_on_ramp == 0:
            return False, None, (
                f"No crops with on-ramp geometry found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_merge_onto_same_road and count_merge == 0:
            return False, None, (
                f"No crops with merge geometry found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_multi_lane and count_multi_lane == 0:
            return False, None, (
                f"No multi-lane crops found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        if spec.needs_oncoming and count_oncoming == 0:
            return False, None, (
                f"No oncoming-pair crops found for {town}; "
                f"total={len(crops)}, on_ramp={count_on_ramp}, merge={count_merge}, "
                f"multi_lane={count_multi_lane}, oncoming={count_oncoming}"
            )
        
        # Create a scenario object for assignment
        scenario = crop_picker.Scenario(sid=scenario_id, text=scenario_text)
        
        # Solve assignment for this single scenario
        res = crop_picker.solve_assignment(
            scenarios=[scenario],
            specs={scenario_id: spec},
            crops=crops,
            domain_k=50,
            capacity_per_crop=10,
            reuse_weight=4000.0,
            junction_penalty=25000.0,
            log_every=0,
        )
        
        assignments = res.detailed.get("assignments", {})
        if scenario_id not in assignments:
            # Retry with a relaxed spec to allow any crop
            # Only relax if there exists at least one crop that satisfies the original spec.
            if not any(crop_satisfies_spec(spec, c) for c in crops):
                return False, None, (
                    f"No crops satisfy geometry spec; "
                    f"needs_on_ramp={spec.needs_on_ramp}, "
                    f"needs_merge_onto_same_road={spec.needs_merge_onto_same_road}, "
                    f"needs_multi_lane={spec.needs_multi_lane}, "
                    f"needs_oncoming={spec.needs_oncoming}"
                )
            relaxed_spec = crop_picker.GeometrySpec(
                topology="unknown",
                degree=0,
                required_maneuvers={"straight": 0, "left": 0, "right": 0},
                needs_oncoming=False,
                needs_merge_onto_same_road=False,
                needs_on_ramp=False,
                needs_multi_lane=False,
                min_lane_count=1,
                min_entry_runup_m=0.0,
                min_exit_runout_m=0.0,
                preferred_entry_cardinals=[],
                avoid_extra_intersections=False,
                confidence=0.0,
                notes="relaxed_fallback",
            )
            relaxed = crop_picker.solve_assignment(
                scenarios=[scenario],
                specs={scenario_id: relaxed_spec},
                crops=crops,
                domain_k=50,
                capacity_per_crop=10,
                reuse_weight=4000.0,
                junction_penalty=25000.0,
                log_every=0,
            )
            assignments = relaxed.detailed.get("assignments", {})
        
        if scenario_id not in assignments:
            # Last-resort fallback: pick the smallest-area crop
            cand = min(crops, key=lambda c: c.area)
            crop_vals = [cand.crop.xmin, cand.crop.xmax, cand.crop.ymin, cand.crop.ymax]
        else:
            crop_vals = assignments[scenario_id]["crop"]
        crop = CropBox(xmin=crop_vals[0], xmax=crop_vals[1], ymin=crop_vals[2], ymax=crop_vals[3])
        
        # Step 2: Generate legal paths
        data = load_nodes(str(nodes_path))
        all_segments = build_segments(data)
        cropped_segments = crop_segments(all_segments, crop)
        
        if not cropped_segments:
            return False, None, "No segments found in crop region"
        
        adj = build_connectivity(
            cropped_segments,
            connect_radius_m=6.0,
            connect_yaw_tol_deg=60.0,
        )
        
        legal_paths = generate_legal_paths(
            cropped_segments,
            adj,
            crop,
            min_path_length=20.0,
            max_paths=100,
            max_depth=5,
            allow_within_region_fallback=False,
        )
        
        if not legal_paths:
            return False, None, "No legal paths found for this crop"
        
        # Build candidates for the path picker
        params = {
            "max_yaw_diff_deg": 60.0,
            "connect_radius_m": 6.0,
            "min_path_length_m": 20.0,
            "max_paths": 100,
            "max_depth": 5,
            "turn_frame": "WORLD_FRAME",
        }
        
        candidates = []
        for i, p in enumerate(legal_paths):
            sig = build_path_signature(p)
            name = make_path_name(i, sig)
            sig["segments_detailed"] = build_segments_detailed_for_path(p, polyline_sample_n=10)
            candidates.append({"name": name, "signature": sig})
        
        save_prompt_file(
            str(legal_prompt_path),
            crop=crop,
            nodes_path=str(nodes_path),
            params=params,
            paths_named=candidates,
        )
        save_aggregated_signatures_json(
            str(legal_json_path),
            crop=crop,
            nodes_path=str(nodes_path),
            params=params,
            paths_named=candidates,
        )
        
        # Step 3: Run path picker
        prompt_text = legal_prompt_path.read_text(encoding="utf-8").strip()
        prompt_text += (
            "\n\nUSER SCENARIO DESCRIPTION:\n"
            + scenario_text
            + "\n(Only assign paths to moving vehicles; ignore static/parked props.)\n"
        )
        picker_prompt_path.write_text(prompt_text, encoding="utf-8")
        
        pick_paths_with_model(
            prompt=prompt_text,
            aggregated_json=str(legal_json_path),
            out_picked_json=str(picked_paths_path),
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.2,
            top_p=0.95,
            require_straight=require_straight,
        )
        
        # Step 4: Run path refiner
        refine_picked_paths_with_model(
            picked_paths_json=str(picked_paths_path),
            description=scenario_text,
            out_json=str(refined_paths_path),
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=self.refiner_max_new_tokens,
            carla_assets=str(parent_dir / "carla_assets.json"),
        )
        
        # Step 5: Run object placer
        from types import SimpleNamespace
        placer_args = SimpleNamespace(
            model="",
            picked_paths=str(refined_paths_path),
            carla_assets=str(parent_dir / "carla_assets.json"),
            description=scenario_text,
            out=str(scene_json_path),
            viz_out=str(scene_png_path),
            viz=self.viz_objects,
            viz_show=False,
            nodes_root=str(nodes_path.parent),
            max_new_tokens=self.placer_max_new_tokens,
            do_sample=self.placer_do_sample,
            temperature=self.placer_temperature,
            top_p=self.placer_top_p,
            placement_mode="llm_anchor",
        )
        
        run_object_placer(placer_args, model=self._model, tokenizer=self._tokenizer)
        
        if scene_json_path.exists():
            # Don't write routes here - only write them for final successful scenarios
            # This avoids creating multiple route directories for failed attempts
            return True, str(scene_json_path), None
        else:
            return False, None, "scene_objects.json not created"

    def run_repair(
        self,
        scenario_text: str,
        scenario_id: str,
        repair_strategy: 'RepairStrategy',
        original_out_dir: Path,
        town: str = "Town05",
        mute: bool = True,
        validation_feedback: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Run a repair attempt from a specific pipeline stage.
        
        Uses cached intermediate results where possible to avoid
        re-running expensive earlier stages.
        
        Args:
            scenario_text: The scenario description
            scenario_id: Unique ID for this scenario
            repair_strategy: The repair strategy to use
            original_out_dir: Directory with original pipeline outputs
            town: CARLA town
            mute: Whether to mute stdout
            validation_feedback: Feedback from validation to include in prompts
            
        Returns:
            (success, scene_path, error_message)
        """
        stage = repair_strategy.stage_to_retry
        
        # Create repair output directory
        repair_dir = original_out_dir.parent / f"{original_out_dir.name}_repair_{repair_strategy.name}"
        repair_dir.mkdir(parents=True, exist_ok=True)
        
        # Build optional repair context (applied only to the retried stage)
        repair_lines: List[str] = []
        if validation_feedback:
            repair_lines.append("VALIDATION ISSUES:")
            repair_lines.append(validation_feedback)
        if repair_strategy.extra_context:
            repair_lines.append(repair_strategy.extra_context)
        repair_context = "\n".join(repair_lines)
        
        try:
            if mute:
                with mute_stdout():
                    return self._run_repair_internal(
                        scenario_text,
                        scenario_id,
                        repair_strategy,
                        original_out_dir,
                        repair_dir,
                        town,
                        repair_context,
                    )
            else:
                return self._run_repair_internal(
                    scenario_text,
                    scenario_id,
                    repair_strategy,
                    original_out_dir,
                    repair_dir,
                    town,
                    repair_context,
                )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg

    def _run_repair_internal(
        self,
        scenario_text: str,
        scenario_id: str,
        strategy: 'RepairStrategy',
        original_dir: Path,
        repair_dir: Path,
        town: str,
        repair_context: str = "",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Internal repair logic."""
        import shutil

        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        from convert_scene_to_routes import convert_scene_to_routes
        from run_path_picker import pick_paths_with_model
        from run_path_refiner import refine_picked_paths_with_model
        from run_object_placer import run_object_placer
        
        self._ensure_model()
        
        stage = strategy.stage_to_retry
        
        # Determine what we can reuse from original run
        legal_json_path = original_dir / "legal_paths_detailed.json"
        legal_prompt_path = original_dir / "legal_paths_prompt.txt"
        picked_paths_path = repair_dir / "picked_paths_detailed.json"
        refined_paths_path = repair_dir / "picked_paths_refined.json"
        scene_json_path = repair_dir / "scene_objects.json"
        scene_png_path = repair_dir / "scene_objects.png"
        
        # Check we have the prerequisite files
        if stage in ("step_03_path_picker", "step_04_path_refiner", "step_05_object_placer"):
            if not legal_json_path.exists() or not legal_prompt_path.exists():
                return False, None, "Missing legal_paths files from original run"
        
        # Apply repair context only to the retried stage
        picker_description = scenario_text
        refiner_description = scenario_text
        placer_description = scenario_text
        if repair_context:
            if stage == "step_03_path_picker":
                picker_description = f"{scenario_text}\n\n{repair_context}"
            elif stage == "step_04_path_refiner":
                refiner_description = f"{scenario_text}\n\n{repair_context}"
            elif stage == "step_05_object_placer":
                placer_description = f"{scenario_text}\n\n{repair_context}"

        # Run from the specified stage
        if stage == "step_03_path_picker":
            # Re-run path picker with new hyperparameters
            prompt_text = legal_prompt_path.read_text()
            prompt_text += (
                "\n\nUSER SCENARIO DESCRIPTION:\n"
                f"{picker_description}\n"
                "(Only assign paths to moving vehicles; ignore static/parked props.)\n"
            )
            
            pick_paths_with_model(
                prompt=prompt_text,
                aggregated_json=str(legal_json_path),
                out_picked_json=str(picked_paths_path),
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=strategy.max_new_tokens,
                do_sample=True,
                temperature=strategy.temperature,
                top_p=strategy.top_p,
            )
            
            # Continue with path refiner and object placer
            refine_picked_paths_with_model(
                picked_paths_json=str(picked_paths_path),
                description=refiner_description,
                out_json=str(refined_paths_path),
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self.refiner_max_new_tokens,
                carla_assets=str(parent_dir / "carla_assets.json"),
            )
            
        elif stage == "step_04_path_refiner":
            # Copy picked paths from original, re-run refiner
            orig_picked = original_dir / "picked_paths_detailed.json"
            if orig_picked.exists():
                shutil.copy(orig_picked, picked_paths_path)
            else:
                return False, None, "Missing picked_paths from original run"
            
            # Note: refiner doesn't support temperature/top_p directly
            refine_picked_paths_with_model(
                picked_paths_json=str(picked_paths_path),
                description=refiner_description,
                out_json=str(refined_paths_path),
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self.refiner_max_new_tokens,
                carla_assets=str(parent_dir / "carla_assets.json"),
            )
            
        elif stage == "step_05_object_placer":
            # Copy refined paths from original, re-run object placer
            orig_refined = original_dir / "picked_paths_refined.json"
            if orig_refined.exists():
                import shutil
                shutil.copy(orig_refined, refined_paths_path)
            else:
                # Try picked paths
                orig_picked = original_dir / "picked_paths_detailed.json"
                if orig_picked.exists():
                    shutil.copy(orig_picked, refined_paths_path)
                else:
                    return False, None, "Missing paths from original run"
        
        # Run object placer for all repair paths
        nodes_path = parent_dir / "town_nodes" / f"{town}.json"
        if not nodes_path.exists():
            nodes_path = parent_dir / "data" / "nodes" / f"{town}.json"
        
        from types import SimpleNamespace
        placer_args = SimpleNamespace(
            model="",
            picked_paths=str(refined_paths_path),
            carla_assets=str(parent_dir / "carla_assets.json"),
            description=placer_description,
            out=str(scene_json_path),
            viz_out=str(scene_png_path),
            viz=self.viz_objects,
            viz_show=False,
            nodes_root=str(nodes_path.parent),
            max_new_tokens=self.placer_max_new_tokens,
            do_sample=self.placer_do_sample,
            temperature=self.placer_temperature,
            top_p=self.placer_top_p,
            placement_mode="llm_anchor",
        )
        
        run_object_placer(placer_args, model=self._model, tokenizer=self._tokenizer)
        
        if scene_json_path.exists():
            # Don't write routes here - only write them for final successful scenarios
            # This avoids creating multiple route directories for failed repair attempts
            return True, str(scene_json_path), None
        else:
            return False, None, "scene_objects.json not created after repair"


# =============================================================================
# GENERATION LOOP
# =============================================================================

@dataclass
class GenerationResult:
    """Result of a single generation attempt."""
    scenario_id: str
    category: str
    difficulty: int
    text: str
    success: bool
    scene_path: Optional[str] = None
    validation_result: Optional[SceneValidationResult] = None
    attempts: int = 0
    error_message: Optional[str] = None


@dataclass
class GenerationLoopConfig:
    """Configuration for the generation loop."""
    # Target count
    target_count: int = 10
    
    # Retry settings
    max_retries_per_scenario: int = 3
    max_text_repair_passes: int = 5
    
    # Category filtering (optional bias)
    categories: Optional[List[str]] = None
    category_weights: Optional[Dict[str, float]] = None
    
    # Difficulty distribution
    difficulties: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    
    # Output settings
    output_dir: str = "log"
    
    # Validation thresholds
    min_validation_score: float = 0.6
    
    # Diversity settings
    similarity_threshold: float = 0.7
    
    # Pipeline settings
    town: str = "Town05"
    mute_pipeline: bool = True
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    viz_objects: bool = True
    routes_out_dir: str = "routes"
    routes_ego_num: Optional[int] = None
    
    # Route alignment settings
    align_routes: bool = True
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    
    # Verbose logging
    verbose: bool = False
    
    # Debug flags
    debug_ir: bool = False
    ir_repair_passes: int = 1
    
    # Stage repair settings
    stage_repairs: bool = True  # If False, skip pipeline stage repairs and regenerate text instead


# =============================================================================
# REPAIR STRATEGIES
# =============================================================================

@dataclass
class RepairStrategy:
    """Configuration for a repair attempt."""
    name: str
    stage_to_retry: str  # Which pipeline stage to retry from
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512
    extra_context: Optional[str] = None  # Additional prompt context


# Repair strategies for different failure types
REPAIR_STRATEGIES = {
    "step_03_path_picker": [
        RepairStrategy(
            name="path_picker_warmer",
            stage_to_retry="step_03_path_picker",
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=768,
            extra_context="IMPORTANT: Carefully match the vehicle relationships in the description. "
                         "Ensure vehicles approach from the correct directions."
        ),
        RepairStrategy(
            name="path_picker_focused",
            stage_to_retry="step_03_path_picker",
            temperature=0.6,
            top_p=0.85,
            max_new_tokens=512,
            extra_context="Focus on the spatial constraints. Pick paths that create intersection conflicts."
        ),
        RepairStrategy(
            name="path_picker_deterministic",
            stage_to_retry="step_03_path_picker",
            temperature=0.3,
            top_p=0.9,
            max_new_tokens=512,
            extra_context="Be precise. Match exactly the number of vehicles and their maneuvers."
        ),
    ],
    "step_05_object_placer": [
        RepairStrategy(
            name="object_placer_warmer",
            stage_to_retry="step_05_object_placer",
            temperature=0.8,
            top_p=0.92,
            max_new_tokens=1024,
            extra_context="Ensure all actors (pedestrians, parked vehicles, etc.) mentioned in the "
                         "description are placed in the scene."
        ),
        RepairStrategy(
            name="object_placer_focused",
            stage_to_retry="step_05_object_placer",
            temperature=0.5,
            top_p=0.85,
            max_new_tokens=1024,
            extra_context="Focus on actor placement. Match the description precisely."
        ),
        RepairStrategy(
            name="object_placer_minimal",
            stage_to_retry="step_05_object_placer",
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=768,
            extra_context="Place only the essential actors mentioned in the description."
        ),
    ],
    "step_04_path_refiner": [
        RepairStrategy(
            name="path_refiner_warmer",
            stage_to_retry="step_04_path_refiner",
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
        ),
        RepairStrategy(
            name="path_refiner_focused",
            stage_to_retry="step_04_path_refiner",
            temperature=0.5,
            top_p=0.85,
            max_new_tokens=512,
        ),
    ],
    "step_01_crop": [
        # For crop failures, we regenerate text with different phrasing
        RepairStrategy(
            name="regenerate_text",
            stage_to_retry="regenerate",
            temperature=0.85,
            extra_context="Generate a simpler version of this scenario that's easier to place."
        ),
    ],
}


class GenerationLoop:
    """
    Main generation loop with validation and retry logic.
    
    Generates N valid scenarios by:
    1. Generating natural language scenario descriptions
    2. Running them through the pipeline
    3. Validating the generated scene matches the description
    4. Retrying on failure (up to max_retries)
    5. Tracking generated scenarios to avoid duplicates
    """
    
    def __init__(self, config: GenerationLoopConfig, model=None, tokenizer=None):
        self.config = config
        self.logger = GenerationLogger(verbose=config.verbose)
        
        # Load model if not provided
        if model is None or tokenizer is None:
            self.logger.info(f"Loading model: {config.model_id}")
            model, tokenizer = self._load_model(config.model_id)
        
        self._model = model
        self._tokenizer = tokenizer
        
        # Initialize components
        self.text_generator = ScenarioGenerator(
            GenerationConfig(
                temperature=0.8,  # Higher for more diversity
                similarity_threshold=config.similarity_threshold,
                model_id=config.model_id,
            ),
            model=model,
            tokenizer=tokenizer,
        )
        self.text_validator = ScenarioValidator()
        self.scene_validator = SceneValidator()
        self.pipeline_runner = PipelineRunner(
            config.output_dir,
            model=model,
            tokenizer=tokenizer,
            viz_objects=config.viz_objects,
            routes_out_dir=config.routes_out_dir,
            routes_ego_num=config.routes_ego_num,
        )
        self.routes_out_dir = config.routes_out_dir
        self.routes_ego_num = config.routes_ego_num
        
        # Filter to feasible categories
        self.categories = config.categories or get_feasible_categories()
        self.categories = [c for c in self.categories if c in get_feasible_categories()]
        
        # Tracking
        self.generated_scenarios: List[GenerationResult] = []
        self.failed_scenarios: List[GenerationResult] = []
        self.seen_texts: Set[str] = set()
        
        # Statistics
        self.total_attempts = 0
        self.total_pipeline_runs = 0
        self.total_validation_failures = 0
    
    def _load_model(self, model_id: str):
        """Load the HuggingFace model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model.eval()
        return model, tokenizer
    
    def _select_category_and_difficulty(self) -> Tuple[str, int]:
        """Select next category and difficulty, with optional weighting."""
        import random
        
        if self.config.category_weights:
            # Weighted selection
            categories = list(self.config.category_weights.keys())
            weights = list(self.config.category_weights.values())
            category = random.choices(categories, weights=weights, k=1)[0]
        else:
            # Uniform selection
            category = random.choice(self.categories)
        
        difficulty = random.choice(self.config.difficulties)
        return category, difficulty
    
    def _is_duplicate(self, text: str) -> bool:
        """Check if this scenario text is too similar to existing ones."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        for seen in self.seen_texts:
            seen_words = set(seen.lower().split())
            if not words or not seen_words:
                continue
            overlap = len(words & seen_words)
            similarity = overlap / max(len(words), len(seen_words))
            if similarity > self.config.similarity_threshold:
                return True
        
        return False
    
    def _extract_and_validate_ir(
        self, 
        scenario_text: str,
    ) -> Tuple[ScenarioIR, List[str], str, str]:
        """
        Extract IR from scenario text and run geometric validation.
        
        Returns:
            (ir, geometric_errors, ir_summary, geometric_feedback)
        """
        # Try LLM extraction first, fall back to regex
        def llm_generate(prompt: str) -> str:
            if self.config.debug_ir:
                self.logger.info(f"  [LLM CALL] Sending IR extraction prompt ({len(prompt)} chars)")
            response = self.text_generator._generate_text(prompt)
            if self.config.debug_ir:
                self.logger.info(f"  [LLM RESPONSE] Received {len(response)} chars")
                if response:
                    self.logger.info(f"  [LLM RESPONSE] First 300 chars: {response[:300]}")
                else:
                    self.logger.info(f"  [LLM RESPONSE] Response is EMPTY!")
            return response
        
        try:
            llm_ir = extract_ir_with_llm(
                scenario_text,
                llm_generate,
                max_repair_attempts=self.config.ir_repair_passes,
                debug=self.config.debug_ir,
            )
        except Exception as e:
            self.logger.info(f"  LLM IR extraction failed: {type(e).__name__}: {e}")
            llm_ir = ScenarioIR(extraction_warnings=[f"LLM extraction failed: {e}"])
        
        # Debug: Print IR extraction details if flag is set
        if self.config.debug_ir:
            self.logger.info(f"\n  === IR EXTRACTION DEBUG ===")
            self.logger.info(f"  Raw LLM response (first 500 chars):")
            preview = llm_ir.raw_llm_response[:500] if llm_ir.raw_llm_response else "(empty)"
            self.logger.info(f"    {preview}")
            self.logger.info(f"  Extracted vehicles: {len(llm_ir.vehicles)}")
            for v in llm_ir.vehicles:
                self.logger.info(f"    - Vehicle {v.vehicle_id}: approach={v.approach_direction.value}, maneuver={v.maneuver}")
            self.logger.info(f"  Extracted constraints: {len(llm_ir.constraints)}")
            self.logger.info(f"  Extraction confidence: {llm_ir.extraction_confidence:.2f}")
            self.logger.info(f"  === END DEBUG ===\n")
        
        # Check if extraction had warnings
        if llm_ir.extraction_warnings:
            self.logger.info(f"  Extraction warnings: {llm_ir.extraction_warnings}")
        
        print("[DEBUG] Starting regex IR extraction...", flush=True)
        regex_ir = extract_ir_with_regex(scenario_text)
        print("[DEBUG] Regex IR done, merging...", flush=True)
        
        # Merge extractions (LLM preferred, fill gaps with regex)
        ir = merge_ir_extractions(llm_ir, regex_ir)
        print("[DEBUG] Merge done, validating geometry...", flush=True)
        
        # Run geometric validation
        geo_result = validate_geometric_consistency(ir)
        print("[DEBUG] Geometry validation done", flush=True)
        
        # Collect geometric errors as strings
        geo_errors = [e.message for e in geo_result.errors]
        
        # Build feedback strings
        ir_summary = ir.to_summary_string()
        geo_feedback = build_geometric_feedback(ir, geo_result)
        
        return ir, geo_errors, ir_summary, geo_feedback
    
    def _generate_text(self, category: str, difficulty: int) -> Tuple[Optional[str], str]:
        """
        Generate scenario text using LLM with IR-based validation.
        
        This method:
        1. Generates free-form text from the LLM
        2. Extracts IR (structured representation) from the text
        3. Runs deterministic geometric validation on the IR
        4. Combines semantic + geometric errors for comprehensive feedback
        5. Uses enhanced repair prompts with IR visibility
        
        Returns:
            (scenario_text, status_message)
        """
        cat_info = CATEGORY_FEASIBILITY[category]
        
        # Get existing scenarios in this category for context
        existing = [
            r.text for r in self.generated_scenarios
            if r.category == category and r.success
        ]
        
        # Also include failed attempts to avoid those patterns
        existing.extend([
            r.text for r in self.failed_scenarios
            if r.category == category
        ][-3:])  # Last 3 failures
        
        # Select variation values
        used_combos = self.text_generator.used_combinations.get(category, set())
        forced_variations = select_variation_values(cat_info, used_combos)
        
        # Build prompt (chain-of-thought optional)
        prompt_builder = (
            build_generation_prompt
            if self.text_generator.config.use_chain_of_thought
            else build_generation_prompt_simple
        )
        prompt = prompt_builder(
            category=category,
            difficulty=difficulty,
            cat_info=cat_info,
            existing_scenarios=existing,
            forced_variations=forced_variations,
        )
        
        try:
            # Generate text
            response = self.text_generator._generate_text(prompt)
            scenario_text = self.text_generator._clean_response(response)
            
            # Always print the full generated text
            print(f"\n{'='*60}")
            print(f"Generated Text:")
            print(f"{'='*60}")
            print(scenario_text)
            print(f"{'='*60}\n")
            
            # === STAGE 1: Semantic Validation ===
            validation = self.text_validator.validate_scenario(
                text=scenario_text,
                category=category,
                difficulty=difficulty,
                scenario_id=f"{category}_{difficulty}",
            )
            semantic_errors = (
                validation.structural_errors + 
                validation.semantic_errors + 
                validation.parse_errors
            )
            print(f"[DEBUG] Semantic validation done, {len(semantic_errors)} errors", flush=True)
            
            # === STAGE 2: IR Extraction & Geometric Validation ===
            ir, geo_errors, ir_summary, geo_feedback = self._extract_and_validate_ir(scenario_text)
            print(f"[DEBUG] IR extraction done, {len(geo_errors)} geo errors", flush=True)
            
            # Print IR extraction results
            if self.config.verbose:
                print(f"\n{'='*60}")
                print(f"IR Extraction:")
                print(f"{'='*60}")
                print(ir_summary)
                print(f"{'='*60}\n")
            
            # Combine all errors
            all_errors = semantic_errors + geo_errors
            print(f"[DEBUG] Total errors: {len(all_errors)}, validation.is_valid={validation.is_valid}", flush=True)
            
            # Check if completely valid
            if validation.is_valid and not geo_errors:
                print(f"[DEBUG] Scenario valid! Checking for duplicate...", flush=True)
                if self._is_duplicate(scenario_text):
                    return None, "Too similar to existing scenario"
                self.logger.debug(f"  First attempt valid (semantic + geometric)!")
                print(f"[DEBUG] Returning OK", flush=True)
                return scenario_text, "OK"
            
            print(f"[DEBUG] Not valid, entering repair loop...", flush=True)

            if all_errors:
                self.logger.info(f"  Validation errors: {all_errors}")
            if geo_errors:
                self.logger.debug(f"  Geometric errors: {geo_errors}")

            last_text = scenario_text
            last_semantic_errors = semantic_errors
            last_geo_errors = geo_errors
            last_ir_summary = ir_summary
            last_geo_feedback = geo_feedback

            # === REPAIR LOOP with IR-based feedback ===
            for pass_idx in range(self.config.max_text_repair_passes):
                self.logger.debug(
                    f"  Text repair pass {pass_idx + 1}/{self.config.max_text_repair_passes}"
                )
                
                # Use enhanced repair prompt with IR feedback
                all_last_errors = last_semantic_errors + last_geo_errors
                fix_prompt = build_repair_prompt_with_ir(
                    scenario=last_text,
                    errors=all_last_errors,
                    category=category,
                    difficulty=difficulty,
                    cat_info=cat_info,
                    ir_summary=last_ir_summary,
                    geometric_feedback=last_geo_feedback,
                )
                
                # Show what errors we're asking the LLM to fix
                all_last_errors_for_prompt = last_semantic_errors + last_geo_errors
                print(f"\n{'='*60}")
                print(f"Repair pass {pass_idx + 1}: Errors being sent to LLM:")
                for e in all_last_errors_for_prompt[:5]:
                    print(f"  - {e}")
                print(f"{'='*60}")
                
                fix_response = self.text_generator._generate_text(fix_prompt)
                fixed_text = self.text_generator._clean_response(fix_response)
                unchanged = " ".join(fixed_text.split()).lower() == " ".join(last_text.split()).lower()
                
                print(f"\n{'='*60}")
                print(f"Repaired Text (pass {pass_idx + 1}){'  [UNCHANGED!]' if unchanged else ''}:")
                print(f"{'='*60}")
                print(fixed_text)
                print(f"{'='*60}\n")
                
                # Re-validate semantically
                fixed_validation = self.text_validator.validate_scenario(
                    text=fixed_text,
                    category=category,
                    difficulty=difficulty,
                    scenario_id=f"{category}_{difficulty}_fix{pass_idx+1}",
                )
                fixed_semantic_errors = (
                    fixed_validation.structural_errors +
                    fixed_validation.semantic_errors +
                    fixed_validation.parse_errors
                )
                
                # Re-extract IR and validate geometrically
                fixed_ir, fixed_geo_errors, fixed_ir_summary, fixed_geo_feedback = self._extract_and_validate_ir(fixed_text)
                
                # Always show validation feedback after each repair pass
                all_fixed_errors = fixed_semantic_errors + fixed_geo_errors
                if all_fixed_errors:
                    self.logger.info(f"  Validation errors: {all_fixed_errors[:3]}")
                elif fixed_validation.is_valid and not fixed_geo_errors:
                    self.logger.info(f"  Pass {pass_idx + 1} valid!")
                
                if self.config.verbose:
                    print(f"\n{'='*60}")
                    print(f"IR Extraction (pass {pass_idx + 1}):")
                    print(f"{'='*60}")
                    print(fixed_ir_summary)
                    if fixed_geo_errors:
                        print(f"\nGeometric errors: {fixed_geo_errors}")
                    print(f"{'='*60}\n")

                # Check if now valid
                if fixed_validation.is_valid and not fixed_geo_errors:
                    if self._is_duplicate(fixed_text):
                        last_text = fixed_text
                        last_semantic_errors = ["Too similar to existing scenario"]
                        last_geo_errors = []
                        last_ir_summary = fixed_ir_summary
                        last_geo_feedback = fixed_geo_feedback
                        self.logger.debug(f"  Pass {pass_idx + 1} valid but duplicate, continuing...")
                        continue
                    self.logger.debug(f"  Pass {pass_idx + 1} valid (semantic + geometric)!")
                    return fixed_text, "OK"

                # Update for next pass
                last_text = fixed_text
                last_semantic_errors = fixed_semantic_errors
                last_geo_errors = fixed_geo_errors
                last_ir_summary = fixed_ir_summary
                last_geo_feedback = fixed_geo_feedback
                
                all_last_errors = last_semantic_errors + last_geo_errors
                self.logger.debug(f"  Pass {pass_idx + 1} still has errors: {all_last_errors[:3]}")
                
                if unchanged and "Revision identical to previous output" not in all_last_errors:
                    last_semantic_errors.append("Revision identical to previous output; you MUST change the text")

            final_errors = last_semantic_errors + last_geo_errors
            return None, f"Text validation failed after {self.config.max_text_repair_passes} passes: {final_errors[:2]}"
            
        except Exception as e:
            import traceback
            return None, f"Text generation error: {e}\n{traceback.format_exc()}"
    
    def _run_and_validate(
        self,
        scenario_text: str,
        category: str,
        difficulty: int,
        scenario_id: str,
    ) -> Tuple[bool, Optional[str], Optional[SceneValidationResult], str]:
        """
        Run pipeline and validate the result.
        
        Returns:
            (success, scene_path, validation_result, status_message)
        """
        self.total_pipeline_runs += 1
        
        # Run pipeline
        success, scene_path, error = self.pipeline_runner.run_full_pipeline(
            scenario_text=scenario_text,
            category=category,
            difficulty=difficulty,
            scenario_id=scenario_id,
            town=self.config.town,
            mute=self.config.mute_pipeline,
        )
        
        if not success:
            return False, None, None, f"Pipeline failed: {error}"
        
        # Validate scene
        validation = self.scene_validator.validate_scene(
            scene_path,
            scenario_text,
            category=category,
            difficulty=difficulty,
        )
        
        # Show validation result
        self.logger.info(f"  Validation score: {validation.score:.2f} / {self.config.min_validation_score:.2f}")
        
        if not validation.is_valid or validation.score < self.config.min_validation_score:
            self.total_validation_failures += 1
            issues = [i.message for i in validation.get_errors()[:3]]
            error_count = len(validation.get_errors())
            # Clarify failure reason: score too low vs hard constraint errors
            if error_count > 0 and validation.score >= self.config.min_validation_score:
                reason = f"Validation failed ({error_count} error(s), score={validation.score:.2f} but has hard constraint violations): {issues}"
            elif validation.score < self.config.min_validation_score:
                reason = f"Validation failed (score={validation.score:.2f} < {self.config.min_validation_score:.2f}): {issues}"
            else:
                reason = f"Validation failed (score={validation.score:.2f}): {issues}"
            return False, scene_path, validation, reason
        
        return True, scene_path, validation, "OK"
    
    def _attempt_repairs(
        self,
        scenario_text: str,
        category: str,
        difficulty: int,
        scenario_id: str,
        validation: SceneValidationResult,
        scene_path: str,
    ) -> Tuple[bool, Optional[str], Optional[SceneValidationResult]]:
        """
        Attempt to repair a failed scenario using multiple strategies.
        
        Identifies the failing stage and tries different repair strategies
        with varying hyperparameters.
        
        Args:
            scenario_text: The scenario description
            category: Scenario category
            difficulty: Difficulty level
            scenario_id: Unique scenario ID
            validation: The failed validation result
            scene_path: Path to the failed scene
            
        Returns:
            (success, new_scene_path, new_validation)
        """
        # Identify the failing stage
        failing_stage = self.scene_validator.get_failing_stage(validation)
        
        if not failing_stage:
            # Can't determine stage, try path picker as default
            failing_stage = "step_03_path_picker"
        
        self.logger.info(f"  Identified failing stage: {failing_stage}")
        
        # Get repair strategies for this stage
        strategies = REPAIR_STRATEGIES.get(failing_stage, [])
        
        if not strategies:
            self.logger.info(f"  No repair strategies for stage: {failing_stage}")
            return False, None, None
        
        # Build validation feedback for repair context
        validation_feedback = self._build_validation_feedback(validation)
        
        # Try each repair strategy
        original_out_dir = Path(scene_path).parent if scene_path else None
        
        for i, strategy in enumerate(strategies):
            self.logger.info(f"  Repair attempt {i + 1}/{len(strategies)}: {strategy.name}")
            self.logger.info(f"    temp={strategy.temperature}, top_p={strategy.top_p}")
            
            if strategy.stage_to_retry == "regenerate":
                # Special case: regenerate text entirely
                self.logger.debug("    Regenerating scenario text...")
                return False, None, None  # Signal to caller to regenerate
            
            if original_out_dir is None:
                continue
            
            # Run repair
            self.total_pipeline_runs += 1
            success, new_scene_path, error = self.pipeline_runner.run_repair(
                scenario_text=scenario_text,
                scenario_id=scenario_id,
                repair_strategy=strategy,
                original_out_dir=original_out_dir,
                town=self.config.town,
                mute=self.config.mute_pipeline,
                validation_feedback=validation_feedback,
            )
            
            if not success:
                self.logger.info(f"    Repair failed: {error}")
                continue
            
            # Validate the repaired scene
            new_validation = self.scene_validator.validate_scene(
                new_scene_path,
                scenario_text,
                category=category,
                difficulty=difficulty,
            )
            
            if new_validation.is_valid and new_validation.score >= self.config.min_validation_score:
                self.logger.info(f"    Repair succeeded! score={new_validation.score:.2f}")
                return True, new_scene_path, new_validation
            else:
                self.logger.info(f"    Repair validation failed: score={new_validation.score:.2f}")
                issues = [i.message for i in new_validation.get_errors()[:2]]
                self.logger.debug(f"    Issues: {issues}")
        
        return False, None, None
    
    def _build_validation_feedback(self, validation: SceneValidationResult) -> str:
        """Build a feedback string from validation issues to help repair."""
        lines = []
        for issue in validation.issues[:5]:  # Top 5 issues
            lines.append(f"- {issue.message}")
            if issue.expected:
                lines.append(f"  Expected: {issue.expected}")
            if issue.actual:
                lines.append(f"  Actual: {issue.actual}")
        return "\n".join(lines)
    
    def generate_single(
        self,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate a single validated scenario with intelligent repair.
        
        Process:
        1. Generate scenario text
        2. Run full pipeline
        3. Validate scene against description
        4. If validation fails:
           a. Identify failing stage
           b. Try repair strategies with different hyperparameters
           c. If all repairs fail, regenerate text and retry
        5. Repeat up to max_retries times
        """
        self.total_attempts += 1
        
        # Select category and difficulty if not provided
        if category is None or difficulty is None:
            category, difficulty = self._select_category_and_difficulty()
        
        scenario_id = f"{category.replace(' ', '_')}_{difficulty}_{self.total_attempts}"
        
        result = GenerationResult(
            scenario_id=scenario_id,
            category=category,
            difficulty=difficulty,
            text="",
            success=False,
        )
        
        for attempt in range(self.config.max_retries_per_scenario):
            result.attempts = attempt + 1
            
            # Step 1: Generate text
            self.logger.debug(f"Attempt {attempt + 1}: Generating text for {category} (difficulty {difficulty})")
            
            scenario_text, text_status = self._generate_text(category, difficulty)
            
            if scenario_text is None:
                self.logger.info(f"  Text generation failed: {text_status}")
                continue
            
            result.text = scenario_text
            self.logger.debug(f"  Generated: {scenario_text[:80]}...")
            
            # Step 2: Run pipeline and validate
            self.logger.debug(f"  Running pipeline...")
            
            success, scene_path, validation, status = self._run_and_validate(
                scenario_text=scenario_text,
                category=category,
                difficulty=difficulty,
                scenario_id=f"{scenario_id}_attempt{attempt + 1}",
            )
            
            result.scene_path = scene_path
            result.validation_result = validation
            
            if success:
                result.success = True
                self.seen_texts.add(scenario_text)
                return result
            
            # Step 3: Validation failed - attempt repairs
            self.logger.info(f"  Initial run failed: {status}")
            
            if validation is not None and scene_path is not None and self.config.stage_repairs:
                self.logger.info(f"  Attempting repairs...")
                
                repair_success, repair_path, repair_validation = self._attempt_repairs(
                    scenario_text=scenario_text,
                    category=category,
                    difficulty=difficulty,
                    scenario_id=f"{scenario_id}_attempt{attempt + 1}",
                    validation=validation,
                    scene_path=scene_path,
                )
                
                if repair_success:
                    result.success = True
                    result.scene_path = repair_path
                    result.validation_result = repair_validation
                    self.seen_texts.add(scenario_text)
                    return result
                else:
                    self.logger.info(f"  All repairs failed, will regenerate text")
            elif not self.config.stage_repairs:
                self.logger.info(f"  Stage repairs disabled, will regenerate text")
            
            result.error_message = status
        
        return result
    
    def run(self) -> List[GenerationResult]:
        """
        Run the full generation loop until target_count scenarios are generated.
        
        Returns:
            List of successful GenerationResult objects
        """
        self.logger.divider()
        self.logger.info(f"Starting generation loop: target={self.config.target_count} scenarios")
        self.logger.info(f"Categories: {len(self.categories)}, Difficulties: {self.config.difficulties}")
        self.logger.divider()
        
        successful_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while successful_count < self.config.target_count:
            # Safety check
            if consecutive_failures >= max_consecutive_failures:
                self.logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping")
                break
            
            if self.total_attempts >= self.config.target_count * 5:
                self.logger.error(f"Reached maximum attempts ({self.total_attempts}), stopping")
                break
            
            # Select category and difficulty first so we can show what we're generating
            selected_category, selected_difficulty = self._select_category_and_difficulty()
            
            # Generate
            self.logger.progress(
                successful_count + 1,
                self.config.target_count,
                f"Generating {selected_category} (difficulty {selected_difficulty})..."
            )
            
            result = self.generate_single(
                category=selected_category,
                difficulty=selected_difficulty,
            )
            
            if result.success:
                successful_count += 1
                consecutive_failures = 0
                self.generated_scenarios.append(result)
                
                self.logger.success(
                    f"Generated: {result.category} (difficulty {result.difficulty}) "
                    f"[attempts: {result.attempts}, score: {result.validation_result.score:.2f}]"
                )
                self.logger.debug(f"  Text: {result.text[:100]}...")
                
                # Write routes for successful scenarios only (with clean scenario ID)
                if self.routes_out_dir and result.scene_path:
                    try:
                        from convert_scene_to_routes import convert_scene_to_routes
                        # Use clean scenario ID without attempt/repair suffixes
                        clean_scenario_id = f"{result.category.replace(' ', '_')}_{result.difficulty}"
                        routes_root = Path(self.routes_out_dir)
                        routes_dir = routes_root / clean_scenario_id
                        convert_scene_to_routes(
                            str(result.scene_path),
                            str(routes_dir),
                            ego_num=self.routes_ego_num,
                            align_routes=self.config.align_routes,
                            carla_host=self.config.carla_host,
                            carla_port=self.config.carla_port,
                        )
                    except Exception as exc:
                        self.logger.warning(f"  Routes generation failed: {exc}")
            else:
                consecutive_failures += 1
                self.failed_scenarios.append(result)
                
                self.logger.warning(
                    f"Failed after {result.attempts} attempts: {result.category} - {result.error_message}"
                )
        
        # Summary
        self.logger.divider()
        self.logger.info("Generation Complete!")
        self.logger.info(f"  Successful: {len(self.generated_scenarios)}")
        self.logger.info(f"  Failed: {len(self.failed_scenarios)}")
        self.logger.info(f"  Total attempts: {self.total_attempts}")
        self.logger.info(f"  Pipeline runs: {self.total_pipeline_runs}")
        self.logger.info(f"  Validation failures: {self.total_validation_failures}")
        self.logger.divider()
        
        # Save results
        self._save_results()
        
        return self.generated_scenarios
    
    def _save_results(self):
        """Save generation results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scenarios in the standard format
        scenarios_by_category: Dict[str, List[Dict]] = {}
        for result in self.generated_scenarios:
            if result.category not in scenarios_by_category:
                scenarios_by_category[result.category] = []
            scenarios_by_category[result.category].append({
                "difficulty": result.difficulty,
                "text": result.text,
            })
        
        with open(output_dir / "generated_scenarios.json", 'w') as f:
            json.dump(scenarios_by_category, f, indent=2)
        
        # Save detailed results
        detailed = {
            "generated_at": datetime.now().isoformat(),
            "config": {
                "target_count": self.config.target_count,
                "categories": self.categories,
                "difficulties": self.config.difficulties,
            },
            "statistics": {
                "successful": len(self.generated_scenarios),
                "failed": len(self.failed_scenarios),
                "total_attempts": self.total_attempts,
                "pipeline_runs": self.total_pipeline_runs,
                "validation_failures": self.total_validation_failures,
            },
            "scenarios": [
                {
                    "id": r.scenario_id,
                    "category": r.category,
                    "difficulty": r.difficulty,
                    "text": r.text,
                    "scene_path": r.scene_path,
                    "attempts": r.attempts,
                    "validation_score": r.validation_result.score if r.validation_result else None,
                }
                for r in self.generated_scenarios
            ],
        }
        
        with open(output_dir / "generation_results.json", 'w') as f:
            json.dump(detailed, f, indent=2)
        
        # Save log
        self.logger.save(str(output_dir / "generation_log.txt"))
        
        self.logger.info(f"Results saved to {output_dir}/")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for the generation loop."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate validated driving scenarios with retry logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10 scenarios across all categories
  python -m scenario_generator.generation_loop --count 10

  # Generate 5 scenarios for specific categories
  python -m scenario_generator.generation_loop --count 5 --categories "Unprotected Left Turn" "Lane Change Negotiation"

  # Generate with verbose output
  python -m scenario_generator.generation_loop --count 10 --verbose

  # Generate with custom output directory
  python -m scenario_generator.generation_loop --count 10 --output-dir my_scenarios
        """
    )
    
    parser.add_argument(
        "--count", "-n", type=int, default=10,
        help="Number of scenarios to generate (default: 10)"
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Specific categories to generate (default: all feasible)"
    )
    parser.add_argument(
        "--difficulties", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Difficulty levels to generate (default: 1-5)"
    )
    parser.add_argument(
        "--output-dir", default="log",
        help="Output directory (default: log)"
    )
    parser.add_argument(
        "--viz-objects", dest="viz_objects", action="store_true", default=True,
        help="Enable scene_objects.png visualization output (default: enabled)"
    )
    parser.add_argument(
        "--no-viz-objects", dest="viz_objects", action="store_false",
        help="Disable scene_objects.png visualization output"
    )
    parser.add_argument(
        "--routes-out-dir", default="routes",
        help="Directory to write routes output (default: routes)"
    )
    parser.add_argument(
        "--routes-ego-num", type=int, default=None,
        help="Override ego vehicle count for route conversion (default: auto-detect)"
    )
    parser.add_argument(
        "--town", default="Town05",
        help="CARLA town to use (default: Town05)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per scenario (default: 3)"
    )
    parser.add_argument(
        "--ir-repair-passes", type=int, default=1,
        help="LLM repair attempts for IR JSON parsing (default: 1)"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum validation score (default: 0.6)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--show-pipeline", action="store_true",
        help="Show pipeline output (not muted)"
    )
    
    args = parser.parse_args()
    
    # Validate categories
    feasible = get_feasible_categories()
    if args.categories:
        invalid = [c for c in args.categories if c not in feasible]
        if invalid:
            print(f"Error: Invalid categories: {invalid}")
            print(f"Available categories: {feasible}")
            sys.exit(1)
    
    # Create config
    config = GenerationLoopConfig(
        target_count=args.count,
        categories=args.categories,
        difficulties=args.difficulties,
        output_dir=args.output_dir,
        town=args.town,
        max_retries_per_scenario=args.max_retries,
        min_validation_score=args.min_score,
        mute_pipeline=not args.show_pipeline,
        verbose=args.verbose,
        viz_objects=args.viz_objects,
        routes_out_dir=args.routes_out_dir,
        routes_ego_num=args.routes_ego_num,
        ir_repair_passes=args.ir_repair_passes,
    )
    
    # Run generation loop
    loop = GenerationLoop(config)
    results = loop.run()
    
    print(f"\nGenerated {len(results)} scenarios")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
