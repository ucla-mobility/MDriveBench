"""
Scenario Generator

LLM-based scenario generation with validation loop.
Generates diverse scenarios within actual pipeline capabilities.

Features:
- Chain-of-thought reasoning before generating scenarios
- Research context integration for domain knowledge
- Optimized sampling for diversity
"""

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .capabilities import (
    CATEGORY_FEASIBILITY, CategoryFeasibility,
    TopologyType, ActorKind, MotionType, ConstraintType, EgoManeuver,
    LateralPosition, TimingPhase,
    get_feasible_categories,
)
from .constraints import (
    ScenarioSpec, EgoVehicleSpec, InterVehicleConstraint, NonEgoActorSpec,
    validate_spec, spec_to_dict,
)
from .validator import ScenarioValidator, ValidationResult


# =============================================================================
# RESEARCH CONTEXT (loaded from file or embedded)
# =============================================================================

def load_research_context() -> str:
    """Load research context from the MD file."""
    context_path = Path(__file__).parent / "research_context.md"
    if context_path.exists():
        return context_path.read_text()
    return ""  # Fallback to empty if not found


# Cache the research context
_RESEARCH_CONTEXT = None

def get_research_context() -> str:
    """Get cached research context."""
    global _RESEARCH_CONTEXT
    if _RESEARCH_CONTEXT is None:
        _RESEARCH_CONTEXT = load_research_context()
    return _RESEARCH_CONTEXT


# =============================================================================
# WHAT THE PIPELINE CAN ACTUALLY EXPRESS (for prompt grounding)
# =============================================================================

EXPRESSIBLE_ELEMENTS = """
WHAT YOU CAN SPECIFY (use ONLY these):

EGO VEHICLES:
- Any number of ego vehicles: "Vehicle 1", "Vehicle 2", etc.
- Maneuvers: straight, left turn, right turn, lane change
- Relative positions via constraints (see below)

CONSTRAINTS BETWEEN EGO VEHICLES (use these exact relationships):
- same_approach_as: "Vehicle 2 approaches from the same direction as Vehicle 1"
- opposite_approach_of: "Vehicle 2 approaches from the opposite direction"
- perpendicular_right_of: "Vehicle 2 approaches from the right of Vehicle 1"
- perpendicular_left_of: "Vehicle 2 approaches from the left of Vehicle 1"
- same_exit_as: "Vehicle 2 exits in the same direction as Vehicle 1"
- same_road_as: "Vehicle 2 is on the same road as Vehicle 1"
- follow_route_of: "Vehicle 2 follows behind Vehicle 1"
- left_lane_of: "Vehicle 2 is in the left lane relative to Vehicle 1"
- right_lane_of: "Vehicle 2 is in the right lane relative to Vehicle 1"
- merges_into_lane_of: "Vehicle 2 merges into the lane of Vehicle 1"
- Avoid cardinal directions (north/south/east/west). If you mention them, also state the relationship
  explicitly using the constraint phrases above (opposite/perpendicular/same).

NON-EGO ACTORS (simple spawn-and-move, NOT ego vehicles):
- parked_vehicle: "a parked vehicle blocking the lane"
- walker: "a pedestrian crossing the road"
- cyclist: "a cyclist in the bike lane"
- static_prop: "traffic cones arranged diagonally"

ACTOR MOTION TYPES:
- static: no movement
- cross_perpendicular: crosses the road (for walkers/cyclists)
- follow_lane: moves along the lane
- straight_line: moves in a straight line

ACTOR POSITIONS:
- Lateral: center, half_right, right_edge, offroad_right, half_left, left_edge, offroad_left
- Timing: on_approach, after_turn, in_intersection

WHAT YOU CANNOT SPECIFY (NEVER use these):
- Exact speeds (NO "30 km/h", "slow", "fast")
- Exact timing (NO "5 seconds later", "after 3 seconds")
- Behavioral traits (NO "aggressive", "hesitant", "distracted", "attentive")
- Dynamic reactions (NO "brakes when X happens", "accelerates to block")
- Traffic signals or lights
- Roundabouts, parking lots, emergency vehicles
- Exact distances in meters
"""


# =============================================================================
# MULTI-AGENT COORDINATION EMPHASIS
# =============================================================================

COORDINATION_EMPHASIS = """
CRITICAL: Your scenarios must require IMPLICIT COORDINATION between vehicles.

WHAT MAKES A GOOD MULTI-AGENT SCENARIO:
1. Multiple vehicles have CONFLICTING goals that require someone to yield/coordinate
2. The resolution is NOT obvious - reasonable drivers could disagree on who goes first
3. DO NOT explicitly say who should yield - let the situation imply it
4. Create STRANGE or RARE situations that humans rarely encounter
5. USE OCCLUSIONS AND PROPS LIBERALLY to create long-tail difficulty

EXAMPLES OF IMPLICIT COORDINATION (GOOD):
- "Vehicle 1 turns left while Vehicle 2 approaches from the opposite direction going straight, 
   and Vehicle 3 follows behind Vehicle 2" (left turner must yield to queue, timing unclear)
- "Vehicle 1, Vehicle 2, and Vehicle 3 each approach from different directions at an intersection, 
   each intending to go straight" (deadlock situation, no clear priority)
- "Vehicle 1 is in the right lane, Vehicle 2 is in the left lane of Vehicle 1, both intend 
   to merge into the same lane that Vehicle 3 already occupies" (merge conflict)
- "Vehicle 1 is blocked from seeing Vehicle 2 by a parked vehicle at the corner, forcing 
   Vehicle 1 to make an unprotected turn blind to oncoming traffic"

EXAMPLES OF BAD SCENARIOS:
- "Vehicle 1 yields to Vehicle 2" (explicitly says who yields - BAD)
- "Vehicle 1 goes first, then Vehicle 2" (specifies sequence - BAD)
- "Vehicle 2 waits for Vehicle 1" (specifies waiting - BAD)

LONG-TAIL SITUATION IDEAS (emphasize these):
- 4+ vehicles at uncontrolled intersection from different approaches
- Multiple vehicles trying to merge into same gap
- Chains of following vehicles meeting chains of oncoming vehicles
- Crossing pedestrian while multiple vehicles negotiate a turn
- Parked vehicle blocks sight lines during multi-way negotiation
- Static props (construction cones) narrow lanes, forcing difficult merges
- Multiple obstacles scattered to force vehicles to navigate tight spaces
- Pedestrian crossing while vehicles compete for merge gap
- Parked vehicles on both sides limit lane width, creating bottleneck
- Hidden vehicle emergent from behind parked vehicle at intersection
"""


# =============================================================================
# PROMPT TEMPLATES FOR LLM-BASED GENERATION
# =============================================================================

def build_system_prompt(include_research_context: bool = True) -> str:
    """Build the system prompt, optionally including research context."""
    
    research_section = ""
    if include_research_context:
        research_content = get_research_context()
        if research_content:
            research_section = f"""
---
RESEARCH CONTEXT ON MULTI-AGENT SCENARIOS:
{research_content}
---
"""
    
    return f"""You are an expert at generating DIFFICULT, MULTI-AGENT driving scenarios for autonomous vehicle testing.

Your goal is to create scenarios where multiple vehicles face IMPLICIT COORDINATION CHALLENGES - situations where it's unclear who should go first, and reasonable drivers could disagree.
{research_section}
{EXPRESSIBLE_ELEMENTS}

{COORDINATION_EMPHASIS}

Your scenarios must:
1. Use "Vehicle 1", "Vehicle 2", etc. for ego vehicles (use 3+ vehicles for difficulty)
2. Describe relationships using the constraint types listed above
3. Use only the actor types, motion types, and positions listed above
4. Create REALISTIC but RARE coordination challenges - the kind humans rarely encounter
5. NEVER explicitly state who yields, waits, or goes first - let the situation imply it"""


# Legacy constant for backwards compatibility
SYSTEM_PROMPT = build_system_prompt(include_research_context=False)


def parse_variation_axes(cat_info: CategoryFeasibility) -> Dict[str, List[str]]:
    """
    Parse variation axes into a dict mapping axis name to list of options.
    """
    axes = {}
    for axis in cat_info.variation_axes:
        if ": " in axis:
            axis_name, options_str = axis.split(": ", 1)
            options = [o.strip() for o in options_str.split(" vs ")]
            if options:
                axes[axis_name] = options
    return axes

def select_variation_values(cat_info: CategoryFeasibility, used_combinations: Optional[Set[str]] = None) -> Dict[str, str]:
    """
    Select values from each variation axis to force diversity.
    Tries to avoid combinations already used.
    Returns a dict mapping axis name to selected value.
    """
    axes = parse_variation_axes(cat_info)
    used_combinations = used_combinations or set()
    
    # Try random selections, avoiding used combinations
    for _ in range(50):  # Max attempts
        selections = {name: random.choice(options) for name, options in axes.items()}
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(selections.items()))
        if combo_key not in used_combinations:
            return selections
    
    # Fallback: just return random (better than nothing)
    return {name: random.choice(options) for name, options in axes.items()}


def get_all_combinations(cat_info: CategoryFeasibility) -> List[Dict[str, str]]:
    """
    Get all possible combinations of variation axis values.
    Useful for systematic generation.
    """
    import itertools
    
    axes = parse_variation_axes(cat_info)
    if not axes:
        return [{}]
    
    keys = list(axes.keys())
    value_lists = [axes[k] for k in keys]
    
    combinations = []
    for values in itertools.product(*value_lists):
        combinations.append(dict(zip(keys, values)))
    
    return combinations


def build_generation_prompt(
    category: str,
    difficulty: int,
    cat_info: CategoryFeasibility,
    existing_scenarios: List[str],
    forced_variations: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for generating a new scenario in this category."""
    
    # Get conflict mechanisms
    conflicts = "\n".join(f"  - {c}" for c in cat_info.conflict_via) if cat_info.conflict_via else "  - (use category-appropriate conflicts)"
    
    # Feature requirements - check difficulty threshold for non-ego actors
    features = []
    if cat_info.needs_oncoming:
        features.append("- Must involve opposite_approach_of constraint for oncoming traffic")
    if cat_info.needs_multi_lane:
        features.append("- Must use left_lane_of or right_lane_of constraints")
    if cat_info.needs_on_ramp:
        features.append("- Must include merge geometry (merges_into_lane_of, same_road_as)")
    
    # Non-ego actors only if difficulty meets threshold
    if cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
        features.append("- Must include non-ego actors (parked_vehicle, walker, cyclist, or static_prop)")
    elif cat_info.uses_non_ego_actors and difficulty < cat_info.non_ego_actors_min_difficulty:
        # Explicitly tell LLM not to include non-ego actors at low difficulty
        features.append(f"- NO non-ego actors (too simple for difficulty {difficulty})")
    
    features_str = "\n".join(features) if features else "- No special topology required"
    
    # Select random variation values for this generation to force diversity
    if forced_variations is None:
        forced_variations = select_variation_values(cat_info)
    
    variation_requirements = ""
    if forced_variations:
        variation_requirements = "\n\nREQUIRED VARIATION CHOICES (you MUST use these specific values):\n"
        for axis, value in forced_variations.items():
            variation_requirements += f"  - {axis}: {value}\n"
    
    # Existing scenarios to avoid repetition
    existing_str = ""
    if existing_scenarios:
        existing_str = "\n\nDO NOT REPEAT THESE EXISTING SCENARIOS (create something DIFFERENT):\n" + "\n".join(
            f"  [{i+1}] {s[:150]}..." if len(s) > 150 else f"  [{i+1}] {s}"
            for i, s in enumerate(existing_scenarios[-5:])
        )
    
    # Vehicle count based on difficulty
    min_vehicles = max(2, difficulty)
    max_vehicles = min_vehicles + 2
    # Minimum explicit constraints by difficulty (aligns with validator expectations)
    min_constraints = 2 if difficulty >= 4 else 1 if difficulty >= 3 else 0
    
    return f"""Generate a DIFFICULT multi-agent driving scenario for: {category}
Difficulty: {difficulty}/5

CATEGORY CONTEXT:
{cat_info.feasibility_notes}

CONFLICT MECHANISMS TO EXPRESS:
{conflicts}

HARD REQUIREMENTS (must satisfy ALL on the first pass):
{features_str}
- Use {min_vehicles} to {max_vehicles} ego vehicles NUMBERED as: Vehicle 1, Vehicle 2, Vehicle 3, etc.
- NEVER use letter names like "Vehicle A" or "Vehicle B" - ONLY use numbers!
- Explicitly mention Vehicle 1 through Vehicle {min_vehicles} by their NUMBER
- Include at least {min_constraints} explicit inter-vehicle constraints from the allowed list
- Include all required vehicles and explicit constraints in the SAME paragraph
- Create a situation requiring IMPLICIT coordination (vehicles must figure out who yields)
- DO NOT explicitly say who yields/waits/goes first
- Make it a LONG-TAIL situation - something unusual but physically possible
{variation_requirements}
{existing_str}

OCCLUSION & PROP GUIDANCE (USE LIBERALLY TO INCREASE DIFFICULTY):
- If a variation asks for occlusion, parked_vehicle, obstacle, or props - INCLUDE IT
- At LOW difficulty (1-2): Simple single obstacles (1-2 parked vehicles OR 3-4 cones)
- At HIGH difficulty (4-5): Complex multi-element arrangements
  * Work zone example: difficulty 2 = cones on one side narrowing one lane
  * Work zone example: difficulty 4+ = cones on BOTH sides creating extreme bottleneck
  * Weaving example: difficulty 2 = 3-4 props in a line
  * Weaving example: difficulty 5 = 6-8 props scattered creating random navigation
- Compound arrangements combine multiple types: cones + parked vehicles + pedestrians
- Scaling rule: More vehicles + narrower spaces + more props = higher difficulty

RULES (non-optional):
- Use ONLY the constraints, actors, and positions from the allowed list
- NO speeds, NO timing in seconds, NO behavioral descriptions
- Create a UNIQUE scenario different from any shown above
- The more vehicles involved in the conflict, the better
- If variation choices mention occlusion/props/obstacles - DEFINITELY USE THEM

THINK STEP-BY-STEP before generating (ensure difficulty requirements are satisfied):
1. First, in <planning> tags, think through:
   - What specific coordination challenge will this create?
   - How will vehicle paths conflict?
   - What makes this scenario ambiguous (no clear right-of-way)?
   - What constraints will create the desired relationships?
   - Are there variation choices about occlusion/props? How can I add them creatively?
   - How is this different from the existing scenarios above?

2. Then, in <scenario> tags, write the final scenario description starting with "Vehicle 1..."
   - Include all required vehicles and explicit constraints in the SAME paragraph

Generate:"""


def build_generation_prompt_simple(
    category: str,
    difficulty: int,
    cat_info: CategoryFeasibility,
    existing_scenarios: List[str],
    forced_variations: Optional[Dict[str, str]] = None,
) -> str:
    """Build a simpler prompt without chain-of-thought (for faster generation)."""
    
    # Get conflict mechanisms
    conflicts = "\n".join(f"  - {c}" for c in cat_info.conflict_via) if cat_info.conflict_via else "  - (use category-appropriate conflicts)"
    
    # Feature requirements
    features = []
    if cat_info.needs_oncoming:
        features.append("- Must involve opposite_approach_of constraint for oncoming traffic")
    if cat_info.needs_multi_lane:
        features.append("- Must use left_lane_of or right_lane_of constraints")
    if cat_info.needs_on_ramp:
        features.append("- Must include merge geometry (merges_into_lane_of, same_road_as)")
    if cat_info.uses_non_ego_actors:
        features.append("- Must include non-ego actors (parked_vehicle, walker, cyclist, or static_prop)")
    features_str = "\n".join(features) if features else "- No special topology required"
    
    # Select random variation values
    if forced_variations is None:
        forced_variations = select_variation_values(cat_info)
    
    variation_requirements = ""
    if forced_variations:
        variation_requirements = "\nRequired: " + ", ".join(f"{k}={v}" for k, v in forced_variations.items())
    
    # Vehicle count based on difficulty
    min_vehicles = max(2, difficulty)
    max_vehicles = min_vehicles + 2
    # Minimum explicit constraints by difficulty (aligns with validator expectations)
    min_constraints = 2 if difficulty >= 4 else 1 if difficulty >= 3 else 0
    
    # Existing scenarios summary
    existing_str = ""
    if existing_scenarios:
        existing_str = f"\nAvoid similarity to: {len(existing_scenarios)} previous scenarios"
    
    return f"""Category: {category} | Difficulty: {difficulty}/5 | Vehicles: {min_vehicles}-{max_vehicles}
{features_str}{variation_requirements}{existing_str}

HARD REQUIREMENTS (must satisfy ALL on the first pass):
- Explicitly mention Vehicle 1 through Vehicle {min_vehicles}
- Include at least {min_constraints} explicit inter-vehicle constraints from the allowed list
- Use {min_vehicles} to {max_vehicles} ego vehicles (Vehicle 1, Vehicle 2, ...)
- Include all required vehicles and explicit constraints in ONE paragraph
- Implicit coordination only (no explicit yields/waits)
Start with "Vehicle 1...":"""


def build_repair_prompt(
    scenario: str,
    errors: List[str],
    category: str,
    difficulty: int,
    cat_info: CategoryFeasibility,
) -> str:
    """Build a detailed critic prompt for text repair."""
    conflicts = "\n".join(f"  - {c}" for c in cat_info.conflict_via) if cat_info.conflict_via else "  - (use category-appropriate conflicts)"

    features = []
    if cat_info.needs_oncoming:
        features.append("- Must include opposite_approach_of (explicit phrase)")
    if cat_info.needs_multi_lane:
        features.append("- Must include left_lane_of or right_lane_of (explicit phrase)")
    if cat_info.needs_on_ramp:
        features.append("- Must mention on-ramp and include merge constraints")
    if cat_info.needs_merge:
        features.append("- Must include merge language and merges_into_lane_of or same_road_as")
    
    # Check difficulty threshold for non-ego actors
    if cat_info.uses_non_ego_actors and difficulty >= cat_info.non_ego_actors_min_difficulty:
        features.append("- Must include non-ego actors if required by the errors")
    elif cat_info.uses_non_ego_actors and difficulty < cat_info.non_ego_actors_min_difficulty:
        features.append(f"- NO non-ego actors (difficulty {difficulty} is too low)")
    
    features_str = "\n".join(features) if features else "- No special topology required"

    min_vehicles = max(2, difficulty)
    max_vehicles = min_vehicles + 2
    min_constraints = 2 if difficulty >= 4 else 1 if difficulty >= 3 else 0

    error_lines = "\n".join(f"  - {e}" for e in errors) if errors else "  - (none)"

    return f"""You are an expert scenario critic and pipeline engineer.
Your task is to REVISE the scenario to satisfy the validator errors and pipeline constraints.

ENGINEERING CONTEXT (what the pipeline can express):
{EXPRESSIBLE_ELEMENTS}

CATEGORY CONTEXT:
{cat_info.feasibility_notes}

CONFLICT MECHANISMS TO EXPRESS:
{conflicts}

REQUIREMENTS:
{features_str}
- Use {min_vehicles} to {max_vehicles} ego vehicles NUMBERED as: Vehicle 1, Vehicle 2, etc.
- NEVER use letter names like "Vehicle A" or "Vehicle B" - ONLY use numbers!
- Explicitly mention Vehicle 1 through Vehicle {min_vehicles} by their NUMBER
- Include at least {min_constraints} explicit inter-vehicle constraints from the allowed list
- If you mention cardinal directions (north/south/east/west), ALSO include an explicit constraint
  phrase like "approaches from the opposite direction" or "approaches from the perpendicular road"
- Keep the same core situation, only fix what is needed
- Do NOT output raw constraint labels like "opposite_approach_of(Vehicle 1, Vehicle 2)" or bullet lists.
  Use natural language sentences only.
- If possible, ADD occlusions (parked_vehicle), pedestrians (walker), or props (static_prop) to increase difficulty

REPAIR RULES (MUST FOLLOW):
- CONVERT any letter-named vehicles (A, B, C) to numbered vehicles (4, 5, 6, etc.)
- Your revision MUST be different from the original text (no copy-paste).
- Do NOT skip numbering (no "Vehicle 1" and "Vehicle 3" without "Vehicle 2").
- If an error says "at least N vehicles", explicitly add and describe Vehicle 1..N.
- If an error says "explicit inter-vehicle constraints", use phrases like:
  "Vehicle 2 approaches from the opposite direction as Vehicle 1",
  "Vehicle 3 is in the left lane of Vehicle 1",
  "Vehicle 4 follows behind Vehicle 2".
- If an error mentions merge/on-ramp/lane-drop, explicitly mention that and add a merge constraint.
- Consider enhancing with parked vehicles at corners, construction cones blocking lanes, or pedestrians crossing

VALIDATION ERRORS:
{error_lines}

SCENARIO TO FIX:
{scenario}

OUTPUT:
- Return ONLY the revised scenario description as a single paragraph
- Start with "Vehicle 1..."
- Preserve vehicle numbering when possible; only add vehicles if required by the errors
"""


def build_repair_prompt_with_ir(
    scenario: str,
    errors: List[str],
    category: str,
    difficulty: int,
    cat_info: CategoryFeasibility,
    ir_summary: str,
    geometric_feedback: str,
) -> str:
    """
    Build an enhanced repair prompt that includes IR extraction and geometric validation.
    
    This shows the LLM:
    1. How its text was interpreted (the IR)
    2. What geometric errors were found
    3. Specific fixes to apply
    
    This is much more effective than the basic repair prompt because it gives
    the LLM actionable, specific feedback rather than vague errors.
    """
    conflicts = "\n".join(f"  - {c}" for c in cat_info.conflict_via) if cat_info.conflict_via else "  - (use category-appropriate conflicts)"

    features = []
    if cat_info.needs_oncoming:
        features.append("- Must include opposite_approach_of (explicit phrase)")
    if cat_info.needs_multi_lane:
        features.append("- Must include left_lane_of or right_lane_of (explicit phrase)")
    if cat_info.needs_on_ramp:
        features.append("- Must mention on-ramp and include merge constraints")
    if cat_info.needs_merge:
        features.append("- Must include merge language and merges_into_lane_of or same_road_as")
    if cat_info.uses_non_ego_actors:
        features.append("- Must include non-ego actors if required by the errors")
    features_str = "\n".join(features) if features else "- No special topology required"

    min_vehicles = max(2, difficulty)
    max_vehicles = min_vehicles + 2
    min_constraints = 2 if difficulty >= 4 else 1 if difficulty >= 3 else 0

    error_lines = "\n".join(f"  - {e}" for e in errors) if errors else "  - (none)"

    return f"""You are an expert scenario critic with deep understanding of spatial geometry.
Your task is to REVISE the scenario to fix ALL geometric and validation errors.

=== HOW YOUR SCENARIO WAS INTERPRETED ===
{ir_summary}

=== GEOMETRIC CONSISTENCY ANALYSIS ===
{geometric_feedback}

=== VALIDATION ERRORS ===
{error_lines}

=== ORIGINAL SCENARIO ===
{scenario}

=== CATEGORY REQUIREMENTS ===
Category: {category}
{cat_info.feasibility_notes}

Conflict mechanisms to use:
{conflicts}

Topology requirements:
{features_str}

Vehicle requirements:
- Use {min_vehicles} to {max_vehicles} ego vehicles (Vehicle 1, Vehicle 2, ...)
- Include at least {min_constraints} explicit inter-vehicle constraints

=== GEOMETRIC RULES (MUST FOLLOW) ===
Understanding: "approach direction" = the direction a vehicle is TRAVELING/HEADING (not where it came from)
- "approaches from the north heading south" = approach direction S (southbound)
- "traveling northward" = approach direction N (northbound)

1. follows_route_of: Both vehicles must be HEADING the SAME direction
   - If V1 is heading South, V2 must also be heading South
   - WRONG: V1 heading South, V2 heading North (these are OPPOSITE, not following)

2. opposite_approach_of: Vehicles must be heading OPPOSITE directions (N/S or E/W)
   - CORRECT: V1 heading North, V2 heading South
   - WRONG: V1 heading North, V2 heading East (these are PERPENDICULAR)

3. perpendicular_left_of/perpendicular_right_of: Vehicles must be heading perpendicular directions
   - CORRECT: V1 heading North, V2 heading East or West
   - WRONG: V1 heading North, V2 heading South (these are OPPOSITE)

4. Maneuver consistency: heading + maneuver = new heading
   - Heading South + left turn = now heading East
   - Heading South + right turn = now heading West
   - Heading South + straight = still heading South

=== YOUR TASK ===
1. Fix ALL geometric errors listed above
2. Keep the creative scenario concept but correct the spatial relationships
3. Make sure constraints match the actual cardinal directions stated
4. Use natural language, NOT raw constraint labels

=== CRITICAL VEHICLE NAMING RULES ===
- ALWAYS use numbered vehicles: Vehicle 1, Vehicle 2, Vehicle 3, etc.
- NEVER use letter-named vehicles like "Vehicle A" or "Vehicle B"
- CONVERT any letter-named vehicles to numbered vehicles (A→4, B→5, C→6, etc.)

=== OUTPUT ===
Return ONLY the revised scenario as a single paragraph starting with "Vehicle 1..."
Do NOT explain your changes. Just output the fixed scenario.
"""


# =============================================================================
# GENERATOR CLASS
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for scenario generation."""
    # Model settings
    model_id: str = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    max_new_tokens: int = 1024  # Increased for chain-of-thought
    temperature: float = 0.85  # Slightly higher for more creativity
    top_p: float = 0.92
    
    # Diversity settings - these help avoid repetitive outputs
    repetition_penalty: float = 1.15  # Penalize repeating tokens
    presence_penalty: float = 0.3     # Penalize tokens that appeared at all
    frequency_penalty: float = 0.3    # Penalize based on frequency
    
    # Retry settings
    max_retries: int = 3
    
    # Diversity settings
    similarity_threshold: float = 0.7
    
    # Output settings
    output_format: str = "scenarios_json"  # only supported format
    
    # Chain-of-thought settings
    use_chain_of_thought: bool = True
    include_research_context: bool = True


class ScenarioGenerator:
    """
    Generates diverse, valid scenarios using LLM with validation loop.
    """
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        model=None,
        tokenizer=None,
    ):
        self.config = config or GenerationConfig()
        if self.config.output_format != "scenarios_json":
            raise ValueError(
                f"output_format '{self.config.output_format}' is not supported; "
                "use 'scenarios_json'."
            )
        self.validator = ScenarioValidator()
        self.feasible_categories = get_feasible_categories()
        
        # Model (lazy load)
        self._model = model
        self._tokenizer = tokenizer
        
        # Tracking for diversity
        self.generated_by_category: Dict[str, List[str]] = {
            cat: [] for cat in self.feasible_categories
        }
        # Track used variation combinations to force exploration
        self.used_combinations: Dict[str, Set[str]] = {
            cat: set() for cat in self.feasible_categories
        }
    
    def _load_model(self):
        """Lazy load the model if not provided."""
        if self._model is not None and self._tokenizer is not None:
            return
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"[INFO] Loading model: {self.config.model_id}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self._model.eval()
        print("[INFO] Model loaded")
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text from the model with optimized sampling for diversity."""
        self._load_model()
        import torch
        from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
        
        # Build system prompt (with or without research context)
        system_prompt = build_system_prompt(
            include_research_context=self.config.include_research_context
        )
        
        # Build chat format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = self._tokenizer(text, return_tensors="pt")
        if hasattr(self._model, 'device'):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        
        # Build generation kwargs with diversity settings
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        
        # Add repetition penalty if model supports it
        if self.config.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty

        # Optional presence/frequency penalties (OpenAI-style) via logits processor
        if self.config.presence_penalty != 0.0 or self.config.frequency_penalty != 0.0:
            class _PresenceFrequencyPenalty(LogitsProcessor):
                def __init__(self, presence_penalty: float, frequency_penalty: float, prompt_len: int):
                    self.presence_penalty = float(presence_penalty)
                    self.frequency_penalty = float(frequency_penalty)
                    self.prompt_len = int(prompt_len)

                def __call__(self, input_ids, scores):
                    if input_ids.size(1) <= self.prompt_len:
                        return scores
                    for batch_idx in range(input_ids.size(0)):
                        generated = input_ids[batch_idx, self.prompt_len:]
                        if generated.numel() == 0:
                            continue
                        unique_tokens, counts = torch.unique(generated, return_counts=True)
                        if self.presence_penalty != 0.0:
                            scores[batch_idx, unique_tokens] -= self.presence_penalty
                        if self.frequency_penalty != 0.0:
                            scores[batch_idx, unique_tokens] -= counts.to(scores.dtype) * self.frequency_penalty
                    return scores

            processors = LogitsProcessorList()
            processors.append(_PresenceFrequencyPenalty(
                presence_penalty=self.config.presence_penalty,
                frequency_penalty=self.config.frequency_penalty,
                prompt_len=prompt_len,
            ))
            gen_kwargs["logits_processor"] = processors

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Extract just the generated part (after the prompt) using token positions
        # This is more reliable than string matching which can fail due to encoding differences
        generated_tokens = outputs[0, prompt_len:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up LLM response to get just the scenario text."""
        # If using chain-of-thought, extract the scenario part
        if "<scenario>" in response.lower():
            # Extract content between <scenario> tags
            match = re.search(r'<scenario>(.*?)</scenario>', response, re.IGNORECASE | re.DOTALL)
            if match:
                response = match.group(1).strip()
        elif "<planning>" in response.lower():
            # If there's planning but no scenario tags, take everything after planning
            parts = re.split(r'</planning>', response, flags=re.IGNORECASE)
            if len(parts) > 1:
                response = parts[1].strip()
        
        # Remove any markdown formatting
        response = re.sub(r'^```.*?\n', '', response)
        response = re.sub(r'\n```$', '', response)
        response = response.strip()
        
        # Remove any leading labels
        response = re.sub(r'^(Scenario:|Description:|Output:)\s*', '', response, flags=re.IGNORECASE)
        
        # Find the first line starting with "Vehicle 1" if present
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip().lower().startswith('vehicle 1'):
                # Take from this line to the end of the paragraph
                remaining = '\n'.join(lines[i:])
                paragraphs = remaining.split('\n\n')
                if paragraphs:
                    response = paragraphs[0].strip()
                    break
        
        # Take first paragraph if multiple
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        if paragraphs:
            response = paragraphs[0]
        
        return response.strip()
    
    def _is_too_similar(self, new_text: str, existing: List[str]) -> bool:
        """Check if new scenario is too similar to existing ones."""
        if not existing:
            return False
        
        # Simple word overlap similarity
        new_words = set(new_text.lower().split())
        
        for old_text in existing:
            old_words = set(old_text.lower().split())
            if not new_words or not old_words:
                continue
            
            overlap = len(new_words & old_words)
            similarity = overlap / max(len(new_words), len(old_words))
            
            if similarity > self.config.similarity_threshold:
                return True
        
        return False
    
    def generate_scenario(
        self,
        category: str,
        difficulty: int,
    ) -> Tuple[Optional[str], ValidationResult]:
        """
        Generate a single scenario with validation loop.
        
        Returns (scenario_text, validation_result) or (None, result) if all retries failed.
        """
        if category not in self.feasible_categories:
            result = ValidationResult(
                scenario_id=f"{category}_{difficulty}",
                category=category,
                difficulty=difficulty,
                text="",
            )
            result.structural_errors = [f"Category '{category}' is not feasible"]
            return None, result
        
        cat_info = CATEGORY_FEASIBILITY[category]
        existing = self.generated_by_category.get(category, [])
        used_combos = self.used_combinations.get(category, set())
        
        # Select variation values, avoiding used combinations
        forced_variations = select_variation_values(cat_info, used_combos)
        combo_key = "|".join(f"{k}={v}" for k, v in sorted(forced_variations.items()))
        
        prompt_builder = build_generation_prompt if self.config.use_chain_of_thought else build_generation_prompt_simple

        for attempt in range(self.config.max_retries):
            # Generate with forced variations
            prompt = prompt_builder(category, difficulty, cat_info, existing, forced_variations)
            response = self._generate_text(prompt)
            scenario_text = self._clean_response(response)

            # Validate
            result = self.validator.validate_scenario(
                text=scenario_text,
                category=category,
                difficulty=difficulty,
                scenario_id=f"{category}_{difficulty}_attempt{attempt+1}",
            )

            if result.is_valid:
                # Check diversity
                if self._is_too_similar(scenario_text, existing):
                    result.parse_errors.append("Too similar to existing scenario")
                    result.is_parseable = False
                else:
                    # Success!
                    self.generated_by_category[category].append(scenario_text)
                    self.used_combinations[category].add(combo_key)
                    return scenario_text, result

            # Try to fix
            if attempt < self.config.max_retries - 1:
                errors = (
                    result.structural_errors +
                    result.semantic_errors +
                    result.parse_errors
                )
                fix_prompt = build_repair_prompt(
                    scenario=scenario_text,
                    errors=errors,
                    category=category,
                    difficulty=difficulty,
                    cat_info=cat_info,
                )
                response = self._generate_text(fix_prompt)
                fixed_text = self._clean_response(response)
                fixed_result = self.validator.validate_scenario(
                    text=fixed_text,
                    category=category,
                    difficulty=difficulty,
                    scenario_id=f"{category}_{difficulty}_attempt{attempt+1}_fix",
                )
                if fixed_result.is_valid:
                    if self._is_too_similar(fixed_text, existing):
                        fixed_result.parse_errors.append("Too similar to existing scenario")
                        fixed_result.is_parseable = False
                    else:
                        self.generated_by_category[category].append(fixed_text)
                        self.used_combinations[category].add(combo_key)
                        return fixed_text, fixed_result
                result = fixed_result
        
        # All retries failed
        return None, result
    
    def generate_batch(
        self,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[int]] = None,
        count_per_combination: int = 1,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate a batch of scenarios.
        
        Returns scenarios in the format matching scenarios.json:
        {
            "Category Name": [
                {"difficulty": 1, "text": "..."},
                ...
            ],
            ...
        }
        """
        categories = categories or self.feasible_categories
        difficulties = difficulties or [1, 2, 3, 4, 5]
        
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        for category in categories:
            if category not in self.feasible_categories:
                print(f"[WARN] Skipping infeasible category: {category}")
                continue
            
            results[category] = []
            
            for difficulty in difficulties:
                for i in range(count_per_combination):
                    print(f"[INFO] Generating: {category}, difficulty {difficulty}, attempt {i+1}/{count_per_combination}")
                    
                    scenario_text, validation = self.generate_scenario(category, difficulty)
                    
                    if scenario_text:
                        results[category].append({
                            "difficulty": difficulty,
                            "text": scenario_text,
                        })
                        print(f"  ✓ Success")
                    else:
                        print(f"  ✗ Failed: {validation.structural_errors + validation.semantic_errors + validation.parse_errors}")
        
        return results
    
    def generate_to_file(
        self,
        output_path: str,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[int]] = None,
        count_per_combination: int = 1,
    ):
        """Generate scenarios and write to file."""
        results = self.generate_batch(
            categories=categories,
            difficulties=difficulties,
            count_per_combination=count_per_combination,
        )
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        total = sum(len(scenarios) for scenarios in results.values())
        print(f"\n[INFO] Generated {total} scenarios to {output_path}")
        for cat, scenarios in results.items():
            print(f"  {cat}: {len(scenarios)}")


# =============================================================================
# TEMPLATE-BASED GENERATION (NO LLM REQUIRED)
# =============================================================================

class TemplateGenerator:
    """
    Generate scenarios from templates without requiring an LLM.
    Useful for rapid prototyping and testing.
    """
    
    def __init__(self):
        self.validator = ScenarioValidator()
        self.feasible_categories = get_feasible_categories()
    
    def _build_vehicle_description(
        self,
        vehicle_id: str,
        maneuver: str,
        entry_road: str,
        lane: Optional[str] = None,
    ) -> str:
        """Build description for a single vehicle."""
        parts = [vehicle_id]
        
        if entry_road == "main":
            parts.append("travels along the main road")
        elif entry_road == "side":
            parts.append("approaches from the side road")
        else:
            parts.append("travels")
        
        if lane:
            parts.append(f"in the {lane}")
        
        if maneuver == "left":
            parts.append("intending to turn left")
        elif maneuver == "right":
            parts.append("intending to turn right")
        elif maneuver == "lane_change":
            parts.append("intending to change lanes")
        else:
            parts.append("straight")
        
        return " ".join(parts)
    
    def _build_conflict_description(
        self,
        conflict_type: str,
        num_vehicles: int,
    ) -> str:
        """Build conflict description."""
        if conflict_type == "merge":
            return "The vehicles reach the merge region with overlapping timing, creating an ambiguous merge interaction."
        elif conflict_type == "intersection":
            return "Their arrival windows overlap at the intersection, creating ambiguous priority negotiation."
        elif conflict_type == "lane_contest":
            return "The lane positioning creates ambiguous negotiation for available gaps."
        elif conflict_type == "crosswalk":
            return "A pedestrian crossing creates a yield decision."
        else:
            return "Their paths interact, creating coordination requirements."
    
    def generate_from_spec(self, spec: ScenarioSpec) -> str:
        """Generate natural language from a ScenarioSpec."""
        return spec.generate_description()
    
    def generate_simple(
        self,
        category: str,
        difficulty: int,
        num_vehicles: int = 2,
    ) -> Optional[str]:
        """Generate a simple scenario using templates."""
        if category not in CATEGORY_FEASIBILITY:
            return None
        
        cat = CATEGORY_FEASIBILITY[category]
        
        # Build vehicle descriptions
        vehicles = []
        for i in range(1, num_vehicles + 1):
            maneuver = "straight"
            entry_road = "main" if i == 1 else "unknown"
            
            # Category-specific defaults
            if "Left Turn" in category and i == 1:
                maneuver = "left"
            if "Side Street" in category and i == 2:
                entry_road = "side"
            
            vehicles.append(self._build_vehicle_description(
                f"Vehicle {i}",
                maneuver,
                entry_road,
            ))
        
        # Build scenario
        parts = [". ".join(vehicles) + "."]
        
        # Add conflict
        conflict_type = "intersection"
        if cat.needs_on_ramp or cat.needs_merge:
            conflict_type = "merge"
        elif cat.needs_multi_lane:
            conflict_type = "lane_contest"
        
        parts.append(self._build_conflict_description(conflict_type, num_vehicles))
        
        return " ".join(parts)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for scenario generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate driving scenarios")
    parser.add_argument("--output", required=True, help="Output path for scenarios.json")
    parser.add_argument("--categories", nargs="+", help="Categories to generate (default: all feasible)")
    parser.add_argument("--difficulties", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Difficulty levels to generate")
    parser.add_argument("--count", type=int, default=1, help="Scenarios per category-difficulty combination")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model ID for generation")
    parser.add_argument("--template-only", action="store_true", help="Use templates instead of LLM")
    
    args = parser.parse_args()
    
    if args.template_only:
        gen = TemplateGenerator()
        results = {}
        for cat in args.categories or get_feasible_categories():
            results[cat] = []
            for diff in args.difficulties:
                text = gen.generate_simple(cat, diff, num_vehicles=min(diff + 1, 4))
                if text:
                    results[cat].append({"difficulty": diff, "text": text})
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Generated {sum(len(v) for v in results.values())} scenarios to {args.output}")
    else:
        config = GenerationConfig(model_id=args.model)
        gen = ScenarioGenerator(config)
        gen.generate_to_file(
            args.output,
            categories=args.categories,
            difficulties=args.difficulties,
            count_per_combination=args.count,
        )


if __name__ == "__main__":
    main()
