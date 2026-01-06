"""
Scenario Validator

Validates scenario descriptions against pipeline capabilities using heuristics.
Does not run the full pipeline.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .capabilities import (
    CATEGORY_FEASIBILITY, TopologyType, ActorKind, MotionType,
    get_feasible_categories, get_infeasible_categories,
)
from .constraints import ScenarioSpec, validate_spec


@dataclass
class ValidationResult:
    """Result of validating a scenario."""
    scenario_id: str
    category: str
    difficulty: int
    text: str
    
    # Validation stages
    is_structurally_valid: bool = False
    is_semantically_valid: bool = False
    is_parseable: bool = False
    
    # Errors at each stage
    structural_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)
    parse_errors: List[str] = field(default_factory=list)
    
    # Extracted info
    detected_vehicles: List[str] = field(default_factory=list)
    detected_actors: List[str] = field(default_factory=list)
    detected_topology: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        return self.is_structurally_valid and self.is_semantically_valid and self.is_parseable
    
    def summary(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        lines = [
            f"{status}: {self.scenario_id} ({self.category}, difficulty {self.difficulty})",
        ]
        
        if self.structural_errors:
            lines.append("  Structural errors:")
            for e in self.structural_errors:
                lines.append(f"    - {e}")
        
        if self.semantic_errors:
            lines.append("  Semantic errors:")
            for e in self.semantic_errors:
                lines.append(f"    - {e}")
        
        if self.parse_errors:
            lines.append("  Parse errors:")
            for e in self.parse_errors:
                lines.append(f"    - {e}")
        
        return "\n".join(lines)


class ScenarioValidator:
    """
    Validates scenarios at multiple levels:
    
    1. Structural validation: Basic format and category checks
    2. Semantic validation: Checks against pipeline capabilities  
    3. Parse validation: Heuristic checks for extractable phrasing
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.feasible_categories = set(get_feasible_categories())
        self.infeasible_categories = set(get_infeasible_categories())
    
    def validate_scenario(
        self,
        text: str,
        category: str,
        difficulty: int,
        scenario_id: str = "unknown",
    ) -> ValidationResult:
        """Validate a single scenario."""
        result = ValidationResult(
            scenario_id=scenario_id,
            category=category,
            difficulty=difficulty,
            text=text,
        )
        
        # Stage 1: Structural validation
        self._validate_structure(result)
        
        # Stage 2: Semantic validation
        if result.is_structurally_valid:
            self._validate_semantics(result)
        
        # Stage 3: Parse validation
        if result.is_semantically_valid:
            self._validate_parseability(result)
        
        return result
    
    def _validate_structure(self, result: ValidationResult):
        """Check basic structure: category, difficulty, text format."""
        errors = []
        
        # Check category
        if result.category in self.infeasible_categories:
            errors.append(f"Category '{result.category}' is NOT FEASIBLE with current pipeline")
        elif result.category not in self.feasible_categories:
            errors.append(f"Unknown category: '{result.category}'")
        
        # Check difficulty
        if not 1 <= result.difficulty <= 5:
            errors.append(f"Difficulty must be 1-5, got {result.difficulty}")
        
        # Check text is not empty
        if not result.text or not result.text.strip():
            errors.append("Scenario text is empty")
        
        # Check text length (too short = probably incomplete, too long = probably won't fit context)
        text_len = len(result.text)
        if text_len < 50:
            errors.append(f"Text too short ({text_len} chars) - may be incomplete")
        if text_len > 1500:
            errors.append(f"Text very long ({text_len} chars) - may cause context issues")
        
        result.structural_errors = errors
        result.is_structurally_valid = len(errors) == 0
    
    def _validate_semantics(self, result: ValidationResult):
        """Check semantic correctness against pipeline capabilities."""
        errors = []
        warnings = []
        text_lower = result.text.lower()

        def _extract_vehicle_cardinals(text: str) -> Dict[str, str]:
            patterns = [
                re.compile(r"vehicle\s*(\d+)[^.]*?\bapproach(?:es|ing)?\s+from\s+(north|south|east|west)\b"),
                re.compile(r"vehicle\s*(\d+)[^.]*?\bcoming\s+from\s+(north|south|east|west)\b"),
                re.compile(r"vehicle\s*(\d+)[^.]*?\bfrom\s+(north|south|east|west)\b"),
            ]
            found: Dict[str, str] = {}
            for pattern in patterns:
                for match in pattern.finditer(text):
                    vehicle = f"Vehicle {match.group(1)}"
                    if vehicle not in found:
                        found[vehicle] = match.group(2)
            return found
        
        # Extract mentioned vehicles
        vehicle_pattern = r'vehicle\s*(\d+)'
        vehicle_matches = re.findall(vehicle_pattern, text_lower)
        result.detected_vehicles = [f"Vehicle {n}" for n in sorted(set(vehicle_matches), key=int)]
        
        # Check vehicle count
        num_vehicles = len(result.detected_vehicles)
        if num_vehicles == 0:
            errors.append("No ego vehicles detected (need 'Vehicle 1', 'Vehicle 2', etc.)")
        elif num_vehicles < 2:
            errors.append("Multi-agent scenarios require at least 2 vehicles")
        # No upper limit on ego vehicles - the pipeline supports any reasonable number
        
        # Check for multi-agent coordination patterns
        # These match both the constraint syntax (with underscores) and natural language variants
        coordination_patterns = [
            (r'opposite[_\s]*(approach|direction)', "opposite_approach_of"),
            (r'perpendicular[_\s]*(left|right)?[_\s]*(of)?', "perpendicular constraint"),
            (r'same[_\s]*(approach|direction|road|exit)[_\s]*(as|of)?', "same_* constraint"),
            (r'follow[s]?[_\s]*(route|behind|vehicle)', "follow_route_of"),
            (r'(left|right)[_\s]*lane[_\s]*(of)?', "lane constraint"),
            (r'merge[s]?[_\s]*into[_\s]*(lane)?', "merges_into_lane_of"),
            (r'oncoming', "opposite_approach (oncoming)"),
            (r'from\s+(the\s+)?(opposite|other)\s+direction', "opposite_approach (from opposite)"),
            (r'approaching\s+from\s+', "approach constraint"),
            (r'in\s+the\s+(left|right)\s+lane\s+of', "lane constraint"),
            (r'behind\s+vehicle', "follow constraint"),
            (r'ahead\s+of\s+vehicle', "follow constraint"),
        ]
        
        detected_constraints = []
        for pattern, constraint_name in coordination_patterns:
            if re.search(pattern, text_lower):
                detected_constraints.append(constraint_name)

        cardinal_map = _extract_vehicle_cardinals(text_lower)
        cardinal_set = set(cardinal_map.values())
        has_opposite_cardinals = ("north" in cardinal_set and "south" in cardinal_set) or (
            "east" in cardinal_set and "west" in cardinal_set
        )
        has_perpendicular_cardinals = any(
            a in cardinal_set and b in cardinal_set
            for a, b in [("north", "east"), ("north", "west"), ("south", "east"), ("south", "west")]
        )
        if has_opposite_cardinals:
            detected_constraints.append("opposite_approach (cardinal)")
        if has_perpendicular_cardinals:
            detected_constraints.append("perpendicular (cardinal)")
        
        # For difficulty 3+, should have explicit coordination
        if result.difficulty >= 3 and len(detected_constraints) == 0:
            errors.append(f"Difficulty {result.difficulty} scenario should have explicit inter-vehicle constraints")
        
        # For difficulty 4+, should have multiple vehicles and complex coordination
        if result.difficulty >= 4:
            if num_vehicles < 3:
                errors.append(f"Difficulty {result.difficulty} scenario should have at least 3 vehicles")
            if len(detected_constraints) < 2:
                errors.append(f"Difficulty {result.difficulty} scenario should have multiple coordination constraints")
        
        # Check for impossible specifications
        impossible_patterns = [
            (r'\d+\s*(m/s|km/h|mph|meters?\s*per\s*second)', "Cannot specify exact speeds"),
            (r'\d+\s*seconds?\s*(later|after|before)', "Cannot specify exact timing in seconds"),
            (r'(traffic\s*light|traffic\s*signal|red\s*light|green\s*light)', "Cannot control traffic signals"),
            (r'roundabout', "Roundabout scenarios not supported (no detection)"),
            (r'(parking\s*lot|parking\s*structure|parking\s*garage)', "Parking lot scenarios not supported"),
            (r'(emergency\s*vehicle|ambulance|fire\s*truck|police)', "Emergency vehicle scenarios not supported"),
            # Only flag explicit prescriptive statements about yielding, not dilemma descriptions
            # OK: "unclear who should yield" (describes dilemma)
            # OK: "must decide whether to yield" (describes decision)
            # BAD: "Vehicle 1 yields to Vehicle 2" (prescribes specific outcome)
            # BAD: "Vehicle 1 must yield" (prescribes specific action)
            (r'vehicle\s*\d\s+(yields?|waits?|goes?\s*first|lets?\s+vehicle)', "Should not explicitly state who yields/waits - let coordination be implicit"),
            (r'vehicle\s*\d\s+(must|should|will)\s+(yield|wait|go\s*first)', "Should not explicitly state who yields/waits - let coordination be implicit"),
        ]
        
        for pattern, error_msg in impossible_patterns:
            if re.search(pattern, text_lower):
                errors.append(error_msg)
        
        # Check category-specific requirements
        cat = CATEGORY_FEASIBILITY.get(result.category)
        if cat:
            if cat.needs_oncoming and not (
                re.search(r"(oncoming|opposing|opposite\s+direction)", text_lower) or has_opposite_cardinals
            ):
                errors.append("Category requires oncoming/opposite approaches but text does not state it")

            if cat.needs_on_ramp and not re.search(r"\bon[-\s]?ramp\b", text_lower):
                errors.append("Category requires an on-ramp merge; include 'on-ramp' wording")

            if cat.needs_merge and not re.search(r"\bmerge\b|\bmerges\b|\bmerging\b", text_lower):
                errors.append("Category requires merge behavior but no merge language was found")

            if cat.needs_multi_lane and not re.search(
                r"(left\s+lane|right\s+lane|adjacent\s+lane|multi[-\s]?lane|two[-\s]?lane|left_lane|right_lane|adjacent_lane)",
                text_lower,
            ):
                errors.append("Category requires multi-lane context (left/right/adjacent lane)")

            if "weaving" in result.category.lower() and not re.search(
                r"(lane\s+change|changes?\s+lane|weav|merge)", text_lower
            ):
                errors.append("Highway weaving scenarios must mention lane changes or weaving/merge behavior")

            if "lane drop" in result.category.lower() and not re.search(
                r"(lane\s+drop|zipper)", text_lower
            ):
                errors.append("Lane drop scenarios must mention lane drop/zipper behavior explicitly")
        
        # Detect non-ego actors
        actor_patterns = [
            (r'pedestrian|walker|person crossing', 'walker'),
            (r'cyclist|bicycle|bicyclist', 'cyclist'),
            (r'parked (car|vehicle|truck|van)', 'parked_vehicle'),
            (r'stopped (car|vehicle|truck)', 'parked_vehicle'),
            (r'(traffic )?cone[s]?', 'static_prop'),
            (r'barrier[s]?', 'static_prop'),
            (r'obstacle', 'static_prop'),
        ]
        
        detected = set()
        for pattern, actor_type in actor_patterns:
            if re.search(pattern, text_lower):
                detected.add(actor_type)
        result.detected_actors = list(detected)
        
        # Infer topology from text
        if 'intersection' in text_lower or 'junction' in text_lower:
            if 't-junction' in text_lower or 't junction' in text_lower or 'side road' in text_lower:
                result.detected_topology = 't_junction'
            else:
                result.detected_topology = 'intersection'
        elif 'merge' in text_lower or 'ramp' in text_lower:
            result.detected_topology = 'corridor'  # with merge features
        elif 'lane' in text_lower or 'straight' in text_lower:
            result.detected_topology = 'corridor'
        else:
            result.detected_topology = 'unknown'
        
        result.semantic_errors = errors
        result.is_semantically_valid = len(errors) == 0
    
    def _validate_parseability(self, result: ValidationResult):
        """Check if the scenario can be parsed by the pipeline extractors."""
        errors = []
        text_lower = result.text.lower()
        
        # Check for clear vehicle descriptions
        for veh in result.detected_vehicles:
            veh_lower = veh.lower()
            # Look for clear action description for this vehicle
            veh_patterns = [
                rf'{veh_lower}\s+(travels|approaches|intends|turns|continues)',
                rf'{veh_lower}\s+is\s+(in|on|behind|ahead)',
            ]
            has_action = any(re.search(p, text_lower) for p in veh_patterns)
            if not has_action:
                # Warning, not error - might still work
                pass
        
        # Check for ambiguous references that confuse extractors
        ambiguous_patterns = [
            (r'it\s+(travels|turns|continues)', "Ambiguous 'it' reference - use explicit vehicle names"),
            (r'the vehicle\s+(travels|turns)', "Ambiguous 'the vehicle' - use 'Vehicle 1', etc."),
            (r'both vehicles', "May be ambiguous - specify which vehicles explicitly"),
        ]
        
        for pattern, warning in ambiguous_patterns:
            if re.search(pattern, text_lower):
                # These are warnings, not hard errors
                pass
        
        # Check for NPC vehicle confusion
        # The pipeline has special handling for "Vehicle X is an NPC"
        npc_pattern = r'vehicle\s+(\d+)\s+is\s+(?:an?\s+)?(?:npc|non-?player)'
        npc_matches = re.findall(npc_pattern, text_lower)
        if npc_matches:
            npc_ids = [f"Vehicle {n}" for n in npc_matches]
            # Check these aren't the only vehicles
            non_npc = [v for v in result.detected_vehicles if v not in npc_ids]
            if not non_npc:
                errors.append("All detected vehicles are NPCs - need at least one ego vehicle")
        
        result.parse_errors = errors
        result.is_parseable = len(errors) == 0
    
    def validate_scenarios_file(
        self,
        scenarios_path: str,
    ) -> Dict[str, List[ValidationResult]]:
        """Validate all scenarios in a scenarios.json file."""
        with open(scenarios_path, 'r') as f:
            scenarios = json.load(f)
        
        results_by_category: Dict[str, List[ValidationResult]] = {}
        
        for category, scenario_list in scenarios.items():
            results_by_category[category] = []
            
            for i, scenario in enumerate(scenario_list):
                difficulty = scenario.get('difficulty', i + 1)
                text = scenario.get('text', '')
                scenario_id = f"{category}_{difficulty}"
                
                result = self.validate_scenario(
                    text=text,
                    category=category,
                    difficulty=difficulty,
                    scenario_id=scenario_id,
                )
                results_by_category[category].append(result)
                
                if self.verbose:
                    print(result.summary())
                    print()
        
        return results_by_category
    
    def print_validation_report(
        self,
        results_by_category: Dict[str, List[ValidationResult]],
    ):
        """Print a summary report of validation results."""
        total = 0
        valid = 0
        invalid_by_category: Dict[str, int] = {}
        
        for category, results in results_by_category.items():
            cat_invalid = 0
            for r in results:
                total += 1
                if r.is_valid:
                    valid += 1
                else:
                    cat_invalid += 1
            if cat_invalid > 0:
                invalid_by_category[category] = cat_invalid
        
        print("=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        print(f"Total scenarios: {total}")
        print(f"Valid: {valid} ({100*valid/total:.1f}%)")
        print(f"Invalid: {total - valid} ({100*(total-valid)/total:.1f}%)")
        
        if invalid_by_category:
            print("\nInvalid by category:")
            for cat, count in sorted(invalid_by_category.items()):
                print(f"  {cat}: {count}")
        
        print("=" * 60)


def main():
    """CLI for scenario validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate scenario descriptions")
    parser.add_argument("--input", required=True, help="Path to scenarios.json")
    parser.add_argument("--verbose", action="store_true", help="Print details for each scenario")
    
    args = parser.parse_args()
    
    validator = ScenarioValidator(verbose=args.verbose)
    results = validator.validate_scenarios_file(args.input)
    validator.print_validation_report(results)


if __name__ == "__main__":
    main()
