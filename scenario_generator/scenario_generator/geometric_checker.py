"""
Geometric Consistency Checker

Deterministic validation of geometric constraints extracted from scenario IR.
These checks CANNOT be wrong - they are pure logic based on cardinal directions.

This module catches the spatial reasoning errors that LLMs commonly make,
such as claiming two vehicles "follow" each other when they're going opposite directions.

CRITICAL CONVENTION (matches pipeline):
- approach_direction = HEADING direction (where vehicle is GOING)
- "approaches from the north heading south" → approach_direction = S (southbound)
- Two vehicles "follow" each other if they have the SAME approach_direction
- Two vehicles are "opposite" if approach_directions are opposite (N/S or E/W)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .scenario_ir import ScenarioIR, VehicleIR, ConstraintIR, Cardinal


# =============================================================================
# GEOMETRIC RULES (100% deterministic)
# =============================================================================

# Opposite pairs: N/S and E/W (for heading directions)
OPPOSITE_PAIRS = frozenset([
    (Cardinal.NORTH, Cardinal.SOUTH),
    (Cardinal.SOUTH, Cardinal.NORTH),
    (Cardinal.EAST, Cardinal.WEST),
    (Cardinal.WEST, Cardinal.EAST),
])

# Perpendicular pairs (for heading directions)
PERPENDICULAR_PAIRS = frozenset([
    (Cardinal.NORTH, Cardinal.EAST),
    (Cardinal.NORTH, Cardinal.WEST),
    (Cardinal.SOUTH, Cardinal.EAST),
    (Cardinal.SOUTH, Cardinal.WEST),
    (Cardinal.EAST, Cardinal.NORTH),
    (Cardinal.EAST, Cardinal.SOUTH),
    (Cardinal.WEST, Cardinal.NORTH),
    (Cardinal.WEST, Cardinal.SOUTH),
])


# =============================================================================
# ERROR DATACLASS
# =============================================================================

@dataclass
class GeometricError:
    """
    A geometric consistency error with specific, actionable feedback.
    """
    error_type: str  # e.g., "constraint_violation", "maneuver_mismatch"
    severity: str  # "error" (must fix) or "warning" (should review)
    message: str  # Human-readable error message
    affected_vehicles: List[int] = field(default_factory=list)
    affected_constraint: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    
    def to_string(self) -> str:
        lines = [f"[ERROR] {self.message}"]
        if self.suggested_fixes:
            lines.append("   FIX OPTIONS:")
            for i, fix in enumerate(self.suggested_fixes, 1):
                lines.append(f"   {i}. {fix}")
        return "\n".join(lines)


@dataclass
class GeometricValidationResult:
    """
    Result of geometric validation.
    """
    is_valid: bool
    errors: List[GeometricError] = field(default_factory=list)
    warnings: List[GeometricError] = field(default_factory=list)
    
    def all_issues(self) -> List[GeometricError]:
        return self.errors + self.warnings
    
    def to_string(self) -> str:
        if self.is_valid and not self.warnings:
            return "Geometric consistency: PASSED"
        
        lines = []
        if self.errors:
            lines.append("GEOMETRIC ERRORS (must fix):")
            for e in self.errors:
                lines.append(e.to_string())
        if self.warnings:
            lines.append("GEOMETRIC WARNINGS (should review):")
            for w in self.warnings:
                lines.append(w.to_string())
        return "\n".join(lines)


# =============================================================================
# CONSTRAINT VALIDATORS
# =============================================================================

def check_follows_route_of(
    v1: VehicleIR,
    v2: VehicleIR,
    constraint: ConstraintIR,
) -> Optional[GeometricError]:
    """
    Check follows_route_of constraint.
    
    RULE: For V1 to follow V2, they must be HEADING the SAME direction.
    - Same approach_direction = same heading = valid
    - Opposite approach_directions = opposite headings = invalid
    
    Convention: approach_direction = heading direction (where vehicle is GOING)
    """
    if v1.approach_direction == Cardinal.UNKNOWN or v2.approach_direction == Cardinal.UNKNOWN:
        return None  # Can't validate without direction info
    
    # Same approach_direction = same heading = valid following
    if v1.approach_direction == v2.approach_direction:
        return None  # Valid
    
    # Check if they're going opposite directions
    pair = (v1.approach_direction, v2.approach_direction)
    if pair in OPPOSITE_PAIRS:
        return GeometricError(
            error_type="constraint_violation",
            severity="error",
            message=(
                f"follows_route_of(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
                f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
                f"They are going OPPOSITE directions, not following."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint="follows_route_of",
            suggested_fixes=[
                f"Change to opposite_approach_of(V{v1.vehicle_id}, V{v2.vehicle_id}) since they're going opposite directions",
                f"Change V{v1.vehicle_id} to head {v2.approach_direction.value} so it actually follows V{v2.vehicle_id}",
                f"Remove the follows_route_of constraint entirely",
            ],
        )
    
    # Perpendicular - also invalid for following
    if pair in PERPENDICULAR_PAIRS:
        return GeometricError(
            error_type="constraint_violation", 
            severity="error",
            message=(
                f"follows_route_of(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
                f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
                f"They are on PERPENDICULAR roads, not following."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint="follows_route_of",
            suggested_fixes=[
                f"Change to perpendicular_left_of or perpendicular_right_of",
                f"Change V{v1.vehicle_id} to head {v2.approach_direction.value} to actually follow",
                f"Remove the follows_route_of constraint",
            ],
        )
    
    return None


def check_opposite_approach_of(
    v1: VehicleIR,
    v2: VehicleIR,
    constraint: ConstraintIR,
) -> Optional[GeometricError]:
    """
    Check opposite_approach_of constraint.
    
    RULE: For opposite approach, vehicles must be HEADING opposite directions (N/S or E/W).
    
    Convention: approach_direction = heading direction
    """
    if v1.approach_direction == Cardinal.UNKNOWN or v2.approach_direction == Cardinal.UNKNOWN:
        return None
    
    pair = (v1.approach_direction, v2.approach_direction)
    if pair in OPPOSITE_PAIRS:
        return None  # Valid
    
    if pair in PERPENDICULAR_PAIRS:
        return GeometricError(
            error_type="constraint_violation",
            severity="error", 
            message=(
                f"opposite_approach_of(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
                f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
                f"These are PERPENDICULAR, not opposite."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint="opposite_approach_of",
            suggested_fixes=[
                f"Change to perpendicular_left_of or perpendicular_right_of",
                f"Change V{v2.vehicle_id} to head {v1.approach_direction.opposite().value} for true opposite",
            ],
        )
    
    if v1.approach_direction == v2.approach_direction:
        return GeometricError(
            error_type="constraint_violation",
            severity="error",
            message=(
                f"opposite_approach_of(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"Both vehicles are heading {v1.approach_direction.value}. "
                f"They are going the SAME direction, not opposite."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint="opposite_approach_of",
            suggested_fixes=[
                f"Change to same_approach_as or follows_route_of",
                f"Change V{v2.vehicle_id} to head {v1.approach_direction.opposite().value}",
            ],
        )
    
    return None


def check_perpendicular_of(
    v1: VehicleIR,
    v2: VehicleIR,
    constraint: ConstraintIR,
) -> Optional[GeometricError]:
    """
    Check perpendicular_left_of or perpendicular_right_of constraints.
    
    RULE: Vehicles must be HEADING perpendicular directions (N/E, N/W, S/E, S/W).
    
    Convention: approach_direction = heading direction
    """
    if v1.approach_direction == Cardinal.UNKNOWN or v2.approach_direction == Cardinal.UNKNOWN:
        return None
    
    pair = (v1.approach_direction, v2.approach_direction)
    if pair in PERPENDICULAR_PAIRS:
        return None  # Valid
    
    if pair in OPPOSITE_PAIRS:
        return GeometricError(
            error_type="constraint_violation",
            severity="error",
            message=(
                f"{constraint.constraint_type}(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
                f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
                f"These are OPPOSITE, not perpendicular."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint=constraint.constraint_type,
            suggested_fixes=[
                f"Change to opposite_approach_of",
                f"Change heading directions to be perpendicular (e.g., N/E or N/W)",
            ],
        )
    
    if v1.approach_direction == v2.approach_direction:
        return GeometricError(
            error_type="constraint_violation",
            severity="error",
            message=(
                f"{constraint.constraint_type}(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"Both are heading {v1.approach_direction.value}. "
                f"They are going the SAME direction, not perpendicular."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint=constraint.constraint_type,
            suggested_fixes=[
                f"Change to same_approach_as or follows_route_of",
                f"Change one vehicle's heading to be perpendicular",
            ],
        )
    
    return None


def check_same_approach_as(
    v1: VehicleIR,
    v2: VehicleIR,
    constraint: ConstraintIR,
) -> Optional[GeometricError]:
    """
    Check same_approach_as constraint.
    
    RULE: Vehicles must be HEADING the same direction.
    
    Convention: approach_direction = heading direction
    """
    if v1.approach_direction == Cardinal.UNKNOWN or v2.approach_direction == Cardinal.UNKNOWN:
        return None
    
    if v1.approach_direction == v2.approach_direction:
        return None  # Valid
    
    pair = (v1.approach_direction, v2.approach_direction)
    if pair in OPPOSITE_PAIRS:
        return GeometricError(
            error_type="constraint_violation",
            severity="error",
            message=(
                f"same_approach_as(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
                f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
                f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
                f"These are OPPOSITE directions."
            ),
            affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
            affected_constraint="same_approach_as",
            suggested_fixes=[
                f"Change to opposite_approach_of",
                f"Change V{v2.vehicle_id} to head {v1.approach_direction.value}",
            ],
        )
    
    return GeometricError(
        error_type="constraint_violation",
        severity="error",
        message=(
            f"same_approach_as(V{v1.vehicle_id}, V{v2.vehicle_id}) is geometrically invalid. "
            f"V{v1.vehicle_id} is heading {v1.approach_direction.value}, "
            f"V{v2.vehicle_id} is heading {v2.approach_direction.value}. "
            f"These are PERPENDICULAR."
        ),
        affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
        affected_constraint="same_approach_as",
        suggested_fixes=[
            f"Change to perpendicular_left_of or perpendicular_right_of",
            f"Change V{v2.vehicle_id} to head {v1.approach_direction.value}",
        ],
    )


def check_maneuver_exit_consistency(v: VehicleIR) -> Optional[GeometricError]:
    """
    Check that maneuver matches stated exit direction.
    
    If heading (approach_direction) and maneuver are known, exit should be deterministic.
    If exit is explicitly stated and doesn't match, that's an error.
    
    Convention: approach_direction = heading direction
    """
    if v.approach_direction == Cardinal.UNKNOWN:
        return None
    if v.maneuver in ("unknown", "lane_change"):
        return None
    if v.exit_direction == Cardinal.UNKNOWN:
        return None
    
    expected_exit = v.compute_exit()
    if expected_exit == Cardinal.UNKNOWN:
        return None
    
    if v.exit_direction != expected_exit:
        return GeometricError(
            error_type="maneuver_mismatch",
            severity="error",
            message=(
                f"Vehicle {v.vehicle_id}: heading {v.approach_direction.value} + "
                f"{v.maneuver} turn should exit {expected_exit.value}, "
                f"but text says exit is {v.exit_direction.value}."
            ),
            affected_vehicles=[v.vehicle_id],
            suggested_fixes=[
                f"Change exit to {expected_exit.value}",
                f"Change maneuver to match the stated exit direction",
                f"Clarify the heading direction",
            ],
        )
    
    return None


def check_left_right_of_perpendicular_correctness(
    v1: VehicleIR,
    v2: VehicleIR,
    constraint: ConstraintIR,
) -> Optional[GeometricError]:
    """
    Check that perpendicular_left_of and perpendicular_right_of are used correctly.
    
    This is more nuanced - it checks if the left/right designation matches the geometry.
    
    Convention: approach_direction = heading direction (where vehicle is GOING)
    """
    if v1.approach_direction == Cardinal.UNKNOWN or v2.approach_direction == Cardinal.UNKNOWN:
        return None
    
    # First verify they are perpendicular (based on heading directions)
    pair = (v1.approach_direction, v2.approach_direction)
    if pair not in PERPENDICULAR_PAIRS:
        return None  # Let the perpendicular check catch this
    
    # Now check left vs right
    # Convention: "V1 is perpendicular_right_of V2" means V1 approaches from V2's right
    # V2's heading direction IS its approach_direction
    # We need to determine which road (origin) V1 comes from relative to V2's perspective
    
    v2_heading = v2.approach_direction  # This is where V2 is GOING
    v1_origin = v1.origin_cardinal()     # This is where V1 is coming FROM
    
    # From V2's perspective (facing its heading direction),
    # what is on V2's right/left?
    # Right side: 90° clockwise from heading
    # Left side: 90° counter-clockwise from heading
    
    # When V2 is heading NORTH, V2's right is EAST, left is WEST
    # When V2 is heading SOUTH, V2's right is WEST, left is EAST
    # When V2 is heading EAST, V2's right is SOUTH, left is NORTH
    # When V2 is heading WEST, V2's right is NORTH, left is SOUTH
    
    right_of_heading = {
        Cardinal.NORTH: Cardinal.EAST,
        Cardinal.SOUTH: Cardinal.WEST,
        Cardinal.EAST: Cardinal.SOUTH,
        Cardinal.WEST: Cardinal.NORTH,
    }
    
    left_of_heading = {
        Cardinal.NORTH: Cardinal.WEST,
        Cardinal.SOUTH: Cardinal.EAST,
        Cardinal.EAST: Cardinal.NORTH,
        Cardinal.WEST: Cardinal.SOUTH,
    }
    
    if constraint.constraint_type == "perpendicular_right_of":
        # V1 should come FROM V2's right side direction
        expected_v1_origin = right_of_heading.get(v2_heading)
        if expected_v1_origin and v1_origin != expected_v1_origin:
            actual_side = "left" if v1_origin == left_of_heading.get(v2_heading) else "unknown"
            return GeometricError(
                error_type="constraint_direction_mismatch",
                severity="warning",
                message=(
                    f"perpendicular_right_of(V{v1.vehicle_id}, V{v2.vehicle_id}) may be incorrect. "
                    f"V{v2.vehicle_id} is heading {v2_heading.value}, so its RIGHT is the {expected_v1_origin.value} road. "
                    f"But V{v1.vehicle_id} comes from {v1_origin.value} (V{v2.vehicle_id}'s {actual_side})."
                ),
                affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
                affected_constraint="perpendicular_right_of",
                suggested_fixes=[
                    f"Change to perpendicular_left_of if V{v1.vehicle_id} is on V{v2.vehicle_id}'s left",
                    f"Change V{v1.vehicle_id}'s approach to come from {expected_v1_origin.value} for true right-of",
                ],
            )
    
    elif constraint.constraint_type == "perpendicular_left_of":
        expected_v1_origin = left_of_heading.get(v2_heading)
        if expected_v1_origin and v1_origin != expected_v1_origin:
            actual_side = "right" if v1_origin == right_of_heading.get(v2_heading) else "unknown"
            return GeometricError(
                error_type="constraint_direction_mismatch",
                severity="warning",
                message=(
                    f"perpendicular_left_of(V{v1.vehicle_id}, V{v2.vehicle_id}) may be incorrect. "
                    f"V{v2.vehicle_id} is heading {v2_heading.value}, so its LEFT is the {expected_v1_origin.value} road. "
                    f"But V{v1.vehicle_id} comes from {v1_origin.value} (V{v2.vehicle_id}'s {actual_side})."
                ),
                affected_vehicles=[v1.vehicle_id, v2.vehicle_id],
                affected_constraint="perpendicular_left_of",
                suggested_fixes=[
                    f"Change to perpendicular_right_of if V{v1.vehicle_id} is on V{v2.vehicle_id}'s right",
                    f"Change V{v1.vehicle_id}'s approach to come from {expected_v1_origin.value} for true left-of",
                ],
            )
    
    return None


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_geometric_consistency(ir: ScenarioIR) -> GeometricValidationResult:
    """
    Validate the geometric consistency of a scenario IR.
    
    This applies deterministic rules that CANNOT be wrong:
    - If vehicles claim to follow each other, they must go the same direction
    - If vehicles claim opposite approach, they must be on opposite cardinals
    - If a vehicle turns left from north, it exits east (deterministic)
    
    Returns:
        GeometricValidationResult with errors and warnings
    """
    result = GeometricValidationResult(is_valid=True)
    
    # Check each constraint
    for constraint in ir.constraints:
        v1 = ir.get_vehicle(constraint.vehicle1_id)
        v2 = ir.get_vehicle(constraint.vehicle2_id)
        
        if not v1 or not v2:
            result.warnings.append(GeometricError(
                error_type="missing_vehicle",
                severity="warning",
                message=f"Constraint {constraint.constraint_type} references missing vehicle",
                affected_constraint=constraint.constraint_type,
            ))
            continue
        
        error = None
        if constraint.constraint_type == "follows_route_of":
            error = check_follows_route_of(v1, v2, constraint)
        elif constraint.constraint_type == "opposite_approach_of":
            error = check_opposite_approach_of(v1, v2, constraint)
        elif constraint.constraint_type in ("perpendicular_left_of", "perpendicular_right_of"):
            error = check_perpendicular_of(v1, v2, constraint)
            if not error:
                # Also check left/right correctness
                error = check_left_right_of_perpendicular_correctness(v1, v2, constraint)
        elif constraint.constraint_type == "same_approach_as":
            error = check_same_approach_as(v1, v2, constraint)
        
        if error:
            if error.severity == "error":
                result.errors.append(error)
                result.is_valid = False
            else:
                result.warnings.append(error)
    
    # Check maneuver-exit consistency for each vehicle
    for v in ir.vehicles:
        error = check_maneuver_exit_consistency(v)
        if error:
            if error.severity == "error":
                result.errors.append(error)
                result.is_valid = False
            else:
                result.warnings.append(error)
    
    # Check for vehicles with conflicting constraints
    # e.g., V1 follows V2 AND V1 opposite of V2 (impossible)
    constraint_pairs = {}
    for c in ir.constraints:
        pair = (min(c.vehicle1_id, c.vehicle2_id), max(c.vehicle1_id, c.vehicle2_id))
        if pair not in constraint_pairs:
            constraint_pairs[pair] = []
        constraint_pairs[pair].append(c.constraint_type)
    
    for pair, types in constraint_pairs.items():
        if "follows_route_of" in types and "opposite_approach_of" in types:
            result.errors.append(GeometricError(
                error_type="conflicting_constraints",
                severity="error",
                message=(
                    f"V{pair[0]} and V{pair[1]} have conflicting constraints: "
                    f"follows_route_of (requires same direction) AND opposite_approach_of (requires opposite). "
                    f"These are mutually exclusive."
                ),
                affected_vehicles=list(pair),
                suggested_fixes=["Remove one of the conflicting constraints"],
            ))
            result.is_valid = False
        
        if "same_approach_as" in types and "opposite_approach_of" in types:
            result.errors.append(GeometricError(
                error_type="conflicting_constraints",
                severity="error",
                message=(
                    f"V{pair[0]} and V{pair[1]} have conflicting constraints: "
                    f"same_approach_as AND opposite_approach_of. These are mutually exclusive."
                ),
                affected_vehicles=list(pair),
                suggested_fixes=["Remove one of the conflicting constraints"],
            ))
            result.is_valid = False
    
    return result


def build_geometric_feedback(
    ir: ScenarioIR,
    validation: GeometricValidationResult,
) -> str:
    """
    Build a formatted feedback string for the critic prompt.
    
    This shows the LLM:
    1. How its text was interpreted (the IR)
    2. What geometric errors were found
    3. Specific fixes to apply
    """
    lines = []
    
    # Show the extracted IR
    lines.append("=" * 60)
    lines.append("EXTRACTED INTERPRETATION:")
    lines.append("=" * 60)
    lines.append(ir.to_summary_string())
    lines.append("")
    
    # Show geometric validation results
    lines.append("=" * 60)
    lines.append("GEOMETRIC CONSISTENCY CHECK:")
    lines.append("=" * 60)
    lines.append(validation.to_string())
    
    return "\n".join(lines)
