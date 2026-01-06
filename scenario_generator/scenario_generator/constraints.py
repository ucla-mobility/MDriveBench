"""
Scenario Constraints - Structured IR Aligned with Pipeline

This module provides structured representations that map DIRECTLY to what
the pipeline can parse and execute. No aspirational features.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import re

from .capabilities import (
    TopologyType, ActorKind, MotionType, SpeedHint, TimingPhase,
    LateralPosition, GroupPattern, ConstraintType, EgoManeuver,
    CATEGORY_FEASIBILITY, get_feasible_categories,
)


# =============================================================================
# EGO VEHICLE SPECIFICATION
# =============================================================================
# Maps directly to what step_03_path_picker/constraints.py can parse

@dataclass
class EgoVehicleSpec:
    """
    Specification for an ego vehicle.
    
    These map directly to the 'vehicles' array in the path picker constraints.
    """
    vehicle_id: str  # "Vehicle 1", "Vehicle 2", etc.
    maneuver: EgoManeuver = EgoManeuver.STRAIGHT
    lane_change_phase: str = "unknown"  # "before_intersection", "after_intersection", "unknown"
    entry_road: str = "unknown"  # "main", "side", "unknown"
    exit_road: str = "unknown"   # "main", "side", "unknown"


@dataclass
class InterVehicleConstraint:
    """
    Constraint between two ego vehicles.
    
    Maps directly to the 'constraints' array in the path picker.
    """
    constraint_type: ConstraintType
    vehicle_a: str  # "Vehicle 1"
    vehicle_b: str  # "Vehicle 2"
    
    def to_natural_language(self) -> str:
        """Convert to natural language for scenario description."""
        mapping = {
            ConstraintType.SAME_APPROACH_AS: f"{self.vehicle_a} approaches from the same direction as {self.vehicle_b}",
            ConstraintType.OPPOSITE_APPROACH_OF: f"{self.vehicle_a} approaches from the opposite direction of {self.vehicle_b}",
            ConstraintType.PERPENDICULAR_RIGHT_OF: f"{self.vehicle_a} approaches from the perpendicular road to the right of {self.vehicle_b}'s approach",
            ConstraintType.PERPENDICULAR_LEFT_OF: f"{self.vehicle_a} approaches from the perpendicular road to the left of {self.vehicle_b}'s approach",
            ConstraintType.SAME_EXIT_AS: f"{self.vehicle_a} exits in the same direction as {self.vehicle_b}",
            ConstraintType.SAME_ROAD_AS: f"{self.vehicle_a} ends up on the same road as {self.vehicle_b}",
            ConstraintType.FOLLOW_ROUTE_OF: f"{self.vehicle_a} follows behind {self.vehicle_b}",
            ConstraintType.LEFT_LANE_OF: f"{self.vehicle_a} is in the lane to the left of {self.vehicle_b}",
            ConstraintType.RIGHT_LANE_OF: f"{self.vehicle_a} is in the lane to the right of {self.vehicle_b}",
            ConstraintType.MERGES_INTO_LANE_OF: f"{self.vehicle_a} changes lanes into {self.vehicle_b}'s lane",
        }
        return mapping.get(self.constraint_type, f"{self.constraint_type.value}({self.vehicle_a}, {self.vehicle_b})")


# =============================================================================
# NON-EGO ACTOR SPECIFICATION  
# =============================================================================
# Maps directly to what step_05_object_placer/prompts.py Stage 1 can parse

@dataclass
class NonEgoActorSpec:
    """
    Specification for a non-ego actor (obstacle, pedestrian, etc.).
    
    Maps directly to the entity schema in object placer Stage 1.
    """
    # Identity
    actor_id: str  # e.g., "parked truck", "pedestrian", "traffic cones"
    kind: ActorKind
    quantity: int = 1
    
    # Spatial placement (relative to an ego vehicle's path)
    affects_vehicle: Optional[str] = None  # "Vehicle 1", etc.
    timing_phase: TimingPhase = TimingPhase.UNKNOWN
    lateral_position: LateralPosition = LateralPosition.CENTER
    
    # For groups (quantity > 1)
    group_pattern: GroupPattern = GroupPattern.UNKNOWN
    start_lateral: Optional[LateralPosition] = None
    end_lateral: Optional[LateralPosition] = None
    
    # Motion
    motion: MotionType = MotionType.STATIC
    speed: SpeedHint = SpeedHint.UNKNOWN
    crossing_direction: Optional[str] = None  # "left" or "right" for crossing motion
    
    # For NPC vehicles only
    direction_relative_to: Optional[str] = None  # "same" or "opposite"
    
    def to_natural_language(self) -> str:
        """Convert to natural language fragment for scenario description."""
        parts = []
        
        # Quantity and type
        if self.quantity > 1:
            parts.append(f"{self.quantity} {self.actor_id}")
        else:
            parts.append(f"a {self.actor_id}")
        
        # Position relative to vehicle
        if self.affects_vehicle:
            if self.timing_phase == TimingPhase.ON_APPROACH:
                parts.append(f"on {self.affects_vehicle}'s approach")
            elif self.timing_phase == TimingPhase.AFTER_TURN:
                parts.append(f"after {self.affects_vehicle}'s turn")
            elif self.timing_phase == TimingPhase.IN_INTERSECTION:
                parts.append(f"in the intersection")
            elif self.timing_phase == TimingPhase.AFTER_EXIT:
                parts.append(f"after {self.affects_vehicle} exits")
            elif self.timing_phase == TimingPhase.AFTER_MERGE:
                parts.append(f"after the merge region")
        
        # Lateral position
        if self.lateral_position != LateralPosition.CENTER:
            pos_map = {
                LateralPosition.HALF_RIGHT: "positioned to the right of center",
                LateralPosition.RIGHT_EDGE: "at the right edge of the lane",
                LateralPosition.OFFROAD_RIGHT: "off the right side of the road",
                LateralPosition.HALF_LEFT: "positioned to the left of center", 
                LateralPosition.LEFT_EDGE: "at the left edge of the lane",
                LateralPosition.OFFROAD_LEFT: "off the left side of the road",
            }
            parts.append(pos_map.get(self.lateral_position, ""))
        
        # Motion
        if self.motion == MotionType.CROSS_PERPENDICULAR:
            dir_str = f"from {'right to left' if self.crossing_direction == 'left' else 'left to right'}" if self.crossing_direction else ""
            parts.append(f"crossing the road {dir_str}")
        elif self.motion == MotionType.FOLLOW_LANE:
            parts.append("moving along the lane")
        
        return " ".join(filter(None, parts))


# =============================================================================
# COMPLETE SCENARIO SPECIFICATION
# =============================================================================

@dataclass
class ScenarioSpec:
    """
    Complete specification for a scenario.
    
    This is the structured IR that:
    1. Can be validated against pipeline capabilities
    2. Can be converted to natural language for the pipeline
    3. Can be generated creatively within bounds
    """
    # Metadata
    category: str
    difficulty: int  # 1-5
    
    # Map requirements
    topology: TopologyType
    needs_oncoming: bool = False
    needs_multi_lane: bool = False
    needs_on_ramp: bool = False
    needs_merge: bool = False
    
    # Ego vehicles and their relationships
    ego_vehicles: List[EgoVehicleSpec] = field(default_factory=list)
    vehicle_constraints: List[InterVehicleConstraint] = field(default_factory=list)
    
    # Non-ego actors
    actors: List[NonEgoActorSpec] = field(default_factory=list)
    
    # Natural language output (generated from above)
    description: str = ""
    
    def generate_description(self) -> str:
        """
        Generate a natural language description from the structured spec.
        This produces text that the pipeline can parse.
        """
        parts = []
        
        # Describe each ego vehicle
        for i, veh in enumerate(self.ego_vehicles):
            veh_parts = [f"{veh.vehicle_id}"]
            
            # Entry road context
            if veh.entry_road == "main":
                veh_parts.append("travels along the main road")
            elif veh.entry_road == "side":
                veh_parts.append("approaches from the side road")
            else:
                veh_parts.append("travels")
            
            # Maneuver
            if veh.maneuver == EgoManeuver.LEFT:
                veh_parts.append("and intends to turn left")
            elif veh.maneuver == EgoManeuver.RIGHT:
                veh_parts.append("and intends to turn right")
            elif veh.maneuver == EgoManeuver.LANE_CHANGE:
                if veh.lane_change_phase == "before_intersection":
                    veh_parts.append("and changes lanes before the intersection")
                elif veh.lane_change_phase == "after_intersection":
                    veh_parts.append("and changes lanes after the intersection")
                else:
                    veh_parts.append("and changes lanes")
            else:
                veh_parts.append("straight")
            
            parts.append(" ".join(veh_parts) + ".")
        
        # Describe vehicle relationships via constraints
        for constraint in self.vehicle_constraints:
            parts.append(constraint.to_natural_language() + ".")
        
        # Describe non-ego actors
        for actor in self.actors:
            parts.append(actor.to_natural_language() + ".")
        
        # Add conflict context for the scenario
        parts.append(self._describe_conflict())
        
        self.description = " ".join(parts)
        return self.description
    
    def _describe_conflict(self) -> str:
        """Generate conflict description based on scenario type."""
        if len(self.ego_vehicles) >= 2:
            if self.needs_on_ramp or self.needs_merge:
                return "The vehicles reach the merge region with overlapping timing, creating an ambiguous merge interaction."
            elif self.needs_oncoming:
                return "Their arrival windows overlap, creating ambiguous priority negotiation."
            elif self.needs_multi_lane:
                return "The lane positioning creates ambiguous negotiation for available gaps."
            else:
                return "Their paths interact, creating coordination requirements."
        return ""


# =============================================================================
# VALIDATION
# =============================================================================

def validate_spec(spec: ScenarioSpec) -> Tuple[bool, List[str]]:
    """
    Validate that a scenario spec is expressible by the pipeline.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Check category is feasible
    if spec.category not in get_feasible_categories():
        errors.append(f"Category '{spec.category}' is not feasible with current pipeline")
    
    # Check difficulty range
    if not 1 <= spec.difficulty <= 5:
        errors.append(f"Difficulty must be 1-5, got {spec.difficulty}")
    
    # Check topology is supported
    if spec.topology not in {TopologyType.INTERSECTION, TopologyType.T_JUNCTION, TopologyType.CORRIDOR}:
        errors.append(f"Topology '{spec.topology.value}' is not supported")
    
    # Check ego vehicle count
    if len(spec.ego_vehicles) < 1:
        errors.append("Must have at least 1 ego vehicle")
    # No upper limit on ego vehicles - the pipeline can handle any reasonable number
    
    # Check actor count
    if len(spec.actors) > 8:
        errors.append(f"Too many non-ego actors: {len(spec.actors)} > 8")
    
    # Validate ego vehicle specs
    for veh in spec.ego_vehicles:
        if not re.match(r"Vehicle \d+", veh.vehicle_id):
            errors.append(f"Invalid vehicle_id format: {veh.vehicle_id}")
    
    # Validate constraints reference valid vehicles
    valid_ids = {v.vehicle_id for v in spec.ego_vehicles}
    for c in spec.vehicle_constraints:
        if c.vehicle_a not in valid_ids:
            errors.append(f"Constraint references unknown vehicle: {c.vehicle_a}")
        if c.vehicle_b not in valid_ids:
            errors.append(f"Constraint references unknown vehicle: {c.vehicle_b}")
    
    # Validate actor specs
    for actor in spec.actors:
        if actor.motion == MotionType.CROSS_PERPENDICULAR and actor.kind not in {ActorKind.WALKER, ActorKind.CYCLIST}:
            errors.append(f"Cross perpendicular motion only valid for walkers/cyclists, not {actor.kind.value}")
    
    # Check that required features are consistent
    cat = CATEGORY_FEASIBILITY.get(spec.category)
    if cat:
        if cat.needs_oncoming and not spec.needs_oncoming:
            errors.append(f"Category {spec.category} requires oncoming traffic")
        if cat.needs_multi_lane and not spec.needs_multi_lane:
            errors.append(f"Category {spec.category} requires multi-lane")
        if cat.needs_on_ramp and not spec.needs_on_ramp:
            errors.append(f"Category {spec.category} requires on-ramp")
    
    return (len(errors) == 0, errors)


def spec_to_dict(spec: ScenarioSpec) -> Dict[str, Any]:
    """Convert spec to dictionary for JSON serialization."""
    return {
        "category": spec.category,
        "difficulty": spec.difficulty,
        "topology": spec.topology.value,
        "needs_oncoming": spec.needs_oncoming,
        "needs_multi_lane": spec.needs_multi_lane,
        "needs_on_ramp": spec.needs_on_ramp,
        "needs_merge": spec.needs_merge,
        "ego_vehicles": [
            {
                "vehicle_id": v.vehicle_id,
                "maneuver": v.maneuver.value,
                "lane_change_phase": v.lane_change_phase,
                "entry_road": v.entry_road,
                "exit_road": v.exit_road,
            }
            for v in spec.ego_vehicles
        ],
        "vehicle_constraints": [
            {
                "type": c.constraint_type.value,
                "a": c.vehicle_a,
                "b": c.vehicle_b,
            }
            for c in spec.vehicle_constraints
        ],
        "actors": [
            {
                "actor_id": a.actor_id,
                "kind": a.kind.value,
                "quantity": a.quantity,
                "affects_vehicle": a.affects_vehicle,
                "timing_phase": a.timing_phase.value,
                "lateral_position": a.lateral_position.value,
                "motion": a.motion.value,
                "speed": a.speed.value,
            }
            for a in spec.actors
        ],
        "description": spec.description,
    }


def spec_from_dict(d: Dict[str, Any]) -> ScenarioSpec:
    """Parse spec from dictionary."""
    ego_vehicles = [
        EgoVehicleSpec(
            vehicle_id=v["vehicle_id"],
            maneuver=EgoManeuver(v.get("maneuver", "straight")),
            lane_change_phase=v.get("lane_change_phase", "unknown"),
            entry_road=v.get("entry_road", "unknown"),
            exit_road=v.get("exit_road", "unknown"),
        )
        for v in d.get("ego_vehicles", [])
    ]
    
    vehicle_constraints = [
        InterVehicleConstraint(
            constraint_type=ConstraintType(c["type"]),
            vehicle_a=c["a"],
            vehicle_b=c["b"],
        )
        for c in d.get("vehicle_constraints", [])
    ]
    
    actors = [
        NonEgoActorSpec(
            actor_id=a["actor_id"],
            kind=ActorKind(a["kind"]),
            quantity=a.get("quantity", 1),
            affects_vehicle=a.get("affects_vehicle"),
            timing_phase=TimingPhase(a.get("timing_phase", "unknown")),
            lateral_position=LateralPosition(a.get("lateral_position", "center")),
            motion=MotionType(a.get("motion", "static")),
            speed=SpeedHint(a.get("speed", "unknown")),
        )
        for a in d.get("actors", [])
    ]
    
    return ScenarioSpec(
        category=d["category"],
        difficulty=d["difficulty"],
        topology=TopologyType(d["topology"]),
        needs_oncoming=d.get("needs_oncoming", False),
        needs_multi_lane=d.get("needs_multi_lane", False),
        needs_on_ramp=d.get("needs_on_ramp", False),
        needs_merge=d.get("needs_merge", False),
        ego_vehicles=ego_vehicles,
        vehicle_constraints=vehicle_constraints,
        actors=actors,
        description=d.get("description", ""),
    )
