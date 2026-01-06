"""
Pipeline Capabilities Definition - GROUND TRUTH

This module defines what the scenario generation pipeline CAN and CANNOT express.
These capabilities are derived from ACTUAL pipeline code analysis.

CRITICAL: This file must ONLY contain capabilities that exist in the actual code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


# =============================================================================
# ACTUAL TOPOLOGICAL FEATURES (from step_01_crop/models.py CropFeatures)
# =============================================================================
# These are the ONLY map features the pipeline can detect and match:

@dataclass
class MapFeatures:
    """Features actually detected by compute_crop_features() in features.py"""
    # Turns detected in the crop (from path signatures)
    turns: List[str]  # ["straight", "left", "right", "uturn"]
    
    # Cardinal directions of path entries/exits
    entry_dirs: List[str]  # ["N", "S", "E", "W"]
    exit_dirs: List[str]
    
    # Junction type booleans
    has_oncoming_pair: bool      # Two straight paths with opposite entry directions
    is_t_junction: bool          # len(dirs) == 3
    is_four_way: bool            # len(dirs) >= 4
    has_merge_onto_same_road: bool  # Multiple entry turns lead to same exit road
    has_on_ramp: bool            # has_merge AND NOT is_four_way (heuristic)
    
    # Lane features
    lane_count_est: int          # 1-3 typically
    has_multi_lane: bool         # lane_count_est >= 2


# =============================================================================
# ACTUAL GEOMETRY SPEC (from step_01_crop/models.py GeometrySpec)
# =============================================================================
# This is what the LLM extractor produces from a description:

class TopologyType(Enum):
    """ONLY these topology types exist in the pipeline."""
    INTERSECTION = "intersection"  # General intersection (3+ way)
    T_JUNCTION = "t_junction"      # Exactly 3 directions
    CORRIDOR = "corridor"          # Straight road segment
    UNKNOWN = "unknown"            # Fallback


@dataclass
class GeometrySpecActual:
    """The ACTUAL geometry spec from llm_extractor.py"""
    topology: TopologyType
    degree: int  # 0, 3, or 4 (0 = unknown)
    
    # Maneuver requirements (counts)
    required_maneuvers: Dict[str, int]  # {"straight": 0-3, "left": 0-3, "right": 0-3}
    
    # Feature requirements
    needs_oncoming: bool
    needs_merge_onto_same_road: bool
    needs_on_ramp: bool
    needs_multi_lane: bool
    min_lane_count: int  # 1-3
    
    # Path length requirements (meters)
    min_entry_runup_m: float   # Default 28.0
    min_exit_runout_m: float   # Default 18.0
    
    # Preferences
    preferred_entry_cardinals: List[str]  # ["N", "S", "E", "W"] or []
    avoid_extra_intersections: bool


# =============================================================================
# ACTUAL NON-EGO ACTOR CAPABILITIES (from step_05_object_placer/prompts.py)
# =============================================================================

class ActorKind(Enum):
    """Actor types the pipeline can spawn."""
    STATIC_PROP = "static_prop"        # cones, barriers, debris, boxes
    PARKED_VEHICLE = "parked_vehicle"  # stationary vehicles blocking lanes
    WALKER = "walker"                  # pedestrians
    CYCLIST = "cyclist"                # bicycles
    NPC_VEHICLE = "npc_vehicle"        # Simple NPC (NOT an ego with its own path)


class MotionType(Enum):
    """Motion types from guardrails.py validation."""
    STATIC = "static"                  # No movement
    CROSS_PERPENDICULAR = "cross_perpendicular"  # Crosses lane perpendicular
    FOLLOW_LANE = "follow_lane"        # Moves along lane direction
    STRAIGHT_LINE = "straight_line"    # Moves between two anchors


class SpeedHint(Enum):
    """Speed hints - qualitative only, NOT precise."""
    STOPPED = "stopped"
    SLOW = "slow"
    NORMAL = "normal"
    FAST = "fast"
    ERRATIC = "erratic"
    UNKNOWN = "unknown"


class TimingPhase(Enum):
    """When an actor appears relative to ego path."""
    ON_APPROACH = "on_approach"
    AFTER_TURN = "after_turn"
    IN_INTERSECTION = "in_intersection"
    AFTER_EXIT = "after_exit"
    AFTER_MERGE = "after_merge"
    UNKNOWN = "unknown"


class LateralPosition(Enum):
    """Lateral positions from constants.py LATERAL_RELATIONS."""
    CENTER = "center"
    HALF_RIGHT = "half_right"
    RIGHT_EDGE = "right_edge"
    OFFROAD_RIGHT = "offroad_right"
    HALF_LEFT = "half_left"
    LEFT_EDGE = "left_edge"
    OFFROAD_LEFT = "offroad_left"


class GroupPattern(Enum):
    """How multiple objects are arranged."""
    ACROSS_LANE = "across_lane"
    ALONG_LANE = "along_lane"
    DIAGONAL = "diagonal"
    UNKNOWN = "unknown"


class CrossingDirection(Enum):
    """For crossing motion only."""
    LEFT = "left"    # Crosses from right to left
    RIGHT = "right"  # Crosses from left to right


# =============================================================================
# ACTUAL EGO VEHICLE CONSTRAINTS (from step_03_path_picker/constraints.py)
# =============================================================================

class ConstraintType(Enum):
    """Inter-vehicle constraints that the path picker understands."""
    SAME_APPROACH_AS = "same_approach_as"
    OPPOSITE_APPROACH_OF = "opposite_approach_of"
    PERPENDICULAR_RIGHT_OF = "perpendicular_right_of"
    PERPENDICULAR_LEFT_OF = "perpendicular_left_of"
    SAME_EXIT_AS = "same_exit_as"
    SAME_ROAD_AS = "same_road_as"
    FOLLOW_ROUTE_OF = "follow_route_of"
    LEFT_LANE_OF = "left_lane_of"
    RIGHT_LANE_OF = "right_lane_of"
    MERGES_INTO_LANE_OF = "merges_into_lane_of"


class EgoManeuver(Enum):
    """Maneuvers the path picker recognizes."""
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"
    LANE_CHANGE = "lane_change"
    UNKNOWN = "unknown"


# =============================================================================
# AVAILABLE CARLA TOWNS (from birdview_v2_cache directory)
# =============================================================================

AVAILABLE_TOWNS = ["Town01", "Town02", "Town05", "Town06", "Town07"]


# =============================================================================
# GROUND TRUTH PIPELINE CAPABILITIES
# =============================================================================

@dataclass
class PipelineCapabilities:
    """
    Complete specification of what the pipeline CAN and CANNOT express.
    This is the GROUND TRUTH derived from actual code analysis.
    """
    
    # === EGO VEHICLES ===
    # No hard limit on ego vehicles - the pipeline supports any reasonable number
    # Practical considerations: more vehicles = more complex path negotiation
    min_ego_vehicles: int = 1
    
    # === NON-EGO ACTORS ===
    # No hard limit - practical considerations for spawn point availability
    actor_kinds: Set[ActorKind] = field(default_factory=lambda: set(ActorKind))
    motion_types: Set[MotionType] = field(default_factory=lambda: set(MotionType))
    
    # === TOPOLOGY MATCHING ===
    # The pipeline matches descriptions to existing map regions
    # It can detect: intersection, t_junction, corridor
    # It CANNOT create: roundabouts (no detection), signalized intersections
    supported_topologies: Set[TopologyType] = field(default_factory=lambda: {
        TopologyType.INTERSECTION,
        TopologyType.T_JUNCTION,
        TopologyType.CORRIDOR,
    })
    
    # Map feature detection
    can_detect_oncoming: bool = True
    can_detect_multi_lane: bool = True
    can_detect_on_ramp: bool = True  # Heuristic only
    can_detect_merge: bool = True
    can_detect_t_junction: bool = True
    can_detect_four_way: bool = True
    
    # === INTER-VEHICLE CONSTRAINTS ===
    constraint_types: Set[ConstraintType] = field(default_factory=lambda: set(ConstraintType))
    
    # === ACTOR PLACEMENT ===
    lateral_positions: Set[LateralPosition] = field(default_factory=lambda: set(LateralPosition))
    timing_phases: Set[TimingPhase] = field(default_factory=lambda: set(TimingPhase))
    s_along_range: Tuple[float, float] = (0.0, 1.0)  # Position along segment
    
    # === WHAT THE PIPELINE CANNOT DO ===
    hard_limitations: List[str] = field(default_factory=lambda: [
        # Map/Topology limitations
        "CANNOT create roundabouts - no roundabout detection in CropFeatures",
        "CANNOT create signalized intersections - no signal phase control",
        "CANNOT create highway diverge/off-ramps - only on-ramp heuristic exists",
        
        # Vehicle behavior limitations
        "CANNOT specify exact vehicle speeds in m/s or km/h - only qualitative hints",
        "CANNOT specify exact timing between vehicles in seconds",
        "CANNOT control NPC behavioral personality (aggressive, hesitant)",
        "CANNOT script dynamic reactions (brake when X happens)",
        "CANNOT specify exact headway distances between vehicles",
        "CANNOT create multi-stage scenarios (do X, then later do Y)",
        
        # Spatial limitations
        "CANNOT reference specific coordinates - only segment-relative positions",
        "CANNOT specify exact distances - only relative positions (s_along 0-1)",
        
        # Actor limitations
        "NPC vehicles are SIMPLE - spawn and move, no complex paths",
        "NPC vehicles do NOT get their own picked paths like ego vehicles",
    ])


# Singleton instance
PIPELINE_CAPABILITIES = PipelineCapabilities()


# =============================================================================
# CATEGORY FEASIBILITY ANALYSIS
# =============================================================================

@dataclass
class CategoryFeasibility:
    """
    Analysis of whether a scenario category is actually implementable.
    """
    name: str
    is_feasible: bool
    feasibility_notes: str
    
    # What map features are required
    required_topology: TopologyType
    needs_oncoming: bool = False
    needs_multi_lane: bool = False
    needs_on_ramp: bool = False
    needs_merge: bool = False
    
    # Non-ego actors usage (now difficulty-dependent)
    uses_non_ego_actors: bool = False
    non_ego_actors_min_difficulty: int = 1  # Minimum difficulty to include non-ego actors (1-5, default 1 = always)
    
    # Conflict creation mechanisms
    conflict_via: List[str] = field(default_factory=list)
    
    # Difficulty scaling mechanisms that actually work
    difficulty_knobs: List[str] = field(default_factory=list)
    
    # Creative variation axes for diverse scenario generation
    variation_axes: List[str] = field(default_factory=list)


# Honest assessment of each category
CATEGORY_FEASIBILITY: Dict[str, CategoryFeasibility] = {
    
    "Highway On-Ramp Merge": CategoryFeasibility(
        name="Highway On-Ramp Merge",
        is_feasible=True,
        feasibility_notes="Uses merge_onto_same_road + multi_lane geometry for on-ramp-like scenarios. The strict has_on_ramp heuristic is too restrictive, so we use more common merge geometry instead.",
        required_topology=TopologyType.CORRIDOR,
        needs_on_ramp=False,  # Heuristic is too strict; use needs_merge_onto_same_road instead
        needs_merge=True,     # Mark as needing merge, but actual spec uses merge_onto_same_road
        uses_non_ego_actors=True,
        non_ego_actors_min_difficulty=1,
        conflict_via=[
            "Multiple ego vehicles with paths converging at merge",
            "follow_route_of constraint for queued vehicles",
            "same_road_as constraint for merge target",
            "Parked vehicle or obstacle blocking lanes increases merge difficulty",
        ],
        difficulty_knobs=[
            "Number of ego vehicles",
            "Constraint combinations",
            "Obstacle placement severity",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vs 6+ vehicles",
            "ramp_queue: single merging vehicle vs follow_route_of chain of 2-3",
            "highway_queue: single mainline vehicle vs follow_route_of chain of 2-4",
            "lane_positions: same lane competition vs adjacent lanes (left_lane_of/right_lane_of)",
            "merge_type: single merges_into_lane_of vs multiple competing merges",
            "obstacle: none vs parked_vehicle vs static_prop on merge approach",
            "obstacle_position: entrance vs merge point vs exit",
        ],
    ),
    
    "Lane Drop Merge (Zipper)": CategoryFeasibility(
        name="Lane Drop Merge (Zipper)",
        is_feasible=True,
        feasibility_notes="Works if map has lane drop region with multi-lane approach. Can include construction cones or parked vehicles to further constrict lanes.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        needs_merge=True,
        uses_non_ego_actors=True,
        non_ego_actors_min_difficulty=2,  # Only include props at difficulty 2+
        conflict_via=[
            "Ego vehicles in adjacent lanes reaching taper",
            "left_lane_of / right_lane_of constraints",
            "merges_into_lane_of for lane change intent",
            "Static props (cones) narrowing available merge space",
        ],
        difficulty_knobs=[
            "Number of vehicles per lane",
            "Taper severity from props",
        ],
        variation_axes=[
            "ego_count: 2-8 vehicles",
            "lane_distribution: all in one lane vs spread across lanes",
            "constraint_pattern: alternating left_lane_of/right_lane_of vs chains",
            "merge_targets: single merges_into_lane_of vs multiple",
            "work_zone_complexity: none vs single_side_channelization vs dual_side_bottleneck",
            "  single_side_channelization: cones create 3-4 cone line on left edge narrowing left lane to 50%",
            "  dual_side_bottleneck (harder): cones on BOTH sides (left edge AND right edge of roadway) creating 1-lane bottleneck forcing extreme merging",
            "parked_vehicle: none vs parked_vehicle in work zone (blocks one merge gap)",
        ],
    ),
    
    "Roundabout Entry": CategoryFeasibility(
        name="Roundabout Entry",
        is_feasible=False,
        feasibility_notes="NOT FEASIBLE - no roundabout detection in CropFeatures.",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=False,
        conflict_via=[],
        difficulty_knobs=[],
        variation_axes=[],
    ),
    
    "Courtesy & Deadlock Negotiation": CategoryFeasibility(
        name="Courtesy & Deadlock Negotiation",
        is_feasible=True,
        feasibility_notes="Works at uncontrolled intersections with perpendicular approaches - HIGHLY MULTI-AGENT",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "perpendicular_right_of / perpendicular_left_of constraints creating ambiguous right-of-way",
            "opposite_approach_of for oncoming standoff",
            "Paths crossing at junction center with no clear priority",
            "Multiple approach directions creating deadlock potential",
        ],
        difficulty_knobs=[
            "Number of approach directions with vehicles",
            "Mix of maneuvers creating crossing conflicts",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vs 6 vehicles from different approaches",
            "approach_pattern: 2-way standoff vs 3-way deadlock vs 4-way gridlock",
            "constraint_web: chain of perpendicular constraints vs mixed perpendicular+opposite",
            "maneuver_clash: all straight vs mix where turns cross each other's paths",
            "complication: none vs walker with cross_perpendicular vs parked_vehicle occlusion",
            "occlusion_position: half_right vs right_edge limiting visibility",
        ],
    ),
    
    "Unprotected Left Turn": CategoryFeasibility(
        name="Unprotected Left Turn",
        is_feasible=True,
        feasibility_notes="Works at intersections with oncoming traffic - classic multi-agent coordination",
        required_topology=TopologyType.INTERSECTION,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "Left-turning ego vs opposite_approach_of oncoming ego chain",
            "Multiple left turners competing for same gap",
            "Pedestrian actor crossing exit segment creates additional conflict",
            "Opposing left turners (same_exit_as) blocking each other",
        ],
        difficulty_knobs=[
            "Number of oncoming vehicles",
            "Number of left turners",
            "Actor combinations",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vs 6 vehicles",
            "oncoming_depth: single oncoming vs follow_route_of chain of 3-4",
            "left_turners: single vs 2-3 with same_approach_as forming queue",
            "opposing_conflict: none vs opposite left turner (same_exit_as clash)",
            "pedestrian: none vs walker cross_perpendicular on exit leg",
            "pedestrian_side: crossing from left vs from right",
            "occlusion: none vs parked_vehicle limiting visibility of pedestrian",
        ],
    ),
    
    "Lane Change Negotiation": CategoryFeasibility(
        name="Lane Change Negotiation",
        is_feasible=True,
        feasibility_notes="Works on multi-lane corridors - multiple vehicles competing for same lane. Can include occlusions to block visibility of neighboring vehicles.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "left_lane_of / right_lane_of constraints positioning vehicles",
            "merges_into_lane_of creating lane change intent conflict",
            "Multiple vehicles trying to merge into same target lane",
            "follow_route_of chains in each lane creating queues",
            "Parked vehicle or obstacle blocking a lane creates forced merge",
        ],
        difficulty_knobs=[
            "Vehicles in target lane",
            "Simultaneous lane change intentions",
            "Occlusion/prop placement limiting visibility",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vs 6 vehicles",
            "lane_layout: 2-lane vs 3-lane (more lanes = more conflict)",
            "merge_conflict: one merges_into_lane_of vs 2 competing merges vs 3-way merge race",
            "queue_depth: no follow_route_of vs chains of 2-3 in each lane",
            "merge_direction: all merging same direction vs opposing merges (left meets right)",
            "occlusion: none vs parked_vehicle blocking view of adjacent vehicle",
            "obstacle: none vs static_prop in one lane forcing swerve",
        ],
    ),
    
    "Highway Weaving": CategoryFeasibility(
        name="Highway Weaving",
        is_feasible=True,
        feasibility_notes="Works for lane weaving on multi-lane corridors. Can include obstacles or parked vehicles to create additional complexity.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        non_ego_actors_min_difficulty=2,  # Start adding props at difficulty 2+
        conflict_via=[
            "Multiple lane change constraints",
            "Vehicles distributed across lanes",
            "Obstacles blocking lanes force adaptive weaving patterns",
        ],
        difficulty_knobs=[
            "Number of lanes and vehicles",
            "Obstacle count and positioning",
        ],
        variation_axes=[
            "ego_count: 3-8 vehicles",
            "lane_span: 2-lane vs 3-lane road",
            "constraint_density: sparse vs dense lane relationships",
            "merge_conflicts: single merges_into_lane_of vs crossing merges",
            "prop_arrangement: none vs linear_spread vs scattered_dense",
            "  linear_spread: 3-4 static_props in a diagonal line forcing coordinated weaving",
            "  scattered_dense: 6-8 static_props scattered randomly (difficulty 4+) forcing reactive navigation",
            "parked_vehicles: none vs 1-2 interspersed with props blocking merge gaps",
        ],
    ),
    
    "Overtaking on Two-Lane Road": CategoryFeasibility(
        name="Overtaking on Two-Lane Road",
        is_feasible=True,
        feasibility_notes="Works on multi-lane same-direction roads. Can include oncoming vehicles, obstacles, and pedestrians to increase complexity.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "follow_route_of for following vehicle",
            "left_lane_of for passing lane vehicle",
            "Oncoming traffic limits overtaking windows",
            "Obstacles in passing lane block overtake attempt",
        ],
        difficulty_knobs=[
            "Number of vehicles in each lane",
            "Oncoming traffic density",
            "Obstacle placement",
        ],
        variation_axes=[
            "ego_count: 2-5 vehicles",
            "follow_depth: single follow_route_of vs chain",
            "lane_occupancy: left_lane_of single vs multiple",
            "maneuver: lane_change included or not",
            "oncoming: none vs opposite_approach_of vehicle",
            "obstacle: none vs parked_vehicle blocking pass lane",
            "pedestrian: none vs walker crossing",
        ],
    ),
    
    "Opposing Traffic on Narrow Road": CategoryFeasibility(
        name="Opposing Traffic on Narrow Road",
        is_feasible=True,
        feasibility_notes="Works with oncoming traffic detection",
        required_topology=TopologyType.CORRIDOR,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "opposite_approach_of for oncoming ego vehicles",
            "parked_vehicle actor to create constriction",
        ],
        difficulty_knobs=[
            "Queue depth each direction",
            "Parked vehicle placement",
        ],
        variation_axes=[
            "ego_count: 2-6 vehicles",
            "direction_split: 1v1 vs 2v1 vs 3v2 etc",
            "follow_chains: none vs follow_route_of on each side",
            "parked_lateral: half_right vs right_edge vs offroad_right",
            "parked_s_along: 0.3 vs 0.5 vs 0.7 positioning",
        ],
    ),
    
    "Pedestrian Crosswalk": CategoryFeasibility(
        name="Pedestrian Crosswalk",
        is_feasible=True,
        feasibility_notes="Works with walker actors using crossing motion",
        required_topology=TopologyType.CORRIDOR,
        uses_non_ego_actors=True,
        conflict_via=[
            "walker actor with cross_perpendicular motion",
            "Timing via when=on_approach",
        ],
        difficulty_knobs=[
            "Number of pedestrians",
            "Occlusion actors",
        ],
        variation_axes=[
            "ego_count: 1-4 vehicles",
            "walker_count: 1 vs 2-3 with group pattern",
            "walker_motion: cross_perpendicular",
            "crossing_direction: left vs right",
            "timing_phase: on_approach vs in_intersection",
            "occlusion: none vs parked_vehicle",
            "parked_lateral: half_right vs right_edge",
        ],
    ),
    
    "Occluded Hazard": CategoryFeasibility(
        name="Occluded Hazard",
        is_feasible=True,
        feasibility_notes="Works with parked_vehicle actor creating occlusion",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "parked_vehicle actor positioned to occlude",
            "Crossing vehicle or walker revealed late",
        ],
        difficulty_knobs=[
            "Occluder size/position",
        ],
        variation_axes=[
            "ego_count: 2-4 vehicles",
            "occluder_type: parked_vehicle vs static_prop",
            "occluder_lateral: half_right vs right_edge vs offroad_right",
            "occluded_actor: another ego via perpendicular constraint vs walker",
            "walker_motion: cross_perpendicular if walker",
        ],
    ),
    
    "Blocked Lane (Obstacle)": CategoryFeasibility(
        name="Blocked Lane (Obstacle)",
        is_feasible=True,
        feasibility_notes="Works with parked_vehicle actor blocking lane",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "parked_vehicle actor blocking ego's lane",
            "Adjacent lane ego via left_lane_of / right_lane_of",
        ],
        difficulty_knobs=[
            "Number of blockages",
            "Adjacent lane vehicles",
        ],
        variation_axes=[
            "ego_count: 2-5 vehicles",
            "blockage_count: 1 vs 2-3 parked_vehicles",
            "blockage_lateral: center vs half_right",
            "blockage_s_along: spread along route",
            "lane_constraints: left_lane_of vs right_lane_of for adjacent traffic",
        ],
    ),
    
    "Construction Zone": CategoryFeasibility(
        name="Construction Zone",
        is_feasible=True,
        feasibility_notes="Works with static_prop actors (cones) creating taper",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "static_prop actors in diagonal pattern",
            "Lane merge constraints between ego vehicles",
        ],
        difficulty_knobs=[
            "Cone count and pattern",
            "Vehicles per lane",
        ],
        variation_axes=[
            "ego_count: 2-6 vehicles",
            "cone_count: 3 vs 6 vs 10 static_props",
            "cone_pattern: along_lane vs diagonal",
            "cone_lateral: series from center to right_edge",
            "lane_constraints: merges_into_lane_of for forced merge",
            "work_vehicle: none vs parked_vehicle",
        ],
    ),
    
    "Multi-Conflict Scenarios": CategoryFeasibility(
        name="Multi-Conflict Scenarios",
        is_feasible=True,
        feasibility_notes="HIGHLY MULTI-AGENT: Combines multiple conflict types - the most challenging scenarios",
        required_topology=TopologyType.INTERSECTION,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "Left turn + oncoming queue + crosswalk pedestrian simultaneously",
            "Occlusion blocks visibility of crossing traffic",
            "Multiple vehicles from perpendicular approaches + conflicting turns",
            "Pedestrian crossing adds third axis of conflict to vehicle negotiations",
        ],
        difficulty_knobs=[
            "Number of conflict types combined",
            "Depth of each conflict (queue lengths)",
        ],
        variation_axes=[
            "ego_count: 4 vs 5 vs 6 vehicles from multiple approaches",
            "conflict_layers: 2 conflict types vs 3 vs 4 overlapping conflicts",
            "constraint_web: perpendicular + opposite_approach + same_exit all present",
            "maneuver_chaos: left turners meeting right turners meeting straight-through",
            "actor_complication: walker crossing + parked_vehicle occlusion together",
            "pedestrian_timing: on_approach vs in_intersection creates different conflicts",
        ],
    ),
    
    "Side Street Entry": CategoryFeasibility(
        name="Side Street Entry",
        is_feasible=True,
        feasibility_notes="Works at T-junctions - classic yield scenario with queue. Parked vehicles can block view of oncoming traffic or limit merge space.",
        required_topology=TopologyType.T_JUNCTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "Side street ego turning onto main road full of traffic",
            "Main road follow_route_of chains creating gaps and queue",
            "Parked vehicles limiting visibility of main road traffic",
            "Pedestrians crossing exit path",
        ],
        difficulty_knobs=[
            "Main road traffic count",
            "Number of side street vehicles",
            "Occlusion/pedestrian complexity",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vehicles",
            "main_road_queue: single vs follow_route_of chain of 2-4",
            "side_street_queue: single vs follow_route_of chain of 2",
            "side_maneuver: left turn (harder gap) vs right turn",
            "constraint: same_road_as vs perpendicular_right_of",
            "parked_vehicle: none vs parked_vehicle at corner blocking sight",
            "pedestrian: none vs walker crossing exit path",
        ],
    ),
    
    "Emergency Vehicle Encounter": CategoryFeasibility(
        name="Emergency Vehicle Encounter",
        is_feasible=False,
        feasibility_notes="NOT FEASIBLE - no emergency vehicle behavior or priority logic.",
        required_topology=TopologyType.CORRIDOR,
        uses_non_ego_actors=False,
        conflict_via=[],
        difficulty_knobs=[],
        variation_axes=[],
    ),
    
    "Wide Turn Negotiation": CategoryFeasibility(
        name="Wide Turn Negotiation",
        is_feasible=True,
        feasibility_notes="Can specify turn maneuver for ego vehicles. Parked vehicles or pedestrians can complicate turn paths and sight lines.",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "Ego with turn maneuver",
            "Adjacent lane ego constraints",
            "Parked vehicles blocking turning paths",
            "Pedestrians crossing turn destination",
        ],
        difficulty_knobs=[
            "Adjacent traffic count",
            "Occlusion/pedestrian complexity",
        ],
        variation_axes=[
            "ego_count: 2-4 vehicles",
            "turn_type: left vs right maneuver",
            "lane_constraints: left_lane_of vs right_lane_of for adjacent",
            "approach_constraints: same_approach_as vs perpendicular",
            "parked_vehicle: none vs parked_vehicle at corner reducing sight lines",
            "pedestrian: none vs walker crossing turn destination",
        ],
    ),
    
    "Shared Left Turn Lane Conflict": CategoryFeasibility(
        name="Shared Left Turn Lane Conflict",
        is_feasible=True,
        feasibility_notes="Works at intersections with opposing left turns",
        required_topology=TopologyType.INTERSECTION,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "Two ego vehicles with left maneuver",
            "opposite_approach_of constraint",
            "same_exit_as targeting same side street",
        ],
        difficulty_knobs=[
            "Queue depth per direction",
        ],
        variation_axes=[
            "ego_count: 2-6 vehicles",
            "queue_pattern: symmetric vs asymmetric follow_route_of chains",
            "exit_conflict: same_exit_as vs different exits",
            "actor: none vs walker on destination leg",
            "walker_timing: on_approach vs after_turn",
        ],
    ),
    
    "Multi-Way Standoff": CategoryFeasibility(
        name="Multi-Way Standoff",
        is_feasible=True,
        feasibility_notes="HIGHLY MULTI-AGENT: 4+ vehicles arriving at intersection simultaneously with no clear priority",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "4-6 vehicles approaching from all directions simultaneously",
            "perpendicular_right_of/left_of creating circular priority ambiguity",
            "Multiple maneuvers that would cross each other's paths",
        ],
        difficulty_knobs=[
            "Number of vehicles (4-6)",
            "Maneuver conflicts (crossing paths)",
        ],
        variation_axes=[
            "ego_count: 4 vs 5 vs 6 vehicles",
            "approach_coverage: vehicles from 3 directions vs all 4 directions",
            "maneuver_conflict: all straight (crossing) vs mix of turns (complex crossing)",
            "constraint_pattern: chain of perpendicular_right_of vs mixed constraints",
            "complication: none vs walker in center vs parked_vehicle occlusion",
        ],
    ),
    
    "Platoon Merge Conflict": CategoryFeasibility(
        name="Platoon Merge Conflict",
        is_feasible=True,
        feasibility_notes="HIGHLY MULTI-AGENT: Two chains of following vehicles must merge into single lane",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        needs_merge=True,
        uses_non_ego_actors=False,
        conflict_via=[
            "Two follow_route_of chains in adjacent lanes",
            "merges_into_lane_of from both chains competing",
            "Lane drop forces interleaving of platoons",
        ],
        difficulty_knobs=[
            "Platoon length per lane",
            "Number of vehicles trying to merge",
        ],
        variation_axes=[
            "ego_count: 4 vs 5 vs 6 vs 7 vehicles total",
            "platoon_size: 2+2 vs 3+2 vs 3+3 split between lanes",
            "merge_pattern: leader merges vs follower merges vs both try",
            "lane_relationship: left_lane_of chain vs right_lane_of chain",
        ],
    ),
    
    "Narrow Passage Negotiation": CategoryFeasibility(
        name="Narrow Passage Negotiation",
        is_feasible=True,
        feasibility_notes="HIGHLY MULTI-AGENT: Opposing traffic queues must negotiate through single-lane bottleneck",
        required_topology=TopologyType.CORRIDOR,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "parked_vehicle creating single-lane constriction",
            "opposite_approach_of queues on each side",
            "follow_route_of chains creating depth",
        ],
        difficulty_knobs=[
            "Queue depth on each side",
            "Number of constrictions",
        ],
        variation_axes=[
            "ego_count: 4 vs 5 vs 6 vehicles",
            "queue_asymmetry: 2v2 vs 3v2 vs 3v3",
            "constriction: single parked_vehicle vs double parked_vehicles",
            "parked_position: same side vs alternating sides",
            "pedestrian: none vs walker crossing at constriction",
        ],
    ),
    
    "Parking Lot Maneuvers": CategoryFeasibility(
        name="Parking Lot Maneuvers",
        is_feasible=False,
        feasibility_notes="NOT FEASIBLE - no parking lot topology detection.",
        required_topology=TopologyType.CORRIDOR,
        uses_non_ego_actors=False,
        conflict_via=[],
        difficulty_knobs=[],
        variation_axes=[],
    ),
    
    "Highway Diverge / Off-Ramp Exit Negotiation": CategoryFeasibility(
        name="Highway Diverge / Off-Ramp Exit Negotiation",
        is_feasible=False,
        feasibility_notes="NOT FEASIBLE - only on-ramp detection exists (has_on_ramp).",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=False,
        conflict_via=[],
        difficulty_knobs=[],
        variation_axes=[],
    ),
}


def get_feasible_categories() -> List[str]:
    """Return only categories that are actually implementable."""
    return [name for name, cat in CATEGORY_FEASIBILITY.items() if cat.is_feasible]


def get_infeasible_categories() -> List[str]:
    """Return categories that cannot be implemented with current pipeline."""
    return [name for name, cat in CATEGORY_FEASIBILITY.items() if not cat.is_feasible]


def print_feasibility_report():
    """Print a human-readable feasibility report."""
    feasible = get_feasible_categories()
    infeasible = get_infeasible_categories()
    
    print("=" * 60)
    print("SCENARIO CATEGORY FEASIBILITY REPORT")
    print("=" * 60)
    
    print(f"\n✓ FEASIBLE ({len(feasible)} categories):")
    for name in feasible:
        cat = CATEGORY_FEASIBILITY[name]
        print(f"  - {name}")
        print(f"    {cat.feasibility_notes}")
    
    print(f"\n✗ NOT FEASIBLE ({len(infeasible)} categories):")
    for name in infeasible:
        cat = CATEGORY_FEASIBILITY[name]
        print(f"  - {name}")
        print(f"    {cat.feasibility_notes}")
    
    print("\n" + "=" * 60)
