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
    HIGHWAY = "highway"            # Multi-lane high-speed road (3+ lanes)
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
    SIDEWALK_RIGHT = "sidewalk_right"  # pedestrian on right sidewalk
    HALF_LEFT = "half_left"
    LEFT_EDGE = "left_edge"
    OFFROAD_LEFT = "offroad_left"
    SIDEWALK_LEFT = "sidewalk_left"    # pedestrian on left sidewalk


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
    SAME_LANE_AS = "same_lane_as"


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
    # It can detect: intersection, t_junction, corridor, highway
    # It CANNOT create: roundabouts (no detection), signalized intersections
    supported_topologies: Set[TopologyType] = field(default_factory=lambda: {
        TopologyType.INTERSECTION,
        TopologyType.T_JUNCTION,
        TopologyType.CORRIDOR,
        TopologyType.HIGHWAY,
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
        "CANNOT specify exact headway distances between vehicles",
        "CANNOT create complex multi-stage scenarios (beyond a single trigger -> action)",
        
        # Spatial limitations
        "CANNOT reference specific coordinates - only segment-relative positions",
        "CANNOT specify exact distances - only relative positions (s_along 0-1)",
        
        # Actor limitations
        "NPC vehicles only support simple trigger actions (start motion, hard brake, single lane change)",
        "NPC vehicles do NOT get their own picked paths like ego vehicles",
    ])


# Singleton instance
PIPELINE_CAPABILITIES = PipelineCapabilities()


# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

@dataclass
class CategoryDefinition:
    """
    Scenario category definition aligned with pipeline capabilities.
    """
    name: str
    notes: str
    
    # What map features are required
    required_topology: TopologyType
    needs_oncoming: bool = False
    needs_multi_lane: bool = False
    needs_on_ramp: bool = False
    needs_merge: bool = False
    
    # Non-ego actors usage
    uses_non_ego_actors: bool = False
    
    # Conflict creation mechanisms
    conflict_via: List[str] = field(default_factory=list)
    
    # Creative variation axes for diverse scenario generation
    variation_axes: List[str] = field(default_factory=list)


# Honest assessment of each category
CATEGORY_DEFINITIONS: Dict[str, CategoryDefinition] = {
    
    "Intersection Deadlock Resolution": CategoryDefinition(
        name="Intersection Deadlock Resolution",
        notes="Works at uncontrolled intersections with perpendicular approaches - HIGHLY MULTI-AGENT, pure vehicle-to-vehicle deadlock resolution",
        required_topology=TopologyType.INTERSECTION,
        uses_non_ego_actors=False,
        conflict_via=[
            "perpendicular_right_of / perpendicular_left_of constraints creating ambiguous right-of-way",
            "opposite_approach_of for oncoming standoff",
            "Paths crossing at junction center with no clear priority",
            "Multiple approach directions creating deadlock potential",
        ],
        variation_axes=[
            "ego_count: 3 vs 4 vs 5 vs 6 vehicles from different approaches",
            "approach_pattern: 2-way standoff vs 3-way deadlock vs 4-way gridlock",
            "constraint_web: chain of perpendicular constraints vs mixed perpendicular+opposite",
            "constraint_count: 2 vs 3 vs 4 vs 5+ inter-vehicle constraints",
            "maneuver_clash: all straight vs mix where turns cross each other's paths",
        ],
    ),
    
    "Unprotected Left Turn": CategoryDefinition(
        name="Unprotected Left Turn",
        notes="Works at intersections with oncoming traffic - classic multi-agent coordination",
        required_topology=TopologyType.INTERSECTION,
        needs_oncoming=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "Left-turning ego vs opposite_approach_of oncoming ego chain",
            "Multiple left turners competing for same gap",
            "Pedestrian actor crossing exit segment creates additional conflict",
            "Opposing left turners (same_exit_as) blocking each other",
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
    
    "Highway On-Ramp Merge": CategoryDefinition(
        name="Highway On-Ramp Merge",
        notes="Multi-lane highway with on-ramp merge points. Requires highway geometry with 3+ lanes and merge capabilities.",
        required_topology=TopologyType.HIGHWAY,
        needs_on_ramp=True,
        needs_merge=True,
        uses_non_ego_actors=False,
        conflict_via=[
            "Multiple ego vehicles with paths converging at merge",
            "follow_route_of constraint for queued vehicles",
            "same_road_as constraint for merge target",
            "Parked vehicle or obstacle blocking lanes increases merge complexity",
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
    
    "Interactive Lane Change": CategoryDefinition(
        name="Interactive Lane Change",
        notes="Lane weaving on multi-lane highways/corridors. Vehicles in adjacent lanes constantly change lanes, crossing into each other's lanes. Pure vehicle-to-vehicle interaction, no props.",
        required_topology=TopologyType.HIGHWAY,
        needs_multi_lane=True,
        uses_non_ego_actors=False,  # NO props - pure vehicle weaving
        conflict_via=[
            "Multiple MERGES_INTO_LANE_OF constraints (vehicles weaving into each other's lanes)",
            "LEFT_LANE_OF and RIGHT_LANE_OF to establish adjacent starting positions",
            "All vehicles perform active lane changes throughout the scenario",
        ],
        variation_axes=[
            "ego_count_d1: 2 vehicles side-by-side with 1 lane change",
            "ego_count_d2: 3 vehicles with 2 lane changes (multiple weave interactions)",
            "ego_count_d3: 4 vehicles with 3 lane changes (complex weaving pattern)",
            "ego_count_d4: 4 vehicles with 4 lane changes (dense weaving)",
            "ego_count_d5: 5 vehicles with 5+ lane changes (maximum density weaving)",
            "lane_span: 2-lane vs 3-lane corridor",
            "constraint_density: increases across variants",
            "weave_pattern: simple adjacent swaps vs complex multi-vehicle weaving",
            "spatial_coupling: all vehicles start in adjacent lanes (LEFT_LANE_OF/RIGHT_LANE_OF)",
        ],
    ),
    
    "Blocked Lane (Obstacle)": CategoryDefinition(
        name="Blocked Lane (Obstacle)",
        notes="Works with parked_vehicle actor blocking lane",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "parked_vehicle actor blocking ego's lane",
            "Adjacent lane ego via left_lane_of / right_lane_of",
        ],
        variation_axes=[
            "ego_count: 2-5 vehicles",
            "blockage_count: 1 vs 2-3 parked_vehicles",
            "blockage_lateral: center vs half_right",
            "blockage_s_along: spread along route",
            "lane_constraints: left_lane_of vs right_lane_of for adjacent traffic",
        ],
    ),
    
    "Lane Drop / Alternating Merge": CategoryDefinition(
        name="Lane Drop / Alternating Merge",
        notes="Works if map has lane drop region with multi-lane approach. Can include construction cones or parked vehicles to further constrict lanes.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        needs_merge=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "Ego vehicles in adjacent lanes reaching taper",
            "left_lane_of / right_lane_of constraints",
            "merges_into_lane_of for lane change intent",
            "Static props (cones) narrowing available merge space",
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
    
    "Major/Minor Unsignalized Entry": CategoryDefinition(
        name="Major/Minor Unsignalized Entry",
        notes="Works at T-junctions - classic yield scenario with queue. Parked vehicles can block view of oncoming traffic or limit merge space.",
        required_topology=TopologyType.T_JUNCTION,
        uses_non_ego_actors=True,
        conflict_via=[
            "Side street ego turning onto main road full of traffic",
            "Main road follow_route_of chains creating gaps and queue",
            "Parked vehicles limiting visibility of main road traffic",
            "Pedestrians crossing exit path",
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
    
    "Construction Zone": CategoryDefinition(
        name="Construction Zone",
        notes="Works with static_prop actors (cones) creating taper",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "static_prop actors in diagonal pattern",
            "Lane merge constraints between ego vehicles",
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
    
    "Pedestrian Crosswalk": CategoryDefinition(
        name="Pedestrian Crosswalk",
        notes="D1-D3: Focus on vehicle count (2/3/4 cars), single pedestrian. D4-D5: Focus on pedestrian count (2/3 pedestrians) spawning behind parked vehicle occlusions. NO moving NPC vehicles or cyclists.",
        required_topology=TopologyType.CORRIDOR,
        uses_non_ego_actors=True,
        conflict_via=[
            "walker actor with cross_perpendicular motion",
            "Timing via when=on_approach",
            "Multiple vehicles encountering the same pedestrian crossing",
            "Occlusion hiding pedestrian from oncoming vehicles (d4-d5)",
        ],
        variation_axes=[
            "ego_count: d1=2, d2=3, d3=4 vehicles (linear scaling for d1-d3)",
            "ego_count: 2-3 vehicles for d4-d5 (pedestrian-focused)",
            "walker_count: 1 for d1-d3, d4=2, d5=3 pedestrians",
            "walker_start: right_edge vs left_edge (ALWAYS start from side of road)",
            "crossing_direction: left vs right",
            "timing_phase: on_approach (pedestrian triggers when vehicle approaches)",
            "trigger_distance: 8-12m (vary occasionally for harder scenarios)",
            "occlusion: none (d1-d3) vs parked_vehicle blocking view (d4-d5 required)",
            "parked_lateral: right_edge vs offroad_right (NOT in driving path)",
            "occlusion_type: large parked vehicles (box truck, van, bus, delivery truck)",
        ],
    ),
    
    "Overtaking on Two-Lane Road": CategoryDefinition(
        name="Overtaking on Two-Lane Road",
        notes="Works on multi-lane same-direction roads. Can include oncoming vehicles, obstacles, and pedestrians to increase complexity.",
        required_topology=TopologyType.CORRIDOR,
        needs_multi_lane=True,
        uses_non_ego_actors=True,
        conflict_via=[
            "follow_route_of for following vehicle",
            "left_lane_of for passing lane vehicle",
            "Oncoming traffic limits overtaking windows",
            "Obstacles in passing lane block overtake attempt",
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
    
}


def get_available_categories() -> List[str]:
    """Return all supported categories."""
    return list(CATEGORY_DEFINITIONS.keys())
