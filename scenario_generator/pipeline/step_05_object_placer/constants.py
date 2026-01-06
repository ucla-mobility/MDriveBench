LATERAL_RELATIONS = [
    "center",
    "half_right",
    "right_edge",
    "offroad_right",
    "half_left",
    "left_edge",
    "offroad_left",
]

# Lane width heuristic (meters)
LANE_WIDTH_M = 3.5

# Map qualitative to meters (right positive using right_normal_world)
LATERAL_TO_M = {
    "center": 0.0,
    "half_right": +0.25 * LANE_WIDTH_M,
    "right_edge": +0.45 * LANE_WIDTH_M,
    "offroad_right": +1.10 * LANE_WIDTH_M,
    "half_left": -0.25 * LANE_WIDTH_M,
    "left_edge": -0.45 * LANE_WIDTH_M,
    "offroad_left": -1.10 * LANE_WIDTH_M,
}

__all__ = [
    "LANE_WIDTH_M",
    "LATERAL_RELATIONS",
    "LATERAL_TO_M",
]
