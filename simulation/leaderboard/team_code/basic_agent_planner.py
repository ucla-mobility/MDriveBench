"""
BasicAgentPlanner — wraps CARLA's built-in BasicAgent so its hazard check
consumes our Detection list (perception output OR ground-truth oracle) instead
of querying ``world.get_actors()``.

This is the experimental scaffold for the perception-as-only-independent-variable
study. CARLA's lateral PID, longitudinal PID, route following, and traffic-light
handling are all reused unchanged — they are battle-tested in the Leaderboard.
The ONE thing that varies between matrix cells is the Detection list passed in:

    perception output  → real run     (bottleneck = detector quality)
    world.get_actors() → GT oracle    (bottleneck = planner only — the control row)

Compared to RulePlanner this trades the home-grown 50-line steering+brake logic
for CARLA's PID controllers. That isolates the comparison: any closed-loop
failure in a "real" detector run that does NOT also fail under GT is attributable
to perception, not the planner.

Usage
-----
    planner = BasicAgentPlanner(actor=hero_vehicle, target_speed_kmh=25)
    planner.set_global_plan(world_plan)            # [(carla.Transform, RoadOption), ...]

    # Per step:
    planner.set_detections(detections_ego_frame)   # List[Detection]
    control = planner.run_step()                   # carla.VehicleControl
    traj    = planner.predict_trajectory(3.0, 0.1) # (T,2) world-frame xy
"""
from __future__ import annotations

import math
from typing import List, Optional

import carla
import numpy as np

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption

from team_code.rule_planner import Detection


# Half-width of a CARLA driving lane in meters; matches RulePlanner's corridor.
DEFAULT_LANE_HALF_WIDTH = 1.6


class BasicAgentPlanner:
    """CARLA BasicAgent with its hazard check rewired to consume a Detection list.

    Detections are in EGO frame (CARLA LH: +x forward, +y right). That matches
    HEAL's perception output natively, and the GT-oracle path projects world
    actors into the same frame — so the planner's view is convention-stable
    across matrix cells.
    """

    def __init__(
        self,
        actor: carla.Vehicle,
        target_speed_kmh: float = 25.0,
        corridor_half_width: float = DEFAULT_LANE_HALF_WIDTH,
        brake_persistence_ticks: int = 3,
        opt_dict: Optional[dict] = None,
    ):
        if actor is None:
            raise ValueError("BasicAgentPlanner: actor is None — defer construction until hero is spawned")
        self._actor = actor
        self._corridor_half_width = float(corridor_half_width)
        self._detections: List[Detection] = []
        # Detection-persistence filter (same idea as RulePlanner): an in-corridor
        # detection must persist for N consecutive ticks before the override
        # asserts hazard_detected=True. Filters single-frame perception FPs
        # which previously locked the ego in dense walker scenes.
        self._brake_persistence_ticks = max(1, int(brake_persistence_ticks))
        self._brake_streak = 0

        opt = dict(opt_dict or {})
        # Slightly larger than BasicAgent's default 5 m so we react earlier at
        # cruise speed (the threshold is then padded by current vehicle_speed).
        opt.setdefault("base_vehicle_threshold", 6.0)
        opt.setdefault("max_throttle", 0.6)
        opt.setdefault("max_brake", 0.5)
        opt.setdefault("max_steering", 0.8)

        self._agent = BasicAgent(actor, target_speed=target_speed_kmh, opt_dict=opt)

        # Replace the world-querying obstacle filter with our detection-driven one.
        # Bound-method assignment is safe: BasicAgent.run_step calls
        # `self._vehicle_obstacle_detected(...)` so attribute lookup finds ours.
        self._agent._vehicle_obstacle_detected = self._detection_obstacle_detected

    # ───── Wiring ─────────────────────────────────────────────────────

    def set_global_plan(self, world_plan):
        """world_plan: list of (carla.Transform, RoadOption) — leaderboard's format."""
        carla_map = self._actor.get_world().get_map()
        plan = []
        for transform, opt in world_plan:
            wp = carla_map.get_waypoint(transform.location)
            if wp is None:
                continue
            plan.append((wp, opt if opt is not None else RoadOption.LANEFOLLOW))
        if not plan:
            raise RuntimeError("BasicAgentPlanner: no waypoints survived map projection")
        self._agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)

    def set_detections(self, detections: List[Detection]):
        """Set the obstacle list for the next run_step. Detections are ego-frame."""
        self._detections = list(detections or [])

    # ───── Step ───────────────────────────────────────────────────────

    def run_step(self) -> carla.VehicleControl:
        return self._agent.run_step()

    def done(self) -> bool:
        return self._agent.done()

    @property
    def target_speed_mps(self) -> float:
        return float(self._agent._local_planner._target_speed) / 3.6

    # ───── Trajectory rollout for open-loop ADE/FDE ───────────────────

    def predict_trajectory(self, horizon_s: float = 3.0, dt: float = 0.1) -> np.ndarray:
        """Walk the local planner's queued waypoints at target_speed.

        Returns (T, 2) world-frame xy. The rollout assumes steady-state cruise —
        no obstacle-aware deceleration — which is fine for ADE/FDE since the
        ground-truth trajectory is also recorded at whatever speed the ego
        actually drove.
        """
        speed = self.target_speed_mps
        queue = list(self._agent._local_planner._waypoints_queue)
        T = max(1, int(round(horizon_s / dt)))
        loc = self._actor.get_location()
        pos = np.array([loc.x, loc.y], dtype=np.float64)

        if not queue:
            return np.tile(pos, (T, 1))

        out = np.zeros((T, 2), dtype=np.float64)
        cursor = 0
        pop_dist = 1.5  # m, when to advance to next waypoint

        for t in range(T):
            while cursor < len(queue) - 1:
                wp = queue[cursor][0].transform.location
                if math.hypot(wp.x - pos[0], wp.y - pos[1]) < pop_dist:
                    cursor += 1
                else:
                    break
            wp = queue[cursor][0].transform.location
            vec = np.array([wp.x - pos[0], wp.y - pos[1]], dtype=np.float64)
            d = float(np.linalg.norm(vec))
            cur_dir = vec / d if d > 1e-6 else np.array([0.0, 0.0])
            pos = pos + cur_dir * speed * dt
            out[t] = pos
        return out

    # ───── The crucial swap: detection-driven hazard check ────────────

    def _detection_obstacle_detected(self, vehicle_list=None, max_distance=None):
        """Replacement for ``BasicAgent._vehicle_obstacle_detected``.

        BasicAgent calls this from ``run_step`` to decide whether to emergency-stop.
        The original implementation iterates ``world.get_actors().filter("*vehicle*")``
        and projects each onto the ego's lane via the CARLA map. Here we use the
        ego-frame Detection list directly:

          * x_ego ∈ [0.5, max_distance]   — in front of us, within range
          * |y_ego| < corridor + radius   — within our corridor

        This is intentionally simpler than the lane-aware original. The
        comparison we care about is "same planner, varying perception" — and
        the simpler corridor check is itself stable across detectors. The
        complexity we lose (lane geometry awareness) was dependent on querying
        the simulator anyway, so it would have been unfair to the perception
        models that don't get that information.
        """
        if self._agent._ignore_vehicles:
            self._brake_streak = 0
            return (False, None)
        if not self._detections:
            self._brake_streak = 0
            return (False, None)

        if max_distance is None:
            max_distance = self._agent._base_vehicle_threshold
        half_w = self._corridor_half_width

        in_corridor = False
        for det in self._detections:
            x_ego = float(det.xy_ego[0])
            y_ego = float(det.xy_ego[1])
            if x_ego < 0.5 or x_ego > max_distance:
                continue
            if abs(y_ego) > half_w + float(det.radius):
                continue
            in_corridor = True
            break

        # Persistence filter: only commit the brake once the in-corridor
        # condition has held for N consecutive ticks. Resets immediately on
        # any clean tick, so genuine hazards still fire after N ticks of
        # consistent observation.
        if in_corridor:
            self._brake_streak += 1
        else:
            self._brake_streak = 0
        if self._brake_streak >= self._brake_persistence_ticks:
            return (True, None)
        return (False, None)
