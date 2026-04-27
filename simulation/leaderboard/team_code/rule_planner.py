"""
RulePlanner — minimal rule-based planner with two output modes.

Single decision core:
    plan(ego_state, detections, dt) -> PlanIntent

Two adapters:
    to_control(intent, ego_state)              -> (steer, throttle, brake)   [closed-loop]
    to_trajectory(ego_state, intent, horizon)  -> (T, 2) world-frame xy      [open-loop]

Conventions:
    * World frame: same frame as the global route waypoints. Right-handed,
      +x east, +y north (matches CARLA / V2X-Real recordings).
    * Ego frame: +x forward, +y right (CARLA is LH, so the math rotation
      R(-yaw) of a CARLA world delta yields a right-positive lateral coord).
    * Detections come in EGO frame — that's what HEAL outputs natively.
    * Yaw is in radians, [-pi, pi], measured from world +x. CARLA's
      Transform.rotation.yaw is positive clockwise from above (LH).
    * Speed is m/s.

This is intentionally simple — ~50 lines of decision logic, ~30 lines of
control adapter, ~30 lines of trajectory adapter. The point is a SINGLE
defensible rule-based baseline that's identical across detector swaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EgoState:
    """Ego vehicle state in WORLD frame at the current step."""
    xy:    np.ndarray   # (2,) world-frame position
    yaw:   float        # radians
    speed: float        # m/s, scalar


@dataclass
class Detection:
    """One detected obstacle, in EGO frame.

    The planner only needs centroid + a coarse footprint for corridor checks.
    Full 3D shape lives in the perception output if anyone needs it later.
    """
    xy_ego:    np.ndarray   # (2,) center xy in ego frame, +x forward
    radius:    float        # m, half-diagonal of footprint — coarse
    score:     float


@dataclass
class PlanIntent:
    """The planner's decision for this step. Mode-agnostic."""
    target_xy_world: np.ndarray   # (2,) waypoint to track in world frame
    target_speed:    float        # m/s
    brake_now:       bool         # hard stop override (TTC violation)
    debug:           dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────
# RulePlanner
# ────────────────────────────────────────────────────────────────────────────

class RulePlanner:
    """
    Parameters
    ----------
    global_route : (W, 2) world-frame waypoints, ordered start → goal.
    cruise_speed : m/s nominal speed when corridor is clear.
    waypoint_pop_dist : pop a route waypoint when ego is within this many m.
    corridor_half_width : m. Detections within |y_ego| < this count as in-path.
    look_ahead : m. Only consider obstacles closer than this.
    ttc_brake_s : s. If any in-corridor obstacle has TTC < this, brake hard.
    safe_gap : m. Subtract from x_ego before computing TTC (collision margin).
    """

    def __init__(
        self,
        global_route:        np.ndarray,
        cruise_speed:        float = 6.0,
        waypoint_pop_dist:   float = 4.0,
        corridor_half_width: float = 1.6,
        look_ahead:          float = 25.0,
        ttc_brake_s:         float = 2.5,
        safe_gap:            float = 4.0,
        brake_persistence_ticks: int = 3,
    ):
        route = np.asarray(global_route, dtype=np.float64)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2:
            raise ValueError(f"global_route must be (W>=2, 2); got {route.shape}")
        self._route                  = route
        self._cursor                 = 0   # index of next-to-track waypoint
        self.cruise_speed            = cruise_speed
        self.waypoint_pop_dist       = waypoint_pop_dist
        self.corridor_half_width     = corridor_half_width
        self.look_ahead              = look_ahead
        self.ttc_brake_s             = ttc_brake_s
        self.safe_gap                = safe_gap
        # Detection-persistence filter: a brake-trigger condition must persist
        # for this many *consecutive* ticks before brake_now actually fires.
        # 1 disables the filter (legacy behavior). 3 (≈150 ms @ 20 Hz) is
        # enough to filter most one-frame perception false positives, which
        # were observed to permanently lock the ego in dense walker scenes.
        self.brake_persistence_ticks = max(1, int(brake_persistence_ticks))
        self._brake_streak           = 0   # ticks of consecutive in-corridor obstacles

    # ─────── Decision core ───────────────────────────────────────────────

    def plan(
        self,
        ego: EgoState,
        detections: List[Detection],
        dt: float = 0.1,
    ) -> PlanIntent:
        # 1. Advance the cursor past any waypoints we've already crossed.
        while (self._cursor < len(self._route) - 1
               and np.linalg.norm(self._route[self._cursor] - ego.xy) < self.waypoint_pop_dist):
            self._cursor += 1
        target_xy_world = self._route[self._cursor]

        # 2. Constant cruise speed. Curve-based slowdown removed in v1 — densified
        # route waypoints produce fake "sharp turns" between adjacent segments
        # that wrecked the brake decision and overwhelmed any signal from
        # perception. Reintroduce only with proper segment smoothing.
        target_speed = self.cruise_speed

        # 3. Corridor check on detections (already in ego frame).
        # `raw_brake` is the per-tick decision; `brake_now` only commits if the
        # condition has held for `brake_persistence_ticks` consecutive ticks.
        # This filters one-frame false positives that previously permanently
        # locked the ego in dense walker scenes.
        raw_brake = False
        min_ttc = np.inf
        for det in detections:
            x_ego, y_ego = det.xy_ego
            if x_ego < 0 or x_ego > self.look_ahead:
                continue                         # behind us or too far
            if abs(y_ego) > self.corridor_half_width + det.radius:
                continue                         # not in our lane
            gap = x_ego - self.safe_gap - det.radius
            if gap <= 0:
                raw_brake = True
                min_ttc = 0.0
                break
            # Treat the obstacle as stationary — pessimistic but safe for v1.
            ttc = gap / max(ego.speed, 0.5)
            if ttc < self.ttc_brake_s:
                raw_brake = True
            if ttc < min_ttc:
                min_ttc = ttc

        if raw_brake:
            self._brake_streak += 1
        else:
            self._brake_streak = 0
        brake_now = self._brake_streak >= self.brake_persistence_ticks

        return PlanIntent(
            target_xy_world=target_xy_world,
            target_speed=0.0 if brake_now else target_speed,
            brake_now=brake_now,
            debug={
                "cursor":       self._cursor,
                "min_ttc":      float(min_ttc),
                "brake_streak": int(self._brake_streak),
                "raw_brake":    bool(raw_brake),
            },
        )

    # ─────── Adapter A — closed-loop CARLA control ─────────────────────

    def to_control(
        self,
        intent: PlanIntent,
        ego: EgoState,
        K_steer:    float = 1.5,
        K_throttle: float = 0.5,
        K_brake:    float = 0.5,
    ) -> Tuple[float, float, float]:
        """Return (steer, throttle, brake) in CARLA's [-1,1]/[0,1]/[0,1] ranges."""
        # Heading error: angle ego→target in ego frame, minus 0 (target should
        # be straight ahead). We compute it directly in ego frame.
        dxw = intent.target_xy_world - ego.xy
        cos_y, sin_y = np.cos(-ego.yaw), np.sin(-ego.yaw)
        x_ego = cos_y * dxw[0] - sin_y * dxw[1]
        y_ego = sin_y * dxw[0] + cos_y * dxw[1]
        # CARLA is LH: with R(-yaw) above, positive y_ego means target is to
        # the vehicle's RIGHT. CARLA's positive steer is also right, so no
        # sign flip — heading_err > 0 ⇒ steer right ⇒ correct.
        heading_err = np.arctan2(y_ego, max(x_ego, 0.5))   # rad
        steer = float(np.clip(K_steer * heading_err / np.pi, -1.0, 1.0))

        if intent.brake_now:
            return steer, 0.0, 1.0

        speed_err = intent.target_speed - ego.speed
        if speed_err >= 0:
            throttle = float(np.clip(K_throttle * speed_err, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-K_brake * speed_err, 0.0, 1.0))
        return steer, throttle, brake

    # ─────── Adapter B — open-loop predicted trajectory ─────────────────

    def to_trajectory(
        self,
        ego: EgoState,
        intent: PlanIntent,
        horizon_s: float = 3.0,
        dt:        float = 0.1,
    ) -> np.ndarray:
        """Roll out a (T, 2) world-frame trajectory from the current intent.

        Behavior:
          * If brake_now: linearly decelerate to 0 over ttc_brake_s, then idle.
          * Otherwise: follow the route at intent.target_speed, advancing
            through the route waypoint list as we cross each one.
        """
        T = int(round(horizon_s / dt))
        traj = np.zeros((T, 2), dtype=np.float64)

        pos = ego.xy.astype(np.float64).copy()
        speed = float(ego.speed)
        cursor = self._cursor

        if intent.brake_now:
            # Decelerate at a rate that hits 0 over ttc_brake_s seconds, then idle.
            decel = ego.speed / max(self.ttc_brake_s, 1e-3)
        else:
            decel = 0.0

        # Heading direction at start of rollout — match ego yaw, so the first
        # few steps stay on the planned heading even if the next waypoint is
        # behind a turn.
        cur_dir = np.array([np.cos(ego.yaw), np.sin(ego.yaw)])

        for t in range(T):
            # 1. Decide direction this step: head toward current route waypoint
            #    if it's still ahead; otherwise re-orient to next.
            while cursor < len(self._route) - 1:
                vec = self._route[cursor] - pos
                if np.linalg.norm(vec) < self.waypoint_pop_dist:
                    cursor += 1
                else:
                    break
            target = self._route[cursor]
            vec = target - pos
            d = np.linalg.norm(vec)
            if d > 1e-6:
                cur_dir = vec / d

            # 2. Update speed (deceleration if braking, else hold target_speed).
            if intent.brake_now:
                speed = max(0.0, speed - decel * dt)
            else:
                speed = intent.target_speed

            # 3. Advance position.
            pos = pos + cur_dir * speed * dt
            traj[t] = pos

        return traj

    # ─────── Helpers ────────────────────────────────────────────────────

    def reset(self):
        """Reset to the start of the route — call this between scenarios."""
        self._cursor = 0
        self._brake_streak = 0

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def route_remaining(self) -> int:
        return max(0, len(self._route) - self._cursor)
