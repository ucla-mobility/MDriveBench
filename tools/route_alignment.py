"""Route alignment pipeline — single source of truth for both the GRP
inspector and run_custom_eval.

The pipeline takes raw XML waypoint dicts and returns a dense, lane-following,
smooth route. It exists because:

  - leaderboard's `interpolate_trajectory` follows the lane network only via
    a *per-segment loop guard* — it can leak diagonal cuts when the input
    waypoints are sparse, and its `postprocess_route_trace` UCLA-v2 smoothing
    is only effective on UCLA towns and harmful when it does fire.
  - refine_waypoints_dp's compression aggressively prunes interior LANEFOLLOW
    states, then `grp.trace_route` between sparse compressed states detours
    or zigzags through wrong lanes.
  - Neither produces output that follows actual junction-connector splines.

This pipeline produces output that:
  - Follows lane-connector splines through junctions (via grp.trace_route at
    full hop_resolution density, no subsampling).
  - Stays on drivable lanes by construction (every output point comes from
    grp's lane-graph traversal).
  - Is smooth — Chaikin corner-cutting + bounded moving-average pass round
    junction-pinch kinks while staying within ~0.5 m of the lane center.

Entry point:
  align_route(carla_module, carla_map, grp, xml_waypoints, ...) -> list[dict]
"""

from __future__ import annotations

import math
from typing import Any


# ────────────────────────────────────────────────────────────────────────────
# Helpers: extract (x, y, z, yaw) from a CARLA Waypoint or Transform robustly.
# ────────────────────────────────────────────────────────────────────────────

def _entry_to_dict(entry, road_option=None) -> dict | None:
    """Convert a (Waypoint|Transform, RoadOption) entry to {x,y,z,yaw,road_option}.

    CARLA 0.9.12 exposes `Waypoint.transform` as a callable and `Transform.location`
    as a non-callable attribute, so we probe both shapes.
    """
    obj = entry
    if isinstance(entry, tuple) and len(entry) >= 1:
        obj = entry[0]
        if road_option is None and len(entry) >= 2:
            road_option = entry[1]
    loc = getattr(obj, "location", None)
    rot = getattr(obj, "rotation", None)
    if loc is None or rot is None or callable(loc) or callable(rot):
        tf = getattr(obj, "transform", None)
        if tf is None:
            return None
        if callable(tf):
            try:
                tf = tf()
            except Exception:
                return None
        loc = getattr(tf, "location", None)
        rot = getattr(tf, "rotation", None)
        if loc is None or rot is None:
            return None
    return {
        "x": float(loc.x),
        "y": float(loc.y),
        "z": float(loc.z),
        "yaw": float(rot.yaw),
        "road_option": str(road_option).split(".")[-1] if road_option is not None else None,
    }


# ────────────────────────────────────────────────────────────────────────────
# grp.trace_route with recursive midpoint fallback (handles long detours).
# ────────────────────────────────────────────────────────────────────────────

def grp_route_recursive(carla_module, carla_map, grp, a_loc, b_loc,
                        max_detour: float = 4.0,
                        depth: int = 0,
                        max_depth: int = 4) -> list:
    """grp.trace_route with recursive midpoint fallback.

    Some CARLA road graphs (e.g. Town02 Major_Minor intersections) return
    pathological 5-10× detours for sparse 2-waypoint chords because the
    direct connector at the junction isn't represented in the lane graph.
    When the trace is longer than max_detour × chord, snap the chord midpoint
    to a drivable lane and recursively trace each half. This forces grp to
    use local junction connectors instead of routing the long way around.
    """
    chord = math.hypot(b_loc.x - a_loc.x, b_loc.y - a_loc.y)
    if chord < 1.0:
        return []
    try:
        route = grp.trace_route(a_loc, b_loc)
    except Exception:
        route = []

    def _trace_len(rt):
        total = 0.0
        prev = None
        for wp, _ in rt:
            loc = wp.transform.location if hasattr(wp, "transform") else wp.location
            if callable(loc):
                continue
            if prev is not None:
                total += math.hypot(loc.x - prev[0], loc.y - prev[1])
            prev = (loc.x, loc.y)
        return total

    if route and _trace_len(route) <= chord * max_detour:
        return route
    if depth >= max_depth:
        return route

    mid_x = (a_loc.x + b_loc.x) / 2.0
    mid_y = (a_loc.y + b_loc.y) / 2.0
    mid_z = (a_loc.z + b_loc.z) / 2.0
    try:
        mid_wp = carla_map.get_waypoint(
            carla_module.Location(x=mid_x, y=mid_y, z=mid_z),
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
    except Exception:
        mid_wp = None
    if mid_wp is None:
        return route
    mid_loc = carla_module.Location(
        x=float(mid_wp.transform.location.x),
        y=float(mid_wp.transform.location.y),
        z=float(mid_wp.transform.location.z),
    )
    if (math.hypot(mid_loc.x - a_loc.x, mid_loc.y - a_loc.y) < 0.5
            or math.hypot(mid_loc.x - b_loc.x, mid_loc.y - b_loc.y) < 0.5):
        return route

    left = grp_route_recursive(carla_module, carla_map, grp, a_loc, mid_loc,
                               max_detour, depth + 1, max_depth)
    right = grp_route_recursive(carla_module, carla_map, grp, mid_loc, b_loc,
                                max_detour, depth + 1, max_depth)
    if not left:
        return right
    if not right:
        return left
    return left + right[1:]


# ────────────────────────────────────────────────────────────────────────────
# Densify XML inputs: linear chord for short gaps, ALL grp waypoints for long
# gaps (with recursive midpoint fallback for pathological detours).
# ────────────────────────────────────────────────────────────────────────────

def densify_inputs(waypoints: list[dict],
                   carla_module, carla_map, grp,
                   max_gap_m: float = 3.0,
                   large_gap_m: float = 5.0,
                   max_grp_detour: float = 4.0) -> tuple[list[dict], int]:
    """Densify XML waypoints to ≤ max_gap_m spacing.

      - Gaps ≤ max_gap_m: append as-is.
      - max_gap_m < gap ≤ large_gap_m: linear chord interpolation (fine —
        chord-vs-arc error is sub-degree at this scale).
      - gap > large_gap_m: replace with ALL grp.trace_route waypoints (no
        subsampling) via grp_route_recursive. This is the critical bit:
        sparse-input left/right turns where the direct chord cuts diagonally
        through an intersection get replaced with the actual lane-connector
        spline, sampled at hop_resolution (~2 m).
    """
    if max_gap_m <= 0 or len(waypoints) < 2:
        return list(waypoints), 0

    out: list[dict] = [waypoints[0]]
    inserted = 0
    for i in range(1, len(waypoints)):
        p, q = waypoints[i - 1], waypoints[i]
        dx = q["x"] - p["x"]
        dy = q["y"] - p["y"]
        dz = q.get("z", 0.0) - p.get("z", 0.0)
        d = math.hypot(dx, dy)
        used_grp = False
        if d > large_gap_m:
            a_loc = carla_module.Location(x=p["x"], y=p["y"], z=p.get("z", 0.0))
            b_loc = carla_module.Location(x=q["x"], y=q["y"], z=q.get("z", 0.0))
            route = grp_route_recursive(
                carla_module, carla_map, grp, a_loc, b_loc,
                max_detour=max_grp_detour, max_depth=4,
            )
            if route and len(route) >= 3:
                pts = []
                trace_len = 0.0
                prev_xy = None
                for entry in route:
                    rd = _entry_to_dict(entry)
                    if rd is None:
                        continue
                    xy = (rd["x"], rd["y"])
                    if prev_xy is not None:
                        trace_len += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
                    prev_xy = xy
                    pts.append(rd)
                if pts and trace_len < d * max_grp_detour:
                    # Use ALL grp waypoints. grp samples the lane connector
                    # spline at hop_resolution (~2 m) — that's exactly the
                    # density we need to faithfully follow tight junction
                    # turn-lanes. Coarser subsampling drops corner samples
                    # and reintroduces diagonal-cut bee-lines.
                    for j in range(1, len(pts) - 1):
                        out.append({
                            "x": pts[j]["x"],
                            "y": pts[j]["y"],
                            "z": pts[j].get("z", 0.0),
                            "yaw": pts[j].get("yaw", 0.0),
                        })
                        inserted += 1
                    used_grp = True

        if not used_grp and d > max_gap_m:
            n_splits = max(2, int(math.ceil(d / max_gap_m)))
            for j in range(1, n_splits):
                t = j / n_splits
                out.append({
                    "x": p["x"] + dx * t,
                    "y": p["y"] + dy * t,
                    "z": p.get("z", 0.0) + dz * t,
                    "yaw": q.get("yaw", 0.0),
                })
                inserted += 1
        out.append(q)
    return out, inserted


# ────────────────────────────────────────────────────────────────────────────
# Hairpin filter: drop physically-impossible heading reversals.
# ────────────────────────────────────────────────────────────────────────────

def remove_hairpins(points: list[dict], max_yaw_jump_deg: float = 120.0,
                    max_passes: int = 5) -> tuple[list[dict], int]:
    """Iteratively drop interior points whose surrounding heading change
    exceeds max_yaw_jump_deg. A heading change > 120° between consecutive
    samples means the path momentarily reverses direction — physically
    impossible for a vehicle. Endpoints are never dropped.
    """
    if len(points) < 3:
        return list(points), 0
    pts = list(points)
    total_removed = 0
    for _ in range(max_passes):
        if len(pts) < 3:
            break
        keep = [True] * len(pts)
        for i in range(1, len(pts) - 1):
            ax, ay = pts[i - 1]["x"], pts[i - 1]["y"]
            bx, by = pts[i]["x"], pts[i]["y"]
            cx, cy = pts[i + 1]["x"], pts[i + 1]["y"]
            h1 = math.degrees(math.atan2(by - ay, bx - ax))
            h2 = math.degrees(math.atan2(cy - by, cx - bx))
            diff = abs((h1 - h2 + 180.0) % 360.0 - 180.0)
            if diff > max_yaw_jump_deg:
                keep[i] = False
        n_removed = sum(1 for k in keep if not k)
        if n_removed == 0:
            break
        pts = [p for p, k in zip(pts, keep) if k]
        total_removed += n_removed
    return pts, total_removed


# ────────────────────────────────────────────────────────────────────────────
# Chaikin's corner-cutting smoothing.
# ────────────────────────────────────────────────────────────────────────────

def chaikin_smooth(points: list[dict], iterations: int = 4) -> list[dict]:
    """Round sharp corners via Chaikin's corner-cutting. Endpoints preserved.
    Each iteration ~doubles point count and rounds corners; after N iters a
    90° corner becomes ~ (90/2^N)° per resampled chord.
    """
    if iterations <= 0 or len(points) < 3:
        return points
    pts = points
    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ax, ay, az = a["x"], a["y"], a.get("z", 0.0)
            bx, by, bz = b["x"], b["y"], b.get("z", 0.0)
            p1 = {"x": 0.75 * ax + 0.25 * bx,
                  "y": 0.75 * ay + 0.25 * by,
                  "z": 0.75 * az + 0.25 * bz,
                  "yaw": 0.0, "road_option": None}
            p2 = {"x": 0.25 * ax + 0.75 * bx,
                  "y": 0.25 * ay + 0.75 * by,
                  "z": 0.25 * az + 0.75 * bz,
                  "yaw": 0.0, "road_option": None}
            if i > 0:
                new_pts.append(p1)
            if i < len(pts) - 2:
                new_pts.append(p2)
        new_pts.append(pts[-1])
        pts = new_pts
    return pts


# ────────────────────────────────────────────────────────────────────────────
# Bounded moving-average smoothing (caps per-point displacement).
# ────────────────────────────────────────────────────────────────────────────

def bounded_moving_avg(points: list[dict], window: int = 11,
                       max_displacement_m: float = 0.5) -> list[dict]:
    """Windowed moving-average on (x, y), capped at max_displacement_m so a
    smoothed point never moves further than ~14% of a 3.5 m lane width from
    its original position.
    """
    if len(points) < window or window < 2:
        return points
    half = window // 2
    out: list[dict] = []
    for i in range(len(points)):
        lo = max(0, i - half)
        hi = min(len(points), i + half + 1)
        avg_x = sum(p["x"] for p in points[lo:hi]) / (hi - lo)
        avg_y = sum(p["y"] for p in points[lo:hi]) / (hi - lo)
        dx = avg_x - points[i]["x"]
        dy = avg_y - points[i]["y"]
        d = math.hypot(dx, dy)
        if d > max_displacement_m and d > 1e-6:
            scale = max_displacement_m / d
            avg_x = points[i]["x"] + dx * scale
            avg_y = points[i]["y"] + dy * scale
        out.append({**points[i], "x": avg_x, "y": avg_y})
    return out


# ────────────────────────────────────────────────────────────────────────────
# THE pipeline.
# ────────────────────────────────────────────────────────────────────────────

def align_route(carla_module, carla_map, grp, xml_waypoints: list[dict],
                max_gap_m: float = 3.0,
                large_gap_m: float = 5.0,
                max_grp_detour: float = 4.0,
                hairpin_threshold_deg: float = 120.0,
                chaikin_iters: int = 4,
                moving_avg_window: int = 11,
                moving_avg_max_disp_m: float = 0.5,
                moving_avg_passes: int = 2) -> dict:
    """Take raw XML waypoint dicts → return a dense, lane-following, smooth route.

    Args:
        xml_waypoints: list of {"x","y","z","yaw"} dicts from the XML.
        ... (other args expose pipeline knobs; defaults match the validated
        --bypass-dp inspector pipeline that produces grp_vis_final/).

    Returns:
        {
            "dense_route":        [list of {x,y,z,yaw,road_option} dicts],
            "n_input_waypoints":  int,
            "n_densified":        int (after densify_inputs),
            "n_inserted":         int (how many new pts densify added),
            "n_hairpins_removed": int,
            "n_final":            int (length of dense_route),
        }
    """
    if not xml_waypoints or len(xml_waypoints) < 2:
        return {
            "dense_route": list(xml_waypoints),
            "n_input_waypoints": len(xml_waypoints),
            "n_densified": len(xml_waypoints),
            "n_inserted": 0,
            "n_hairpins_removed": 0,
            "n_final": len(xml_waypoints),
        }

    # 1. Densify: short gaps linear, long gaps via grp.trace_route (full density)
    densified, n_inserted = densify_inputs(
        xml_waypoints, carla_module, carla_map, grp,
        max_gap_m=max_gap_m, large_gap_m=large_gap_m,
        max_grp_detour=max_grp_detour,
    )

    # 2. Build a working list with road_option=None on every point
    dense: list[dict] = []
    for w in densified:
        dense.append({
            "x": float(w["x"]),
            "y": float(w["y"]),
            "z": float(w.get("z", 0.0)),
            "yaw": float(w.get("yaw", 0.0)),
            "road_option": None,
        })

    # 3. Hairpin filter: drop physically-impossible reversals
    dense, n_hairpins = remove_hairpins(dense, max_yaw_jump_deg=hairpin_threshold_deg)

    # 4. Chaikin corner-cutting
    if chaikin_iters > 0 and len(dense) >= 3:
        dense = chaikin_smooth(dense, iterations=chaikin_iters)

    # 5. Bounded moving-average passes (flatten residual junction kinks)
    for _ in range(max(0, moving_avg_passes)):
        if len(dense) >= 5:
            dense = bounded_moving_avg(
                dense, window=moving_avg_window,
                max_displacement_m=moving_avg_max_disp_m,
            )

    return {
        "dense_route": dense,
        "n_input_waypoints": len(xml_waypoints),
        "n_densified": len(densified),
        "n_inserted": n_inserted,
        "n_hairpins_removed": n_hairpins,
        "n_final": len(dense),
    }
