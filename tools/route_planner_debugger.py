#!/usr/bin/env python3
"""
Standalone debugger for CARLA's GlobalRoutePlanner.

Given a route XML (e.g. generated via tools/setup_scenario_from_zip.py),
this script connects to a CARLA server, reconstructs the waypoint graph,
and traces the global route between the route's start/end points.

It emits rich JSON logs describing the reference trajectory vs. the planner
output and, optionally, draws matplotlib overlays for quick inspection.

Usage example:
    # Start CARLA separately: ./external_paths/carla_root/CarlaUE4.sh --world-port=2000
    python tools/route_planner_debugger.py \
        --route-xml simulation/leaderboard/data/CustomRoutes/B/routes_training.xml \
        --town Town05 \
        --route-id 247 \
        --carla-port 2000 \
        --out-dir debug_routes/B --draw
"""

import argparse
import json
import math
import pathlib
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np

try:
    import carla
except ImportError as exc:  # pragma: no cover - informative message
    raise RuntimeError(
        "carla module not found. Activate the CoLMDriver environment "
        "and ensure CARLA's PythonAPI is on PYTHONPATH."
    ) from exc

from leaderboard.utils.route_parser import RouteParser
from agents.navigation.local_planner import RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


def _serialize_location(loc: carla.Location) -> Dict[str, float]:
    return {"x": loc.x, "y": loc.y, "z": loc.z}


def _serialize_transform(transform: carla.Transform) -> Dict[str, float]:
    return {
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
        "pitch": transform.rotation.pitch,
        "roll": transform.rotation.roll,
    }


def _describe_waypoint(wp: carla.Waypoint) -> Dict[str, float]:
    return {
        "road_id": wp.road_id,
        "section_id": wp.section_id,
        "lane_id": wp.lane_id,
        "lane_type": str(wp.lane_type),
        "is_junction": wp.is_junction,
        "transform": _serialize_transform(wp.transform),
    }


def _neighbors(wp: carla.Waypoint, step: float = 2.0) -> List[Dict[str, float]]:
    """
    Enumerate immediate path choices from a waypoint for debugging context.
    """
    neigh = []
    for nxt in wp.next(step):
        neigh.append(_describe_waypoint(nxt))
    if not neigh:
        for nxt in wp.previous(step):
            neigh.append(_describe_waypoint(nxt))
    return neigh


def _plan_route(
    grp: GlobalRoutePlanner,
    start: carla.Location,
    end: carla.Location,
) -> List[Tuple[carla.Waypoint, RoadOption]]:
    """
    Helper to compute the planner trace with defensive checks.
    """
    route = grp.trace_route(start, end)
    if not route:
        raise RuntimeError("GlobalRoutePlanner returned an empty route.")
    return route


def _compute_rmse(ref: List[carla.Location], plan: List[carla.Waypoint]) -> float:
    """
    Rough measure of how closely the plan follows the provided trajectory.
    """
    if not ref or not plan:
        return math.inf
    ref_arr = np.array([[loc.x, loc.y] for loc in ref])
    plan_arr = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in plan])
    # Align lengths by sampling
    idx = np.linspace(0, len(plan_arr) - 1, num=len(ref_arr), dtype=int)
    sampled = plan_arr[idx]
    mse = np.mean((ref_arr - sampled) ** 2)
    return float(math.sqrt(mse))


def _extract_topology_lines(carla_map: carla.Map) -> np.ndarray:
    """
    Precompute line segments describing the drivable lane graph for plotting.
    """
    lines = []
    for seg_start, seg_end in carla_map.get_topology():
        s = seg_start.transform.location
        e = seg_end.transform.location
        lines.append(((s.x, s.y), (e.x, e.y)))
    return np.array(lines)


def _candidate_waypoints(
    carla_map: carla.Map,
    loc: carla.Location,
    radius: float = 1.5,
    max_candidates: int = 3,
) -> List[carla.Waypoint]:
    """
    Gather nearby lane centerlines around a reference location.
    CARLA doesn't expose a k-NN query directly, so we sample a grid in a small radius.
    """
    offsets = [0.0, -radius, radius]
    seen_ids = set()
    candidates = []
    for dx in offsets:
        for dy in offsets:
            sample = carla.Location(x=loc.x + dx, y=loc.y + dy, z=loc.z)
            wp = carla_map.get_waypoint(
                sample, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            if wp and wp.id not in seen_ids:
                seen_ids.add(wp.id)
                candidates.append((wp, wp.transform.location.distance(sample)))
    candidates.sort(key=lambda x: x[1])
    return [wp for wp, _ in candidates[:max_candidates]]


def _trace_cost(
    grp: GlobalRoutePlanner,
    start: carla.Waypoint,
    end: carla.Waypoint,
    ref_start: carla.Location,
    ref_end: carla.Location,
    left_penalty: float = 50.0,
    deviation_weight: float = 2.0,
) -> Tuple[List[Tuple[carla.Waypoint, RoadOption]], float, int]:
    """
    Trace a subroute and compute a cost that penalizes left/right turns and deviation from refs.
    """
    subroute = grp.trace_route(start.transform.location, end.transform.location)
    if not subroute:
        return [], math.inf, 0

    # Length cost
    length_cost = 0.0
    for (wp_curr, _), (wp_next, _) in zip(subroute[:-1], subroute[1:]):
        length_cost += wp_curr.transform.location.distance(wp_next.transform.location)

    # Turn penalties
    left_count = sum(1 for _, opt in subroute if opt == RoadOption.LEFT)
    right_count = sum(1 for _, opt in subroute if opt == RoadOption.RIGHT)
    turn_penalty = left_penalty * left_count + (left_penalty * 0.5) * right_count

    # Deviation: how far the end points are from the reference polyline segment
    dev_start = start.transform.location.distance(ref_start)
    dev_end = end.transform.location.distance(ref_end)
    deviation = dev_start + dev_end

    total_cost = length_cost + turn_penalty + deviation_weight * deviation
    return subroute, total_cost, left_count


def _best_fit_route(
    carla_map: carla.Map,
    grp: GlobalRoutePlanner,
    ref_locations: List[carla.Location],
    max_candidates: int = 3,
    keep_top: int = 5,
) -> Tuple[List[Tuple[carla.Waypoint, RoadOption]], Dict]:
    """
    Multi-start, DP-based selection: for each reference waypoint, consider several nearby
    lane centerlines and keep the lowest-cost concatenation.
    """
    if len(ref_locations) < 2:
        return [], {}

    # Seed candidates for the first ref
    first_loc = ref_locations[0]
    first_wps = _candidate_waypoints(carla_map, first_loc, max_candidates=max_candidates)
    state = []
    for wp in first_wps:
        state.append(
            {
                "wp": wp,
                "cost": 0.0 + wp.transform.location.distance(first_loc),
                "path": [(wp, RoadOption.LANEFOLLOW)],
                "lefts": 0,
            }
        )

    # DP over segments
    for idx in range(1, len(ref_locations)):
        ref_prev = ref_locations[idx - 1]
        ref_curr = ref_locations[idx]
        curr_candidates = _candidate_waypoints(
            carla_map, ref_curr, max_candidates=max_candidates
        )
        next_state = []
        for prev_state in state:
            for cand_wp in curr_candidates:
                subroute, sub_cost, lefts = _trace_cost(
                    grp,
                    prev_state["wp"],
                    cand_wp,
                    ref_prev,
                    ref_curr,
                )
                if not subroute or math.isinf(sub_cost):
                    continue
                new_cost = prev_state["cost"] + sub_cost
                new_path = prev_state["path"][:-1] + subroute  # avoid duplicate start
                next_state.append(
                    {
                        "wp": cand_wp,
                        "cost": new_cost,
                        "path": new_path,
                        "lefts": prev_state["lefts"] + lefts,
                    }
                )
        if not next_state:
            break
        # Keep best few candidates
        next_state.sort(key=lambda s: s["cost"])
        state = next_state[:keep_top]

    if not state:
        return [], {}

    best = min(state, key=lambda s: s["cost"])
    meta = {
        "total_cost": best["cost"],
        "left_turns": best["lefts"],
        "segments": len(ref_locations) - 1,
    }
    return best["path"], meta


def _plot_route(
    out_path: pathlib.Path,
    ref: List[carla.Location],
    plan: List[Tuple[carla.Waypoint, RoadOption]],
    title: str,
    topology_lines: np.ndarray,
    hotspots: List[Dict[str, float]],
    actors: List[Dict[str, float]],
) -> None:
    import matplotlib.pyplot as plt  # Local import: optional dependency

    ref_xy = np.array([[loc.x, loc.y] for loc in ref])
    plan_xy = np.array(
        [[wp.transform.location.x, wp.transform.location.y] for wp, _ in plan]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    if len(topology_lines):
        for line in topology_lines:
            plt.plot(
                [line[0][0], line[1][0]],
                [line[0][1], line[1][1]],
                color="lightgray",
                linewidth=0.3,
                alpha=0.35,
            )
    if ref_xy.size:
        plt.plot(ref_xy[:, 0], ref_xy[:, 1], "k--", label="Reference (route XML)")
    plt.plot(plan_xy[:, 0], plan_xy[:, 1], "r-", label="Planner trace")
    for idx, (x, y) in enumerate(plan_xy[:: max(len(plan_xy) // 20, 1)]):
        plt.text(x, y, str(idx), fontsize=6, color="blue")
    if hotspots:
        hx = [h["x"] for h in hotspots]
        hy = [h["y"] for h in hotspots]
        colors = ["orange" if h["type"] == "junction" else "green" for h in hotspots]
        plt.scatter(hx, hy, c=colors, s=25, marker="o", label="Ambiguous segments")
    if actors:
        ax = [a["x"] for a in actors]
        ay = [a["y"] for a in actors]
        plt.scatter(ax, ay, c="purple", marker="x", s=35, label="Scenario actors")
        for actor in actors:
            plt.text(actor["x"], actor["y"], actor["name"], fontsize=6, color="purple")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route-xml",
        required=True,
        help="Path to a routes XML, or a directory containing multiple XMLs plus actors_manifest.json",
    )
    parser.add_argument("--route-id", action="append", help="Specific route IDs to debug")
    parser.add_argument("--town", required=True, help="CARLA town name (e.g., Town05)")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--sampling-resolution", type=float, default=2.0)
    parser.add_argument("--actors-manifest", help="Optional actors_manifest.json path")
    parser.add_argument("--out-dir", default="route_debug", help="Directory for logs/plots")
    parser.add_argument("--draw", action="store_true", help="Save matplotlib overlays")
    parser.add_argument(
        "--multi-snap",
        action="store_true",
        help="Use multi-candidate snapping with cost-based selection to reduce wrong-lane snaps",
    )
    parser.add_argument("--candidate-radius", type=float, default=1.5)
    parser.add_argument("--candidate-k", type=int, default=3)
    parser.add_argument("--keep-top", type=int, default=5)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(10.0)

    world = client.load_world(args.town)
    carla_map = world.get_map()

    dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution=args.sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    topology_lines = _extract_topology_lines(carla_map)

    route_ids = args.route_id
    route_path = pathlib.Path(args.route_xml)
    if route_path.is_dir():
        xml_files = sorted(route_path.glob("*.xml"))
        if not xml_files:
            raise RuntimeError(f"No XML files found under directory {route_path}")
        manifest_path = (
            pathlib.Path(args.actors_manifest)
            if args.actors_manifest
            else route_path / "actors_manifest.json"
        )
    else:
        xml_files = [route_path]
        manifest_path = (
            pathlib.Path(args.actors_manifest)
            if args.actors_manifest
            else route_path.parent / "actors_manifest.json"
        )

    actors_manifest = {}
    if manifest_path.exists():
        try:
            actors_manifest = json.loads(manifest_path.read_text())
            print(f"Loaded actors manifest from {manifest_path}")
        except Exception as exc:
            print(f"Warning: failed to parse actors manifest {manifest_path}: {exc}")

    def parse_actor_positions(manifest: Dict, scenario_root: pathlib.Path) -> Dict[str, List[Dict[str, float]]]:
        mapping: Dict[str, List[Dict[str, float]]] = {}
        for entries in manifest.values():
            for entry in entries:
                route_id = entry.get("route_id")
                file_rel = entry.get("file")
                if not route_id or not file_rel:
                    continue
                actor_path = scenario_root / file_rel
                try:
                    tree = ET.parse(actor_path)
                    waypoint = tree.find(".//waypoint")
                    if waypoint is None:
                        continue
                    loc = {
                        "x": float(waypoint.attrib["x"]),
                        "y": float(waypoint.attrib["y"]),
                        "z": float(waypoint.attrib.get("z", 0.0)),
                        "name": entry.get("name", pathlib.Path(file_rel).stem),
                        "kind": entry.get("kind", "actor"),
                    }
                    mapping.setdefault(route_id, []).append(loc)
                except Exception as exc:
                    print(f"Warning: failed to parse actor file {actor_path}: {exc}")
        return mapping

    actor_positions = (
        parse_actor_positions(
            actors_manifest,
            scenario_root=route_path if route_path.is_dir() else route_path.parent,
        )
        if actors_manifest
        else {}
    )

    json_records = []
    seen_route_ids = set()
    for xml_file in xml_files:
        configs = RouteParser.parse_routes_file(
            str(xml_file),
            scenario_file="",
            single_route=None if not route_ids or len(route_ids) != 1 else route_ids[0],
        )
        if route_ids and len(route_ids) != 1:
            configs = [cfg for cfg in configs if cfg.name.split("_")[-1] in route_ids]
        if not configs:
            print(f"[RoutePlannerDebugger] No routes matched in {xml_file}, skipping.")
            continue

        for cfg in configs:
            ref_locations = cfg.trajectory
            start = ref_locations[0]
            end = ref_locations[-1]
            if args.multi_snap:
                plan, meta_cost = _best_fit_route(
                    carla_map,
                    grp,
                    ref_locations,
                    max_candidates=args.candidate_k,
                    keep_top=args.keep_top,
                )
            else:
                plan = _plan_route(grp, start, end)
                meta_cost = {}
            plan_waypoints = [wp for wp, _ in plan]
            rmse = _compute_rmse(ref_locations, plan_waypoints)

            route_id = cfg.name.split("_")[-1]
            base_name = f"{cfg.name}_{xml_file.stem}"
            duplicate_route_id = route_id in seen_route_ids
            seen_route_ids.add(route_id)

            lanefollow_indices = [
                idx for idx, (_, option) in enumerate(plan) if option == RoadOption.LANEFOLLOW
            ]
            lanefollow_junctions = [
                idx
                for idx, (wp, option) in enumerate(plan)
                if option == RoadOption.LANEFOLLOW and wp.is_junction
            ]
            hotspot_points = []
            for idx in lanefollow_junctions:
                loc = plan[idx][0].transform.location
                hotspot_points.append({"type": "junction", "index": idx, "x": loc.x, "y": loc.y})
            branching_points = []
            for idx, (wp, _) in enumerate(plan):
                neighbor_count = len(_neighbors(wp))
                if neighbor_count > 1:
                    loc = wp.transform.location
                    branching_points.append(
                        {"type": "branch", "index": idx, "x": loc.x, "y": loc.y, "choices": neighbor_count}
                    )
            hotspot_points.extend(branching_points)

            start_delta = math.hypot(
                start.x - plan_waypoints[0].transform.location.x,
                start.y - plan_waypoints[0].transform.location.y,
            )
            end_delta = math.hypot(
                end.x - plan_waypoints[-1].transform.location.x,
                end.y - plan_waypoints[-1].transform.location.y,
            )

            actors_for_route = actor_positions.get(route_id, [])

            planner_details = [
                {
                    "waypoint": _describe_waypoint(wp),
                    "road_option": option.name,
                    "neighbors": _neighbors(wp),
                }
                for wp, option in plan
            ]

            record = {
                "route_name": cfg.name,
                "route_id": route_id,
                "town": cfg.town,
                "sampling_resolution": args.sampling_resolution,
                "rmse_xy": rmse,
                "reference_path": [_serialize_location(loc) for loc in ref_locations],
                "planner_path": planner_details,
                "actors": actors_for_route,
                "source_xml": str(xml_file),
                "analysis": {
                    "duplicate_route_id_warning": duplicate_route_id,
                    "lanefollow_segments": len(lanefollow_indices),
                    "lanefollow_in_junctions": len(lanefollow_junctions),
                    "max_branching_factor": max(
                        (len(step["neighbors"]) for step in planner_details), default=0
                    ),
                    "start_alignment_error_m": start_delta,
                    "end_alignment_error_m": end_delta,
                    "cost_search": meta_cost,
                },
            }
            json_records.append(record)

            json_path = out_dir / f"{base_name}.json"
            json_path.write_text(json.dumps(record, indent=2))

            if args.draw:
                img_path = out_dir / f"{base_name}.png"
                _plot_route(
                    img_path,
                    ref_locations,
                    plan,
                    base_name,
                    topology_lines,
                    hotspot_points,
                    actors_for_route,
                )

            print(
                f"[RoutePlannerDebugger] {base_name}: "
                f"{len(plan)} planner waypoints, RMSE={rmse:.2f}m -> {json_path}"
            )

    if not json_records:
        raise RuntimeError("No routes processed; check --route-xml and --route-id filters.")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(json_records, indent=2))
    print(f"Wrote {len(json_records)} route logs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
