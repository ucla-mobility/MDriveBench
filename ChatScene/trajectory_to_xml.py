#!/usr/bin/env python3
"""
Convert SafeBench trajectory logs (CSV/NPZ) to CoLMDriver XML route format.

This script reads trajectory data logged by SafeBench's TrajectoryLogger and converts
it to the XML format expected by run_custom_eval.py for CoLMDriver evaluation.

Usage:
    python trajectory_to_xml.py <npz_or_csv_file> --output <output_dir> --town Town05
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import csv
import math
import zipfile
import os
from safebench.gym_carla.trajectory_logger import TrajectoryReader

# CARLA global route planner (same as used at runtime)
try:
    import carla
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    carla = None
    GlobalRoutePlannerDAO = None
    GlobalRoutePlanner = None


def load_csv_trajectories(csv_path: Path) -> Dict[str, List[Dict]]:
    """Load trajectories from CSV file"""
    trajectories = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            actor_id = row['actor_id']
            if actor_id not in trajectories:
                trajectories[actor_id] = []
            
            # Convert numeric fields
            actor_type = row.get('actor_type')
            vehicle_model = row.get('vehicle_model')  # CARLA blueprint e.g. vehicle.yamaha.yzf
            trajectories[actor_id].append({
                'step': int(row['step']),
                'timestamp': float(row['timestamp']),
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z']),
                'vx': float(row.get('vx', 0)),
                'vy': float(row.get('vy', 0)),
                'vz': float(row.get('vz', 0)),
                'velocity': float(row['velocity']),
                'ax': float(row.get('ax', 0)),
                'ay': float(row.get('ay', 0)),
                'az': float(row.get('az', 0)),
                'roll': float(row['roll']),
                'pitch': float(row['pitch']),
                'yaw': float(row['yaw']),
                'actor_type': actor_type,
                'vehicle_model': vehicle_model,
            })
    
    return trajectories


def load_npz_trajectories(npz_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Load trajectories from NPZ file along with metadata"""
    reader = TrajectoryReader(npz_path)
    return reader.get_all_actors(), reader.get_metadata()


def load_route_plan_csv(route_plan_path: Path) -> List[Dict]:
    """
    Load a coarse ego route plan CSV. Expected columns: x, y, z (yaw optional).
    Returns a list of dicts shaped like trajectory rows with yaw defaulting to 0.
    """
    plan = []
    with open(route_plan_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yaw = float(row.get('yaw', 0.0))
            plan.append({
                'step': len(plan),  # synthetic step index
                'timestamp': 0.0,
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row.get('z', 0.0)),
                'vx': 0.0,
                'vy': 0.0,
                'vz': 0.0,
                'velocity': 0.0,
                'ax': 0.0,
                'ay': 0.0,
                'az': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': yaw,
                'actor_type': 'ego',
            })
    return plan


def npz_to_dict_trajectories(
    trajectories: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Dict]] = None,
) -> Dict[str, List[Dict]]:
    """Convert NPZ structured arrays to list of dicts"""
    result = {}
    agent_meta: Dict[str, Dict] = {}
    if metadata:
        agent_meta = metadata.get('agents', {})
    for actor_id, traj_array in trajectories.items():
        result[actor_id] = []
        actor_type = None
        vehicle_model = None
        if agent_meta:
            actor_info = agent_meta.get(actor_id, {})
            actor_type = actor_info.get('type')
            vehicle_model = actor_info.get('vehicle_model')
        for i, row in enumerate(traj_array):
            result[actor_id].append({
                'step': i,
                'timestamp': float(row['timestamp']),
                'x': float(row['x']),
                'y': float(row['y']),
                'z': float(row['z']),
                'vx': float(row.get('vx', 0)) if 'vx' in traj_array.dtype.names else 0.0,
                'vy': float(row.get('vy', 0)) if 'vy' in traj_array.dtype.names else 0.0,
                'vz': float(row.get('vz', 0)) if 'vz' in traj_array.dtype.names else 0.0,
                'velocity': float(row['velocity']),
                'ax': float(row.get('ax', 0)) if 'ax' in traj_array.dtype.names else 0.0,
                'ay': float(row.get('ay', 0)) if 'ay' in traj_array.dtype.names else 0.0,
                'az': float(row.get('az', 0)) if 'az' in traj_array.dtype.names else 0.0,
                'roll': float(row['roll']),
                'pitch': float(row['pitch']),
                'yaw': float(row['yaw']),
                'actor_type': actor_type,
                'vehicle_model': vehicle_model,
            })
    return result


def create_route_xml(
    actor_id: str,
    actor_role: str,
    trajectory: List[Dict],
    town: str,
    route_id: str = "0",
    downsample_factor: int = 1,
    vehicle_model: Optional[str] = None
) -> ET.Element:
    """
    Create XML route element from trajectory data.
    
    Args:
        actor_id: Identifier for the actor (e.g., 'ego_0', 'adv_agent_0')
        actor_role: Role of actor (e.g., ego, pedestrian, bike, nonego, npc)
        trajectory: List of trajectory states with x, y, z, yaw
        town: Town name (e.g., 'Town05')
        route_id: Route identifier
        downsample_factor: Sample every Nth point (to reduce waypoint count)
        vehicle_model: CARLA blueprint ID (e.g., 'static.prop.trafficcone01', 'vehicle.toyota.prius')
    
    Returns:
        ElementTree.Element with route XML
    """
    # Create route element
    route_attrs = {
        'id': route_id,
        'town': town,
        'role': actor_role,
    }
    # Include CARLA blueprint for spawning props/vehicles correctly
    if vehicle_model:
        route_attrs['model'] = vehicle_model
    route_elem = ET.Element('route', route_attrs)
    
    # Add waypoints, downsampling as needed
    sampled_trajectory = trajectory[::downsample_factor]
    
    for i, state in enumerate(sampled_trajectory):
        # Create waypoint element
        waypoint_elem = ET.SubElement(route_elem, 'waypoint', {
            'x': f"{state['x']:.6f}",
            'y': f"{state['y']:.6f}",
            'z': f"{state['z']:.6f}",
            'yaw': f"{state['yaw']:.6f}",
        })

        # For ego, keep XML minimal: no velocity/accel; only first orientation
        is_ego = actor_role.lower() == "ego"
        if not is_ego:
            # Add velocity info as sub-element
            ET.SubElement(waypoint_elem, 'velocity', {
                'x': f"{state.get('vx', 0):.6f}",
                'y': f"{state.get('vy', 0):.6f}",
                'z': f"{state.get('vz', 0):.6f}",
                'magnitude': f"{state.get('velocity', 0):.6f}",
            })
            # Add acceleration info as sub-element
            ET.SubElement(waypoint_elem, 'acceleration', {
                'x': f"{state.get('ax', 0):.6f}",
                'y': f"{state.get('ay', 0):.6f}",
                'z': f"{state.get('az', 0):.6f}",
            })
            # Add orientation info as sub-element
            ET.SubElement(waypoint_elem, 'orientation', {
                'roll': f"{state.get('roll', 0):.6f}",
                'pitch': f"{state.get('pitch', 0):.6f}",
                'yaw': f"{state.get('yaw', 0):.6f}",
            })
        else:
            if i == 0:
                ET.SubElement(waypoint_elem, 'orientation', {
                    'roll': f"{state.get('roll', 0):.6f}",
                    'pitch': f"{state.get('pitch', 0):.6f}",
                    'yaw': f"{state.get('yaw', 0):.6f}",
                })
    
    return route_elem


def trajectories_to_xml(
    trajectories: Dict[str, List[Dict]],
    town: str,
    output_dir: Path,
    downsample_factor: int = 1,
    promote_set: set = None
) -> List[Path]:
    """
    Convert trajectories to XML route files.
    
    Args:
        trajectories: Dict mapping actor_id to trajectory (list of states)
        town: Town name
        output_dir: Output directory for XML files
        downsample_factor: Sample every Nth waypoint
        promote_set: Set of actor IDs (lowercase) to treat as ego agents
    
    Returns:
        List of paths to created XML files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    
    # Initialize promote_set
    if promote_set is None:
        promote_set = set()
    
    # Create manifest data
    actors_manifest = {}
    
    town_lower = town.lower()
    
    def derive_role_and_index(actor_id: str, actor_type: Optional[str], promote_set: set = None) -> Tuple[str, str]:
        promote_set = promote_set or set()
        actor_id_lower = actor_id.lower()
        # Check if this actor should be promoted to ego
        is_promoted = actor_id_lower in promote_set
        
        if actor_type:
            parts = actor_type.split('_')
            base_role = parts[0] if parts else actor_type
            idx = parts[-1] if len(parts) > 1 else None
        else:
            base_role = 'ego' if actor_id.startswith('ego') else 'npc'
            idx = None
        
        # Override role if promoted to ego
        if is_promoted:
            base_role = 'ego'
        
        if idx is None and '_' in actor_id:
            idx = actor_id.split('_')[-1]
        if idx is None:
            idx = '0'
        return base_role, idx
    
    def indent_xml(tree: ET.ElementTree) -> None:
        """Indent XML even on Python versions without ET.indent (pre-3.9)."""
        indent_fn = getattr(ET, "indent", None)
        if indent_fn:
            indent_fn(tree, space="  ", level=0)
            return
        def _indent(elem: ET.Element, level: int = 0) -> None:
            i = "\n" + level * "  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                for child in elem:
                    _indent(child, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        _indent(tree.getroot())
    
    # Track ego index globally to ensure unique filenames for multiple egos
    ego_index_counter = 0
    
    for actor_id, trajectory in trajectories.items():
        if not trajectory:
            print(f"  Skipping empty trajectory for {actor_id}")
            continue
        
        original_actor_type = trajectory[0].get('actor_type')
        vehicle_model = trajectory[0].get('vehicle_model') if trajectory else None
        actor_role, _ = derive_role_and_index(actor_id, original_actor_type, promote_set)
        
        # Assign sequential index for ego actors to avoid filename collisions
        if actor_role.lower() == "ego":
            actor_index = str(ego_index_counter)
            ego_index_counter += 1
        else:
            # For non-ego actors, use original index from actor_id
            if '_' in actor_id:
                actor_index = actor_id.split('_')[-1]
            else:
                actor_index = '0'
        
        # Create XML route
        route_elem = create_route_xml(
            actor_id=actor_id,
            actor_role=actor_role,
            trajectory=trajectory,
            town=town,
            route_id="0",
            downsample_factor=downsample_factor,
            vehicle_model=vehicle_model
        )
        
        # Create XML tree and write to file
        root = ET.Element('routes')
        root.append(route_elem)
        tree = ET.ElementTree(root)
        
        # Pedestrians and other roles live under actors/<role> to avoid ego collision.
        role_dir = output_dir if actor_role.lower() == "ego" else output_dir / "actors" / actor_role
        role_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{town_lower}_custom_{actor_role}_actor_{actor_index}.xml"
        output_path = role_dir / filename
        
        # Write XML with proper formatting
        indent_xml(tree)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"  Created {filename} with {len(trajectory)} total waypoints ({len(trajectory)//downsample_factor} after downsampling)")
        output_files.append(output_path)
        
        # Track in manifest
        role_key = actor_role
        if role_key not in actors_manifest:
            actors_manifest[role_key] = []
        
        rel_path = os.path.relpath(output_path, output_dir).replace(os.sep, '/')

        # For promoted actors, actor_type should be 'ego' not the original type
        manifest_actor_type = 'ego' if actor_role.lower() == 'ego' else (original_actor_type or actor_role)

        # Get vehicle_model from trajectory metadata (e.g., vehicle.kawasaki.ninja)
        vehicle_model = trajectory[0].get('vehicle_model') if trajectory else None

        manifest_entry = {
            'file': rel_path,
            'route_id': '0',
            'town': town,
            'name': filename.replace('.xml', ''),
            'kind': actor_role,
            'actor_id': actor_id,
            'actor_type': manifest_actor_type,
        }
        # Include vehicle model if available (for spawning correct vehicle type)
        if vehicle_model:
            manifest_entry['model'] = vehicle_model
        
        actors_manifest[role_key].append(manifest_entry)
    
    # Save manifest
    manifest_path = output_dir / 'actors_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(actors_manifest, f, indent=2)
    
    print(f"  Saved actors manifest to {manifest_path}")
    
    return output_files


def _get_carla_map(host: str, port: int, town: str) -> Optional["carla.Map"]:
    """Connect to CARLA and ensure the requested town is loaded."""
    if carla is None:
        print("CARLA not available; skipping global route planner plotting.")
        return None
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    if world.get_map().name != town:
        print(f"Loading {town} on CARLA server for route plotting...")
        world = client.load_world(town)
    return world.get_map()


def _build_global_plan(
    carla_map: "carla.Map",
    trajectory: List[Dict],
    sampling_resolution: float = 2.0,
) -> List[Tuple["carla.Waypoint", object]]:
    """Recreate runtime global plan by chaining trace_route across waypoints."""
    dao = GlobalRoutePlannerDAO(carla_map, sampling_resolution)
    grp = GlobalRoutePlanner(dao)
    grp.setup()
    plan: List[Tuple["carla.Waypoint", object]] = []
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        start_loc = carla.Location(x=start["x"], y=start["y"], z=start["z"])
        end_loc = carla.Location(x=end["x"], y=end["y"], z=end["z"])
        segment = grp.trace_route(start_loc, end_loc)
        if plan and segment:
            segment = segment[1:]  # avoid duplicate seam
        plan.extend(segment)
    return plan


def _save_plan_artifacts(
    actor_id: str,
    debug_dir: Path,
    trajectory: List[Dict],
    plan: List[Tuple["carla.Waypoint", object]],
) -> None:
    """Write plan JSON and a quick visual overlay."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    plan_json = []
    for wp, opt in plan:
        loc = wp.transform.location
        plan_json.append(
            {
                "x": loc.x,
                "y": loc.y,
                "z": loc.z,
                "road_option": getattr(opt, "name", str(opt)),
            }
        )
    with open(debug_dir / f"global_plan_{actor_id}.json", "w") as f:
        json.dump(plan_json, f, indent=2)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot export.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    raw_x = [s["x"] for s in trajectory]
    raw_y = [s["y"] for s in trajectory]
    ax.plot(raw_x, raw_y, "k--", label="input trajectory")

    colors = {
        "LANEFOLLOW": "gray",
        "STRAIGHT": "green",
        "LEFT": "red",
        "RIGHT": "blue",
        "CHANGELANELEFT": "orange",
        "CHANGELANERIGHT": "purple",
    }
    plan_x = []
    plan_y = []
    for wp, opt in plan:
        loc = wp.transform.location
        plan_x.append(loc.x)
        plan_y.append(loc.y)
        opt_name = getattr(opt, "name", str(opt))
        ax.scatter(loc.x, loc.y, s=8, c=colors.get(opt_name, "cyan"), alpha=0.6)
    ax.plot(plan_x, plan_y, "-", color="cyan", label="GlobalRoutePlanner plan", alpha=0.7)

    ax.set_title(f"Global route plan for {actor_id}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(debug_dir / f"global_plan_{actor_id}.png", dpi=200)
    plt.close(fig)


def _route_metrics(route: List[Tuple["carla.Waypoint", object]]) -> Tuple[float, bool, List[str]]:
    """Return total length, whether a turn occurs, and option list for a GRP route."""
    length = 0.0
    has_turn = False
    options: List[str] = []
    if not route:
        return length, has_turn, options
    prev_loc = route[0][0].transform.location
    for wp, opt in route[1:]:
        loc = wp.transform.location
        length += loc.distance(prev_loc)
        prev_loc = loc
        opt_name = getattr(opt, "name", str(opt))
        options.append(opt_name)
        if opt_name.upper() not in ("LANEFOLLOW", "VOID"):
            has_turn = True
    return length, has_turn, options


def _recompute_headings(states: List[Dict]) -> None:
    """Overwrite yaw based on neighbor waypoint direction."""
    if len(states) < 2:
        return

    def heading(a: Dict, b: Dict) -> float:
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        if dx == 0 and dy == 0:
            return a.get("yaw", 0.0)
        return math.degrees(math.atan2(dy, dx))

    for idx, state in enumerate(states):
        if idx == 0:
            yaw = heading(state, states[idx + 1])
        elif idx == len(states) - 1:
            yaw = heading(states[idx - 1], state)
        else:
            yaw = heading(states[idx - 1], states[idx + 1])
        state["yaw"] = yaw


def _max_segment_deviation(
    trajectory: List[Dict],
    start_idx: int,
    end_idx: int,
    start_loc: "carla.Location",
    end_loc: "carla.Location",
) -> float:
    """Maximum lateral deviation of intermediate samples from the line between start/end."""
    if end_idx - start_idx <= 1:
        return 0.0
    sx, sy = start_loc.x, start_loc.y
    ex, ey = end_loc.x, end_loc.y
    dx = ex - sx
    dy = ey - sy
    seg_len_sq = dx * dx + dy * dy
    max_dev = 0.0
    for idx in range(start_idx + 1, end_idx):
        px = trajectory[idx]["x"]
        py = trajectory[idx]["y"]
        if seg_len_sq == 0:
            dist = math.hypot(px - sx, py - sy)
        else:
            t = ((px - sx) * dx + (py - sy) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = sx + t * dx
            proj_y = sy + t * dy
            dist = math.hypot(px - proj_x, py - proj_y)
        if dist > max_dev:
            max_dev = dist
    return max_dev


def _refine_snapping_dp(
    carla_map: "carla.Map",
    trajectory: List[Dict],
    grp: GlobalRoutePlanner,
    radius: float,
    k: int,
    w_deviation: float,
    w_route: float,
    turn_penalty: float,
    skip_penalty: float,
    shape_penalty: float,
    max_skip: int,
    waypoint_step: float = 1.0,
) -> Tuple[List[Dict], List[Tuple["carla.Waypoint", object]], List[Dict]]:
    """
    Search over candidate map waypoints, allowing waypoint skipping, and retain only
    the minimal set of nodes that still captures genuine lane/road changes.
    """
    if not trajectory:
        return [], [], []

    _recompute_headings(trajectory)

    def yaw_diff_deg(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    heading_thresh = 30.0
    straight_thresh = 15.0
    lane_change_penalty = 30.0
    straight_turn_penalty = 30.0

    all_wps = carla_map.generate_waypoints(waypoint_step)
    if not all_wps:
        raise RuntimeError("No waypoints generated from CARLA map.")
    coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in all_wps])
    rad2 = radius * radius
    n = len(trajectory)

    def candidate_indices(idx: int, xy: np.ndarray) -> List[int]:
        traj_yaw = trajectory[idx]["yaw"]
        diff = coords - xy[None, :]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idxs = np.where(dist2 <= rad2)[0]
        if idxs.size == 0:
            idxs = np.array([int(np.argmin(dist2))])
        filtered = []
        for ci in idxs:
            wp_yaw = all_wps[ci].transform.rotation.yaw
            if yaw_diff_deg(wp_yaw, traj_yaw) <= heading_thresh:
                filtered.append((ci, dist2[ci]))
        if not filtered:
            filtered = [(ci, dist2[ci]) for ci in idxs]
        filtered.sort(key=lambda x: x[1])
        return [ci for ci, _ in filtered[:k]]

    candidates_per_step: List[List[int]] = []
    for idx, state in enumerate(trajectory):
        xy = np.array([state["x"], state["y"]])
        candidates_per_step.append(candidate_indices(idx, xy))

    best_cost: List[Dict[int, float]] = [dict() for _ in range(n)]
    backref: List[Dict[int, Optional[Tuple[int, int]]]] = [dict() for _ in range(n)]

    for ci in candidates_per_step[0]:
        dev = np.linalg.norm(np.array([trajectory[0]["x"], trajectory[0]["y"]]) - coords[ci])
        best_cost[0][ci] = w_deviation * dev
        backref[0][ci] = None

    for i in range(1, n):
        obs = np.array([trajectory[i]["x"], trajectory[i]["y"]])
        traj_yaw_i = trajectory[i]["yaw"]
        for ci in candidates_per_step[i]:
            dev = np.linalg.norm(obs - coords[ci])
            base_cost = w_deviation * dev
            best_val = None
            best_prev: Optional[Tuple[int, int]] = None
            j_start = max(0, i - max_skip)
            for j in range(i - 1, j_start - 1, -1):
                if not best_cost[j]:
                    continue
                traj_yaw_j = trajectory[j]["yaw"]
                delta_heading = yaw_diff_deg(traj_yaw_i, traj_yaw_j)
                is_straight_traj = delta_heading < straight_thresh
                gap = i - j
                for prev_ci, prev_cost in best_cost[j].items():
                    prev_wp = all_wps[prev_ci]
                    curr_wp = all_wps[ci]
                    route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
                    if not route:
                        continue
                    route_len, has_turn, _ = _route_metrics(route)
                    skip_cost = skip_penalty * max(0, gap - 1)
                    shape_cost = 0.0
                    if gap > 1:
                        shape_cost = shape_penalty * _max_segment_deviation(
                            trajectory, j, i, prev_wp.transform.location, curr_wp.transform.location
                        )
                    same_lane = (
                        prev_wp.road_id == curr_wp.road_id and prev_wp.lane_id == curr_wp.lane_id
                    )
                    total = (
                        prev_cost
                        + base_cost
                        + w_route * route_len
                        + (turn_penalty if has_turn else 0.0)
                        + skip_cost
                        + shape_cost
                        + (0.0 if same_lane else lane_change_penalty)
                    )
                    if is_straight_traj and has_turn:
                        total += straight_turn_penalty
                    if best_val is None or total < best_val:
                        best_val = total
                        best_prev = (j, prev_ci)
            if best_val is not None:
                best_cost[i][ci] = best_val
                backref[i][ci] = best_prev

    if not best_cost[-1]:
        raise RuntimeError("Failed to align trajectory to map (no terminal state).")

    end_ci = min(best_cost[-1], key=lambda ci: best_cost[-1][ci])
    path: List[Tuple[int, int]] = []
    idx = n - 1
    ci = end_ci
    while True:
        path.append((idx, ci))
        prev = backref[idx].get(ci)
        if prev is None:
            break
        idx, ci = prev
    path.reverse()

    state_records: List[Dict] = []
    for sample_idx, cand_idx in path:
        wp = all_wps[cand_idx]
        loc = wp.transform.location
        yaw = wp.transform.rotation.yaw
        refined = dict(trajectory[sample_idx])
        refined.update({"x": loc.x, "y": loc.y, "z": loc.z, "yaw": yaw})
        state_records.append(
            {
                "sample_index": sample_idx,
                "candidate_index": cand_idx,
                "wp": wp,
                "state": refined,
            }
        )

    if len(state_records) < 2:
        raise RuntimeError("Refined trajectory must contain at least start and end waypoints.")

    segment_data: List[Dict] = []
    for prev_state, curr_state in zip(state_records[:-1], state_records[1:]):
        prev_wp = prev_state["wp"]
        curr_wp = curr_state["wp"]
        route = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
        route_len, has_turn, options = _route_metrics(route)
        segment_data.append(
            {
                "route": route,
                "options": options,
                "has_turn": has_turn,
                "start_wp": prev_wp,
                "end_wp": curr_wp,
            }
        )

    keep_flags = [False] * len(state_records)
    keep_flags[0] = True
    keep_flags[-1] = True
    for mid in range(1, len(state_records) - 1):
        prev_state = state_records[mid - 1]
        curr_state = state_records[mid]
        next_state = state_records[mid + 1]
        prev_wp = prev_state["wp"]
        curr_wp = curr_state["wp"]
        next_wp = next_state["wp"]

        lane_change = (
            curr_wp.road_id != prev_wp.road_id
            or curr_wp.lane_id != prev_wp.lane_id
            or next_wp.road_id != curr_wp.road_id
            or next_wp.lane_id != curr_wp.lane_id
        )
        turn_near = segment_data[mid - 1]["has_turn"] or segment_data[mid]["has_turn"]

        if lane_change or turn_near:
            keep_flags[mid] = True
            continue

        merged_route = grp.trace_route(prev_wp.transform.location, next_wp.transform.location)
        if not merged_route:
            keep_flags[mid] = True
            continue
        _, merged_turn, _ = _route_metrics(merged_route)
        if merged_turn:
            keep_flags[mid] = True
            continue
        # else drop this waypoint

    compressed_states = [state for flag, state in zip(keep_flags, state_records) if flag]
    if len(compressed_states) < 2:
        compressed_states = state_records

    final_traj: List[Dict] = [state["state"] for state in compressed_states]
    _recompute_headings(final_traj)
    final_mapping: List[Dict] = []
    for state in compressed_states:
        wp = state["wp"]
        sample_idx = state["sample_index"]
        final_mapping.append(
            {
                "sample_index": int(sample_idx),
                "candidate_index": int(state["candidate_index"]),
                "orig": {
                    "x": trajectory[sample_idx]["x"],
                    "y": trajectory[sample_idx]["y"],
                    "z": trajectory[sample_idx]["z"],
                },
                "snapped": {"x": wp.transform.location.x, "y": wp.transform.location.y, "z": wp.transform.location.z},
                "road_id": wp.road_id,
                "lane_id": wp.lane_id,
                "s": getattr(wp, "s", None),
            }
        )

    plan: List[Tuple["carla.Waypoint", object]] = []
    for prev_state, curr_state in zip(compressed_states[:-1], compressed_states[1:]):
        prev_wp = prev_state["wp"]
        curr_wp = curr_state["wp"]
        route_seg = grp.trace_route(prev_wp.transform.location, curr_wp.transform.location)
        if plan and route_seg:
            route_seg = route_seg[1:]
        plan.extend(route_seg)

    return final_traj, plan, final_mapping


def package_routes_zip(output_dir: Path, zip_path: Optional[Path] = None) -> Path:
    """
    Package the generated XML routes (and manifest, if present) into a ZIP
    that can be consumed by tools/setup_scenario_from_zip.py.
    """
    output_dir = output_dir.resolve()
    if zip_path is None:
        zip_path = output_dir.with_suffix(".zip")
    else:
        zip_path = zip_path.resolve()

    xml_files = sorted(output_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"No XML route files found in {output_dir}")

    manifest = output_dir / "actors_manifest.json"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for xml_file in xml_files:
            archive.write(xml_file, arcname=xml_file.name)
        if manifest.exists():
            archive.write(manifest, arcname=manifest.name)

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert SafeBench trajectory logs to CoLMDriver XML format'
    )
    parser.add_argument(
        'trajectory_file',
        type=Path,
        help='Path to NPZ or CSV trajectory file, OR a directory containing trajectory files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory (default: trajectory_file parent / xml_routes)'
    )
    parser.add_argument(
        '--town',
        type=str,
        default='Town05',
        help='CARLA town name (default: Town05)'
    )
    parser.add_argument(
        '--promote-to-ego',
        type=str,
        nargs='+',
        default=[],
        metavar='ACTOR_ID',
        help='Treat these NPC actors as additional ego agents (run alignment, output as ego role). '
             'E.g., --promote-to-ego adv_agent_0 static_prop_0'
    )
    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Downsample waypoints: keep every Nth point (default: 1, keep all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--zip',
        action='store_true',
        help='Also package the generated routes into a ZIP for setup_scenario_from_zip.py'
    )
    parser.add_argument(
        '--zip-output',
        type=Path,
        help='Optional path for the ZIP (default: <output_dir>.zip)'
    )
    parser.add_argument(
        '--carla-host',
        type=str,
        default='127.0.0.1',
        help='CARLA server host for plotting with GlobalRoutePlanner (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--carla-port',
        type=int,
        default=2000,
        help='CARLA server port for plotting with GlobalRoutePlanner (default: 2000)'
    )
    parser.add_argument(
        '--plan-sampling',
        type=float,
        default=2.0,
        help='Sampling resolution (m) for GlobalRoutePlanner (default: 2.0)'
    )
    parser.add_argument(
        '--plot-plan',
        dest='plot_plan',
        action='store_true',
        help='Plot and save GlobalRoutePlanner plan overlays (default: on)'
    )
    parser.add_argument(
        '--no-plot-plan',
        dest='plot_plan',
        action='store_false',
        help='Disable GlobalRoutePlanner plotting'
    )
    parser.add_argument(
        '--refine-snap',
        dest='refine_snap',
        action='store_true',
        help='Use dynamic snapping to minimize detours when building plans (default: on)'
    )
    parser.add_argument(
        '--no-refine-snap',
        dest='refine_snap',
        action='store_false',
        help='Disable snapping refinement'
    )
    parser.add_argument(
        '--snap-radius',
        type=float,
        default=1.5,
        help='Search radius (m) for candidate map waypoints'
    )
    parser.add_argument(
        '--snap-k',
        type=int,
        default=5,
        help='Max candidate waypoints per trajectory point'
    )
    parser.add_argument(
        '--snap-w-deviation',
        type=float,
        default=1.0,
        help='Weight for deviation from original waypoint during snapping'
    )
    parser.add_argument(
        '--snap-w-route',
        type=float,
        default=1.0,
        help='Weight for GlobalRoutePlanner route length between snapped points'
    )
    parser.add_argument(
        '--snap-turn-penalty',
        type=float,
        default=5.0,
        help='Penalty added when a segment introduces LEFT/RIGHT/CHANGELANE options'
    )
    parser.add_argument(
        '--snap-skip-penalty',
        type=float,
        default=0.5,
        help='Penalty per skipped waypoint between kept nodes'
    )
    parser.add_argument(
        '--snap-shape-penalty',
        type=float,
        default=0.2,
        help='Penalty multiplier for deviation of skipped samples from the snapped segment'
    )
    parser.add_argument(
        '--snap-max-skip',
        type=int,
        default=20,
        help='Maximum number of original waypoints that can be skipped in one jump'
    )
    parser.add_argument(
        '--ego-route-plan',
        type=Path,
        help='Optional coarse ego route CSV to override ego trajectory (CSV columns x,y,z[,yaw])'
    )
    parser.set_defaults(plot_plan=True, refine_snap=True)
    
    args = parser.parse_args()
    
    # Validate input - can be file or directory
    input_path = args.trajectory_file.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")
    
    # If directory, auto-detect trajectory file, manifest, and coarse route
    if input_path.is_dir():
        print(f"Directory provided, auto-detecting files in: {input_path}")
        # Find trajectory file (prefer CSV over NPZ)
        csv_files = list(input_path.glob("*_trajectory.csv"))
        npz_files = list(input_path.glob("*_trajectory.npz"))
        if csv_files:
            trajectory_file = csv_files[0]
        elif npz_files:
            trajectory_file = npz_files[0]
        else:
            raise FileNotFoundError(f"No trajectory file (*_trajectory.csv or *_trajectory.npz) found in {input_path}")
        print(f"  Found trajectory: {trajectory_file.name}")
        
        # Auto-detect coarse route if not specified
        if args.ego_route_plan is None:
            coarse_routes = list(input_path.glob("*_coarse_route.csv"))
            if coarse_routes:
                args.ego_route_plan = coarse_routes[0]
                print(f"  Found coarse route: {args.ego_route_plan.name}")
        
        # Manifest will be auto-detected later based on trajectory file name
    else:
        trajectory_file = input_path
    
    # Determine output directory
    if args.output:
        output_dir = args.output.resolve()
    else:
        output_dir = trajectory_file.parent / 'xml_routes'
    
    # Load trajectories
    print("Loading trajectories...")
    metadata = None
    if trajectory_file.suffix == '.npz':
        trajectories_npz, metadata = load_npz_trajectories(trajectory_file)
        trajectories = npz_to_dict_trajectories(trajectories_npz, metadata)
    elif trajectory_file.suffix == '.csv':
        trajectories = load_csv_trajectories(trajectory_file)
        # Try to load manifest from corresponding JSON
        manifest_path = trajectory_file.parent / trajectory_file.name.replace('_trajectory.csv', '_manifest.json')
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                metadata = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {trajectory_file.suffix}")

    # Auto-detect town from manifest if not explicitly specified (or using default)
    town = args.town
    manifest_town = metadata.get('town') if metadata else None
    if manifest_town:
        # If user explicitly specified a different town than manifest, warn
        if args.town != 'Town05' and args.town != manifest_town:
            print(f"  Warning: --town {args.town} differs from manifest town '{manifest_town}'")
            print(f"  Using explicitly specified --town {args.town}")
        elif args.town == 'Town05':  # Default value - use manifest
            town = manifest_town
            print(f"  Auto-detected town from manifest: {town}")
    elif args.town == 'Town05':
        print(f"  Note: Using default town '{town}'. No town found in manifest.")
    
    print(f"\n=== SafeBench Trajectory to CoLMDriver XML Converter ===")
    print(f"Input file: {trajectory_file}")
    print(f"Output dir: {output_dir}")
    print(f"Town: {town}")
    print(f"Downsample factor: {args.downsample}")
    print()

    # Optionally override ego trajectory with a coarse route plan
    if args.ego_route_plan:
        ego_plan = load_route_plan_csv(args.ego_route_plan)
        if ego_plan:
            trajectories['ego_0'] = ego_plan
            print(f"  Replaced ego trajectory with route plan from {args.ego_route_plan}")
        else:
            print(f"  Warning: ego route plan {args.ego_route_plan} was empty; keeping logged ego trajectory.")
    
    print(f"  Loaded {len(trajectories)} actor trajectories")
    for actor_id, traj in trajectories.items():
        print(f"    - {actor_id}: {len(traj)} states")

    # Optional snapping refinement using the runtime GlobalRoutePlanner
    if args.refine_snap:
        if carla is None or GlobalRoutePlannerDAO is None or GlobalRoutePlanner is None:
            print("Snapping refinement requested, but CARLA Python API is not available.")
        else:
            print("\nRefining waypoint snapping to minimize detours...")
            carla_map = _get_carla_map(args.carla_host, args.carla_port, town)
            if carla_map:
                grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, args.plan_sampling))
                grp.setup()
                # Build set of actors to treat as ego (original egos + promoted NPCs)
                promote_set = set(p.lower() for p in getattr(args, 'promote_to_ego', []))
                for actor_id, traj in list(trajectories.items()):
                    if not traj:
                        continue
                    is_ego = actor_id.lower().startswith("ego") or actor_id.lower() in promote_set
                    if not is_ego:
                        continue
                    try:
                        refined_traj, refined_plan, mapping = _refine_snapping_dp(
                            carla_map,
                            traj,
                            grp,
                            radius=args.snap_radius,
                            k=args.snap_k,
                            w_deviation=args.snap_w_deviation,
                            w_route=args.snap_w_route,
                            turn_penalty=args.snap_turn_penalty,
                            skip_penalty=args.snap_skip_penalty,
                            shape_penalty=args.snap_shape_penalty,
                            max_skip=args.snap_max_skip,
                        )
                        trajectories[actor_id] = refined_traj
                        # Save mapping and plan overlay under a debug folder
                        debug_dir = output_dir / "plan_debug"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        with open(debug_dir / f"snapping_mapping_{actor_id}.json", "w") as f:
                            json.dump(mapping, f, indent=2)
                        _save_plan_artifacts(actor_id, debug_dir, refined_traj, refined_plan)
                        print(f"  Refined snapping for {actor_id} (saved mapping and plan overlay)")
                    except Exception as e:  # pragma: no cover - defensive
                        print(f"  Failed to refine snapping for {actor_id}: {e}")
    
    # Build promote_set from args
    promote_set = set(p.lower() for p in getattr(args, 'promote_to_ego', []))
    if promote_set:
        print(f"  Promoting to ego: {', '.join(sorted(promote_set))}")
    
    # Convert to XML
    print("\nConverting to XML routes...")
    output_files = trajectories_to_xml(
        trajectories=trajectories,
        town=town,
        output_dir=output_dir,
        downsample_factor=args.downsample,
        promote_set=promote_set
    )
    
    print(f"\n✓ Successfully created {len(output_files)} XML route files")
    print(f"✓ Routes ready for CoLMDriver evaluation at: {output_dir}")

    # Optional: visualize the runtime GlobalRoutePlanner path that will be used
    if args.plot_plan:
        if carla is None or GlobalRoutePlannerDAO is None or GlobalRoutePlanner is None:
            print("CARLA route plotting requested, but CARLA Python API is not available.")
        else:
            print("\nBuilding GlobalRoutePlanner paths for visualization...")
            carla_map = _get_carla_map(args.carla_host, args.carla_port, town)
            if carla_map:
                debug_dir = output_dir / "plan_debug"
                for actor_id, traj in trajectories.items():
                    if not traj:
                        continue
                    try:
                        plan = _build_global_plan(carla_map, traj, sampling_resolution=args.plan_sampling)
                        _save_plan_artifacts(actor_id, debug_dir, traj, plan)
                        print(f"  Saved plan overlay for {actor_id}")
                    except Exception as e:  # pragma: no cover - defensive logging
                        print(f"  Failed to build plan for {actor_id}: {e}")
    
    zip_path = None
    if args.zip:
        zip_path = package_routes_zip(output_dir, args.zip_output)
        print(f"\n✓ Packaged routes ZIP for setup_scenario_from_zip: {zip_path}")
        print("\nTo install the routes:")
        print(f"  python tools/setup_scenario_from_zip.py {zip_path} --scenario-name <name> --overwrite")
    else:
        print("\n(To create a packager-ready ZIP, rerun with --zip)")

    # Show how to use
    print(f"\nTo run evaluation with CoLMDriver:")
    if zip_path:
        print(f"  python tools/run_custom_eval.py --zip {zip_path}")
    else:
        print(f"  python tools/run_custom_eval.py --routes-dir <prepared_dir>")


if __name__ == '__main__':
    main()
