#!/usr/bin/env python3
"""
Convert a V2XPnP-style YAML sequence into CARLA leaderboard-ready XML routes.

Outputs:
  - ego_route.xml (optional) with the ego trajectory as role="ego".
  - actors/<name>.xml for every non-ego object, role="npc", snap_to_road="false".
  - actors_manifest.json describing all actors (route_id, model, speed, etc.).
  - Optional GIF visualizing the replay frames.

Usage (typical):
  python v2xpnp/scripts/yaml_to_carla_log.py \\
      --scenario-dir /data2/marco/CoLMDriver/v2xpnp/Sample_Dataset/2023-03-17-16-12-12_3_0 \\
      --subdir -1 \\
      --town ucla_v2 \\
      --out-dir /data2/marco/CoLMDriver/v2xpnp/out_log_replay \\
      --gif

Notes:
  - All custom actors are marked snap_to_road="false" by default; override with --snap-to-road
    if you want snapping.
  - Apply global transforms with --tx/--ty/--tz and --yaw-deg to align to the CARLA map.
  - If --subdir is omitted (or set to "all") and multiple subfolders exist, all YAML subfolders
    are used for actor locations. Only non-negative subfolders are treated as ego vehicles;
    negative subfolders contribute actors only.
  - Use --spawn-viz with --xodr and/or --map-pkl/--use-carla-map to visualize spawn vs aligned points.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import xml.etree.ElementTree as ET
import pickle  # optional; used for map caches

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import patches, transforms
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    plt = None
    patches = None
    transforms = None
    imageio = None

# Optional CARLA client (only needed when --use-carla-map)
try:  # pragma: no cover
    import carla  # type: ignore
except Exception:
    carla = None


# ---------------------- Helpers ---------------------- #

def list_yaml_timesteps(folder: Path) -> List[Path]:
    """Return sorted YAML files in a folder (by stem as int if possible)."""
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".yaml"]
    def sort_key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.stem
    return sorted(files, key=sort_key)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def yaw_from_angle(angle: Sequence[float] | None) -> float:
    """Dataset provides [roll?, yaw?, pitch?]; middle component works for provided samples."""
    if isinstance(angle, Sequence) and len(angle) >= 2:
        return float(angle[1])
    if isinstance(angle, Sequence) and angle:
        return float(angle[-1])
    return 0.0


def yaw_from_pose(pose: Sequence[float] | None) -> float:
    """true_ego_pose is [x, y, z, roll, yaw, pitch]."""
    if isinstance(pose, Sequence) and len(pose) >= 5:
        return float(pose[4])
    return 0.0


def apply_se2(point: Tuple[float, float], yaw_deg: float, tx: float, ty: float, flip_y: bool = False) -> Tuple[float, float]:
    """Rotate then translate (yaw in degrees), with optional Y flip."""
    rad = math.radians(yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    x, y = point
    if flip_y:
        y = -y
    xr = c * x - s * y + tx
    yr = s * x + c * y + ty
    return xr, yr


def invert_se2(point: Tuple[float, float], yaw_deg: float, tx: float, ty: float, flip_y: bool = False) -> Tuple[float, float]:
    """Inverse of apply_se2 (undo translation, rotation, and optional Y flip)."""
    x = point[0] - tx
    y = point[1] - ty
    rad = math.radians(-yaw_deg)
    c, s = math.cos(rad), math.sin(rad)
    xr = c * x - s * y
    yr = s * x + c * y
    if flip_y:
        yr = -yr
    return xr, yr


def euclid3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """3D euclidean distance (Python 3.7-compatible; math.dist is 3.8+)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_vehicle_type(obj_type: str | None) -> bool:
    """Check if obj_type represents a vehicle (not trash, signs, etc.)."""
    if not obj_type:
        return True  # Default to vehicle
    ot = obj_type.lower()
    
    # Exclude non-vehicle, non-pedestrian types (static props)
    excluded_keywords = [
        "trash", "can", "barrel", "cone", "barrier",
        "sign", "pole", "light", "bench", "tree", "plant"
    ]
    
    for keyword in excluded_keywords:
        if keyword in ot:
            return False
    
    # Allow vehicles and pedestrians
    return True


def is_pedestrian_type(obj_type: str | None) -> bool:
    """Check if obj_type represents a pedestrian/walker."""
    if not obj_type:
        return False
    ot = obj_type.lower()
    
    pedestrian_keywords = ["pedestrian", "walker", "person", "people"]
    
    for keyword in pedestrian_keywords:
        if keyword in ot:
            return True
    
    return False


# Walker blueprint policy: exclude child-sized models by default.
ADULT_WALKER_BLUEPRINTS = [
    "walker.pedestrian.0002",
    "walker.pedestrian.0003",
    "walker.pedestrian.0004",
    "walker.pedestrian.0005",
    "walker.pedestrian.0006",
    "walker.pedestrian.0007",
    "walker.pedestrian.0008",
    "walker.pedestrian.0009",
    "walker.pedestrian.0011",
    "walker.pedestrian.0012",
    "walker.pedestrian.0013",
    "walker.pedestrian.0014",
]
CHILD_WALKER_BLUEPRINTS = {
    "walker.pedestrian.0001",
    "walker.pedestrian.0010",
}


def _is_child_walker_blueprint(bp_id: str) -> bool:
    return str(bp_id) in CHILD_WALKER_BLUEPRINTS


def map_obj_type(obj_type: str | None) -> str:
    """Map dataset obj_type to a CARLA 0.9.12 blueprint (vehicle or walker)."""
    # Define blueprint pools for random selection (excluding children)
    walker_blueprints = ADULT_WALKER_BLUEPRINTS
    
    bus_blueprints = [
        "vehicle.volkswagen.t2",
        "vehicle.mitsubishi.fusorosa",
    ]
    
    truck_blueprints = [
        "vehicle.carlamotors.carlacola",
    ]
    
    firetruck_blueprints = [
        "vehicle.carlamotors.firetruck",
    ]
    
    van_blueprints = [
        "vehicle.mercedes.sprinter",
        "vehicle.volkswagen.t2",
    ]
    
    ambulance_blueprints = [
        "vehicle.ford.ambulance",
    ]
    
    police_blueprints = [
        "vehicle.dodge.charger_police",
        "vehicle.dodge.charger_police_2020",
    ]
    
    motorcycle_blueprints = [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
        "vehicle.vespa.zx125",
    ]
    
    bicycle_blueprints = [
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
        "vehicle.bh.crossbike",
    ]
    
    suv_blueprints = [
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.nissan.patrol",
        "vehicle.nissan.patrol_2021",
        "vehicle.lincoln.mkz_2020",
    ]
    
    # General car/sedan blueprints (default pool)
    car_blueprints = [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.audi.tt",
        "vehicle.audi.etron",
        "vehicle.bmw.grandtourer",
        "vehicle.chevrolet.impala",
        "vehicle.citroen.c3",
        "vehicle.dodge.charger_2020",
        "vehicle.ford.crown",
        "vehicle.ford.mustang",
        "vehicle.lincoln.mkz_2017",
        "vehicle.mercedes.coupe",
        "vehicle.mercedes.coupe_2020",
        "vehicle.mini.cooper_s",
        "vehicle.mini.cooper_s_2021",
        "vehicle.nissan.micra",
        "vehicle.seat.leon",
        "vehicle.toyota.prius",
        "vehicle.volkswagen.t2",
    ]
    
    if not obj_type:
        return random.choice(car_blueprints)
    ot = obj_type.lower()
    
    # Pedestrians/Walkers
    if is_pedestrian_type(obj_type):
        return random.choice(walker_blueprints)
    
    # Buses and large vehicles
    if "bus" in ot:
        return random.choice(bus_blueprints)
    
    # Trucks and vans
    if "truck" in ot:
        if "fire" in ot:
            return random.choice(firetruck_blueprints)
        return random.choice(truck_blueprints)
    if "van" in ot or "sprinter" in ot:
        return random.choice(van_blueprints)
    
    # Emergency vehicles
    if "ambulance" in ot:
        return random.choice(ambulance_blueprints)
    if "police" in ot:
        return random.choice(police_blueprints)
    
    # Motorcycles
    if "motor" in ot or "motorcycle" in ot:
        return random.choice(motorcycle_blueprints)
    if "bike" in ot and "bicycle" not in ot:
        return random.choice(motorcycle_blueprints)
    
    # Bicycles (with rider)
    if "bicycle" in ot or "cycl" in ot:
        return random.choice(bicycle_blueprints)
    
    # SUVs and larger cars
    if "suv" in ot or "jeep" in ot:
        return random.choice(suv_blueprints)
    if "patrol" in ot:
        return random.choice(suv_blueprints)
    
    # Sedans and cars (default category)
    return random.choice(car_blueprints)


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


# ---------------------- Spawn Preprocess ---------------------- #

@dataclass
class SpawnCandidate:
    dx: float
    dy: float
    source: str
    base_cost: float
    valid: bool = False
    reason: str | None = None
    spawn_loc: Optional[Tuple[float, float, float]] = None
    dz: float = 0.0
    z_source: Optional[str] = None


def _path_distance(traj: List[Waypoint]) -> float:
    dist = 0.0
    for a, b in zip(traj, traj[1:]):
        dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
    return dist


def _classify_actor_kind(traj: List[Waypoint], obj_type_raw: str) -> Tuple[str, bool]:
    is_pedestrian = is_pedestrian_type(obj_type_raw)
    if is_pedestrian:
        kind = "walker"
        if len(traj) >= 2 and _path_distance(traj) < 0.5:
            kind = "walker_static"
        return kind, True

    kind = "npc"
    if len(traj) <= 1:
        kind = "static"
    elif len(traj) >= 2 and _path_distance(traj) < 0.5:
        kind = "static"
    return kind, False


def _actor_radius(kind: str, length: Optional[float], width: Optional[float], model: str) -> float:
    if length is not None or width is not None:
        dim = max(float(length or 0.0), float(width or 0.0))
        if dim > 0.0:
            return max(0.2, 0.5 * dim)
    model_lower = str(model or "").lower()
    if kind.startswith("walker"):
        return 0.4
    if "bicycle" in model_lower or "cycl" in model_lower:
        return 0.9
    if kind == "static":
        return 0.6
    return 1.5


def _ensure_times(traj: List[Waypoint], times: List[float] | None, default_dt: float) -> List[float]:
    if times and len(times) == len(traj):
        return [float(t) for t in times]
    return [float(i) * float(default_dt) for i in range(len(traj))]


def _build_time_grid(all_times: List[float], sample_dt: float) -> List[float]:
    if not all_times:
        return [0.0]
    max_time = max(all_times)
    if max_time <= 0.0:
        return [0.0]
    sample_dt = max(0.05, float(sample_dt))
    n = int(math.floor(max_time / sample_dt)) + 1
    return [i * sample_dt for i in range(n + 1)]


def _sample_positions(
    traj: List[Waypoint],
    times: List[float],
    sample_times: List[float],
    always_active: bool,
) -> List[Optional[Tuple[float, float]]]:
    if not traj:
        return [None for _ in sample_times]

    if len(traj) == 1 or always_active:
        pos = (traj[0].x, traj[0].y)
        return [pos for _ in sample_times]

    positions: List[Optional[Tuple[float, float]]] = []
    idx = 0
    last = len(times) - 1
    for t in sample_times:
        if t < times[0] or t > times[-1]:
            positions.append(None)
            continue
        while idx + 1 < last and times[idx + 1] < t:
            idx += 1
        if idx + 1 >= len(times):
            positions.append((traj[-1].x, traj[-1].y))
            continue
        t0 = times[idx]
        t1 = times[idx + 1]
        if t1 <= t0:
            alpha = 0.0
        else:
            alpha = (t - t0) / (t1 - t0)
        x = traj[idx].x + (traj[idx + 1].x - traj[idx].x) * alpha
        y = traj[idx].y + (traj[idx + 1].y - traj[idx].y) * alpha
        positions.append((x, y))
    return positions


def _resolve_ground_z(world, location) -> Optional[float]:
    if world is None:
        return None
    ground_projection = getattr(world, "ground_projection", None)
    if callable(ground_projection):
        try:
            probe = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            result = ground_projection(probe, 100.0)
            if result is not None:
                if hasattr(result, "z"):
                    return float(result.z)
                if isinstance(result, (tuple, list)) and result:
                    first = result[0]
                    if hasattr(first, "z"):
                        return float(first.z)
        except Exception:
            pass
    cast_ray = getattr(world, "cast_ray", None)
    if callable(cast_ray):
        try:
            start = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z + 50.0,
            )
            end = carla.Location(
                x=location.x,
                y=location.y,
                z=location.z - 50.0,
            )
            hits = cast_ray(start, end)
            if hits:
                best_z = None
                best_dist = None
                for hit in hits:
                    hit_loc = getattr(hit, "location", None) or getattr(hit, "point", None)
                    if hit_loc is None:
                        continue
                    dz = abs(float(hit_loc.z) - float(location.z))
                    if best_dist is None or dz < best_dist:
                        best_dist = dz
                        best_z = float(hit_loc.z)
                if best_z is not None:
                    return best_z
        except Exception:
            pass
    return None


def _candidate_z_offsets(
    world,
    world_map,
    base_loc,
    normalize_z: bool,
) -> List[Tuple[float, str]]:
    """Return (dz, source) candidates ordered by |dz|, starting with authored z."""
    offsets: List[Tuple[float, str]] = [(0.0, "authored")]
    if not normalize_z:
        return offsets

    candidates: List[Tuple[str, float]] = []
    ground_z = _resolve_ground_z(world, base_loc) if world is not None else None
    if ground_z is not None:
        candidates.append(("ground", float(ground_z)))
    if world_map is not None:
        try:
            wp_any = world_map.get_waypoint(
                base_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Any,
            )
        except Exception:
            wp_any = None
        if wp_any is not None:
            candidates.append(("waypoint_any", float(wp_any.transform.location.z)))

    orig_z = float(base_loc.z)
    for label, z in candidates:
        offsets.append((float(z) - orig_z, label))

    uniq: List[Tuple[float, str]] = []
    seen = set()
    for dz, label in offsets:
        key = round(float(dz), 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((float(dz), label))

    uniq.sort(key=lambda it: abs(it[0]))
    return uniq


def _try_spawn_candidate(
    world,
    world_map,
    blueprint,
    base_wp: Waypoint,
    cand: SpawnCandidate,
    normalize_z: bool,
) -> None:
    base_loc = carla.Location(
        x=base_wp.x + cand.dx,
        y=base_wp.y + cand.dy,
        z=base_wp.z,
    )
    z_offsets = _candidate_z_offsets(world, world_map, base_loc, normalize_z)

    first_loc = None
    first_dz = 0.0
    first_src = None
    last_exc: Optional[Exception] = None

    for dz, z_src in z_offsets:
        spawn_loc = carla.Location(
            x=base_loc.x,
            y=base_loc.y,
            z=base_loc.z + dz,
        )
        if first_loc is None:
            first_loc = spawn_loc
            first_dz = float(dz)
            first_src = z_src
        spawn_tf = carla.Transform(
            spawn_loc,
            carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw, roll=base_wp.roll),
        )
        actor = None
        try:
            actor = world.try_spawn_actor(blueprint, spawn_tf)
        except Exception as exc:
            last_exc = exc
            continue
        if actor is not None:
            cand.valid = True
            cand.spawn_loc = (float(spawn_loc.x), float(spawn_loc.y), float(spawn_loc.z))
            cand.dz = float(dz)
            cand.z_source = z_src
            try:
                actor.destroy()
            except Exception:
                pass
            return

    cand.valid = False
    if first_loc is not None:
        cand.spawn_loc = (float(first_loc.x), float(first_loc.y), float(first_loc.z))
        cand.dz = float(first_dz)
        cand.z_source = first_src
    if last_exc is not None:
        cand.reason = f"spawn_exception: {last_exc}"
    else:
        cand.reason = "spawn_failed"


def _lane_type_value(name: str):
    return getattr(carla.LaneType, name, None) if carla is not None else None


def _generate_spawn_candidates(
    base_wp: Waypoint,
    role: str,
    world_map,
    max_shift: float,
    grid_steps: List[float],
    lateral_margin: float,
    random_samples: int = 0,
    rng: Optional[random.Random] = None,
) -> List[SpawnCandidate]:
    candidates: Dict[Tuple[int, int], SpawnCandidate] = {}

    def _add_candidate(dx: float, dy: float, source: str, bias: float) -> None:
        dist = math.hypot(dx, dy)
        if dist > max_shift:
            return
        key = (int(round(dx * 100)), int(round(dy * 100)))
        base_cost = dist + bias
        existing = candidates.get(key)
        if existing is None or base_cost < existing.base_cost:
            candidates[key] = SpawnCandidate(dx=dx, dy=dy, source=source, base_cost=base_cost)

    _add_candidate(0.0, 0.0, "authored", 0.0)

    # Micro-jitter grid around authored pose
    for dx in grid_steps:
        for dy in grid_steps:
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                continue
            _add_candidate(dx, dy, "grid", 0.05)

    # Radial rings to increase coverage without biasing too far
    if max_shift > 0.0:
        ring_radii = []
        step = 0.5
        r = step
        while r <= max_shift + 1e-6:
            ring_radii.append(round(r, 2))
            r += step
        angles = [i * 30 for i in range(12)]
        for r in ring_radii:
            for ang in angles:
                rad = math.radians(float(ang))
                dx = r * math.cos(rad)
                dy = r * math.sin(rad)
                _add_candidate(dx, dy, f"ring_{r:.1f}", 0.08 + 0.01 * r)

    if world_map is None:
        # Random offsets (if requested)
        if random_samples > 0:
            rng = rng or random.Random(0)
            for _ in range(random_samples):
                r = max_shift * math.sqrt(rng.random())
                theta = 2.0 * math.pi * rng.random()
                dx = r * math.cos(theta)
                dy = r * math.sin(theta)
                _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)
        return list(candidates.values())

    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)

    lane_candidates: List[Tuple[str, object]] = []
    if role in ("npc", "static"):
        for lane_name in ("Driving", "Shoulder", "Parking"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))
    else:
        for lane_name in ("Sidewalk", "Shoulder"):
            lane_val = _lane_type_value(lane_name)
            if lane_val is not None:
                lane_candidates.append((lane_name, lane_val))

    driving_wp = None
    for lane_name, lane_val in lane_candidates:
        try:
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=lane_val)
        except Exception:
            wp = None
        if wp is None:
            continue
        dx = float(wp.transform.location.x) - base_wp.x
        dy = float(wp.transform.location.y) - base_wp.y
        _add_candidate(dx, dy, f"lane_{lane_name.lower()}", 0.1)
        if lane_name == "Driving":
            driving_wp = wp

    # Lateral offsets from driving lane (captures shoulder-like positions)
    if driving_wp is not None:
        try:
            yaw = math.radians(float(driving_wp.transform.rotation.yaw))
            right = (math.sin(yaw), -math.cos(yaw))
            lane_width = getattr(driving_wp, "lane_width", 3.5) or 3.5
            for mult in (0.5, 1.0, 1.5):
                offset = mult * float(lane_width) + float(lateral_margin)
                for sign in (-1.0, 1.0):
                    dx = float(driving_wp.transform.location.x) + sign * right[0] * offset - base_wp.x
                    dy = float(driving_wp.transform.location.y) + sign * right[1] * offset - base_wp.y
                    _add_candidate(dx, dy, f"lane_lateral_{mult:.1f}", 0.12 + 0.03 * mult)
            # along-lane offsets (forward/back)
            try:
                for dist in (0.5, 1.0, 2.0, 3.0, 4.0):
                    nxt = driving_wp.next(dist)
                    if nxt:
                        loc = nxt[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_forward", 0.15)
                    prev = driving_wp.previous(dist)
                    if prev:
                        loc = prev[0].transform.location
                        _add_candidate(loc.x - base_wp.x, loc.y - base_wp.y, "lane_backward", 0.15)
            except Exception:
                pass
        except Exception:
            pass

    # Random offsets (if requested)
    if random_samples > 0:
        rng = rng or random.Random(0)
        for _ in range(random_samples):
            r = max_shift * math.sqrt(rng.random())
            theta = 2.0 * math.pi * rng.random()
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            _add_candidate(dx, dy, "random", 0.2 + 0.01 * r)

    return list(candidates.values())


def _connect_carla_for_spawn(
    host: str,
    port: int,
    expected_town: Optional[str],
):
    if carla is None:
        raise RuntimeError("carla module not available")
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    cmap = world.get_map()
    if expected_town and expected_town not in (cmap.name or ""):
        available_maps = client.get_available_maps()
        candidates = [m for m in available_maps if expected_town in m]
        if candidates:
            target_map = candidates[0]
            print(f"[INFO] Loading map '{target_map}' for spawn preprocessing")
            world = client.load_world(target_map)
            cmap = world.get_map()
    return client, world, cmap


def _extend_grid_steps(base_steps: List[float], max_shift: float) -> List[float]:
    steps = list(base_steps)
    # Add finer steps up to 1.0
    for val in (0.6, 0.8, 1.0):
        steps.extend([val, -val])
    # Add coarser steps up to max_shift
    if max_shift > 1.0:
        step = 0.5
        v = 1.5
        while v <= max_shift + 1e-6:
            steps.extend([v, -v])
            v += step
    # Deduplicate and clamp
    uniq = []
    seen = set()
    for v in steps:
        if abs(v) > max_shift + 1e-6:
            continue
        key = round(float(v), 3)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _select_blueprint(
    blueprint_lib,
    model: str,
    kind: str,
    obj_type_raw: str,
) -> Tuple[Optional[object], str, str]:
    """
    Return (blueprint, model_used, reason).
    Tries exact match, pattern match, then role-aware fallbacks.
    """
    if blueprint_lib is None:
        return None, model, "no_blueprint_lib"

    # Exact match
    try:
        bp = blueprint_lib.find(model)
        if bp is not None:
            if kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", "")):
                bp = None
            else:
                return bp, model, "exact"
    except Exception:
        pass

    # Pattern match
    try:
        matches = blueprint_lib.filter(model)
        if matches:
            if kind.startswith("walker"):
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
            if matches:
                return matches[0], matches[0].id, "pattern"
    except Exception:
        pass

    obj_lower = str(obj_type_raw or "").lower()
    fallback_models: List[str] = []

    if kind.startswith("walker") or "pedestrian" in obj_lower or "walker" in obj_lower:
        fallback_models = list(ADULT_WALKER_BLUEPRINTS)[:3]
    elif "bicycle" in obj_lower or "cycl" in obj_lower:
        fallback_models = [
            "vehicle.diamondback.century",
            "vehicle.gazelle.omafiets",
            "vehicle.bh.crossbike",
        ]
    elif "motor" in obj_lower or "motorcycle" in obj_lower or ("bike" in obj_lower and "bicycle" not in obj_lower):
        fallback_models = [
            "vehicle.harley-davidson.low_rider",
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.vespa.zx125",
        ]
    elif "bus" in obj_lower:
        fallback_models = [
            "vehicle.volkswagen.t2",
            "vehicle.mitsubishi.fusorosa",
        ]
    elif "truck" in obj_lower:
        fallback_models = [
            "vehicle.carlamotors.carlacola",
        ]
    elif "van" in obj_lower or "sprinter" in obj_lower:
        fallback_models = [
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.t2",
        ]
    elif "ambulance" in obj_lower:
        fallback_models = [
            "vehicle.ford.ambulance",
        ]
    elif "police" in obj_lower:
        fallback_models = [
            "vehicle.dodge.charger_police",
            "vehicle.dodge.charger_police_2020",
        ]
    else:
        fallback_models = [
            "vehicle.tesla.model3",
            "vehicle.audi.a2",
            "vehicle.lincoln.mkz_2017",
            "vehicle.nissan.micra",
        ]

    for fallback in fallback_models:
        try:
            bp = blueprint_lib.find(fallback)
            if bp is not None and not (kind.startswith("walker") and _is_child_walker_blueprint(getattr(bp, "id", ""))):
                return bp, fallback, "fallback"
        except Exception:
            pass
        try:
            matches = blueprint_lib.filter(fallback)
            if matches:
                if kind.startswith("walker"):
                    matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_pattern"
        except Exception:
            pass

    # Final generic fallback
    try:
        if kind.startswith("walker"):
            matches = blueprint_lib.filter("walker.pedestrian.*")
            if matches:
                matches = [m for m in matches if not _is_child_walker_blueprint(getattr(m, "id", ""))]
                if matches:
                    return matches[0], matches[0].id, "fallback_any_walker"
    except Exception:
        pass
    try:
        matches = blueprint_lib.filter("vehicle.*")
        if matches:
            return matches[0], matches[0].id, "fallback_any_vehicle"
    except Exception:
        pass

    return None, model, "missing_blueprint"


def _preprocess_spawn_positions(
    vehicles: Dict[int, List[Waypoint]],
    vehicle_times: Dict[int, List[float]],
    actor_meta: Dict[int, Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    report: Dict[str, object] = {
        "settings": {},
        "actors": {},
        "summary": {},
    }

    if carla is None:
        print("[WARN] spawn preprocess requested but CARLA Python module is unavailable; skipping.")
        report["summary"]["status"] = "skipped_no_carla"
        return report

    try:
        client, world, world_map = _connect_carla_for_spawn(
            host=args.carla_host,
            port=args.carla_port,
            expected_town=args.expected_town,
        )
    except Exception as exc:
        print(f"[WARN] spawn preprocess failed to connect to CARLA: {exc}")
        report["summary"]["status"] = "skipped_carla_connect"
        return report

    blueprint_lib = world.get_blueprint_library() if world else None

    existing_actors = 0
    cleared_actors = 0
    try:
        existing_actors = len(world.get_actors()) if world else 0
    except Exception:
        existing_actors = 0

    # Clear dynamic actors to reduce spawn-test interference.
    if world is not None and existing_actors:
        try:
            to_destroy = []
            for actor in world.get_actors():
                try:
                    tid = actor.type_id or ""
                except Exception:
                    tid = ""
                if (
                    tid.startswith("vehicle.")
                    or tid.startswith("walker.")
                    or tid.startswith("sensor.")
                    or tid.startswith("controller.ai.")
                ):
                    to_destroy.append(actor.id)
            if to_destroy:
                try:
                    if hasattr(carla, "command") and client is not None:
                        commands = [carla.command.DestroyActor(aid) for aid in to_destroy]
                        client.apply_batch_sync(commands, True)
                    else:
                        for actor in world.get_actors(to_destroy):
                            try:
                                actor.destroy()
                            except Exception:
                                pass
                except Exception:
                    for actor in world.get_actors(to_destroy):
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                cleared_actors = len(to_destroy)
                try:
                    settings = world.get_settings()
                    if getattr(settings, "synchronous_mode", False):
                        world.tick()
                    else:
                        world.wait_for_tick()
                except Exception:
                    pass
        except Exception:
            pass

    if cleared_actors:
        print(f"[SPAWN_PRE] Cleared {cleared_actors} dynamic actors from CARLA world before spawn checks.")
    elif existing_actors > 10:
        print(f"[WARN] CARLA world has {existing_actors} existing actors; spawn tests may be affected.")

    max_shift = max(0.0, float(args.spawn_preprocess_max_shift))
    grid_steps = []
    for token in re.split(r"[,\s]+", str(args.spawn_preprocess_grid or "").strip()):
        if not token:
            continue
        try:
            grid_steps.append(float(token))
        except Exception:
            continue
    if not grid_steps:
        grid_steps = [0.0, 0.2, -0.2, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2]
    grid_steps = _extend_grid_steps(grid_steps, max_shift)
    lateral_margin = 0.6

    sample_dt = float(args.spawn_preprocess_sample_dt)
    grid_size = max(1.0, float(args.spawn_preprocess_grid_size))
    max_candidates = max(5, int(args.spawn_preprocess_max_candidates))
    collision_weight = float(args.spawn_preprocess_collision_weight)
    normalize_z = bool(args.spawn_preprocess_normalize_z)
    random_samples = max(0, int(args.spawn_preprocess_random_samples))
    debug_radius = float(args.spawn_preprocess_debug_radius)
    debug_max_items = int(args.spawn_preprocess_debug_max_items)

    all_times: List[float] = []
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        all_times.extend(times)
    sample_times = _build_time_grid(all_times, sample_dt)

    report["settings"] = {
        "max_shift": max_shift,
        "grid_steps": grid_steps,
        "sample_dt": sample_dt,
        "grid_size": grid_size,
        "max_candidates": max_candidates,
        "collision_weight": collision_weight,
        "normalize_z": normalize_z,
        "random_samples": random_samples,
        "debug_radius": debug_radius,
        "debug_max_items": debug_max_items,
        "cleared_dynamic_actors": cleared_actors,
        "sample_times": len(sample_times),
    }

    # Precompute base positions and radii
    base_positions: Dict[int, List[Optional[Tuple[float, float]]]] = {}
    radii: Dict[int, float] = {}
    times_cache: Dict[int, List[float]] = {}
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        times = _ensure_times(traj, vehicle_times.get(vid), args.dt)
        times_cache[vid] = times
        kind = str(meta.get("kind"))
        always_active = kind in ("static", "walker_static")
        base_positions[vid] = _sample_positions(traj, times, sample_times, always_active)
        radii[vid] = _actor_radius(
            kind,
            meta.get("length"),
            meta.get("width"),
            str(meta.get("model", "")),
        )

    # Cache world actors/env objects for debug
    actor_items: List[Dict[str, object]] = []
    env_items: List[Dict[str, object]] = []
    if world is not None:
        try:
            for actor in world.get_actors():
                try:
                    loc = actor.get_location()
                    tf = actor.get_transform()
                    bbox = actor.bounding_box
                except Exception:
                    continue
                actor_items.append(
                    {
                        "id": int(actor.id),
                        "type": getattr(actor, "type_id", "actor"),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass
        try:
            label_any = getattr(carla.CityObjectLabel, "Any", None)
            env_objs = world.get_environment_objects(label_any) if label_any is not None else world.get_environment_objects()
            for env in env_objs:
                try:
                    tf = env.transform
                    loc = tf.location
                    bbox = env.bounding_box
                except Exception:
                    continue
                env_items.append(
                    {
                        "id": int(getattr(env, "id", -1)),
                        "type": getattr(env, "type_id", getattr(env, "type", "env")),
                        "loc": loc,
                        "bbox": _bbox_corners_2d(bbox, tf),
                    }
                )
        except Exception:
            pass

    # Build candidate lists with spawn validity
    candidates_by_actor: Dict[int, List[SpawnCandidate]] = {}
    bp_by_actor: Dict[int, object] = {}
    for vid, traj in vehicles.items():
        meta = actor_meta.get(vid)
        if meta is None or not traj:
            continue
        kind = str(meta.get("kind"))
        role = "npc" if kind in ("npc", "static") else "walker"
        model = str(meta.get("model") or "")
        actor_report = {
            "kind": kind,
            "model": model,
            "model_used": model,
            "candidates": [],
            "chosen": None,
        }
        report["actors"][str(vid)] = actor_report

        base_wp = traj[0]
        candidates = _generate_spawn_candidates(
            base_wp=base_wp,
            role=role,
            world_map=world_map,
            max_shift=max_shift,
            grid_steps=grid_steps,
            lateral_margin=lateral_margin,
            random_samples=random_samples,
            rng=random.Random(vid),
        )
        # Sort by base_cost and distance to keep the best candidates first
        candidates.sort(key=lambda c: c.base_cost)
        candidates = candidates[:max_candidates]

        if blueprint_lib is None:
            print(f"[WARN] Blueprint library unavailable; skipping spawn validation for actor {vid}.")
            for cand in candidates:
                cand.valid = True
                cand.reason = "no_blueprint_lib"
                cand.spawn_loc = (float(base_wp.x + cand.dx), float(base_wp.y + cand.dy), float(base_wp.z))
                cand.dz = 0.0
                cand.z_source = "authored"
            candidates_by_actor[vid] = candidates
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": c.valid,
                    "reason": c.reason,
                    "base_cost": c.base_cost,
                    "spawn_loc": c.spawn_loc,
                    "dz": c.dz,
                    "z_source": c.z_source,
                }
                for c in candidates
            ]
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": len(candidates),
                "invalid": 0,
                "source_counts": src_counts,
                "failure_reasons": {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue

        bp, model_used, reason = _select_blueprint(blueprint_lib, model, kind, str(meta.get("obj_type") or ""))
        actor_report["model_used"] = model_used
        actor_report["blueprint_reason"] = reason
        if bp is None:
            print(f"[WARN] No blueprint found for actor {vid} model '{model}'; leaving trajectory unchanged.")
            candidates_by_actor[vid] = []
            src_counts = {}
            for c in candidates:
                src_counts[c.source] = src_counts.get(c.source, 0) + 1
            actor_report["candidates"] = [
                {
                    "dx": c.dx,
                    "dy": c.dy,
                    "source": c.source,
                    "valid": False,
                    "reason": "missing_blueprint",
                    "base_cost": c.base_cost,
                    "spawn_loc": (float(base_wp.x + c.dx), float(base_wp.y + c.dy), float(base_wp.z)),
                    "dz": 0.0,
                    "z_source": "authored",
                }
                for c in candidates
            ]
            actor_report["status"] = "missing_blueprint"
            actor_report["candidate_stats"] = {
                "total": len(candidates),
                "valid": 0,
                "invalid": len(candidates),
                "source_counts": src_counts,
                "failure_reasons": {"missing_blueprint": len(candidates)} if candidates else {},
            }
            actor_report["spawn_base"] = {
                "x": float(base_wp.x),
                "y": float(base_wp.y),
                "z": float(base_wp.z),
                "yaw": float(base_wp.yaw),
            }
            continue
        bp_by_actor[vid] = bp
        if model_used and model_used != model:
            print(f"[WARN] Blueprint '{model}' unavailable; using '{model_used}' for actor {vid}.")
            meta["model"] = model_used
            model = model_used
            actor_report["model"] = model_used

        for cand in candidates:
            _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)

        # If no valid candidates, expand search once more (try harder)
        if not any(c.valid for c in candidates):
            hard_max_shift = min(max_shift * 2.0, max_shift + 4.0)
            hard_grid_steps = _extend_grid_steps(grid_steps, hard_max_shift)
            hard_candidates = _generate_spawn_candidates(
                base_wp=base_wp,
                role=role,
                world_map=world_map,
                max_shift=hard_max_shift,
                grid_steps=hard_grid_steps,
                lateral_margin=lateral_margin,
                random_samples=random_samples * 2,
                rng=random.Random(vid + 100000),
            )
            hard_candidates.sort(key=lambda c: c.base_cost)
            hard_candidates = hard_candidates[: max_candidates * 3]
            for cand in hard_candidates:
                _try_spawn_candidate(world, world_map, bp, base_wp, cand, normalize_z)
            # merge candidates (keep best cost per offset)
            merged: Dict[Tuple[int, int], SpawnCandidate] = {}
            for cand in candidates + hard_candidates:
                key = (int(round(cand.dx * 100)), int(round(cand.dy * 100)))
                existing = merged.get(key)
                if existing is None or cand.base_cost < existing.base_cost:
                    merged[key] = cand
            candidates = sorted(merged.values(), key=lambda c: c.base_cost)
            candidates = candidates[: max_candidates * 2]

        candidates_by_actor[vid] = candidates
        valid_count = sum(1 for c in candidates if c.valid)
        invalid_count = len(candidates) - valid_count
        reasons: Dict[str, int] = {}
        sources: Dict[str, int] = {}
        for c in candidates:
            sources[c.source] = sources.get(c.source, 0) + 1
            if not c.valid:
                key = c.reason or "spawn_failed"
                reasons[key] = reasons.get(key, 0) + 1
        actor_report["candidate_stats"] = {
            "total": len(candidates),
            "valid": valid_count,
            "invalid": invalid_count,
            "source_counts": sources,
            "failure_reasons": reasons,
        }
        actor_report["spawn_base"] = {
            "x": float(base_wp.x),
            "y": float(base_wp.y),
            "z": float(base_wp.z),
            "yaw": float(base_wp.yaw),
        }
        actor_report["candidates"] = [
            {
                "dx": c.dx,
                "dy": c.dy,
                "source": c.source,
                "valid": c.valid,
                "reason": c.reason,
                "base_cost": c.base_cost,
                "spawn_loc": c.spawn_loc,
                "dz": c.dz,
                "z_source": c.z_source,
            }
            for c in candidates
        ]

    # Global assignment with spatiotemporal collision avoidance
    occupancy: List[Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]] = [
        defaultdict(list) for _ in sample_times
    ]

    def _collision_score(vid: int, cand: SpawnCandidate) -> float:
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        score = 0.0
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + cand.dx
            y = pos[1] + cand.dy
            cell_x = int(math.floor(x / grid_size))
            cell_y = int(math.floor(y / grid_size))
            cell_map = occupancy[t_idx]
            for gx in range(cell_x - 1, cell_x + 2):
                for gy in range(cell_y - 1, cell_y + 2):
                    for ox, oy, orad, oid in cell_map.get((gx, gy), []):
                        dist = math.hypot(x - ox, y - oy)
                        if dist < (radius + orad):
                            score += 1.0
        return score

    chosen_offsets: Dict[int, SpawnCandidate] = {}
    actor_order = sorted(
        [vid for vid in vehicles.keys() if vid in candidates_by_actor],
        key=lambda vid: len([c for c in candidates_by_actor[vid] if c.valid]) or 9999,
    )

    for vid in actor_order:
        meta = actor_meta.get(vid)
        if meta is None:
            continue
        cands = [c for c in candidates_by_actor.get(vid, []) if c.valid]
        if not cands:
            # fallback to no shift
            fallback = SpawnCandidate(
                dx=0.0,
                dy=0.0,
                source="fallback",
                base_cost=0.0,
                valid=False,
                reason="no_valid_candidates",
                dz=0.0,
                z_source="fallback",
            )
            chosen_offsets[vid] = fallback
            report["actors"][str(vid)]["chosen"] = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "z_source": "fallback",
                "source": "fallback",
                "collision_score": None,
                "status": "no_valid_candidates",
            }
            cand_all = candidates_by_actor.get(vid, [])
            if cand_all:
                best_invalid = min(cand_all, key=lambda c: c.base_cost)
                report["actors"][str(vid)]["best_invalid_candidate"] = {
                    "dx": best_invalid.dx,
                    "dy": best_invalid.dy,
                    "source": best_invalid.source,
                    "base_cost": best_invalid.base_cost,
                    "reason": best_invalid.reason,
                    "spawn_loc": best_invalid.spawn_loc,
                    "dz": best_invalid.dz,
                    "z_source": best_invalid.z_source,
                }
            # add debug info for failed spawns
            base_wp = vehicles.get(vid, [None])[0]
            entry = report["actors"].get(str(vid), {})
            if base_wp is not None:
                bp = bp_by_actor.get(vid)
                probe_yaw = str(entry.get("kind", "")).startswith("npc") or str(entry.get("kind", "")).startswith("static")
                entry["debug"] = _collect_spawn_debug(
                    actor_id=vid,
                    base_wp=base_wp,
                    entry=entry,
                    world=world,
                    world_map=world_map,
                    blueprint=bp,
                    actor_items=actor_items,
                    env_items=env_items,
                    max_dist=debug_radius,
                    max_items=debug_max_items,
                    probe_yaw=probe_yaw,
                )
            continue

        best = None
        best_score = None
        best_collision = None
        for cand in cands:
            collision = _collision_score(vid, cand)
            total = cand.base_cost + collision_weight * collision
            if best_score is None or total < best_score:
                best = cand
                best_score = total
                best_collision = collision

        if best is None:
            continue

        chosen_offsets[vid] = best
        report["actors"][str(vid)]["chosen"] = {
            "dx": best.dx,
            "dy": best.dy,
            "dz": best.dz,
            "z_source": best.z_source,
            "source": best.source,
            "collision_score": best_collision,
            "status": "ok",
        }

        # Update occupancy
        positions = base_positions.get(vid, [])
        radius = radii.get(vid, 1.0)
        for t_idx, pos in enumerate(positions):
            if pos is None:
                continue
            x = pos[0] + best.dx
            y = pos[1] + best.dy
            cell = (int(math.floor(x / grid_size)), int(math.floor(y / grid_size)))
            occupancy[t_idx][cell].append((x, y, radius, vid))

        if args.spawn_preprocess_verbose:
            print(
                f"[SPAWN_PRE] actor {vid} kind={meta.get('kind')} model={meta.get('model')} "
                f"chosen dx={best.dx:.3f} dy={best.dy:.3f} dz={best.dz:.3f} "
                f"z_src={best.z_source} source={best.source} "
                f"collision={best_collision}"
            )

    # Apply offsets to trajectories
    total_shifted = 0
    total_z_shifted = 0
    for vid, cand in chosen_offsets.items():
        if abs(cand.dx) < 1e-6 and abs(cand.dy) < 1e-6 and abs(cand.dz) < 1e-6:
            continue
        traj = vehicles.get(vid)
        if not traj:
            continue
        for wp in traj:
            wp.x += cand.dx
            wp.y += cand.dy
            wp.z += cand.dz
        total_shifted += 1
        if abs(cand.dz) >= 1e-6:
            total_z_shifted += 1

    missing_bp = 0
    no_valid = 0
    fallback_bp = 0
    valid_counts: List[int] = []
    reason_totals: Dict[str, int] = {}
    source_totals: Dict[str, int] = {}
    for entry in report.get("actors", {}).values():
        if entry.get("status") == "missing_blueprint":
            missing_bp += 1
        chosen = entry.get("chosen") or {}
        if chosen.get("status") == "no_valid_candidates":
            no_valid += 1
        reason = str(entry.get("blueprint_reason") or "")
        if reason.startswith("fallback"):
            fallback_bp += 1
        stats = entry.get("candidate_stats") or {}
        if stats:
            valid_counts.append(int(stats.get("valid", 0)))
            for src, cnt in (stats.get("source_counts") or {}).items():
                source_totals[src] = source_totals.get(src, 0) + int(cnt)
            for r, cnt in (stats.get("failure_reasons") or {}).items():
                reason_totals[r] = reason_totals.get(r, 0) + int(cnt)

    report["summary"] = {
        "status": "ok",
        "actors_considered": len(actor_meta),
        "actors_shifted": total_shifted,
        "actors_z_shifted": total_z_shifted,
        "actors_missing_blueprint": missing_bp,
        "actors_no_valid_candidates": no_valid,
        "actors_with_fallback_blueprint": fallback_bp,
        "candidate_valid_counts": {
            "min": min(valid_counts) if valid_counts else 0,
            "max": max(valid_counts) if valid_counts else 0,
            "avg": (sum(valid_counts) / max(1, len(valid_counts))) if valid_counts else 0.0,
        },
        "candidate_source_totals": source_totals,
        "candidate_failure_reasons": reason_totals,
    }
    ok_count = max(0, len(actor_meta) - missing_bp - no_valid)
    print(
        "[SPAWN_PRE] Summary: "
        f"actors={len(actor_meta)} ok={ok_count} "
        f"no_valid={no_valid} missing_bp={missing_bp} "
        f"fallback_bp={fallback_bp} shifted={total_shifted} z_shifted={total_z_shifted}"
    )
    if no_valid:
        samples = []
        for actor_id, entry in report.get("actors", {}).items():
            chosen = entry.get("chosen") or {}
            if chosen.get("status") != "no_valid_candidates":
                continue
            best = entry.get("best_invalid_candidate") or {}
            model_used = entry.get("model_used") or entry.get("model")
            samples.append(
                f"id={actor_id} kind={entry.get('kind')} model={model_used} "
                f"best_src={best.get('source')} cost={best.get('base_cost')}"
            )
            if len(samples) >= 10:
                break
        if samples:
            print("[SPAWN_PRE] No-valid examples: " + "; ".join(samples))
    return report

# ---------------------- Core conversion ---------------------- #

def build_trajectories(
    yaml_dir: Path,
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool = False,
) -> Tuple[
    Dict[int, List[Waypoint]],
    Dict[int, List[float]],
    List[Waypoint],
    List[float],
    Dict[int, Dict[str, object]],
]:
    """Parse YAML sequence into per-vehicle trajectories and ego path, plus per-waypoint times."""
    yaml_paths = list_yaml_timesteps(yaml_dir)
    if not yaml_paths:
        raise SystemExit(f"No YAML files found under {yaml_dir}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_traj: List[Waypoint] = []
    ego_times: List[float] = []
    obj_info: Dict[int, Dict[str, object]] = {}
    spawn_report: Dict[str, object] | None = None

    for idx, path in enumerate(yaml_paths):
        try:
            frame_idx = int(path.stem)
        except Exception:
            frame_idx = idx
        frame_time = float(frame_idx) * float(dt)
        data = load_yaml(path)
        ego_pose = data.get("true_ego_pose") or data.get("lidar_pose")
        if ego_pose:
            ex, ey, ez = float(ego_pose[0]), float(ego_pose[1]), float(ego_pose[2])
            ex, ey = apply_se2((ex, ey), yaw_deg, tx, ty, flip_y=flip_y)
            ego_yaw = yaw_from_pose(ego_pose)
            if flip_y:
                ego_yaw = -ego_yaw
            ego_traj.append(
                Waypoint(
                    x=ex,
                    y=ey,
                    z=ez + tz,
                    yaw=ego_yaw + yaw_deg,
                    pitch=float(ego_pose[3]) if len(ego_pose) > 3 else 0.0,
                    roll=float(ego_pose[5]) if len(ego_pose) > 5 else 0.0,
                )
            )
            ego_times.append(frame_time)

        vehs = data.get("vehicles", {}) or {}
        for vid_str, payload in vehs.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
            if isinstance(payload, dict):
                existing = obj_info.get(vid, {})
                obj_type = payload.get("obj_type")
                if obj_type and (not existing.get("obj_type")):
                    existing["obj_type"] = obj_type
                    existing["model"] = map_obj_type(obj_type)
                elif obj_type and existing.get("obj_type") and obj_type != existing.get("obj_type"):
                    # Keep first seen obj_type but note mismatch once
                    if not existing.get("_obj_type_conflict"):
                        print(f"[WARN] obj_type conflict for id {vid}: '{existing.get('obj_type')}' vs '{obj_type}' (keeping first)")
                        existing["_obj_type_conflict"] = True
                ext = payload.get("extent") or []
                if isinstance(ext, Sequence):
                    length = float(ext[0]) * 2 if len(ext) > 0 else None
                    width = float(ext[1]) * 2 if len(ext) > 1 else None
                    if length is not None and existing.get("length") is None:
                        existing["length"] = length
                    if width is not None and existing.get("width") is None:
                        existing["width"] = width
                if existing:
                    obj_info[vid] = existing
            loc = payload.get("location") or [0, 0, 0]
            ang = payload.get("angle") or [0, 0, 0]
            pitch = float(ang[0]) if len(ang) > 0 else 0.0
            yaw = yaw_from_angle(ang)
            if flip_y:
                yaw = -yaw
            yaw += yaw_deg
            roll = float(ang[2]) if len(ang) > 2 else 0.0
            x, y = apply_se2((float(loc[0]), float(loc[1])), yaw_deg, tx, ty, flip_y=flip_y)
            z = float(loc[2]) + tz if len(loc) > 2 else tz
            wp = Waypoint(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)
            vehicles.setdefault(vid, []).append(wp)
            vehicle_times.setdefault(vid, []).append(frame_time)

    # Compute simple average speed (m/s) per vehicle from path length
    speeds: Dict[int, float] = {}
    for vid, traj in vehicles.items():
        dist = 0.0
        for a, b in zip(traj, traj[1:]):
            dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
        speeds[vid] = dist / max(dt * max(len(traj) - 1, 1), 1e-6)

    return vehicles, vehicle_times, ego_traj, ego_times, obj_info


def extract_obj_info(yaml_dir: Path) -> Dict[int, Dict[str, object]]:
    """Gather obj_type/model/size from the first timestep in a YAML directory."""
    obj_info: Dict[int, Dict[str, object]] = {}
    first_yaml = next(iter(list_yaml_timesteps(yaml_dir)), None)
    if not first_yaml:
        return obj_info
    data0 = load_yaml(first_yaml)
    vehs0 = data0.get("vehicles", {}) or {}
    for vid_str, payload in vehs0.items():
        try:
            vid = int(vid_str)
        except Exception:
            continue
        obj_type = payload.get("obj_type") or "npc"
        model = map_obj_type(obj_type)
        ext = payload.get("extent") or []
        length = float(ext[0]) * 2 if len(ext) > 0 else None
        width = float(ext[1]) * 2 if len(ext) > 1 else None
        obj_info[vid] = {
            "obj_type": obj_type,
            "model": model,
            "length": length,
            "width": width,
        }
    return obj_info


def write_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
    times: List[float] | None = None,
    snap_to_road: bool = False,
    xml_tx: float = 0.0,
    xml_ty: float = 0.0,
) -> None:
    root = ET.Element("routes")
    route = ET.SubElement(
        root,
        "route",
        {
            "id": str(route_id),
            "town": town,
            "role": role,
            "snap_to_road": "true" if snap_to_road else "false",
        },
    )
    for idx, wp in enumerate(waypoints):
        # Normalize pitch and roll: CARLA XML expects pitch=360.0 (or 0.0) and roll=0.0
        # Using 360.0 for pitch as seen in reference XML files
        attrs = {
            "x": f"{wp.x + xml_tx:.6f}",
            "y": f"{wp.y + xml_ty:.6f}",
            "z": f"{wp.z:.6f}",
            "yaw": f"{wp.yaw:.6f}",
            "pitch": "360.000000",  # Normalized for CARLA compatibility
            "roll": "0.000000",     # Normalized for CARLA compatibility
        }
        if times and idx < len(times):
            try:
                attrs["time"] = f"{float(times[idx]):.6f}"
            except (TypeError, ValueError):
                pass
        ET.SubElement(
            route,
            "waypoint",
            attrs,
        )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def parse_route_xml(path: Path) -> List[Waypoint]:
    """Load waypoints (x,y,z,yaw) from a CARLA route XML."""
    tree = ET.parse(path)
    root = tree.getroot()
    wps: List[Waypoint] = []
    for node in root.findall(".//waypoint"):
        try:
            x = float(node.attrib.get("x", 0.0))
            y = float(node.attrib.get("y", 0.0))
            z = float(node.attrib.get("z", 0.0))
            yaw = float(node.attrib.get("yaw", 0.0))
            wps.append(Waypoint(x=x, y=y, z=z, yaw=yaw))
        except Exception:
            continue
    return wps


def save_manifest(
    manifest_path: Path,
    actors_by_kind: Dict[str, List[dict]],
    ego_entries: List[dict],
) -> None:
    """Save manifest with actors organized by kind (ego, npc, static, etc.)."""
    manifest: Dict[str, List[dict]] = {}
    if ego_entries:
        manifest["ego"] = ego_entries
    # Add all other actor kinds
    for kind, entries in sorted(actors_by_kind.items()):
        manifest[kind] = entries
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------- Visualization ---------------------- #

def plot_frame(
    timestep: int,
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    out_path: Path,
    axes_limits: Tuple[float, float, float, float] | None = None,
    map_lines: List[List[Tuple[float, float]]] | None = None,
    invert_plot_y: bool = False,
):
    if plt is None or patches is None or transforms is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib imageio")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Timestep {timestep:06d}")

    xs: List[float] = []
    ys: List[float] = []
    for vid, traj in actors_by_id.items():
        if timestep >= len(traj):
            continue
        wp = traj[timestep]
        width = 2.0
        height = 4.0
        rect = patches.Rectangle(
            (wp.x - width / 2, wp.y - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor="C0",
            facecolor="C0",
            alpha=0.4,
        )
        rot = transforms.Affine2D().rotate_deg_around(wp.x, wp.y, wp.yaw) + ax.transData
        rect.set_transform(rot)
        ax.add_patch(rect)
        ax.text(wp.x, wp.y, f"{vid}", fontsize=7, ha="center", va="center")
        xs.append(wp.x)
        ys.append(wp.y)

    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            idx = min(timestep, len(ego_traj) - 1)
            ego = ego_traj[idx]
            color = "orange" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            tri = patches.RegularPolygon(
                (ego.x, ego.y),
                numVertices=3,
                radius=2.5,
                orientation=math.radians(ego.yaw),
                color=color,
                alpha=0.6,
            )
            ax.add_patch(tri)
            ax.text(ego.x, ego.y, f"ego{ego_idx}", fontsize=7, ha="center", va="center")
            xs.append(ego.x)
            ys.append(ego.y)

    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color="gray", linewidth=1.0, alpha=0.5)
            xs.extend(lx)
            ys.extend(ly)

    if axes_limits:
        minx, maxx, miny, maxy = axes_limits
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    elif xs and ys:
        pad = 10.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_gif(frames_dir: Path, gif_path: Path, fps: float = 10.0) -> None:
    if imageio is None:
        raise RuntimeError("imageio is required for GIF output; install imageio")
    imgs = []
    for png in sorted(frames_dir.glob("frame_*.png")):
        imgs.append(imageio.imread(png))
    if not imgs:
        raise RuntimeError("No frames produced for GIF")
    duration_ms = 1000.0 / float(fps)
    imageio.mimsave(gif_path, imgs, duration=duration_ms / 1000.0)


def write_paths_png(
    actors_by_id: Dict[int, List[Waypoint]],
    ego_trajs: Sequence[List[Waypoint]],
    map_lines: List[List[Tuple[float, float]]],
    out_path: Path,
    axis_pad: float = 10.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None or patches is None:
        raise RuntimeError("matplotlib is required for --paths-png; install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Actor Paths")

    xs: List[float] = []
    ys: List[float] = []

    # Map
    for line in map_lines:
        if len(line) < 2:
            continue
        lx = [p[0] for p in line]
        ly = [p[1] for p in line]
        ax.plot(lx, ly, color="gray", linewidth=0.8, alpha=0.5, zorder=0)
        xs.extend(lx)
        ys.extend(ly)

    # Actors
    for vid, traj in actors_by_id.items():
        if len(traj) < 2:
            continue
        lx = [wp.x for wp in traj]
        ly = [wp.y for wp in traj]
        ax.plot(lx, ly, linewidth=1.5, alpha=0.9, label=f"id {vid}")
        ax.scatter(lx[0], ly[0], s=15, marker="o")
        xs.extend(lx)
        ys.extend(ly)

    # Ego(s)
    if ego_trajs:
        for ego_idx, ego_traj in enumerate(ego_trajs):
            if not ego_traj:
                continue
            lx = [wp.x for wp in ego_traj]
            ly = [wp.y for wp in ego_traj]
            color = "black" if ego_idx == 0 else f"C{(ego_idx + 1) % 10}"
            label = "ego" if ego_idx == 0 else f"ego{ego_idx}"
            ax.plot(lx, ly, color=color, linewidth=2.0, alpha=0.8, label=label)
            ax.scatter(lx[0], ly[0], s=30, marker="*", color=color)
            xs.extend(lx)
            ys.extend(ly)

    if xs and ys:
        pad = max(0.0, axis_pad)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    if len(actors_by_id) <= 20:  # avoid huge legends
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_actor_yaw_viz(
    actor_id: int,
    gt_traj: List[Waypoint],
    xml_traj: List[Waypoint],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    arrow_step: int = 5,
    arrow_len: float = 0.8,
    pad: float = 5.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} yaw: GT vs XML")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    # Paths
    gt_x = [wp.x for wp in gt_traj]
    gt_y = [wp.y for wp in gt_traj]
    xml_x = [wp.x for wp in xml_traj]
    xml_y = [wp.y for wp in xml_traj]
    # Draw XML first, then GT on top with markers so overlap is visible
    ax.plot(xml_x, xml_y, color="#d95f0e", linewidth=2.0, alpha=0.9, label="XML path", zorder=2)
    ax.plot(
        gt_x,
        gt_y,
        color="#2c7fb8",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=2.5,
        markevery=max(1, int(len(gt_x) / 20)),
        label="GT path",
        zorder=3,
    )

    # If GT and XML are effectively identical, annotate it
    min_len = min(len(gt_traj), len(xml_traj))
    if min_len > 0:
        max_diff = 0.0
        for i in range(min_len):
            dx = gt_traj[i].x - xml_traj[i].x
            dy = gt_traj[i].y - xml_traj[i].y
            max_diff = max(max_diff, math.hypot(dx, dy))
        if max_diff < 1e-3:
            ax.text(
                0.02,
                0.98,
                "GT == XML (overlapping paths)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                zorder=10,
            )

    # Yaw arrows
    def _quiver(traj: List[Waypoint], color: str, label: str) -> None:
        if not traj:
            return
        step = max(1, int(arrow_step))
        xs = [wp.x for i, wp in enumerate(traj) if i % step == 0]
        ys = [wp.y for i, wp in enumerate(traj) if i % step == 0]
        us = [math.cos(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        vs = [math.sin(math.radians(wp.yaw)) * arrow_len for i, wp in enumerate(traj) if i % step == 0]
        ax.quiver(
            xs,
            ys,
            us,
            vs,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=color,
            alpha=0.7,
            width=0.002,
            label=label,
        )

    _quiver(gt_traj, "#2c7fb8", "GT yaw")
    _quiver(xml_traj, "#d95f0e", "XML yaw")

    # Bounds
    xs_all = gt_x + xml_x
    ys_all = gt_y + xml_y
    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_actor_raw_yaml_viz(
    actor_id: int,
    points_by_subdir: Dict[str, List[Tuple[float, float, float]]],
    map_lines: List[List[Tuple[float, float]]] | None,
    out_path: Path,
    pad: float = 20.0,
    invert_plot_y: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization; install matplotlib")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Actor {actor_id} raw YAML points by subfolder")

    # Map layer
    if map_lines:
        for line in map_lines:
            if len(line) < 2:
                continue
            xs, ys = zip(*line)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.7, alpha=0.6, zorder=0)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]

    xs_all: List[float] = []
    ys_all: List[float] = []
    for idx, (subdir, pts) in enumerate(sorted(points_by_subdir.items(), key=lambda kv: _yaml_dir_sort_key(Path(kv[0])))):
        if not pts:
            continue
        pts_sorted = sorted(pts, key=lambda p: p[2])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        xs_all.extend(xs)
        ys_all.extend(ys)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.8, zorder=2)
        ax.scatter(xs, ys, s=14, color=color, marker=marker, alpha=0.85, zorder=3, label=f"{subdir} (n={len(xs)})")
        # Start/end annotations
        ax.scatter([xs[0]], [ys[0]], s=40, color=color, marker="o", zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=40, color=color, marker="x", zorder=4)
        ax.annotate(f"t={pts_sorted[0][2]:.1f}", (xs[0], ys[0]), textcoords="offset points", xytext=(6, 6), fontsize=8, color=color)
        ax.annotate(f"t={pts_sorted[-1][2]:.1f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -10), fontsize=8, color=color)

    if xs_all and ys_all:
        pad_val = max(0.0, float(pad))
        ax.set_xlim(min(xs_all) - pad_val, max(xs_all) + pad_val)
        ax.set_ylim(min(ys_all) - pad_val, max(ys_all) + pad_val)

    if invert_plot_y:
        ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert V2XPnP YAML logs to CARLA route XML + manifest")
    p.add_argument("--scenario-dir", required=True, help="Path to the scenario folder containing subfolders with YAML frames")
    p.add_argument(
        "--subdir",
        default="all",
        help=(
            "Specific subfolder inside scenario-dir to use (e.g., -1). "
            "Use 'all' to process all subfolders for actor locations. If omitted and multiple "
            "subfolders exist, behavior is the same. Non-negative subfolders produce ego routes; "
            "negative subfolders contribute actors only."
        ),
    )
    p.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/carla_log_export)")
    p.add_argument("--route-id", default="0", help="Route id to assign to ego and actors (default: 0)")
    p.add_argument("--town", default="ucla_v2", help="CARLA town/map name to embed in XML (default: ucla_v2)")
    p.add_argument("--ego-name", default="ego", help="Name for ego vehicle")
    p.add_argument("--ego-model", default="vehicle.lincoln.mkz2017", help="Blueprint for ego vehicle")
    p.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds (for speed estimation)")
    p.add_argument(
        "--encode-timing",
        action="store_true",
        help="Embed per-waypoint timing in XML using frame index * dt (enables log replay).",
    )
    p.add_argument("--tx", type=float, default=0.0, help="Translation X to apply to all coordinates")
    p.add_argument("--ty", type=float, default=0.0, help="Translation Y to apply to all coordinates")
    p.add_argument("--tz", type=float, default=0.0, help="Translation Z to apply to all coordinates")
    p.add_argument("--xml-tx", type=float, default=0.0, help="Additional X offset applied only when writing XML outputs")
    p.add_argument("--xml-ty", type=float, default=0.0, help="Additional Y offset applied only when writing XML outputs")
    p.add_argument(
        "--coord-json",
        default="/data2/marco/CoLMDriver/v2xpnp/map/ucla_map_offset_carla.json",
        help="Optional JSON file containing transform keys like tx, ty, theta_deg/rad, flip_y; applied to all coordinates",
    )
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Global yaw rotation (degrees, applied before translation)")
    p.add_argument("--snap-to-road", action="store_true", default=True, help="Enable road snapping for actors (defaults to on)")
    p.add_argument("--no-ego", action="store_true", help="Skip writing ego_route.xml")
    p.add_argument("--gif", action="store_true", help="Generate GIF visualization")
    p.add_argument("--gif-path", default=None, help="Path for GIF (default: <out-dir>/replay.gif)")
    p.add_argument("--paths-png", default=None, help="If set, render a single PNG with each actor's full path as a polyline")
    p.add_argument(
        "--actor-yaw-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot GT vs XML yaw over the CARLA map.",
    )
    p.add_argument(
        "--actor-yaw-viz-dir",
        default=None,
        help="Output directory for actor yaw visualizations (default: <out-dir>/actor_yaw_viz).",
    )
    p.add_argument(
        "--actor-yaw-viz-step",
        type=int,
        default=10,
        help="Stride for yaw arrows in actor visualizations (default: 10).",
    )
    p.add_argument(
        "--actor-yaw-viz-arrow-len",
        type=float,
        default=0.8,
        help="Arrow length (meters) for yaw visualizations (default: 0.8).",
    )
    p.add_argument(
        "--actor-yaw-viz-pad",
        type=float,
        default=5.0,
        help="Padding (meters) around GT/XML path extents for actor yaw visualizations.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-ids",
        default="",
        help="Comma/space-separated actor ids to plot raw YAML points by subfolder.",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-dir",
        default=None,
        help="Output directory for raw YAML actor visualizations (default: <out-dir>/actor_raw_yaml_viz).",
    )
    p.add_argument(
        "--actor-raw-yaml-viz-pad",
        type=float,
        default=20.0,
        help="Padding (meters) around raw YAML points for actor visualizations.",
    )
    p.add_argument("--map-pkl", default=None, help="Optional pickle containing vector map polylines to overlay")
    p.add_argument("--use-carla-map", default=True, action="store_true", help="Connect to CARLA to fetch map polylines for overlay")
    p.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    p.add_argument("--carla-port", type=int, default=2010, help="CARLA port (default: 2010)")
    p.add_argument("--carla-sample", type=float, default=2.0, help="Waypoint sampling distance in meters (default: 2.0)")
    p.add_argument("--carla-cache", default=None, help="Path to cache map polylines (default: <out-dir>/carla_map_cache.pkl)")
    p.add_argument("--expected-town", default="ucla_v2", help="Assert CARLA map name contains this string when using --use-carla-map")
    p.add_argument("--axis-pad", type=float, default=10.0, help="Padding (meters) around actor/ego extents for visualization axes")
    p.add_argument("--flip-y", action="store_true", help="Mirror dataset Y axis and negate yaw (useful if overlay appears upside-down)")
    p.add_argument("--invert-plot-y", action="store_true", help="Invert matplotlib Y axis for visualization only")
    p.add_argument(
        "--spawn-viz",
        action="store_true",
        help="Generate a spawn-vs-aligned visualization over CARLA map and XODR layers.",
    )
    p.add_argument(
        "--spawn-viz-path",
        default=None,
        help="Output path for spawn-vs-aligned visualization (default: <out-dir>/spawn_alignment_viz.png).",
    )
    p.add_argument(
        "--xodr",
        default=None,
        help="Path to the OpenDRIVE XODR file for spawn visualization overlay.",
    )
    p.add_argument(
        "--xodr-step",
        type=float,
        default=2.0,
        help="Sampling step size (meters) for XODR geometry (default: 2.0).",
    )
    p.add_argument(
        "--map-image",
        default=None,
        help="Optional raster map image to use as the CARLA background layer (PNG/JPG).",
    )
    p.add_argument(
        "--map-image-bounds",
        nargs=4,
        type=float,
        default=None,
        metavar=("MINX", "MAXX", "MINY", "MAXY"),
        help="World bounds for the map image (minx maxx miny maxy). If omitted, bounds are inferred.",
    )
    p.add_argument(
        "--spawn-preprocess",
        action="store_true",
        help="Run CARLA-in-the-loop spawn preprocessing to improve actor spawn success.",
    )
    p.add_argument(
        "--spawn-preprocess-report",
        default=None,
        help="Optional JSON report path for spawn preprocessing results.",
    )
    p.add_argument(
        "--spawn-preprocess-max-shift",
        type=float,
        default=4.0,
        help="Maximum XY shift (meters) when generating spawn candidates (default: 4.0).",
    )
    p.add_argument(
        "--spawn-preprocess-random-samples",
        type=int,
        default=80,
        help="Number of random candidate offsets per actor (default: 80).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz",
        action="store_true",
        help="Generate visualization for actors that failed to spawn (over CARLA map).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dir",
        default=None,
        help="Output directory for failed spawn visualizations (default: <out-dir>/spawn_preprocess_fail_viz).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-window",
        type=float,
        default=60.0,
        help="Window size (meters) for per-actor failed spawn plots (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-dpi",
        type=int,
        default=220,
        help="DPI for failed spawn visualizations (default: 220).",
    )
    p.add_argument(
        "--spawn-preprocess-fail-viz-sample",
        type=float,
        default=1.0,
        help="CARLA map sampling distance for failed spawn visualizations (default: 1.0).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-radius",
        type=float,
        default=30.0,
        help="Radius (meters) for collecting nearby actors/env objects in failed spawn debug (default: 30).",
    )
    p.add_argument(
        "--spawn-preprocess-debug-max-items",
        type=int,
        default=10,
        help="Max nearby actors/env objects to record per failed spawn (default: 10).",
    )
    p.add_argument(
        "--spawn-preprocess-grid",
        default="0.0,0.2,-0.2,0.4,-0.4,0.8,-0.8,1.2,-1.2",
        help="Comma/space-separated XY offsets (meters) for local candidate grid.",
    )
    p.add_argument(
        "--spawn-preprocess-sample-dt",
        type=float,
        default=0.5,
        help="Sampling timestep (seconds) for collision scoring (default: 0.5).",
    )
    p.add_argument(
        "--spawn-preprocess-grid-size",
        type=float,
        default=5.0,
        help="Spatial hash grid size (meters) for collision checks (default: 5.0).",
    )
    p.add_argument(
        "--spawn-preprocess-max-candidates",
        type=int,
        default=60,
        help="Maximum candidate offsets per actor (default: 60).",
    )
    p.add_argument(
        "--spawn-preprocess-collision-weight",
        type=float,
        default=50.0,
        help="Weight for collision penalty in candidate scoring (default: 50.0).",
    )
    p.add_argument(
        "--spawn-preprocess-verbose",
        action="store_true",
        help="Enable verbose spawn preprocessing logs.",
    )
    p.add_argument(
        "--spawn-preprocess-normalize-z",
        action="store_true",
        default=True,
        help="Use ground projection when validating spawn candidates (default: on).",
    )
    p.add_argument(
        "--no-spawn-preprocess-normalize-z",
        dest="spawn_preprocess_normalize_z",
        action="store_false",
        help="Disable ground projection during spawn candidate validation.",
    )
    p.add_argument("--run-custom-eval", action="store_true", help="After export, call tools/run_custom_eval.py with the generated routes dir")
    p.add_argument(
        "--eval-planner",
        default="",
        help="Planner for run_custom_eval (empty string means no planner flag; e.g., pass 'tcp' or 'log_replay')",
    )
    p.add_argument("--eval-port", type=int, default=2014, help="CARLA port for run_custom_eval (default: 2014)")
    return p.parse_args()


def _yaml_dir_sort_key(path: Path) -> Tuple[int, object]:
    name = path.name
    try:
        return (0, int(name))
    except Exception:
        return (1, name)


def _is_negative_subdir(path: Path) -> bool:
    try:
        return int(path.name) < 0
    except Exception:
        return False


def _parse_id_list(raw: str) -> List[int]:
    ids: List[int] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            ids.append(int(token))
        except Exception:
            continue
    return ids


def pick_yaml_dirs(scenario_dir: Path, chosen: str | None) -> List[Path]:
    subdirs = [d for d in scenario_dir.iterdir() if d.is_dir()]

    if chosen:
        if str(chosen).lower() == "all":
            yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
            numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
            if numeric_subdirs:
                yaml_subdirs = numeric_subdirs
            if not yaml_subdirs:
                raise SystemExit(f"No YAML subfolders found under {scenario_dir}")
            return sorted(yaml_subdirs, key=_yaml_dir_sort_key)
        cand = scenario_dir / chosen
        if not cand.is_dir():
            raise SystemExit(f"--subdir {chosen} not found under {scenario_dir}")
        return [cand]

    if list_yaml_timesteps(scenario_dir):
        return [scenario_dir]

    yaml_subdirs = [d for d in subdirs if list_yaml_timesteps(d)]
    numeric_subdirs = [d for d in yaml_subdirs if re.fullmatch(r"-?\d+", d.name or "")]
    if numeric_subdirs:
        yaml_subdirs = numeric_subdirs
    if len(yaml_subdirs) == 1:
        return yaml_subdirs
    if len(yaml_subdirs) > 1:
        return sorted(yaml_subdirs, key=_yaml_dir_sort_key)

    raise SystemExit(f"No YAML files found under {scenario_dir}")


def _extract_map_lines(obj, depth=0, out: List[List[Tuple[float, float]]] | None = None):
    """Heuristic extractor for vector map polylines from arbitrary pickle structures."""
    if out is None:
        out = []
    if obj is None or depth > 10:
        return out

    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            try:
                out.append([(float(obj["x"]), float(obj["y"]))])
            except Exception:
                pass
        for v in obj.values():
            _extract_map_lines(v, depth + 1, out)
        return out

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2 and all(hasattr(it, "__len__") and len(it) >= 2 for it in obj if it is not None):
            try:
                pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                if len(pts) >= 2:
                    out.append(pts)
                    return out
            except Exception:
                pass
        for v in obj:
            _extract_map_lines(v, depth + 1, out)
        return out

    if hasattr(obj, "x") and hasattr(obj, "y"):
        try:
            out.append([(float(obj.x), float(obj.y))])
        except Exception:
            pass
        return out

    if hasattr(obj, "__dict__"):
        _extract_map_lines(obj.__dict__, depth + 1, out)
    return out


def _integrate_geometry(
    x0: float,
    y0: float,
    hdg: float,
    length: float,
    curv_fn,
    step: float,
) -> List[Tuple[float, float]]:
    if length <= 0.0:
        return []
    step = max(step, 0.1)
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    x = x0
    y = y0
    theta = hdg
    points = [(x, y)]
    for i in range(n):
        s_mid = (i + 0.5) * ds
        kappa = curv_fn(s_mid)
        theta_mid = theta + 0.5 * kappa * ds
        x += ds * math.cos(theta_mid)
        y += ds * math.sin(theta_mid)
        theta += kappa * ds
        points.append((x, y))
    return points


def _sample_geometry(geom: ET.Element, step: float) -> List[Tuple[float, float]]:
    x0 = float(geom.attrib.get("x", 0.0))
    y0 = float(geom.attrib.get("y", 0.0))
    hdg = float(geom.attrib.get("hdg", 0.0))
    length = float(geom.attrib.get("length", 0.0))

    child = next(iter(geom), None)
    if child is None:
        return [(x0, y0)]

    if child.tag == "line":
        curv_fn = lambda s: 0.0
    elif child.tag == "arc":
        curvature = float(child.attrib.get("curvature", 0.0))
        curv_fn = lambda s, k=curvature: k
    elif child.tag == "spiral":
        curv_start = float(child.attrib.get("curvStart", 0.0))
        curv_end = float(child.attrib.get("curvEnd", 0.0))

        def curv_fn(s: float, cs=curv_start, ce=curv_end, total=length) -> float:
            if total <= 0.0:
                return cs
            return cs + (ce - cs) * (s / total)
    else:
        curv_fn = lambda s: 0.0

    return _integrate_geometry(x0, y0, hdg, length, curv_fn, step)


def load_xodr_points(path: Path, step: float) -> List[Tuple[float, float]]:
    root = ET.parse(path).getroot()
    points: List[Tuple[float, float]] = []
    for geom in root.findall(".//planView/geometry"):
        points.extend(_sample_geometry(geom, step))
    return points


def _bounds_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), max(xs), min(ys), max(ys))


def _merge_bounds(bounds_list: Sequence[Tuple[float, float, float, float] | None]) -> Tuple[float, float, float, float] | None:
    mins = []
    maxs = []
    for b in bounds_list:
        if not b:
            continue
        mins.append((b[0], b[2]))
        maxs.append((b[1], b[3]))
    if not mins or not maxs:
        return None
    minx = min(m[0] for m in mins)
    miny = min(m[1] for m in mins)
    maxx = max(m[0] for m in maxs)
    maxy = max(m[1] for m in maxs)
    return (minx, maxx, miny, maxy)


def _plot_background_lines(ax, lines: List[List[Tuple[float, float]]], color: str, lw: float, alpha: float):
    for line in lines:
        if len(line) < 2:
            continue
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=1)


def _crop_lines_to_bounds(
    lines: List[List[Tuple[float, float]]],
    bounds: Tuple[float, float, float, float],
) -> List[List[Tuple[float, float]]]:
    minx, maxx, miny, maxy = bounds
    cropped: List[List[Tuple[float, float]]] = []
    for line in lines:
        if len(line) < 2:
            continue
        keep = False
        for x, y in line:
            if minx <= x <= maxx and miny <= y <= maxy:
                keep = True
                break
        if keep:
            cropped.append(line)
    return cropped


def _bbox_corners_2d(bbox, tf) -> List[Tuple[float, float]]:
    corners = []
    try:
        ext = bbox.extent
        center = bbox.location
        for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            loc = carla.Location(
                x=center.x + sx * ext.x,
                y=center.y + sy * ext.y,
                z=center.z,
            )
            world_loc = tf.transform(loc)
            corners.append((float(world_loc.x), float(world_loc.y)))
    except Exception:
        return []
    return corners


def _nearest_items(
    items: List[Dict[str, object]],
    center: carla.Location,
    max_dist: float,
    limit: int,
) -> List[Dict[str, object]]:
    out = []
    for item in items:
        loc = item.get("loc")
        if loc is None:
            continue
        try:
            dist = float(loc.distance(center))
        except Exception:
            continue
        if dist > max_dist:
            continue
        out.append((dist, item))
    out.sort(key=lambda x: x[0])
    results = []
    for dist, it in out[:limit]:
        payload = dict(it)
        payload["dist"] = float(dist)
        results.append(payload)
    return results


def _collect_spawn_debug(
    actor_id: int,
    base_wp: Waypoint,
    entry: Dict[str, object],
    world,
    world_map,
    blueprint,
    actor_items: List[Dict[str, object]],
    env_items: List[Dict[str, object]],
    max_dist: float,
    max_items: int,
    probe_yaw: bool,
) -> Dict[str, object]:
    debug: Dict[str, object] = {}
    loc = carla.Location(x=base_wp.x, y=base_wp.y, z=base_wp.z)
    ground_z = _resolve_ground_z(world, loc) if world is not None else None
    if ground_z is not None:
        debug["ground_z"] = float(ground_z)
        debug["z_delta"] = float(base_wp.z) - float(ground_z)
    try:
        wp_any = world_map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any) if world_map else None
        wp_drive = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving) if world_map else None
        wp_sidewalk = world_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Sidewalk) if world_map else None
    except Exception:
        wp_any = wp_drive = wp_sidewalk = None
    if wp_any is not None:
        debug["wp_any"] = {
            "road_id": int(getattr(wp_any, "road_id", -1)),
            "lane_id": int(getattr(wp_any, "lane_id", -1)),
            "lane_type": str(getattr(wp_any, "lane_type", "")),
            "is_junction": bool(getattr(wp_any, "is_junction", False)),
        }
    if wp_drive is not None:
        debug["wp_drive"] = {
            "road_id": int(getattr(wp_drive, "road_id", -1)),
            "lane_id": int(getattr(wp_drive, "lane_id", -1)),
            "lane_type": str(getattr(wp_drive, "lane_type", "")),
            "is_junction": bool(getattr(wp_drive, "is_junction", False)),
        }
    if wp_sidewalk is not None:
        debug["wp_sidewalk"] = {
            "road_id": int(getattr(wp_sidewalk, "road_id", -1)),
            "lane_id": int(getattr(wp_sidewalk, "lane_id", -1)),
            "lane_type": str(getattr(wp_sidewalk, "lane_type", "")),
            "is_junction": bool(getattr(wp_sidewalk, "is_junction", False)),
        }

    near_actors = _nearest_items(actor_items, loc, max_dist, limit=max_items)
    near_env = _nearest_items(env_items, loc, max_dist, limit=max_items)
    debug["nearest_actors"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_actors
    ]
    debug["nearest_env_objects"] = [
        {
            "id": it.get("id"),
            "type": it.get("type"),
            "dist": it.get("dist"),
            "bbox": it.get("bbox"),
        }
        for it in near_env
    ]

    probe_results = []
    if blueprint is not None and world is not None:
        for dz in (0.0, 0.2, 0.5, 1.0, 2.0):
            spawn_loc = carla.Location(x=loc.x, y=loc.y, z=loc.z + dz)
            spawn_tf = carla.Transform(
                spawn_loc,
                carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw, roll=base_wp.roll),
            )
            ok = False
            actor = None
            try:
                actor = world.try_spawn_actor(blueprint, spawn_tf)
                ok = actor is not None
            except Exception:
                ok = False
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
            probe_results.append({"dz": float(dz), "ok": bool(ok)})

        if probe_yaw:
            yaw_results = []
            for dyaw in (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0):
                spawn_tf = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z),
                    carla.Rotation(pitch=base_wp.pitch, yaw=base_wp.yaw + dyaw, roll=base_wp.roll),
                )
                ok = False
                actor = None
                try:
                    actor = world.try_spawn_actor(blueprint, spawn_tf)
                    ok = actor is not None
                except Exception:
                    ok = False
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        pass
                yaw_results.append({"dyaw": float(dyaw), "ok": bool(ok)})
            debug["probe_yaw"] = yaw_results

    debug["probe_z"] = probe_results
    return debug


def _plot_failed_spawn_visualizations(
    report: Dict[str, object],
    map_lines: List[List[Tuple[float, float]]],
    out_dir: Path,
    window_m: float,
    dpi: int,
) -> None:
    if plt is None:
        print("[WARN] matplotlib not available; skipping failed spawn visualization.")
        return

    actors = report.get("actors") or {}
    failed = []
    for actor_id, entry in actors.items():
        chosen = entry.get("chosen") or {}
        if chosen.get("status") != "no_valid_candidates":
            continue
        base = entry.get("spawn_base") or {}
        if not base:
            continue
        failed.append((actor_id, entry, base))

    if not failed:
        print("[INFO] No failed actors to visualize.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Overview plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    if map_lines:
        _plot_background_lines(ax, map_lines, color="#9e9e9e", lw=0.6, alpha=0.5)
    xs = []
    ys = []
    for actor_id, entry, base in failed:
        x = float(base.get("x", 0.0))
        y = float(base.get("y", 0.0))
        xs.append(x)
        ys.append(y)
        ax.scatter([x], [y], c="#d62728", s=30, marker="x", zorder=5)
        ax.text(x, y, str(actor_id), fontsize=6, color="#111111", zorder=6)
    if xs and ys:
        pad = max(10.0, 0.5 * window_m)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_title("Failed Spawns Overview")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "failed_spawn_overview.png", dpi=dpi)
    plt.close(fig)

    # Per-actor zoomed plots
    half = max(10.0, 0.5 * float(window_m))
    for actor_id, entry, base in failed:
        cx = float(base.get("x", 0.0))
        cy = float(base.get("y", 0.0))
        bounds = (cx - half, cx + half, cy - half, cy + half)
        local_lines = _crop_lines_to_bounds(map_lines, bounds) if map_lines else []

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal", adjustable="box")
        if local_lines:
            _plot_background_lines(ax, local_lines, color="#b0b0b0", lw=0.7, alpha=0.6)

        debug = entry.get("debug") or {}
        if patches is not None:
            for obj in debug.get("nearest_env_objects", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#ff9896",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )
            for obj in debug.get("nearest_actors", []):
                poly = obj.get("bbox")
                if poly:
                    ax.add_patch(
                        patches.Polygon(
                            poly,
                            closed=True,
                            fill=False,
                            edgecolor="#1f77b4",
                            linewidth=0.8,
                            alpha=0.8,
                            zorder=1,
                        )
                    )

        # Candidate points
        candidates = entry.get("candidates") or []
        invalid_x = []
        invalid_y = []
        valid_x = []
        valid_y = []
        for cand in candidates:
            loc = cand.get("spawn_loc")
            if not loc:
                continue
            if cand.get("valid"):
                valid_x.append(loc[0])
                valid_y.append(loc[1])
            else:
                invalid_x.append(loc[0])
                invalid_y.append(loc[1])
        if invalid_x:
            ax.scatter(invalid_x, invalid_y, s=8, c="#808080", alpha=0.45, label="invalid candidates", zorder=2)
        if valid_x:
            ax.scatter(valid_x, valid_y, s=12, c="#2ca02c", alpha=0.8, label="valid candidates", zorder=3)

        # Base spawn
        ax.scatter([cx], [cy], s=60, marker="x", c="#d62728", label="spawn base", zorder=5)

        # Best invalid
        best = entry.get("best_invalid_candidate") or {}
        if best:
            bl = best.get("spawn_loc")
            if bl:
                ax.scatter([bl[0]], [bl[1]], s=40, marker="o", c="#ff7f0e", label="best invalid", zorder=4)

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_title(f"Failed Spawn Actor {actor_id}")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        meta = f"kind={entry.get('kind')} model={entry.get('model_used') or entry.get('model')}"
        stats = entry.get("candidate_stats") or {}
        detail = f"candidates={stats.get('total')} valid={stats.get('valid')} invalid={stats.get('invalid')}"
        ax.text(bounds[0], bounds[3], meta, fontsize=7, va="top")
        ax.text(bounds[0], bounds[3] - 0.05 * (bounds[3] - bounds[2]), detail, fontsize=7, va="top")
        if debug.get("probe_z"):
            z_ok = [str(r["dz"]) for r in debug.get("probe_z") if r.get("ok")]
            ax.text(
                bounds[0],
                bounds[3] - 0.10 * (bounds[3] - bounds[2]),
                f"probe_z_ok: {', '.join(z_ok) if z_ok else 'none'}",
                fontsize=7,
                va="top",
            )

        # Debug text: nearest actors/env objects
        lines = []
        near_actors = debug.get("nearest_actors") or []
        near_env = debug.get("nearest_env_objects") or []
        if near_actors:
            lines.append("nearest actors:")
            for item in near_actors[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if near_env:
            lines.append("nearest env:")
            for item in near_env[:5]:
                lines.append(
                    f"  id={item.get('id')} type={item.get('type')} d={item.get('dist'):.2f}"
                )
        if lines:
            ax.text(
                bounds[1],
                bounds[3],
                "\n".join(lines),
                fontsize=6,
                va="top",
                ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5),
            )

        out_path = out_dir / f"failed_actor_{actor_id}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def _plot_offset_annotation(
    ax,
    aligned_pt: Tuple[float, float],
    spawn_pt: Tuple[float, float],
    label: str | None = None,
):
    dx = spawn_pt[0] - aligned_pt[0]
    dy = spawn_pt[1] - aligned_pt[1]
    if dx == 0.0 and dy == 0.0:
        return

    x_step = (aligned_pt[0] + dx, aligned_pt[1])

    # Highlight the pre-alignment (spawn) reference point.
    ax.scatter(
        [spawn_pt[0]],
        [spawn_pt[1]],
        s=130,
        marker="X",
        c="#ff7f0e",
        edgecolors="#111111",
        linewidths=0.6,
        label=label,
        zorder=8,
    )

    # Draw axis-aligned offset components.
    ax.plot(
        [aligned_pt[0], x_step[0]],
        [aligned_pt[1], x_step[1]],
        color="#ff7f0e",
        linewidth=1.8,
        zorder=7,
    )
    ax.plot(
        [x_step[0], spawn_pt[0]],
        [x_step[1], spawn_pt[1]],
        color="#1f77b4",
        linewidth=1.8,
        zorder=7,
    )

    bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    ax.annotate(
        f"dx={dx:+.2f}m",
        xy=((aligned_pt[0] + x_step[0]) * 0.5, aligned_pt[1]),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="#ff7f0e",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )
    ax.annotate(
        f"dy={dy:+.2f}m",
        xy=(x_step[0], (x_step[1] + spawn_pt[1]) * 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color="#1f77b4",
        fontsize=9,
        bbox=bbox,
        zorder=9,
    )


def _plot_spawn_alignment(
    ax,
    aligned_points: Dict[int, Tuple[float, float]],
    spawn_points: Dict[int, Tuple[float, float]],
    actor_kind_by_id: Dict[int, str],
    ego_aligned: List[Tuple[float, float]],
    ego_spawn: List[Tuple[float, float]],
    title: str,
    show_offsets: bool = True,
    offset_pair: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    offset_label: str | None = None,
):
    kind_markers = {
        "npc": "o",
        "static": "s",
        "walker": "^",
        "walker_static": "^",
    }
    aligned_color = "#2ca02c"
    spawn_color = "#d62728"

    for kind, marker in kind_markers.items():
        ids = [vid for vid, k in actor_kind_by_id.items() if k == kind and vid in aligned_points]
        if not ids:
            continue
        a_pts = [aligned_points[vid] for vid in ids]
        s_pts = [spawn_points[vid] for vid in ids if vid in spawn_points]
        ax.scatter(
            [p[0] for p in a_pts],
            [p[1] for p in a_pts],
            s=20,
            marker=marker,
            c=aligned_color,
            alpha=0.7,
            label=f"{kind} aligned",
            zorder=3,
        )
        if s_pts:
            ax.scatter(
                [p[0] for p in s_pts],
                [p[1] for p in s_pts],
                s=40,
                marker=marker,
                facecolors="none",
                edgecolors=spawn_color,
                linewidths=1.0,
                label=f"{kind} spawn",
                zorder=4,
            )

    if ego_aligned:
        ax.scatter(
            [p[0] for p in ego_aligned],
            [p[1] for p in ego_aligned],
            s=80,
            marker="*",
            c="#111111",
            label="ego aligned",
            zorder=5,
        )
    if ego_spawn:
        ax.scatter(
            [p[0] for p in ego_spawn],
            [p[1] for p in ego_spawn],
            s=110,
            marker="*",
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.5,
            label="ego spawn",
            zorder=6,
        )

    if show_offsets:
        for vid, a_pt in aligned_points.items():
            s_pt = spawn_points.get(vid)
            if not s_pt:
                continue
            if a_pt == s_pt:
                continue
            ax.plot([a_pt[0], s_pt[0]], [a_pt[1], s_pt[1]], color="#555555", alpha=0.3, linewidth=0.8, zorder=2)

    if offset_pair is not None:
        _plot_offset_annotation(ax, offset_pair[0], offset_pair[1], label=offset_label)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)


def _pick_offset_reference(
    aligned_points: Dict[int, Tuple[float, float]],
    ref_points: Dict[int, Tuple[float, float]],
    ego_aligned: List[Tuple[float, float]],
    ego_ref: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float], str] | None:
    if ego_aligned and ego_ref:
        return ego_aligned[0], ego_ref[0], "ego"
    for vid in sorted(aligned_points.keys()):
        r_pt = ref_points.get(vid)
        if r_pt is None:
            continue
        return aligned_points[vid], r_pt, f"id {vid}"
    return None


def load_vector_map_from_pickle(path: Path) -> List[List[Tuple[float, float]]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return _extract_map_lines(obj, out=[])


def fetch_carla_map_lines(host: str, port: int, sample: float, cache_path: Path | None, expected_town: str | None = None) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float] | None]:
    """Connect to CARLA, ensure the desired map is loaded, sample waypoints, and optionally cache."""
    # Cache reuse only if map name matches expectation (if provided).
    # Load cache before touching CARLA to avoid crashes in environments without a running server.
    if cache_path and cache_path.exists():
        try:
            cached = pickle.load(cache_path.open("rb"))
            if (
                isinstance(cached, dict)
                and "lines" in cached
                and (expected_town is None or expected_town in str(cached.get("map_name", "")))
            ):
                print(f"[INFO] Using cached map polylines from {cache_path} (map={cached.get('map_name')})")
                return cached["lines"], cached.get("bounds")
        except Exception:
            pass

    if carla is None:
        raise SystemExit("carla Python module not available; install CARLA egg/wheel or omit --use-carla-map")

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    available_maps = client.get_available_maps()
    world = client.get_world()
    cmap = world.get_map()
    print(f"[INFO] CARLA current map: {cmap.name}")
    print(f"[INFO] CARLA available maps: {', '.join(available_maps)}")

    if expected_town and expected_town not in (cmap.name or ""):
        candidates = [m for m in available_maps if expected_town in m]
        if not candidates:
            raise RuntimeError(f"CARLA map '{cmap.name}' does not match expected '{expected_town}', and no available map matches")
        target_map = candidates[0]
        print(f"[INFO] Loading map '{target_map}' to satisfy expected substring '{expected_town}'")
        world = client.load_world(target_map)
        cmap = world.get_map()
        print(f"[INFO] Loaded map: {cmap.name}")

    wps = cmap.generate_waypoints(distance=sample)
    buckets: Dict[Tuple[int, int], List[carla.Waypoint]] = {}
    for wp in wps:
        key = (wp.road_id, wp.lane_id)
        buckets.setdefault(key, []).append(wp)

    lines: List[List[Tuple[float, float]]] = []
    bounds = [float("inf"), -float("inf"), float("inf"), -float("inf")]  # minx, maxx, miny, maxy
    for _, seq in buckets.items():
        seq.sort(key=lambda w: w.s)  # along-lane distance
        line = []
        for w in seq:
            x, y = float(w.transform.location.x), float(w.transform.location.y)
            line.append((x, y))
            bounds[0] = min(bounds[0], x)
            bounds[1] = max(bounds[1], x)
            bounds[2] = min(bounds[2], y)
            bounds[3] = max(bounds[3], y)
        if len(line) >= 2:
            lines.append(line)

    btuple = None if bounds[0] == float("inf") else tuple(bounds)  # type: ignore

    if cache_path:
        try:
            pickle.dump({"lines": lines, "bounds": btuple, "map_name": cmap.name}, cache_path.open("wb"))
            print(f"[INFO] Cached map polylines to {cache_path} (map={cmap.name})")
        except Exception:
            pass

    return lines, btuple


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.scenario_dir).expanduser().resolve()
    yaml_dirs = pick_yaml_dirs(scenario_dir, args.subdir)
    out_dir = Path(args.out_dir or (scenario_dir / "carla_log_export")).resolve()
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)

    # Optional transform overrides from JSON
    if args.coord_json:
        try:
            cfg = json.loads(Path(args.coord_json).read_text(encoding="utf-8"))
            json_tx = float(cfg.get("tx", 0.0))
            json_ty = float(cfg.get("ty", 0.0))
            json_tz = float(cfg.get("tz", 0.0)) if "tz" in cfg else 0.0
            json_theta_deg = (
                float(cfg.get("theta_deg", 0.0))
                if "theta_deg" in cfg
                else float(cfg.get("theta_rad", 0.0)) * 180.0 / math.pi if "theta_rad" in cfg else 0.0
            )
            json_flip = bool(cfg.get("flip_y", False) or cfg.get("y_flip", False))

            # Inverse transform: JSON describes CARLA->PKL; we need PKL->CARLA for XML
            if json_flip:
                args.tx += -json_tx
                args.ty += json_ty
                args.flip_y = True
            else:
                args.tx += -json_tx
                args.ty += -json_ty
            args.tz += -json_tz
            args.yaw_deg += -json_theta_deg

            # Allow XML-only offsets from the same file if present
            args.xml_tx += float(cfg.get("xml_tx", 0.0))
            args.xml_ty += float(cfg.get("xml_ty", 0.0))
        except Exception as exc:
            raise SystemExit(f"Failed to read coord_json {args.coord_json}: {exc}") from exc

    if len(yaml_dirs) > 1:
        print("[INFO] Using multiple YAML subfolders for actor locations:")
        for yd in yaml_dirs:
            print(f"  - {yd}")
        pos_subdirs = [yd for yd in yaml_dirs if not _is_negative_subdir(yd)]
        neg_subdirs = [yd for yd in yaml_dirs if _is_negative_subdir(yd)]
        if pos_subdirs:
            print("[INFO] Ego subfolders (non-negative):")
            for yd in pos_subdirs:
                print(f"  - {yd}")
        if neg_subdirs:
            print("[INFO] Non-ego subfolders (negative):")
            for yd in neg_subdirs:
                print(f"  - {yd}")

    vehicles: Dict[int, List[Waypoint]] = {}
    vehicle_times: Dict[int, List[float]] = {}
    ego_trajs: List[List[Waypoint]] = []
    ego_times_list: List[List[float]] = []
    obj_info: Dict[int, Dict[str, object]] = {}

    for yd in yaml_dirs:
        is_negative_subdir = _is_negative_subdir(yd)
        v_map, v_times, ego_traj, ego_times, v_info = build_trajectories(
            yaml_dir=yd,
            dt=args.dt,
            tx=args.tx,
            ty=args.ty,
            tz=args.tz,
            yaw_deg=args.yaw_deg,
            flip_y=args.flip_y,
        )
        if ego_traj and not is_negative_subdir:
            ego_trajs.append(ego_traj)
            ego_times_list.append(ego_times)
        for vid, meta in v_info.items():
            existing = obj_info.get(vid, {})
            if not existing:
                obj_info[vid] = meta
                continue
            # Fill missing fields without overwriting existing obj_type/model
            if not existing.get("obj_type") and meta.get("obj_type"):
                existing["obj_type"] = meta.get("obj_type")
                if meta.get("model"):
                    existing["model"] = meta.get("model")
            if existing.get("length") is None and meta.get("length") is not None:
                existing["length"] = meta.get("length")
            if existing.get("width") is None and meta.get("width") is not None:
                existing["width"] = meta.get("width")
            obj_info[vid] = existing
        for vid, traj in v_map.items():
            if vid not in vehicles or len(traj) > len(vehicles[vid]):
                vehicles[vid] = traj
                vehicle_times[vid] = v_times.get(vid, [])

    # Build actor metadata (used for preprocessing and export)
    actor_meta_by_id: Dict[int, Dict[str, object]] = {}
    skipped_non_vehicles = 0
    for vid, traj in vehicles.items():
        if not traj:
            continue
        info = obj_info.get(vid, {})
        obj_type_val = info.get("obj_type")
        if not obj_type_val:
            print(f"[WARN] Missing obj_type for actor id {vid}; defaulting to npc")
            obj_type_raw = "npc"
        else:
            obj_type_raw = str(obj_type_val)
        if not is_vehicle_type(obj_type_raw):
            skipped_non_vehicles += 1
            continue
        kind, is_ped = _classify_actor_kind(traj, obj_type_raw)
        model = info.get("model") or map_obj_type(obj_type_raw)
        actor_meta_by_id[vid] = {
            "kind": kind,
            "is_pedestrian": is_ped,
            "obj_type": obj_type_raw,
            "model": model,
            "length": info.get("length"),
            "width": info.get("width"),
        }

    if skipped_non_vehicles > 0:
        print(f"[INFO] Skipped {skipped_non_vehicles} non-actor objects (props, static objects, etc.)")

    if args.spawn_preprocess:
        report = _preprocess_spawn_positions(vehicles, vehicle_times, actor_meta_by_id, args)
        spawn_report = report
        if args.spawn_preprocess_report:
            report_path = Path(args.spawn_preprocess_report)
            if not report_path.is_absolute():
                report_path = out_dir / report_path
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"[INFO] Spawn preprocess report written to {report_path}")
            except Exception as exc:
                print(f"[WARN] Failed to write spawn preprocess report: {exc}")

    # Write ego route (optional)
    ego_entries: List[dict] = []
    if not args.no_ego and ego_trajs:
        # Remove legacy ego_route.xml to avoid double-counting egos
        legacy_ego = out_dir / "ego_route.xml"
        if legacy_ego.exists():
            try:
                legacy_ego.unlink()
                print(f"[INFO] Removed legacy ego file {legacy_ego}")
            except Exception:
                pass
        for ego_idx, ego_traj in enumerate(ego_trajs):
            ego_times = ego_times_list[ego_idx] if ego_idx < len(ego_times_list) else []
            # Follow CustomRoutes naming: {town}_custom_ego_vehicle_{i}.xml
            ego_xml = out_dir / f"{args.town.lower()}_custom_ego_vehicle_{ego_idx}.xml"
            write_route_xml(
                ego_xml,
                route_id=args.route_id,
                role="ego",
                town=args.town,
                waypoints=ego_traj,
                times=ego_times if args.encode_timing else None,
                snap_to_road=False,
                xml_tx=args.xml_tx,
                xml_ty=args.xml_ty,
            )
            ego_entries.append({
                "file": ego_xml.name,
                "route_id": str(args.route_id),
                "town": args.town,
                "name": ego_xml.stem,
                "kind": "ego",
                "model": args.ego_model,
            })

    # Build actor entries after we know obj_type/model
    # Group by kind (npc, static, etc.) for manifest
    actors_by_kind: Dict[str, List[dict]] = {}
    actor_kind_by_id: Dict[int, str] = {}
    actor_xml_by_id: Dict[int, Path] = {}

    for vid, traj in vehicles.items():
        if not traj:
            continue
        meta = actor_meta_by_id.get(vid)
        if meta is None:
            continue
        obj_type_raw = str(meta.get("obj_type") or "npc")
        kind = str(meta.get("kind"))
        model = meta.get("model") or map_obj_type(obj_type_raw)
        length = meta.get("length")
        width = meta.get("width")
        
        # Use obj_type directly for actor type in filename
        # Clean it up to make it suitable for filenames
        actor_type = obj_type_raw.replace(" ", "_").replace("-", "_").title()
        if not actor_type or actor_type.lower() == "npc":
            actor_type = "Vehicle"
        
        # Follow CustomRoutes naming: {town}_custom_{ActorType}_{id}_{kind}.xml
        name = f"{args.town.lower()}_custom_{actor_type}_{vid}_{kind}"
        
        # Create subdirectory for actor kind
        kind_dir = actors_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        actor_xml = kind_dir / f"{name}.xml"

        write_route_xml(
            actor_xml,
            route_id=args.route_id,
            role=kind,
            town=args.town,
            waypoints=traj,
            times=vehicle_times.get(vid) if args.encode_timing else None,
            snap_to_road=args.snap_to_road is True,
            xml_tx=args.xml_tx,
            xml_ty=args.xml_ty,
        )
        actor_xml_by_id[vid] = actor_xml
        speed = 0.0
        if len(traj) >= 2:
            dist = 0.0
            for a, b in zip(traj, traj[1:]):
                dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
            if args.encode_timing:
                times = vehicle_times.get(vid)
                if times and len(times) == len(traj):
                    total_time = times[-1] - times[0]
                    if total_time > 1e-6:
                        speed = dist / total_time
                    else:
                        speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
                else:
                    speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
            else:
                speed = dist / max(args.dt * (len(traj) - 1), 1e-6)
        
        entry = {
            "file": str(actor_xml.relative_to(out_dir)),
            "route_id": str(args.route_id),
            "town": args.town,
            "name": name,
            "kind": kind,
            "model": model,
        }
        
        # Add optional fields
        if speed > 0:
            entry["speed"] = speed
        if length is not None:
            entry["length"] = str(length) if isinstance(length, (int, float)) else length
        if width is not None:
            entry["width"] = str(width) if isinstance(width, (int, float)) else width
        
        if kind not in actors_by_kind:
            actors_by_kind[kind] = []
        actors_by_kind[kind].append(entry)
        actor_kind_by_id[vid] = kind

    save_manifest(out_dir / "actors_manifest.json", actors_by_kind, ego_entries)

    # Optional visualization
    if (
        args.gif
        or args.paths_png
        or args.spawn_viz
        or args.actor_yaw_viz_ids
        or args.actor_raw_yaml_viz_ids
        or args.spawn_preprocess_fail_viz
    ):
        if plt is None or (args.gif and imageio is None):
            raise SystemExit("matplotlib (and imageio for GIF) are required for visualization")
        map_lines: List[List[Tuple[float, float]]] = []
        map_bounds = None
        # Priority: explicit map pickle -> CARLA live map (with cache) -> none
        if args.map_pkl:
            try:
                map_lines = load_vector_map_from_pickle(Path(args.map_pkl).expanduser())
                print(f"[INFO] Loaded {len(map_lines)} polylines from {args.map_pkl}")
            except Exception as exc:
                print(f"[WARN] Failed to load map pickle {args.map_pkl}: {exc}")
        elif args.use_carla_map:
            cache_path = Path(args.carla_cache or (out_dir / "carla_map_cache.pkl"))
            try:
                sample = float(args.carla_sample)
                if args.spawn_preprocess_fail_viz:
                    sample = min(sample, float(args.spawn_preprocess_fail_viz_sample))
                map_lines, map_bounds = fetch_carla_map_lines(
                    host=args.carla_host,
                    port=args.carla_port,
                    sample=sample,
                    cache_path=cache_path,
                    expected_town=args.expected_town,
                )
                if map_lines:
                    print(f"[INFO] Loaded {len(map_lines)} map polylines from CARLA ({args.carla_host}:{args.carla_port})")
            except Exception as exc:
                print(f"[WARN] Failed to fetch map from CARLA: {exc}")
                map_bounds = None
        else:
            map_bounds = None
        if args.gif:
            frames_dir = out_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            max_len = max((len(t) for t in vehicles.values()), default=0)
            for et in ego_trajs:
                max_len = max(max_len, len(et))
            axes_limits = None
            # Precompute global limits for stable camera
            xs: List[float] = []
            ys: List[float] = []
            for traj in vehicles.values():
                xs.extend([wp.x for wp in traj])
                ys.extend([wp.y for wp in traj])
            for et in ego_trajs:
                for wp in et:
                    xs.append(wp.x)
                    ys.append(wp.y)
            for line in map_lines:
                for x, y in line:
                    xs.append(x)
                    ys.append(y)
            if xs and ys:
                pad = max(0.0, float(args.axis_pad))
                axes_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

            for i in range(max_len):
                plot_frame(
                    i,
                    vehicles,
                    ego_trajs,
                    frames_dir / f"frame_{i:06d}.png",
                    axes_limits,
                    map_lines=map_lines,
                    invert_plot_y=args.invert_plot_y,
                )
            gif_path = Path(args.gif_path or (out_dir / "replay.gif"))
            write_gif(frames_dir, gif_path)
            print(f"[OK] GIF written to {gif_path}")

        if args.paths_png:
            png_path = Path(args.paths_png).expanduser()
            write_paths_png(
                actors_by_id=vehicles,
                ego_trajs=ego_trajs,
                map_lines=map_lines,
                out_path=png_path,
                axis_pad=float(args.axis_pad),
                invert_plot_y=args.invert_plot_y,
            )
            print(f"[OK] Paths PNG written to {png_path}")

        if args.actor_yaw_viz_ids:
            actor_ids = _parse_id_list(args.actor_yaw_viz_ids)
            out_dir_yaw = Path(args.actor_yaw_viz_dir or (out_dir / "actor_yaw_viz")).expanduser()
            out_dir_yaw.mkdir(parents=True, exist_ok=True)
            for vid in actor_ids:
                gt_traj = vehicles.get(vid) or []
                xml_path = actor_xml_by_id.get(vid)
                if xml_path is None:
                    # best-effort fallback search
                    matches = list(actors_dir.rglob(f"*_{vid}_*.xml"))
                    xml_path = matches[0] if matches else None
                if not gt_traj:
                    print(f"[WARN] No GT trajectory found for actor id {vid}")
                    continue
                if xml_path is None or not xml_path.exists():
                    print(f"[WARN] No XML found for actor id {vid}")
                    continue
                xml_traj = parse_route_xml(xml_path)
                if not xml_traj:
                    print(f"[WARN] XML had no waypoints for actor id {vid}: {xml_path}")
                    continue
                out_path = out_dir_yaw / f"actor_{vid}_yaw_viz.png"
                write_actor_yaw_viz(
                    actor_id=vid,
                    gt_traj=gt_traj,
                    xml_traj=xml_traj,
                    map_lines=map_lines,
                    out_path=out_path,
                    arrow_step=max(1, int(args.actor_yaw_viz_step)),
                    arrow_len=float(args.actor_yaw_viz_arrow_len),
                    pad=float(args.actor_yaw_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor yaw viz written to {out_path}")

        if args.actor_raw_yaml_viz_ids:
            actor_ids = _parse_id_list(args.actor_raw_yaml_viz_ids)
            out_dir_raw = Path(args.actor_raw_yaml_viz_dir or (out_dir / "actor_raw_yaml_viz")).expanduser()
            out_dir_raw.mkdir(parents=True, exist_ok=True)

            # Collect per-subdir points directly from YAML (with transform + XML offsets applied)
            points_by_actor: Dict[int, Dict[str, List[Tuple[float, float, float]]]] = {vid: {} for vid in actor_ids}
            for yd in yaml_dirs:
                sub_name = yd.name
                yaml_paths = list_yaml_timesteps(yd)
                for idx, path in enumerate(yaml_paths):
                    try:
                        frame_idx = int(path.stem)
                    except Exception:
                        frame_idx = idx
                    t = float(frame_idx) * float(args.dt)
                    data = load_yaml(path)
                    vehs = data.get("vehicles", {}) or {}
                    for vid in actor_ids:
                        payload = vehs.get(vid) if vid in vehs else vehs.get(str(vid))
                        if not payload:
                            continue
                        loc = payload.get("location") or [0, 0, 0]
                        x0 = float(loc[0]) if len(loc) > 0 else 0.0
                        y0 = float(loc[1]) if len(loc) > 1 else 0.0
                        x, y = apply_se2((x0, y0), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)
                        x += float(args.xml_tx)
                        y += float(args.xml_ty)
                        points_by_actor.setdefault(vid, {}).setdefault(sub_name, []).append((x, y, t))

            for vid in actor_ids:
                points = points_by_actor.get(vid, {})
                if not points:
                    print(f"[WARN] No YAML points found for actor id {vid}")
                    continue
                out_path = out_dir_raw / f"actor_{vid}_raw_yaml_points.png"
                write_actor_raw_yaml_viz(
                    actor_id=vid,
                    points_by_subdir=points,
                    map_lines=map_lines,
                    out_path=out_path,
                    pad=float(args.actor_raw_yaml_viz_pad),
                    invert_plot_y=args.invert_plot_y,
                )
                print(f"[OK] Actor raw YAML viz written to {out_path}")

        if args.spawn_viz:
            spawn_viz_path = Path(args.spawn_viz_path or (out_dir / "spawn_alignment_viz.png")).expanduser()

            aligned_points: Dict[int, Tuple[float, float]] = {}
            spawn_points: Dict[int, Tuple[float, float]] = {}
            pre_align_points: Dict[int, Tuple[float, float]] = {}
            for vid, traj in vehicles.items():
                if not traj:
                    continue
                wp0 = traj[0]
                aligned_points[vid] = (wp0.x, wp0.y)
                spawn_points[vid] = (wp0.x + args.xml_tx, wp0.y + args.xml_ty)
                pre_align_points[vid] = invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y)

            ego_aligned: List[Tuple[float, float]] = []
            ego_spawn: List[Tuple[float, float]] = []
            ego_pre_align: List[Tuple[float, float]] = []
            for ego_traj in ego_trajs:
                if not ego_traj:
                    continue
                wp0 = ego_traj[0]
                ego_aligned.append((wp0.x, wp0.y))
                ego_spawn.append((wp0.x + args.xml_tx, wp0.y + args.xml_ty))
                ego_pre_align.append(invert_se2((wp0.x, wp0.y), args.yaw_deg, args.tx, args.ty, flip_y=args.flip_y))

            offset_pair = None
            offset_label = None
            offset_ref = _pick_offset_reference(aligned_points, pre_align_points, ego_aligned, ego_pre_align)
            if offset_ref:
                offset_pair = (offset_ref[0], offset_ref[1])
                offset_label = f"pre-align ref ({offset_ref[2]})"

            xodr_points: List[Tuple[float, float]] = []
            xodr_path = Path(args.xodr).expanduser() if args.xodr else None
            if xodr_path and xodr_path.exists():
                try:
                    xodr_points = load_xodr_points(xodr_path, args.xodr_step)
                    print(f"[INFO] Loaded {len(xodr_points)} XODR points from {xodr_path}")
                except Exception as exc:
                    print(f"[WARN] Failed to load XODR {xodr_path}: {exc}")
                    xodr_points = []
            else:
                # Best-effort default XODR next to repo root (if present)
                default_xodr = Path(__file__).resolve().parents[2] / "ucla_v2.xodr"
                if default_xodr.exists():
                    try:
                        xodr_points = load_xodr_points(default_xodr, args.xodr_step)
                        print(f"[INFO] Loaded {len(xodr_points)} XODR points from {default_xodr}")
                    except Exception as exc:
                        print(f"[WARN] Failed to load XODR {default_xodr}: {exc}")

            map_image = None
            map_image_bounds = None
            if args.map_image:
                try:
                    map_image = plt.imread(args.map_image)
                    if args.map_image_bounds:
                        map_image_bounds = tuple(args.map_image_bounds)  # type: ignore
                except Exception as exc:
                    print(f"[WARN] Failed to load map image {args.map_image}: {exc}")
                    map_image = None

            map_points: List[Tuple[float, float]] = []
            for line in map_lines:
                map_points.extend(line)

            kind_by_id = dict(actor_kind_by_id)
            for vid in aligned_points.keys():
                kind_by_id.setdefault(vid, "npc")

            bounds = _merge_bounds(
                [
                    map_bounds,
                    _bounds_from_points(map_points),
                    _bounds_from_points(xodr_points),
                    _bounds_from_points(
                        list(aligned_points.values())
                        + list(spawn_points.values())
                        + list(pre_align_points.values())
                        + ego_aligned
                        + ego_spawn
                        + ego_pre_align
                    ),
                    map_image_bounds,
                ]
            )
            if bounds:
                pad = max(0.0, float(args.axis_pad))
                minx, maxx, miny, maxy = bounds
                bounds = (minx - pad, maxx + pad, miny - pad, maxy + pad)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            ax_map, ax_xodr = axes

            # Left: CARLA map layer (image if provided, else polylines)
            if map_image is not None:
                if map_image_bounds:
                    minx, maxx, miny, maxy = map_image_bounds
                elif bounds:
                    minx, maxx, miny, maxy = bounds
                else:
                    minx, maxx, miny, maxy = 0.0, 1.0, 0.0, 1.0
                ax_map.imshow(map_image, extent=(minx, maxx, miny, maxy), origin="lower", alpha=0.8, zorder=0)
            elif map_lines:
                _plot_background_lines(ax_map, map_lines, color="#9e9e9e", lw=0.8, alpha=0.6)
            _plot_spawn_alignment(
                ax_map,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="CARLA Map Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            # Right: XODR layer
            if xodr_points:
                ax_xodr.scatter(
                    [p[0] for p in xodr_points],
                    [p[1] for p in xodr_points],
                    s=1,
                    c="#1f77b4",
                    alpha=0.5,
                    label="XODR geometry",
                    zorder=1,
                )
            _plot_spawn_alignment(
                ax_xodr,
                aligned_points,
                spawn_points,
                kind_by_id,
                ego_aligned,
                ego_spawn,
                title="XODR Layer",
                offset_pair=offset_pair,
                offset_label=offset_label,
            )

            for ax in axes:
                if bounds:
                    minx, maxx, miny, maxy = bounds
                    ax.set_xlim(minx, maxx)
                    ax.set_ylim(miny, maxy)
                if args.invert_plot_y:
                    ax.invert_yaxis()

            # Global legend and info
            handles, labels = ax_map.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", frameon=True)

            info_lines = [
                f"Actors: {len(aligned_points)} (npc={sum(1 for k in kind_by_id.values() if k == 'npc')}, "
                f"static={sum(1 for k in kind_by_id.values() if k == 'static')}, "
                f"walker={sum(1 for k in kind_by_id.values() if k in ('walker', 'walker_static'))})",
                f"Egos: {len(ego_aligned)}",
                f"Alignment tx/ty/yaw: {args.tx:.2f}, {args.ty:.2f}, {args.yaw_deg:.2f}",
                f"XML offset xml_tx/xml_ty: {args.xml_tx:.2f}, {args.xml_ty:.2f}",
                f"flip_y: {args.flip_y}",
            ]
            fig.text(0.01, 0.01, "\n".join(info_lines), fontsize=9, ha="left", va="bottom")
            fig.suptitle("Spawn vs Aligned Positions (CARLA vs XODR)", fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(spawn_viz_path, dpi=180)
            plt.close(fig)
            print(f"[OK] Spawn alignment visualization written to {spawn_viz_path}")

        if args.spawn_preprocess_fail_viz:
            if spawn_report is None:
                print("[WARN] Spawn preprocess visualization requested but no report available.")
            else:
                fail_dir = args.spawn_preprocess_fail_viz_dir
                if not fail_dir:
                    fail_dir = str(out_dir / "spawn_preprocess_fail_viz")
                out_path = Path(fail_dir).expanduser()
                _plot_failed_spawn_visualizations(
                    report=spawn_report,
                    map_lines=map_lines,
                    out_dir=out_path,
                    window_m=float(args.spawn_preprocess_fail_viz_window),
                    dpi=int(args.spawn_preprocess_fail_viz_dpi),
                )
                print(f"[OK] Failed spawn visualization written to {out_path}")

    # Optional: run custom eval with generated routes
    if args.run_custom_eval:
        repo_root = Path(__file__).resolve().parents[2]
        python_bin = sys.executable
        cmd = [
            python_bin,
            str(repo_root / "tools" / "run_custom_eval.py"),
            "--routes-dir",
            str(out_dir),
            "--port",
            str(args.eval_port),
            "--overwrite",
        ]
        if args.eval_planner:
            cmd.extend(["--planner", args.eval_planner])
        print("[INFO] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] run_custom_eval failed with exit code {exc.returncode}")

    print(f"[OK] Export complete -> {out_dir}")
    print("Files:")
    if ego_entries:
        for entry in ego_entries:
            print(f"  - {entry['file']}")
    print(f"  - actors_manifest.json")
    total_actors = sum(len(entries) for entries in actors_by_kind.values())
    print(f"  - actors/*/*.xml ({total_actors} actors across {len(actors_by_kind)} categories)")
    for kind, entries in sorted(actors_by_kind.items()):
        print(f"    - {kind}: {len(entries)} actors")
    if args.gif:
        print(f"  - replay.gif")


if __name__ == "__main__":
    main()
