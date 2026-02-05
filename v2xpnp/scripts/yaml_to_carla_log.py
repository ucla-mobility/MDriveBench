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
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def map_obj_type(obj_type: str | None) -> str:
    """Map dataset obj_type to a CARLA 0.9.12 blueprint (vehicle or walker)."""
    if not obj_type:
        return "vehicle.tesla.model3"
    ot = obj_type.lower()
    
    # Pedestrians/Walkers
    if is_pedestrian_type(obj_type):
        return "walker.pedestrian.0001"
    
    # Buses and large vehicles
    if "bus" in ot:
        return "vehicle.volkswagen.t2"
    
    # Trucks and vans
    if "truck" in ot:
        if "fire" in ot:
            return "vehicle.carlamotors.firetruck"
        return "vehicle.carlamotors.carlacola"
    if "van" in ot or "sprinter" in ot:
        return "vehicle.mercedes.sprinter"
    
    # Emergency vehicles
    if "ambulance" in ot:
        return "vehicle.ford.ambulance"
    if "police" in ot:
        return "vehicle.dodge.charger_police_2020"
    
    # Motorcycles
    if "motor" in ot or "motorcycle" in ot:
        return "vehicle.harley-davidson.low_rider"
    if "bike" in ot and "bicycle" not in ot:
        return "vehicle.yamaha.yzf"
    
    # Bicycles (with rider)
    if "bicycle" in ot or "cycl" in ot:
        return "vehicle.diamondback.century"
    
    # SUVs and larger cars
    if "suv" in ot or "jeep" in ot:
        return "vehicle.jeep.wrangler_rubicon"
    if "patrol" in ot:
        return "vehicle.nissan.patrol_2021"
    
    # Sedans and cars (default category)
    return "vehicle.tesla.model3"


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0
    roll: float = 0.0


# ---------------------- Core conversion ---------------------- #

def build_trajectories(
    yaml_dir: Path,
    dt: float,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    flip_y: bool = False,
) -> Tuple[Dict[int, List[Waypoint]], List[Waypoint]]:
    """Parse YAML sequence into per-vehicle trajectories and ego path."""
    yaml_paths = list_yaml_timesteps(yaml_dir)
    if not yaml_paths:
        raise SystemExit(f"No YAML files found under {yaml_dir}")

    vehicles: Dict[int, List[Waypoint]] = {}
    ego_traj: List[Waypoint] = []

    for path in yaml_paths:
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

        vehs = data.get("vehicles", {}) or {}
        for vid_str, payload in vehs.items():
            try:
                vid = int(vid_str)
            except Exception:
                continue
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

    # Compute simple average speed (m/s) per vehicle from path length
    speeds: Dict[int, float] = {}
    for vid, traj in vehicles.items():
        dist = 0.0
        for a, b in zip(traj, traj[1:]):
            dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
        speeds[vid] = dist / max(dt * max(len(traj) - 1, 1), 1e-6)

    return vehicles, ego_traj


def write_route_xml(
    path: Path,
    route_id: str,
    role: str,
    town: str,
    waypoints: List[Waypoint],
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
    for wp in waypoints:
        # Normalize pitch and roll: CARLA XML expects pitch=360.0 (or 0.0) and roll=0.0
        # Using 360.0 for pitch as seen in reference XML files
        ET.SubElement(
            route,
            "waypoint",
            {
                "x": f"{wp.x + xml_tx:.6f}",
                "y": f"{wp.y + xml_ty:.6f}",
                "z": f"{wp.z:.6f}",
                "yaw": f"{wp.yaw:.6f}",
                "pitch": "360.000000",  # Normalized for CARLA compatibility
                "roll": "0.000000",     # Normalized for CARLA compatibility
            },
        )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


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
    ego_traj: List[Waypoint],
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

    if ego_traj:
        idx = min(timestep, len(ego_traj) - 1)
        ego = ego_traj[idx]
        tri = patches.RegularPolygon(
            (ego.x, ego.y),
            numVertices=3,
            radius=2.5,
            orientation=math.radians(ego.yaw),
            color="orange",
            alpha=0.6,
        )
        ax.add_patch(tri)
        ax.text(ego.x, ego.y, "ego", fontsize=7, ha="center", va="center")
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
    ego_traj: List[Waypoint],
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

    # Ego
    if ego_traj:
        lx = [wp.x for wp in ego_traj]
        ly = [wp.y for wp in ego_traj]
        ax.plot(lx, ly, color="black", linewidth=2.0, alpha=0.8, label="ego")
        ax.scatter(lx[0], ly[0], s=30, marker="*", color="black")
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


# ---------------------- CLI ---------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert V2XPnP YAML logs to CARLA route XML + manifest")
    p.add_argument("--scenario-dir", required=True, help="Path to the scenario folder containing subfolders with YAML frames")
    p.add_argument("--subdir", default=None, help="Specific subfolder inside scenario-dir to use (e.g., -1). If omitted and only one subfolder exists, it is used; if YAMLs live directly in scenario-dir, they are used.")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <scenario-dir>/carla_log_export)")
    p.add_argument("--route-id", default="0", help="Route id to assign to ego and actors (default: 0)")
    p.add_argument("--town", default="ucla_v2", help="CARLA town/map name to embed in XML (default: ucla_v2)")
    p.add_argument("--ego-name", default="ego", help="Name for ego vehicle")
    p.add_argument("--ego-model", default="vehicle.lincoln.mkz2017", help="Blueprint for ego vehicle")
    p.add_argument("--dt", type=float, default=0.1, help="Timestep spacing in seconds (for speed estimation)")
    p.add_argument("--tx", type=float, default=0.0, help="Translation X to apply to all coordinates")
    p.add_argument("--ty", type=float, default=0.0, help="Translation Y to apply to all coordinates")
    p.add_argument("--tz", type=float, default=0.0, help="Translation Z to apply to all coordinates")
    p.add_argument("--xml-tx", type=float, default=0.0, help="Additional X offset applied only when writing XML outputs")
    p.add_argument("--xml-ty", type=float, default=0.0, help="Additional Y offset applied only when writing XML outputs")
    p.add_argument(
        "--coord-json",
        default=None,
        help="Optional JSON file containing transform keys like tx, ty, theta_deg/rad, flip_y; applied to all coordinates",
    )
    p.add_argument("--yaw-deg", type=float, default=0.0, help="Global yaw rotation (degrees, applied before translation)")
    p.add_argument("--snap-to-road", action="store_true", help="Enable road snapping for actors (defaults to off)")
    p.add_argument("--no-ego", action="store_true", help="Skip writing ego_route.xml")
    p.add_argument("--gif", action="store_true", help="Generate GIF visualization")
    p.add_argument("--gif-path", default=None, help="Path for GIF (default: <out-dir>/replay.gif)")
    p.add_argument("--paths-png", default=None, help="If set, render a single PNG with each actor's full path as a polyline")
    p.add_argument("--map-pkl", default=None, help="Optional pickle containing vector map polylines to overlay")
    p.add_argument("--use-carla-map", action="store_true", help="Connect to CARLA to fetch map polylines for overlay")
    p.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    p.add_argument("--carla-port", type=int, default=2010, help="CARLA port (default: 2010)")
    p.add_argument("--carla-sample", type=float, default=2.0, help="Waypoint sampling distance in meters (default: 2.0)")
    p.add_argument("--carla-cache", default=None, help="Path to cache map polylines (default: <out-dir>/carla_map_cache.pkl)")
    p.add_argument("--expected-town", default="ucla_v2", help="Assert CARLA map name contains this string when using --use-carla-map")
    p.add_argument("--axis-pad", type=float, default=10.0, help="Padding (meters) around actor/ego extents for visualization axes")
    p.add_argument("--flip-y", action="store_true", help="Mirror dataset Y axis and negate yaw (useful if overlay appears upside-down)")
    p.add_argument("--invert-plot-y", action="store_true", help="Invert matplotlib Y axis for visualization only")
    p.add_argument("--run-custom-eval", action="store_true", help="After export, call tools/run_custom_eval.py with the generated routes dir")
    p.add_argument(
        "--eval-planner",
        default="",
        help="Planner for run_custom_eval (empty string means no planner flag; e.g., pass 'tcp' or 'log_replay')",
    )
    p.add_argument("--eval-port", type=int, default=2014, help="CARLA port for run_custom_eval (default: 2014)")
    return p.parse_args()


def pick_yaml_dir(scenario_dir: Path, chosen: str | None) -> Path:
    subdirs = [d for d in scenario_dir.iterdir() if d.is_dir()]
    if chosen:
        cand = scenario_dir / chosen
        if not cand.is_dir():
            raise SystemExit(f"--subdir {chosen} not found under {scenario_dir}")
        return cand
    if list_yaml_timesteps(scenario_dir):
        return scenario_dir
    if len(subdirs) == 1:
        return subdirs[0]
    raise SystemExit("Multiple subfolders found; specify one with --subdir")


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


def load_vector_map_from_pickle(path: Path) -> List[List[Tuple[float, float]]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    return _extract_map_lines(obj, out=[])


def fetch_carla_map_lines(host: str, port: int, sample: float, cache_path: Path | None, expected_town: str | None = None) -> Tuple[List[List[Tuple[float, float]]], Tuple[float, float, float, float] | None]:
    """Connect to CARLA, ensure the desired map is loaded, sample waypoints, and optionally cache."""
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

    # Cache reuse only if map name matches expectation (if provided)
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
    yaml_dir = pick_yaml_dir(scenario_dir, args.subdir)
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

    vehicles, ego_traj = build_trajectories(
        yaml_dir=yaml_dir,
        dt=args.dt,
        tx=args.tx,
        ty=args.ty,
        tz=args.tz,
        yaw_deg=args.yaw_deg,
        flip_y=args.flip_y,
    )

    # Write ego route (optional)
    ego_entries: List[dict] = []
    if not args.no_ego and ego_traj:
        # Remove legacy ego_route.xml to avoid double-counting egos
        legacy_ego = out_dir / "ego_route.xml"
        if legacy_ego.exists():
            try:
                legacy_ego.unlink()
                print(f"[INFO] Removed legacy ego file {legacy_ego}")
            except Exception:
                pass

        # Follow CustomRoutes naming: {town}_custom_ego_vehicle_0.xml
        ego_xml = out_dir / f"{args.town.lower()}_custom_ego_vehicle_0.xml"
        write_route_xml(
            ego_xml,
            route_id=args.route_id,
            role="ego",
            town=args.town,
            waypoints=ego_traj,
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

    # Gather first-timestep info to set obj_type/model/size
    obj_info: Dict[int, Dict[str, object]] = {}
    first_yaml = next(iter(list_yaml_timesteps(yaml_dir)), None)
    if first_yaml:
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

    # Build actor entries after we know obj_type/model
    # Group by kind (npc, static, etc.) for manifest
    actors_by_kind: Dict[str, List[dict]] = {}
    skipped_non_vehicles = 0
    
    for vid, traj in vehicles.items():
        if not traj:
            continue
        info = obj_info.get(vid, {})
        obj_type_raw = str(info.get("obj_type", "npc"))
        
        # Skip non-vehicle, non-pedestrian types (trash cans, props, etc.)
        if not is_vehicle_type(obj_type_raw):
            skipped_non_vehicles += 1
            continue
        
        # Check if this is a pedestrian/walker
        is_pedestrian = is_pedestrian_type(obj_type_raw)
        
        # Determine kind: pedestrians get "walker", vehicles get "npc" or "static"
        if is_pedestrian:
            kind = "walker"  # Pedestrians are always walkers
            # Optionally distinguish stationary walkers
            if len(traj) >= 2:
                dist = 0.0
                for a, b in zip(traj, traj[1:]):
                    dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
                if dist < 0.5:
                    kind = "walker_static"
        else:
            # For vehicles: determine if static or dynamic
            kind = "npc"  # default
            if len(traj) <= 1:
                kind = "static"
            elif len(traj) >= 2:
                # Check if vehicle is essentially stationary
                dist = 0.0
                for a, b in zip(traj, traj[1:]):
                    dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
                if dist < 0.5:  # Less than 0.5m total movement = static
                    kind = "static"
        
        model = info.get("model") or map_obj_type(obj_type_raw)
        length = info.get("length")
        width = info.get("width")
        
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
            snap_to_road=args.snap_to_road is True,
            xml_tx=args.xml_tx,
            xml_ty=args.xml_ty,
        )
        speed = 0.0
        if len(traj) >= 2:
            dist = 0.0
            for a, b in zip(traj, traj[1:]):
                dist += euclid3((a.x, a.y, a.z), (b.x, b.y, b.z))
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

    if skipped_non_vehicles > 0:
        print(f"[INFO] Skipped {skipped_non_vehicles} non-actor objects (props, static objects, etc.)")

    save_manifest(out_dir / "actors_manifest.json", actors_by_kind, ego_entries)

    # Optional visualization
    if args.gif or args.paths_png:
        if plt is None or imageio is None:
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
                map_lines, map_bounds = fetch_carla_map_lines(
                    host=args.carla_host,
                    port=args.carla_port,
                    sample=args.carla_sample,
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
            max_len = max(max_len, len(ego_traj))
            axes_limits = None
            # Precompute global limits for stable camera
            xs: List[float] = []
            ys: List[float] = []
            for traj in vehicles.values():
                xs.extend([wp.x for wp in traj])
                ys.extend([wp.y for wp in traj])
            for wp in ego_traj:
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
                    ego_traj,
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
                ego_traj=ego_traj,
                map_lines=map_lines,
                out_path=png_path,
                axis_pad=float(args.axis_pad),
                invert_plot_y=args.invert_plot_y,
            )
            print(f"[OK] Paths PNG written to {png_path}")

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
