#!/usr/bin/env python3
"""Aggregate vehicle annotations across V2XPnP sample folders into a wide CSV and optional viz.

Given a top-level folder (e.g., /data2/marco/CoLMDriver/v2xpnp/V2XPnP_Sample_Data/2023-03-17-16-12-12_3_0),
this script loads every YAML under each immediate child folder (e.g., -1, -2, 1, 2).
It produces a CSV where:
  - Each row is a vehicle id.
  - The second column is the obj_type (from the first sighting).
  - Each subsequent column corresponds to a timestep (basename of the YAML, like 000001).
    The cell value is a JSON object containing location, extent, angle, attribute, and the
    source folder that contributed that timestep.
Missing detections are left blank.

If --viz is provided, a PNG is written per timestep showing top-down boxes colored by obj_type.

Usage:
    python tools/aggregate_vehicles.py \
        --base-dir /data2/marco/CoLMDriver/v2xpnp/V2XPnP_Sample_Data/2023-03-17-16-12-12_3_0 \
                --output vehicles_by_timestep.csv \
                --viz
"""

import argparse
import csv
import json
import math
import os
import pickle
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches, transforms
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:  # optional CARLA client for map overlay
    import carla  # type: ignore
except Exception:
    carla = None

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate vehicles across V2XPnP subfolders")
    parser.add_argument("--base-dir", required=True, help="Top-level directory containing subfolders (-1, -2, 1, 2, ...)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <base-dir>/vehicles_by_timestep.csv",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Allowed numeric difference when checking for consistency across folders",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If set, generate per-timestep visualizations",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="Directory to save visualization PNGs (defaults to <base-dir>/viz)",
    )
    parser.add_argument(
        "--viz-over-time",
        action="store_true",
        help="If set, also stitch per-timestep frames into a single GIF",
    )
    parser.add_argument(
        "--viz-video",
        default=None,
        help="Path for the GIF/animation (defaults to <base-dir>/vehicles_over_time.gif)",
    )
    parser.add_argument(
        "--viz-video-full",
        default=None,
        help="Path for a second GIF zoomed out to include all map lines (defaults to <base-dir>/vehicles_over_time_full.gif)",
    )
    parser.add_argument(
        "--plot-ego",
        action="store_true",
        default=True,
        help="Plot ego pose for each timestep (drawn as a triangle)",
    )
    parser.add_argument(
        "--no-plot-ego",
        action="store_false",
        dest="plot_ego",
        help="Disable ego pose plotting",
    )
    parser.add_argument(
        "--map-pkl",
        default=None,
        help="Optional PKL path with map polylines to overlay on visualization",
    )
    parser.add_argument(
        "--map-color-pkl",
        default="gray",
        help="Color for PKL map overlay (default: gray)",
    )
    parser.add_argument(
        "--use-carla-map",
        action="store_true",
        help="Fetch map polylines directly from a running CARLA server",
    )
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--carla-sample",
        type=float,
        default=2.0,
        help="Waypoint sampling distance in meters when pulling map from CARLA (default: 2.0)",
    )
    parser.add_argument(
        "--carla-cache",
        default=None,
        help="Cache file for CARLA map polylines (default: <base-dir>/carla_map_cache.pkl)",
    )
    parser.add_argument(
        "--carla-map-offset-json",
        default=None,
        help="Optional JSON file with tx/ty to offset CARLA map overlay (fields: tx, ty)",
    )
    parser.add_argument(
        "--carla-map-flip-y",
        action="store_true",
        help="Flip CARLA map Y coordinates before applying offset (useful if axis is inverted)",
    )
    parser.add_argument(
        "--expected-town",
        default=None,
        help="If set, require CARLA map name to contain this substring (e.g., ucla_v2)",
    )
    parser.add_argument(
        "--map-color-carla",
        default="#cfa93a",
        help="Color for CARLA/UCLA map overlay (default: #cfa93a)",
    )
    return parser.parse_args()


def list_subfolders(base_dir: str) -> List[str]:
    children = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            children.append(path)
    return children


def list_timesteps(folder: str) -> List[str]:
    steps = set()
    for name in os.listdir(folder):
        if name.endswith(".yaml"):
            steps.add(os.path.splitext(name)[0])
    return sorted(steps)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def nearly_equal(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, abs_tol=tol, rel_tol=0)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(nearly_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(nearly_equal(a[k], b[k], tol) for k in a.keys())
    return a == b


def merge_vehicle_entry(
    existing: Dict[str, Any],
    new_entry: Dict[str, Any],
    tol: float,
    warnings: List[str],
    source: str,
    timestep: str,
) -> Dict[str, Any]:
    # If we already stored data for this timestep, prefer consistency and keep the first.
    if timestep in existing:
        old = existing[timestep]
        if not nearly_equal(old.get("location"), new_entry.get("location"), tol):
            old_loc = old.get("location") or []
            new_loc = new_entry.get("location") or []
            diffs = []
            if isinstance(old_loc, list) and isinstance(new_loc, list) and len(old_loc) == len(new_loc):
                try:
                    diffs = [float(n) - float(o) for o, n in zip(old_loc, new_loc)]
                except Exception:
                    diffs = []
            max_abs = max((abs(d) for d in diffs), default=float("nan"))
            l2 = math.sqrt(sum(d * d for d in diffs)) if diffs else float("nan")
            warnings.append(
                f"Mismatch at timestep {timestep} for vehicle {existing.get('vehicle_id', 'unknown')} between {old.get('source')} and {source}; "
                f"loc delta max_abs={max_abs:.3f}, l2={l2:.3f}"
            )
        return existing
    existing[timestep] = {**new_entry, "source": source}
    return existing


def yaw_from_pose(pose: Any) -> float:
    """Extract yaw (degrees) from a 6-DOF pose array."""

    if isinstance(pose, Sequence) and len(pose) >= 5:
        return float(pose[4])
    return 0.0


def yaw_from_angle(angle: Any) -> float:
    """Extract yaw in degrees from the angle field.

    The dataset provides three Euler components; the middle one represents yaw in these files.
    Fallback to the last component if the length is unexpected.
    """

    if isinstance(angle, list) and len(angle) >= 2:
        return float(angle[1])
    if isinstance(angle, list) and angle:
        return float(angle[-1])
    return 0.0


def ensure_color(obj_type: str, palette: Dict[str, str], color_cycle: List[str]) -> str:
    if obj_type in palette:
        return palette[obj_type]
    color = color_cycle[len(palette) % len(color_cycle)]
    palette[obj_type] = color
    return color


def plot_timestep(
    timestep: str,
    vehicles: Dict[int, Dict[str, Any]],
    out_path: str,
    palette: Dict[str, str],
    color_cycle: List[str],
    ego_pose: Any = None,
    axes_limits: Tuple[float, float, float, float] = None,
    pkl_lines: List[List[Tuple[float, float]]] = None,
    carla_lines: List[List[Tuple[float, float]]] = None,
    map_color_pkl: str = "gray",
    map_color_carla: str = "#cfa93a",
    invert_plot_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Timestep {timestep}")
    ax.set_aspect("equal", adjustable="box")

    xs: List[float] = []
    ys: List[float] = []
    legend_handles: Dict[str, Any] = {}

    for vid, data in vehicles.items():
        if timestep not in data:
            continue
        vdata = data[timestep]
        loc = vdata.get("location")
        extent = vdata.get("extent") or [0.5, 0.5, 0.5]
        obj_type = data.get("obj_type", "") or "Unknown"
        if not loc or len(loc) < 2:
            continue

        x, y = float(loc[0]), float(loc[1])
        yaw_deg = yaw_from_angle(vdata.get("angle"))
        width = float(extent[1]) * 2 if len(extent) > 1 else 1.0
        height = float(extent[0]) * 2 if len(extent) > 0 else 1.0

        color = ensure_color(obj_type, palette, color_cycle)

        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.4,
        )
        rot = transforms.Affine2D().rotate_deg_around(x, y, yaw_deg) + ax.transData
        rect.set_transform(rot)
        ax.add_patch(rect)

        ax.text(x, y, f"{vid}", ha="center", va="center", fontsize=8, color="black")

        xs.append(x)
        ys.append(y)

        if obj_type not in legend_handles:
            legend_handles[obj_type] = patches.Patch(color=color, label=obj_type)

    if ego_pose:
        ex, ey = float(ego_pose[0]), float(ego_pose[1])
        yaw_deg = yaw_from_pose(ego_pose)
        ego_length = 4.5
        ego_width = 2.0
        color = ensure_color("ego", palette, color_cycle)
        triangle = patches.RegularPolygon(
            (ex, ey),
            numVertices=3,
            radius=ego_length / 2,
            orientation=math.radians(yaw_deg),
            color=color,
            alpha=0.6,
        )
        ax.add_patch(triangle)
        ax.text(ex, ey, "ego", ha="center", va="center", fontsize=7, color="black")
        xs.append(ex)
        ys.append(ey)
        if "ego" not in legend_handles:
            legend_handles["ego"] = patches.Patch(color=color, label="ego")

    if pkl_lines:
        for line in pkl_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color=map_color_pkl, linewidth=1.0, alpha=0.6, label="map_pkl" if "map_pkl" not in legend_handles else None)
            xs.extend(lx)
            ys.extend(ly)
            if "map_pkl" not in legend_handles:
                legend_handles["map_pkl"] = patches.Patch(color=map_color_pkl, label="map (PKL)")

    if carla_lines:
        for line in carla_lines:
            if len(line) < 2:
                continue
            lx = [p[0] for p in line]
            ly = [p[1] for p in line]
            ax.plot(lx, ly, color=map_color_carla, linewidth=1.0, alpha=0.6, label="map_carla" if "map_carla" not in legend_handles else None)
            xs.extend(lx)
            ys.extend(ly)
            if "map_carla" not in legend_handles:
                legend_handles["map_carla"] = patches.Patch(color=map_color_carla, label="map (CARLA)")

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
    if legend_handles:
        ax.legend(handles=list(legend_handles.values()), loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = os.path.abspath(args.base_dir)
    output_csv = args.output or os.path.join(base_dir, "vehicles_by_timestep.csv")
    viz_dir = args.viz_dir or os.path.join(base_dir, "viz")
    viz_video = args.viz_video or os.path.join(base_dir, "vehicles_over_time.gif")
    viz_video_full = args.viz_video_full or os.path.join(base_dir, "vehicles_over_time_full.gif")
    subfolders = list_subfolders(base_dir)
    if not subfolders:
        raise SystemExit(f"No subfolders found in {base_dir}")

    # Collect all timesteps across subfolders.
    timesteps: List[str] = []
    for folder in subfolders:
        timesteps.extend(list_timesteps(folder))
    timesteps = sorted(set(timesteps))
    if not timesteps:
        raise SystemExit("No YAML files found under subfolders")

    vehicles: Dict[int, Dict[str, Any]] = {}
    ego_by_ts: Dict[str, Any] = {}
    warnings: List[str] = []
    pkl_lines: List[List[Tuple[float, float]]] = []
    carla_lines: List[List[Tuple[float, float]]] = []
    map_bounds = None

    if args.map_pkl:
        map_obj = None

        class _StubClass:
            """Stub for missing classes during unpickling."""
            def __init__(self, *args, **kwargs):
                self.__dict__.update(kwargs)
                for i, arg in enumerate(args):
                    setattr(self, f"_arg{i}", arg)

        class _SafeUnpickler(pickle.Unpickler):
            def find_class(self, module, name):  # pragma: no cover
                try:
                    return super().find_class(module, name)
                except Exception:
                    return _StubClass

        def _try_load(loader):
            with open(args.map_pkl, "rb") as pf:
                return loader(pf)

        try:
            map_obj = _try_load(pickle.load)
        except Exception:
            try:
                map_obj = _try_load(lambda f: pickle.load(f, encoding="latin1"))
            except Exception:
                try:
                    map_obj = _SafeUnpickler(open(args.map_pkl, "rb")).load()
                except Exception as exc:  # pragma: no cover
                    warnings.append(f"Failed to load map PKL {args.map_pkl}: {exc}")
                    map_obj = None

        if map_obj is not None:

            def _extract_lines(obj, depth=0):
                if obj is None or depth > 10:
                    return
                
                # Check if it's a stub with attributes we can iterate
                if hasattr(obj, "__dict__") and not isinstance(obj, (dict, list, tuple)):
                    _extract_lines(obj.__dict__, depth + 1)
                
                if isinstance(obj, dict):
                    # Try to extract x, y coordinates from dict keys
                    if "x" in obj and "y" in obj:
                        try:
                            x, y = float(obj["x"]), float(obj["y"])
                            # Single point - collect for potential line building
                            return [(x, y)]
                        except Exception:
                            pass
                    # Look for coordinate-like keys
                    for key in ["centerline", "boundary", "points", "nodes", "polyline", "coordinates", "line"]:
                        if key in obj:
                            result = _extract_lines(obj[key], depth + 1)
                            if result:
                                return result
                    # Recurse into all values
                    for v in obj.values():
                        _extract_lines(v, depth + 1)
                        
                elif isinstance(obj, (list, tuple)):
                    if not obj:
                        return
                    
                    # Check if it's a list of coordinate pairs/tuples
                    try:
                        if all(hasattr(item, "__len__") and len(item) >= 2 for item in obj[:3] if item is not None):
                            pts = [(float(p[0]), float(p[1])) for p in obj if p is not None and len(p) >= 2]
                            if len(pts) >= 2:
                                pkl_lines.append(pts)
                                return pts
                    except Exception:
                        pass
                    
                    # Check if it's a list of objects with x, y attributes
                    try:
                        pts = []
                        for item in obj:
                            if hasattr(item, "x") and hasattr(item, "y"):
                                pts.append((float(item.x), float(item.y)))
                            elif hasattr(item, "__dict__") and "x" in item.__dict__ and "y" in item.__dict__:
                                pts.append((float(item.__dict__["x"]), float(item.__dict__["y"])))
                            elif isinstance(item, dict) and "x" in item and "y" in item:
                                pts.append((float(item["x"]), float(item["y"])))
                        if len(pts) >= 2:
                            pkl_lines.append(pts)
                            return pts
                    except Exception:
                        pass
                    
                    # Recurse into list items
                    for v in obj:
                        _extract_lines(v, depth + 1)
                        
                else:
                    # Check for numpy arrays
                    try:
                        import numpy as np  # type: ignore
                        if hasattr(obj, "shape") and len(getattr(obj, "shape", [])) == 2:
                            arr = np.asarray(obj)
                            if arr.shape[1] >= 2 and arr.shape[0] >= 2:
                                pts = [(float(p[0]), float(p[1])) for p in arr]
                                pkl_lines.append(pts)
                                return pts
                    except Exception:
                        pass
                    
                    # Check for objects with x, y attributes
                    if hasattr(obj, "x") and hasattr(obj, "y"):
                        try:
                            return [(float(obj.x), float(obj.y))]
                        except Exception:
                            pass

            _extract_lines(map_obj)
            if not pkl_lines:
                warnings.append(f"Map PKL {args.map_pkl} loaded but no polylines recognized; skipping map overlay")

    if args.use_carla_map:
        if carla is None:
            warnings.append("carla module not available; cannot fetch map from CARLA")
        else:
            cache_path = args.carla_cache or os.path.join(base_dir, "carla_map_cache.pkl")
            cache_loaded = False
            tx_offset = 0.0
            ty_offset = 0.0
            if args.carla_map_offset_json and os.path.isfile(args.carla_map_offset_json):
                try:
                    with open(args.carla_map_offset_json, "r", encoding="utf-8") as jf:
                        offset_cfg = json.load(jf) or {}
                        tx_offset = float(offset_cfg.get("tx", 0.0))
                        ty_offset = float(offset_cfg.get("ty", 0.0))
                        warnings.append(f"Applied CARLA map offset tx={tx_offset}, ty={ty_offset} from {args.carla_map_offset_json}")
                except Exception as exc:
                    warnings.append(f"Failed to read carla_map_offset_json {args.carla_map_offset_json}: {exc}")
            if cache_path and os.path.isfile(cache_path):
                try:
                    cached = pickle.load(open(cache_path, "rb"))
                    if isinstance(cached, dict) and "lines" in cached:
                        lines_cached = cached["lines"]
                        # Apply optional flip then offset to cached lines
                        offset_lines = []
                        for line in lines_cached:
                            new_line = []
                            for (x, y) in line:
                                y_flipped = -y if args.carla_map_flip_y else y
                                new_line.append((x + tx_offset, y_flipped + ty_offset))
                            offset_lines.append(new_line)
                        carla_lines.extend(offset_lines)
                        map_bounds = cached.get("bounds")
                        cache_loaded = True
                        warnings.append(f"Using cached CARLA map from {cache_path}")
                except Exception:
                    pass
            if not cache_loaded:
                try:
                    client = carla.Client(args.carla_host, args.carla_port)
                    client.set_timeout(10.0)
                    world = client.get_world()
                    cmap = world.get_map()
                    if args.expected_town and args.expected_town not in (cmap.name or ""):
                        candidates = [m for m in client.get_available_maps() if args.expected_town in m]
                        if not candidates:
                            warnings.append(
                                f"CARLA map '{cmap.name}' does not match expected '{args.expected_town}' and no match found"
                            )
                        else:
                            world = client.load_world(candidates[0])
                            cmap = world.get_map()
                    wps = cmap.generate_waypoints(distance=args.carla_sample)
                    buckets: Dict[Tuple[int, int], List[Any]] = {}
                    for wp in wps:
                        buckets.setdefault((wp.road_id, wp.lane_id), []).append(wp)
                    carla_lines_local: List[List[Tuple[float, float]]] = []
                    bounds = [float("inf"), -float("inf"), float("inf"), -float("inf")]
                    for _, seq in buckets.items():
                        seq.sort(key=lambda w: w.s)
                        pts = []
                        for w in seq:
                            x = float(w.transform.location.x)
                            y = float(w.transform.location.y)
                            if args.carla_map_flip_y:
                                y = -y
                            x += tx_offset
                            y += ty_offset
                            pts.append((x, y))
                            bounds[0] = min(bounds[0], x)
                            bounds[1] = max(bounds[1], x)
                            bounds[2] = min(bounds[2], y)
                            bounds[3] = max(bounds[3], y)
                        if len(pts) >= 2:
                            carla_lines_local.append(pts)
                    map_bounds = None if bounds[0] == float("inf") else tuple(bounds)
                    carla_lines.extend(carla_lines_local)
                    if cache_path:
                        try:
                            pickle.dump({"lines": carla_lines_local, "bounds": map_bounds, "map_name": cmap.name}, open(cache_path, "wb"))
                        except Exception:
                            pass
                except Exception as exc:
                    warnings.append(f"Failed to fetch map from CARLA: {exc}")

    for folder in subfolders:
        for timestep in timesteps:
            yaml_path = os.path.join(folder, f"{timestep}.yaml")
            if not os.path.isfile(yaml_path):
                continue
            data = load_yaml(yaml_path)
            ego_pose = data.get("true_ego_pose") or data.get("lidar_pose")
            if ego_pose and timestep not in ego_by_ts:
                ego_by_ts[timestep] = ego_pose
            vehs = data.get("vehicles", {}) or {}
            for vid_str, payload in vehs.items():
                try:
                    vid = int(vid_str)
                except (TypeError, ValueError):
                    warnings.append(f"Skipping non-integer vehicle id {vid_str} in {yaml_path}")
                    continue

                entry = {
                    "vehicle_id": vid,
                    "obj_type": payload.get("obj_type", ""),
                    "location": payload.get("location"),
                    "extent": payload.get("extent"),
                    "angle": payload.get("angle"),
                    "attribute": payload.get("attribute", ""),
                }

                if vid not in vehicles:
                    vehicles[vid] = {"vehicle_id": vid, "obj_type": entry["obj_type"]}
                # Preserve first obj_type; if a new one conflicts, keep the first and warn.
                if entry["obj_type"] and vehicles[vid].get("obj_type") and entry["obj_type"] != vehicles[vid]["obj_type"]:
                    warnings.append(
                        f"Conflicting obj_type for vehicle {vid} at {yaml_path}: {entry['obj_type']} vs {vehicles[vid]['obj_type']}"
                    )

                merge_vehicle_entry(vehicles[vid], entry, args.tolerance, warnings, source=os.path.basename(folder), timestep=timestep)

    # Compute axes limits (vehicle-centric) and full-map limits (vehicles + maps)
    all_points: List[Tuple[float, float]] = []
    for vdata in vehicles.values():
        for ts in timesteps:
            if ts in vdata and vdata[ts].get("location"):
                loc = vdata[ts]["location"]
                try:
                    all_points.append((float(loc[0]), float(loc[1])))
                except Exception:
                    pass
    for ts, pose in ego_by_ts.items():
        if pose and len(pose) >= 2:
            try:
                all_points.append((float(pose[0]), float(pose[1])))
            except Exception:
                pass
    axes_limits: Tuple[float, float, float, float] = None
    if all_points:
        xs, ys = zip(*all_points)
        pad = 10.0
        axes_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    map_limits: Tuple[float, float, float, float] = axes_limits
    if pkl_lines or carla_lines:
        xs = []
        ys = []
        for line in pkl_lines + carla_lines:
            xs.extend([p[0] for p in line])
            ys.extend([p[1] for p in line])
        if xs and ys:
            pad = 10.0
            map_limits = (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    # Write CSV
    header = ["vehicle_id", "obj_type"] + [f"t_{ts}" for ts in timesteps]
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for vid in sorted(vehicles.keys()):
            row = {"vehicle_id": vid, "obj_type": vehicles[vid].get("obj_type", "")}
            for ts in timesteps:
                if ts in vehicles[vid]:
                    cell = {
                        k: vehicles[vid][ts][k]
                        for k in ["location", "extent", "angle", "attribute", "source"]
                        if k in vehicles[vid][ts]
                    }
                    row[f"t_{ts}"] = json.dumps(cell, separators=(",", ":"))
                else:
                    row[f"t_{ts}"] = ""
            writer.writerow(row)

    print(f"CSV written: {output_csv}")

    generated_frames: List[str] = []

    if args.viz or args.viz_over_time:
        os.makedirs(viz_dir, exist_ok=True)
        palette: Dict[str, str] = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
        for ts in timesteps:
            out_path = os.path.join(viz_dir, f"t_{ts}.png")
            plot_timestep(
                ts,
                vehicles,
                out_path,
                palette,
                color_cycle,
                ego_pose=ego_by_ts.get(ts) if args.plot_ego else None,
                axes_limits=axes_limits,
                pkl_lines=pkl_lines,
                carla_lines=carla_lines,
                map_color_pkl=args.map_color_pkl,
                map_color_carla=args.map_color_carla,
                invert_plot_y=False,
            )
            generated_frames.append(out_path)
        if args.viz:
            print(f"Frames written: {viz_dir}")

    if args.viz_over_time:
        if not args.viz and not generated_frames:
            # Generate frames on the fly if user only requested the GIF
            os.makedirs(viz_dir, exist_ok=True)
            palette: Dict[str, str] = {}
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
            for ts in timesteps:
                out_path = os.path.join(viz_dir, f"t_{ts}.png")
                plot_timestep(
                    ts,
                    vehicles,
                    out_path,
                    palette,
                    color_cycle,
                    ego_pose=ego_by_ts.get(ts) if args.plot_ego else None,
                    axes_limits=axes_limits,
                    pkl_lines=pkl_lines,
                    carla_lines=carla_lines,
                    map_color_pkl=args.map_color_pkl,
                    map_color_carla=args.map_color_carla,
                    invert_plot_y=False,
                )
                generated_frames.append(out_path)

        if imageio is None:
            print("imageio not available; skipping GIF creation. Install imageio to enable this feature.")
        elif not generated_frames:
            print("No frames available to stitch; skipping GIF creation.")
        else:
            frames = [imageio.imread(p) for p in generated_frames]
            imageio.mimsave(viz_video, frames, duration=0.5)
            print(f"GIF written: {viz_video}")

        # Full-map GIF (zoomed out to include maps)
        if imageio is not None and generated_frames:
            tmp_full = []
            for ts in timesteps:
                out_path = os.path.join(viz_dir, f"t_{ts}_full.png")
                plot_timestep(
                    ts,
                    vehicles,
                    out_path,
                    palette,
                    color_cycle,
                    ego_pose=ego_by_ts.get(ts) if args.plot_ego else None,
                    axes_limits=map_limits or axes_limits,
                    pkl_lines=pkl_lines,
                    carla_lines=carla_lines,
                    map_color_pkl=args.map_color_pkl,
                    map_color_carla=args.map_color_carla,
                    invert_plot_y=False,
                )
                tmp_full.append(out_path)
            frames_full = [imageio.imread(p) for p in tmp_full]
            imageio.mimsave(viz_video_full, frames_full, duration=0.5)
            print(f"Full-map GIF written: {viz_video_full}")

    if warnings:
        print("Completed with warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(f"Wrote {output_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
