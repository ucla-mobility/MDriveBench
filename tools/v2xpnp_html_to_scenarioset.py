"""Export a pipeline-runtime HTML dataset into the scenarioset/v2xpnp/<scenario>/
layout (route XMLs + manifest + control config).

Reads the embedded `<script id="dataset" type="application/json">` block from a
pipeline_runtime HTML, applies the PKL→CARLA coordinate transform, and writes:

  <out>/<scenario>/
      ucla_v2_custom_ego_vehicle_<i>.xml      # ego routes (one per ego)
      actors/npc/ucla_v2_custom_Vehicle_<id>_npc.xml
      actors/walker/ucla_v2_custom_Walker_<id>_walker.xml
      actors/static/ucla_v2_custom_Vehicle_<id>_static.xml
      actors_manifest.json
      carla_control_config.json

The format mirrors the reference 21 already-converted v2xpnp scenarios. Z,
pitch, and roll are set to 0.0 — run carla_ground_align.py separately to set
real ground elevation.

Usage:
    python3 -m tools.v2xpnp_html_to_scenarioset \
        --html /tmp/eval_v3/2023-03-17-15-53-02_1_0.html \
        --out scenarioset/v2xpnp \
        --map-offset-json v2xpnp/map/ucla_map_offset_carla.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def parse_html_dataset(html_path: Path) -> Dict[str, Any]:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r'<script[^>]+id="dataset"[^>]*type="application/json"[^>]*>(.*?)</script>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"No embedded dataset found in {html_path}")
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Coordinate transform: PKL → CARLA
# ---------------------------------------------------------------------------
def load_pkl_to_carla_transform(json_path: Optional[Path]) -> Dict[str, float]:
    """The map-offset JSON describes PKL→CARLA. Returns a dict with `tx`, `ty`,
    `flip_y`, and `theta_deg` (currently only theta=0 supported)."""
    out: Dict[str, float] = {"tx": 0.0, "ty": 0.0, "flip_y": False, "theta_deg": 0.0}
    if json_path is None or not json_path.exists():
        return out
    cfg = json.loads(json_path.read_text(encoding="utf-8"))
    out["tx"] = float(cfg.get("tx", 0.0))
    out["ty"] = float(cfg.get("ty", 0.0))
    out["flip_y"] = bool(cfg.get("flip_y", False))
    out["theta_deg"] = float(cfg.get("theta_deg", 0.0))
    return out


def pkl_to_carla(x: float, y: float, yaw_deg: float, t: Dict[str, float]) -> Tuple[float, float, float]:
    """Apply the inverse transform (PKL → CARLA) — note JSON describes
    CARLA→PKL so we negate translations as in `trajectory_ingest`."""
    tx = -t["tx"]
    ty = t["ty"] if t["flip_y"] else -t["ty"]
    if t["flip_y"]:
        x_out = x + tx
        y_out = -y + ty
        yaw_out = -yaw_deg  # mirror y → flip yaw sign
    else:
        x_out = x + tx
        y_out = y + ty
        yaw_out = yaw_deg
    if abs(t["theta_deg"]) > 1e-9:
        # rotate xy + adjust yaw
        c = math.cos(math.radians(-t["theta_deg"]))
        s = math.sin(math.radians(-t["theta_deg"]))
        x_out, y_out = x_out * c - y_out * s, x_out * s + y_out * c
        yaw_out -= t["theta_deg"]
    return x_out, y_out, yaw_out


# ---------------------------------------------------------------------------
# Actor classification
# ---------------------------------------------------------------------------
def classify_actor(track: Dict[str, Any]) -> str:
    """Returns one of: 'ego', 'npc', 'walker', 'static'."""
    role = str(track.get("role", "")).strip().lower()
    if role == "ego":
        return "ego"
    if role == "walker":
        # walkers that don't move much — call them walker_static
        # for now treat all as walker
        return "walker"
    # vehicle role: split into npc / static
    if bool(track.get("low_motion_vehicle", False)):
        return "static"
    # Check raw displacement
    fr = track.get("frames") or []
    if len(fr) < 2:
        return "static"
    x0, y0 = _safe_float(fr[0].get("x")), _safe_float(fr[0].get("y"))
    xN, yN = _safe_float(fr[-1].get("x")), _safe_float(fr[-1].get("y"))
    if math.isfinite(x0) and math.isfinite(y0) and math.isfinite(xN) and math.isfinite(yN):
        if math.hypot(xN - x0, yN - y0) < 2.0:
            return "static"
    return "npc"


# ---------------------------------------------------------------------------
# XML writers
# ---------------------------------------------------------------------------
def _write_xml(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_ego_route(
    path: Path,
    route_id: str,
    town: str,
    waypoints: List[Tuple[float, float, float, float]],  # x, y, yaw_deg, t
) -> None:
    """Ego XML format (matches reference)."""
    root = ET.Element("routes", {"actor_control_mode": "replay", "log_replay_actors": "true"})
    route = ET.SubElement(root, "route", {
        "id": route_id,
        "role": "ego",
        "snap_to_road": "false",
        "town": town,
    })
    for x, y, yaw, _t in waypoints:
        ET.SubElement(route, "waypoint", {
            "pitch": "0.000000",
            "roll": "0.000000",
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "yaw": f"{yaw:.6f}",
            "z": "0.000000",
        })
    _write_xml(root, path)


def write_actor_route(
    path: Path,
    route_id: str,
    town: str,
    role: str,
    model: str,
    waypoints: List[Tuple[float, float, float, float]],  # x, y, yaw_deg, t
    control_mode: str = "replay",
) -> None:
    """NPC / walker / static XML format (matches reference)."""
    root = ET.Element("routes")
    route = ET.SubElement(root, "route", {
        "control_mode": control_mode,
        "id": route_id,
        "model": model,
        "role": role,
        "snap_to_road": "false",
        "town": town,
    })
    for x, y, yaw, t in waypoints:
        ET.SubElement(route, "waypoint", {
            "pitch": "0.000000",
            "roll": "0.000000",
            "time": f"{t:.6f}",
            "x": f"{x:.6f}",
            "y": f"{y:.6f}",
            "yaw": f"{yaw:.6f}",
            "z": "0.000000",
        })
    _write_xml(root, path)


# ---------------------------------------------------------------------------
# Model name guessing — matches the conventions used by trajectory_ingest
# ---------------------------------------------------------------------------
DEFAULT_VEHICLE_MODEL = "vehicle.tesla.model3"
DEFAULT_EGO_MODEL = "vehicle.lincoln.mkz_2020"
DEFAULT_WALKER_MODEL = "walker.pedestrian.0001"


def actor_model(track: Dict[str, Any], kind: str) -> str:
    if kind == "walker":
        return str(track.get("model") or DEFAULT_WALKER_MODEL)
    if kind == "ego":
        return str(track.get("model") or DEFAULT_EGO_MODEL)
    return str(track.get("model") or DEFAULT_VEHICLE_MODEL)


def actor_filename(scenario_name: str, town: str, track_id: str, kind: str) -> str:
    actor_type = "Walker" if kind == "walker" else "Vehicle"
    return f"{town.lower()}_custom_{actor_type}_{track_id}_{kind}.xml"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_dataset(
    dataset: Dict[str, Any],
    scenario_name: str,
    out_root: Path,
    transform: Dict[str, float],
    town: str = "ucla_v2",
) -> Dict[str, Any]:
    """Convert one HTML dataset to a scenarioset XML directory.

    Returns the manifest dict that was written.
    """
    out_dir = out_root / scenario_name
    actors_dir = out_dir / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("npc", "walker", "static"):
        (actors_dir / sub).mkdir(parents=True, exist_ok=True)

    tracks = dataset.get("tracks") or []
    ego_entries: List[Dict[str, Any]] = []
    actors_by_kind: Dict[str, List[Dict[str, Any]]] = {"npc": [], "walker": [], "static": []}

    ego_count = 0
    for track in tracks:
        tid = str(track.get("id", "?"))
        role = str(track.get("role", "")).strip().lower()
        if role not in ("ego", "vehicle", "walker"):
            continue
        kind = classify_actor(track)
        frames = track.get("frames") or []
        if not frames:
            continue
        # Build waypoints in CARLA frame
        waypoints: List[Tuple[float, float, float, float]] = []
        for f in frames:
            cx = _safe_float(f.get("cx"))
            cy = _safe_float(f.get("cy"))
            cyaw = _safe_float(f.get("cyaw"), default=_safe_float(f.get("yaw")))
            t = _safe_float(f.get("t"), 0.0)
            if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cyaw)):
                # Fall back to raw if snap missing
                cx = _safe_float(f.get("x"))
                cy = _safe_float(f.get("y"))
                cyaw = _safe_float(f.get("yaw"))
                if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cyaw)):
                    continue
            xc, yc, yawc = pkl_to_carla(cx, cy, cyaw, transform)
            waypoints.append((xc, yc, yawc, t))
        if not waypoints:
            continue
        # For static actors, freeze pose at first waypoint (avoid replaying jitter)
        if kind == "static" and len(waypoints) > 1:
            anchor = waypoints[0]
            waypoints = [(anchor[0], anchor[1], anchor[2], wp[3]) for wp in waypoints]

        if kind == "ego":
            xml_name = f"{town.lower()}_custom_ego_vehicle_{ego_count}.xml"
            xml_path = out_dir / xml_name
            write_ego_route(
                xml_path,
                route_id=scenario_name,
                town=town,
                waypoints=waypoints,
            )
            ego_entries.append({
                "file": xml_name,
                "route_id": scenario_name,
                "town": town,
                "name": f"ego_{ego_count}",
                "kind": "ego",
                "model": actor_model(track, "ego"),
            })
            ego_count += 1
        else:
            xml_name = actor_filename(scenario_name, town, tid, kind)
            xml_path = actors_dir / kind / xml_name
            write_actor_route(
                xml_path,
                route_id=scenario_name,
                town=town,
                role=kind,
                model=actor_model(track, kind),
                waypoints=waypoints,
                control_mode="replay",
            )
            entry = {
                "file": str(xml_path.relative_to(out_dir)),
                "route_id": scenario_name,
                "town": town,
                "name": tid,
                "kind": kind,
                "model": actor_model(track, kind),
                "control_mode": "replay",
            }
            length = track.get("length")
            width = track.get("width")
            if length is not None:
                entry["length"] = str(length) if isinstance(length, (int, float)) else length
            if width is not None:
                entry["width"] = str(width) if isinstance(width, (int, float)) else width
            actors_by_kind[kind].append(entry)

    manifest: Dict[str, Any] = {}
    if ego_entries:
        manifest["ego"] = ego_entries
    for kind in sorted(actors_by_kind.keys()):
        if actors_by_kind[kind]:
            manifest[kind] = actors_by_kind[kind]
    (out_dir / "actors_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # carla_control_config.json
    config = {
        "ego_path_source": "auto",
        "actor_control_mode": "replay",
        "walker_control_mode": "replay",
        "encode_timing": True,
        "snap_to_road": False,
        "static_spawn_only": False,
        "town": town,
        "route_id": scenario_name,
        "manifest_path": "actors_manifest.json",
        "ego_count": len(ego_entries),
        "npc_count": len(actors_by_kind.get("npc", [])),
        "walker_count": len(actors_by_kind.get("walker", [])),
        "static_count": len(actors_by_kind.get("static", [])),
    }
    (out_dir / "carla_control_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--html", nargs="+", required=True, help="HTML files produced by pipeline_runtime")
    p.add_argument("--out", required=True, help="Output root (e.g., scenarioset/v2xpnp_new)")
    p.add_argument("--map-offset-json", required=True, help="ucla_map_offset_carla.json")
    p.add_argument("--town", default="ucla_v2")
    p.add_argument("--route-id-from", choices=["scenario_name", "html_stem"], default="scenario_name",
                   help="How to derive the route_id (default: dataset.scenario_name)")
    args = p.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    transform = load_pkl_to_carla_transform(Path(args.map_offset_json))
    print(f"Transform: tx={transform['tx']} ty={transform['ty']} flip_y={transform['flip_y']}")

    for hp in args.html:
        hp_path = Path(hp).resolve()
        ds = parse_html_dataset(hp_path)
        if args.route_id_from == "html_stem":
            scenario_name = hp_path.stem
        else:
            scenario_name = ds.get("scenario_name") or hp_path.stem
        print(f"\n=== {scenario_name} ===")
        m = convert_dataset(ds, scenario_name, out_root, transform, town=args.town)
        n_ego = len(m.get("ego", []))
        n_npc = len(m.get("npc", []))
        n_walker = len(m.get("walker", []))
        n_static = len(m.get("static", []))
        print(f"  ego={n_ego} npc={n_npc} walker={n_walker} static={n_static}")
        print(f"  out: {out_root / scenario_name}")


if __name__ == "__main__":
    main()
