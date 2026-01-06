#!/usr/bin/env python3
"""Dump available CARLA blueprints (vehicles, props, walkers, etc.) to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("carla_assets.json"),
        help="Output JSON path (default: carla_assets.json).",
    )
    parser.add_argument(
        "--pretty",
        default=True,
        action="store_true",
        help="Indent the JSON output for readability.",
    )
    parser.add_argument(
        "--no-attributes",
        action="store_true",
        help="Only list blueprint IDs (skip per-attribute metadata).",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Skip spawning actors to measure bounding boxes (enabled by default).",
    )
    return parser.parse_args()


def categorize_blueprint(bp_id: str) -> str:
    """Group blueprints into broad categories for readability."""
    if bp_id.startswith("vehicle."):
        return "vehicle"
    if bp_id.startswith("walker."):
        return "walker"
    if bp_id.startswith("traffic."):
        return "traffic"
    if bp_id.startswith("sensor."):
        return "sensor"
    if bp_id.startswith("static."):
        return "static"
    return "other"


def serialize_attribute(attr: carla.ActorAttribute) -> Dict[str, Any]:
    """Convert a CARLA attribute to a JSON-friendly dict."""
    attr_type = getattr(attr.type, "name", str(attr.type))
    return {
        "id": attr.id,
        "type": attr_type,
        "is_modifiable": bool(attr.is_modifiable),
        "recommended_values": list(attr.recommended_values),
        "default": getattr(attr, "default", None),
    }


def serialize_blueprint(bp: carla.ActorBlueprint, include_attributes: bool) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "id": bp.id,
        "tags": list(bp.tags),
    }
    if include_attributes:
        attrs: List[Dict[str, Any]] = []
        for attr in bp:
            attrs.append(serialize_attribute(attr))
        data["attributes"] = attrs
    return data


def _actor_locations(world: carla.World) -> List[carla.Location]:
    locs: List[carla.Location] = []
    try:
        for actor in world.get_actors():
            if "sensor." in actor.type_id:
                continue
            locs.append(actor.get_transform().location)
    except Exception:
        return []
    return locs


def _spawn_transforms(
    world: carla.World,
    z_offsets: List[float],
    max_points: int = 30,
    min_dist_m: float = 6.0,
    avoid_locs: Optional[List[carla.Location]] = None,
) -> List[carla.Transform]:
    spawn_points = []
    if world.get_map():
        spawn_points = world.get_map().get_spawn_points() or []
    candidates: List[carla.Transform] = []
    if spawn_points:
        for sp in spawn_points:
            if avoid_locs:
                too_close = False
                for loc in avoid_locs:
                    if sp.location.distance(loc) < min_dist_m:
                        too_close = True
                        break
                if too_close:
                    continue
            candidates.append(sp)
            if len(candidates) >= max_points:
                break
        if not candidates:
            candidates = spawn_points[:max_points]
    transforms: List[carla.Transform] = []
    for sp in candidates:
        loc = sp.location
        for z_off in z_offsets:
            transforms.append(carla.Transform(
                carla.Location(float(loc.x), float(loc.y), float(loc.z) + float(z_off)),
                sp.rotation,
            ))
    if not transforms:
        transforms = [carla.Transform(carla.Location(0.0, 0.0, float(z_offsets[0] if z_offsets else 0.0)), carla.Rotation())]
    return transforms


def _walker_transforms(world: carla.World, fallback: List[carla.Transform], max_points: int = 5) -> List[carla.Transform]:
    locs = []
    for _ in range(max_points):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            locs.append(loc)
    if not locs:
        locs = [t.location for t in fallback]
    return [carla.Transform(carla.Location(float(l.x), float(l.y), float(l.z) + 1.0), carla.Rotation()) for l in locs]


def _get_actor_bbox(actor: carla.Actor) -> Optional[carla.BoundingBox]:
    bbox = getattr(actor, "bounding_box", None)
    if bbox is not None:
        return bbox
    getter = getattr(actor, "get_bounding_box", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            return None
    return None


def measure_bbox(
    bp: carla.ActorBlueprint,
    world: carla.World,
    transforms: List[carla.Transform],
) -> Optional[Dict[str, float]]:
    for tf in transforms:
        actor = None
        try:
            actor = world.try_spawn_actor(bp, tf)
            if actor is None:
                continue
            bbox = _get_actor_bbox(actor)
            if bbox is None:
                return None
            ex = float(bbox.extent.x)
            ey = float(bbox.extent.y)
            ez = float(bbox.extent.z)
            return {
                "extent_x": ex,
                "extent_y": ey,
                "extent_z": ez,
                "length": 2.0 * ex,
                "width": 2.0 * ey,
                "height": 2.0 * ez,
            }
        finally:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
    return None


def main() -> None:
    args = parse_args()
    include_attributes = not args.no_attributes
    include_bbox = not args.no_bbox

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world = client.get_world()

    bp_library = world.get_blueprint_library()
    avoid_locs = _actor_locations(world) if include_bbox else []
    spawn_transforms = _spawn_transforms(world, z_offsets=[0.0, 20.0], avoid_locs=avoid_locs) if include_bbox else []
    walker_transforms = _walker_transforms(world, spawn_transforms) if include_bbox else []
    assets: Dict[str, List[Dict[str, Any]]] = {}
    bbox_success = 0
    for bp in bp_library:
        category = categorize_blueprint(bp.id)
        data = serialize_blueprint(bp, include_attributes)
        if include_bbox:
            tf_list = walker_transforms if category == "walker" else spawn_transforms
            data["bbox"] = measure_bbox(bp, world, tf_list)
            if data["bbox"] is not None:
                bbox_success += 1
        assets.setdefault(category, []).append(data)

    # Sort for deterministic output.
    for items in assets.values():
        items.sort(key=lambda x: x["id"])

    map_name = world.get_map().name if world.get_map() else None
    output_data = {
        "meta": {
            "host": args.host,
            "port": args.port,
            "map": map_name.split("/")[-1] if map_name else None,
            "attribute_details": include_attributes,
            "bbox_attempted": include_bbox,
        },
        "summary": {category: len(items) for category, items in assets.items()},
        "assets": assets,
    }
    if include_bbox:
        output_data["meta"]["bbox_units"] = "m"
        output_data["meta"]["bbox_source"] = "actor.bounding_box.extent"
        output_data["meta"]["bbox_success"] = bbox_success
        output_data["meta"]["bbox_total"] = len(bp_library)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(output_data, indent=2 if args.pretty else None)
    args.output.write_text(json_text, encoding="utf-8")
    print(f"Wrote asset catalog with {len(bp_library)} blueprints to {args.output}")


if __name__ == "__main__":
    main()
