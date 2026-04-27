#!/usr/bin/env python3
"""Browser-based scenario validation and editing tool for final dataset curation."""

from __future__ import annotations

import argparse
import atexit
import base64
import copy
import csv
import hashlib
import importlib
import io
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEADERBOARD_ROOT = REPO_ROOT / "simulation" / "leaderboard"

from scenario_generator.carla_validation import (  # noqa: E402
    _load_world_for_town,
    validate_xml_manifest_contract,
)
from scenario_generator.pipeline.step_07_route_alignment import main as align_mod  # noqa: E402


REVIEW_STATE_FILENAME = ".scenario_review_state.json"
REVIEW_STATUS_CSV_FILENAME = "scenario_review_status.csv"
ROLE_ORDER = {"ego": 0, "npc": 1, "pedestrian": 2, "walker": 2, "bicycle": 3, "cyclist": 3, "static": 4}
ROLE_ALIASES = {"walker": "pedestrian", "cyclist": "bicycle", "static_prop": "static"}
ROUTE_ATTR_ORDER = [
    "id",
    "town",
    "role",
    "snap_to_road",
    "snap_spawn_to_road",
    "control_mode",
    "model",
    "target_speed",
    "speed",
]
WEATHER_ATTR_ORDER = [
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "wetness",
    "fog_distance",
    "fog_density",
]
WAYPOINT_ATTR_ORDER = ["x", "y", "z", "yaw", "pitch", "roll", "time", "speed"]
DEFAULT_REVIEW = {"status": "pending", "note": "", "updated_at": None, "last_saved_at": None}
GRP_POSTPROCESS_MODES = ("none", "seam", "kink", "legacy")
DEFAULT_MODEL_BY_ROLE = {
    "ego": "vehicle.lincoln.mkz2017",
    "npc": "vehicle.audi.a2",
    "pedestrian": "walker.pedestrian.0001",
    "walker": "walker.pedestrian.0001",
    "bicycle": "vehicle.bh.crossbike",
    "cyclist": "vehicle.bh.crossbike",
    "static": "static.prop.trafficcone",
}


@dataclass(frozen=True)
class GRPPreviewConfig:
    sampling_resolution: float = 2.0
    postprocess_mode: str | None = None
    postprocess_ignore_endpoints: bool = True

    def env_overrides(self) -> dict[str, str]:
        overrides = {
            "CARLA_GRP_PP_IGNORE_ENDPOINTS": "1" if self.postprocess_ignore_endpoints else "0",
        }
        if self.postprocess_mode is not None:
            mode = str(self.postprocess_mode).strip().lower()
            if mode == "none":
                overrides["CARLA_GRP_PP_ENABLE"] = "0"
            else:
                overrides["CARLA_GRP_PP_ENABLE"] = "1"
                overrides["CARLA_GRP_PP_MODE"] = mode
        return overrides

    def as_payload(self) -> dict[str, Any]:
        return {
            "sampling_resolution": round(float(self.sampling_resolution), 4),
            "postprocess_mode": self.postprocess_mode,
            "postprocess_ignore_endpoints": bool(self.postprocess_ignore_endpoints),
        }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _safe_float(value: Any, *, allow_none: bool = False, default: float | None = None) -> float | None:
    if value is None:
        return None if allow_none else default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None if allow_none else default
    if not math.isfinite(result):
        return None if allow_none else default
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_asset_id(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _clean_bbox_payload(payload: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    bbox: dict[str, float] = {}
    for key in ("extent_x", "extent_y", "extent_z", "length", "width", "height"):
        value = _safe_float(payload.get(key), allow_none=True)
        if value is not None:
            bbox[key] = round(float(value), 6)
    return bbox or None


def _load_asset_bbox_lookup(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    assets_root = payload.get("assets")
    if not isinstance(assets_root, dict):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for entries in assets_root.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            asset_id = str(entry.get("id") or "").strip()
            if not asset_id:
                continue
            bbox = _clean_bbox_payload(entry.get("bbox"))
            if bbox is None:
                continue
            lookup[_canonical_asset_id(asset_id)] = {
                "id": asset_id,
                "bbox": bbox,
            }
    return lookup


ASSET_BBOX_LOOKUP = _load_asset_bbox_lookup(REPO_ROOT / "scenario_generator" / "carla_assets.json")


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


def _et_to_pretty_xml(root: ET.Element) -> str:
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    text = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + text + "\n"


def _route_sort_key(route: dict[str, Any]) -> tuple[int, str, str]:
    role = str(route.get("kind") or route.get("role") or route.get("route_attrs", {}).get("role", "")).lower()
    order = ROLE_ORDER.get(role, 99)
    name = str(route.get("name") or "")
    rel_path = str(route.get("file") or "")
    return (order, name, rel_path)


def _role_group(role: str) -> str:
    role_norm = ROLE_ALIASES.get(str(role).strip().lower(), str(role).strip().lower())
    return role_norm or "npc"


def _bool_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return "true"
    if text in {"0", "false", "no", "off"}:
        return "false"
    return text


def _ordered_keys(payload: dict[str, Any], preferred: list[str]) -> list[str]:
    remaining = [key for key in payload.keys() if key not in preferred]
    return [key for key in preferred if key in payload] + sorted(remaining)


def _coerce_route_attrs(payload: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, raw in payload.items():
        if raw is None:
            continue
        if key in {"snap_to_road", "snap_spawn_to_road"}:
            value = _bool_text(raw)
            if value is not None:
                out[key] = value
            continue
        if key in {"target_speed", "speed"}:
            value = _safe_float(raw, allow_none=True)
            if value is not None:
                out[key] = f"{value:.2f}"
            continue
        text = str(raw).strip()
        if text:
            out[key] = text
    return out


def _coerce_weather_attrs(payload: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(payload, dict):
        return out
    for key in WEATHER_ATTR_ORDER:
        raw = payload.get(key)
        value = _safe_float(raw, allow_none=True)
        if value is None:
            continue
        out[key] = f"{float(value):.2f}"
    for key, raw in payload.items():
        if key in out or key not in WEATHER_ATTR_ORDER:
            continue
        value = _safe_float(raw, allow_none=True)
        if value is not None:
            out[key] = f"{float(value):.2f}"
    return out


def _coerce_waypoint(payload: dict[str, Any]) -> dict[str, Any]:
    extras = payload.get("extras", {})
    if extras is None or not isinstance(extras, dict):
        extras = {}
    result = {
        "x": float(_safe_float(payload.get("x"), default=0.0) or 0.0),
        "y": float(_safe_float(payload.get("y"), default=0.0) or 0.0),
        "z": float(_safe_float(payload.get("z"), default=0.0) or 0.0),
        "yaw": float(_safe_float(payload.get("yaw"), default=0.0) or 0.0),
        "pitch": _safe_float(payload.get("pitch"), allow_none=True),
        "roll": _safe_float(payload.get("roll"), allow_none=True),
        "time": _safe_float(payload.get("time"), allow_none=True),
        "speed": _safe_float(payload.get("speed"), allow_none=True),
        "extras": {str(k): str(v) for k, v in extras.items() if v not in (None, "")},
    }
    return result


def _serialise_waypoint_for_client(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": round(float(payload["x"]), 6),
        "y": round(float(payload["y"]), 6),
        "z": round(float(payload["z"]), 6),
        "yaw": round(float(payload["yaw"]), 6),
        "pitch": None if payload.get("pitch") is None else round(float(payload["pitch"]), 6),
        "roll": None if payload.get("roll") is None else round(float(payload["roll"]), 6),
        "time": None if payload.get("time") is None else round(float(payload["time"]), 6),
        "speed": None if payload.get("speed") is None else round(float(payload["speed"]), 4),
        "extras": dict(payload.get("extras") or {}),
    }


def _parse_route_xml(xml_path: Path) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    route = root.find("route")
    if route is None:
        raise ValueError(f"{xml_path} does not contain a <route> node")
    route_attrs = dict(route.attrib)
    weather_attrs = _coerce_weather_attrs(dict(route.find("weather").attrib) if route.find("weather") is not None else {})
    waypoints: list[dict[str, Any]] = []
    for waypoint in route.findall("waypoint"):
        attrs = dict(waypoint.attrib)
        waypoints.append(
            {
                "x": float(_safe_float(attrs.pop("x"), default=0.0) or 0.0),
                "y": float(_safe_float(attrs.pop("y"), default=0.0) or 0.0),
                "z": float(_safe_float(attrs.pop("z"), default=0.0) or 0.0),
                "yaw": float(_safe_float(attrs.pop("yaw", 0.0), default=0.0) or 0.0),
                "pitch": _safe_float(attrs.pop("pitch", None), allow_none=True),
                "roll": _safe_float(attrs.pop("roll", None), allow_none=True),
                "time": _safe_float(attrs.pop("time", attrs.pop("t", None)), allow_none=True),
                "speed": _safe_float(attrs.pop("speed", None), allow_none=True),
                "extras": attrs,
            }
        )
    return route_attrs, weather_attrs, waypoints


def _route_xml_text(route_attrs: dict[str, str], weather_attrs: dict[str, str], waypoints: list[dict[str, Any]]) -> str:
    root = ET.Element("routes")
    route_el = ET.SubElement(root, "route")
    for key in _ordered_keys(route_attrs, ROUTE_ATTR_ORDER):
        value = route_attrs.get(key)
        if value is not None and str(value).strip():
            route_el.set(key, str(value))
    if weather_attrs:
        weather_el = ET.SubElement(route_el, "weather")
        for key in _ordered_keys(weather_attrs, WEATHER_ATTR_ORDER):
            value = weather_attrs.get(key)
            if value is not None and str(value).strip():
                weather_el.set(key, str(value))
    for waypoint in waypoints:
        attrs: dict[str, str] = {
            "x": f"{float(waypoint['x']):.6f}",
            "y": f"{float(waypoint['y']):.6f}",
            "z": f"{float(waypoint['z']):.6f}",
            "yaw": f"{float(waypoint['yaw']):.6f}",
        }
        if waypoint.get("pitch") is not None:
            attrs["pitch"] = f"{float(waypoint['pitch']):.6f}"
        if waypoint.get("roll") is not None:
            attrs["roll"] = f"{float(waypoint['roll']):.6f}"
        if waypoint.get("time") is not None:
            attrs["time"] = f"{float(waypoint['time']):.6f}"
        if waypoint.get("speed") is not None:
            attrs["speed"] = f"{float(waypoint['speed']):.4f}"
        extras = waypoint.get("extras") or {}
        for key in sorted(extras.keys()):
            if key not in attrs and extras[key] is not None and str(extras[key]).strip():
                attrs[key] = str(extras[key])
        ET.SubElement(route_el, "waypoint", attrs)
    return _et_to_pretty_xml(root)


def _validate_route_payload(route: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]], list[str]]:
    route_attrs = _coerce_route_attrs(dict(route.get("route_attrs") or {}))
    if not route_attrs.get("id"):
        raise ValueError(f"{route.get('file') or route.get('actor_id')}: route id is required")
    if not route_attrs.get("town"):
        raise ValueError(f"{route.get('file') or route.get('actor_id')}: route town is required")
    role = _role_group(route_attrs.get("role", route.get("kind", "npc")))
    route_attrs["role"] = role

    waypoints_raw = route.get("waypoints")
    if not isinstance(waypoints_raw, list) or not waypoints_raw:
        raise ValueError(f"{route.get('file') or route.get('actor_id')}: at least one waypoint is required")
    waypoints = [_coerce_waypoint(item if isinstance(item, dict) else {}) for item in waypoints_raw]

    warnings: list[str] = []
    times = [waypoint.get("time") for waypoint in waypoints]
    present_times = [value for value in times if value is not None]
    if present_times and len(present_times) != len(times):
        raise ValueError(f"{route.get('file') or route.get('actor_id')}: waypoint timing must be set for every waypoint or none")
    if present_times:
        for idx in range(1, len(present_times)):
            if present_times[idx] < present_times[idx - 1] - 1e-6:
                raise ValueError(f"{route.get('file') or route.get('actor_id')}: waypoint times must be non-decreasing")

    if role in {"ego", "npc", "pedestrian", "walker", "bicycle", "cyclist"} and len(waypoints) < 2:
        warnings.append("single_waypoint_dynamic_route")
    return route_attrs, waypoints, warnings


def _recompute_headings(waypoints: list[dict[str, Any]]) -> None:
    if len(waypoints) < 2:
        return
    for idx in range(len(waypoints)):
        if idx < len(waypoints) - 1:
            dx = float(waypoints[idx + 1]["x"]) - float(waypoints[idx]["x"])
            dy = float(waypoints[idx + 1]["y"]) - float(waypoints[idx]["y"])
        else:
            dx = float(waypoints[idx]["x"]) - float(waypoints[idx - 1]["x"])
            dy = float(waypoints[idx]["y"]) - float(waypoints[idx - 1]["y"])
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            waypoints[idx]["yaw"] = math.degrees(math.atan2(dy, dx))


def _angle_delta_deg(a: float, b: float) -> float:
    diff = (float(a) - float(b) + 180.0) % 360.0 - 180.0
    return abs(diff)


def _heading_from_locations(points: list[Any], min_dist: float = 2.0) -> float | None:
    if len(points) < 2:
        return None
    p0 = points[0]
    for idx in range(1, len(points)):
        dx = float(points[idx].x) - float(p0.x)
        dy = float(points[idx].y) - float(p0.y)
        if (dx * dx + dy * dy) >= (float(min_dist) * float(min_dist)):
            return math.degrees(math.atan2(dy, dx))
    return None


def _copy_location(carla_module: Any, location: Any) -> Any:
    return carla_module.Location(
        x=float(location.x),
        y=float(location.y),
        z=float(location.z),
    )


def _load_leaderboard_route_manipulation() -> Any:
    # Reuse the exact runtime route interpolation path used by the evaluator.
    align_mod._ensure_carla()
    if str(LEADERBOARD_ROOT) not in sys.path:
        sys.path.insert(0, str(LEADERBOARD_ROOT))
    return importlib.import_module("leaderboard.utils.route_manipulation")


def _align_preview_start_waypoint(
    *,
    carla_module: Any,
    carla_map: Any,
    trajectory: list[Any],
    waypoint_payloads: list[dict[str, Any]],
) -> bool:
    if not trajectory:
        return False

    xml_yaw = None
    if waypoint_payloads:
        xml_yaw = _safe_float(waypoint_payloads[0].get("yaw"), allow_none=True)

    xml_heading = _heading_from_locations(trajectory)
    desired_yaw = None
    if xml_heading is not None:
        if xml_yaw is None or _angle_delta_deg(xml_heading, float(xml_yaw)) > 45.0:
            desired_yaw = float(xml_heading)
        else:
            desired_yaw = float(xml_yaw)
    elif xml_yaw is not None:
        desired_yaw = float(xml_yaw)

    if desired_yaw is None:
        return False

    try:
        waypoint = carla_map.get_waypoint(
            trajectory[0],
            project_to_road=True,
            lane_type=carla_module.LaneType.Driving,
        )
    except Exception:
        waypoint = None
    if waypoint is None:
        return False

    candidates = [waypoint]
    for neighbor_fn in ("get_left_lane", "get_right_lane"):
        neighbor = None
        try:
            neighbor = getattr(waypoint, neighbor_fn)()
        except Exception:
            neighbor = None
        if neighbor is not None and getattr(neighbor, "lane_type", None) == carla_module.LaneType.Driving:
            candidates.append(neighbor)

    best = min(
        candidates,
        key=lambda candidate: _angle_delta_deg(
            desired_yaw,
            float(candidate.transform.rotation.yaw),
        ),
    )
    try:
        if best.transform.location.distance(trajectory[0]) > 5.0:
            return False
    except Exception:
        return False

    trajectory[0] = _copy_location(carla_module, best.transform.location)
    return True


def _route_trace_to_dense_points(route: list[tuple[Any, Any]]) -> tuple[list[dict[str, Any]], float]:
    dense_points: list[dict[str, Any]] = []
    trace_length_m = 0.0
    prev_loc = None
    for waypoint, road_option in route:
        transform = getattr(waypoint, "transform", None)
        if transform is None:
            continue
        location = transform.location
        if prev_loc is not None:
            trace_length_m += float(location.distance(prev_loc))
        prev_loc = location
        dense_points.append(
            {
                "x": round(float(location.x), 3),
                "y": round(float(location.y), 3),
                "z": round(float(location.z), 3),
                "yaw": round(float(transform.rotation.yaw), 6),
                "road_option": getattr(road_option, "name", str(road_option)),
            }
        )
    return dense_points, trace_length_m


def _build_followed_dense_route(
    *,
    carla_module: Any,
    world: Any,
    carla_map: Any,
    waypoints_payload: list[dict[str, Any]],
    sampling_resolution: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route_manipulation = _load_leaderboard_route_manipulation()
    trajectory = [
        carla_module.Location(
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item["z"]),
        )
        for item in waypoints_payload
    ]
    start_snapped = _align_preview_start_waypoint(
        carla_module=carla_module,
        carla_map=carla_map,
        trajectory=trajectory,
        waypoint_payloads=waypoints_payload,
    )
    _gps_route, route = route_manipulation.interpolate_trajectory(
        world,
        trajectory,
        hop_resolution=float(sampling_resolution),
    )
    dense_points, trace_length_m = _route_trace_to_dense_points(route)
    debug = dict(getattr(route_manipulation.interpolate_trajectory, "last_debug", {}) or {})
    debug["start_snapped"] = bool(start_snapped)
    debug["trace_length_m"] = round(float(trace_length_m), 3)
    return dense_points, debug


# Version counter — bump when GRP algorithm changes to invalidate stale caches.
_GRP_ALGORITHM_VERSION = 6


def _make_grp_like_run_custom_eval(carla_map: Any, sampling_resolution: float) -> Any:
    _, GlobalRoutePlanner, GlobalRoutePlannerDAO = align_mod._ensure_carla()
    try:
        grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    except TypeError:
        grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
    if hasattr(grp, "setup"):
        grp.setup()
    return grp


@contextmanager
def _temporary_env(overrides: dict[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def classify_lane_direction(wp: Any, carla_map: Any) -> str:
    lane_type = wp.lane_type
    driving_flag = getattr(type(lane_type), "Driving", None) or getattr(lane_type, "Driving", None)
    if driving_flag is not None and hasattr(lane_type, "__and__") and not lane_type & driving_flag:
        return "neutral"

    reference_wp = None
    try:
        reference_wp = carla_map.get_waypoint_xodr(wp.road_id, 0, wp.s)
    except RuntimeError:
        reference_wp = None

    if reference_wp is not None:
        yaw = math.radians(wp.transform.rotation.yaw)
        ref_yaw = math.radians(reference_wp.transform.rotation.yaw)
        dot = math.cos(yaw - ref_yaw)
        if dot > 1e-3:
            return "forward"
        if dot < -1e-3:
            return "opposing"
    return "forward" if getattr(wp, "lane_id", 0) < 0 else "opposing"


def _scenario_dir_candidates(root: Path) -> list[Path]:
    if _looks_like_scenario_dir(root):
        return [root]
    results: list[Path] = []
    for current_root, dirnames, _filenames in os.walk(root, topdown=True):
        current_path = Path(current_root)
        dirnames[:] = sorted(
            [
                dirname
                for dirname in dirnames
                if not dirname.startswith(".")
            ]
        )
        if current_path == root:
            continue
        if _looks_like_scenario_dir(current_path):
            results.append(current_path)
            dirnames[:] = []
            continue
    return results


def _looks_like_scenario_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "actors_manifest.json").exists():
        return True
    if list(path.glob("*.xml")):
        return True
    actors_dir = path / "actors"
    return actors_dir.exists() and any(actors_dir.rglob("*.xml"))


@dataclass
class ManifestEntryRef:
    group: str
    entry: dict[str, Any]


class ScenarioStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()
        self._scenario_paths: dict[str, Path] = {}
        self._review_path = (root if root.is_dir() else root.parent) / REVIEW_STATE_FILENAME
        self._review_csv_path = (root if root.is_dir() else root.parent) / REVIEW_STATUS_CSV_FILENAME
        self._review_state = _load_json(self._review_path)
        if "scenarios" not in self._review_state or not isinstance(self._review_state["scenarios"], dict):
            self._review_state = {"scenarios": {}}
        self._summaries: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self._scenario_paths.clear()
            self._summaries.clear()
            for scenario_dir in _scenario_dir_candidates(self.root):
                scenario_id = str(scenario_dir.relative_to(self.root)) if scenario_dir != self.root else scenario_dir.name
                self._scenario_paths[scenario_id] = scenario_dir
                self._summaries.append(self._build_summary(scenario_id, scenario_dir))
            self._summaries.sort(key=lambda item: item["id"].lower())
            self._sync_review_tracking_locked(persist=True)

    def _build_summary(self, scenario_id: str, scenario_dir: Path) -> dict[str, Any]:
        manifest = self._load_manifest(scenario_dir)
        xml_paths = self._collect_active_xmls(scenario_dir, manifest)
        counts: dict[str, int] = {}
        town = None
        warnings: list[str] = []
        for xml_path in xml_paths:
            try:
                route_attrs, _, _ = _parse_route_xml(xml_path)
            except Exception as exc:
                warnings.append(f"{xml_path.name}: {exc}")
                continue
            role = _role_group(route_attrs.get("role", "npc"))
            counts[role] = counts.get(role, 0) + 1
            if town is None:
                town = route_attrs.get("town")
        review = self.get_review_state(scenario_id)
        return {
            "id": scenario_id,
            "name": scenario_dir.name,
            "relative_path": scenario_id,
            "parent_path": "" if "/" not in scenario_id else scenario_id.rsplit("/", 1)[0],
            "path": str(scenario_dir),
            "town": town,
            "route_count": sum(counts.values()),
            "ego_route_count": counts.get("ego", 0),
            "actor_counts": counts,
            "status": review["status"],
            "note": review.get("note", ""),
            "warnings": warnings,
        }

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._summaries)

    def get_scenario_path(self, scenario_id: str) -> Path:
        with self._lock:
            try:
                return self._scenario_paths[scenario_id]
            except KeyError as exc:
                raise FileNotFoundError(f"Unknown scenario id: {scenario_id}") from exc

    def get_review_state(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            payload = self._review_state["scenarios"].get(scenario_id)
            if not isinstance(payload, dict):
                payload = {}
            review = dict(DEFAULT_REVIEW)
            review.update(payload)
            return review

    def _sync_review_tracking_locked(self, *, persist: bool) -> None:
        scenarios = self._review_state.get("scenarios")
        if not isinstance(scenarios, dict):
            scenarios = {}
        next_state: dict[str, dict[str, Any]] = {}
        for summary in self._summaries:
            scenario_id = str(summary["id"])
            payload = scenarios.get(scenario_id)
            if not isinstance(payload, dict):
                payload = {}
            review = dict(DEFAULT_REVIEW)
            review.update(payload)
            next_state[scenario_id] = review
            summary["status"] = review["status"]
            summary["note"] = review.get("note", "")
        self._review_state = {
            "root": str(self.root),
            "updated_at": _now_iso(),
            "scenarios": next_state,
        }
        if persist:
            self._persist_review_tracking_locked()

    def _persist_review_tracking_locked(self) -> None:
        self._review_state["updated_at"] = _now_iso()
        _atomic_write_json(self._review_path, self._review_state)
        fieldnames = [
            "scenario_id",
            "scenario_name",
            "status",
            "note",
            "town",
            "route_count",
            "updated_at",
            "last_saved_at",
            "path",
        ]
        rows: list[dict[str, Any]] = []
        for summary in self._summaries:
            review = self._review_state["scenarios"].get(str(summary["id"]), {})
            rows.append(
                {
                    "scenario_id": summary["id"],
                    "scenario_name": summary["name"],
                    "status": review.get("status", "pending"),
                    "note": review.get("note", ""),
                    "town": summary.get("town", ""),
                    "route_count": summary.get("route_count", 0),
                    "updated_at": review.get("updated_at", ""),
                    "last_saved_at": review.get("last_saved_at", ""),
                    "path": summary.get("path", ""),
                }
            )
        _atomic_write_csv(self._review_csv_path, rows, fieldnames)

    def update_review_state(self, scenario_id: str, *, status: str | None = None, note: str | None = None, mark_saved: bool = False) -> dict[str, Any]:
        with self._lock:
            self.get_scenario_path(scenario_id)
            review = self.get_review_state(scenario_id)
            if status is not None:
                review["status"] = str(status)
            if note is not None:
                review["note"] = str(note)
            review["updated_at"] = _now_iso()
            if mark_saved:
                review["last_saved_at"] = review["updated_at"]
                if review["status"] == "pending":
                    review["status"] = "edited"
            self._review_state["scenarios"][scenario_id] = review
            for summary in self._summaries:
                if summary["id"] == scenario_id:
                    summary["status"] = review["status"]
                    summary["note"] = review.get("note", "")
            self._persist_review_tracking_locked()
            return copy.deepcopy(review)

    def _load_manifest(self, scenario_dir: Path) -> dict[str, Any] | None:
        manifest_path = scenario_dir / "actors_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _manifest_lookup(self, manifest: dict[str, Any] | None) -> dict[str, ManifestEntryRef]:
        lookup: dict[str, ManifestEntryRef] = {}
        if not manifest:
            return lookup
        if isinstance(manifest.get("actors"), list):
            for entry in manifest["actors"]:
                if isinstance(entry, dict) and entry.get("file"):
                    rel_path = str(entry["file"]).replace("\\", "/")
                    group = _role_group(entry.get("kind", entry.get("role", "npc")))
                    lookup[rel_path] = ManifestEntryRef(group=group, entry=entry)
            return lookup
        for group, entries in manifest.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict) and entry.get("file"):
                    rel_path = str(entry["file"]).replace("\\", "/")
                    lookup[rel_path] = ManifestEntryRef(group=str(group), entry=entry)
        return lookup

    def _collect_active_xmls(self, scenario_dir: Path, manifest: dict[str, Any] | None) -> list[Path]:
        scenario_dir = scenario_dir.resolve()
        manifest_lookup = self._manifest_lookup(manifest)
        candidates: set[Path] = set()
        for rel_path in manifest_lookup.keys():
            xml_path = (scenario_dir / rel_path).resolve()
            if xml_path.exists() and xml_path.is_file():
                candidates.add(xml_path)
        for xml_path in scenario_dir.glob("*.xml"):
            if "_REPLAY" in xml_path.stem and xml_path.with_name(xml_path.stem.replace("_REPLAY", "") + xml_path.suffix).exists():
                continue
            candidates.add(xml_path.resolve())
        actors_dir = scenario_dir / "actors"
        if actors_dir.exists():
            for xml_path in actors_dir.rglob("*.xml"):
                if "_REPLAY" in xml_path.stem and xml_path.with_name(xml_path.stem.replace("_REPLAY", "") + xml_path.suffix).exists():
                    continue
                candidates.add(xml_path.resolve())
        active = [path for path in candidates if path.exists() and path.suffix == ".xml"]
        active.sort(key=lambda path: path.relative_to(scenario_dir).as_posix())
        return active

    def load_scenario(self, scenario_id: str) -> dict[str, Any]:
        scenario_dir = self.get_scenario_path(scenario_id).resolve()
        manifest = self._load_manifest(scenario_dir)
        manifest_lookup = self._manifest_lookup(manifest)
        warnings: list[str] = []
        routes: list[dict[str, Any]] = []
        town = None
        scenario_weather: dict[str, str] = {}
        weather_source_file: str | None = None

        for xml_path in self._collect_active_xmls(scenario_dir, manifest):
            rel_path = xml_path.relative_to(scenario_dir).as_posix()
            try:
                route_attrs, weather_attrs, waypoints = _parse_route_xml(xml_path)
            except Exception as exc:
                warnings.append(f"{rel_path}: {exc}")
                continue
            manifest_ref = manifest_lookup.get(rel_path)
            role = _role_group(route_attrs.get("role", manifest_ref.group if manifest_ref else "npc"))
            route_attrs["role"] = role
            town = town or route_attrs.get("town")
            if weather_attrs:
                if not scenario_weather:
                    scenario_weather = dict(weather_attrs)
                    weather_source_file = rel_path
                elif scenario_weather != weather_attrs:
                    warnings.append(
                        f"{rel_path}: weather differs from {weather_source_file}; editor will use scenario-level weather from the first route"
                    )
            name = None
            if manifest_ref is not None:
                name = manifest_ref.entry.get("name")
            if not name:
                name = xml_path.stem
            model, bbox = self._resolve_route_model_and_bbox(route_attrs, manifest_ref.entry if manifest_ref else None, role)
            route_payload = {
                "actor_id": rel_path,
                "file": rel_path,
                "kind": role,
                "name": str(name),
                "route_attrs": dict(route_attrs),
                "resolved_model": model,
                "bbox": bbox,
                "waypoints": [_serialise_waypoint_for_client(waypoint) for waypoint in waypoints],
                "original_waypoints": [_serialise_waypoint_for_client(waypoint) for waypoint in waypoints],
                "manifest_entry": copy.deepcopy(manifest_ref.entry) if manifest_ref else None,
                "supports_grp": role in {"ego", "npc"},
            }
            routes.append(route_payload)

        routes.sort(key=_route_sort_key)
        review = self.get_review_state(scenario_id)
        actor_counts = self._count_roles(routes)
        return {
            "id": scenario_id,
            "name": scenario_dir.name,
            "relative_path": scenario_id,
            "parent_path": "" if "/" not in scenario_id else scenario_id.rsplit("/", 1)[0],
            "path": str(scenario_dir),
            "town": town,
            "weather": dict(scenario_weather),
            "routes": routes,
            "warnings": warnings,
            "review": review,
            "manifest_present": manifest is not None,
            "actor_counts": actor_counts,
            "ego_route_count": actor_counts.get("ego", 0),
        }

    def _resolve_route_model_and_bbox(
        self,
        route_attrs: dict[str, Any],
        manifest_entry: dict[str, Any] | None,
        role: str,
    ) -> tuple[str | None, dict[str, float] | None]:
        candidates = [
            route_attrs.get("model"),
            None if not isinstance(manifest_entry, dict) else manifest_entry.get("model"),
            DEFAULT_MODEL_BY_ROLE.get(_role_group(role)),
        ]
        for candidate in candidates:
            model = str(candidate or "").strip()
            if not model:
                continue
            asset = ASSET_BBOX_LOOKUP.get(_canonical_asset_id(model))
            if asset is not None:
                return asset.get("id"), copy.deepcopy(asset.get("bbox"))
            return model, None
        return None, None

    def rename_scenario(self, scenario_id: str, new_name: str) -> None:
        with self._lock:
            scenario_dir = self.get_scenario_path(scenario_id)
            new_name = str(new_name or "").strip()
            if not new_name:
                raise ValueError("New name is required")
            if "/" in new_name or "\\" in new_name or new_name.startswith("."):
                raise ValueError(f"Invalid scenario name: {new_name!r}")
            parent = scenario_dir.parent
            new_dir = parent / new_name
            if new_dir.exists() and new_dir.resolve() != scenario_dir.resolve():
                raise ValueError(f"A scenario named {new_name!r} already exists")
            os.rename(scenario_dir, new_dir)
            # Compute new id and migrate review state
            new_id = str(new_dir.relative_to(self.root)) if new_dir.resolve() != self.root.resolve() else new_name
            old_review = self._review_state["scenarios"].pop(scenario_id, {})
            if old_review:
                self._review_state["scenarios"][new_id] = old_review
        self.reload()

    def _count_roles(self, routes: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for route in routes:
            role = _role_group(route.get("kind", "npc"))
            counts[role] = counts.get(role, 0) + 1
        return counts

    def save_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario_id = str(payload.get("id") or "").strip()
        if not scenario_id:
            raise ValueError("Scenario id is required")
        scenario_dir = self.get_scenario_path(scenario_id).resolve()
        manifest = self._load_manifest(scenario_dir)
        existing_xml_paths = self._collect_active_xmls(scenario_dir, manifest)
        routes = payload.get("routes")
        if not isinstance(routes, list) or not routes:
            raise ValueError("Scenario payload must include routes")

        route_texts: dict[Path, str] = {}
        normalised_routes: list[dict[str, Any]] = []
        warnings: list[str] = []
        weather_attrs = _coerce_weather_attrs(payload.get("weather"))
        for route in routes:
            if not isinstance(route, dict):
                raise ValueError("Route payload must be an object")
            rel_path = str(route.get("file") or route.get("actor_id") or "").replace("\\", "/").strip()
            if not rel_path:
                raise ValueError("Route payload missing file path")
            route_attrs, waypoints, route_warnings = _validate_route_payload(route)
            target_path = (scenario_dir / rel_path).resolve()
            if scenario_dir.resolve() not in target_path.parents and target_path != scenario_dir.resolve():
                raise ValueError(f"Refusing to write outside scenario directory: {rel_path}")
            route_texts[target_path] = _route_xml_text(route_attrs, weather_attrs, waypoints)
            normalised_routes.append(
                {
                    "file": rel_path,
                    "name": str(route.get("name") or Path(rel_path).stem),
                    "kind": _role_group(route_attrs.get("role", route.get("kind", "npc"))),
                    "route_attrs": route_attrs,
                    "waypoints": waypoints,
                }
            )
            warnings.extend([f"{rel_path}: {item}" for item in route_warnings])

        manifest_text = None
        manifest_path = scenario_dir / "actors_manifest.json"
        if manifest is not None:
            manifest_text = json.dumps(self._sync_manifest(copy.deepcopy(manifest), normalised_routes), indent=2, ensure_ascii=True) + "\n"
            try:
                json.loads(manifest_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Updated manifest is invalid JSON: {exc}") from exc

        target_xml_paths = set(route_texts.keys())
        stale_xml_paths: list[Path] = []
        for existing_path in existing_xml_paths:
            resolved = existing_path.resolve()
            if resolved in target_xml_paths:
                continue
            if scenario_dir not in resolved.parents and resolved != scenario_dir:
                continue
            try:
                _parse_route_xml(resolved)
            except Exception:
                continue
            stale_xml_paths.append(resolved)

        for xml_text in route_texts.values():
            ET.fromstring(xml_text)

        for path, xml_text in route_texts.items():
            _atomic_write_text(path, xml_text)
        if manifest_text is not None:
            _atomic_write_text(manifest_path, manifest_text)
        for stale_path in stale_xml_paths:
            try:
                stale_path.unlink()
            except OSError as exc:
                warnings.append(f"{stale_path.relative_to(scenario_dir).as_posix()}: failed to delete stale route XML ({exc})")

        if manifest_text is not None:
            validation = validate_xml_manifest_contract(scenario_dir)
            if not validation.get("ok", False):
                warnings.extend(validation.get("errors", []))
            warnings.extend(validation.get("warnings", []))

        review = self.update_review_state(scenario_id, note=payload.get("review", {}).get("note"), mark_saved=True)
        fresh = self.load_scenario(scenario_id)
        fresh["review"] = review
        if warnings:
            fresh["warnings"] = fresh.get("warnings", []) + warnings
        self.reload()
        return fresh

    def _sync_manifest(self, manifest: dict[str, Any], routes: list[dict[str, Any]]) -> dict[str, Any]:
        active_paths = {str(route["file"]).replace("\\", "/") for route in routes}
        flat = isinstance(manifest.get("actors"), list)
        if flat:
            entries = manifest.get("actors")
            if isinstance(entries, list):
                filtered_entries = []
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("file"):
                        rel_path = str(entry["file"]).replace("\\", "/")
                        if rel_path not in active_paths:
                            continue
                    filtered_entries.append(entry)
                manifest["actors"] = filtered_entries
        else:
            for group, entries in list(manifest.items()):
                if not isinstance(entries, list):
                    continue
                filtered_entries = []
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("file"):
                        rel_path = str(entry["file"]).replace("\\", "/")
                        if rel_path not in active_paths:
                            continue
                    filtered_entries.append(entry)
                manifest[group] = filtered_entries

        lookup = self._manifest_lookup(manifest)
        for route in routes:
            rel_path = str(route["file"]).replace("\\", "/")
            route_attrs = route["route_attrs"]
            role = _role_group(route_attrs.get("role", route["kind"]))
            manifest_group = role if not flat else "actors"

            if rel_path in lookup:
                entry = lookup[rel_path].entry
                if not flat and lookup[rel_path].group != manifest_group:
                    old_group = manifest.setdefault(lookup[rel_path].group, [])
                    if isinstance(old_group, list):
                        old_group[:] = [item for item in old_group if item is not entry]
                    entry = {}
                    manifest.setdefault(manifest_group, []).append(entry)
            else:
                entry = {}
                if flat:
                    manifest.setdefault("actors", []).append(entry)
                else:
                    manifest.setdefault(manifest_group, []).append(entry)
                lookup[rel_path] = ManifestEntryRef(group=manifest_group, entry=entry)

            entry["file"] = rel_path
            entry["kind"] = role
            entry["name"] = route.get("name") or Path(rel_path).stem
            entry["route_id"] = route_attrs.get("id", "")
            entry["town"] = route_attrs.get("town", "")

            if route_attrs.get("model"):
                entry["model"] = route_attrs["model"]
            elif "model" in entry:
                entry.pop("model")

            if route_attrs.get("control_mode"):
                entry["control_mode"] = route_attrs["control_mode"]
            elif "control_mode" in entry:
                entry.pop("control_mode")

            target_speed = route_attrs.get("target_speed") or route_attrs.get("speed")
            if target_speed:
                speed_value = _safe_float(target_speed, allow_none=True)
                if speed_value is not None:
                    entry["speed"] = round(float(speed_value), 2)
                    entry["target_speed"] = round(float(speed_value), 2)
            else:
                entry.pop("speed", None)
                entry.pop("target_speed", None)
        return manifest


GRP_DISK_CACHE_FILENAME = ".grp_cache.json"


def _numpy_rgba_to_png(rgba: Any) -> bytes:
    """Encode a numpy (H, W, 4) uint8 RGBA array as a valid PNG using only stdlib (zlib/struct)."""
    h, w = rgba.shape[:2]
    # PNG file signature
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return length + tag + data + crc

    # IHDR: width, height, bit_depth=8, colortype=6 (RGBA), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)

    # Raw image data: prepend filter byte 0 (None) to each row
    rows = []
    for y in range(h):
        rows.append(b"\x00" + rgba[y].tobytes())
    raw = b"".join(rows)
    compressed = zlib.compress(raw, level=6)
    idat = chunk(b"IDAT", compressed)

    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


class CarlaSessionManager:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        auto_launch: bool,
        carla_root: Path | None,
        extra_args: list[str],
        startup_timeout_s: float,
        post_start_buffer_s: float,
        sampling_distance: float,
        grp_config: GRPPreviewConfig,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.auto_launch = bool(auto_launch)
        self.carla_root = carla_root
        self.extra_args = list(extra_args)
        self.startup_timeout_s = max(5.0, float(startup_timeout_s))
        self.post_start_buffer_s = max(0.0, float(post_start_buffer_s))
        self.sampling_distance = max(0.5, float(sampling_distance))
        self.grp_config = grp_config
        self._managed_process: subprocess.Popen[Any] | None = None
        self._log_handle: Any | None = None
        self._lock = threading.RLock()
        self._last_error: str | None = None
        self._map_cache: dict[tuple[str, float], dict[str, Any]] = {}
        self._grp_cache: dict[str, dict[str, Any]] = {}
        self._grp_disk_cache_path: Path | None = None

    def set_disk_cache_root(self, root: Path) -> None:
        """Point the GRP disk cache at <root>/.grp_cache.json and load any existing entries."""
        self._grp_disk_cache_path = root / GRP_DISK_CACHE_FILENAME
        self._load_disk_cache()

    def _load_disk_cache(self) -> None:
        if self._grp_disk_cache_path is None or not self._grp_disk_cache_path.exists():
            return
        try:
            with self._grp_disk_cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            entries = data.get("entries") if isinstance(data, dict) else None
            if isinstance(entries, dict):
                with self._lock:
                    loaded = {k: v for k, v in entries.items() if isinstance(v, dict)}
                    self._grp_cache.update(loaded)
                print(f"[GRP disk cache] Loaded {len(loaded)} cached entries from {self._grp_disk_cache_path}", flush=True)
        except Exception as exc:
            print(f"[GRP disk cache] Failed to load: {exc}", flush=True)

    def _save_disk_cache_locked(self) -> None:
        if self._grp_disk_cache_path is None:
            return
        try:
            data = {"version": "1", "updated_at": _now_iso(), "entries": dict(self._grp_cache)}
            _atomic_write_text(self._grp_disk_cache_path, json.dumps(data, ensure_ascii=True, separators=(",", ":")))
        except Exception as exc:
            print(f"[GRP disk cache] Failed to save: {exc}", flush=True)

    @staticmethod
    def discover_carla_root(explicit_root: Path | None) -> Path | None:
        candidates = []
        if explicit_root is not None:
            candidates.append(explicit_root)
        candidates.extend(
            [
                REPO_ROOT / "carla912",
                REPO_ROOT / "carla",
                REPO_ROOT / "external_paths" / "carla_root",
            ]
        )
        for candidate in candidates:
            if candidate is None:
                continue
            if (candidate / "CarlaUE4.sh").exists():
                return candidate.resolve()
        return None

    def _ensure_carla_import(self) -> tuple[Any, Any, Any]:
        return align_mod._ensure_carla()

    def _rpc_probe(self, timeout_s: float = 3.0) -> tuple[bool, str | None, str | None]:
        try:
            carla, _, _ = self._ensure_carla_import()
        except Exception as exc:
            return False, None, f"CARLA import failed: {exc}"
        client = None
        try:
            client = carla.Client(self.host, self.port)
            client.set_timeout(float(timeout_s))
            world = client.get_world()
            map_name = world.get_map().name
            map_base = map_name.split("/")[-1] if "/" in map_name else map_name
            return True, map_base, None
        except Exception as exc:
            return False, None, str(exc)
        finally:
            client = None

    def status(self) -> dict[str, Any]:
        with self._lock:
            connected, town, error = self._rpc_probe(timeout_s=1.0)
            if error:
                self._last_error = error
            return {
                "connected": connected,
                "managed": self._managed_process is not None and self._managed_process.poll() is None,
                "host": self.host,
                "port": self.port,
                "town": town,
                "pid": None if self._managed_process is None else self._managed_process.pid,
                "last_error": None if connected else (error or self._last_error),
                "carla_root": None if self.carla_root is None else str(self.carla_root),
                "auto_launch": self.auto_launch,
                "grp_config": self.grp_config.as_payload(),
                "grp_cache_size": len(self._grp_cache),
            }

    def ensure_ready(self, *, desired_town: str | None = None, restart_if_managed: bool = True) -> None:
        with self._lock:
            connected, _, error = self._rpc_probe(timeout_s=2.0)
            if connected:
                return
            if self._managed_process is not None and self._managed_process.poll() is not None:
                self._managed_process = None
            if self._managed_process is not None and restart_if_managed:
                self._stop_locked()
            if not self.auto_launch:
                raise RuntimeError(error or "CARLA is not reachable")
            if self.carla_root is None:
                raise RuntimeError("CARLA is not reachable and no CARLA root was found for auto-launch")
            self._start_locked()
            if desired_town and self.post_start_buffer_s > 0:
                time.sleep(self.post_start_buffer_s)

    @contextmanager
    def world_context(self, town: str | None = None) -> Iterator[tuple[Any, Any, Any]]:
        with self._lock:
            self.ensure_ready(desired_town=town)
            carla, _, _ = self._ensure_carla_import()
            try:
                client, world = _load_world_for_town(
                    carla,
                    self.host,
                    self.port,
                    town,
                    timeout_s=max(10.0, self.startup_timeout_s),
                )
            except Exception:
                if self._managed_process is not None:
                    self._stop_locked()
                    self._start_locked()
                    client, world = _load_world_for_town(
                        carla,
                        self.host,
                        self.port,
                        town,
                        timeout_s=max(10.0, self.startup_timeout_s),
                    )
                else:
                    raise
            yield carla, client, world

    def reconnect(self, *, restart_managed: bool = False) -> dict[str, Any]:
        with self._lock:
            if restart_managed and self._managed_process is not None:
                self._stop_locked()
            self.ensure_ready(restart_if_managed=restart_managed)
            return self.status()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        proc = self._managed_process
        if proc is None:
            if self._log_handle is not None:
                try:
                    self._log_handle.close()
                except OSError:
                    pass
                self._log_handle = None
            return
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and proc.poll() is None:
                time.sleep(0.25)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
        if self._log_handle is not None:
            try:
                self._log_handle.write(f"=== SCENARIO BUILDER CARLA STOP {_now_iso()} ===\n")
                self._log_handle.flush()
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        self._managed_process = None

    def _start_locked(self) -> None:
        if self.carla_root is None:
            raise RuntimeError("CARLA root is not configured")
        if self._managed_process is not None and self._managed_process.poll() is None:
            return
        carla_script = self.carla_root / "CarlaUE4.sh"
        if not carla_script.exists():
            raise FileNotFoundError(carla_script)

        cmd = [str(carla_script), f"--world-port={self.port}"]
        extra_args = list(self.extra_args)
        lowered = {arg.lower() for arg in extra_args}
        if not os.environ.get("DISPLAY") and not any("renderoffscreen" in arg for arg in lowered):
            extra_args.append("-RenderOffScreen")
        cmd.extend(extra_args)
        print(
            f"Launching managed CARLA from {self.carla_root} on {self.host}:{self.port}...",
            flush=True,
        )

        log_path = REPO_ROOT / "logs" / "scenario_builder_carla_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n=== SCENARIO BUILDER CARLA START {_now_iso()} host={self.host} port={self.port} ===\n"
        )
        self._log_handle.write("COMMAND: " + " ".join(cmd) + "\n")
        self._log_handle.flush()
        self._managed_process = subprocess.Popen(
            cmd,
            cwd=str(self.carla_root),
            env=dict(os.environ),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + self.startup_timeout_s
        last_error = None
        while time.monotonic() < deadline:
            if self._managed_process.poll() is not None:
                raise RuntimeError(f"Managed CARLA exited with code {self._managed_process.returncode}")
            connected, _, error = self._rpc_probe(timeout_s=2.0)
            if connected:
                self._last_error = None
                print(f"Managed CARLA is reachable on {self.host}:{self.port}.", flush=True)
                return
            last_error = error
            time.sleep(1.0)
        self._last_error = last_error
        self._stop_locked()
        raise RuntimeError(f"Timed out waiting for CARLA RPC readiness on {self.host}:{self.port}: {last_error}")

    def get_map_payload(self, town: str | None) -> dict[str, Any] | None:
        if not town:
            return None
        cache_key = (str(town).lower(), self.sampling_distance)
        with self._lock:
            if cache_key in self._map_cache:
                return copy.deepcopy(self._map_cache[cache_key])
        with self.world_context(town) as (_, _, world):
            payload = self._sample_town(world)
        with self._lock:
            self._map_cache[cache_key] = payload
        return copy.deepcopy(payload)

    def _grp_cache_key(self, town: str, role: str, waypoints_payload: list[dict[str, Any]]) -> str:
        normalized = [_serialise_waypoint_for_client(_coerce_waypoint(item)) for item in waypoints_payload]
        payload = {
            "town": str(town).strip(),
            "role": _role_group(role),
            "grp_ver": _GRP_ALGORITHM_VERSION,
            "config": self.grp_config.as_payload(),
            "waypoints": normalized,
        }
        raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _compute_grp_preview_locked(self, town: str, role: str, route_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
        role_norm = _role_group(role)
        with _temporary_env(self.grp_config.env_overrides()):
            with self.world_context(town) as (carla, _, world):
                carla_map = world.get_map()
                grp = _make_grp_like_run_custom_eval(carla_map, float(self.grp_config.sampling_resolution))
                metrics = {
                    "segment_count": 0,
                    "trace_points": 0,
                    "trace_length_m": 0.0,
                    "current_length_m": 0.0,
                    "trace_source": "runtime_interpolated_current_waypoints",
                }

                aligned_waypoints = route_waypoints
                plan: list[Any] = []
                try:
                    # Keep builder alignment exactly on the same path used by run_custom_eval:
                    # align_mod.refine_waypoints_dp with the same GRP object/init logic.
                    aligned_waypoints, plan = align_mod.refine_waypoints_dp(
                        carla_map,
                        copy.deepcopy(route_waypoints),
                        grp,
                    )
                except Exception as exc:
                    metrics["alignment_error"] = str(exc)

                for idx in range(len(route_waypoints) - 1):
                    p0 = route_waypoints[idx]
                    p1 = route_waypoints[idx + 1]
                    metrics["current_length_m"] += math.hypot(
                        float(p1["x"]) - float(p0["x"]),
                        float(p1["y"]) - float(p0["y"]),
                    )
                    metrics["segment_count"] += 1

                dense_points: list[dict[str, Any]] = []
                display_debug: dict[str, Any] = {}
                try:
                    dense_points, display_debug = _build_followed_dense_route(
                        carla_module=carla,
                        world=world,
                        carla_map=carla_map,
                        waypoints_payload=copy.deepcopy(route_waypoints),
                        sampling_resolution=float(self.grp_config.sampling_resolution),
                    )
                    metrics["trace_length_m"] = float(display_debug.get("trace_length_m") or 0.0)
                except Exception as exc:
                    metrics["followed_route_error"] = str(exc)
                    metrics["trace_source"] = "aligned_plan_fallback"
                    dense_points, trace_length = _route_trace_to_dense_points(plan)
                    metrics["trace_length_m"] = float(trace_length)
                metrics["trace_points"] = len(dense_points)
                postprocess_meta = display_debug.get("postprocess_meta")
                if isinstance(postprocess_meta, dict):
                    metrics["postprocess_meta"] = copy.deepcopy(postprocess_meta)
                if "start_snapped" in display_debug:
                    metrics["start_snapped"] = bool(display_debug["start_snapped"])

                return {
                    "supported": True,
                    "cached": False,
                    "config": self.grp_config.as_payload(),
                    "dense_points": dense_points,
                    "aligned_waypoints": [_serialise_waypoint_for_client(item) for item in aligned_waypoints],
                    "metrics": {
                        "segment_count": metrics["segment_count"],
                        "trace_points": metrics["trace_points"],
                        "trace_length_m": round(float(metrics["trace_length_m"]), 3),
                        "current_length_m": round(float(metrics["current_length_m"]), 3),
                        "detour_ratio": round(
                            float(metrics["trace_length_m"]) / max(1e-6, float(metrics["current_length_m"])),
                            3,
                        ),
                        "trace_source": metrics.get("trace_source"),
                        "start_snapped": metrics.get("start_snapped"),
                        "postprocess_meta": metrics.get("postprocess_meta"),
                        "alignment_error": metrics.get("alignment_error"),
                        "followed_route_error": metrics.get("followed_route_error"),
                    },
                }

    def warm_grp_previews(self, town: str | None, routes: Iterable[dict[str, Any]]) -> dict[str, Any]:
        if not town:
            return {"by_actor_id": {}, "warmed_count": 0}
        previews: dict[str, Any] = {}
        warmed_count = 0
        for route in routes:
            if not isinstance(route, dict):
                continue
            actor_id = str(route.get("actor_id") or route.get("file") or "").strip()
            role = str(route.get("route_attrs", {}).get("role") or route.get("kind") or "").strip()
            if _role_group(role) != "ego":
                continue
            waypoints = route.get("waypoints")
            if not actor_id or not isinstance(waypoints, list) or len(waypoints) < 2:
                continue
            try:
                previews[actor_id] = self.grp_preview(str(town), role, waypoints)
                warmed_count += 1
            except Exception:
                continue
        return {"by_actor_id": previews, "warmed_count": warmed_count}

    def _sample_town(self, world: Any) -> dict[str, Any]:
        carla_map = world.get_map()
        waypoints = carla_map.generate_waypoints(distance=self.sampling_distance)
        xs: list[float] = []
        ys: list[float] = []
        colors: list[str] = []
        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")
        for waypoint in waypoints:
            loc = waypoint.transform.location
            x = float(loc.x)
            y = float(loc.y)
            xs.append(round(x, 3))
            ys.append(round(y, 3))
            direction = classify_lane_direction(waypoint, carla_map)
            if direction == "forward":
                colors.append("#1b8d66")
            elif direction == "opposing":
                colors.append("#9a4b2f")
            else:
                colors.append("#44505e")
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        if not xs:
            xmin = ymin = -10.0
            xmax = ymax = 10.0
        pad_x = max((xmax - xmin) * 0.05, 10.0)
        pad_y = max((ymax - ymin) * 0.05, 10.0)
        map_name = carla_map.name
        map_base = map_name.split("/")[-1] if "/" in map_name else map_name
        return {
            "town": map_base,
            "x": xs,
            "y": ys,
            "colors": colors,
            "sampling_distance": self.sampling_distance,
            "xmin": xmin - pad_x,
            "xmax": xmax + pad_x,
            "ymin": ymin - pad_y,
            "ymax": ymax + pad_y,
            # Raw extents (before padding) used for BEV image alignment.
            # The BEV cache is generated with margin=300, so:
            #   bev_origin = (xmin_raw - 300, ymin_raw - 300)
            "xmin_raw": xmin,
            "ymin_raw": ymin,
        }

    def grp_preview(self, town: str, role: str, waypoints_payload: list[dict[str, Any]]) -> dict[str, Any]:
        role_norm = _role_group(role)
        if role_norm not in {"ego", "npc"}:
            return {"supported": False, "reason": f"GRP preview is only supported for driving routes, not '{role_norm}'"}
        if len(waypoints_payload) < 2:
            return {"supported": False, "reason": "At least two waypoints are required"}

        route_waypoints = [_coerce_waypoint(item) for item in waypoints_payload]
        cache_key = self._grp_cache_key(town, role_norm, route_waypoints)
        with self._lock:
            cached = self._grp_cache.get(cache_key)
            if cached is not None:
                cached_payload = copy.deepcopy(cached)
                cached_payload["cached"] = True
                return cached_payload
            preview = self._compute_grp_preview_locked(str(town), role_norm, route_waypoints)
            self._grp_cache[cache_key] = copy.deepcopy(preview)
            self._save_disk_cache_locked()
            return preview


class ScenarioBuilderApp:
    def __init__(self, args: argparse.Namespace) -> None:
        root = Path(args.scenarios_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        grp_config = GRPPreviewConfig(
            sampling_resolution=max(0.1, float(args.align_ego_sampling_resolution)),
            postprocess_mode=args.grp_postprocess_mode,
            postprocess_ignore_endpoints=bool(args.grp_postprocess_ignore_endpoints),
        )
        carla_root = CarlaSessionManager.discover_carla_root(
            None if args.carla_root is None else Path(args.carla_root).expanduser().resolve()
        )
        self.store = ScenarioStore(root)
        self.carla = CarlaSessionManager(
            host=args.carla_host,
            port=args.carla_port,
            auto_launch=bool(args.auto_launch_carla),
            carla_root=carla_root,
            extra_args=list(args.carla_arg),
            startup_timeout_s=args.carla_startup_timeout,
            post_start_buffer_s=args.carla_post_start_buffer,
            sampling_distance=args.sampling_distance,
            grp_config=grp_config,
        )
        self.carla.set_disk_cache_root(root)
        atexit.register(self.carla.stop)
        self._prewarm_lock = threading.Lock()
        self._prewarm_status: dict[str, Any] = {"running": False, "done": 0, "total": 0, "errors": 0, "current": ""}
        # BEV (birdview) cache: scan for .npy files at startup
        bev_dir_arg = getattr(args, "bev_cache", None)
        bev_dir = Path(bev_dir_arg).expanduser().resolve() if bev_dir_arg else self._default_bev_dir()
        self._bev_paths: dict[str, Path] = self._scan_bev_cache(bev_dir)
        self._bev_png_cache: dict[str, bytes] = {}
        if self._bev_paths:
            print(f"[BEV] Found top-down images for: {', '.join(sorted(self._bev_paths))}", flush=True)
        if getattr(args, "prewarm_grp", False):
            self.start_prewarm_all_grp()

    @staticmethod
    def _default_bev_dir() -> Path:
        # Look for birdview_v2_cache relative to the script / repo root
        candidates = [
            REPO_ROOT / "birdview_v2_cache" / "Carla" / "Maps",
            REPO_ROOT / "birdview_v2_cache",
            Path("/data2/marco/CoLMDriver/birdview_v2_cache/Carla/Maps"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return REPO_ROOT / "birdview_v2_cache" / "Carla" / "Maps"

    @staticmethod
    def _scan_bev_cache(bev_dir: Path) -> dict[str, Path]:
        result: dict[str, Path] = {}
        if not bev_dir.exists():
            return result
        for npy in bev_dir.glob("*.npy"):
            # Filename format: Town05__px_per_meter=5__opendrive_hash=....__margin=300.npy
            town = npy.name.split("__")[0]
            if town and town not in result:
                result[town] = npy
        return result

    def get_town_image_png(self, town: str) -> bytes | None:
        """Return a PNG (bytes) for the given town's BEV image, cached in memory."""
        if town in self._bev_png_cache:
            return self._bev_png_cache[town]
        npy_path = self._bev_paths.get(town)
        if npy_path is None:
            return None
        try:
            import numpy as np  # type: ignore
            arr = np.load(str(npy_path))  # (3, H, W), values 0/1
            _, h, w = arr.shape
            # Build RGBA image: channel 0=road, 1=sidewalk, 2=other, rest=background
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # background – matches canvas background #0d141b
            rgba[:, :, 0] = 13
            rgba[:, :, 1] = 20
            rgba[:, :, 2] = 27
            rgba[:, :, 3] = 255
            # drivable road (channel 0) – slate blue-grey, clearly visible
            road = arr[0] > 0
            rgba[road, 0] = 48
            rgba[road, 1] = 68
            rgba[road, 2] = 88
            # sidewalk (channel 1)
            side = arr[1] > 0
            rgba[side, 0] = 30
            rgba[side, 1] = 44
            rgba[side, 2] = 58
            # other marked surface (channel 2)
            other = arr[2] > 0
            rgba[other, 0] = 36
            rgba[other, 1] = 52
            rgba[other, 2] = 68
            # Flip vertically: BEV rows go south→north (y_pixel = (world_y - origin_y)*ppx),
            # but canvas draws y downward. Flipping makes the image align without JS transforms.
            rgba = np.flipud(rgba)
            png_bytes = _numpy_rgba_to_png(rgba)
            self._bev_png_cache[town] = png_bytes
            print(f"[BEV] Encoded {town} image: {len(png_bytes)//1024} KB", flush=True)
            return png_bytes
        except Exception as exc:  # noqa: BLE001
            print(f"[BEV] Failed to encode {town}: {exc}", flush=True)
            return None

    def prewarm_status(self) -> dict[str, Any]:
        with self._prewarm_lock:
            return dict(self._prewarm_status)

    def start_prewarm_all_grp(self) -> dict[str, Any]:
        with self._prewarm_lock:
            if self._prewarm_status.get("running"):
                return dict(self._prewarm_status)
            self._prewarm_status = {"running": True, "done": 0, "total": 0, "errors": 0, "current": "starting…"}
        thread = threading.Thread(target=self._prewarm_worker, daemon=True, name="grp-prewarm")
        thread.start()
        return dict(self._prewarm_status)

    def _prewarm_worker(self) -> None:
        summaries = self.store.summaries()
        # Only warm scenarios that have ego routes (GRP only runs on ego/npc)
        candidates = [s for s in summaries if s.get("ego_route_count", 0) > 0]
        total = len(candidates)
        with self._prewarm_lock:
            self._prewarm_status["total"] = total
        print(f"[GRP prewarm] Starting: {total} scenarios with ego routes", flush=True)
        done = 0
        errors = 0
        for summary in candidates:
            scenario_id = summary["id"]
            with self._prewarm_lock:
                self._prewarm_status["current"] = scenario_id
            try:
                scenario = self.store.load_scenario(scenario_id)
                town = scenario.get("town")
                if not town:
                    continue
                for route in scenario.get("routes", []):
                    role = str(route.get("route_attrs", {}).get("role") or route.get("kind") or "").strip()
                    if _role_group(role) != "ego":
                        continue
                    waypoints = route.get("waypoints")
                    if not isinstance(waypoints, list) or len(waypoints) < 2:
                        continue
                    try:
                        self.carla.grp_preview(town, role, waypoints)
                    except Exception as exc:
                        print(f"[GRP prewarm] {scenario_id} route {route.get('file','?')}: {exc}", flush=True)
                        errors += 1
            except Exception as exc:
                print(f"[GRP prewarm] {scenario_id}: {exc}", flush=True)
                errors += 1
            done += 1
            with self._prewarm_lock:
                self._prewarm_status["done"] = done
                self._prewarm_status["errors"] = errors
            print(f"[GRP prewarm] {done}/{total} done ({errors} errors)", flush=True)
        with self._prewarm_lock:
            self._prewarm_status["running"] = False
            self._prewarm_status["current"] = f"done – {done}/{total} scenarios, {errors} errors"
        print(f"[GRP prewarm] Complete: {done}/{total} scenarios, {errors} errors", flush=True)

    def scenario_index(self) -> list[dict[str, Any]]:
        return self.store.summaries()

    def scenario_payload(self, scenario_id: str) -> dict[str, Any]:
        """Return scenario data immediately without blocking on CARLA."""
        payload = self.store.load_scenario(scenario_id)
        payload["map_payload"] = None
        payload["grp_previews"] = {}
        payload["grp_warmed_count"] = 0
        payload["carla_status"] = self.carla.status()
        return payload

    def scenario_bg_payload(self, scenario_id: str) -> dict[str, Any]:
        """Return CARLA-dependent data (map, GRP) for a scenario – may be slow."""
        payload = self.store.load_scenario(scenario_id)
        result: dict[str, Any] = {"id": scenario_id}
        try:
            result["map_payload"] = self.carla.get_map_payload(payload.get("town"))
        except Exception as exc:
            result["map_payload"] = None
            result["map_error"] = str(exc)
        try:
            grp_payload = self.carla.warm_grp_previews(payload.get("town"), payload.get("routes", []))
            result["grp_previews"] = grp_payload.get("by_actor_id", {})
            result["grp_warmed_count"] = int(grp_payload.get("warmed_count", 0))
        except Exception as exc:
            result["grp_previews"] = {}
            result["grp_error"] = str(exc)
        result["carla_status"] = self.carla.status()
        return result

    def save_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        saved = self.store.save_scenario(payload)
        saved["map_payload"] = None
        saved["grp_previews"] = {}
        saved["grp_warmed_count"] = 0
        saved["carla_status"] = self.carla.status()
        return saved

    def rename_scenario(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario_id = str(payload.get("id") or "").strip()
        new_name = str(payload.get("name") or "").strip()
        if not scenario_id:
            raise ValueError("Scenario id is required")
        if not new_name:
            raise ValueError("New name is required")
        self.store.rename_scenario(scenario_id, new_name)
        return {"ok": True, "scenarios": self.store.summaries()}

    def update_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        scenario_id = str(payload.get("id") or "").strip()
        if not scenario_id:
            raise ValueError("Scenario id is required")
        review = self.store.update_review_state(
            scenario_id,
            status=payload.get("status"),
            note=payload.get("note"),
        )
        return {"ok": True, "review": review, "scenarios": self.store.summaries()}

    def grp_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        town = str(payload.get("town") or "").strip()
        role = str(payload.get("role") or "").strip()
        waypoints = payload.get("waypoints")
        if not town:
            raise ValueError("Town is required for GRP preview")
        if not isinstance(waypoints, list):
            raise ValueError("Waypoints are required for GRP preview")
        return self.carla.grp_preview(town, role, waypoints)

    def grp_prewarm(self) -> dict[str, Any]:
        return self.start_prewarm_all_grp()


APP: ScenarioBuilderApp | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
                return
            if path == "/api/scenarios":
                self._send_json(200, APP.scenario_index())
                return
            if path == "/api/scenario":
                scenario_id = str(query.get("id", [""])[0])
                self._send_json(200, APP.scenario_payload(scenario_id))
                return
            if path == "/api/scenario_bg":
                scenario_id = str(query.get("id", [""])[0])
                self._send_json(200, APP.scenario_bg_payload(scenario_id))
                return
            if path == "/api/carla_status":
                status = APP.carla.status()
                status["prewarm"] = APP.prewarm_status()
                self._send_json(200, status)
                return
            if path == "/api/town_image":
                town = str(query.get("town", [""])[0])
                if not town:
                    self._send_text(400, "missing town")
                    return
                png = APP.get_town_image_png(town)
                if png is None:
                    self._send_text(404, f"no BEV image for {town}")
                    return
                self._send(200, "image/png", png)
                return
            self._send_text(404, "not found")
        except Exception as exc:
            self._error_response(exc)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            payload = self._read_json_body()
            if path == "/api/save":
                self._send_json(200, APP.save_scenario(payload))
                return
            if path == "/api/review":
                self._send_json(200, APP.update_review(payload))
                return
            if path == "/api/grp_preview":
                self._send_json(200, APP.grp_preview(payload))
                return
            if path == "/api/rename":
                self._send_json(200, APP.rename_scenario(payload))
                return
            if path == "/api/grp_prewarm":
                self._send_json(200, APP.grp_prewarm())
                return
            if path == "/api/carla_reconnect":
                restart_managed = bool(payload.get("restart_managed", False))
                self._send_json(200, APP.carla.reconnect(restart_managed=restart_managed))
                return
            self._send_text(404, "not found")
        except Exception as exc:
            self._error_response(exc)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _send(self, status_code: int, content_type: str, payload: bytes) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status_code: int, payload: Any) -> None:
        self._send(status_code, "application/json; charset=utf-8", _json_dumps(payload))

    def _send_text(self, status_code: int, text: str) -> None:
        self._send(status_code, "text/plain; charset=utf-8", text.encode("utf-8"))

    def _error_response(self, exc: Exception) -> None:
        traceback.print_exc()
        self._send_json(
            400,
            {
                "error": str(exc),
                "type": type(exc).__name__,
            },
        )


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Scenario Validation Editor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root{
      --bg:#10161d;
      --panel:#17212b;
      --panel-2:#111923;
      --panel-3:#0c1218;
      --line:#294052;
      --text:#e8ecef;
      --muted:#8fa1ad;
      --accent:#f2a65a;
      --accent-2:#4fb3bf;
      --good:#6fcf97;
      --bad:#eb5757;
      --warn:#f2c94c;
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;background:var(--bg);color:var(--text);font:13px/1.4 "IBM Plex Sans","Segoe UI",system-ui,sans-serif;overflow:hidden}
    body{display:flex;flex-direction:column}
    button,input,select,textarea{font:inherit}
    button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:7px 11px;border-radius:9px;cursor:pointer}
    button:hover{border-color:var(--accent);color:#fff}
    button.primary{background:linear-gradient(135deg,#a85d2b,#f2a65a);border-color:#e08a43;color:#161616;font-weight:700}
    button.ghost{background:transparent}
    button.good{border-color:#4e8b69;color:var(--good)}
    button.bad{border-color:#984a4a;color:#ffb3b3}
    button.warn{border-color:#91753a;color:#ffe7a3}
    button:disabled{opacity:.45;cursor:not-allowed}
    input,select,textarea{width:100%;border:1px solid var(--line);border-radius:9px;background:var(--panel-2);color:var(--text);padding:7px 9px}
    textarea{min-height:72px;resize:vertical}
    .topbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:linear-gradient(135deg,#141f2b,#1d2a36);border-bottom:1px solid var(--line)}
    .topbar .title{font:700 16px/1.2 "IBM Plex Mono","SFMono-Regular",monospace;letter-spacing:.04em;text-transform:uppercase;color:#ffd6b0}
    .topbar .group{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
    .topbar .spacer{flex:1}
    .status-pill{display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border-radius:999px;border:1px solid var(--line);background:rgba(0,0,0,.18);font:600 12px/1 "IBM Plex Mono",monospace}
    .status-dot{width:9px;height:9px;border-radius:50%;background:#556}
    .status-dot.ok{background:var(--good)}
    .status-dot.err{background:var(--bad)}
    .layout{display:grid;grid-template-columns:minmax(0,1fr) 720px;min-height:0;flex:1}
    .map-panel{position:relative;overflow:hidden;background:radial-gradient(circle at top left,#152433,#0d141b 55%)}
    #mapCanvas{display:block;width:100%;height:100%}
    .overlay{position:absolute;left:18px;top:18px;padding:12px 14px;max-width:340px;border:1px solid rgba(255,255,255,.08);border-radius:14px;background:rgba(9,13,18,.72);backdrop-filter:blur(6px);box-shadow:0 18px 50px rgba(0,0,0,.32)}
    .overlay h1{margin:0 0 6px 0;font:700 14px/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;color:#ffd6b0}
    .overlay p{margin:0;color:var(--muted)}
    .side{display:grid;grid-template-rows:minmax(240px,31vh) minmax(0,1fr);min-height:0;border-left:1px solid var(--line);background:linear-gradient(180deg,#141c25,#0f151d)}
    .queue-pane,.inspector{min-height:0}
    .queue-pane{display:flex;flex-direction:column;border-bottom:1px solid rgba(255,255,255,.06)}
    .pane-head{padding:14px 16px 10px 16px}
    h2{margin:0;font:700 12px/1.2 "IBM Plex Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:#ffd6b0}
    .pane-title-row{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
    .queue-meta,.muted{color:var(--muted)}
    .scenario-list{flex:1;overflow:auto;padding:0 16px 14px 16px;display:flex;flex-direction:column;gap:8px}
    .scenario-tree{display:flex;flex-direction:column;gap:8px}
    .folder-node{border-left:1px solid rgba(255,255,255,.08);margin-left:6px;padding-left:10px}
    .folder-group{display:flex;flex-direction:column;gap:8px}
    .folder-summary{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:6px 4px;color:#c7d1d8;cursor:pointer;list-style:none}
    .folder-summary::-webkit-details-marker{display:none}
    .folder-summary::before{content:'▾';color:var(--muted);font:600 12px/1 "IBM Plex Mono",monospace}
    details:not([open]) > .folder-summary::before{content:'▸'}
    .folder-name{font:600 12px/1.2 "IBM Plex Mono",monospace;color:#b8c6cf}
    .folder-meta{color:var(--muted)}
    .scenario-item,.actor-item{padding:10px 12px;border:1px solid rgba(255,255,255,.06);border-radius:14px;background:rgba(255,255,255,.02);cursor:pointer}
    .scenario-item:hover,.actor-item:hover{border-color:#3b556a}
    .scenario-item.active,.actor-item.active{border-color:#d98a46;background:rgba(242,166,90,.08)}
    .scenario-head,.actor-head{display:flex;justify-content:space-between;align-items:center;gap:10px}
    .scenario-name,.actor-name{font-weight:700}
    .actor-head-main{display:flex;align-items:center;gap:10px;min-width:0}
    .color-chip{width:12px;height:12px;border-radius:999px;flex:0 0 auto;border:1px solid rgba(255,255,255,.28);box-shadow:0 0 0 1px rgba(0,0,0,.25) inset}
    .scenario-meta,.actor-meta{color:var(--muted)}
    .scenario-badges{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
    .badge{display:inline-flex;align-items:center;padding:4px 8px;border-radius:999px;background:#1d2a36;border:1px solid var(--line);font:600 11px/1 "IBM Plex Mono",monospace;text-transform:uppercase}
    .badge.pending{color:#c8d3db}
    .badge.edited{color:var(--accent)}
    .badge.approved{color:var(--good)}
    .badge.rejected{color:#ffb3b3}
    .badge.warn{color:var(--warn)}
    .badge.grp{background:#12202b}
    .badge.grp.loading{color:#f8ddb5}
    .badge.grp.ready{color:var(--good)}
    .badge.grp.queued{color:#c8d3db}
    .badge.grp.stale{color:var(--accent)}
    .badge.grp.unavailable{color:#ffb3b3}
    .badge.grp.na{color:#7f919d}
    .active-preset{border-color:var(--accent) !important;color:var(--accent) !important;background:rgba(242,166,90,.12) !important}
    .layer-bar{position:absolute;right:18px;top:18px;display:flex;gap:6px;z-index:4}
    .layer-btn{padding:5px 10px;font:600 11px/1 "IBM Plex Mono",monospace;border-radius:999px;border:1px solid rgba(255,255,255,.15);background:rgba(9,13,18,.72);color:#8fa1ad;backdrop-filter:blur(4px);cursor:pointer}
    .layer-btn.on{border-color:var(--accent-2);color:var(--accent-2);background:rgba(79,179,191,.12)}
    .button-row{display:flex;flex-wrap:wrap;gap:8px}
    .button-row.tight button{padding:6px 9px}
    .details-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    .details-grid label,.field label{display:block;font:600 11px/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;color:#9cb0be;margin:0 0 5px 0}
    .field{margin:0}
    .small{font-size:11px}
    .filter-row{display:grid;grid-template-columns:1fr;gap:8px}
    .inspector{padding:14px 16px 16px 16px;overflow-y:auto}
    .inspector-grid{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:12px;min-height:min-content}
    .summary-card,.actor-card,.route-card,.prop-card,.warning-card,.waypoint-card{border:1px solid rgba(255,255,255,.06);border-radius:16px;background:rgba(255,255,255,.025)}
    .summary-card{display:grid;grid-template-columns:minmax(0,1fr) 240px;gap:12px;padding:14px}
    .summary-copy{display:flex;flex-direction:column;gap:10px}
    .meta-strip{display:flex;flex-wrap:wrap;gap:8px}
    .weather-card{padding:12px;border-radius:14px;border:1px solid rgba(255,255,255,.06);background:rgba(12,18,24,.68)}
    .weather-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    .weather-actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .metric-chip{display:inline-flex;align-items:center;gap:6px;padding:7px 10px;border-radius:999px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);color:#d7e1e8}
    .metric-chip strong{font:600 11px/1 "IBM Plex Mono",monospace;color:#fff3e1;text-transform:uppercase;letter-spacing:.04em}
    .actor-card{padding:14px;display:flex;flex-direction:column;gap:10px;min-height:0}
    .actor-tools{display:flex;flex-wrap:wrap;gap:8px}
    .actor-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;overflow:auto;max-height:152px}
    .editor-grid{display:grid;grid-template-columns:300px minmax(0,1fr);gap:12px;min-height:0}
    .route-stack{display:flex;flex-direction:column;gap:12px;min-height:0}
    .route-card,.prop-card,.warning-card{padding:14px;display:flex;flex-direction:column;gap:10px;min-height:0}
    .route-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    .stat-block{padding:10px 11px;border-radius:12px;border:1px solid rgba(255,255,255,.06);background:rgba(12,18,24,.7)}
    .stat-label{display:block;font:600 10px/1.2 "IBM Plex Mono",monospace;text-transform:uppercase;color:#91a4b3;margin-bottom:5px}
    .stat-value{display:block;font-weight:700;color:#f7fbfd;word-break:break-word}
    table{width:100%;border-collapse:collapse;min-width:760px}
    th,td{border-bottom:1px solid rgba(255,255,255,.06);padding:6px 4px;text-align:left;vertical-align:top}
    th{font:600 11px/1.2 "IBM Plex Mono",monospace;color:#9cb0be;text-transform:uppercase;position:sticky;top:0;background:#10161d}
    td input{min-width:72px}
    .table-wrap{overflow:auto;border:1px solid rgba(255,255,255,.05);border-radius:12px;background:rgba(0,0,0,.1)}
    .mini-table{width:100%;min-width:0}
    .mini-table th,.mini-table td{padding:5px 6px}
    .mini-table th{position:static;background:transparent}
    .mini-table td input{min-width:0}
    .section-note{margin:0 0 8px 0;color:var(--muted)}
    .waypoint-card{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:10px;padding:14px;min-height:0}
    .waypoint-selection{padding:11px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.06);background:rgba(12,18,24,.72)}
    .waypoint-tools{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
    .waypoint-fields{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-top:10px}
    .waypoint-wrap{min-height:0}
    .waypoint-row.changed{background:rgba(79,179,191,.08)}
    .waypoint-row.selected{background:rgba(242,166,90,.1)}
    .row-delta{display:inline-flex;align-items:center;padding:2px 6px;border-radius:999px;background:rgba(79,179,191,.15);border:1px solid rgba(79,179,191,.2);font:600 10px/1 "IBM Plex Mono",monospace;color:#9fe0e7}
    .msg{padding:10px 12px;border-top:1px solid var(--line);background:#0c1218;color:var(--muted);min-height:42px}
    .warn-list{display:flex;flex-direction:column;gap:6px}
    .warn-item{padding:8px 10px;border-radius:10px;background:rgba(242,201,76,.07);border:1px solid rgba(242,201,76,.18);color:#f8ddb5}
    .hidden{display:none !important}
    .map-help{position:absolute;right:18px;bottom:18px;padding:10px 12px;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(9,13,18,.72);color:var(--muted);max-width:300px}
    .nudge-menu{position:absolute;z-index:5;min-width:156px;padding:12px;border-radius:14px;border:1px solid rgba(255,255,255,.1);background:rgba(8,12,17,.92);box-shadow:0 16px 42px rgba(0,0,0,.38)}
    .nudge-grid{display:grid;grid-template-columns:repeat(3,36px);gap:6px;justify-content:center;margin-top:8px}
    .nudge-grid button{padding:7px 0;min-width:0}
    .nudge-head{display:flex;align-items:center;justify-content:space-between;gap:10px}
    .nudge-meta{color:var(--muted)}
    .helper-note{color:var(--muted)}
    button.toggle-on{border-color:var(--accent-2);color:var(--accent-2);background:rgba(79,179,191,.12)}
    body.old-flow{
      --bg:#111;
      --panel:#1a1a1a;
      --panel-2:#222;
      --panel-3:#181818;
      --line:#2b2b2b;
      --text:#e5e5e5;
      --muted:#b4b4b4;
      --accent:#2979ff;
      --accent-2:#2979ff;
      --good:#2ecc71;
      --bad:#ff6b6b;
      --warn:#f2c94c;
      font:13px/1.4 "Segoe UI",Arial,sans-serif;
    }
    body.old-flow .topbar{
      background:#161616;
      border-bottom:1px solid #2b2b2b;
      gap:8px;
      padding:8px 10px;
    }
    body.old-flow .topbar .title{
      font:700 15px/1.2 "Segoe UI",Arial,sans-serif;
      letter-spacing:0;
      text-transform:none;
      color:#e8e8e8;
      margin-right:6px;
    }
    body.old-flow .layout{grid-template-columns:minmax(0,1fr) 560px}
    body.old-flow .side{
      background:#1a1a1a;
      border-left:1px solid #2b2b2b;
      grid-template-rows:minmax(210px,34vh) minmax(0,1fr);
    }
    body.old-flow .map-panel{
      background:#111;
    }
    body.old-flow .overlay{
      border-radius:4px;
      padding:8px 10px;
      max-width:320px;
      background:rgba(17,17,17,.85);
      border:1px solid #2b2b2b;
      backdrop-filter:none;
      box-shadow:none;
    }
    body.old-flow .overlay h1{
      font:700 14px/1.2 "Segoe UI",Arial,sans-serif;
      text-transform:none;
      color:#e5e5e5;
      letter-spacing:0;
    }
    body.old-flow .pane-head,.old-flow .inspector{padding:10px}
    body.old-flow .pane-title-row{margin-bottom:8px}
    body.old-flow h2{
      font:700 12px/1.2 "Segoe UI",Arial,sans-serif;
      letter-spacing:0;
      text-transform:none;
      color:#e8e8e8;
    }
    body.old-flow button{
      border-radius:4px;
      border:1px solid #3a3a3a;
      background:#2a2a2a;
      color:#f0f0f0;
      padding:6px 10px;
    }
    body.old-flow button:hover{
      border-color:#4f8efc;
      color:#fff;
    }
    body.old-flow button.primary{
      background:#2979ff;
      border-color:#3f88ff;
      color:#fff;
      font-weight:600;
    }
    body.old-flow button.good{border-color:#2f8f5a;color:#8de7b3}
    body.old-flow button.bad{border-color:#9a4d4d;color:#ffc0c0}
    body.old-flow button.warn{border-color:#8a7031;color:#ffe9a6}
    body.old-flow input, body.old-flow select, body.old-flow textarea{
      border-radius:4px;
      border:1px solid #3a3a3a;
      background:#222;
      color:#f0f0f0;
      padding:6px 8px;
    }
    body.old-flow .summary-card,
    body.old-flow .actor-card,
    body.old-flow .route-card,
    body.old-flow .prop-card,
    body.old-flow .warning-card,
    body.old-flow .waypoint-card{
      border-radius:4px;
      border:1px solid #2f2f2f;
      background:#1f1f1f;
      padding:10px;
    }
    body.old-flow .scenario-item,
    body.old-flow .actor-item{
      border-radius:4px;
      border:1px solid #343434;
      background:#202020;
      padding:8px 10px;
    }
    body.old-flow .scenario-item.active,
    body.old-flow .actor-item.active{
      border-color:#2979ff;
      background:rgba(41,121,255,.12);
    }
    body.old-flow .stat-block{
      border-radius:4px;
      background:#202020;
      border:1px solid #343434;
      padding:8px;
    }
    body.old-flow .stat-label{
      font:600 11px/1.2 "Segoe UI",Arial,sans-serif;
      text-transform:none;
      color:#bdbdbd;
    }
    body.old-flow .metric-chip{
      border-radius:4px;
      background:#202020;
      border:1px solid #343434;
      color:#dddddd;
      padding:6px 8px;
    }
    body.old-flow .metric-chip strong{
      font:600 11px/1 "Segoe UI",Arial,sans-serif;
      text-transform:none;
      color:#f0f0f0;
      letter-spacing:0;
    }
    body.old-flow .table-wrap{
      border-radius:4px;
      border:1px solid #343434;
      background:#181818;
    }
    body.old-flow th{
      background:#222;
      color:#d0d0d0;
      font:600 11px/1.2 "Segoe UI",Arial,sans-serif;
      text-transform:none;
    }
    body.old-flow .weather-card{
      border-radius:4px;
      border:1px solid #343434;
      background:#202020;
      padding:10px;
    }
    body.old-flow .msg{
      background:#161616;
      border-top:1px solid #2b2b2b;
      color:#c9c9c9;
      min-height:38px;
      padding:8px 10px;
    }
    body.old-flow .status-pill{
      border-radius:4px;
      border:1px solid #3a3a3a;
      background:#1d1d1d;
      font:600 11px/1 "Segoe UI",Arial,sans-serif;
    }
    body.old-flow .layer-btn{
      border-radius:4px;
      background:rgba(17,17,17,.85);
      border:1px solid #3a3a3a;
      color:#bdbdbd;
      font:600 11px/1 "Segoe UI",Arial,sans-serif;
    }
    body.old-flow .layer-btn.on{
      border-color:#3f88ff;
      color:#8ab6ff;
      background:rgba(41,121,255,.15);
    }
    body.old-flow .map-help{
      max-width:380px;
      border-radius:4px;
      border:1px solid #343434;
      background:rgba(17,17,17,.88);
      padding:8px 10px;
    }
    body.old-flow .folder-node{border-left:1px solid #343434}
    body.old-flow .folder-summary{padding:4px 2px}
    body.old-flow .folder-name{font:600 12px/1.2 "Segoe UI",Arial,sans-serif;color:#d4d4d4}
    body.old-flow .editor-grid{grid-template-columns:minmax(0,1fr)}
    body.old-flow .waypoint-card{grid-template-rows:auto auto minmax(220px,1fr)}
    body.old-flow .summary-card{display:none}
    body.old-flow:not(.show-advanced) .weather-card,
    body.old-flow:not(.show-advanced) .prop-card,
    body.old-flow:not(.show-advanced) .warning-card,
    body.old-flow:not(.show-advanced) .helper-note,
    body.old-flow:not(.show-advanced) #grpSummary,
    body.old-flow:not(.show-advanced) #grpStateBadge,
    body.old-flow:not(.show-advanced) .route-stats{display:none}
    body.old-flow:not(.show-advanced) .route-card .details-grid{
      grid-template-columns:repeat(2,minmax(0,1fr));
    }
    body.old-flow:not(.show-advanced) .route-card .details-grid .field:nth-child(n+5){display:none}
    body.old-flow:not(.show-advanced) .route-card .button-row button:nth-child(-n+3){display:none}
    @media (max-width: 1560px){
      .layout{grid-template-columns:minmax(0,1fr) 660px}
      .editor-grid{grid-template-columns:280px minmax(0,1fr)}
    }
    @media (max-width: 1180px){
      .layout{grid-template-columns:minmax(0,1fr)}
      .side{border-left:none;border-top:1px solid var(--line);grid-template-rows:minmax(200px,34vh) minmax(0,1fr);max-height:56vh}
      .overlay{max-width:260px}
      .summary-card{grid-template-columns:minmax(0,1fr)}
      .weather-grid{grid-template-columns:minmax(0,1fr)}
      .editor-grid{grid-template-columns:minmax(0,1fr)}
      .actor-grid{grid-template-columns:minmax(0,1fr)}
      .waypoint-fields{grid-template-columns:repeat(2,minmax(0,1fr))}
    }
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">Scenario Validation Editor</div>
    <div class="group">
      <button class="ghost" onclick="goPrev()">Prev</button>
      <button class="ghost" onclick="goNext()">Next</button>
      <button class="ghost" onclick="undoEdit()">Undo</button>
      <button class="ghost" onclick="redoEdit()">Redo</button>
      <button class="primary" onclick="saveScenario()">Save In Place</button>
      <button class="primary" id="saveNextBtn" onclick="saveAndNext()">Save + Next</button>
      <button class="ghost" onclick="reloadCurrent(true)">Reload</button>
      <button class="ghost" onclick="focusEgoRoutes()">Focus Ego</button>
      <button class="ghost" onclick="fitView()">Fit All</button>
      <button class="ghost" id="drawWaypointBtn" onclick="toggleDrawWaypointMode()">Draw Waypoints</button>
      <button class="ghost" onclick="refreshGrp()">Run / Refresh GRP</button>
      <button class="ghost" onclick="adoptGrpWaypoints()">Adopt GRP</button>
      <button class="ghost" id="advancedPanelsBtn" onclick="toggleAdvancedPanels()">Advanced Panels</button>
      <button class="ghost" id="uiModeBtn" onclick="toggleUiMode()">Modern UI</button>
    </div>
    <div class="spacer"></div>
    <div class="group">
      <button class="good" onclick="setReviewStatus('approved')">Approve</button>
      <button class="bad" onclick="setReviewStatus('rejected')">Reject</button>
      <button class="warn" onclick="setReviewStatus('edited')">Needs Edit</button>
    </div>
    <div class="status-pill"><span id="carlaDot" class="status-dot"></span><span id="carlaText">CARLA</span></div>
    <button class="ghost" onclick="reconnectCarla(false)">Reconnect</button>
    <button class="ghost" onclick="reconnectCarla(true)">Restart Managed</button>
  </div>

  <div class="layout">
    <div class="map-panel">
      <canvas id="mapCanvas"></canvas>
      <div class="overlay">
        <h1 id="overlayTitle">No Scenario Loaded</h1>
        <p id="overlayMeta">Select a scenario to begin review.</p>
      </div>
      <div id="mapNudgeMenu" class="nudge-menu hidden"></div>
      <div class="layer-bar">
        <button id="layerMapBtn" class="layer-btn on" onclick="toggleLayer('map')" title="Toggle road map">Map</button>
        <button id="layerActorsBtn" class="layer-btn on" onclick="toggleLayer('actors')" title="Toggle actor paths">Actors</button>
        <button id="layerGrpBtn" class="layer-btn on" onclick="toggleLayer('grp')" title="Toggle GRP overlay">GRP</button>
      </div>
      <div id="mapHelp" class="map-help small"></div>
    </div>

    <aside class="side">
      <section class="queue-pane">
        <div class="pane-head">
          <div class="pane-title-row">
            <h2>Scenario Queue</h2>
            <div style="display:flex;align-items:center;gap:10px">
              <div id="scenarioQueueSummary" class="queue-meta small"></div>
              <label style="display:flex;align-items:center;gap:5px;font:600 11px/1 'IBM Plex Mono',monospace;color:var(--muted);cursor:pointer"><input type="checkbox" id="hideReviewedCheck" onchange="toggleHideReviewed(this.checked)" /> Hide reviewed</label>
            </div>
          </div>
          <div class="filter-row">
            <input id="scenarioFilterInput" placeholder="Filter scenarios by folder, name, town, status" oninput="updateScenarioFilter(this.value)" />
          </div>
        </div>
        <div id="scenarioList" class="scenario-list"></div>
      </section>

      <section class="inspector">
        <div class="inspector-grid">
          <div class="summary-card" style="grid-template-columns:minmax(0,1fr)">
            <div class="summary-copy">
              <div class="pane-title-row">
                <div style="display:flex;align-items:center;gap:8px;min-width:0">
                  <h2 id="scenarioNameDisplay" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">No Scenario</h2>
                  <button class="ghost small" id="renameBtn" onclick="renameScenario()" style="display:none;flex-shrink:0">Rename</button>
                </div>
                <div id="scenarioContextMeta" class="small muted" style="flex-shrink:0"></div>
              </div>
              <div id="scenarioSummaryChips" class="meta-strip"></div>
              <div class="weather-card">
                <div class="pane-title-row" style="margin-bottom:8px">
                  <h2>Time / Weather</h2>
                  <div id="weatherSummary" class="small muted"></div>
                </div>
                <div class="weather-actions">
                  <button class="ghost small" data-weather-preset="dawn" onclick="applyTimeOfDayPreset('dawn')">Dawn</button>
                  <button class="ghost small" data-weather-preset="morning" onclick="applyTimeOfDayPreset('morning')">Morning</button>
                  <button class="ghost small" data-weather-preset="noon" onclick="applyTimeOfDayPreset('noon')">Noon</button>
                  <button class="ghost small" data-weather-preset="dusk" onclick="applyTimeOfDayPreset('dusk')">Dusk</button>
                  <button class="ghost small" data-weather-preset="night" onclick="applyTimeOfDayPreset('night')">Night</button>
                  <button class="ghost small" data-weather-preset="default" onclick="applyTimeOfDayPreset('default')">Clear</button>
                </div>
              </div>
            </div>
          </div>

          <div class="actor-card">
            <div class="pane-title-row">
              <h2>Actors</h2>
              <div id="actorSummary" class="small muted"></div>
            </div>
            <div class="actor-tools button-row tight">
              <button class="ghost small" onclick="addActor('ego')">Add Ego</button>
              <button class="ghost small" onclick="addActor('npc')">Add NPC</button>
              <button class="ghost small" onclick="addActor('pedestrian')">Add Ped</button>
              <button class="ghost small" onclick="addActor('bicycle')">Add Bike</button>
              <button class="ghost small" onclick="addActor('static')">Add Prop</button>
            </div>
            <div class="filter-row">
              <input id="actorFilterInput" placeholder="Filter actors by file, role, model, or name" oninput="updateActorFilter(this.value)" />
            </div>
            <div id="actorList" class="actor-grid"></div>
          </div>

          <div class="editor-grid">
            <div class="route-stack">
              <div class="route-card">
                <div class="pane-title-row">
                  <h2>Route</h2>
                  <div id="grpStateBadge"></div>
                </div>
                <div id="routeStats" class="route-stats"></div>
                <div class="details-grid">
                  <div class="field"><label for="routeIdInput">Route Id</label><input id="routeIdInput" onchange="updateRouteAttr('id',this.value)" /></div>
                  <div class="field"><label for="routeTownInput">Town</label><input id="routeTownInput" onchange="updateRouteAttr('town',this.value)" /></div>
                  <div class="field"><label for="routeRoleInput">Role</label><input id="routeRoleInput" onchange="updateRouteAttr('role',this.value)" /></div>
                  <div class="field"><label for="routeModelInput">Model</label><input id="routeModelInput" onchange="updateRouteAttr('model',this.value)" /></div>
                  <div class="field"><label for="routeControlInput">Control</label><select id="routeControlInput" onchange="updateRouteAttr('control_mode',this.value)"><option value=""></option><option value="policy">policy</option><option value="replay">replay</option></select></div>
                  <div class="field"><label for="routeTargetSpeedInput">Target Speed</label><input id="routeTargetSpeedInput" type="number" step="0.1" onchange="updateTargetSpeed(this.value)" /></div>
                  <div class="field"><label for="routeSnapInput">Snap To Road</label><select id="routeSnapInput" onchange="updateRouteAttr('snap_to_road',this.value)"><option value=""></option><option value="true">true</option><option value="false">false</option></select></div>
                  <div class="field"><label for="routeSpawnSnapInput">Snap Spawn</label><select id="routeSpawnSnapInput" onchange="updateRouteAttr('snap_spawn_to_road',this.value)"><option value=""></option><option value="true">true</option><option value="false">false</option></select></div>
                </div>
                <div class="button-row">
                  <button class="ghost" onclick="recomputeSelectedYaws()">Recompute Yaw</button>
                  <button class="ghost" onclick="shiftSelectedTimes()">Shift Times</button>
                  <button class="ghost" id="shiftEgoWaypointsBtn" onclick="shiftSelectedEgoWaypoints()">Shift Ego XY</button>
                  <button class="ghost" onclick="normalizeSelectedTimes()">Set t0</button>
                  <button class="ghost" onclick="appendWaypoint()">Append Waypoint</button>
                  <button class="ghost" id="routeDrawWaypointBtn" onclick="toggleDrawWaypointMode()">Draw On Map</button>
                  <button class="ghost" onclick="deleteSelectedActor()">Delete Actor</button>
                </div>
                <div class="helper-note small">Target speed is the route behavior speed. Spawn speed is only explicit when the first waypoint carries a speed value.</div>
                <div id="grpSummary" class="small muted">GRP preview not loaded.</div>
              </div>

              <div class="prop-card">
                <div class="pane-title-row">
                  <h2>Route Props</h2>
                  <button class="ghost small" onclick="addRouteProp()">Add Prop</button>
                </div>
                <div class="table-wrap" style="max-height:190px">
                  <table id="routeAttrTable" class="mini-table"></table>
                </div>
              </div>

              <div class="warning-card">
                <div class="pane-title-row">
                  <h2>Warnings</h2>
                </div>
                <div id="warningList" class="warn-list"></div>
              </div>
            </div>

            <div class="waypoint-card">
              <div class="pane-title-row">
                <h2>Waypoints</h2>
                <div id="waypointSummary" class="small muted"></div>
              </div>
              <div id="waypointInspector" class="waypoint-selection small">Select a waypoint to edit it directly, duplicate it, or delete it.</div>
              <div class="table-wrap waypoint-wrap">
                <table id="waypointTable"></table>
              </div>
            </div>
          </div>
        </div>
      </section>
    </aside>
  </div>

  <div id="message" class="msg">Loading scenario index…</div>

  <script>
    const state = {
      scenarios: [],
      scenario: null,
      currentIndex: -1,
      selectedActorId: null,
      selectedWaypointIndex: -1,
      grpPreview: null,
      grpPreviewCache: {},
      dirty: false,
      view: {scale: 1, tx: 0, ty: 0},
      drag: null,
      hoverActorId: null,
      hoverWaypointIndex: -1,
      carla: null,
      grpConfig: null,
      scenarioFilter: '',
      actorFilter: '',
      hideReviewed: false,
      grpScenarioStatus: {},
      historyUndo: [],
      historyRedo: [],
      nudgeStep: 0.5,
      nudgeMenu: null,
      drawWaypointMode: false,
      uiMode: 'classic',
      showAdvancedPanels: false,
      layers: {map: true, actors: true, grp: true},
      townImageEl: null,     // HTMLImageElement for the current town's BEV
      townImageMeta: null,   // {town, origin_x, origin_y, px_per_meter}
    };

    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');
    const mapPanelEl = document.querySelector('.map-panel');
    const nudgeMenuEl = document.getElementById('mapNudgeMenu');
    const messageEl = document.getElementById('message');
    const EGO_ROUTE_COLORS = ['#5ab0ff','#ff8b7b','#7edc8b','#f6c85f','#c792ea','#61d6d6','#f78fb3','#8ac6ff','#ffd166','#95e06c'];
    const DEFAULT_ROUTE_MODEL = {
      ego: 'vehicle.lincoln.mkz2017',
      npc: 'vehicle.audi.a2',
      pedestrian: 'walker.pedestrian.0001',
      bicycle: 'vehicle.bh.crossbike',
      static: 'static.prop.trafficcone',
    };
    const DEFAULT_ROUTE_BBOX = {
      'vehicle.lincoln.mkz2017': {length:4.901683,width:2.128324,height:1.510746},
      'vehicle.audi.a2': {length:3.705369,width:1.788679,height:1.547087},
      'walker.pedestrian.0001': {length:0.68,width:0.68,height:1.86},
      'vehicle.bh.crossbike': {length:1.487289,width:0.859257,height:1.079579},
      'static.prop.trafficcone': {length:0.456843,width:0.456843,height:0.824513},
    };
    const TIME_OF_DAY_PRESETS = {
      dawn: {sun_altitude_angle: -2, sun_azimuth_angle: 80, cloudiness: 20, precipitation: 0, wetness: 0},
      morning: {sun_altitude_angle: 18, sun_azimuth_angle: 110, cloudiness: 20, precipitation: 0, wetness: 0},
      noon: {sun_altitude_angle: 65, sun_azimuth_angle: 180, cloudiness: 10, precipitation: 0, wetness: 0},
      dusk: {sun_altitude_angle: -2, sun_azimuth_angle: 260, cloudiness: 25, precipitation: 0, wetness: 0},
      night: {sun_altitude_angle: -15, sun_azimuth_angle: 300, cloudiness: 20, precipitation: 0, wetness: 10},
    };
    const UI_MODE_STORAGE_KEY = 'scenario_builder_ui_mode';
    const ADVANCED_PANELS_STORAGE_KEY = 'scenario_builder_show_advanced';
    const DRAW_MODE_STORAGE_KEY = 'scenario_builder_draw_waypoint_mode';
    let dpr = window.devicePixelRatio || 1;

    function clone(v){ return JSON.parse(JSON.stringify(v)); }
    function setMessage(text){ messageEl.textContent = text; }
    function readStoredString(key, fallback){
      try{
        const value = window.localStorage.getItem(key);
        return value == null ? fallback : value;
      }catch(_err){
        return fallback;
      }
    }
    function readStoredBool(key, fallback){
      const value = readStoredString(key, fallback ? '1' : '0');
      return value === '1' || value === 'true';
    }
    function writeStoredValue(key, value){
      try{
        window.localStorage.setItem(key, String(value));
      }catch(_err){
        // ignored
      }
    }
    function isTypingField(){
      const el = document.activeElement;
      if(!el) return false;
      const tag = String(el.tagName || '').toUpperCase();
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
    }
    function renderMapHelp(){
      const helpEl = document.getElementById('mapHelp');
      if(!helpEl) return;
      const drawText = state.drawWaypointMode ? 'Draw mode ON: click map to append waypoint.' : 'Draw mode OFF: click selects.';
      helpEl.textContent = `${drawText} Shift-drag moves waypoint. Double-click inserts waypoint. Delete removes selected. W toggles draw mode. Ctrl/Cmd+S saves.`;
    }
    function refreshUiToggleButtons(){
      const uiBtn = document.getElementById('uiModeBtn');
      if(uiBtn){
        uiBtn.textContent = state.uiMode === 'classic' ? 'Modern UI' : 'Classic UI';
      }
      const advBtn = document.getElementById('advancedPanelsBtn');
      if(advBtn){
        advBtn.textContent = state.showAdvancedPanels ? 'Hide Advanced' : 'Advanced Panels';
        advBtn.classList.toggle('toggle-on', !!state.showAdvancedPanels);
      }
      const drawButtons = [document.getElementById('drawWaypointBtn'), document.getElementById('routeDrawWaypointBtn')];
      for(const btn of drawButtons){
        if(!btn) continue;
        btn.textContent = state.drawWaypointMode ? 'Draw Waypoints: ON' : 'Draw Waypoints: OFF';
        btn.classList.toggle('toggle-on', !!state.drawWaypointMode);
      }
    }
    function applyUiPreferences(){
      document.body.classList.toggle('old-flow', state.uiMode === 'classic');
      document.body.classList.toggle('show-advanced', !!state.showAdvancedPanels);
      refreshUiToggleButtons();
      renderMapHelp();
      drawMap();
    }
    function toggleUiMode(){
      state.uiMode = state.uiMode === 'classic' ? 'modern' : 'classic';
      writeStoredValue(UI_MODE_STORAGE_KEY, state.uiMode);
      applyUiPreferences();
    }
    function toggleAdvancedPanels(){
      state.showAdvancedPanels = !state.showAdvancedPanels;
      writeStoredValue(ADVANCED_PANELS_STORAGE_KEY, state.showAdvancedPanels ? '1' : '0');
      applyUiPreferences();
    }
    function toggleDrawWaypointMode(){
      state.drawWaypointMode = !state.drawWaypointMode;
      writeStoredValue(DRAW_MODE_STORAGE_KEY, state.drawWaypointMode ? '1' : '0');
      refreshUiToggleButtons();
      renderMapHelp();
    }
    function currentRoute(){ return state.scenario ? state.scenario.routes.find(r => r.actor_id === state.selectedActorId) || null : null; }
    function selectedOriginal(){ const route = currentRoute(); return route ? route.original_waypoints : []; }
    function currentScenarioSummary(){ return state.scenario ? state.scenarios.find(item => item.id === state.scenario.id) || null : null; }
    function markDirty(reason){ state.dirty = true; refreshCurrentScenarioGrpState(); renderTopSummary(reason || 'Unsaved edits'); }
    function badgeClass(status){ return ['badge', status || 'pending'].join(' '); }
    function grpBadgeClass(status){ return ['badge', 'grp', status || 'queued'].join(' '); }
    function hashString(text){
      let hash = 0;
      const source = String(text || '');
      for(let i = 0; i < source.length; i += 1){
        hash = ((hash << 5) - hash) + source.charCodeAt(i);
        hash |= 0;
      }
      return Math.abs(hash);
    }
    function withAlpha(color, alpha){
      const hex = String(color || '').replace('#', '');
      if(hex.length !== 6) return color;
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }
    function darkenColor(color, factor){
      // factor 0=black, 1=original
      const hex = String(color || '').replace('#', '');
      if(hex.length !== 6) return color;
      const r = Math.round(parseInt(hex.slice(0, 2), 16) * factor);
      const g = Math.round(parseInt(hex.slice(2, 4), 16) * factor);
      const b = Math.round(parseInt(hex.slice(4, 6), 16) * factor);
      return `rgb(${r}, ${g}, ${b})`;
    }
    function normalizeSearch(text){
      return String(text || '').trim().toLowerCase();
    }
    function safeNumber(value){
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }
    function formatMetric(value, digits=2, suffix=''){
      const num = safeNumber(value);
      return num == null ? 'n/a' : `${num.toFixed(digits)}${suffix}`;
    }
    function formatSpeed(value){
      const num = safeNumber(value);
      return num == null ? 'not set' : `${num.toFixed(1)} m/s`;
    }
    function scenarioWeather(){
      if(!state.scenario) return {};
      if(!state.scenario.weather || typeof state.scenario.weather !== 'object'){
        state.scenario.weather = {};
      }
      return state.scenario.weather;
    }
    function weatherPresetLabel(weather){
      const altitude = safeNumber(weather && weather.sun_altitude_angle);
      if(altitude == null) return 'runtime default';
      if(altitude < -8) return 'night';
      if(altitude < 4) return 'dawn/dusk';
      if(altitude < 32) return 'morning/evening';
      return 'day';
    }
    function detectWeatherPreset(weather){
      const attrs = weather || {};
      for(const [name, preset] of Object.entries(TIME_OF_DAY_PRESETS)){
        const keys = Object.keys(preset);
        let match = true;
        for(const key of keys){
          const left = safeNumber(attrs[key]);
          const right = safeNumber(preset[key]);
          if(left == null || right == null || Math.abs(left - right) > 0.05){
            match = false;
            break;
          }
        }
        if(match) return name;
      }
      return Object.keys(attrs).length ? 'custom' : 'default';
    }
    function routeRole(route){
      return String(route && ((route.route_attrs && route.route_attrs.role) || route.kind) || '').toLowerCase();
    }
    function routeBaseColor(route){
      const role = routeRole(route);
      if(role === 'ego'){
        const egoRoutes = state.scenario && Array.isArray(state.scenario.routes) ? state.scenario.routes.filter(item => routeRole(item) === 'ego') : [];
        const idx = egoRoutes.findIndex(item => item.actor_id === (route && route.actor_id));
        if(idx >= 0) return EGO_ROUTE_COLORS[idx % EGO_ROUTE_COLORS.length];
        const key = route && (route.actor_id || route.file || route.name || (route.route_attrs && route.route_attrs.id)) || 'ego';
        return EGO_ROUTE_COLORS[hashString(key) % EGO_ROUTE_COLORS.length];
      }
      return {npc:'#f2a65a', pedestrian:'#86d39c', walker:'#86d39c', bicycle:'#d49cff', cyclist:'#d49cff', static:'#9aa7b4'}[role] || '#c7d1d8';
    }
    function routeColorValue(route, selected=false){
      const base = routeBaseColor(route);
      return selected ? base : withAlpha(base, 0.92);
    }
    function routeEffectiveModel(route){
      return (route && route.route_attrs && route.route_attrs.model) || (route && route.resolved_model) || (route && route.manifest_entry && route.manifest_entry.model) || '';
    }
    function isGenericRouteName(name){
      const text = String(name || '').trim().toLowerCase();
      if(!text) return true;
      if(/^(entity|actor|static|npc|pedestrian|bicycle|walker|cyclist|ego)(_[a-z0-9]+)+$/.test(text)) return true;
      if(/^town[0-9a-z]+_(entity|actor|static|npc|pedestrian|bicycle|walker|cyclist|ego)_/.test(text)) return true;
      return false;
    }
    function titleCaseWords(text){
      return String(text || '')
        .split(' ')
        .filter(Boolean)
        .map(token => token.charAt(0).toUpperCase() + token.slice(1))
        .join(' ');
    }
    function modelLabelFromId(modelId, role){
      const model = String(modelId || '').trim();
      if(!model) return '';
      const parts = model.split('.').filter(Boolean);
      if(!parts.length) return '';
      let text = '';
      if(parts[0] === 'static' && parts[1] === 'prop'){
        text = parts.slice(2).join(' ');
      }else if(parts[0] === 'vehicle'){
        text = parts.slice(1).join(' ');
      }else if(parts[0] === 'walker'){
        text = parts.slice(1).join(' ');
      }else if(role === 'pedestrian' || role === 'walker'){
        text = 'pedestrian';
      }else if(role === 'bicycle' || role === 'cyclist'){
        text = 'bicycle';
      }else{
        text = parts.slice(-1)[0];
      }
      return titleCaseWords(text.replace(/[._-]+/g, ' ').trim());
    }
    function routeDisplayName(route){
      const rawName = String(route && route.name || '').trim();
      const modelLabel = modelLabelFromId(routeEffectiveModel(route), routeRole(route));
      if(!rawName) return modelLabel || 'Actor';
      if(modelLabel && isGenericRouteName(rawName)) return `${modelLabel} (${rawName})`;
      return rawName;
    }
    function defaultBBoxForRoute(route){
      const model = routeEffectiveModel(route) || DEFAULT_ROUTE_MODEL[routeRole(route)] || '';
      return clone(DEFAULT_ROUTE_BBOX[model] || null);
    }
    function routeBbox(route){
      if(route && route.bbox && safeNumber(route.bbox.width) != null) return route.bbox;
      return route ? defaultBBoxForRoute(route) : null;
    }
    function routeBboxText(route){
      const bbox = routeBbox(route);
      return bbox ? `${formatMetric(bbox.length, 2)} x ${formatMetric(bbox.width, 2)} x ${formatMetric(bbox.height, 2)} m` : 'n/a';
    }
    function routeTargetSpeedValue(route){
      return route ? safeNumber(route.route_attrs && (route.route_attrs.target_speed ?? route.route_attrs.speed)) : null;
    }
    function routeStartSpeedValue(route){
      const first = route && Array.isArray(route.waypoints) && route.waypoints.length ? route.waypoints[0] : null;
      return first ? safeNumber(first.speed) : null;
    }
    function routeStartTimeText(route){
      const first = route && Array.isArray(route.waypoints) && route.waypoints.length ? route.waypoints[0] : null;
      return first && safeNumber(first.time) != null ? `${Number(first.time).toFixed(2)} s` : 'untimed';
    }
    function routeVehicleWidthMeters(route){
      const bbox = routeBbox(route);
      if(bbox && safeNumber(bbox.width) != null && Number(bbox.width) > 0){
        return Number(bbox.width);
      }
      return {ego:2.13, npc:1.95, pedestrian:0.7, walker:0.7, bicycle:0.85, cyclist:0.85, static:1.2}[routeRole(route)] || 1.4;
    }
    function routeStrokeWidth(route, selected=false){
      const scaled = routeVehicleWidthMeters(route) * state.view.scale;
      const minPx = selected ? 4.5 : 3.0;
      const maxPx = selected ? 34 : 28;
      return Math.max(minPx, Math.min(maxPx, scaled));
    }
    function editableSnapshot(){
      return {
        routes: clone(state.scenario ? state.scenario.routes : []),
        weather: clone(state.scenario ? state.scenario.weather || {} : {}),
        selectedActorId: state.selectedActorId,
        selectedWaypointIndex: state.selectedWaypointIndex,
        grpPreviewCache: clone(state.grpPreviewCache || {}),
      };
    }
    function restoreEditableSnapshot(snapshot, reason){
      if(!state.scenario || !snapshot) return;
      state.scenario.routes = clone(snapshot.routes || []);
      state.scenario.weather = clone(snapshot.weather || {});
      state.selectedActorId = snapshot.selectedActorId || (state.scenario.routes[0] && state.scenario.routes[0].actor_id) || null;
      state.selectedWaypointIndex = Number.isInteger(snapshot.selectedWaypointIndex) ? snapshot.selectedWaypointIndex : -1;
      state.grpPreviewCache = clone(snapshot.grpPreviewCache || {});
      syncSelectedGrpPreviewFromCache();
      state.dirty = true;
      hideNudgeMenu();
      renderAll(reason || 'Restored edit');
    }
    function pushHistory(label){
      if(!state.scenario) return;
      state.historyUndo.push({label: label || 'edit', snapshot: editableSnapshot()});
      if(state.historyUndo.length > 80){
        state.historyUndo.shift();
      }
      state.historyRedo = [];
    }
    function undoEdit(){
      if(!state.scenario || !state.historyUndo.length) return;
      const entry = state.historyUndo.pop();
      state.historyRedo.push({label: entry.label, snapshot: editableSnapshot()});
      restoreEditableSnapshot(entry.snapshot, `Undo ${entry.label}`);
    }
    function redoEdit(){
      if(!state.scenario || !state.historyRedo.length) return;
      const entry = state.historyRedo.pop();
      state.historyUndo.push({label: entry.label, snapshot: editableSnapshot()});
      restoreEditableSnapshot(entry.snapshot, `Redo ${entry.label}`);
    }
    function hideNudgeMenu(){
      state.nudgeMenu = null;
      nudgeMenuEl.classList.add('hidden');
      nudgeMenuEl.innerHTML = '';
    }
    function showNudgeMenu(clientX, clientY, actorId, waypointIndex){
      state.nudgeMenu = {actorId, waypointIndex};
      const rect = mapPanelEl.getBoundingClientRect();
      const left = Math.min(Math.max(16, clientX - rect.left + 12), Math.max(16, rect.width - 180));
      const top = Math.min(Math.max(16, clientY - rect.top + 12), Math.max(16, rect.height - 196));
      nudgeMenuEl.style.left = `${left}px`;
      nudgeMenuEl.style.top = `${top}px`;
      const canDelete = currentRoute() && currentRoute().waypoints.length > 1;
      nudgeMenuEl.innerHTML = `
        <div class="nudge-head">
          <strong>Waypoint ${waypointIndex}</strong>
          <button class="ghost small" onclick="hideNudgeMenu()">Close</button>
        </div>
        <div class="nudge-meta small">Step <select onchange="state.nudgeStep=Number(this.value)"><option value="0.25" ${state.nudgeStep===0.25?'selected':''}>0.25m</option><option value="0.5" ${state.nudgeStep===0.5?'selected':''}>0.5m</option><option value="1" ${state.nudgeStep===1?'selected':''}>1m</option><option value="2" ${state.nudgeStep===2?'selected':''}>2m</option></select></div>
        <div class="nudge-grid">
          <span></span>
          <button class="ghost" onclick="nudgeWaypoint(0,1)">↑</button>
          <span></span>
          <button class="ghost" onclick="nudgeWaypoint(-1,0)">←</button>
          <button class="ghost" onclick="nudgeWaypoint(0,0,true)">⟲</button>
          <button class="ghost" onclick="nudgeWaypoint(1,0)">→</button>
          <span></span>
          <button class="ghost" onclick="nudgeWaypoint(0,-1)">↓</button>
          <span></span>
        </div>
        <div style="display:flex;gap:6px;margin-top:8px">
          <button class="ghost small" onclick="insertWaypointAfter(${waypointIndex});hideNudgeMenu()">Insert After</button>
          <button class="ghost small" style="color:#f87171" onclick="deleteWaypoint(${waypointIndex});hideNudgeMenu()" ${canDelete?'':'disabled'}>Delete</button>
        </div>
      `;
      nudgeMenuEl.classList.remove('hidden');
    }
    function renderScenarioWeather(){
      const weather = scenarioWeather();
      const summaryEl = document.getElementById('weatherSummary');
      const preset = detectWeatherPreset(weather);
      if(summaryEl) summaryEl.textContent = Object.keys(weather).length ? `${weatherPresetLabel(weather)} | XML override` : 'runtime default';
      document.querySelectorAll('[data-weather-preset]').forEach(btn => {
        btn.classList.toggle('active-preset', btn.dataset.weatherPreset === preset);
      });
    }
    function updateScenarioWeatherAttr(key, value){
      if(!state.scenario) return;
      pushHistory('scenario weather');
      const weather = scenarioWeather();
      if(value === ''){
        delete weather[key];
      }else{
        const num = Number(value);
        if(Number.isFinite(num)){
          weather[key] = Number(num.toFixed(2));
        }
      }
      if(!Object.keys(weather).length){
        state.scenario.weather = {};
      }
      markDirty('Scenario weather updated');
      renderScenarioWeather();
      renderScenarioContext();
    }
    function applyTimeOfDayPreset(name){
      if(!state.scenario) return;
      if(!name) return;
      pushHistory('time of day');
      if(name === 'default'){
        state.scenario.weather = {};
      }else if(name === 'custom'){
        renderScenarioWeather();
        return;
      }else{
        state.scenario.weather = clone(TIME_OF_DAY_PRESETS[name] || {});
      }
      markDirty('Scenario weather updated');
      renderScenarioWeather();
      renderScenarioContext();
    }
    function nudgeWaypoint(dx, dy, recenter=false){
      const menu = state.nudgeMenu;
      const route = currentRoute();
      if(!route) return;
      const idx = menu && route.actor_id === menu.actorId ? menu.waypointIndex : state.selectedWaypointIndex;
      if(idx == null || idx < 0) return;
      const wp = route.waypoints[idx];
      if(!wp) return;
      pushHistory('nudge waypoint');
      if(recenter){
        const original = route.original_waypoints && route.original_waypoints[idx];
        if(original){
          wp.x = Number(original.x);
          wp.y = Number(original.y);
          if(original.z != null) wp.z = Number(original.z);
        }
      }else{
        const step = Number(state.nudgeStep || 0.5);
        wp.x = Number(wp.x) + dx * step;
        wp.y = Number(wp.y) + dy * step;
      }
      recomputeRouteYaws(route);
      invalidateSelectedGrpCache();
      markDirty('Waypoint nudged');
      renderWaypointInspector();
      renderWaypointTable();
      renderRouteDetails();
      drawMap();
    }
    function routeMatchesFilter(route){
      const filter = normalizeSearch(state.actorFilter);
      if(!filter) return true;
      const haystack = [
        routeDisplayName(route),
        route.name,
        route.file,
        route.kind,
        routeEffectiveModel(route),
        route.route_attrs && route.route_attrs.role,
        route.route_attrs && route.route_attrs.id,
        route.route_attrs && route.route_attrs.town,
      ].map(normalizeSearch).join(' ');
      return haystack.includes(filter);
    }
    function scenarioMatchesFilter(item){
      if(state.hideReviewed && (item.status === 'approved' || item.status === 'rejected')) return false;
      const filter = normalizeSearch(state.scenarioFilter);
      if(!filter) return true;
      const haystack = [
        item.name,
        item.id,
        item.relative_path,
        item.parent_path,
        item.status,
        item.town,
      ].map(normalizeSearch).join(' ');
      return haystack.includes(filter);
    }
    function scenarioEgoRouteCount(item){
      if(!item) return 0;
      if(Array.isArray(item.routes)){
        return item.routes.filter(route => routeRole(route) === 'ego').length;
      }
      return Number(item.ego_route_count || 0);
    }
    function grpStatusLabel(status, item){
      const total = scenarioEgoRouteCount(item);
      if(total <= 0) return 'no grp';
      return {queued:'grp queued', loading:'grp loading', ready:'grp ready', stale:'grp stale', unavailable:'grp error', na:'no grp'}[status] || 'grp queued';
    }
    function markScenarioGrpState(id, status){
      if(!id) return;
      state.grpScenarioStatus[id] = status;
    }
    function cachedGrpPreviewCount(){
      return Object.values(state.grpPreviewCache || {}).filter(payload => payload && payload.supported).length;
    }
    function deriveCurrentScenarioGrpState(){
      if(!state.scenario) return 'queued';
      const total = scenarioEgoRouteCount(state.scenario);
      if(total <= 0) return 'na';
      const readyCount = cachedGrpPreviewCount();
      if(readyCount >= total) return 'ready';
      if(readyCount > 0 || state.dirty) return 'stale';
      return 'queued';
    }
    function refreshCurrentScenarioGrpState(){
      if(state.scenario){
        markScenarioGrpState(state.scenario.id, deriveCurrentScenarioGrpState());
        renderScenarioList();
      }
    }
    function initScenarioGrpStates(){
      const next = {};
      for(const item of state.scenarios){
        next[item.id] = state.grpScenarioStatus[item.id] || (scenarioEgoRouteCount(item) > 0 ? 'queued' : 'na');
      }
      state.grpScenarioStatus = next;
    }
    function emptyBounds(){
      return {xmin: Infinity, xmax: -Infinity, ymin: Infinity, ymax: -Infinity};
    }
    function expandBounds(bounds, x, y){
      if(!Number.isFinite(x) || !Number.isFinite(y)) return;
      bounds.xmin = Math.min(bounds.xmin, x);
      bounds.xmax = Math.max(bounds.xmax, x);
      bounds.ymin = Math.min(bounds.ymin, y);
      bounds.ymax = Math.max(bounds.ymax, y);
    }
    function boundsFromRouteSet(routes){
      const bounds = emptyBounds();
      for(const route of routes || []){
        for(const wp of route.waypoints || []){
          expandBounds(bounds, Number(wp.x), Number(wp.y));
        }
      }
      return Number.isFinite(bounds.xmin) ? bounds : null;
    }
    function paddedBounds(bounds, padRatio=0.12, minPad=12){
      if(!bounds) return null;
      const width = Math.max(1, bounds.xmax - bounds.xmin);
      const height = Math.max(1, bounds.ymax - bounds.ymin);
      const padX = Math.max(minPad, width * padRatio);
      const padY = Math.max(minPad, height * padRatio);
      return {xmin: bounds.xmin - padX, xmax: bounds.xmax + padX, ymin: bounds.ymin - padY, ymax: bounds.ymax + padY};
    }
    function fitBounds(bounds, scaleFactor=0.92){
      const rect = canvas.getBoundingClientRect();
      if(!bounds) return;
      const width = Math.max(1, bounds.xmax - bounds.xmin);
      const height = Math.max(1, bounds.ymax - bounds.ymin);
      const scale = Math.min(rect.width / width, rect.height / height) * scaleFactor;
      state.view.scale = Number.isFinite(scale) && scale > 0 ? scale : 1;
      state.view.tx = rect.width * 0.5 - ((bounds.xmin + bounds.xmax) * 0.5) * state.view.scale;
      state.view.ty = rect.height * 0.5 + ((bounds.ymin + bounds.ymax) * 0.5) * state.view.scale;
      drawMap();
    }
    function syncSelectedGrpPreviewFromCache(){
      const actorId = state.selectedActorId;
      if(actorId && state.grpPreviewCache && state.grpPreviewCache[actorId]){
        state.grpPreview = clone(state.grpPreviewCache[actorId]);
      }else{
        state.grpPreview = null;
      }
    }
    function updateScenarioFilter(value){
      state.scenarioFilter = value || '';
      renderScenarioList();
    }
    function updateActorFilter(value){
      state.actorFilter = value || '';
      renderActorList();
    }
    function toggleHideReviewed(value){
      state.hideReviewed = !!value;
      renderScenarioList();
    }
    function toggleLayer(name){
      state.layers[name] = !state.layers[name];
      const btn = document.getElementById('layer' + name.charAt(0).toUpperCase() + name.slice(1) + 'Btn');
      if(btn) btn.classList.toggle('on', state.layers[name]);
      drawMap();
    }
    function invalidateSelectedGrpCache(){
      const route = currentRoute();
      if(route && route.actor_id && state.grpPreviewCache){
        delete state.grpPreviewCache[route.actor_id];
      }
      state.grpPreview = null;
      refreshCurrentScenarioGrpState();
    }

    function formatGrpConfig(config){
      if(!config) return '';
      const res = config.sampling_resolution != null ? `res ${Number(config.sampling_resolution).toFixed(2)}m` : 'res ?';
      const mode = config.postprocess_mode ? `pp ${config.postprocess_mode}` : 'pp default';
      const endpoints = `ignore endpoints ${config.postprocess_ignore_endpoints ? 'on' : 'off'}`;
      return [res, mode, endpoints].join(' | ');
    }

    async function fetchJson(url, opts){
      const response = await fetch(url, opts);
      const payload = await response.json().catch(() => ({}));
      if(!response.ok){
        throw new Error(payload.error || response.statusText || 'Request failed');
      }
      if(payload && payload.error){
        throw new Error(payload.error);
      }
      return payload;
    }

    function resizeCanvas(){
      const rect = canvas.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      ctx.setTransform(dpr,0,0,dpr,0,0);
      drawMap();
    }

    function worldToCanvas(x, y){
      return [state.view.tx + x * state.view.scale, state.view.ty - y * state.view.scale];
    }

    function canvasToWorld(x, y){
      return [(x - state.view.tx) / state.view.scale, -(y - state.view.ty) / state.view.scale];
    }

    function scenarioBounds(){
      if(state.scenario && state.scenario.map_payload){
        const map = state.scenario.map_payload;
        return {xmin: map.xmin, xmax: map.xmax, ymin: map.ymin, ymax: map.ymax};
      }
      const routeBounds = boundsFromRouteSet(state.scenario ? state.scenario.routes : []);
      return routeBounds ? paddedBounds(routeBounds, 0.08, 10) : {xmin: -10, xmax: 10, ymin: -10, ymax: 10};
    }

    function fitView(){
      fitBounds(scenarioBounds(), 0.92);
    }

    function focusEgoRoutes(){
      if(!state.scenario) return;
      const egoRoutes = state.scenario.routes.filter(route => routeRole(route) === 'ego');
      const selected = currentRoute();
      const focusRoutes = egoRoutes.length ? egoRoutes : (selected ? [selected] : state.scenario.routes);
      const bounds = paddedBounds(boundsFromRouteSet(focusRoutes), 0.28, 16);
      fitBounds(bounds || scenarioBounds(), 0.86);
    }

    function drawPolyline(points, color, width, dash){
      if(!points || points.length === 0) return;
      ctx.save();
      ctx.beginPath();
      if(Array.isArray(dash) && dash.length) ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const first = worldToCanvas(Number(points[0].x), Number(points[0].y));
      ctx.moveTo(first[0], first[1]);
      for(let i = 1; i < points.length; i += 1){
        const p = worldToCanvas(Number(points[i].x), Number(points[i].y));
        ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
      ctx.restore();
    }

    function drawMapBackground(){
      const map = state.scenario && state.scenario.map_payload;

      // BEV satellite-style town image (drawn first, below everything)
      if(state.layers.map && state.townImageEl && state.townImageMeta && state.townImageEl.complete){
        const m = state.townImageMeta;
        const imgW = state.townImageEl.naturalWidth;
        const imgH = state.townImageEl.naturalHeight;
        if(imgW > 0 && imgH > 0){
          // After vertical flip in Python: pixel row 0 = world_y_max, pixel row H-1 = world_y_min
          const worldYMax = m.origin_y + imgH / m.px_per_meter;
          const worldXMin = m.origin_x;
          const worldXMax = m.origin_x + imgW / m.px_per_meter;
          const worldYMin = m.origin_y;
          const [cx0, cy0] = worldToCanvas(worldXMin, worldYMax); // top-left on canvas
          const [cx1, cy1] = worldToCanvas(worldXMax, worldYMin); // bottom-right on canvas
          ctx.drawImage(state.townImageEl, cx0, cy0, cx1 - cx0, cy1 - cy0);
        }
      }

      if(!map || !Array.isArray(map.x)) return;
      const xs = map.x, ys = map.y, colors = map.colors || [];
      ctx.save();
      for(let i = 0; i < xs.length; i += 1){
        const p = worldToCanvas(Number(xs[i]), Number(ys[i]));
        ctx.fillStyle = colors[i] || '#33424f';
        ctx.fillRect(p[0], p[1], 2, 2);
      }
      ctx.restore();
      // Lane direction arrows: one every ~20m; only draw when zoomed in enough
      const sd = Number(map.sampling_distance || 4.0);
      const arrowStep = Math.max(1, Math.round(20 / sd));
      const worldSpacingPx = sd * arrowStep * state.view.scale;
      if(worldSpacingPx < 14) return; // too zoomed out, skip
      ctx.save();
      for(let i = 0; i < xs.length - 1; i += arrowStep){
        const j = i + 1;
        const wx0 = Number(xs[i]), wy0 = Number(ys[i]);
        const wx1 = Number(xs[j]), wy1 = Number(ys[j]);
        // Only draw if consecutive points are on the same lane
        if(Math.hypot(wx1 - wx0, wy1 - wy0) > sd * 1.8) continue;
        const pa = worldToCanvas(wx0, wy0);
        const pb = worldToCanvas(wx1, wy1);
        const angle = Math.atan2(pb[1] - pa[1], pb[0] - pa[0]);
        const sz = Math.max(3.5, Math.min(6.5, state.view.scale * 1.3));
        ctx.save();
        ctx.translate(pa[0], pa[1]);
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(-sz * 0.4, -sz * 0.55);
        ctx.lineTo(sz * 0.55, 0);
        ctx.lineTo(-sz * 0.4, sz * 0.55);
        ctx.strokeStyle = withAlpha(colors[i] || '#44505e', 0.65);
        ctx.lineWidth = 1.2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke();
        ctx.restore();
      }
      ctx.restore();
    }

    function drawRouteWaypoints(route, selected){
      for(let i = 0; i < route.waypoints.length; i += 1){
        const wp = route.waypoints[i];
        const p = worldToCanvas(Number(wp.x), Number(wp.y));
        const isSelectedWp = selected && i === state.selectedWaypointIndex;
        const changed = waypointChanged(route, i);
        const radius = isSelectedWp ? 8 : (selected ? 5.6 : 3.8);
        if(changed){
          ctx.beginPath();
          ctx.fillStyle = withAlpha(routeBaseColor(route), 0.18);
          ctx.arc(p[0], p[1], radius + 3.2, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.beginPath();
        ctx.fillStyle = isSelectedWp ? '#fff1df' : routeColorValue(route, selected);
        ctx.arc(p[0], p[1], radius, 0, Math.PI * 2);
        ctx.fill();
        if(selected){
          ctx.strokeStyle = isSelectedWp ? '#ffdfb8' : '#0f151d';
          ctx.lineWidth = isSelectedWp ? 2.1 : 1.5;
          ctx.stroke();
        }
      }
    }
    function drawActorFootprint(route, selected){
      const bbox = routeBbox(route);
      if(!bbox || !route || !Array.isArray(route.waypoints) || !route.waypoints.length) return;
      const focusIdx = selected && state.selectedWaypointIndex >= 0 ? Math.min(state.selectedWaypointIndex, route.waypoints.length - 1) : 0;
      const wp = route.waypoints[focusIdx];
      const length = Math.max(0.2, Number(bbox.length || 0));
      const width = Math.max(0.2, Number(bbox.width || 0));
      const yaw = Number(wp.yaw || 0) * Math.PI / 180;
      const cos = Math.cos(yaw);
      const sin = Math.sin(yaw);
      const hx = length * 0.5;
      const hy = width * 0.5;
      const corners = [
        {x: Number(wp.x) + cos * hx - sin * hy, y: Number(wp.y) + sin * hx + cos * hy},
        {x: Number(wp.x) - cos * hx - sin * hy, y: Number(wp.y) - sin * hx + cos * hy},
        {x: Number(wp.x) - cos * hx + sin * hy, y: Number(wp.y) - sin * hx - cos * hy},
        {x: Number(wp.x) + cos * hx + sin * hy, y: Number(wp.y) + sin * hx - cos * hy},
      ].map(point => worldToCanvas(point.x, point.y));
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for(let i = 1; i < corners.length; i += 1){
        ctx.lineTo(corners[i][0], corners[i][1]);
      }
      ctx.closePath();
      const base = routeBaseColor(route);
      ctx.fillStyle = withAlpha(base, routeRole(route) === 'static' ? 0.26 : (selected ? 0.22 : 0.12));
      ctx.strokeStyle = withAlpha(base, selected ? 0.95 : 0.7);
      ctx.lineWidth = selected ? 2.2 : 1.4;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function drawDirectionArrows(route, selected){
      const wps = route.waypoints;
      if(!wps || wps.length < 2) return;
      const color = routeBaseColor(route);
      const sz = selected ? 13 : 9;
      const alpha = selected ? 1.0 : 0.88;
      const outerWidth = selected ? 4.0 : 3.2;
      const innerWidth = selected ? 2.2 : 1.8;
      const minSpacingPx = 72;
      let accDist = minSpacingPx * 0.4; // first arrow appears early in the path
      let prevP = worldToCanvas(Number(wps[0].x), Number(wps[0].y));
      for(let i = 1; i < wps.length; i += 1){
        const p = worldToCanvas(Number(wps[i].x), Number(wps[i].y));
        const segLen = Math.hypot(p[0] - prevP[0], p[1] - prevP[1]);
        accDist += segLen;
        if(accDist >= minSpacingPx){
          accDist = 0;
          const mx = (prevP[0] + p[0]) * 0.5;
          const my = (prevP[1] + p[1]) * 0.5;
          const angle = Math.atan2(p[1] - prevP[1], p[0] - prevP[0]);
          ctx.save();
          ctx.translate(mx, my);
          ctx.rotate(angle);
          // Open chevron >
          ctx.beginPath();
          ctx.moveTo(-sz * 0.45, -sz * 0.58);
          ctx.lineTo(sz * 0.55, 0);
          ctx.lineTo(-sz * 0.45, sz * 0.58);
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          // Dark outline for contrast against map/path
          ctx.strokeStyle = 'rgba(0,0,0,0.65)';
          ctx.lineWidth = outerWidth;
          ctx.stroke();
          // Colored chevron on top
          ctx.strokeStyle = withAlpha(color, alpha);
          ctx.lineWidth = innerWidth;
          ctx.stroke();
          ctx.restore();
        }
        prevP = p;
      }
    }

    function drawMap(){
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.fillStyle = '#0d141b';
      ctx.fillRect(0, 0, rect.width, rect.height);
      if(state.layers.map) drawMapBackground();

      if(!state.scenario) return;

      const route = currentRoute();

      if(state.layers.actors){
        // Dynamic actors (ego/npc) first so static props render on top
        const dynamicRoutes = state.scenario.routes.filter(r => routeRole(r) !== 'static');
        const staticRoutes  = state.scenario.routes.filter(r => routeRole(r) === 'static');

        // --- dynamic: footprints, paths, arrows ---
        for(const r of dynamicRoutes) drawActorFootprint(r, r.actor_id === state.selectedActorId);
        for(const r of dynamicRoutes){
          const sel = r.actor_id === state.selectedActorId;
          drawPolyline(r.waypoints, routeColorValue(r, sel), routeStrokeWidth(r, sel), []);
        }
        for(const r of dynamicRoutes) drawDirectionArrows(r, r.actor_id === state.selectedActorId);

        // original-waypoints ghost for selected actor
        if(route){
          const original = selectedOriginal();
          if(original && original.length){
            drawPolyline(original, withAlpha(routeBaseColor(route), 0.32), Math.max(2.0, routeStrokeWidth(route, false) * 0.55), [10, 7]);
          }
        }

        // GRP for ALL actors (drawn before static props so props sit on top)
        if(state.layers.grp && state.grpPreviewCache){
          for(const r of state.scenario.routes){
            const grp = state.grpPreviewCache[r.actor_id];
            if(!grp || !grp.supported) continue;
            const base = routeBaseColor(r);
            const grpColor = darkenColor(base, 0.55);
            const grpWidth = Math.max(1.2, routeStrokeWidth(r, false) * 0.30);
            // Draw only the dense GRP trace (road-following path). The
            // aligned_waypoints are sparse key-positions shown as dots below
            // so they don't look like a second competing route path.
            drawPolyline(grp.dense_points || [], withAlpha(grpColor, 0.75), grpWidth, [4, 6]);
            // Draw aligned waypoints as small filled circles, not a polyline.
            const dotColor = darkenColor(base, 0.72);
            const dotR = Math.max(2.5, grpWidth * 0.9);
            for(const pt of (grp.aligned_waypoints || [])){
              const sc = worldToCanvas(pt.x, pt.y);
              ctx.beginPath();
              ctx.arc(sc[0], sc[1], dotR, 0, Math.PI * 2);
              ctx.fillStyle = withAlpha(dotColor, 0.85);
              ctx.fill();
            }
          }
        }

        // --- static props on top of everything so they're visible ---
        for(const r of staticRoutes) drawActorFootprint(r, r.actor_id === state.selectedActorId);
        for(const r of staticRoutes){
          const sel = r.actor_id === state.selectedActorId;
          drawPolyline(r.waypoints, routeColorValue(r, sel), routeStrokeWidth(r, sel), []);
        }
        for(const r of staticRoutes) drawDirectionArrows(r, r.actor_id === state.selectedActorId);

        // waypoint dots for all routes on top
        for(const r of state.scenario.routes) drawRouteWaypoints(r, r.actor_id === state.selectedActorId);
      }
    }

    function renderTopSummary(extra){
      if(!state.scenario){
        document.getElementById('overlayTitle').textContent = 'No Scenario Loaded';
        document.getElementById('overlayMeta').textContent = 'Select a scenario to begin review.';
        const nameEl = document.getElementById('scenarioNameDisplay');
        if(nameEl) nameEl.textContent = 'No Scenario';
        const renameBtn = document.getElementById('renameBtn');
        if(renameBtn) renameBtn.style.display = 'none';
        return;
      }
      const review = state.scenario.review || {status:'pending'};
      const dirtyText = state.dirty ? ' | unsaved edits' : '';
      const town = state.scenario.town || 'unknown town';
      const actorCount = (state.scenario.routes || []).length;
      document.getElementById('overlayTitle').textContent = state.scenario.name;
      document.getElementById('overlayMeta').textContent = `${town} | ${actorCount} routes | ${review.status}${dirtyText}${extra ? ' | ' + extra : ''}`;
      const nameEl = document.getElementById('scenarioNameDisplay');
      if(nameEl) nameEl.textContent = state.scenario.name;
      const renameBtn = document.getElementById('renameBtn');
      if(renameBtn) renameBtn.style.display = '';
    }

    function renderScenarioContext(){
      const queueSummaryEl = document.getElementById('scenarioQueueSummary');
      const readyCount = state.scenarios.filter(item => (state.grpScenarioStatus[item.id] || 'queued') === 'ready').length;
      const loadingCount = state.scenarios.filter(item => (state.grpScenarioStatus[item.id] || 'queued') === 'loading').length;
      queueSummaryEl.textContent = `${state.scenarios.length} total | ${readyCount} ready${loadingCount ? ` | ${loadingCount} loading` : ''}`;

      const metaEl = document.getElementById('scenarioContextMeta');
      const chipsEl = document.getElementById('scenarioSummaryChips');
      if(!state.scenario){
        metaEl.textContent = '';
        chipsEl.innerHTML = '<span class="metric-chip"><strong>Scenario</strong><span>Load one from the queue</span></span>';
        return;
      }
      const grpState = state.grpScenarioStatus[state.scenario.id] || deriveCurrentScenarioGrpState();
      const review = state.scenario.review || {status:'pending'};
      const scenarioIndexText = state.currentIndex >= 0 ? `${state.currentIndex + 1}/${state.scenarios.length}` : `${state.scenarios.length}`;
      metaEl.textContent = `${scenarioIndexText} | ${state.scenario.relative_path || state.scenario.id}`;
      const chips = [
        ['status', review.status || 'pending'],
        ['town', state.scenario.town || 'unknown'],
        ['routes', String((state.scenario.routes || []).length)],
        ['ego grp', grpStatusLabel(grpState, state.scenario)],
        ['time', weatherPresetLabel(state.scenario.weather || {})],
        ['warnings', String((state.scenario.warnings || []).length)],
      ];
      if(state.dirty){
        chips.push(['edits', 'unsaved']);
      }
      chipsEl.innerHTML = chips.map(([label, value]) => `<span class="metric-chip"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></span>`).join('');
    }

    function buildScenarioTree(items){
      const root = {folders:{}, scenarios:[]};
      for(const item of items){
        const relPath = String(item.relative_path || item.id || item.name || '').replaceAll('\\', '/');
        const parts = relPath.split('/').filter(Boolean);
        if(!parts.length){
          root.scenarios.push(item);
          continue;
        }
        let node = root;
        for(let i = 0; i < parts.length - 1; i += 1){
          const part = parts[i];
          if(!node.folders[part]){
            const folderPath = parts.slice(0, i + 1).join('/');
            node.folders[part] = {name: part, path: folderPath, folders:{}, scenarios:[]};
          }
          node = node.folders[part];
        }
        node.scenarios.push(item);
      }
      return root;
    }
    function annotateScenarioTree(node){
      let total = Array.isArray(node.scenarios) ? node.scenarios.length : 0;
      let approved = Array.isArray(node.scenarios) ? node.scenarios.filter(item => item.status === 'approved').length : 0;
      let containsCurrent = Array.isArray(node.scenarios) ? node.scenarios.some(item => state.scenario && item.id === state.scenario.id) : false;
      for(const child of Object.values(node.folders || {})){
        const summary = annotateScenarioTree(child);
        child.summary = summary;
        total += summary.total;
        approved += summary.approved;
        containsCurrent = containsCurrent || summary.containsCurrent;
      }
      node.summary = {total, approved, allApproved: total > 0 && approved === total, containsCurrent};
      return node.summary;
    }

    function countTreeScenarios(node){
      let total = Array.isArray(node.scenarios) ? node.scenarios.length : 0;
      for(const child of Object.values(node.folders || {})){
        total += countTreeScenarios(child);
      }
      return total;
    }

    function renderScenarioTree(node, container, depth){
      const folderNames = Object.keys(node.folders || {}).sort((a, b) => a.localeCompare(b));
      for(const folderName of folderNames){
        const child = node.folders[folderName];
        const details = document.createElement('details');
        details.className = 'folder-node';
        const childSummary = child.summary || {allApproved:false, containsCurrent:false, approved:0, total:countTreeScenarios(child)};
        details.open = !!normalizeSearch(state.scenarioFilter) || !!childSummary.containsCurrent || (depth < 1 && !childSummary.allApproved);
        const summary = document.createElement('summary');
        summary.className = 'folder-summary';
        summary.innerHTML = `
          <span class="folder-name">${escapeHtml(child.name)}</span>
          <span class="folder-meta small">${childSummary.approved || 0}/${childSummary.total || countTreeScenarios(child)} approved</span>
        `;
        details.appendChild(summary);
        const group = document.createElement('div');
        group.className = 'folder-group';
        renderScenarioTree(child, group, depth + 1);
        details.appendChild(group);
        container.appendChild(details);
      }

      const scenarios = Array.isArray(node.scenarios) ? [...node.scenarios] : [];
      scenarios.sort((a, b) => String(a.relative_path || a.id || '').localeCompare(String(b.relative_path || b.id || '')));
      for(const item of scenarios){
        const div = document.createElement('div');
        div.className = 'scenario-item' + (state.scenario && state.scenario.id === item.id ? ' active' : '');
        div.onclick = () => loadScenario(item.id, false);
        const relPath = String(item.relative_path || item.id || item.name || '');
        const town = item.town || 'unknown town';
        const grpState = state.grpScenarioStatus[item.id] || (scenarioEgoRouteCount(item) > 0 ? 'queued' : 'na');
        const egoCount = scenarioEgoRouteCount(item);
        div.innerHTML = `
          <div class="scenario-head">
            <div class="scenario-name">${escapeHtml(item.name)}</div>
            <div class="scenario-badges">
              <span class="${badgeClass(item.status)}">${escapeHtml(item.status || 'pending')}</span>
              <span class="${grpBadgeClass(grpState)}">${escapeHtml(grpStatusLabel(grpState, item))}</span>
            </div>
          </div>
          <div class="scenario-meta small">${escapeHtml(relPath)} | ${escapeHtml(town)} | ${item.route_count || 0} routes${egoCount ? ` | ${egoCount} ego` : ''}</div>
        `;
        container.appendChild(div);
      }
    }

    function renderScenarioList(){
      const root = document.getElementById('scenarioList');
      root.innerHTML = '';
      const filtered = state.scenarios.filter(item => scenarioMatchesFilter(item));
      if(!filtered.length){
        root.innerHTML = '<div class="muted small">No scenarios match the current filter.</div>';
        renderScenarioContext();
        return;
      }
      const tree = document.createElement('div');
      tree.className = 'scenario-tree';
      const treeData = buildScenarioTree(filtered);
      annotateScenarioTree(treeData);
      renderScenarioTree(treeData, tree, 0);
      root.appendChild(tree);
      renderScenarioContext();
    }

    function renderActorList(){
      const root = document.getElementById('actorList');
      root.innerHTML = '';
      document.getElementById('actorSummary').textContent = '';
      if(!state.scenario) return;
      const visibleRoutes = state.scenario.routes.filter(route => routeMatchesFilter(route));
      document.getElementById('actorSummary').textContent = `${visibleRoutes.length}/${state.scenario.routes.length} shown`;
      for(const route of visibleRoutes){
        const div = document.createElement('div');
        div.className = 'actor-item' + (route.actor_id === state.selectedActorId ? ' active' : '');
        div.onclick = () => selectActor(route.actor_id);
        const displayName = routeDisplayName(route);
        const bbox = routeBboxText(route);
        const targetSpeed = formatSpeed(routeTargetSpeedValue(route));
        const startSpeed = formatSpeed(routeStartSpeedValue(route));
        div.innerHTML = `
          <div class="actor-head">
            <div class="actor-head-main">
              <span class="color-chip" style="background:${routeBaseColor(route)}"></span>
              <div class="actor-name">${escapeHtml(displayName)}</div>
            </div>
            <span class="badge">${escapeHtml(route.kind)}</span>
          </div>
          <div class="actor-meta small">${escapeHtml(route.file)} | ${escapeHtml(routeEffectiveModel(route) || 'model unknown')}</div>
          <div class="actor-meta small">${route.waypoints.length} wp | start ${escapeHtml(routeStartTimeText(route))} | target ${escapeHtml(targetSpeed)}</div>
          <div class="actor-meta small">bbox ${escapeHtml(bbox)} | first wp speed ${escapeHtml(startSpeed)}</div>
        `;
        root.appendChild(div);
      }
      if(!visibleRoutes.length){
        root.innerHTML = '<div class="muted small">No actors match the current filter.</div>';
      }
    }

    function renderWarnings(){
      const root = document.getElementById('warningList');
      root.innerHTML = '';
      const warnings = state.scenario && Array.isArray(state.scenario.warnings) ? state.scenario.warnings : [];
      if(!warnings.length){
        root.innerHTML = '<div class="muted small">No warnings.</div>';
        return;
      }
      for(const warning of warnings){
        const div = document.createElement('div');
        div.className = 'warn-item small';
        div.textContent = warning;
        root.appendChild(div);
      }
    }

    function selectedRouteGrpState(route){
      if(!route || !route.supports_grp) return 'na';
      if(state.grpPreview && state.grpPreview.supported) return 'ready';
      if(state.grpPreviewCache && state.grpPreviewCache[route.actor_id] && state.grpPreviewCache[route.actor_id].supported) return 'ready';
      return state.dirty ? 'stale' : 'queued';
    }

    function renderRouteDetails(){
      const route = currentRoute();
      const statsEl = document.getElementById('routeStats');
      const grpBadgeEl = document.getElementById('grpStateBadge');
      document.getElementById('routeIdInput').value = route ? (route.route_attrs.id || '') : '';
      document.getElementById('routeTownInput').value = route ? (route.route_attrs.town || '') : '';
      document.getElementById('routeRoleInput').value = route ? (route.route_attrs.role || route.kind || '') : '';
      document.getElementById('routeModelInput').value = route ? (route.route_attrs.model || route.resolved_model || '') : '';
      document.getElementById('routeControlInput').value = route ? (route.route_attrs.control_mode || '') : '';
      document.getElementById('routeTargetSpeedInput').value = route ? (route.route_attrs.target_speed || route.route_attrs.speed || '') : '';
      document.getElementById('routeSnapInput').value = route ? (route.route_attrs.snap_to_road || '') : '';
      document.getElementById('routeSpawnSnapInput').value = route ? (route.route_attrs.snap_spawn_to_road || '') : '';
      const shiftEgoBtn = document.getElementById('shiftEgoWaypointsBtn');
      if(shiftEgoBtn){
        const hasWaypoints = !!(route && Array.isArray(route.waypoints) && route.waypoints.length);
        shiftEgoBtn.disabled = !(route && routeRole(route) === 'ego' && hasWaypoints);
      }
      const grpSummary = document.getElementById('grpSummary');
      if(!route){
        statsEl.innerHTML = '<div class="stat-block"><span class="stat-label">Selection</span><span class="stat-value">No actor selected</span></div>';
        grpBadgeEl.innerHTML = '';
        grpSummary.textContent = 'No actor selected.';
        return;
      }
      const statRows = [
        ['model', routeEffectiveModel(route) || 'unknown'],
        ['bbox', routeBboxText(route)],
        ['target speed', formatSpeed(routeTargetSpeedValue(route))],
        ['first wp speed', formatSpeed(routeStartSpeedValue(route))],
        ['start time', routeStartTimeText(route)],
        ['waypoints', String((route.waypoints || []).length)],
      ];
      statsEl.innerHTML = statRows.map(([label, value]) => `
        <div class="stat-block">
          <span class="stat-label">${escapeHtml(label)}</span>
          <span class="stat-value">${escapeHtml(value)}</span>
        </div>
      `).join('');
      const grpState = selectedRouteGrpState(route);
      grpBadgeEl.innerHTML = route.supports_grp ? `<span class="${grpBadgeClass(grpState)}">${escapeHtml(grpStatusLabel(grpState, {ego_route_count:1}))}</span>` : '<span class="badge grp na">no grp</span>';
      if(!route.supports_grp){
        grpSummary.textContent = 'GRP preview is only available for ego/NPC driving routes.';
        return;
      }
      if(state.grpPreview && state.grpPreview.supported){
        const metrics = state.grpPreview.metrics || {};
        const ratio = metrics.detour_ratio != null ? `ratio ${Number(metrics.detour_ratio).toFixed(3)}` : 'ratio n/a';
        const configSummary = formatGrpConfig(state.grpPreview.config || state.grpConfig);
        const cacheText = state.grpPreview.cached ? 'cache hit' : 'fresh';
        grpSummary.textContent = `GRP ${cacheText} | trace ${metrics.trace_points || 0} pts | current ${metrics.current_length_m || 0}m | grp ${metrics.trace_length_m || 0}m | ${ratio}${configSummary ? ' | ' + configSummary : ''}`;
        return;
      }
      const configSummary = formatGrpConfig(state.grpConfig);
      grpSummary.textContent = configSummary ? `GRP preview not loaded. ${configSummary}` : 'GRP preview not loaded.';
    }

    function routeAttrKeys(route){
      if(!route || !route.route_attrs) return [];
      const hidden = new Set(['id', 'town', 'role', 'model', 'control_mode', 'target_speed', 'speed', 'snap_to_road', 'snap_spawn_to_road']);
      return Object.keys(route.route_attrs).filter(key => !hidden.has(key)).sort((a, b) => a.localeCompare(b));
    }

    function renderRouteAttrTable(){
      const table = document.getElementById('routeAttrTable');
      const route = currentRoute();
      if(!route){
        table.innerHTML = '<tr><td class="muted">No actor selected.</td></tr>';
        return;
      }
      const keys = routeAttrKeys(route);
      if(!keys.length){
        table.innerHTML = '<tr><td class="muted">No extra route properties. Add one when you need a non-standard field.</td></tr>';
        return;
      }
      table.innerHTML = `
        <thead>
          <tr>
            <th>Key</th>
            <th>Value</th>
            <th></th>
          </tr>
        </thead>
      `;
      const tbody = document.createElement('tbody');
      for(const key of keys){
        const tr = document.createElement('tr');
        const keyTd = document.createElement('td');
        keyTd.innerHTML = `<span class="small" style="font-family:'IBM Plex Mono',monospace">${escapeHtml(key)}</span>`;
        const valueTd = document.createElement('td');
        const input = document.createElement('input');
        input.value = route.route_attrs[key] == null ? '' : String(route.route_attrs[key]);
        input.onchange = () => updateRouteAttr(key, input.value);
        valueTd.appendChild(input);
        const buttonTd = document.createElement('td');
        const button = document.createElement('button');
        button.className = 'ghost small';
        button.textContent = 'Remove';
        button.onclick = (event) => {
          event.preventDefault();
          updateRouteAttr(key, '');
        };
        buttonTd.appendChild(button);
        tr.appendChild(keyTd);
        tr.appendChild(valueTd);
        tr.appendChild(buttonTd);
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
    }

    function waypointChanged(route, idx){
      if(!route || !Array.isArray(route.original_waypoints)) return false;
      const original = route.original_waypoints[idx];
      const current = route.waypoints[idx];
      if(!original || !current) return true;
      const keys = ['x','y','z','yaw','pitch','roll','time','speed'];
      for(const key of keys){
        const currentValue = current[key];
        const originalValue = original[key];
        if(currentValue == null && originalValue == null) continue;
        if(currentValue == null || originalValue == null) return true;
        if(Math.abs(Number(currentValue) - Number(originalValue)) > 1e-4) return true;
      }
      return false;
    }

    function renderWaypointInspector(){
      const root = document.getElementById('waypointInspector');
      const route = currentRoute();
      const summaryEl = document.getElementById('waypointSummary');
      if(!route){
        summaryEl.textContent = '';
        root.textContent = 'Select a waypoint to edit it directly, duplicate it, or delete it.';
        return;
      }
      const changedCount = route.waypoints.filter((_wp, idx) => waypointChanged(route, idx)).length;
      summaryEl.textContent = `${route.waypoints.length} total | ${changedCount} changed`;
      const idx = state.selectedWaypointIndex;
      if(idx < 0 || idx >= route.waypoints.length){
        root.innerHTML = `
          <div>Select a waypoint row or drag a waypoint on the map.</div>
          <div class="waypoint-tools">
            <button class="ghost small" onclick="appendWaypoint()">Append Waypoint</button>
            <button class="ghost small" onclick="recomputeSelectedYaws()">Recompute Yaw</button>
          </div>
        `;
        return;
      }
      const wp = route.waypoints[idx];
      const changed = waypointChanged(route, idx);
      root.innerHTML = `
        <div><strong>Waypoint ${idx}</strong>${changed ? ' <span class="row-delta">changed</span>' : ''}</div>
        <div class="waypoint-tools">
          <button class="ghost small" onclick="selectWaypoint(${Math.max(0, idx - 1)})" ${idx <= 0 ? 'disabled' : ''}>Prev</button>
          <button class="ghost small" onclick="selectWaypoint(${Math.min(route.waypoints.length - 1, idx + 1)})" ${idx >= route.waypoints.length - 1 ? 'disabled' : ''}>Next</button>
          <button class="ghost small" onclick="insertWaypointBefore(${idx})">Insert Before</button>
          <button class="ghost small" onclick="insertWaypointAfter(${idx})">Insert After</button>
          <button class="ghost small" onclick="duplicateWaypoint(${idx})">Duplicate</button>
          <button class="ghost small" onclick="deleteWaypoint(${idx})" ${route.waypoints.length <= 1 ? 'disabled' : ''}>Delete</button>
          <button class="ghost small" onclick="showNudgeMenu(canvas.getBoundingClientRect().left + 80, canvas.getBoundingClientRect().top + 80, '${escapeHtml(route.actor_id)}', ${idx})">Nudge</button>
          <button class="ghost small" onclick="nudgeWaypoint(0,0,true)">Reset Pos</button>
        </div>
        <div class="waypoint-fields">
          <div class="field"><label>x</label><input type="number" step="0.01" value="${Number(wp.x).toFixed(3)}" onchange="updateWaypointField(${idx},'x',this.value)" /></div>
          <div class="field"><label>y</label><input type="number" step="0.01" value="${Number(wp.y).toFixed(3)}" onchange="updateWaypointField(${idx},'y',this.value)" /></div>
          <div class="field"><label>z</label><input type="number" step="0.01" value="${Number(wp.z).toFixed(3)}" onchange="updateWaypointField(${idx},'z',this.value)" /></div>
          <div class="field"><label>yaw</label><input type="number" step="0.01" value="${Number(wp.yaw).toFixed(3)}" onchange="updateWaypointField(${idx},'yaw',this.value)" /></div>
          <div class="field"><label>time</label><input type="number" step="0.01" value="${wp.time == null ? '' : Number(wp.time).toFixed(3)}" onchange="updateWaypointField(${idx},'time',this.value)" /></div>
          <div class="field"><label>speed</label><input type="number" step="0.01" value="${wp.speed == null ? '' : Number(wp.speed).toFixed(3)}" onchange="updateWaypointField(${idx},'speed',this.value)" /></div>
          <div class="field"><label>pitch</label><input type="number" step="0.01" value="${wp.pitch == null ? '' : Number(wp.pitch).toFixed(3)}" onchange="updateWaypointField(${idx},'pitch',this.value)" /></div>
          <div class="field"><label>roll</label><input type="number" step="0.01" value="${wp.roll == null ? '' : Number(wp.roll).toFixed(3)}" onchange="updateWaypointField(${idx},'roll',this.value)" /></div>
        </div>
      `;
    }

    function renderWaypointTable(){
      const table = document.getElementById('waypointTable');
      const route = currentRoute();
      if(!route){
        table.innerHTML = '<tr><td class="muted">No actor selected.</td></tr>';
        return;
      }
      let html = `
        <thead>
          <tr>
            <th>#</th>
            <th>Δ</th>
            <th>x</th>
            <th>y</th>
            <th>z</th>
            <th>yaw</th>
            <th>pitch</th>
            <th>roll</th>
            <th>time</th>
            <th>speed</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
      `;
      route.waypoints.forEach((wp, idx) => {
        const changed = waypointChanged(route, idx);
        const rowClass = ['waypoint-row', idx === state.selectedWaypointIndex ? 'selected' : '', changed ? 'changed' : ''].filter(Boolean).join(' ');
        html += `
          <tr class="${rowClass}" onclick="selectWaypoint(${idx})">
            <td>${idx}</td>
            <td>${changed ? '<span class="row-delta">edit</span>' : ''}</td>
            <td><input type="number" step="0.01" value="${Number(wp.x).toFixed(3)}" onchange="updateWaypointField(${idx},'x',this.value)" /></td>
            <td><input type="number" step="0.01" value="${Number(wp.y).toFixed(3)}" onchange="updateWaypointField(${idx},'y',this.value)" /></td>
            <td><input type="number" step="0.01" value="${Number(wp.z).toFixed(3)}" onchange="updateWaypointField(${idx},'z',this.value)" /></td>
            <td><input type="number" step="0.01" value="${Number(wp.yaw).toFixed(3)}" onchange="updateWaypointField(${idx},'yaw',this.value)" /></td>
            <td><input type="number" step="0.01" value="${wp.pitch == null ? '' : Number(wp.pitch).toFixed(3)}" onchange="updateWaypointField(${idx},'pitch',this.value)" /></td>
            <td><input type="number" step="0.01" value="${wp.roll == null ? '' : Number(wp.roll).toFixed(3)}" onchange="updateWaypointField(${idx},'roll',this.value)" /></td>
            <td><input type="number" step="0.01" value="${wp.time == null ? '' : Number(wp.time).toFixed(3)}" onchange="updateWaypointField(${idx},'time',this.value)" /></td>
            <td><input type="number" step="0.01" value="${wp.speed == null ? '' : Number(wp.speed).toFixed(3)}" onchange="updateWaypointField(${idx},'speed',this.value)" /></td>
            <td>
              <div class="button-row">
                <button class="ghost small" onclick="event.stopPropagation(); insertWaypointBefore(${idx})">Before</button>
                <button class="ghost small" onclick="event.stopPropagation(); insertWaypointAfter(${idx})">After</button>
                <button class="ghost small" onclick="event.stopPropagation(); duplicateWaypoint(${idx})">Copy</button>
                <button class="ghost small" onclick="event.stopPropagation(); deleteWaypoint(${idx})" ${route.waypoints.length <= 1 ? 'disabled' : ''}>Del</button>
              </div>
            </td>
          </tr>
        `;
      });
      html += '</tbody>';
      table.innerHTML = html;
    }

    function renderReview(){
      // Review note textarea removed; review state tracked via approve/reject buttons
    }

    function renderAll(reason){
      renderScenarioList();
      renderActorList();
      renderScenarioWeather();
      renderRouteDetails();
      renderRouteAttrTable();
      renderWaypointInspector();
      renderWaypointTable();
      renderWarnings();
      renderReview();
      renderTopSummary(reason);
      drawMap();
    }

    async function loadIndex(){
      state.scenarios = await fetchJson('/api/scenarios');
      initScenarioGrpStates();
      renderScenarioList();
      if(state.scenarios.length){
        await loadScenario(state.scenarios[0].id, true);
      }else{
        setMessage('No scenario directories found.');
      }
    }

    async function loadScenario(id, force){
      if(state.dirty && !force){
        const proceed = window.confirm('Discard unsaved edits and load another scenario?');
        if(!proceed) return;
      }
      setMessage('Loading scenario…');
      markScenarioGrpState(id, 'loading');
      renderScenarioList();
      try{
        // Phase 1: fast load – scenario data only (no CARLA blocking)
        const payload = await fetchJson('/api/scenario?id=' + encodeURIComponent(id));
        state.scenario = payload;
        state.currentIndex = state.scenarios.findIndex(item => item.id === id);
        const firstEgo = payload.routes.find(route => routeRole(route) === 'ego');
        state.selectedActorId = firstEgo ? firstEgo.actor_id : (payload.routes.length ? payload.routes[0].actor_id : null);
        state.selectedWaypointIndex = -1;
        state.grpPreviewCache = {};
        state.historyUndo = [];
        state.historyRedo = [];
        syncSelectedGrpPreviewFromCache();
        state.dirty = false;
        hideNudgeMenu();
        markScenarioGrpState(id, 'queued');
        focusEgoRoutes();
        renderAll();
        setMessage(`Loaded ${payload.name} | fetching map & GRP…`);

        // Phase 2: async CARLA data (map background + GRP warmup)
        try{
          const bg = await fetchJson('/api/scenario_bg?id=' + encodeURIComponent(id));
          if(state.scenario && state.scenario.id === id){
            state.scenario.map_payload = bg.map_payload || null;
            state.grpPreviewCache = clone(bg.grp_previews || {});
            syncSelectedGrpPreviewFromCache();
            markScenarioGrpState(id, deriveCurrentScenarioGrpState());
            updateCarlaStatus(bg.carla_status || null);
            // Load BEV town image if available and not already loaded for this town
            const mp = state.scenario.map_payload;
            if(mp && mp.town && mp.xmin_raw != null && mp.ymin_raw != null){
              const town = mp.town;
              const BEV_MARGIN = 300;
              const bev_origin_x = mp.xmin_raw - BEV_MARGIN;
              const bev_origin_y = mp.ymin_raw - BEV_MARGIN;
              const BEV_PX_PER_METER = 5;
              const needsLoad = !state.townImageMeta || state.townImageMeta.town !== town;
              if(needsLoad){
                state.townImageEl = null;
                state.townImageMeta = null;
                const img = new Image();
                img.onload = () => {
                  if(state.scenario && state.scenario.id === id){
                    state.townImageEl = img;
                    state.townImageMeta = {town, origin_x: bev_origin_x, origin_y: bev_origin_y, px_per_meter: BEV_PX_PER_METER};
                    drawMap();
                  }
                };
                img.onerror = () => { /* BEV not available for this town – no-op */ };
                img.src = '/api/town_image?town=' + encodeURIComponent(town);
              }
            }
            // Do NOT re-fit: Phase 1 already focused on the scenario; re-fitting
            // here would zoom out to full-map level once map_payload arrives.
            renderAll();
            const warmed = bg.grp_warmed_count ? ` | warmed ${bg.grp_warmed_count} ego GRP` : '';
            setMessage(`Loaded ${payload.name}${warmed}`);
          }
        }catch(bgErr){
          if(state.scenario && state.scenario.id === id){
            markScenarioGrpState(id, 'unavailable');
            renderScenarioList();
            setMessage(`Loaded ${payload.name} | CARLA unavailable: ${bgErr.message}`);
          }
        }
      }catch(err){
        markScenarioGrpState(id, 'unavailable');
        renderScenarioList();
        setMessage('Loading failed: ' + err.message);
        throw err;
      }
    }

    async function reloadCurrent(force){
      if(!state.scenario) return;
      await loadScenario(state.scenario.id, !!force);
    }

    function currentViewCenterWorld(){
      const rect = canvas.getBoundingClientRect();
      return canvasToWorld(rect.width * 0.5, rect.height * 0.5);
    }
    function nextActorOrdinal(kind){
      if(!state.scenario) return 1;
      const role = String(kind || '').toLowerCase();
      const matches = state.scenario.routes.filter(route => routeRole(route) === role);
      return matches.length + 1;
    }
    function uniqueRouteFile(kind, town){
      const townSlug = String(town || 'town').trim().toLowerCase();
      const ordinal = nextActorOrdinal(kind);
      const role = String(kind || '').toLowerCase();
      const candidateBase = role === 'ego' ? `${townSlug}_ego_vehicle_${ordinal}.xml` : `actors/${role}/${townSlug}_${role}_${ordinal}.xml`;
      const existing = new Set((state.scenario && state.scenario.routes || []).map(route => route.file));
      if(!existing.has(candidateBase)) return candidateBase;
      let index = ordinal + 1;
      while(true){
        const nextCandidate = role === 'ego' ? `${townSlug}_ego_vehicle_${index}.xml` : `actors/${role}/${townSlug}_${role}_${index}.xml`;
        if(!existing.has(nextCandidate)) return nextCandidate;
        index += 1;
      }
    }
    function makeDefaultRoute(kind){
      const role = String(kind || 'npc').toLowerCase();
      const town = (state.scenario && state.scenario.town) || (currentRoute() && currentRoute().route_attrs && currentRoute().route_attrs.town) || 'Town01';
      const center = currentRoute() && currentRoute().waypoints && currentRoute().waypoints[0] ? currentRoute().waypoints[0] : null;
      const worldCenter = center ? [Number(center.x), Number(center.y)] : currentViewCenterWorld();
      const yaw = center ? Number(center.yaw || 0) : 0;
      const forward = {x: Math.cos(yaw * Math.PI / 180), y: Math.sin(yaw * Math.PI / 180)};
      const model = DEFAULT_ROUTE_MODEL[role] || DEFAULT_ROUTE_MODEL.npc;
      const bbox = clone(DEFAULT_ROUTE_BBOX[model] || null);
      const file = uniqueRouteFile(role, town);
      const ordinal = nextActorOrdinal(role);
      const baseName = {ego:'Ego', npc:'NPC', pedestrian:'Pedestrian', bicycle:'Bicycle', static:'Prop'}[role] || 'Actor';
      const defaultTargetSpeed = {ego:'0.0', npc:'4.0', pedestrian:'1.5', bicycle:'4.0'}[role];
      const waypoints = role === 'static' ? [
        {x: Number(worldCenter[0]), y: Number(worldCenter[1]), z: 0, yaw, pitch: null, roll: null, time: null, speed: 0, extras: {}},
      ] : [
        {x: Number(worldCenter[0]), y: Number(worldCenter[1]), z: 0, yaw, pitch: null, roll: null, time: 0, speed: safeNumber(defaultTargetSpeed), extras: {}},
        {x: Number(worldCenter[0]) + forward.x * 8, y: Number(worldCenter[1]) + forward.y * 8, z: 0, yaw, pitch: null, roll: null, time: 1, speed: safeNumber(defaultTargetSpeed), extras: {}},
      ];
      return {
        actor_id: file,
        file,
        kind: role,
        name: `${baseName} ${ordinal}`,
        route_attrs: {
          id: `${town.toLowerCase()}_${role}_${ordinal}`,
          town,
          role,
          model,
          ...(defaultTargetSpeed ? {target_speed: defaultTargetSpeed} : {}),
        },
        resolved_model: model,
        bbox,
        waypoints: clone(waypoints),
        original_waypoints: clone(waypoints),
        manifest_entry: null,
        supports_grp: role === 'ego' || role === 'npc',
      };
    }
    function addActor(kind){
      if(!state.scenario) return;
      pushHistory(`add ${kind}`);
      const route = makeDefaultRoute(kind);
      state.scenario.routes.push(route);
      state.scenario.routes.sort((a, b) => {
        const roleOrder = {ego:0, npc:1, pedestrian:2, bicycle:3, static:4};
        const diff = (roleOrder[routeRole(a)] ?? 99) - (roleOrder[routeRole(b)] ?? 99);
        if(diff !== 0) return diff;
        return String(a.file || '').localeCompare(String(b.file || ''));
      });
      state.selectedActorId = route.actor_id;
      state.selectedWaypointIndex = 0;
      invalidateSelectedGrpCache();
      markDirty(`Added ${kind}`);
      renderAll();
    }
    function deleteSelectedActor(){
      if(!state.scenario) return;
      const route = currentRoute();
      if(!route) return;
      const confirmDelete = window.confirm(`Delete actor ${routeDisplayName(route)}?`);
      if(!confirmDelete) return;
      pushHistory('delete actor');
      state.scenario.routes = state.scenario.routes.filter(item => item.actor_id !== route.actor_id);
      const nextRoute = state.scenario.routes[0] || null;
      state.selectedActorId = nextRoute ? nextRoute.actor_id : null;
      state.selectedWaypointIndex = -1;
      invalidateSelectedGrpCache();
      markDirty('Deleted actor');
      renderAll();
    }

    function selectActor(actorId){
      state.selectedActorId = actorId;
      state.selectedWaypointIndex = -1;
      hideNudgeMenu();
      syncSelectedGrpPreviewFromCache();
      renderAll();
    }

    function selectWaypoint(index){
      state.selectedWaypointIndex = index;
      hideNudgeMenu();
      drawMap();
      renderWaypointInspector();
      renderWaypointTable();
    }

    function updateRouteAttr(key, value){
      const route = currentRoute();
      if(!route) return;
      pushHistory('route field');
      if(value === ''){
        delete route.route_attrs[key];
      }else{
        route.route_attrs[key] = value;
      }
      if(key === 'town'){
        state.scenario.town = value || state.scenario.town;
      }
      if(key === 'model'){
        route.resolved_model = value || route.resolved_model || '';
        route.bbox = defaultBBoxForRoute(route);
      }
      invalidateSelectedGrpCache();
      if(key === 'role'){
        const nextRole = String(value || route.kind || '').toLowerCase();
        route.kind = nextRole || route.kind;
        route.supports_grp = nextRole === 'ego' || nextRole === 'npc';
      }
      markDirty('Route updated');
      renderRouteDetails();
      renderRouteAttrTable();
      renderScenarioContext();
      renderActorList();
      drawMap();
    }

    function updateTargetSpeed(value){
      const route = currentRoute();
      if(!route) return;
      pushHistory('target speed');
      const hasTarget = route.route_attrs && Object.prototype.hasOwnProperty.call(route.route_attrs, 'target_speed');
      const hasLegacy = route.route_attrs && Object.prototype.hasOwnProperty.call(route.route_attrs, 'speed');
      if(value === ''){
        if(hasTarget) delete route.route_attrs.target_speed;
        if(hasLegacy) delete route.route_attrs.speed;
      }else if(hasTarget){
        route.route_attrs.target_speed = value;
        if(hasLegacy) route.route_attrs.speed = value;
      }else if(hasLegacy){
        route.route_attrs.speed = value;
      }else{
        route.route_attrs.target_speed = value;
      }
      invalidateSelectedGrpCache();
      markDirty('Route updated');
      renderRouteDetails();
      renderRouteAttrTable();
      renderActorList();
      drawMap();
    }

    function updateWaypointField(index, key, rawValue){
      const route = currentRoute();
      if(!route) return;
      const wp = route.waypoints[index];
      if(!wp) return;
      pushHistory('waypoint field');
      if(rawValue === ''){
        wp[key] = null;
      }else{
        const num = Number(rawValue);
        if(Number.isFinite(num)) wp[key] = num;
      }
      invalidateSelectedGrpCache();
      markDirty('Waypoint updated');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
      renderActorList();
      drawMap();
    }

    function recomputeRouteYaws(route){
      if(!route || !Array.isArray(route.waypoints) || route.waypoints.length < 2) return;
      for(let i = 0; i < route.waypoints.length; i += 1){
        const current = route.waypoints[i];
        let dx = 0, dy = 0;
        if(i < route.waypoints.length - 1){
          dx = Number(route.waypoints[i + 1].x) - Number(current.x);
          dy = Number(route.waypoints[i + 1].y) - Number(current.y);
        }else if(i > 0){
          dx = Number(current.x) - Number(route.waypoints[i - 1].x);
          dy = Number(current.y) - Number(route.waypoints[i - 1].y);
        }
        if(Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6){
          current.yaw = Math.atan2(dy, dx) * 180 / Math.PI;
        }
      }
    }

    function recomputeSelectedYaws(){
      const route = currentRoute();
      if(!route) return;
      pushHistory('recompute yaw');
      recomputeRouteYaws(route);
      invalidateSelectedGrpCache();
      markDirty('Yaw recomputed');
      renderWaypointTable();
      drawMap();
    }

    function insertWaypointAfter(index, skipHistory=false){
      const route = currentRoute();
      if(!route) return;
      if(!skipHistory) pushHistory('insert waypoint');
      const base = route.waypoints[index];
      let next = route.waypoints[index + 1];
      let insert;
      if(next){
        insert = clone(base);
        insert.x = (Number(base.x) + Number(next.x)) * 0.5;
        insert.y = (Number(base.y) + Number(next.y)) * 0.5;
        insert.z = (Number(base.z) + Number(next.z)) * 0.5;
        insert.yaw = (Number(base.yaw) + Number(next.yaw)) * 0.5;
        if(base.time != null && next.time != null) insert.time = (Number(base.time) + Number(next.time)) * 0.5;
      }else{
        insert = clone(base);
        const yawRad = Number(base.yaw || 0) * Math.PI / 180;
        insert.x = Number(base.x) + Math.cos(yawRad) * 4.0;
        insert.y = Number(base.y) + Math.sin(yawRad) * 4.0;
        if(base.time != null) insert.time = Number(base.time) + 0.5;
      }
      route.waypoints.splice(index + 1, 0, insert);
      state.selectedWaypointIndex = index + 1;
      invalidateSelectedGrpCache();
      markDirty('Waypoint inserted');
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function insertWaypointBefore(index){
      const route = currentRoute();
      if(!route) return;
      pushHistory('insert waypoint');
      if(index <= 0){
        const first = clone(route.waypoints[0]);
        const yawRad = Number(first.yaw || 0) * Math.PI / 180;
        first.x = Number(first.x) - Math.cos(yawRad) * 4.0;
        first.y = Number(first.y) - Math.sin(yawRad) * 4.0;
        if(first.time != null) first.time = Number(first.time) - 0.5;
        route.waypoints.splice(0, 0, first);
        state.selectedWaypointIndex = 0;
      }else{
        insertWaypointAfter(index - 1, true);
        return;
      }
      invalidateSelectedGrpCache();
      markDirty('Waypoint inserted');
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function duplicateWaypoint(index){
      const route = currentRoute();
      if(!route) return;
      const source = route.waypoints[index];
      if(!source) return;
      pushHistory('duplicate waypoint');
      const copyWp = clone(source);
      route.waypoints.splice(index + 1, 0, copyWp);
      state.selectedWaypointIndex = index + 1;
      invalidateSelectedGrpCache();
      markDirty('Waypoint duplicated');
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function appendWaypoint(){
      const route = currentRoute();
      if(!route || !route.waypoints.length) return;
      insertWaypointAfter(route.waypoints.length - 1);
    }

    function appendWaypointAt(x, y){
      const route = currentRoute();
      if(!route) return;
      pushHistory('draw waypoint');
      let waypoint;
      if(route.waypoints.length){
        const last = route.waypoints[route.waypoints.length - 1];
        waypoint = clone(last);
        waypoint.x = Number(x);
        waypoint.y = Number(y);
        if(last.time != null){
          waypoint.time = Number(last.time) + 0.5;
        }
      }else{
        waypoint = {
          x: Number(x),
          y: Number(y),
          z: 0,
          yaw: 0,
          pitch: null,
          roll: null,
          time: 0,
          speed: routeTargetSpeedValue(route),
          extras: {},
        };
      }
      route.waypoints.push(waypoint);
      recomputeRouteYaws(route);
      state.selectedWaypointIndex = route.waypoints.length - 1;
      invalidateSelectedGrpCache();
      markDirty('Waypoint appended');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function deleteWaypoint(index){
      const route = currentRoute();
      if(!route || route.waypoints.length <= 1) return;
      pushHistory('delete waypoint');
      route.waypoints.splice(index, 1);
      state.selectedWaypointIndex = Math.min(index, route.waypoints.length - 1);
      invalidateSelectedGrpCache();
      markDirty('Waypoint deleted');
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function shiftSelectedTimes(){
      const route = currentRoute();
      if(!route) return;
      const value = window.prompt('Shift all waypoint times by seconds:', '0.5');
      if(value == null) return;
      const delta = Number(value);
      if(!Number.isFinite(delta)) return;
      pushHistory('shift waypoint times');
      for(const wp of route.waypoints){
        if(wp.time == null) wp.time = 0;
        wp.time = Number(wp.time) + delta;
      }
      invalidateSelectedGrpCache();
      markDirty('Waypoint timing shifted');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
    }

    function shiftSelectedEgoWaypoints(){
      const route = currentRoute();
      if(!route) return;
      if(routeRole(route) !== 'ego'){
        window.alert('Global XY shift is only available for ego routes.');
        return;
      }
      if(!Array.isArray(route.waypoints) || !route.waypoints.length){
        window.alert('Selected ego route has no waypoints.');
        return;
      }
      const value = window.prompt('Shift all selected ego waypoints by Δx,Δy (meters):', '0,0');
      if(value == null) return;
      const pieces = String(value).split(',');
      if(pieces.length !== 2){
        window.alert('Enter shift as two numbers: dx,dy');
        return;
      }
      const dx = Number(String(pieces[0]).trim());
      const dy = Number(String(pieces[1]).trim());
      if(!Number.isFinite(dx) || !Number.isFinite(dy)){
        window.alert('Enter shift as two numbers: dx,dy');
        return;
      }
      if(Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return;
      pushHistory('shift ego waypoints');
      for(const waypoint of route.waypoints){
        waypoint.x = Number(waypoint.x) + dx;
        waypoint.y = Number(waypoint.y) + dy;
      }
      invalidateSelectedGrpCache();
      markDirty('Ego waypoints shifted');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
      renderActorList();
      drawMap();
    }

    function normalizeSelectedTimes(){
      const route = currentRoute();
      if(!route || !route.waypoints.length) return;
      const value = window.prompt('Set first waypoint time to:', '0');
      if(value == null) return;
      const target = Number(value);
      if(!Number.isFinite(target)) return;
      pushHistory('normalize waypoint times');
      const first = route.waypoints[0].time == null ? 0 : Number(route.waypoints[0].time);
      const delta = target - first;
      for(const wp of route.waypoints){
        if(wp.time == null) wp.time = 0;
        wp.time = Number(wp.time) + delta;
      }
      invalidateSelectedGrpCache();
      markDirty('Waypoint timing normalized');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
    }

    function addRouteProp(){
      const route = currentRoute();
      if(!route) return;
      const key = window.prompt('New route property key:', 'trigger_condition');
      if(key == null) return;
      const normalizedKey = String(key).trim();
      if(!normalizedKey) return;
      const value = window.prompt(`Value for ${normalizedKey}:`, route.route_attrs[normalizedKey] || '');
      if(value == null) return;
      pushHistory('add route prop');
      route.route_attrs[normalizedKey] = value;
      invalidateSelectedGrpCache();
      markDirty('Route property added');
      renderRouteDetails();
      renderRouteAttrTable();
      renderScenarioContext();
      renderActorList();
      drawMap();
    }

    async function refreshGrp(){
      const route = currentRoute();
      if(!route) return;
      if(!route.supports_grp){
        setMessage('GRP preview is only supported for ego/NPC routes.');
        return;
      }
      markScenarioGrpState(state.scenario && state.scenario.id, 'loading');
      renderScenarioList();
      setMessage('Running GRP preview…');
      try{
        state.grpPreview = await fetchJson('/api/grp_preview', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            town: route.route_attrs.town || state.scenario.town,
            role: route.route_attrs.role || route.kind,
            waypoints: route.waypoints,
          }),
        });
        state.grpPreviewCache[route.actor_id] = clone(state.grpPreview);
        refreshCurrentScenarioGrpState();
        renderRouteDetails();
        drawMap();
        setMessage(state.grpPreview.cached ? 'GRP preview loaded from cache.' : 'GRP preview ready.');
      }catch(err){
        state.grpPreview = null;
        markScenarioGrpState(state.scenario && state.scenario.id, 'unavailable');
        renderScenarioList();
        renderRouteDetails();
        setMessage('GRP preview failed: ' + err.message);
      }
    }

    function adoptGrpWaypoints(){
      const route = currentRoute();
      if(!route || !state.grpPreview || !state.grpPreview.supported || !Array.isArray(state.grpPreview.aligned_waypoints) || !state.grpPreview.aligned_waypoints.length){
        return;
      }
      pushHistory('adopt grp');
      route.waypoints = clone(state.grpPreview.aligned_waypoints);
      state.selectedWaypointIndex = -1;
      invalidateSelectedGrpCache();
      markDirty('Applied GRP alignment');
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
      setMessage('Applied GRP alignment. Keep tweaking the preprocessed route and run GRP again whenever needed.');
    }

    async function saveScenario(){
      if(!state.scenario) return false;
      setMessage('Saving scenario in place…');
      try{
        const payload = await fetchJson('/api/save', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            id: state.scenario.id,
            weather: state.scenario.weather || {},
            routes: state.scenario.routes,
            review: {note: (state.scenario.review && state.scenario.review.note) || ''},
          }),
        });
        state.scenario = payload;
        state.dirty = false;
        state.grpPreviewCache = clone(payload.grp_previews || {});
        state.historyUndo = [];
        state.historyRedo = [];
        if(!state.scenario.routes.some(route => route.actor_id === state.selectedActorId)){
          const firstEgo = state.scenario.routes.find(route => routeRole(route) === 'ego');
          state.selectedActorId = firstEgo ? firstEgo.actor_id : (state.scenario.routes.length ? state.scenario.routes[0].actor_id : null);
          state.selectedWaypointIndex = -1;
        }
        syncSelectedGrpPreviewFromCache();
        state.scenarios = await fetchJson('/api/scenarios');
        initScenarioGrpStates();
        markScenarioGrpState(payload.id, deriveCurrentScenarioGrpState());
        updateCarlaStatus(payload.carla_status || null);
        renderAll();
        setMessage('Scenario saved in place.');
        return true;
      }catch(err){
        markScenarioGrpState(state.scenario && state.scenario.id, 'unavailable');
        renderScenarioList();
        setMessage('Save failed: ' + err.message);
        return false;
      }
    }

    async function saveAndNext(){
      if(!state.scenario) return;
      const currentId = state.scenario.id;
      const ok = await saveScenario();
      if(!ok) return;
      const idx = state.scenarios.findIndex(item => item.id === currentId);
      if(idx >= 0 && idx < state.scenarios.length - 1){
        await loadScenario(state.scenarios[idx + 1].id, true);
      }
    }

    async function saveReviewNote(){
      if(!state.scenario) return;
      const note = (state.scenario.review && state.scenario.review.note) || '';
      try{
        const payload = await fetchJson('/api/review', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id: state.scenario.id, note}),
        });
        state.scenarios = payload.scenarios || state.scenarios;
        initScenarioGrpStates();
        renderScenarioList();
        renderScenarioContext();
        renderTopSummary();
        setMessage('Review note saved.');
      }catch(err){
        setMessage('Saving note failed: ' + err.message);
      }
    }

    async function setReviewStatus(status){
      if(!state.scenario) return;
      try{
        const payload = await fetchJson('/api/review', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id: state.scenario.id, status, note: (state.scenario.review && state.scenario.review.note) || ''}),
        });
        state.scenarios = payload.scenarios || state.scenarios;
        initScenarioGrpStates();
        state.scenario.review = payload.review || {status};
        renderScenarioList();
        renderScenarioContext();
        renderTopSummary();
        setMessage('Review state updated to ' + status + '.');
      }catch(err){
        setMessage('Updating review failed: ' + err.message);
      }
    }

    function goPrev(){
      if(state.currentIndex > 0){
        loadScenario(state.scenarios[state.currentIndex - 1].id, false);
      }
    }

    function goNext(){
      if(state.currentIndex >= 0 && state.currentIndex < state.scenarios.length - 1){
        loadScenario(state.scenarios[state.currentIndex + 1].id, false);
      }
    }

    function updateCarlaStatus(status){
      state.carla = status;
      if(status && status.grp_config){
        state.grpConfig = status.grp_config;
      }
      const dot = document.getElementById('carlaDot');
      const text = document.getElementById('carlaText');
      if(!status){
        dot.className = 'status-dot';
        text.textContent = 'CARLA unknown';
        return;
      }
      dot.className = 'status-dot ' + (status.connected ? 'ok' : 'err');
      const managed = status.managed ? 'managed' : 'external';
      const town = status.town ? ` | ${status.town}` : '';
      const cacheSize = status.grp_cache_size != null ? ` | grp cache ${status.grp_cache_size}` : '';
      const detail = status.connected ? `${status.host}:${status.port}${town} | ${managed}${cacheSize}` : (status.last_error || 'unreachable');
      text.textContent = detail;
      // Update prewarm button
      const prewarm = status.prewarm || null;
      const btn = document.getElementById('prewarmBtn');
      if(btn && prewarm){
        if(prewarm.running){
          btn.textContent = `Prewarming… ${prewarm.done}/${prewarm.total}`;
          btn.disabled = true;
        }else{
          const total = prewarm.total || 0;
          const errors = prewarm.errors || 0;
          if(total > 0){
            btn.textContent = errors ? `Prewarm done (${errors} err)` : `Prewarm done ✓`;
          }else{
            btn.textContent = 'Prewarm All GRP';
          }
          btn.disabled = false;
        }
      }
    }

    async function triggerPrewarm(){
      const btn = document.getElementById('prewarmBtn');
      if(btn) btn.disabled = true;
      setMessage('Starting GRP prewarm for all scenarios…');
      try{
        const payload = await fetchJson('/api/grp_prewarm', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({}),
        });
        if(payload.running){
          setMessage(`GRP prewarm started: ${payload.total} scenarios queued.`);
        }else{
          setMessage('GRP prewarm already running or just finished.');
        }
      }catch(err){
        if(btn) btn.disabled = false;
        setMessage('Prewarm failed: ' + err.message);
      }
    }

    async function pollCarla(){
      try{
        const payload = await fetchJson('/api/carla_status');
        updateCarlaStatus(payload);
      }catch(_err){
        updateCarlaStatus(null);
      }
    }

    async function reconnectCarla(restartManaged){
      setMessage(restartManaged ? 'Restarting managed CARLA…' : 'Reconnecting to CARLA…');
      try{
        const payload = await fetchJson('/api/carla_reconnect', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({restart_managed: !!restartManaged}),
        });
        updateCarlaStatus(payload);
        setMessage('CARLA is reachable again.');
        if(state.scenario){
          reloadCurrent(true);
        }
      }catch(err){
        setMessage('CARLA reconnect failed: ' + err.message);
      }
    }

    async function renameScenario(){
      if(!state.scenario) return;
      const currentName = state.scenario.name;
      const newName = window.prompt('Rename scenario folder:', currentName);
      if(newName == null) return;
      const trimmed = newName.trim();
      if(!trimmed || trimmed === currentName) return;
      setMessage('Renaming scenario…');
      try{
        const payload = await fetchJson('/api/rename', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({id: state.scenario.id, name: trimmed}),
        });
        state.scenarios = payload.scenarios || state.scenarios;
        initScenarioGrpStates();
        // Find the new scenario id (the one with matching name)
        const newItem = state.scenarios.find(item => item.name === trimmed);
        if(newItem){
          await loadScenario(newItem.id, true);
        }else{
          renderScenarioList();
          setMessage('Renamed. Could not locate new scenario id.');
        }
      }catch(err){
        setMessage('Rename failed: ' + err.message);
      }
    }

    function escapeHtml(value){
      return String(value == null ? '' : value)
        .replaceAll('&','&amp;')
        .replaceAll('<','&lt;')
        .replaceAll('>','&gt;')
        .replaceAll('"','&quot;')
        .replaceAll("'",'&#39;');
    }

    function hitWaypoint(mx, my){
      if(!state.scenario) return null;
      const orderedRoutes = [...state.scenario.routes].sort((a, b) => (a.actor_id === state.selectedActorId ? -1 : 0) - (b.actor_id === state.selectedActorId ? -1 : 0));
      let best = null;
      let bestDist = Infinity;
      for(const route of orderedRoutes){
        for(let i = 0; i < route.waypoints.length; i += 1){
          const wp = route.waypoints[i];
          const p = worldToCanvas(Number(wp.x), Number(wp.y));
          const dist = Math.hypot(mx - p[0], my - p[1]);
          const threshold = Math.max(13, routeStrokeWidth(route, route.actor_id === state.selectedActorId) * 0.55);
          if(dist <= threshold && dist < bestDist){
            best = {actorId: route.actor_id, index: i};
            bestDist = dist;
          }
        }
      }
      return best;
    }

    function insertWaypointAtCanvasPos(mx, my){
      const route = currentRoute();
      if(!route || route.waypoints.length < 1) return;
      const world = canvasToWorld(mx, my);
      // Find the segment on this route that is closest to the click
      let bestSeg = 0;
      let bestDist = Infinity;
      for(let i = 0; i < route.waypoints.length - 1; i += 1){
        const a = worldToCanvas(Number(route.waypoints[i].x), Number(route.waypoints[i].y));
        const b = worldToCanvas(Number(route.waypoints[i + 1].x), Number(route.waypoints[i + 1].y));
        const d = distToSegment(mx, my, a[0], a[1], b[0], b[1]);
        if(d < bestDist){ bestDist = d; bestSeg = i; }
      }
      // If only 1 waypoint, just append at the last index
      const insertIdx = route.waypoints.length === 1 ? 0 : bestSeg;
      pushHistory('insert waypoint');
      const base = route.waypoints[insertIdx];
      const next = route.waypoints[insertIdx + 1];
      const insert = clone(base);
      insert.x = world[0];
      insert.y = world[1];
      if(next){
        insert.z = (Number(base.z) + Number(next.z)) * 0.5;
        if(base.time != null && next.time != null)
          insert.time = (Number(base.time) + Number(next.time)) * 0.5;
      }
      route.waypoints.splice(insertIdx + 1, 0, insert);
      recomputeRouteYaws(route);
      state.selectedWaypointIndex = insertIdx + 1;
      invalidateSelectedGrpCache();
      markDirty('Waypoint inserted');
      renderWaypointInspector();
      renderWaypointTable();
      drawMap();
    }

    function distToSegment(px, py, ax, ay, bx, by){
      const abx = bx - ax, aby = by - ay;
      const ab2 = abx * abx + aby * aby;
      if(ab2 < 1e-6) return Math.hypot(px - ax, py - ay);
      let t = ((px - ax) * abx + (py - ay) * aby) / ab2;
      t = Math.max(0, Math.min(1, t));
      const qx = ax + t * abx, qy = ay + t * aby;
      return Math.hypot(px - qx, py - qy);
    }

    function hitActor(mx, my){
      if(!state.scenario) return null;
      let best = null;
      let bestDist = Infinity;
      let bestThreshold = 10;
      for(const route of state.scenario.routes){
        for(let i = 0; i < route.waypoints.length - 1; i += 1){
          const a = worldToCanvas(Number(route.waypoints[i].x), Number(route.waypoints[i].y));
          const b = worldToCanvas(Number(route.waypoints[i + 1].x), Number(route.waypoints[i + 1].y));
          const d = distToSegment(mx, my, a[0], a[1], b[0], b[1]);
          if(d < bestDist){
            bestDist = d;
            best = route.actor_id;
            bestThreshold = Math.max(10, routeStrokeWidth(route, false) * 0.65);
          }
        }
      }
      return bestDist <= bestThreshold ? best : null;
    }

    canvas.addEventListener('mousedown', event => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if(event.button === 1 || (event.button === 0 && event.altKey)){
        state.drag = {type:'pan', startX:x, startY:y, tx:state.view.tx, ty:state.view.ty};
        return;
      }
      if(event.button !== 0) return;
      hideNudgeMenu();
      const hit = hitWaypoint(x, y);
      if(hit){
        state.selectedActorId = hit.actorId;
        state.selectedWaypointIndex = hit.index;
        syncSelectedGrpPreviewFromCache();
        renderWaypointInspector();
        renderWaypointTable();
        renderRouteDetails();
        renderActorList();
        drawMap();
        if(event.shiftKey){
          state.drag = {type:'waypoint', actorId: hit.actorId, index: hit.index, snapshotPushed:false};
        }
        return;
      }
      const actorId = hitActor(x, y);
      if(actorId){
        selectActor(actorId);
        return;
      }
      if(state.drawWaypointMode && currentRoute()){
        const world = canvasToWorld(x, y);
        appendWaypointAt(world[0], world[1]);
      }
    });

    window.addEventListener('mouseup', () => {
      const hadWaypointDrag = state.drag && state.drag.type === 'waypoint';
      state.drag = null;
      if(hadWaypointDrag){
        renderWaypointInspector();
        renderRouteDetails();
        renderActorList();
        renderWaypointTable();
      }
    });

    canvas.addEventListener('mousemove', event => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      if(!state.drag) return;
      if(state.drag.type === 'pan'){
        state.view.tx = state.drag.tx + (x - state.drag.startX);
        state.view.ty = state.drag.ty + (y - state.drag.startY);
        drawMap();
        return;
      }
      if(state.drag.type === 'waypoint'){
        const route = currentRoute();
        if(!route || route.actor_id !== state.drag.actorId) return;
        const idx = state.drag.index;
        if(idx < 0 || idx >= route.waypoints.length) return;
        if(!state.drag.snapshotPushed){
          pushHistory('drag waypoint');
          state.drag.snapshotPushed = true;
        }
        const world = canvasToWorld(x, y);
        route.waypoints[idx].x = world[0];
        route.waypoints[idx].y = world[1];
        recomputeRouteYaws(route);
        invalidateSelectedGrpCache();
        state.selectedWaypointIndex = idx;
        markDirty('Waypoint dragged');
        renderWaypointInspector();
        drawMap();
      }
    });

    canvas.addEventListener('dblclick', event => {
      if(!state.scenario || !currentRoute()) return;
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      // If double-click landed on an existing waypoint, ignore (let click-select handle it)
      const hit = hitWaypoint(x, y);
      if(hit && hit.actorId === state.selectedActorId) return;
      insertWaypointAtCanvasPos(x, y);
    });

    canvas.addEventListener('contextmenu', event => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = hitWaypoint(x, y);
      if(!hit){
        hideNudgeMenu();
        return;
      }
      state.selectedActorId = hit.actorId;
      state.selectedWaypointIndex = hit.index;
      syncSelectedGrpPreviewFromCache();
      renderRouteDetails();
      renderWaypointInspector();
      renderWaypointTable();
      renderActorList();
      drawMap();
      showNudgeMenu(event.clientX, event.clientY, hit.actorId, hit.index);
    });

    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const cx = event.clientX - rect.left;
      const cy = event.clientY - rect.top;
      const before = canvasToWorld(cx, cy);
      const factor = event.deltaY < 0 ? 1.12 : 0.89;
      state.view.scale = Math.max(0.05, Math.min(600, state.view.scale * factor));
      state.view.tx = cx - before[0] * state.view.scale;
      state.view.ty = cy + before[1] * state.view.scale;
      drawMap();
    }, {passive:false});

    window.addEventListener('keydown', event => {
      const isMod = event.ctrlKey || event.metaKey;
      if(isMod && event.key.toLowerCase() === 's'){
        event.preventDefault();
        saveScenario();
        return;
      }
      if(isMod && event.key === 'Enter'){
        event.preventDefault();
        saveAndNext();
        return;
      }
      if(isMod && event.key.toLowerCase() === 'z' && !event.shiftKey){
        event.preventDefault();
        undoEdit();
        return;
      }
      if((isMod && event.key.toLowerCase() === 'y') || (isMod && event.shiftKey && event.key.toLowerCase() === 'z')){
        event.preventDefault();
        redoEdit();
        return;
      }
      if(event.key === 'Escape'){
        hideNudgeMenu();
      }
      const typing = isTypingField();
      if(!typing){
        const lower = event.key.toLowerCase();
        if(lower === 'w'){
          event.preventDefault();
          toggleDrawWaypointMode();
          return;
        }
        if(lower === 'n'){
          event.preventDefault();
          goNext();
          return;
        }
        if(lower === 'p'){
          event.preventDefault();
          goPrev();
          return;
        }
        if(event.key === '[' || event.key === ']'){
          event.preventDefault();
          const route = currentRoute();
          if(route && route.waypoints.length){
            const delta = event.key === '[' ? -1 : 1;
            const nextIndex = state.selectedWaypointIndex < 0
              ? (delta > 0 ? 0 : route.waypoints.length - 1)
              : Math.max(0, Math.min(route.waypoints.length - 1, state.selectedWaypointIndex + delta));
            selectWaypoint(nextIndex);
          }
          return;
        }
      }
      // Delete / Backspace — delete the selected waypoint
      if((event.key === 'Delete' || event.key === 'Backspace') && !event.ctrlKey && !event.metaKey){
        if(!typing){
          event.preventDefault();
          const route = currentRoute();
          if(route && state.selectedWaypointIndex >= 0){
            deleteWaypoint(state.selectedWaypointIndex);
          }
        }
      }
    });
    window.addEventListener('mousedown', event => {
      if(!state.nudgeMenu) return;
      if(nudgeMenuEl.contains(event.target)) return;
      hideNudgeMenu();
    });
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('beforeunload', event => {
      if(state.dirty){
        event.preventDefault();
        event.returnValue = '';
      }
    });

    async function init(){
      state.uiMode = readStoredString(UI_MODE_STORAGE_KEY, 'classic') === 'modern' ? 'modern' : 'classic';
      state.showAdvancedPanels = readStoredBool(ADVANCED_PANELS_STORAGE_KEY, false);
      const storedDrawMode = readStoredString(DRAW_MODE_STORAGE_KEY, '');
      state.drawWaypointMode = storedDrawMode === ''
        ? state.uiMode === 'classic'
        : (storedDrawMode === '1' || storedDrawMode === 'true');
      resizeCanvas();
      applyUiPreferences();
      await loadIndex();
      await pollCarla();
      window.setInterval(pollCarla, 5000);
    }

    init().catch(err => {
      setMessage('Initialization failed: ' + err.message);
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios_dir", help="Directory containing scenario subdirectories, or a single scenario directory")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8890, help="HTTP port to bind (default: 8890)")
    parser.add_argument("--carla-host", default="127.0.0.1", help="CARLA host (default: 127.0.0.1)")
    parser.add_argument("--carla-port", type=int, default=2000, help="CARLA RPC port (default: 2000)")
    parser.add_argument("--carla-root", default=None, help="Optional CARLA root for auto-launch")
    parser.add_argument(
        "--no-auto-launch-carla",
        dest="auto_launch_carla",
        action="store_false",
        help="Disable automatic CARLA launch when the configured host:port is unreachable",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra argument to pass to CarlaUE4.sh (repeatable)",
    )
    parser.add_argument(
        "--carla-startup-timeout",
        type=float,
        default=90.0,
        help="Seconds to wait for CARLA RPC readiness after auto-launch",
    )
    parser.add_argument(
        "--carla-post-start-buffer",
        type=float,
        default=2.0,
        help="Extra settle time after CARLA auto-launch (seconds)",
    )
    parser.add_argument(
        "--sampling-distance",
        type=float,
        default=4.0,
        help="Lane centerline sampling distance in metres for map background rendering",
    )
    parser.add_argument(
        "--align-ego-sampling-resolution",
        type=float,
        default=2.0,
        help="GRP sampling resolution (meters) used for ego route alignment.",
    )
    parser.add_argument(
        "--grp-postprocess-mode",
        choices=GRP_POSTPROCESS_MODES,
        default=None,
        help=(
            "Override CARLA GRP postprocess mode for route interpolation. "
            "'none' disables postprocessing (CARLA_GRP_PP_ENABLE=0); "
            "other modes force CARLA_GRP_PP_ENABLE=1 and set CARLA_GRP_PP_MODE. "
            "If omitted, keep existing environment/default behavior."
        ),
    )
    parser.add_argument(
        "--grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_true",
        default=True,
        help=(
            "Set CARLA_GRP_PP_IGNORE_ENDPOINTS=1 so GRP postprocessing excludes "
            "the first/last route nodes from consideration (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_false",
        help="Set CARLA_GRP_PP_IGNORE_ENDPOINTS=0 (disable default endpoint ignoring).",
    )
    parser.add_argument(
        "--prewarm-grp",
        action="store_true",
        default=False,
        help="On startup, launch a background thread to pre-compute GRP previews for all ego routes across all scenarios.",
    )
    parser.add_argument(
        "--bev-cache",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to the birdview_v2_cache Carla/Maps directory containing .npy top-down town images. "
             "Auto-detected if not provided.",
    )
    parser.set_defaults(auto_launch_carla=True)
    return parser.parse_args()


def main() -> None:
    global APP
    args = parse_args()
    scenario_root = Path(args.scenarios_dir).expanduser().resolve()
    print("Starting scenario validation editor...", flush=True)
    print(f"Scenario root: {scenario_root}", flush=True)
    APP = ScenarioBuilderApp(args)

    startup_launch_error: str | None = None
    if APP.carla.auto_launch:
        carla_root_text = str(APP.carla.carla_root) if APP.carla.carla_root is not None else "not found"
        print(
            f"Checking CARLA at {APP.carla.host}:{APP.carla.port} (auto-launch enabled, root={carla_root_text})...",
            flush=True,
        )
        try:
            APP.carla.ensure_ready(restart_if_managed=False)
        except Exception as exc:
            startup_launch_error = str(exc)
            print(f"CARLA warm startup failed: {startup_launch_error}", flush=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"Scenario validation editor listening on http://{args.host}:{args.port}")
    carla_status = APP.carla.status()
    print(
        "CARLA: "
        + (
            f"{carla_status['host']}:{carla_status['port']} reachable"
            if carla_status["connected"]
            else f"{carla_status['host']}:{carla_status['port']} unavailable ({carla_status.get('last_error')})"
        )
    )
    if startup_launch_error is not None:
        print(f"CARLA auto-launch attempt failed: {startup_launch_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        APP.carla.stop()


if __name__ == "__main__":
    main()
