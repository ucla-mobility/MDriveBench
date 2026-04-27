#!/usr/bin/env python3
"""Legacy-style CARLA scenario builder with scenario queue + GRP preview."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_scenario_builder_module() -> Any:
    module_path = Path(__file__).resolve().parent / "scenario_builder.py"
    spec = importlib.util.spec_from_file_location("scenario_builder_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SB = _load_scenario_builder_module()
APP: Any = None
LEGACY_BEV: "LegacyBEVStore | None" = None


def _json_dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def _read_legacy_bev_metadata(bev_path: Path) -> dict[str, float] | None:
    candidate_paths = [
        bev_path.with_suffix(bev_path.suffix + ".meta.json"),
        bev_path.with_suffix(".json"),
    ]
    for meta_path in candidate_paths:
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bounds = metadata.get("world_bounds")
        if isinstance(bounds, dict):
            try:
                return {
                    "min_x": float(bounds["min_x"]),
                    "max_x": float(bounds["max_x"]),
                    "min_y": float(bounds["min_y"]),
                    "max_y": float(bounds["max_y"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _find_default_legacy_bev_dir() -> Path | None:
    candidates = [
        REPO_ROOT / "results" / "carla_town_bev",
        REPO_ROOT / "tools" / "results" / "carla_town_bev",
        REPO_ROOT / "carla_town_bev",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


ASSET_REGISTRY_PATH = REPO_ROOT / "scenario_generator" / "carla_assets.json"
STATIC_PROP_BBOX_CACHE: dict[str, dict[str, float]] = {}
STATIC_PROP_BBOX_CACHE_LOCK = threading.RLock()


def _canonical_asset_id(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _clean_bbox_payload(payload: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(payload, dict):
        return None
    extent_x = _safe_float(payload.get("extent_x"))
    extent_y = _safe_float(payload.get("extent_y"))
    extent_z = _safe_float(payload.get("extent_z"))
    length = _safe_float(payload.get("length"))
    width = _safe_float(payload.get("width"))
    height = _safe_float(payload.get("height"))
    if length is None and extent_x is not None:
        length = extent_x * 2.0
    if width is None and extent_y is not None:
        width = extent_y * 2.0
    if height is None and extent_z is not None:
        height = extent_z * 2.0
    if length is None or width is None:
        return None
    bbox = {
        "length": round(float(length), 6),
        "width": round(float(width), 6),
    }
    if height is not None:
        bbox["height"] = round(float(height), 6)
    if extent_x is not None:
        bbox["extent_x"] = round(float(extent_x), 6)
    if extent_y is not None:
        bbox["extent_y"] = round(float(extent_y), 6)
    if extent_z is not None:
        bbox["extent_z"] = round(float(extent_z), 6)
    return bbox


def _bbox_from_actor(actor: Any) -> dict[str, float] | None:
    try:
        bbox = actor.bounding_box
        extent_x = float(bbox.extent.x)
        extent_y = float(bbox.extent.y)
        extent_z = float(bbox.extent.z)
    except Exception:
        return None
    if extent_x <= 0.0 or extent_y <= 0.0:
        return None
    return {
        "extent_x": round(extent_x, 6),
        "extent_y": round(extent_y, 6),
        "extent_z": round(extent_z, 6),
        "length": round(extent_x * 2.0, 6),
        "width": round(extent_y * 2.0, 6),
        "height": round(extent_z * 2.0, 6),
    }


def _format_prop_label(blueprint_id: str) -> str:
    token = str(blueprint_id or "").strip().split(".")[-1]
    token = token.replace("_", " ").replace("-", " ")
    token = " ".join(part for part in token.split(" ") if part)
    if not token:
        return str(blueprint_id or "Static prop")
    return " ".join(piece[:1].upper() + piece[1:] for piece in token.split(" "))


def _read_repo_static_props() -> list[str]:
    try:
        payload = json.loads(ASSET_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    assets = payload.get("assets", {})
    entries = assets.get("static", []) if isinstance(assets, dict) else []
    if not isinstance(entries, list):
        return []
    ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        asset_id = str(entry.get("id") or "").strip()
        if not asset_id:
            continue
        ids.append(asset_id)
    # Keep deterministic ordering and dedupe.
    seen: set[str] = set()
    unique_ids: list[str] = []
    for asset_id in ids:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        unique_ids.append(asset_id)
    return unique_ids


def _load_bbox_from_asset_lookup(model: str) -> dict[str, float] | None:
    lookup = getattr(SB, "ASSET_BBOX_LOOKUP", {})
    if not isinstance(lookup, dict):
        return None
    entry = lookup.get(_canonical_asset_id(model))
    if not isinstance(entry, dict):
        return None
    return _clean_bbox_payload(entry.get("bbox"))


def _get_cached_bbox(model: str) -> dict[str, float] | None:
    with STATIC_PROP_BBOX_CACHE_LOCK:
        cached = STATIC_PROP_BBOX_CACHE.get(_canonical_asset_id(model))
        if cached is None:
            return None
        return dict(cached)


def _cache_bbox(model: str, bbox: dict[str, float]) -> None:
    canonical = _canonical_asset_id(model)
    if not canonical:
        return
    payload = dict(bbox)
    with STATIC_PROP_BBOX_CACHE_LOCK:
        STATIC_PROP_BBOX_CACHE[canonical] = payload


def _live_static_props(town: str | None = None) -> list[str]:
    if APP is None:
        return []
    status = APP.carla.status()
    if not status.get("connected"):
        return []
    with APP.carla.world_context(town or None) as (_carla, _client, world):
        library = world.get_blueprint_library()
        ids = {
            str(blueprint.id).strip()
            for blueprint in library.filter("static.prop.*")
            if str(getattr(blueprint, "id", "")).strip()
        }
    return sorted(ids)


def _measure_asset_bbox_live(model: str, town: str | None = None) -> tuple[str, dict[str, float]]:
    if APP is None:
        raise RuntimeError("Scenario builder app is not initialised")
    status = APP.carla.status()
    if not status.get("connected"):
        raise RuntimeError("CARLA is not connected")
    model = str(model or "").strip()
    if not model:
        raise ValueError("Model id is required")

    with APP.carla.world_context(town or None) as (carla, _client, world):
        library = world.get_blueprint_library()
        blueprint = None
        try:
            blueprint = library.find(model)
        except Exception:
            blueprint = None
        if blueprint is None:
            lower = model.lower()
            candidates = [bp for bp in library.filter("static.prop.*") if str(getattr(bp, "id", "")).strip().lower() == lower]
            if not candidates:
                raise ValueError(f"Unknown CARLA blueprint: {model}")
            blueprint = candidates[0]

        if hasattr(blueprint, "has_attribute") and blueprint.has_attribute("role_name"):
            try:
                blueprint.set_attribute("role_name", "prop")
            except Exception:
                pass

        attempts: list[Any] = []
        try:
            spawn_points = list(world.get_map().get_spawn_points() or [])
        except Exception:
            spawn_points = []
        for transform in spawn_points[:12]:
            tf = carla.Transform(transform.location, transform.rotation)
            tf.location.z = float(tf.location.z) + 2.5
            attempts.append(tf)
        attempts.append(carla.Transform(carla.Location(x=0.0, y=0.0, z=3.0), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)))

        actor = None
        for transform in attempts:
            try:
                actor = world.try_spawn_actor(blueprint, transform)
            except Exception:
                actor = None
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f"Failed to spawn actor for bbox measurement: {model}")
        try:
            bbox = _bbox_from_actor(actor)
        finally:
            try:
                actor.destroy()
            except Exception:
                pass
    if bbox is None:
        raise RuntimeError(f"CARLA returned no bounding box for {model}")
    resolved_model = str(getattr(blueprint, "id", model)).strip() or model
    return resolved_model, bbox


def _static_prop_presets_payload(town: str | None = None) -> dict[str, Any]:
    repo_props = _read_repo_static_props()
    source = "repository"
    live_error: str | None = None
    try:
        live_props = _live_static_props(town)
    except Exception as exc:  # noqa: BLE001
        live_props = []
        live_error = str(exc)
    if live_props:
        props = live_props
        source = "carla_live"
    else:
        props = repo_props

    presets: list[dict[str, Any]] = []
    for blueprint in props:
        bbox = _get_cached_bbox(blueprint)
        if bbox is None:
            bbox = _load_bbox_from_asset_lookup(blueprint)
            if bbox is not None:
                _cache_bbox(blueprint, bbox)
        preset = {
            "id": blueprint,
            "label": _format_prop_label(blueprint),
            "blueprint": blueprint,
        }
        if bbox is not None:
            preset["bbox"] = bbox
            preset["length"] = bbox.get("length")
            preset["width"] = bbox.get("width")
            preset["height"] = bbox.get("height")
        presets.append(preset)

    presets.sort(key=lambda item: str(item.get("label", "")).lower())
    payload = {
        "source": source,
        "count": len(presets),
        "presets": presets,
    }
    if live_error and source != "carla_live":
        payload["warning"] = f"Failed live CARLA prop query; using repository list: {live_error}"
    return payload


def _asset_bbox_payload(model: str, town: str | None = None) -> dict[str, Any]:
    model = str(model or "").strip()
    if not model:
        raise ValueError("Model id is required")
    cached = _get_cached_bbox(model)
    if cached is not None:
        return {"model": model, "bbox": cached, "source": "cache"}
    from_registry = _load_bbox_from_asset_lookup(model)
    if from_registry is not None:
        _cache_bbox(model, from_registry)
        return {"model": model, "bbox": from_registry, "source": "registry"}
    resolved_model, measured = _measure_asset_bbox_live(model, town)
    _cache_bbox(resolved_model, measured)
    _cache_bbox(model, measured)
    return {"model": resolved_model, "bbox": measured, "source": "carla_live"}


WEATHER_ATTR_KEYS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "wetness",
    "fog_distance",
    "fog_density",
)

CARLA_WEATHER_PRESET_IDS: list[tuple[str, str]] = [
    ("1", "ClearNoon"),
    ("2", "ClearSunset"),
    ("3", "CloudyNoon"),
    ("4", "CloudySunset"),
    ("5", "WetNoon"),
    ("6", "WetSunset"),
    ("7", "MidRainyNoon"),
    ("8", "MidRainSunset"),
    ("9", "WetCloudyNoon"),
    ("10", "WetCloudySunset"),
    ("11", "HardRainNoon"),
    ("12", "HardRainSunset"),
    ("13", "SoftRainNoon"),
    ("14", "SoftRainSunset"),
]

# Fallback values aligned with weather presets used by existing CARLA route generation in-repo.
# Source: simulation/assets/TCP/tools/generate_random_routes.py
FALLBACK_WEATHER_PRESETS: dict[str, dict[str, float]] = {
    "ClearNoon": {"cloudiness": 15.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 0.35, "sun_azimuth_angle": 0.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "ClearSunset": {"cloudiness": 15.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 0.35, "sun_azimuth_angle": 45.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "CloudyNoon": {"cloudiness": 80.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 0.35, "sun_azimuth_angle": 45.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "CloudySunset": {"cloudiness": 80.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "wind_intensity": 0.35, "sun_azimuth_angle": 270.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "WetNoon": {"cloudiness": 20.0, "precipitation": 0.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 45.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "WetSunset": {"cloudiness": 20.0, "precipitation": 0.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 270.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "MidRainyNoon": {"cloudiness": 80.0, "precipitation": 30.0, "precipitation_deposits": 50.0, "wind_intensity": 0.4, "sun_azimuth_angle": 0.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "MidRainSunset": {"cloudiness": 80.0, "precipitation": 30.0, "precipitation_deposits": 50.0, "wind_intensity": 0.4, "sun_azimuth_angle": 270.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "WetCloudyNoon": {"cloudiness": 90.0, "precipitation": 0.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "WetCloudySunset": {"cloudiness": 90.0, "precipitation": 0.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 0.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "HardRainNoon": {"cloudiness": 90.0, "precipitation": 60.0, "precipitation_deposits": 100.0, "wind_intensity": 1.0, "sun_azimuth_angle": 90.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "HardRainSunset": {"cloudiness": 80.0, "precipitation": 60.0, "precipitation_deposits": 100.0, "wind_intensity": 1.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "SoftRainNoon": {"cloudiness": 90.0, "precipitation": 15.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 315.0, "sun_altitude_angle": 75.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
    "SoftRainSunset": {"cloudiness": 90.0, "precipitation": 15.0, "precipitation_deposits": 50.0, "wind_intensity": 0.35, "sun_azimuth_angle": 270.0, "sun_altitude_angle": 15.0, "wetness": 0.0, "fog_distance": 0.0, "fog_density": 0.0},
}


def _extract_weather_payload(obj: Any) -> dict[str, float]:
    payload: dict[str, float] = {}
    for key in WEATHER_ATTR_KEYS:
        if not hasattr(obj, key):
            continue
        value = _safe_float(getattr(obj, key))
        if value is None:
            continue
        payload[key] = round(float(value), 2)
    return payload


def _weather_preset_payload() -> dict[str, Any]:
    presets: list[dict[str, Any]] = []
    source = "fallback"
    warning: str | None = None

    carla_module = None
    if APP is not None:
        try:
            carla_module, _, _ = APP.carla._ensure_carla_import()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            warning = str(exc)

    for preset_id, preset_name in CARLA_WEATHER_PRESET_IDS:
        weather_payload: dict[str, float] | None = None
        if carla_module is not None:
            try:
                preset_obj = getattr(carla_module.WeatherParameters, preset_name)
                weather_payload = _extract_weather_payload(preset_obj)
            except Exception:
                weather_payload = None
        if not weather_payload:
            fallback = FALLBACK_WEATHER_PRESETS.get(preset_name)
            weather_payload = {} if fallback is None else {key: round(float(value), 2) for key, value in fallback.items()}
        if weather_payload:
            presets.append(
                {
                    "id": preset_id,
                    "name": preset_name,
                    "label": f"{preset_id}. {preset_name}",
                    "weather": weather_payload,
                }
            )
    if carla_module is not None and len(presets) == len(CARLA_WEATHER_PRESET_IDS):
        source = "carla_runtime"
    payload: dict[str, Any] = {"source": source, "presets": presets}
    if warning and source != "carla_runtime":
        payload["warning"] = f"CARLA weather preset introspection unavailable; using fallback values: {warning}"
    return payload


class LegacyBEVStore:
    def __init__(self, bev_dir: Path | None) -> None:
        self.bev_dir = bev_dir
        self._png_cache: dict[str, bytes] = {}

    def _town_png_path(self, town: str) -> Path | None:
        if self.bev_dir is None:
            return None
        direct = self.bev_dir / f"{town}.png"
        if direct.exists():
            return direct
        matches = sorted(self.bev_dir.glob(f"{town}*.png"))
        if matches:
            return matches[0]
        return None

    def town_info(self, town: str) -> dict[str, Any]:
        png_path = self._town_png_path(town)
        if png_path is None:
            return {"found": False, "town": town}
        bounds = _read_legacy_bev_metadata(png_path)
        return {
            "found": True,
            "town": town,
            "image_name": png_path.name,
            "bounds": bounds,
            "url": f"/api/legacy_bev_image?town={town}",
        }

    def town_png_bytes(self, town: str) -> bytes | None:
        if town in self._png_cache:
            return self._png_cache[town]
        png_path = self._town_png_path(town)
        if png_path is None:
            return None
        try:
            bev_bytes = png_path.read_bytes()
        except OSError:
            return None
        if Image is not None:
            try:
                with Image.open(png_path) as img:
                    rotated = img.rotate(90, expand=True)
                    # Use in-memory buffer only.
                    import io

                    buffer = io.BytesIO()
                    rotated.save(buffer, format="PNG")
                    bev_bytes = buffer.getvalue()
            except Exception:
                pass
        self._png_cache[town] = bev_bytes
        return bev_bytes


LEGACY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CARLA Scenario Builder (Legacy)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/FileSaver.js/2.0.5/FileSaver.min.js"></script>
  <style>
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background-color: #111;
      color: #e5e5e5;
    }
    .container {
      display: flex;
      height: 100vh;
    }
    .left-panel {
      flex: 1 1 70%;
      position: relative;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .right-panel {
      flex: 1 1 30%;
      background-color: #1a1a1a;
      padding: 16px;
      overflow-y: auto;
      border-left: 1px solid #2b2b2b;
      min-width: 460px;
      max-width: 580px;
    }
    #map {
      width: 100%;
      height: 100%;
    }
    h2, h3 {
      margin-top: 0;
      margin-bottom: 8px;
    }
    label {
      display: block;
      margin-bottom: 6px;
      font-weight: 600;
    }
    input[type="text"], input[type="number"], select {
      width: 100%;
      padding: 6px;
      margin-bottom: 12px;
      border: 1px solid #333;
      border-radius: 4px;
      background-color: #222;
      color: #eee;
      box-sizing: border-box;
    }
    button {
      background-color: #2979ff;
      border: none;
      color: white;
      padding: 8px 12px;
      text-align: center;
      font-size: 14px;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 6px;
      margin-bottom: 6px;
    }
    button.secondary {
      background-color: #555;
    }
    button.danger {
      background-color: #c0392b;
    }
    button.good {
      background-color: #1f7f4c;
    }
    button.warn {
      background-color: #8a6e2b;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      border: 1px solid #333;
      padding: 6px;
      text-align: left;
    }
    th {
      background-color: #252525;
    }
    td input {
      width: 100%;
      box-sizing: border-box;
    }
    textarea.xml-editor {
      width: 100%;
      min-height: 140px;
      margin-top: 6px;
      background-color: #181818;
      color: #dcdcdc;
      border: 1px solid #333;
      border-radius: 4px;
      padding: 6px;
      font-family: "Courier New", monospace;
      box-sizing: border-box;
    }
    details {
      margin-bottom: 14px;
      background-color: #202020;
      border-radius: 4px;
      padding: 6px 10px;
    }
    summary {
      cursor: pointer;
      font-weight: 600;
    }
    .info {
      margin-bottom: 12px;
      font-size: 13px;
      color: #bbb;
    }
    .status-msg {
      font-size: 12px;
      margin-top: 4px;
    }
    .status-error {
      color: #ff6b6b;
    }
    .status-success {
      color: #2ecc71;
    }
    .lane-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 6px 0 12px 0;
      font-size: 12px;
      color: #bbb;
    }
    .lane-legend .lane-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .lane-swatch {
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
    }
    .lane-forward { background-color: #3498db; }
    .lane-opposite { background-color: #e74c3c; }
    .lane-neutral { background-color: #95a5a6; }
    .lane-arrow {
      font-size: 14px;
      font-weight: 600;
    }
    .lane-arrow.forward { color: #3498db; }
    .lane-arrow.opposite { color: #e74c3c; }
    .panel {
      margin: 12px 0 18px 0;
      padding: 10px;
      border: 1px solid #2b2b2b;
      border-radius: 4px;
      background-color: #1a1a1a;
    }
    .panel h3 {
      margin: 0 0 8px 0;
    }
    .offset-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 12px;
      align-items: center;
      margin-bottom: 8px;
    }
    .offset-grid label {
      margin: 0;
      font-size: 13px;
      font-weight: 600;
    }
    .offset-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .static-model-table td input {
      width: 100%;
    }
    button.model-placement-active {
      background-color: #1e5fd8;
    }
    button.weather-preset-btn.active {
      background-color: #1f3a2d;
      border-color: #2ecc71;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
    }
    .status-line {
      margin-top: 6px;
      margin-bottom: 8px;
      color: #9dc1ff;
      font-size: 12px;
      min-height: 18px;
    }
    .scenario-row {
      display: grid;
      grid-template-columns: 1fr auto auto auto auto;
      gap: 6px;
      align-items: center;
      margin-bottom: 10px;
    }
    .scenario-row button {
      margin: 0;
      padding: 7px 10px;
      font-size: 13px;
    }
    .nudge-menu {
      position: absolute;
      z-index: 2100;
      min-width: 156px;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,.1);
      background: rgba(8,12,17,.92);
      box-shadow: 0 16px 42px rgba(0,0,0,.38);
    }
    .nudge-grid {
      display: grid;
      grid-template-columns: repeat(3, 36px);
      gap: 6px;
      justify-content: center;
      margin-top: 8px;
    }
    .nudge-grid button {
      padding: 7px 0;
      min-width: 0;
      margin: 0;
      background-color: #2d2d2d;
    }
    .nudge-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 6px;
    }
    .nudge-meta {
      color: #bdbdbd;
      font-size: 12px;
      margin-bottom: 4px;
    }
    .bev-preview {
      margin: 12px 0 18px 0;
      padding: 10px;
      border: 1px solid #2b2b2b;
      border-radius: 4px;
      background-color: #181818;
    }
    .bev-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }
    .bev-header h3 {
      margin: 0;
      font-size: 16px;
    }
    .icon-button {
      background: transparent;
      color: #e5e5e5;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 4px 8px;
      cursor: pointer;
      font-size: 14px;
      margin: 0;
    }
    .icon-button-large {
      font-size: 20px;
      padding: 4px 10px;
    }
    .icon-button:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }
    #bevImage {
      width: 100%;
      max-height: 320px;
      object-fit: contain;
      border-radius: 4px;
      border: 1px solid #333;
      background-color: #0f0f0f;
    }
    .bev-placeholder {
      font-size: 13px;
      color: #8a8a8a;
      margin: 0;
    }
    .bev-open-notice {
      font-size: 12px;
      color: #8cb4ff;
      margin: 4px 0 0 0;
    }
    .hidden {
      display: none !important;
    }
    .bev-overlay {
      position: fixed;
      top: 80px;
      left: 80px;
      width: 560px;
      height: 560px;
      background-color: #111;
      border: 1px solid #2b2b2b;
      border-radius: 6px;
      box-shadow: 0 10px 35px rgba(0, 0, 0, 0.6);
      z-index: 2000;
      display: flex;
      flex-direction: column;
    }
    .bev-overlay-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 12px;
      cursor: move;
      user-select: none;
      border-bottom: 1px solid #2b2b2b;
    }
    .bev-overlay-header span {
      font-weight: 600;
    }
    .bev-overlay-content {
      flex: 1;
      padding: 8px;
      overflow: auto;
      position: relative;
    }
    .bev-overlay-content img {
      width: 100%;
      height: auto;
      border-radius: 4px;
      transform-origin: center;
      transition: transform 0.1s ease-out;
    }
    .bev-overlay-zoom-floating {
      position: absolute;
      top: 12px;
      right: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      z-index: 5;
    }
    .bev-overlay-zoom-floating button {
      padding: 6px 10px;
      font-size: 18px;
      line-height: 1;
      background-color: rgba(0, 0, 0, 0.6);
    }
    .bev-overlay-resize-handle {
      position: absolute;
      width: 18px;
      height: 18px;
      right: 4px;
      bottom: 4px;
      cursor: se-resize;
      border-right: 2px solid #666;
      border-bottom: 2px solid #666;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="left-panel">
      <div id="map"></div>
      <div id="mapNudgeMenu" class="nudge-menu hidden"></div>
    </div>
    <div class="right-panel">
      <h2>Scenario Builder (Legacy)</h2>
      <p class="info">Old workflow restored. Added: scenario selection + GRP preview/adopt.</p>

      <div class="scenario-row">
        <select id="scenarioSelect"></select>
        <button id="prevScenarioBtn" class="secondary">Prev</button>
        <button id="nextScenarioBtn" class="secondary">Next</button>
        <button id="reloadScenarioBtn" class="secondary">Reload</button>
        <button id="saveScenarioBtn" class="good">Save In Place</button>
      </div>
      <div class="scenario-row">
        <button id="saveNextScenarioBtn" class="good">Save + Next</button>
        <button id="undoBtn" class="secondary">Undo</button>
        <button id="undoAllBtn" class="secondary">Undo All</button>
        <button id="approveBtn" class="good">Approve</button>
        <button id="rejectBtn" class="danger">Reject</button>
        <button id="editedBtn" class="warn">Needs Edit</button>
      </div>

      <div class="status-line" id="scenarioStatusLine"></div>

      <label for="scenarioName">Scenario / folder name</label>
      <input type="text" id="scenarioName" value="" />

      <div class="panel">
        <h3>Weather preset</h3>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <button id="weatherPresetNightBtn" class="secondary weather-preset-btn">Night</button>
          <button id="weatherPresetCloudyBtn" class="secondary weather-preset-btn">Cloudy</button>
          <button id="weatherPresetRainBtn" class="secondary weather-preset-btn">Rain</button>
          <button id="weatherPresetDefaultBtn" class="secondary weather-preset-btn">Default</button>
        </div>
        <p id="weatherPresetStatus" class="info" style="margin:8px 0 0 0;"></p>
      </div>

      <label for="routeId">Route ID (active agent)</label>
      <input type="text" id="routeId" value="247" />

      <label for="actorSelect">Active agent</label>
      <select id="actorSelect"></select>

      <div>
        <button id="addEgoBtn">Add ego</button>
        <button id="addNpcBtn">Add NPC vehicle</button>
        <button id="addPedBtn">Add pedestrian</button>
        <button id="addBikeBtn">Add bicycle</button>
        <button id="removeActorBtn" class="danger">Remove agent</button>
        <button id="clearActorBtn" class="secondary">Clear current path</button>
        <button id="resetAllBtn" class="secondary">Reset all</button>
      </div>

      <div style="margin-top: 8px;">
        <button id="runGrpBtn" class="secondary">Run GRP (all egos)</button>
        <button id="adoptGrpBtn" class="secondary">Adopt GRP</button>
        <span id="grpStatus" class="status-msg"></span>
      </div>

      <p class="info">
        Click to drop waypoints. After placing one, click again in the desired direction to set its heading.
      </p>
      <div class="lane-legend">
        <span class="lane-item">
          <span class="lane-swatch lane-forward"></span>
          <span>Forward lane <span class="lane-arrow forward">&#8594;</span></span>
        </span>
        <span class="lane-item">
          <span class="lane-swatch lane-opposite"></span>
          <span>Opposing lane <span class="lane-arrow opposite">&#8592;</span></span>
        </span>
        <span class="lane-item">
          <span class="lane-swatch lane-neutral"></span>
          <span>Neutral / shoulder</span>
        </span>
      </div>

      <div class="panel">
        <h3>Waypoint offset</h3>
        <div class="offset-grid">
          <label for="offsetXInput">&#916;x [m]</label>
          <input type="number" id="offsetXInput" step="0.1" value="0" />
          <label for="offsetYInput">&#916;y [m]</label>
          <input type="number" id="offsetYInput" step="0.1" value="0" />
        </div>
        <div class="offset-actions">
          <button id="applyOffsetBtn" class="secondary">Apply to active agent</button>
          <button id="resetOffsetInputs" class="secondary">Reset offsets</button>
        </div>
      </div>

      <div class="panel">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:10px">
          <h3>Static 3D models</h3>
          <div>
            <button id="placeStaticModelBtn" class="secondary">Place 3D model</button>
            <button id="clearStaticModelsBtn" class="secondary">Clear models</button>
          </div>
        </div>
        <div>
          <label for="staticModelPreset">Model preset</label>
          <select id="staticModelPreset"></select>
        </div>
        <p id="staticModelHint" class="info" style="display:none;">
          Click on the map to place the selected model. Use the table to fine-tune position and rotation.
        </p>
        <table class="static-model-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>x [m]</th>
              <th>y [m]</th>
              <th>yaw [°]</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="staticModelTableBody">
            <tr><td colspan="5" style="text-align:center; padding:8px;">No static models yet.</td></tr>
          </tbody>
        </table>
      </div>

      <div class="bev-preview">
        <div class="bev-header">
          <h3>Town BEV preview</h3>
          <button id="openBevFullscreen" class="icon-button icon-button-large" title="Open floating viewer">&#x2922;</button>
        </div>
        <img id="bevImage" class="hidden" alt="Bird's-eye preview" />
        <p id="bevOpenNotice" class="bev-open-notice hidden">
          Preview shown in floating window.
        </p>
        <p id="bevFallback" class="bev-placeholder">
          No BEV preview found for this town.
        </p>
      </div>

      <table id="waypointTable">
        <thead>
          <tr>
            <th>#</th>
            <th>x [m]</th>
            <th>y [m]</th>
            <th>yaw [°]</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="waypointTableBody">
          <tr><td colspan="5" style="text-align:center; padding:8px;">No waypoints yet.</td></tr>
        </tbody>
      </table>

      <h3>Generated XML</h3>
      <div id="xmlOutputs"></div>
      <button id="downloadAllBtn">Download all as ZIP</button>
    </div>
  </div>

  <div id="bevOverlay" class="bev-overlay hidden">
    <div id="bevOverlayHeader" class="bev-overlay-header">
      <span id="bevOverlayTitle">Town BEV preview</span>
      <button id="bevOverlayClose" class="icon-button" title="Close">&times;</button>
    </div>
    <div id="bevOverlayContent" class="bev-overlay-content">
      <div class="bev-overlay-zoom-floating">
        <button id="bevZoomInBtn" class="icon-button" title="Zoom in">+</button>
        <button id="bevZoomOutBtn" class="icon-button" title="Zoom out">−</button>
      </div>
      <img id="bevOverlayImage" alt="Town BEV enlarged preview" />
    </div>
    <div id="bevOverlayResize" class="bev-overlay-resize-handle"></div>
  </div>

  <script>
    const colorPalette = [
      "#8e44ad","#f39c12","#2ecc71","#d35400","#1abc9c",
      "#f1c40f","#27ae60","#9b59b6","#ff6f61","#2c3e50"
    ];
    const MIRROR_X = true;
    const MIRROR_Y = false;
    const STATIC_MODEL_FALLBACK_PRESETS = [
      {id:'static.prop.constructioncone',label:'Construction cone',blueprint:'static.prop.constructioncone',length:0.45,width:0.45,height:0.86},
      {id:'static.prop.trafficcone01',label:'Traffic cone 01',blueprint:'static.prop.trafficcone01',length:0.4,width:0.4,height:0.82},
      {id:'static.prop.trafficcone02',label:'Traffic cone 02',blueprint:'static.prop.trafficcone02',length:0.4,width:0.4,height:0.82},
      {id:'static.prop.streetbarrier',label:'Street barrier',blueprint:'static.prop.streetbarrier',length:2.0,width:0.45,height:1.0},
      {id:'static.prop.trafficwarning',label:'Traffic warning',blueprint:'static.prop.trafficwarning',length:1.2,width:0.5,height:1.3},
      {id:'static.prop.container',label:'Container',blueprint:'static.prop.container',length:6.1,width:2.44,height:2.6}
    ];
    let STATIC_MODEL_PRESETS = STATIC_MODEL_FALLBACK_PRESETS.map(preset => ({...preset}));
    let STATIC_MODEL_PRESET_MAP = {};
    let STATIC_MODEL_DEFAULT_PRESET = STATIC_MODEL_PRESETS[0];
    const STATIC_MODEL_BBOX_CACHE = {};
    const WEATHER_ATTR_KEYS = [
      'cloudiness',
      'precipitation',
      'precipitation_deposits',
      'wind_intensity',
      'sun_azimuth_angle',
      'sun_altitude_angle',
      'wetness',
      'fog_distance',
      'fog_density',
    ];
    const PLOTLY_CLICK_FLAG = '__legacyScenarioBuilderPlotlyClick';
    const OVERLAY_MIN_WIDTH = 260;
    const OVERLAY_DEFAULT_WIDTH = 600;
    const ICON_EXPAND = '\u2922';
    const ICON_COLLAPSE = '\u00D7';
    const OVERLAY_MIN_SCALE = 0.1;
    const OVERLAY_MAX_SCALE = 3.0;
    const OVERLAY_SCALE_STEP = 0.1;

    const state = {
      scenarios: [],
      scenarioById: {},
      scenarioId: null,
      scenario: null,
      scenarioWeather: {},
      activeTown: '',
      mapPayload: null,
      actors: [],
      staticModels: [],
      dirty: false,
      grpPreviewByActorId: {},
      grpStatusText: '',
      activeBevBounds: null,
      nudgeStep: 0.5,
      nudgeTarget: null,
      historyUndo: [],
      baselineSnapshot: null,
      weatherPresets: [],
      weatherPresetSource: 'unknown',
    };

    let activeActorId = null;
    let actorIdCounter = 0;
    let staticModelIdCounter = 0;
    let egoCounter = 0;
    let npcCounter = 0;
    let pedestrianCounter = 0;
    let bicycleCounter = 0;
    let staticCounter = 0;
    let colorIndex = 0;
    let pendingHeading = null;
    let orientationPreview = null;
    let plotReady = false;
    let modelPlacementMode = false;
    let activeStaticModelPresetId = STATIC_MODEL_DEFAULT_PRESET.id;
    let bevOverlayDragState = null;
    let bevOverlayResizeState = null;
    let bevOverlayAspect = 1;
    let bevOverlayOpen = false;
    let bevOverlayScale = 1;

    const mapDiv = document.getElementById('map');
    const nudgeMenuEl = document.getElementById('mapNudgeMenu');
    const scenarioSelectEl = document.getElementById('scenarioSelect');
    const scenarioStatusLine = document.getElementById('scenarioStatusLine');
    const weatherPresetNightBtn = document.getElementById('weatherPresetNightBtn');
    const weatherPresetCloudyBtn = document.getElementById('weatherPresetCloudyBtn');
    const weatherPresetRainBtn = document.getElementById('weatherPresetRainBtn');
    const weatherPresetDefaultBtn = document.getElementById('weatherPresetDefaultBtn');
    const weatherPresetStatusEl = document.getElementById('weatherPresetStatus');
    const staticModelHint = document.getElementById('staticModelHint');
    const staticModelTableBody = document.getElementById('staticModelTableBody');
    const staticModelPresetSelect = document.getElementById('staticModelPreset');
    const placeStaticModelBtn = document.getElementById('placeStaticModelBtn');
    const clearStaticModelsBtn = document.getElementById('clearStaticModelsBtn');
    const clearActorBtn = document.getElementById('clearActorBtn');
    const undoBtn = document.getElementById('undoBtn');
    const undoAllBtn = document.getElementById('undoAllBtn');
    const offsetXInput = document.getElementById('offsetXInput');
    const offsetYInput = document.getElementById('offsetYInput');
    const applyOffsetBtn = document.getElementById('applyOffsetBtn');
    const resetOffsetBtn = document.getElementById('resetOffsetInputs');
    const grpStatusEl = document.getElementById('grpStatus');
    const bevImageEl = document.getElementById('bevImage');
    const bevFallbackEl = document.getElementById('bevFallback');
    const bevOpenNotice = document.getElementById('bevOpenNotice');
    const openBevBtn = document.getElementById('openBevFullscreen');
    const bevOverlay = document.getElementById('bevOverlay');
    const bevOverlayHeader = document.getElementById('bevOverlayHeader');
    const bevOverlayTitle = document.getElementById('bevOverlayTitle');
    const bevOverlayImage = document.getElementById('bevOverlayImage');
    const bevOverlayClose = document.getElementById('bevOverlayClose');
    const bevOverlayResize = document.getElementById('bevOverlayResize');
    const bevZoomInBtn = document.getElementById('bevZoomInBtn');
    const bevZoomOutBtn = document.getElementById('bevZoomOutBtn');

    const baseTrace = {
      x: [],
      y: [],
      mode: 'markers',
      marker: {
        size: 5,
        color: 'rgba(200, 200, 200, 0.45)',
        line: {width: 0}
      },
      customdata: [],
      hovertemplate:
        'x=%{x:.2f} m' +
        '<br>y=%{y:.2f} m' +
        '<br>z=%{customdata[0]:.2f} m' +
        '<br>yaw=%{customdata[1]:.1f}°' +
        '<br>lane=%{customdata[2]} (%{customdata[4]})' +
        '<br>road=%{customdata[3]}' +
        '<extra></extra>',
      showlegend: false
    };

    const layout = {
      paper_bgcolor: '#111',
      plot_bgcolor: '#111',
      xaxis: {
        gridcolor: '#222',
        zerolinecolor: '#333',
        title: 'x [m]',
        autorange: true
      },
      yaxis: {
        gridcolor: '#222',
        zerolinecolor: '#333',
        title: 'y [m]',
        scaleanchor: 'x',
        scaleratio: 1,
        autorange: true
      },
      dragmode: 'zoom',
      hovermode: 'closest',
      margin: {l: 60, r: 20, t: 30, b: 60},
      images: []
    };

    const config = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['select2d', 'lasso2d'],
      scrollZoom: true
    };

    Plotly.newPlot(mapDiv, [baseTrace], layout, config).then(() => { plotReady = true; });

    function clone(v) { return JSON.parse(JSON.stringify(v)); }
    function slug(v) {
      return String(v || '').trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '') || 'item';
    }
    function escapeXmlAttr(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }
    function canonicalAssetId(value) {
      return String(value || '').trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    }
    function finiteNumber(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function normalizeBBoxPayload(payload) {
      if (!payload || typeof payload !== 'object') return null;
      const extentX = finiteNumber(payload.extent_x);
      const extentY = finiteNumber(payload.extent_y);
      const extentZ = finiteNumber(payload.extent_z);
      let length = finiteNumber(payload.length);
      let width = finiteNumber(payload.width);
      let height = finiteNumber(payload.height);
      if (length == null && extentX != null) length = extentX * 2;
      if (width == null && extentY != null) width = extentY * 2;
      if (height == null && extentZ != null) height = extentZ * 2;
      if (length == null || width == null) return null;
      const bbox = {
        length: Number(length),
        width: Number(width),
      };
      if (height != null) bbox.height = Number(height);
      if (extentX != null) bbox.extent_x = Number(extentX);
      if (extentY != null) bbox.extent_y = Number(extentY);
      if (extentZ != null) bbox.extent_z = Number(extentZ);
      return bbox;
    }
    function bboxFromRouteAttrs(attrs) {
      if (!attrs || typeof attrs !== 'object') return null;
      return normalizeBBoxPayload({
        length: attrs.bbox_length,
        width: attrs.bbox_width,
        height: attrs.bbox_height,
      });
    }
    function bboxFromPreset(preset) {
      if (!preset || typeof preset !== 'object') return null;
      const fromPreset = normalizeBBoxPayload(preset.bbox);
      if (fromPreset) return fromPreset;
      return normalizeBBoxPayload({
        length: preset.length,
        width: preset.width,
        height: preset.height,
      });
    }
    function bboxFromSizeFields(model) {
      if (!model || typeof model !== 'object') return null;
      return normalizeBBoxPayload({
        length: model.length,
        width: model.width,
        height: model.height,
      });
    }
    function cacheBboxForModel(modelName, bbox) {
      const key = canonicalAssetId(modelName);
      const normalized = normalizeBBoxPayload(bbox);
      if (!key || !normalized) return;
      STATIC_MODEL_BBOX_CACHE[key] = clone(normalized);
    }
    function cachedBboxForModel(modelName) {
      const key = canonicalAssetId(modelName);
      if (!key) return null;
      const cached = STATIC_MODEL_BBOX_CACHE[key];
      return cached ? clone(cached) : null;
    }
    function applyBboxToModel(model, bbox, options) {
      if (!model || !bbox) return false;
      const opts = options || {};
      const normalized = normalizeBBoxPayload(bbox);
      if (!normalized) return false;
      const beforeL = finiteNumber(model.length);
      const beforeW = finiteNumber(model.width);
      const beforeH = finiteNumber(model.height);
      model.length = Number(normalized.length);
      model.width = Number(normalized.width);
      if (normalized.height != null) model.height = Number(normalized.height);
      cacheBboxForModel(model.blueprint || model.model || '', normalized);
      if (opts.persistAttrs !== false) {
        model.routeAttrs = model.routeAttrs || {};
        model.routeAttrs.bbox_length = Number(normalized.length).toFixed(6);
        model.routeAttrs.bbox_width = Number(normalized.width).toFixed(6);
        if (normalized.height != null) model.routeAttrs.bbox_height = Number(normalized.height).toFixed(6);
      }
      return (
        beforeL == null || Math.abs(beforeL - Number(normalized.length)) > 1e-6 ||
        beforeW == null || Math.abs(beforeW - Number(normalized.width)) > 1e-6 ||
        (normalized.height != null && (beforeH == null || Math.abs(beforeH - Number(normalized.height)) > 1e-6))
      );
    }
    function findStaticPresetByBlueprint(blueprint) {
      const target = String(blueprint || '').trim();
      if (!target) return null;
      for (let i = 0; i < STATIC_MODEL_PRESETS.length; i += 1) {
        const preset = STATIC_MODEL_PRESETS[i];
        if (String(preset.blueprint || '').trim() === target) return preset;
      }
      return null;
    }
    function chooseDefaultStaticPresetId(presets) {
      if (!Array.isArray(presets) || presets.length === 0) return null;
      const preferred = ['static.prop.constructioncone', 'static.prop.trafficcone01', 'static.prop.trafficcone02'];
      for (let i = 0; i < preferred.length; i += 1) {
        const hit = presets.find(item => String(item.blueprint || item.id || '').trim() === preferred[i]);
        if (hit) return String(hit.id || hit.blueprint);
      }
      return String(presets[0].id || presets[0].blueprint || '');
    }
    function rebuildStaticPresetIndex() {
      STATIC_MODEL_PRESET_MAP = {};
      STATIC_MODEL_PRESETS.forEach(preset => {
        if (!preset || typeof preset !== 'object') return;
        const id = String(preset.id || preset.blueprint || '').trim();
        const blueprint = String(preset.blueprint || id).trim();
        if (!id || !blueprint) return;
        const merged = {
          id: id,
          blueprint: blueprint,
          label: String(preset.label || blueprint),
        };
        const bbox = bboxFromPreset(preset);
        if (bbox) {
          merged.bbox = clone(bbox);
          merged.length = bbox.length;
          merged.width = bbox.width;
          if (bbox.height != null) merged.height = bbox.height;
          cacheBboxForModel(blueprint, bbox);
        } else {
          if (finiteNumber(preset.length) != null) merged.length = Number(preset.length);
          if (finiteNumber(preset.width) != null) merged.width = Number(preset.width);
          if (finiteNumber(preset.height) != null) merged.height = Number(preset.height);
        }
        STATIC_MODEL_PRESET_MAP[id] = merged;
      });
      STATIC_MODEL_PRESETS = Object.values(STATIC_MODEL_PRESET_MAP).sort((a, b) => String(a.label).localeCompare(String(b.label)));
      STATIC_MODEL_DEFAULT_PRESET = STATIC_MODEL_PRESETS.length ? STATIC_MODEL_PRESETS[0] : {id: 'static.prop.trafficcone01', label: 'Traffic cone 01', blueprint: 'static.prop.trafficcone01', length: 0.4, width: 0.4};
      if (!activeStaticModelPresetId || !STATIC_MODEL_PRESET_MAP[activeStaticModelPresetId]) {
        activeStaticModelPresetId = chooseDefaultStaticPresetId(STATIC_MODEL_PRESETS) || STATIC_MODEL_DEFAULT_PRESET.id;
      }
    }
    function renderStaticModelPresetOptions() {
      if (!staticModelPresetSelect) return;
      staticModelPresetSelect.innerHTML = '';
      STATIC_MODEL_PRESETS.forEach(preset => {
        const option = document.createElement('option');
        option.value = preset.id;
        option.textContent = preset.label + ' (' + preset.blueprint + ')';
        staticModelPresetSelect.appendChild(option);
      });
      if (STATIC_MODEL_PRESETS.length === 0) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'No CARLA props available';
        staticModelPresetSelect.appendChild(option);
      }
      staticModelPresetSelect.value = activeStaticModelPresetId;
    }
    async function resolveAssetBBox(modelName, forceRefresh) {
      const blueprint = String(modelName || '').trim();
      if (!blueprint) return null;
      if (!forceRefresh) {
        const cached = cachedBboxForModel(blueprint);
        if (cached) return cached;
      }
      const query = new URLSearchParams({model: blueprint});
      if (state.activeTown) query.set('town', state.activeTown);
      const payload = await fetchJson('/api/asset_bbox?' + query.toString());
      const bbox = normalizeBBoxPayload(payload && payload.bbox ? payload.bbox : payload);
      if (!bbox) return null;
      const resolvedModel = payload && payload.model ? String(payload.model) : blueprint;
      cacheBboxForModel(resolvedModel, bbox);
      cacheBboxForModel(blueprint, bbox);
      return bbox;
    }
    async function refreshStaticModelBBox(model, options) {
      const opts = options || {};
      if (!model || !model.blueprint) return null;
      try {
        const bbox = await resolveAssetBBox(model.blueprint, !!opts.forceRefresh);
        if (!bbox) return null;
        const changed = applyBboxToModel(model, bbox, {persistAttrs: opts.persistAttrs !== false});
        if (changed) {
          renderStaticModelTable();
          updatePlot();
          if (opts.markDirty) setDirty(true);
        }
        return bbox;
      } catch (_err) {
        return null;
      }
    }
    function queueStaticModelBBoxRefresh(model, options) {
      if (!model || !model.blueprint) return;
      const opts = options || {};
      const attrsBBox = bboxFromRouteAttrs(model.routeAttrs || {});
      if (attrsBBox) return;
      const cached = cachedBboxForModel(model.blueprint);
      if (cached) {
        const changed = applyBboxToModel(model, cached, {persistAttrs: true});
        if (changed) {
          renderStaticModelTable();
          updatePlot();
          if (opts.markDirty) setDirty(true);
        }
        return;
      }
      window.setTimeout(() => {
        refreshStaticModelBBox(model, {
          markDirty: !!opts.markDirty,
          forceRefresh: !!opts.forceRefresh,
          persistAttrs: opts.persistAttrs !== false,
        });
      }, 0);
    }
    async function loadStaticModelPresets() {
      let presets = [];
      try {
        const townQuery = state.activeTown ? ('?town=' + encodeURIComponent(state.activeTown)) : '';
        const payload = await fetchJson('/api/static_prop_presets' + townQuery);
        if (payload && Array.isArray(payload.presets)) presets = payload.presets;
      } catch (_err) {
        presets = [];
      }
      if (!presets.length) {
        presets = STATIC_MODEL_FALLBACK_PRESETS.map(item => ({...item}));
      }
      STATIC_MODEL_PRESETS = presets.map(item => ({...item}));
      rebuildStaticPresetIndex();
      renderStaticModelPresetOptions();
      updateStaticModelPlacementUi();
    }
    function normalizeWeatherPayload(payload) {
      const result = {};
      if (!payload || typeof payload !== 'object') return result;
      WEATHER_ATTR_KEYS.forEach((key) => {
        const value = finiteNumber(payload[key]);
        if (value == null) return;
        result[key] = Number(value.toFixed(2));
      });
      return result;
    }
    function weatherPayloadEqual(a, b) {
      const aa = normalizeWeatherPayload(a);
      const bb = normalizeWeatherPayload(b);
      const aKeys = Object.keys(aa).sort();
      const bKeys = Object.keys(bb).sort();
      if (aKeys.length !== bKeys.length) return false;
      for (let i = 0; i < aKeys.length; i += 1) {
        if (aKeys[i] !== bKeys[i]) return false;
      }
      return aKeys.every((key) => Math.abs(Number(aa[key]) - Number(bb[key])) <= 0.05);
    }
    const QUICK_WEATHER_FALLBACKS = {
      night: {
        cloudiness: 20.0,
        precipitation: 0.0,
        precipitation_deposits: 0.0,
        wind_intensity: 5.0,
        sun_azimuth_angle: 300.0,
        sun_altitude_angle: -15.0,
        wetness: 10.0,
        fog_distance: 80.0,
        fog_density: 5.0,
      },
      cloudy: {
        cloudiness: 80.0,
        precipitation: 0.0,
        precipitation_deposits: 0.0,
        wind_intensity: 0.35,
        sun_azimuth_angle: 45.0,
        sun_altitude_angle: 75.0,
        wetness: 0.0,
        fog_distance: 0.0,
        fog_density: 0.0,
      },
      rain: {
        cloudiness: 90.0,
        precipitation: 60.0,
        precipitation_deposits: 100.0,
        wind_intensity: 1.0,
        sun_azimuth_angle: 90.0,
        sun_altitude_angle: 75.0,
        wetness: 0.0,
        fog_distance: 0.0,
        fog_density: 0.0,
      },
    };
    function weatherPresetByName(name) {
      const target = String(name || '').trim().toLowerCase();
      if (!target) return null;
      return (state.weatherPresets || []).find(item => String(item.name || '').trim().toLowerCase() === target) || null;
    }
    function quickWeatherPresetPayload(key) {
      const normalizedKey = String(key || '').trim().toLowerCase();
      if (normalizedKey === 'default') return {};
      if (normalizedKey === 'night') return normalizeWeatherPayload(QUICK_WEATHER_FALLBACKS.night);
      if (normalizedKey === 'cloudy') {
        const cloudy = weatherPresetByName('CloudyNoon');
        return normalizeWeatherPayload(cloudy && cloudy.weather ? cloudy.weather : QUICK_WEATHER_FALLBACKS.cloudy);
      }
      if (normalizedKey === 'rain') {
        const hard = weatherPresetByName('HardRainNoon');
        const soft = weatherPresetByName('SoftRainNoon');
        const mid = weatherPresetByName('MidRainyNoon');
        return normalizeWeatherPayload(
          (hard && hard.weather) || (soft && soft.weather) || (mid && mid.weather) || QUICK_WEATHER_FALLBACKS.rain
        );
      }
      return {};
    }
    function detectQuickWeatherPreset(payload) {
      const current = normalizeWeatherPayload(payload);
      if (!Object.keys(current).length) return 'default';
      if (weatherPayloadEqual(current, quickWeatherPresetPayload('night'))) return 'night';
      if (weatherPayloadEqual(current, quickWeatherPresetPayload('cloudy'))) return 'cloudy';
      if (weatherPayloadEqual(current, quickWeatherPresetPayload('rain'))) return 'rain';
      return 'custom';
    }
    function weatherSummaryText(payload) {
      const weather = normalizeWeatherPayload(payload);
      const keys = Object.keys(weather);
      if (!keys.length) return 'No weather override in editor; evaluator/runtime default route weather applies.';
      const ordered = [
        'cloudiness',
        'precipitation',
        'precipitation_deposits',
        'sun_altitude_angle',
        'sun_azimuth_angle',
      ];
      const preview = ordered
        .filter(key => Object.prototype.hasOwnProperty.call(weather, key))
        .slice(0, 5)
        .map(key => `${key}=${weather[key].toFixed(2)}`)
        .join(', ');
      return preview || keys.map(key => `${key}=${weather[key].toFixed(2)}`).join(', ');
    }
    function renderWeatherPresetUi() {
      const currentWeather = normalizeWeatherPayload(state.scenarioWeather || {});
      const active = detectQuickWeatherPreset(currentWeather);
      if (weatherPresetNightBtn) weatherPresetNightBtn.classList.toggle('active', active === 'night');
      if (weatherPresetCloudyBtn) weatherPresetCloudyBtn.classList.toggle('active', active === 'cloudy');
      if (weatherPresetRainBtn) weatherPresetRainBtn.classList.toggle('active', active === 'rain');
      if (weatherPresetDefaultBtn) weatherPresetDefaultBtn.classList.toggle('active', active === 'default');

      if (weatherPresetStatusEl) {
        const source = state.weatherPresetSource ? `preset source: ${state.weatherPresetSource}` : 'preset source: unknown';
        if (active === 'default') {
          weatherPresetStatusEl.textContent = `Active weather: default/no editor override. ${source}.`;
        } else if (active === 'custom') {
          weatherPresetStatusEl.textContent = `Active weather: custom override. ${weatherSummaryText(currentWeather)}.`;
        } else {
          weatherPresetStatusEl.textContent = `Active weather: ${active}. ${weatherSummaryText(currentWeather)}.`;
        }
      }
    }
    function setScenarioWeather(nextWeather, statusText) {
      const current = normalizeWeatherPayload(state.scenarioWeather || {});
      const normalized = normalizeWeatherPayload(nextWeather || {});
      if (weatherPayloadEqual(current, normalized)) return false;
      pushHistory();
      state.scenarioWeather = normalized;
      renderWeatherPresetUi();
      setDirty(true);
      if (statusText) setStatus(statusText);
      return true;
    }
    async function loadWeatherPresets() {
      try {
        const payload = await fetchJson('/api/weather_presets');
        const presets = Array.isArray(payload && payload.presets) ? payload.presets : [];
        state.weatherPresetSource = String((payload && payload.source) || 'unknown');
        state.weatherPresets = presets
          .map((item, idx) => {
            const weather = normalizeWeatherPayload(item && item.weather ? item.weather : {});
            if (!Object.keys(weather).length) return null;
            return {
              id: String(item.id == null ? idx + 1 : item.id),
              name: String(item.name || item.label || item.id || `Preset ${idx + 1}`),
              label: String(item.label || item.name || `Preset ${idx + 1}`),
              weather: weather,
            };
          })
          .filter(Boolean);
      } catch (_err) {
        state.weatherPresets = [];
        state.weatherPresetSource = 'unavailable';
      }
      renderWeatherPresetUi();
    }
    rebuildStaticPresetIndex();
    function updateUndoButtons() {
      if (undoBtn) undoBtn.disabled = state.historyUndo.length === 0;
      if (undoAllBtn) undoAllBtn.disabled = !state.baselineSnapshot;
    }
    function setDirty(value) {
      state.dirty = !!value;
      updateUndoButtons();
      renderScenarioStatus();
    }
    function editableSnapshot() {
      return clone({
        actors: state.actors,
        staticModels: state.staticModels,
        scenarioWeather: state.scenarioWeather,
        activeActorId: activeActorId,
        grpPreviewByActorId: state.grpPreviewByActorId
      });
    }
    function snapshotKey(snapshot) {
      if (!snapshot) return '';
      return JSON.stringify(snapshot);
    }
    function snapshotsEqual(a, b) {
      return snapshotKey(a) === snapshotKey(b);
    }
    function commitBaselineSnapshot() {
      state.baselineSnapshot = editableSnapshot();
      state.historyUndo = [];
      updateUndoButtons();
    }
    function restoreSnapshot(snapshot, options) {
      if (!snapshot) return false;
      const opts = options || {};
      state.actors = clone(snapshot.actors || []);
      state.staticModels = clone(snapshot.staticModels || []);
      state.scenarioWeather = clone(snapshot.scenarioWeather || {});
      state.grpPreviewByActorId = clone(snapshot.grpPreviewByActorId || {});
      activeActorId = snapshot.activeActorId == null ? null : snapshot.activeActorId;
      if (activeActorId != null && !state.actors.some(actor => actor.id === activeActorId)) {
        activeActorId = state.actors.length ? state.actors[0].id : null;
      }
      pendingHeading = null;
      orientationPreview = null;
      hideNudgeMenu();
      setModelPlacementMode(false);
      const actor = getActiveActor();
      document.getElementById('routeId').value = actor ? routeIdForActor(actor) : '';
      renderActorSelect();
      renderWaypointTable();
      renderStaticModelTable();
      renderWeatherPresetUi();
      renderXmlOutputs();
      updatePlot();
      const dirty = Object.prototype.hasOwnProperty.call(opts, 'dirty')
        ? !!opts.dirty
        : !snapshotsEqual(snapshot, state.baselineSnapshot);
      setDirty(dirty);
      if (opts.status) setStatus(opts.status);
      return true;
    }
    function pushHistory() {
      state.historyUndo.push(editableSnapshot());
      if (state.historyUndo.length > 200) state.historyUndo.shift();
      updateUndoButtons();
    }
    function undoEdit() {
      if (state.historyUndo.length === 0) return false;
      const snapshot = state.historyUndo.pop();
      const dirty = !snapshotsEqual(snapshot, state.baselineSnapshot);
      const ok = restoreSnapshot(snapshot, {dirty: dirty, status: 'Undid last change.'});
      updateUndoButtons();
      return ok;
    }
    function undoAllEdits() {
      if (!state.baselineSnapshot) return false;
      const ok = restoreSnapshot(state.baselineSnapshot, {dirty: false, status: 'Reverted all edits for this scenario.'});
      state.historyUndo = [];
      updateUndoButtons();
      return ok;
    }
    function activeScenarioSummary() {
      return state.scenarioById[state.scenarioId] || null;
    }
    function setStatus(text, cls) {
      scenarioStatusLine.textContent = text || '';
      scenarioStatusLine.className = 'status-line' + (cls ? (' ' + cls) : '');
    }
    function setGrpStatus(text, ok=false) {
      grpStatusEl.textContent = text || '';
      grpStatusEl.className = 'status-msg ' + (ok ? 'status-success' : '');
    }
    function hasLoadedBevImage() {
      return !!(bevImageEl && bevImageEl.complete && bevImageEl.naturalWidth > 0 && bevImageEl.naturalHeight > 0);
    }
    function updateBevButtonState(hasImage) {
      if (!openBevBtn) return;
      if (!hasImage) {
        openBevBtn.disabled = true;
        openBevBtn.textContent = ICON_EXPAND;
        openBevBtn.title = 'Open floating viewer';
        return;
      }
      openBevBtn.disabled = false;
      if (bevOverlayOpen) {
        openBevBtn.textContent = ICON_COLLAPSE;
        openBevBtn.title = 'Hide floating viewer';
      } else {
        openBevBtn.textContent = ICON_EXPAND;
        openBevBtn.title = 'Open floating viewer';
      }
    }
    function syncBevPreviewVisibility(hasImage) {
      if (!bevImageEl || !bevFallbackEl || !bevOpenNotice) return;
      if (!hasImage) {
        bevImageEl.classList.add('hidden');
        bevOpenNotice.classList.add('hidden');
        bevFallbackEl.classList.remove('hidden');
        return;
      }
      bevFallbackEl.classList.add('hidden');
      if (bevOverlayOpen) {
        bevImageEl.classList.add('hidden');
        bevOpenNotice.classList.remove('hidden');
      } else {
        bevImageEl.classList.remove('hidden');
        bevOpenNotice.classList.add('hidden');
      }
    }
    async function loadBevForTown(town) {
      if (!town || !bevImageEl) {
        if (bevOverlayOpen) closeBevOverlay();
        state.activeBevBounds = null;
        syncBevPreviewVisibility(false);
        updateBevButtonState(false);
        if (bevImageEl) bevImageEl.removeAttribute('src');
        layout.images = [];
        if (plotReady) updatePlot();
        return;
      }
      let info = null;
      try {
        info = await fetchJson('/api/legacy_bev?town=' + encodeURIComponent(town));
      } catch (_err) {
        info = null;
      }
      if (!info || !info.found || !info.url) {
        if (bevOverlayOpen) closeBevOverlay();
        state.activeBevBounds = null;
        syncBevPreviewVisibility(false);
        updateBevButtonState(false);
        bevImageEl.removeAttribute('src');
        layout.images = [];
        if (plotReady) updatePlot();
        return;
      }
      state.activeBevBounds = info.bounds || null;
      const src = info.url + '&_=' + Date.now();
      bevImageEl.onload = () => {
        syncBevPreviewVisibility(true);
        updateBevButtonState(true);
        updateMapBackgroundImage(town);
        if (plotReady) updatePlot();
      };
      bevImageEl.onerror = () => {
        if (bevOverlayOpen) closeBevOverlay();
        state.activeBevBounds = null;
        syncBevPreviewVisibility(false);
        updateBevButtonState(false);
        layout.images = [];
        if (plotReady) updatePlot();
      };
      bevImageEl.src = src;
      if (bevOverlayTitle) bevOverlayTitle.textContent = town + ' BEV';
      if (bevOverlayImage && bevOverlayOpen) {
        bevOverlayImage.src = src;
      }
      syncBevPreviewVisibility(false);
      updateBevButtonState(false);
    }
    function closeBevOverlay() {
      if (!bevOverlay) return;
      bevOverlay.classList.add('hidden');
      bevOverlayDragState = null;
      bevOverlayResizeState = null;
      bevOverlayOpen = false;
      const hasImage = hasLoadedBevImage();
      syncBevPreviewVisibility(hasImage);
      updateBevButtonState(hasImage);
    }
    function clampOverlayScale(value) {
      return Math.min(OVERLAY_MAX_SCALE, Math.max(OVERLAY_MIN_SCALE, value));
    }
    function updateOverlayZoomUI() {
      if (bevOverlayImage) {
        bevOverlayImage.style.transform = 'scale(' + bevOverlayScale + ')';
      }
    }
    function setOverlayScale(scale) {
      bevOverlayScale = clampOverlayScale(scale);
      updateOverlayZoomUI();
    }
    function applyOverlaySize(width) {
      if (!bevOverlay) return;
      const aspect = Math.max(bevOverlayAspect, 1e-3);
      const clampedWidth = Math.max(OVERLAY_MIN_WIDTH, width);
      bevOverlay.style.width = clampedWidth + 'px';
      bevOverlay.style.height = Math.max(200, clampedWidth / aspect) + 'px';
      bevOverlay.dataset.width = String(clampedWidth);
    }
    function showBevOverlay() {
      if (!bevOverlay || !bevOverlayImage || !bevImageEl || !hasLoadedBevImage()) return;
      bevOverlayImage.src = bevImageEl.src;
      if (!bevOverlay.dataset.positioned) {
        bevOverlay.style.left = '80px';
        bevOverlay.style.top = '80px';
        bevOverlay.dataset.positioned = '1';
      }
      const storedWidth = bevOverlay.dataset.width ? parseFloat(bevOverlay.dataset.width) : OVERLAY_DEFAULT_WIDTH;
      applyOverlaySize(storedWidth || OVERLAY_DEFAULT_WIDTH);
      bevOverlay.classList.remove('hidden');
      bevOverlayOpen = true;
      syncBevPreviewVisibility(true);
      updateBevButtonState(true);
      setOverlayScale(bevOverlayScale || 1);
    }
    function updateMapBackgroundImage(town) {
      const bounds = state.activeBevBounds;
      if (!town || !bounds) {
        layout.images = [];
        return;
      }
      const source = '/api/legacy_bev_image?town=' + encodeURIComponent(town) + '&_=' + Date.now();
      layout.images = [{
        source: source,
        xref: 'x',
        yref: 'y',
        x: Number(bounds.min_x),
        y: Number(bounds.max_y),
        sizex: Number(bounds.max_x) - Number(bounds.min_x),
        sizey: Number(bounds.max_y) - Number(bounds.min_y),
        sizing: 'stretch',
        opacity: 0.8,
        layer: 'below',
      }];
    }
    function fetchJson(url, opts) {
      return fetch(url, opts).then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || response.statusText || 'Request failed');
        if (payload && payload.error) throw new Error(payload.error);
        return payload;
      });
    }
    function getActiveActor() {
      return state.actors.find(a => a.id === activeActorId) || null;
    }
    function ensureActorOffset(actor) {
      if (!actor) return null;
      if (!actor.offsetTotal) actor.offsetTotal = {x: 0, y: 0};
      return actor.offsetTotal;
    }
    function hasActiveOffset(actor) {
      const offset = ensureActorOffset(actor);
      return !!offset && (Math.abs(offset.x) > 1e-6 || Math.abs(offset.y) > 1e-6);
    }
    function resetActorOffsetState(actor) {
      const offset = ensureActorOffset(actor);
      if (!offset) return;
      offset.x = 0;
      offset.y = 0;
    }
    function routeRole(actor) {
      return String(actor && (actor.kind || (actor.routeAttrs && actor.routeAttrs.role)) || 'npc').toLowerCase();
    }
    function isEgoActor(actor) {
      return routeRole(actor) === 'ego';
    }
    function isGenericEntityName(value) {
      const text = String(value || '').trim().toLowerCase();
      if (!text) return true;
      if (/^(entity|actor|static|npc|pedestrian|bicycle|walker|cyclist|ego)(_[a-z0-9]+)+$/.test(text)) return true;
      if (/^town[0-9a-z]+_(entity|actor|static|npc|pedestrian|bicycle|walker|cyclist|ego)_/.test(text)) return true;
      return false;
    }
    function titleCaseWords(value) {
      return String(value || '')
        .split(' ')
        .filter(Boolean)
        .map(token => token.charAt(0).toUpperCase() + token.slice(1))
        .join(' ');
    }
    function modelLabelFromId(modelId, roleHint) {
      const model = String(modelId || '').trim();
      if (!model) return '';
      const parts = model.split('.').filter(Boolean);
      if (!parts.length) return '';
      const role = String(roleHint || '').toLowerCase();
      let text = '';
      if (parts[0] === 'static' && parts[1] === 'prop') {
        const preset = findStaticPresetByBlueprint(model);
        if (preset && preset.label) return String(preset.label);
        text = parts.slice(2).join(' ');
      } else if (parts[0] === 'vehicle') {
        text = parts.slice(1).join(' ');
      } else if (parts[0] === 'walker') {
        text = parts.slice(1).join(' ');
      } else if (role === 'pedestrian' || role === 'walker') {
        text = 'pedestrian';
      } else if (role === 'bicycle' || role === 'cyclist') {
        text = 'bicycle';
      } else {
        text = parts.slice(-1)[0];
      }
      text = text.replace(/[._-]+/g, ' ').trim();
      return titleCaseWords(text);
    }
    function actorDisplayName(actor) {
      const rawName = String(actor && actor.name || '').trim();
      const modelName =
        String(
          (actor && actor.routeAttrs && actor.routeAttrs.model)
          || (actor && actor.resolvedModel)
          || ''
        ).trim();
      const modelLabel = modelLabelFromId(modelName, routeRole(actor));
      if (!rawName) return modelLabel || 'Actor';
      if (modelLabel && isGenericEntityName(rawName)) return `${modelLabel} (${rawName})`;
      return rawName;
    }
    function staticModelDisplayName(model) {
      const rawName = String(model && model.name || '').trim();
      const modelName = String((model && model.blueprint) || (model && model.routeAttrs && model.routeAttrs.model) || '').trim();
      const modelLabel = modelLabelFromId(modelName, 'static');
      if (!rawName) return modelLabel || 'Static';
      if (modelLabel && isGenericEntityName(rawName)) return `${modelLabel} (${rawName})`;
      return rawName;
    }
    function parseCssColorToRgb(color) {
      const text = String(color || '').trim();
      if (!text) return null;
      const hex3 = /^#([0-9a-f]{3})$/i.exec(text);
      if (hex3) {
        const h = hex3[1];
        return {
          r: parseInt(h[0] + h[0], 16),
          g: parseInt(h[1] + h[1], 16),
          b: parseInt(h[2] + h[2], 16),
        };
      }
      const hex6 = /^#([0-9a-f]{6})$/i.exec(text);
      if (hex6) {
        const h = hex6[1];
        return {
          r: parseInt(h.slice(0, 2), 16),
          g: parseInt(h.slice(2, 4), 16),
          b: parseInt(h.slice(4, 6), 16),
        };
      }
      const rgb = /^rgba?\(([^)]+)\)$/i.exec(text);
      if (rgb) {
        const parts = rgb[1].split(',').map(v => Number(v.trim()));
        if (parts.length >= 3 && parts.slice(0, 3).every(Number.isFinite)) {
          return {
            r: Math.max(0, Math.min(255, parts[0])),
            g: Math.max(0, Math.min(255, parts[1])),
            b: Math.max(0, Math.min(255, parts[2])),
          };
        }
      }
      return null;
    }
    function darkerShade(color, factor = 0.72) {
      const rgb = parseCssColorToRgb(color);
      if (!rgb) return color || '#00b8cc';
      const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
      return `rgb(${clamp(rgb.r * factor)}, ${clamp(rgb.g * factor)}, ${clamp(rgb.b * factor)})`;
    }
    function getMarkerStyle(actor) {
      if (actor.kind === 'pedestrian') return {symbol: 'circle', size: 14};
      if (actor.kind === 'bicycle') return {symbol: 'diamond', size: 16};
      return {symbol: 'triangle-up', size: 20};
    }
    function nextActorName(kind) {
      if (kind === 'ego') return 'ego_vehicle_' + (egoCounter++);
      if (kind === 'pedestrian') return 'pedestrian_' + (pedestrianCounter++);
      if (kind === 'bicycle') return 'bicycle_' + (bicycleCounter++);
      return 'npc_vehicle_' + (npcCounter++);
    }
    function uniqueRouteFile(kind, actorName) {
      const townSlug = slug(state.activeTown || 'town');
      const base = kind === 'ego'
        ? `${townSlug}_${slug(actorName)}.xml`
        : `actors/${kind}/${townSlug}_${slug(actorName)}.xml`;
      const existing = new Set();
      state.actors.forEach(a => existing.add(String(a.file || '')));
      state.staticModels.forEach(s => existing.add(String(s.file || '')));
      if (!existing.has(base)) return base;
      let i = 2;
      while (true) {
        const candidate = kind === 'ego'
          ? `${townSlug}_${slug(actorName)}_${i}.xml`
          : `actors/${kind}/${townSlug}_${slug(actorName)}_${i}.xml`;
        if (!existing.has(candidate)) return candidate;
        i += 1;
      }
    }
    function defaultModelForKind(kind) {
      if (kind === 'ego') return 'vehicle.lincoln.mkz2017';
      if (kind === 'npc') return 'vehicle.audi.a2';
      if (kind === 'pedestrian') return 'walker.pedestrian.0001';
      if (kind === 'bicycle') return 'vehicle.bh.crossbike';
      return '';
    }
    function createActor(kind) {
      const name = nextActorName(kind);
      const actor = {
        id: actorIdCounter++,
        kind: kind,
        name: name,
        color: colorPalette[colorIndex++ % colorPalette.length],
        waypoints: [],
        offsetTotal: {x: 0, y: 0},
        file: uniqueRouteFile(kind, name),
        routeAttrs: {
          id: slug(`${state.activeTown || 'town'}_${kind}_${name}`),
          town: state.activeTown || 'Town01',
          role: kind,
          model: defaultModelForKind(kind)
        },
        supportsGrp: kind === 'ego' || kind === 'npc',
      };
      return actor;
    }
    function formatWaypoint(wp, indent) {
      const i = indent || '      ';
      const attrs = [
        `x="${Number(wp.x).toFixed(6)}"`,
        `y="${Number(wp.y).toFixed(6)}"`,
        `z="${Number((wp.z == null ? 0 : wp.z)).toFixed(6)}"`,
        `yaw="${Number((wp.yaw == null ? 0 : wp.yaw)).toFixed(6)}"`,
      ];
      if (wp.pitch != null) attrs.push(`pitch="${Number(wp.pitch).toFixed(6)}"`);
      if (wp.roll != null) attrs.push(`roll="${Number(wp.roll).toFixed(6)}"`);
      if (wp.time != null) attrs.push(`time="${Number(wp.time).toFixed(6)}"`);
      if (wp.speed != null) attrs.push(`speed="${Number(wp.speed).toFixed(4)}"`);
      return i + '<waypoint ' + attrs.join(' ') + ' />';
    }
    function generateXmlForActor(actor) {
      const attrs = actor.routeAttrs || {};
      const routeId = attrs.id || '0';
      const town = attrs.town || state.activeTown || 'Town01';
      const role = attrs.role || actor.kind || 'npc';
      const model = attrs.model ? ` model="${attrs.model}"` : '';
      const snap = attrs.snap_to_road ? ` snap_to_road="${attrs.snap_to_road}"` : '';
      const spawnSnap = attrs.snap_spawn_to_road ? ` snap_spawn_to_road="${attrs.snap_spawn_to_road}"` : '';
      const control = attrs.control_mode ? ` control_mode="${attrs.control_mode}"` : '';
      const targetSpeed = attrs.target_speed ? ` target_speed="${attrs.target_speed}"` : '';
      const lines = [
        "<?xml version='1.0' encoding='utf-8'?>",
        '<routes>',
        `  <route id="${routeId}" town="${town}" role="${role}"${model}${snap}${spawnSnap}${control}${targetSpeed}>`
      ];
      actor.waypoints.forEach(wp => { lines.push(formatWaypoint(wp, '    ')); });
      lines.push('  </route>');
      lines.push('</routes>');
      return lines.join('\n');
    }
    function parseXmlToWaypoints(xmlText) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(xmlText, 'application/xml');
      const errorNode = doc.getElementsByTagName('parsererror');
      if (errorNode.length) throw new Error(errorNode[0].textContent || 'Invalid XML');
      const nodes = Array.from(doc.getElementsByTagName('waypoint'));
      if (!nodes.length) throw new Error('No <waypoint> elements found.');
      return nodes.map(node => {
        const x = parseFloat(node.getAttribute('x'));
        const y = parseFloat(node.getAttribute('y'));
        const yaw = parseFloat(node.getAttribute('yaw') || '0');
        const z = parseFloat(node.getAttribute('z') || '0');
        const pitch = node.hasAttribute('pitch') ? parseFloat(node.getAttribute('pitch')) : null;
        const roll = node.hasAttribute('roll') ? parseFloat(node.getAttribute('roll')) : null;
        const time = node.hasAttribute('time') ? parseFloat(node.getAttribute('time')) : null;
        const speed = node.hasAttribute('speed') ? parseFloat(node.getAttribute('speed')) : null;
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(yaw)) {
          throw new Error('Waypoints must include numeric x, y, yaw.');
        }
        return {
          x, y, z: Number.isFinite(z) ? z : 0, yaw,
          pitch: Number.isFinite(pitch) ? pitch : null,
          roll: Number.isFinite(roll) ? roll : null,
          time: Number.isFinite(time) ? time : null,
          speed: Number.isFinite(speed) ? speed : null,
          extras: {}
        };
      });
    }
    function sanitizeFileComponent(value) {
      return slug(value || 'item');
    }
    function routeIdForActor(actor) {
      return String((actor.routeAttrs && actor.routeAttrs.id) || '');
    }
    function setRouteIdForActor(actor, routeId) {
      if (!actor.routeAttrs) actor.routeAttrs = {};
      if (routeId) actor.routeAttrs.id = routeId;
      else delete actor.routeAttrs.id;
      setDirty(true);
      renderXmlOutputs();
    }
    function updateWaypointActionButtons() {
      const actor = getActiveActor();
      const hasWaypoints = !!(actor && actor.waypoints.length > 0);
      if (clearActorBtn) clearActorBtn.disabled = !hasWaypoints;
      if (applyOffsetBtn) applyOffsetBtn.disabled = !hasWaypoints;
      if (resetOffsetBtn) resetOffsetBtn.disabled = !hasWaypoints || !hasActiveOffset(actor);
    }
    function setActiveActor(id) {
      pendingHeading = null;
      orientationPreview = null;
      hideNudgeMenu();
      activeActorId = id;
      renderActorSelect();
      renderWaypointTable();
      renderXmlOutputs();
      updatePlot();
      const actor = getActiveActor();
      if (actor) document.getElementById('routeId').value = routeIdForActor(actor);
      setGrpStatus('');
    }
    function removeActorById(id) {
      const idx = state.actors.findIndex(a => a.id === id);
      if (idx === -1) return false;
      pushHistory();
      const removed = state.actors.splice(idx, 1)[0];
      if (removed) delete state.grpPreviewByActorId[removed.id];
      const fallback = state.actors[idx] || state.actors[idx - 1] || state.actors[0] || null;
      setActiveActor(fallback ? fallback.id : null);
      setDirty(true);
      return true;
    }
    function offsetActiveActorWaypoints(dx, dy) {
      const actor = getActiveActor();
      if (!actor || actor.waypoints.length === 0) return false;
      if (!Number.isFinite(dx) || !Number.isFinite(dy)) return false;
      pushHistory();
      const offset = ensureActorOffset(actor);
      actor.waypoints.forEach(wp => {
        wp.x = Number(wp.x) + dx;
        wp.y = Number(wp.y) + dy;
      });
      if (offset) {
        offset.x += dx;
        offset.y += dy;
      }
      setDirty(true);
      return true;
    }
    function resetActiveActorOffset() {
      const actor = getActiveActor();
      if (!actor || actor.waypoints.length === 0) return false;
      const offset = ensureActorOffset(actor);
      if (!offset) return false;
      if (Math.abs(offset.x) < 1e-6 && Math.abs(offset.y) < 1e-6) return false;
      pushHistory();
      actor.waypoints.forEach(wp => {
        wp.x = Number(wp.x) - offset.x;
        wp.y = Number(wp.y) - offset.y;
      });
      offset.x = 0;
      offset.y = 0;
      setDirty(true);
      return true;
    }
    function getStaticModelPresetById(id) {
      return STATIC_MODEL_PRESET_MAP[id] || null;
    }
    function getActiveStaticModelPreset() {
      return getStaticModelPresetById(activeStaticModelPresetId) || STATIC_MODEL_DEFAULT_PRESET;
    }
    function nextStaticModelName(preset) {
      return (preset.label || 'Static') + ' ' + (++staticCounter);
    }
    function updateStaticModelPlacementUi() {
      const preset = getActiveStaticModelPreset();
      const suffix = preset ? (' (' + preset.label + ')') : '';
      if (placeStaticModelBtn) placeStaticModelBtn.textContent = (modelPlacementMode ? 'Click map…' : 'Place 3D model') + suffix;
      if (staticModelHint) {
        staticModelHint.style.display = modelPlacementMode ? 'block' : 'none';
        staticModelHint.textContent = 'Click on the map to place the selected model' + suffix + '. Use the table to fine-tune position and rotation.';
      }
    }
    function setModelPlacementMode(enabled) {
      modelPlacementMode = !!enabled;
      if (placeStaticModelBtn) placeStaticModelBtn.classList.toggle('model-placement-active', modelPlacementMode);
      pendingHeading = null;
      orientationPreview = null;
      updateStaticModelPlacementUi();
      updateOrientationOverlay();
    }
    function createStaticModel(x, y, presetId) {
      pushHistory();
      const preset = getStaticModelPresetById(presetId) || getActiveStaticModelPreset();
      const blueprint = String((preset && preset.blueprint) || 'static.prop.trafficcone01').trim();
      const presetBBox = bboxFromPreset(preset);
      const modelName = nextStaticModelName(preset);
      const model = {
        id: staticModelIdCounter++,
        name: modelName,
        x: x,
        y: y,
        z: 0,
        yaw: 0,
        length: presetBBox ? Number(presetBBox.length) : (Number.isFinite(preset.length) ? preset.length : 1.0),
        width: presetBBox ? Number(presetBBox.width) : (Number.isFinite(preset.width) ? preset.width : 1.0),
        height: presetBBox && presetBBox.height != null ? Number(presetBBox.height) : (Number.isFinite(preset.height) ? preset.height : null),
        blueprint: blueprint,
        file: uniqueRouteFile('static', modelName),
        routeAttrs: {
          id: slug(`${state.activeTown || 'town'}_static_${staticModelIdCounter}`),
          town: state.activeTown || 'Town01',
          role: 'static',
          model: blueprint
        }
      };
      if (presetBBox) applyBboxToModel(model, presetBBox, {persistAttrs: true});
      state.staticModels.push(model);
      renderStaticModelTable();
      updatePlot();
      setDirty(true);
      queueStaticModelBBoxRefresh(model, {markDirty: false, persistAttrs: true});
    }
    function removeStaticModel(id) {
      const idx = state.staticModels.findIndex(m => m.id === id);
      if (idx >= 0) {
        pushHistory();
        state.staticModels.splice(idx, 1);
        renderStaticModelTable();
        updatePlot();
        setDirty(true);
      }
    }
    function clearStaticModels() {
      pushHistory();
      state.staticModels = [];
      staticModelIdCounter = 0;
      staticCounter = 0;
      renderStaticModelTable();
      updatePlot();
      setDirty(true);
    }
    function updateStaticModelField(model, field, value) {
      if (!Number.isFinite(value)) return;
      if (Number(model[field]) === Number(value)) return;
      pushHistory();
      model[field] = value;
      updatePlot();
      renderStaticModelTable();
      setDirty(true);
    }
    function renderStaticModelTable() {
      staticModelTableBody.innerHTML = '';
      if (clearStaticModelsBtn) clearStaticModelsBtn.disabled = state.staticModels.length === 0;
      if (state.staticModels.length === 0) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.style.textAlign = 'center';
        cell.style.padding = '8px';
        cell.textContent = 'No static models yet.';
        row.appendChild(cell);
        staticModelTableBody.appendChild(row);
        return;
      }
      state.staticModels.forEach(model => {
        const row = document.createElement('tr');
        const nameCell = document.createElement('td');
        nameCell.textContent = staticModelDisplayName(model);
        const modelBBox = bboxFromSizeFields(model);
        const bboxText = modelBBox ? `${Number(modelBBox.length).toFixed(2)} x ${Number(modelBBox.width).toFixed(2)}${modelBBox.height != null ? ' x ' + Number(modelBBox.height).toFixed(2) : ''} m` : 'n/a';
        nameCell.title = (model.blueprint || 'unknown model') + ' | bbox ' + bboxText;
        row.appendChild(nameCell);
        const xCell = document.createElement('td');
        const xInput = document.createElement('input');
        xInput.type = 'number';
        xInput.step = '0.1';
        xInput.value = Number(model.x).toFixed(2);
        xInput.addEventListener('change', () => updateStaticModelField(model, 'x', parseFloat(xInput.value)));
        xCell.appendChild(xInput);
        row.appendChild(xCell);
        const yCell = document.createElement('td');
        const yInput = document.createElement('input');
        yInput.type = 'number';
        yInput.step = '0.1';
        yInput.value = Number(model.y).toFixed(2);
        yInput.addEventListener('change', () => updateStaticModelField(model, 'y', parseFloat(yInput.value)));
        yCell.appendChild(yInput);
        row.appendChild(yCell);
        const yawCell = document.createElement('td');
        const yawInput = document.createElement('input');
        yawInput.type = 'number';
        yawInput.step = '1';
        yawInput.value = Number(model.yaw).toFixed(1);
        yawInput.addEventListener('change', () => updateStaticModelField(model, 'yaw', parseFloat(yawInput.value)));
        yawCell.appendChild(yawInput);
        row.appendChild(yawCell);
        const actionsCell = document.createElement('td');
        const removeBtn = document.createElement('button');
        removeBtn.className = 'danger';
        removeBtn.textContent = 'Delete';
        removeBtn.addEventListener('click', () => removeStaticModel(model.id));
        actionsCell.appendChild(removeBtn);
        row.appendChild(actionsCell);
        staticModelTableBody.appendChild(row);
      });
    }
    function computeStaticModelPolygon(model) {
      const halfL = Number(model.length) * 0.5;
      const halfW = Number(model.width) * 0.5;
      const localPoints = [
        {x: halfL, y: halfW},
        {x: -halfL, y: halfW},
        {x: -halfL, y: -halfW},
        {x: halfL, y: -halfW}
      ];
      const rad = Number(model.yaw) * Math.PI / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      return localPoints.map(pt => ({
        x: Number(model.x) + pt.x * cos - pt.y * sin,
        y: Number(model.y) + pt.x * sin + pt.y * cos
      }));
    }
    function updateMapFromPayload(mapPayload) {
      state.mapPayload = mapPayload || null;
      if (!mapPayload) {
        baseTrace.x = [];
        baseTrace.y = [];
        baseTrace.marker.color = 'rgba(200, 200, 200, 0.45)';
        baseTrace.customdata = [];
        layout.xaxis.autorange = true;
        layout.yaxis.autorange = true;
        layout.images = [];
        return;
      }
      baseTrace.x = mapPayload.x || [];
      baseTrace.y = mapPayload.y || [];
      baseTrace.marker.color = mapPayload.colors || mapPayload.lane_colors || baseTrace.marker.color;
      const z = mapPayload.z || [];
      const yaw = mapPayload.yaw || [];
      const lane = mapPayload.lane_id || [];
      const road = mapPayload.road_id || [];
      const direction = mapPayload.lane_direction || [];
      baseTrace.customdata = z.map((_, idx) => [z[idx] || 0, yaw[idx] || 0, lane[idx] || 0, road[idx] || 0, direction[idx] || '']);
      if (MIRROR_X) layout.xaxis.range = [mapPayload.xmax, mapPayload.xmin];
      else layout.xaxis.range = [mapPayload.xmin, mapPayload.xmax];
      if (MIRROR_Y) layout.yaxis.range = [mapPayload.ymax, mapPayload.ymin];
      else layout.yaxis.range = [mapPayload.ymin, mapPayload.ymax];
      layout.xaxis.autorange = false;
      layout.yaxis.autorange = false;
      updateMapBackgroundImage(state.activeTown);
    }
    function updateOrientationPreviewMarker() {
      if (!plotReady) return;
      let traceIndex = 1;
      state.actors.forEach(actor => {
        if (actor.waypoints.length === 0) return;
        const angles = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90);
        const yawValues = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx));
        Plotly.restyle(mapDiv, {'marker.angle': [angles], 'customdata': [yawValues]}, traceIndex);
        traceIndex += 1;
      });
    }
    function screenToData(event) {
      const xaxis = mapDiv._fullLayout.xaxis;
      const yaxis = mapDiv._fullLayout.yaxis;
      if (!xaxis || !yaxis || !Plotly || !Plotly.Axes) return null;
      const rect = mapDiv.getBoundingClientRect();
      const xPixel = event.clientX - rect.left;
      const yPixel = event.clientY - rect.top;
      const xData = Plotly.Axes.p2c(xaxis, xPixel);
      const yData = Plotly.Axes.p2c(yaxis, yPixel);
      if (!isFinite(xData) || !isFinite(yData)) return null;
      return {x: xData, y: yData};
    }
    function dataToScreen(x, y) {
      const xaxis = mapDiv._fullLayout.xaxis;
      const yaxis = mapDiv._fullLayout.yaxis;
      if (!xaxis || !yaxis || !Plotly || !Plotly.Axes) return null;
      const xPixel = Plotly.Axes.c2p(xaxis, Number(x));
      const yPixel = Plotly.Axes.c2p(yaxis, Number(y));
      if (!isFinite(xPixel) || !isFinite(yPixel)) return null;
      return {x: xPixel, y: yPixel};
    }
    function recomputeActorHeadings(actor) {
      if (!actor || !Array.isArray(actor.waypoints) || actor.waypoints.length < 2) return;
      for (let i = 0; i < actor.waypoints.length; i += 1) {
        const cur = actor.waypoints[i];
        let dx = 0, dy = 0;
        if (i < actor.waypoints.length - 1) {
          dx = Number(actor.waypoints[i + 1].x) - Number(cur.x);
          dy = Number(actor.waypoints[i + 1].y) - Number(cur.y);
        } else if (i > 0) {
          dx = Number(cur.x) - Number(actor.waypoints[i - 1].x);
          dy = Number(cur.y) - Number(actor.waypoints[i - 1].y);
        }
        if (Math.abs(dx) > 1e-6 || Math.abs(dy) > 1e-6) {
          cur.yaw = Math.atan2(dy, dx) * 180 / Math.PI;
        }
      }
    }
    function hideNudgeMenu() {
      state.nudgeTarget = null;
      if (!nudgeMenuEl) return;
      nudgeMenuEl.classList.add('hidden');
      nudgeMenuEl.innerHTML = '';
    }
    function hitWaypointForContextMenu(clientX, clientY) {
      const rect = mapDiv.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;
      let best = null;
      let bestDist = Infinity;
      state.actors.forEach(actor => {
        actor.waypoints.forEach((wp, idx) => {
          const p = dataToScreen(wp.x, wp.y);
          if (!p) return;
          const d = Math.hypot(mx - p.x, my - p.y);
          if (d < bestDist) {
            bestDist = d;
            best = {actorId: actor.id, waypointIndex: idx};
          }
        });
      });
      if (best && bestDist <= 16) return best;
      return null;
    }
    function nudgeWaypoint(dx, dy, recenter=false) {
      if (!state.nudgeTarget) return;
      const actor = state.actors.find(a => a.id === state.nudgeTarget.actorId);
      if (!actor) return;
      const idx = state.nudgeTarget.waypointIndex;
      const wp = actor.waypoints[idx];
      if (!wp) return;
      const step = Number(state.nudgeStep || 0.5);
      if (recenter) {
        // no-op recenter in legacy mode (kept for parity of menu button)
      } else {
        pushHistory();
        wp.x = Number(wp.x) + dx * step;
        wp.y = Number(wp.y) + dy * step;
      }
      recomputeActorHeadings(actor);
      setDirty(true);
      renderWaypointTable();
      renderXmlOutputs();
      updatePlot();
    }
    function showNudgeMenu(clientX, clientY, actorId, waypointIndex) {
      if (!nudgeMenuEl) return;
      state.nudgeTarget = {actorId, waypointIndex};
      if (activeActorId !== actorId) setActiveActor(actorId);
      const panelRect = mapDiv.parentElement.getBoundingClientRect();
      const left = Math.min(Math.max(16, clientX - panelRect.left + 12), Math.max(16, panelRect.width - 186));
      const top = Math.min(Math.max(16, clientY - panelRect.top + 12), Math.max(16, panelRect.height - 196));
      nudgeMenuEl.style.left = left + 'px';
      nudgeMenuEl.style.top = top + 'px';
      nudgeMenuEl.innerHTML = `
        <div class="nudge-head">
          <strong>Waypoint ${waypointIndex}</strong>
          <button class="secondary" style="margin:0;padding:3px 8px" onclick="hideNudgeMenu()">Close</button>
        </div>
        <div class="nudge-meta">Step
          <select onchange="state.nudgeStep=Number(this.value)">
            <option value="0.25" ${state.nudgeStep===0.25?'selected':''}>0.25m</option>
            <option value="0.5" ${state.nudgeStep===0.5?'selected':''}>0.5m</option>
            <option value="1" ${state.nudgeStep===1?'selected':''}>1m</option>
            <option value="2" ${state.nudgeStep===2?'selected':''}>2m</option>
          </select>
        </div>
        <div class="nudge-grid">
          <span></span>
          <button onclick="nudgeWaypoint(0,1)">↑</button>
          <span></span>
          <button onclick="nudgeWaypoint(-1,0)">←</button>
          <button onclick="nudgeWaypoint(0,0,true)">⟲</button>
          <button onclick="nudgeWaypoint(1,0)">→</button>
          <span></span>
          <button onclick="nudgeWaypoint(0,-1)">↓</button>
          <span></span>
        </div>
      `;
      nudgeMenuEl.classList.remove('hidden');
    }
    function computeHeadingDegrees(origin, target) {
      return Math.atan2(Number(target.y) - Number(origin.y), Number(target.x) - Number(origin.x)) * 180 / Math.PI;
    }
    function yawForDisplay(actor, index) {
      const wp = actor.waypoints[index];
      if (!wp) return 0;
      if (pendingHeading && orientationPreview && pendingHeading.actorId === actor.id && pendingHeading.index === index) {
        return computeHeadingDegrees(wp, orientationPreview);
      }
      return Number(wp.yaw || 0);
    }
    function updateOrientationOverlay() {
      if (!plotReady) return;
      const shapes = [];
      if (pendingHeading) {
        const actor = state.actors.find(a => a.id === pendingHeading.actorId);
        if (actor) {
          const wp = actor.waypoints[pendingHeading.index];
          if (wp) {
            const radius = 4;
            shapes.push({
              type: 'circle',
              xref: 'x', yref: 'y',
              x0: Number(wp.x) - radius,
              x1: Number(wp.x) + radius,
              y0: Number(wp.y) - radius,
              y1: Number(wp.y) + radius,
              line: {color: '#f1c40f', width: 2, dash: 'dot'}
            });
          }
        }
      }
      Plotly.relayout(mapDiv, {shapes: shapes, annotations: []});
    }
    function handlePlotClick(x, y) {
      const actor = getActiveActor();
      if (!actor) return;
      if (pendingHeading && pendingHeading.actorId === actor.id) {
        const base = actor.waypoints[pendingHeading.index];
        if (base) {
          pushHistory();
          const yaw = Math.atan2(y - Number(base.y), x - Number(base.x)) * 180 / Math.PI;
          if (isFinite(yaw)) base.yaw = yaw;
        }
        pendingHeading = null;
        orientationPreview = null;
        updatePlot();
        renderWaypointTable();
        renderXmlOutputs();
        setDirty(true);
        return;
      }
      pushHistory();
      const prev = actor.waypoints[actor.waypoints.length - 1];
      const yaw = prev ? Number(prev.yaw || 0) : 0;
      const time = prev && prev.time != null ? Number(prev.time) + 1.0 : null;
      const speed = prev && prev.speed != null ? Number(prev.speed) : null;
      actor.waypoints.push({x: x, y: y, z: 0.0, yaw: yaw, pitch: null, roll: null, time: time, speed: speed, extras: {}});
      pendingHeading = {actorId: actor.id, index: actor.waypoints.length - 1};
      orientationPreview = null;
      updatePlot();
      renderWaypointTable();
      renderXmlOutputs();
      setDirty(true);
    }
    function updatePlot() {
      const traces = [baseTrace];
      state.actors.forEach(actor => {
        if (actor.waypoints.length === 0) return;
        const displayName = actorDisplayName(actor);
        const orderLabels = actor.waypoints.map((_, idx) => String(idx + 1));
        const markerStyle = getMarkerStyle(actor);
        traces.push({
          x: actor.waypoints.map(w => Number(w.x)),
          y: actor.waypoints.map(w => Number(w.y)),
          customdata: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx)),
          text: orderLabels,
          textposition: 'middle center',
          textfont: {color: '#111', size: 12, family: '"Segoe UI Semibold", "Segoe UI", Arial, sans-serif'},
          mode: 'lines+markers+text',
          name: displayName + ' (' + actor.kind.toUpperCase() + ')',
          line: {color: actor.color, width: 3},
          marker: {
            color: actor.color,
            size: markerStyle.size,
            symbol: markerStyle.symbol,
            angle: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90),
            line: {color: '#000', width: 0.5}
          },
          hovertemplate:
            'Agent: ' + displayName + ' (' + actor.kind.toUpperCase() + ')<br>' +
            'x=%{x:.2f} m<br>' +
            'y=%{y:.2f} m<br>' +
            'yaw=%{customdata:.1f}°<extra></extra>'
        });
      });
      state.staticModels.forEach(model => {
        const corners = computeStaticModelPolygon(model);
        if (corners.length === 0) return;
        const displayName = staticModelDisplayName(model);
        const bbox = bboxFromSizeFields(model);
        const bboxText = bbox ? `${Number(bbox.length).toFixed(2)} x ${Number(bbox.width).toFixed(2)}${bbox.height != null ? ' x ' + Number(bbox.height).toFixed(2) : ''} m` : 'n/a';
        const xs = corners.map(pt => pt.x);
        const ys = corners.map(pt => pt.y);
        xs.push(corners[0].x);
        ys.push(corners[0].y);
        traces.push({
          x: xs,
          y: ys,
          mode: 'lines',
          fill: 'toself',
          name: displayName + ' (STATIC)',
          line: {color: 'rgba(255, 165, 0, 0.9)', width: 2},
          fillcolor: 'rgba(255, 165, 0, 0.2)',
          hovertemplate:
            displayName +
            '<br>model=' + (model.blueprint || 'unknown') +
            '<br>bbox=' + bboxText +
            '<br>x=%{x:.2f} m<br>y=%{y:.2f} m<br>yaw=' + Number(model.yaw).toFixed(1) + '°<extra></extra>',
          showlegend: false
        });
      });
      state.actors.forEach(actor => {
        const grp = state.grpPreviewByActorId[actor.id];
        if (!grp || !grp.supported || !Array.isArray(grp.dense_points) || grp.dense_points.length < 2) return;
        const displayName = actorDisplayName(actor);
        traces.push({
          x: grp.dense_points.map(p => Number(p.x)),
          y: grp.dense_points.map(p => Number(p.y)),
          mode: 'lines',
          name: displayName + ' GRP',
          line: {color: darkerShade(actor.color, 0.72), width: 3, dash: 'dash'},
          hovertemplate: displayName + ' GRP<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>',
          showlegend: false
        });
      });
      Plotly.react(mapDiv, traces, layout, config).then(() => {
        updateOrientationOverlay();
        updateOrientationPreviewMarker();
      });
    }
    function renderScenarioStatus() {
      const summary = activeScenarioSummary();
      if (!summary) {
        setStatus('No scenario loaded.');
        return;
      }
      const dirty = state.dirty ? ' | unsaved edits' : '';
      setStatus(`${summary.name} | ${state.activeTown || 'unknown town'} | ${state.actors.length + state.staticModels.length} routes${dirty}`);
    }
    function renderActorSelect() {
      const select = document.getElementById('actorSelect');
      select.innerHTML = '';
      state.actors.forEach(actor => {
        const opt = document.createElement('option');
        opt.value = actor.id;
        opt.textContent = actorDisplayName(actor) + ' (' + actor.kind.toUpperCase() + ')';
        if (actor.id === activeActorId) opt.selected = true;
        select.appendChild(opt);
      });
      select.disabled = state.actors.length === 0;
      document.getElementById('removeActorBtn').disabled = state.actors.length === 0;
      updateWaypointActionButtons();
    }
    function renderWaypointTable() {
      const tbody = document.getElementById('waypointTableBody');
      tbody.innerHTML = '';
      const actor = getActiveActor();
      if (!actor || actor.waypoints.length === 0) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');
        cell.colSpan = 5;
        cell.style.textAlign = 'center';
        cell.style.padding = '8px';
        cell.textContent = 'No waypoints yet.';
        row.appendChild(cell);
        tbody.appendChild(row);
        updateWaypointActionButtons();
        return;
      }
      actor.waypoints.forEach((wp, idx) => {
        const row = document.createElement('tr');
        const cellIdx = document.createElement('td');
        cellIdx.textContent = idx;
        row.appendChild(cellIdx);
        const cellX = document.createElement('td');
        const xInput = document.createElement('input');
        xInput.type = 'number';
        xInput.step = '0.01';
        xInput.value = Number(wp.x).toFixed(3);
        xInput.addEventListener('change', () => {
          const val = parseFloat(xInput.value);
          if (!Number.isFinite(val)) { xInput.value = Number(wp.x).toFixed(3); return; }
          if (Number(wp.x) === val) return;
          pushHistory();
          wp.x = val;
          updatePlot();
          renderXmlOutputs();
          setDirty(true);
        });
        cellX.appendChild(xInput);
        row.appendChild(cellX);
        const cellY = document.createElement('td');
        const yInput = document.createElement('input');
        yInput.type = 'number';
        yInput.step = '0.01';
        yInput.value = Number(wp.y).toFixed(3);
        yInput.addEventListener('change', () => {
          const val = parseFloat(yInput.value);
          if (!Number.isFinite(val)) { yInput.value = Number(wp.y).toFixed(3); return; }
          if (Number(wp.y) === val) return;
          pushHistory();
          wp.y = val;
          updatePlot();
          renderXmlOutputs();
          setDirty(true);
        });
        cellY.appendChild(yInput);
        row.appendChild(cellY);
        const cellYaw = document.createElement('td');
        const yawInput = document.createElement('input');
        yawInput.type = 'number';
        yawInput.step = '0.1';
        yawInput.value = Number(wp.yaw || 0).toFixed(2);
        yawInput.addEventListener('change', () => {
          const val = parseFloat(yawInput.value);
          const nextYaw = Number.isFinite(val) ? val : 0;
          if (Number(wp.yaw || 0) === nextYaw) return;
          pushHistory();
          wp.yaw = nextYaw;
          updatePlot();
          renderXmlOutputs();
          setDirty(true);
        });
        cellYaw.appendChild(yawInput);
        row.appendChild(cellYaw);
        const cellActions = document.createElement('td');
        const removeBtn = document.createElement('button');
        removeBtn.textContent = 'Delete';
        removeBtn.className = 'danger';
        removeBtn.addEventListener('click', () => {
          pushHistory();
          actor.waypoints.splice(idx, 1);
          pendingHeading = null;
          orientationPreview = null;
          renderWaypointTable();
          updatePlot();
          renderXmlOutputs();
          setDirty(true);
        });
        cellActions.appendChild(removeBtn);
        row.appendChild(cellActions);
        tbody.appendChild(row);
      });
      updateWaypointActionButtons();
    }
    function renderXmlOutputs() {
      const container = document.getElementById('xmlOutputs');
      container.innerHTML = '';
      state.actors.forEach(actor => {
        const details = document.createElement('details');
        details.open = state.actors.length <= 1;
        const summary = document.createElement('summary');
        summary.textContent = actorDisplayName(actor) + ' [' + actor.kind.toUpperCase() + '] (' + actor.waypoints.length + ' waypoints)';
        details.appendChild(summary);
        const textarea = document.createElement('textarea');
        textarea.className = 'xml-editor';
        let programmaticUpdate = true;
        textarea.value = generateXmlForActor(actor);
        programmaticUpdate = false;
        const status = document.createElement('div');
        status.className = 'status-msg';
        let debounceId = null;
        textarea.addEventListener('input', () => {
          if (programmaticUpdate) { programmaticUpdate = false; return; }
          clearTimeout(debounceId);
          status.textContent = 'Applying...';
          status.className = 'status-msg';
          debounceId = setTimeout(() => {
            try {
              const waypoints = parseXmlToWaypoints(textarea.value);
              pushHistory();
              actor.waypoints = waypoints;
              resetActorOffsetState(actor);
              pendingHeading = null;
              orientationPreview = null;
              renderWaypointTable();
              updatePlot();
              programmaticUpdate = true;
              textarea.value = generateXmlForActor(actor);
              programmaticUpdate = false;
              status.textContent = 'Applied';
              status.className = 'status-msg status-success';
              setDirty(true);
            } catch (err) {
              status.textContent = 'Error: ' + (err.message || err);
              status.className = 'status-msg status-error';
            }
          }, 400);
        });
        const buttonsRow = document.createElement('div');
        const downloadBtn = document.createElement('button');
        downloadBtn.textContent = 'Download XML';
        downloadBtn.addEventListener('click', () => {
          const blob = new Blob([textarea.value], {type: 'application/xml'});
          const fileName = sanitizeFileComponent((state.scenario && state.scenario.name) || 'scenario') + '_' + sanitizeFileComponent(actor.name) + '.xml';
          saveAs(blob, fileName);
        });
        buttonsRow.appendChild(downloadBtn);
        details.appendChild(textarea);
        details.appendChild(buttonsRow);
        details.appendChild(status);
        container.appendChild(details);
      });
    }
    function renderScenarioSelect() {
      const current = state.scenarioId;
      scenarioSelectEl.innerHTML = '';
      state.scenarios.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.id;
        const status = item.status ? ` | ${item.status}` : '';
        const town = item.town ? ` | ${item.town}` : '';
        opt.textContent = item.name + town + status;
        if (item.id === current) opt.selected = true;
        scenarioSelectEl.appendChild(opt);
      });
    }
    function currentScenarioIndex() {
      return state.scenarios.findIndex(item => item.id === state.scenarioId);
    }
    function goPrevScenario() {
      const idx = currentScenarioIndex();
      if (idx > 0) loadScenarioById(state.scenarios[idx - 1].id, false);
    }
    function goNextScenario() {
      const idx = currentScenarioIndex();
      if (idx >= 0 && idx < state.scenarios.length - 1) loadScenarioById(state.scenarios[idx + 1].id, false);
    }
    async function refreshScenarioList() {
      const scenarios = await fetchJson('/api/scenarios');
      state.scenarios = scenarios;
      state.scenarioById = {};
      state.scenarios.forEach(item => { state.scenarioById[item.id] = item; });
      renderScenarioSelect();
    }
    function mapActorFromRoute(route) {
      const kind = String(route.kind || (route.route_attrs && route.route_attrs.role) || 'npc').toLowerCase();
      const resolvedModel = String((route.route_attrs && route.route_attrs.model) || route.resolved_model || '').trim();
      if (kind === 'ego') egoCounter += 1;
      else if (kind === 'pedestrian') pedestrianCounter += 1;
      else if (kind === 'bicycle') bicycleCounter += 1;
      else npcCounter += 1;
      return {
        id: actorIdCounter++,
        kind: kind,
        name: route.name || route.file || ('actor_' + actorIdCounter),
        color: colorPalette[colorIndex++ % colorPalette.length],
        waypoints: clone(route.waypoints || []),
        offsetTotal: {x: 0, y: 0},
        file: route.file || uniqueRouteFile(kind, route.name || ('actor_' + actorIdCounter)),
        routeAttrs: clone(route.route_attrs || {}),
        resolvedModel: resolvedModel,
        supportsGrp: !!route.supports_grp || kind === 'ego' || kind === 'npc'
      };
    }
    function mapStaticFromRoute(route) {
      const wp = (route.waypoints && route.waypoints[0]) ? route.waypoints[0] : {x:0,y:0,z:0,yaw:0};
      const attrs = clone(route.route_attrs || {});
      const blueprint = String(attrs.model || route.resolved_model || 'static.prop.trafficcone01').trim();
      const preset = findStaticPresetByBlueprint(blueprint);
      const persistedBBox = bboxFromRouteAttrs(attrs);
      const routeBBox = normalizeBBoxPayload(route.bbox);
      const presetBBox = bboxFromPreset(preset);
      const bbox = persistedBBox || routeBBox || presetBBox;
      staticCounter += 1;
      const model = {
        id: staticModelIdCounter++,
        name: route.name || ('Static ' + staticCounter),
        x: Number(wp.x || 0),
        y: Number(wp.y || 0),
        z: Number(wp.z || 0),
        yaw: Number(wp.yaw || 0),
        length: bbox ? Number(bbox.length) : (finiteNumber(route.length) != null ? Number(route.length) : (preset && finiteNumber(preset.length) != null ? Number(preset.length) : 1.5)),
        width: bbox ? Number(bbox.width) : (finiteNumber(route.width) != null ? Number(route.width) : (preset && finiteNumber(preset.width) != null ? Number(preset.width) : 0.7)),
        height: bbox && bbox.height != null ? Number(bbox.height) : (preset && finiteNumber(preset.height) != null ? Number(preset.height) : null),
        blueprint: blueprint,
        file: route.file || uniqueRouteFile('static', route.name || ('static_' + staticCounter)),
        routeAttrs: attrs
      };
      if (bbox) {
        applyBboxToModel(model, bbox, {persistAttrs: true});
      } else {
        queueStaticModelBBoxRefresh(model, {markDirty: false, persistAttrs: true});
      }
      return model;
    }
    async function loadScenarioById(id, force) {
      if (!id) return;
      if (state.dirty && !force) {
        const proceed = confirm('Discard unsaved edits and load another scenario?');
        if (!proceed) return;
      }
      setStatus('Loading scenario...');
      try {
        const payload = await fetchJson('/api/scenario?id=' + encodeURIComponent(id));
        let bgPayload = {};
        try {
          bgPayload = await fetchJson('/api/scenario_bg?id=' + encodeURIComponent(id));
        } catch (bgErr) {
          setStatus('Loaded scenario without CARLA map/GRP warmup: ' + bgErr.message, 'status-error');
        }
        state.scenarioId = id;
        state.scenario = payload;
        state.scenarioWeather = clone(payload.weather || {});
        state.activeTown = payload.town || '';
        await loadStaticModelPresets();
        await loadBevForTown(state.activeTown);
        document.getElementById('scenarioName').value = payload.name || '';
        state.actors = [];
        state.staticModels = [];
        state.grpPreviewByActorId = {};
        hideNudgeMenu();
        actorIdCounter = 0;
        staticModelIdCounter = 0;
        colorIndex = 0;
        pendingHeading = null;
        orientationPreview = null;
        (payload.routes || []).forEach(route => {
          const kind = String(route.kind || (route.route_attrs && route.route_attrs.role) || '').toLowerCase();
          if (kind === 'static') state.staticModels.push(mapStaticFromRoute(route));
          else state.actors.push(mapActorFromRoute(route));
        });
        if (state.actors.length) activeActorId = state.actors[0].id;
        else activeActorId = null;
        const actor = getActiveActor();
        document.getElementById('routeId').value = actor ? routeIdForActor(actor) : '';
        updateMapFromPayload(bgPayload.map_payload || null);
        renderActorSelect();
        renderWaypointTable();
        renderStaticModelTable();
        renderWeatherPresetUi();
        renderXmlOutputs();
        updatePlot();
        setDirty(false);
        renderScenarioStatus();
        setStatus(`Loaded ${payload.name} (${state.actors.length + state.staticModels.length} routes).`);
        setGrpStatus('');
        commitBaselineSnapshot();
      } catch (err) {
        setStatus('Failed loading scenario: ' + err.message, 'status-error');
      }
    }
    function normalizeWaypointForSave(wp) {
      return {
        x: Number(wp.x || 0),
        y: Number(wp.y || 0),
        z: Number(wp.z || 0),
        yaw: Number(wp.yaw || 0),
        pitch: wp.pitch == null ? null : Number(wp.pitch),
        roll: wp.roll == null ? null : Number(wp.roll),
        time: wp.time == null ? null : Number(wp.time),
        speed: wp.speed == null ? null : Number(wp.speed),
        extras: (wp.extras && typeof wp.extras === 'object') ? clone(wp.extras) : {}
      };
    }
    function actorToRoute(actor) {
      const attrs = clone(actor.routeAttrs || {});
      attrs.id = attrs.id || slug(`${state.activeTown || 'town'}_${actor.kind}_${actor.name}`);
      attrs.town = attrs.town || state.activeTown || 'Town01';
      attrs.role = actor.kind || attrs.role || 'npc';
      if (!attrs.model) {
        const model = defaultModelForKind(actor.kind);
        if (model) attrs.model = model;
      }
      return {
        actor_id: actor.file,
        file: actor.file,
        kind: actor.kind,
        name: actor.name,
        route_attrs: attrs,
        waypoints: actor.waypoints.map(normalizeWaypointForSave)
      };
    }
    function staticToRoute(model) {
      const attrs = clone(model.routeAttrs || {});
      attrs.id = attrs.id || slug(`${state.activeTown || 'town'}_static_${model.name}`);
      attrs.town = attrs.town || state.activeTown || 'Town01';
      attrs.role = 'static';
      attrs.model = attrs.model || model.blueprint || 'static.prop.trafficcone01';
      const bbox = bboxFromSizeFields(model);
      if (bbox) {
        attrs.bbox_length = Number(bbox.length).toFixed(6);
        attrs.bbox_width = Number(bbox.width).toFixed(6);
        if (bbox.height != null) attrs.bbox_height = Number(bbox.height).toFixed(6);
      }
      return {
        actor_id: model.file,
        file: model.file,
        kind: 'static',
        name: model.name,
        route_attrs: attrs,
        waypoints: [{
          x: Number(model.x || 0),
          y: Number(model.y || 0),
          z: Number(model.z || 0),
          yaw: Number(model.yaw || 0),
          pitch: null,
          roll: null,
          time: null,
          speed: null,
          extras: {}
        }]
      };
    }
    async function saveScenarioInPlace() {
      if (!state.scenarioId || !state.scenario) return false;
      if (!state.actors.length && !state.staticModels.length) {
        alert('No routes to save.');
        return false;
      }
      const routes = [];
      state.actors.forEach(actor => routes.push(actorToRoute(actor)));
      state.staticModels.forEach(model => routes.push(staticToRoute(model)));
      setStatus('Saving scenario in place...');
      try {
        await fetchJson('/api/save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            id: state.scenarioId,
            weather: state.scenarioWeather || {},
            routes: routes,
            review: {note: (state.scenario.review && state.scenario.review.note) || ''}
          })
        });
        await refreshScenarioList();
        await loadScenarioById(state.scenarioId, true);
        commitBaselineSnapshot();
        setStatus('Saved scenario in place.');
        return true;
      } catch (err) {
        setStatus('Save failed: ' + err.message, 'status-error');
        return false;
      }
    }
    async function saveAndNext() {
      const idx = currentScenarioIndex();
      const ok = await saveScenarioInPlace();
      if (!ok) return;
      if (idx >= 0 && idx < state.scenarios.length - 1) {
        await loadScenarioById(state.scenarios[idx + 1].id, true);
      }
    }
    async function setReviewStatus(status) {
      if (!state.scenarioId) return;
      try {
        await fetchJson('/api/review', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({id: state.scenarioId, status: status})
        });
        await refreshScenarioList();
        setStatus('Review status set to ' + status + '.');
      } catch (err) {
        setStatus('Review update failed: ' + err.message, 'status-error');
      }
    }
    async function runGrpForActiveActor() {
      const egoActors = state.actors.filter(actor => isEgoActor(actor));
      if (!egoActors.length) {
        setGrpStatus('No ego routes found for GRP.');
        return;
      }
      const runnable = egoActors.filter(actor => actor.supportsGrp && actor.waypoints && actor.waypoints.length >= 2);
      if (!runnable.length) {
        setGrpStatus('Need at least 2 waypoints on ego routes for GRP.');
        return;
      }
      setGrpStatus(`Running GRP for ${runnable.length} ego route${runnable.length === 1 ? '' : 's'}…`);

      let ready = 0;
      let failed = 0;
      let skipped = 0;
      let tracePoints = 0;

      for (const actor of egoActors) {
        if (!actor.supportsGrp || !actor.waypoints || actor.waypoints.length < 2) {
          skipped += 1;
          continue;
        }
        try {
          const preview = await fetchJson('/api/grp_preview', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              town: (actor.routeAttrs && actor.routeAttrs.town) || state.activeTown,
              role: (actor.routeAttrs && actor.routeAttrs.role) || actor.kind,
              waypoints: actor.waypoints.map(normalizeWaypointForSave)
            })
          });
          state.grpPreviewByActorId[actor.id] = preview;
          if (preview.supported) {
            ready += 1;
            tracePoints += Number((preview.metrics && preview.metrics.trace_points) || 0);
          } else {
            skipped += 1;
          }
        } catch (_err) {
          failed += 1;
        }
      }

      updatePlot();
      if (failed > 0) {
        setGrpStatus(`GRP done | ready ${ready} | skipped ${skipped} | failed ${failed}`);
      } else {
        setGrpStatus(`GRP ready for ${ready} ego route${ready === 1 ? '' : 's'} | total trace ${tracePoints} pts`, true);
      }
    }
    function adoptGrpForActiveActor() {
      const actor = getActiveActor();
      if (!actor) return;
      const preview = state.grpPreviewByActorId[actor.id];
      if (!preview || !preview.supported || !Array.isArray(preview.aligned_waypoints) || !preview.aligned_waypoints.length) {
        setGrpStatus('No GRP result to adopt.');
        return;
      }
      pushHistory();
      actor.waypoints = clone(preview.aligned_waypoints);
      pendingHeading = null;
      orientationPreview = null;
      renderWaypointTable();
      renderXmlOutputs();
      updatePlot();
      setDirty(true);
      setGrpStatus('Adopted GRP-aligned waypoints.', true);
    }

    if (staticModelPresetSelect) {
      renderStaticModelPresetOptions();
      staticModelPresetSelect.addEventListener('change', () => {
        const nextPreset = getStaticModelPresetById(staticModelPresetSelect.value);
        if (!nextPreset) return;
        activeStaticModelPresetId = nextPreset.id;
        updateStaticModelPlacementUi();
      });
    }
    if (weatherPresetNightBtn) {
      weatherPresetNightBtn.addEventListener('click', () => {
        setScenarioWeather(quickWeatherPresetPayload('night'), 'Applied weather preset: night.');
      });
    }
    if (weatherPresetCloudyBtn) {
      weatherPresetCloudyBtn.addEventListener('click', () => {
        setScenarioWeather(quickWeatherPresetPayload('cloudy'), 'Applied weather preset: cloudy.');
      });
    }
    if (weatherPresetRainBtn) {
      weatherPresetRainBtn.addEventListener('click', () => {
        setScenarioWeather(quickWeatherPresetPayload('rain'), 'Applied weather preset: rain.');
      });
    }
    if (weatherPresetDefaultBtn) {
      weatherPresetDefaultBtn.addEventListener('click', () => {
        setScenarioWeather({}, 'Cleared weather override.');
      });
    }
    if (openBevBtn) {
      openBevBtn.addEventListener('click', () => {
        if (openBevBtn.disabled) return;
        if (bevOverlayOpen) closeBevOverlay();
        else showBevOverlay();
      });
    }
    if (bevOverlayClose) {
      bevOverlayClose.addEventListener('click', closeBevOverlay);
    }
    if (bevOverlayHeader) {
      bevOverlayHeader.addEventListener('mousedown', (event) => {
        if (event.button !== 0 || !bevOverlay || bevOverlay.classList.contains('hidden')) return;
        event.preventDefault();
        const rect = bevOverlay.getBoundingClientRect();
        bevOverlayDragState = {
          offsetX: event.clientX - rect.left,
          offsetY: event.clientY - rect.top
        };
      });
    }
    if (bevOverlayResize) {
      bevOverlayResize.addEventListener('mousedown', (event) => {
        if (event.button !== 0 || !bevOverlay || bevOverlay.classList.contains('hidden')) return;
        event.preventDefault();
        const rect = bevOverlay.getBoundingClientRect();
        bevOverlayResizeState = {
          startWidth: rect.width,
          startX: event.clientX
        };
      });
    }
    if (bevOverlayImage) {
      bevOverlayImage.addEventListener('load', () => {
        const naturalWidth = bevOverlayImage.naturalWidth;
        const naturalHeight = bevOverlayImage.naturalHeight;
        if (naturalWidth > 0 && naturalHeight > 0) bevOverlayAspect = naturalWidth / naturalHeight;
        else bevOverlayAspect = 1;
        if (bevOverlay && !bevOverlay.classList.contains('hidden')) {
          const currentWidth = bevOverlay.dataset.width ? parseFloat(bevOverlay.dataset.width) : undefined;
          applyOverlaySize(currentWidth || OVERLAY_DEFAULT_WIDTH);
        }
        updateOverlayZoomUI();
      });
      bevOverlayImage.addEventListener('wheel', (event) => {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.1 : 0.9;
        setOverlayScale(bevOverlayScale * factor);
      }, {passive: false});
    }
    if (bevZoomInBtn) {
      bevZoomInBtn.addEventListener('click', () => {
        setOverlayScale(bevOverlayScale + OVERLAY_SCALE_STEP);
      });
    }
    if (bevZoomOutBtn) {
      bevZoomOutBtn.addEventListener('click', () => {
        setOverlayScale(bevOverlayScale - OVERLAY_SCALE_STEP);
      });
    }
    window.addEventListener('mousemove', (event) => {
      if (bevOverlayDragState && bevOverlay && !bevOverlay.classList.contains('hidden')) {
        const left = event.clientX - bevOverlayDragState.offsetX;
        const top = event.clientY - bevOverlayDragState.offsetY;
        bevOverlay.style.left = Math.max(10, left) + 'px';
        bevOverlay.style.top = Math.max(10, top) + 'px';
      } else if (bevOverlayResizeState && bevOverlay && !bevOverlay.classList.contains('hidden')) {
        const deltaX = event.clientX - bevOverlayResizeState.startX;
        const newWidth = bevOverlayResizeState.startWidth + deltaX;
        applyOverlaySize(newWidth);
      }
    });
    window.addEventListener('mouseup', () => {
      bevOverlayDragState = null;
      bevOverlayResizeState = null;
    });
    updateBevButtonState(false);
    updateOverlayZoomUI();

    mapDiv.addEventListener('mousemove', function(event) {
      if (!pendingHeading) return;
      const coords = screenToData(event);
      if (!coords) return;
      orientationPreview = coords;
      updateOrientationOverlay();
      updateOrientationPreviewMarker();
    });

    mapDiv.on('plotly_click', function(data) {
      if (data.event) data.event[PLOTLY_CLICK_FLAG] = true;
      const evt = data.event || {};
      if (evt.button && evt.button !== 0) return;
      if (!data.points || data.points.length === 0) return;
      const pt = data.points[0];
      if (typeof pt.x !== 'number' || typeof pt.y !== 'number') return;
      if (modelPlacementMode) {
        createStaticModel(pt.x, pt.y, activeStaticModelPresetId);
        setModelPlacementMode(false);
        return;
      }
      handlePlotClick(pt.x, pt.y);
    });

    mapDiv.addEventListener('click', function(event) {
      if (!(event instanceof MouseEvent)) return;
      if (event.button !== 0) return;
      if (event.detail && event.detail > 1) return;
      if (event[PLOTLY_CLICK_FLAG]) return;
      hideNudgeMenu();
      if (modelPlacementMode) return;
      const target = event.target;
      if (target && typeof target.closest === 'function') {
        if (target.closest('.modebar')) return;
        if (target.closest('.scatterlayer')) return;
      }
      const actor = getActiveActor();
      if (!actor || actor.kind !== 'pedestrian') return;
      const coords = screenToData(event);
      if (!coords) return;
      handlePlotClick(coords.x, coords.y);
    }, true);
    mapDiv.addEventListener('contextmenu', (event) => {
      event.preventDefault();
      const hit = hitWaypointForContextMenu(event.clientX, event.clientY);
      if (!hit) {
        hideNudgeMenu();
        return;
      }
      showNudgeMenu(event.clientX, event.clientY, hit.actorId, hit.waypointIndex);
    });

    document.getElementById('actorSelect').addEventListener('change', (evt) => {
      setActiveActor(parseInt(evt.target.value));
    });
    document.getElementById('routeId').addEventListener('input', () => {
      const actor = getActiveActor();
      if (!actor) return;
      const nextRouteId = document.getElementById('routeId').value;
      const currentRouteId = routeIdForActor(actor);
      if (nextRouteId === currentRouteId) return;
      pushHistory();
      setRouteIdForActor(actor, nextRouteId);
    });
    document.getElementById('scenarioName').addEventListener('input', () => {
      setDirty(true);
    });
    document.getElementById('addEgoBtn').addEventListener('click', () => {
      pushHistory();
      const actor = createActor('ego');
      state.actors.push(actor);
      setActiveActor(actor.id);
      renderXmlOutputs();
      updatePlot();
      setDirty(true);
    });
    document.getElementById('addNpcBtn').addEventListener('click', () => {
      pushHistory();
      const actor = createActor('npc');
      state.actors.push(actor);
      setActiveActor(actor.id);
      renderXmlOutputs();
      updatePlot();
      setDirty(true);
    });
    document.getElementById('addPedBtn').addEventListener('click', () => {
      pushHistory();
      const actor = createActor('pedestrian');
      state.actors.push(actor);
      setActiveActor(actor.id);
      renderXmlOutputs();
      updatePlot();
      setDirty(true);
    });
    document.getElementById('addBikeBtn').addEventListener('click', () => {
      pushHistory();
      const actor = createActor('bicycle');
      state.actors.push(actor);
      setActiveActor(actor.id);
      renderXmlOutputs();
      updatePlot();
      setDirty(true);
    });
    document.getElementById('removeActorBtn').addEventListener('click', () => {
      const actor = getActiveActor();
      if (!actor) return;
      removeActorById(actor.id);
      renderWaypointTable();
      renderXmlOutputs();
      updatePlot();
    });
    if (clearActorBtn) {
      clearActorBtn.addEventListener('click', () => {
        const actor = getActiveActor();
        if (!actor) return;
        pushHistory();
        actor.waypoints = [];
        resetActorOffsetState(actor);
        pendingHeading = null;
        orientationPreview = null;
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
        setDirty(true);
      });
    }
    document.getElementById('resetAllBtn').addEventListener('click', () => {
      if (!confirm('Clear all agents and waypoints for this scenario view?')) return;
      pushHistory();
      state.actors = [];
      state.staticModels = [];
      state.grpPreviewByActorId = {};
      actorIdCounter = 0;
      staticModelIdCounter = 0;
      egoCounter = 0;
      npcCounter = 0;
      pedestrianCounter = 0;
      bicycleCounter = 0;
      staticCounter = 0;
      colorIndex = 0;
      activeActorId = null;
      pendingHeading = null;
      orientationPreview = null;
      renderActorSelect();
      renderWaypointTable();
      renderStaticModelTable();
      updatePlot();
      renderXmlOutputs();
      setDirty(true);
      setGrpStatus('');
    });
    if (undoBtn) undoBtn.addEventListener('click', undoEdit);
    if (undoAllBtn) undoAllBtn.addEventListener('click', undoAllEdits);
    document.getElementById('runGrpBtn').addEventListener('click', runGrpForActiveActor);
    document.getElementById('adoptGrpBtn').addEventListener('click', adoptGrpForActiveActor);
    if (placeStaticModelBtn) {
      placeStaticModelBtn.addEventListener('click', () => setModelPlacementMode(!modelPlacementMode));
    }
    if (clearStaticModelsBtn) {
      clearStaticModelsBtn.addEventListener('click', () => {
        if (state.staticModels.length === 0) return;
        if (!confirm('Remove all static models?')) return;
        clearStaticModels();
        setModelPlacementMode(false);
      });
    }
    if (applyOffsetBtn) {
      applyOffsetBtn.addEventListener('click', () => {
        const dx = parseFloat(offsetXInput.value);
        const dy = parseFloat(offsetYInput.value);
        if (!Number.isFinite(dx) || !Number.isFinite(dy)) {
          alert('Offsets must be numeric values.');
          return;
        }
        if (!offsetActiveActorWaypoints(dx, dy)) {
          alert('No waypoints available for the active agent.');
          return;
        }
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
      });
    }
    if (resetOffsetBtn) {
      resetOffsetBtn.addEventListener('click', () => {
        const changed = resetActiveActorOffset();
        if (changed) {
          renderWaypointTable();
          updatePlot();
          renderXmlOutputs();
        }
        offsetXInput.value = '0';
        offsetYInput.value = '0';
      });
    }
    document.getElementById('downloadAllBtn').addEventListener('click', () => {
      if (state.actors.length === 0 && state.staticModels.length === 0) {
        alert('No routes to export.');
        return;
      }
      const scenarioNameRaw = document.getElementById('scenarioName').value || 'legacy_scenario';
      const scenarioNameSafe = sanitizeFileComponent(scenarioNameRaw);
      const zip = new JSZip();
      state.actors.forEach(actor => {
        const xmlText = generateXmlForActor(actor);
        const actorSafe = sanitizeFileComponent(actor.name);
        zip.file(scenarioNameSafe + '_' + actorSafe + '.xml', xmlText);
      });
      state.staticModels.forEach(model => {
        const attrs = model.routeAttrs || {};
        const routeId = attrs.id || '0';
        const town = attrs.town || state.activeTown || 'Town01';
        const modelName = attrs.model || model.blueprint || 'static.prop.trafficcone01';
        const bbox = bboxFromSizeFields(model);
        const routeAttrs = {...attrs, id: routeId, town: town, role: 'static', model: modelName};
        if (bbox) {
          routeAttrs.bbox_length = Number(bbox.length).toFixed(6);
          routeAttrs.bbox_width = Number(bbox.width).toFixed(6);
          if (bbox.height != null) routeAttrs.bbox_height = Number(bbox.height).toFixed(6);
        }
        const routeAttrText = Object.keys(routeAttrs)
          .filter(key => routeAttrs[key] != null && String(routeAttrs[key]).trim() !== '')
          .map(key => `${key}="${escapeXmlAttr(routeAttrs[key])}"`)
          .join(' ');
        const xmlText = [
          "<?xml version='1.0' encoding='utf-8'?>",
          '<routes>',
          `  <route ${routeAttrText}>`,
          `    <waypoint x="${Number(model.x).toFixed(6)}" y="${Number(model.y).toFixed(6)}" z="${Number(model.z || 0).toFixed(6)}" yaw="${Number(model.yaw).toFixed(6)}" />`,
          '  </route>',
          '</routes>'
        ].join('\n');
        zip.file(scenarioNameSafe + '_' + sanitizeFileComponent(model.name) + '_static.xml', xmlText);
      });
      zip.generateAsync({type:'blob'}).then(content => {
        saveAs(content, scenarioNameSafe + '_routes.zip');
      });
    });
    document.getElementById('prevScenarioBtn').addEventListener('click', goPrevScenario);
    document.getElementById('nextScenarioBtn').addEventListener('click', goNextScenario);
    document.getElementById('reloadScenarioBtn').addEventListener('click', () => {
      if (state.scenarioId) loadScenarioById(state.scenarioId, true);
    });
    document.getElementById('saveScenarioBtn').addEventListener('click', saveScenarioInPlace);
    document.getElementById('saveNextScenarioBtn').addEventListener('click', saveAndNext);
    document.getElementById('approveBtn').addEventListener('click', () => setReviewStatus('approved'));
    document.getElementById('rejectBtn').addEventListener('click', () => setReviewStatus('rejected'));
    document.getElementById('editedBtn').addEventListener('click', () => setReviewStatus('edited'));
    scenarioSelectEl.addEventListener('change', () => {
      const id = scenarioSelectEl.value;
      if (id) loadScenarioById(id, false);
    });
    window.addEventListener('beforeunload', (evt) => {
      if (state.dirty) {
        evt.preventDefault();
        evt.returnValue = '';
      }
    });
    window.addEventListener('mousedown', (event) => {
      if (!state.nudgeTarget) return;
      if (nudgeMenuEl && nudgeMenuEl.contains(event.target)) return;
      hideNudgeMenu();
    });
    window.addEventListener('keydown', (event) => {
      const isMod = event.ctrlKey || event.metaKey;
      if (isMod && event.key.toLowerCase() === 's') {
        event.preventDefault();
        saveScenarioInPlace();
      }
      if (isMod && event.key.toLowerCase() === 'z' && !event.shiftKey) {
        event.preventDefault();
        undoEdit();
      }
      if (isMod && event.key === 'Enter') {
        event.preventDefault();
        saveAndNext();
      }
      if (event.key === 'Escape') {
        if (bevOverlayOpen) closeBevOverlay();
        else if (modelPlacementMode) setModelPlacementMode(false);
        hideNudgeMenu();
      }
    });

    async function init() {
      updateUndoButtons();
      await loadWeatherPresets();
      await loadStaticModelPresets();
      updateStaticModelPlacementUi();
      renderStaticModelTable();
      renderWeatherPresetUi();
      await refreshScenarioList();
      if (state.scenarios.length > 0) {
        await loadScenarioById(state.scenarios[0].id, true);
      } else {
        setStatus('No scenarios found.');
      }
    }

    init().catch(err => {
      setStatus('Initialization failed: ' + err.message, 'status-error');
    });
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                self._send(200, "text/html; charset=utf-8", LEGACY_HTML.encode("utf-8"))
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
            if path == "/api/static_prop_presets":
                town = str(query.get("town", [""])[0]).strip() or None
                self._send_json(200, _static_prop_presets_payload(town))
                return
            if path == "/api/asset_bbox":
                model = str(query.get("model", [""])[0]).strip()
                town = str(query.get("town", [""])[0]).strip() or None
                if not model:
                    self._send_json(400, {"error": "missing model"})
                    return
                self._send_json(200, _asset_bbox_payload(model, town))
                return
            if path == "/api/weather_presets":
                self._send_json(200, _weather_preset_payload())
                return
            if path == "/api/carla_status":
                status = APP.carla.status()
                status["prewarm"] = APP.prewarm_status()
                self._send_json(200, status)
                return
            if path == "/api/legacy_bev":
                town = str(query.get("town", [""])[0])
                if not town:
                    self._send_json(400, {"error": "missing town"})
                    return
                info = {"found": False, "town": town} if LEGACY_BEV is None else LEGACY_BEV.town_info(town)
                self._send_json(200, info)
                return
            if path == "/api/legacy_bev_image":
                town = str(query.get("town", [""])[0])
                if not town:
                    self._send_text(400, "missing town")
                    return
                png = None if LEGACY_BEV is None else LEGACY_BEV.town_png_bytes(town)
                if png is None:
                    self._send_text(404, f"no legacy BEV image for {town}")
                    return
                self._send(200, "image/png", png)
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
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios_dir", help="Directory containing scenario subdirectories, or a single scenario directory")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8892, help="HTTP port to bind (default: 8892)")
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
        choices=("none", "seam", "kink", "legacy"),
        default=None,
        help=(
            "Override CARLA GRP postprocess mode for route interpolation. "
            "'none' disables postprocessing."
        ),
    )
    parser.add_argument(
        "--grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_true",
        default=True,
        help="Set CARLA_GRP_PP_IGNORE_ENDPOINTS=1 (default).",
    )
    parser.add_argument(
        "--no-grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_false",
        help="Set CARLA_GRP_PP_IGNORE_ENDPOINTS=0.",
    )
    parser.add_argument(
        "--prewarm-grp",
        action="store_true",
        default=False,
        help="On startup, pre-compute GRP previews for all ego routes in background.",
    )
    parser.add_argument(
        "--bev-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing legacy top-down BEV PNGs named <Town>.png "
            "and optional metadata sidecars (<Town>.png.meta.json or <Town>.json). "
            "If omitted, auto-detects common legacy output locations."
        ),
    )
    parser.add_argument(
        "--bev-cache",
        type=str,
        default=None,
        metavar="DIR",
        help="Path to birdview_v2_cache Carla/Maps directory containing .npy top-down town images.",
    )
    parser.set_defaults(auto_launch_carla=True)
    return parser.parse_args()


def main() -> None:
    global APP, LEGACY_BEV
    args = parse_args()
    scenario_root = Path(args.scenarios_dir).expanduser().resolve()
    print("Starting legacy scenario builder...", flush=True)
    print(f"Scenario root: {scenario_root}", flush=True)
    APP = SB.ScenarioBuilderApp(args)
    bev_dir = None
    if args.bev_dir is not None:
        bev_dir = args.bev_dir.expanduser().resolve()
    else:
        bev_dir = _find_default_legacy_bev_dir()
    LEGACY_BEV = LegacyBEVStore(bev_dir)
    if bev_dir is not None:
        print(f"Legacy BEV directory: {bev_dir}", flush=True)
    else:
        print("Legacy BEV directory: not found (preview disabled unless --bev-dir is provided)", flush=True)

    startup_launch_error: str | None = None
    if APP.carla.auto_launch:
        carla_root_text = str(APP.carla.carla_root) if APP.carla.carla_root is not None else "not found"
        print(
            f"Checking CARLA at {APP.carla.host}:{APP.carla.port} (auto-launch enabled, root={carla_root_text})...",
            flush=True,
        )
        try:
            APP.carla.ensure_ready(restart_if_managed=False)
        except Exception as exc:  # noqa: BLE001
            startup_launch_error = str(exc)
            print(f"CARLA warm startup failed: {startup_launch_error}", flush=True)

    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(f"Legacy scenario builder listening on http://{args.host}:{args.port}")
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
