#!/usr/bin/env python3
"""
Generate top-down renderings for every installed CARLA town.

Two modes are available:
- rendered: place an RGB camera high above the town and capture an actual
  Unreal Engine screenshot (default).
- mask: reuse bird's-eye masks to create a schematic lane/road visualization.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, NamedTuple, Tuple

import numpy as np
from PIL import Image

try:
    import carla  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


# Allow importing the vendored carla_birdeye_view utilities.
REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "simulation" / "leaderboard" / "team_code" / "utils"
if str(UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(UTILS_PATH))

try:
    from carla_birdeye_view.colors import RGB
    from carla_birdeye_view.mask import COLOR_ON, MapMaskGenerator
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(
        "Failed to import carla_birdeye_view utilities. "
        "Install the dependencies listed in simulation/leaderboard/requirements.txt."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--towns",
        nargs="*",
        default=None,
        help="Optional subset of towns to process (e.g., Town05 Town10HD).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/carla_town_bev"),
        help="Directory where BEV images are stored (default: results/carla_town_bev).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if the destination file already exists.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Optional scaling factor applied after rendering (default: 1.0 - no change).",
    )
    parser.add_argument(
        "--mode",
        choices=("rendered", "mask"),
        default="rendered",
        help="Type of output to generate. 'rendered' captures a real CARLA screenshot.",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=1.0,
        help="Mask mode only: rendering density for the static masks.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=2048,
        help="Rendered mode: camera width in pixels.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=2048,
        help="Rendered mode: camera height in pixels.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Rendered mode: vertical field-of-view in degrees.",
    )
    parser.add_argument(
        "--altitude-buffer",
        type=float,
        default=40.0,
        help="Rendered mode: extra metres added to the computed camera height.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=120.0,
        help="Rendered mode: minimum camera height above the map centre.",
    )
    parser.add_argument(
        "--waypoint-spacing",
        type=float,
        default=25.0,
        help="Rendered mode: spacing in metres for sampling map bounds.",
    )
    parser.add_argument(
        "--spectator",
        action="store_true",
        help="Rendered mode: also position the spectator at the capture transform.",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="ClearNoon",
        help="Rendered mode: CARLA WeatherParameters preset name (e.g., ClearNoon, CloudySunset). Ignored when --keep-weather is set.",
    )
    parser.add_argument(
        "--keep-weather",
        action="store_true",
        help="Rendered mode: do not force the weather to ClearNoon.",
    )
    parser.add_argument(
        "--exposure-mode",
        type=str,
        default="auto",
        help="Rendered mode: camera exposure mode (auto, manual, histogram, logarithmic).",
    )
    parser.add_argument(
        "--exposure-compensation",
        type=float,
        default=-0.1,
        help="Rendered mode: exposure compensation applied when supported (lower values darken the image).",
    )
    parser.add_argument(
        "--shutter-speed",
        type=float,
        default=100.0,
        help="Rendered mode: manual shutter speed in CARLA units (microseconds).",
    )
    parser.add_argument(
        "--iso",
        type=float,
        default=220.0,
        help="Rendered mode: manual ISO value (lower = darker).",
    )
    parser.add_argument(
        "--fstop",
        type=float,
        default=7.1,
        help="Rendered mode: manual aperture f-stop (higher = darker).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.1,
        help="Rendered mode: tone-mapping gamma value.",
    )
    parser.add_argument(
        "--disable-postprocess",
        action="store_true",
        help="Rendered mode: disable postprocess effects on the capture sensor.",
    )
    return parser.parse_args()


def gather_towns(client: carla.Client, requested: Iterable[str] | None) -> List[str]:
    available = sorted({path.split("/")[-1] for path in client.get_available_maps()})
    print("Available CARLA towns:", ", ".join(available))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested towns not installed: {', '.join(missing)}")
        return [town for town in available if town in requested]
    return available


def compose_static_bev(road_mask: np.ndarray, lane_mask: np.ndarray, center_mask: np.ndarray) -> np.ndarray:
    """Combine road, lane, and centerline masks into an RGB BEV canvas."""
    height, width = road_mask.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas[road_mask == COLOR_ON] = RGB.DARK_GRAY
    canvas[lane_mask == COLOR_ON] = RGB.WHITE
    canvas[center_mask == COLOR_ON] = RGB.CHOCOLATE
    return canvas


def maybe_downscale(image: Image.Image, factor: float) -> Image.Image:
    if factor >= 0.999:
        return image
    if factor <= 0.0:
        raise ValueError("--downscale must be positive.")
    width = max(1, int(round(image.width * factor)))
    height = max(1, int(round(image.height * factor)))
    return image.resize((width, height), resample=Image.BILINEAR)


def render_mask_bev(
    client: carla.Client,
    town: str,
    pixels_per_meter: float,
) -> tuple[Image.Image, MapBounds]:
    world = client.load_world(town)
    world.wait_for_tick()

    ppm = max(0.1, pixels_per_meter)
    mask_generator = MapMaskGenerator(client, pixels_per_meter=ppm)
    mask_generator.disable_local_rendering_mode()

    road_mask = mask_generator.road_mask()
    lane_mask = mask_generator.lanes_mask()
    center_mask = mask_generator.centerlines_mask()

    rgb = compose_static_bev(road_mask, lane_mask, center_mask)

    boundaries = mask_generator._map_boundaries
    bounds = MapBounds(
        min_x=boundaries.min_x,
        max_x=boundaries.max_x,
        min_y=boundaries.min_y,
        max_y=boundaries.max_y,
    )
    return Image.fromarray(rgb), bounds


class MapBounds(NamedTuple):
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.min_x + self.max_x) * 0.5, (self.min_y + self.max_y) * 0.5


@dataclass
class RenderCapture:
    image: Image.Image
    camera_transform: carla.Transform
    intrinsic: np.ndarray
    bounds: MapBounds


def compute_map_bounds(carla_map: carla.Map, spacing: float) -> MapBounds:
    if spacing <= 0.0:
        raise ValueError("--waypoint-spacing must be positive.")
    waypoints = carla_map.generate_waypoints(distance=spacing)
    if not waypoints:
        raise RuntimeError("Failed to sample map waypoints; cannot compute bounds.")

    xs = [wp.transform.location.x for wp in waypoints]
    ys = [wp.transform.location.y for wp in waypoints]
    margin = 20.0
    return MapBounds(
        min_x=min(xs) - margin,
        max_x=max(xs) + margin,
        min_y=min(ys) - margin,
        max_y=max(ys) + margin,
    )


def ensure_synchronous(world: carla.World) -> Tuple[bool, carla.WorldSettings]:
    original = world.get_settings()
    if original.no_rendering_mode:
        raise RuntimeError("World is in no_rendering_mode; cannot capture screenshots.")

    if original.synchronous_mode:
        return False, original

    new_settings = carla.WorldSettings()
    if hasattr(new_settings, "no_rendering_mode"):
        new_settings.no_rendering_mode = getattr(original, "no_rendering_mode", False)
    new_settings.synchronous_mode = True
    if hasattr(new_settings, "fixed_delta_seconds"):
        new_settings.fixed_delta_seconds = getattr(original, "fixed_delta_seconds", None) or 0.05

    for attr in ("substepping", "max_substeps", "max_substep_delta_time"):
        if hasattr(new_settings, attr) and hasattr(original, attr):
            setattr(new_settings, attr, getattr(original, attr))

    world.apply_settings(new_settings)
    return True, original


def build_intrinsic_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    fov_rad = math.radians(fov_deg)
    focal = height / (2.0 * math.tan(fov_rad / 2.0))
    fx = focal
    fy = focal
    cx = width * 0.5
    cy = height * 0.5
    intrinsic = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsic


def safe_set_attribute(blueprint: carla.ActorBlueprint, name: str, value: object) -> None:
    if blueprint.has_attribute(name):
        blueprint.set_attribute(name, str(value))


def resolve_weather(preset_name: str) -> carla.WeatherParameters:
    preset_name = preset_name.strip()
    candidates = {
        key: getattr(carla.WeatherParameters, key)
        for key in dir(carla.WeatherParameters)
        if key[0].isupper() and isinstance(getattr(carla.WeatherParameters, key), carla.WeatherParameters)
    }
    lookup = {key.lower(): key for key in candidates.keys()}
    chosen_key = lookup.get(preset_name.lower())
    if chosen_key is None:
        raise ValueError(
            f"Unknown weather preset '{preset_name}'. "
            f"Available: {', '.join(sorted(candidates.keys()))}"
        )
    return candidates[chosen_key]


def capture_rendered_bev(
    client: carla.Client,
    town: str,
    *,
    width: int,
    height: int,
    fov: float,
    altitude_buffer: float,
    min_height: float,
    waypoint_spacing: float,
    set_spectator: bool,
    keep_weather: bool,
    weather: str,
    exposure_mode: str,
    exposure_compensation: float,
    shutter_speed: float,
    iso: float,
    fstop: float,
    gamma: float,
    disable_postprocess: bool,
) -> RenderCapture:
    world = client.load_world(town)
    world.wait_for_tick()

    if not keep_weather:
        weather_params = resolve_weather(weather)
        world.set_weather(weather_params)

    carla_map = world.get_map()
    bounds = compute_map_bounds(carla_map, waypoint_spacing)
    centre_x, centre_y = bounds.centre

    max_half_extent = 0.5 * max(bounds.width, bounds.height)
    fov_rad = math.radians(max(1.0, min(fov, 175.0)))
    tan_half = math.tan(fov_rad / 2.0)
    altitude = max_half_extent / max(tan_half, 1e-3) + altitude_buffer
    altitude = max(altitude, min_height)

    camera_transform = carla.Transform(
        carla.Location(x=centre_x, y=centre_y, z=altitude),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )

    if set_spectator:
        world.get_spectator().set_transform(camera_transform)

    synchronous_enabled, original_settings = ensure_synchronous(world)
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", f"{fov:.2f}")
    camera_bp.set_attribute("sensor_tick", "0.0")
    safe_set_attribute(camera_bp, "exposure_mode", exposure_mode)
    safe_set_attribute(camera_bp, "exposure_compensation", exposure_compensation)
    safe_set_attribute(camera_bp, "shutter_speed", shutter_speed)
    safe_set_attribute(camera_bp, "iso", iso)
    safe_set_attribute(camera_bp, "fstop", fstop)
    safe_set_attribute(camera_bp, "gamma", max(0.1, gamma))
    if disable_postprocess:
        safe_set_attribute(camera_bp, "enable_postprocess_effects", "false")

    sensor = world.spawn_actor(camera_bp, camera_transform)
    image_queue: queue.Queue[carla.Image] = queue.Queue()
    sensor.listen(image_queue.put)

    try:
        # Warm up a few frames to allow textures to settle.
        for _ in range(3):
            if synchronous_enabled:
                world.tick()
            else:
                world.wait_for_tick()
                time.sleep(0.05)

        try:
            image = image_queue.get(timeout=5.0)
        except queue.Empty as exc:
            raise RuntimeError(f"Timed out waiting for camera data in {town}") from exc
    finally:
        sensor.stop()
        sensor.destroy()
        if synchronous_enabled:
            world.apply_settings(original_settings)

    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    intrinsic = build_intrinsic_matrix(width, height, fov)
    image = Image.fromarray(rgb)
    return RenderCapture(
        image=image,
        camera_transform=camera_transform,
        intrinsic=intrinsic,
        bounds=bounds,
    )


def save_bev_metadata(
    image_path: Path,
    bounds: MapBounds,
    image: Image.Image,
    *,
    mode: str,
) -> Path:
    """Write a sidecar JSON file with spatial metadata for the BEV image."""
    width = max(1, image.width)
    height = max(1, image.height)
    world_bounds = bounds._asdict()
    metadata = {
        "town": image_path.stem,
        "mode": mode,
        "image_width": width,
        "image_height": height,
        "world_bounds": world_bounds,
        "meters_per_pixel": {
            "x": (bounds.width) / width if width else None,
            "y": (bounds.height) / height if height else None,
        },
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    meta_path = image_path.with_suffix(image_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def main() -> None:
    args = parse_args()

    if args.mode == "mask" and args.pixels_per_meter <= 0:
        raise ValueError("--pixels-per-meter must be positive.")
    if args.downscale <= 0:
        raise ValueError("--downscale must be positive.")
    if args.mode == "rendered":
        if args.image_width <= 0 or args.image_height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if args.fov <= 0:
            raise ValueError("--fov must be positive.")
        if args.shutter_speed <= 0 or args.iso <= 0 or args.fstop <= 0:
            raise ValueError("Exposure parameters must be positive.")
        if args.gamma <= 0:
            raise ValueError("--gamma must be positive.")

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    towns = gather_towns(client, args.towns)
    if not towns:
        raise RuntimeError("No CARLA towns found.")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for town in towns:
        dst_path = output_root / f"{town}.png"
        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] {town} already rendered at {dst_path}")
            continue

        print(f"[INFO] Rendering BEV for {town} ...")
        if args.mode == "mask":
            bev_image, bounds = render_mask_bev(client, town, args.pixels_per_meter)
        else:
            capture = capture_rendered_bev(
                client,
                town,
                width=args.image_width,
                height=args.image_height,
                fov=args.fov,
                altitude_buffer=args.altitude_buffer,
                min_height=args.min_height,
                waypoint_spacing=args.waypoint_spacing,
                set_spectator=args.spectator,
                keep_weather=args.keep_weather,
                weather=args.weather,
                exposure_mode=args.exposure_mode,
                exposure_compensation=args.exposure_compensation,
                shutter_speed=args.shutter_speed,
                iso=args.iso,
                fstop=args.fstop,
                gamma=args.gamma,
                disable_postprocess=args.disable_postprocess,
            )
            bev_image = capture.image
            bounds = capture.bounds
        bev_image = maybe_downscale(bev_image, args.downscale)
        bev_image.save(dst_path)
        save_bev_metadata(dst_path, bounds, bev_image, mode=args.mode)
        print(f"[DONE] Saved {town} top-down map to {dst_path}")


if __name__ == "__main__":
    main()
