#!/usr/bin/env python3
"""Interactive CARLA scenario builder."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
from string import Template
from typing import Any, Iterable, List

import pandas as pd
try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]
try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


COLOR_PALETTE = [
    "#8e44ad",
    "#f39c12",
    "#2ecc71",
    "#d35400",
    "#1abc9c",
    "#f1c40f",
    "#27ae60",
    "#9b59b6",
    "#ff6f61",
    "#2c3e50",
]


def classify_lane_direction(wp: carla.Waypoint, carla_map: carla.Map) -> str:
    """Return 'forward', 'opposing', or 'neutral' for the provided waypoint."""
    lane_type = wp.lane_type
    if not lane_type & carla.LaneType.Driving:
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

    return "forward" if wp.lane_id < 0 else "opposing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Sampling distance between reference waypoints in metres.",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help="Legacy flag for a single town (use --towns for multiple).",
    )
    parser.add_argument(
        "--towns",
        nargs="*",
        default=None,
        help="Optional list of towns (e.g., Town05 Town07). When omitted, all installed towns are shown.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scenario_builder.html"),
        help="Path to the generated HTML file.",
    )
    parser.add_argument(
        "--bev-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with BEV PNGs (and *.meta.json sidecars) named <town>.png "
            "(generated via tools/generate_carla_town_bev.py)."
        ),
    )
    return parser.parse_args()


def list_available_towns(client: carla.Client) -> List[str]:
    return sorted({path.split("/")[-1] for path in client.get_available_maps()})


def sample_town(world: carla.World, distance: float) -> pd.DataFrame:
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(distance=distance)
    rows = []
    for wp in waypoints:
        tf = wp.transform
        loc = tf.location
        rows.append(
            dict(
                x=loc.x,
                y=loc.y,
                z=loc.z,
                yaw=tf.rotation.yaw,
                lane_id=wp.lane_id,
                road_id=wp.road_id,
                section_id=wp.section_id,
                lane_direction=classify_lane_direction(wp, carla_map),
            )
        )
    return pd.DataFrame(rows)


def compute_payload(df: pd.DataFrame) -> dict[str, Any]:
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    pad_x = max((xmax - xmin) * 0.05, 10.0)
    pad_y = max((ymax - ymin) * 0.05, 10.0)
    direction_colors: list[str] = []
    for direction in df["lane_direction"].tolist():
        if direction == "forward":
            direction_colors.append("#3498db")
        elif direction == "opposing":
            direction_colors.append("#e74c3c")
        else:
            direction_colors.append("#95a5a6")
    return {
        "x": df["x"].round(3).tolist(),
        "y": df["y"].round(3).tolist(),
        "z": df["z"].round(3).tolist(),
        "yaw": df["yaw"].round(3).tolist(),
        "lane_id": df["lane_id"].tolist(),
        "road_id": df["road_id"].tolist(),
        "section_id": df["section_id"].tolist(),
        "lane_direction": df["lane_direction"].tolist(),
        "lane_colors": direction_colors,
        "xmin": xmin - pad_x,
        "xmax": xmax + pad_x,
        "ymin": ymin - pad_y,
        "ymax": ymax + pad_y,
    }


def read_bev_metadata(bev_path: Path) -> dict[str, float] | None:
    """Load the spatial bounds stored next to the BEV PNG."""
    candidate_paths = [
        bev_path.with_suffix(bev_path.suffix + ".meta.json"),
        bev_path.with_suffix(".json"),
    ]
    for meta_path in candidate_paths:
        if not meta_path.exists():
            continue
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            print(f"[WARN] Failed to parse BEV metadata at {meta_path}: {exc}")  # noqa: T201
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
                print(f"[WARN] Invalid world_bounds in {meta_path}")  # noqa: T201
    return None


def load_bev_assets(bev_dir: Path | None, town: str) -> tuple[str | None, dict[str, float] | None]:
    """Return (image_data_uri, bounds_dict) for the BEV if available."""
    if bev_dir is None:
        return None, None
    bev_path = bev_dir / f"{town}.png"
    if not bev_path.exists():
        print(f"[WARN] Missing BEV preview for {town} at {bev_path}")  # noqa: T201
        return None, None
    bev_bytes = bev_path.read_bytes()
    if Image is not None:
        try:
            with Image.open(io.BytesIO(bev_bytes)) as img:
                rotated = img.rotate(90, expand=True)
                buffer = io.BytesIO()
                rotated.save(buffer, format="PNG")
                bev_bytes = buffer.getvalue()
        except OSError as exc:  # pragma: no cover - file specific
            print(f"[WARN] Failed to rotate BEV for {town}: {exc}")  # noqa: T201
    else:  # pragma: no cover - optional dependency missing
        print("[WARN] Pillow not installed; BEV rotation skipped.")  # noqa: T201
    data = base64.b64encode(bev_bytes).decode("ascii")
    metadata = read_bev_metadata(bev_path)
    if metadata is None:
        print(f"[WARN] No metadata for {town}; BEV preview disabled.")  # noqa: T201
    return f"data:image/png;base64,{data}", metadata


def generate_html(town_payloads: dict[str, dict[str, Any]],
                   distance: float,
                   colors: Iterable[str],
                   output_path: Path) -> None:
    if not town_payloads:
        raise ValueError("No town payloads provided.")

    colors_json = json.dumps(list(colors))
    towns_json = json.dumps(town_payloads)
    initial_town = next(iter(town_payloads.keys()))

    template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>CARLA Scenario Builder</title>
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
        }
        #map {
            width: 100%;
            height: 100%;
        }
        h2 {
            margin-top: 0;
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
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
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
        summary.xml-summary {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }
        .xml-summary-label {
            flex: 1;
            min-width: 0;
        }
        .xml-remove-btn {
            padding: 2px 8px;
            line-height: 1;
            font-size: 14px;
        }
        .info {
            margin-bottom: 12px;
            font-size: 13px;
            color: #bbb;
        }
        .town-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .town-tab {
            background-color: #2d2d2d;
            border: none;
            color: #d0d0d0;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        .town-tab.active {
            background-color: #2979ff;
            color: #fff;
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
        .offset-panel {
            margin: 12px 0 18px 0;
            padding: 10px;
            border: 1px solid #2b2b2b;
            border-radius: 4px;
            background-color: #1a1a1a;
        }
        .offset-panel h3 {
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
        .waypoint-coord-input {
            width: 100%;
            box-sizing: border-box;
        }
        .static-model-panel {
            margin: 16px 0;
            padding: 10px;
            border: 1px solid #2b2b2b;
            border-radius: 4px;
            background-color: #1a1a1a;
        }
        .static-model-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .static-model-header h3 {
            margin: 0;
        }
        .static-model-controls {
            margin-bottom: 12px;
        }
        .static-model-hint {
            font-size: 12px;
            color: #8cb4ff;
            margin: 8px 0;
        }
        .static-model-table td input {
            width: 100%;
        }
        .static-model-table th {
            font-size: 13px;
        }
        button.model-placement-active {
            background-color: #1e5fd8;
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
            <div class="town-tabs" id="townTabs"></div>
            <div id="map"></div>
        </div>
        <div class="right-panel">
            <h2>Scenario Builder</h2>
            <p class="info">Active town: <strong id="activeTownLabel"></strong> | Sampling: ${distance} m</p>
            <p class="info">Switching towns resets all agents and waypoints.</p>
            <label for="scenarioName">Scenario / folder name</label>
            <input type="text" id="scenarioName" value="" />

            <label for="routeId">Route ID</label>
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

            <div class="offset-panel">
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

            <div class="static-model-panel">
                <div class="static-model-header">
                    <h3>Static 3D models</h3>
                    <div>
                        <button id="placeStaticModelBtn" class="secondary">Place 3D model</button>
                        <button id="clearStaticModelsBtn" class="secondary">Clear models</button>
                    </div>
                </div>
                <div class="static-model-controls">
                    <label for="staticModelPreset">Model preset</label>
                    <select id="staticModelPreset"></select>
                </div>
                <p id="staticModelHint" class="static-model-hint hidden">
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
                    No BEV preview found for this town. Run tools/generate_carla_town_bev.py.
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
    const colorPalette = ${colors_json};
    const townsData = ${towns_json};
    const MIRROR_X = true;
    const MIRROR_Y = false;
    const OVERLAY_MIN_WIDTH = 260;
    const OVERLAY_DEFAULT_WIDTH = 600;
    const ICON_EXPAND = '\u2922';
    const ICON_COLLAPSE = '\u00D7';
    const OVERLAY_MIN_SCALE = 0.1;
    const OVERLAY_MAX_SCALE = 3.0;
    const OVERLAY_SCALE_STEP = 0.1;
    const STATIC_MODEL_PRESETS = [
        {id: 'truck', label: 'Truck', blueprint: 'vehicle.carlamotors.carlacola', length: 12.0, width: 3.0},
        {id: 'streetbarrier', label: 'Street barrier', blueprint: 'static.prop.streetbarrier', length: 2.0, width: 0.45},
        {id: 'constructioncone', label: 'Construction cone', blueprint: 'static.prop.constructioncone', length: 0.45, width: 0.45},
        {id: 'trafficcone01', label: 'Traffic cone 01', blueprint: 'static.prop.trafficcone01', length: 0.4, width: 0.4},
        {id: 'trafficcone02', label: 'Traffic cone 02', blueprint: 'static.prop.trafficcone02', length: 0.4, width: 0.4},
        {id: 'trafficwarning', label: 'Traffic warning', blueprint: 'static.prop.trafficwarning', length: 1.2, width: 0.5},
        {id: 'warningaccident', label: 'Warning accident', blueprint: 'static.prop.warningaccident', length: 1.0, width: 0.5}
    ];
    const STATIC_MODEL_PRESET_MAP = {};
    STATIC_MODEL_PRESETS.forEach(preset => {
        STATIC_MODEL_PRESET_MAP[preset.id] = preset;
    });
    const STATIC_MODEL_DEFAULT_PRESET = STATIC_MODEL_PRESETS[0] || {
        id: 'default',
        label: 'Generic static',
        blueprint: 'vehicle.carlamotors.carlacola',
        length: 12.0,
        width: 3.0
    };
    const PLOTLY_CLICK_FLAG = '__scenarioBuilderPlotlyClick';

    const townNames = Object.keys(townsData);
    let activeTown = ${initial_town};
    if (!townNames.includes(activeTown) && townNames.length > 0) {
        activeTown = townNames[0];
    }

    let actors = [];
    let activeActorId = null;
    let actorIdCounter = 0;
    let egoCounter = 0;
    let npcCounter = 0;
    let pedestrianCounter = 0;
    let bicycleCounter = 0;
    let colorIndex = 0;
    let pendingHeading = null;
    let orientationPreview = null;
    let plotReady = false;
    let bevOverlayDragState = null;
    let bevOverlayResizeState = null;
    let bevOverlayAspect = 1;
    let bevOverlayOpen = false;
    let bevOverlayScale = 1;
    let staticModels = [];
    let staticModelIdCounter = 0;
    let modelPlacementMode = false;
    let activeStaticModelPresetId = STATIC_MODEL_DEFAULT_PRESET.id;
    let staticModelCounters = {};

    function nextActorName(kind) {
        if (kind === 'ego') {
            const name = 'ego_vehicle_' + egoCounter;
            egoCounter += 1;
            return name;
        }
        if (kind === 'pedestrian') {
            const name = 'pedestrian_' + pedestrianCounter;
            pedestrianCounter += 1;
            return name;
        }
        if (kind === 'bicycle') {
            const name = 'bicycle_' + bicycleCounter;
            bicycleCounter += 1;
            return name;
        }
        const name = 'npc_vehicle_' + npcCounter;
        npcCounter += 1;
        return name;
    }

    function createActor(kind) {
        const color = colorPalette[colorIndex % colorPalette.length];
        colorIndex += 1;
        const actor = {
            id: actorIdCounter,
            kind: kind,
            name: nextActorName(kind),
            color: color,
            waypoints: [],
            offsetTotal: {x: 0, y: 0}
        };
        actorIdCounter += 1;
        return actor;
    }

    function getActiveActor() {
        return actors.find(a => a.id === activeActorId) || null;
    }

    function ensureActorOffset(actor) {
        if (!actor) return null;
        if (!actor.offsetTotal) {
            actor.offsetTotal = {x: 0, y: 0};
        }
        return actor.offsetTotal;
    }

    function hasActiveOffset(actor) {
        const offset = ensureActorOffset(actor);
        if (!offset) return false;
        return Math.abs(offset.x) > 1e-6 || Math.abs(offset.y) > 1e-6;
    }

    function resetActorOffsetState(actor) {
        const offset = ensureActorOffset(actor);
        if (!offset) return;
        offset.x = 0;
        offset.y = 0;
    }

    function setActiveActor(id) {
        resetOrientationState();
        activeActorId = id;
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    }

    function updateWaypointActionButtons() {
        const actor = getActiveActor();
        const hasWaypoints = !!(actor && actor.waypoints.length > 0);
        if (clearActorBtn) {
            clearActorBtn.disabled = !hasWaypoints;
        }
        if (applyOffsetBtn) {
            applyOffsetBtn.disabled = !hasWaypoints;
        }
        if (resetOffsetBtn) {
            resetOffsetBtn.disabled = !hasWaypoints || !hasActiveOffset(actor);
        }
    }

    function removeActorById(id) {
        const idx = actors.findIndex(a => a.id === id);
        if (idx === -1) return false;
        actors.splice(idx, 1);
        const fallback = actors[idx] || actors[idx - 1] || actors[0] || null;
        const nextId = fallback ? fallback.id : null;
        setActiveActor(nextId);
        return true;
    }

    function offsetActiveActorWaypoints(dx, dy) {
        const actor = getActiveActor();
        if (!actor || actor.waypoints.length === 0) return false;
        if (!Number.isFinite(dx) || !Number.isFinite(dy)) return false;
        const offset = ensureActorOffset(actor);
        actor.waypoints.forEach(wp => {
            wp.x += dx;
            wp.y += dy;
        });
        if (offset) {
            offset.x += dx;
            offset.y += dy;
        }
        return true;
    }

    function resetActiveActorOffset() {
        const actor = getActiveActor();
        if (!actor || actor.waypoints.length === 0) return false;
        const offset = ensureActorOffset(actor);
        if (!offset) return false;
        if (Math.abs(offset.x) < 1e-6 && Math.abs(offset.y) < 1e-6) {
            return false;
        }
        actor.waypoints.forEach(wp => {
            wp.x -= offset.x;
            wp.y -= offset.y;
        });
        offset.x = 0;
        offset.y = 0;
        return true;
    }

    function townHasBev(townName) {
        const data = townsData[townName];
        return !!(data && data.bev_image);
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

    function updateBevPreview(townData, townName) {
        const hasImage = !!(townData && townData.bev_image);
        if (bevImageEl) {
            if (hasImage) {
                bevImageEl.src = townData.bev_image;
            } else {
                bevImageEl.removeAttribute('src');
            }
        }
        syncBevPreviewVisibility(hasImage);
        if (!hasImage) {
            closeBevOverlay();
        }
        updateBevButtonState(hasImage);
        if (hasImage && bevOverlayTitle) {
            bevOverlayTitle.textContent = townName + ' BEV';
        }
        if (hasImage && bevOverlay && !bevOverlay.classList.contains('hidden') && bevOverlayImage) {
            bevOverlayImage.src = townData.bev_image;
        }
    }

    function closeBevOverlay() {
        if (!bevOverlay) return;
        bevOverlay.classList.add('hidden');
        bevOverlayDragState = null;
        bevOverlayResizeState = null;
        bevOverlayOpen = false;
        const hasImage = townHasBev(activeTown);
        syncBevPreviewVisibility(hasImage);
        updateBevButtonState(hasImage);
    }

    function showBevOverlay() {
        if (!bevOverlay || !bevOverlayImage) return;
        const data = townsData[activeTown];
        if (!data || !data.bev_image) return;
        bevOverlayImage.src = data.bev_image;
        if (bevOverlayTitle) {
            bevOverlayTitle.textContent = activeTown + ' BEV';
        }
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

    function getStaticModelPresetById(id) {
        if (!id) return null;
        return STATIC_MODEL_PRESET_MAP[id] || null;
    }

    function getActiveStaticModelPreset() {
        return getStaticModelPresetById(activeStaticModelPresetId) || STATIC_MODEL_DEFAULT_PRESET;
    }

    function nextStaticModelName(preset) {
        const key = preset.id || preset.label || 'static';
        staticModelCounters[key] = (staticModelCounters[key] || 0) + 1;
        return preset.label + ' ' + staticModelCounters[key];
    }

    function resetStaticModelCounters() {
        staticModelCounters = {};
    }

    function updateStaticModelPlacementUi() {
        const preset = getActiveStaticModelPreset();
        const presetLabel = preset ? preset.label : null;
        const suffix = presetLabel ? ' (' + presetLabel + ')' : '';
        if (placeStaticModelBtn) {
            const base = modelPlacementMode ? 'Click map…' : 'Place 3D model';
            placeStaticModelBtn.textContent = base + suffix;
        }
        if (staticModelHint) {
            staticModelHint.classList.toggle('hidden', !modelPlacementMode);
            const hintText = 'Click on the map to place the selected model' +
                (presetLabel ? ' (' + presetLabel + ')' : '') +
                '. Use the table to fine-tune position and rotation.';
            staticModelHint.textContent = hintText;
        }
    }

    function setModelPlacementMode(enabled) {
        modelPlacementMode = enabled;
        if (modelPlacementMode) {
            resetOrientationState();
        }
        if (placeStaticModelBtn) {
            placeStaticModelBtn.classList.toggle('model-placement-active', enabled);
        }
        updateStaticModelPlacementUi();
    }

    function createStaticModel(x, y, presetId) {
        const preset = getStaticModelPresetById(presetId) || getActiveStaticModelPreset();
        if (!preset) return;
        const model = {
            id: staticModelIdCounter,
            name: nextStaticModelName(preset),
            x,
            y,
            yaw: 0,
            length: (Number.isFinite(preset.length) ? preset.length : STATIC_MODEL_DEFAULT_PRESET.length),
            width: (Number.isFinite(preset.width) ? preset.width : STATIC_MODEL_DEFAULT_PRESET.width),
            blueprint: preset.blueprint || STATIC_MODEL_DEFAULT_PRESET.blueprint
        };
        staticModelIdCounter += 1;
        staticModels.push(model);
        renderStaticModelTable();
        updatePlot();
    }

    function removeStaticModel(id) {
        const idx = staticModels.findIndex(m => m.id === id);
        if (idx >= 0) {
            staticModels.splice(idx, 1);
            renderStaticModelTable();
            updatePlot();
        }
    }

    function clearStaticModels() {
        staticModels = [];
        staticModelIdCounter = 0;
        resetStaticModelCounters();
        renderStaticModelTable();
        updatePlot();
    }

    function updateStaticModelField(model, field, value) {
        if (!Number.isFinite(value)) return;
        model[field] = value;
        updatePlot();
        renderStaticModelTable();
    }

    function renderStaticModelTable() {
        if (!staticModelTableBody) return;
        staticModelTableBody.innerHTML = '';
        if (clearStaticModelsBtn) {
            clearStaticModelsBtn.disabled = staticModels.length === 0;
        }
        if (staticModels.length === 0) {
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
        staticModels.forEach(model => {
            const row = document.createElement('tr');

            const nameCell = document.createElement('td');
            nameCell.textContent = model.name;
            row.appendChild(nameCell);

            const xCell = document.createElement('td');
            const xInput = document.createElement('input');
            xInput.type = 'number';
            xInput.step = '0.1';
            xInput.value = model.x.toFixed(2);
            xInput.addEventListener('change', () => {
                updateStaticModelField(model, 'x', parseFloat(xInput.value));
            });
            xCell.appendChild(xInput);
            row.appendChild(xCell);

            const yCell = document.createElement('td');
            const yInput = document.createElement('input');
            yInput.type = 'number';
            yInput.step = '0.1';
            yInput.value = model.y.toFixed(2);
            yInput.addEventListener('change', () => {
                updateStaticModelField(model, 'y', parseFloat(yInput.value));
            });
            yCell.appendChild(yInput);
            row.appendChild(yCell);

            const yawCell = document.createElement('td');
            const yawInput = document.createElement('input');
            yawInput.type = 'number';
            yawInput.step = '1';
            yawInput.value = model.yaw.toFixed(1);
            yawInput.addEventListener('change', () => {
                updateStaticModelField(model, 'yaw', parseFloat(yawInput.value));
            });
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
        const halfL = model.length * 0.5;
        const halfW = model.width * 0.5;
        const localPoints = [
            {x: halfL, y: halfW},
            {x: -halfL, y: halfW},
            {x: -halfL, y: -halfW},
            {x: halfL, y: -halfW}
        ];
        const rad = model.yaw * Math.PI / 180;
        const cos = Math.cos(rad);
        const sin = Math.sin(rad);
        return localPoints.map(pt => ({
            x: model.x + pt.x * cos - pt.y * sin,
            y: model.y + pt.x * sin + pt.y * cos
        }));
    }

    const mapDiv = document.getElementById('map');
    const bevImageEl = document.getElementById('bevImage');
    const bevFallbackEl = document.getElementById('bevFallback');
    const bevOpenNotice = document.getElementById('bevOpenNotice');
    const openBevBtn = document.getElementById('openBevFullscreen');
    const staticModelHint = document.getElementById('staticModelHint');
    const staticModelTableBody = document.getElementById('staticModelTableBody');
    const staticModelPresetSelect = document.getElementById('staticModelPreset');
    const placeStaticModelBtn = document.getElementById('placeStaticModelBtn');
    const clearStaticModelsBtn = document.getElementById('clearStaticModelsBtn');
    const bevOverlay = document.getElementById('bevOverlay');
    const bevOverlayHeader = document.getElementById('bevOverlayHeader');
    const bevOverlayTitle = document.getElementById('bevOverlayTitle');
    const bevOverlayImage = document.getElementById('bevOverlayImage');
    const bevOverlayClose = document.getElementById('bevOverlayClose');
    const bevOverlayResize = document.getElementById('bevOverlayResize');
    const bevZoomInBtn = document.getElementById('bevZoomInBtn');
    const bevZoomOutBtn = document.getElementById('bevZoomOutBtn');
    const clearActorBtn = document.getElementById('clearActorBtn');
    const offsetXInput = document.getElementById('offsetXInput');
    const offsetYInput = document.getElementById('offsetYInput');
    const applyOffsetBtn = document.getElementById('applyOffsetBtn');
    const resetOffsetBtn = document.getElementById('resetOffsetInputs');

    if (staticModelPresetSelect) {
        staticModelPresetSelect.innerHTML = '';
        STATIC_MODEL_PRESETS.forEach(preset => {
            const option = document.createElement('option');
            option.value = preset.id;
            option.textContent = preset.label;
            staticModelPresetSelect.appendChild(option);
        });
        if (activeStaticModelPresetId && STATIC_MODEL_PRESET_MAP[activeStaticModelPresetId]) {
            staticModelPresetSelect.value = activeStaticModelPresetId;
        }
        staticModelPresetSelect.addEventListener('change', () => {
            const nextPreset = getStaticModelPresetById(staticModelPresetSelect.value);
            if (!nextPreset) return;
            activeStaticModelPresetId = nextPreset.id;
            updateStaticModelPlacementUi();
        });
    }
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
    Plotly.newPlot(mapDiv, [baseTrace], layout, config).then(() => {
        plotReady = true;
    });

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

    if (openBevBtn) {
        openBevBtn.addEventListener('click', () => {
            if (openBevBtn.disabled) return;
            if (bevOverlayOpen) {
                closeBevOverlay();
            } else {
                showBevOverlay();
            }
        });
    }
    if (bevOverlayClose) {
        bevOverlayClose.addEventListener('click', closeBevOverlay);
    }
    window.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            if (bevOverlayOpen) {
                closeBevOverlay();
            } else if (modelPlacementMode) {
                setModelPlacementMode(false);
            }
        }
    });
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
    function applyOverlaySize(width) {
        if (!bevOverlay) return;
        const aspect = Math.max(bevOverlayAspect, 1e-3);
        const clampedWidth = Math.max(OVERLAY_MIN_WIDTH, width);
        bevOverlay.style.width = clampedWidth + 'px';
        bevOverlay.style.height = Math.max(200, clampedWidth / aspect) + 'px';
        bevOverlay.dataset.width = String(clampedWidth);
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
    if (bevOverlayImage) {
        bevOverlayImage.addEventListener('load', () => {
            const naturalWidth = bevOverlayImage.naturalWidth;
            const naturalHeight = bevOverlayImage.naturalHeight;
            if (naturalWidth > 0 && naturalHeight > 0) {
                bevOverlayAspect = naturalWidth / naturalHeight;
            } else {
                bevOverlayAspect = 1;
            }
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

    window.addEventListener('mouseup', () => {
        bevOverlayDragState = null;
        bevOverlayResizeState = null;
    });

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

    if (placeStaticModelBtn) {
        placeStaticModelBtn.addEventListener('click', () => {
            setModelPlacementMode(!modelPlacementMode);
        });
    }

    if (clearStaticModelsBtn) {
        clearStaticModelsBtn.addEventListener('click', () => {
            if (staticModels.length === 0) return;
            if (!confirm('Remove all static models?')) return;
            clearStaticModels();
            setModelPlacementMode(false);
        });
    }

    renderStaticModelTable();
    setModelPlacementMode(false);
    updateOverlayZoomUI();

    if (applyOffsetBtn) {
        applyOffsetBtn.addEventListener('click', () => {
            const dx = offsetXInput ? parseFloat(offsetXInput.value) : 0;
            const dy = offsetYInput ? parseFloat(offsetYInput.value) : 0;
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
            if (offsetXInput) offsetXInput.value = '0';
            if (offsetYInput) offsetYInput.value = '0';
        });
    }

    mapDiv.addEventListener('mousemove', function(event) {
        if (!pendingHeading) return;
        const coords = screenToData(event);
        if (!coords) return;
        orientationPreview = coords;
        updateOrientationOverlay();
        updateOrientationPreviewMarker();
    });

    mapDiv.on('plotly_click', function(data) {
        if (data.event) {
            data.event[PLOTLY_CLICK_FLAG] = true;
        }
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

    function computeHeadingDegrees(origin, target) {
        return Math.atan2(target.y - origin.y, target.x - origin.x) * 180 / Math.PI;
    }

    function yawForDisplay(actor, index) {
        const wp = actor.waypoints[index];
        if (!wp) return 0;
        if (
            pendingHeading &&
            orientationPreview &&
            pendingHeading.actorId === actor.id &&
            pendingHeading.index === index
        ) {
            return computeHeadingDegrees(wp, orientationPreview);
        }
        return wp.yaw;
    }

    function getMarkerStyle(actor) {
        if (actor.kind === 'pedestrian') {
            return {symbol: 'circle', size: 14};
        }
        if (actor.kind === 'bicycle') {
            return {symbol: 'diamond', size: 16};
        }
        return {symbol: 'triangle-up', size: 20};
    }

    function updateOrientationPreviewMarker() {
        if (!plotReady) return;
        let traceIndex = 1; // baseTrace is index 0
        actors.forEach((actor) => {
            if (actor.waypoints.length === 0) return;
            const angles = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90);
            const yawValues = actor.waypoints.map((_, idx) => yawForDisplay(actor, idx));
            Plotly.restyle(mapDiv, {
                'marker.angle': [angles],
                'customdata': [yawValues]
            }, traceIndex);
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

    function resetOrientationState() {
        pendingHeading = null;
        orientationPreview = null;
        updateOrientationOverlay();
        updateOrientationPreviewMarker();
    }

    function updateOrientationOverlay() {
        if (!plotReady) return;
        const shapes = [];
        if (pendingHeading) {
            const actor = actors.find(a => a.id === pendingHeading.actorId);
            if (actor) {
                const wp = actor.waypoints[pendingHeading.index];
                if (wp) {
                    const radius = 4;
                    shapes.push({
                        type: 'circle',
                        xref: 'x',
                        yref: 'y',
                        x0: wp.x - radius,
                        x1: wp.x + radius,
                        y0: wp.y - radius,
                        y1: wp.y + radius,
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
                const yaw = Math.atan2(y - base.y, x - base.x) * 180 / Math.PI;
                if (isFinite(yaw)) {
                    base.yaw = yaw;
                }
            }
            pendingHeading = null;
            orientationPreview = null;
            updatePlot();
            renderWaypointTable();
            renderXmlOutputs();
            return;
        }

        const prev = actor.waypoints[actor.waypoints.length - 1];
        let yaw = prev ? prev.yaw : 0;
        actor.waypoints.push({x: x, y: y, z: 0.0, yaw: yaw});
        pendingHeading = {actorId: actor.id, index: actor.waypoints.length - 1};
        orientationPreview = null;
        updatePlot();
        renderWaypointTable();
        renderXmlOutputs();
    }

    function updatePlot() {
        const traces = [baseTrace];
        actors.forEach(actor => {
            if (actor.waypoints.length === 0) return;
            const orderLabels = actor.waypoints.map((_, idx) => String(idx + 1));
            const markerStyle = getMarkerStyle(actor);
            traces.push({
                x: actor.waypoints.map(w => w.x),
                y: actor.waypoints.map(w => w.y),
                customdata: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx)),
                text: orderLabels,
                textposition: 'middle center',
                textfont: {
                    color: '#111',
                    size: 12,
                    family: '"Segoe UI Semibold", "Segoe UI", Arial, sans-serif'
                },
                mode: 'lines+markers+text',
                name: actor.name + ' (' + actor.kind.toUpperCase() + ')',
                line: {
                    color: actor.color,
                    width: 3
                },
                marker: {
                    color: actor.color,
                    size: markerStyle.size,
                    symbol: markerStyle.symbol,
                    angle: actor.waypoints.map((_, idx) => yawForDisplay(actor, idx) - 90),
                    line: {color: '#000', width: 0.5}
                },
                hovertemplate:
                    'Agent: ' + actor.name + ' (' + actor.kind.toUpperCase() + ')<br>' +
                    'x=%{x:.2f} m<br>' +
                    'y=%{y:.2f} m<br>' +
                    'yaw=%{customdata:.1f}°<extra></extra>'
            });
        });
        staticModels.forEach(model => {
            const corners = computeStaticModelPolygon(model);
            if (corners.length === 0) return;
            const xs = corners.map(pt => pt.x);
            const ys = corners.map(pt => pt.y);
            xs.push(corners[0].x);
            ys.push(corners[0].y);
            traces.push({
                x: xs,
                y: ys,
                mode: 'lines',
                fill: 'toself',
                name: model.name + ' (STATIC)',
                line: {color: 'rgba(255, 165, 0, 0.9)', width: 2},
                fillcolor: 'rgba(255, 165, 0, 0.2)',
                hovertemplate:
                    model.name + '<br>' +
                    'x=%{x:.2f} m<br>' +
                    'y=%{y:.2f} m<br>' +
                    'yaw=' + model.yaw.toFixed(1) + '°' +
                    '<extra></extra>',
                showlegend: false
            });
        });
        Plotly.react(mapDiv, traces, layout, config).then(() => {
            updateOrientationOverlay();
            updateOrientationPreviewMarker();
        });
    }
    function renderActorSelect() {
        const select = document.getElementById('actorSelect');
        select.innerHTML = '';
        actors.forEach(actor => {
            const opt = document.createElement('option');
            opt.value = actor.id;
            opt.textContent = actor.name + ' (' + actor.kind.toUpperCase() + ')';
            if (actor.id === activeActorId) opt.selected = true;
            select.appendChild(opt);
        });
        select.disabled = actors.length === 0;
        document.getElementById('removeActorBtn').disabled = actors.length === 0;
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
            xInput.className = 'waypoint-coord-input';
            xInput.value = wp.x.toFixed(3);
            xInput.addEventListener('change', () => {
                const val = parseFloat(xInput.value);
                if (!Number.isFinite(val)) {
                    xInput.value = wp.x.toFixed(3);
                    return;
                }
                wp.x = val;
                updatePlot();
                renderXmlOutputs();
            });
            cellX.appendChild(xInput);
            row.appendChild(cellX);

            const cellY = document.createElement('td');
            const yInput = document.createElement('input');
            yInput.type = 'number';
            yInput.step = '0.01';
            yInput.className = 'waypoint-coord-input';
            yInput.value = wp.y.toFixed(3);
            yInput.addEventListener('change', () => {
                const val = parseFloat(yInput.value);
                if (!Number.isFinite(val)) {
                    yInput.value = wp.y.toFixed(3);
                    return;
                }
                wp.y = val;
                updatePlot();
                renderXmlOutputs();
            });
            cellY.appendChild(yInput);
            row.appendChild(cellY);

            const cellYaw = document.createElement('td');
            const yawInput = document.createElement('input');
            yawInput.type = 'number';
            yawInput.step = '0.1';
            yawInput.value = wp.yaw.toFixed(2);
            yawInput.addEventListener('change', () => {
                const val = parseFloat(yawInput.value);
                wp.yaw = Number.isFinite(val) ? val : 0;
                updatePlot();
                renderXmlOutputs();
            });
            cellYaw.appendChild(yawInput);
            row.appendChild(cellYaw);

            const cellActions = document.createElement('td');
            const removeBtn = document.createElement('button');
            removeBtn.textContent = 'Delete';
            removeBtn.className = 'danger';
            removeBtn.addEventListener('click', () => {
                actor.waypoints.splice(idx, 1);
                resetOrientationState();
                renderWaypointTable();
                updatePlot();
                renderXmlOutputs();
            });
            cellActions.appendChild(removeBtn);
            row.appendChild(cellActions);

            tbody.appendChild(row);
        });
        updateWaypointActionButtons();
    }

    function generateXml(actor) {
        const scenarioName = document.getElementById('scenarioName').value || 'custom_scenario';
        const routeId = document.getElementById('routeId').value || '0';
        const lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            '<routes>',
            '  <route id="' + routeId + '" town="' + activeTown + '" role="' + actor.kind + '">'
        ];
        actor.waypoints.forEach(wp => {
            lines.push(formatWaypoint(wp, '    '));
        });
        lines.push('  </route>');
        lines.push('</routes>');
        return lines.join('');
    }

    function generateStaticModelXml(model) {
        const routeId = document.getElementById('routeId').value || '0';
        const blueprint = model.blueprint ||
            (STATIC_MODEL_DEFAULT_PRESET ? STATIC_MODEL_DEFAULT_PRESET.blueprint : null) ||
            'vehicle.carlamotors.carlacola';
        const lines = [
            "<?xml version='1.0' encoding='utf-8'?>",
            '<routes>',
            '  <route id="' + routeId + '" town="' + activeTown + '" role="static" model="' + blueprint + '" length="' + model.length + '" width="' + model.width + '">',
            '    <waypoint x="' + model.x.toFixed(6) + '" y="' + model.y.toFixed(6) + '" z="0.0" yaw="' + model.yaw.toFixed(6) + '" />',
            '  </route>',
            '</routes>'
        ];
        return lines.join('');
    }

    function sanitizeFileComponent(value) {
        return (value || '')
            .trim()
            .replace(/[^a-zA-Z0-9._-]+/g, '_')
            .replace(/_+/g, '_')
            .replace(/^_+|_+$$/g, '') || 'item';
    }

    function formatWaypoint(wp, indent) {
        if (indent === undefined) indent = '      ';
        const yaw = wp.yaw.toFixed(6);
        const x = wp.x.toFixed(6);
        const y = wp.y.toFixed(6);
        const z = (wp.z || 0).toFixed(6);
        return indent + '<waypoint pitch="360.000000" roll="0.000000" x="' + x + '" y="' + y + '" yaw="' + yaw + '" z="' + z + '" />';
    }

    function parseXmlToWaypoints(xmlText) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(xmlText, 'application/xml');
        const errorNode = doc.getElementsByTagName('parsererror');
        if (errorNode.length) {
            throw new Error(errorNode[0].textContent || 'Invalid XML');
        }
        const nodes = Array.from(doc.getElementsByTagName('waypoint'));
        if (!nodes.length) {
            throw new Error('No <waypoint> elements found.');
        }
        return nodes.map(node => {
            const x = parseFloat(node.getAttribute('x'));
            const y = parseFloat(node.getAttribute('y'));
            const yaw = parseFloat(node.getAttribute('yaw') || '0');
            const z = parseFloat(node.getAttribute('z') || '0');
            if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(yaw)) {
                throw new Error('Waypoints must include numeric x, y, yaw.');
            }
            return { x: x, y: y, z: Number.isFinite(z) ? z : 0, yaw: yaw };
        });
    }

    function renderXmlOutputs() {
        const container = document.getElementById('xmlOutputs');
        container.innerHTML = '';
        const scenarioName = document.getElementById('scenarioName').value || 'custom_scenario';

        actors.forEach(actor => {
            const details = document.createElement('details');
            details.open = actors.length <= 1;
            const summary = document.createElement('summary');
            summary.className = 'xml-summary';
            const summaryLabel = document.createElement('span');
            summaryLabel.className = 'xml-summary-label';
            summaryLabel.textContent = actor.name + ' [' + actor.kind.toUpperCase() + '] (' + actor.waypoints.length + ' waypoints)';
            const summaryRemoveBtn = document.createElement('button');
            summaryRemoveBtn.type = 'button';
            summaryRemoveBtn.className = 'icon-button danger xml-remove-btn';
            summaryRemoveBtn.textContent = '×';
            summaryRemoveBtn.title = 'Delete this agent';
            summaryRemoveBtn.addEventListener('click', (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                removeActorById(actor.id);
            });
            summary.appendChild(summaryLabel);
            summary.appendChild(summaryRemoveBtn);
            details.appendChild(summary);

            const textarea = document.createElement('textarea');
            textarea.className = 'xml-editor';
            let programmaticUpdate = true;
            textarea.value = generateXml(actor);
            programmaticUpdate = false;

            const status = document.createElement('div');
            status.className = 'status-msg';

            let debounceId = null;
            textarea.addEventListener('input', () => {
                if (programmaticUpdate) {
                    programmaticUpdate = false;
                    return;
                }
                clearTimeout(debounceId);
                status.textContent = 'Applying...';
                status.className = 'status-msg';
                debounceId = setTimeout(() => {
                    try {
                        const waypoints = parseXmlToWaypoints(textarea.value);
                        actor.waypoints = waypoints;
                        resetActorOffsetState(actor);
                        resetOrientationState();
                        renderWaypointTable();
                        updatePlot();
                        programmaticUpdate = true;
                        textarea.value = generateXml(actor);
                        programmaticUpdate = false;
                        status.textContent = 'Applied';
                        status.className = 'status-msg status-success';
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
                const fileName = scenarioName + '_' + actor.name + '.xml';
                saveAs(blob, fileName);
            });
            buttonsRow.appendChild(downloadBtn);

            details.appendChild(textarea);
            details.appendChild(buttonsRow);
            details.appendChild(status);
            container.appendChild(details);
        });
    }

    function renderTownTabs() {
        const tabs = document.getElementById('townTabs');
        tabs.innerHTML = '';
        townNames.forEach(name => {
            const btn = document.createElement('button');
            btn.textContent = name;
            btn.className = 'town-tab' + (name === activeTown ? ' active' : '');
            btn.addEventListener('click', () => loadTown(name));
            tabs.appendChild(btn);
        });
        document.getElementById('activeTownLabel').textContent = activeTown;
    }

    function loadTown(name) {
        activeTown = name;
        const data = townsData[name];
        updateBevPreview(data, name);
        staticModels = [];
        staticModelIdCounter = 0;
        resetStaticModelCounters();
        renderStaticModelTable();
        setModelPlacementMode(false);
        baseTrace.x = data.x;
        baseTrace.y = data.y;
        baseTrace.marker.color = data.lane_colors || baseTrace.marker.color;
        baseTrace.customdata = data.z.map((_, idx) => [
            data.z[idx],
            data.yaw[idx],
            data.lane_id[idx],
            data.road_id[idx],
            data.lane_direction ? data.lane_direction[idx] : ''
        ]);
        if (MIRROR_X) {
            layout.xaxis.range = [data.xmax, data.xmin];
        } else {
            layout.xaxis.range = [data.xmin, data.xmax];
        }
        if (MIRROR_Y) {
            layout.yaxis.range = [data.ymax, data.ymin];
        } else {
            layout.yaxis.range = [data.ymin, data.ymax];
        }
        layout.xaxis.autorange = false;
        layout.yaxis.autorange = false;

        actors = [];
        actorIdCounter = 0;
        egoCounter = 0;
        npcCounter = 0;
        pedestrianCounter = 0;
        bicycleCounter = 0;
        colorIndex = 0;
        const actor = createActor('ego');
        actors.push(actor);
        activeActorId = actor.id;

        document.getElementById('scenarioName').value = name.toLowerCase() + '_custom';
        document.getElementById('routeId').value = '247';

        pendingHeading = null;
        orientationPreview = null;

        renderTownTabs();
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    }

    function init() {
        renderTownTabs();
        if (townNames.length > 0) {
            loadTown(activeTown);
        } else {
            document.getElementById('activeTownLabel').textContent = 'None';
        }
    }

    document.getElementById('actorSelect').addEventListener('change', (evt) => {
        setActiveActor(parseInt(evt.target.value));
    });

    document.getElementById('addEgoBtn').addEventListener('click', () => {
        const actor = createActor('ego');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addNpcBtn').addEventListener('click', () => {
        const actor = createActor('npc');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addPedBtn').addEventListener('click', () => {
        const actor = createActor('pedestrian');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('addBikeBtn').addEventListener('click', () => {
        const actor = createActor('bicycle');
        actors.push(actor);
        setActiveActor(actor.id);
    });

    document.getElementById('removeActorBtn').addEventListener('click', () => {
        const actor = getActiveActor();
        if (!actor) return;
        removeActorById(actor.id);
    });

    if (clearActorBtn) {
        clearActorBtn.addEventListener('click', () => {
            const actor = getActiveActor();
            if (!actor) return;
            actor.waypoints = [];
            resetActorOffsetState(actor);
            resetOrientationState();
            renderWaypointTable();
            updatePlot();
            renderXmlOutputs();
        });
    }

    document.getElementById('resetAllBtn').addEventListener('click', () => {
        if (!confirm('Clear all agents and waypoints?')) return;
        if (townNames.length === 0) return;
        actors = [];
        actorIdCounter = 0;
        egoCounter = 0;
        npcCounter = 0;
        pedestrianCounter = 0;
        bicycleCounter = 0;
        colorIndex = 0;
        const actor = createActor('ego');
        actors.push(actor);
        activeActorId = actor.id;
        staticModels = [];
        staticModelIdCounter = 0;
        resetStaticModelCounters();
        renderStaticModelTable();
        setModelPlacementMode(false);
        resetOrientationState();
        renderActorSelect();
        renderWaypointTable();
        updatePlot();
        renderXmlOutputs();
    });

    document.getElementById('scenarioName').addEventListener('input', renderXmlOutputs);
    document.getElementById('routeId').addEventListener('input', renderXmlOutputs);

    document.getElementById('downloadAllBtn').addEventListener('click', () => {
        if (actors.length === 0) {
            alert('No agents to export.');
            return;
        }
        const scenarioNameRaw = document.getElementById('scenarioName').value || 'custom_scenario';
        const scenarioNameSafe = sanitizeFileComponent(scenarioNameRaw);
        const zip = new JSZip();

        actors.forEach(actor => {
            const xmlText = generateXml(actor);
            const actorSafe = sanitizeFileComponent(actor.name);
            const fileName = scenarioNameSafe + '_' + actorSafe + '.xml';
            zip.file(fileName, xmlText);
        });

        staticModels.forEach(model => {
            const xmlText = generateStaticModelXml(model);
            const modelSafe = sanitizeFileComponent(model.name + '_static');
            const fileName = scenarioNameSafe + '_' + modelSafe + '.xml';
            zip.file(fileName, xmlText);
        });

        zip.generateAsync({type:'blob'}).then(content => {
            saveAs(content, scenarioNameSafe + '_routes.zip');
        });
    });

    init();
    </script>
</body>
</html>
""")

    html = template.substitute(
        colors_json=colors_json,
        towns_json=towns_json,
        initial_town=json.dumps(initial_town),
        distance=f"{distance:.2f}",
    )

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved scenario builder to {output_path}")


def main() -> None:
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    available = list_available_towns(client)
    if not available:
        raise RuntimeError("No CARLA maps found on the server.")

    if args.towns:
        requested = args.towns
    elif args.town:
        requested = [args.town]
    else:
        requested = available

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested town(s) not available: {', '.join(missing)}")

    print("Available CARLA towns:", ", ".join(available))
    print("Building scenario builder for:", ", ".join(requested))

    payloads: dict[str, dict[str, Any]] = {}
    for town in requested:
        print(f"Sampling {town} with spacing {args.distance:.2f} m ...")
        world = client.load_world(town)
        df = sample_town(world, args.distance)
        print(f"  collected {len(df)} reference points.")
        payload = compute_payload(df)
        bev_image, bev_bounds = load_bev_assets(args.bev_dir, town)
        payload["bev_image"] = bev_image
        payload["bev_bounds"] = bev_bounds
        payloads[town] = payload

    generate_html(payloads, args.distance, COLOR_PALETTE, args.output)


if __name__ == "__main__":
    main()
