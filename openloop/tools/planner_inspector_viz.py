#!/usr/bin/env python3
"""planner_inspector_viz.py  —  Comprehensive planner data inspector

Reads a debug_frames JSONL file (produced by openloop_eval.py --dump-frame-debug)
and an optional scenario YAML directory, then generates:

  <out_dir>/summary.png          — Full run overview (trajectory + ADE timeline)
  <out_dir>/frame_NNNN.png       — Per-frame detail pages for sampled frames

Per-frame pages show EVERYTHING fed into and produced by the planner:
  • Bird's-eye world view: ego GT path, predicted waypoints, GT canonical targets,
    PID control rollout, surrounding actor bounding boxes
  • Coordinate transform chain: raw planner output → carla_bev → world → resampled
  • VAD-only: all 6 command branch trajectories in ego-local frame
  • Input audit panel: measurement fields, input file paths, run_step payload schema,
    camera/lidar sanity statistics, VAD branch consistency checks
  • LiDAR top-down panel (from recorded .bin input path)
  • Ego state: speed, compass rose, steer/throttle/brake, latency, staleness
  • ADE breakdown: per-step L2 bar chart (native vs extrapolated steps colored differently)
  • Per-step canonical table: predicted xy, GT xy, validity flags, L2 distance
  • Provenance debug notes from pipeline
  • (Optional) Camera images (loaded from debug-recorded input paths, with scenario fallback)

Usage:
  python planner_inspector_viz.py \\
    --debug-frames /path/to/frames.jsonl \\
    --planner <planner_name> \\
    --out-dir /tmp/inspector_tcp \\
    [--scenario-dir /path/to/yamldir] \\
    [--frame-idx 123]      # exact frame index (repeatable)
    [--frame-stride N]       # sample every N-th frame
    [--max-frames N]         # cap on per-frame figures (default 40)
"""

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── colour palette ─────────────────────────────────────────────────────────────
C_GT_PATH  = '#95a5a6'   # gray        — full ego GT trajectory
C_GT_PTS   = '#27ae60'   # dark green  — GT canonical 6-point targets
C_PRED     = '#e74c3c'   # red         — pred_world_used
C_ROLLOUT  = '#2980b9'   # blue        — PID rollout
C_EGO      = '#f39c12'   # orange      — current ego position
C_NATIVE   = '#c0392b'   # dark red    — native-horizon prediction points
C_EXTRAP   = '#e67e22'   # dark orange — extrapolated prediction points
C_STALE    = '#bdc3c7'   # light gray  — stale frame marker
C_ACTOR    = '#8e44ad'   # purple      — surrounding vehicle bounding boxes

# VAD branch index ordering follows RoadOption(value)-1:
#   0=LEFT, 1=RIGHT, 2=STRAIGHT, 3=LANEFOLLOW, 4=CHANGELANELEFT, 5=CHANGELANERIGHT
VAD_CMD_NAMES  = ['Turn Left', 'Turn Right', 'Go Straight',
                  'Lane Follow', 'Lane Change L', 'Lane Change R']
VAD_CMD_COLORS = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']


def _planner_family(planner: str) -> str:
    tag = str(planner or '').lower()
    if 'colmdriver_rulebase' in tag:
        return 'colmdriver'
    if 'colmdriver' in tag:
        return 'colmdriver'
    if 'vad' in tag:
        return 'vad'
    if 'tcp' in tag:
        return 'tcp'
    return 'generic'


def _first_existing_key(payload: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        if key in payload:
            arr = _arr(payload.get(key))
            if arr is not None:
                return key
    return candidates[0]

# ── geometry helpers ───────────────────────────────────────────────────────────

def _arr(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=float)
    return a if a.size > 0 else None


def _rot2d_deg(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([[c, -s], [s, c]])


def _obb_corners(cx: float, cy: float, half_l: float, half_w: float,
                 yaw_deg: float) -> np.ndarray:
    """Return 5×2 closed polygon of an oriented bounding box."""
    corners = np.array([[half_l, half_w], [half_l, -half_w],
                        [-half_l, -half_w], [-half_l, half_w],
                        [half_l, half_w]])
    R = _rot2d_deg(yaw_deg)
    return corners @ R.T + np.array([cx, cy])


def _pad_limits(ax, *arrays, frac: float = 0.15):
    """Set equal-aspect axis limits with fractional padding over provided point arrays."""
    pts = []
    for a in arrays:
        if a is not None:
            a = np.asarray(a).reshape(-1, 2)
            valid = a[np.isfinite(a).all(axis=1)]
            if len(valid):
                pts.append(valid)
    if not pts:
        return
    all_pts = np.vstack(pts)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    ax.set_xlim(xmin - dx * frac, xmax + dx * frac)
    ax.set_ylim(ymin - dy * frac, ymax + dy * frac)


# ── data loading ───────────────────────────────────────────────────────────────

def load_frames(path: str) -> List[Dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return sorted(frames, key=lambda x: x['frame_idx'])


def _load_yaml(scenario_dir: str, frame_idx: int) -> Optional[Dict]:
    try:
        import yaml
    except ImportError:
        return None
    p = pathlib.Path(scenario_dir) / f"{frame_idx:06d}.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def _load_yaml_from_frame(
    frame: Dict[str, Any],
    scenario_dir: Optional[str],
    frame_idx: int,
) -> Optional[Dict]:
    """Best-effort YAML loader using explicit scenario_dir or debug-recorded path."""
    if scenario_dir:
        y = _load_yaml(scenario_dir, frame_idx)
        if y is not None:
            return y
    input_files = frame.get('input_files') or {}
    yaml_cur = input_files.get('yaml') if isinstance(input_files, dict) else None
    if not yaml_cur:
        return None
    try:
        p = pathlib.Path(str(yaml_cur))
        yp = p.with_name(f"{int(frame_idx):06d}.yaml")
        if not yp.exists():
            return None
        with open(yp) as f:
            import yaml
            return yaml.safe_load(f)
    except Exception:
        return None


# ── shared drawing primitives ─────────────────────────────────────────────────

def _draw_ego_arrow(ax, x: float, y: float, yaw_deg: float,
                    color=C_EGO, size: float = 2.0, label: str = None):
    r = math.radians(yaw_deg)
    dx, dy = math.cos(r) * size, math.sin(r) * size
    kw = dict(arrowstyle='->', color=color, lw=2.0)
    ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y), arrowprops=kw)
    ax.scatter([x], [y], color=color, s=70, zorder=6, label=label)


def _draw_actor_boxes(ax, yaml_data: Optional[Dict], label: bool = True):
    if yaml_data is None:
        return
    vehicles = yaml_data.get('vehicles') or {}
    # vehicles may be a dict keyed by integer IDs or a plain list
    veh_iter = vehicles.values() if isinstance(vehicles, dict) else vehicles
    first = True
    for v in veh_iter:
        if not isinstance(v, dict):
            continue
        loc = v.get('location', [0, 0, 0])
        ext = v.get('extent', [1, 0.5, 0.5])
        ang = v.get('angle', [0, 0, 0])
        yaw = ang[1] if len(ang) > 1 else 0.0
        corners = _obb_corners(loc[0], loc[1], ext[0], ext[1], yaw)
        kw = dict(color=C_ACTOR, alpha=0.65, lw=0.9)
        if first and label:
            ax.plot(corners[:, 0], corners[:, 1], label='Actors', **kw)
            first = False
        else:
            ax.plot(corners[:, 0], corners[:, 1], **kw)


def _get_vehicle_entry(yaml_data: Optional[Dict], actor_id: Optional[int]) -> Optional[Dict]:
    if yaml_data is None or actor_id is None:
        return None
    vehicles = yaml_data.get('vehicles') or {}
    if not isinstance(vehicles, dict):
        return None
    if actor_id in vehicles:
        v = vehicles.get(actor_id)
        return v if isinstance(v, dict) else None
    s = str(actor_id)
    if s in vehicles:
        v = vehicles.get(s)
        return v if isinstance(v, dict) else None
    return None


def _draw_collision_actor_box(
    ax,
    yaml_data: Optional[Dict],
    actor_id: Optional[int],
    *,
    color: str = '#ff2d2d',
    label: bool = True,
) -> bool:
    v = _get_vehicle_entry(yaml_data, actor_id)
    if not isinstance(v, dict):
        return False
    loc = v.get('location', [0, 0, 0])
    ext = v.get('extent', [1, 0.5, 0.5])
    ang = v.get('angle', [0, 0, 0])
    yaw = ang[1] if len(ang) > 1 else 0.0
    corners = _obb_corners(loc[0], loc[1], ext[0], ext[1], yaw)
    kw = dict(color=color, alpha=0.95, lw=2.2, linestyle='--')
    if label:
        ax.plot(corners[:, 0], corners[:, 1], label=f'Collision actor id={actor_id}', **kw)
    else:
        ax.plot(corners[:, 0], corners[:, 1], **kw)
    ax.scatter([float(loc[0])], [float(loc[1])], s=35, color=color, marker='x', zorder=7)
    ax.annotate(
        f'coll actor {actor_id}',
        (float(loc[0]), float(loc[1])),
        fontsize=6,
        color=color,
        xytext=(3, 3),
        textcoords='offset points',
    )
    return True


def _infer_collision_actor_from_frame(
    frame: Dict[str, Any],
    scenario_dir: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Fallback collision-actor inference for legacy debug files without collision_debug."""
    can = frame.get('canonical_scoring_metrics') or {}
    if not bool(can.get('collision')):
        return None
    pred = _arr(can.get('predicted_canonical_points'))
    if pred is None or pred.ndim != 2 or len(pred) < 1:
        return None
    pose = frame.get('pose_world') or []
    if len(pose) < 5:
        return None
    fi = int(frame.get('frame_idx', 0))
    ego_id = frame.get('ego_actor_id')
    try:
        ego_id_int = int(ego_id)
    except Exception:
        ego_id_int = None

    step_index = 0
    frame_delta = 5  # 0.5 s at 10 Hz
    future_fi = fi + frame_delta
    yaml_future = _load_yaml_from_frame(frame, scenario_dir, future_fi)
    if yaml_future is None:
        return None
    vehicles = yaml_future.get('vehicles') or {}
    if not isinstance(vehicles, dict) or not vehicles:
        return None

    yaw = math.radians(float(pose[4]))
    sin_y = math.sin(yaw)
    cos_y = math.cos(yaw)
    pred0 = np.asarray(pred[0, :2], dtype=float)
    best = None
    for aid_raw, v in vehicles.items():
        if not isinstance(v, dict):
            continue
        try:
            aid = int(aid_raw)
        except Exception:
            continue
        if ego_id_int is not None and aid == ego_id_int:
            continue
        loc = v.get('location')
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        actor_xy = np.asarray([float(loc[0]), float(loc[1])], dtype=float)
        diff = pred0 - actor_xy
        lateral = abs(-sin_y * diff[0] + cos_y * diff[1])
        if lateral >= 3.5:
            continue
        longitudinal = float(cos_y * diff[0] + sin_y * diff[1])
        euclidean = float(np.linalg.norm(diff))
        cand = {
            "collision": True,
            "actor_id": int(aid),
            "step_index": int(step_index),
            "frame_delta": int(frame_delta),
            "t_seconds": 0.5,
            "lateral_m": float(lateral),
            "longitudinal_m": float(longitudinal),
            "euclidean_m": euclidean,
            "pred_world_xy": [float(pred0[0]), float(pred0[1])],
            "actor_world_xy": [float(actor_xy[0]), float(actor_xy[1])],
            "inferred_from_legacy_record": True,
        }
        if best is None or float(cand["lateral_m"]) < float(best["lateral_m"]):
            best = cand
    return best


def _compass_rose(ax, heading_rad: float):
    """Draw a self-contained compass rose on the given axis."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    circle = plt.Circle((0, 0), 1.0, fill=False, color='gray', lw=0.8)
    ax.add_patch(circle)
    for label, deg in [('N', 0), ('E', 90), ('S', 180), ('W', 270)]:
        r = math.radians(deg)
        ax.text(1.35 * math.sin(r), 1.35 * math.cos(r), label,
                ha='center', va='center', fontsize=6, color='#555')
        ax.plot([0.85 * math.sin(r), math.sin(r)],
                [0.85 * math.cos(r), math.cos(r)], color='#ccc', lw=0.5)
    hx, hy = math.sin(heading_rad), math.cos(heading_rad)
    ax.annotate('', xy=(hx * 0.82, hy * 0.82), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='navy', lw=2.0))
    ax.scatter([hx * 0.82], [hy * 0.82], color='navy', s=20, zorder=5)


# ── per-frame layout helper ────────────────────────────────────────────────────

def _make_perframe_fig(frame: Dict, all_frames: List[Dict], planner: str,
                        scenario_dir: Optional[str]) -> plt.Figure:
    fi      = frame['frame_idx']
    met     = frame.get('metrics') or {}
    can     = frame.get('canonical_scoring_metrics') or {}
    pad     = frame.get('planner_adapter_debug') or {}
    meas    = frame.get('measurements') or {}
    planner_family = _planner_family(planner)
    is_vad  = (planner_family == 'vad')
    input_files = frame.get('input_files') or {}
    debug_cam_paths = input_files.get('camera_images') if isinstance(input_files, dict) else {}
    has_cams = bool(scenario_dir) or any(
        isinstance(debug_cam_paths, dict) and debug_cam_paths.get(d)
        for d in ('front', 'left', 'right', 'rear')
    )
    has_vad  = is_vad and bool(pad.get('candidate_local_wps_by_command'))

    # Row heights (in inches)
    row_h = [5.0,   # BEV
             3.5,   # transform chain
             2.8,   # ego state + ADE breakdown
             3.1,   # input audit + lidar topdown
             2.5,   # canonical table + provenance
             ]
    if has_vad:
        row_h.insert(2, 3.2)
    if has_cams:
        row_h.append(2.5)

    n_rows = len(row_h)
    fig = plt.figure(figsize=(22, sum(row_h) + 0.6))
    gs  = gridspec.GridSpec(n_rows, 4, figure=fig,
                            height_ratios=row_h,
                            hspace=0.52, wspace=0.35)
    row = 0

    # ── Row 0: Bird's-eye view (full width) ──────────────────────────────────
    ax_bev = fig.add_subplot(gs[row, :])
    _panel_bev(ax_bev, frame, all_frames, planner, scenario_dir)
    row += 1

    # ── Row 1: Coordinate transform chain (4 equal subpanels) ────────────────
    axes_chain = [fig.add_subplot(gs[row, c]) for c in range(4)]
    _panel_transform_chain(axes_chain, frame, planner)
    row += 1

    # ── Row 2 (VAD only): all command branches ────────────────────────────────
    if has_vad:
        ax_vad = fig.add_subplot(gs[row, :])
        _panel_vad_branches(ax_vad, frame)
        row += 1

    # ── Row 2/3: left=ego state+compass | right=ADE breakdown ────────────────
    ax_state = fig.add_subplot(gs[row, :2])
    ax_ade   = fig.add_subplot(gs[row, 2:])
    _panel_ego_state(ax_state, frame)
    _panel_ade_breakdown(ax_ade, frame, planner)
    row += 1

    # ── Row 3/4: left=input audit | right=lidar topdown ──────────────────────
    ax_input = fig.add_subplot(gs[row, :2])
    ax_lidar = fig.add_subplot(gs[row, 2:])
    _panel_input_audit(ax_input, frame, planner)
    _panel_lidar_topdown(ax_lidar, frame)
    row += 1

    # ── Row 4/5: left=canonical table | right=provenance notes ───────────────
    ax_table = fig.add_subplot(gs[row, :2])
    ax_prov  = fig.add_subplot(gs[row, 2:])
    _panel_canonical_table(ax_table, frame)
    _panel_provenance(ax_prov, frame)
    row += 1

    # ── Row N (optional): camera images ──────────────────────────────────────
    if has_cams:
        axes_cams = [fig.add_subplot(gs[row, c]) for c in range(4)]
        _panel_cameras(axes_cams, frame, scenario_dir)
        row += 1

    # ── suptitle ──────────────────────────────────────────────────────────────
    stale_str  = 'STALE ⚠' if met.get('stale') else 'FRESH'
    scored_str = 'SCORED' if can.get('scored') else 'NOT SCORED'
    ade_str    = (f"ADE={can['plan_ade']:.3f}m  FDE={can['plan_fde']:.3f}m"
                  if can.get('plan_ade') is not None else 'ADE=n/a')
    reason = can.get('inclusion_reason', 'unknown')
    extrap = '(+extrap)' if frame.get('trajectory_extrapolation') else ''
    fig.suptitle(
        f'[{planner.upper()}] Frame {fi}  |  {stale_str}  |  {scored_str}  '
        f'|  {ade_str}  |  reason={reason}  {extrap}',
        fontsize=13, fontweight='bold', y=0.998)

    return fig


# ── Panel: Bird's-eye view ────────────────────────────────────────────────────

def _panel_bev(ax, frame, all_frames, planner, scenario_dir):
    fi   = frame['frame_idx']
    can  = frame.get('canonical_scoring_metrics') or {}
    met  = frame.get('metrics') or {}
    pose = frame.get('pose_world', [])
    yaw_deg = pose[4] if len(pose) > 4 else 0.0

    # Full ego GT trajectory
    gt_path = np.array([f['ego_world_xy'] for f in all_frames
                        if f.get('ego_world_xy') is not None], dtype=float)
    if len(gt_path):
        ax.plot(gt_path[:, 0], gt_path[:, 1], color=C_GT_PATH, lw=1.2,
                alpha=0.6, zorder=1, label='Full GT path')
        # Mark frame index every 10 frames for orientation
        for f in all_frames[::10]:
            xy = f.get('ego_world_xy')
            if xy:
                ax.annotate(str(f['frame_idx']), xy, fontsize=4,
                            color='#888', ha='center', va='center', zorder=2)

    # Current ego position + heading
    ego_xy = _arr(frame.get('ego_world_xy'))
    if ego_xy is not None:
        _draw_ego_arrow(ax, ego_xy[0], ego_xy[1], yaw_deg,
                        color=C_EGO, size=2.5, label=f'Ego (fr {fi})')

    # Surrounding actors from YAML
    yaml_d = _load_yaml_from_frame(frame, scenario_dir, fi)
    _draw_actor_boxes(ax, yaml_d, label=True)

    # Collision trigger actor highlight (if collision debug identifies one).
    coll_dbg = can.get('collision_debug') or {}
    if not coll_dbg and bool(can.get('collision')):
        coll_dbg = _infer_collision_actor_from_frame(frame, scenario_dir) or {}
    coll_actor_id = coll_dbg.get('actor_id')
    coll_step = coll_dbg.get('step_index')
    coll_frame_delta = coll_dbg.get('frame_delta')
    if (
        can.get('collision')
        and isinstance(coll_actor_id, (int, np.integer))
        and isinstance(coll_step, (int, np.integer))
    ):
        if not isinstance(coll_frame_delta, (int, np.integer)):
            coll_frame_delta = int((int(coll_step) + 1) * 5)
        coll_frame_idx = int(fi) + int(coll_frame_delta)
        yaml_coll = _load_yaml_from_frame(frame, scenario_dir, coll_frame_idx)
        drew = _draw_collision_actor_box(
            ax,
            yaml_coll,
            int(coll_actor_id),
            label=True,
        )
        if drew:
            lat = coll_dbg.get('lateral_m')
            lon = coll_dbg.get('longitudinal_m')
            euc = coll_dbg.get('euclidean_m')
            tsec = coll_dbg.get('t_seconds')
            ax.text(
                0.01, 0.02,
                (
                    f"collision trigger: actor={int(coll_actor_id)}  "
                    f"step={int(coll_step)}  t={float(tsec):.2f}s  "
                    f"lat={float(lat):.2f}m  lon={float(lon):.2f}m  "
                    f"dist={float(euc):.2f}m"
                    if isinstance(lat, (int, float))
                    and isinstance(lon, (int, float))
                    and isinstance(euc, (int, float))
                    and isinstance(tsec, (int, float))
                    else f"collision trigger: actor={int(coll_actor_id)} step={int(coll_step)}"
                ),
                transform=ax.transAxes,
                fontsize=6.5,
                color='#b30000',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.88),
            )

    # GT canonical 6-point targets
    gt_pts = _arr(can.get('gt_canonical_points') or frame.get('gt_future_world'))
    if gt_pts is not None and gt_pts.ndim == 2:
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], color=C_GT_PTS, s=60, zorder=5,
                   marker='x', linewidths=2, label='GT canonical (6 pts)')
        ax.plot(gt_pts[:, 0], gt_pts[:, 1], color=C_GT_PTS, lw=1.0, alpha=0.7, zorder=4)
        for i, pt in enumerate(gt_pts):
            ax.annotate(f'gt{i}', pt, fontsize=5, color=C_GT_PTS,
                        xytext=(3, 3), textcoords='offset points')

    # Predicted waypoints (pred_world_used) — native vs extrapolated coloring
    pred = _arr(frame.get('pred_world_used'))
    native_len = frame.get('planner_raw_length', 4 if _planner_family(planner) == 'tcp' else 6)
    extra_legend_handles = []
    if pred is not None and pred.ndim == 2:
        ax.plot(pred[:, 0], pred[:, 1], color=C_PRED, lw=1.8, alpha=0.85, zorder=4,
                label='Pred (pred_world_used)')
        for i, pt in enumerate(pred):
            if not np.isfinite(pt).all():
                continue
            color  = C_EXTRAP if i >= native_len else C_NATIVE
            marker = '^'       if i >= native_len else 'o'
            ax.scatter([pt[0]], [pt[1]], color=color, s=55, marker=marker, zorder=6)
            ax.annotate(f'p{i}', pt, fontsize=5, color=color,
                        xytext=(3, 3), textcoords='offset points')
        # Legend proxy patches (do NOT call add_patch — they are artists, not patches)
        extra_legend_handles = [
            mpatches.Patch(color=C_NATIVE, label=f'Pred native (0–{min(native_len,6)-1})'),
            mpatches.Patch(color=C_EXTRAP, label=f'Pred extrap ({native_len}+)'),
        ]

    # PID control rollout
    rollout = _arr(frame.get('control_rollout_world'))
    if rollout is not None and rollout.ndim == 2 and np.isfinite(rollout).all():
        ax.plot(rollout[:, 0], rollout[:, 1], color=C_ROLLOUT, lw=1.2,
                linestyle='--', alpha=0.75, zorder=3, label='PID rollout')

    # Decoration
    stale_str = 'STALE ⚠' if met.get('stale') else 'FRESH'
    speed_mps = (frame.get('measurements') or {}).get('speed_mps', 0.0) or 0.0
    ax.set_xlabel('X world (m, CARLA)', fontsize=8)
    ax.set_ylabel('Y world (m, CARLA)', fontsize=8)
    ax.set_title(f'Bird\'s-Eye World View — Frame {fi}  [{stale_str}]  '
                 f'speed={speed_mps:.2f} m/s ({speed_mps*3.6:.1f} km/h)',
                 fontsize=9)
    ax.legend(loc='upper left', fontsize=6, framealpha=0.7, ncol=3,
              handles=([h for h in ax.get_legend_handles_labels()[0]]
                       + extra_legend_handles))
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')
    _pad_limits(ax, gt_path if len(gt_path) else None,
                gt_pts, pred, rollout, frac=0.08)


# ── Panel: Coordinate transform chain ────────────────────────────────────────

def _panel_transform_chain(axes, frame, planner):
    pad        = frame.get('planner_adapter_debug') or {}
    ego_world  = _arr(frame.get('ego_world_xy'))
    planner_family = _planner_family(planner)
    is_vad = (planner_family == 'vad')
    native_len = frame.get('planner_raw_length', 4 if planner_family == 'tcp' else 6)

    if planner_family == 'vad':
        steps = [
            ('local_wps',         'local_wps\n(ego-local, VAD output)',     'local'),
            ('carla_bev_wps',     'carla_bev_wps\n(axis-swapped / BEV)',    'local'),
            ('world_wps',         'world_wps\n(world, native horizon)',      'world'),
            ('resampled_world_wps',
             f'resampled_world_wps\n(world, 6 canonical pts,\n'
             f'≥{native_len} = extrapolated)', 'world'),
        ]
    elif planner_family == 'colmdriver':
        steps = [
            ('raw_wps',               'raw_wps\n(mapped world-like planner frame)', 'mapped'),
            ('corrected_world_wps',   'corrected_world_wps\n(world after origin correction)', 'world'),
            ('world_wps',             'world_wps\n(world, native horizon)',          'world'),
            ('resampled_world_wps',
             f'resampled_world_wps\n(world, 6 canonical pts,\n'
             f'≥{native_len} = extrapolated)', 'world'),
        ]
    else:
        raw_key = _first_existing_key(pad, ['tcp_wps_raw', 'local_wps', 'raw_wps'])
        if raw_key == 'tcp_wps_raw':
            raw_label = 'tcp_wps_raw\n(ego-local, before axis swap)'
            raw_frame = 'local'
        elif raw_key == 'local_wps':
            raw_label = 'local_wps\n(ego-local planner output)'
            raw_frame = 'local'
        else:
            raw_label = 'raw_wps\n(planner-native frame)'
            raw_frame = 'mapped'
        mid_key = _first_existing_key(pad, ['carla_bev_wps', 'corrected_world_wps'])
        if mid_key == 'carla_bev_wps':
            mid_label = 'carla_bev_wps\n(axis-swapped / BEV)'
            mid_frame = 'local'
        else:
            mid_label = 'corrected_world_wps\n(world corrected)'
            mid_frame = 'world'
        steps = [
            (raw_key,             raw_label,                                 raw_frame),
            (mid_key,             mid_label,                                 mid_frame),
            ('world_wps',         'world_wps\n(world, native horizon)',      'world'),
            ('resampled_world_wps',
             f'resampled_world_wps\n(world, 6 canonical pts,\n'
             f'≥{native_len} = extrapolated)', 'world'),
        ]

    for ax, (key, title, frame_type) in zip(axes, steps):
        pts = _arr(pad.get(key))
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xlabel('X (m)', fontsize=5)
        ax.set_ylabel('Y (m)', fontsize=5)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        if pts is None or pts.ndim != 2:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            continue

        if key == 'resampled_world_wps':
            # Color native vs extrapolated
            for i, pt in enumerate(pts):
                color  = C_EXTRAP if i >= native_len else C_NATIVE
                marker = '^'       if i >= native_len else 'o'
                ax.scatter([pt[0]], [pt[1]], color=color, s=40, marker=marker, zorder=4)
                ax.annotate(f't{i}', pt, fontsize=5, color=color,
                            xytext=(2, 2), textcoords='offset points')
            ax.plot(pts[:, 0], pts[:, 1], color=C_PRED, lw=1.2, alpha=0.8, zorder=3)
        else:
            ax.scatter(pts[:, 0], pts[:, 1], color=C_PRED, s=38, zorder=4)
            ax.plot(pts[:, 0], pts[:, 1], color=C_PRED, lw=1.2, alpha=0.8, zorder=3)
            for i, pt in enumerate(pts):
                ax.annotate(str(i), pt, fontsize=5, color='#555',
                            xytext=(2, 2), textcoords='offset points')

        if frame_type == 'local':
            ax.scatter([0], [0], color=C_EGO, s=60, marker='*', zorder=5)
            ax.axhline(0, color='gray', lw=0.4, alpha=0.35)
            ax.axvline(0, color='gray', lw=0.4, alpha=0.35)
            ax.annotate('ego', (0, 0), fontsize=5, color=C_EGO,
                        xytext=(2, 3), textcoords='offset points')
        else:
            if ego_world is not None:
                ax.scatter([ego_world[0]], [ego_world[1]], color=C_EGO,
                           s=60, marker='*', zorder=5, label='ego')
                ax.annotate('ego', ego_world, fontsize=5, color=C_EGO,
                            xytext=(2, 3), textcoords='offset points')
        _pad_limits(ax, pts, frac=0.2)


# ── Panel: VAD command branches ───────────────────────────────────────────────

def _panel_vad_branches(ax, frame):
    pad      = frame.get('planner_adapter_debug') or {}
    branches = pad.get('candidate_local_wps_by_command')
    selected = pad.get('selected_command_index', -1)

    if branches is None:
        ax.text(0.5, 0.5, 'No branch data', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_title('VAD: All Command Branches (ego-local frame)', fontsize=8)
        return

    branches = np.asarray(branches, dtype=float)  # (n_cmds, n_steps, 2)
    all_pts  = []
    for cmd_i, branch in enumerate(branches):
        name   = VAD_CMD_NAMES[cmd_i] if cmd_i < len(VAD_CMD_NAMES) else f'cmd{cmd_i}'
        color  = VAD_CMD_COLORS[cmd_i % len(VAD_CMD_COLORS)]
        is_sel = (cmd_i == selected)
        alpha  = 1.0 if is_sel else 0.35
        lw     = 2.8 if is_sel else 0.9
        marker = 'D' if is_sel else 'o'
        lbl    = f'[{cmd_i}] {name}' + (' ◀ SELECTED' if is_sel else '')

        ax.plot(branch[:, 0], branch[:, 1], color=color, lw=lw, alpha=alpha,
                label=lbl, zorder=4 if is_sel else 2)
        ax.scatter(branch[:, 0], branch[:, 1], color=color, s=25,
                   alpha=alpha, marker=marker, zorder=5 if is_sel else 3)
        if is_sel:
            for i, pt in enumerate(branch):
                ax.annotate(f't{i}', pt, fontsize=5, color=color,
                            xytext=(2, 2), textcoords='offset points')
        all_pts.append(branch)

    ax.scatter([0], [0], color=C_EGO, s=80, marker='*', zorder=6, label='ego origin')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.3)
    ax.axvline(0, color='gray', lw=0.4, alpha=0.3)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X local (m)', fontsize=7)
    ax.set_ylabel('Y local (m)', fontsize=7)
    sel_name = VAD_CMD_NAMES[selected] if 0 <= selected < len(VAD_CMD_NAMES) else '?'
    vad_diag = frame.get('vad_branch_diagnostics') or {}
    best_idx = vad_diag.get('best_branch_index_by_local_ade')
    if isinstance(best_idx, (int, np.integer)) and 0 <= int(best_idx) < len(VAD_CMD_NAMES):
        best_name = VAD_CMD_NAMES[int(best_idx)]
        ax.set_title(
            f'VAD: All {len(branches)} Command Branches in Ego-Local Frame  '
            f'—  selected={selected} ({sel_name})  best={int(best_idx)} ({best_name})',
            fontsize=9,
        )
    else:
        ax.set_title(
            f'VAD: All {len(branches)} Command Branches in Ego-Local Frame  '
            f'—  selected={selected} ({sel_name})',
            fontsize=9,
        )
    ax.legend(loc='upper left', fontsize=6, ncol=3, framealpha=0.7)
    ax.grid(True, alpha=0.2)
    _pad_limits(ax, *all_pts, frac=0.15)


# ── Panel: Ego state ──────────────────────────────────────────────────────────

def _panel_ego_state(ax, frame):
    meas    = frame.get('measurements') or {}
    ctrl    = frame.get('control') or {}
    met     = frame.get('metrics') or {}
    can     = frame.get('canonical_scoring_metrics') or {}
    pose    = frame.get('pose_world') or []

    speed     = meas.get('speed_mps',  0.0) or 0.0
    compass   = frame.get('resolved_compass_rad', 0.0) or 0.0
    steer     = ctrl.get('steer',    0.0) or 0.0
    throttle  = ctrl.get('throttle', 0.0) or 0.0
    brake     = ctrl.get('brake',    0.0) or 0.0
    stale     = met.get('stale', False)
    latency   = frame.get('latency_s', None)
    src       = frame.get('traj_source_selected', 'unknown')
    extrap    = frame.get('trajectory_extrapolation', False)
    raw_len   = frame.get('planner_raw_length', '?')

    yaw_deg   = pose[4] if len(pose) > 4 else 0.0
    pose_str  = (f"[{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]  "
                 f"roll={pose[3]:.2f}°  yaw={pose[4]:.2f}°  pitch={pose[5]:.2f}°"
                 if len(pose) >= 6 else str(pose))

    lat_str   = f"{latency*1000:.1f} ms" if latency is not None else "n/a"

    lines = [
        f"  Frame idx : {frame['frame_idx']}",
        f"  Timestamp : {frame.get('timestamp_s', 0.0):.4f} s  (dt={frame.get('runtime_dt_s',0.0):.4f} s)",
        f"  Speed     : {speed:.4f} m/s  ({speed*3.6:.2f} km/h)",
        f"  Compass   : {math.degrees(compass):.2f}°  ({compass:.5f} rad)",
        f"  GPS       : x={meas.get('gps_x',0.0):.6f}  y={meas.get('gps_y',0.0):.6f}",
        f"  Pose(xyz) : {pose_str}",
        "",
        f"  Steer     : {steer:+.4f}",
        f"  Throttle  : {throttle:.4f}",
        f"  Brake     : {brake:.4f}",
        "",
        f"  Stale     : {'YES ⚠' if stale else 'no'}",
        f"  Traj src  : {src}",
        f"  Extrap    : {'YES — last steps linearly extrapolated' if extrap else 'no'}",
        f"  Raw len   : {raw_len} native wps",
        f"  Latency   : {lat_str}",
        f"  Scored    : {can.get('scored', False)}  ({can.get('inclusion_reason', '?')})",
    ]

    ax.axis('off')
    ax.text(0.02, 0.97, '\n'.join(lines), va='top', ha='left',
            transform=ax.transAxes, fontsize=7.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffff0', alpha=0.9))

    # Compass rose inset
    axins = ax.inset_axes([0.70, 0.05, 0.27, 0.90])
    _compass_rose(axins, compass)
    axins.set_title('Heading\n(CARLA: 0=N, CW+)', fontsize=5.5)
    ax.set_title('Ego State & Control Inputs', fontsize=8)


def _panel_input_audit(ax, frame, planner):
    """Dense text panel for input provenance and consistency checks."""
    meas = frame.get('measurements') or {}
    stats = frame.get('sensor_stats') or {}
    input_files = frame.get('input_files') or {}
    input_data_overview = frame.get('input_data_overview') or {}
    pad = frame.get('planner_adapter_debug') or {}

    def _fmt_float(v, nd=3):
        try:
            fv = float(v)
        except Exception:
            return "n/a"
        if not np.isfinite(fv):
            return "n/a"
        return f"{fv:.{nd}f}"

    def _fmt_path(p):
        if not p:
            return "n/a"
        s = str(p)
        return s if len(s) <= 95 else ("..." + s[-92:])

    lines = []
    lines.append("Input Provenance")
    lines.append(f"  yaml      : {_fmt_path(input_files.get('yaml'))}")
    lines.append(f"  lidar_bin : {_fmt_path(input_files.get('lidar_bin'))}")
    cam_imgs = input_files.get('camera_images') or {}
    for d in ("front", "left", "right", "rear"):
        lines.append(f"  cam_{d:<5}: {_fmt_path(cam_imgs.get(d))}")

    lines.append("")
    lines.append("Planner Inputs (measurement fields)")
    lines.append(
        f"  speed={_fmt_float(meas.get('speed_mps'))} m/s  "
        f"gps=({_fmt_float(meas.get('gps_x'))}, {_fmt_float(meas.get('gps_y'))})  "
        f"compass={_fmt_float(meas.get('compass'), nd=5)} rad"
    )
    lines.append(
        f"  command={meas.get('command', 'n/a')}  "
        f"target_point={meas.get('target_point', 'n/a')}"
    )
    lines.append(
        f"  lidar_pose=({_fmt_float(meas.get('lidar_pose_x'))}, "
        f"{_fmt_float(meas.get('lidar_pose_y'))}, {_fmt_float(meas.get('lidar_pose_z'))})"
    )

    lidar_stats = stats.get('lidar') or {}
    if lidar_stats:
        lines.append("")
        lines.append("Sensor Stats")
        lines.append(
            f"  lidar shape={lidar_stats.get('shape', '?')}  "
            f"pts={lidar_stats.get('num_points', '?')}  "
            f"xy_range=[{_fmt_float(lidar_stats.get('xy_range_min'))}, "
            f"{_fmt_float(lidar_stats.get('xy_range_max'))}] m"
        )
        if lidar_stats.get('xyz_min') is not None and lidar_stats.get('xyz_max') is not None:
            xyz_min = lidar_stats.get('xyz_min')
            xyz_max = lidar_stats.get('xyz_max')
            lines.append(
                f"  lidar xyz_min={xyz_min}  xyz_max={xyz_max}"
            )
    cam_stats = (stats.get('cameras') or {}) if isinstance(stats, dict) else {}
    for d in ("front", "left", "right", "rear"):
        c = cam_stats.get(d) or {}
        if c.get('available'):
            lines.append(
                f"  cam_{d:<5} shape={c.get('shape', '?')}  "
                f"mean={_fmt_float(c.get('mean'), nd=2)}  "
                f"std={_fmt_float(c.get('std'), nd=2)}  "
                f"nz={_fmt_float(c.get('nonzero_fraction'), nd=3)}"
            )
        else:
            lines.append(f"  cam_{d:<5} unavailable")

    lines.append("")
    keys = sorted(input_data_overview.keys())
    preview = ", ".join(keys[:10]) + (" ..." if len(keys) > 10 else "")
    lines.append(f"run_step input_data keys ({len(keys)}): {preview}")

    if _planner_family(planner) == 'vad':
        local = _arr(pad.get('local_wps'))
        branches = _arr(pad.get('candidate_local_wps_by_command'))
        selected = pad.get('selected_command_index')
        vad_diag = frame.get('vad_branch_diagnostics') or {}
        branch_msg = "n/a"
        try:
            cmd = int(selected)
        except Exception:
            cmd = -1
        if local is not None and branches is not None and branches.ndim == 3 and 0 <= cmd < len(branches):
            n = min(len(local), len(branches[cmd]))
            if n > 0:
                diff = np.linalg.norm(local[:n] - branches[cmd][:n], axis=1)
                branch_msg = (
                    f"selected={cmd}  mean_l2={np.mean(diff):.4f} m  "
                    f"max_l2={np.max(diff):.4f} m"
                )
        lines.append(f"VAD branch consistency: {branch_msg}")
        if vad_diag:
            route_val = vad_diag.get('route_command_value')
            route_name = vad_diag.get('route_command_name')
            sel_idx = vad_diag.get('selected_branch_index')
            sel_name = vad_diag.get('selected_command_name')
            sel_raw = vad_diag.get('selected_command_raw_value')
            sel_fallback = vad_diag.get('selected_command_fallback_to_lanefollow')
            best_idx = vad_diag.get('best_branch_index_by_local_ade')
            best_name = vad_diag.get('best_command_name_by_local_ade')
            sel_ade = vad_diag.get('selected_branch_local_ade_m')
            best_ade = vad_diag.get('best_branch_local_ade_m')
            gap = vad_diag.get('selected_minus_best_local_ade_m')

            def _f(v):
                try:
                    fv = float(v)
                except Exception:
                    return "n/a"
                return f"{fv:.3f}" if np.isfinite(fv) else "n/a"

            lines.append(
                "VAD route/select: "
                f"route_cmd={route_val}({route_name})  "
                f"selected={sel_idx}({sel_name})  "
                f"best={best_idx}({best_name})"
            )
            lines.append(
                "VAD command raw/fallback: "
                f"raw={sel_raw}  fallback_to_lanefollow={sel_fallback}"
            )
            lines.append(
                "VAD local-ADE: "
                f"selected={_f(sel_ade)} m  best={_f(best_ade)} m  "
                f"gap={_f(gap)} m"
            )

    can = frame.get('canonical_scoring_metrics') or {}
    coll_dbg = can.get('collision_debug') or {}
    if not coll_dbg and bool(can.get('collision')):
        coll_dbg = _infer_collision_actor_from_frame(frame, None) or {}
    coll_actor = coll_dbg.get('actor_id')
    coll_step = coll_dbg.get('step_index')
    if (
        bool(can.get('collision'))
        and isinstance(coll_actor, (int, np.integer))
        and isinstance(coll_step, (int, np.integer))
    ):
        frame_delta = coll_dbg.get('frame_delta')
        if not isinstance(frame_delta, (int, np.integer)):
            frame_delta = int((int(coll_step) + 1) * 5)
        collision_frame_idx = int(frame.get('frame_idx', 0)) + int(frame_delta)
        yaml_coll = _load_yaml_from_frame(frame, None, collision_frame_idx)
        veh = _get_vehicle_entry(yaml_coll, int(coll_actor))
        lines.append("")
        lines.append("Collision Trigger Actor")
        lines.append(
            f"  actor_id={int(coll_actor)}  step={int(coll_step)}  "
            f"frame={collision_frame_idx}  t={_fmt_float(coll_dbg.get('t_seconds'))} s"
        )
        if bool(coll_dbg.get('inferred_from_legacy_record')):
            lines.append("  source=legacy fallback inference (collision_debug missing)")
        lines.append(
            f"  lateral={_fmt_float(coll_dbg.get('lateral_m'))} m  "
            f"longitudinal={_fmt_float(coll_dbg.get('longitudinal_m'))} m  "
            f"euclidean={_fmt_float(coll_dbg.get('euclidean_m'))} m"
        )
        if isinstance(veh, dict):
            lines.append(f"  obj_type={veh.get('obj_type', 'n/a')}  attr={veh.get('attribute', 'n/a')}")
            lines.append(f"  location={veh.get('location', 'n/a')}")
            lines.append(f"  angle={veh.get('angle', 'n/a')}")
            lines.append(f"  extent={veh.get('extent', 'n/a')}")
            for k in sorted(veh.keys()):
                if k in {"obj_type", "attribute", "location", "angle", "extent"}:
                    continue
                lines.append(f"  {k}={veh.get(k)}")
        else:
            lines.append("  actor metadata unavailable in collision-step YAML")

    ax.axis('off')
    ax.text(0.02, 0.97, "\n".join(lines),
            va='top', ha='left', transform=ax.transAxes,
            fontsize=6.4, family='monospace',
            bbox=dict(boxstyle='round,pad=0.42', facecolor='#f8fbff', alpha=0.9))
    ax.set_title('Input Audit (Files, Measurements, Sensor Stats)', fontsize=8)


def _panel_lidar_topdown(ax, frame):
    """Top-down LiDAR quicklook for one frame (best-effort)."""
    input_files = frame.get('input_files') or {}
    lidar_path = input_files.get('lidar_bin')
    if not lidar_path:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No lidar path in debug frame',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='gray')
        ax.set_title('LiDAR Top-Down', fontsize=8)
        return
    p = pathlib.Path(str(lidar_path))
    if not p.exists():
        ax.axis('off')
        ax.text(0.5, 0.5, f'LiDAR file missing:\n{p}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=7, color='gray')
        ax.set_title('LiDAR Top-Down', fontsize=8)
        return

    try:
        arr = np.fromfile(str(p), dtype=np.float32)
        if arr.size < 4:
            raise ValueError("empty or malformed .bin file")
        arr = arr.reshape(-1, 4)
        xyz = arr[:, :3]
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        if len(xyz) == 0:
            raise ValueError("no finite xyz points")

        max_pts = 60000
        if len(xyz) > max_pts:
            keep = np.linspace(0, len(xyz) - 1, max_pts).astype(int)
            xyz = xyz[keep]

        sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2],
                        s=0.8, alpha=0.55, cmap='viridis', linewidths=0)
        plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label='z (m)')
        ax.scatter([0], [0], c='red', s=35, marker='x', label='ego lidar origin', zorder=5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('lidar x (m)', fontsize=7)
        ax.set_ylabel('lidar y (m)', fontsize=7)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=6, framealpha=0.7)
        ax.set_title(f'LiDAR Top-Down (from {p.name})', fontsize=8)
    except Exception as exc:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Failed to load LiDAR:\n{exc}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='gray')
        ax.set_title('LiDAR Top-Down', fontsize=8)


# ── Panel: ADE breakdown ──────────────────────────────────────────────────────

def _panel_ade_breakdown(ax, frame, planner):
    can         = frame.get('canonical_scoring_metrics') or {}
    native_len  = frame.get('planner_raw_length', 4 if _planner_family(planner) == 'tcp' else 6)
    breakdown   = can.get('ade_breakdown_m')

    if not breakdown:
        ax.text(0.5, 0.5, 'Not scored\n(stale or insufficient horizon)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        ax.set_title('Per-Step ADE Breakdown', fontsize=8)
        ax.set_visible(True)
        return

    ade  = np.asarray(breakdown, dtype=float)
    n    = len(ade)
    xs   = list(range(n))
    clrs = [C_EXTRAP if i >= native_len else C_NATIVE for i in xs]

    bars = ax.bar(xs, ade, color=clrs, alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ade):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05, f'{val:.2f}',
                ha='center', va='bottom', fontsize=6.5)

    ts = can.get('canonical_timestamps', [0.5 * (i + 1) for i in range(n)])
    ax.set_xticks(xs)
    ax.set_xticklabels([f't{i}\n({ts[i]:.1f}s)' for i in xs], fontsize=6)
    ax.set_ylabel('L2 distance to GT (m)', fontsize=7)
    ax.set_xlabel('Canonical step', fontsize=7)
    mean_ade = can.get('plan_ade')
    fde      = can.get('plan_fde')
    suf = (f'  —  ADE={mean_ade:.3f}m  FDE={fde:.3f}m'
           if mean_ade is not None else '')
    ax.set_title(f'Per-Step ADE Breakdown{suf}', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    lh = [mpatches.Patch(color=C_NATIVE,
                         label=f'Native (steps 0–{min(native_len, n) - 1})'),
          mpatches.Patch(color=C_EXTRAP,
                         label=f'Extrapolated (steps {native_len}+)')]
    ax.legend(handles=lh, fontsize=6.5, loc='upper left')


# ── Panel: Canonical table ────────────────────────────────────────────────────

def _panel_canonical_table(ax, frame):
    can     = frame.get('canonical_scoring_metrics') or {}
    gt_pts  = _arr(can.get('gt_canonical_points'))
    pred_pts= _arr(can.get('predicted_canonical_points'))
    valid   = can.get('valid_mask',    [])
    gt_v    = can.get('gt_valid_mask', [])
    ts      = can.get('canonical_timestamps', [0.5 * (i + 1) for i in range(6)])
    bkdn    = can.get('ade_breakdown_m', [])

    rows = []
    for i in range(6):
        t  = f'{ts[i]:.1f}s' if i < len(ts) else '?'
        if pred_pts is not None and i < len(pred_pts):
            px, py = f'{pred_pts[i,0]:.2f}', f'{pred_pts[i,1]:.2f}'
        else:
            px, py = '?', '?'
        if gt_pts is not None and i < len(gt_pts):
            gx, gy = f'{gt_pts[i,0]:.2f}', f'{gt_pts[i,1]:.2f}'
        else:
            gx, gy = '?', '?'
        v  = '✓' if i < len(valid)  and valid[i]  else '✗'
        gv = '✓' if i < len(gt_v)   and gt_v[i]   else '✗'
        l2 = '—'
        if i < len(bkdn):
            try:
                l2_val = float(bkdn[i])
            except (TypeError, ValueError):
                l2_val = float('nan')
            if np.isfinite(l2_val):
                l2 = f'{l2_val:.3f}'
        rows.append([str(i), t, px, py, gx, gy, v, gv, l2])

    ax.axis('off')
    col_labels = ['Step', 'ts', 'pred_x', 'pred_y', 'gt_x', 'gt_y',
                  'valid', 'gt_valid', 'L2 (m)']
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.25)

    # Colour L2 cells by magnitude
    for i, row in enumerate(rows):
        try:
            val = float(row[-1])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        cell = tbl[i + 1, len(col_labels) - 1]  # +1 for header
        r = min(val / 15.0, 1.0)   # 0→white, 15m→red
        cell.set_facecolor((1.0, 1.0 - 0.7 * r, 1.0 - 0.7 * r))

    ax.set_title('Per-Step Canonical Points, GT, Validity, L2', fontsize=8)


# ── Panel: Provenance notes ───────────────────────────────────────────────────

def _panel_provenance(ax, frame):
    can   = frame.get('canonical_scoring_metrics') or {}
    notes = can.get('provenance_debug_notes', [])
    if isinstance(notes, str):
        notes = [notes]
    if not notes:
        notes = ['(no provenance notes recorded)']

    ax.axis('off')
    bullet_text = '\n'.join(f'• {n}' for n in notes)
    ax.text(0.02, 0.97, 'Provenance / Pipeline Notes:\n\n' + bullet_text,
            va='top', ha='left', transform=ax.transAxes,
            fontsize=5.5, wrap=False,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', alpha=0.85))
    ax.set_title('Pipeline Provenance Debug Notes', fontsize=8)


# ── Panel: Camera images ──────────────────────────────────────────────────────

def _panel_cameras(axes, frame, scenario_dir):
    fi = frame['frame_idx']
    try:
        from PIL import Image as PILImage
        pil_ok = True
    except ImportError:
        pil_ok = False

    yaml_d = _load_yaml(scenario_dir, fi) if scenario_dir else None
    input_files = frame.get('input_files') or {}
    cam_paths = input_files.get('camera_images') if isinstance(input_files, dict) else {}

    debug_has_directional = (
        isinstance(cam_paths, dict)
        and any(cam_paths.get(d) for d in ('front', 'left', 'right', 'rear'))
    )
    if debug_has_directional:
        camera_slots = [('front', cam_paths.get('front')),
                        ('left', cam_paths.get('left')),
                        ('right', cam_paths.get('right')),
                        ('rear', cam_paths.get('rear'))]
    else:
        camera_slots = [('cam1', None), ('cam2', None), ('cam3', None), ('cam4', None)]

    for ax, (cam_key, debug_path) in zip(axes, camera_slots):
        ax.axis('off')
        if not pil_ok:
            ax.text(0.5, 0.5, f'{cam_key}\n(Pillow not installed)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(cam_key, fontsize=7)
            continue

        # Resolve image path from frame-debug manifest first, then scenario fallback.
        img_path = None
        if debug_path and os.path.exists(str(debug_path)):
            img_path = str(debug_path)
        if img_path is None and yaml_d and cam_key in {'cam1', 'cam2', 'cam3', 'cam4'}:
            raw = yaml_d.get(cam_key, '')
            if isinstance(raw, dict):
                raw = raw.get('file', '')
            if raw:
                cand = raw if os.path.isabs(raw) else os.path.join(scenario_dir, raw)
                if os.path.exists(cand):
                    img_path = cand
        if img_path is None and scenario_dir:
            for ext in ('.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG'):
                cand = os.path.join(scenario_dir, f'{fi:06d}_{cam_key}{ext}')
                if os.path.exists(cand):
                    img_path = cand
                    break

        if img_path:
            try:
                img = np.array(PILImage.open(img_path))
                ax.imshow(img)
                ax.set_title(f'{cam_key}  ({os.path.basename(img_path)})', fontsize=6)
            except Exception as e:
                ax.text(0.5, 0.5, f'{cam_key}\nerr: {e}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=6)
                ax.set_title(cam_key, fontsize=6)
        else:
            ax.text(0.5, 0.5, f'{cam_key}\nnot found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(cam_key, fontsize=7)


# ── Summary figure ────────────────────────────────────────────────────────────

def make_summary_figure(frames: List[Dict], planner: str, out_path: str):
    fig = plt.figure(figsize=(24, 20))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    ax_traj  = fig.add_subplot(gs[0, :2])   # trajectory coloured by ADE (wide)
    ax_meta  = fig.add_subplot(gs[0, 2])    # run metadata text box
    ax_ade   = fig.add_subplot(gs[1, :2])   # ADE/FDE timeline
    ax_speed = fig.add_subplot(gs[1, 2])    # speed + control history
    ax_inc   = fig.add_subplot(gs[2, 0])    # inclusion reason histogram
    ax_fit   = fig.add_subplot(gs[2, 1])    # transform fidelity scatter
    ax_heat  = fig.add_subplot(gs[2, 2])    # per-step ADE heatmap

    _sum_trajectory(ax_traj, frames, planner)
    _sum_metadata(ax_meta, frames, planner)
    _sum_ade_timeline(ax_ade, frames)
    _sum_speed_profile(ax_speed, frames)
    _sum_inclusion_histogram(ax_inc, frames)
    _sum_transform_fidelity(ax_fit, frames, planner)
    _sum_ade_heatmap(ax_heat, frames)

    scen_id = frames[0].get('scenario', '?')
    fig.suptitle(f'[{planner.upper()}] Run Summary  —  {scen_id}  '
                 f'({len(frames)} frames)', fontsize=14, fontweight='bold')
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  Summary saved: {out_path}')


def _sum_trajectory(ax, frames, planner):
    gt_path = np.array([f['ego_world_xy'] for f in frames
                        if f.get('ego_world_xy')], dtype=float)
    if not len(gt_path):
        return
    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    ade_vals = np.array([c.get('plan_ade', np.nan) for c in can_list], dtype=float)

    ax.plot(gt_path[:, 0], gt_path[:, 1], color=C_GT_PATH, lw=1.0, alpha=0.45,
            zorder=1, label='GT path')
    # Stale frames
    stale_mask = np.array([( f.get('metrics') or {} ).get('stale', False) for f in frames])
    if stale_mask.any():
        ax.scatter(gt_path[stale_mask, 0], gt_path[stale_mask, 1],
                   color=C_STALE, s=8, alpha=0.4, zorder=2, label='Stale')

    # Scored frames coloured by ADE
    scored_mask = np.isfinite(ade_vals)
    if scored_mask.any():
        vmax = np.nanpercentile(ade_vals, 95)
        sc = ax.scatter(gt_path[scored_mask, 0], gt_path[scored_mask, 1],
                        c=ade_vals[scored_mask], cmap='RdYlGn_r',
                        s=45, zorder=4, vmin=0, vmax=vmax, label='Scored (ADE)')
        plt.colorbar(sc, ax=ax, label='ADE (m)', shrink=0.65, pad=0.01)

    # Mark start and end
    ax.annotate('START', gt_path[0], fontsize=7, color='green',
                xytext=(4, 4), textcoords='offset points')
    ax.annotate('END', gt_path[-1], fontsize=7, color='red',
                xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('X world (m)', fontsize=8)
    ax.set_ylabel('Y world (m)', fontsize=8)
    ax.set_title('Full Ego Trajectory (scored frames coloured by ADE)', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='datalim')


def _sum_metadata(ax, frames, planner):
    ax.axis('off')
    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    n_scored  = sum(1 for c in can_list if c.get('scored'))
    n_stale   = sum(1 for f in frames if ( f.get('metrics') or {} ).get('stale'))
    n_insuff  = sum(1 for c in can_list if c.get('inclusion_reason') == 'insufficient_horizon')
    n_fresh   = sum(1 for c in can_list if c.get('is_fresh_plan'))
    ade_vals  = [c.get('plan_ade') for c in can_list if c.get('plan_ade') is not None]
    fde_vals  = [c.get('plan_fde') for c in can_list if c.get('plan_fde') is not None]
    mean_ade  = np.mean(ade_vals) if ade_vals else float('nan')
    mean_fde  = np.mean(fde_vals) if fde_vals else float('nan')

    scen = frames[0].get('scenario', '?')
    lines = [
        f'Scenario  : {scen}',
        f'Planner   : {planner.upper()}',
        f'Frames    : {len(frames)}',
        '',
        f'Scored    : {n_scored}',
        f'Stale     : {n_stale}',
        f'Insuff.   : {n_insuff}',
        f'Fresh     : {n_fresh}',
        '',
        f'Mean ADE  : {mean_ade:.4f} m',
        f'Mean FDE  : {mean_fde:.4f} m',
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), va='top', ha='left',
            transform=ax.transAxes, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='#eaf4fb', alpha=0.9))
    ax.set_title('Run Summary', fontsize=9)


def _sum_ade_timeline(ax, frames):
    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    fi_arr   = np.array([f['frame_idx'] for f in frames])

    scored_fi, scored_ade, scored_fde = [], [], []
    for f, c in zip(frames, can_list):
        if c.get('scored') and c.get('plan_ade') is not None:
            scored_fi.append(f['frame_idx'])
            scored_ade.append(c['plan_ade'])
            scored_fde.append(c.get('plan_fde', np.nan))

    if scored_fi:
        ax.plot(scored_fi, scored_ade, color=C_PRED, lw=1.8, marker='o',
                ms=4, zorder=4, label='ADE (scored)')
        ax.plot(scored_fi, scored_fde, color=C_GT_PTS, lw=1.2, ls='--',
                marker='s', ms=3, zorder=3, label='FDE (scored)')

    # Shade inclusion reason background
    reason_colors = {
        'stale': ('#bdc3c7', 0.08),
        'insufficient_horizon': ('#9b59b6', 0.10),
        'fresh': ('#2ecc71', 0.06),
    }
    for f, c in zip(frames, can_list):
        reason = c.get('inclusion_reason', 'unknown')
        if reason in reason_colors:
            col, alpha = reason_colors[reason]
            ax.axvspan(f['frame_idx'] - 0.5, f['frame_idx'] + 0.5,
                       alpha=alpha, color=col)

    # Legend patches for background zones
    handles, labels = ax.get_legend_handles_labels()
    for label, (col, alpha) in reason_colors.items():
        handles.append(mpatches.Patch(color=col, alpha=0.4, label=label))
        labels.append(label)
    ax.legend(handles=handles, labels=labels, fontsize=6, loc='upper left', ncol=2)
    ax.set_xlabel('Frame index', fontsize=8)
    ax.set_ylabel('Error (m)', fontsize=8)
    ax.set_title('ADE / FDE vs Frame Index\n(background shaded by inclusion reason)',
                 fontsize=9)
    ax.grid(True, alpha=0.25)


def _sum_speed_profile(ax, frames):
    fi      = [f['frame_idx'] for f in frames]
    speeds  = [(f.get('measurements') or {}).get('speed_mps', 0.0) or 0.0 for f in frames]
    throt   = [(f.get('control') or {}).get('throttle', 0.0) or 0.0 for f in frames]
    brk     = [(f.get('control') or {}).get('brake', 0.0) or 0.0 for f in frames]
    steer   = [(f.get('control') or {}).get('steer', 0.0) or 0.0 for f in frames]

    ax.plot(fi, speeds, color=C_EGO, lw=1.8, label='Speed (m/s)', zorder=3)
    ax2 = ax.twinx()
    ax2.fill_between(fi, throt, alpha=0.25, color='green', label='Throttle')
    ax2.fill_between(fi, [-b for b in brk], alpha=0.25, color='red',  label='Brake')
    ax2.plot(fi, steer, color='#3498db', lw=0.9, linestyle=':', alpha=0.7, label='Steer')
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('Control', fontsize=7)

    ax.set_xlabel('Frame', fontsize=7)
    ax.set_ylabel('Speed (m/s)', fontsize=7)
    ax.set_title('Speed & Control History', fontsize=9)

    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.2)


def _sum_inclusion_histogram(ax, frames):
    from collections import Counter
    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    reasons  = [c.get('inclusion_reason', 'unknown') for c in can_list]
    counts   = Counter(reasons)

    colour_map = {
        'stale': C_STALE,
        'fresh': C_GT_PTS,
        'insufficient_horizon': '#9b59b6',
        'unknown': 'lightgray',
    }
    items = sorted(counts.items(), key=lambda x: -x[1])
    labels, vals = zip(*items) if items else ([], [])
    colors = [colour_map.get(l, '#3498db') for l in labels]

    bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.3, str(v), ha='center', fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha='right', fontsize=7)
    ax.set_ylabel('Frame count', fontsize=7)
    ax.set_title('Frame Inclusion Reasons', fontsize=9)
    ax.grid(axis='y', alpha=0.25)


def _sum_transform_fidelity(ax, frames, planner):
    """Scatter: planner-frame distance from ego vs world-frame distance from ego."""
    planner_family = _planner_family(planner)
    raw_dist, world_dist = [], []
    for f in frames:
        pad  = f.get('planner_adapter_debug') or {}
        if planner_family == 'colmdriver':
            raw = _arr(pad.get('corrected_world_wps'))
        else:
            raw = _arr(pad.get('tcp_wps_raw') or pad.get('local_wps'))
        wld  = _arr(pad.get('world_wps'))
        ego  = _arr(f.get('ego_world_xy'))
        if raw is None or wld is None or ego is None:
            continue
        if raw.ndim != 2 or wld.ndim != 2:
            continue
        n = min(len(raw), len(wld))
        for i in range(n):
            if planner_family == 'colmdriver':
                raw_dist.append(float(np.linalg.norm(raw[i] - ego)))
            else:
                raw_dist.append(float(np.linalg.norm(raw[i])))
            world_dist.append(float(np.linalg.norm(wld[i] - ego)))

    if raw_dist:
        ax.scatter(raw_dist, world_dist, alpha=0.5, s=10, color=C_PRED)
        mn = min(min(raw_dist), min(world_dist))
        mx = max(max(raw_dist), max(world_dist))
        ax.plot([mn, mx], [mn, mx], 'k--', lw=0.9, alpha=0.6, label='y=x (ideal)')
        # Residuals
        residuals = np.array(raw_dist) - np.array(world_dist)
        ax.text(0.05, 0.95, f'Max residual: {np.abs(residuals).max():.3f} m\n'
                             f'Mean |residual|: {np.abs(residuals).mean():.4f} m',
                va='top', ha='left', transform=ax.transAxes, fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7))
    if planner_family == 'colmdriver':
        ax.set_xlabel('||corrected_world_pt_i − ego|| (m)', fontsize=7)
        ax.set_ylabel('||world_pt_i − ego|| (m)', fontsize=7)
        ax.set_title('Transform Fidelity: corrected-world vs world', fontsize=9)
    else:
        ax.set_xlabel('||raw_pt_i|| — dist from ego, local (m)', fontsize=7)
        ax.set_ylabel('||world_pt_i − ego|| — dist from ego, world (m)', fontsize=7)
        ax.set_title('Transform Fidelity: local dist = world dist?', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)


def _sum_ade_heatmap(ax, frames):
    """Per-frame × per-step ADE heatmap."""
    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    fi_arr   = np.array([f['frame_idx'] for f in frames])
    n_steps  = 6

    matrix = np.full((len(frames), n_steps), np.nan)
    for i, c in enumerate(can_list):
        bkdn = c.get('ade_breakdown_m')
        if bkdn:
            for j, v in enumerate(bkdn[:n_steps]):
                matrix[i, j] = v

    vmax = np.nanpercentile(matrix[np.isfinite(matrix)], 95) if np.isfinite(matrix).any() else 1
    im = ax.imshow(matrix.T, aspect='auto', cmap='RdYlGn_r',
                   extent=[fi_arr[0] - 0.5, fi_arr[-1] + 0.5,
                           -0.5, n_steps - 0.5],
                   origin='lower', interpolation='nearest',
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='ADE (m)', shrink=0.75)
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels([f't{i} (+{0.5*(i+1):.1f}s)' for i in range(n_steps)], fontsize=6)
    ax.set_xlabel('Frame index', fontsize=7)
    ax.set_ylabel('Canonical step', fontsize=7)
    ax.set_title('Per-Step ADE Heatmap\n(grey = not scored)', fontsize=9)


# ── Frame selection ────────────────────────────────────────────────────────────

def _select_frames(frames: List[Dict], stride: Optional[int],
                   max_frames: int) -> List[Dict]:
    if stride is not None:
        return frames[::stride][:max_frames]

    # Default: all scored frames ± 1 neighbour, plus first/last 3
    fm = {f['frame_idx']: f for f in frames}
    scored_ids = {f['frame_idx'] for f in frames
                  if ( f.get('canonical_scoring_metrics') or {} ).get('scored')}
    wanted = set()
    for fi in scored_ids:
        wanted.update([fi - 1, fi, fi + 1])
    # First/last 3 frames for context
    sorted_ids = sorted(fm.keys())
    wanted |= set(sorted_ids[:3]) | set(sorted_ids[-3:])
    # Valid only
    selected = sorted(wanted & set(fm.keys()))
    return [fm[fi] for fi in selected][:max_frames]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Comprehensive planner data inspector visualization')
    p.add_argument('--debug-frames', required=True,
                   help='Path to debug_frames JSONL file')
    p.add_argument('--planner', required=True,
                   help='Planner name (any label; unknown families use generic panels)')
    p.add_argument('--out-dir', required=True,
                   help='Output directory for PNG figures')
    p.add_argument('--scenario-dir', default=None,
                   help='Optional path to scenario YAML/image directory '
                        '(for camera images and actor bounding boxes)')
    p.add_argument('--frame-idx', type=int, action='append', default=None,
                   help='Exact frame index to render (repeatable). '
                        'When set, takes precedence over stride/all-frames.')
    p.add_argument('--frame-stride', type=int, default=None,
                   help='Sample every N-th frame for per-frame pages '
                        '(default: scored ±1 + first/last 3)')
    p.add_argument('--max-frames', type=int, default=40,
                   help='Maximum number of per-frame figures (default 40)')
    p.add_argument('--all-frames', action='store_true',
                   help='Generate a page for every frame (ignores --max-frames cap)')
    return p.parse_args()


def main():
    args  = _parse_args()
    out   = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading debug frames: {args.debug_frames}')
    frames = load_frames(args.debug_frames)
    print(f'  Loaded {len(frames)} frames for '
          f'scenario {frames[0].get("scenario","?")}')

    can_list = [f.get('canonical_scoring_metrics') or {} for f in frames]
    n_scored = sum(1 for c in can_list if c.get('scored'))
    n_stale  = sum(1 for f in frames if ( f.get('metrics') or {} ).get('stale'))
    n_insuff = sum(1 for c in can_list
                   if c.get('inclusion_reason') == 'insufficient_horizon')
    print(f'  scored={n_scored}  stale={n_stale}  insufficient={n_insuff}')

    # ── Summary figure ────────────────────────────────────────────────────────
    print('Generating summary figure ...')
    make_summary_figure(frames, args.planner, str(out / 'summary.png'))

    # ── Per-frame figures ─────────────────────────────────────────────────────
    if args.frame_idx:
        wanted = set(int(v) for v in args.frame_idx)
        selected = [f for f in frames if int(f.get('frame_idx', -1)) in wanted]
        selected = sorted(selected, key=lambda x: x['frame_idx'])
        found = {int(f['frame_idx']) for f in selected}
        missing = sorted(wanted - found)
        if missing:
            print(f'  [warn] requested frame(s) not found: {missing}')
    elif args.all_frames:
        selected = frames
    else:
        selected = _select_frames(frames, args.frame_stride, args.max_frames)
    print(f'Generating {len(selected)} per-frame figures ...')

    for idx, frame in enumerate(selected):
        fi       = frame['frame_idx']
        out_path = str(out / f'frame_{fi:04d}.png')
        print(f'  [{idx+1:3d}/{len(selected)}] frame {fi:4d} ... ', end='', flush=True)
        fig = None
        try:
            fig = _make_perframe_fig(frame, frames, args.planner, args.scenario_dir)
            fig.savefig(out_path, dpi=110, bbox_inches='tight')
            print('OK')
        except Exception:
            import traceback
            print('ERROR')
            traceback.print_exc()
        finally:
            if fig is not None:
                plt.close(fig)

    all_out = sorted(out.glob('*.png'))
    print(f'\nDone.  {len(all_out)} figures written to {out}/')
    for p in all_out:
        sz = p.stat().st_size // 1024
        print(f'  {p.name:<30s}  {sz:5d} KB')


if __name__ == '__main__':
    main()
