"""
Single-frame detection probe on a v2xpnp scenario in CARLA.

Loads `actors_manifest.json` from `scenarioset/v2xpnp/<scene>/`, spawns every
actor at its frame-0 pose in CARLA's `ucla_v2` town, attaches our LiDAR to
ego_0, captures one sweep, runs the detector, and compares predictions against
the spawned actor positions as GT.

Reuses CarlaSceneSession + build_actor_spawn_list from the openloop prototype.

Usage (CARLA must be running on --port):
    /data/miniconda3/envs/colmdrivermarco2/bin/python tools/detection_probe_v2xpnp.py \\
        --scenario-dir scenarioset/v2xpnp/2023-03-17-16-03-02_11_0 \\
        --detector fcooper --port 6000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# CARLA egg.
_CARLA_EGG = "/data2/marco/CoLMDriver/carla912/PythonAPI/carla/dist/carla-0.9.12-py3.7-linux-x86_64.egg"
if os.path.isfile(_CARLA_EGG) and _CARLA_EGG not in sys.path:
    sys.path.append(_CARLA_EGG)

import torch
import carla
from tools.perception_runner import PerceptionRunner

# Reuse the prototype's spawn machinery.
from openloop.tools.carla_openloop_prototype import (
    CarlaSceneSession,
    build_actor_spawn_list,
)

# IoU-based AP utilities (the same OPV2V/HEAL papers report against).
from opencood.utils.eval_utils import caluclate_tp_fp, calculate_ap


def _carla_actor_to_corners_ego(actor, ego_tf, T_ego_world):
    """8-corner BB of a CARLA actor in ego LiDAR frame, OpenCOOD-friendly order.

    OpenCOOD's `convert_format` builds a 2D BEV polygon from corners[:, :4, :2]
    and expects those four points to trace a clean (CCW) quadrilateral.
    CARLA's `get_world_vertices()` doesn't follow that convention — its first 4
    vertices are corners of one x-face (mixing bottom + top z), not the
    bottom face. So we:

      1. Project all 8 CARLA corners into ego frame.
      2. Pick the 4 with the lowest z → that's the bottom face.
      3. Sort them by polar angle around their centroid → CCW.
      4. Match top-face corners (highest z) to the same CCW order.
    """
    world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    corners = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
    if corners.shape != (8, 3):
        return None
    ones = np.ones((8, 1), dtype=np.float64)
    hom_w = np.concatenate([corners, ones], axis=1)
    hom_e = (T_ego_world @ hom_w.T).T
    pts = hom_e[:, :3]                                          # (8, 3)

    # Identify bottom (4 lowest z) vs top.
    order_z = np.argsort(pts[:, 2])
    bot_idx = order_z[:4]
    top_idx = order_z[4:]
    bot = pts[bot_idx]                                          # (4, 3)
    top = pts[top_idx]                                          # (4, 3)

    # Sort bottom 4 CCW around their centroid (xy plane).
    bot_c = bot[:, :2].mean(axis=0)
    angles_bot = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
    bot_order = np.argsort(angles_bot)
    bot = bot[bot_order]

    # Match each top corner to the closest bottom corner (xy distance) so the
    # 4→8 pairing is consistent.
    matched_top = np.zeros_like(top)
    used = np.zeros(4, dtype=bool)
    for i in range(4):
        d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        matched_top[i] = top[j]

    out = np.concatenate([bot, matched_top], axis=0).astype(np.float32)
    return out


def _ego_world_inverse_transform(ego_tf):
    """Build the 4x4 transform world → ego LiDAR (sensor) frame."""
    yaw_rad = np.radians(ego_tf.rotation.yaw)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[cy, -sy, 0],
                  [sy,  cy, 0],
                  [0,   0,  1]], dtype=np.float64)
    t = np.array([ego_tf.location.x, ego_tf.location.y, ego_tf.location.z], dtype=np.float64)
    T_world_ego = np.eye(4)
    T_world_ego[:3, :3] = R
    T_world_ego[:3, 3]  = t
    return np.linalg.inv(T_world_ego)


def compute_iou_metrics(pred_corners_ego, pred_scores,
                        gt_corners_ego, lidar_range,
                        iou_thresholds=(0.3, 0.5, 0.7)):
    """Run the published OPV2V/HEAL metric on a single frame's pred + GT.

    Returns a dict per-IoU with TP/FP/GT counts plus precision and recall.
    AP from a single frame is informative but noisy; we report it anyway.
    """
    # Range filter: pred and GT both filtered identically (centroid in lidar range).
    def _in_range(corners):
        if len(corners) == 0:
            return np.zeros((0,), dtype=bool)
        c = corners.mean(axis=1)
        return ((c[:, 0] >= lidar_range[0]) & (c[:, 0] <= lidar_range[3]) &
                (c[:, 1] >= lidar_range[1]) & (c[:, 1] <= lidar_range[4]))

    keep_p = _in_range(pred_corners_ego)
    pred_corners_ego = pred_corners_ego[keep_p]
    pred_scores      = pred_scores[keep_p]
    keep_g = _in_range(gt_corners_ego)
    gt_corners_ego   = gt_corners_ego[keep_g]

    # caluclate_tp_fp expects torch tensors and uses .is_cuda.
    pc = torch.from_numpy(pred_corners_ego) if len(pred_corners_ego) else None
    ps = torch.from_numpy(pred_scores)      if len(pred_scores)      else None
    gc = torch.from_numpy(gt_corners_ego)

    result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []} for iou in iou_thresholds}
    for iou in iou_thresholds:
        caluclate_tp_fp(pc, ps, gc, result_stat, iou_thresh=iou)

    out = {}
    for iou in iou_thresholds:
        n_tp = int(np.sum(result_stat[iou]["tp"]))
        n_fp = int(np.sum(result_stat[iou]["fp"]))
        n_gt = int(result_stat[iou]["gt"])
        ap_val = 0.0
        if n_gt > 0 and (n_tp + n_fp) > 0:
            try:
                ap_val, _, _ = calculate_ap(result_stat, iou)
                ap_val = float(ap_val)
            except Exception:
                ap_val = 0.0
        precision = n_tp / max(1, n_tp + n_fp)
        recall    = n_tp / max(1, n_gt)
        out[iou] = {"tp": n_tp, "fp": n_fp, "gt": n_gt,
                    "precision": precision, "recall": recall, "ap": ap_val}
    return out


def _attach_lidar(world, ego, channels=64, lr=100, pps=2304000):
    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    for k, v in {
        "channels": str(channels), "range": str(lr),
        "rotation_frequency": "20", "points_per_second": str(pps),
        "upper_fov": "4", "lower_fov": "-20",
    }.items():
        bp.set_attribute(k, v)
    tf = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=2.5),
        carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
    )
    return world.spawn_actor(bp, tf, attach_to=ego)


class _Latest:
    pcd = None


def _project_actor_to_ego(actor, ego_tf, ego_yaw_rad):
    """Centroid of actor in ego LiDAR frame (xy)."""
    a_loc = actor.get_transform().location
    dx = a_loc.x - ego_tf.location.x
    dy = a_loc.y - ego_tf.location.y
    cos_y = np.cos(-ego_yaw_rad)
    sin_y = np.sin(-ego_yaw_rad)
    return cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy


def _actor_dim(actor):
    """xy dim (length × width) of actor's bounding box."""
    bb = actor.bounding_box
    # extent is half-dims; full dims are 2*extent.
    return 2.0 * bb.extent.x, 2.0 * bb.extent.y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario-dir", required=True, type=Path,
                    help="Path to scenarioset/v2xpnp/<scene> directory.")
    ap.add_argument("--detector", default="fcooper",
                    choices=["fcooper", "attfuse", "disco", "cobevt"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--score-thresh", type=float, default=0.20)
    ap.add_argument("--t-target-s", type=float, default=0.0,
                    help="Frame timestamp (seconds) to spawn the actors at.")
    ap.add_argument("--lidar-range", type=float, default=100.0)
    ap.add_argument("--top-k", type=int, default=20,
                    help="How many top-score predictions to print.")
    ap.add_argument("--cooperative", action="store_true",
                    help="Use ego_0 + ego_1 LiDARs in cooperative fusion mode "
                         "(if the scene has 2 egos). Default: single-agent.")
    args = ap.parse_args()

    if not args.scenario_dir.is_dir():
        raise SystemExit(f"scenario dir not found: {args.scenario_dir}")
    if not (args.scenario_dir / "actors_manifest.json").exists():
        raise SystemExit(f"no actors_manifest.json in {args.scenario_dir}")

    print(f"=== detection_probe_v2xpnp ===")
    print(f"  scenario : {args.scenario_dir.name}")
    print(f"  detector : {args.detector}")
    print(f"  CARLA    : {args.host}:{args.port}")
    print(f"  t_target : {args.t_target_s} s")

    specs = build_actor_spawn_list(args.scenario_dir, args.t_target_s)
    n_ego = sum(1 for s in specs if s.role == "ego")
    n_npc = sum(1 for s in specs if s.role == "npc")
    n_walker = sum(1 for s in specs if s.role == "walker")
    n_static = sum(1 for s in specs if s.role == "static")
    print(f"  manifest : {n_ego} ego, {n_npc} npc, {n_walker} walker, {n_static} static")

    box_a, box_b = {"pcd": None}, {"pcd": None}
    with CarlaSceneSession(host=args.host, port=args.port) as session:
        ok, skip = session.spawn_scene(specs)
        print(f"  spawned  : {ok} OK, {skip} skipped")

        # Find all egos in role_name order (ego_0, ego_1, ...).
        egos = [a for a in session.world.get_actors()
                if a.attributes.get("role_name", "").startswith("ego_")]
        egos.sort(key=lambda a: a.attributes.get("role_name", ""))
        if not egos:
            raise SystemExit("no ego actors spawned; check manifest")
        ego = egos[0]
        ego_tf = ego.get_transform()
        ego_yaw_rad = np.radians(ego_tf.rotation.yaw)
        print(f"  primary  : {ego.attributes.get('role_name')} at "
              f"({ego_tf.location.x:.2f},{ego_tf.location.y:.2f}) yaw={ego_tf.rotation.yaw:.1f}°")

        # Attach LiDAR to primary ego.
        lidar_a = _attach_lidar(session.world, ego, lr=args.lidar_range)
        session._spawned.append(lidar_a)
        lidar_a.listen(lambda m: box_a.__setitem__("pcd",
            np.frombuffer(bytes(m.raw_data), dtype=np.float32).reshape(-1, 4).copy()))

        # If cooperative requested AND there's a 2nd ego, attach a LiDAR to it too.
        ego_b = None
        lidar_b = None
        if args.cooperative:
            if len(egos) < 2:
                print(f"  [WARN] --cooperative requested but only {len(egos)} ego(s) "
                      f"in scene — falling back to single-agent.")
            else:
                ego_b = egos[1]
                tf_b = ego_b.get_transform()
                print(f"  cooperator: {ego_b.attributes.get('role_name')} at "
                      f"({tf_b.location.x:.2f},{tf_b.location.y:.2f}) yaw={tf_b.rotation.yaw:.1f}°")
                lidar_b = _attach_lidar(session.world, ego_b, lr=args.lidar_range)
                session._spawned.append(lidar_b)
                lidar_b.listen(lambda m: box_b.__setitem__("pcd",
                    np.frombuffer(bytes(m.raw_data), dtype=np.float32).reshape(-1, 4).copy()))

        # Tick to let the lidar(s) complete a sweep.
        for _ in range(5):
            session.world.tick()
        need_b = lidar_b is not None
        t0 = time.time()
        while time.time() - t0 < 5.0:
            if box_a["pcd"] is not None and (not need_b or box_b["pcd"] is not None):
                break
            session.world.tick()
        if box_a["pcd"] is None or (need_b and box_b["pcd"] is None):
            raise SystemExit("no LiDAR data after tick budget")
        pcd = box_a["pcd"]
        print(f"\n  LiDAR ego_0 : {pcd.shape}  intensity∈[{pcd[:,3].min():.2f},{pcd[:,3].max():.2f}]")
        if need_b:
            pcd_b = box_b["pcd"]
            print(f"  LiDAR ego_1 : {pcd_b.shape}  intensity∈[{pcd_b[:,3].min():.2f},{pcd_b[:,3].max():.2f}]")

        # Run detector — single or cooperative.
        runner = PerceptionRunner(detector=args.detector,
                                  score_threshold=args.score_thresh)
        if need_b:
            tf_a = ego.get_transform()
            tf_b = ego_b.get_transform()
            poses = [
                [tf_a.location.x, tf_a.location.y, tf_a.location.z,
                 tf_a.rotation.roll, tf_a.rotation.yaw, tf_a.rotation.pitch],
                [tf_b.location.x, tf_b.location.y, tf_b.location.z,
                 tf_b.rotation.roll, tf_b.rotation.yaw, tf_b.rotation.pitch],
            ]
            det = runner.detect_cooperative(
                lidars=[pcd, box_b["pcd"]],
                agent_poses_world=poses,
                primary_idx=0,
                comm_range_m=70.0,
                max_cooperators=4,
            )
            mode_label = "cooperative (2 egos)"
        else:
            det = runner.detect(pcd)
            mode_label = "single-agent"
        print(f"  preds    : P={det.P}  mode={mode_label}"
              + (f"  score=[{det.scores.min():.3f},{det.scores.max():.3f}]"
                 if det.P else "  (none)"))

        # Build GT in ego LiDAR frame from spawned vehicle actors (excluding ego).
        ego_id = ego.id
        all_vehicles = list(session.world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_vehicles if a.id != ego_id]
        gt_xy = []
        for a in gt_actors:
            x, y = _project_actor_to_ego(a, ego_tf, ego_yaw_rad)
            length, width = _actor_dim(a)
            gt_xy.append({"id": a.id, "type": a.type_id,
                          "role": a.attributes.get("role_name", ""),
                          "xy": (x, y), "dim": (length, width),
                          "dist": float(np.hypot(x, y))})
        gt_xy.sort(key=lambda g: g["dist"])

        # Per-pred: nearest-GT distance & IoU-style flag.
        if det.P > 0:
            print(f"\n  TOP-{min(args.top_k, det.P)} predictions (score-sorted) → nearest GT:")
            order = np.argsort(-det.scores)[:args.top_k]
            for rank, i in enumerate(order):
                px, py = det.centers_xy[i]
                if gt_xy:
                    ds = [(g["dist"] if False else
                           float(np.hypot(g["xy"][0] - px, g["xy"][1] - py)),
                           g) for g in gt_xy]
                    d_min, g_min = min(ds, key=lambda t: t[0])
                    note = (f"id={g_min['id']:5d} role={g_min['role']:12s} "
                            f"xy_gt=({g_min['xy'][0]:+6.1f},{g_min['xy'][1]:+6.1f}) "
                            f"Δ={d_min:5.2f} m")
                else:
                    note = "(no GT)"
                print(f"    [{rank:2d}] xy=({px:+6.1f},{py:+6.1f}) "
                      f"score={det.scores[i]:.3f}   {note}")

        # Per-GT: closest pred + score, sorted by distance from ego.
        print(f"\n  Per-GT match (closest pred), sorted by distance from ego:")
        print(f"  {'role':<14} {'gt_xy':>14} {'dist':>5}  {'pred_xy':>14} {'Δ':>5} {'score':>6}")
        if det.P > 0:
            for g in gt_xy[:25]:                        # limit to nearest 25 GT
                ds = np.hypot(det.centers_xy[:, 0] - g["xy"][0],
                              det.centers_xy[:, 1] - g["xy"][1])
                j = int(np.argmin(ds))
                px, py = det.centers_xy[j]
                sc = float(det.scores[j])
                print(f"  {g['role']:<14} ({g['xy'][0]:+5.1f},{g['xy'][1]:+5.1f})  "
                      f"{g['dist']:>5.1f}m ({px:+5.1f},{py:+5.1f}) {ds[j]:>5.2f} {sc:>5.2f}")
        else:
            for g in gt_xy[:25]:
                print(f"  {g['role']:<14} ({g['xy'][0]:+5.1f},{g['xy'][1]:+5.1f})  "
                      f"{g['dist']:>5.1f}m   (no preds)")

        # Quick TP@2m count.
        if det.P > 0 and gt_xy:
            tp_2m = 0
            preds_xy = det.centers_xy
            for g in gt_xy:
                ds = np.hypot(preds_xy[:, 0] - g["xy"][0],
                              preds_xy[:, 1] - g["xy"][1])
                if ds.min() < 2.0:
                    tp_2m += 1
            print(f"\n  GT recovered within 2 m by ANY pred: {tp_2m} / {len(gt_xy)}")

        # ── Proper BEV IoU-based metrics (matches the published OPV2V/HEAL eval) ──
        # Build GT corner boxes from CARLA actors.
        T_ego_world = _ego_world_inverse_transform(ego_tf)
        # Two GT pools: full (vehicles + statics), real-vehicles only (NPCs+ego_1).
        all_vehicles_acts = list(session.world.get_actors().filter("vehicle.*"))
        all_acts = [a for a in all_vehicles_acts if a.id != ego_id]
        real_acts = [a for a in all_acts
                     if a.attributes.get("role_name", "").startswith("npc_") or
                        a.attributes.get("role_name", "").startswith("ego_")]
        def _build_gt_corners(actors):
            arr = []
            for a in actors:
                c = _carla_actor_to_corners_ego(a, ego_tf, T_ego_world)
                if c is not None:
                    arr.append(c)
            return np.stack(arr, 0) if arr else np.zeros((0, 8, 3), dtype=np.float32)

        # The detector's pred_corners_ego has the corner convention OpenCOOD trains in.
        pred_corners = det.corners.astype(np.float32) if det.P else np.zeros((0, 8, 3), dtype=np.float32)
        pred_scores  = det.scores.astype(np.float32)  if det.P else np.zeros((0,), dtype=np.float32)
        # Drop ego self-detections (predictions whose center is within 2 m of origin).
        if len(pred_corners) > 0:
            c = pred_corners.mean(axis=1)
            keep = (c[:, 0] ** 2 + c[:, 1] ** 2) > 4.0
            pred_corners = pred_corners[keep]
            pred_scores  = pred_scores[keep]

        gt_all  = _build_gt_corners(all_acts)
        gt_real = _build_gt_corners(real_acts)

        for label, gt_corners in [
            ("ALL vehicle.* (vehicles + statics)", gt_all),
            ("REAL movers only (NPCs + ego_1)",     gt_real),
        ]:
            print(f"\n  ── BEV IoU metrics — {label} ──")
            metrics = compute_iou_metrics(
                pred_corners_ego=pred_corners,
                pred_scores=pred_scores,
                gt_corners_ego=gt_corners,
                lidar_range=[-100, -100, -3, 100, 100, 1],
            )
            print(f"    {'IoU':>5} {'TP':>4} {'FP':>4} {'GT':>4}  {'precision':>9} {'recall':>7} {'AP':>6}")
            for iou, m in metrics.items():
                print(f"    {iou:.2f}  {m['tp']:>4} {m['fp']:>4} {m['gt']:>4}  "
                      f"{m['precision']:>9.3f} {m['recall']:>7.3f} {m['ap']:>6.3f}")


if __name__ == "__main__":
    main()
