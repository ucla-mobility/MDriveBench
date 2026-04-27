"""
PerceptionSwapAgent — leaderboard agent that pairs a swappable HEAL detector
with a fixed rule-based planner. Multi-ego cooperative mode supported.

Purpose: measure planner sensitivity to perception quality. Same scenario,
same planner — only the detector backbone changes.

Configuration YAML (path_to_conf_file):

    detector: fcooper             # one of: fcooper, attfuse, disco
    cooperative: true             # true → fuse all egos in comm range
                                  # false → degenerate single-agent forward pass
    comm_range_m: 70              # default from HEAL configs (V2V comm distance)
    max_cooperators: 4            # cap fused agents (besides primary)
    cruise_speed: 6.0             # m/s
    score_threshold: 0.20         # detection confidence cutoff
    detection_radius: 2.0         # m, planner treats each det as a circle of this radius

For closed-loop CARLA via run_custom_eval.py: the harness applies the returned
controls directly. Per-ego perception runs once per tick (one cooperative
forward pass per ego, with itself as primary).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.perception_runner import PerceptionRunner
from team_code.rule_planner import RulePlanner, EgoState, Detection
from team_code.basic_agent_planner import BasicAgentPlanner


# ─── Env-var knobs for the perception-ablation experiment ───────────────────
# PERCEPTION_PLANNER:
#   "rule"        → original RulePlanner (default; preserves prior behaviour).
#   "basicagent"  → CARLA BasicAgent with hazard check rewired to our Detection
#                   list. This is the methodologically clean planner for the
#                   "perception is the only independent variable" study.
# PERCEPTION_USE_GT:
#   "1" / "true" → bypass the perception model and feed Detections derived from
#                  CARLA's world.get_actors() (range-gated to LiDAR sensor).
#                  This is the GT-oracle control row of the matrix.
def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "").lower() in ("1", "true", "yes", "on")


def get_entry_point():
    return "PerceptionSwapAgent"


class PerceptionSwapAgent(AutonomousAgent):
    """Per-ego perception+planner. v1: single ego only."""

    # ─────── Lifecycle ────────────────────────────────────────────────

    def setup(self, path_to_conf_file: str, ego_vehicles_num: int = 1):
        self.track = Track.SENSORS
        self.ego_vehicles_num = int(ego_vehicles_num)
        self.agent_name = "perception_swap"

        cfg = _load_conf(path_to_conf_file)
        self._detector_name    = str(cfg.get("detector", "fcooper"))
        # Tag used to prefix the agent's /tmp/* output files (diag JSONL + AP
        # summary). Defaults to detector_name so existing flows keep their
        # filenames; override with `log_tag: <unique>` in the YAML when two
        # configs would otherwise collide (e.g. perception_swap_gt and
        # perception_swap_fcooper both load detector=fcooper for the model
        # weights but should write to distinct logs in a parallel matrix).
        self._log_tag          = str(cfg.get("log_tag", self._detector_name))
        self._cooperative      = bool(cfg.get("cooperative", True))
        self._comm_range_m     = float(cfg.get("comm_range_m", 70.0))
        self._max_cooperators  = int(cfg.get("max_cooperators", 4))
        self._cruise_speed     = float(cfg.get("cruise_speed", 6.0))
        # Persistence filter: an in-corridor detection must hold for N
        # consecutive ticks before brake_now commits. Filters single-frame
        # perception FPs that previously locked the ego in dense walker
        # scenes. 1 disables the filter (legacy); 3 (~150 ms @ 20 Hz) is the
        # default and works for both RulePlanner and BasicAgentPlanner.
        self._brake_persistence_ticks = int(cfg.get("brake_persistence_ticks", 3))
        self._score_threshold  = float(cfg.get("score_threshold", 0.20))
        self._detection_radius = float(cfg.get("detection_radius", 2.0))
        self._collect_ap       = bool(cfg.get("collect_ap_metrics", True))
        # When disable_brake is True, we hand the planner an empty detection list
        # so it never triggers brake_now. Useful for AP-only runs where we want
        # the ego to keep moving through the scenario regardless of perception
        # noise. Independent of AP collection (which always sees true preds).
        self._disable_brake    = bool(cfg.get("disable_brake", False))

        # Planner selection + GT-perception toggle (see module-level docstring
        # for the experimental rationale).
        self._planner_kind     = os.environ.get(
            "PERCEPTION_PLANNER", str(cfg.get("planner", "rule"))
        ).lower().strip()
        if self._planner_kind not in ("rule", "basicagent"):
            print(f"[perception_swap] unknown PERCEPTION_PLANNER={self._planner_kind!r}, falling back to 'rule'")
            self._planner_kind = "rule"
        self._use_gt_perception = _env_flag(
            "PERCEPTION_USE_GT", bool(cfg.get("use_gt_perception", False))
        )
        # Stored routes-per-ego, used to defer BasicAgent construction until
        # the hero actors have been spawned (the harness calls set_global_plan
        # before the first run_step, but actors only exist on/after run_step).
        self._stored_world_plans: List[Optional[list]] = [None] * self.ego_vehicles_num
        # Target speed for BasicAgent (km/h). Reuses cruise_speed (m/s) so the
        # config knob name stays consistent with the RulePlanner path.
        self._target_speed_kmh = float(cfg.get("target_speed_kmh", self._cruise_speed * 3.6))
        print(
            f"[perception_swap] planner={self._planner_kind} "
            f"use_gt_perception={self._use_gt_perception} "
            f"target_speed_kmh={self._target_speed_kmh:.1f} "
            f"disable_brake={self._disable_brake}"
        )
        if self._disable_brake:
            print(
                "[perception_swap] WARNING: disable_brake=True hides ALL detections "
                "from the planner. Closed-loop perception ablation is meaningless in "
                "this mode (every detector run, including GT, behaves identically — "
                "open-loop cruise). Set disable_brake=false in the YAML for the "
                "ablation matrix; keep it true ONLY for pure AP-collection runs."
            )

        # Single perception model loaded once. Reused across all egos and frames.
        self.perception = PerceptionRunner(
            detector=self._detector_name,
            score_threshold=self._score_threshold,
        )

        # One planner per ego — each ego has its own route + state.
        # Type is RulePlanner | BasicAgentPlanner depending on self._planner_kind.
        self.planners: List[Optional[Any]] = [None] * self.ego_vehicles_num

        # One AP collector per ego (we measure detection quality per-ego since
        # each ego is the primary of its own cooperative forward pass).
        self.ap_collectors: List[Optional[CarlaGTApCollector]] = (
            [CarlaGTApCollector() for _ in range(self.ego_vehicles_num)]
            if self._collect_ap else [None] * self.ego_vehicles_num
        )

        # Cached LiDAR range from the model config — used by the GT projector.
        self._lidar_range = list(self.perception.config["heter"]["modality_setting"]["m1"]["preprocess"]["cav_lidar_range"])

        # Will be filled with the CARLA actor IDs of the egos (so the GT
        # projector can exclude them). Populated lazily on first run_step.
        self._ego_actor_ids: Optional[set] = None

        # Where to write the AP summary JSON. Resolved lazily based on env var.
        self._ap_out_path: Optional[str] = None

        # Diagnostics — last_step_info is keyed by ego index. Also write a JSONL
        # log of per-frame info so we can debug post-run.
        self.last_step_info: Dict[int, Dict[str, Any]] = {}
        diag_path = os.environ.get("PERCEPTION_SWAP_DIAG_LOG",
                                   f"/tmp/perception_swap_diag_{self._log_tag}.jsonl")
        try:
            os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
            self._diag_fp = open(diag_path, "w")
            print(f"[perception_swap] per-frame diag → {diag_path}")
        except Exception:
            self._diag_fp = None
        self._tick_count = 0

        # BEV viz: save a top-down PNG every N ticks showing route, predicted
        # 3s trajectory, predicted detections, and GT actors. Resolution order:
        #   1. PERCEPTION_SWAP_VIZ_DIR env (explicit override)
        #   2. SAVE_PATH env (run_custom_eval sets this to results/<tag>/image/<scene>)
        #      — matches the convention every other agent in this repo uses
        #   3. disabled
        viz_dir_override = os.environ.get("PERCEPTION_SWAP_VIZ_DIR", "")
        save_path_root  = os.environ.get("SAVE_PATH", "")
        if viz_dir_override:
            self._viz_dir = viz_dir_override
        elif save_path_root:
            self._viz_dir = os.path.join(save_path_root, "perception_swap_viz")
        else:
            self._viz_dir = ""
        self._viz_every = int(os.environ.get("PERCEPTION_SWAP_VIZ_EVERY", "10"))
        if self._viz_dir:
            try:
                os.makedirs(self._viz_dir, exist_ok=True)
                print(f"[perception_swap] BEV viz → {self._viz_dir} (every {self._viz_every} ticks)")
            except Exception as exc:
                print(f"[perception_swap] viz dir setup failed: {exc}")
                self._viz_dir = ""

        # Per-tick prediction-file output (JSON + NPZ) compatible with
        # carla_openloop_prototype's compute_metrics_from_run_dir reader.
        # Off by default — closed-loop baseline runs don't need it. Enable via
        # PERCEPTION_SWAP_WRITE_PRED_FILES=1; the openloop prototype's
        # run_planner_evaluation sets it automatically.
        self._write_pred_files = os.environ.get(
            "PERCEPTION_SWAP_WRITE_PRED_FILES", ""
        ).lower() in ("1", "true", "yes", "on")
        # Stride between prediction-file dumps. Codriving emits at 0.2s real
        # cadence (every 4 control ticks @ 20 Hz); we mirror that to give the
        # metric reader the same temporal density.
        self._pred_file_stride = int(os.environ.get("PERCEPTION_SWAP_PRED_FILE_STRIDE", "4"))
        if self._write_pred_files and save_path_root:
            self._pred_dirs = [
                os.path.join(save_path_root, f"ego_vehicle_{i}")
                for i in range(self.ego_vehicles_num)
            ]
            for d in self._pred_dirs:
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception as exc:
                    print(f"[perception_swap] pred-dir setup failed for {d}: {exc}")
            print(
                f"[perception_swap] per-tick prediction files → "
                f"<SAVE_PATH>/ego_vehicle_<i>/<step>.json + <step>_pred.npz "
                f"(stride={self._pred_file_stride})"
            )
        else:
            self._pred_dirs = []

    def destroy(self):
        # Defensive snapshot at the START of destroy() — if anything in the
        # cleanup chain below hard-exits the process (e.g. scenario_runner's
        # remove_all_actors hitting its 30 s budget on dense scenes), at least
        # we have the latest AP state on disk. The periodic per-tick snapshot
        # also covers this, but does so at 10-tick granularity; this catches
        # the final 1-9 ticks that wouldn't otherwise be flushed.
        try:
            self._snapshot_ap_summary()
        except Exception:
            pass

        # Close diag log.
        try:
            if getattr(self, "_diag_fp", None) is not None:
                self._diag_fp.close()
        except Exception:
            pass

        # Save AP summary, if collected.
        if self._collect_ap and any(c is not None for c in self.ap_collectors):
            out_path = self._ap_out_path or os.environ.get(
                "PERCEPTION_SWAP_AP_OUT",
                f"/tmp/perception_swap_ap_{self._log_tag}.json",
            )
            for ego_i, coll in enumerate(self.ap_collectors):
                if coll is None:
                    continue
                p = out_path.replace(".json", f"_ego{ego_i}.json")
                try:
                    coll.save(p, extra_meta={
                        "detector":     self._detector_name,
                        "cooperative":  self._cooperative,
                        "comm_range_m": self._comm_range_m,
                        "ego_index":    ego_i,
                    })
                    print(f"[perception_swap] saved AP summary → {p}")
                except Exception as exc:
                    print(f"[perception_swap] AP save failed for ego {ego_i}: {exc}")

        # Free CUDA memory if torch sets up persistent caches.
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _snapshot_ap_summary(self):
        """Write the current AP summary to disk so a kill mid-run doesn't lose data."""
        if not self._collect_ap:
            return
        out_path = self._ap_out_path or os.environ.get(
            "PERCEPTION_SWAP_AP_OUT",
            f"/tmp/perception_swap_ap_{self._log_tag}.json",
        )
        for ego_i, coll in enumerate(self.ap_collectors):
            if coll is None:
                continue
            p = out_path.replace(".json", f"_ego{ego_i}.json")
            try:
                coll.save(p, extra_meta={
                    "detector":     self._detector_name,
                    "cooperative":  self._cooperative,
                    "comm_range_m": self._comm_range_m,
                    "ego_index":    ego_i,
                    "snapshot_tick": self._tick_count,
                    "is_snapshot":   True,
                })
            except Exception:
                pass

    def _save_per_step_pred_files(
        self, primary_idx, ego, det, traj, ego_pose_world,
    ):
        """Per-tick JSON + NPZ in the format carla_openloop_prototype.compute_metrics_from_run_dir
        consumes (codriving-compatible).

        JSON fields: lidar_pose_x/y, theta (≈ yaw + π/2), waypoints (10 ego-local
        waypoints at 0.2s cadence), speed.
        NPZ fields:  pred_box_world (N, 8, 3) world-frame BB corners,
                     pred_score (N,), lidar_pose_x/y, theta, ego_x, ego_y.

        Coord conventions copied from carla_openloop_prototype.local_waypoints_to_world:
            carla-body forward = -local_y
            carla-body left    = -local_x
        i.e. inverse:
            forward = cos(yaw)·(wx-lx) + sin(yaw)·(wy-ly)
            left    = -sin(yaw)·(wx-lx) + cos(yaw)·(wy-ly)
            local_y = -forward,  local_x = -left
        """
        if not self._write_pred_files or not self._pred_dirs:
            return
        if self._tick_count % self._pred_file_stride != 0:
            return
        try:
            from tools.perception_runner import Detections as _Det  # noqa: F401  (type cue)
            _, _, _, _, x_to_world = _lazy_eval_imports()

            # Ego world position + yaw.
            lx = float(ego_pose_world[0])
            ly = float(ego_pose_world[1])
            yaw_deg = float(ego_pose_world[4])
            yaw_rad = float(np.radians(yaw_deg))
            theta = yaw_rad + (np.pi / 2.0)   # codriving's compass convention

            # Sub-sample our 30-pt @ 0.1s trajectory to 10-pt @ 0.2s to match the
            # metric reader's pred_horizon_steps=10 / pred_dt_s=0.2 default.
            traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
            stride = max(1, int(round(0.2 / 0.1)))
            sub = traj[stride - 1::stride][:10]   # 10 waypoints @ 0.2s
            if len(sub) < 10:
                # Pad with last waypoint if trajectory is shorter than 2 s.
                pad = np.tile(sub[-1:] if len(sub) else np.zeros((1, 2)), (10 - len(sub), 1))
                sub = np.concatenate([sub, pad], axis=0)

            # World → ego-local in codriving convention.
            dx = sub[:, 0] - lx
            dy = sub[:, 1] - ly
            c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
            forward = c * dx + s * dy
            left    = -s * dx + c * dy
            local_y = -forward
            local_x = -left
            local_wps = np.column_stack([local_x, local_y]).tolist()

            # Predicted boxes in ego frame → world frame.
            T_world_ego = x_to_world(ego_pose_world)
            pred_corners_ego = np.asarray(det.corners, dtype=np.float64)   # (N, 8, 3)
            if pred_corners_ego.size:
                hom = np.concatenate(
                    [pred_corners_ego.reshape(-1, 3),
                     np.ones((pred_corners_ego.shape[0] * 8, 1))],
                    axis=1,
                )
                pred_world = (T_world_ego @ hom.T).T[:, :3].reshape(-1, 8, 3).astype(np.float32)
            else:
                pred_world = np.zeros((0, 8, 3), dtype=np.float32)
            pred_score = np.asarray(det.scores, dtype=np.float32).reshape(-1)

            # Output files.
            out_dir = self._pred_dirs[primary_idx]
            stem = f"{self._tick_count:04d}"
            json_path = os.path.join(out_dir, f"{stem}.json")
            npz_path  = os.path.join(out_dir, f"{stem}_pred.npz")

            payload = {
                "speed":          float(ego.speed),
                "waypoints":      local_wps,
                "lidar_pose_x":   lx,
                "lidar_pose_y":   ly,
                "theta":          float(theta),
                "yaw_rad":        yaw_rad,    # explicit, in addition to compass theta
            }
            with open(json_path, "w") as f:
                json.dump(payload, f)
            np.savez(
                npz_path,
                pred_box_world=pred_world,
                pred_score=pred_score,
                lidar_pose_x=np.float64(lx),
                lidar_pose_y=np.float64(ly),
                theta=np.float64(theta),
                ego_x=np.float64(lx),
                ego_y=np.float64(ly),
            )
        except Exception as exc:
            self._pred_file_err_count = getattr(self, "_pred_file_err_count", 0) + 1
            if self._pred_file_err_count <= 3:
                print(f"[perception_swap] pred-file save failed (ego {primary_idx}): {exc}")

    def _save_camera_frame(self, primary_idx, cam_img):
        """Write the raw front-camera RGB to a PNG, one per save-tick."""
        if not self._viz_dir:
            return
        if self._tick_count % self._viz_every != 0:
            return
        if cam_img is None or not hasattr(cam_img, "shape") or cam_img.size == 0:
            return
        try:
            # CARLA gives us BGRA uint8. Drop alpha + reorder to RGB for PIL.
            if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                rgb = cam_img[..., [2, 1, 0]]
            else:
                rgb = cam_img
            from PIL import Image
            out = os.path.join(
                self._viz_dir,
                f"tick_{self._tick_count:05d}_ego{primary_idx}_cam.png",
            )
            Image.fromarray(rgb.astype(np.uint8)).save(out)
        except Exception as exc:
            self._cam_err_count = getattr(self, "_cam_err_count", 0) + 1
            if self._cam_err_count <= 3:
                print(f"[perception_swap] camera save failed: {exc}")

    def _save_bev_frame(self, primary_idx, ego_state, det, traj, cam_img=None):
        """Side-by-side: front camera (left) + top-down BEV (right).

        BEV shows ego + route + 3-s plan + GT actors + predictions.
        Camera shows the live driver-view RGB if a camera sensor is declared.
        """
        if not self._viz_dir:
            return
        if self._tick_count % self._viz_every != 0:
            return
        try:
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except Exception:
            return
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            if world is None:
                return

            ex, ey = float(ego_state.xy[0]), float(ego_state.xy[1])
            yaw = float(ego_state.yaw)
            cy, sy = np.cos(yaw), np.sin(yaw)

            # Predictions: ego frame → world (rotation + translation by ego pose).
            pred_world = np.zeros((det.P, 2), dtype=np.float64) if det.P else None
            if det.P:
                for i, (px, py) in enumerate(det.centers_xy):
                    pred_world[i, 0] = ex + cy * px - sy * py
                    pred_world[i, 1] = ey + sy * px + cy * py

            # GT vehicles (excluding any hero/ego).
            ego_ids = self._ego_actor_ids or set()
            gt_xy = []
            for a in world.get_actors().filter("vehicle.*"):
                if a.id in ego_ids:
                    continue
                loc = a.get_location()
                gt_xy.append((loc.x, loc.y, a.attributes.get("role_name", "")))
            # Other heroes (cooperators) — show separately so we can see them.
            other_heroes = []
            for a in self._get_hero_actors():
                if a.id == self._get_hero_actors()[primary_idx].id:
                    continue
                loc = a.get_location()
                other_heroes.append((loc.x, loc.y))

            route_xy = self.last_step_info.get(primary_idx, {}).get("route_xy")

            # Two-panel figure: camera (left) + BEV (right).
            fig, (ax_cam, ax) = plt.subplots(1, 2, figsize=(16, 8),
                                              gridspec_kw={"width_ratios": [1, 1]})
            if cam_img is not None and hasattr(cam_img, "shape") and cam_img.size > 0:
                # CARLA delivers BGRA uint8; convert to RGB for matplotlib.
                if cam_img.ndim == 3 and cam_img.shape[-1] == 4:
                    cam_rgb = cam_img[..., [2, 1, 0]]
                else:
                    cam_rgb = cam_img
                ax_cam.imshow(cam_rgb)
            ax_cam.set_title(f"front cam · ego_{primary_idx}")
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])
            if route_xy is not None and len(route_xy) > 1:
                ax.plot(route_xy[:, 0], route_xy[:, 1], color="gray",
                        linestyle="--", linewidth=1, alpha=0.6, label="route")
            if traj is not None and len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], color="orange",
                        linewidth=2.4, label=f"3 s plan (n={len(traj)})")
                ax.plot(traj[-1, 0], traj[-1, 1], "o", color="orange",
                        markersize=6, alpha=0.7)
            if gt_xy:
                gx = [g[0] for g in gt_xy]
                gy = [g[1] for g in gt_xy]
                ax.scatter(gx, gy, s=42, c="green", marker="s",
                           edgecolors="black", linewidths=0.5, label=f"GT ({len(gt_xy)})")
            if pred_world is not None and len(pred_world) > 0:
                ax.scatter(pred_world[:, 0], pred_world[:, 1], s=80, c="red",
                           marker="+", linewidths=2.0, label=f"pred ({len(pred_world)})")
            for (ox, oy) in other_heroes:
                ax.plot(ox, oy, "^", color="purple", markersize=14, alpha=0.7,
                        label="cooperator" if other_heroes.index((ox, oy)) == 0 else None)

            # Draw ego as a small triangle pointing in ego.yaw direction.
            arrow_len = 4.0
            ax.plot(ex, ey, "^", color="blue", markersize=14)
            ax.arrow(ex, ey, arrow_len * cy, arrow_len * sy,
                     head_width=1.5, head_length=1.0, color="blue",
                     length_includes_head=True)

            # Window centred on ego, ±60 m.
            R = 60.0
            ax.set_xlim(ex - R, ex + R)
            ax.set_ylim(ey - R, ey + R)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            info = self.last_step_info.get(primary_idx, {})
            ax.set_title(
                f"tick {self._tick_count:04d} ego_{primary_idx}  "
                f"spd={ego_state.speed:.1f} m/s  "
                f"P={det.P}  GT={len(gt_xy)}  "
                f"brake={'Y' if info.get('brake_now') else 'N'}",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=8)

            out_path = os.path.join(
                self._viz_dir,
                f"tick_{self._tick_count:05d}_ego{primary_idx}.png",
            )
            fig.savefig(out_path, dpi=80, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            # Don't let viz crash the drive.
            self._viz_err_count = getattr(self, "_viz_err_count", 0) + 1
            if self._viz_err_count <= 3:
                print(f"[perception_swap] viz frame failed: {exc}")

    def _resolve_ego_actor_ids(self) -> set:
        """Find the CARLA actor IDs for the hero (ego) vehicles. Used to
        exclude them from GT during AP collection."""
        actors = self._get_hero_actors()
        return {a.id for a in actors}

    def _get_hero_actors(self) -> List[Any]:
        """Hero vehicles in CARLA, ordered by their numeric role_name suffix.

        Leaderboard convention: ego vehicles have role_name 'hero', 'hero_1',
        'hero_2', ... corresponding to ego slots 0, 1, 2, ...
        """
        try:
            from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
            world = CarlaDataProvider.get_world()
            if world is None:
                return []
            heroes = [a for a in world.get_actors().filter("vehicle.*")
                      if a.attributes.get("role_name", "").startswith("hero")]
            def _slot(a):
                rn = a.attributes.get("role_name", "hero")
                if rn == "hero":
                    return 0
                if rn.startswith("hero_"):
                    try:
                        return int(rn.split("_", 1)[1])
                    except Exception:
                        return 0
                return 0
            heroes.sort(key=_slot)
            return heroes
        except Exception:
            return []

    def _gt_perception_pass(self, primary_pose_world, ego_actor_ids):
        """Replace one perception forward pass with a CARLA-ground-truth one.

        Returns a ``Detections``-shaped result (same fields as PerceptionRunner
        outputs) so all downstream code (Detection list, AP collector, BEV viz,
        diagnostics) keeps working unchanged.

        Range-gated to ``self._lidar_range`` so the oracle isn't unfairly
        omniscient — perception models only see things within sensor range,
        and GT must respect the same horizon for the comparison to be valid.
        """
        # Imports gated to avoid eager carla_data_provider load.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from tools.perception_runner import Detections as _DetResult
        _, _, _, _, x_to_world = _lazy_eval_imports()

        world = CarlaDataProvider.get_world()
        empty = _DetResult(
            corners=np.zeros((0, 8, 3), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            centers_xy=np.zeros((0, 2), dtype=np.float32),
            P=0,
        )
        if world is None:
            return empty

        all_v = list(world.get_actors().filter("vehicle.*"))
        gt_actors = [a for a in all_v if a.id not in (ego_actor_ids or set())]
        if not gt_actors:
            return empty

        T_ego_world = np.linalg.inv(x_to_world(primary_pose_world))
        lr = self._lidar_range  # [xmin, ymin, zmin, xmax, ymax, zmax]

        corners_list: List[np.ndarray] = []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            cw = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if cw.shape != (8, 3):
                continue
            hom = np.concatenate([cw, np.ones((8, 1))], axis=1)
            ce = (T_ego_world @ hom.T).T[:, :3]                        # (8, 3) ego frame
            # Reorder to clean CCW bottom face (same fix as the AP collector).
            order_z = np.argsort(ce[:, 2])
            bot, top = ce[order_z[:4]], ce[order_z[4:]]
            bc = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bc[1], bot[:, 0] - bc[0])
            bot = bot[np.argsort(ang)]
            mt = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                mt[i] = top[j]
            cor = np.concatenate([bot, mt], axis=0)                    # (8, 3)

            c = cor.mean(axis=0)
            if not (lr[0] <= c[0] <= lr[3] and lr[1] <= c[1] <= lr[4]):
                continue
            corners_list.append(cor)

        if not corners_list:
            return empty
        corners = np.stack(corners_list, axis=0).astype(np.float32)
        centers_xy = corners.mean(axis=1)[:, :2].astype(np.float32)
        scores = np.ones((len(corners),), dtype=np.float32)
        return _DetResult(corners=corners, scores=scores,
                          centers_xy=centers_xy, P=len(corners))

    # ─────── Sensor declaration ───────────────────────────────────────
    # The harness reads this and feeds matching keys via input_data.
    # IDs are suffixed _<slot_id> by the harness; for ego 0 they become
    # `lidar_0`, `gps_0`, `imu_0`, `speed_0`.

    def sensors(self):
        # LiDAR config — yaw=0 so the sensor frame equals the ego frame and
        # the detector's predictions land directly in ego coords. Camera is
        # declared so we can save real driver-view images alongside the BEV
        # plot for visualization (only used by viz, not by the model).
        return [
            {"type": "sensor.lidar.ray_cast", "id": "lidar",
             "x": 0.0, "y": 0.0, "z": 2.5,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "range": 100,
             "rotation_frequency": 20,
             "channels": 64,
             "upper_fov": 4,
             "lower_fov": -20,
             "points_per_second": 2304000},
            {"type": "sensor.camera.rgb", "id": "rgb_front",
             "x": 1.3, "y": 0.0, "z": 1.5,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 800, "height": 600, "fov": 100},
            {"type": "sensor.other.gnss",     "id": "gps",
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"type": "sensor.other.imu",      "id": "imu",
             "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            {"type": "sensor.speedometer",    "id": "speed",
             "reading_frequency": 20},
        ]

    # ─────── Global plan ──────────────────────────────────────────────
    # The harness calls this once before the first run_step. world_plan is a
    # list-of-lists (one list per ego) of (carla.Transform, RoadOption) pairs.

    def set_global_plan(self, gps_plan, world_plan):
        if not world_plan:
            raise RuntimeError("PerceptionSwapAgent: empty world_plan")
        if len(world_plan) != self.ego_vehicles_num:
            # Common pattern in this repo: world_plan is one route per ego.
            # If the harness gave us fewer routes than egos, broadcast the first.
            # Most cleanly: error out so we notice.
            raise RuntimeError(
                f"PerceptionSwapAgent: got {len(world_plan)} routes for "
                f"{self.ego_vehicles_num} egos"
            )

        for ego_i, ego_route in enumerate(world_plan):
            if not ego_route or len(ego_route) < 2:
                raise RuntimeError(
                    f"PerceptionSwapAgent: ego {ego_i} route too short "
                    f"({len(ego_route)} waypoints)"
                )
            route_xy = np.array([
                [float(transform.location.x), float(transform.location.y)]
                for transform, _cmd in ego_route
            ], dtype=np.float64)
            self.last_step_info.setdefault(ego_i, {})["route_xy"] = route_xy

            if self._planner_kind == "rule":
                self.planners[ego_i] = RulePlanner(
                    global_route=route_xy,
                    cruise_speed=self._cruise_speed,
                    brake_persistence_ticks=self._brake_persistence_ticks,
                )
            else:
                # BasicAgent needs the hero actor in its constructor; the
                # actor isn't spawned yet on the first set_global_plan call,
                # so we stash the route and build the planner lazily on the
                # first run_step that sees the actor.
                self._stored_world_plans[ego_i] = ego_route

    # ─────── Per-frame step ───────────────────────────────────────────

    def run_step(self, input_data: Dict[str, Any], timestamp: float):
        # Bail out cleanly if set_global_plan never landed. For the rule
        # planner we built it eagerly there; for basicagent we stashed the
        # route and will build the planner lazily once heroes are spawned.
        if self._planner_kind == "rule":
            if any(p is None for p in self.planners):
                return [_idle_control() for _ in range(self.ego_vehicles_num)]
        else:
            if any(p is None for p in self._stored_world_plans):
                return [_idle_control() for _ in range(self.ego_vehicles_num)]

        # 1. Read each ego's pose from CARLA directly (more reliable than
        # converting GNSS lat/lon → world meters), and read LiDAR from sensor.
        hero_actors = self._get_hero_actors()
        if len(hero_actors) < self.ego_vehicles_num:
            # Fall back to idle if we can't see the heroes yet — happens on
            # the very first tick before CARLA has registered them.
            return [_idle_control() for _ in range(self.ego_vehicles_num)]

        ego_states: List[EgoState] = []
        ego_lidars: List[np.ndarray] = []
        ego_poses_world: List[List[float]] = []

        for slot in range(self.ego_vehicles_num):
            actor = hero_actors[slot]
            tf = actor.get_transform()
            vel = actor.get_velocity()
            xy = np.array([tf.location.x, tf.location.y], dtype=np.float64)
            # CARLA Transform.rotation.yaw is in DEGREES (LH frame: +y is the
            # vehicle's RIGHT, so positive yaw rotates +x toward +y, which is
            # clockwise viewed from above). The numeric value (cos yaw, sin yaw)
            # gives the forward direction in CARLA world coords either way.
            yaw_deg = float(tf.rotation.yaw)
            yaw_rad = float(np.radians(yaw_deg))
            yaw_rad = float(((yaw_rad + np.pi) % (2.0 * np.pi)) - np.pi)
            speed = float(np.hypot(vel.x, vel.y))

            lidar = _read_sensor(input_data, "lidar", slot,
                                 default=np.zeros((0, 4), dtype=np.float32))

            # One-shot sensor sanity print so we can verify the sensor config
            # produced the expected point count + intensity scale.
            if not getattr(self, f"_sensor_logged_{slot}", False) and len(lidar) > 0:
                print(f"[perception_swap] sensor sanity (ego {slot}, tick {self._tick_count}): "
                      f"lidar shape={lidar.shape}  "
                      f"x∈[{lidar[:,0].min():.1f},{lidar[:,0].max():.1f}] "
                      f"y∈[{lidar[:,1].min():.1f},{lidar[:,1].max():.1f}] "
                      f"z∈[{lidar[:,2].min():.1f},{lidar[:,2].max():.1f}] "
                      f"intensity∈[{lidar[:,3].min():.3f},{lidar[:,3].max():.3f}]")
                setattr(self, f"_sensor_logged_{slot}", True)

            ego_states.append(EgoState(xy=xy, yaw=yaw_rad, speed=speed))
            ego_lidars.append(np.asarray(lidar, dtype=np.float32))
            ego_poses_world.append([
                float(tf.location.x), float(tf.location.y), float(tf.location.z),
                float(tf.rotation.roll), yaw_deg, float(tf.rotation.pitch),
            ])

        # Lazy-build the BasicAgentPlanner now that heroes exist (one-shot).
        if self._planner_kind == "basicagent":
            for slot in range(self.ego_vehicles_num):
                if self.planners[slot] is None:
                    plan_for_slot = self._stored_world_plans[slot]
                    self.planners[slot] = BasicAgentPlanner(
                        actor=hero_actors[slot],
                        target_speed_kmh=self._target_speed_kmh,
                        brake_persistence_ticks=self._brake_persistence_ticks,
                    )
                    self.planners[slot].set_global_plan(plan_for_slot)
                    print(
                        f"[perception_swap] BasicAgentPlanner built for ego {slot} "
                        f"(target_speed={self._target_speed_kmh:.1f} km/h, "
                        f"route_len={len(plan_for_slot)})"
                    )

        # GT-perception oracle path: skip the perception model entirely and
        # build Detections directly from world.get_actors(). One forward-pass-
        # equivalent per ego (so per-ego ego frames stay consistent).
        if self._use_gt_perception and self._ego_actor_ids is None:
            self._ego_actor_ids = self._resolve_ego_actor_ids()

        # 2. Run perception (or GT oracle) per ego, then plan.
        ctrls: List[carla.VehicleControl] = []
        for primary_idx in range(self.ego_vehicles_num):
            ego = ego_states[primary_idx]
            planner = self.planners[primary_idx]

            if self._use_gt_perception:
                det = self._gt_perception_pass(
                    ego_poses_world[primary_idx], self._ego_actor_ids
                )
            elif self._cooperative and self.ego_vehicles_num > 1:
                det = self.perception.detect_cooperative(
                    lidars=ego_lidars,
                    agent_poses_world=ego_poses_world,
                    primary_idx=primary_idx,
                    comm_range_m=self._comm_range_m,
                    max_cooperators=self._max_cooperators,
                )
            else:
                det = self.perception.detect(ego_lidars[primary_idx])

            # AP collection (CARLA GT). Skip in GT-oracle mode: pred == GT by
            # construction, so AP is trivially 1.0 and just costs cycles.
            # Lazy-init the ego ID set on first call so it picks up egos after
            # they're spawned.
            if (self._collect_ap and not self._use_gt_perception
                    and self.ap_collectors[primary_idx] is not None):
                if self._ego_actor_ids is None:
                    self._ego_actor_ids = self._resolve_ego_actor_ids()
                try:
                    self.ap_collectors[primary_idx].update(
                        primary_pose_world=ego_poses_world[primary_idx],
                        pred_corners_ego=det.corners,
                        pred_scores=det.scores,
                        lidar_range=self._lidar_range,
                        ego_actor_ids=self._ego_actor_ids,
                        debug_dump_path=os.environ.get("PERCEPTION_SWAP_DEBUG_DUMP",
                            f"/tmp/perception_swap_smoke_run/debug_dump_ego{primary_idx}.json"),
                    )
                except Exception as exc:
                    # Don't let AP-collection failures kill the drive. Rate-limit
                    # the printout so we don't drown the log.
                    self._ap_err_count = getattr(self, "_ap_err_count", 0) + 1
                    if self._ap_err_count <= 3 or self._ap_err_count % 100 == 0:
                        print(f"[perception_swap] AP update fail #{self._ap_err_count} "
                              f"(ego {primary_idx}): {exc}")

                # Snapshot AP summary every 10 ticks (~0.5 s @ 20 Hz / 1 s @ 10 Hz)
                # so the latest state survives even if the harness hard-exits before
                # destroy() completes — observed when scenario_runner's
                # remove_all_actors hits its 30 s budget on dense scenes (90+ walkers)
                # and bypasses the agent's destroy().
                if (self._tick_count + 1) % 10 == 0:
                    self._snapshot_ap_summary()

            det_list = [
                Detection(
                    xy_ego=det.centers_xy[i].astype(np.float64),
                    radius=self._detection_radius,
                    score=float(det.scores[i]),
                )
                for i in range(det.P)
            ]

            # Optionally hide all detections from the planner so it never brakes.
            # AP collector still sees the real predictions below.
            planner_dets = [] if self._disable_brake else det_list

            # Plan + control. Two backends; plan_extras carries the path-specific
            # diagnostic fields so last_step_info / diag JSONL stay consistent.
            if self._planner_kind == "rule":
                intent = planner.plan(ego, planner_dets, dt=0.1)
                steer, throttle, brake = planner.to_control(intent, ego)
                traj = planner.to_trajectory(ego, intent, horizon_s=3.0, dt=0.1)
                plan_extras = {
                    "intent_speed": float(intent.target_speed),
                    "brake_now":    bool(intent.brake_now),
                    "min_ttc":      float(intent.debug.get("min_ttc", float("inf"))),
                    "cursor":       int(planner.cursor),
                }
            else:
                planner.set_detections(planner_dets)
                ctrl_carla = planner.run_step()
                steer    = float(ctrl_carla.steer)
                throttle = float(ctrl_carla.throttle)
                brake    = float(ctrl_carla.brake)
                traj     = planner.predict_trajectory(horizon_s=3.0, dt=0.1)
                plan_extras = {
                    "intent_speed": planner.target_speed_mps,
                    # BasicAgent's add_emergency_stop writes brake=max_brake (0.5).
                    # >= catches that exactly.
                    "brake_now":    brake >= 0.5,
                    "min_ttc":      float("inf"),
                    "cursor":       -1,
                }
            self.last_planned_waypoints_world = traj  # (T, 2) world-frame xy

            # Optional per-tick prediction-file output for the openloop prototype's
            # metric reader (off unless PERCEPTION_SWAP_WRITE_PRED_FILES=1).
            self._save_per_step_pred_files(
                primary_idx, ego, det, traj, ego_poses_world[primary_idx]
            )

            # Optional viz dump: front camera (raw) + combined BEV+cam every N ticks.
            if self._viz_dir:
                cam_img = _read_sensor(input_data, "rgb_front", primary_idx,
                                       default=None)
                # One-shot print so we know whether the camera sensor is alive.
                if not getattr(self, f"_cam_logged_{primary_idx}", False):
                    if cam_img is None or not hasattr(cam_img, "shape"):
                        print(f"[perception_swap] WARN: rgb_front_{primary_idx} not delivered by harness")
                    else:
                        print(f"[perception_swap] rgb_front_{primary_idx} shape={cam_img.shape} dtype={cam_img.dtype}")
                    setattr(self, f"_cam_logged_{primary_idx}", True)
                self._save_camera_frame(primary_idx, cam_img)
                self._save_bev_frame(primary_idx, ego, det, traj, cam_img)

            self.last_step_info[primary_idx] = {
                "route_xy":      self.last_step_info.get(primary_idx, {}).get("route_xy"),
                "ego_xy":        ego.xy,
                "ego_yaw":       ego.yaw,
                "ego_speed":     ego.speed,
                "n_detections":  det.P,
                "intent_speed":  plan_extras["intent_speed"],
                "brake_now":     plan_extras["brake_now"],
                "min_ttc":       plan_extras["min_ttc"],
                "cursor":        plan_extras["cursor"],
                "cooperative":   self._cooperative and self.ego_vehicles_num > 1,
                "planner_kind":  self._planner_kind,
                "use_gt":        self._use_gt_perception,
                "traj_first":    traj[0].tolist(),
                "traj_last":     traj[-1].tolist(),
                "horizon_s":     3.0,
            }

            ctrl = carla.VehicleControl()
            ctrl.steer    = float(steer)
            ctrl.throttle = float(throttle)
            ctrl.brake    = float(brake)
            ctrl.hand_brake = False
            ctrl.manual_gear_shift = False
            ctrls.append(ctrl)

            # Per-frame diagnostic dump (one JSON line per ego per tick).
            if self._diag_fp is not None:
                try:
                    # Corridor stats only meaningful for the rule planner (it
                    # exposes look_ahead / corridor_half_width). For BasicAgent
                    # we use a reasonable proxy (BasicAgent.base_vehicle_threshold).
                    if self._planner_kind == "rule":
                        in_corridor_x_max = float(planner.look_ahead)
                        corridor_half_w   = float(planner.corridor_half_width)
                    else:
                        in_corridor_x_max = float(planner._agent._base_vehicle_threshold) + float(ego.speed)
                        corridor_half_w   = float(planner._corridor_half_width)
                    self._diag_fp.write(json.dumps({
                        "tick":         self._tick_count,
                        "ego":          int(primary_idx),
                        "planner_kind": self._planner_kind,
                        "use_gt":       self._use_gt_perception,
                        "ego_xy":       [float(ego.xy[0]), float(ego.xy[1])],
                        "ego_yaw":      float(ego.yaw),
                        "ego_speed":    float(ego.speed),
                        "n_dets":       int(det.P),
                        "n_dets_in_corridor": int(sum(
                            1 for d in det_list
                            if 0 <= d.xy_ego[0] <= in_corridor_x_max and
                               abs(d.xy_ego[1]) <= corridor_half_w + d.radius
                        )),
                        "intent_speed": float(plan_extras["intent_speed"]),
                        "brake_now":    bool(plan_extras["brake_now"]),
                        "min_ttc":      float(plan_extras["min_ttc"]),
                        "cursor":       int(plan_extras["cursor"]),
                        "ctrl":         {"steer": float(steer), "throttle": float(throttle), "brake": float(brake)},
                    }) + "\n")
                    self._diag_fp.flush()
                except Exception:
                    pass

        self._tick_count += 1
        return ctrls


# ──────────────────────────────────────────────────────────────────────────
# Helpers (module-level so they're easy to unit-test)
# ──────────────────────────────────────────────────────────────────────────


def _load_conf(path: str) -> Dict[str, Any]:
    """Read a tiny YAML conf file. Empty/missing path → defaults."""
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PerceptionSwapAgent: missing conf file {path!r}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_sensor(input_data: Dict[str, Any], sensor_id: str, slot: int, default):
    """Return the data tensor of the (frame_idx, payload) tuple for `<sensor_id>_<slot>`."""
    key = f"{sensor_id}_{slot}"
    entry = input_data.get(key)
    if entry is None:
        return default
    return np.asarray(entry[1])


def _read_speed(input_data: Dict[str, Any], slot: int) -> float:
    key = f"speed_{slot}"
    entry = input_data.get(key)
    if entry is None:
        return 0.0
    payload = entry[1]
    if isinstance(payload, dict):
        return float(payload.get("speed", 0.0))
    try:
        return float(payload)
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# CARLA-GT AP Collector
# ──────────────────────────────────────────────────────────────────────────
# In closed-loop CARLA we have perfect GT every tick: every spawned vehicle's
# pose and bounding box are queryable from CarlaDataProvider. We query each
# tick, project to the primary ego's LiDAR frame, and accumulate TP/FP using
# OpenCOOD's eval utilities. AP is computed at scenario end.

# Lazy-imported to avoid pulling torch/opencood when this file is loaded for
# argument parsing or doc generation.
def _lazy_eval_imports():
    import torch
    from opencood.utils.eval_utils import caluclate_tp_fp, calculate_ap
    from opencood.utils.box_utils import boxes_to_corners_3d
    from opencood.utils.transformation_utils import x_to_world
    return torch, caluclate_tp_fp, calculate_ap, boxes_to_corners_3d, x_to_world


class CarlaGTApCollector:
    """Per-detector AP accumulator using CARLA ground truth as oracle.

    Usage pattern inside the agent:
        collector = CarlaGTApCollector(iou_thresholds=(0.3, 0.5, 0.7))
        ...each tick, per primary ego...
            collector.update(
                primary_pose_world=[x, y, z, roll, yaw_deg, pitch],
                pred_corners_ego=det.corners,
                pred_scores=det.scores,
                lidar_range=runner.config[...]['cav_lidar_range'],
                ego_actor_ids={ego_id_0, ego_id_1, ...},  # to exclude egos from GT
            )
        ...on destroy()...
            collector.save(json_path)
    """
    def __init__(self, iou_thresholds=(0.1, 0.3, 0.5, 0.7),
                 self_filter_radius_m: float = 2.0):
        self.iou_thresholds = tuple(iou_thresholds)
        self.self_filter_radius_m = float(self_filter_radius_m)
        self.result_stat = {iou: {"tp": [], "fp": [], "gt": 0, "score": []}
                            for iou in self.iou_thresholds}
        self.n_frames = 0
        self.cum_n_pred = 0
        self.cum_n_pred_after_self_filter = 0
        self.cum_n_gt = 0
        self.cum_n_self_dets = 0

    def update(self,
               primary_pose_world,
               pred_corners_ego,
               pred_scores,
               lidar_range,
               ego_actor_ids,
               debug_dump_path: Optional[str] = None):
        """Append one frame's TP/FP stats. If `debug_dump_path` is given AND we
        haven't dumped yet AND this frame has both preds and GTs, dump them."""
        torch, caluclate_tp_fp, _, _, _ = _lazy_eval_imports()

        gt_corners_ego = self._build_gt_corners_ego_frame(
            primary_pose_world=primary_pose_world,
            lidar_range=lidar_range,
            ego_actor_ids=ego_actor_ids,
        )

        # ── One-shot debug dump: first frame with both preds AND GT ──
        if (debug_dump_path is not None
                and not getattr(self, "_debug_dumped", False)
                and len(pred_corners_ego) > 0 and len(gt_corners_ego) > 0):
            try:
                pred_centers = pred_corners_ego.mean(axis=1)
                gt_centers   = gt_corners_ego.mean(axis=1)
                # For EACH pred (sorted by score desc), the closest GT.
                order = np.argsort(-pred_scores)
                preds_dump = []
                for rank, i in enumerate(order):
                    d = np.linalg.norm(gt_centers[:, :2] - pred_centers[i, :2], axis=1)
                    j = int(np.argmin(d))
                    preds_dump.append({
                        "rank":            rank,
                        "pred_xy":         pred_centers[i, :2].tolist(),
                        "pred_score":      float(pred_scores[i]),
                        "pred_dim_xy":     [
                            float(pred_corners_ego[i, :, 0].max() - pred_corners_ego[i, :, 0].min()),
                            float(pred_corners_ego[i, :, 1].max() - pred_corners_ego[i, :, 1].min()),
                        ],
                        "nearest_gt_xy":   gt_centers[j, :2].tolist(),
                        "nearest_gt_dim":  [
                            float(gt_corners_ego[j, :, 0].max() - gt_corners_ego[j, :, 0].min()),
                            float(gt_corners_ego[j, :, 1].max() - gt_corners_ego[j, :, 1].min()),
                        ],
                        "dist_m":          float(d[j]),
                    })
                # All GT centers + dims for full picture.
                gt_dump = []
                for k in range(len(gt_centers)):
                    gt_dump.append({
                        "gt_xy":  gt_centers[k, :2].tolist(),
                        "gt_dim": [
                            float(gt_corners_ego[k, :, 0].max() - gt_corners_ego[k, :, 0].min()),
                            float(gt_corners_ego[k, :, 1].max() - gt_corners_ego[k, :, 1].min()),
                        ],
                    })
                with open(debug_dump_path, "w") as f:
                    json.dump({
                        "n_pred":     int(len(pred_corners_ego)),
                        "n_gt":       int(len(gt_corners_ego)),
                        "primary_pose_world": list(primary_pose_world),
                        "lidar_range": list(lidar_range),
                        "all_preds_sorted_by_score": preds_dump,
                        "all_gt": gt_dump,
                    }, f, indent=2)
                print(f"[perception_swap] one-shot debug dump → {debug_dump_path}")
                self._debug_dumped = True
            except Exception as exc:
                print(f"[perception_swap] debug dump failed: {exc}")

        # Filter pred to within lidar range to match what GT does, and drop
        # ego self-detections (predictions whose center is near origin —
        # the model often emits a high-confidence box for the ego itself
        # because the ego's LiDAR sees its own car body returns).
        n_self_filtered = 0
        if len(pred_corners_ego) > 0:
            centers = pred_corners_ego.mean(axis=1)
            in_range = (
                (centers[:, 0] >= lidar_range[0]) & (centers[:, 0] <= lidar_range[3]) &
                (centers[:, 1] >= lidar_range[1]) & (centers[:, 1] <= lidar_range[4])
            )
            r2 = self.self_filter_radius_m ** 2
            not_self = (centers[:, 0] ** 2 + centers[:, 1] ** 2) > r2
            n_self_filtered = int((~not_self).sum())
            keep = in_range & not_self
            pred_corners_ego = pred_corners_ego[keep]
            pred_scores      = pred_scores[keep]
        self.cum_n_self_dets += n_self_filtered
        self.cum_n_pred_after_self_filter += int(len(pred_corners_ego))

        pc = torch.from_numpy(pred_corners_ego) if len(pred_corners_ego) else None
        ps = torch.from_numpy(pred_scores)      if len(pred_scores)      else None
        gc = torch.from_numpy(gt_corners_ego)   # (M, 8, 3)

        for iou in self.iou_thresholds:
            caluclate_tp_fp(pc, ps, gc, self.result_stat, iou_thresh=iou)

        self.n_frames   += 1
        self.cum_n_pred += int(len(pred_corners_ego))
        self.cum_n_gt   += int(len(gt_corners_ego))

    def _build_gt_corners_ego_frame(self, primary_pose_world, lidar_range, ego_actor_ids):
        """Query the CARLA world for vehicles, project to primary ego LiDAR frame."""
        # Imports gated to avoid eager carla_data_provider load.
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        _, _, _, _, x_to_world = _lazy_eval_imports()

        world = CarlaDataProvider.get_world()
        if world is None:
            return np.zeros((0, 8, 3), dtype=np.float32)

        all_vehicles = list(world.get_actors().filter("vehicle.*"))
        # Exclude egos from GT (we don't want to "detect" ourselves).
        gt_actors = [a for a in all_vehicles if a.id not in ego_actor_ids]
        if not gt_actors:
            return np.zeros((0, 8, 3), dtype=np.float32)

        # Build world→ego_lidar transform.
        T_world_ego = x_to_world(primary_pose_world)        # ego→world
        T_ego_world = np.linalg.inv(T_world_ego)            # world→ego

        # CARLA's get_world_vertices() does NOT return a clean (-,-)→(+,-)→
        # (+,+)→(-,+) bottom face. The first 4 vertices alternate between
        # bottom and top of one x-face. The simple constant permutation we
        # tried earlier did not produce a valid CCW quadrilateral. Robust fix:
        #   1) project all 8 corners into ego frame,
        #   2) take the 4 with smallest z = bottom face,
        #   3) sort them CCW around their centroid in xy,
        #   4) match top corners to the same CCW order.
        # This guarantees corners[:4] is a clean polygon for shapely's IoU.

        all_corners_ego: List[np.ndarray] = []
        for actor in gt_actors:
            try:
                world_verts = actor.bounding_box.get_world_vertices(actor.get_transform())
            except Exception:
                continue
            corners_w = np.array([[v.x, v.y, v.z] for v in world_verts], dtype=np.float64)
            if corners_w.shape != (8, 3):
                continue

            ones = np.ones((8, 1), dtype=np.float64)
            hom_w = np.concatenate([corners_w, ones], axis=1)
            hom_e = (T_ego_world @ hom_w.T).T
            pts = hom_e[:, :3]                                           # (8, 3)

            order_z = np.argsort(pts[:, 2])
            bot = pts[order_z[:4]]
            top = pts[order_z[4:]]
            bot_c = bot[:, :2].mean(axis=0)
            ang = np.arctan2(bot[:, 1] - bot_c[1], bot[:, 0] - bot_c[0])
            bot = bot[np.argsort(ang)]
            matched_top = np.zeros_like(top)
            used = np.zeros(4, dtype=bool)
            for i in range(4):
                d = np.linalg.norm(top[:, :2] - bot[i, :2], axis=1)
                d[used] = np.inf
                j = int(np.argmin(d))
                used[j] = True
                matched_top[i] = top[j]
            corn_ego = np.concatenate([bot, matched_top], axis=0)

            # Filter to lidar range based on centroid.
            c = corn_ego.mean(axis=0)
            if not (lidar_range[0] <= c[0] <= lidar_range[3] and
                    lidar_range[1] <= c[1] <= lidar_range[4]):
                continue
            all_corners_ego.append(corn_ego.astype(np.float32))

        if not all_corners_ego:
            return np.zeros((0, 8, 3), dtype=np.float32)
        return np.stack(all_corners_ego, axis=0)

    def save(self, out_path: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Compute final AP and write JSON."""
        _, _, calculate_ap, _, _ = _lazy_eval_imports()
        ap_table: Dict[str, Any] = {}
        for iou in self.iou_thresholds:
            n_tp = int(np.sum(self.result_stat[iou]["tp"]))
            n_fp = int(np.sum(self.result_stat[iou]["fp"]))
            n_gt = int(self.result_stat[iou]["gt"])
            # calculate_ap divides by gt and by precision-curve area; both blow up on empty data.
            if n_gt == 0 or (n_tp + n_fp) == 0:
                ap_value = 0.0
            else:
                try:
                    ap_value, _, _ = calculate_ap(self.result_stat, iou)
                    ap_value = float(ap_value)
                except ZeroDivisionError:
                    ap_value = 0.0
            ap_table[f"ap_{int(iou*100):02d}"] = {
                "ap":   ap_value,
                "iou":  float(iou),
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_gt": n_gt,
            }
        summary = {
            "n_frames":   self.n_frames,
            "cum_n_pred": self.cum_n_pred,
            "cum_n_pred_after_self_filter": self.cum_n_pred_after_self_filter,
            "cum_n_self_dets": self.cum_n_self_dets,
            "cum_n_gt":   self.cum_n_gt,
            "ap":         ap_table,
        }
        if extra_meta:
            summary.update(extra_meta)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)


def _idle_control():
    c = carla.VehicleControl()
    c.steer = 0.0
    c.throttle = 0.0
    c.brake = 0.0
    c.hand_brake = False
    c.manual_gear_shift = False
    return c
