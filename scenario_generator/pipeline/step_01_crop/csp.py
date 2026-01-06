import os
from typing import Any, Dict, List, Tuple

from .models import AssignmentResult, CropFeatures, GeometrySpec, Scenario
from .scoring import crop_base_cost, crop_satisfies_spec
from .viz import save_viz


def solve_assignment(
    scenarios: List[Scenario],
    specs: Dict[str, GeometrySpec],
    crops: List[CropFeatures],
    domain_k: int,
    capacity_per_crop: int,
    reuse_weight: float,
    junction_penalty: float,
    log_every: int = 0,
    viz_out_dir: str = "",
    viz_invert_x: bool = False,
    viz_dpi: int = 150,
    viz_max: int = 0,
) -> AssignmentResult:

    domain: Dict[str, List[Tuple[CropFeatures, float]]] = {}
    total = len(scenarios)
    for i, sc in enumerate(scenarios, start=1):
        spec = specs[sc.sid]
        feas = [c for c in crops if crop_satisfies_spec(spec, c)]
        scored = [(c, crop_base_cost(spec, c, junction_penalty=junction_penalty)) for c in feas]
        scored.sort(key=lambda x: x[1])
        domain[sc.sid] = scored[: max(1, domain_k)]
        if log_every and (i % log_every == 0 or i == total):
            print(f"[INFO] domain {i}/{total}")

    order = sorted(scenarios, key=lambda sc: (len(domain[sc.sid]), -specs[sc.sid].confidence))
    scenarios_by_id = {sc.sid: sc for sc in scenarios}

    assigned: Dict[str, CropFeatures] = {}
    used: set = set()
    load: Dict[Tuple[str, str], int] = {}

    def key(c: CropFeatures) -> Tuple[str, str]:
        return (c.town, c.crop.to_str())

    def incremental_cost(cand: CropFeatures, base: float) -> float:
        inc = base
        if key(cand) not in used:
            inc += reuse_weight
        if capacity_per_crop > 0 and load.get(key(cand), 0) >= capacity_per_crop:
            inc += 1e12
        return inc

    total_order = len(order)
    viz_enabled = bool(viz_out_dir)
    viz_pending: List[Tuple[str, str, CropFeatures]] = []
    viz_count = 0
    viz_rendered: Dict[str, Tuple[str, str]] = {}

    def flush_viz(force: bool = False) -> None:
        nonlocal viz_enabled, viz_count
        if not viz_enabled:
            viz_pending.clear()
            return
        if not viz_pending and not force:
            return
        for sid, text, feat in list(viz_pending):
            if viz_max and viz_count >= viz_max:
                viz_enabled = False
                break
            out_png = os.path.join(
                viz_out_dir, f"{sid}__{feat.town}__{feat.crop.to_str()}.png"
            )
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=text,
                    crop=feat.crop,
                    crop_feat=feat,
                    invert_x=viz_invert_x,
                    dpi=viz_dpi,
                )
                viz_count += 1
                viz_rendered[sid] = (feat.town, feat.crop.to_str())
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")
        viz_pending.clear()

    for i, sc in enumerate(order, start=1):
        sid = sc.sid
        options = domain[sid]
        if not options:
            continue
        best = None
        best_val = float("inf")
        for cand, base in options:
            val = incremental_cost(cand, base)
            if val < best_val:
                best_val = val
                best = (cand, base)
        if best is None or best_val >= 1e11:
            cand, base = options[0]
        else:
            cand, base = best
        assigned[sid] = cand
        used.add(key(cand))
        load[key(cand)] = load.get(key(cand), 0) + 1
        if viz_enabled:
            viz_pending.append((sid, sc.text, cand))
        if log_every and (i % log_every == 0 or i == total_order):
            print(f"[INFO] assigned {i}/{total_order}")
            flush_viz()

    def total_objective() -> float:
        base_sum = 0.0
        used_now = set()
        load_now: Dict[Tuple[str, str], int] = {}
        for sid, c in assigned.items():
            spec = specs[sid]
            base_sum += crop_base_cost(spec, c, junction_penalty=junction_penalty)
            used_now.add(key(c))
            load_now[key(c)] = load_now.get(key(c), 0) + 1
        cap_pen = 0.0
        if capacity_per_crop > 0:
            for _, v in load_now.items():
                if v > capacity_per_crop:
                    cap_pen += 1e9 * (v - capacity_per_crop)
        return base_sum + reuse_weight * len(used_now) + cap_pen

    for _ in range(2):
        improved = False
        cur_obj = total_objective()
        for sc in order:
            sid = sc.sid
            if sid not in assigned:
                continue
            cur_crop = assigned[sid]
            cur_key = key(cur_crop)
            for cand, _ in domain[sid]:
                if key(cand) == cur_key:
                    continue
                if capacity_per_crop > 0:
                    cur_load = sum(1 for _, c2 in assigned.items() if key(c2) == key(cand))
                    if cur_load >= capacity_per_crop:
                        continue
                assigned[sid] = cand
                new_obj = total_objective()
                if new_obj + 1e-6 < cur_obj:
                    improved = True
                    cur_obj = new_obj
                    cur_crop = cand
                    cur_key = key(cand)
                else:
                    assigned[sid] = cur_crop
        if not improved:
            break

    flush_viz(force=True)
    if viz_rendered:
        for sid, prev_key in list(viz_rendered.items()):
            feat = assigned.get(sid)
            if feat is None:
                continue
            new_key = (feat.town, feat.crop.to_str())
            if new_key == prev_key:
                continue
            sc = scenarios_by_id.get(sid)
            if sc is None:
                continue
            out_png = os.path.join(
                viz_out_dir, f"{sid}__{feat.town}__{feat.crop.to_str()}.png"
            )
            try:
                save_viz(
                    out_png=out_png,
                    scenario_id=sid,
                    scenario_text=sc.text,
                    crop=feat.crop,
                    crop_feat=feat,
                    invert_x=viz_invert_x,
                    dpi=viz_dpi,
                )
                viz_rendered[sid] = new_key
            except Exception as e:
                print(f"[WARN] viz failed for {sid}: {e}")

    mapping: Dict[str, Dict[str, List[str]]] = {}
    detailed: Dict[str, Any] = {"assignments": {}, "unassigned": []}

    for sc in scenarios:
        sid = sc.sid
        if sid not in assigned:
            detailed["unassigned"].append({"id": sid, "text": sc.text, "source": sc.source})
            continue
        c = assigned[sid]
        t = c.town
        ck = c.crop.to_str()
        mapping.setdefault(t, {}).setdefault(ck, []).append(sid)

        spec = specs[sid]
        detailed["assignments"][sid] = {
            "scenario": sc.text,
            "source": sc.source,
            "town": t,
            "crop": [c.crop.xmin, c.crop.xmax, c.crop.ymin, c.crop.ymax],
            "center_xy": list(c.center_xy),
            "geometry_spec": spec.__dict__,
            "crop_features": {
                "dirs": c.dirs,
                "turns": c.turns,
                "entry_dirs": c.entry_dirs,
                "exit_dirs": c.exit_dirs,
                "has_oncoming_pair": c.has_oncoming_pair,
                "is_t_junction": c.is_t_junction,
                "is_four_way": c.is_four_way,
                "has_merge_onto_same_road": c.has_merge_onto_same_road,
                "has_on_ramp": c.has_on_ramp,
                "lane_count_est": c.lane_count_est,
                "junction_count": c.junction_count,
                "maneuver_stats": c.maneuver_stats,
                "n_paths": c.n_paths,
                "area": c.area,
            },
        }

    for t in list(mapping.keys()):
        mapping[t] = dict(sorted(mapping[t].items(), key=lambda kv: kv[0]))
        for k0 in mapping[t]:
            mapping[t][k0] = sorted(mapping[t][k0])

    return AssignmentResult(mapping=mapping, detailed=detailed)
