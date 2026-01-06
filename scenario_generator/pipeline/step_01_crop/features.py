import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import generate_legal_paths as glp

from .models import CropFeatures, CropKey, GeometrySpec


def _opposite_dir(d: str) -> str:
    return {"E": "W", "W": "E", "N": "S", "S": "N"}.get(d, d)


def _cluster_points(points: np.ndarray, eps: float = 12.0) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    parents = list(range(len(points)))

    def find(a: int) -> int:
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for i in range(len(points)):
        nbrs = tree.query_ball_point(points[i], r=eps)
        for j in nbrs:
            if j != i:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(points)):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def detect_junction_centers(segments: List[Any], adj: List[List[int]]) -> List[np.ndarray]:
    n = len(segments)
    indeg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in adj[i]:
            indeg[j] += 1

    pts = []
    for i, s in enumerate(segments):
        if len(adj[i]) >= 2 or indeg[i] >= 2:
            pts.append(s.points[-1])
        for j in adj[i]:
            if segments[j].road_id != s.road_id:
                pts.append(s.points[-1])
                break

    if not pts:
        return []
    pts = np.asarray(pts, dtype=float)
    clusters = _cluster_points(pts, eps=14.0)
    centers = [pts[idxs].mean(axis=0) for idxs in clusters if len(idxs) >= 3]
    return centers


def _crop_contains_point(c: CropKey, p: np.ndarray) -> bool:
    return (c.xmin <= float(p[0]) <= c.xmax) and (c.ymin <= float(p[1]) <= c.ymax)


def _point_xy(p: Any) -> Tuple[float, float]:
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"])
    return float(p[0]), float(p[1])


def _estimate_lane_count(segments_crop: List[Any]) -> int:
    by: Dict[Tuple[int, int], set] = {}
    for s in segments_crop:
        key = (int(s.road_id), int(s.section_id))
        by.setdefault(key, set()).add(int(s.lane_id))
    if not by:
        return 1
    return max(1, max(len(v) for v in by.values()))


def compute_crop_features(
    town_name: str,
    segments_full: List[Any],
    junction_centers: List[np.ndarray],
    center_xy: Tuple[float, float],
    crop: CropKey,
    min_path_len: float,
    max_paths: int,
    max_depth: int,
) -> Optional[CropFeatures]:
    cb = glp.CropBox(crop.xmin, crop.xmax, crop.ymin, crop.ymax)
    segs_crop = glp.crop_segments(segments_full, cb)
    if len(segs_crop) < 6:
        return None

    adj_crop = glp.build_connectivity(segs_crop)
    paths = glp.generate_legal_paths(
        segs_crop, adj_crop, cb,
        min_path_length=min_path_len,
        max_paths=max_paths,
        max_depth=max_depth,
        allow_within_region_fallback=False
    )
    if len(paths) < 6:
        return None

    sigs = [glp.build_path_signature(p) for p in paths]
    entry_dirs = sorted(set(s["entry"]["cardinal4"] for s in sigs))
    exit_dirs = sorted(set(s["exit"]["cardinal4"] for s in sigs))
    dirs = sorted(set(entry_dirs) | set(exit_dirs))
    turns = sorted(set(s["entry_to_exit_turn"] for s in sigs))

    straights = [s for s in sigs if s["entry_to_exit_turn"] == "straight"]
    entry_set = set(s["entry"]["cardinal4"] for s in straights)
    has_oncoming = any((_opposite_dir(d) in entry_set) for d in entry_set)

    is_t = (len(dirs) == 3)
    is_four = (len(dirs) >= 4)

    by_exit: Dict[Tuple[int, int], set] = {}
    for s in sigs:
        key = (int(s["exit"]["road_id"]), int(s["exit"]["section_id"]))
        by_exit.setdefault(key, set()).add(s["entry_to_exit_turn"])
    has_merge = any(("straight" in v and ("left" in v or "right" in v)) for v in by_exit.values())
    
    # On-ramp heuristic: improved detection
    # An on-ramp is a merge point where:
    # 1. Multiple distinct entry/exit directions exist (not just a 2-way road)
    # 2. There's merge geometry (multiple maneuvers to same exit)
    # 3. Has multi-lane capability for merging
    # 4. NOT a full four-way intersection (more than 2 distinct directions on entry OR exit)
    has_multi_distinct_entries = len(entry_set) >= 2
    has_multi_distinct_exits = len(set(s["exit"]["cardinal4"] for s in sigs)) >= 2
    has_merge_geometry = has_merge or len(by_exit) > 1  # Multiple exit roads is also merge-like
    has_on_ramp = (
        has_merge_geometry 
        and (has_multi_distinct_entries or has_multi_distinct_exits)
        and not is_four
    )

    lane_count = _estimate_lane_count(segs_crop)
    has_ml = lane_count >= 2

    jct_count = sum(1 for jc in junction_centers if _crop_contains_point(crop, jc))

    cx, cy = center_xy
    man_stats: Dict[str, Dict[str, float]] = {}
    for man in ["straight", "left", "right", "uturn"]:
        man_stats[man] = {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0}

    for s in sigs:
        man = s["entry_to_exit_turn"]
        ent = s["entry"]["point"]
        ex = s["exit"]["point"]
        ent_x, ent_y = _point_xy(ent)
        ex_x, ex_y = _point_xy(ex)
        ent_d = float(math.hypot(ent_x - cx, ent_y - cy))
        ex_d = float(math.hypot(ex_x - cx, ex_y - cy))
        st = man_stats.setdefault(man, {"count": 0.0, "max_entry_dist": 0.0, "max_exit_dist": 0.0})
        st["count"] += 1.0
        st["max_entry_dist"] = max(st["max_entry_dist"], ent_d)
        st["max_exit_dist"] = max(st["max_exit_dist"], ex_d)

    area = (crop.xmax - crop.xmin) * (crop.ymax - crop.ymin)
    return CropFeatures(
        town=town_name,
        crop=crop,
        center_xy=center_xy,
        turns=turns,
        entry_dirs=entry_dirs,
        exit_dirs=exit_dirs,
        dirs=dirs,
        has_oncoming_pair=has_oncoming,
        is_t_junction=is_t,
        is_four_way=is_four,
        has_merge_onto_same_road=has_merge,
        has_on_ramp=has_on_ramp,
        lane_count_est=lane_count,
        has_multi_lane=has_ml,
        maneuver_stats=man_stats,
        n_paths=len(paths),
        junction_count=jct_count,
        area=float(area),
        _segments_full=segments_full,
        _junction_centers=junction_centers,
    )
