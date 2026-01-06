import json
from typing import Any, Dict, List, Tuple

import numpy as np

from .geometry import wrap180


def load_nodes(nodes_path: str) -> Dict[str, Any]:
    with open(nodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{nodes_path} missing top-level 'payload'")
    return data


def _unit_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    r = np.radians(wrap180(yaw_deg))
    return np.array([np.cos(r), np.sin(r)], dtype=float)


def _orient_polyline(points_xy: np.ndarray, yaws_deg: np.ndarray, orig_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    yaws = np.asarray(yaws_deg, dtype=float)
    idxs = np.asarray(orig_idx, dtype=int)
    if len(pts) < 2:
        return pts, yaws, idxs

    vecs = pts[1:] - pts[:-1]
    norms = (np.linalg.norm(vecs, axis=1) + 1e-9)
    dir_vecs = vecs / norms[:, None]
    yaw_vecs = np.vstack([_unit_from_yaw_deg(y) for y in yaws[:-1]])
    dots = np.sum(dir_vecs * yaw_vecs, axis=1)
    if float(np.nanmean(dots)) < 0.0:
        return pts[::-1].copy(), yaws[::-1].copy(), idxs[::-1].copy()
    return pts, yaws, idxs


def _split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws: np.ndarray, gap_m: float = 6.0):
    if len(pts) < 2:
        return [(idxs_sorted, pts, yaws)] if len(pts) > 0 else []
    jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cuts = [0]
    for i, d in enumerate(jumps):
        if float(d) > gap_m:
            cuts.append(i + 1)
    cuts.append(len(pts))
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            out.append((idxs_sorted[a:b], pts[a:b], yaws[a:b]))
    return out


def build_segments_from_nodes(nodes: Dict[str, Any], min_points: int = 6) -> List[Dict[str, Any]]:
    payload = nodes["payload"]
    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    from collections import defaultdict
    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    segments: List[Dict[str, Any]] = []
    seg_id_counter = 0

    # IMPORTANT: iteration order determines seg_id assignment (insertion order of grouped).
    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]

        for idxs_chunk, pts_chunk, yaws_chunk in _split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = _orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue
            segments.append({
                "seg_id": int(seg_id_counter),
                "road_id": int(rid),
                "lane_id": int(lid),
                "section_id": int(sid),
                "points": pts_o.astype(float),
                "yaws": np.asarray([wrap180(v) for v in yaws_o], dtype=float),
                "orig_idx": idxs_o.astype(int),
            })
            seg_id_counter += 1

    return segments


def _override_seg_points_with_picked(
    picked_list: List[Dict[str, Any]], seg_by_id: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    """
    Replace/augment seg_by_id geometry with any polyline_samples present in picked_list.
    This lets refined start/end slices or synthetic segments propagate into placement/viz.
    """
    out = dict(seg_by_id)
    for p in picked_list:
        sig = p.get("signature", {}) if isinstance(p.get("signature", {}), dict) else {}
        segs = sig.get("segments_detailed", []) if isinstance(sig.get("segments_detailed", []), list) else []
        for s in segs:
            if not isinstance(s, dict):
                continue
            seg_id = s.get("seg_id", None)
            pl = s.get("polyline_sample") or []
            if seg_id is None or len(pl) < 2:
                continue
            try:
                sid = int(seg_id)
            except Exception:
                continue
            pts = np.array([[float(pt["x"]), float(pt["y"])] for pt in pl if "x" in pt and "y" in pt], dtype=float)
            if len(pts) >= 2:
                out[sid] = pts
    return out


__all__ = [
    "_override_seg_points_with_picked",
    "_orient_polyline",
    "_split_by_gaps",
    "_unit_from_yaw_deg",
    "build_segments_from_nodes",
    "load_nodes",
]
