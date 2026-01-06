import difflib
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import LATERAL_RELATIONS


def _wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def _heading_deg(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _infer_heading_change_deg(points: np.ndarray) -> float:
    """Approximate how much a segment turns (deg) from its start tangent to end tangent."""
    if points is None or len(points) < 3:
        return 0.0
    v0 = points[1] - points[0]
    v1 = points[-1] - points[-2]
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-6 or n1 < 1e-6:
        return 0.0
    h0 = _heading_deg(v0 / n0)
    h1 = _heading_deg(v1 / n1)
    return abs(_wrap180(h1 - h0))


def _segment_length_m(points: np.ndarray) -> float:
    if points is None or len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(points[1:] - points[:-1], axis=1)))


def _infer_vehicle_turn_exit_indices(
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
    turn_deg_threshold: float = 35.0,
) -> Dict[int, Dict[str, Any]]:
    """Infer which segments are 'turn' vs 'exit' per vehicle using geometry."""
    out: Dict[int, Dict[str, Any]] = {}
    for p in picked_list:
        veh_str = p.get("vehicle", "")
        m = re.search(r"(\d+)", veh_str)
        if not m:
            continue
        veh_num = int(m.group(1))
        seg_ids = [int(x) for x in p.get("signature", {}).get("segment_ids", [])]
        seg_lens: List[float] = []
        heading_changes: List[float] = []
        for sid in seg_ids:
            pts = seg_by_id.get(int(sid))
            seg_lens.append(_segment_length_m(pts))
            heading_changes.append(_infer_heading_change_deg(pts))

        turn_indices = {i + 1 for i, d in enumerate(heading_changes) if d >= turn_deg_threshold}

        # Fallback: if the path is non-straight but geometry didn't cross the threshold (rare),
        # assume segment 2 is the turn connector.
        entry_to_exit_turn = p.get("signature", {}).get("entry_to_exit_turn", "straight")
        if not turn_indices and entry_to_exit_turn in ("left", "right", "u_turn") and len(seg_ids) >= 2:
            turn_indices = {2}

        exit_index = None
        if turn_indices:
            last_turn = max(turn_indices)
            if last_turn < len(seg_ids):
                exit_index = last_turn + 1
            else:
                exit_index = last_turn
        elif len(seg_ids) >= 2:
            # For straight-through paths with no detected turns, the last segment is still the "exit"
            # (the road after the intersection, even though there's no turn)
            exit_index = len(seg_ids)

        out[veh_num] = {
            "turn_indices": turn_indices,
            "exit_index": exit_index,
            "seg_ids": seg_ids,
            "seg_lens": seg_lens,
        }
    return out


def _best_entity_match(actor_semantic: str, entities: List[Dict[str, Any]], target_vehicle: int) -> Optional[Dict[str, Any]]:
    """Match a Stage2 actor back to a Stage1 entity (for 'when' semantics)."""
    if not actor_semantic or not entities:
        return None
    a = actor_semantic.strip().lower()
    best = None
    best_score = 0.0
    for e in entities:
        if not isinstance(e, dict):
            continue
        ev = e.get("affects_vehicle", None)
        if ev is not None and ev != target_vehicle:
            continue
        mention = str(e.get("mention", "")).strip().lower()
        if not mention:
            continue
        a_tokens = set(re.findall(r"[a-z0-9]+", a))
        m_tokens = set(re.findall(r"[a-z0-9]+", mention))
        jacc = (len(a_tokens & m_tokens) / max(1, len(a_tokens | m_tokens)))
        seq = difflib.SequenceMatcher(None, a, mention).ratio()
        score = 0.6 * seq + 0.4 * jacc
        if score > best_score:
            best_score = score
            best = e
    if best_score < 0.30:
        return None
    return best


def apply_after_turn_segment_corrections(
    actors: List[Dict[str, Any]],
    stage1_entities: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> None:
    """In-place correction for a common failure mode:
    If an entity says 'after_turn' or 'after_merge' but Stage2 placed the actor on a turning segment,
    shift it to the inferred post-turn (exit) segment, preserving distance-from-segment-start in meters.
    """
    veh_info = _infer_vehicle_turn_exit_indices(picked_list, seg_by_id)

    for a in actors:
        placement = a.get("placement", {})
        try:
            tv = int(placement.get("target_vehicle", 0))
            seg_idx = int(placement.get("segment_index", 0))
            s_along = float(placement.get("s_along", 0.0))
        except Exception:
            continue

        if tv <= 0 or seg_idx <= 0:
            continue

        e = _best_entity_match(str(a.get("semantic", "")), stage1_entities, tv)
        if not e:
            continue

        when_phase = e.get("when", None)
        if when_phase not in ("after_turn", "after_merge"):
            continue

        info = veh_info.get(tv)
        if not info:
            continue

        turn_indices = info.get("turn_indices", set())
        exit_idx = info.get("exit_index", None)
        if not exit_idx:
            continue

        if seg_idx not in turn_indices or exit_idx == seg_idx:
            continue

        seg_lens = info.get("seg_lens", [])
        if seg_idx - 1 >= len(seg_lens) or exit_idx - 1 >= len(seg_lens):
            continue
        old_len = float(seg_lens[seg_idx - 1])
        new_len = float(seg_lens[exit_idx - 1])
        if old_len < 1e-6 or new_len < 1e-6:
            continue

        offset_m = float(np.clip(s_along * old_len, 0.0, old_len))
        new_s = float(np.clip(offset_m / new_len, 0.02, 0.98))

        placement["segment_index"] = int(exit_idx)
        placement["s_along"] = new_s

        motion = a.get("motion", {})
        if isinstance(motion, dict) and "anchor_s_along" in motion:
            motion["anchor_s_along"] = new_s


def apply_in_intersection_segment_corrections(
    actors: List[Dict[str, Any]],
    stage1_entities: List[Dict[str, Any]],
    picked_list: List[Dict[str, Any]],
    seg_by_id: Dict[int, np.ndarray],
) -> None:
    """In-place correction: if an entity says 'in_intersection' but was placed off the turn connector,
    shift it onto the inferred turn segment (or a middle segment if unknown).
    """
    veh_info = _infer_vehicle_turn_exit_indices(picked_list, seg_by_id)

    for a in actors:
        placement = a.get("placement", {})
        try:
            tv = int(placement.get("target_vehicle", 0))
            seg_idx = int(placement.get("segment_index", 0))
            s_along = float(placement.get("s_along", 0.0))
        except Exception:
            continue

        if tv <= 0 or seg_idx <= 0:
            continue

        e = _best_entity_match(str(a.get("semantic", "")), stage1_entities, tv)
        if not e:
            continue

        when_phase = e.get("when", None)
        if when_phase != "in_intersection":
            continue

        info = veh_info.get(tv)
        if not info:
            continue

        seg_ids = info.get("seg_ids", [])
        if not seg_ids:
            continue

        turn_indices = info.get("turn_indices", set())
        if seg_idx in turn_indices:
            continue

        if turn_indices:
            new_idx = min(turn_indices, key=lambda i: abs(int(i) - seg_idx))
        else:
            mid = int(round((len(seg_ids) + 1) / 2.0))
            new_idx = max(1, min(mid, len(seg_ids)))

        if len(seg_ids) >= 3 and new_idx == len(seg_ids):
            new_idx = len(seg_ids) - 1

        if new_idx == seg_idx:
            continue

        seg_lens = info.get("seg_lens", [])
        if seg_idx - 1 < len(seg_lens) and new_idx - 1 < len(seg_lens):
            old_len = float(seg_lens[seg_idx - 1])
            new_len = float(seg_lens[new_idx - 1])
            if old_len > 1e-6 and new_len > 1e-6:
                offset_m = float(np.clip(s_along * old_len, 0.0, old_len))
                new_s = float(np.clip(offset_m / new_len, 0.02, 0.98))
                placement["s_along"] = new_s
                motion = a.get("motion", {})
                if isinstance(motion, dict) and "anchor_s_along" in motion:
                    motion["anchor_s_along"] = new_s

        placement["segment_index"] = int(new_idx)


def validate_stage2_output(actors: List[Dict[str, Any]], vehicle_segments: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for i, a in enumerate(actors):
        if not isinstance(a, dict):
            errors.append(f"actors[{i}] is not an object")
            continue

        for k in ("id", "semantic", "category", "asset_id", "placement", "motion", "confidence"):
            if k not in a:
                errors.append(f"actors[{i}] missing key '{k}'")

        placement = a.get("placement", {})
        if not isinstance(placement, dict):
            errors.append(f"actors[{i}].placement must be an object")
            continue

        if placement.get("frame") != "segment":
            errors.append(f"actors[{i}].placement.frame must be 'segment' (for now)")
            continue

        tv = placement.get("target_vehicle")
        if tv not in vehicle_segments:
            errors.append(f"actors[{i}] target_vehicle '{tv}' not found in vehicle_segments")
            continue

        seg_idx = placement.get("segment_index")
        if not isinstance(seg_idx, int):
            errors.append(f"actors[{i}] segment_index must be int")
            continue

        num = vehicle_segments[tv].get("num_segments")
        if isinstance(num, int) and (seg_idx < 1 or seg_idx > num):
            errors.append(f"actors[{i}] segment_index {seg_idx} out of range [1,{num}] for {tv}")

        s_along = placement.get("s_along")
        if not isinstance(s_along, (int, float)) or not (0.0 <= float(s_along) <= 1.0):
            errors.append(f"actors[{i}] s_along must be in [0,1]")

        lat = placement.get("lateral_relation")
        if lat not in LATERAL_RELATIONS:
            errors.append(f"actors[{i}] lateral_relation '{lat}' invalid")

        conf = a.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            errors.append(f"actors[{i}] confidence must be in [0,1]")

        motion = a.get("motion", {})
        if not isinstance(motion, dict):
            errors.append(f"actors[{i}].motion must be an object")
            continue
        mtype = motion.get("type")
        # Accept common alias and normalize.
        if mtype == "following_lane":
            motion["type"] = "follow_lane"
            mtype = "follow_lane"
        if mtype not in ("static", "cross_perpendicular", "follow_lane", "straight_line"):
            errors.append(f"actors[{i}].motion.type invalid")

    return errors


def build_repair_prompt(bad_json_text: str, errors: List[str]) -> str:
    return (
        "You returned JSON but it failed validation.\n"
        "Fix ONLY the JSON to satisfy the constraints. Return JSON ONLY.\n"
        "\n"
        "VALIDATION ERRORS:\n"
        + "\n".join([f"- {e}" for e in errors])
        + "\n\nBAD JSON:\n"
        + bad_json_text
        + "\n"
    )


__all__ = [
    "_best_entity_match",
    "_heading_deg",
    "_infer_heading_change_deg",
    "_infer_vehicle_turn_exit_indices",
    "_segment_length_m",
    "_wrap180",
    "apply_after_turn_segment_corrections",
    "apply_in_intersection_segment_corrections",
    "build_repair_prompt",
    "validate_stage2_output",
]
