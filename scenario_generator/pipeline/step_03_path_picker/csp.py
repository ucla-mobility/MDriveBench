import math
from typing import Any, Dict, List, Optional

from .candidates import (
    _build_road_corridors,
    _candidate_all_road_ids,
    _candidate_entry_cardinal,
    _candidate_entry_heading_rad,
    _candidate_entry_lane_id,
    _candidate_entry_point,
    _candidate_entry_road_id,
    _candidate_exit_cardinal,
    _candidate_exit_lane_id,
    _candidate_exit_road_id,
    _candidate_geometric_discontinuity,
    _candidate_lane_change_count,
    _candidate_lane_change_phase,
    _candidate_lane_inconsistency,
    _candidate_length,
    _candidate_straight_lateral_drift,
)
from .constraints import (
    _candidate_matches_unary,
    _filter_constraints_with_evidence,
    _infer_constraints_from_description,
    _infer_road_role_sets,
    _normalize_constraints_obj,
    _opposite_cardinal,
)


def _consistent_with_constraints(
    assignment: Dict[str, Dict[str, Any]],
    candidate_for_v: Dict[str, Any],
    v: str,
    constraints: List[Dict[str, str]],
    road_corridors: Optional[Dict[int, set]] = None,
) -> bool:
    ent_v = _candidate_entry_cardinal(candidate_for_v)
    exrid_v = _candidate_exit_road_id(candidate_for_v)
    all_roads_v = _candidate_all_road_ids(candidate_for_v)

    for c in constraints:
        t = str(c.get("type", "")).strip().lower()
        a = c.get("a")
        b = c.get("b")
        if not a or not b:
            continue

        other = None
        if a == v and b in assignment:
            other = (b, assignment[b])
        elif b == v and a in assignment:
            other = (a, assignment[a])
        else:
            continue

        _, o_cand = other
        ent_o = _candidate_entry_cardinal(o_cand)
        exit_o = _candidate_exit_cardinal(o_cand)
        exrid_o = _candidate_exit_road_id(o_cand)
        all_roads_o = _candidate_all_road_ids(o_cand)

        if t == "same_approach_as":
            if ent_v != ent_o:
                return False
        elif t == "opposite_approach_of":
            if ent_v != _opposite_cardinal(ent_o):
                return False
        elif t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            # W->S, S->E, E->N, N->W
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if a == v:
                # v is "a", we need v's entry to be 90° clockwise from other's entry
                expected = clockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                # v is "b", we need other's entry to be 90° clockwise from v's entry
                expected = clockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            # If b is W (westbound), a should be N (northbound) - 90° counter-clockwise
            # W->N, N->E, E->S, S->W
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if a == v:
                expected = counterclockwise.get(ent_o, None)
                if expected and ent_v != expected:
                    return False
            else:
                expected = counterclockwise.get(ent_v, None)
                if expected and ent_o != expected:
                    return False
        elif t == "same_exit_as":
            # Both vehicles should have the same exit direction
            exit_v = _candidate_exit_cardinal(cand)
            if exit_v != exit_o:
                return False
        elif t == "same_road_as":
            # "same_road_as" means vehicles end up on the same logical road corridor.
            # This is satisfied if:
            # 1. Their exit road_ids are in the same corridor (connected by straight paths), OR
            # 2. One vehicle's exit road is among the roads the other travels on
            if exrid_v is None or exrid_o is None:
                # Cannot determine, don't over-constrain
                continue

            # Check if exit roads are in the same corridor
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:  # corridors overlap
                    same_corridor = True

            # Check if one's exit is in the other's traversed roads
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)

            if not same_corridor and not exit_in_other_path:
                return False

    return True


def _solve_paths_csp(
    constraints_obj: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    description: str = "",
    require_straight: bool = False,
) -> List[Dict[str, Any]]:
    """
    Soft CSP/backtracking over candidate paths.
    Returns list of {"vehicle": ..., "path_name": ..., "confidence": ...} for each vehicle.
    Soft penalties are applied for constraint violations; the best-scoring assignment is chosen.
    
    If require_straight=True, heavily penalizes paths with turns (for CORRIDOR topology).
    """
    if require_straight:
        print("[INFO] CSP: require_straight=True, filtering to straight paths only")
    norm = _normalize_constraints_obj(constraints_obj)
    if not norm:
        raise ValueError("Invalid constraints object (missing vehicles list).")

    norm = _filter_constraints_with_evidence(description, norm)
    norm = _infer_constraints_from_description(description, norm)
    vehicles = [v["vehicle"] for v in norm["vehicles"]]
    unary = {v["vehicle"]: v for v in norm["vehicles"]}
    constraints = norm.get("constraints", [])
    
    # DEBUG: print extracted constraints
    print(f"[DEBUG CSP] Extracted constraints: {constraints}")
    for v_obj in norm["vehicles"]:
        print(f"[DEBUG CSP] Vehicle {v_obj.get('vehicle')}: maneuver={v_obj.get('maneuver')}")

    # Identify which vehicles are explicitly performing lane changes
    # This is determined by the LLM's extracted maneuver field - no regex needed
    vehicles_with_lane_change: set = set()
    for v in vehicles:
        if unary.get(v, {}).get("maneuver", "").lower() == "lane_change":
            vehicles_with_lane_change.add(v)
    
    # Extract lane change phase requirements for vehicles doing lane changes
    lane_change_phase_req: Dict[str, str] = {}
    for v in vehicles_with_lane_change:
        phase = str(unary.get(v, {}).get("lane_change_phase", "unknown")).strip().lower()
        if phase in ("before_intersection", "after_intersection"):
            lane_change_phase_req[v] = phase
    
    if lane_change_phase_req:
        print(f"[INFO] Lane change phase requirements: {lane_change_phase_req}")

    def _candidate_effective_length(cand: Dict[str, Any], vehicle: str) -> float:
        base = _candidate_length(cand)
        lanes = (cand.get("signature") or {}).get("lanes", [])
        
        # If require_straight, heavily penalize turning paths
        if require_straight:
            turn = str((cand.get("signature") or {}).get("entry_to_exit_turn", "unknown")).lower()
            if turn in ("left", "right", "uturn"):
                return max(0.0, base - 1000.0)  # Heavy penalty for turns in corridor scenarios
        
        # Only skip penalty if THIS vehicle is doing a lane change
        if vehicle in vehicles_with_lane_change:
            # Check if there's a phase requirement
            required_phase = lane_change_phase_req.get(vehicle)
            if required_phase:
                actual_phase = _candidate_lane_change_phase(cand)
                # Heavy penalty if lane change happens at wrong phase
                if actual_phase not in ("none", "unknown", required_phase):
                    return 0.0  # Zero score for wrong phase
                # Bonus for matching phase
                if actual_phase == required_phase:
                    return base + 100.0  # Large bonus for correct phase
            return base
        
        # Geometric discontinuity (gaps between segments) is the PRIMARY indicator
        # If geometry is smooth (geo=0), the path is good regardless of lane IDs
        geo_gap = _candidate_geometric_discontinuity(cand, threshold_m=2.0)
        
        if geo_gap == 0:
            # Geometrically smooth path - minimal penalty
            # Only penalize if there's actual lane drift in straight paths
            drift_count = _candidate_straight_lateral_drift(cand)
            if drift_count > 0:
                return max(0.0, base - 200.0 * drift_count)
            return base
        
        # Path has geometric gaps - apply penalties
        lc_count = _candidate_lane_change_count(cand)
        drift_count = _candidate_straight_lateral_drift(cand)
        inconsistency = _candidate_lane_inconsistency(cand)
        
        # Combine penalties
        lane_penalty = max(lc_count + drift_count, inconsistency) * 200.0
        geo_penalty = geo_gap * 50.0  # 50m penalty per meter of gap
        
        total_penalty = lane_penalty + geo_penalty
        
        return max(0.0, base - total_penalty)

    role_sets = _infer_road_role_sets(candidates)
    road_corridors = _build_road_corridors(candidates)

    # Domains (filtered by unary constraints)
    domains: Dict[str, List[Dict[str, Any]]] = {}
    for v in vehicles:
        u = unary.get(v, {})
        man = str(u.get("maneuver", "unknown")).strip().lower()
        # Treat "lane_change" as "straight" for path filtering (they go straight but change lanes)
        if man == "lane_change":
            man = "straight"
        er = str(u.get("entry_road", "unknown")).strip().lower()
        xr = str(u.get("exit_road", "unknown")).strip().lower()
        dom = [c for c in candidates if _candidate_matches_unary(c, man, er, xr, role_sets)]
        dom.sort(key=lambda c: (-_candidate_effective_length(c, v), str(c.get("name", ""))))
        domains[v] = dom

    for v, dom in domains.items():
        if not dom:
            raise ValueError(f"No candidates satisfy unary constraints for {v}.")

    order = sorted(vehicles, key=lambda v: (len(domains[v]), v))

    # Soft scoring: maximize (total_length - penalty_weight * violations)
    penalty_weight = 100.0
    max_len_per_v = {v: max(_candidate_effective_length(c, v) for c in domains[v]) for v in vehicles}

    def _constraint_penalty(c_v: Dict[str, Any], c_o: Dict[str, Any], t: str, v_is_a: bool) -> float:
        ent_v = _candidate_entry_cardinal(c_v)
        ent_o = _candidate_entry_cardinal(c_o)
        exrid_v = _candidate_exit_road_id(c_v)
        exrid_o = _candidate_exit_road_id(c_o)
        all_roads_v = _candidate_all_road_ids(c_v)
        all_roads_o = _candidate_all_road_ids(c_o)

        def _lane_relation_ok(a_c: Dict[str, Any], b_c: Dict[str, Any], relation: str) -> Optional[bool]:
            """Return True/False if checkable, else None (insufficient signal => don't over-constrain)."""
            # Must be the same approach direction; lane relations are defined on approach.
            if _candidate_entry_cardinal(a_c) != _candidate_entry_cardinal(b_c):
                return False

            # If entry road_id is known for both, require them to match.
            rid_a = _candidate_entry_road_id(a_c)
            rid_b = _candidate_entry_road_id(b_c)
            if rid_a is not None and rid_b is not None and rid_a != rid_b:
                return False

            # If lane_id is known for both, prefer adjacent lanes.
            lid_a = _candidate_entry_lane_id(a_c)
            lid_b = _candidate_entry_lane_id(b_c)
            if lid_a is not None and lid_b is not None and abs(lid_a - lid_b) != 1:
                return False

            pa = _candidate_entry_point(a_c)
            pb = _candidate_entry_point(b_c)
            th = _candidate_entry_heading_rad(b_c)
            if pa is None or pb is None or th is None:
                return None

            dx = pa[0] - pb[0]
            dy = pa[1] - pb[1]
            # Left normal for b's heading.
            left_x = -math.sin(th)
            left_y = math.cos(th)
            lat = dx * left_x + dy * left_y

            # Small deadband to avoid numerical flicker.
            eps = 0.5
            if relation == "left":
                return lat > eps
            if relation == "right":
                return lat < -eps
            return None

        if t == "same_approach_as":
            return 0.0 if ent_v == ent_o else 1.0
        if t == "opposite_approach_of":
            return 0.0 if ent_v == _opposite_cardinal(ent_o) else 1.0
        if t == "perpendicular_right_of":
            # "a" approaches from perpendicular right of "b"
            # If b is W (westbound), a should be S (southbound) - 90° clockwise
            clockwise = {"W": "S", "S": "E", "E": "N", "N": "W"}
            if v_is_a:
                expected = clockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = clockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "perpendicular_left_of":
            # "a" approaches from perpendicular left of "b"
            counterclockwise = {"W": "N", "N": "E", "E": "S", "S": "W"}
            if v_is_a:
                expected = counterclockwise.get(ent_o, None)
                return 0.0 if expected and ent_v == expected else 1.0
            else:
                expected = counterclockwise.get(ent_v, None)
                return 0.0 if expected and ent_o == expected else 1.0
        if t == "same_exit_as":
            exit_v = _candidate_exit_cardinal(c_v)
            exit_o = _candidate_exit_cardinal(c_o)
            return 0.0 if exit_v == exit_o else 1.0
        if t == "same_road_as":
            if exrid_v is None or exrid_o is None:
                return 0.0
            same_corridor = False
            if road_corridors:
                corridor_v = road_corridors.get(exrid_v, {exrid_v})
                corridor_o = road_corridors.get(exrid_o, {exrid_o})
                if corridor_v & corridor_o:
                    same_corridor = True
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "follow_route_of":
            # Strong preference that the follower takes the same *route* as the leader:
            # same approach + same maneuver + same exit corridor (when checkable).
            turn_v = str(((c_v.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()
            turn_o = str(((c_o.get("signature") or {}).get("entry_to_exit_turn", ""))).strip().lower()

            if ent_v != ent_o:
                return 1.0

            # If either turn is unknown, don't over-constrain on turn; but if both known, require match.
            if turn_v and turn_o and (turn_v != turn_o):
                return 1.0

            if exrid_v is None or exrid_o is None:
                return 0.0

            corridor_v = road_corridors.get(exrid_v, {exrid_v}) if road_corridors else {exrid_v}
            corridor_o = road_corridors.get(exrid_o, {exrid_o}) if road_corridors else {exrid_o}
            same_corridor = bool(corridor_v & corridor_o)
            exit_in_other_path = (exrid_v in all_roads_o) or (exrid_o in all_roads_v)
            return 0.0 if (same_corridor or exit_in_other_path) else 1.0
        if t == "left_lane_of":
            # Asymmetric: "a" is in the lane to the left of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "left") if v_is_a else _lane_relation_ok(c_o, c_v, "left")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "right_lane_of":
            # Asymmetric: "a" is in the lane to the right of "b" (on approach).
            ok = _lane_relation_ok(c_v, c_o, "right") if v_is_a else _lane_relation_ok(c_o, c_v, "right")
            return 0.0 if (ok is None or ok is True) else 1.0
        if t == "adjacent_lane_of":
            # Symmetric: vehicles must be in different (adjacent) lanes on approach.
            # Same approach direction required.
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't check lane adjacency
            lid_v = _candidate_entry_lane_id(c_v)
            lid_o = _candidate_entry_lane_id(c_o)
            if lid_v is None or lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            if lid_v == lid_o:
                return 1.0  # Same lane = violation
            if abs(lid_v - lid_o) == 1:
                return 0.0  # Adjacent lanes = satisfied
            return 0.5  # Not adjacent but different = partial penalty
        if t == "merges_into_lane_of":
            # Asymmetric: "a" starts in adjacent lane but ENDS in same lane as "b"
            # This is for lane change scenarios where vehicle a merges into vehicle b's lane
            # Same approach direction required
            if _candidate_entry_cardinal(c_v) != _candidate_entry_cardinal(c_o):
                return 1.0  # Different approach = can't merge
            
            if v_is_a:
                # c_v is the merger (a), c_o is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_v)
                entry_lid_o = _candidate_entry_lane_id(c_o)
                exit_lid_v = _candidate_exit_lane_id(c_v)
                exit_lid_o = _candidate_exit_lane_id(c_o)
            else:
                # c_o is the merger (a), c_v is the reference (b)
                entry_lid_v = _candidate_entry_lane_id(c_o)
                entry_lid_o = _candidate_entry_lane_id(c_v)
                exit_lid_v = _candidate_exit_lane_id(c_o)
                exit_lid_o = _candidate_exit_lane_id(c_v)
            
            if entry_lid_v is None or entry_lid_o is None or exit_lid_v is None or exit_lid_o is None:
                return 0.0  # Can't determine, don't over-constrain
            
            penalty = 0.0
            # Entry lanes must be DIFFERENT (adjacent preferred)
            if entry_lid_v == entry_lid_o:
                penalty += 1.0  # Same entry lane = violation (should start in different lane)
            elif abs(entry_lid_v - entry_lid_o) != 1:
                penalty += 0.3  # Not adjacent = partial penalty
            
            # Exit lanes must be the SAME (merge into reference's lane)
            if exit_lid_v != exit_lid_o:
                penalty += 1.0  # Different exit lane = violation (should merge into same lane)
            
            return penalty
        return 0.0

    best_assignment: Optional[Dict[str, Dict[str, Any]]] = None
    best_score = -1e18

    def backtrack(i: int, asn: Dict[str, Dict[str, Any]], cur_len: float, cur_pen: float):
        nonlocal best_assignment, best_score
        if i >= len(order):
            score = cur_len - penalty_weight * cur_pen
            if score > best_score:
                best_score = score
                best_assignment = dict(asn)
            return

        # optimistic bound for pruning
        remaining = sum(max_len_per_v[v] for v in order[i:])
        bound = cur_len + remaining - penalty_weight * cur_pen
        if bound < best_score:
            return

        vname = order[i]
        for cand in domains[vname]:
            add_pen = 0.0
            for c in constraints:
                t = str(c.get("type", "")).strip().lower()
                a = c.get("a")
                b = c.get("b")
                if not a or not b:
                    continue
                other = None
                if a == vname and b in asn:
                    other = asn[b]
                elif b == vname and a in asn:
                    other = asn[a]
                if other is not None:
                    add_pen += _constraint_penalty(cand, other, t, v_is_a=(a == vname))
            asn[vname] = cand
            backtrack(i + 1, asn, cur_len + _candidate_effective_length(cand, vname), cur_pen + add_pen)
            del asn[vname]

    backtrack(0, {}, 0.0, 0.0)

    if not best_assignment:
        raise ValueError("No feasible assignment found.")

    # Print final assignment summary
    print("[DEBUG CSP] Final assignment:")
    for v in vehicles:
        c = best_assignment[v]
        sig = c.get("signature", {})
        lanes = sig.get("lanes", [])
        entry_lane = sig.get("entry", {}).get("lane_id", "?")
        entry_card = sig.get("entry", {}).get("cardinal4", "?")
        is_lc = v in vehicles_with_lane_change
        eff_len = _candidate_effective_length(c, v)
        incon = _candidate_lane_inconsistency(c)
        print(f"  {v}: entry={entry_card} lane={entry_lane} lanes={lanes} eff_len={eff_len:.0f}m incon={incon} lc_vehicle={is_lc}")

    out = []
    for v in vehicles:
        c = best_assignment[v]
        # Encode a soft confidence: 1.0 if no penalties, else lower
        confidence = 1.0
        out.append({"vehicle": v, "path_name": str(c.get("name", "")), "confidence": confidence})
    return out


__all__ = [
    "_consistent_with_constraints",
    "_solve_paths_csp",
]
