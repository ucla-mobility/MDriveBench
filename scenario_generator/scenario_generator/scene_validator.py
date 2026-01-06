"""
Scene Validator

Deep validation of generated scenes against natural language descriptions.
Compares scene_objects.json content to what was requested in the scenario text.

Enhanced validation includes:
1. Basic structure validation (vehicle count, maneuvers, actors)
2. Constraint satisfaction (from picked_paths_detailed.json)
3. Path intersection analysis (critical for multi-agent scenarios)
4. Entry/exit direction validation
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum

from .capabilities import CATEGORY_FEASIBILITY, TopologyType

class ValidationIssue(Enum):
    """Types of validation issues that can occur."""
    # Vehicle count issues
    VEHICLE_COUNT_MISMATCH = "vehicle_count_mismatch"
    MISSING_VEHICLE = "missing_vehicle"
    
    # Maneuver issues
    WRONG_MANEUVER = "wrong_maneuver"
    MISSING_MANEUVER = "missing_maneuver"
    
    # Constraint issues
    MISSING_RELATIONSHIP = "missing_relationship"
    WRONG_APPROACH_DIRECTION = "wrong_approach_direction"
    CONSTRAINT_NOT_SATISFIED = "constraint_not_satisfied"
    
    # Path issues (multi-agent specific)
    NO_PATH_INTERSECTION = "no_path_intersection"
    PATHS_NOT_CONFLICTING = "paths_not_conflicting"

    # Topology/geometry issues
    TOPOLOGY_MISMATCH = "topology_mismatch"
    MISSING_MERGE_FEATURE = "missing_merge_feature"
    MISSING_MULTI_LANE = "missing_multi_lane"
    MISSING_LANE_CHANGE = "missing_lane_change"
    MISSING_LANE_DROP = "missing_lane_drop"
    
    # Actor issues
    MISSING_ACTOR = "missing_actor"
    WRONG_ACTOR_TYPE = "wrong_actor_type"
    WRONG_ACTOR_MOTION = "wrong_actor_motion"
    WRONG_ACTOR_POSITION = "wrong_actor_position"
    
    # General issues
    SCENE_EMPTY = "scene_empty"
    PARSE_ERROR = "parse_error"


@dataclass
class ValidationIssueDetail:
    """Detailed information about a validation issue."""
    issue_type: ValidationIssue
    severity: str  # "error", "warning", "info"
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    suggestion: Optional[str] = None
    pipeline_stage: Optional[str] = None  # Which stage likely caused this


@dataclass
class SceneValidationResult:
    """Result of validating a scene against its description."""
    is_valid: bool
    score: float  # 0.0 to 1.0, how well scene matches description
    issues: List[ValidationIssueDetail] = field(default_factory=list)
    
    # Extracted info from description
    expected_vehicles: int = 0
    expected_maneuvers: Dict[str, str] = field(default_factory=dict)
    expected_actors: List[Dict[str, Any]] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)
    
    # Extracted info from scene
    actual_vehicles: int = 0
    actual_maneuvers: Dict[str, str] = field(default_factory=dict)
    actual_actors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Path intersection info (critical for multi-agent scenarios)
    paths_intersect: bool = False
    intersection_region: Optional[Dict[str, float]] = None
    
    def add_issue(self, issue_type: ValidationIssue, severity: str, message: str, 
                  expected: str = None, actual: str = None, 
                  suggestion: str = None, pipeline_stage: str = None):
        self.issues.append(ValidationIssueDetail(
            issue_type=issue_type,
            severity=severity,
            message=message,
            expected=expected,
            actual=actual,
            suggestion=suggestion,
            pipeline_stage=pipeline_stage,
        ))
    
    def get_errors(self) -> List[ValidationIssueDetail]:
        return [i for i in self.issues if i.severity == "error"]
    
    def get_warnings(self) -> List[ValidationIssueDetail]:
        return [i for i in self.issues if i.severity == "warning"]
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Validation: {'PASS' if self.is_valid else 'FAIL'} (score: {self.score:.2f})",
            f"Vehicles: expected {self.expected_vehicles}, got {self.actual_vehicles}",
        ]
        
        # Show path intersection status for multi-agent scenarios
        if self.expected_vehicles >= 2:
            if self.paths_intersect:
                region = self.intersection_region
                if region:
                    lines.append(f"Paths intersect: YES (near x={region['x']:.1f}, y={region['y']:.1f})")
                else:
                    lines.append("Paths intersect: YES")
            else:
                lines.append("Paths intersect: NO ⚠️")
        
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for issue in self.issues:
                prefix = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
                lines.append(f"  {prefix} {issue.message}")
                if issue.suggestion:
                    lines.append(f"      → {issue.suggestion}")
        return "\n".join(lines)


class SceneValidator:
    """
    Validates generated scenes against their natural language descriptions.
    
    Enhanced features:
    - Path intersection detection (critical for multi-agent scenarios)
    - Constraint satisfaction verification from picked_paths
    - Multi-agent coordination validation
    """
    
    # Relationship keywords that imply paths MUST intersect
    INTERSECTION_REQUIRED_PATTERNS = [
        r'intersect', r'cross(?:ing)?(?:\s+path)?', r'conflict(?:ing)?',
        r'negotiate', r'yield', r'right[\s-]?of[\s-]?way',
        r'deadlock', r'standoff', r'competition',
        r'unprotected\s+(?:left|right)', r'perpendicular',
        r'intersection', r'junction',
    ]
    
    def __init__(self):
        # Patterns for extracting info from natural language
        self.vehicle_pattern = re.compile(r'Vehicle\s+(\d+)', re.IGNORECASE)
        self.maneuver_patterns = {
            'left': re.compile(r'Vehicle\s+(\d+)[^.]*(?:turns?\s+left|left\s+turn|turning\s+left)', re.IGNORECASE),
            'right': re.compile(r'Vehicle\s+(\d+)[^.]*(?:turns?\s+right|right\s+turn|turning\s+right)', re.IGNORECASE),
            'straight': re.compile(r'Vehicle\s+(\d+)[^.]*(?:goes?\s+straight|straight\s+through|continues?\s+straight|travels?\s+straight)', re.IGNORECASE),
            'lane_change': re.compile(r'Vehicle\s+(\d+)[^.]*(?:changes?\s+lane|lane\s+change|merges?|merging)', re.IGNORECASE),
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'opposite_approach': re.compile(r'(?:opposite|oncoming|opposing)[\s_]+(?:direction|approach|traffic)', re.IGNORECASE),
            'perpendicular': re.compile(r'(?:perpendicular|from\s+the\s+(?:left|right)|cross(?:ing)?)', re.IGNORECASE),
            'same_approach': re.compile(r'(?:same\s+(?:direction|approach)|behind|following|follows?)', re.IGNORECASE),
            'adjacent_lane': re.compile(r'(?:adjacent\s+lane|left\s+lane|right\s+lane|next\s+lane)', re.IGNORECASE),
            'merging': re.compile(r'(?:merg(?:es?|ing)|lane\s+drop|zipper)', re.IGNORECASE),
        }
        
        # Compiled intersection requirement pattern
        self.intersection_required_pattern = re.compile(
            '|'.join(self.INTERSECTION_REQUIRED_PATTERNS), re.IGNORECASE
        )
        
        # Actor patterns
        self.actor_patterns = {
            'walker': re.compile(r'(?:pedestrian|walker|person|crossing)\s*(?:cross(?:es|ing)?)?', re.IGNORECASE),
            'parked_vehicle': re.compile(r'(?:parked\s+(?:vehicle|car|truck)|stopped\s+vehicle|blocking|obstruction)', re.IGNORECASE),
            'cyclist': re.compile(r'(?:cyclist|bicycle|bike|cycling)', re.IGNORECASE),
            'static_prop': re.compile(r'(?:cone|barrier|traffic\s+cone|construction|debris)', re.IGNORECASE),
        }
        
        # Motion patterns for actors
        self.motion_patterns = {
            'cross_perpendicular': re.compile(r'(?:cross(?:es|ing)?(?:\s+the\s+road)?|perpendicular)', re.IGNORECASE),
            'static': re.compile(r'(?:parked|stopped|stationary|blocking|static)', re.IGNORECASE),
            'follow_lane': re.compile(r'(?:follow(?:s|ing)?\s+(?:the\s+)?lane|along\s+(?:the\s+)?lane)', re.IGNORECASE),
        }
        
        # Lateral position patterns
        self.lateral_patterns = {
            'center': re.compile(r'(?:center|middle|in\s+the\s+lane)', re.IGNORECASE),
            'right_edge': re.compile(r'(?:right\s+(?:edge|side)|at\s+the\s+right)', re.IGNORECASE),
            'left_edge': re.compile(r'(?:left\s+(?:edge|side)|at\s+the\s+left)', re.IGNORECASE),
            'half_right': re.compile(r'(?:half\s+right|partially\s+right)', re.IGNORECASE),
            'half_left': re.compile(r'(?:half\s+left|partially\s+left)', re.IGNORECASE),
        }
    
    def _requires_path_intersection(self, text: str) -> bool:
        """
        Determine if the scenario description implies paths must intersect.
        This is critical for multi-agent coordination scenarios.
        """
        return bool(self.intersection_required_pattern.search(text))
    
    def _extract_polylines_from_scene(self, scene_data: Dict[str, Any]) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract polylines for each vehicle's path from scene_objects.json.
        Returns dict mapping vehicle name to list of (x, y) points.
        """
        polylines = {}
        
        for ego in scene_data.get('ego_picked', []):
            vehicle_name = ego.get('vehicle', 'unknown')
            points = []
            
            # Get polylines from segments_detailed
            for segment in ego.get('signature', {}).get('segments_detailed', []):
                for pt in segment.get('polyline_sample', []):
                    points.append((pt.get('x', 0), pt.get('y', 0)))
            
            if points:
                polylines[vehicle_name] = points
        
        return polylines
    
    def _point_segment_distance(
        self,
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        ax, ay = a
        bx, by = b
        px, py = p
        vx, vy = (bx - ax), (by - ay)
        denom = vx * vx + vy * vy
        if denom < 1e-9:
            return math.hypot(px - ax, py - ay), a
        t = ((px - ax) * vx + (py - ay) * vy) / denom
        t = max(0.0, min(1.0, t))
        cx = ax + t * vx
        cy = ay + t * vy
        return math.hypot(px - cx, py - cy), (cx, cy)

    def _segment_segment_distance(
        self,
        a0: Tuple[float, float],
        a1: Tuple[float, float],
        b0: Tuple[float, float],
        b1: Tuple[float, float],
    ) -> Tuple[float, Tuple[float, float]]:
        candidates = []
        d, cp = self._point_segment_distance(a0, b0, b1)
        candidates.append((d, a0, cp))
        d, cp = self._point_segment_distance(a1, b0, b1)
        candidates.append((d, a1, cp))
        d, cp = self._point_segment_distance(b0, a0, a1)
        candidates.append((d, cp, b0))
        d, cp = self._point_segment_distance(b1, a0, a1)
        candidates.append((d, cp, b1))
        dmin, pa, pb = min(candidates, key=lambda x: x[0])
        mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)
        return dmin, mid

    def _check_polylines_intersect(
        self, 
        poly1: List[Tuple[float, float]], 
        poly2: List[Tuple[float, float]],
        threshold_m: float = 5.0,  # How close paths need to be to "intersect"
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """
        Check if two polylines come within threshold distance of each other.
        Returns (intersects, intersection_point or None).
        
        This is a simplified check - finds minimum distance between polylines.
        """
        if len(poly1) < 2 or len(poly2) < 2:
            return (False, None)

        min_dist = float('inf')
        closest_point = None

        for i in range(len(poly1) - 1):
            a0, a1 = poly1[i], poly1[i + 1]
            for j in range(len(poly2) - 1):
                b0, b1 = poly2[j], poly2[j + 1]
                dist, mid = self._segment_segment_distance(a0, a1, b0, b1)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = mid
                    if min_dist <= threshold_m:
                        return (True, closest_point)

        return (min_dist <= threshold_m, closest_point if min_dist <= threshold_m else None)
    
    def _validate_path_intersections(
        self, 
        scene_data: Dict[str, Any],
        scenario_text: str,
        result: 'SceneValidationResult',
        expected_flags: Optional[Dict[str, Any]] = None,
        expected_relationships: Optional[List[str]] = None,
    ) -> float:
        """
        Validate that paths intersect if the scenario requires it.
        Returns a score component (0.0 to 1.0).
        """
        requires_intersection = self._requires_path_intersection(scenario_text)
        if expected_flags and expected_flags.get("required_topology") == TopologyType.CORRIDOR:
            if expected_flags.get("needs_oncoming") or ("opposite_approach" in (expected_relationships or [])):
                return 1.0
        
        if not requires_intersection:
            # No intersection required, skip this check
            return 1.0
        
        polylines = self._extract_polylines_from_scene(scene_data)
        
        if len(polylines) < 2:
            # Can't check intersection with fewer than 2 vehicles
            result.add_issue(
                ValidationIssue.PATHS_NOT_CONFLICTING, "warning",
                "Cannot verify path intersection with fewer than 2 vehicles",
                pipeline_stage="step_03_path_picker"
            )
            return 0.7
        
        # Check all pairs of vehicles for intersection
        vehicle_names = list(polylines.keys())
        any_intersection = False
        intersection_points = []
        
        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                v1, v2 = vehicle_names[i], vehicle_names[j]
                intersects, point = self._check_polylines_intersect(
                    polylines[v1], polylines[v2], threshold_m=6.0
                )
                if intersects:
                    any_intersection = True
                    if point:
                        intersection_points.append(point)
        
        if any_intersection:
            result.paths_intersect = True
            if intersection_points:
                # Store approximate intersection region
                avg_x = sum(p[0] for p in intersection_points) / len(intersection_points)
                avg_y = sum(p[1] for p in intersection_points) / len(intersection_points)
                result.intersection_region = {'x': avg_x, 'y': avg_y}
            return 1.0
        else:
            result.paths_intersect = False
            result.add_issue(
                ValidationIssue.NO_PATH_INTERSECTION, "error",
                "Scenario requires path intersection but paths do not cross",
                expected="paths within 6m of each other",
                actual="paths do not intersect",
                suggestion="Re-run path picker with intersection-causing constraints (opposite_approach, perpendicular)",
                pipeline_stage="step_03_path_picker"
            )
            return 0.3
    
    def _validate_constraint_satisfaction(
        self,
        scene_dir: Path,
        expected: Dict[str, Any],
        result: 'SceneValidationResult',
    ) -> float:
        """
        Validate constraint satisfaction by reading picked_paths_detailed.json.
        This checks if the paths actually satisfy the spatial constraints.
        """
        picked_paths_path = scene_dir / "picked_paths_detailed.json"
        refined_paths_path = scene_dir / "picked_paths_refined.json"
        
        # Try refined first, then picked
        paths_file = refined_paths_path if refined_paths_path.exists() else picked_paths_path
        
        if not paths_file.exists():
            # Can't validate without picked paths
            return 0.8  # Give benefit of the doubt
        
        try:
            with open(paths_file, 'r') as f:
                paths_data = json.load(f)
        except Exception:
            return 0.8
        
        picked = paths_data.get('picked', paths_data.get('ego_picked', []))
        if len(picked) < 2:
            return 0.9  # Single vehicle, no constraints to validate
        
        # Extract entry/exit info for constraint checking
        entries = {}
        exits = {}
        roads = {}
        lanes = {}
        
        for p in picked:
            name = p.get('vehicle', p.get('name', 'unknown'))
            sig = p.get('signature', {})
            
            entry = sig.get('entry', {})
            exit_info = sig.get('exit', {})
            
            entries[name] = {
                'cardinal': entry.get('cardinal4', 'unknown'),
                'road_id': entry.get('road_id'),
                'lane_id': entry.get('lane_id'),
            }
            exits[name] = {
                'cardinal': exit_info.get('cardinal4', 'unknown'),
                'road_id': exit_info.get('road_id'),
                'lane_id': exit_info.get('lane_id'),
            }
            roads[name] = sig.get('roads', [])
            lanes[name] = sig.get('lanes', [])
        
        score = 1.0
        vehicle_names = list(entries.keys())
        
        # Check expected relationships
        for rel in expected.get('relationships', []):
            if rel == 'opposite_approach':
                # At least two vehicles should have opposite entry cardinals
                cardinals = [
                    entries[v]['cardinal'] for v in vehicle_names
                    if entries[v]['cardinal'] and entries[v]['cardinal'] != 'unknown'
                ]
                if len(cardinals) < 2:
                    continue
                opposite_map = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
                has_opposite = any(
                    opposite_map.get(cardinals[i]) == cardinals[j]
                    for i in range(len(cardinals))
                    for j in range(i+1, len(cardinals))
                )
                if not has_opposite:
                    score -= 0.2
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "warning",
                        "opposite_approach constraint not satisfied",
                        expected="vehicles from opposite directions (N/S or E/W)",
                        actual=f"entry cardinals: {cardinals}",
                        pipeline_stage="step_03_path_picker"
                    )
            
            elif rel == 'perpendicular':
                # At least two vehicles should have perpendicular entry cardinals
                cardinals = [
                    entries[v]['cardinal'] for v in vehicle_names
                    if entries[v]['cardinal'] and entries[v]['cardinal'] != 'unknown'
                ]
                if len(cardinals) < 2:
                    continue
                perp_pairs = {('N', 'E'), ('N', 'W'), ('S', 'E'), ('S', 'W'),
                              ('E', 'N'), ('E', 'S'), ('W', 'N'), ('W', 'S')}
                has_perp = any(
                    (cardinals[i], cardinals[j]) in perp_pairs
                    for i in range(len(cardinals))
                    for j in range(i+1, len(cardinals))
                )
                if not has_perp:
                    score -= 0.2
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "warning",
                        "perpendicular constraint not satisfied",
                        expected="vehicles from perpendicular directions",
                        actual=f"entry cardinals: {cardinals}",
                        pipeline_stage="step_03_path_picker"
                    )
            
            elif rel in ('same_approach', 'adjacent_lane'):
                # Vehicles should share some road segments
                all_roads = [set(roads[v]) for v in vehicle_names]
                if len(all_roads) >= 2:
                    shared = all_roads[0].intersection(*all_roads[1:])
                    if not shared:
                        score -= 0.15
                        result.add_issue(
                            ValidationIssue.CONSTRAINT_NOT_SATISFIED, "info",
                            f"{rel} constraint: no shared road segments found",
                            pipeline_stage="step_03_path_picker"
                        )
            
            elif rel == 'merging':
                # Check if vehicles converge (entries different, exits share road)
                entry_roads = [entries[v]['road_id'] for v in vehicle_names if entries[v]['road_id'] is not None]
                exit_roads = [exits[v]['road_id'] for v in vehicle_names if exits[v]['road_id'] is not None]
                if len(entry_roads) < 2 or len(exit_roads) < 1:
                    continue
                
                # Different entry roads, same exit road suggests merge
                if len(set(entry_roads)) > 1 and len(set(exit_roads)) == 1:
                    pass  # Good!
                else:
                    score -= 0.1
                    result.add_issue(
                        ValidationIssue.CONSTRAINT_NOT_SATISFIED, "info",
                        "merging constraint: paths don't converge as expected",
                        actual=f"entry roads: {entry_roads}, exit roads: {exit_roads}",
                        pipeline_stage="step_03_path_picker"
                    )
        
        return max(0, score)

    def _extract_vehicle_cardinals(self, text: str) -> Dict[str, str]:
        text_l = text.lower()
        patterns = [
            re.compile(r"vehicle\s*(\d+)[^.]*?\bapproach(?:es|ing)?\s+from\s+(north|south|east|west)\b"),
            re.compile(r"vehicle\s*(\d+)[^.]*?\bcoming\s+from\s+(north|south|east|west)\b"),
            re.compile(r"vehicle\s*(\d+)[^.]*?\bfrom\s+(north|south|east|west)\b"),
        ]
        found: Dict[str, str] = {}
        for pattern in patterns:
            for match in pattern.finditer(text_l):
                vehicle = f"Vehicle {match.group(1)}"
                if vehicle not in found:
                    found[vehicle] = match.group(2)
        return found

    def _infer_expected_features(self, scenario_text: str, category: Optional[str]) -> Dict[str, Any]:
        text_l = scenario_text.lower()
        flags = {
            "required_topology": None,
            "needs_oncoming": False,
            "needs_on_ramp": bool(re.search(r"\bon[-\s]?ramp\b", text_l)),
            "needs_merge": bool(re.search(r"\bmerge\b|\bmerges\b|\bmerging\b", text_l)),
            "needs_multi_lane": bool(re.search(
                r"(left\s+lane|right\s+lane|adjacent\s+lane|multi[-\s]?lane|two[-\s]?lane)",
                text_l,
            )),
            "needs_lane_change": bool(re.search(r"(lane\s+change|changes?\s+lane|weav)", text_l)),
            "needs_lane_drop": bool(re.search(r"(lane\s+drop|zipper)", text_l)),
        }
        if category:
            cat = CATEGORY_FEASIBILITY.get(category)
            if cat:
                flags["required_topology"] = cat.required_topology
                flags["needs_oncoming"] = flags["needs_oncoming"] or cat.needs_oncoming
                flags["needs_on_ramp"] = flags["needs_on_ramp"] or cat.needs_on_ramp
                flags["needs_merge"] = flags["needs_merge"] or cat.needs_merge
                flags["needs_multi_lane"] = flags["needs_multi_lane"] or cat.needs_multi_lane
                if "weaving" in cat.name.lower():
                    flags["needs_lane_change"] = True
                if "lane drop" in cat.name.lower():
                    flags["needs_lane_drop"] = True
        if flags["needs_on_ramp"]:
            flags["needs_merge"] = True
        return flags

    def _analyze_scene_geometry(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        entry_roads: List[Optional[int]] = []
        exit_roads: List[Optional[int]] = []
        entry_lanes: List[Optional[int]] = []
        exit_lanes: List[Optional[int]] = []
        entry_dirs: List[str] = []
        exit_dirs: List[str] = []
        turns: List[str] = []
        lane_changes: List[bool] = []

        for ego in scene_data.get("ego_picked", []):
            sig = ego.get("signature", {})
            entry = sig.get("entry", {})
            exit_info = sig.get("exit", {})

            entry_roads.append(entry.get("road_id"))
            exit_roads.append(exit_info.get("road_id"))
            entry_lanes.append(entry.get("lane_id"))
            exit_lanes.append(exit_info.get("lane_id"))
            entry_dirs.append(str(entry.get("cardinal4", "unknown")))
            exit_dirs.append(str(exit_info.get("cardinal4", "unknown")))
            turns.append(str(sig.get("entry_to_exit_turn", "unknown")).lower())

            lanes = sig.get("lanes", [])
            lane_changes.append(len(set(lanes)) > 1 if lanes else False)

        unique_entry_roads = {r for r in entry_roads if r is not None}
        unique_exit_roads = {r for r in exit_roads if r is not None}
        unique_entry_lanes = {l for l in entry_lanes if l is not None}
        unique_exit_lanes = {l for l in exit_lanes if l is not None}

        has_merge_onto_same_road = len(unique_entry_roads) > 1 and len(unique_exit_roads) == 1
        has_lane_change = any(lane_changes)
        has_multi_lane = len(unique_entry_lanes) > 1 or len(unique_exit_lanes) > 1 or has_lane_change
        has_turns = any(t in ("left", "right", "uturn") for t in turns)
        lane_drop_like = (
            len(unique_entry_roads) == 1
            and len(unique_exit_roads) == 1
            and len(unique_entry_lanes) >= 2
            and len(unique_exit_lanes) < len(unique_entry_lanes)
        )

        return {
            "entry_roads": entry_roads,
            "exit_roads": exit_roads,
            "entry_lanes": entry_lanes,
            "exit_lanes": exit_lanes,
            "entry_dirs": entry_dirs,
            "exit_dirs": exit_dirs,
            "turns": turns,
            "distinct_entry_roads": len(unique_entry_roads),
            "distinct_exit_roads": len(unique_exit_roads),
            "distinct_entry_lanes": len(unique_entry_lanes),
            "distinct_exit_lanes": len(unique_exit_lanes),
            "has_merge_onto_same_road": has_merge_onto_same_road,
            "has_multi_lane": has_multi_lane,
            "has_lane_change": has_lane_change,
            "has_turns": has_turns,
            "lane_drop_like": lane_drop_like,
        }

    def extract_expected_from_text(self, text: str) -> Dict[str, Any]:
        """Extract expected scene elements from natural language description."""
        result = {
            'vehicles': [],
            'maneuvers': {},
            'relationships': [],
            'actors': [],
        }
        
        # Extract vehicles
        vehicles = set(self.vehicle_pattern.findall(text))
        result['vehicles'] = sorted([int(v) for v in vehicles])
        
        # Extract maneuvers
        for maneuver, pattern in self.maneuver_patterns.items():
            matches = pattern.findall(text)
            for vehicle_num in matches:
                result['maneuvers'][f"Vehicle {vehicle_num}"] = maneuver
        
        # Extract relationships
        for rel_type, pattern in self.relationship_patterns.items():
            if pattern.search(text):
                result['relationships'].append(rel_type)

        cardinals = self._extract_vehicle_cardinals(text)
        if cardinals:
            card_set = set(cardinals.values())
            has_opposite = ("north" in card_set and "south" in card_set) or ("east" in card_set and "west" in card_set)
            has_perp = any(
                a in card_set and b in card_set
                for a, b in [("north", "east"), ("north", "west"), ("south", "east"), ("south", "west")]
            )
            if has_opposite and "opposite_approach" not in result["relationships"]:
                result["relationships"].append("opposite_approach")
            if has_perp and "perpendicular" not in result["relationships"]:
                result["relationships"].append("perpendicular")
        
        # Extract actors
        for actor_type, pattern in self.actor_patterns.items():
            if pattern.search(text):
                actor = {'kind': actor_type}
                
                # Try to determine motion
                for motion, motion_pattern in self.motion_patterns.items():
                    if motion_pattern.search(text):
                        actor['motion'] = motion
                        break
                
                # Try to determine lateral position
                for lateral, lateral_pattern in self.lateral_patterns.items():
                    if lateral_pattern.search(text):
                        actor['lateral'] = lateral
                        break
                
                result['actors'].append(actor)
        
        return result
    
    def extract_actual_from_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actual scene elements from scene_objects.json."""
        result = {
            'vehicles': [],
            'maneuvers': {},
            'actors': [],
            'entry_directions': {},
            'exit_directions': {},
            'lane_changes': {},
        }
        
        # Extract ego vehicles
        for ego in scene_data.get('ego_picked', []):
            vehicle_name = ego.get('vehicle', '')
            result['vehicles'].append(vehicle_name)
            
            sig = ego.get('signature', {})
            maneuver = sig.get('entry_to_exit_turn', 'unknown')
            result['maneuvers'][vehicle_name] = maneuver
            lanes = sig.get('lanes')
            if isinstance(lanes, list) and lanes:
                result['lane_changes'][vehicle_name] = len(set(lanes)) > 1
            else:
                result['lane_changes'][vehicle_name] = None
            
            entry = sig.get('entry', {})
            exit_info = sig.get('exit', {})
            result['entry_directions'][vehicle_name] = entry.get('cardinal4', 'unknown')
            result['exit_directions'][vehicle_name] = exit_info.get('cardinal4', 'unknown')
        
        # Extract actors
        for actor in scene_data.get('actors', []):
            result['actors'].append({
                'kind': actor.get('category', actor.get('kind', 'unknown')),
                'motion': actor.get('motion', {}).get('type', 'unknown') if isinstance(actor.get('motion'), dict) else 'unknown',
                'lateral': actor.get('placement', {}).get('lateral_relation', 'unknown') if isinstance(actor.get('placement'), dict) else 'unknown',
                'semantic': actor.get('semantic', ''),
            })
        
        return result
    
    def validate_scene(
        self,
        scene_path: str,
        scenario_text: str,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
    ) -> SceneValidationResult:
        """
        Validate a generated scene against its description.
        
        Args:
            scene_path: Path to scene_objects.json
            scenario_text: The natural language scenario description
            
        Returns:
            SceneValidationResult with detailed validation info
        """
        result = SceneValidationResult(is_valid=True, score=1.0)
        
        # Load scene
        try:
            with open(scene_path, 'r') as f:
                scene_data = json.load(f)
        except Exception as e:
            result.is_valid = False
            result.score = 0.0
            result.add_issue(
                ValidationIssue.PARSE_ERROR, "error",
                f"Failed to load scene_objects.json: {e}",
                pipeline_stage="step_05_object_placer"
            )
            return result
        
        # Extract expected elements from text
        expected = self.extract_expected_from_text(scenario_text)
        expected_flags = self._infer_expected_features(scenario_text, category)
        if expected_flags.get("needs_oncoming") and "opposite_approach" not in expected["relationships"]:
            expected["relationships"].append("opposite_approach")
        result.expected_vehicles = len(expected['vehicles'])
        result.expected_maneuvers = expected['maneuvers']
        result.expected_actors = expected['actors']
        result.expected_relationships = expected['relationships']
        
        # Extract actual elements from scene
        actual = self.extract_actual_from_scene(scene_data)
        result.actual_vehicles = len(actual['vehicles'])
        result.actual_maneuvers = actual['maneuvers']
        result.actual_actors = actual['actors']
        geometry = self._analyze_scene_geometry(scene_data)
        
        # Calculate score components
        score_components = []
        
        # 1. Validate vehicle count
        expected_count = len(expected['vehicles'])
        actual_count = len(actual['vehicles'])
        
        if actual_count == 0:
            result.is_valid = False
            result.add_issue(
                ValidationIssue.SCENE_EMPTY, "error",
                "Scene has no ego vehicles",
                expected=f"{expected_count} vehicles",
                actual="0 vehicles",
                suggestion="Check path picker stage",
                pipeline_stage="step_03_path_picker"
            )
            score_components.append(0.0)
        elif actual_count != expected_count:
            severity = "error" if abs(actual_count - expected_count) > 1 else "warning"
            if severity == "error":
                result.is_valid = False
            result.add_issue(
                ValidationIssue.VEHICLE_COUNT_MISMATCH, severity,
                f"Vehicle count mismatch: expected {expected_count}, got {actual_count}",
                expected=str(expected_count),
                actual=str(actual_count),
                suggestion="Re-run path picker with correct vehicle count",
                pipeline_stage="step_03_path_picker"
            )
            score_components.append(min(actual_count, expected_count) / max(actual_count, expected_count))
        else:
            score_components.append(1.0)
        
        # 2. Validate maneuvers
        maneuver_score = 1.0
        for vehicle, expected_maneuver in expected['maneuvers'].items():
            actual_maneuver = actual['maneuvers'].get(vehicle)
            if str(expected_maneuver).lower() == "lane_change":
                lane_change = actual.get('lane_changes', {}).get(vehicle)
                if lane_change is None:
                    result.add_issue(
                        ValidationIssue.MISSING_MANEUVER, "warning",
                        f"{vehicle} lane change could not be verified (no lane data)",
                        expected=expected_maneuver,
                        pipeline_stage="step_03_path_picker"
                    )
                    maneuver_score -= 0.2
                elif not lane_change:
                    result.add_issue(
                        ValidationIssue.WRONG_MANEUVER, "warning",
                        f"{vehicle} did not change lanes as expected",
                        expected=expected_maneuver,
                        actual=actual_maneuver,
                        suggestion="Re-run path picker to include a lane change",
                        pipeline_stage="step_03_path_picker"
                    )
                    maneuver_score -= 0.3
                continue
            if actual_maneuver is None:
                result.add_issue(
                    ValidationIssue.MISSING_MANEUVER, "warning",
                    f"{vehicle} maneuver could not be verified (vehicle not found)",
                    expected=expected_maneuver,
                    pipeline_stage="step_03_path_picker"
                )
                maneuver_score -= 0.2
            elif actual_maneuver.lower() != expected_maneuver.lower():
                result.add_issue(
                    ValidationIssue.WRONG_MANEUVER, "warning",
                    f"{vehicle} has wrong maneuver",
                    expected=expected_maneuver,
                    actual=actual_maneuver,
                    suggestion=f"Re-run path picker to get {expected_maneuver} maneuver",
                    pipeline_stage="step_03_path_picker"
                )
                maneuver_score -= 0.3
        score_components.append(max(0, maneuver_score))
        
        # 3. Validate actors
        actor_score = 1.0
        expected_actor_types = {a['kind'] for a in expected['actors']}
        actual_actor_types = {a['kind'] for a in actual['actors']}
        
        # Map category names to object_placer categories
        category_map = {
            'walker': 'walker',
            'parked_vehicle': 'vehicle',
            'cyclist': 'cyclist',
            'static_prop': 'static',
        }
        
        for expected_type in expected_actor_types:
            mapped_type = category_map.get(expected_type, expected_type)
            if mapped_type not in actual_actor_types and expected_type not in actual_actor_types:
                result.add_issue(
                    ValidationIssue.MISSING_ACTOR, "warning",
                    f"Expected {expected_type} actor not found in scene",
                    expected=expected_type,
                    actual=str(list(actual_actor_types)) if actual_actor_types else "none",
                    suggestion=f"Re-run object placer to add {expected_type}",
                    pipeline_stage="step_05_object_placer"
                )
                actor_score -= 0.3
        
        # Check for actor motion
        if expected['actors']:
            for expected_actor in expected['actors']:
                if 'motion' in expected_actor:
                    expected_motion = expected_actor['motion']
                    found_motion = False
                    for actual_actor in actual['actors']:
                        if actual_actor.get('motion') == expected_motion:
                            found_motion = True
                            break
                    if not found_motion and actual['actors']:
                        result.add_issue(
                            ValidationIssue.WRONG_ACTOR_MOTION, "info",
                            f"Expected actor motion '{expected_motion}' not found",
                            expected=expected_motion,
                            pipeline_stage="step_05_object_placer"
                        )
                        actor_score -= 0.1
        
        score_components.append(max(0, actor_score))
        
        # 4. Validate relationships (basic check based on entry directions)
        relationship_score = 1.0
        if 'opposite_approach' in expected['relationships']:
            # Check if any two vehicles have opposite entry directions
            entry_dirs = [d for d in actual['entry_directions'].values() if d and d != "unknown"]
            opposite_pairs = [('N', 'S'), ('S', 'N'), ('E', 'W'), ('W', 'E')]
            has_opposite = any(
                (entry_dirs[i], entry_dirs[j]) in opposite_pairs
                for i in range(len(entry_dirs))
                for j in range(i+1, len(entry_dirs))
            ) if len(entry_dirs) >= 2 else False
            
            if len(entry_dirs) >= 2 and not has_opposite:
                result.add_issue(
                    ValidationIssue.MISSING_RELATIONSHIP, "warning",
                    "Expected opposite approach relationship not found",
                    expected="vehicles approaching from opposite directions",
                    actual=f"entry directions: {entry_dirs}",
                    suggestion="Re-run path picker with opposite_approach constraint",
                    pipeline_stage="step_03_path_picker"
                )
                relationship_score -= 0.3
        
        if 'perpendicular' in expected['relationships']:
            entry_dirs = [d for d in actual['entry_directions'].values() if d and d != "unknown"]
            perp_pairs = [('N', 'E'), ('N', 'W'), ('S', 'E'), ('S', 'W'),
                         ('E', 'N'), ('E', 'S'), ('W', 'N'), ('W', 'S')]
            has_perp = any(
                (entry_dirs[i], entry_dirs[j]) in perp_pairs
                for i in range(len(entry_dirs))
                for j in range(i+1, len(entry_dirs))
            ) if len(entry_dirs) >= 2 else False
            
            if len(entry_dirs) >= 2 and not has_perp:
                result.add_issue(
                    ValidationIssue.MISSING_RELATIONSHIP, "warning",
                    "Expected perpendicular approach relationship not found",
                    expected="vehicles approaching from perpendicular directions",
                    actual=f"entry directions: {entry_dirs}",
                    suggestion="Re-run path picker with perpendicular constraint",
                    pipeline_stage="step_03_path_picker"
                )
                relationship_score -= 0.3
        
        score_components.append(max(0, relationship_score))

        # 5. Validate topology/geometry requirements
        feature_score = 1.0
        if expected_flags.get("required_topology") == TopologyType.CORRIDOR:
            allow_two_way = expected_flags.get("needs_oncoming") or ("opposite_approach" in expected['relationships'])
            if geometry["distinct_exit_roads"] > 1:
                if allow_two_way and geometry["distinct_exit_roads"] <= 2:
                    pass
                else:
                    expected_msg = "single corridor exit road"
                    if allow_two_way:
                        expected_msg = "at most two exit roads for oncoming corridor"
                    result.add_issue(
                        ValidationIssue.TOPOLOGY_MISMATCH, "error",
                        "Corridor scenario exits to multiple roads",
                        expected=expected_msg,
                        actual=f"exit roads: {geometry['distinct_exit_roads']}",
                        suggestion="Re-run crop picker for corridor geometry",
                        pipeline_stage="step_01_crop",
                    )
                    feature_score -= 0.6
            elif geometry["has_turns"] and not expected_flags.get("needs_merge"):
                result.add_issue(
                    ValidationIssue.TOPOLOGY_MISMATCH, "error",
                    "Corridor scenario contains turning paths",
                    expected="straight paths on a corridor",
                    actual=f"turns: {geometry['turns']}",
                    suggestion="Re-run crop picker for corridor geometry",
                    pipeline_stage="step_01_crop",
                )
                feature_score -= 0.4

        if expected_flags.get("needs_multi_lane") and not geometry["has_multi_lane"]:
            result.add_issue(
                ValidationIssue.MISSING_MULTI_LANE, "error",
                "Multi-lane geometry not present in picked paths",
                expected="at least 2 lanes",
                actual=f"entry lanes: {geometry['distinct_entry_lanes']}, exit lanes: {geometry['distinct_exit_lanes']}, lane_changes: {geometry['has_lane_change']}",
                suggestion="Re-run crop picker for multi-lane geometry",
                pipeline_stage="step_01_crop",
            )
            feature_score -= 0.4

        has_merge_geom = geometry["has_merge_onto_same_road"] or geometry["lane_drop_like"] or geometry["has_lane_change"]
        if expected_flags.get("needs_merge") and not has_merge_geom:
            result.add_issue(
                ValidationIssue.MISSING_MERGE_FEATURE, "error",
                "Merge geometry not found",
                expected="multiple entry roads merging to one exit road",
                actual=(
                    f"entry roads: {geometry['distinct_entry_roads']}, "
                    f"exit roads: {geometry['distinct_exit_roads']}, "
                    f"lane_changes: {geometry['has_lane_change']}, lane_drop: {geometry['lane_drop_like']}"
                ),
                suggestion="Re-run crop picker for merge geometry",
                pipeline_stage="step_01_crop",
            )
            feature_score -= 0.5

        if expected_flags.get("needs_on_ramp") and geometry["distinct_entry_roads"] < 2:
            result.add_issue(
                ValidationIssue.MISSING_MERGE_FEATURE, "error",
                "On-ramp scenario lacks a distinct ramp entry road",
                expected="ramp + mainline entry roads",
                actual=f"entry roads: {geometry['distinct_entry_roads']}",
                suggestion="Re-run crop picker for on-ramp geometry",
                pipeline_stage="step_01_crop",
            )
            feature_score -= 0.3

        if expected_flags.get("needs_lane_change") and not geometry["has_lane_change"]:
            result.add_issue(
                ValidationIssue.MISSING_LANE_CHANGE, "error",
                "Weaving scenario lacks lane changes in picked paths",
                expected="lane changes along at least one path",
                actual=f"lane changes: {geometry['has_lane_change']}",
                suggestion="Re-run path picker with lane-change constraints",
                pipeline_stage="step_03_path_picker",
            )
            feature_score -= 0.4

        if expected_flags.get("needs_lane_drop") and not geometry["lane_drop_like"]:
            result.add_issue(
                ValidationIssue.MISSING_LANE_DROP, "error",
                "Lane drop scenario lacks converging lanes",
                expected="multiple entry lanes converging to fewer exit lanes",
                actual=f"entry lanes: {geometry['distinct_entry_lanes']}, exit lanes: {geometry['distinct_exit_lanes']}",
                suggestion="Re-run crop picker for lane-drop geometry or add cones to create a drop",
                pipeline_stage="step_01_crop",
            )
            feature_score -= 0.4

        if category:
            entry_dirs = [d for d in actual['entry_directions'].values() if d and d != "unknown"]
            unique_dirs = set(entry_dirs)
            perp_pairs = {('N', 'E'), ('N', 'W'), ('S', 'E'), ('S', 'W'),
                          ('E', 'N'), ('E', 'S'), ('W', 'N'), ('W', 'S')}
            has_perp = any(
                (entry_dirs[i], entry_dirs[j]) in perp_pairs
                for i in range(len(entry_dirs))
                for j in range(i + 1, len(entry_dirs))
            ) if len(entry_dirs) >= 2 else False
            has_turn = any(
                m in ("left", "right", "uturn")
                for m in actual['maneuvers'].values()
            )

            if category == "Courtesy & Deadlock Negotiation":
                if len(unique_dirs) < 3 and not has_perp and not has_turn:
                    result.add_issue(
                        ValidationIssue.PATHS_NOT_CONFLICTING, "error",
                        "Courtesy/Deadlock requires perpendicular or turning conflict",
                        expected="perpendicular approaches, turns that cross, or 3+ approach directions",
                        actual=f"entry dirs: {sorted(unique_dirs)}; maneuvers: {list(actual['maneuvers'].values())}",
                        suggestion="Re-run path picker with perpendicular constraints or add turning maneuvers",
                        pipeline_stage="step_03_path_picker",
                    )
                    feature_score -= 0.4

            if category == "Multi-Way Standoff":
                if len(unique_dirs) < 3:
                    result.add_issue(
                        ValidationIssue.PATHS_NOT_CONFLICTING, "error",
                        "Multi-Way Standoff requires 3+ approach directions",
                        expected="vehicles from at least three distinct entry directions",
                        actual=f"entry dirs: {sorted(unique_dirs)}",
                        suggestion="Re-run path picker with perpendicular constraints",
                        pipeline_stage="step_03_path_picker",
                    )
                    feature_score -= 0.4

        score_components.append(max(0, feature_score))

        # 6. Validate path intersections (critical for multi-agent scenarios)
        scene_dir = Path(scene_path).parent
        intersection_score = self._validate_path_intersections(
            scene_data,
            scenario_text,
            result,
            expected_flags=expected_flags,
            expected_relationships=expected['relationships'],
        )
        score_components.append(intersection_score)

        # 7. Validate constraint satisfaction from picked_paths
        constraint_score = self._validate_constraint_satisfaction(scene_dir, expected, result)
        score_components.append(constraint_score)
        
        # Calculate final score (weighted average)
        # Give more weight to critical components
        weights = [
            1.0,  # vehicle count
            0.8,  # maneuvers
            0.6,  # actors
            0.9,  # relationships
            1.0,  # topology/geometry
            1.2,  # path intersection (critical for multi-agent)
            0.9,  # constraint satisfaction
        ]
        
        if len(score_components) == len(weights):
            weighted_sum = sum(s * w for s, w in zip(score_components, weights))
            total_weight = sum(weights)
            result.score = weighted_sum / total_weight
        else:
            result.score = sum(score_components) / len(score_components) if score_components else 0.0
        
        # Determine overall validity
        error_count = len(result.get_errors())
        if error_count > 0:
            result.is_valid = False
        elif result.score < 0.6:
            result.is_valid = False
        
        return result
    
    def get_failing_stage(self, result: SceneValidationResult) -> Optional[str]:
        """
        Determine which pipeline stage likely caused the validation failure.
        Returns the stage that should be re-run.
        """
        if not result.issues:
            return None
        
        # Priority order for re-running stages
        stage_priority = [
            "step_03_path_picker",
            "step_05_object_placer",
            "step_04_path_refiner",
            "step_01_crop",
        ]
        
        # Count issues per stage
        stage_counts = {}
        for issue in result.issues:
            if issue.pipeline_stage:
                stage_counts[issue.pipeline_stage] = stage_counts.get(issue.pipeline_stage, 0) + 1
        
        # Return highest priority stage with issues
        for stage in stage_priority:
            if stage in stage_counts:
                return stage
        
        return None


def validate_scene_file(scene_path: str, scenario_text: str) -> SceneValidationResult:
    """Convenience function to validate a scene file."""
    validator = SceneValidator()
    return validator.validate_scene(scene_path, scenario_text)
