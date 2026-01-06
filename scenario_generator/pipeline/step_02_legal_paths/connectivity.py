from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import ang_diff_deg
from .models import CropBox, LaneSegment, LegalPath


def build_connectivity(segments: List[LaneSegment],
                       connect_radius_m: float = 6.0,
                       connect_yaw_tol_deg: float = 60.0) -> List[List[int]]:
    """
    Build segment-to-segment connectivity graph.
    Two segments are connected if:
    1. End of segment A is close to start of segment B (within connect_radius_m)
    2. Heading at end of A aligns with heading at start of B (within connect_yaw_tol_deg)
    """
    n = len(segments)
    adj: List[List[int]] = [[] for _ in range(n)]
    if n == 0:
        return adj

    starts = np.vstack([seg.points[0] for seg in segments])
    tree = cKDTree(starts)

    for i, seg_a in enumerate(segments):
        end_pt = seg_a.points[-1]
        end_heading = seg_a.heading_at_end()

        candidates = tree.query_ball_point(end_pt, r=connect_radius_m)
        for j in candidates:
            if i == j:
                continue
            seg_b = segments[j]
            start_heading = seg_b.heading_at_start()
            if ang_diff_deg(end_heading, start_heading) <= connect_yaw_tol_deg:
                adj[i].append(j)

    return adj


def identify_boundary_segments(segments: List[LaneSegment],
                               crop: CropBox) -> Tuple[List[int], List[int]]:
    """
    Identify segments that cross the crop boundary.

    Entry segments: start outside, enter inside
    Exit segments: start inside, exit outside
    """
    entry_segments = []
    exit_segments = []

    for i, seg in enumerate(segments):
        start_inside = crop.contains(seg.points[0])
        end_inside = crop.contains(seg.points[-1])

        if not start_inside and end_inside:
            entry_segments.append(i)
        elif not start_inside and not end_inside:
            for pt in seg.points[1:-1]:
                if crop.contains(pt):
                    entry_segments.append(i)
                    break

        if start_inside and not end_inside:
            exit_segments.append(i)
        elif not start_inside and not end_inside:
            if i not in entry_segments:
                for pt in seg.points[1:-1]:
                    if crop.contains(pt):
                        exit_segments.append(i)
                        break

    return entry_segments, exit_segments


def generate_legal_paths(segments: List[LaneSegment],
                         adj: List[List[int]],
                         crop: CropBox,
                         min_path_length: float = 20.0,
                         max_paths: int = 100,
                         max_depth: int = 10,
                         allow_within_region_fallback: bool = True) -> List[LegalPath]:
    """
    Generate legal paths that go from outside the crop area to outside.
    """
    legal_paths: List[LegalPath] = []

    entry_segments, exit_segments = identify_boundary_segments(segments, crop)
    print(f"[INFO] Found {len(entry_segments)} entry segments and {len(exit_segments)} exit segments")

    if len(entry_segments) == 0 or len(exit_segments) == 0:
        print("[WARNING] No entry or exit segments found. Paths must cross crop boundary.")
        if not allow_within_region_fallback:
            print("[INFO] Returning 0 legal paths for this crop (requires boundary-crossing paths).")
            return []
        print("[INFO] Falling back to any paths within the region...")
        entry_segments = list(range(len(segments)))
        exit_segments = list(range(len(segments)))

    exit_set = set(exit_segments)

    def dfs(current_idx: int, path: List[int], total_length: float, depth: int):
        if len(legal_paths) >= max_paths:
            return

        if current_idx in exit_set and len(path) >= 2:
            if total_length >= min_path_length:
                path_segments = [segments[i] for i in path]
                legal_paths.append(LegalPath(path_segments, total_length))
                return

        if depth >= max_depth:
            return

        for next_idx in adj[current_idx]:
            if next_idx in path:
                continue
            next_seg = segments[next_idx]
            new_length = total_length + next_seg.length()
            dfs(next_idx, path + [next_idx], new_length, depth + 1)

    for entry_idx in entry_segments:
        if len(legal_paths) >= max_paths:
            break
        dfs(entry_idx, [entry_idx], segments[entry_idx].length(), 1)

    return legal_paths
