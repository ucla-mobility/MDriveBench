#!/usr/bin/env python3
"""
run_path_refiner.py

Wrapper for pipeline/step_04_path_refiner.
"""

from pipeline.step_04_path_refiner.csp import (
    _candidate_speeds,
    _default_start_end_in_crop,
    _eval_spawn_relation,
    _speed_class_to_mps,
    refine_spawn_and_speeds_soft_csp,
)
from pipeline.step_04_path_refiner.geometry import (
    CropBox,
    _build_segment_payload_from_polyline,
    _closest_point_on_segment,
    _cross2,
    _dist,
    _dot,
    _find_closest_idx,
    _first_last_idx_in_crop,
    _forward_axis_at,
    _heading_deg,
    _polyline_bbox,
    _polyline_cumdist,
    _polyline_length,
    _polyline_slice,
    _resample_polyline,
    _segment_segment_closest,
    _segments_to_polyline_with_map,
    _slice_segments_detailed,
    _unit,
    find_conflict_between_polylines,
)
from pipeline.step_04_path_refiner.lane_change import apply_merge_into_lane_of
from pipeline.step_04_path_refiner.llm import (
    _chat_template,
    _extract_first_json_object,
    _llm_generate_json,
    extract_refinement_constraints,
)
from pipeline.step_04_path_refiner.main import load_model, main, refine_picked_paths_with_model
from pipeline.step_04_path_refiner.viz import plt, visualize_refinement

__all__ = [
    "CropBox",
    "_build_segment_payload_from_polyline",
    "_candidate_speeds",
    "_chat_template",
    "_closest_point_on_segment",
    "_cross2",
    "_default_start_end_in_crop",
    "_dist",
    "_dot",
    "_eval_spawn_relation",
    "_extract_first_json_object",
    "_find_closest_idx",
    "_first_last_idx_in_crop",
    "_forward_axis_at",
    "_heading_deg",
    "_llm_generate_json",
    "_polyline_bbox",
    "_polyline_cumdist",
    "_polyline_length",
    "_polyline_slice",
    "_resample_polyline",
    "_segment_segment_closest",
    "_segments_to_polyline_with_map",
    "_slice_segments_detailed",
    "_speed_class_to_mps",
    "_unit",
    "apply_merge_into_lane_of",
    "extract_refinement_constraints",
    "find_conflict_between_polylines",
    "load_model",
    "main",
    "plt",
    "refine_picked_paths_with_model",
    "refine_spawn_and_speeds_soft_csp",
    "visualize_refinement",
]


if __name__ == "__main__":
    main()
