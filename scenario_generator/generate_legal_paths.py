#!/usr/bin/env python3
"""
visualize_legal_paths.py

Wrapper for pipeline/step_02_legal_paths.
"""

from pipeline.step_02_legal_paths.connectivity import build_connectivity, generate_legal_paths
from pipeline.step_02_legal_paths.geometry import ang_diff_deg, cumulative_dist, heading_deg_from_vec, sanitize, unit_from_yaw_deg, wrap180
from pipeline.step_02_legal_paths.main import main
from pipeline.step_02_legal_paths.models import CropBox, LaneSegment, LegalPath
from pipeline.step_02_legal_paths.prompt import save_aggregated_signatures_json, save_prompt_file
from pipeline.step_02_legal_paths.segments import build_segments, crop_segments, load_nodes
from pipeline.step_02_legal_paths.signatures import (
    bound_word,
    build_path_signature,
    build_segments_detailed_for_path,
    classify_turn_world,
    heading_to_cardinal4_world,
    make_path_name,
)
from pipeline.step_02_legal_paths.viz import visualize_individual_paths, visualize_legal_paths

__all__ = [
    "CropBox",
    "LaneSegment",
    "LegalPath",
    "ang_diff_deg",
    "build_connectivity",
    "build_path_signature",
    "build_segments",
    "build_segments_detailed_for_path",
    "classify_turn_world",
    "cumulative_dist",
    "crop_segments",
    "generate_legal_paths",
    "heading_deg_from_vec",
    "heading_to_cardinal4_world",
    "load_nodes",
    "make_path_name",
    "sanitize",
    "save_aggregated_signatures_json",
    "save_prompt_file",
    "unit_from_yaw_deg",
    "visualize_individual_paths",
    "visualize_legal_paths",
    "wrap180",
    "bound_word",
]


if __name__ == "__main__":
    main()
