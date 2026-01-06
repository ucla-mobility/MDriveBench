#!/usr/bin/env python3
"""
convert_scene_to_routes.py

Wrapper for pipeline/step_06_scene_to_routes.
"""

from pipeline.step_06_scene_to_routes.main import (
    convert_scene_to_routes,
    extract_town_from_nodes_path,
    create_route_xml,
    extract_waypoints_from_path,
    prettify_xml,
    safe_filename,
    write_xml_file,
    main,
)

__all__ = [
    "convert_scene_to_routes",
    "extract_town_from_nodes_path",
    "create_route_xml",
    "extract_waypoints_from_path",
    "prettify_xml",
    "safe_filename",
    "write_xml_file",
    "main",
]


if __name__ == "__main__":
    exit(main())
