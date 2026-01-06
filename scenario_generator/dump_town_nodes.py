#!/usr/bin/env python3
"""Dump CARLA scenario builder node data for a single town to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc

from scenario_builder import compute_payload, list_available_towns, sample_town


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--distance",
        type=float,
        default=2.0,
        help="Sampling distance between reference waypoints in metres.",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help="Optional town name (e.g., Town05). Defaults to whichever map is currently loaded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("town_nodes.json"),
        help="Output JSON path (default: town_nodes.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("town_nodes"),
        help="Directory for multiple towns when --town is omitted (default: town_nodes/).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the JSON output for readability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    available = list_available_towns(client)
    if args.town:
        if args.town not in available:
            available_list = ", ".join(available)
            raise ValueError(f"Requested town {args.town!r} is not available. Found: {available_list}")
        target_towns = [args.town]
    else:
        target_towns = available
        if not target_towns:
            raise RuntimeError("No CARLA maps found on the server.")

    if len(target_towns) == 1 and args.town:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_paths = {target_towns[0]: args.output}
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {town: args.output_dir / f"{town}.json" for town in target_towns}

    for town_name in target_towns:
        print(f"Sampling {town_name} with spacing {args.distance:.2f} m ...")
        world = client.load_world(town_name)
        df = sample_town(world, args.distance)
        payload = compute_payload(df)

        output_data = {
            "town": town_name,
            "distance": args.distance,
            "node_count": len(df),
            "payload": payload,
        }

        json_text = json.dumps(output_data, indent=2 if args.pretty else None)
        output_path = output_paths[town_name]
        output_path.write_text(json_text, encoding="utf-8")
        print(f"Wrote {len(df)} nodes for {town_name} to {output_path}")


if __name__ == "__main__":
    main()
