#!/usr/bin/env python3
"""
Sample a CARLA town and create a static scatter plot saved to disk.
If mplcursors is installed, hovering in the interactive window will show
exact coordinates; otherwise the script simply saves a PNG.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import matplotlib.pyplot as plt

try:
    import mplcursors  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mplcursors = None

try:
    import carla
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import CARLA. Ensure the CARLA PythonAPI egg is on PYTHONPATH."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2000, help="CARLA port (default: 2000)")
    parser.add_argument(
        "--distance",
        type=float,
        default=3.0,
        help="Sampling distance between waypoints in metres.",
    )
    parser.add_argument(
        "--town",
        default=None,
        help="Optional town to load (e.g., Town05). If omitted, uses the current world.",
    )
    parser.add_argument(
        "--output",
        default="carla_map.png",
        help="Output image path. If omitted, a window is shown instead.",
    )
    return parser.parse_args()


def gather_maps(client: carla.Client, requested: Iterable[str] | None) -> list[str]:
    available = sorted({path.split("/")[-1] for path in client.get_available_maps()})
    print("Available CARLA towns:", ", ".join(available))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested towns not installed: {', '.join(missing)}")
    return available


def main() -> None:
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    gather_maps(client, None)

    if args.town:
        world = client.load_world(args.town)
    else:
        world = client.get_world()

    cmap = world.get_map()
    waypoints = cmap.generate_waypoints(distance=args.distance)

    xs, ys, yaws, lane_ids, road_ids = [], [], [], [], []
    for wp in waypoints:
        transform = wp.transform
        loc = transform.location
        xs.append(loc.x)
        ys.append(loc.y)
        yaws.append(transform.rotation.yaw)
        lane_ids.append(wp.lane_id)
        road_ids.append(wp.road_id)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(xs, ys, s=6, c=lane_ids, cmap="coolwarm", linewidths=0.0, alpha=0.9)
    ax.set_aspect("equal", "box")
    ax.set_title(f"{cmap.name} – sampled every {args.distance} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("lane_id")

    if mplcursors:
        cursor = mplcursors.cursor(scatter, hover=True)

        @cursor.connect("add")
        def _(sel):
            idx = sel.target.index
            sel.annotation.set_text(
                f"x={xs[idx]:.3f}\n"
                f"y={ys[idx]:.3f}\n"
                f"yaw={yaws[idx]:.2f}\n"
                f"lane_id={lane_ids[idx]}\n"
                f"road_id={road_ids[idx]}"
            )
    else:
        print("Tip: install 'mplcursors' (`pip install mplcursors`) for hover tooltips.")

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches="tight")
        print(f"Saved map preview to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
