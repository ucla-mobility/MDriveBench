#!/usr/bin/env python3
"""
Generate an interactive HTML map viewer for every installed CARLA town.
Each town gets its own tab; hover over points to inspect coordinates, yaw,
and lane metadata. CARLA must be running but the script itself is headless.
"""

from __future__ import annotations

import argparse
import textwrap
from typing import Iterable, List, Tuple

import pandas as pd
import plotly.express as px

try:
    import carla
except ImportError as exc:  # pragma: no cover - runtime dependency
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
        default=2.0,
        help="Sampling distance between waypoints in metres (smaller = denser plot).",
    )
    parser.add_argument(
        "--towns",
        nargs="*",
        default=None,
        help="Optional list of towns to include (e.g. Town05 Town07). If omitted, all installed towns are processed.",
    )
    parser.add_argument(
        "--output",
        default="carla_maps.html",
        help="Output HTML file path (default: carla_maps.html).",
    )
    return parser.parse_args()


def gather_towns(client: carla.Client, requested: Iterable[str] | None) -> List[str]:
    available = sorted({path.split("/")[-1] for path in client.get_available_maps()})
    print("Available CARLA towns:", ", ".join(available))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested towns not installed: {', '.join(missing)}")
        return [town for town in available if town in requested]
    return available


def sample_town(world: carla.World, distance: float) -> pd.DataFrame:
    cmap = world.get_map()
    waypoints = cmap.generate_waypoints(distance=distance)
    rows = []
    for wp in waypoints:
        tf = wp.transform
        loc = tf.location
        rows.append(
            dict(
                x=loc.x,
                y=loc.y,
                z=loc.z,
                yaw=tf.rotation.yaw,
                lane_id=wp.lane_id,
                road_id=wp.road_id,
                section_id=wp.section_id,
            )
        )
    return pd.DataFrame(rows)


def build_plot(df: pd.DataFrame, town: str, distance: float) -> str:
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="lane_id",
        color_continuous_scale="Turbo",
        title=f"{town} – sampled every {distance} m",
        hover_data={
            "x": ":.3f",
            "y": ":.3f",
            "z": ":.3f",
            "yaw": ":.2f",
            "lane_id": True,
            "road_id": True,
            "section_id": True,
        },
        height=800,
        width=800,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"plot_{town}")


def write_html(figures: List[Tuple[str, str]], dst: str) -> None:
    tabs = []
    contents = []
    for idx, (town, fig_html) in enumerate(figures):
        active = " active" if idx == 0 else ""
        display = "block" if idx == 0 else "none"
        tabs.append(
            f'<button class="tablink{active}" onclick="openTab(event, \'{town}\')">{town}</button>'
        )
        contents.append(
            f'<div id="{town}" class="tabcontent" style="display:{display}">{fig_html}</div>'
        )

    html = textwrap.dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <title>CARLA Map Viewer</title>
            <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #101010;
                    color: #f0f0f0;
                    margin: 0;
                    padding: 0;
                }}
                .tab {{
                    background-color: #1f1f1f;
                    padding: 10px;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 6px;
                }}
                .tablink {{
                    background-color: #2d2d2d;
                    border: none;
                    color: #d0d0d0;
                    padding: 10px 18px;
                    cursor: pointer;
                    border-radius: 4px 4px 0 0;
                    font-size: 15px;
                    transition: background-color 0.2s;
                }}
                .tablink:hover {{
                    background-color: #444;
                }}
                .tablink.active {{
                    background-color: #2979ff;
                    color: white;
                }}
                .tabcontent {{
                    display: none;
                    padding: 10px 20px 30px 20px;
                }}
            </style>
        </head>
        <body>
            <div class="tab">
                {''.join(tabs)}
            </div>
            {''.join(contents)}
            <script>
                function openTab(evt, tabName) {{
                    const tabs = document.getElementsByClassName("tabcontent");
                    const links = document.getElementsByClassName("tablink");
                    for (let i = 0; i < tabs.length; i++) {{
                        tabs[i].style.display = "none";
                    }}
                    for (let i = 0; i < links.length; i++) {{
                        links[i].className = links[i].className.replace(" active", "");
                    }}
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </body>
        </html>
        """
    )

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Saved interactive map viewer to {dst}")


def main() -> None:
    args = parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)

    towns = gather_towns(client, args.towns)
    if not towns:
        raise RuntimeError("No CARLA towns found.")

    figures: List[Tuple[str, str]] = []
    for town in towns:
        print(f"[INFO] Processing {town} ...")
        world = client.load_world(town)
        df = sample_town(world, args.distance)
        print(f"  sampled {len(df)} points")
        figures.append((town, build_plot(df, town, args.distance)))

    write_html(figures, args.output)


if __name__ == "__main__":
    main()
