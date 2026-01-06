#!/usr/bin/env python3
"""Convenience launcher for running custom CARLA leaderboard evaluations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from setup_scenario_from_zip import prepare_routes_from_zip, parse_route_metadata


NIGHT_WEATHER = {
    "cloudiness": 20.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 5.0,
    "sun_azimuth_angle": 300.0,
    "sun_altitude_angle": -15.0,
    "wetness": 10.0,
    "fog_distance": 80.0,
    "fog_density": 5.0,
    "fog_falloff": 2.0,
}

CLOUDY_WEATHER = {
    "cloudiness": 100.0,
    "precipitation": 15.0,
    "precipitation_deposits": 25.0,
    "wind_intensity": 45.0,
    "sun_azimuth_angle": 210.0,
    "sun_altitude_angle": 20.0,
    "wetness": 35.0,
    "fog_distance": 70.0,
    "fog_density": 18.0,
    "fog_falloff": 1.5,
}

DEFAULT_NPC_PARAMETER = (
    "simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_npc.yaml"
)


@dataclass
class VariantSpec:
    label: str
    suffix: str | None
    description: str
    weather: dict[str, float] | None = None
    scenario_parameter: str | None = None


@dataclass
class NegotiationMode:
    label: str
    suffix: str | None
    description: str
    disable_comm: bool


@dataclass(frozen=True)
class PlannerSpec:
    agent: str
    agent_config: str


PLANNER_SPECS: dict[str, PlannerSpec] = {
    "colmdriver": PlannerSpec(
        agent="simulation/leaderboard/team_code/colmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
    ),
    "colmdriver_rulebase": PlannerSpec(
        agent="simulation/leaderboard/team_code/colmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_rulebase_config.yaml",
    ),
    "codriving": PlannerSpec(
        agent="simulation/leaderboard/team_code/pnp_agent_e2e_v2v.py",
        agent_config="simulation/leaderboard/team_code/agent_config/pnp_config_codriving_5_10.yaml",
    ),
    "tcp": PlannerSpec(
        agent="simulation/leaderboard/team_code/tcp_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/tcp_5_10_config.yaml",
    ),
    "vad": PlannerSpec(
        agent="simulation/leaderboard/team_code/vad_b2d_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/pnp_config_vad.yaml",
    ),
    "uniad": PlannerSpec(
        agent="simulation/leaderboard/team_code/uniad_b2d_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/uniad.yaml",
    ),
    "lmdrive": PlannerSpec(
        agent="simulation/leaderboard/team_code/lmdriver_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/lmdriver_config_8_10.py",
    ),
}

DEFAULT_PLANNER = "colmdriver"
DEFAULT_AGENT = PLANNER_SPECS[DEFAULT_PLANNER].agent
DEFAULT_AGENT_CONFIG = PLANNER_SPECS[DEFAULT_PLANNER].agent_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", type=Path, help="ZIP produced by scenario_builder.")
    group.add_argument(
        "--routes-dir",
        type=Path,
        help="Existing routes directory prepared for the leaderboard.",
    )

    parser.add_argument(
        "--scenario-name",
        help="Name for the scenario (defaults to ZIP stem or routes-dir name).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("simulation/leaderboard/data/CustomRoutes"),
        help="Where to place extracted routes when using --zip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing scenario directory if it already exists.",
    )
    parser.add_argument(
        "--results-tag",
        help="Folder tag under results/. Defaults to scenario name.",
    )
    parser.add_argument(
        "--ego-num",
        type=int,
        help="Override the number of ego vehicles (auto-detected otherwise).",
    )
    parser.add_argument("--port", type=int, default=2002, help="CARLA server port.")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        dest="tm_port",
        help="Traffic Manager port (defaults to port + 5).",
    )
    parser.add_argument(
        "--scenario-parameter",
        default="simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_no_npc.yaml",
        help="Scenario parameter YAML.",
    )
    parser.add_argument(
        "--scenarios",
        default="simulation/leaderboard/data/scenarios/no_scenarios.json",
        help="Scenario JSON to load alongside routes.",
    )
    parser.add_argument(
        "--planner",
        choices=sorted(PLANNER_SPECS.keys()),
        help=(
            "Select a predefined planner preset (sets --agent/--agent-config defaults). "
            f"Defaults to {DEFAULT_PLANNER} if not set."
        ),
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Path to the evaluation agent Python file (overrides --planner default).",
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Agent configuration file (overrides --planner default).",
    )
    parser.add_argument("--repetitions", type=int, default=1, help="Route repetitions.")
    parser.add_argument("--track", default="SENSORS", help="Leaderboard track name.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Route timeout.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint if available.",
    )
    parser.add_argument(
        "--skip-existed",
        action="store_true",
        default=True,
        help="Skip routes that already have results (default: on).",
    )
    parser.add_argument(
        "--no-skip-existed",
        dest="skip_existed",
        action="store_false",
        help="Force rerun even if results exist.",
    )
    parser.add_argument(
        "--carla-seed", type=int, default=2000, help="Seed for CARLA world randomisation."
    )
    parser.add_argument(
        "--traffic-seed",
        type=int,
        default=2000,
        help="Seed for CARLA Traffic Manager.",
    )
    parser.add_argument(
        "--record",
        default="",
        help="Optional CARLA recording file path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command and exit without running the evaluator.",
    )
    parser.add_argument(
        "--add-night",
        action="store_true",
        help="Also evaluate the same scenario with a forced night-time weather override.",
    )
    parser.add_argument(
        "--add-cloudy",
        action="store_true",
        help="Also evaluate the same scenario under heavy-cloud conditions.",
    )
    parser.add_argument(
        "--add-npcs",
        action="store_true",
        help="Also evaluate with CARLA's random traffic/pedestrians enabled.",
    )
    parser.add_argument(
        "--npc-scenario-parameter",
        default=DEFAULT_NPC_PARAMETER,
        help=(
            "Scenario parameter YAML to use whenever --add-npcs is supplied "
            f"(default: {DEFAULT_NPC_PARAMETER})."
        ),
    )
    parser.add_argument(
        "--add-no-negotiation",
        action="store_true",
        help=(
            "For every scenario variant, also run a copy with inter-ego negotiation disabled."
        ),
    )
    return parser.parse_args()


def append_pythonpath(env: Dict[str, str], path: Path) -> None:
    path_str = str(path)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path_str}:{current}" if current else path_str


def find_carla_egg(carla_root: Path) -> Path:
    dist_dir = carla_root / "PythonAPI" / "carla" / "dist"
    eggs = sorted(dist_dir.glob("carla-*.egg"))
    if not eggs:
        raise FileNotFoundError(f"No CARLA egg found under {dist_dir}")
    py3_eggs = [egg for egg in eggs if "-py3" in egg.name]
    if py3_eggs:
        return py3_eggs[0]
    return eggs[0]


def detect_ego_routes(routes_dir: Path) -> int:
    count = 0
    for xml_path in routes_dir.rglob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            count += 1
    return count


def read_manifest(routes_dir: Path) -> tuple[Path | None, Dict[str, List[Dict[str, str]]] | None]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return manifest_path, None
    return manifest_path, data


def apply_weather_override(routes_dir: Path, weather: dict[str, float]) -> None:
    """Inject/replace weather blocks in every route XML under routes_dir."""
    for xml_path in routes_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()
        route_nodes = list(root.iter("route"))
        if not route_nodes:
            continue
        for route in route_nodes:
            for weather_node in list(route.findall("weather")):
                route.remove(weather_node)
            weather_node = ET.SubElement(route, "weather")
            for key, value in weather.items():
                weather_node.set(key, f"{value}")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def clone_routes_with_weather_override(
    src: Path, weather: dict[str, float] | None
) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """Create a temporary copy of routes with an optional weather override applied."""
    if weather is None:
        return src, None
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{src.name}_")
    clone_root = Path(temp_dir.name) / src.name
    shutil.copytree(src, clone_root)
    apply_weather_override(clone_root, weather)
    return clone_root, temp_dir


def main() -> None:
    args = parse_args()
    if args.planner:
        planner = PLANNER_SPECS[args.planner]
        if args.agent is None:
            args.agent = planner.agent
        if args.agent_config is None:
            args.agent_config = planner.agent_config
    else:
        if args.agent is None:
            args.agent = DEFAULT_AGENT
        if args.agent_config is None:
            args.agent_config = DEFAULT_AGENT_CONFIG
    repo_root = Path(__file__).resolve().parents[1]

    scenario_summary = None
    routes_dir: Path
    scenario_name = args.scenario_name
    manifest_path: Path | None = None
    actors_manifest: Dict[str, List[Dict[str, str]]] | None = None

    if args.zip is not None:
        scenario_summary = prepare_routes_from_zip(
            zip_path=args.zip,
            scenario_name=args.scenario_name,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        routes_dir = scenario_summary["output_dir"]
        scenario_name = scenario_summary["scenario_name"]
        auto_ego_num = scenario_summary["ego_count"]
        manifest_path = scenario_summary.get("manifest_path")
        actors_manifest = scenario_summary.get("actors_manifest")
    else:
        routes_dir = args.routes_dir.expanduser().resolve()
        if not routes_dir.exists():
            raise FileNotFoundError(routes_dir)
        auto_ego_num = detect_ego_routes(routes_dir)
        if auto_ego_num == 0:
            raise RuntimeError(
                f"No ego routes located under {routes_dir}. "
                "Ensure your XML files include role=\"ego\"."
            )
        manifest_path, actors_manifest = read_manifest(routes_dir)

    ego_num = args.ego_num or auto_ego_num
    scenario_name = scenario_name or routes_dir.name
    results_tag = args.results_tag or scenario_name

    variants: list[VariantSpec] = [
        VariantSpec(
            label="baseline",
            suffix=None,
            description="Default daylight / clear run.",
            scenario_parameter=args.scenario_parameter,
            weather=None,
        )
    ]
    if args.add_night:
        variants.append(
            VariantSpec(
                label="night",
                suffix="night",
                description="Night-time lighting (sun altitude override).",
                weather=NIGHT_WEATHER,
                scenario_parameter=args.scenario_parameter,
            )
        )
    if args.add_cloudy:
        variants.append(
            VariantSpec(
                label="cloudy",
                suffix="cloudy",
                description="Daylight with heavy cloud cover.",
                weather=CLOUDY_WEATHER,
                scenario_parameter=args.scenario_parameter,
            )
        )
    if args.add_npcs:
        variants.append(
            VariantSpec(
                label="npcs",
                suffix="npcs",
                description="Interdrive NPC parameter file (random traffic/pedestrians).",
                scenario_parameter=args.npc_scenario_parameter,
                weather=None,
            )
        )

    negotiation_modes: list[NegotiationMode] = [
        NegotiationMode(
            label="negotiation",
            suffix=None,
            description="Negotiation enabled (default).",
            disable_comm=False,
        )
    ]
    if args.add_no_negotiation:
        negotiation_modes.append(
            NegotiationMode(
                label="no-negotiation",
                suffix="nonego",
                description="Negotiation disabled; egos act independently.",
                disable_comm=True,
            )
        )

    carla_root = repo_root / "carla"
    leaderboard_root = repo_root / "simulation" / "leaderboard"
    scenario_runner_root = repo_root / "simulation" / "scenario_runner"
    data_root = repo_root / "simulation" / "assets" / "v2xverse_debug"

    env_template = os.environ.copy()
    env_template["CARLA_ROOT"] = str(carla_root)
    env_template["LEADERBOARD_ROOT"] = str(leaderboard_root)
    env_template["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_root)
    env_template["DATA_ROOT"] = str(data_root)
    env_template.setdefault("COLMDRIVER_OFFLINE", "0")

    append_pythonpath(env_template, scenario_runner_root)
    append_pythonpath(env_template, leaderboard_root)
    append_pythonpath(env_template, leaderboard_root / "team_code")
    append_pythonpath(env_template, carla_root / "PythonAPI")
    append_pythonpath(env_template, carla_root / "PythonAPI" / "carla")
    append_pythonpath(env_template, find_carla_egg(carla_root))

    tm_port = args.tm_port or (args.port + 5)

    evaluator = (
        leaderboard_root
        / "leaderboard"
        / "leaderboard_evaluator_parameter.py"
    )
    agent_path = (repo_root / args.agent).resolve()
    agent_config_path = (repo_root / args.agent_config).resolve()
    scenarios_json = (repo_root / args.scenarios).resolve()

    print("Scenario directory:", routes_dir)
    print("Ego vehicles:", ego_num)
    if scenario_summary is not None:
        print("Actors discovered:")
        for path, role, route_id, town in scenario_summary["routes"]:
            rel = path.relative_to(routes_dir)
            print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    elif actors_manifest:
        print("Actors discovered:")
        for role, entries in actors_manifest.items():
            for entry in entries:
                rel = entry.get("file", "?")
                route_id = entry.get("route_id", "?")
                town = entry.get("town", "?")
                print(f"  - {role:11s} route_id={route_id:>4s} town={town} file={rel}")
    if actors_manifest:
        non_ego = {
            role: len(entries)
            for role, entries in actors_manifest.items()
            if role != "ego"
        }
        if non_ego:
            print(
                "Non-ego actor counts:",
                ", ".join(f"{role}={count}" for role, count in sorted(non_ego.items())),
            )
    if manifest_path:
        print("Actor manifest:", manifest_path)

    run_matrix = [(spec, nego) for spec in variants for nego in negotiation_modes]
    if run_matrix:
        print("\nConfigured evaluation variants:")
        for spec, nego in run_matrix:
            base_name = spec.suffix or "baseline"
            mode_note = " (no negotiation)" if nego.disable_comm else ""
            print(f"  - {base_name}{mode_note}: {spec.description} [{nego.description}]")

    for spec, negotiation in run_matrix:
        variant_routes_dir, temp_dir = clone_routes_with_weather_override(routes_dir, spec.weather)
        try:
            env = env_template.copy()
            env["ROUTES"] = str(variant_routes_dir)
            env["ROUTES_DIR"] = str(variant_routes_dir)

            variant_manifest = variant_routes_dir / "actors_manifest.json"
            if variant_manifest.exists():
                env["CUSTOM_ACTOR_MANIFEST"] = str(variant_manifest)
                actors_root = variant_manifest.parent / "actors"
                if actors_root.exists():
                    env["CUSTOM_ACTOR_ROOT"] = str(actors_root)
                else:
                    env.pop("CUSTOM_ACTOR_ROOT", None)
            else:
                env.pop("CUSTOM_ACTOR_MANIFEST", None)
                env.pop("CUSTOM_ACTOR_ROOT", None)

            if negotiation.disable_comm:
                env["COLMDRIVER_DISABLE_NEGOTIATION"] = "1"
            else:
                env.pop("COLMDRIVER_DISABLE_NEGOTIATION", None)

            suffix_parts: list[str] = []
            if spec.suffix:
                suffix_parts.append(spec.suffix)
            if negotiation.suffix:
                suffix_parts.append(negotiation.suffix)
            variant_suffix = "_".join(suffix_parts) if suffix_parts else None
            variant_results_tag = results_tag if not variant_suffix else f"{results_tag}_{variant_suffix}"
            result_root = repo_root / "results" / "results_driving_custom" / variant_results_tag
            save_path = result_root / "image"
            checkpoint_endpoint = result_root / "results.json"
            save_path.mkdir(parents=True, exist_ok=True)
            checkpoint_endpoint.parent.mkdir(parents=True, exist_ok=True)

            env["RESULT_ROOT"] = str(result_root)
            env["SAVE_PATH"] = str(save_path)
            env["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint)

            scenario_parameter_rel = spec.scenario_parameter or args.scenario_parameter
            scenario_parameter_path = (repo_root / scenario_parameter_rel).resolve()

            cmd: List[str] = [
                sys.executable,
                str(evaluator),
                "--routes_dir",
                str(variant_routes_dir),
                "--ego-num",
                str(ego_num),
                "--scenario_parameter",
                str(scenario_parameter_path),
                "--agent",
                str(agent_path),
                "--agent-config",
                str(agent_config_path),
                "--port",
                str(args.port),
                "--trafficManagerPort",
                str(tm_port),
                "--scenarios",
                str(scenarios_json),
                "--repetitions",
                str(args.repetitions),
                "--track",
                args.track,
                "--checkpoint",
                str(checkpoint_endpoint),
                "--debug",
                "1" if args.debug else "0",
                "--record",
                args.record,
                "--resume",
                "1" if args.resume else "0",
                "--carlaProviderSeed",
                str(args.carla_seed),
                "--trafficManagerSeed",
                str(args.traffic_seed),
                "--skip_existed",
                "1" if args.skip_existed else "0",
                "--timeout",
                str(args.timeout),
            ]

            variant_name = spec.suffix or "baseline"
            if negotiation.disable_comm:
                variant_name = f"{variant_name} (no negotiation)"
            print(f"\n=== Running variant: {variant_name} ===")
            print("Scenario parameter:", scenario_parameter_path)
            if spec.weather:
                print("Weather override applied via:", variant_routes_dir)
            print("Results will be stored under:", result_root)
            print("Command:")
            print("  " + " ".join(cmd))

            if args.dry_run:
                continue

            subprocess.run(cmd, check=True, env=env)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


if __name__ == "__main__":
    main()
