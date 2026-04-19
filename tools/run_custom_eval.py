#!/usr/bin/env python3
"""Convenience launcher for running custom CARLA leaderboard evaluations."""

from __future__ import annotations

import atexit
import argparse
from collections import deque
import datetime as dt
import importlib.util
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, TextIO, Tuple

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

# Ensure repository-root imports (e.g. common/*) resolve when launched as
# "python tools/run_custom_eval.py" from arbitrary working directories.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT_IMPORT = _SCRIPT_DIR.parent
if str(_REPO_ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_IMPORT))

from setup_scenario_from_zip import prepare_routes_from_zip, parse_route_metadata

try:
    from common.carla_connection_events import (
        create_logged_client,
        get_logged_client_id,
        install_process_lifecycle_logging,
        log_carla_event,
        log_client_release_begin,
        log_client_release_end,
        log_client_retry,
        log_client_rpc_ready,
        log_process_exception,
        set_connection_events_logdir,
        set_planner_context,
    )
except Exception:  # pragma: no cover - logger is optional outside launcher workflows
    def create_logged_client(
        carla_module: Any,
        host: str,
        port: int,
        *,
        timeout_s: float | None = None,
        context: str = "",
        attempt: int | None = None,
        process_name: str | None = None,
    ) -> Any:
        del context, attempt, process_name
        client = carla_module.Client(host, int(port))
        if timeout_s is not None:
            client.set_timeout(float(timeout_s))
        return client

    def install_process_lifecycle_logging(
        process_name: str,
        *,
        env_keys: Iterable[str] | None = None,
    ) -> None:
        del process_name, env_keys

    def log_carla_event(event_type: str, *, process_name: str | None = None, **fields: Any) -> None:
        del event_type, process_name, fields

    def get_logged_client_id(client: Any | None) -> str | None:
        del client
        return None

    def log_client_release_begin(
        client: Any | None,
        *,
        context: str = "",
        process_name: str | None = None,
        reason: str = "",
    ) -> str | None:
        del client, context, process_name, reason
        return None

    def log_client_release_end(
        client_id: str | None,
        *,
        context: str = "",
        process_name: str | None = None,
        reason: str = "",
    ) -> None:
        del client_id, context, process_name, reason

    def log_client_retry(
        *,
        host: str,
        port: int,
        context: str = "",
        attempt: int | None = None,
        sleep_s: float | None = None,
        reason: str = "",
        process_name: str | None = None,
    ) -> None:
        del host, port, context, attempt, sleep_s, reason, process_name

    def log_client_rpc_ready(
        *,
        host: str,
        port: int,
        context: str = "",
        attempt: int | None = None,
        process_name: str | None = None,
    ) -> None:
        del host, port, context, attempt, process_name

    def log_process_exception(exc: BaseException, *, process_name: str, where: str = "") -> None:
        del exc, process_name, where

    def set_connection_events_logdir(logdir: str | Path, *, process_name: str | None = None) -> Path:
        del process_name
        root = Path(logdir)
        return root / "carla_connection_events.log"

    def set_planner_context(
        planner_name: str,
        *,
        world_port: int,
        tm_port: int,
        stream_port: int | None = None,
        route_id: str = "",
        gpu_id: str = "",
        process_name: str | None = None,
    ) -> None:
        """Fallback stub: just export the context via env vars."""
        import os as _os
        _os.environ["CARLA_PLANNER_NAME"] = str(planner_name)
        _os.environ["CARLA_WORLD_PORT"]   = str(world_port)
        _os.environ["CARLA_STREAM_PORT"]  = str(stream_port if stream_port is not None else world_port + 1)
        _os.environ["CARLA_TM_PORT"]      = str(tm_port)


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

OVERCAST_WEATHER = {
    "cloudiness": 100.0,
    "precipitation": 0.0,
    "precipitation_deposits": 0.0,
    "wind_intensity": 25.0,
    "sun_azimuth_angle": 210.0,
    "sun_altitude_angle": 45.0,
    "wetness": 20.0,
    "fog_distance": 60.0,
    "fog_density": 20.0,
    "fog_falloff": 1.5,
}

DEFAULT_NPC_PARAMETER = (
    "simulation/leaderboard/leaderboard/scenarios/scenario_parameter_Interdrive_npc.yaml"
)

CARLA_STARTUP_TIMEOUT = 60.0
CARLA_RESTART_WAIT = 2.0
CARLA_POST_START_BUFFER_DEFAULT = 5.0
CARLA_CRASH_MULTIPLIER = 2.0
CARLA_PORT_TRIES = 8
CARLA_PORT_STEP = 1
CARLA_STREAM_PORT_OFFSET_DEFAULT = 1
CARLA_TM_PORT_OFFSET_DEFAULT = 5
CARLA_STREAM_DEBUG_SUMMARY_EVERY_DEFAULT = 200
CARLA_FORENSICS_MALLOC_CHECK_DEFAULT = "3"
CARLA_FORENSICS_MALLOC_PERTURB_DEFAULT = "137"
CARLA_ROOTCAUSE_MODE_DEFAULT = "auto"
CARLA_ROOTCAUSE_DIAG_INTERVAL_DEFAULT = 2
CARLA_ROOTCAUSE_STARTUP_TIMEOUT_DEFAULT = 180
CARLA_ROOTCAUSE_CORE_WAIT_SECONDS_DEFAULT = 15
COLMDRIVER_VLLM_ENV_DEFAULT = "vllm"
COLMDRIVER_VLM_MODEL_DEFAULT = "ckpt/colmdriver/VLM"
COLMDRIVER_LLM_MODEL_DEFAULT = "ckpt/colmdriver/LLM"
COLMDRIVER_VLM_MAX_MODEL_LEN_DEFAULT = 8192
COLMDRIVER_LLM_MAX_MODEL_LEN_DEFAULT = 4096
COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT = 300.0
COLMDRIVER_VLLM_PLANNERS = frozenset({"colmdriver", "colmdriver_rulebase"})
AUTO_WORKERS_PER_GPU_SETTLE_TIMEOUT_DEFAULT = 20.0
AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT = 6144
AUTO_WORKERS_PER_GPU_STABLE_SAMPLES_DEFAULT = 2
AUTO_WORKERS_PER_GPU_POLL_SECONDS_DEFAULT = 2.0
AUTO_WORKERS_PER_GPU_STABLE_TOLERANCE_MIB_DEFAULT = 512
AUTO_WORKERS_MAX_PER_GPU_DEFAULT = 2
OOM_CONCURRENCY_BACKOFF_STEP_DEFAULT = 1
OOM_FREE_MIB_PENALTY_STEP_DEFAULT = 2048
OOM_RECOVERY_SUCCESS_STREAK_DEFAULT = 2
OOM_RECOVERY_PENALTY_DECAY_STEP_DEFAULT = 1024
OOM_BACKOFF_DECAY_SECONDS_DEFAULT = 600.0  # stale OOM penalty fully decays after this many seconds

# --- VRAM-aware scheduler defaults ------------------------------------------------
CARLA_FIXED_VRAM_MIB_DEFAULT = 4096
VRAM_SAFETY_BUFFER_MIB_DEFAULT = 2048
VRAM_SAFETY_BUFFER_AGGRESSIVE_MIB_DEFAULT = 512
MIN_FREE_AGGRESSIVE_MIB_DEFAULT = 1024
GPU_MONITOR_POLL_SECONDS_DEFAULT = 2.0
SCHEDULER_TICK_POLL_TIMEOUT_S_DEFAULT = 0.5
VRAM_ESTIMATE_CACHE_DEFAULT = str(
    Path.home() / ".cache" / "colmdriver" / "planner_vram_estimates.json"
)
VRAM_ESTIMATE_EMA_ALPHA_DEFAULT = 0.4  # weight for the new observation when updating

# Conservative a-priori VRAM estimates (MiB) for each planner's client footprint
# on top of the managed CARLA's fixed overhead.  These are used as the initial
# seed when no persistent observation exists yet; the scheduler refines them
# automatically via :class:`PlannerVramEstimator`.
PLANNER_VRAM_ESTIMATE_MIB: Dict[str, int] = {
    "autopilot": 1500,
    "log-replay": 1500,
    "minimal": 1500,
    "tcp": 3500,
    "codriving": 6000,
    "vad": 10000,
    "uniad": 12000,
    "lmdrive": 20000,
    "colmdriver": 6000,
    "colmdriver_rulebase": 6000,
}
PLANNER_VRAM_DEFAULT_FALLBACK_MIB = 8000
PLANNER_RETRY_LIMIT_DEFAULT = 3
GPU_QUERY_CACHE_TTL_S_DEFAULT = 1.0
MULTI_PLANNER_STATUS_SNAPSHOT_INTERVAL_S_DEFAULT = 2.0
UCLA_V2_CROSSWALK_MAP_TOKEN = "ucla_v2"
UCLA_V2_CROSSWALK_Z_MODE_DEFAULT = "hybrid_raycast"
UCLA_V2_CROSSWALK_TRIM_DEFAULT = 1
UCLA_V2_CROSSWALK_HIGH_Z_PRUNE_DEFAULT = 4
UCLA_V2_CROSSWALK_POLL_SEC_DEFAULT = 2.0
UCLA_V2_CROSSWALK_CACHE_DEFAULT = "v2xpnp/map/ucla_v2_crosswalk_surface_z_cache.json"
SCENARIO_STALL_OUTPUT_TIMEOUT_DEFAULT = 30.0
TICK_HEARTBEAT_FREEZE_TIMEOUT_S = 300.0  # 5 min — kill frozen sim (no tick activity)
SCENARIO_STARTUP_QUIET_TIMEOUT_DEFAULT = 0.0
SCENARIO_RETRY_LIMIT_DEFAULT = 0
STALLED_NEAR_END_MIN_ROUTE_SCORE_DEFAULT = 90.0
STALLED_NEAR_END_LOG_TAIL_BYTES_DEFAULT = 512 * 1024
RESULT_LOG_FALLBACK_TAIL_BYTES_DEFAULT = 2 * 1024 * 1024
MULTI_PLANNER_CHILD_SCENARIO_RETRY_BURST_DEFAULT = 3
SCENARIO_RETRY_HARD_CEILING = 5  # absolute max retries even when --scenario-retry-limit is unlimited
CARLA_DOWN_ABORT_GRACE_S = 15.0
CARLA_TEARDOWN_GATE_TIMEOUT_DEFAULT = 20.0
CARLA_TEARDOWN_GATE_POLL_S_DEFAULT = 0.25
CARLA_TEARDOWN_GATE_STABLE_SAMPLES_DEFAULT = 4
CARLA_ACTIVE_SOCKET_STATES = frozenset(
    {
        "ESTAB",
        "SYN-SENT",
        "SYN-RECV",
        "FIN-WAIT-1",
        "FIN-WAIT-2",
        "CLOSE-WAIT",
        "CLOSING",
        "LAST-ACK",
    }
)
MONITORED_OUTPUT_TAIL_LINES = 80
CARLA_DIAG_MARKER = "[CARLA_DIAG]"
DASHBOARD_STATUS_ENV = "RUN_CUSTOM_EVAL_STATUS_FILE"
CARLA_PROCESS_MATCH_TOKENS = (
    "carlaue4",
    "carlaue4.sh",
    "carlaue4-linux-shipping",
)
VLLM_PROCESS_MATCH_TOKENS = ("vllm", "api_server", "openai.api_server")
GENERIC_SCENARIO_FAILURE_RETRY_ATTEMPTS = 2
RUN_CUSTOM_EVAL_PROCESS_NAME = "run_custom_eval"
_TEARDOWN_GATE_BASELINES_LOCK = threading.Lock()
_TEARDOWN_GATE_BASELINES: Dict[str, int] = {}
_PERF_LOCK = threading.Lock()
_PERF_COUNTERS: "PerformanceCounters | None" = None
_GPU_QUERY_CACHE_LOCK = threading.Lock()
_GPU_QUERY_CACHE: Dict[str, tuple[float, Dict[str, "GPUMemoryStats"]]] = {}
_GPU_QUERY_CACHE_TTL_S = GPU_QUERY_CACHE_TTL_S_DEFAULT


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


@dataclass
class PlannerRequest:
    name: str
    conda_env: str | None = None
    run_id: str | None = None
    scenario_shard: str | None = None  # "INDEX/COUNT" for scenario-level sharding


@dataclass(frozen=True)
class PortBundle:
    port: int
    tm_port: int


@dataclass(frozen=True)
class CarlaRootCauseConfig:
    script_path: Path
    mode: str
    diag_interval: int
    startup_timeout_s: int
    core_wait_seconds: int
    log_root: Path


@dataclass(frozen=True)
class PlannerSlot:
    slot_index: int
    gpu: str | None
    ports: PortBundle


@dataclass(frozen=True)
class MonitoredRunResult:
    returncode: int
    stalled: bool
    elapsed_s: float
    last_output_age_s: float
    output_tail: List[str]
    terminated_reason: str | None = None


@dataclass(frozen=True)
class ColmDriverVLLMReservation:
    scheduler_gpus: Tuple[str, ...]
    vlm_gpu: str
    llm_gpu: str
    selection_source: str


@dataclass(frozen=True)
class GPUMemoryStats:
    index: str
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass
class GPUPlannerState:
    gpu: str | None
    active_count: int = 0
    launch_capacity: int = 1
    telemetry_available: bool = False
    last_stats: GPUMemoryStats | None = None
    last_probe_source: str = "static"
    max_concurrency: int = 1
    oom_penalty_mib: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    last_failure_kind: str = ""
    last_failure_monotonic: float = 0.0
    # MiB that the scheduler has "promised" to the planners currently running
    # or launching on this GPU.  Used to bin-pack more jobs before nvidia-smi
    # catches up to reflect the new allocation.
    reserved_mib: int = 0
    # Peak observed VRAM during the most recent planner lifecycle (per-planner
    # observations are merged into PlannerVramEstimator).
    last_baseline_mib: int | None = None
    # Tracks which run_ids are currently occupying this GPU so the peak
    # observation loop can attribute usage.
    active_run_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class PlannerThreadResult:
    planner_request: PlannerRequest
    slot_index: int
    gpu: str | None
    ports: PortBundle
    returncode: int | None = None
    error: str | None = None
    failure_kind: str = ""


@dataclass
class PlannerDashboardState:
    planner_request: PlannerRequest
    results_dir: Path
    log_path: Path
    status_path: Path
    default_results_subdir: str = ""
    phase: str = "queued"
    slot_index: int | None = None
    gpu: str | None = None
    ports: PortBundle | None = None
    runner: "MonitoredSubprocessRunner | None" = None
    start_time: float | None = None
    end_time: float | None = None
    returncode: int | None = None
    error: str | None = None
    route_progress: Tuple[int, int] = (0, 1)
    entry_status: str = ""
    last_output_age_s: float | None = None
    ego_route_scores: Dict[int, float] = field(default_factory=dict)
    ego_status: Dict[int, str] = field(default_factory=dict)
    carla_up: bool | None = None
    tm_up: bool | None = None
    scenario_index: int = 0
    scenario_total: int = 1
    completed_scenarios: int = 0
    current_scenario: str = ""
    current_scenario_results_subdir: str = ""
    current_attempt: int | None = None
    attempt_limit: int | None = None


@dataclass
class ResultRootProgress:
    route_progress: Tuple[int, int] = (0, 1)
    entry_status: str = ""
    ego_route_scores: Dict[int, float] = field(default_factory=dict)
    ego_status: Dict[int, str] = field(default_factory=dict)
    ego_record_counts: Dict[int, int] = field(default_factory=dict)
    has_structured_records: bool = False
    route_recovered: bool = False
    recovery_count: int = 0
    planner_continuity_modes: List[str] = field(default_factory=list)
    recovered_approximate: bool = False
    checkpoint_id: int | None = None
    recovery_reason: str = ""
    recovery_failure_kind: str = ""
    invalid_for_benchmark: bool = False
    benchmark_eligible_candidate: bool = True
    terminal_state: str = ""
    started: bool = False
    completed: bool = False


@dataclass
class PerformanceCounters:
    nvidia_smi_calls: int = 0
    nvidia_smi_time_s: float = 0.0
    gpu_settle_calls: int = 0
    gpu_settle_time_s: float = 0.0
    gpu_settle_timeouts: int = 0
    teardown_gate_calls: int = 0
    teardown_gate_time_s: float = 0.0
    teardown_gate_timeouts: int = 0
    carla_start_calls: int = 0
    carla_start_time_s: float = 0.0
    carla_restart_calls: int = 0
    carla_restart_time_s: float = 0.0
    child_runs: int = 0
    child_run_time_s: float = 0.0
    child_output_lines: int = 0
    child_diag_lines: int = 0
    retry_events: int = 0
    retries_by_kind: Dict[str, int] = field(default_factory=dict)
    retries_by_outcome: Dict[str, int] = field(default_factory=dict)


_PERF_COUNTERS = PerformanceCounters()


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RESULT_SECTION_RE = re.compile(
    r"Results of RouteScenario.*?------\s*(SUCCESS|FAILURE)\s*=+",
    re.IGNORECASE,
)
ROUTE_COMPLETION_RE = re.compile(
    r"RouteCompletionTest.*?([0-9]+(?:\.[0-9]+)?)\s*%",
    re.IGNORECASE,
)


PLANNER_SPECS: dict[str, PlannerSpec] = {
    "autopilot": PlannerSpec(
        agent="simulation/leaderboard/team_code/auto_pilot.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
    ),
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
    "log-replay": PlannerSpec(
        # Minimal agent that returns neutral controls; actual movement is handled by
        # LogReplayFollower in route_scenario.py when CUSTOM_EGO_LOG_REPLAY=1 is set
        agent="simulation/leaderboard/team_code/logreplay_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/colmdriver_config.yaml",
    ),
    "minimal": PlannerSpec(
        agent="simulation/leaderboard/team_code/minimal_agent.py",
        agent_config="simulation/leaderboard/team_code/agent_config/example_config.yaml",
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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
        "--results-subdir",
        help=(
            "Optional subfolder appended under results tag. Useful for multi-scenario runs "
            "to keep a shared results tag with per-scenario subfolders."
        ),
    )
    parser.add_argument(
        "--routes-leaf",
        default=None,
        help=(
            "Optional leaf subfolder name to locate routes inside each scenario folder. "
            "If set, will look for <routes-dir>/*/<routes-leaf> before falling back to <routes-dir>/*."
        ),
    )
    parser.add_argument(
        "--ego-num",
        type=int,
        help="Override the number of ego vehicles (auto-detected otherwise).",
    )
    parser.add_argument("--port", type=int, default=2014, help="CARLA server port.")
    parser.add_argument(
        "--traffic-manager-port",
        type=int,
        dest="tm_port",
        help="Traffic Manager port (defaults to port + 5).",
    )
    parser.add_argument(
        "--start-carla",
        action="store_true",
        help="Launch a local CARLA server from CARLA_ROOT/carla912 before evaluation.",
    )
    parser.add_argument(
        "--carla-rootcause",
        action="store_true",
        help=(
            "When --start-carla is enabled, launch CARLA through "
            "tools/run_carla_rootcause_capture.sh so server lifecycle/logs use root-cause capture "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-script",
        default="tools/run_carla_rootcause_capture.sh",
        help=(
            "Path to CARLA root-cause capture wrapper script "
            "(default: tools/run_carla_rootcause_capture.sh)."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-mode",
        choices=("auto", "gdb", "core"),
        default=CARLA_ROOTCAUSE_MODE_DEFAULT,
        help=(
            "Capture mode passed to the root-cause wrapper when --carla-rootcause is enabled "
            f"(default: {CARLA_ROOTCAUSE_MODE_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-diag-interval",
        type=int,
        default=CARLA_ROOTCAUSE_DIAG_INTERVAL_DEFAULT,
        help=(
            "Diagnostics polling interval (seconds) for root-cause CARLA launches "
            f"(default: {CARLA_ROOTCAUSE_DIAG_INTERVAL_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-startup-timeout",
        type=int,
        default=CARLA_ROOTCAUSE_STARTUP_TIMEOUT_DEFAULT,
        help=(
            "Seconds to wait for CARLA child PID ownership during root-cause launch "
            f"(default: {CARLA_ROOTCAUSE_STARTUP_TIMEOUT_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-core-wait-seconds",
        type=int,
        default=CARLA_ROOTCAUSE_CORE_WAIT_SECONDS_DEFAULT,
        help=(
            "Seconds to wait for a core file after CARLA exits in root-cause core mode "
            f"(default: {CARLA_ROOTCAUSE_CORE_WAIT_SECONDS_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--carla-rootcause-log-root",
        default=None,
        help=(
            "Directory where per-launch root-cause log directories are created when "
            "--carla-rootcause is enabled (default: repository root)."
        ),
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        help="Extra arguments to pass to CarlaUE4.sh (repeatable).",
    )
    parser.add_argument(
        "--carla-port-tries",
        type=int,
        default=CARLA_PORT_TRIES,
        help="How many ports to try if the desired CARLA port is already in use.",
    )
    parser.add_argument(
        "--carla-port-step",
        type=int,
        default=CARLA_PORT_STEP,
        help="Port increment to use when searching for a free CARLA port.",
    )
    parser.add_argument(
        "--carla-crash-multiplier",
        type=float,
        default=CARLA_CRASH_MULTIPLIER,
        help=(
            "If a scenario runs longer than timeout * multiplier (and other factors), "
            "assume CARLA has crashed and restart it (multi-scenario runs only)."
        ),
    )
    parser.add_argument(
        "--carla-restart-wait",
        type=float,
        default=CARLA_RESTART_WAIT,
        help="Seconds to wait after restarting CARLA before starting the next scenario.",
    )
    parser.add_argument(
        "--carla-post-start-buffer",
        type=float,
        default=CARLA_POST_START_BUFFER_DEFAULT,
        help=(
            "Extra settle time in seconds after managed CARLA reports its port is open and before "
            "launching the evaluator."
        ),
    )
    parser.add_argument(
        "--carla-forensics",
        action="store_true",
        help=(
            "Enable crash-forensics defaults (CARLA stream debug counters, glibc malloc checks, "
            "core-dump soft-limit attempt, and extra CARLA logging flags when --start-carla is used) "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--carla-stream-debug-summary-every",
        type=int,
        default=CARLA_STREAM_DEBUG_SUMMARY_EVERY_DEFAULT,
        help=(
            "When --carla-forensics is enabled, emit stream debug summaries every N dispatcher events "
            f"(default: {CARLA_STREAM_DEBUG_SUMMARY_EVERY_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--carla-forensics-log-file",
        default=None,
        help=(
            "Optional file path to append forensics runtime messages from this launcher "
            "(in addition to printing to stdout)."
        ),
    )
    parser.add_argument(
        "--carla-forensics-server-log-file",
        default=None,
        help=(
            "Optional file path for CARLA server stdout/stderr when --start-carla is enabled. "
            "Useful for capturing stream/malloc diagnostics."
        ),
    )
    parser.add_argument(
        "--inject-ucla-v2-crosswalk",
        dest="inject_ucla_v2_crosswalk",
        action="store_true",
        help="Automatically draw the UCLA v2 debug crosswalk overlay while evaluator runs.",
    )
    parser.add_argument(
        "--no-inject-ucla-v2-crosswalk",
        dest="inject_ucla_v2_crosswalk",
        action="store_false",
        help="Disable automatic UCLA v2 debug crosswalk overlay injection.",
    )
    parser.add_argument(
        "--ucla-v2-crosswalk-cache-file",
        default=UCLA_V2_CROSSWALK_CACHE_DEFAULT,
        help="Path to persistent UCLA v2 crosswalk surface-z cache JSON.",
    )
    parser.add_argument(
        "--ucla-v2-crosswalk-trim-outermost-per-side",
        type=int,
        default=UCLA_V2_CROSSWALK_TRIM_DEFAULT,
        help="Outer stripes removed per side for UCLA v2 crosswalk overlay.",
    )
    parser.add_argument(
        "--ucla-v2-crosswalk-high-z-prune-count",
        type=int,
        default=UCLA_V2_CROSSWALK_HIGH_Z_PRUNE_DEFAULT,
        help="Highest-z stripe count pruned for UCLA v2 crosswalk overlay.",
    )
    parser.add_argument(
        "--ucla-v2-crosswalk-z-mode",
        default=UCLA_V2_CROSSWALK_Z_MODE_DEFAULT,
        choices=("hybrid_raycast", "raycast", "ground_projection", "waypoint"),
        help="Z sampling mode for UCLA v2 crosswalk overlay.",
    )
    parser.add_argument(
        "--ucla-v2-crosswalk-poll-sec",
        type=float,
        default=UCLA_V2_CROSSWALK_POLL_SEC_DEFAULT,
        help="Polling interval (seconds) for UCLA v2 overlay worker.",
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
        action="append",
        default=[],
        help=(
            "Planner preset to run. Repeat to launch multiple planners. Each value can be "
            "<planner> or [<planner>,<conda_env>]. "
            f"Available planners: {', '.join(sorted(PLANNER_SPECS.keys()))}. "
            f"Defaults to {DEFAULT_PLANNER} if not set."
        ),
    )
    parser.add_argument(
        "--gpu",
        "--gpus",
        dest="gpus",
        action="append",
        default=[],
        help=(
            "GPU ids for planner runs. Repeat the flag or pass a comma-separated list. "
            "With multiple planners, runs are distributed across these GPU slots."
        ),
    )
    parser.add_argument(
        "--auto-workers-per-gpu",
        dest="auto_workers_per_gpu",
        action="store_true",
        default=True,
        help=(
            "Use live GPU free-memory probes to launch more than one planner on the same GPU "
            "when headroom remains after services/processes settle."
        ),
    )
    parser.add_argument(
        "--no-auto-workers-per-gpu",
        dest="auto_workers_per_gpu",
        action="store_false",
        help="Disable adaptive per-GPU planner concurrency and run at most one planner per GPU.",
    )
    parser.add_argument(
        "--max-multiprocess",
        action="store_true",
        default=False,
        help=(
            "Enable maximal safe multiprocessing. Aggressively packs evaluations onto GPUs "
            "up to the VRAM limit with reduced safety margins. Implies "
            "--allow-multiple-carlas-per-gpu. Lowers --vram-safety-buffer-mib and "
            "--auto-workers-min-free-mib to aggressive defaults for maximum concurrency."
        ),
    )
    parser.add_argument(
        "--scenario-shard",
        type=str,
        default=None,
        metavar="INDEX/COUNT",
        help=(
            "Run only a subset of scenarios.  INDEX/COUNT means this worker "
            "handles scenarios where (scenario_index %% COUNT) == INDEX.  "
            "E.g. --scenario-shard 0/4 runs scenarios 0, 4, 8, … "
            "Used internally by --max-multiprocess for scenario-level parallelism."
        ),
    )
    parser.add_argument(
        "--allow-multiple-carlas-per-gpu",
        action="store_true",
        help=(
            "Permit more than one managed CARLA/planner run on the same GPU in multi-planner mode. "
            "By default, --start-carla keeps this at one per GPU for stability."
        ),
    )
    parser.add_argument(
        "--auto-workers-min-free-mib",
        type=int,
        default=AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT,
        help=(
            "Minimum free VRAM (MiB) that must remain after a settle probe before another planner "
            f"is launched on that GPU (default: {AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--auto-workers-settle-timeout",
        type=float,
        default=AUTO_WORKERS_PER_GPU_SETTLE_TIMEOUT_DEFAULT,
        help=(
            "Seconds to wait for GPU memory usage to settle after starting shared services or a "
            f"planner child (default: {AUTO_WORKERS_PER_GPU_SETTLE_TIMEOUT_DEFAULT:.0f})."
        ),
    )
    parser.add_argument(
        "--auto-workers-poll-seconds",
        type=float,
        default=AUTO_WORKERS_PER_GPU_POLL_SECONDS_DEFAULT,
        help=(
            "Polling interval in seconds for adaptive GPU free-memory probes "
            f"(default: {AUTO_WORKERS_PER_GPU_POLL_SECONDS_DEFAULT:.1f})."
        ),
    )
    parser.add_argument(
        "--auto-workers-stable-samples",
        type=int,
        default=AUTO_WORKERS_PER_GPU_STABLE_SAMPLES_DEFAULT,
        help=(
            "Number of consecutive GPU memory samples within tolerance required to treat memory "
            f"usage as settled (default: {AUTO_WORKERS_PER_GPU_STABLE_SAMPLES_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--auto-workers-stable-tolerance-mib",
        type=int,
        default=AUTO_WORKERS_PER_GPU_STABLE_TOLERANCE_MIB_DEFAULT,
        help=(
            "Allowed change in free VRAM (MiB) between consecutive settle samples "
            f"(default: {AUTO_WORKERS_PER_GPU_STABLE_TOLERANCE_MIB_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--auto-workers-max-per-gpu",
        type=int,
        default=AUTO_WORKERS_MAX_PER_GPU_DEFAULT,
        help=(
            "Upper bound on concurrent planner workers per GPU when auto-workers are enabled "
            f"(default: {AUTO_WORKERS_MAX_PER_GPU_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--oom-concurrency-backoff-step",
        type=int,
        default=OOM_CONCURRENCY_BACKOFF_STEP_DEFAULT,
        help=(
            "How many worker slots to remove from a GPU after an OOM-class failure "
            f"(default: {OOM_CONCURRENCY_BACKOFF_STEP_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--oom-free-mib-penalty-step",
        type=int,
        default=OOM_FREE_MIB_PENALTY_STEP_DEFAULT,
        help=(
            "Extra free-VRAM requirement (MiB) added per OOM on the same GPU "
            f"(default: {OOM_FREE_MIB_PENALTY_STEP_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--oom-recovery-success-streak",
        type=int,
        default=OOM_RECOVERY_SUCCESS_STREAK_DEFAULT,
        help=(
            "Number of consecutive successful completions on a GPU before relaxing OOM penalties "
            f"(default: {OOM_RECOVERY_SUCCESS_STREAK_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--oom-recovery-penalty-decay-step",
        type=int,
        default=OOM_RECOVERY_PENALTY_DECAY_STEP_DEFAULT,
        help=(
            "How much OOM VRAM penalty (MiB) to remove after each recovery streak "
            f"(default: {OOM_RECOVERY_PENALTY_DECAY_STEP_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--oom-backoff-decay-seconds",
        type=float,
        default=OOM_BACKOFF_DECAY_SECONDS_DEFAULT,
        help=(
            "Seconds after which a stale OOM penalty on an idle GPU is fully forgotten "
            f"(default: {OOM_BACKOFF_DECAY_SECONDS_DEFAULT:.0f}s). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--vram-aware-scheduler",
        dest="vram_aware_scheduler",
        action="store_true",
        default=True,
        help=(
            "Use the dynamic, bin-packing, VRAM-aware multi-planner scheduler. "
            "This is the default; disable with --no-vram-aware-scheduler for the "
            "legacy one-planner-per-GPU code path."
        ),
    )
    parser.add_argument(
        "--no-vram-aware-scheduler",
        dest="vram_aware_scheduler",
        action="store_false",
        help="Disable the dynamic VRAM-aware scheduler (fall back to legacy policy).",
    )
    parser.add_argument(
        "--planner-vram-estimate-cache",
        default=VRAM_ESTIMATE_CACHE_DEFAULT,
        help=(
            "Path to persistent per-planner VRAM estimate cache (JSON). "
            "Observations from each run are merged back into this file. "
            f"Default: {VRAM_ESTIMATE_CACHE_DEFAULT}"
        ),
    )
    parser.add_argument(
        "--planner-vram-estimate-mib",
        action="append",
        default=[],
        metavar="NAME=MIB",
        help=(
            "Override a planner's VRAM estimate for this run (repeatable). "
            "Example: --planner-vram-estimate-mib uniad=14000"
        ),
    )
    parser.add_argument(
        "--carla-vram-estimate-mib",
        type=int,
        default=CARLA_FIXED_VRAM_MIB_DEFAULT,
        help=(
            "Estimated VRAM (MiB) that a single managed CARLA server consumes on a GPU. "
            "Subtracted from per-GPU headroom when the first planner on that GPU launches "
            f"(default: {CARLA_FIXED_VRAM_MIB_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--vram-safety-buffer-mib",
        type=int,
        default=VRAM_SAFETY_BUFFER_MIB_DEFAULT,
        help=(
            "Extra VRAM safety margin (MiB) held back on every GPU on top of "
            "--auto-workers-min-free-mib. Raises this floor to reduce OOM risk "
            f"(default: {VRAM_SAFETY_BUFFER_MIB_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--gpu-monitor-poll-seconds",
        type=float,
        default=GPU_MONITOR_POLL_SECONDS_DEFAULT,
        help=(
            "Background GPU-memory monitor polling interval (seconds). Replaces the "
            "inline blocking nvidia-smi settle waits in the scheduler hot path "
            f"(default: {GPU_MONITOR_POLL_SECONDS_DEFAULT:.1f}s)."
        ),
    )
    parser.add_argument(
        "--scheduler-tick-log",
        default=None,
        help=(
            "Optional path to write per-tick JSONL scheduler decisions and snapshots. "
            "Intended for verifying that GPUs are saturated and jobs aren't stalling."
        ),
    )
    parser.add_argument(
        "--scheduler-tick-interval",
        type=float,
        default=SCHEDULER_TICK_POLL_TIMEOUT_S_DEFAULT,
        help=(
            "Maximum idle time (seconds) the scheduler spends waiting for completions "
            "before re-evaluating the packing plan. Lower values react faster to freed "
            f"VRAM (default: {SCHEDULER_TICK_POLL_TIMEOUT_S_DEFAULT:.2f}s)."
        ),
    )
    parser.add_argument(
        "--gpu-query-cache-ttl",
        type=float,
        default=GPU_QUERY_CACHE_TTL_S_DEFAULT,
        help=(
            "Reuse nvidia-smi samples younger than this many seconds to reduce probing overhead "
            f"(default: {GPU_QUERY_CACHE_TTL_S_DEFAULT:.1f}s)."
        ),
    )
    parser.add_argument(
        "--planner-status-snapshot-interval",
        type=float,
        default=MULTI_PLANNER_STATUS_SNAPSHOT_INTERVAL_S_DEFAULT,
        help=(
            "Minimum interval in seconds between planner child status-file reads in the dashboard "
            f"(default: {MULTI_PLANNER_STATUS_SNAPSHOT_INTERVAL_S_DEFAULT:.1f}s)."
        ),
    )
    parser.add_argument(
        "--planner-dashboard",
        dest="planner_dashboard",
        action="store_true",
        default=True,
        help=(
            "In multi-planner mode, mute child stdout and render a live tqdm dashboard with "
            "planner, ego, GPU, and CARLA status."
        ),
    )
    parser.add_argument(
        "--no-planner-dashboard",
        dest="planner_dashboard",
        action="store_false",
        help="Disable the multi-planner tqdm dashboard and stream planner stdout normally.",
    )
    parser.add_argument(
        "--planner-dashboard-refresh-seconds",
        type=float,
        default=1.0,
        help="Refresh interval in seconds for the multi-planner dashboard.",
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
    parser.add_argument(
        "--repro-reruns",
        "--rerun-same-scenario",
        dest="repro_reruns",
        type=int,
        default=1,
        help=(
            "Launcher-level reruns of the same planner/routes invocation for reproduction. "
            "Unlike --repetitions, this reruns the full run_custom_eval harness as separate "
            "attempts with distinct result subdirectories."
        ),
    )
    parser.add_argument(
        "--repro-rerun-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repro-rerun-total",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--track", default="SENSORS", help="Leaderboard track name.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Route timeout.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA server host passed to leaderboard evaluator (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--capture-actor-images",
        action="store_true",
        help="Save a snapshot image for each spawned custom actor.",
    )
    parser.add_argument(
        "--normalize-actor-z",
        default=True,
        action="store_true",
        help="Adjust custom actor Z to nearest ground height without changing x/y or rotation.",
    )
    parser.add_argument(
        "--spawn-retry-xy-offsets",
        default="0.0,0.25,-0.25,0.5,-0.5,1.0,-1.0",
        help=(
            "Comma-separated XY offsets (meters) used for custom-actor spawn retries. "
            "All pairwise (dx,dy) combinations are attempted in increasing offset cost."
        ),
    )
    parser.add_argument(
        "--spawn-retry-z-offsets",
        default="0.0,0.2,-0.2,0.5,-0.5,1.0",
        help=(
            "Comma-separated Z offsets (meters) used with XY retry combinations for "
            "custom-actor spawn retries."
        ),
    )
    parser.add_argument(
        "--spawn-retry-max-attempts",
        type=int,
        default=180,
        help="Maximum number of custom-actor spawn retry offset candidates per actor.",
    )
    parser.add_argument(
        "--image-save-interval",
        type=int,
        default=None,
        help="Frames between saved images for TCP/log-replay (env: TCP_SAVE_INTERVAL).",
    )
    parser.add_argument(
        "--capture-every-sensor-frame",
        action="store_true",
        help=(
            "Save RGB/BEV images directly from camera callbacks for every sensor frame "
            "(decoupled from agent-step save cadence)."
        ),
    )
    parser.add_argument(
        "--capture-logreplay-images",
        dest="capture_logreplay_images",
        action="store_true",
        help=(
            "Save extra log-replay camera images into logreplayimages/ at the same cadence "
            "as normal agent image saves (controlled by --image-save-interval)."
        ),
    )
    parser.add_argument(
        "--capture-calibration-at-save-interval",
        dest="capture_logreplay_images",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--capture-overhead-per-tick",
        action="store_true",
        help=(
            "Deprecated no-op. Overhead per-tick capture has been removed; "
            "flag is accepted for compatibility."
        ),
    )
    parser.add_argument(
        "--overhead-save-every-n",
        type=int,
        default=1,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-draw-grp-boxes",
        dest="overhead_draw_grp_boxes",
        action="store_true",
        default=True,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--no-overhead-draw-grp-boxes",
        dest="overhead_draw_grp_boxes",
        action="store_false",
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-output-subdir",
        default="overhead_tick",
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-image-width",
        type=int,
        default=2048,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-image-height",
        type=int,
        default=2048,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-image-fov",
        type=float,
        default=90.0,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-image-margin-m",
        type=float,
        default=30.0,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-image-z-padding",
        type=float,
        default=20.0,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-min-camera-z",
        type=float,
        default=80.0,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-size-xy",
        type=float,
        default=0.8,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-height",
        type=float,
        default=0.6,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-z-offset",
        type=float,
        default=0.2,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-thickness",
        type=float,
        default=0.08,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-life-time",
        type=float,
        default=600.0,
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-color-regular-rgb",
        default="255,40,40",
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-color-corrected-rgb",
        default="255,255,0",
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--overhead-box-color-sanitized-rgb",
        default="40,255,40",
        help="Deprecated no-op. Accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--rgb-width",
        type=int,
        default=None,
        help="Override TCP front RGB width (env: TCP_RGB_WIDTH).",
    )
    parser.add_argument(
        "--rgb-height",
        type=int,
        default=None,
        help="Override TCP front RGB height (env: TCP_RGB_HEIGHT).",
    )
    parser.add_argument(
        "--rgb-fov",
        type=float,
        default=None,
        help="Override TCP front RGB FOV (env: TCP_RGB_FOV).",
    )
    parser.add_argument(
        "--rgb-x",
        type=float,
        default=None,
        help="Override TCP front RGB camera X (env: TCP_RGB_X).",
    )
    parser.add_argument(
        "--rgb-y",
        type=float,
        default=None,
        help="Override TCP front RGB camera Y (env: TCP_RGB_Y).",
    )
    parser.add_argument(
        "--rgb-z",
        type=float,
        default=None,
        help="Override TCP front RGB camera Z (env: TCP_RGB_Z).",
    )
    parser.add_argument(
        "--rgb-roll",
        type=float,
        default=None,
        help="Override TCP front RGB camera roll (env: TCP_RGB_ROLL).",
    )
    parser.add_argument(
        "--rgb-pitch",
        type=float,
        default=None,
        help="Override TCP front RGB camera pitch (env: TCP_RGB_PITCH).",
    )
    parser.add_argument(
        "--rgb-yaw",
        type=float,
        default=None,
        help="Override TCP front RGB camera yaw (env: TCP_RGB_YAW).",
    )
    parser.add_argument(
        "--logreplay-rgb-width",
        type=int,
        default=None,
        help="Override logreplayimages RGB width (env: TCP_LOGREPLAY_RGB_WIDTH).",
    )
    parser.add_argument(
        "--logreplay-rgb-height",
        type=int,
        default=None,
        help="Override logreplayimages RGB height (env: TCP_LOGREPLAY_RGB_HEIGHT).",
    )
    parser.add_argument(
        "--logreplay-rgb-fov",
        type=float,
        default=None,
        help="Override logreplayimages RGB FOV (env: TCP_LOGREPLAY_RGB_FOV).",
    )
    parser.add_argument(
        "--logreplay-rgb-x",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera X (env: TCP_LOGREPLAY_RGB_X).",
    )
    parser.add_argument(
        "--logreplay-rgb-y",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera Y (env: TCP_LOGREPLAY_RGB_Y).",
    )
    parser.add_argument(
        "--logreplay-rgb-z",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera Z (env: TCP_LOGREPLAY_RGB_Z).",
    )
    parser.add_argument(
        "--logreplay-rgb-roll",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera roll (env: TCP_LOGREPLAY_RGB_ROLL).",
    )
    parser.add_argument(
        "--logreplay-rgb-pitch",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera pitch (env: TCP_LOGREPLAY_RGB_PITCH).",
    )
    parser.add_argument(
        "--logreplay-rgb-yaw",
        type=float,
        default=None,
        help="Override logreplayimages RGB camera yaw (env: TCP_LOGREPLAY_RGB_YAW).",
    )
    parser.add_argument(
        "--bev-width",
        type=int,
        default=None,
        help="Override TCP BEV camera width (env: TCP_BEV_WIDTH).",
    )
    parser.add_argument(
        "--bev-height",
        type=int,
        default=None,
        help="Override TCP BEV camera height (env: TCP_BEV_HEIGHT).",
    )
    parser.add_argument(
        "--bev-fov",
        type=float,
        default=None,
        help="Override TCP BEV camera FOV (env: TCP_BEV_FOV).",
    )
    parser.add_argument(
        "--log-replay-actors",
        action="store_true",
        help="Replay custom actors using per-waypoint timing if present in XML.",
    )
    parser.add_argument(
        "--custom-actor-control-mode",
        choices=("policy", "replay"),
        default="policy",
        help=(
            "Control mode for custom actors. "
            "'policy' uses regular controller behavior (WaypointFollower). "
            "'replay' uses transform/timing log replay."
        ),
    )
    parser.add_argument(
        "--npc-only-fake-ego",
        action="store_true",
        help=(
            "No-ego mode: keep ego XML files as planner-compatible artifacts, but run this evaluation "
            "with ego trajectories injected as custom NPC actors and capture images from those fake-ego actors."
        ),
    )
    parser.add_argument(
        "--fake-ego-control-mode",
        choices=("policy", "replay"),
        default=None,
        help=(
            "Control mode used specifically for fake-ego actors when --npc-only-fake-ego is enabled. "
            "If omitted, fake-egos use --custom-actor-control-mode (same as regular NPCs)."
        ),
    )
    parser.add_argument(
        "--smooth-log-replay-vehicles",
        dest="smooth_log_replay_vehicles",
        action="store_true",
        default=None,
        help="Enable adaptive smoothing for logged NPC vehicle transforms.",
    )
    parser.add_argument(
        "--no-smooth-log-replay-vehicles",
        dest="smooth_log_replay_vehicles",
        action="store_false",
        help="Disable adaptive smoothing for logged NPC vehicle transforms.",
    )
    parser.add_argument(
        "--log-replay-smooth-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle x/y smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle x/y smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-yaw-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle yaw smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_YAW_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-yaw-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle yaw smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_YAW_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-z-min-cutoff",
        type=float,
        default=None,
        help="One-Euro min cutoff for vehicle z smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_Z_MIN_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-smooth-z-beta",
        type=float,
        default=None,
        help="One-Euro beta for vehicle z smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_Z_BETA).",
    )
    parser.add_argument(
        "--log-replay-smooth-d-cutoff",
        type=float,
        default=None,
        help="One-Euro derivative cutoff for vehicle smoothing (env: CUSTOM_LOG_REPLAY_SMOOTH_D_CUTOFF).",
    )
    parser.add_argument(
        "--log-replay-lateral-damping",
        type=float,
        default=None,
        help="Lateral jitter damping strength in [0,1] (env: CUSTOM_LOG_REPLAY_LATERAL_DAMPING).",
    )
    parser.add_argument(
        "--log-replay-lateral-passes",
        type=int,
        default=None,
        help="Number of lateral jitter suppression passes (env: CUSTOM_LOG_REPLAY_LATERAL_PASSES).",
    )
    parser.add_argument(
        "--log-replay-lateral-max-correction",
        type=float,
        default=None,
        help="Max per-pass lateral correction in meters (env: CUSTOM_LOG_REPLAY_LATERAL_MAX_CORRECTION).",
    )
    parser.add_argument(
        "--log-replay-lateral-turn-keep",
        type=float,
        default=None,
        help="Keep factor in turns in [0,1] (env: CUSTOM_LOG_REPLAY_LATERAL_TURN_KEEP).",
    )
    parser.add_argument(
        "--log-replay-lateral-turn-angle-deg",
        type=float,
        default=None,
        help="Turn-angle threshold in degrees (env: CUSTOM_LOG_REPLAY_LATERAL_TURN_ANGLE_DEG).",
    )
    parser.add_argument(
        "--log-replay-time-lead",
        type=float,
        default=None,
        help=(
            "Replay time lead in seconds to compensate one-tick sensor/update lag "
            "(env: CUSTOM_LOG_REPLAY_TIME_LEAD)."
        ),
    )
    parser.add_argument(
        "--stabilize-static-log-replay-vehicles",
        dest="stabilize_static_log_replay_vehicles",
        action="store_true",
        default=None,
        help="Freeze near-static replay vehicle segments to remove parked-car jitter.",
    )
    parser.add_argument(
        "--no-stabilize-static-log-replay-vehicles",
        dest="stabilize_static_log_replay_vehicles",
        action="store_false",
        help="Disable near-static replay vehicle segment stabilization.",
    )
    parser.add_argument(
        "--static-log-replay-total-disp",
        type=float,
        default=None,
        help="Global static hold threshold for start-end XY displacement in meters.",
    )
    parser.add_argument(
        "--static-log-replay-total-path",
        type=float,
        default=None,
        help="Global static hold threshold for cumulative XY path length in meters.",
    )
    parser.add_argument(
        "--static-log-replay-window-min-duration",
        type=float,
        default=None,
        help="Minimum near-static segment duration in seconds before freezing.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-disp",
        type=float,
        default=None,
        help="Maximum net XY displacement (meters) for a near-static segment window.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-speed",
        type=float,
        default=None,
        help="Maximum XY speed (m/s) for near-static segment detection.",
    )
    parser.add_argument(
        "--static-log-replay-window-max-yaw-delta",
        type=float,
        default=None,
        help="Maximum per-step yaw delta (deg) for near-static segment detection.",
    )
    parser.add_argument(
        "--animate-walkers",
        dest="animate_walkers",
        action="store_true",
        default=None,
        help="Animate walker replay via WalkerControl instead of teleport-only transforms.",
    )
    parser.add_argument(
        "--no-animate-walkers",
        dest="animate_walkers",
        action="store_false",
        help="Disable walker replay animation and use transform replay only.",
    )
    parser.add_argument(
        "--walker-max-speed",
        type=float,
        default=None,
        help="Max walker replay speed in m/s (env: CUSTOM_LOG_REPLAY_WALKER_MAX_SPEED).",
    )
    parser.add_argument(
        "--walker-teleport-distance",
        type=float,
        default=None,
        help=(
            "Distance threshold in meters where walker replay snaps to target instead of control "
            "(env: CUSTOM_LOG_REPLAY_WALKER_TELEPORT_DIST)."
        ),
    )
    parser.add_argument(
        "--walker-ground-lift",
        type=float,
        default=None,
        help="Extra z offset applied after ground snap for walkers (env: CUSTOM_WALKER_GROUND_LIFT).",
    )
    parser.add_argument(
        "--normalize-ego-z",
        default=True,
        action="store_true",
        help="Snap ego vehicle Z to nearest driving road height before spawning.",
    )
    parser.add_argument(
        "--align-ego-routes",
        action="store_true",
        help=(
            "Align ego route waypoints to the CARLA map using the scenario_generator "
            "route alignment module (uses CARLA GRP to snap headings/lanes)."
        ),
    )
    parser.add_argument(
        "--align-ego-host",
        default="127.0.0.1",
        help="CARLA host for ego route alignment (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--align-ego-port",
        type=int,
        default=None,
        help="CARLA port for ego route alignment (default: --port).",
    )
    parser.add_argument(
        "--align-ego-sampling-resolution",
        type=float,
        default=2.0,
        help="GRP sampling resolution (meters) used for ego route alignment.",
    )
    parser.add_argument(
        "--llm-port",
        type=int,
        default=8888,
        help="Port for LLM server (CoLMDriver only, default: 8888).",
    )
    parser.add_argument(
        "--vlm-port",
        type=int,
        default=1111,
        help="Port for VLM server (CoLMDriver only, default: 1111).",
    )
    parser.add_argument(
        "--start-colmdriver-vllm",
        action="store_true",
        help=(
            "When a CoLMDriver planner is requested, start the shared CoLMDriver VLM/LLM "
            "vLLM servers automatically before evaluation."
        ),
    )
    parser.add_argument(
        "--colmdriver-vllm-env",
        default=COLMDRIVER_VLLM_ENV_DEFAULT,
        help=(
            "Conda environment containing the `vllm` CLI for CoLMDriver server startup "
            f"(default: {COLMDRIVER_VLLM_ENV_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--colmdriver-vllm-startup-timeout",
        type=float,
        default=COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT,
        help=(
            "Seconds to wait for the auto-started CoLMDriver VLM/LLM ports to come up "
            f"(default: {COLMDRIVER_VLLM_STARTUP_TIMEOUT_DEFAULT:.0f})."
        ),
    )
    parser.add_argument(
        "--defer-colmdriver-vllm-start",
        dest="defer_colmdriver_vllm_start",
        action="store_true",
        default=True,
        help=(
            "In mixed multi-planner batches, queue CoLMDriver planners after non-CoLMDriver "
            "planners and delay CoLMDriver vLLM startup until that deferred phase begins."
        ),
    )
    parser.add_argument(
        "--no-defer-colmdriver-vllm-start",
        dest="defer_colmdriver_vllm_start",
        action="store_false",
        help="Disable deferred CoLMDriver phase scheduling and start shared vLLM services immediately.",
    )
    parser.add_argument(
        "--show-carla-diag",
        action="store_true",
        help=(
            "Print CARLA diagnostic lines tagged with [CARLA_DIAG]. "
            "By default they are muted from stdout but still remain in service/log files."
        ),
    )
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
        "--route-plots-only",
        action="store_true",
        help=(
            "Generate route plots from the exact evaluator/RouteScenario path only "
            "(skip scenario driving execution)."
        ),
    )
    parser.add_argument(
        "--disable-stall-detection",
        dest="disable_stall_detection",
        default=True,
        action="store_true",
        help=(
            "Disable fallback position-based stall detection in ScenarioManager "
            "(prevents auto-terminating routes when all egos appear stationary). "
            "This is the default behavior."
        ),
    )
    parser.add_argument(
        "--enable-stall-detection",
        dest="disable_stall_detection",
        action="store_false",
        help=(
            "Re-enable ScenarioManager position-based stall detection "
            "(overrides the default of stall detection being disabled)."
        ),
    )
    parser.add_argument(
        "--grp-postprocess-mode",
        choices=("none", "seam", "kink", "legacy"),
        default=None,
        help=(
            "Override CARLA GRP postprocess mode for route interpolation. "
            "'none' disables postprocessing (CARLA_GRP_PP_ENABLE=0); "
            "other modes force CARLA_GRP_PP_ENABLE=1 and set CARLA_GRP_PP_MODE. "
            "If omitted, keep existing environment/default behavior."
        ),
    )
    parser.add_argument(
        "--grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_true",
        default=True,
        help=(
            "Set CARLA_GRP_PP_IGNORE_ENDPOINTS=1 so GRP postprocessing excludes "
            "the first/last route nodes from consideration (default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-grp-postprocess-ignore-endpoints",
        dest="grp_postprocess_ignore_endpoints",
        action="store_false",
        help="Set CARLA_GRP_PP_IGNORE_ENDPOINTS=0 (disable default endpoint ignoring).",
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
        "--weather",
        choices=["clear", "cloudy", "overcast", "night"],
        default=None,
        help="Run a single weather variant and tag results with the weather name.",
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
    parser.add_argument(
        "--scenario-stall-output-timeout",
        type=float,
        default=SCENARIO_STALL_OUTPUT_TIMEOUT_DEFAULT,
        help=(
            "If a scenario exceeds its watchdog budget, require this many seconds of quiet stdout "
            "before killing it as stuck."
        ),
    )
    parser.add_argument(
        "--scenario-startup-quiet-timeout",
        type=float,
        default=SCENARIO_STARTUP_QUIET_TIMEOUT_DEFAULT,
        help=(
            "During startup, restart the scenario if there is no non-CARLA-diagnostic planner "
            "output and no route completion for this many seconds. "
            "Set <=0 to disable this startup quiet checker."
        ),
    )
    parser.add_argument(
        "--scenario-retry-limit",
        type=int,
        default=SCENARIO_RETRY_LIMIT_DEFAULT,
        help=(
            "Maximum restart attempts for one stuck scenario before giving up. "
            "Use 0 or a negative value for unlimited retries."
        ),
    )
    parser.add_argument(
        "--planner-retry-limit",
        type=int,
        default=PLANNER_RETRY_LIMIT_DEFAULT,
        help=(
            "Maximum parent-level relaunch attempts per planner worker in multi-planner mode. "
            "Use 0 or a negative value for unlimited relaunches."
        ),
    )
    parser.add_argument(
        "--perf-report-file",
        default="_launcher_perf.json",
        help=(
            "File name (or absolute path) for launcher bottleneck timing report. "
            "Use empty string to disable."
        ),
    )
    parser.add_argument(
        "--disable-checkpoint-recovery",
        action="store_true",
        help=(
            "Disable in-evaluator checkpoint-based CARLA crash recovery. "
            "Recovery is off by default; this flag force-disables it even if recovery-mode is set elsewhere."
        ),
    )
    parser.add_argument(
        "--recovery-mode",
        choices=("off", "engineering", "publication_safe"),
        default="off",
        help=(
            "Recovery policy mode. "
            "'off' disables in-evaluator recovery; "
            "'engineering' allows best-effort continuity fallbacks; "
            "'publication_safe' enforces strict benchmark-safe continuity constraints."
        ),
    )
    parser.add_argument(
        "--checkpoint-interval-frames",
        type=int,
        default=20,
        help="Fixed checkpoint cadence in logical evaluator frames.",
    )
    parser.add_argument(
        "--checkpoint-max-recoveries",
        type=int,
        default=5,
        help="Maximum number of in-process crash recoveries per route before failing the route.",
    )
    parser.add_argument(
        "--checkpoint-max-count",
        type=int,
        default=0,
        help="Optional retention cap for on-disk checkpoints (0 keeps all checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-restart-cmd",
        default="",
        help=(
            "Optional shell command executed by the evaluator child to restart CARLA after a crash. "
            "If empty, evaluator waits for CARLA to come back on its configured host/port."
        ),
    )
    parser.add_argument(
        "--checkpoint-nearby-radius-m",
        type=float,
        default=90.0,
        help="Radius used when checkpointing/restoring nearby dynamic actors.",
    )
    parser.add_argument(
        "--checkpoint-tl-radius-m",
        type=float,
        default=120.0,
        help="Radius used when checkpointing/restoring traffic lights near ego vehicles.",
    )
    parser.add_argument(
        "--checkpoint-actor-match-m",
        type=float,
        default=6.0,
        help="Maximum XY distance for matching checkpointed actors to respawned actors.",
    )
    parser.add_argument(
        "--planner-replay-window",
        type=int,
        default=40,
        help=(
            "Planner replay warm-start buffer length in frames for engineering-mode fallback "
            "(ignored in publication_safe mode)."
        ),
    )
    parser.add_argument(
        "--launcher-carladown-abort-with-child-recovery",
        action="store_true",
        help=(
            "Keep launcher CARLA-down hard-abort active even when child checkpoint recovery is enabled."
        ),
    )
    parser.add_argument(
        "--accept-stalled-near-end-failures",
        action="store_true",
        help=(
            "Accept and keep a non-zero evaluator run when runtime-failed egos are already "
            "near route completion (useful for end-of-route stall/crash recovery)."
        ),
    )
    parser.add_argument(
        "--stalled-near-end-min-route-score",
        type=float,
        default=STALLED_NEAR_END_MIN_ROUTE_SCORE_DEFAULT,
        help=(
            "Minimum per-ego route score (0-100) required to accept a runtime failure under "
            "--accept-stalled-near-end-failures."
        ),
    )
    parser.add_argument(
        "--stalled-near-end-require-stall-evidence",
        action="store_true",
        help=(
            "When accepting stalled near-end failures, require explicit stall markers in "
            "scenario logs/output."
        ),
    )
    parser.set_defaults(
        inject_ucla_v2_crosswalk=False,
        disable_stall_detection=True,
    )
    # Be tolerant to pasted unicode dashes/zero-width chars in CLI flags.
    dash_map = str.maketrans(
        {
            "\u2010": "-",  # hyphen
            "\u2011": "-",  # non-breaking hyphen
            "\u2012": "-",  # figure dash
            "\u2013": "-",  # en dash
            "\u2014": "-",  # em dash
            "\u2212": "-",  # minus sign
            "\ufe58": "-",  # small em dash
            "\ufe63": "-",  # small hyphen-minus
            "\uff0d": "-",  # fullwidth hyphen-minus
        }
    )
    normalized_argv = [
        arg.translate(dash_map).replace("\u200b", "").replace("\ufeff", "")
        for arg in sys.argv[1:]
    ]
    args = parser.parse_args(normalized_argv)
    if args.carla_rootcause and not args.start_carla:
        parser.error("--carla-rootcause requires --start-carla.")
    if int(args.carla_rootcause_diag_interval) <= 0:
        parser.error("--carla-rootcause-diag-interval must be a positive integer.")
    if int(args.carla_rootcause_startup_timeout) <= 0:
        parser.error("--carla-rootcause-startup-timeout must be a positive integer.")
    if int(args.carla_rootcause_core_wait_seconds) < 0:
        parser.error("--carla-rootcause-core-wait-seconds must be non-negative.")
    if not (0.0 <= float(args.stalled_near_end_min_route_score) <= 100.0):
        parser.error("--stalled-near-end-min-route-score must be within [0, 100].")
    if int(args.checkpoint_interval_frames) <= 0:
        parser.error("--checkpoint-interval-frames must be positive.")
    if int(args.checkpoint_max_recoveries) < 0:
        parser.error("--checkpoint-max-recoveries must be >= 0.")
    if int(args.checkpoint_max_count) < 0:
        parser.error("--checkpoint-max-count must be >= 0.")
    if int(args.planner_replay_window) <= 0:
        parser.error("--planner-replay-window must be positive.")
    if int(args.repro_reruns) <= 0:
        parser.error("--repro-reruns must be positive.")
    if int(args.repro_rerun_index) < 0:
        parser.error("--repro-rerun-index must be non-negative.")
    if int(args.repro_rerun_total) < 0:
        parser.error("--repro-rerun-total must be non-negative.")
    if int(args.repro_rerun_index) > 0 and int(args.repro_rerun_total) <= 0:
        parser.error("--repro-rerun-total must be positive when --repro-rerun-index is set.")
    if int(args.auto_workers_max_per_gpu) <= 0:
        parser.error("--auto-workers-max-per-gpu must be positive.")
    if int(args.oom_concurrency_backoff_step) <= 0:
        parser.error("--oom-concurrency-backoff-step must be positive.")
    if int(args.oom_free_mib_penalty_step) < 0:
        parser.error("--oom-free-mib-penalty-step must be non-negative.")
    if int(args.oom_recovery_success_streak) <= 0:
        parser.error("--oom-recovery-success-streak must be positive.")
    if int(args.oom_recovery_penalty_decay_step) < 0:
        parser.error("--oom-recovery-penalty-decay-step must be non-negative.")
    if float(args.gpu_query_cache_ttl) < 0.0:
        parser.error("--gpu-query-cache-ttl must be >= 0.")
    if float(args.planner_status_snapshot_interval) <= 0.0:
        parser.error("--planner-status-snapshot-interval must be > 0.")
    args.recovery_mode = str(args.recovery_mode or "off").strip().lower()
    if args.disable_checkpoint_recovery:
        args.recovery_mode = "off"
    if args.recovery_mode == "publication_safe" and args.accept_stalled_near_end_failures:
        parser.error("--accept-stalled-near-end-failures is not allowed in publication_safe recovery mode.")
    args.perf_report_file = str(args.perf_report_file or "").strip()

    # --max-multiprocess: apply aggressive defaults for maximal safe concurrency.
    if getattr(args, "max_multiprocess", False):
        args.allow_multiple_carlas_per_gpu = True
        args.vram_aware_scheduler = True
        args.auto_workers_per_gpu = True
        # Lower safety margins to pack GPUs more aggressively.
        if args.vram_safety_buffer_mib == VRAM_SAFETY_BUFFER_MIB_DEFAULT:
            args.vram_safety_buffer_mib = VRAM_SAFETY_BUFFER_AGGRESSIVE_MIB_DEFAULT
        if args.auto_workers_min_free_mib == AUTO_WORKERS_PER_GPU_MIN_FREE_MIB_DEFAULT:
            args.auto_workers_min_free_mib = MIN_FREE_AGGRESSIVE_MIB_DEFAULT

    # Parse --scenario-shard INDEX/COUNT into (index, count) or None.
    args._scenario_shard_parsed = None
    if args.scenario_shard is not None:
        _shard_parts = str(args.scenario_shard).split("/")
        if len(_shard_parts) == 2:
            try:
                _shard_idx = int(_shard_parts[0])
                _shard_cnt = int(_shard_parts[1])
                if 0 <= _shard_idx < _shard_cnt and _shard_cnt > 0:
                    args._scenario_shard_parsed = (_shard_idx, _shard_cnt)
                else:
                    raise ValueError(
                        f"--scenario-shard INDEX must be in [0, COUNT): got {args.scenario_shard}"
                    )
            except ValueError as exc:
                raise ValueError(f"Invalid --scenario-shard value: {args.scenario_shard}") from exc
        else:
            raise ValueError(
                f"--scenario-shard must be INDEX/COUNT (e.g. 0/4): got {args.scenario_shard}"
            )

    setattr(args, "_normalized_argv", normalized_argv)
    return args


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def wait_for_port_close(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_port_open(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def build_carla_down_abort_checker(
    *,
    host: str,
    world_port: int,
    carla_manager: "CarlaProcessManager | None" = None,
    grace_s: float = CARLA_DOWN_ABORT_GRACE_S,
) -> Callable[[], str | None]:
    grace_s = max(1.0, float(grace_s))
    down_since: float | None = None

    def _checker() -> str | None:
        nonlocal down_since
        manager_running = None if carla_manager is None else carla_manager.is_running()
        world_up = is_port_open(host, world_port, timeout=0.2)
        carla_down = (manager_running is False) or (not world_up)
        if not carla_down:
            down_since = None
            return None
        if down_since is None:
            down_since = time.monotonic()
            return None
        elapsed = time.monotonic() - down_since
        if elapsed < grace_s:
            return None
        details: List[str] = []
        if manager_running is False:
            details.append("process-exited")
        if not world_up:
            details.append(f"world-port={world_port}-down")
        return (
            "[WARN] Managed CARLA became unavailable for "
            f"{elapsed:.1f}s ({', '.join(details) or 'unknown'}). "
            "Terminating the evaluator early so the scenario retry path can restart CARLA."
        )

    return _checker


def combine_health_checks(
    *checks: Callable[[], str | None] | None,
) -> Callable[[], str | None] | None:
    active_checks = [check for check in checks if check is not None]
    if not active_checks:
        return None

    def _checker() -> str | None:
        for check in active_checks:
            reason = check()
            if reason:
                return reason
        return None

    return _checker


def build_startup_stall_abort_checker(
    *,
    result_root: Path,
    ego_count: int,
    runner_provider: Callable[[], "MonitoredSubprocessRunner | None"],
    quiet_timeout_s: float,
) -> Callable[[], str | None]:
    result_root = Path(result_root)
    ego_count = max(0, int(ego_count))
    quiet_timeout_s = float(quiet_timeout_s)
    if quiet_timeout_s <= 0.0:
        return lambda: None

    def _checker() -> str | None:
        runner = runner_provider()
        if runner is None:
            return None
        snapshot = runner.snapshot()
        elapsed_s = float(snapshot.get("elapsed_s", 0.0) or 0.0)
        quiet_for = float(snapshot.get("last_output_age_s", 0.0) or 0.0)
        if elapsed_s < quiet_timeout_s or quiet_for < quiet_timeout_s:
            return None
        summary = collect_result_root_progress(result_root, ego_count=ego_count)
        route_done, route_total = summary.route_progress
        # Startup-stall detection should only guard the pre-route phase. Once the
        # route has started or terminal outcomes exist, defer to normal outcome
        # handling instead of force-killing the evaluator.
        if summary.started or summary.completed or has_only_terminal_route_outcomes(summary):
            return None
        if route_done > 0:
            return None
        return (
            "[WARN] Scenario startup produced no non-CARLA planner output and no route progress "
            f"for {elapsed_s:.1f}s (quiet_for={quiet_for:.1f}s, routes={route_done}/{route_total}). "
            "Terminating it so managed CARLA and the route can be restarted cleanly."
        )

    return _checker


def build_tick_heartbeat_checker(
    heartbeat_path: Path,
    *,
    freeze_timeout_s: float = TICK_HEARTBEAT_FREEZE_TIMEOUT_S,
    grace_s: float = 90.0,
) -> Callable[[], str | None]:
    """Return a health-check that fires when the sim tick heartbeat goes stale.

    scenario_manager touches *heartbeat_path* at most once per second while the
    simulation is running (set via CARLA_TICK_HEARTBEAT_FILE env var).  If the
    file has not been updated for *freeze_timeout_s* seconds the simulator is
    considered deadlocked and the evaluator is terminated.

    A *grace_s* startup window (default 90 s) is not enforced to allow for CARLA
    map load time before the first tick.
    """
    heartbeat_path = Path(heartbeat_path)
    freeze_timeout_s = max(30.0, float(freeze_timeout_s))
    grace_s = max(0.0, float(grace_s))
    start_time = time.monotonic()

    def _checker() -> str | None:
        if time.monotonic() - start_time < grace_s:
            return None
        try:
            age = time.time() - heartbeat_path.stat().st_mtime
        except OSError:
            return None  # file not yet created — scenario hasn't started ticking
        if age < freeze_timeout_s:
            return None
        return (
            f"[WARN] Simulation tick heartbeat has not been updated for {age:.0f}s "
            f"(freeze_timeout={freeze_timeout_s:.0f}s). "
            "Terminating frozen scenario so CARLA and the route can be restarted."
        )

    return _checker


def wait_for_process_port(
    proc: subprocess.Popen[Any],
    host: str,
    port: int,
    timeout: float,
) -> bool:
    log_carla_event(
        "PORT_WAIT_BEGIN",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        host=host,
        port=int(port),
        timeout_s=float(timeout),
        pid=None if proc is None else proc.pid,
    )
    deadline = time.monotonic() + timeout
    checks = 0
    while time.monotonic() < deadline:
        checks += 1
        if is_port_open(host, port, timeout=1.0):
            log_carla_event(
                "PORT_WAIT_READY",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                host=host,
                port=int(port),
                pid=None if proc is None else proc.pid,
                checks=checks,
            )
            return True
        if proc.poll() is not None:
            log_carla_event(
                "PORT_WAIT_PROCESS_EXIT",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                host=host,
                port=int(port),
                pid=None if proc is None else proc.pid,
                returncode=proc.poll(),
                checks=checks,
            )
            return False
        time.sleep(0.5)
    log_carla_event(
        "PORT_WAIT_TIMEOUT",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        host=host,
        port=int(port),
        pid=None if proc is None else proc.pid,
        checks=checks,
        timeout_s=float(timeout),
    )
    return False


def wait_for_carla_rpc_ready(host: str, port: int, timeout: float = 60.0) -> bool:
    """Active RPC readiness check: retries carla.Client.get_world() until it succeeds.

    TCP port open does not mean CARLA's RPC layer is ready.  This function
    confirms the RPC handshake by calling get_world(), which is the first
    call any evaluator or planner will make.  Returns True once confirmed
    ready, False on timeout.
    """
    try:
        import carla  # type: ignore[import]
    except ImportError:
        print("[WARN][CARLA CONNECT] carla package not importable; skipping RPC readiness check.")
        return True  # can't verify; don't block startup unnecessarily

    deadline = time.monotonic() + timeout
    attempt = 0
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        attempt += 1
        client = None
        try:
            client = create_logged_client(
                carla,
                host,
                port,
                timeout_s=3.0,
                context="startup_rpc_probe",
                attempt=attempt,
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            )
            probe_client_id = get_logged_client_id(client)
            client.get_world()
            log_client_rpc_ready(
                host=host,
                port=port,
                context="startup_rpc_probe",
                attempt=attempt,
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                client_id=probe_client_id,
            )
            print(f"[CARLA CONNECT attempt={attempt}] RPC ready on {host}:{port}")
            return True
        except Exception as exc:
            last_exc = exc
            exc_type = type(exc).__name__
            log_client_retry(
                host=host,
                port=port,
                context="startup_rpc_probe",
                attempt=attempt,
                sleep_s=1.0,
                reason=f"{exc_type}: {exc}",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            )
            print(f"[CARLA CONNECT attempt={attempt}] exception={exc_type}: {exc}")
            time.sleep(1.0)
        finally:
            release_client_id = log_client_release_begin(
                client,
                context="startup_rpc_probe",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                reason="startup_rpc_probe_attempt_end",
            )
            client = None
            log_client_release_end(
                release_client_id,
                context="startup_rpc_probe",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                reason="startup_rpc_probe_attempt_end",
            )

    print(
        f"[CARLA CONNECT] RPC not ready after {attempt} attempt(s) "
        f"(timeout={timeout:.0f}s); last_exception={type(last_exc).__name__ if last_exc else 'none'}"
    )
    return False


def is_local_host(host: str) -> bool:
    normalized = str(host).strip().lower()
    return normalized in {"127.0.0.1", "localhost", "0.0.0.0", "::1", "::"}


def read_process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    try:
        proc = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def find_listening_pids(port: int) -> List[int]:
    try:
        proc = subprocess.run(
            [
                "lsof",
                "-nP",
                "-t",
                f"-iTCP:{int(port)}",
                "-sTCP:LISTEN",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    pids: List[int] = []
    for line in proc.stdout.splitlines():
        token = line.strip()
        if not token:
            continue
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid not in pids:
            pids.append(pid)
    return pids


def _extract_endpoint_port(endpoint: str) -> int | None:
    token = str(endpoint or "").strip()
    if not token:
        return None
    if token.startswith("[") and "]:" in token:
        token = token.rsplit("]:", 1)[-1]
    elif ":" in token:
        token = token.rsplit(":", 1)[-1]
    token = token.strip()
    if token.endswith("*"):
        token = token[:-1]
    if not token.isdigit():
        return None
    try:
        return int(token)
    except Exception:
        return None


def _sample_local_tcp_state_counts(port: int) -> Tuple[Dict[str, int], str, str | None]:
    """
    Return TCP state counts for sockets where src/dst port == *port*.
    """
    target_port = int(port)
    try:
        proc = subprocess.run(
            ["ss", "-tan"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return {}, "ss", f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return {}, "ss", (proc.stderr or proc.stdout or "").strip() or f"returncode={proc.returncode}"

    counts: Dict[str, int] = {}
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("state") or line.lower().startswith("netid"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        state = parts[0].strip().upper()
        local_ep = parts[3]
        peer_ep = parts[4]
        local_port = _extract_endpoint_port(local_ep)
        peer_port = _extract_endpoint_port(peer_ep)
        if local_port != target_port and peer_port != target_port:
            continue
        counts[state] = counts.get(state, 0) + 1
    return counts, "ss", None


def _count_active_connections(state_counts: Dict[str, int]) -> int:
    active = 0
    for state, count in state_counts.items():
        if str(state).upper() in CARLA_ACTIVE_SOCKET_STATES:
            active += int(count)
    return int(active)


def _count_time_wait_connections(state_counts: Dict[str, int]) -> int:
    return int(state_counts.get("TIME-WAIT", 0) + state_counts.get("TIME_WAIT", 0))


def _format_state_counts(state_counts: Dict[str, int]) -> str:
    if not state_counts:
        return ""
    ordered = sorted(state_counts.items(), key=lambda item: item[0])
    return ",".join(f"{state}:{count}" for state, count in ordered)


def wait_for_evaluator_start_gate(
    *,
    host: str,
    world_port: int,
    context: str,
    timeout_s: float = CARLA_TEARDOWN_GATE_TIMEOUT_DEFAULT,
    poll_s: float = CARLA_TEARDOWN_GATE_POLL_S_DEFAULT,
    stable_samples: int = CARLA_TEARDOWN_GATE_STABLE_SAMPLES_DEFAULT,
) -> bool:
    """
    Ensure prior evaluator/client sockets have quiesced before launching the next evaluator.
    """
    timeout_s = max(0.0, float(timeout_s))
    poll_s = max(0.05, float(poll_s))
    stable_samples = max(1, int(stable_samples))
    host_is_local = is_local_host(host)
    log_carla_event(
        "EVALUATOR_START_GATE_BEGIN",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        context=context,
        host=host,
        world_port=int(world_port),
        timeout_s=timeout_s,
        poll_s=poll_s,
        stable_samples=stable_samples,
        local_host=int(bool(host_is_local)),
    )
    if not host_is_local:
        log_carla_event(
            "EVALUATOR_START_GATE_END",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            context=context,
            host=host,
            world_port=int(world_port),
            status="skip_remote_host",
        )
        return True

    deadline = time.monotonic() + timeout_s
    stable = 0
    samples = 0
    while time.monotonic() <= deadline:
        samples += 1
        state_counts, source, sample_error = _sample_local_tcp_state_counts(world_port)
        if sample_error is not None:
            # If socket sampling is unavailable, do not block evaluator progress.
            log_carla_event(
                "EVALUATOR_START_GATE_END",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                host=host,
                world_port=int(world_port),
                status="sampling_unavailable",
                source=source,
                error=sample_error,
            )
            print(
                "[WARN] Evaluator teardown gate sampling unavailable "
                f"for {host}:{int(world_port)} ({sample_error}); proceeding without socket gate."
            )
            return True

        active_count = _count_active_connections(state_counts)
        time_wait_count = _count_time_wait_connections(state_counts)
        if active_count <= 0:
            stable += 1
        else:
            stable = 0

        if samples == 1 or active_count > 0 or stable >= stable_samples:
            log_carla_event(
                "EVALUATOR_START_GATE_SAMPLE",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                host=host,
                world_port=int(world_port),
                sample=samples,
                active_connections=active_count,
                time_wait_connections=time_wait_count,
                stable_samples_observed=stable,
                state_counts=_format_state_counts(state_counts),
            )

        if stable >= stable_samples:
            log_carla_event(
                "EVALUATOR_START_GATE_END",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                host=host,
                world_port=int(world_port),
                status="ready",
                samples=samples,
                active_connections=active_count,
                time_wait_connections=time_wait_count,
                state_counts=_format_state_counts(state_counts),
            )
            return True

        time.sleep(poll_s)

    final_counts, _, _ = _sample_local_tcp_state_counts(world_port)
    log_carla_event(
        "EVALUATOR_START_GATE_END",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        context=context,
        host=host,
        world_port=int(world_port),
        status="timeout",
        samples=samples,
        active_connections=_count_active_connections(final_counts),
        time_wait_connections=_count_time_wait_connections(final_counts),
        state_counts=_format_state_counts(final_counts),
    )
    return False


def process_matches_tokens(pid: int, match_tokens: Sequence[str]) -> bool:
    if not match_tokens:
        return True
    cmdline = read_process_cmdline(pid).lower()
    return any(token.lower() in cmdline for token in match_tokens)


def terminate_pid_or_group(pid: int, grace_s: float = 15.0) -> None:
    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = None

    terminated = False
    if pgid is not None and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
            terminated = True
        except Exception:
            terminated = False
    if not terminated:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            return

    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except Exception:
            break
        time.sleep(0.2)

    if pgid is not None and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def cleanup_listening_processes(
    *,
    host: str,
    ports: Iterable[int],
    service_name: str,
    match_tokens: Sequence[str],
    grace_s: float = 15.0,
) -> Tuple[bool, List[str]]:
    if not is_local_host(host):
        return False, []

    all_pids: List[int] = []
    unmatched_details: List[str] = []
    for port in sorted(set(int(port) for port in ports)):
        for pid in find_listening_pids(port):
            if pid in all_pids:
                continue
            if process_matches_tokens(pid, match_tokens):
                all_pids.append(pid)
            else:
                cmdline = read_process_cmdline(pid) or "<unknown>"
                unmatched_details.append(
                    f"port={port} pid={pid} cmd={cmdline}"
                )

    for pid in all_pids:
        terminate_pid_or_group(pid, grace_s=grace_s)

    closed = True
    deadline = time.monotonic() + max(1.0, grace_s)
    target_ports = tuple(sorted(set(int(port) for port in ports)))
    while time.monotonic() < deadline:
        if all(not is_port_open(host, port, timeout=0.2) for port in target_ports):
            return True, unmatched_details
        time.sleep(0.2)
    closed = all(not is_port_open(host, port, timeout=0.2) for port in target_ports)
    return closed, unmatched_details


def find_available_port(host: str, preferred_port: int, tries: int, step: int) -> int:
    attempts = max(1, tries)
    step = max(1, step)
    for idx in range(attempts):
        port = preferred_port + idx * step
        if port > 65535:
            break
        if not is_port_open(host, port, timeout=1.0):
            return port
    raise RuntimeError(
        f"No free CARLA port found starting at {preferred_port} "
        f"(tries={attempts}, step={step})."
    )


def compute_tm_port_offset(world_port: int, tm_port: int | None) -> int:
    if tm_port is None:
        return CARLA_TM_PORT_OFFSET_DEFAULT
    offset = int(tm_port) - int(world_port)
    if offset == 0:
        raise ValueError("Traffic Manager port must differ from CARLA world port.")
    return offset


def carla_service_ports(
    world_port: int,
    tm_port_offset: int,
    *,
    stream_port_offset: int = CARLA_STREAM_PORT_OFFSET_DEFAULT,
) -> tuple[int, ...]:
    ports: list[int] = []
    for candidate in (
        int(world_port),
        int(world_port) + int(stream_port_offset),
        int(world_port) + int(tm_port_offset),
    ):
        if candidate not in ports:
            ports.append(candidate)
    return tuple(ports)


def _parse_addr_port(address: str) -> int | None:
    token = str(address).strip()
    if not token:
        return None
    if token.startswith("["):
        end = token.rfind("]")
        if end < 0:
            return None
        suffix = token[end + 1 :]
        if suffix.startswith(":"):
            suffix = suffix[1:]
        if not suffix or suffix == "*":
            return None
        try:
            return int(suffix)
        except ValueError:
            return None
    if ":" not in token:
        return None
    suffix = token.rsplit(":", 1)[-1]
    if not suffix or suffix == "*":
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def snapshot_socket_state_counts_for_ports(ports: Iterable[int]) -> Dict[str, int] | None:
    tracked_ports = {int(port) for port in ports if int(port) > 0}
    if not tracked_ports:
        return {}
    try:
        proc = subprocess.run(
            ["ss", "-tan"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None

    counts: Dict[str, int] = {}
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("State"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        state = str(parts[0]).upper()
        local_port = _parse_addr_port(parts[3])
        peer_port = _parse_addr_port(parts[4])
        if local_port not in tracked_ports and peer_port not in tracked_ports:
            continue
        counts[state] = counts.get(state, 0) + 1
    return counts


def _teardown_gate_key(host: str, ports: Iterable[int]) -> str:
    normalized_host = str(host).strip().lower()
    normalized_ports = sorted({int(port) for port in ports if int(port) > 0})
    return "{}|{}".format(
        normalized_host,
        ",".join(str(port) for port in normalized_ports),
    )


def _get_teardown_gate_baseline(gate_key: str) -> int | None:
    with _TEARDOWN_GATE_BASELINES_LOCK:
        baseline = _TEARDOWN_GATE_BASELINES.get(str(gate_key))
    return None if baseline is None else int(baseline)


def _update_teardown_gate_baseline(
    gate_key: str,
    observed_active: int,
) -> tuple[int | None, int, bool]:
    observed = max(0, int(observed_active))
    with _TEARDOWN_GATE_BASELINES_LOCK:
        previous = _TEARDOWN_GATE_BASELINES.get(str(gate_key))
        if previous is None:
            _TEARDOWN_GATE_BASELINES[str(gate_key)] = observed
            return None, observed, True
        previous_int = int(previous)
        # Keep the baseline monotonic per process so transient low snapshots do not
        # ratchet the gate threshold down and create false teardown-gate timeouts.
        if observed > previous_int:
            _TEARDOWN_GATE_BASELINES[str(gate_key)] = observed
            return previous_int, observed, True
        return previous_int, previous_int, False


def wait_for_carla_teardown_gate(
    *,
    host: str,
    world_port: int,
    tm_port: int,
    context: str,
    stream_port: int | None = None,
    timeout_s: float = CARLA_TEARDOWN_GATE_TIMEOUT_DEFAULT,
    poll_s: float = CARLA_TEARDOWN_GATE_POLL_S_DEFAULT,
    stable_samples: int = CARLA_TEARDOWN_GATE_STABLE_SAMPLES_DEFAULT,
    quiet: bool = False,
) -> bool:
    begin_t = time.monotonic()
    _perf_add("teardown_gate_calls", 1)
    tracked_ports = {
        int(world_port),
        int(tm_port),
        int(stream_port) if stream_port is not None else int(world_port) + CARLA_STREAM_PORT_OFFSET_DEFAULT,
    }
    tracked_ports = {port for port in tracked_ports if port > 0}
    if not tracked_ports:
        _perf_add("teardown_gate_time_s", time.monotonic() - begin_t)
        return True

    gate_key = _teardown_gate_key(str(host), tracked_ports)
    baseline_active = _get_teardown_gate_baseline(gate_key)

    if not is_local_host(host):
        log_carla_event(
            "EVALUATOR_START_GATE_SKIP",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            context=context,
            reason="remote_host",
            host=str(host),
            world_port=int(world_port),
            tm_port=int(tm_port),
        )
        _perf_add("teardown_gate_time_s", time.monotonic() - begin_t)
        return True

    begin_message = (
        "[INFO] Teardown gate: waiting for CARLA socket quiescence "
        f"(context={context}, ports={sorted(tracked_ports)}, baseline_active="
        f"{'unset' if baseline_active is None else int(baseline_active)})."
    )
    if not quiet:
        print(begin_message)
    log_carla_event(
        "EVALUATOR_START_GATE_BEGIN",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        context=context,
        host=str(host),
        ports=",".join(str(port) for port in sorted(tracked_ports)),
        timeout_s=float(timeout_s),
        poll_s=float(poll_s),
        stable_samples=int(stable_samples),
        baseline_active=-1 if baseline_active is None else int(baseline_active),
    )

    deadline = time.monotonic() + max(0.5, float(timeout_s))
    stable = 0
    last_counts: Dict[str, int] | None = None
    last_active = -1
    snapshots = 0
    while time.monotonic() < deadline:
        snapshots += 1
        counts = snapshot_socket_state_counts_for_ports(sorted(tracked_ports))
        if counts is None:
            log_carla_event(
                "EVALUATOR_START_GATE_STATE",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                status="unavailable",
                snapshots=int(snapshots),
            )
            if not quiet:
                print(
                    "[WARN] Teardown gate socket snapshot unavailable; "
                    "continuing without strict socket-state gating."
                )
            _perf_add("teardown_gate_time_s", time.monotonic() - begin_t)
            return True

        active = sum(counts.get(state, 0) for state in CARLA_ACTIVE_SOCKET_STATES)
        time_wait = counts.get("TIME-WAIT", 0)
        baseline_prev, baseline_active, baseline_changed = _update_teardown_gate_baseline(
            gate_key,
            int(active),
        )
        if baseline_changed:
            log_carla_event(
                (
                    "EVALUATOR_START_GATE_BASELINE_SET"
                    if baseline_prev is None
                    else "EVALUATOR_START_GATE_BASELINE_UPDATE"
                ),
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                gate_key=gate_key,
                observed_active=int(active),
                baseline_active=int(baseline_active),
                previous_baseline=-1 if baseline_prev is None else int(baseline_prev),
            )

        ready_active_threshold = max(0, int(baseline_active))
        if counts != last_counts or active != last_active:
            log_carla_event(
                "EVALUATOR_START_GATE_STATE",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                active_connections=int(active),
                time_wait_connections=int(time_wait),
                baseline_active=int(baseline_active),
                ready_active_threshold=int(ready_active_threshold),
                states=";".join(
                    f"{state}:{counts[state]}" for state in sorted(counts.keys())
                ),
                snapshots=int(snapshots),
            )
            last_counts = dict(counts)
            last_active = int(active)

        if active <= ready_active_threshold:
            stable += 1
        else:
            stable = 0

        effective_stable_samples = max(1, int(stable_samples))
        if int(baseline_active) == 0 and int(active) == 0:
            # Fast-path: no active CARLA sockets at all. Waiting for extra
            # stable samples adds ~poll_s * (stable_samples-1) launch latency
            # without improving safety in this all-idle case.
            effective_stable_samples = 1

        if stable >= effective_stable_samples:
            duration_s = max(0.0, float(timeout_s) - max(0.0, deadline - time.monotonic()))
            log_carla_event(
                "EVALUATOR_START_GATE_END",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                context=context,
                status="ready",
                active_connections=int(active),
                time_wait_connections=int(time_wait),
                baseline_active=int(baseline_active),
                ready_active_threshold=int(ready_active_threshold),
                stable_samples_required=int(effective_stable_samples),
                duration_s=duration_s,
                snapshots=int(snapshots),
            )
            if not quiet:
                print(
                    "[INFO] Teardown gate passed: CARLA connections quiesced "
                    f"(context={context}, active={active}, time_wait={time_wait}, "
                    f"baseline_active={baseline_active})."
                )
            _perf_add("teardown_gate_time_s", time.monotonic() - begin_t)
            return True
        time.sleep(max(0.05, float(poll_s)))

    final_counts = snapshot_socket_state_counts_for_ports(sorted(tracked_ports)) or {}
    final_active = sum(final_counts.get(state, 0) for state in CARLA_ACTIVE_SOCKET_STATES)
    final_time_wait = final_counts.get("TIME-WAIT", 0)
    final_baseline = _get_teardown_gate_baseline(gate_key)
    if final_baseline is None:
        final_baseline = 0
    log_carla_event(
        "EVALUATOR_START_GATE_END",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        context=context,
        status="timeout",
        active_connections=int(final_active),
        time_wait_connections=int(final_time_wait),
        baseline_active=int(final_baseline),
        ready_active_threshold=int(max(0, int(final_baseline))),
        snapshots=int(snapshots),
    )
    if not quiet:
        print(
            "[WARN] Teardown gate timed out; not-ready "
            f"(context={context}, active={final_active}, time_wait={final_time_wait}, "
            f"baseline_active={final_baseline})."
        )
    _perf_add("teardown_gate_time_s", time.monotonic() - begin_t)
    _perf_add("teardown_gate_timeouts", 1)
    return False


def are_ports_available(
    host: str,
    ports: Iterable[int],
    *,
    reserved_ports: set[int] | None = None,
) -> bool:
    reserved = reserved_ports or set()
    for port in ports:
        if port < 1 or port > 65535:
            return False
        if port in reserved:
            return False
        if is_port_open(host, port, timeout=1.0):
            return False
    return True


def scan_port_availability(
    host: str,
    ports: Iterable[int],
    *,
    reserved_ports: set[int] | None = None,
) -> Dict[int, bool]:
    availability: Dict[int, bool] = {}
    reserved = reserved_ports or set()
    for port in sorted(set(int(port) for port in ports)):
        if port < 1 or port > 65535 or port in reserved:
            availability[port] = False
            continue
        availability[port] = not is_port_open(host, port, timeout=1.0)
    return availability


def find_available_port_bundles(
    host: str,
    preferred_port: int,
    tries: int,
    step: int,
    *,
    bundle_count: int = 1,
    stream_port_offset: int = CARLA_STREAM_PORT_OFFSET_DEFAULT,
    tm_port_offset: int = CARLA_TM_PORT_OFFSET_DEFAULT,
    reserved_ports: set[int] | None = None,
) -> List[PortBundle]:
    attempts = max(1, tries)
    step = max(1, step)
    bundle_count = max(1, int(bundle_count))
    offsets = (0, int(stream_port_offset), int(tm_port_offset))
    min_world_port = max(1, 1 - min(offsets))
    max_world_port = 65535 - max(offsets)
    scan_start = max(min_world_port, int(preferred_port))
    block_attempts = max(
        attempts,
        bundle_count * max(8, max(abs(offset) for offset in offsets) + 3),
        64,
    )
    reserved = set(reserved_ports or set())
    bundles: List[PortBundle] = []
    initial_window_reported = False

    while scan_start <= max_world_port:
        max_index = max(0, (max_world_port - scan_start) // step)
        current_attempts = min(block_attempts, max_index + 1)
        world_ports = [scan_start + idx * step for idx in range(current_attempts)]
        scan_ports = set()
        for world_port in world_ports:
            scan_ports.update(carla_service_ports(world_port, tm_port_offset, stream_port_offset=stream_port_offset))

        availability = scan_port_availability(
            host,
            scan_ports,
            reserved_ports=reserved,
        )

        for world_port in world_ports:
            required_ports = carla_service_ports(
                world_port,
                tm_port_offset,
                stream_port_offset=stream_port_offset,
            )
            if not all(availability.get(port, False) for port in required_ports):
                continue
            tm_port = world_port + tm_port_offset
            bundle = PortBundle(port=world_port, tm_port=tm_port)
            bundles.append(bundle)
            reserved.update(required_ports)
            for port in required_ports:
                availability[port] = False
            if len(bundles) >= bundle_count:
                return bundles

        if not initial_window_reported and scan_start == preferred_port:
            initial_window_reported = True
            print(
                "[INFO] No free CARLA port set found in the initial search window "
                f"({current_attempts} attempts from {preferred_port}). Continuing broader scan."
            )
        if not world_ports:
            break
        scan_start = world_ports[-1] + step

    raise RuntimeError(
        f"No free CARLA port set found starting at {preferred_port} "
        f"(initial_tries={attempts}, step={step}, stream_offset={stream_port_offset}, "
        f"tm_offset={tm_port_offset}, "
        f"requested_bundles={bundle_count})."
    )


def find_available_port_bundle(
    host: str,
    preferred_port: int,
    tries: int,
    step: int,
    *,
    stream_port_offset: int = CARLA_STREAM_PORT_OFFSET_DEFAULT,
    tm_port_offset: int = CARLA_TM_PORT_OFFSET_DEFAULT,
    reserved_ports: set[int] | None = None,
) -> PortBundle:
    return find_available_port_bundles(
        host,
        preferred_port,
        tries,
        step,
        bundle_count=1,
        stream_port_offset=stream_port_offset,
        tm_port_offset=tm_port_offset,
        reserved_ports=reserved_ports,
    )[0]


def should_add_offscreen(extra_args: List[str]) -> bool:
    if os.environ.get("DISPLAY"):
        return False
    lowered = {arg.lower() for arg in extra_args}
    if any("renderoffscreen" in arg for arg in lowered):
        return False
    if any("windowed" in arg for arg in lowered):
        return False
    return True


def _ensure_carla_arg(extra_args: List[str], arg: str) -> None:
    arg_lower = arg.lower()
    for current in extra_args:
        if current.lower() == arg_lower:
            return
    extra_args.append(arg)


def _enable_core_dumps_best_effort() -> tuple[bool, str]:
    if resource is None:
        return False, "resource module unavailable"
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        desired_soft = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        resource.setrlimit(resource.RLIMIT_CORE, (desired_soft, hard))
        new_soft, new_hard = resource.getrlimit(resource.RLIMIT_CORE)
        return True, f"core_limit soft={new_soft} hard={new_hard} (previous_soft={soft})"
    except Exception as exc:  # pragma: no cover - platform dependent
        return False, str(exc)


def _build_runtime_env(args: argparse.Namespace, base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)
    if not args.carla_forensics:
        # The local CARLA wrapper enables diagnostics/strace by default.
        # Disable that for normal evaluation runs so launcher-managed CARLA stays lightweight.
        env["CARLA_SERVER_DIAG"] = "0"
        env["CARLA_DIAG_ENABLE_STRACE"] = "0"
        return env

    env["CARLA_SERVER_DIAG"] = "1"
    env["CARLA_DIAG_ENABLE_STRACE"] = "1"
    env.setdefault("CARLA_STREAM_DEBUG", "1")
    env.setdefault(
        "CARLA_STREAM_DEBUG_SUMMARY_EVERY",
        str(max(1, int(args.carla_stream_debug_summary_every))),
    )
    env.setdefault("MALLOC_CHECK_", CARLA_FORENSICS_MALLOC_CHECK_DEFAULT)
    env.setdefault("MALLOC_PERTURB_", CARLA_FORENSICS_MALLOC_PERTURB_DEFAULT)
    return env


def _strip_path_entry(path_value: str, entry: str) -> str:
    target = os.path.abspath(entry)
    kept = [item for item in path_value.split(os.pathsep) if os.path.abspath(item) != target]
    return os.pathsep.join(item for item in kept if item)


def _is_venv_bin_path(entry: str) -> bool:
    """Return True if *entry* is inside a .venv or venv activation tree.

    Checks for the literal component ``.venv`` anywhere in the absolute path,
    which covers the common ``project/.venv/bin`` layout regardless of whether
    ``VIRTUAL_ENV`` is set.  A plain ``venv/`` directory is also matched so
    both naming conventions are caught.
    """
    try:
        parts = Path(os.path.abspath(entry)).parts
    except Exception:
        return False
    return ".venv" in parts or (len(parts) > 1 and parts[-2] in ("venv", ".venv"))


def _build_conda_launch_env(base_env: Dict[str, str]) -> Dict[str, str]:
    """Build an environment suitable for launching a conda-managed child process.

    Removes all virtual-environment contamination from PATH regardless of
    whether ``VIRTUAL_ENV`` is set.  This prevents ``.venv/bin`` from leaking
    into a conda child and causing the wrong Python / wrong packages to be
    loaded.
    """
    env = dict(base_env)
    virtual_env = env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env.pop("__PYVENV_LAUNCHER__", None)

    # --- strip VIRTUAL_ENV-reported bin dir (legacy path) ---
    if virtual_env:
        venv_bin = Path(virtual_env) / ("Scripts" if os.name == "nt" else "bin")
        env["PATH"] = _strip_path_entry(env.get("PATH", ""), str(venv_bin))

    # --- strip ANY remaining .venv/bin entries, with or without VIRTUAL_ENV ---
    raw_path = env.get("PATH", "")
    kept: list[str] = []
    stripped: list[str] = []
    for entry in raw_path.split(os.pathsep):
        if entry and _is_venv_bin_path(entry):
            stripped.append(entry)
        elif entry:
            kept.append(entry)
    if stripped:
        print(f"[EVAL ENV] Stripped venv PATH entries: {stripped}")
    env["PATH"] = os.pathsep.join(kept)
    return env


def _append_text_log_line(log_path: Path | None, line: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip("\n") + "\n")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to append log line to {log_path}: {exc}")


def terminate_process_tree(proc: subprocess.Popen[Any], grace_s: float = 15.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def wait_for_carla_post_start_buffer(buffer_s: float) -> None:
    buffer_s = max(0.0, float(buffer_s))
    if buffer_s <= 0:
        return
    print(
        "[INFO] Waiting "
        f"{buffer_s:.1f}s for managed CARLA to settle before launching the evaluator."
    )
    time.sleep(buffer_s)


class MonitoredSubprocessRunner:
    def __init__(
        self,
        cmd: Sequence[str],
        *,
        env: Dict[str, str],
        cwd: Path | None = None,
        timeout_s: float | None = None,
        stall_output_timeout_s: float = SCENARIO_STALL_OUTPUT_TIMEOUT_DEFAULT,
        output_prefix: str = "",
        tail_lines: int = MONITORED_OUTPUT_TAIL_LINES,
        show_carla_diag: bool = False,
        emit_stdout: bool = True,
        log_path: Path | None = None,
        health_check: Callable[[], str | None] | None = None,
    ) -> None:
        self.cmd = list(cmd)
        self.env = dict(env)
        self.cwd = None if cwd is None else str(cwd)
        self.timeout_s = timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        self.stall_output_timeout_s = max(0.0, float(stall_output_timeout_s))
        self.output_prefix = output_prefix
        self.tail_lines = max(1, int(tail_lines))
        self.show_carla_diag = bool(show_carla_diag)
        self.emit_stdout = bool(emit_stdout)
        self.log_path = None if log_path is None else Path(log_path)
        self.health_check = health_check
        self.process: subprocess.Popen[str] | None = None
        self._start_time = 0.0
        self._last_output_time = 0.0
        self._last_any_output_time = 0.0
        self._last_output_lock = threading.Lock()
        self._tail: deque[str] = deque(maxlen=self.tail_lines)
        self._tail_any: deque[str] = deque(maxlen=self.tail_lines)
        self._stop_event = threading.Event()
        self._stalled = threading.Event()
        self._terminated = threading.Event()
        self._termination_reason: str | None = None
        self._log_handle: TextIO | None = None
        self._run_log_state_path = (
            None
            if self.log_path is None
            else self.log_path.with_name(f".{self.log_path.name}.run_state.json")
        )
        self._run_log_seq: int | None = None
        self._run_log_started_at: str | None = None

    @staticmethod
    def _format_run_log_field(name: str, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            rendered = json.dumps(str(value), ensure_ascii=True)
        return f" {name}={rendered}"

    @staticmethod
    def _parse_run_log_field(line: str, name: str) -> Any:
        match = re.search(
            rf"\b{name}=((?:\"(?:[^\"\\\\]|\\\\.)*\")|[^\s=]+)",
            line,
        )
        if match is None:
            return None
        raw = match.group(1)
        if raw.startswith('"'):
            try:
                return json.loads(raw)
            except Exception:
                return raw.strip('"')
        if raw == "true":
            return True
        if raw == "false":
            return False
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    def _recover_run_log_state_from_file(self) -> Dict[str, Any]:
        if self.log_path is None or not self.log_path.exists():
            return {}

        start_count = 0
        last_start: Dict[str, Any] | None = None
        last_end: Dict[str, Any] | None = None
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line.startswith("=== RUN START "):
                        start_count += 1
                        timestamp = line[len("=== RUN START ") :].split(" ", 1)[0].strip()
                        run_seq = self._parse_run_log_field(line, "run_seq")
                        if not isinstance(run_seq, int):
                            run_seq = start_count
                        last_start = {
                            "run_seq": run_seq,
                            "started_at": timestamp,
                        }
                    elif line.startswith("=== RUN END "):
                        timestamp = line[len("=== RUN END ") :].split(" ", 1)[0].strip()
                        run_seq = self._parse_run_log_field(line, "run_seq")
                        returncode = self._parse_run_log_field(line, "returncode")
                        status = self._parse_run_log_field(line, "status")
                        if status is None:
                            status = "finished" if returncode == 0 else "failed"
                        last_end = {
                            "run_seq": run_seq if isinstance(run_seq, int) else None,
                            "ended_at": timestamp,
                            "status": str(status),
                            "returncode": returncode if isinstance(returncode, int) else None,
                            "terminated_reason": self._parse_run_log_field(line, "terminated_reason"),
                        }
        except Exception:
            return {}

        recovered: Dict[str, Any] = {}
        next_run_seq = 0
        if last_start is not None:
            next_run_seq = max(next_run_seq, int(last_start.get("run_seq") or 0))
        if last_end is not None and isinstance(last_end.get("run_seq"), int):
            next_run_seq = max(next_run_seq, int(last_end["run_seq"]))
        if next_run_seq > 0:
            recovered["next_run_seq"] = next_run_seq

        if last_start is None:
            if last_end is not None:
                recovered["last_run"] = last_end
            return recovered

        last_start_seq = int(last_start.get("run_seq") or 0)
        last_end_seq = int(last_end.get("run_seq") or 0) if last_end is not None else 0
        if last_start_seq > last_end_seq:
            recovered["open_run"] = last_start
        elif last_end is not None:
            recovered["last_run"] = last_end
        return recovered

    def _load_run_log_state(self) -> Dict[str, Any]:
        path = self._run_log_state_path
        if path is None:
            return {}
        state = load_json_file(path)
        if state:
            return state
        return self._recover_run_log_state_from_file()

    def _write_run_log_state(self, payload: Dict[str, Any]) -> None:
        path = self._run_log_state_path
        if path is None:
            return
        try:
            write_json_file_atomic(path, payload)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to update run-log state {path}: {exc}")

    def _write_run_log_marker(self, marker: str, timestamp: str, **fields: Any) -> None:
        if self._log_handle is None:
            return
        line = f"=== {marker} {timestamp}"
        for key, value in fields.items():
            line += self._format_run_log_field(key, value)
        line += " ===\n"
        self._log_handle.write(line)

    def _begin_run_log(self) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")

        state = self._load_run_log_state()
        now_iso = dt.datetime.now().isoformat(timespec="seconds")
        previous_open = state.get("open_run")
        if isinstance(previous_open, dict):
            previous_run_seq = previous_open.get("run_seq")
            previous_started_at = str(previous_open.get("started_at") or "")
            self._log_handle.write("\n")
            self._write_run_log_marker(
                "RUN STOPPED",
                now_iso,
                previous_run_seq=previous_run_seq,
                previous_started_at=previous_started_at or None,
                reason="new_run_started_before_previous_finished",
            )
            state["last_run"] = {
                "run_seq": previous_run_seq,
                "started_at": previous_started_at,
                "ended_at": now_iso,
                "status": "stopped",
                "reason": "new_run_started_before_previous_finished",
            }
            state["open_run"] = None

        previous_last = state.get("last_run")
        if isinstance(previous_last, dict):
            self._write_run_log_marker(
                "RUN RESTART",
                now_iso,
                previous_run_seq=previous_last.get("run_seq"),
                previous_status=previous_last.get("status"),
                previous_returncode=previous_last.get("returncode"),
                previous_ended_at=previous_last.get("ended_at"),
                previous_reason=previous_last.get("reason"),
            )

        next_run_seq = int(state.get("next_run_seq", 0) or 0) + 1
        self._run_log_seq = next_run_seq
        self._run_log_started_at = now_iso
        state["next_run_seq"] = next_run_seq
        state["open_run"] = {
            "run_seq": next_run_seq,
            "started_at": now_iso,
            "command": list(self.cmd),
        }
        self._write_run_log_state(state)

        self._log_handle.write("\n")
        self._write_run_log_marker(
            "RUN START",
            now_iso,
            run_seq=next_run_seq,
        )
        self._log_handle.write("COMMAND: " + " ".join(self.cmd) + "\n")
        self._log_handle.flush()

    def _finish_run_log(
        self,
        *,
        status: str,
        returncode: int | None,
        error: str | None = None,
    ) -> None:
        now_iso = dt.datetime.now().isoformat(timespec="seconds")
        state = self._load_run_log_state()
        state["open_run"] = None
        state["last_run"] = {
            "run_seq": self._run_log_seq,
            "started_at": self._run_log_started_at,
            "ended_at": now_iso,
            "status": str(status),
            "returncode": returncode,
            "stalled": bool(self._stalled.is_set()),
            "terminated_reason": self._termination_reason,
        }
        if error:
            state["last_run"]["error"] = str(error)
        self._write_run_log_state(state)
        if self._log_handle is not None:
            self._write_run_log_marker(
                "RUN END",
                now_iso,
                run_seq=self._run_log_seq,
                status=status,
                returncode=returncode,
                stalled=bool(self._stalled.is_set()),
                terminated_reason=self._termination_reason,
                error=error,
            )
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None

    def _derive_run_log_status(self, returncode: int | None) -> str:
        if self._stalled.is_set():
            return "stopped"
        if self._termination_reason is not None:
            return "stopped"
        if returncode is not None and returncode < 0:
            return "stopped"
        if returncode == 0:
            return "finished"
        return "failed"

    def _note_output(self, line: str, *, diagnostic: bool = False) -> None:
        now = time.monotonic()
        line = line.rstrip("\n")
        with self._last_output_lock:
            self._last_any_output_time = now
            self._tail_any.append(line)
            if not diagnostic:
                self._last_output_time = now
                self._tail.append(line)

    def _get_elapsed_s(self) -> float:
        if self._start_time <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self._start_time)

    def _get_last_output_age_s(self, *, include_diagnostic: bool = False) -> float:
        with self._last_output_lock:
            last_output_time = self._last_any_output_time if include_diagnostic else self._last_output_time
        return max(0.0, time.monotonic() - last_output_time)

    def _get_output_tail(self, *, include_diagnostic: bool = False) -> List[str]:
        with self._last_output_lock:
            return list(self._tail_any if include_diagnostic else self._tail)

    def snapshot(self) -> Dict[str, Any]:
        proc = self.process
        return {
            "running": proc is not None and proc.poll() is None,
            "returncode": None if proc is None else proc.poll(),
            "elapsed_s": self._get_elapsed_s(),
            "last_output_age_s": self._get_last_output_age_s(),
            "last_any_output_age_s": self._get_last_output_age_s(include_diagnostic=True),
            "stalled": self._stalled.is_set(),
            "terminated_reason": self._termination_reason,
        }

    def _emit_tail(self) -> None:
        tail = self._get_output_tail()
        if not tail:
            tail = self._get_output_tail(include_diagnostic=True)
        if not tail:
            message = "[WARN] No recent stdout/stderr captured before termination."
            if self.emit_stdout:
                print(message)
            elif self._log_handle is not None:
                self._log_handle.write(message + "\n")
                self._log_handle.flush()
            return
        if self.emit_stdout:
            print("[WARN] Recent stdout/stderr before termination:")
            for line in tail[-min(20, len(tail)) :]:
                print(f"[TAIL] {line}")
        elif self._log_handle is not None:
            self._log_handle.write("[WARN] Recent stdout/stderr before termination:\n")
            for line in tail[-min(20, len(tail)) :]:
                self._log_handle.write(f"[TAIL] {line}\n")
            self._log_handle.flush()

    def terminate(self, reason: str | None = None) -> None:
        proc = self.process
        if proc is None or self._terminated.is_set():
            return
        self._terminated.set()
        if reason:
            if self._termination_reason is None:
                self._termination_reason = reason.rstrip("\n")
            if self.emit_stdout:
                print(reason)
            elif self._log_handle is not None:
                self._log_handle.write(reason.rstrip("\n") + "\n")
                self._log_handle.flush()
            log_carla_event(
                "CHILD_TERMINATE_REQUEST",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                reason=self._termination_reason,
                pid=None if proc is None else proc.pid,
                command=" ".join(self.cmd),
            )
        terminate_process_tree(proc)

    def _reader_loop(self) -> None:
        proc = self.process
        if proc is None or proc.stdout is None:
            return
        line_count = 0
        diag_line_count = 0
        try:
            for raw_line in iter(proc.stdout.readline, ""):
                if raw_line == "":
                    break
                line_count += 1
                is_diagnostic = CARLA_DIAG_MARKER in raw_line
                if is_diagnostic:
                    diag_line_count += 1
                self._note_output(raw_line, diagnostic=is_diagnostic)
                if self._log_handle is not None:
                    self._log_handle.write(raw_line)
                    self._log_handle.flush()
                if self.emit_stdout:
                    if not self.show_carla_diag and is_diagnostic:
                        continue
                    if self.output_prefix:
                        sys.stdout.write(self.output_prefix + raw_line)
                    else:
                        sys.stdout.write(raw_line)
                    sys.stdout.flush()
        finally:
            if line_count:
                _perf_add("child_output_lines", line_count)
            if diag_line_count:
                _perf_add("child_diag_lines", diag_line_count)
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _watchdog_loop(self) -> None:
        if (self.timeout_s is None or self.timeout_s <= 0) and self.health_check is None:
            return
        while not self._stop_event.wait(1.0):
            proc = self.process
            if proc is None or proc.poll() is not None:
                return
            if self.health_check is not None:
                health_reason = self.health_check()
                if health_reason:
                    self._emit_tail()
                    self.terminate(health_reason)
                    return
            if self.timeout_s is None or self.timeout_s <= 0:
                continue
            elapsed = time.monotonic() - self._start_time
            if elapsed <= self.timeout_s:
                continue
            quiet_for = self._get_last_output_age_s()
            if quiet_for < self.stall_output_timeout_s:
                continue
            self._stalled.set()
            warning = (
                "[WARN] Process exceeded watchdog budget and produced no output for "
                f"{quiet_for:.1f}s. Terminating it as stuck."
            )
            if self.emit_stdout:
                print(warning)
            elif self._log_handle is not None:
                self._log_handle.write(warning + "\n")
                self._log_handle.flush()
            self._emit_tail()
            log_carla_event(
                "CHILD_WATCHDOG_STALL",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                command=" ".join(self.cmd),
                pid=None if proc is None else proc.pid,
                elapsed_s=elapsed,
                quiet_for_s=quiet_for,
                timeout_s=self.timeout_s,
                stall_output_timeout_s=self.stall_output_timeout_s,
            )
            self.terminate()
            return

    def run(self) -> MonitoredRunResult:
        _perf_add("child_runs", 1)
        self._start_time = time.monotonic()
        self._last_output_time = self._start_time
        self._last_any_output_time = self._start_time
        with _active_monitored_runners_lock:
            _active_monitored_runners.add(self)
        self._begin_run_log()
        log_carla_event(
            "CHILD_RUN_START",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            command=" ".join(self.cmd),
            cwd=self.cwd,
            timeout_s=self.timeout_s,
            log_path=None if self.log_path is None else str(self.log_path),
        )
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except Exception as exc:
            log_carla_event(
                "CHILD_RUN_SPAWN_FAIL",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                command=" ".join(self.cmd),
                cwd=self.cwd,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._finish_run_log(
                status="failed",
                returncode=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            with _active_monitored_runners_lock:
                _active_monitored_runners.discard(self)
            raise
        reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        reader_thread.start()
        watchdog_thread.start()
        try:
            try:
                returncode = self.process.wait()
            except BaseException as exc:
                if self.process.poll() is None:
                    self.terminate(
                        self._termination_reason
                        or f"[INFO] Stopping child process after {type(exc).__name__}."
                    )
                    try:
                        self.process.wait(timeout=5.0)
                    except Exception:
                        pass
                self._stop_event.set()
                # Close the read-end of stdout PIPE from the parent side so the
                # reader thread's readline() unblocks even when grandchild
                # processes still hold the write-end open.  The reader's own
                # finally block calls close() again which is a safe no-op.
                if self.process.stdout is not None:
                    try:
                        self.process.stdout.close()
                    except Exception:
                        pass
                reader_thread.join(timeout=5.0)
                watchdog_thread.join(timeout=5.0)
                self._finish_run_log(
                    status="stopped",
                    returncode=self.process.poll(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                _perf_add("child_run_time_s", time.monotonic() - self._start_time)
                raise

            self._stop_event.set()
            # Same rationale as the exception path above: close stdout before
            # joining so the reader thread is guaranteed to drain and exit.
            if self.process.stdout is not None:
                try:
                    self.process.stdout.close()
                except Exception:
                    pass
            reader_thread.join(timeout=5.0)
            watchdog_thread.join(timeout=5.0)
            elapsed_s = time.monotonic() - self._start_time
            self._finish_run_log(
                status=self._derive_run_log_status(returncode),
                returncode=returncode,
            )
            log_carla_event(
                "CHILD_RUN_END",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                pid=None if self.process is None else self.process.pid,
                command=" ".join(self.cmd),
                returncode=returncode,
                elapsed_s=elapsed_s,
                stalled=bool(self._stalled.is_set()),
                terminated_reason=self._termination_reason,
            )
            _perf_add("child_run_time_s", elapsed_s)
            return MonitoredRunResult(
                returncode=returncode,
                stalled=self._stalled.is_set(),
                elapsed_s=elapsed_s,
                last_output_age_s=self._get_last_output_age_s(),
                output_tail=self._get_output_tail(),
                terminated_reason=self._termination_reason,
            )
        finally:
            with _active_monitored_runners_lock:
                _active_monitored_runners.discard(self)


class CarlaProcessManager:
    def __init__(
        self,
        carla_root: Path,
        host: str,
        port: int,
        tm_port_offset: int,
        extra_args: List[str],
        env: Dict[str, str],
        port_tries: int,
        port_step: int,
        server_log_path: Path | None = None,
        rootcause_config: CarlaRootCauseConfig | None = None,
    ) -> None:
        self.carla_root = carla_root
        self.host = host
        self.port = port
        self.tm_port_offset = tm_port_offset
        self.tm_port = self.port + self.tm_port_offset
        self.extra_args = list(extra_args)
        self.env = env
        self.port_tries = port_tries
        self.port_step = port_step
        self.rootcause_config = rootcause_config
        self.server_log_path = server_log_path
        if self.rootcause_config is not None and self.server_log_path is not None:
            print(
                "[WARN] Ignoring --carla-forensics-server-log-file because --carla-rootcause "
                "keeps CARLA server logs inside the root-cause logdir."
            )
            self.server_log_path = None
        self.process: subprocess.Popen | None = None
        self._server_log_handle: TextIO | None = None
        self.rootcause_last_logdir: Path | None = None
        self._rootcause_launch_count = 0
        self._pending_logdir_outcome: str | None = None
        self._stopped = False

    def plan_logdir_outcome(self, outcome: str) -> None:
        """Schedule the current rootcause logdir to be renamed with *outcome* when stop() is called."""
        self._pending_logdir_outcome = outcome
        log_carla_event(
            "ROOTCAUSE_OUTCOME_PLANNED",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            outcome=outcome,
            logdir=None if self.rootcause_last_logdir is None else str(self.rootcause_last_logdir),
        )

    def _annotate_current_logdir(self) -> None:
        """Rename rootcause_last_logdir by appending the pending outcome (or SUCCESS if none set)."""
        logdir = self.rootcause_last_logdir
        if logdir is None or self.rootcause_config is None:
            self._pending_logdir_outcome = None
            return
        if not logdir.exists():
            self._pending_logdir_outcome = None
            return
        outcome = self._pending_logdir_outcome if self._pending_logdir_outcome is not None else "SUCCESS"
        self._pending_logdir_outcome = None
        safe_outcome = re.sub(r"[^A-Za-z0-9_]", "_", outcome)
        candidate = logdir.parent / f"{logdir.name}_{safe_outcome}"
        counter = 1
        while candidate.exists():
            candidate = logdir.parent / f"{logdir.name}_{safe_outcome}_{counter:02d}"
            counter += 1
        try:
            logdir.rename(candidate)
            self.rootcause_last_logdir = candidate
            events_log_path = set_connection_events_logdir(
                candidate,
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            )
            self.env["CARLA_ROOTCAUSE_LOGDIR"] = str(candidate)
            self.env["CARLA_CONNECTION_EVENTS_LOG"] = str(events_log_path)
            print(f"[INFO] Rootcause logdir annotated: {logdir.name!r} -> {candidate.name!r}")
            log_carla_event(
                "ROOTCAUSE_LOGDIR_ANNOTATED",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                source=str(logdir),
                target=str(candidate),
                outcome=safe_outcome,
                events_log=str(events_log_path),
            )
        except OSError as exc:
            print(f"[WARN] Could not rename rootcause logdir {logdir}: {exc}")
            log_carla_event(
                "ROOTCAUSE_LOGDIR_ANNOTATE_FAIL",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                source=str(logdir),
                target=str(candidate),
                outcome=safe_outcome,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _read_rootcause_run_meta(run_meta_path: Path) -> Dict[str, str]:
        if not run_meta_path.exists():
            return {}
        values: Dict[str, str] = {}
        try:
            for raw_line in run_meta_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        except Exception:
            return {}
        return values

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _next_rootcause_logdir(self, bundle: PortBundle) -> Path:
        config = self.rootcause_config
        if config is None:
            raise RuntimeError("Root-cause configuration is not set.")
        self._rootcause_launch_count += 1
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (
            f"rootcause_{timestamp}_p{bundle.port}_tm{bundle.tm_port}_"
            f"launch{self._rootcause_launch_count:03d}"
        )
        base_dir = Path(config.log_root)
        candidate = base_dir / prefix
        suffix = 1
        while candidate.exists():
            candidate = base_dir / f"{prefix}_{suffix:02d}"
            suffix += 1
        return candidate

    def _cleanup_stale_port_owners(self, bundle: PortBundle, *, grace_s: float = 15.0) -> bool:
        ports = carla_service_ports(bundle.port, self.tm_port_offset)
        cleaned, unmatched = cleanup_listening_processes(
            host=self.host,
            ports=ports,
            service_name="CARLA",
            match_tokens=CARLA_PROCESS_MATCH_TOKENS,
            grace_s=grace_s,
        )
        for detail in unmatched:
            print(
                "[WARN] Port conflict while preparing CARLA, but listener does not look like CARLA: "
                + detail
            )
        return cleaned

    def _launch_bundle_with_rootcause(self, bundle: PortBundle) -> bool:
        config = self.rootcause_config
        if config is None:
            raise RuntimeError("Root-cause configuration is not set.")
        if not config.script_path.exists():
            raise FileNotFoundError(config.script_path)

        rootcause_logdir = self._next_rootcause_logdir(bundle)
        rootcause_logdir.mkdir(parents=True, exist_ok=True)
        self.rootcause_last_logdir = rootcause_logdir
        events_log_path = set_connection_events_logdir(
            rootcause_logdir,
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        )
        self.env["CARLA_ROOTCAUSE_LOGDIR"] = str(rootcause_logdir)
        self.env["CARLA_CONNECTION_EVENTS_LOG"] = str(events_log_path)
        log_carla_event(
            "PROCESS_START",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            context="rootcause_launch",
        )
        log_carla_event(
            "PROCESS_ENV",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            rootcause_logdir=str(rootcause_logdir),
            events_log=str(events_log_path),
            host=self.host,
            requested_port=bundle.port,
            requested_tm_port=bundle.tm_port,
        )
        log_carla_event(
            "ROOTCAUSE_LAUNCH_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            requested_port=bundle.port,
            requested_tm_port=bundle.tm_port,
            logdir=str(rootcause_logdir),
            script=str(config.script_path),
        )
        wrapper_console_log = rootcause_logdir / "rootcause_wrapper_stdout.log"
        run_meta_path = rootcause_logdir / "run_meta.txt"

        cmd = [
            str(config.script_path),
            "--carla-root",
            str(self.carla_root),
            "--logdir",
            str(rootcause_logdir),
            "--port",
            str(bundle.port),
            "--traffic-manager-port",
            str(bundle.tm_port),
            "--port-tries",
            str(max(1, int(self.port_tries))),
            "--port-step",
            str(max(1, int(self.port_step))),
            "--mode",
            str(config.mode),
            "--diag-interval",
            str(max(1, int(config.diag_interval))),
            "--startup-timeout",
            str(max(1, int(config.startup_timeout_s))),
            "--core-wait-seconds",
            str(max(0, int(config.core_wait_seconds))),
        ]
        gpu_value = str(self.env.get("CUDA_VISIBLE_DEVICES", "")).strip()
        if gpu_value:
            cmd.extend(["--gpu", gpu_value])
        carla_extra_args = list(self.extra_args)
        if should_add_offscreen(carla_extra_args):
            carla_extra_args.append("-RenderOffScreen")
        if carla_extra_args:
            cmd.extend(["--", *carla_extra_args])

        print(f"[INFO] Starting CARLA via root-cause wrapper: {' '.join(cmd)}")
        print(f"[INFO] Root-cause logdir: {rootcause_logdir}")

        popen_kwargs: Dict[str, object] = {}
        self._server_log_handle = wrapper_console_log.open("a", encoding="utf-8")
        self._server_log_handle.write(
            f"\n=== ROOTCAUSE WRAPPER START {dt.datetime.now().isoformat(timespec='seconds')} "
            f"requested_port={bundle.port} requested_tm_port={bundle.tm_port} ===\n"
        )
        self._server_log_handle.flush()
        popen_kwargs["stdout"] = self._server_log_handle
        popen_kwargs["stderr"] = subprocess.STDOUT

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.carla_root.parent),
                env=self.env,
                start_new_session=True,
                **popen_kwargs,
            )
        except Exception:
            if self._server_log_handle is not None:
                self._server_log_handle.close()
                self._server_log_handle = None
            raise
        log_carla_event(
            "ROOTCAUSE_LAUNCH_SPAWNED",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            pid=None if self.process is None else self.process.pid,
            requested_port=bundle.port,
            requested_tm_port=bundle.tm_port,
            logdir=str(rootcause_logdir),
            cmd=" ".join(cmd),
        )

        selected_port = int(bundle.port)
        selected_tm_port = int(bundle.tm_port)
        selected_reason = ""
        startup_wait_timeout = max(
            float(CARLA_STARTUP_TIMEOUT),
            float(max(1, int(config.startup_timeout_s))) + 10.0,
        )
        deadline = time.monotonic() + startup_wait_timeout
        while time.monotonic() < deadline:
            proc = self.process
            if proc is None:
                break
            if proc.poll() is not None:
                break
            meta = self._read_rootcause_run_meta(run_meta_path)
            if meta:
                meta_port = self._parse_int(meta.get("selected_port") or meta.get("port"))
                meta_tm_port = self._parse_int(meta.get("selected_tm_port") or meta.get("tm_port"))
                if meta_port is not None:
                    selected_port = meta_port
                if meta_tm_port is not None:
                    selected_tm_port = meta_tm_port
                selected_reason = str(
                    meta.get("port_selection_reason") or meta.get("selection_reason") or ""
                ).strip()
            if is_port_open(self.host, selected_port, timeout=1.0):
                self.port = int(selected_port)
                self.tm_port = int(selected_tm_port)
                if self.port != bundle.port or self.tm_port != bundle.tm_port:
                    print(
                        "[INFO] Root-cause wrapper switched CARLA ports from "
                        f"{bundle.port}/{bundle.tm_port} to {self.port}/{self.tm_port}."
                    )
                if selected_reason:
                    print(f"[INFO] Root-cause port selection reason: {selected_reason}")
                log_carla_event(
                    "ROOTCAUSE_LAUNCH_READY",
                    process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                    pid=None if self.process is None else self.process.pid,
                    host=self.host,
                    selected_port=self.port,
                    selected_tm_port=self.tm_port,
                    selected_reason=selected_reason,
                    logdir=str(rootcause_logdir),
                )
                return True
            time.sleep(0.5)

        returncode = None if self.process is None else self.process.poll()
        meta = self._read_rootcause_run_meta(run_meta_path)
        selected_port_text = meta.get("selected_port") or meta.get("port") or str(bundle.port)
        selected_port_int = self._parse_int(selected_port_text)
        if selected_port_int is None:
            selected_port_int = int(bundle.port)
        selected_tm_text = (
            meta.get("selected_tm_port")
            or meta.get("tm_port")
            or str(int(selected_port_int) + int(self.tm_port_offset))
        )
        selected_reason = str(meta.get("port_selection_reason") or meta.get("selection_reason") or "").strip()
        log_excerpt = ""
        try:
            log_excerpt = "\n".join(
                wrapper_console_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            )
        except Exception:
            log_excerpt = ""
        print(
            "[WARN] Root-cause CARLA failed to open RPC port "
            f"(requested={bundle.port}, selected={selected_port_text}, selected_tm={selected_tm_text}, "
            f"reason={selected_reason or 'unknown'}) within {startup_wait_timeout:.0f}s"
            + (
                f" (exit_code={returncode})."
                if returncode is not None
                else "."
            )
        )
        if log_excerpt:
            print("[WARN] Root-cause wrapper tail:")
            for line in log_excerpt.splitlines():
                print(f"[TAIL] {line}")
            log_carla_event(
                "ROOTCAUSE_LAUNCH_TIMEOUT",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                pid=None if self.process is None else self.process.pid,
                host=self.host,
                requested_port=bundle.port,
                selected_port=selected_port_int,
                selected_tm_port=selected_tm_text,
                selection_reason=selected_reason,
                startup_wait_timeout_s=startup_wait_timeout,
                wrapper_exit_code=returncode,
            )
        self.stop(timeout=5.0)
        return False

    def _launch_bundle(self, bundle: PortBundle) -> bool:
        if self.rootcause_config is not None:
            return self._launch_bundle_with_rootcause(bundle)
        carla_script = self.carla_root / "CarlaUE4.sh"
        cmd = [str(carla_script), f"--world-port={bundle.port}"]
        if should_add_offscreen(self.extra_args):
            cmd.append("-RenderOffScreen")
        cmd.extend(self.extra_args)
        print(
            f"[CARLA START] host={self.host} port={bundle.port} tm_port={bundle.tm_port}"
        )
        print(f"[INFO] Starting CARLA: {' '.join(cmd)}")
        log_carla_event(
            "CARLA_LAUNCH_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=bundle.port,
            tm_port=bundle.tm_port,
            cmd=" ".join(cmd),
            rootcause_mode=0,
        )

        popen_kwargs: Dict[str, object] = {}
        if self.server_log_path is not None:
            self.server_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._server_log_handle = self.server_log_path.open("a", encoding="utf-8")
            self._server_log_handle.write(
                f"\n=== CARLA START {dt.datetime.now().isoformat(timespec='seconds')} "
                f"port={bundle.port} tm_port={bundle.tm_port} ===\n"
            )
            self._server_log_handle.flush()
            popen_kwargs["stdout"] = self._server_log_handle
            popen_kwargs["stderr"] = subprocess.STDOUT

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.carla_root),
                env=self.env,
                start_new_session=True,
                **popen_kwargs,
            )
        except Exception:
            if self._server_log_handle is not None:
                self._server_log_handle.close()
                self._server_log_handle = None
            raise
        log_carla_event(
            "CARLA_LAUNCH_SPAWNED",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=bundle.port,
            tm_port=bundle.tm_port,
            pid=None if self.process is None else self.process.pid,
        )

        if not wait_for_process_port(
            self.process,
            self.host,
            bundle.port,
            timeout=CARLA_STARTUP_TIMEOUT,
        ):
            returncode = self.process.poll()
            print(
                f"[WARN] CARLA failed to open port {bundle.port} within "
                f"{CARLA_STARTUP_TIMEOUT:.0f}s"
                + (
                    f" (exit_code={returncode})."
                    if returncode is not None
                    else "."
                )
            )
            log_carla_event(
                "CARLA_LAUNCH_PORT_FAIL",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                host=self.host,
                port=bundle.port,
                tm_port=bundle.tm_port,
                pid=None if self.process is None else self.process.pid,
                startup_timeout_s=CARLA_STARTUP_TIMEOUT,
                returncode=returncode,
            )
            self.stop(timeout=5.0)
            return False

        # TCP port is open.  Verify the RPC layer is actually responding before
        # declaring startup successful — opening the socket does not mean
        # CARLA's UE4 RPC server is ready to service get_world() calls.
        rpc_timeout = max(30.0, CARLA_STARTUP_TIMEOUT)
        if not wait_for_carla_rpc_ready(self.host, bundle.port, timeout=rpc_timeout):
            print(
                f"[WARN] CARLA port {bundle.port} is open but RPC not ready "
                f"within {rpc_timeout:.0f}s; treating as failed startup."
            )
            log_carla_event(
                "CARLA_LAUNCH_RPC_FAIL",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                host=self.host,
                port=bundle.port,
                tm_port=bundle.tm_port,
                pid=None if self.process is None else self.process.pid,
                rpc_timeout_s=rpc_timeout,
            )
            self.stop(timeout=5.0)
            return False

        self.port = bundle.port
        self.tm_port = bundle.tm_port
        log_carla_event(
            "CARLA_LAUNCH_READY",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=self.port,
            tm_port=self.tm_port,
            pid=None if self.process is None else self.process.pid,
            rootcause_mode=0,
        )
        return True

    def start(self) -> PortBundle:
        start_t = time.monotonic()
        _perf_add("carla_start_calls", 1)
        self._stopped = False
        log_carla_event(
            "CARLA_MANAGER_START_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            requested_port=self.port,
            requested_tm_port=self.tm_port,
            rootcause_mode=1 if self.rootcause_config is not None else 0,
        )
        if self.is_running():
            _perf_add("carla_start_time_s", time.monotonic() - start_t)
            return PortBundle(port=self.port, tm_port=self.tm_port)
        if self.rootcause_config is not None:
            desired_bundle = PortBundle(port=self.port, tm_port=self.tm_port)
            if self._launch_bundle(desired_bundle):
                log_carla_event(
                    "CARLA_MANAGER_START_READY",
                    process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                    host=self.host,
                    selected_port=self.port,
                    selected_tm_port=self.tm_port,
                    rootcause_mode=1,
                )
                _perf_add("carla_start_time_s", time.monotonic() - start_t)
                return PortBundle(port=self.port, tm_port=self.tm_port)
            _perf_add("carla_start_time_s", time.monotonic() - start_t)
            raise RuntimeError(
                "CARLA root-cause wrapper failed to start the CARLA server "
                f"(requested {desired_bundle.port}/{desired_bundle.tm_port})."
            )
        carla_script = self.carla_root / "CarlaUE4.sh"
        if not carla_script.exists():
            raise FileNotFoundError(carla_script)
        desired_bundle = PortBundle(port=self.port, tm_port=self.tm_port)
        desired_ports = carla_service_ports(desired_bundle.port, self.tm_port_offset)
        candidate_bundles: List[PortBundle] = []
        if are_ports_available(self.host, desired_ports):
            candidate_bundles.append(desired_bundle)
        elif self._cleanup_stale_port_owners(desired_bundle, grace_s=10.0) and are_ports_available(
            self.host, desired_ports
        ):
            print(
                "[INFO] Freed stale CARLA listener(s) on "
                f"{desired_bundle.port}/{desired_bundle.tm_port} before startup."
            )
            candidate_bundles.append(desired_bundle)

        additional_needed = max(0, int(self.port_tries) - len(candidate_bundles))
        if additional_needed > 0:
            additional_bundles = find_available_port_bundles(
                self.host,
                desired_bundle.port,
                tries=max(1, self.port_tries),
                step=self.port_step,
                bundle_count=additional_needed,
                tm_port_offset=self.tm_port_offset,
                reserved_ports={
                    port
                    for bundle in candidate_bundles
                    for port in carla_service_ports(bundle.port, self.tm_port_offset)
                },
            )
            for bundle in additional_bundles:
                if bundle not in candidate_bundles:
                    candidate_bundles.append(bundle)

        last_error = None
        for bundle in candidate_bundles:
            if bundle != desired_bundle:
                print(
                    "[INFO] CARLA port set is busy or failed previously. Switching from "
                    f"{desired_bundle.port}/{desired_bundle.tm_port} to "
                    f"{bundle.port}/{bundle.tm_port}."
                )
            if self._launch_bundle(bundle):
                log_carla_event(
                    "CARLA_MANAGER_START_READY",
                    process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                    host=self.host,
                    selected_port=self.port,
                    selected_tm_port=self.tm_port,
                    rootcause_mode=0,
                )
                _perf_add("carla_start_time_s", time.monotonic() - start_t)
                return PortBundle(port=self.port, tm_port=self.tm_port)
            last_error = (
                f"CARLA failed to open RPC port on {bundle.port}/{bundle.tm_port}. "
                "Trying the next free port set."
            )
            wait_for_port_close(self.host, bundle.port, timeout=5.0)

        _perf_add("carla_start_time_s", time.monotonic() - start_t)
        raise RuntimeError(
            last_error
            or "CARLA failed to start on all candidate port sets."
        )

    def stop(self, timeout: float = 15.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        current_bundle = PortBundle(port=self.port, tm_port=self.tm_port)
        log_carla_event(
            "CARLA_MANAGER_STOP_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=current_bundle.port,
            tm_port=current_bundle.tm_port,
            pid=None if self.process is None else self.process.pid,
            timeout_s=float(timeout),
            rootcause_mode=1 if self.rootcause_config is not None else 0,
        )
        if self.process is None:
            if self._server_log_handle is not None:
                self._server_log_handle.close()
                self._server_log_handle = None
        elif self.process.poll() is None:
            print("[INFO] Stopping CARLA...")
            terminate_process_tree(self.process, grace_s=timeout)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._server_log_handle is not None:
            if self.rootcause_config is None:
                self._server_log_handle.write(
                    f"=== CARLA STOP {dt.datetime.now().isoformat(timespec='seconds')} ===\n"
                )
            self._server_log_handle.flush()
            self._server_log_handle.close()
            self._server_log_handle = None
        if self.rootcause_config is not None:
            self._annotate_current_logdir()
        self._cleanup_stale_port_owners(current_bundle, grace_s=max(5.0, timeout))
        log_carla_event(
            "CARLA_MANAGER_STOP_END",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=current_bundle.port,
            tm_port=current_bundle.tm_port,
            rootcause_mode=1 if self.rootcause_config is not None else 0,
        )
        self.process = None

    def restart(self, wait_seconds: float, *, failure_reason: str | None = None) -> PortBundle:
        restart_t = time.monotonic()
        _perf_add("carla_restart_calls", 1)
        log_carla_event(
            "CARLA_MANAGER_RESTART_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            port=self.port,
            tm_port=self.tm_port,
            wait_seconds=float(wait_seconds),
            failure_reason=failure_reason,
            rootcause_mode=1 if self.rootcause_config is not None else 0,
        )
        if self.rootcause_config is not None and failure_reason is not None:
            self.plan_logdir_outcome(f"FAILED_{failure_reason}")
        self.stop()
        wait_for_port_close(self.host, self.port, timeout=10.0)
        time.sleep(max(0.0, wait_seconds))
        bundle = self.start()
        log_carla_event(
            "CARLA_MANAGER_RESTART_END",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            host=self.host,
            selected_port=bundle.port,
            selected_tm_port=bundle.tm_port,
            rootcause_mode=1 if self.rootcause_config is not None else 0,
        )
        _perf_add("carla_restart_time_s", time.monotonic() - restart_t)
        return bundle


class BackgroundServiceProcess:
    def __init__(
        self,
        *,
        name: str,
        cmd: Sequence[str],
        env: Dict[str, str],
        cwd: Path,
        host: str,
        port: int,
        startup_timeout_s: float,
        log_path: Path,
        match_tokens: Sequence[str],
    ) -> None:
        self.name = str(name)
        self.cmd = list(cmd)
        self.env = dict(env)
        self.cwd = str(cwd)
        self.host = str(host)
        self.port = int(port)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.log_path = Path(log_path)
        self.match_tokens = tuple(str(token) for token in match_tokens)
        self.process: subprocess.Popen[Any] | None = None
        self._log_handle: TextIO | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self) -> None:
        if self.is_running():
            return
        log_carla_event(
            "AUX_SERVICE_START_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            service=self.name,
            host=self.host,
            port=self.port,
            cwd=self.cwd,
            command=" ".join(self.cmd),
        )
        if is_port_open(self.host, self.port, timeout=1.0):
            cleaned, unmatched = cleanup_listening_processes(
                host=self.host,
                ports=(self.port,),
                service_name=self.name,
                match_tokens=self.match_tokens,
                grace_s=10.0,
            )
            if cleaned and not is_port_open(self.host, self.port, timeout=1.0):
                print(
                    f"[INFO] Freed stale {self.name} listener(s) on port {self.port} before startup."
                )
            else:
                detail = ""
                if unmatched:
                    detail = " Conflicting listener(s): " + "; ".join(unmatched)
                raise RuntimeError(
                    f"{self.name} port {self.port} is already in use. "
                    "Choose a different port or stop the existing service."
                    + detail
                )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        self._log_handle.write(
            f"\n=== {self.name} START {dt.datetime.now().isoformat(timespec='seconds')} "
            f"port={self.port} ===\n"
        )
        self._log_handle.write("COMMAND: " + " ".join(self.cmd) + "\n")
        self._log_handle.flush()

        print(f"[INFO] Starting {self.name} on port {self.port}: {' '.join(self.cmd)}")
        try:
            self.process = subprocess.Popen(
                self.cmd,
                cwd=self.cwd,
                env=self.env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            raise
        log_carla_event(
            "AUX_SERVICE_SPAWNED",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            service=self.name,
            host=self.host,
            port=self.port,
            pid=None if self.process is None else self.process.pid,
        )

        if not wait_for_process_port(
            self.process,
            self.host,
            self.port,
            timeout=self.startup_timeout_s,
        ):
            returncode = self.process.poll()
            self.stop(timeout=5.0)
            detail = (
                f"process exited with code {returncode}"
                if returncode is not None
                else f"port did not open within {self.startup_timeout_s:.0f}s"
            )
            raise RuntimeError(
                f"{self.name} failed to start: {detail}. "
                f"See {self.log_path}."
            )
        log_carla_event(
            "AUX_SERVICE_READY",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            service=self.name,
            host=self.host,
            port=self.port,
            pid=None if self.process is None else self.process.pid,
        )

    def stop(self, timeout: float = 15.0) -> None:
        log_carla_event(
            "AUX_SERVICE_STOP_BEGIN",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            service=self.name,
            host=self.host,
            port=self.port,
            pid=None if self.process is None else self.process.pid,
            timeout_s=float(timeout),
        )
        if self.process is not None and self.process.poll() is None:
            print(f"[INFO] Stopping {self.name}...")
            terminate_process_tree(self.process, grace_s=timeout)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        if self._log_handle is not None:
            self._log_handle.write(
                f"=== {self.name} STOP {dt.datetime.now().isoformat(timespec='seconds')} ===\n"
            )
            self._log_handle.flush()
            self._log_handle.close()
            self._log_handle = None
        cleanup_listening_processes(
            host=self.host,
            ports=(self.port,),
            service_name=self.name,
            match_tokens=self.match_tokens,
            grace_s=max(5.0, timeout),
        )
        log_carla_event(
            "AUX_SERVICE_STOP_END",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            service=self.name,
            host=self.host,
            port=self.port,
        )
        self.process = None


class ColmDriverVLLMManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        base_env: Dict[str, str],
        conda_env: str,
        reservation: ColmDriverVLLMReservation,
        llm_port: int,
        vlm_port: int,
        startup_timeout_s: float,
        log_dir: Path,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.base_env = _build_conda_launch_env(base_env)
        self.conda_env = str(conda_env)
        self.reservation = reservation
        self.llm_port = int(llm_port)
        self.vlm_port = int(vlm_port)
        self.startup_timeout_s = max(1.0, float(startup_timeout_s))
        self.log_dir = Path(log_dir)
        self.host = "127.0.0.1"
        self.vlm_service = self._build_service(
            name="CoLMDriver VLM vLLM",
            model_rel=COLMDRIVER_VLM_MODEL_DEFAULT,
            max_model_len=COLMDRIVER_VLM_MAX_MODEL_LEN_DEFAULT,
            gpu=self.reservation.vlm_gpu,
            port=self.vlm_port,
            log_name="colmdriver_vlm_vllm.log",
        )
        self.llm_service = self._build_service(
            name="CoLMDriver LLM vLLM",
            model_rel=COLMDRIVER_LLM_MODEL_DEFAULT,
            max_model_len=COLMDRIVER_LLM_MAX_MODEL_LEN_DEFAULT,
            gpu=self.reservation.llm_gpu,
            port=self.llm_port,
            log_name="colmdriver_llm_vllm.log",
        )

    def _build_service(
        self,
        *,
        name: str,
        model_rel: str,
        max_model_len: int,
        gpu: str,
        port: int,
        log_name: str,
    ) -> BackgroundServiceProcess:
        model_path = (self.repo_root / model_rel).resolve()
        if not model_path.exists():
            raise FileNotFoundError(
                f"Required CoLMDriver model checkpoint not found: {model_path}"
            )
        env = dict(self.base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.conda_env,
            "vllm",
            "serve",
            str(model_path),
            "--port",
            str(port),
            "--max-model-len",
            str(int(max_model_len)),
            "--trust-remote-code",
            "--enable-prefix-caching",
        ]
        return BackgroundServiceProcess(
            name=name,
            cmd=cmd,
            env=env,
            cwd=self.repo_root,
            host=self.host,
            port=int(port),
            startup_timeout_s=self.startup_timeout_s,
            log_path=self.log_dir / log_name,
            match_tokens=VLLM_PROCESS_MATCH_TOKENS,
        )

    def start(self) -> None:
        if self.llm_port == self.vlm_port:
            raise ValueError("--llm-port and --vlm-port must be different.")
        try:
            self.vlm_service.start()
            self.llm_service.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self.llm_service.stop()
        self.vlm_service.stop()


_active_carla_manager: CarlaProcessManager | None = None
_active_crosswalk_worker: "UCLAV2CrosswalkWorker | None" = None
_active_colmdriver_vllm_manager: ColmDriverVLLMManager | None = None
_active_monitored_runners: set[MonitoredSubprocessRunner] = set()
_active_monitored_runners_lock = threading.Lock()
_signal_handlers_installed = False


def _cleanup_active_managers() -> None:
    global _active_carla_manager, _active_crosswalk_worker, _active_colmdriver_vllm_manager
    log_carla_event("LAUNCHER_CLEANUP_BEGIN", process_name=RUN_CUSTOM_EVAL_PROCESS_NAME)
    with _active_monitored_runners_lock:
        active_runners = list(_active_monitored_runners)
    for runner in active_runners:
        runner.terminate("[INFO] Stopping active child process during launcher shutdown.")
    if _active_carla_manager is not None:
        _active_carla_manager.stop()
        _active_carla_manager = None
    if _active_crosswalk_worker is not None:
        _active_crosswalk_worker.stop()
        _active_crosswalk_worker = None
    if _active_colmdriver_vllm_manager is not None:
        _active_colmdriver_vllm_manager.stop()
        _active_colmdriver_vllm_manager = None
    log_carla_event("LAUNCHER_CLEANUP_END", process_name=RUN_CUSTOM_EVAL_PROCESS_NAME)


atexit.register(_cleanup_active_managers)


def _install_signal_handlers() -> None:
    global _signal_handlers_installed
    if _signal_handlers_installed:
        return

    def _handler(signum: int, frame: object | None) -> None:
        del frame
        log_carla_event(
            "LAUNCHER_SIGNAL_HANDLER",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            signal=signum,
            signal_name=getattr(signal.Signals(signum), "name", str(signum)),
        )
        _cleanup_active_managers()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    _signal_handlers_installed = True


def count_variants(args: argparse.Namespace) -> int:
    if args.weather:
        return 1
    count = 1
    if args.add_night:
        count += 1
    if args.add_cloudy:
        count += 1
    if args.add_npcs:
        count += 1
    return count


def count_negotiation_modes(args: argparse.Namespace) -> int:
    return 2 if args.add_no_negotiation else 1


def estimate_scenario_timeout(
    args: argparse.Namespace,
    routes_dir: Path,
    *,
    planner_name: str | None = None,
    run_matrix_size: int | None = None,
) -> float:
    ego_routes = detect_ego_routes(
        routes_dir,
        replay_mode=((planner_name or args.planner) == "log-replay"),
    )
    route_count = max(1, ego_routes)
    if run_matrix_size is None:
        run_matrix_size = count_variants(args) * count_negotiation_modes(args)
    expected_runs = max(1, args.repetitions) * max(1, int(run_matrix_size))
    base = float(args.timeout) * route_count * expected_runs
    return base * float(args.carla_crash_multiplier)


def estimate_planner_child_timeout(
    args: argparse.Namespace,
    routes_dir: Path,
    *,
    planner_name: str | None = None,
    retry_limit: int | None = None,
) -> float | None:
    timeout_s = estimate_scenario_timeout(args, routes_dir, planner_name=planner_name)
    normalized_retry_limit = normalize_retry_limit(
        args.scenario_retry_limit if retry_limit is None else retry_limit
    )
    if normalized_retry_limit is None:
        return None
    timeout_s *= retry_limit_multiplier(normalized_retry_limit)
    timeout_s += max(60.0, float(args.timeout) * 0.25)
    return timeout_s


def append_pythonpath(env: Dict[str, str], path: Path) -> None:
    path_str = str(path)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{path_str}:{current}" if current else path_str


def sanitize_run_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def parse_planner_requests(raw_values: Sequence[str] | None) -> List[PlannerRequest]:
    requests: List[PlannerRequest] = []
    for raw in raw_values or []:
        token = str(raw).strip()
        if not token:
            continue
        if token.startswith("[") and token.endswith("]"):
            parts = [part.strip() for part in token[1:-1].split(",")]
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "Planner list entries must be formatted as [planner,conda_env]."
                )
            planner_name, conda_env = parts
        else:
            planner_name, conda_env = token, None
        if planner_name not in PLANNER_SPECS:
            raise ValueError(
                f"Unknown planner '{planner_name}'. "
                f"Available planners: {', '.join(sorted(PLANNER_SPECS))}"
            )
        requests.append(PlannerRequest(name=planner_name, conda_env=conda_env))

    if not requests:
        requests.append(PlannerRequest(name=DEFAULT_PLANNER))

    name_counts: dict[str, int] = {}
    for req in requests:
        name_counts[req.name] = name_counts.get(req.name, 0) + 1

    run_id_counts: dict[str, int] = {}
    finalized: List[PlannerRequest] = []
    for req in requests:
        base_id = req.name
        if req.conda_env and name_counts.get(req.name, 0) > 1:
            base_id = f"{req.name}_{req.conda_env}"
        base_id = sanitize_run_id(base_id)
        run_id_counts[base_id] = run_id_counts.get(base_id, 0) + 1
        run_id = base_id if run_id_counts[base_id] == 1 else f"{base_id}_{run_id_counts[base_id]}"
        finalized.append(
            PlannerRequest(name=req.name, conda_env=req.conda_env, run_id=run_id)
        )
    return finalized


def parse_gpu_requests(raw_values: Sequence[str] | None) -> List[str]:
    gpus: List[str] = []
    for raw in raw_values or []:
        for token in str(raw).split(","):
            gpu = token.strip()
            if gpu:
                gpus.append(gpu)
    return gpus


def planner_requests_need_colmdriver_vllm(
    planner_requests: Sequence[PlannerRequest],
) -> bool:
    return any(req.name in COLMDRIVER_VLLM_PLANNERS for req in planner_requests)


def query_gpu_memory_stats(
    requested_gpus: Sequence[str],
) -> Dict[str, GPUMemoryStats]:
    gpu_tokens = [str(gpu).strip() for gpu in requested_gpus if str(gpu).strip()]
    if not gpu_tokens:
        return {}

    all_stats = _query_all_gpu_memory_stats(include_uuid_keys=True)
    if not all_stats:
        return {}

    stats_by_requested: Dict[str, GPUMemoryStats] = {}
    for requested_gpu in gpu_tokens:
        stats = all_stats.get(requested_gpu)
        if stats is not None:
            stats_by_requested[requested_gpu] = stats

    # Fallback: when GPU IDs are remapped (e.g., container-visible IDs), match
    # by order against the sorted physical indices that nvidia-smi reports.
    if not stats_by_requested:
        index_only = sorted(
            {
                stats.index: stats
                for key, stats in all_stats.items()
                if key == stats.index
            }.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0],
        )
        if len(index_only) >= len(gpu_tokens):
            for requested_gpu, (_idx, stats) in zip(gpu_tokens, index_only):
                stats_by_requested[requested_gpu] = stats

    return stats_by_requested


def _query_all_gpu_memory_stats(
    *,
    include_uuid_keys: bool = False,
    force_refresh: bool = False,
) -> Dict[str, GPUMemoryStats]:
    """Query VRAM stats for every GPU visible to nvidia-smi."""
    cache_ttl_s = max(0.0, float(_GPU_QUERY_CACHE_TTL_S))
    cache_key = "all_gpu_memory_stats"
    now = time.monotonic()
    if not force_refresh and cache_ttl_s > 0.0:
        with _GPU_QUERY_CACHE_LOCK:
            cached = _GPU_QUERY_CACHE.get(cache_key)
        if cached is not None:
            cached_ts, cached_payload = cached
            if now - cached_ts <= cache_ttl_s:
                if include_uuid_keys:
                    return dict(cached_payload)
                return {
                    key: value
                    for key, value in cached_payload.items()
                    if key == value.index
                }

    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    query_start = time.monotonic()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10.0)
    except Exception:
        _perf_add("nvidia_smi_calls", 1)
        _perf_add("nvidia_smi_time_s", time.monotonic() - query_start)
        return {}
    _perf_add("nvidia_smi_calls", 1)
    _perf_add("nvidia_smi_time_s", time.monotonic() - query_start)
    stats: Dict[str, GPUMemoryStats] = {}
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        index, uuid, total_mib, used_mib, free_mib = parts[:5]
        try:
            gpu_stats = GPUMemoryStats(
                index=index,
                total_mib=int(total_mib),
                used_mib=int(used_mib),
                free_mib=int(free_mib),
            )
            stats[index] = gpu_stats
            stats[uuid] = gpu_stats
        except ValueError:
            continue

    with _GPU_QUERY_CACHE_LOCK:
        _GPU_QUERY_CACHE[cache_key] = (time.monotonic(), dict(stats))

    if include_uuid_keys:
        return stats
    return {
        key: value
        for key, value in stats.items()
        if key == value.index
    }


def _pick_fallback_gpu(
    current_gpu: str | None,
    all_gpus: Sequence[str],
    min_free_mib: int = 8192,
) -> str | None:
    """Query live VRAM and return the GPU with the most free memory.

    First searches within *all_gpus*.  If no better candidate is found there
    (e.g. *all_gpus* only contains the one GPU that just failed), expands the
    search to ALL physical GPUs reported by nvidia-smi so that a completely
    different GPU on the machine can be used as a last resort.

    Args:
        current_gpu: The GPU that just failed (excluded from preference, but
            included in the pool as a last resort if it is the only option).
        all_gpus: All candidate GPU IDs (from ``--gpu``/``--gpus``).
        min_free_mib: Minimum free VRAM (MiB) required when searching within
            *all_gpus*.  The system-wide fallback ignores this floor and simply
            picks the GPU with the most free VRAM (OOMing again is still better
            than not retrying at all).

    Returns:
        The GPU ID with the most free VRAM, or ``None`` if nvidia-smi is
        unavailable.
    """
    # --- Tier 1: search within the explicitly provided GPU list ---
    if all_gpus:
        preferred = [g for g in all_gpus if g != current_gpu] or list(all_gpus)
        stats = query_gpu_memory_stats(preferred)
        if stats:
            best_gpu, best_stats = max(stats.items(), key=lambda kv: kv[1].free_mib)
            if best_stats.free_mib >= min_free_mib and best_gpu != current_gpu:
                return best_gpu
            if best_stats.free_mib >= min_free_mib:
                # Only the current (failed) GPU meets the threshold — fall through
                # to the system-wide search below.
                pass
            else:
                print(
                    f"[WARN] _pick_fallback_gpu: best GPU in provided list ({best_gpu}) "
                    f"only has {best_stats.free_mib} MiB free "
                    f"(threshold: {min_free_mib} MiB). "
                    "Expanding search to all system GPUs."
                )

    # --- Tier 2: scan all physical GPUs on the machine ---
    all_stats = _query_all_gpu_memory_stats()
    if not all_stats:
        return None
    # Prefer GPUs other than the one that just OOMed.
    candidates = {k: v for k, v in all_stats.items() if k != current_gpu}
    pool = candidates if candidates else all_stats
    best_gpu, best_stats = max(pool.items(), key=lambda kv: kv[1].free_mib)
    if best_gpu == current_gpu:
        print(
            f"[WARN] _pick_fallback_gpu: system-wide search found no GPU better than "
            f"the current one ({current_gpu}, {best_stats.free_mib} MiB free). "
            "Retrying on the same GPU."
        )
    else:
        print(
            f"[INFO] _pick_fallback_gpu: system-wide fallback selected GPU {best_gpu} "
            f"({best_stats.free_mib} MiB free) instead of {current_gpu}."
        )
    return best_gpu


def _pick_best_gpu(
    all_gpus: Sequence[str],
    min_free_mib: int = 8192,
) -> str | None:
    """Return the GPU with the most free VRAM from *all_gpus*.

    Used for proactive GPU selection before launching the planner to avoid
    assigning to an already-saturated device.

    Returns ``None`` if ``all_gpus`` is empty or nvidia-smi is unavailable.
    """
    if not all_gpus:
        return None
    stats = query_gpu_memory_stats(list(all_gpus))
    if not stats:
        return None
    best_gpu, best_stats = max(stats.items(), key=lambda kv: kv[1].free_mib)
    print(
        f"[INFO] _pick_best_gpu: selected GPU {best_gpu} "
        f"({best_stats.free_mib} MiB free / {best_stats.total_mib} MiB total) "
        f"from candidates: {list(all_gpus)}"
    )
    if best_stats.free_mib < min_free_mib:
        print(
            f"[WARN] _pick_best_gpu: best GPU {best_gpu} only has "
            f"{best_stats.free_mib} MiB free — below minimum {min_free_mib} MiB."
        )
    return best_gpu


def wait_for_gpu_memory_settle(
    gpu: str,
    *,
    settle_timeout_s: float,
    poll_seconds: float,
    stable_samples: int,
    stable_tolerance_mib: int,
) -> GPUMemoryStats | None:
    begin_t = time.monotonic()
    _perf_add("gpu_settle_calls", 1)
    stable_samples = max(1, int(stable_samples))
    settle_timeout_s = max(0.0, float(settle_timeout_s))
    poll_seconds = max(0.1, float(poll_seconds))
    stable_tolerance_mib = max(0, int(stable_tolerance_mib))

    deadline = time.monotonic() + settle_timeout_s
    stable_count = 0
    last_free_mib: int | None = None
    latest_stats: GPUMemoryStats | None = None

    while True:
        stats = query_gpu_memory_stats([gpu]).get(gpu)
        if stats is None:
            _perf_add("gpu_settle_time_s", time.monotonic() - begin_t)
            return latest_stats
        latest_stats = stats

        if last_free_mib is None:
            stable_count = 1
        elif abs(stats.free_mib - last_free_mib) <= stable_tolerance_mib:
            stable_count += 1
        else:
            stable_count = 1
        last_free_mib = stats.free_mib

        if stable_count >= stable_samples:
            _perf_add("gpu_settle_time_s", time.monotonic() - begin_t)
            return latest_stats
        if time.monotonic() >= deadline:
            _perf_add("gpu_settle_time_s", time.monotonic() - begin_t)
            _perf_add("gpu_settle_timeouts", 1)
            return latest_stats
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def select_colmdriver_vllm_gpus(
    requested_gpus: Sequence[str],
) -> ColmDriverVLLMReservation:
    scheduler_gpus = tuple(str(gpu).strip() for gpu in requested_gpus if str(gpu).strip())
    if len(scheduler_gpus) < 2:
        raise RuntimeError(
            "--start-colmdriver-vllm requires at least 2 distinct GPUs in --gpus "
            "so VLM and LLM can be placed separately."
        )
    if len(set(scheduler_gpus)) != len(scheduler_gpus):
        raise RuntimeError(
            "--start-colmdriver-vllm requires distinct GPU ids in --gpus."
        )

    live_stats = query_gpu_memory_stats(scheduler_gpus)
    if live_stats:
        requested_order = {gpu: idx for idx, gpu in enumerate(scheduler_gpus)}
        ranked = sorted(
            live_stats.items(),
            key=lambda item: (
                item[1].free_mib,
                item[1].total_mib,
                -requested_order.get(item[0], 0),
            ),
            reverse=True,
        )
        chosen = [gpu for gpu, _stats in ranked[:2]]
        if len(chosen) == 2:
            return ColmDriverVLLMReservation(
                scheduler_gpus=scheduler_gpus,
                vlm_gpu=chosen[0],
                llm_gpu=chosen[1],
                selection_source="live_free_vram",
            )

    return ColmDriverVLLMReservation(
        scheduler_gpus=scheduler_gpus,
        vlm_gpu=scheduler_gpus[-2],
        llm_gpu=scheduler_gpus[-1],
        selection_source="requested_order_fallback",
    )


class PlannerVramEstimator:
    """Persistent per-planner VRAM estimator.

    Stores exponentially-smoothed peak-memory observations on disk so that
    future scheduler runs bin-pack with realistic estimates from the start.
    Thread-safe: internal lock guards both the in-memory dict and the file.
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        ema_alpha: float = VRAM_ESTIMATE_EMA_ALPHA_DEFAULT,
        seed: Dict[str, int] | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._ema_alpha = max(0.01, min(1.0, float(ema_alpha)))
        self._lock = threading.Lock()
        self._estimates: Dict[str, float] = {}
        self._observation_counts: Dict[str, int] = {}
        if seed:
            for name, value in seed.items():
                self._estimates[name] = float(max(0, int(value)))
        self._load()

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    def _load(self) -> None:
        try:
            if not self._cache_path.exists():
                return
            payload = json.loads(self._cache_path.read_text("utf-8"))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[WARN] Unable to load planner VRAM estimate cache {self._cache_path}: {exc}"
            )
            return
        entries = payload.get("planners") if isinstance(payload, dict) else None
        if not isinstance(entries, dict):
            return
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            try:
                mib = float(entry.get("estimate_mib", 0.0))
                if mib <= 0:
                    continue
                self._estimates[str(name)] = mib
                self._observation_counts[str(name)] = int(entry.get("observation_count", 0))
            except (TypeError, ValueError):
                continue

    def _save_locked(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # pylint: disable=broad-except
            return
        payload = {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "planners": {
                name: {
                    "estimate_mib": int(round(value)),
                    "observation_count": int(self._observation_counts.get(name, 0)),
                }
                for name, value in self._estimates.items()
            },
        }
        try:
            tmp_path = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), "utf-8")
            tmp_path.replace(self._cache_path)
        except Exception as exc:  # pylint: disable=broad-except
            print(
                f"[WARN] Unable to persist planner VRAM estimate cache {self._cache_path}: {exc}"
            )

    def estimate(self, planner_name: str) -> int:
        with self._lock:
            if planner_name in self._estimates:
                return int(round(self._estimates[planner_name]))
        return int(PLANNER_VRAM_ESTIMATE_MIB.get(planner_name, PLANNER_VRAM_DEFAULT_FALLBACK_MIB))

    def observe_peak(self, planner_name: str, observed_mib: int) -> None:
        """Merge an observed peak MiB into the running estimate."""
        if observed_mib <= 0:
            return
        with self._lock:
            current = float(self._estimates.get(planner_name, 0.0))
            if current <= 0.0:
                new_value = float(observed_mib)
            else:
                # Track the *upper envelope*: we can't under-estimate without
                # risking OOM, so bias upward and decay only slowly.
                if observed_mib > current:
                    new_value = float(observed_mib)
                else:
                    new_value = (
                        self._ema_alpha * float(observed_mib)
                        + (1.0 - self._ema_alpha) * current
                    )
            self._estimates[planner_name] = new_value
            self._observation_counts[planner_name] = (
                int(self._observation_counts.get(planner_name, 0)) + 1
            )
            self._save_locked()

    def bump_after_oom(self, planner_name: str, *, bump_mib: int) -> None:
        """Raise the floor for a planner after an OOM."""
        bump_mib = max(0, int(bump_mib))
        if bump_mib <= 0:
            return
        with self._lock:
            current = float(self._estimates.get(planner_name, 0.0))
            if current <= 0.0:
                current = float(
                    PLANNER_VRAM_ESTIMATE_MIB.get(
                        planner_name, PLANNER_VRAM_DEFAULT_FALLBACK_MIB
                    )
                )
            new_value = current + float(bump_mib)
            self._estimates[planner_name] = new_value
            self._observation_counts[planner_name] = (
                int(self._observation_counts.get(planner_name, 0)) + 1
            )
            self._save_locked()

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {name: int(round(val)) for name, val in self._estimates.items()}


class GPUMemoryMonitor(threading.Thread):
    """Background nvidia-smi sampler.

    Dispatchers read the latest snapshot via :meth:`snapshot` with O(1) cost.
    Also tracks per-GPU peak ``used_mib`` since a baseline timestamp so the
    scheduler can derive planner peak observations on completion.
    """

    def __init__(
        self,
        gpu_ids: Sequence[str],
        *,
        poll_seconds: float = GPU_MONITOR_POLL_SECONDS_DEFAULT,
    ) -> None:
        super().__init__(name="GPUMemoryMonitor", daemon=True)
        self._gpu_ids = [str(g) for g in gpu_ids]
        self._poll_seconds = max(0.25, float(poll_seconds))
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Dict[str, GPUMemoryStats] = {}
        # baselines[gpu] = (baseline_used_mib, peak_used_mib_since_baseline)
        self._peaks: Dict[str, Tuple[int, int]] = {}
        self._last_update_monotonic: float = 0.0

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                stats_map = _query_all_gpu_memory_stats(self._gpu_ids) or {}
            except Exception:  # pylint: disable=broad-except
                stats_map = {}
            now = time.monotonic()
            with self._lock:
                self._latest = dict(stats_map)
                self._last_update_monotonic = now
                for gpu_id, stats in stats_map.items():
                    baseline, peak = self._peaks.get(gpu_id, (stats.used_mib, stats.used_mib))
                    if stats.used_mib > peak:
                        peak = stats.used_mib
                    self._peaks[gpu_id] = (baseline, peak)
            self._stop.wait(self._poll_seconds)

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> Dict[str, GPUMemoryStats]:
        with self._lock:
            return dict(self._latest)

    def stats_for(self, gpu_id: str | None) -> GPUMemoryStats | None:
        if gpu_id is None:
            return None
        with self._lock:
            return self._latest.get(str(gpu_id))

    def last_update_age_s(self) -> float:
        with self._lock:
            if self._last_update_monotonic <= 0.0:
                return float("inf")
            return max(0.0, time.monotonic() - self._last_update_monotonic)

    def reset_baseline(self, gpu_id: str) -> None:
        with self._lock:
            stats = self._latest.get(str(gpu_id))
            baseline = int(stats.used_mib) if stats is not None else 0
            self._peaks[str(gpu_id)] = (baseline, baseline)

    def consume_peak_delta(self, gpu_id: str) -> int:
        """Return and reset the peak delta (MiB) since the last baseline."""
        with self._lock:
            baseline, peak = self._peaks.get(str(gpu_id), (0, 0))
            delta = max(0, int(peak) - int(baseline))
            # Reset baseline to the latest live value so the next window
            # starts fresh.
            stats = self._latest.get(str(gpu_id))
            new_baseline = int(stats.used_mib) if stats is not None else 0
            self._peaks[str(gpu_id)] = (new_baseline, new_baseline)
        return delta


class SchedulerTickLogger:
    """JSONL sink for scheduler decisions and snapshots."""

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._enabled = path is not None
        if self._enabled:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # pylint: disable=broad-except
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, event: str, **fields: Any) -> None:
        if not self._enabled:
            return
        payload = {
            "ts": time.time(),
            "ts_monotonic": time.monotonic(),
            "event": event,
        }
        payload.update(fields)
        line = json.dumps(payload, default=str, sort_keys=True)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except Exception:  # pylint: disable=broad-except
                pass


def refresh_gpu_launch_capacity(
    gpu_state: GPUPlannerState,
    *,
    min_free_mib: int,
    max_concurrency: int | None = None,
    extra_min_free_mib: int = 0,
    settle_timeout_s: float,
    poll_seconds: float,
    stable_samples: int,
    stable_tolerance_mib: int,
) -> GPUPlannerState:
    capped_max = None if max_concurrency is None else max(1, int(max_concurrency))
    effective_min_free_mib = max(0, int(min_free_mib) + max(0, int(extra_min_free_mib)))
    if gpu_state.gpu is None:
        gpu_state.telemetry_available = False
        gpu_state.last_stats = None
        gpu_state.last_probe_source = "no_gpu"
        gpu_state.launch_capacity = max(1, gpu_state.active_count + 1)
        if capped_max is not None:
            gpu_state.launch_capacity = min(gpu_state.launch_capacity, capped_max)
        return gpu_state

    stats = wait_for_gpu_memory_settle(
        gpu_state.gpu,
        settle_timeout_s=settle_timeout_s,
        poll_seconds=poll_seconds,
        stable_samples=stable_samples,
        stable_tolerance_mib=stable_tolerance_mib,
    )
    if stats is None:
        gpu_state.telemetry_available = False
        gpu_state.last_stats = None
        gpu_state.last_probe_source = "nvidia_smi_unavailable"
        gpu_state.launch_capacity = max(1, gpu_state.active_count)
        if capped_max is not None:
            gpu_state.launch_capacity = min(gpu_state.launch_capacity, capped_max)
        return gpu_state

    gpu_state.telemetry_available = True
    gpu_state.last_stats = stats
    gpu_state.last_probe_source = "live"
    has_capacity = stats.free_mib >= effective_min_free_mib
    gpu_state.launch_capacity = gpu_state.active_count + (1 if has_capacity else 0)
    if capped_max is not None:
        gpu_state.launch_capacity = min(gpu_state.launch_capacity, capped_max)
    # Warn when a GPU passes the threshold but its free headroom is suspiciously
    # low relative to the total size (e.g. a 46 GiB GPU with only 8 GiB free
    # will not fit a 44 GiB model even though 8 GiB > auto_workers_min_free_mib
    # default of 6 GiB).  This helps diagnose OOM failures early.
    if has_capacity and stats.free_mib < stats.total_mib * 0.5:
        print(
            f"[WARN] GPU {gpu_state.gpu}: only {stats.free_mib} MiB free out of "
            f"{stats.total_mib} MiB total ({100*stats.free_mib//stats.total_mib}% free). "
            "This GPU passed the --auto-workers-min-free-mib threshold "
            f"({effective_min_free_mib} MiB) but may still OOM if the planner model is large. "
            "Consider raising --auto-workers-min-free-mib (e.g. to 40000 for 44 GiB models)."
        )
    return gpu_state


def choose_force_launch_gpu(
    gpu_states: Dict[str | None, GPUPlannerState],
) -> GPUPlannerState:
    return max(
        gpu_states.values(),
        key=lambda state: (
            state.last_stats.free_mib if state.last_stats is not None else -1,
            -(state.active_count),
        ),
    )


def _perf_add(
    field: str,
    value: int | float = 1,
) -> None:
    with _PERF_LOCK:
        current = getattr(_PERF_COUNTERS, field)
        setattr(_PERF_COUNTERS, field, current + value)


def _perf_add_retry(failure_kind: str, outcome: str) -> None:
    normalized_kind = str(failure_kind or "unknown").strip() or "unknown"
    normalized_outcome = str(outcome or "unknown").strip() or "unknown"
    with _PERF_LOCK:
        _PERF_COUNTERS.retry_events += 1
        _PERF_COUNTERS.retries_by_kind[normalized_kind] = (
            _PERF_COUNTERS.retries_by_kind.get(normalized_kind, 0) + 1
        )
        _PERF_COUNTERS.retries_by_outcome[normalized_outcome] = (
            _PERF_COUNTERS.retries_by_outcome.get(normalized_outcome, 0) + 1
        )


def snapshot_performance_counters() -> Dict[str, Any]:
    with _PERF_LOCK:
        return {
            "nvidia_smi_calls": int(_PERF_COUNTERS.nvidia_smi_calls),
            "nvidia_smi_time_s": float(_PERF_COUNTERS.nvidia_smi_time_s),
            "gpu_settle_calls": int(_PERF_COUNTERS.gpu_settle_calls),
            "gpu_settle_time_s": float(_PERF_COUNTERS.gpu_settle_time_s),
            "gpu_settle_timeouts": int(_PERF_COUNTERS.gpu_settle_timeouts),
            "teardown_gate_calls": int(_PERF_COUNTERS.teardown_gate_calls),
            "teardown_gate_time_s": float(_PERF_COUNTERS.teardown_gate_time_s),
            "teardown_gate_timeouts": int(_PERF_COUNTERS.teardown_gate_timeouts),
            "carla_start_calls": int(_PERF_COUNTERS.carla_start_calls),
            "carla_start_time_s": float(_PERF_COUNTERS.carla_start_time_s),
            "carla_restart_calls": int(_PERF_COUNTERS.carla_restart_calls),
            "carla_restart_time_s": float(_PERF_COUNTERS.carla_restart_time_s),
            "child_runs": int(_PERF_COUNTERS.child_runs),
            "child_run_time_s": float(_PERF_COUNTERS.child_run_time_s),
            "child_output_lines": int(_PERF_COUNTERS.child_output_lines),
            "child_diag_lines": int(_PERF_COUNTERS.child_diag_lines),
            "retry_events": int(_PERF_COUNTERS.retry_events),
            "retries_by_kind": dict(_PERF_COUNTERS.retries_by_kind),
            "retries_by_outcome": dict(_PERF_COUNTERS.retries_by_outcome),
        }


def format_performance_summary_lines(
    *,
    wall_clock_s: float | None = None,
) -> List[str]:
    perf = snapshot_performance_counters()
    wall = max(0.0, float(wall_clock_s or 0.0))
    child_time = float(perf.get("child_run_time_s", 0.0))
    launch_overhead = max(0.0, wall - child_time)
    lines = [
        "[PERF] Launcher timing summary:",
        (
            f"[PERF] wall_clock_s={wall:.1f} child_runtime_s={child_time:.1f} "
            f"launcher_overhead_s={launch_overhead:.1f}"
        ),
        (
            f"[PERF] gpu_probes: nvidia_smi_calls={int(perf.get('nvidia_smi_calls', 0))} "
            f"nvidia_smi_time_s={float(perf.get('nvidia_smi_time_s', 0.0)):.1f} "
            f"settle_calls={int(perf.get('gpu_settle_calls', 0))} "
            f"settle_time_s={float(perf.get('gpu_settle_time_s', 0.0)):.1f} "
            f"settle_timeouts={int(perf.get('gpu_settle_timeouts', 0))}"
        ),
        (
            f"[PERF] teardown_gate: calls={int(perf.get('teardown_gate_calls', 0))} "
            f"time_s={float(perf.get('teardown_gate_time_s', 0.0)):.1f} "
            f"timeouts={int(perf.get('teardown_gate_timeouts', 0))}"
        ),
        (
            f"[PERF] carla_lifecycle: starts={int(perf.get('carla_start_calls', 0))} "
            f"start_time_s={float(perf.get('carla_start_time_s', 0.0)):.1f} "
            f"restarts={int(perf.get('carla_restart_calls', 0))} "
            f"restart_time_s={float(perf.get('carla_restart_time_s', 0.0)):.1f}"
        ),
        (
            f"[PERF] child_io: runs={int(perf.get('child_runs', 0))} "
            f"output_lines={int(perf.get('child_output_lines', 0))} "
            f"diag_lines={int(perf.get('child_diag_lines', 0))}"
        ),
        (
            f"[PERF] retries: total={int(perf.get('retry_events', 0))} "
            f"by_kind={json.dumps(perf.get('retries_by_kind', {}), sort_keys=True)} "
            f"by_outcome={json.dumps(perf.get('retries_by_outcome', {}), sort_keys=True)}"
        ),
    ]
    return lines


def write_performance_report(
    report_path: Path | None,
    *,
    wall_clock_s: float,
    context: Dict[str, Any] | None = None,
) -> None:
    if report_path is None:
        return
    payload = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "wall_clock_s": max(0.0, float(wall_clock_s)),
        "performance": snapshot_performance_counters(),
        "summary_lines": format_performance_summary_lines(wall_clock_s=wall_clock_s),
    }
    if context:
        payload["context"] = dict(context)
    try:
        write_json_file_atomic(report_path, payload)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to write performance report {report_path}: {exc}")


def append_retry_event(
    retry_log_path: Path | None,
    *,
    scope: str,
    attempt: int | None,
    retry_limit: int | None,
    failure_kind: str,
    outcome: str,
    details: str = "",
    planner_run_id: str = "",
    scenario_name: str = "",
    gpu: str | None = None,
    ports: PortBundle | None = None,
) -> None:
    _perf_add_retry(failure_kind, outcome)
    if retry_log_path is None:
        return
    retry_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "scope": str(scope),
        "attempt": None if attempt is None else int(attempt),
        "retry_limit": None if retry_limit is None else int(retry_limit),
        "failure_kind": str(failure_kind or "unknown"),
        "outcome": str(outcome or "unknown"),
        "details": str(details or ""),
        "planner_run_id": str(planner_run_id or ""),
        "scenario_name": str(scenario_name or ""),
        "gpu": None if gpu is None else str(gpu),
        "world_port": None if ports is None else int(ports.port),
        "tm_port": None if ports is None else int(ports.tm_port),
    }
    try:
        with retry_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to append retry event {retry_log_path}: {exc}")


def classify_failure_text_kind(text: str) -> str:
    lowered = str(text or "").lower()
    if "cuda_oom" in lowered or ("cuda" in lowered and "out of memory" in lowered):
        return "cuda_oom"
    if "no cuda gpus are available" in lowered or "cuda_unavailable" in lowered:
        return "cuda_unavailable"
    if "teardown_gate_timeout" in lowered:
        return "teardown_gate_timeout"
    if "connection refused" in lowered or "connection reset by peer" in lowered:
        return "rpc_timeout"
    if "sensorreceivednodata" in lowered or "sensor took too long" in lowered:
        return "sensor_failure"
    if "watchdog" in lowered or "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "carla" in lowered and "crash" in lowered:
        return "carla_crash"
    if "planner_env_error" in lowered or "modulenotfounderror" in lowered:
        return "planner_env_error"
    return "generic"


def infer_planner_thread_failure_kind(
    completed: PlannerThreadResult,
    *,
    dashboard_state: PlannerDashboardState,
) -> str:
    if completed.failure_kind:
        return str(completed.failure_kind)
    reason = str(completed.error or "")
    if not reason and completed.returncode is not None:
        reason = f"exit_code={completed.returncode}"
    inferred = classify_failure_text_kind(reason)
    if inferred != "generic":
        return inferred
    tail = _read_text_tail(
        dashboard_state.log_path,
        max_bytes=RESULT_LOG_FALLBACK_TAIL_BYTES_DEFAULT,
    )
    return classify_failure_text_kind(tail)


def load_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def write_json_file_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def get_dashboard_status_path(env: Dict[str, str] | None = None) -> Path | None:
    value = (env or os.environ).get(DASHBOARD_STATUS_ENV)
    if not value:
        return None
    return Path(value).expanduser()


def update_dashboard_status(status_path: Path | None, **fields: Any) -> None:
    if status_path is None:
        return
    payload = load_json_file(status_path)
    payload.update(fields)
    payload["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    try:
        write_json_file_atomic(status_path, payload)
    except Exception:
        return


def _parse_route_progress(top_level: Dict[str, Any]) -> Tuple[int, int]:
    checkpoint = top_level.get("_checkpoint", {}) if isinstance(top_level, dict) else {}
    progress = checkpoint.get("progress", [])
    if isinstance(progress, list) and len(progress) == 2:
        try:
            current_routes = max(0, int(progress[0]))
            total_routes = max(1, int(progress[1]))
            return min(current_routes, total_routes), total_routes
        except Exception:
            return (0, 1)
    return (0, 1)


def _strip_ansi_sequences(text: str) -> str:
    if not text:
        return ""
    return ANSI_ESCAPE_RE.sub("", text)


def _recovery_mode_from_env() -> str:
    mode = str(
        os.environ.get(
            "RUN_CUSTOM_EVAL_RECOVERY_MODE",
            os.environ.get("CARLA_RECOVERY_MODE", "off"),
        )
    ).strip().lower()
    if mode not in ("off", "engineering", "publication_safe"):
        return "off"
    return mode


def strict_record_accounting_enabled() -> bool:
    if str(os.environ.get("CARLA_RECOVERY_STRICT_RECORD_ACCOUNTING", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    return _recovery_mode_from_env() == "publication_safe"


def classify_terminal_accounting_state(summary: ResultRootProgress) -> str:
    if not bool(summary.has_structured_records):
        return "invalid_no_record"
    if not has_only_terminal_route_outcomes(summary):
        if str(summary.recovery_failure_kind or "").strip():
            return "failed_recovery"
        if did_scenario_record_runtime_failure(summary):
            return "failed_runtime"
        return "invalid_no_record"
    if did_scenario_record_runtime_failure(summary):
        if str(summary.recovery_failure_kind or "").strip():
            return "failed_recovery"
        return "failed_runtime"
    if bool(summary.route_recovered):
        return "completed_recovered"
    return "completed_clean"


def _infer_ego_outcomes_from_log(
    log_path: Path,
    *,
    ego_count: int,
) -> List[Tuple[str, float]]:
    if ego_count <= 0:
        return []
    log_tail = _read_text_tail(log_path, max_bytes=RESULT_LOG_FALLBACK_TAIL_BYTES_DEFAULT)
    if not log_tail:
        return []

    cleaned = _strip_ansi_sequences(log_tail)
    lines = cleaned.splitlines()
    if not lines:
        return []

    lower_text = cleaned.lower()
    saw_agent_crash = (
        "agent has crashed" in lower_text
        or "agenterror" in lower_text
        or "run scenario crashed" in lower_text
        or "simulation crashed" in lower_text
    )
    saw_blocked = "vehicle_blocked" in lower_text or "agentblockedtest" in lower_text
    saw_route_timeout = "route_timeout" in lower_text

    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        header_match = RESULT_SECTION_RE.search(line)
        if header_match:
            if current is not None:
                sections.append(current)
            current = {"header": header_match.group(1).upper(), "route_score": None}
            continue
        if current is None:
            continue
        route_match = ROUTE_COMPLETION_RE.search(line)
        if route_match:
            try:
                current["route_score"] = float(route_match.group(1))
            except Exception:
                current["route_score"] = None
            continue
    if current is not None:
        sections.append(current)
    if not sections:
        return []

    selected_sections = sections[-ego_count:]
    inferred: List[Tuple[str, float]] = []
    for section in selected_sections:
        header = str(section.get("header") or "").upper()
        raw_score = section.get("route_score")
        try:
            route_score = float(raw_score)
        except Exception:
            route_score = 100.0 if header == "SUCCESS" else 0.0
        route_score = max(0.0, min(100.0, route_score))

        if header == "SUCCESS":
            status = "Completed"
        else:
            if saw_agent_crash:
                status = "Failed - Agent crashed"
            elif saw_route_timeout:
                status = "Failed - Route timeout"
            elif saw_blocked:
                status = "Failed - Agent blocked"
            else:
                status = "Failed"
        inferred.append((status, route_score))
    return inferred


def collect_result_root_progress(
    result_root: Path,
    *,
    ego_count: int,
    fallback_status: str = "",
    allow_log_fallback: bool | None = None,
) -> ResultRootProgress:
    summary = ResultRootProgress()
    if allow_log_fallback is None:
        allow_log_fallback = not strict_record_accounting_enabled()
    top_level = load_json_file(result_root / "results.json")
    if top_level:
        summary.started = True
        summary.route_progress = _parse_route_progress(top_level)
        summary.entry_status = str(top_level.get("entry_status") or fallback_status or "")
    else:
        summary.entry_status = str(fallback_status or "")

    for ego_idx in range(ego_count):
        ego_data = load_json_file(result_root / f"ego_vehicle_{ego_idx}" / "results.json")
        if ego_data:
            summary.started = True
        ego_records = ego_data.get("_checkpoint", {}).get("records", [])
        if isinstance(ego_records, list):
            summary.ego_record_counts[ego_idx] = len(ego_records)
            if len(ego_records) > 0:
                summary.has_structured_records = True
        else:
            summary.ego_record_counts[ego_idx] = 0
        last_record = ego_records[-1] if ego_records else {}
        scores = last_record.get("scores", {}) if isinstance(last_record, dict) else {}
        status = ""
        if isinstance(last_record, dict):
            status = str(last_record.get("status") or "")
        if not status:
            status = str(ego_data.get("entry_status") or summary.entry_status or fallback_status or "")
        try:
            summary.ego_route_scores[ego_idx] = float(scores.get("score_route", 0.0))
        except Exception:
            summary.ego_route_scores[ego_idx] = 0.0
        summary.ego_status[ego_idx] = status
        meta = last_record.get("meta", {}) if isinstance(last_record, dict) and isinstance(last_record.get("meta"), dict) else {}
        if meta:
            if bool(meta.get("route_recovered", False)):
                summary.route_recovered = True
            try:
                summary.recovery_count = max(
                    int(summary.recovery_count),
                    int(meta.get("recovery_count", meta.get("crash_generation", 0)) or 0),
                )
            except Exception:
                pass
            planner_mode = str(meta.get("planner_continuity_mode", "") or "").strip()
            if planner_mode:
                summary.planner_continuity_modes.append(planner_mode)
            if bool(meta.get("recovered_approximate", False)):
                summary.recovered_approximate = True
            checkpoint_id = meta.get("checkpoint_id", None)
            if checkpoint_id is not None:
                try:
                    ckpt_int = int(checkpoint_id)
                    if summary.checkpoint_id is None or ckpt_int > int(summary.checkpoint_id):
                        summary.checkpoint_id = ckpt_int
                except Exception:
                    pass
            if not summary.recovery_reason:
                summary.recovery_reason = str(meta.get("recovery_reason", "") or "")
            if not summary.recovery_failure_kind:
                summary.recovery_failure_kind = str(meta.get("recovery_failure_kind", "") or "")
            if bool(meta.get("invalid_for_benchmark", False)):
                summary.invalid_for_benchmark = True
            if meta.get("benchmark_eligible_candidate", True) is False:
                summary.benchmark_eligible_candidate = False

    needs_log_fallback = summary.route_progress[0] <= 0
    if not needs_log_fallback:
        for ego_idx in range(ego_count):
            status = str(summary.ego_status.get(ego_idx) or "").strip()
            if not status:
                needs_log_fallback = True
                break
            try:
                route_score = float(summary.ego_route_scores.get(ego_idx, 0.0))
            except Exception:
                route_score = 0.0
            if route_score <= 0.0:
                needs_log_fallback = True
                break

    inferred_outcomes: List[Tuple[str, float]] = []
    if bool(allow_log_fallback) and needs_log_fallback:
        inferred_outcomes = _infer_ego_outcomes_from_log(
            result_root / "log" / "log.log",
            ego_count=ego_count,
        )
    if inferred_outcomes:
        summary.started = True
        if summary.route_progress[0] <= 0:
            summary.route_progress = (1, max(1, summary.route_progress[1]))
        for ego_idx, (inferred_status, inferred_route_score) in enumerate(inferred_outcomes):
            current_status = str(summary.ego_status.get(ego_idx) or "").strip()
            if not current_status:
                summary.ego_status[ego_idx] = inferred_status
            try:
                current_route_score = float(summary.ego_route_scores.get(ego_idx, 0.0))
            except Exception:
                current_route_score = 0.0
            if current_route_score <= 0.0:
                summary.ego_route_scores[ego_idx] = inferred_route_score
        if not str(summary.entry_status or "").strip():
            if any(
                is_runtime_failure_status(status)
                for status in summary.ego_status.values()
            ):
                summary.entry_status = "Failed - Agent crashed"
            elif any(
                is_terminal_route_outcome_status(status)
                for status in summary.ego_status.values()
            ):
                summary.entry_status = "Completed"

    summary.completed = (
        summary.started and summary.route_progress[1] > 0 and summary.route_progress[0] >= summary.route_progress[1]
    )
    summary.terminal_state = classify_terminal_accounting_state(summary)
    return summary


def parse_live_ego_route_scores(payload: Dict[str, Any]) -> Dict[int, float]:
    scores_payload = payload.get("ego_route_scores", {}) if isinstance(payload, dict) else {}
    if not isinstance(scores_payload, dict):
        return {}

    live_scores: Dict[int, float] = {}
    for raw_ego_idx, raw_score in scores_payload.items():
        try:
            ego_idx = int(raw_ego_idx)
            route_score = float(raw_score)
        except Exception:
            continue
        live_scores[ego_idx] = max(0.0, min(100.0, route_score))
    return live_scores


def filter_scenario_entries_for_shard(
    entries: Sequence[Tuple[str, str]],
    shard_str: str | None,
) -> List[Tuple[str, str]]:
    """Return the subset of *entries* that belongs to the given shard.

    *shard_str* has the form ``"INDEX/COUNT"`` (0-based index).  When *None*
    or when *entries* is empty, the full list is returned unchanged.
    """
    if not shard_str or not entries:
        return list(entries)
    try:
        idx_s, cnt_s = shard_str.split("/", 1)
        shard_idx, shard_cnt = int(idx_s), int(cnt_s)
    except (ValueError, TypeError):
        return list(entries)
    if shard_cnt <= 1:
        return list(entries)
    return [
        entry for i, entry in enumerate(entries) if i % shard_cnt == shard_idx
    ]


def abbreviate_dashboard_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def format_dashboard_ego_status(status: str, fallback_phase: str) -> str:
    text = str(status or fallback_phase or "").strip()
    if not text:
        return str(fallback_phase or "")
    lower = text.lower()
    if text.startswith("Failed - "):
        detail = text[len("Failed - ") :].strip()
        if is_runtime_failure_status(lower):
            return f"Runtime issue - {detail}"
        return f"Route outcome - {detail}"
    if lower == "failed":
        return "Route outcome - failed"
    return text


def normalize_retry_limit(retry_limit: int | None) -> int | None:
    if retry_limit is None:
        return None
    value = int(retry_limit)
    if value <= 0:
        return None
    return value


def retry_limit_for_status(retry_limit: int | None) -> int | None:
    return normalize_retry_limit(retry_limit)


def retry_limit_multiplier(retry_limit: int | None) -> int:
    normalized = normalize_retry_limit(retry_limit)
    return normalized if normalized is not None else 1


def should_continue_retrying(current_attempt: int, retry_limit: int | None) -> bool:
    normalized = normalize_retry_limit(retry_limit)
    if normalized is None:
        return current_attempt < SCENARIO_RETRY_HARD_CEILING
    return current_attempt < normalized


def format_attempt_counter(current_attempt: int, retry_limit: int | None) -> str:
    normalized = normalize_retry_limit(retry_limit)
    if normalized is None:
        return f"{current_attempt}/{SCENARIO_RETRY_HARD_CEILING}"
    return f"{current_attempt}/{normalized}"


def is_runtime_failure_status(status: str) -> bool:
    lower = str(status or "").strip().lower()
    if not lower:
        return False
    runtime_markers = (
        "run scenario crashed",
        "agent crashed",
        "simulation crashed",
        "sensor",
        "carla",
        "failed - recovery",
        "failed_recovery",
        "recovery failed",
        "couldn't be set up",
        "sensors were invalid",
        "finished with missing data",
        "rejected",
        "crashed",
    )
    return any(marker in lower for marker in runtime_markers)


def is_terminal_route_outcome_status(status: str) -> bool:
    lower = str(status or "").strip().lower()
    if not lower or is_runtime_failure_status(lower):
        return False
    if lower in {"queued", "launching", "running"}:
        return False
    return (
        lower.startswith("completed")
        or lower.startswith("finished")
        or lower.startswith("failed")
    )


_SIGNAL_NAMES: dict[int, str] = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    4: "SIGILL",
    6: "SIGABRT",
    7: "SIGBUS",
    8: "SIGFPE",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    14: "SIGALRM",
    15: "SIGTERM",
}


def _describe_failure_detail(
    failure_kind: str,
    result: MonitoredRunResult,
    carla_manager: "CarlaProcessManager | None" = None,
    *,
    stage: str | None = None,
    progress_summary: ResultRootProgress | None = None,
) -> str:
    """Return a concise diagnostic string for a scenario failure, including CARLA crash details."""
    parts: list[str] = [f"reason={failure_kind}"]
    if stage:
        parts.append(f"stage={stage}")
    parts.append(f"classification={failure_kind}")
    if progress_summary is not None:
        route_done, route_total = progress_summary.route_progress
        parts.append(f"route_progress={route_done}/{route_total}")
        parts.append(f"started={progress_summary.started}")
        parts.append(f"completed={progress_summary.completed}")
        parts.append(f"ego_vehicle_count={len(progress_summary.ego_status)}")
        entry_status = str(progress_summary.entry_status or "").strip()
        if entry_status:
            parts.append(f"entry_status={entry_status!r}")

    # Check output tail for specific sensor/connection errors (most actionable)
    tail_text = "\n".join(result.output_tail).lower()
    if "a sensor took too long to send their data" in tail_text or "sensorreceivednodata" in tail_text:
        parts.append("cause=sensor_data_timeout (sensor did not respond in time)")
    elif "invalid session: no stream available" in tail_text or "error retrieving stream id" in tail_text:
        parts.append("cause=sensor_stream_disconnected (CARLA stream lost)")
    elif "connection refused" in tail_text:
        parts.append("cause=connection_refused")
    elif "connection reset by peer" in tail_text:
        parts.append("cause=connection_reset")
    elif "failed to connect" in tail_text:
        parts.append("cause=failed_to_connect")

    # CUDA-specific diagnostics
    if failure_kind == "cuda_oom":
        import re as _re
        oom_match = _re.search(
            r"tried to allocate ([0-9.]+ \w+).*?([0-9.]+ \w+) free",
            "\n".join(result.output_tail),
            _re.IGNORECASE | _re.DOTALL,
        )
        if oom_match:
            parts.append(
                f"cause=CUDA_OOM (tried to allocate {oom_match.group(1)}, "
                f"only {oom_match.group(2)} free)"
            )
        else:
            parts.append("cause=CUDA_OOM")
    elif failure_kind == "cuda_unavailable":
        parts.append("cause=no_CUDA_GPUs_visible (check CUDA_VISIBLE_DEVICES / driver)")

    # For CARLA crashes, read rootcause metadata for signal/exit-code info
    if failure_kind in ("carla_crash",) and carla_manager is not None:
        logdir = carla_manager.rootcause_last_logdir
        if logdir is not None and logdir.exists():
            meta = CarlaProcessManager._read_rootcause_run_meta(logdir / "run_meta.txt")
            exit_code_str = meta.get("carla_exit_code") or meta.get("gdb_exit_code")
            if exit_code_str is not None:
                try:
                    exit_code = int(exit_code_str)
                    if exit_code == -11 or exit_code == 139:
                        parts.append("cause=Signal_11_SIGSEGV (segmentation fault / memory corruption)")
                    elif exit_code < 0:
                        sig_num = -exit_code
                        sig_name = _SIGNAL_NAMES.get(sig_num, f"signal_{sig_num}")
                        parts.append(f"cause={sig_name} (killed by signal {sig_num}, exit_code={exit_code})")
                    elif exit_code != 0:
                        parts.append(f"carla_exit_code={exit_code} (non-zero, non-signal exit)")
                except ValueError:
                    pass
            core_file = meta.get("core_file")
            if core_file:
                parts.append(f"core_file={core_file}")

    if result.terminated_reason:
        parts.append(f"terminated={result.terminated_reason!r}")

    return ", ".join(parts)


def _print_failure_banner(
    heading: str,
    detail: str,
) -> None:
    """Print a clearly visible failure banner to stdout."""
    separator = "!" * 70
    print(f"\n{separator}")
    print(f"[FAILURE] {heading}")
    print(f"          {detail}")
    print(f"{separator}\n")


def _log_failure_classification(detected_error: str, assigned_label: str) -> None:
    print(
        f"[FAILURE CLASSIFICATION] detected_error={detected_error!r} "
        f"assigned_label={assigned_label!r}"
    )


def classify_scenario_failure(
    result: MonitoredRunResult,
    *,
    host: str,
    world_port: int,
    tm_port: int,  # kept for API compatibility; no longer used for reachability
    carla_manager: CarlaProcessManager | None = None,
    progress_summary: ResultRootProgress | None = None,
) -> str:
    """Classify a scenario failure into a specific label.

    Label hierarchy (most-specific first):
      cuda_oom           – CUDA out-of-memory error (planner needs more VRAM)
      cuda_unavailable   – No CUDA GPUs visible to the planner process
      planner_env_error  – Python import / package errors (evaluator never contacted CARLA)
      sensor_failure     – sensor timeout / no-data errors
      stalled            – output stalled (evaluator hung)
      no_results_written – evaluator never produced any result files
      no_route_execution – evaluator started but no route progress was recorded
      carla_crash        – CARLA process exited unexpectedly
      carla_process_down – world port closed (CARLA not listening)
      rpc_timeout        – RPC connection-level errors in logs
      carla_unreachable  – CARLA markers in logs but world port still open
      generic            – none of the above
    """
    tail_text = "\n".join(result.output_tail).lower()

    # --- 0. CUDA errors (checked first — most actionable for GPU rotation) ---
    if "cuda out of memory" in tail_text or "out of memory" in tail_text and "cuda" in tail_text:
        _log_failure_classification("CUDA out-of-memory in output", "cuda_oom")
        return "cuda_oom"
    if "no cuda gpus are available" in tail_text:
        _log_failure_classification("no CUDA GPUs available in output", "cuda_unavailable")
        return "cuda_unavailable"

    # --- 1. Python environment errors (import failures, missing packages) ---
    # These fire before CARLA is ever contacted so they must not be labelled
    # as a CARLA connectivity problem.
    # Exclude the harmless site.py/_distutils_hack .pth warning that Python
    # emits on startup when setuptools installs distutils-precedence.pth; it
    # contains "No module named '_distutils_hack'" but is non-fatal (Python
    # itself prints "Remainder of file ignored" and continues normally).
    _has_module_error = (
        "modulenotfounderror" in tail_text
        or "importerror" in tail_text
        or "no module named" in tail_text
    )
    _only_distutils_hack = (
        "_distutils_hack" in tail_text
        and "remainder of file ignored" in tail_text
        and not any(
            marker in tail_text
            for marker in ("modulenotfounderror", "importerror")
            if "_distutils_hack" not in marker
        )
        # Only suppress if the sole "no module named" hit is the distutils one
        and tail_text.count("no module named") <= tail_text.count("_distutils_hack")
    )
    if _has_module_error and not _only_distutils_hack:
        _log_failure_classification("import/module error in output", "planner_env_error")
        return "planner_env_error"

    # --- 2. Sensor failures ---
    if (
        "a sensor took too long to send their data" in tail_text
        or "sensorreceivednodata" in tail_text
    ):
        _log_failure_classification("sensor timeout/no-data in output", "sensor_failure")
        return "sensor_failure"

    # --- 3. Stalled output ---
    if result.stalled:
        _log_failure_classification("evaluator output stalled", "stalled")
        return "stalled"

    # --- 4. Health-check terminated the run ---
    if result.terminated_reason:
        reason_text = result.terminated_reason.lower()
        if "managed carla became unavailable" in reason_text:
            label = (
                "carla_crash"
                if carla_manager is not None and not carla_manager.is_running()
                else "carla_unreachable"
            )
            _log_failure_classification("health-check: CARLA became unavailable", label)
            return label
        if "no non-carla planner output and no route progress" in reason_text:
            _log_failure_classification("health-check: no output/progress", "no_route_execution")
            return "no_route_execution"

    # --- 5. CARLA process exited (manager knows about it) ---
    if carla_manager is not None and not carla_manager.is_running():
        _log_failure_classification("CARLA manager process not running", "carla_crash")
        return "carla_crash"

    # --- 6. CARLA world port not reachable (process down / never started) ---
    # TM port is NOT checked here: TM not being up is not the same as CARLA
    # being unreachable, and checking it was the root-cause of false positives.
    world_up = is_port_open(host, world_port, timeout=0.2)
    if not world_up:
        _log_failure_classification(f"world port {world_port} closed", "carla_process_down")
        return "carla_process_down"

    # --- 7. RPC-level connection errors in log output ---
    rpc_markers = (
        "connection refused",
        "connection reset by peer",
        "failed to connect",
        "rpc",
    )
    if any(marker in tail_text for marker in rpc_markers):
        _log_failure_classification("RPC connection error in output", "rpc_timeout")
        return "rpc_timeout"

    # --- 8. No route progress despite the evaluator starting ---
    if progress_summary is not None:
        if strict_record_accounting_enabled() and not bool(progress_summary.has_structured_records):
            _log_failure_classification("no structured route records", "invalid_no_record")
            return "invalid_no_record"
        route_done, _route_total = progress_summary.route_progress
        if route_done <= 0:
            label = "no_route_execution" if progress_summary.started else "no_results_written"
            detected_error = (
                "result progress with no route execution"
                if label == "no_route_execution"
                else "result progress with no results written"
            )
            if did_scenario_record_runtime_failure(progress_summary):
                detected_error += " (runtime failure recorded before route progress)"
            _log_failure_classification(
                detected_error,
                label,
            )
            return label

    # --- 9. Other CARLA-related markers (world port open but CARLA misbehaving) ---
    carla_markers = (
        "make sure the simulator is ready and connected",
        "waiting for the simulator",
        "error retrieving stream id",
        "invalid session: no stream available",
        "trafficmanager",
        "carla",
        "simulator is ready and connected",
    )
    if any(marker in tail_text for marker in carla_markers):
        _log_failure_classification("CARLA marker in output (world port open)", "carla_unreachable")
        return "carla_unreachable"

    _log_failure_classification("no specific marker matched", "generic")
    return "generic"


def did_scenario_record_runtime_failure(summary: ResultRootProgress) -> bool:
    for raw_status in summary.ego_status.values():
        status = str(raw_status or "").strip().lower()
        if not status:
            continue
        if is_runtime_failure_status(status):
            return True
    entry_status = str(summary.entry_status or "").strip().lower()
    return is_runtime_failure_status(entry_status)


def has_only_terminal_route_outcomes(summary: ResultRootProgress) -> bool:
    if not bool(summary.has_structured_records):
        return False
    statuses = [str(raw_status or "").strip() for raw_status in summary.ego_status.values()]
    statuses = [status for status in statuses if status]
    if not statuses:
        return False
    return all(is_terminal_route_outcome_status(status) for status in statuses)


def classify_missing_terminal_results(summary: ResultRootProgress) -> str:
    if strict_record_accounting_enabled() and not bool(summary.has_structured_records):
        return "invalid_no_record"
    if not summary.started:
        return "no_results_written"
    route_done, _route_total = summary.route_progress
    if route_done <= 0:
        return "no_route_execution"
    return "missing_terminal_results"


def classify_retryable_soft_failure(
    result: MonitoredRunResult,
    *,
    result_root: Path,
    ego_count: int,
    host: str,
    world_port: int,
    tm_port: int,  # kept for API compatibility; no longer used for reachability
    carla_manager: CarlaProcessManager | None = None,
) -> str | None:
    summary = collect_result_root_progress(result_root, ego_count=ego_count)
    if has_only_terminal_route_outcomes(summary):
        return None
    recorded_runtime_failure = did_scenario_record_runtime_failure(summary)

    tail_text = "\n".join(result.output_tail).lower()
    retryable_markers = (
        "a sensor took too long to send their data",
        "sensorreceivednodata",
        "invalid session: no stream available",
        "error retrieving stream id",
        "waiting for the simulator",
        "connection refused",
        "connection reset by peer",
        "failed to connect",
    )
    if any(marker in tail_text for marker in retryable_markers):
        return classify_scenario_failure(
            result,
            host=host,
            world_port=world_port,
            tm_port=tm_port,
            carla_manager=carla_manager,
            progress_summary=summary,
        )

    if recorded_runtime_failure:
        return classify_scenario_failure(
            result,
            host=host,
            world_port=world_port,
            tm_port=tm_port,
            carla_manager=carla_manager,
            progress_summary=summary,
        )

    if carla_manager is not None and not carla_manager.is_running():
        return "carla_crash"
    # Only check world port — TM port not being up does not mean CARLA is down.
    world_up = is_port_open(host, world_port, timeout=0.2)
    if not world_up:
        return "carla_process_down"
    missing_terminal_kind = classify_missing_terminal_results(summary)
    if missing_terminal_kind in ("no_results_written", "no_route_execution", "invalid_no_record"):
        detected_error = (
            "result root recorded no route progress"
            if missing_terminal_kind == "no_route_execution"
            else (
                "result root has no structured route records"
                if missing_terminal_kind == "invalid_no_record"
                else "result root produced no results"
            )
        )
        _log_failure_classification(
            detected_error,
            missing_terminal_kind,
        )
        return missing_terminal_kind
    return None


def validate_planner_child_results(
    *,
    results_dir: Path,
    ego_count: int,
    scenario_entries: Sequence[Tuple[str, str]] = (),
    default_results_subdir: str = "",
) -> str | None:
    result_roots: List[Tuple[str, Path]] = []
    if scenario_entries:
        result_roots.extend(
            (label, Path(results_dir) / relative_dir)
            for label, relative_dir in scenario_entries
        )
    else:
        active_root = (
            Path(results_dir) / default_results_subdir
            if str(default_results_subdir or "").strip()
            else Path(results_dir)
        )
        result_roots.append(("scenario", active_root))

    issues: List[str] = []
    for label, result_root in result_roots:
        summary = collect_result_root_progress(result_root, ego_count=ego_count)
        if summary.terminal_state in ("completed_clean", "completed_recovered"):
            continue
        if did_scenario_record_runtime_failure(summary):
            issues.append(f"{label}: {summary.terminal_state or 'failed_runtime'}")
            continue
        missing_terminal_kind = classify_missing_terminal_results(summary)
        if missing_terminal_kind == "no_results_written":
            issues.append(f"{label}: no results written")
        elif missing_terminal_kind == "no_route_execution":
            issues.append(f"{label}: no route executed")
        elif missing_terminal_kind == "invalid_no_record":
            issues.append(f"{label}: invalid no structured route record")
        else:
            issues.append(f"{label}: no terminal ego outcomes")

    if not issues:
        return None
    if len(issues) == 1:
        return f"Planner child exited 0 but {issues[0]}."
    return (
        "Planner child exited 0 but result validation failed: "
        + "; ".join(issues)
        + "."
    )


def result_root_has_terminal_outcomes(result_root: Path, *, ego_count: int) -> bool:
    summary = collect_result_root_progress(result_root, ego_count=ego_count)
    return summary.terminal_state in (
        "completed_clean",
        "completed_recovered",
        "failed_runtime",
    )


def result_roots_have_terminal_outcomes(
    result_roots: Sequence[Path],
    *,
    ego_count: int,
) -> bool:
    if not result_roots:
        return False
    saw_terminal_outcome = False
    for result_root in result_roots:
        if not result_root_has_terminal_outcomes(result_root, ego_count=ego_count):
            return False
        saw_terminal_outcome = True
    return saw_terminal_outcome


def _read_text_tail(path: Path, *, max_bytes: int = STALLED_NEAR_END_LOG_TAIL_BYTES_DEFAULT) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            handle.seek(max(0, file_size - int(max_bytes)), os.SEEK_SET)
            data = handle.read()
        return data.decode("utf-8", errors="ignore").lower()
    except Exception:
        return ""


def _has_stall_evidence(result: MonitoredRunResult, *, result_root: Path) -> bool:
    def _scan_lines_for_stall_markers(text: str) -> bool:
        if not text:
            return False
        cleaned = _strip_ansi_sequences(text)
        lowered = cleaned.lower()
        if "[stall detection]" in lowered or "all vehicles have been stationary" in lowered:
            return True
        if "vehicle_blocked" in lowered or "route_timeout" in lowered:
            return True
        for raw_line in lowered.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "agentblockedtest" in line and "failure" in line:
                return True
            if "[stall debug]" in line and "not stalled" not in line:
                return True
        return False

    tail_text = "\n".join(result.output_tail)
    if _scan_lines_for_stall_markers(tail_text):
        return True

    log_text = _read_text_tail(result_root / "log" / "log.log")
    if _scan_lines_for_stall_markers(log_text):
        return True
    return False


def should_accept_stalled_near_end_failure(
    *,
    result: MonitoredRunResult,
    result_root: Path,
    summary: ResultRootProgress,
    min_route_score: float,
    require_stall_evidence: bool,
) -> Tuple[bool, str]:
    failed_egos: List[int] = []
    runtime_failure_egos: List[int] = []
    non_runtime_failed_egos: List[int] = []
    low_progress_failed_egos: List[int] = []
    unresolved_egos: List[int] = []

    threshold = float(min_route_score)
    for ego_idx, raw_status in summary.ego_status.items():
        status = str(raw_status or "").strip()
        if not status:
            unresolved_egos.append(int(ego_idx))
            continue
        lower_status = status.lower()
        is_terminal = (
            is_terminal_route_outcome_status(status)
            or is_runtime_failure_status(status)
            or lower_status.startswith("failed")
        )
        if not is_terminal:
            unresolved_egos.append(int(ego_idx))
            continue
        is_failed = lower_status.startswith("failed") or is_runtime_failure_status(status)
        if not is_failed:
            continue
        failed_egos.append(int(ego_idx))
        if is_runtime_failure_status(status):
            runtime_failure_egos.append(int(ego_idx))
        else:
            non_runtime_failed_egos.append(int(ego_idx))
        route_score = float(summary.ego_route_scores.get(int(ego_idx), 0.0))
        if route_score < threshold:
            low_progress_failed_egos.append(int(ego_idx))

    if unresolved_egos:
        return False, "unresolved_ego_status:" + ",".join(str(idx) for idx in unresolved_egos)
    if not failed_egos:
        return False, "no_failed_egos"
    if low_progress_failed_egos:
        return False, (
            "failed_ego_below_threshold:"
            + ",".join(str(idx) for idx in low_progress_failed_egos)
        )

    stall_evidence = _has_stall_evidence(result, result_root=result_root)
    if non_runtime_failed_egos and not stall_evidence:
        return False, "terminal_failure_without_stall_evidence"
    if bool(require_stall_evidence) and not stall_evidence:
        return False, "missing_explicit_stall_evidence"

    route_scores = [float(summary.ego_route_scores.get(idx, 0.0)) for idx in failed_egos]
    detail = (
        "failed_egos="
        + ",".join(str(idx) for idx in failed_egos)
        + "; runtime_failure_egos="
        + ",".join(str(idx) for idx in runtime_failure_egos)
        + "; non_runtime_failed_egos="
        + ",".join(str(idx) for idx in non_runtime_failed_egos)
        + f", min_route_score={min(route_scores):.2f}"
        + f", threshold={threshold:.2f}"
        + f", stall_evidence={int(bool(stall_evidence))}"
    )
    return True, detail


def planner_child_retry_limit(parent_retry_limit: int | None) -> int:
    normalized = normalize_retry_limit(parent_retry_limit)
    if normalized is None:
        return MULTI_PLANNER_CHILD_SCENARIO_RETRY_BURST_DEFAULT
    return normalized


def rotate_scheduler_gpu_order_after_failure(
    gpu_order: Sequence[str | None],
    failed_gpu: str | None,
) -> List[str | None]:
    ordered = list(gpu_order)
    if failed_gpu is None or failed_gpu not in ordered or len(ordered) <= 1:
        return ordered
    ordered = [gpu for gpu in ordered if gpu != failed_gpu]
    ordered.append(failed_gpu)
    return ordered


def should_retry_failure(
    attempt: int,
    retry_limit: int | None,
    failure_kind: str,
) -> bool:
    normalized = normalize_retry_limit(retry_limit)
    if normalized is not None and attempt >= normalized:
        return False
    if failure_kind in ("no_results_written", "no_route_execution", "invalid_no_record", "planner_env_error"):
        return False
    if normalized is None:
        return attempt < SCENARIO_RETRY_HARD_CEILING
    if failure_kind == "generic":
        return attempt < min(normalized, GENERIC_SCENARIO_FAILURE_RETRY_ATTEMPTS)
    return True


def apply_gpu_failure_backoff(
    gpu_state: GPUPlannerState,
    *,
    failure_kind: str,
    args: argparse.Namespace,
) -> None:
    kind = str(failure_kind or "").strip().lower()
    gpu_state.failure_streak += 1
    gpu_state.success_streak = 0
    gpu_state.last_failure_kind = kind
    gpu_state.last_failure_monotonic = time.monotonic()
    if kind not in {"cuda_oom", "cuda_unavailable"}:
        return
    backoff_step = max(1, int(args.oom_concurrency_backoff_step))
    penalty_step = max(0, int(args.oom_free_mib_penalty_step))
    old_cap = int(gpu_state.max_concurrency)
    old_penalty = int(gpu_state.oom_penalty_mib)
    gpu_state.max_concurrency = max(1, int(gpu_state.max_concurrency) - backoff_step)
    gpu_state.oom_penalty_mib = max(0, int(gpu_state.oom_penalty_mib) + penalty_step)
    gpu_state.launch_capacity = min(
        gpu_state.launch_capacity,
        max(1, int(gpu_state.max_concurrency)),
    )
    print(
        "[WARN] GPU OOM backoff applied: "
        f"gpu={gpu_state.gpu} failure={kind} "
        f"max_concurrency {old_cap}->{gpu_state.max_concurrency}, "
        f"oom_penalty_mib {old_penalty}->{gpu_state.oom_penalty_mib}."
    )


def apply_gpu_success_recovery(
    gpu_state: GPUPlannerState,
    *,
    args: argparse.Namespace,
) -> None:
    gpu_state.success_streak += 1
    gpu_state.failure_streak = 0
    target_streak = max(1, int(args.oom_recovery_success_streak))
    if gpu_state.success_streak < target_streak:
        return
    gpu_state.success_streak = 0
    old_cap = int(gpu_state.max_concurrency)
    old_penalty = int(gpu_state.oom_penalty_mib)
    cap_limit = max(1, int(args.auto_workers_max_per_gpu))
    penalty_decay = max(0, int(args.oom_recovery_penalty_decay_step))
    gpu_state.max_concurrency = min(cap_limit, int(gpu_state.max_concurrency) + 1)
    if penalty_decay > 0:
        gpu_state.oom_penalty_mib = max(0, int(gpu_state.oom_penalty_mib) - penalty_decay)
    if gpu_state.max_concurrency != old_cap or gpu_state.oom_penalty_mib != old_penalty:
        print(
            "[INFO] GPU recovery applied: "
            f"gpu={gpu_state.gpu} max_concurrency {old_cap}->{gpu_state.max_concurrency}, "
            f"oom_penalty_mib {old_penalty}->{gpu_state.oom_penalty_mib}."
        )


def decay_stale_oom_penalties(
    gpu_states: Dict[str | None, GPUPlannerState],
    *,
    decay_seconds: float,
    cap_limit: int,
    tick_logger: "SchedulerTickLogger | None" = None,
) -> None:
    """Age out OOM penalties that have been idle for a while.

    Prevents a single early OOM from permanently throttling a GPU that would
    otherwise be healthy.  Fully restores ``max_concurrency`` to ``cap_limit``
    and ``oom_penalty_mib`` to zero once no new failure has arrived in the
    last ``decay_seconds`` and the GPU is currently idle.
    """
    if decay_seconds <= 0.0:
        return
    now = time.monotonic()
    for state in gpu_states.values():
        if state.gpu is None:
            continue
        if state.last_failure_monotonic <= 0.0:
            continue
        age_s = now - state.last_failure_monotonic
        if age_s < decay_seconds:
            continue
        if state.active_count > 0:
            # Don't relax while jobs are still running; wait for them to drain.
            continue
        old_cap = int(state.max_concurrency)
        old_penalty = int(state.oom_penalty_mib)
        if old_cap >= cap_limit and old_penalty <= 0:
            continue
        state.max_concurrency = max(1, int(cap_limit))
        state.oom_penalty_mib = 0
        state.failure_streak = 0
        state.last_failure_kind = ""
        state.last_failure_monotonic = 0.0
        print(
            "[INFO] GPU OOM backoff decayed: "
            f"gpu={state.gpu} age={age_s:.0f}s "
            f"max_concurrency {old_cap}->{state.max_concurrency} "
            f"oom_penalty_mib {old_penalty}->{state.oom_penalty_mib}."
        )
        if tick_logger is not None and tick_logger.enabled:
            tick_logger.log(
                "oom_backoff_decayed",
                gpu=str(state.gpu),
                age_seconds=round(age_s, 2),
                old_max_concurrency=old_cap,
                new_max_concurrency=state.max_concurrency,
                old_penalty_mib=old_penalty,
            )


def bin_pack_launch_plan(
    *,
    pending_requests: Sequence["PlannerRequest"],
    gpu_states: Dict[str | None, GPUPlannerState],
    gpu_monitor: "GPUMemoryMonitor | None",
    vram_estimator: PlannerVramEstimator,
    carla_fixed_mib: int,
    safety_buffer_mib: int,
    min_free_mib: int,
    active_non_colmdriver_threads: int,
    deferred_run_ids: set[str],
    managed_carla_single_worker_mode: bool,
    allow_multi_carla: bool,
    start_carla: bool,
    tick_logger: "SchedulerTickLogger | None",
) -> Tuple[List[Tuple["PlannerRequest", str | None, int]], List[Dict[str, Any]]]:
    """Compute which pending planners should launch this tick.

    Returns a list of ``(planner_request, gpu_id, vram_reservation_mib)`` tuples
    and a parallel list of rejection records (for debugging / instrumentation).

    The packing strategy is First-Fit-Decreasing on estimated VRAM: the
    biggest planner is placed on the GPU with the most remaining headroom
    first, then smaller ones fill whatever gaps remain.  Reserved MiB is
    tracked locally so a single dispatch tick can place multiple jobs
    against a GPU before nvidia-smi catches up.
    """
    decisions: List[Tuple["PlannerRequest", str | None, int]] = []
    rejections: List[Dict[str, Any]] = []

    # Snapshot live GPU stats once for consistency across the tick.
    gpu_snapshot: Dict[str | None, GPUMemoryStats | None] = {}
    for gpu_id, state in gpu_states.items():
        if gpu_id is None:
            gpu_snapshot[None] = None
            continue
        if gpu_monitor is not None:
            gpu_snapshot[gpu_id] = gpu_monitor.stats_for(gpu_id)
        else:
            gpu_snapshot[gpu_id] = state.last_stats

    # Local copy of reservations so we can tentatively commit within this tick.
    local_reserved: Dict[str | None, int] = {
        gpu_id: int(state.reserved_mib) for gpu_id, state in gpu_states.items()
    }
    local_active: Dict[str | None, int] = {
        gpu_id: int(state.active_count) for gpu_id, state in gpu_states.items()
    }
    # Track whether a CARLA server has already been counted on this GPU
    # (either because one is running, or because we plan to launch one
    # alongside a planner on it).
    # When start_carla is True: each child launches its own CARLA, so we
    # must account for CARLA VRAM per launch.
    # When start_carla is False: external CARLA VRAM is already reflected
    # in nvidia-smi's free_mib, so we add zero.
    per_launch_carla_mib = int(carla_fixed_mib) if start_carla else 0

    # Sort pending biggest-first; ties broken by original order for
    # reproducibility.
    sorted_pending = sorted(
        enumerate(pending_requests),
        key=lambda item: (
            -int(vram_estimator.estimate(item[1].name)),
            item[0],
        ),
    )

    for _, req in sorted_pending:
        # Deferred CoLMDriver gating: colmdriver variants cannot launch until
        # all non-colmdriver planners have drained.
        if req.run_id in deferred_run_ids and active_non_colmdriver_threads > 0:
            rejections.append(
                {
                    "run_id": req.run_id,
                    "reason": "deferred_phase_gating",
                    "active_non_colmdriver_threads": active_non_colmdriver_threads,
                }
            )
            continue

        planner_vram = int(vram_estimator.estimate(req.name))
        best_gpu_id: str | None = None
        best_free_after: int | None = None
        gpu_reject_reasons: List[Dict[str, Any]] = []

        for gpu_id, state in gpu_states.items():
            if state.gpu is None:
                # No CUDA slot (e.g. dry run); allow at most one job total.
                if local_active[gpu_id] >= state.max_concurrency:
                    gpu_reject_reasons.append(
                        {"gpu": None, "reason": "no_gpu_active_full"}
                    )
                    continue
                best_gpu_id = None
                best_free_after = None
                break

            max_concurrency = max(1, int(state.max_concurrency))
            if managed_carla_single_worker_mode and not allow_multi_carla:
                # Even in managed-single-carla mode, we can still bin-pack
                # multiple planners against the same CARLA (they share it),
                # but we cap at max_concurrency.
                pass
            if local_active[gpu_id] >= max_concurrency:
                gpu_reject_reasons.append(
                    {
                        "gpu": gpu_id,
                        "reason": "max_concurrency_reached",
                        "active": local_active[gpu_id],
                        "cap": max_concurrency,
                    }
                )
                continue

            stats = gpu_snapshot.get(gpu_id)
            if stats is None:
                gpu_reject_reasons.append(
                    {"gpu": gpu_id, "reason": "no_vram_snapshot"}
                )
                continue

            free_mib = int(stats.free_mib)
            # Subtract reservations we haven't yet seen reflected by nvidia-smi.
            # Active reservations already consume VRAM visible in stats.free_mib,
            # so only the *delta* from launches within this tick matters.
            pending_reservation = int(local_reserved[gpu_id]) - int(state.reserved_mib)
            effective_free = free_mib - max(0, pending_reservation)

            cost = planner_vram + per_launch_carla_mib
            cost += int(state.oom_penalty_mib)

            required_floor = int(min_free_mib) + int(safety_buffer_mib)
            if effective_free - cost < required_floor:
                gpu_reject_reasons.append(
                    {
                        "gpu": gpu_id,
                        "reason": "insufficient_free_mib",
                        "free_mib": free_mib,
                        "effective_free_mib": effective_free,
                        "planner_cost_mib": cost,
                        "required_floor_mib": required_floor,
                    }
                )
                continue

            free_after = effective_free - cost
            # Prefer the GPU that leaves the largest absolute headroom after
            # placement (worst-fit from the chosen planner's perspective).
            # This tends to leave room on the same GPU for a *different*
            # planner later while still keeping concurrency high elsewhere.
            if best_free_after is None or free_after > best_free_after:
                best_free_after = free_after
                best_gpu_id = gpu_id

        if best_gpu_id is None:
            rejections.append(
                {
                    "run_id": req.run_id,
                    "planner": req.name,
                    "planner_vram_estimate_mib": planner_vram,
                    "gpu_rejections": gpu_reject_reasons,
                }
            )
            continue

        # Commit locally.
        state = gpu_states[best_gpu_id]
        cost = planner_vram + per_launch_carla_mib
        cost += int(state.oom_penalty_mib)
        local_reserved[best_gpu_id] = int(local_reserved[best_gpu_id]) + int(cost)
        local_active[best_gpu_id] = int(local_active[best_gpu_id]) + 1
        decisions.append((req, best_gpu_id, cost))

        if tick_logger is not None and tick_logger.enabled:
            stats = gpu_snapshot.get(best_gpu_id)
            tick_logger.log(
                "launch_decision",
                run_id=req.run_id,
                planner=req.name,
                gpu=str(best_gpu_id) if best_gpu_id is not None else None,
                planner_vram_estimate_mib=planner_vram,
                reserved_mib=int(cost),
                gpu_free_mib=int(stats.free_mib) if stats is not None else None,
                gpu_used_mib=int(stats.used_mib) if stats is not None else None,
                gpu_total_mib=int(stats.total_mib) if stats is not None else None,
                active_after=int(local_active[best_gpu_id]),
            )

    return decisions, rejections


class MultiPlannerDashboard:
    def __init__(
        self,
        *,
        planner_states: Dict[str, PlannerDashboardState],
        planner_order: Sequence[str],
        ego_count: int,
        host: str,
        gpu_ids: Sequence[str],
        scenario_entries: Sequence[Tuple[str, str]],
        per_planner_scenario_entries: Dict[str, List[Tuple[str, str]]] | None = None,
        state_lock: threading.Lock,
        refresh_seconds: float,
        status_snapshot_interval_s: float,
        stall_output_timeout_s: float,
        enable: bool,
    ) -> None:
        self.planner_states = planner_states
        self.planner_order = list(planner_order)
        self.ego_count = max(0, int(ego_count))
        self.host = str(host)
        self.gpu_ids = [str(gpu) for gpu in gpu_ids if str(gpu).strip()]
        self.scenario_entries = list(scenario_entries)
        self._per_planner_scenario_entries: Dict[str, List[Tuple[str, str]]] = dict(per_planner_scenario_entries or {})
        self.state_lock = state_lock
        self.refresh_seconds = max(0.2, float(refresh_seconds))
        self.status_snapshot_interval_s = max(0.0, float(status_snapshot_interval_s))
        self.stall_output_timeout_s = max(1.0, float(stall_output_timeout_s))
        self.enable = bool(enable and tqdm is not None)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status_snapshot_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._overview_bar = None
        self._planner_bars: Dict[str, Any] = {}
        self._ego_bars: Dict[Tuple[str, int], Any] = {}
        self._planner_cache: Dict[str, Tuple[str, float, float]] = {}
        self._ego_cache: Dict[Tuple[str, int], Tuple[str, int, int]] = {}
        self._overview_cache: Tuple[str, int, int] | None = None

    def start(self) -> None:
        if not self.enable:
            return

        position = 0
        self._overview_bar = tqdm(
            total=1,
            position=position,
            leave=True,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        )
        position += 1
        for run_id in self.planner_order:
            self._planner_bars[run_id] = tqdm(
                total=1,
                position=position,
                leave=True,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
            )
            position += 1

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enable:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for bar in [self._overview_bar, *self._planner_bars.values()]:
            if bar is not None:
                try:
                    bar.close()
                except Exception:  # pylint: disable=broad-except
                    pass

    def _loop(self) -> None:
        while not self._stop_event.wait(self.refresh_seconds):
            self._refresh()
        self._refresh()

    def _refresh(self) -> None:
        if not self.enable:
            return

        gpu_stats = query_gpu_memory_stats(self.gpu_ids)
        queued = 0
        running = 0
        finished = 0
        failed = 0
        interrupted = 0

        with self.state_lock:
            snapshot = {
                run_id: PlannerDashboardState(
                    planner_request=state.planner_request,
                    results_dir=state.results_dir,
                    log_path=state.log_path,
                    status_path=state.status_path,
                    default_results_subdir=state.default_results_subdir,
                    phase=state.phase,
                    slot_index=state.slot_index,
                    gpu=state.gpu,
                    ports=state.ports,
                    runner=state.runner,
                    start_time=state.start_time,
                    end_time=state.end_time,
                    returncode=state.returncode,
                    error=state.error,
                    route_progress=state.route_progress,
                    entry_status=state.entry_status,
                    last_output_age_s=state.last_output_age_s,
                    ego_route_scores=dict(state.ego_route_scores),
                    ego_status=dict(state.ego_status),
                    carla_up=state.carla_up,
                    tm_up=state.tm_up,
                    scenario_index=state.scenario_index,
                    scenario_total=state.scenario_total,
                    completed_scenarios=state.completed_scenarios,
                    current_scenario=state.current_scenario,
                    current_scenario_results_subdir=state.current_scenario_results_subdir,
                    current_attempt=state.current_attempt,
                    attempt_limit=state.attempt_limit,
                )
                for run_id, state in self.planner_states.items()
            }

        for run_id in self.planner_order:
            state = snapshot[run_id]
            now_mono = time.monotonic()
            cached_status = self._status_snapshot_cache.get(run_id)
            if (
                cached_status is not None
                and self.status_snapshot_interval_s > 0.0
                and (now_mono - cached_status[0]) < self.status_snapshot_interval_s
            ):
                status_payload = dict(cached_status[1])
            else:
                status_payload = load_json_file(state.status_path)
                self._status_snapshot_cache[run_id] = (now_mono, dict(status_payload))
            live_ego_route_scores: Dict[int, float] = {}
            if status_payload:
                try:
                    state.scenario_total = max(1, int(status_payload.get("scenario_total", state.scenario_total)))
                except Exception:
                    pass
                try:
                    state.scenario_index = max(0, int(status_payload.get("scenario_index", state.scenario_index)))
                except Exception:
                    pass
                try:
                    state.completed_scenarios = max(
                        0,
                        int(status_payload.get("completed_scenarios", state.completed_scenarios)),
                    )
                except Exception:
                    pass
                state.current_scenario = str(
                    status_payload.get("scenario_name") or state.current_scenario or ""
                )
                state.current_scenario_results_subdir = str(
                    status_payload.get("scenario_results_subdir")
                    or state.current_scenario_results_subdir
                    or ""
                )
                current_attempt = status_payload.get("current_attempt")
                attempt_limit = status_payload.get("attempt_limit")
                try:
                    state.current_attempt = None if current_attempt is None else int(current_attempt)
                except Exception:
                    pass
                try:
                    state.attempt_limit = None if attempt_limit is None else int(attempt_limit)
                except Exception:
                    pass
                live_ego_route_scores = parse_live_ego_route_scores(status_payload)

            _planner_entries = self._per_planner_scenario_entries.get(run_id, self.scenario_entries)
            if _planner_entries:
                scenario_summaries: List[Tuple[str, str, ResultRootProgress]] = []
                completed_scenarios = 0
                for label, relative_dir in _planner_entries:
                    summary = collect_result_root_progress(
                        state.results_dir / relative_dir,
                        ego_count=self.ego_count,
                    )
                    if summary.completed:
                        completed_scenarios += 1
                    scenario_summaries.append((label, relative_dir, summary))

                preferred_rel_dir = state.current_scenario_results_subdir or ""
                selected_idx = None
                if preferred_rel_dir:
                    for idx, (_, relative_dir, _) in enumerate(scenario_summaries):
                        if relative_dir == preferred_rel_dir:
                            selected_idx = idx
                            break
                if selected_idx is None:
                    for idx, (_, _, summary) in enumerate(scenario_summaries):
                        if summary.started and not summary.completed:
                            selected_idx = idx
                            break
                if selected_idx is None:
                    for idx, (_, _, summary) in enumerate(scenario_summaries):
                        if not summary.started:
                            selected_idx = idx
                            break
                if selected_idx is None and scenario_summaries:
                    selected_idx = len(scenario_summaries) - 1

                state.scenario_total = max(1, len(scenario_summaries))
                state.completed_scenarios = max(state.completed_scenarios, completed_scenarios)
                if selected_idx is not None:
                    label, relative_dir, summary = scenario_summaries[selected_idx]
                    state.scenario_index = selected_idx + 1
                    state.current_scenario = label
                    state.current_scenario_results_subdir = relative_dir
                    state.route_progress = summary.route_progress
                    state.entry_status = summary.entry_status or state.entry_status or state.phase
                    state.ego_route_scores = {}
                    state.ego_status = {}
            else:
                active_relative_dir = state.current_scenario_results_subdir or state.default_results_subdir
                active_root = state.results_dir / active_relative_dir if active_relative_dir else state.results_dir
                summary = collect_result_root_progress(
                    active_root,
                    ego_count=self.ego_count,
                    fallback_status=state.entry_status or state.phase,
                )
                state.route_progress = summary.route_progress
                state.entry_status = summary.entry_status or state.entry_status or state.phase
                state.ego_route_scores = {}
                state.ego_status = {}
                state.scenario_total = max(1, state.scenario_total)
                if state.scenario_total <= 1:
                    state.scenario_index = 1 if state.phase != "queued" else 0

            if live_ego_route_scores:
                for ego_idx, route_score in live_ego_route_scores.items():
                    state.ego_route_scores[ego_idx] = max(
                        route_score,
                        float(state.ego_route_scores.get(ego_idx, 0.0)),
                    )

            phase = state.phase
            if phase in ("launching", "running") and state.ports is not None:
                state.carla_up = is_port_open(self.host, state.ports.port, timeout=0.2)
                state.tm_up = is_port_open(self.host, state.ports.tm_port, timeout=0.2)
            else:
                state.carla_up = None
                state.tm_up = None

            if state.runner is not None:
                runner_snapshot = state.runner.snapshot()
                state.last_output_age_s = runner_snapshot["last_output_age_s"]

            if phase == "queued":
                queued += 1
            elif phase in ("launching", "running"):
                running += 1
            elif phase == "finished":
                if state.returncode == 0:
                    finished += 1
                else:
                    failed += 1
            elif phase == "failed":
                failed += 1
            elif phase == "interrupted":
                interrupted += 1

            planner_bar = self._planner_bars.get(run_id)
            if planner_bar is None:
                continue
            route_total = max(1, int(state.route_progress[1]))
            route_value = max(0, min(int(state.route_progress[0]), route_total))
            live_route_scores = [
                max(0.0, min(100.0, float(route_score)))
                for route_score in state.ego_route_scores.values()
            ]
            live_route_score = (
                sum(live_route_scores) / float(len(live_route_scores))
                if live_route_scores
                else None
            )
            if state.scenario_total > 1:
                route_fraction = float(route_value) / float(route_total)
                planner_total = float(max(1, state.scenario_total))
                planner_value = min(
                    planner_total,
                    max(0.0, float(state.completed_scenarios)) + route_fraction,
                )
            elif live_route_score is not None:
                planner_total = 100.0
                planner_value = float(int(round(live_route_score)))
            else:
                planner_total = float(route_total)
                planner_value = float(route_value)
            health = "queued"
            if phase in ("launching", "running"):
                if state.carla_up is False:
                    health = "carla-down"
                elif state.last_output_age_s is not None and state.last_output_age_s >= self.stall_output_timeout_s:
                    health = "quiet"
                else:
                    health = "ok"
            elif phase == "finished":
                health = "ok" if state.returncode == 0 else "failed"
            elif phase == "failed":
                health = "failed"
            elif phase == "interrupted":
                health = "interrupted"
            scenario_label = ""
            if state.scenario_total > 1:
                scenario_idx = state.scenario_index or min(
                    state.completed_scenarios + 1,
                    state.scenario_total,
                )
                scenario_label = f" scn={scenario_idx}/{state.scenario_total}"
            attempt_label = ""
            if state.current_attempt is not None:
                attempt_label = (
                    f" attempt={format_attempt_counter(state.current_attempt, state.attempt_limit)}"
                )
            carla_status = "-"
            if phase in ("launching", "running"):
                carla_status = "up" if state.carla_up else "down" if state.carla_up is not None else "-"
            planner_line = (
                f"{run_id} {phase} gpu={state.gpu or '-'} "
                f"carla={carla_status} health={health}"
                f"{scenario_label}"
            )
            if state.current_scenario:
                planner_line += f" [{state.current_scenario}]"
            if live_route_score is not None:
                planner_line += f" rc={live_route_score:4.1f}%"
            planner_line += attempt_label
            planner_key = (planner_line, planner_value, planner_total)
            if self._planner_cache.get(run_id) != planner_key:
                planner_bar.total = planner_total
                planner_bar.n = planner_value
                planner_bar.set_description_str(planner_line)
                planner_bar.refresh()
                self._planner_cache[run_id] = planner_key

        # Overview bar: aggregate scenario progress across all planners.
        total_scenario_work = 0
        completed_scenario_work = 0
        with self.state_lock:
            for st in self.planner_states.values():
                total_scenario_work += max(1, st.scenario_total)
                completed_scenario_work += st.completed_scenarios
        gpu_summary_parts: List[str] = []
        for gpu_id in self.gpu_ids:
            stats = gpu_stats.get(gpu_id)
            if stats is not None:
                used_pct = 100 * stats.used_mib // max(1, stats.total_mib)
                gpu_summary_parts.append(
                    f"GPU{gpu_id}:{stats.used_mib}MiB({used_pct}%)"
                )
        gpu_suffix = f"  {' '.join(gpu_summary_parts)}" if gpu_summary_parts else ""
        overview_desc = (
            f"Overall run={running} q={queued} done={finished} fail={failed}"
            f"{gpu_suffix}"
        )
        overview_key = (overview_desc, completed_scenario_work, max(1, total_scenario_work))
        if self._overview_bar is not None and self._overview_cache != overview_key:
            self._overview_bar.total = max(1, total_scenario_work)
            self._overview_bar.n = min(completed_scenario_work, max(1, total_scenario_work))
            self._overview_bar.set_description_str(overview_desc)
            self._overview_bar.refresh()
            self._overview_cache = overview_key

def infer_base_results_tag(args: argparse.Namespace) -> str:
    if args.results_tag:
        return str(args.results_tag)
    if args.scenario_name:
        return str(args.scenario_name)
    if args.zip is not None:
        return Path(args.zip).stem
    if args.routes_dir is not None:
        return Path(args.routes_dir).name
    return "custom_eval"


def combine_results_subdir(base_subdir: str | None, leaf: str) -> str:
    if not base_subdir:
        return leaf
    return f"{base_subdir.rstrip('/')}/{leaf}"


def format_repro_rerun_label(index: int) -> str:
    return f"repro_rerun_{max(1, int(index)):03d}"


def iter_variant_result_tags(
    effective_results_tag: str,
    args: argparse.Namespace,
) -> List[str]:
    tags: List[str] = []
    weather_suffixes: List[str | None] = []
    if args.weather:
        weather_suffixes.append(args.weather.lower())
    else:
        weather_suffixes.append(None)
        if args.add_night:
            weather_suffixes.append("night")
        if args.add_cloudy:
            weather_suffixes.append("cloudy")
        if args.add_npcs:
            weather_suffixes.append("npcs")

    negotiation_suffixes: List[str | None] = [None]
    if args.add_no_negotiation:
        negotiation_suffixes.append("nonego")

    for weather_suffix in weather_suffixes:
        for negotiation_suffix in negotiation_suffixes:
            suffix_parts = [part for part in (weather_suffix, negotiation_suffix) if part]
            suffix = "_".join(suffix_parts) if suffix_parts else None
            tags.append(
                effective_results_tag if not suffix else f"{effective_results_tag}_{suffix}"
            )
    return tags


def planned_result_roots(
    repo_root: Path,
    effective_results_tag: str,
    args: argparse.Namespace,
) -> List[Path]:
    roots: List[Path] = []
    for tag in iter_variant_result_tags(effective_results_tag, args):
        root = repo_root / "results" / "results_driving_custom" / tag
        if root not in roots:
            roots.append(root)
    return roots


def delete_result_roots(result_roots: Sequence[Path]) -> None:
    """Rename partial result directories instead of deleting them.

    The renamed directory gets a ``_partial_<timestamp>`` suffix so that
    partial artifacts are preserved for debugging while still clearing the
    path for the next attempt.
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for root in result_roots:
        if not root.exists():
            continue
        dest = root.parent / f"{root.name}_partial_{timestamp}"
        # Avoid collisions if multiple retries happen in the same second.
        seq = 0
        while dest.exists():
            seq += 1
            dest = root.parent / f"{root.name}_partial_{timestamp}_{seq}"
        print(f"[INFO] Renaming partial result directory: {root} -> {dest}")
        try:
            root.rename(dest)
        except OSError as exc:
            print(f"[WARN] Could not rename {root}: {exc}; falling back to removal.")
            shutil.rmtree(root, ignore_errors=True)


def _matches_value_option(token: str, option: str) -> bool:
    return token == option or token.startswith(f"{option}=")


def strip_cli_options(
    argv: Sequence[str],
    *,
    value_options: Sequence[str] = (),
    flag_options: Sequence[str] = (),
) -> List[str]:
    stripped: List[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        matched_value_option = None
        for option in value_options:
            if _matches_value_option(token, option):
                matched_value_option = option
                break
        if matched_value_option is not None:
            idx += 1 if "=" in token else 2
            continue
        if token in flag_options:
            idx += 1
            continue
        stripped.append(token)
        idx += 1
    return stripped


def build_child_argv(
    base_argv: Sequence[str],
    *,
    replacements: Sequence[str],
    value_options_to_remove: Sequence[str] = (),
    flag_options_to_remove: Sequence[str] = (),
) -> List[str]:
    child_argv = strip_cli_options(
        base_argv,
        value_options=value_options_to_remove,
        flag_options=flag_options_to_remove,
    )
    child_argv.extend(replacements)
    return child_argv


def dispatch_repro_reruns(
    *,
    args: argparse.Namespace,
    base_argv: Sequence[str],
    repo_root: Path,
) -> int:
    total = max(1, int(args.repro_reruns))
    if total <= 1:
        return 0

    if not args.start_carla:
        print(
            "[WARN] --repro-reruns is active without --start-carla. "
            "Each rerun will reuse the same external CARLA session unless you restart it yourself."
        )

    script_path = Path(__file__).resolve()
    failures: List[Tuple[int, int | None]] = []
    print(
        "[INFO] Reproduction mode enabled: "
        f"launching {total} isolated run_custom_eval reruns of the same invocation."
    )
    for rerun_index in range(1, total + 1):
        rerun_label = format_repro_rerun_label(rerun_index)
        rerun_results_subdir = combine_results_subdir(args.results_subdir, rerun_label)
        child_argv = build_child_argv(
            base_argv,
            replacements=[
                "--results-subdir",
                rerun_results_subdir,
                "--repro-rerun-index",
                str(rerun_index),
                "--repro-rerun-total",
                str(total),
            ],
            value_options_to_remove=[
                "--results-subdir",
                "--repro-reruns",
                "--rerun-same-scenario",
                "--repro-rerun-index",
                "--repro-rerun-total",
            ],
        )
        cmd = [sys.executable, str(script_path), *child_argv]
        print(f"\n{'=' * 72}")
        print(f"REPRO RERUN {rerun_index}/{total}")
        print(f"{'=' * 72}")
        print("Results subdir:", rerun_results_subdir)
        print("Command:")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        result = MonitoredSubprocessRunner(
            cmd,
            env=os.environ.copy(),
            cwd=repo_root,
            timeout_s=None,
            stall_output_timeout_s=0.0,
            output_prefix=f"[repro:{rerun_index:03d}] ",
            show_carla_diag=False,
        ).run()
        if result.returncode != 0:
            failures.append((rerun_index, result.returncode))

    if args.dry_run:
        print("[INFO] Dry run: reproduction rerun commands were printed but not executed.")
        return 0

    if failures:
        summary = ", ".join(
            f"{format_repro_rerun_label(index)}:returncode={returncode}"
            for index, returncode in failures
        )
        print(f"[ERROR] Reproduction reruns completed with failures: {summary}")
        first_code = failures[0][1]
        return int(first_code) if first_code not in (None, 0) else 1

    print(f"[INFO] Completed {total} reproduction reruns successfully.")
    return 0


def add_suffix_to_path(path: Path, suffix: str) -> Path:
    suffix = sanitize_run_id(suffix)
    if not path.suffix:
        return path.with_name(f"{path.name}_{suffix}")
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def find_carla_egg(carla_root: Path) -> Path:
    """
    Locate a CARLA Python client build compatible with the current interpreter.

    Priority:
    1) Wheels/eggs under PythonAPI/carla/dist whose filename matches our py3{minor} or cp3{minor}.
    2) Same search in PythonAPI_docker/carla/dist (useful when only docker-built eggs exist).
    3) Any py3*/cp3* egg/wheel under those dirs.
    4) Fallback to the first available artifact.
    """
    search_dirs = [
        carla_root / "PythonAPI" / "carla" / "dist",
        carla_root / "PythonAPI_docker" / "carla" / "dist",
    ]
    artifacts: list[Path] = []
    for dist_dir in search_dirs:
        artifacts.extend(sorted(dist_dir.glob("carla-*.egg")))
        artifacts.extend(sorted(dist_dir.glob("carla-*.whl")))

    if not artifacts:
        raise FileNotFoundError(
            f"No CARLA egg/wheel found under {search_dirs[0]} or {search_dirs[1]}"
        )

    minor = sys.version_info.minor
    def matches_minor(art: Path) -> bool:
        name = art.name
        return (
            f"cp3{minor}" in name
            or f"py3{minor}" in name
            or f"py3.{minor}" in name
        )

    preferred = [art for art in artifacts if matches_minor(art)]
    if preferred:
        return preferred[0]

    py3 = [art for art in artifacts if "py3" in art.name or "cp3" in art.name]
    if py3:
        return py3[0]

    return artifacts[0]


def _ensure_carla_import_for_overlay(carla_root: Path) -> object:
    carla_python = carla_root / "PythonAPI"
    carla_pkg = carla_root / "PythonAPI" / "carla"
    carla_artifact = find_carla_egg(carla_root)
    for path in (carla_python, carla_pkg, carla_artifact):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
    import carla  # type: ignore

    return carla


def _load_crosswalk_experiment_module(repo_root: Path):
    module_path = repo_root / "v2xpnp" / "scripts" / "crosswalk_experiment.py"
    module_name = "crosswalk_experiment_runtime"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    # Register before exec so dataclass/type resolution can find module globals.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


class UCLAV2CrosswalkWorker:
    def __init__(
        self,
        repo_root: Path,
        carla_root: Path,
        host: str,
        port: int,
        cache_file: Path,
        trim_outermost_per_side: int,
        high_z_prune_count: int,
        z_mode: str,
        poll_sec: float,
    ) -> None:
        self.repo_root = repo_root
        self.carla_root = carla_root
        self.host = str(host)
        self.port = int(port)
        self.cache_file = Path(cache_file)
        self.trim_outermost_per_side = max(0, int(trim_outermost_per_side))
        self.high_z_prune_count = max(0, int(high_z_prune_count))
        self.z_mode = str(z_mode)
        self.poll_sec = max(0.2, float(poll_sec))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Use bounded-life non-persistent debug lines to prevent unbounded haze buildup
        # even if CARLA keeps a persistent debug cache across world transitions.
        self._draw_refresh_sec = max(3.0, 2.0 * self.poll_sec)
        self._draw_life_time_sec = self._draw_refresh_sec + 0.75
        self._last_apply_monotonic = 0.0
        self._warned_cache_miss = False
        self._apply_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ucla_v2_crosswalk_worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        try:
            carla = _ensure_carla_import_for_overlay(self.carla_root)
            cw = _load_crosswalk_experiment_module(self.repo_root)
            cache = cw.load_surface_z_cache(self.cache_file)
        except Exception as exc:
            print(f"[WARN] Crosswalk worker init failed: {exc}")
            return

        client = create_logged_client(
            carla,
            self.host,
            self.port,
            timeout_s=2.0,
            context="ucla_v2_crosswalk_worker",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        )
        try:
            while not self._stop_event.is_set():
                frame = -1
                try:
                    world = client.get_world()
                    map_name = str(world.get_map().name).lower()
                    if UCLA_V2_CROSSWALK_MAP_TOKEN not in map_name:
                        time.sleep(self.poll_sec)
                        continue

                    snapshot = world.get_snapshot()
                    frame = int(snapshot.frame) if snapshot is not None else -1
                    now = time.monotonic()
                    if self._last_apply_monotonic > 0.0 and (
                        now - self._last_apply_monotonic
                    ) < self._draw_refresh_sec:
                        time.sleep(self.poll_sec)
                        continue

                    cw.apply_ucla_v2_crosswalk(
                        world=world,
                        carla_map_cache=self.repo_root / "v2xpnp" / "map" / "carla_map_cache.pkl",
                        stripe_step_override=0.0,
                        cid_z_boosts={},
                        z_mode=self.z_mode,
                        z_cache=cache,
                        trim_outermost_per_side=self.trim_outermost_per_side,
                        high_z_prune_count=self.high_z_prune_count,
                        # Match standalone script behavior: use cache first, but allow
                        # live sampling fallback for robust geometry.
                        surface_z_cache_only=False,
                        draw_life_time_s=self._draw_life_time_sec,
                        draw_persistent_lines=False,
                    )
                    self._last_apply_monotonic = now
                    self._apply_count += 1
                    print(
                        f"[INFO] UCLA v2 crosswalk applied on {world.get_map().name} "
                        f"frame={frame} apply_count={self._apply_count} "
                        f"life={self._draw_life_time_sec:.2f}s refresh={self._draw_refresh_sec:.2f}s"
                    )
                except Exception as exc:
                    # Keep retries bounded after failures; non-persistent lines prevent
                    # unbounded stacking, but we still rate-limit retries.
                    self._last_apply_monotonic = time.monotonic()
                    if not self._warned_cache_miss:
                        print(
                            "[WARN] UCLA v2 crosswalk injection failed "
                            f"(cache/load/sampling issue): {exc}"
                        )
                        self._warned_cache_miss = True
                time.sleep(self.poll_sec)
        finally:
            release_client_id = log_client_release_begin(
                client,
                context="ucla_v2_crosswalk_worker",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                reason="crosswalk_worker_scope_end",
            )
            # Release the libcarla C++ object so its destructor closes the TCP
            # socket synchronously (CPython refcount drops to zero here).
            client = None  # noqa: F841
            log_client_release_end(
                release_client_id,
                context="ucla_v2_crosswalk_worker",
                process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
                reason="crosswalk_worker_scope_end",
            )

        try:
            cw.save_surface_z_cache(self.cache_file, cache)
        except Exception:
            pass


def detect_ego_routes(routes_dir: Path, replay_mode: bool = False) -> int:
    """
    Count ego route XML files for the active planner mode.

    replay_mode=False -> count non-REPLAY ego routes (default planners, e.g. TCP)
    replay_mode=True  -> count *_REPLAY ego routes (log-replay planner)
    """
    count = 0
    for xml_path in routes_dir.rglob("*.xml"):
        if "actors" in os.path.normpath(str(xml_path)).split(os.sep):
            continue
        is_replay_file = "_REPLAY" in xml_path.stem
        if replay_mode and not is_replay_file:
            continue
        if not replay_mode and is_replay_file:
            continue
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            count += 1
    return count


def detect_scenario_subfolders(routes_dir: Path, routes_leaf: str | None = None) -> List[tuple[Path, str]]:
    """
    Check if routes_dir contains multiple scenario subfolders (each with their own ego routes).

    Returns a list of subfolders that each contain at least one ego route.
    If routes_dir itself contains ego routes directly (not in subfolders), returns empty list.
    """
    # First check if there are ego routes directly in routes_dir (not in subfolders)
    for xml_path in routes_dir.glob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
            if role == "ego":
                # routes_dir itself is a scenario directory
                return []
        except Exception:
            continue

    # Check subfolders for ego routes
    scenario_folders: list[tuple[Path, str]] = []
    for subdir in sorted(routes_dir.iterdir()):
        if not subdir.is_dir():
            continue
        candidates = []
        if routes_leaf:
            leaf = subdir / routes_leaf
            if leaf.is_dir():
                candidates.append(leaf)
        candidates.append(subdir)

        selected = None
        selected_name = None
        for candidate in candidates:
            has_ego = False
            for xml_path in candidate.rglob("*.xml"):
                try:
                    _, _, role = parse_route_metadata(xml_path.read_bytes())
                    if role == "ego":
                        has_ego = True
                        break
                except Exception:
                    continue
            if has_ego:
                selected = candidate
                selected_name = subdir.name
                break

        if selected is not None and selected_name is not None:
            scenario_folders.append((selected, selected_name))

    return scenario_folders


def read_manifest(routes_dir: Path) -> tuple[Path | None, Dict[str, List[Dict[str, str]]] | None]:
    manifest_path = routes_dir / "actors_manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return manifest_path, None
    return manifest_path, data


def _extract_trailing_index(path: Path) -> int | None:
    stem = path.stem
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _collect_ego_route_records(
    routes_dir: Path,
    replay_mode: bool = False,
) -> List[Tuple[int, Path, str, str, str | None]]:
    """
    Collect ego route XML records from routes_dir.

    replay_mode=True  → only collect *_REPLAY.xml files (for --planner log-replay)
    replay_mode=False → skip *_REPLAY.xml files (for all other planners, use GRP-simplified)
    """
    records: List[Tuple[int, Path, str, str, str | None]] = []
    fallback_idx = 0
    for xml_path in sorted(routes_dir.rglob("*.xml")):
        if "actors" in os.path.normpath(str(xml_path)).split(os.sep):
            continue
        is_replay_file = "_REPLAY" in xml_path.stem
        if replay_mode and not is_replay_file:
            continue   # log-replay: only want _REPLAY files
        if not replay_mode and is_replay_file:
            continue   # other planners: skip _REPLAY files
        try:
            route_id, town, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role != "ego":
            continue

        # Strip _REPLAY suffix before extracting the numeric ego index
        stem_clean = xml_path.stem.replace("_REPLAY", "")
        ego_idx = _extract_trailing_index(Path(stem_clean))
        if ego_idx is None:
            ego_idx = fallback_idx
            fallback_idx += 1

        model = None
        try:
            xml_root = ET.parse(str(xml_path)).getroot()
            route_node = xml_root.find("route")
            if route_node is not None:
                model = route_node.attrib.get("model")
        except ET.ParseError:
            pass

        records.append((int(ego_idx), xml_path, str(route_id), str(town), model))

    records.sort(key=lambda item: (item[0], str(item[1])))
    return records


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_waypoint_speeds(
    waypoints: List[Tuple[float, float, float]],
    times: List[float],
    default_dt: float = 0.1,
) -> List[float]:
    """
    Compute per-waypoint linear speeds (m/s) from waypoint positions and timestamps.
    """
    n = len(waypoints)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    speeds = [0.0] * n
    for i in range(n - 1):
        x0, y0, z0 = waypoints[i]
        x1, y1, z1 = waypoints[i + 1]
        dt = default_dt
        if i < len(times) - 1:
            cand_dt = float(times[i + 1]) - float(times[i])
            if cand_dt > 1e-3:
                dt = cand_dt
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
        speeds[i] = max(0.0, float(dist) / max(dt, 1e-3))
    speeds[-1] = speeds[-2]
    return speeds


def _suppress_initial_jitter_for_stationary_start(
    waypoints: List[Tuple[float, float, float]],
    times: List[float],
    speeds: List[float],
) -> List[float]:
    """
    Force zero speed for an initial near-static prefix and clamp tiny crawl speeds.
    This prevents parked actors from rolling early due waypoint jitter.
    """
    if not waypoints or not speeds:
        return speeds
    if len(waypoints) != len(speeds):
        return speeds

    n = len(waypoints)
    # Keep suppression conservative: only remove startup creep in a short prefix.
    move_speed_thr = 0.25  # m/s
    move_disp_thr = 0.6  # m from initial point
    max_hold_seconds = 1.5

    t_vals: List[float] = []
    if len(times) == n:
        t_vals = [float(t) for t in times]
    else:
        t_vals = [0.1 * float(i) for i in range(n)]

    t0 = float(t_vals[0]) if t_vals else 0.0
    cap_idx = n
    for i in range(1, n):
        if float(t_vals[i]) - t0 >= max_hold_seconds:
            cap_idx = i
            break

    x0, y0, z0 = waypoints[0]
    move_idx = None
    for i in range(1, n):
        xi, yi, zi = waypoints[i]
        disp = ((xi - x0) ** 2 + (yi - y0) ** 2 + (zi - z0) ** 2) ** 0.5
        seg_speed = max(float(speeds[i - 1]), float(speeds[i]))
        if disp >= move_disp_thr or seg_speed >= move_speed_thr:
            move_idx = i
            break

    if move_idx is None:
        # Never zero an entire trajectory; return original profile.
        return speeds

    hold_until = max(0, min(int(move_idx), int(cap_idx)))

    out = list(speeds)
    for i in range(min(hold_until, len(out))):
        out[i] = 0.0

    # Clamp tiny crawl speeds globally to avoid PID creeping on noisy tracks.
    for i, s in enumerate(out):
        if float(s) < 0.05:
            out[i] = 0.0

    return out


def _estimate_target_speed_from_speeds(speeds: List[float]) -> float | None:
    if not speeds:
        return None
    moving = [float(s) for s in speeds if float(s) >= 0.3]
    sample = moving if moving else [float(s) for s in speeds]
    if not sample:
        return None
    sample_sorted = sorted(sample)
    idx = min(len(sample_sorted) - 1, max(0, int(0.75 * (len(sample_sorted) - 1))))
    return max(0.0, float(sample_sorted[idx]))


def _build_fake_ego_npc_xml(
    src_xml: Path,
    dst_xml: Path,
    control_mode: str,
    model: str | None,
) -> float | None:
    """
    Convert an ego route XML into an NPC-style route XML suitable for custom actor policy/replay control.
    Returns the computed target speed (m/s) when available.
    """
    root = ET.parse(str(src_xml)).getroot()
    route_node = root.find("route")
    if route_node is None:
        raise RuntimeError(f"Missing <route> in {src_xml}")

    town = route_node.attrib.get("town", "Town01")
    route_id = route_node.attrib.get("id", "0")
    waypoints = list(route_node.iter("waypoint"))
    if not waypoints:
        raise RuntimeError(f"No <waypoint> entries in {src_xml}")

    xyz: List[Tuple[float, float, float]] = []
    times: List[float] = []
    for wp in waypoints:
        x = _safe_float(wp.attrib.get("x")) or 0.0
        y = _safe_float(wp.attrib.get("y")) or 0.0
        z = _safe_float(wp.attrib.get("z")) or 0.0
        xyz.append((float(x), float(y), float(z)))
        t = _safe_float(wp.attrib.get("time"))
        if t is None:
            t = _safe_float(wp.attrib.get("t"))
        times.append(float(t) if t is not None else 0.0)

    speeds = _compute_waypoint_speeds(xyz, times, default_dt=0.1)
    speeds = _suppress_initial_jitter_for_stationary_start(xyz, times, speeds)
    target_speed = _estimate_target_speed_from_speeds(speeds)

    out_root = ET.Element("routes")
    route_attrs = {
        "id": str(route_id),
        "town": str(town),
        "role": "npc",
        # Let fake-egos use the same snap_to_road behavior as regular NPCs (defaults to true)
        # This ensures they are properly aligned to road lanes
        "control_mode": str(control_mode),
    }
    if model:
        route_attrs["model"] = str(model)
    if target_speed is not None and target_speed > 0.0:
        route_attrs["target_speed"] = f"{float(target_speed):.2f}"

    out_route = ET.SubElement(out_root, "route", route_attrs)
    for i, wp in enumerate(waypoints):
        attrs = dict(wp.attrib)
        if i < len(speeds):
            attrs["speed"] = f"{float(speeds[i]):.4f}"
        ET.SubElement(out_route, "waypoint", attrs)

    out_tree = ET.ElementTree(out_root)
    if hasattr(ET, "indent"):
        ET.indent(out_tree, space="  ")
    out_tree.write(str(dst_xml), encoding="utf-8", xml_declaration=True)
    return target_speed


def prepare_npc_only_fake_ego_routes(
    routes_dir: Path,
    fake_ego_control_mode: str,
    replay_mode: bool = False,
) -> Tuple[Path, tempfile.TemporaryDirectory, List[Tuple[int, str]], Path]:
    """
    Create an isolated routes clone where role=ego XML files are also injected as NPC custom actors.

    This keeps original planner-compatible ego XML artifacts unchanged while enabling a no-ego runtime
    that drives those trajectories through the custom-actor stack.

    replay_mode=True  → uses *_REPLAY.xml ego files (for --planner log-replay)
    replay_mode=False → uses GRP-simplified ego files, skips *_REPLAY.xml
    """
    temp_dir = tempfile.TemporaryDirectory(prefix=f"{routes_dir.name}_npc_only_")
    clone_root = Path(temp_dir.name) / routes_dir.name
    shutil.copytree(routes_dir, clone_root)

    ego_records = _collect_ego_route_records(clone_root, replay_mode=replay_mode)
    if not ego_records:
        temp_dir.cleanup()
        raise RuntimeError(
            f"--npc-only-fake-ego requested, but no role=\"ego\" routes were found under {routes_dir}."
        )

    manifest_path = clone_root / "actors_manifest.json"
    manifest_data: Dict[str, List[Dict[str, str]]]
    if manifest_path.exists():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data = raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            manifest_data = {}
    else:
        manifest_data = {}

    npc_entries = manifest_data.get("npc")
    if not isinstance(npc_entries, list):
        npc_entries = []
        manifest_data["npc"] = npc_entries

    ego_model_by_file: Dict[str, str] = {}
    ego_entries = manifest_data.get("ego")
    if isinstance(ego_entries, list):
        for entry in ego_entries:
            if not isinstance(entry, dict):
                continue
            rel_file = entry.get("file")
            rel_model = entry.get("model")
            if rel_file and rel_model:
                ego_model_by_file[str(rel_file)] = str(rel_model)

    existing_names = set()
    for role_entries in manifest_data.values():
        if not isinstance(role_entries, list):
            continue
        for entry in role_entries:
            if isinstance(entry, dict):
                name = entry.get("name")
                if name:
                    existing_names.add(str(name))

    fake_ego_views: List[Tuple[int, str]] = []
    used_camera_ids = set()

    fake_npc_dir = clone_root / "actors" / "npc"
    fake_npc_dir.mkdir(parents=True, exist_ok=True)
    for ego_idx, ego_xml, route_id, town, model in ego_records:
        base_name = f"fake_ego_{ego_idx}"
        name = base_name
        suffix = 1
        while name in existing_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        existing_names.add(name)

        fake_xml_name = f"{str(town).lower()}_custom_FakeEgo_{ego_idx}_npc.xml"
        fake_xml_path = fake_npc_dir / fake_xml_name

        if not model:
            rel_ego_file = ego_xml.relative_to(clone_root).as_posix()
            model = ego_model_by_file.get(rel_ego_file)
        if not model or str(model).strip().lower() in ("ego", "vehicle.ego", "hero"):
            model = "vehicle.lincoln.mkz_2017"

        target_speed = _build_fake_ego_npc_xml(
            src_xml=ego_xml,
            dst_xml=fake_xml_path,
            control_mode=str(fake_ego_control_mode),
            model=str(model),
        )
        rel_file = fake_xml_path.relative_to(clone_root).as_posix()
        npc_entry: Dict[str, str] = {
            "file": rel_file,
            "route_id": str(route_id),
            "town": str(town),
            "name": name,
            "kind": "npc",
            "control_mode": str(fake_ego_control_mode),
        }
        if model:
            npc_entry["model"] = str(model)
        if target_speed is not None and target_speed > 0.0:
            npc_entry["speed"] = round(float(target_speed), 2)
            npc_entry["target_speed"] = round(float(target_speed), 2)
        npc_entries.append(npc_entry)

        camera_id = int(ego_idx)
        while camera_id in used_camera_ids:
            camera_id += 1
        used_camera_ids.add(camera_id)
        fake_ego_views.append((camera_id, name))

    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    fake_ego_views.sort(key=lambda item: item[0])
    return clone_root, temp_dir, fake_ego_views, manifest_path


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


def align_ego_routes_in_directory(
    routes_dir: Path,
    town: str,
    carla_host: str,
    carla_port: int,
    sampling_resolution: float,
) -> None:
    """
    Align only ego route XMLs in routes_dir using the scenario_generator alignment module.
    """
    try:
        from scenario_generator.pipeline.step_07_route_alignment import align_route_file
        from scenario_generator.pipeline.step_07_route_alignment import main as align_mod
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment unavailable (import failed): {exc}")
        return

    try:
        carla, GlobalRoutePlanner, GlobalRoutePlannerDAO = align_mod._ensure_carla()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment unavailable (CARLA import failed): {exc}")
        return

    client = None
    try:
        print(f"[INFO] Aligning ego routes with CARLA at {carla_host}:{carla_port} (town={town})...")
        client = create_logged_client(
            carla,
            carla_host,
            carla_port,
            timeout_s=30.0,
            context="align_ego_routes",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        )
        world = client.get_world()
        current_map = world.get_map().name
        current_map_base = current_map.split("/")[-1] if "/" in current_map else current_map
        if current_map_base != town:
            world = client.load_world(town)
        carla_map = world.get_map()

        try:
            grp = GlobalRoutePlanner(carla_map, sampling_resolution)
        except TypeError:
            grp = GlobalRoutePlanner(GlobalRoutePlannerDAO(carla_map, sampling_resolution))
        if hasattr(grp, "setup"):
            grp.setup()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Ego route alignment failed to initialize CARLA/GRP: {exc}")
        return
    finally:
        release_client_id = log_client_release_begin(
            client,
            context="align_ego_routes",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            reason="align_ego_routes_scope_end",
        )
        # Release libcarla C++ object so its destructor closes the TCP socket
        # synchronously. carla_map/grp/world are kept alive by their own
        # references and do not hold the client socket open.
        client = None  # noqa: F841
        log_client_release_end(
            release_client_id,
            context="align_ego_routes",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            reason="align_ego_routes_scope_end",
        )

    ego_xmls: List[Path] = []
    for xml_path in routes_dir.glob("*.xml"):
        try:
            _, _, role = parse_route_metadata(xml_path.read_bytes())
        except Exception:
            continue
        if role == "ego":
            ego_xmls.append(xml_path)

    if not ego_xmls:
        print(f"[WARN] No ego route XMLs found for alignment in {routes_dir}.")
        return

    for xml_path in ego_xmls:
        try:
            orig, aligned = align_route_file(
                xml_path,
                carla_map,
                grp,
                backup=False,
            )
            if orig != aligned:
                print(f"[INFO] Aligned {xml_path.name}: {orig} -> {aligned} waypoints")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Failed to align {xml_path.name}: {exc}")


def set_effective_ports(
    args: argparse.Namespace,
    *,
    world_port: int,
    tm_port_offset: int,
) -> PortBundle:
    args.port = int(world_port)
    args.tm_port = int(world_port) + int(tm_port_offset)
    log_carla_event(
        "SET_EFFECTIVE_PORTS",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        world_port=args.port,
        tm_port=args.tm_port,
        tm_port_offset=int(tm_port_offset),
    )
    return PortBundle(port=int(args.port), tm_port=int(args.tm_port))


def _nonempty_env_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def sync_connection_event_env(
    env: Dict[str, str],
    *,
    carla_manager: "CarlaProcessManager | None",
) -> None:
    managed_logdir = None if carla_manager is None else carla_manager.rootcause_last_logdir
    inherited_logdir = _nonempty_env_text(env.get("CARLA_ROOTCAUSE_LOGDIR"))
    inherited_events_log = _nonempty_env_text(env.get("CARLA_CONNECTION_EVENTS_LOG"))

    if managed_logdir is not None:
        events_log = set_connection_events_logdir(
            managed_logdir,
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        )
        env["CARLA_ROOTCAUSE_LOGDIR"] = str(managed_logdir)
        env["CARLA_CONNECTION_EVENTS_LOG"] = str(events_log)
        log_carla_event(
            "SYNC_CONNECTION_EVENT_ENV_SET",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            source="managed",
            rootcause_logdir=str(managed_logdir),
            events_log=str(events_log),
        )
        return

    if inherited_logdir or inherited_events_log:
        if inherited_logdir and not inherited_events_log:
            inherited_events_log = str(
                Path(inherited_logdir).expanduser().resolve() / "carla_connection_events.log"
            )
        elif inherited_events_log and not inherited_logdir:
            inherited_logdir = str(Path(inherited_events_log).expanduser().resolve().parent)

        if inherited_logdir:
            env["CARLA_ROOTCAUSE_LOGDIR"] = inherited_logdir
        else:
            env.pop("CARLA_ROOTCAUSE_LOGDIR", None)
        if inherited_events_log:
            env["CARLA_CONNECTION_EVENTS_LOG"] = inherited_events_log
        else:
            env.pop("CARLA_CONNECTION_EVENTS_LOG", None)
        log_carla_event(
            "SYNC_CONNECTION_EVENT_ENV_PRESERVED",
            process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
            source="inherited",
            rootcause_logdir=inherited_logdir,
            events_log=inherited_events_log,
        )
        return

    env.pop("CARLA_ROOTCAUSE_LOGDIR", None)
    env.pop("CARLA_CONNECTION_EVENTS_LOG", None)
    log_carla_event(
        "SYNC_CONNECTION_EVENT_ENV_CLEARED",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        source="none",
    )


def sync_child_evaluator_context_env(
    env: Dict[str, str],
    *,
    planner_name: str,
    world_port: int,
    tm_port: int,
    route_id: str = "",
    gpu_id: str = "",
    routes_dir: Path | str | None = None,
    result_root: Path | str | None = None,
    save_path: Path | str | None = None,
    checkpoint_endpoint: Path | str | None = None,
) -> None:
    set_planner_context(
        planner_name,
        world_port=int(world_port),
        tm_port=int(tm_port),
        route_id=str(route_id),
        gpu_id=str(gpu_id),
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
    )
    env["CARLA_PLANNER_NAME"] = str(planner_name)
    env["CARLA_WORLD_PORT"] = str(int(world_port))
    env["CARLA_STREAM_PORT"] = str(int(world_port) + 1)
    env["CARLA_TM_PORT"] = str(int(tm_port))
    if routes_dir is not None:
        env["ROUTES"] = str(routes_dir)
        env["ROUTES_DIR"] = str(routes_dir)
    if result_root is not None:
        env["RESULT_ROOT"] = str(result_root)
    if save_path is not None:
        env["SAVE_PATH"] = str(save_path)
    if checkpoint_endpoint is not None:
        env["CHECKPOINT_ENDPOINT"] = str(checkpoint_endpoint)
    log_carla_event(
        "SYNC_CHILD_EVAL_CONTEXT_ENV_SET",
        process_name=RUN_CUSTOM_EVAL_PROCESS_NAME,
        planner_name=str(planner_name),
        route_id=str(route_id),
        gpu_id=str(gpu_id),
        world_port=int(world_port),
        stream_port=int(world_port) + 1,
        tm_port=int(tm_port),
        routes_dir=None if routes_dir is None else str(routes_dir),
        result_root=None if result_root is None else str(result_root),
        save_path=None if save_path is None else str(save_path),
        checkpoint_endpoint=None if checkpoint_endpoint is None else str(checkpoint_endpoint),
    )


def allocate_planner_slots(
    *,
    host: str,
    preferred_port: int,
    tm_port_offset: int,
    slot_count: int,
    tries: int,
    step: int,
) -> List[PlannerSlot]:
    slots: List[PlannerSlot] = []
    requested_slots = max(1, slot_count)
    attempts = max(int(tries), requested_slots * 2)
    bundles = find_available_port_bundles(
        host,
        preferred_port,
        tries=attempts,
        step=step,
        bundle_count=requested_slots,
        tm_port_offset=tm_port_offset,
    )
    for slot_index, bundle in enumerate(bundles):
        slots.append(
            PlannerSlot(
                slot_index=slot_index,
                gpu=None,
                ports=bundle,
            )
        )
    return slots


def main() -> None:
    global _active_carla_manager, _active_crosswalk_worker, _active_colmdriver_vllm_manager
    global _GPU_QUERY_CACHE_TTL_S

    # ---- FD tracker: activate via env var FD_TRACK=1 ----
    try:
        import fd_tracker as _fdt
        _fdt.maybe_install_from_env()
    except Exception:
        _fdt = None  # type: ignore[assignment]

    run_start_monotonic = time.monotonic()
    args = parse_args()
    _GPU_QUERY_CACHE_TTL_S = max(0.0, float(args.gpu_query_cache_ttl))
    with _GPU_QUERY_CACHE_LOCK:
        _GPU_QUERY_CACHE.clear()
    os.environ["RUN_CUSTOM_EVAL_RECOVERY_MODE"] = str(args.recovery_mode or "off")
    os.environ["CARLA_RECOVERY_STRICT_RECORD_ACCOUNTING"] = (
        "1" if str(args.recovery_mode or "").strip().lower() == "publication_safe" else "0"
    )
    install_process_lifecycle_logging(
        RUN_CUSTOM_EVAL_PROCESS_NAME,
        env_keys=(
            "CARLA_ROOTCAUSE_LOGDIR",
            "CARLA_CONNECTION_EVENTS_LOG",
            "CUDA_VISIBLE_DEVICES",
            DASHBOARD_STATUS_ENV,
        ),
    )
    # FD tracker baseline: record open FDs at the very start of main()
    try:
        _fdt.checkpoint("startup_baseline")  # type: ignore[union-attr]
    except Exception:
        pass
    base_argv = list(getattr(args, "_normalized_argv", sys.argv[1:]))
    if any(
        token == "--capture-overhead-per-tick" or token.startswith("--overhead-")
        for token in base_argv
    ):
        print(
            "[WARN] Overhead per-tick capture flags are deprecated and ignored. "
            "This run will not produce overhead_tick images."
        )
    planner_requests = parse_planner_requests(args.planner)
    requested_gpus = parse_gpu_requests(args.gpus)
    if requested_gpus:
        requested_gpus = list(dict.fromkeys(requested_gpus))
    planner_gpu_requests = list(requested_gpus)
    tm_port_offset = compute_tm_port_offset(int(args.port), args.tm_port)

    if args.npc_only_fake_ego:
        if any(req.name != "log-replay" for req in planner_requests):
            print(
                "[WARN] --npc-only-fake-ego requires log-replay planner semantics. "
                "Ignoring requested planners and forcing a single log-replay run."
            )
        retained_env = planner_requests[0].conda_env if planner_requests else None
        planner_requests = [
            PlannerRequest(
                name="log-replay",
                conda_env=retained_env,
                run_id=sanitize_run_id("log-replay"),
            )
        ]

    primary_planner = planner_requests[0]
    args.planner = primary_planner.name
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
    # Allow overriding CARLA install via environment; fallback to bundled carla912.
    carla_root = Path(os.environ.get("CARLA_ROOT", repo_root / "carla912")).expanduser().resolve()
    if not carla_root.exists():
        raise FileNotFoundError(
            f"CARLA_ROOT '{carla_root}' does not exist. "
            "Set CARLA_ROOT to a valid CARLA installation (with PythonAPI/carla/dist)."
        )

    forensics_log_path: Path | None = None
    if args.carla_forensics_log_file:
        candidate = Path(args.carla_forensics_log_file).expanduser()
        forensics_log_path = candidate if candidate.is_absolute() else (repo_root / candidate).resolve()

    forensics_server_log_path: Path | None = None
    if args.carla_forensics_server_log_file:
        candidate = Path(args.carla_forensics_server_log_file).expanduser()
        forensics_server_log_path = (
            candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
        )

    rootcause_log_root = repo_root
    if args.carla_rootcause_log_root:
        candidate = Path(args.carla_rootcause_log_root).expanduser()
        rootcause_log_root = (
            candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
        )
    carla_rootcause_config: CarlaRootCauseConfig | None = None
    if args.carla_rootcause:
        script_candidate = Path(args.carla_rootcause_script).expanduser()
        rootcause_script_path = (
            script_candidate
            if script_candidate.is_absolute()
            else (repo_root / script_candidate).resolve()
        )
        if not rootcause_script_path.exists():
            raise FileNotFoundError(
                "CARLA root-cause script was not found at "
                f"'{rootcause_script_path}'. Set --carla-rootcause-script accordingly."
            )
        if not os.access(rootcause_script_path, os.X_OK):
            raise PermissionError(
                f"CARLA root-cause script is not executable: {rootcause_script_path}"
            )
        rootcause_log_root.mkdir(parents=True, exist_ok=True)
        carla_rootcause_config = CarlaRootCauseConfig(
            script_path=rootcause_script_path,
            mode=str(args.carla_rootcause_mode),
            diag_interval=int(args.carla_rootcause_diag_interval),
            startup_timeout_s=int(args.carla_rootcause_startup_timeout),
            core_wait_seconds=int(args.carla_rootcause_core_wait_seconds),
            log_root=rootcause_log_root,
        )
        print(
            "[INFO] CARLA root-cause integration enabled: "
            f"script={rootcause_script_path} mode={carla_rootcause_config.mode} "
            f"log_root={rootcause_log_root}"
        )

    def _forensics_note(message: str) -> None:
        print(message)
        if args.carla_forensics:
            _append_text_log_line(forensics_log_path, message)

    if args.carla_forensics:
        args.carla_stream_debug_summary_every = max(1, int(args.carla_stream_debug_summary_every))
        _ensure_carla_arg(args.carla_arg, "-log")
        _ensure_carla_arg(args.carla_arg, "-forcelogflush")
        core_ok, core_detail = _enable_core_dumps_best_effort()
        _forensics_note("[INFO] CARLA forensics mode enabled.")
        _forensics_note(
            "[INFO] Forensics env defaults: "
            f"CARLA_STREAM_DEBUG=1, "
            f"CARLA_STREAM_DEBUG_SUMMARY_EVERY={args.carla_stream_debug_summary_every}, "
            f"MALLOC_CHECK_={CARLA_FORENSICS_MALLOC_CHECK_DEFAULT}, "
            f"MALLOC_PERTURB_={CARLA_FORENSICS_MALLOC_PERTURB_DEFAULT}"
        )
        if forensics_log_path is not None:
            _forensics_note(f"[INFO] Forensics launcher log file: {forensics_log_path}")
        if args.start_carla:
            _forensics_note("[INFO] Forensics CARLA args enforced: " + " ".join(args.carla_arg))
            if args.carla_rootcause:
                _forensics_note(
                    "[INFO] --carla-rootcause is enabled; CARLA server stdout/stderr is captured "
                    "by the root-cause logdir."
                )
            elif forensics_server_log_path is not None:
                _forensics_note(
                    f"[INFO] CARLA server stdout/stderr log file: {forensics_server_log_path}"
                )
        else:
            _forensics_note(
                "[WARN] --carla-forensics is active, but --start-carla is not set. "
                "External CARLA must be started with matching env vars for server-side stream diagnostics."
            )
            if forensics_server_log_path is not None:
                _forensics_note(
                    "[WARN] --carla-forensics-server-log-file is set but unused because --start-carla is disabled."
                )
        if core_ok:
            _forensics_note(f"[INFO] Core dump limit update: {core_detail}")
        else:
            _forensics_note(f"[WARN] Core dump limit update failed: {core_detail}")

    if int(args.repro_rerun_total) > 0 and int(args.repro_rerun_index) > 0:
        print(
            "[INFO] Reproduction rerun context: "
            f"{int(args.repro_rerun_index)}/{int(args.repro_rerun_total)} "
            f"label={format_repro_rerun_label(int(args.repro_rerun_index))}"
        )

    if int(args.repro_reruns) > 1 and int(args.repro_rerun_index) == 0:
        raise SystemExit(
            dispatch_repro_reruns(
                args=args,
                base_argv=base_argv,
                repo_root=repo_root,
            )
        )

    runtime_env = _build_runtime_env(args, os.environ.copy())
    dashboard_status_path = get_dashboard_status_path(runtime_env)
    if dashboard_status_path is not None:
        update_dashboard_status(
            dashboard_status_path,
            phase="initializing",
            planner=str(args.planner or ""),
            scenario_total=1,
            scenario_index=0,
            completed_scenarios=0,
            scenario_name="",
            scenario_results_subdir=str(args.results_subdir or ""),
            current_attempt=0,
            attempt_limit=retry_limit_for_status(args.scenario_retry_limit),
        )
    set_effective_ports(
        args,
        world_port=int(args.port),
        tm_port_offset=tm_port_offset,
    )

    multi_planner_mode = len(planner_requests) > 1 or any(
        req.conda_env for req in planner_requests
    )
    colmdriver_vllm_manager: ColmDriverVLLMManager | None = None
    colmdriver_vllm_needed = planner_requests_need_colmdriver_vllm(planner_requests)
    defer_colmdriver_vllm_phase = False
    colmdriver_vllm_started = True
    colmdriver_planner_run_ids = {
        req.run_id for req in planner_requests if req.name in COLMDRIVER_VLLM_PLANNERS
    }
    if args.start_colmdriver_vllm and not colmdriver_vllm_needed:
        print(
            "[WARN] --start-colmdriver-vllm was provided, but no CoLMDriver planner "
            "was requested. Ignoring it."
        )
    if args.start_colmdriver_vllm and colmdriver_vllm_needed:
        has_non_colmdriver_planners = any(
            req.name not in COLMDRIVER_VLLM_PLANNERS for req in planner_requests
        )
        defer_colmdriver_vllm_phase = bool(
            args.defer_colmdriver_vllm_start
            and multi_planner_mode
            and has_non_colmdriver_planners
        )
        reservation = select_colmdriver_vllm_gpus(requested_gpus)
        planner_gpu_requests = list(reservation.scheduler_gpus)
        print(
            "[INFO] Selected CoLMDriver vLLM GPUs: "
            f"scheduler_gpus={', '.join(planner_gpu_requests)} "
            f"VLM={reservation.vlm_gpu} LLM={reservation.llm_gpu} "
            f"(selection={reservation.selection_source})"
        )
        colmdriver_vllm_manager = ColmDriverVLLMManager(
            repo_root=repo_root,
            base_env=runtime_env.copy(),
            conda_env=str(args.colmdriver_vllm_env),
            reservation=reservation,
            llm_port=int(args.llm_port),
            vlm_port=int(args.vlm_port),
            startup_timeout_s=float(args.colmdriver_vllm_startup_timeout),
            log_dir=(
                repo_root
                / "results"
                / "results_driving_custom"
                / infer_base_results_tag(args)
                / "_service_logs"
            ),
        )
        if args.dry_run:
            if defer_colmdriver_vllm_phase:
                print(
                    "[INFO] Dry run: would defer shared CoLMDriver vLLM startup until "
                    "the deferred CoLMDriver planner phase begins "
                    f"(VLM gpu={reservation.vlm_gpu} port={args.vlm_port}, "
                    f"LLM gpu={reservation.llm_gpu} port={args.llm_port})."
                )
            else:
                print(
                    "[INFO] Dry run: would start shared CoLMDriver vLLM services on "
                    f"VLM gpu={reservation.vlm_gpu} port={args.vlm_port} and "
                    f"LLM gpu={reservation.llm_gpu} port={args.llm_port}."
                )
            colmdriver_vllm_started = True
        elif defer_colmdriver_vllm_phase:
            print(
                "[INFO] Deferring shared CoLMDriver vLLM startup until non-CoLMDriver "
                "planner queues are drained."
            )
            colmdriver_vllm_started = False
        else:
            colmdriver_vllm_manager.start()
            _active_colmdriver_vllm_manager = colmdriver_vllm_manager
            _install_signal_handlers()
            colmdriver_vllm_started = True
    if not multi_planner_mode and planner_gpu_requests:
        if len(planner_gpu_requests) == 1:
            chosen_gpu = planner_gpu_requests[0]
        else:
            # Proactively pick the GPU with the most free VRAM to avoid OOM at startup.
            chosen_gpu = _pick_best_gpu(planner_gpu_requests) or planner_gpu_requests[0]
            if chosen_gpu != planner_gpu_requests[0]:
                print(
                    f"[INFO] Proactive GPU selection: chose GPU {chosen_gpu} over "
                    f"{planner_gpu_requests[0]} based on available VRAM."
                )
            print(
                "[WARN] Multiple GPUs were provided for a single direct run. "
                f"Selected GPU {chosen_gpu} based on VRAM; ignoring the rest: "
                f"{', '.join(g for g in planner_gpu_requests if g != chosen_gpu)}"
            )
        runtime_env["CUDA_VISIBLE_DEVICES"] = chosen_gpu
        print(f"[INFO] Using GPU {chosen_gpu} for this evaluation run.")

    if multi_planner_mode:
        # Decouple slot count from planner count.  We allocate enough CARLA
        # port bundles to support running up to ``per_gpu_cap * len(gpus)``
        # concurrent jobs, so the scheduler can pack multiple planners onto
        # the same GPU (and multiple CARLAs onto the same GPU) without ever
        # hitting an artificial slot ceiling.  Each planner still lives in
        # one slot at a time, so the effective ceiling is ``min(launchable,
        # per_gpu_cap * num_gpus)``.
        _per_gpu_cap_hint = max(1, int(args.auto_workers_max_per_gpu))
        _num_gpus_hint = max(1, len(requested_gpus) if requested_gpus else 1)
        slot_count = max(
            len(planner_requests),
            _per_gpu_cap_hint * _num_gpus_hint,
        )
        slot_count = max(1, slot_count)
        slot_templates = allocate_planner_slots(
            host=str(args.host),
            preferred_port=int(args.port),
            tm_port_offset=tm_port_offset,
            slot_count=slot_count,
            tries=int(args.carla_port_tries),
            step=int(args.carla_port_step),
        )
        slot_pool: deque[PlannerSlot] = deque(slot_templates)
        external_carla_mode = bool(not args.start_carla and not args.dry_run)
        if len(planner_requests) > 1 and not args.start_carla:
            print(
                "[WARN] Multi-planner mode without --start-carla assumes external CARLA servers "
                "exist on the assigned port pairs."
            )
        if external_carla_mode:
            reachable_slots: List[PlannerSlot] = []
            unreachable_world_ports: List[int] = []
            for slot in slot_templates:
                world_port = int(slot.ports.port)
                if is_port_open(str(args.host), world_port, timeout=1.0):
                    reachable_slots.append(slot)
                else:
                    unreachable_world_ports.append(world_port)
            if not reachable_slots:
                checked_ports = ", ".join(
                    str(int(slot.ports.port)) for slot in slot_templates
                )
                raise RuntimeError(
                    "External CARLA preflight failed in multi-planner mode: "
                    f"no world ports are reachable on host={args.host}. "
                    f"Checked ports: [{checked_ports}]. "
                    "Start external CARLA on these ports or rerun with --start-carla."
                )
            if len(reachable_slots) < len(slot_templates):
                slot_pool = deque(reachable_slots)
                dropped = ", ".join(str(port) for port in unreachable_world_ports)
                print(
                    "[WARN] External CARLA preflight filtered unavailable world ports: "
                    f"{dropped}. Scheduler will use only reachable slots."
                )

        planner_routes_dir: Path | None = None
        planner_scenario_name = args.scenario_name
        if args.zip is not None:
            prepared_summary = prepare_routes_from_zip(
                zip_path=args.zip,
                scenario_name=args.scenario_name,
                output_root=args.output_root,
                overwrite=args.overwrite,
            )
            planner_routes_dir = prepared_summary["output_dir"]
            planner_scenario_name = prepared_summary["scenario_name"]
            print(f"[INFO] Prepared shared routes for planner batch: {planner_routes_dir}")

        planner_base_results_tag = (
            str(args.results_tag)
            if args.results_tag
            else (
                str(planner_scenario_name)
                if planner_scenario_name
                else (
                    planner_routes_dir.name
                    if planner_routes_dir is not None
                    else infer_base_results_tag(args)
                )
            )
        )

        active_runners: list[MonitoredSubprocessRunner] = []
        active_runners_lock = threading.Lock()
        planner_state_lock = threading.Lock()
        deferred_colmdriver_run_ids: set[str] = set()
        if defer_colmdriver_vllm_phase:
            non_colmdriver_requests = [
                req for req in planner_requests if req.run_id not in colmdriver_planner_run_ids
            ]
            deferred_colmdriver_requests = [
                req for req in planner_requests if req.run_id in colmdriver_planner_run_ids
            ]
            planner_pending: deque[PlannerRequest] = deque(non_colmdriver_requests)
            planner_pending_deferred: deque[PlannerRequest] = deque(deferred_colmdriver_requests)
            deferred_colmdriver_run_ids = {
                req.run_id for req in deferred_colmdriver_requests
            }
            print(
                "[INFO] Deferred CoLMDriver scheduler phase enabled: "
                f"non_colmdriver={len(non_colmdriver_requests)} "
                f"colmdriver_deferred={len(deferred_colmdriver_requests)}."
            )
        else:
            planner_pending = deque(planner_requests)
            planner_pending_deferred = deque()
        active_non_colmdriver_threads = 0

        def has_pending_planner_requests() -> bool:
            return bool(planner_pending or planner_pending_deferred)

        def has_launchable_planner_requests() -> bool:
            if planner_pending:
                return True
            return bool(planner_pending_deferred) and active_non_colmdriver_threads <= 0

        def pop_next_pending_planner() -> PlannerRequest | None:
            if planner_pending:
                return planner_pending.popleft()
            if planner_pending_deferred and active_non_colmdriver_threads <= 0:
                return planner_pending_deferred.popleft()
            return None

        def requeue_pending_planner(request: PlannerRequest) -> None:
            if request.run_id in deferred_colmdriver_run_ids:
                planner_pending_deferred.append(request)
            else:
                planner_pending.append(request)

        planner_relaunch_counts: Dict[str, int] = {
            planner_request.run_id: 0 for planner_request in planner_requests
        }
        completion_queue: "queue.Queue[PlannerThreadResult]" = queue.Queue()
        active_threads: Dict[int, threading.Thread] = {}
        planner_child_retry_budget = planner_child_retry_limit(args.scenario_retry_limit)
        planner_timeout_routes_dir = (
            planner_routes_dir
            if planner_routes_dir is not None
            else (
                args.routes_dir.expanduser().resolve()
                if args.routes_dir is not None
                else None
            )
        )
        planner_results_root = (
            repo_root / "results" / "results_driving_custom" / planner_base_results_tag
        )
        planner_retry_log_path = planner_results_root / "_retry_events.jsonl"
        planner_retry_limit = normalize_retry_limit(args.planner_retry_limit)
        planner_perf_report_path: Path | None = None
        if args.perf_report_file:
            perf_candidate = Path(str(args.perf_report_file))
            planner_perf_report_path = (
                perf_candidate
                if perf_candidate.is_absolute()
                else planner_results_root / perf_candidate
            )
        planner_logs_dir = planner_results_root / "_planner_logs"
        dashboard_enabled = bool(args.planner_dashboard and not args.dry_run and tqdm is not None)
        if args.planner_dashboard and not args.dry_run and tqdm is None:
            print("[WARN] tqdm is unavailable; disabling the multi-planner dashboard.")
        dashboard_ego_count = 0
        dashboard_scenario_entries: List[Tuple[str, str]] = []
        if planner_timeout_routes_dir is not None:
            scenario_subfolders = detect_scenario_subfolders(
                planner_timeout_routes_dir,
                args.routes_leaf,
            )
            if scenario_subfolders:
                dashboard_scenario_entries = [
                    (sub_name, combine_results_subdir(args.results_subdir, sub_name))
                    for _, sub_name in scenario_subfolders
                ]
                for subfolder, _ in scenario_subfolders:
                    dashboard_ego_count = max(
                        dashboard_ego_count,
                        detect_ego_routes(subfolder, replay_mode=False),
                        detect_ego_routes(subfolder, replay_mode=True),
                    )
            else:
                dashboard_ego_count = detect_ego_routes(planner_timeout_routes_dir, replay_mode=False)
                if dashboard_ego_count == 0:
                    dashboard_ego_count = detect_ego_routes(planner_timeout_routes_dir, replay_mode=True)
        planner_validation_ego_count = dashboard_ego_count
        if planner_validation_ego_count <= 0:
            planner_validation_ego_count = 1
            print(
                "[WARN] Unable to auto-detect ego count for multi-planner validation; "
                "defaulting to 1."
            )

        # Compute per-planner scenario entries (sharded planners get their subset).
        _per_planner_scenario_entries: Dict[str, List[Tuple[str, str]]] = {}
        for planner_request in planner_requests:
            _per_planner_scenario_entries[planner_request.run_id] = (
                filter_scenario_entries_for_shard(
                    dashboard_scenario_entries,
                    planner_request.scenario_shard,
                )
            )

        planner_dashboard_states: Dict[str, PlannerDashboardState] = {}
        for planner_request in planner_requests:
            _entries = _per_planner_scenario_entries[planner_request.run_id]
            planner_dashboard_states[planner_request.run_id] = PlannerDashboardState(
                planner_request=planner_request,
                results_dir=planner_results_root / planner_request.run_id,
                log_path=planner_logs_dir / f"{planner_request.run_id}.log",
                status_path=planner_results_root / planner_request.run_id / "_dashboard_status.json",
                default_results_subdir=str(args.results_subdir or ""),
                scenario_total=max(1, len(_entries)),
                current_scenario=_entries[0][0] if _entries else "",
                current_scenario_results_subdir=(
                    _entries[0][1]
                    if _entries
                    else str(args.results_subdir or "")
                ),
            )
        for dashboard_state in planner_dashboard_states.values():
            if not args.dry_run and dashboard_state.status_path.exists():
                try:
                    dashboard_state.status_path.unlink()
                except FileNotFoundError:
                    pass
        planner_dashboard = MultiPlannerDashboard(
            planner_states=planner_dashboard_states,
            planner_order=[planner_request.run_id for planner_request in planner_requests],
            ego_count=dashboard_ego_count,
            host=str(args.host),
            gpu_ids=planner_gpu_requests,
            scenario_entries=dashboard_scenario_entries,
            per_planner_scenario_entries=_per_planner_scenario_entries,
            state_lock=planner_state_lock,
            refresh_seconds=float(args.planner_dashboard_refresh_seconds),
            status_snapshot_interval_s=float(args.planner_status_snapshot_interval),
            stall_output_timeout_s=float(args.scenario_stall_output_timeout),
            enable=dashboard_enabled,
        )

        managed_carla_single_worker_mode = (
            args.start_carla and not args.allow_multiple_carlas_per_gpu
        )
        scheduler_gpu_ids: List[str | None] = planner_gpu_requests[:] if planner_gpu_requests else [None]
        default_gpu_max_concurrency = max(1, int(args.auto_workers_max_per_gpu))
        # When the VRAM-aware scheduler is active (default), always start with
        # the full cap — the bin-packer's VRAM checks are the real gate.
        # Only fall back to 1 if VRAM-aware is disabled AND managed_carla mode
        # would restrict concurrency.
        _init_max_concurrency = default_gpu_max_concurrency
        if not args.vram_aware_scheduler and managed_carla_single_worker_mode:
            _init_max_concurrency = 1
        gpu_states: Dict[str | None, GPUPlannerState] = {
            gpu: GPUPlannerState(
                gpu=gpu,
                launch_capacity=1,
                max_concurrency=_init_max_concurrency,
            )
            for gpu in scheduler_gpu_ids
        }

        if planner_gpu_requests:
            # When the VRAM-aware scheduler is active, skip expensive blocking
            # settle probes.  A single fast nvidia-smi call primes the GPU
            # stats; the background GPUMemoryMonitor refines them continuously.
            if args.vram_aware_scheduler and not args.dry_run:
                _fast_stats = query_gpu_memory_stats(planner_gpu_requests)
                for gpu in planner_gpu_requests:
                    gpu_state = gpu_states[gpu]
                    stats = _fast_stats.get(gpu)
                    if stats is not None:
                        gpu_state.last_stats = stats
                        gpu_state.telemetry_available = True
                        gpu_state.last_probe_source = "fast_prime"
                    else:
                        gpu_state.last_probe_source = "fast_prime_unavailable"
                    if stats is not None and not dashboard_enabled:
                        print(
                            f"[INFO] GPU {gpu}: free={stats.free_mib} MiB "
                            f"used={stats.used_mib} MiB "
                            f"total={stats.total_mib} MiB "
                            f"max_concurrency={gpu_state.max_concurrency}"
                        )
            elif managed_carla_single_worker_mode:
                if not dashboard_enabled:
                    print(
                        "[INFO] Managed CARLA multi-planner mode caps concurrency at one planner "
                        "per GPU by default. Pass --allow-multiple-carlas-per-gpu to override."
                    )
                    print(
                        "[INFO] Probing GPU free VRAM to select which GPUs can host one "
                        "managed CARLA each."
                    )
                for gpu in planner_gpu_requests:
                    gpu_state = gpu_states[gpu]
                    state = refresh_gpu_launch_capacity(
                        gpu_state,
                        min_free_mib=int(args.auto_workers_min_free_mib),
                        max_concurrency=gpu_state.max_concurrency,
                        extra_min_free_mib=gpu_state.oom_penalty_mib,
                        settle_timeout_s=float(args.auto_workers_settle_timeout),
                        poll_seconds=float(args.auto_workers_poll_seconds),
                        stable_samples=int(args.auto_workers_stable_samples),
                        stable_tolerance_mib=int(args.auto_workers_stable_tolerance_mib),
                    )
                    state.launch_capacity = min(state.launch_capacity, 1)
                    state.last_probe_source = "managed_carla_single_worker"
                    if state.last_stats is not None and not dashboard_enabled:
                        print(
                            "[INFO] GPU "
                            f"{gpu}: free={state.last_stats.free_mib} MiB "
                            f"used={state.last_stats.used_mib} MiB "
                            f"total={state.last_stats.total_mib} MiB "
                            f"managed_carla_capacity={state.launch_capacity}"
                        )
                    elif not dashboard_enabled:
                        print(
                            "[WARN] GPU "
                            f"{gpu}: live memory probe unavailable; allowing at most one "
                            "managed CARLA on it."
                        )
                if not any(
                    state.launch_capacity > state.active_count
                    for state in gpu_states.values()
                ):
                    forced_state = choose_force_launch_gpu(gpu_states)
                    forced_state.launch_capacity = min(
                        1,
                        max(forced_state.launch_capacity, forced_state.active_count + 1),
                    )
                    if not dashboard_enabled:
                        print(
                            "[WARN] No GPU met the current free-VRAM threshold for managed CARLA. "
                            f"Forcing one launch on GPU {forced_state.gpu} to keep the batch moving."
                        )
            elif args.auto_workers_per_gpu and not args.dry_run:
                if not dashboard_enabled:
                    print(
                        "[INFO] Probing GPU free VRAM for adaptive planner concurrency "
                        f"(min_free={int(args.auto_workers_min_free_mib)} MiB, "
                        f"settle_timeout={float(args.auto_workers_settle_timeout):.1f}s)."
                    )
                for gpu in planner_gpu_requests:
                    gpu_state = gpu_states[gpu]
                    state = refresh_gpu_launch_capacity(
                        gpu_state,
                        min_free_mib=int(args.auto_workers_min_free_mib),
                        max_concurrency=gpu_state.max_concurrency,
                        extra_min_free_mib=gpu_state.oom_penalty_mib,
                        settle_timeout_s=float(args.auto_workers_settle_timeout),
                        poll_seconds=float(args.auto_workers_poll_seconds),
                        stable_samples=int(args.auto_workers_stable_samples),
                        stable_tolerance_mib=int(args.auto_workers_stable_tolerance_mib),
                    )
                    if state.last_stats is not None and not dashboard_enabled:
                        print(
                            "[INFO] GPU "
                            f"{gpu}: free={state.last_stats.free_mib} MiB "
                            f"used={state.last_stats.used_mib} MiB "
                            f"total={state.last_stats.total_mib} MiB "
                            f"launch_capacity={state.launch_capacity} "
                            f"max_concurrency={state.max_concurrency} "
                            f"oom_penalty_mib={state.oom_penalty_mib}"
                        )
                    elif not dashboard_enabled:
                        print(
                            "[WARN] GPU "
                            f"{gpu}: live memory probe unavailable; starting conservatively "
                            "with one planner at a time."
                        )
                if not any(
                    state.launch_capacity > state.active_count
                    for state in gpu_states.values()
                ):
                    forced_state = choose_force_launch_gpu(gpu_states)
                    forced_state.launch_capacity = max(
                        forced_state.launch_capacity,
                        forced_state.active_count + 1,
                    )
                    forced_state.launch_capacity = min(
                        forced_state.launch_capacity,
                        max(1, int(forced_state.max_concurrency)),
                    )
                    if not dashboard_enabled:
                        print(
                            "[WARN] No GPU met the current free-VRAM threshold. "
                            f"Forcing one planner launch on GPU {forced_state.gpu} to keep the batch moving."
                        )
            else:
                if args.auto_workers_per_gpu and args.dry_run:
                    print(
                        "[INFO] Dry run: skipping live GPU VRAM probes; planner concurrency "
                        "defaults to one worker per listed GPU."
                    )
                for gpu in planner_gpu_requests:
                    gpu_states[gpu].launch_capacity = 1
                    gpu_states[gpu].max_concurrency = 1
                    gpu_states[gpu].last_probe_source = (
                        "dry_run" if args.dry_run else "disabled"
                    )
        else:
            gpu_states[None].last_probe_source = "no_gpu"

        # --- Sort scheduler GPUs by free VRAM (most free first) so that the
        #     emptiest GPUs are used before partially-occupied ones, reducing
        #     the chance of CUDA OOM when other processes share the same GPUs.
        if planner_gpu_requests and len(scheduler_gpu_ids) > 1:
            def _free_mib_for_sort(gpu_id: str | None) -> int:
                st = gpu_states.get(gpu_id)
                if st is not None and st.last_stats is not None:
                    return st.last_stats.free_mib
                return 0  # unknown → sort last

            sorted_ids = sorted(
                scheduler_gpu_ids,
                key=_free_mib_for_sort,
                reverse=True,
            )
            if sorted_ids != scheduler_gpu_ids:
                if not dashboard_enabled:
                    order_str = ", ".join(
                        f"{g}({_free_mib_for_sort(g)}MiB)" for g in sorted_ids
                    )
                    print(
                        f"[INFO] Re-ordered scheduler GPUs by free VRAM: {order_str}"
                    )
                scheduler_gpu_ids = sorted_ids

        # ---- VRAM-aware scheduler infrastructure --------------------------------
        vram_estimator_cache_path = Path(str(args.planner_vram_estimate_cache)).expanduser()
        vram_estimator = PlannerVramEstimator(
            cache_path=vram_estimator_cache_path,
            seed=PLANNER_VRAM_ESTIMATE_MIB,
        )
        # Apply any command-line overrides.
        for raw in args.planner_vram_estimate_mib or []:
            raw_str = str(raw).strip()
            if not raw_str or "=" not in raw_str:
                continue
            name_part, _, value_part = raw_str.partition("=")
            name_part = name_part.strip()
            try:
                value_int = int(float(value_part.strip()))
            except (TypeError, ValueError):
                continue
            if name_part and value_int > 0:
                # Re-seed the in-memory estimate; the bump preserves the
                # observation count so the cache stays meaningful.
                vram_estimator.observe_peak(name_part, value_int)

        # Background GPU memory monitor — replaces inline blocking nvidia-smi
        # settle waits in the scheduler hot path.
        gpu_monitor: GPUMemoryMonitor | None = None
        if planner_gpu_requests and not args.dry_run:
            gpu_monitor = GPUMemoryMonitor(
                planner_gpu_requests,
                poll_seconds=float(args.gpu_monitor_poll_seconds),
            )
            gpu_monitor.start()
            # Pre-seed the monitor so the first scheduler tick has VRAM data
            # immediately instead of waiting up to poll_seconds for the
            # background thread's first cycle.
            _preseed_stats = query_gpu_memory_stats(planner_gpu_requests)
            if _preseed_stats:
                with gpu_monitor._lock:
                    gpu_monitor._latest = dict(_preseed_stats)
                    gpu_monitor._last_update_monotonic = time.monotonic()
                    for gpu_id, stats in _preseed_stats.items():
                        gpu_monitor._peaks[gpu_id] = (stats.used_mib, stats.used_mib)

        # Scheduler tick log (optional JSONL instrumentation).
        _tick_log_path: Path | None = None
        if args.scheduler_tick_log:
            _raw_tick_path = Path(str(args.scheduler_tick_log)).expanduser()
            _tick_log_path = (
                _raw_tick_path
                if _raw_tick_path.is_absolute()
                else (planner_results_root / _raw_tick_path)
            )
        scheduler_tick_logger = SchedulerTickLogger(_tick_log_path)

        # Human-readable orchestration log — always written, even with dashboard.
        _sched_log_path: Path = planner_logs_dir / "_scheduler.log"

        def _sched_log(msg: str) -> None:
            """Append a timestamped line to the orchestration log."""
            ts = time.strftime("%H:%M:%S")
            _append_text_log_line(_sched_log_path, f"[{ts}] {msg}")

        print(f"[INFO] Scheduler log: {_sched_log_path}")

        vram_aware_scheduler = bool(args.vram_aware_scheduler) and not args.dry_run
        if vram_aware_scheduler and not dashboard_enabled:
            _max_mp_label = " [--max-multiprocess]" if getattr(args, "max_multiprocess", False) else ""
            print(
                f"[INFO] VRAM-aware scheduler enabled{_max_mp_label}: "
                f"carla_vram={int(args.carla_vram_estimate_mib)} MiB "
                f"(per_launch={'yes' if args.start_carla else 'external'}), "
                f"safety_buffer={int(args.vram_safety_buffer_mib)} MiB, "
                f"min_free={int(args.auto_workers_min_free_mib)} MiB, "
                f"per_gpu_max_concurrency={int(args.auto_workers_max_per_gpu)}, "
                f"slot_pool={len(slot_pool)}."
            )
            seed_snap = vram_estimator.snapshot()
            if seed_snap:
                print(
                    "[INFO] Planner VRAM estimates (cached): "
                    + ", ".join(f"{k}={v}MiB" for k, v in sorted(seed_snap.items()))
                )
        _sched_log(
            f"SCHEDULER START workers={len(planner_requests)} "
            f"gpus={[str(g) for g in scheduler_gpu_ids]} "
            f"slots={len(slot_pool)} "
            f"per_gpu_max={int(args.auto_workers_max_per_gpu)} "
            f"vram_aware={vram_aware_scheduler} "
            f"max_multiprocess={getattr(args, 'max_multiprocess', False)}"
        )
        if scheduler_tick_logger.enabled:
            scheduler_tick_logger.log(
                "scheduler_started",
                gpu_ids=[str(g) for g in scheduler_gpu_ids],
                slot_pool_size=len(slot_pool),
                planner_count=len(planner_requests),
                per_gpu_max_concurrency=int(args.auto_workers_max_per_gpu),
                managed_carla_single_worker_mode=bool(managed_carla_single_worker_mode),
                allow_multi_carla=bool(args.allow_multiple_carlas_per_gpu),
                start_carla=bool(args.start_carla),
                carla_vram_per_launch=bool(args.start_carla),
                max_multiprocess=bool(getattr(args, "max_multiprocess", False)),
                safety_buffer_mib=int(args.vram_safety_buffer_mib),
                min_free_mib=int(args.auto_workers_min_free_mib),
                carla_fixed_mib=int(args.carla_vram_estimate_mib),
            )

        # When VRAM-aware scheduling is enabled, lift the legacy "one planner
        # per GPU" cap that ``managed_carla_single_worker_mode`` imposed.
        # The bin-packer will make packing decisions against real VRAM
        # snapshots and per-planner estimates instead.
        if vram_aware_scheduler and planner_gpu_requests:
            cap_limit = max(1, int(args.auto_workers_max_per_gpu))
            for gpu_state in gpu_states.values():
                gpu_state.max_concurrency = cap_limit
                gpu_state.launch_capacity = max(gpu_state.launch_capacity, 1)

        def _ensure_colmdriver_vllm_started(*, trigger: str) -> None:
            nonlocal colmdriver_vllm_started
            global _active_colmdriver_vllm_manager
            if colmdriver_vllm_manager is None or colmdriver_vllm_started:
                return
            if args.dry_run:
                colmdriver_vllm_started = True
                return
            print(
                "[INFO] Starting deferred shared CoLMDriver vLLM services "
                f"(trigger={trigger})."
            )
            colmdriver_vllm_manager.start()
            _active_colmdriver_vllm_manager = colmdriver_vllm_manager
            _install_signal_handlers()
            colmdriver_vllm_started = True

            # Shared CoLMDriver services consume substantial VRAM; refresh GPU
            # launch capacities immediately so the scheduler rebalances safely.
            if planner_gpu_requests and (
                args.auto_workers_per_gpu or managed_carla_single_worker_mode
            ):
                for gpu in planner_gpu_requests:
                    gpu_state = gpu_states.get(gpu)
                    if gpu_state is None:
                        continue
                    state = refresh_gpu_launch_capacity(
                        gpu_state,
                        min_free_mib=int(args.auto_workers_min_free_mib),
                        max_concurrency=gpu_state.max_concurrency,
                        extra_min_free_mib=gpu_state.oom_penalty_mib,
                        settle_timeout_s=float(args.auto_workers_settle_timeout),
                        poll_seconds=float(args.auto_workers_poll_seconds),
                        stable_samples=int(args.auto_workers_stable_samples),
                        stable_tolerance_mib=int(args.auto_workers_stable_tolerance_mib),
                    )
                    if managed_carla_single_worker_mode and not vram_aware_scheduler:
                        state.launch_capacity = min(state.launch_capacity, 1)
                if not dashboard_enabled:
                    print(
                        "[INFO] Refreshed GPU launch capacities after deferred CoLMDriver "
                        "vLLM startup."
                    )

        def _collect_reserved_planner_ports(
            *,
            exclude_slot_index: int | None = None,
        ) -> set[int]:
            reserved_ports: set[int] = set()
            for queued_slot in slot_pool:
                if exclude_slot_index is not None and queued_slot.slot_index == exclude_slot_index:
                    continue
                reserved_ports.update(carla_service_ports(queued_slot.ports.port, tm_port_offset))
            with planner_state_lock:
                for state in planner_dashboard_states.values():
                    if (
                        state.slot_index is None
                        or state.ports is None
                        or state.phase not in {"launching", "running"}
                    ):
                        continue
                    if exclude_slot_index is not None and state.slot_index == exclude_slot_index:
                        continue
                    reserved_ports.update(carla_service_ports(state.ports.port, tm_port_offset))
            return reserved_ports

        def _allocate_replacement_slot(completed: PlannerThreadResult) -> PlannerSlot:
            if not args.start_carla:
                return PlannerSlot(
                    slot_index=completed.slot_index,
                    gpu=None,
                    ports=completed.ports,
                )
            reserved_ports = _collect_reserved_planner_ports(
                exclude_slot_index=completed.slot_index,
            )
            reserved_ports.update(carla_service_ports(completed.ports.port, tm_port_offset))
            preferred_port = completed.ports.port + max(1, int(args.carla_port_step))
            try:
                bundle = find_available_port_bundle(
                    str(args.host),
                    preferred_port,
                    tries=max(int(args.carla_port_tries), len(planner_requests) * 2),
                    step=int(args.carla_port_step),
                    tm_port_offset=tm_port_offset,
                    reserved_ports=reserved_ports,
                )
            except Exception:
                bundle = completed.ports
            return PlannerSlot(
                slot_index=completed.slot_index,
                gpu=None,
                ports=bundle,
            )

        def _queue_planner_retry(
            completed: PlannerThreadResult,
            *,
            reason: str,
            failure_kind: str,
        ) -> None:
            nonlocal scheduler_gpu_ids
            run_id = completed.planner_request.run_id
            current_attempt = int(planner_relaunch_counts.get(run_id, 0))
            dashboard_state = planner_dashboard_states[completed.planner_request.run_id]
            can_retry = should_continue_retrying(current_attempt, planner_retry_limit)
            gpu_state = gpu_states.get(completed.gpu)
            if gpu_state is not None:
                apply_gpu_failure_backoff(
                    gpu_state,
                    failure_kind=failure_kind,
                    args=args,
                )
            replacement_slot = _allocate_replacement_slot(completed)
            slot_pool.append(replacement_slot)

            if can_retry:
                # NOTE: We intentionally do NOT rename/delete the results
                # directory here.  The child's skip-completed-scenario logic
                # will detect scenarios that already have terminal outcomes
                # and only re-run the incomplete ones.  Previously this call
                # wiped completed scenario results on every parent retry.
                try:
                    dashboard_state.status_path.unlink()
                except FileNotFoundError:
                    pass
                requeue_pending_planner(completed.planner_request)
                scheduler_gpu_ids = rotate_scheduler_gpu_order_after_failure(
                    scheduler_gpu_ids,
                    completed.gpu,
                )
                append_retry_event(
                    planner_retry_log_path,
                    scope="multi_planner_parent",
                    attempt=current_attempt,
                    retry_limit=planner_retry_limit,
                    failure_kind=failure_kind,
                    outcome="retry_queued",
                    details=reason,
                    planner_run_id=run_id,
                    gpu=completed.gpu,
                    ports=completed.ports,
                )
            else:
                append_retry_event(
                    planner_retry_log_path,
                    scope="multi_planner_parent",
                    attempt=current_attempt,
                    retry_limit=planner_retry_limit,
                    failure_kind=failure_kind,
                    outcome="retry_exhausted",
                    details=reason,
                    planner_run_id=run_id,
                    gpu=completed.gpu,
                    ports=completed.ports,
                )
            with planner_state_lock:
                dashboard_state.phase = (
                    "queued"
                    if can_retry
                    else "failed"
                )
                dashboard_state.slot_index = None
                dashboard_state.gpu = None
                dashboard_state.ports = None
                dashboard_state.runner = None
                dashboard_state.start_time = None
                dashboard_state.end_time = None
                dashboard_state.returncode = (
                    None
                    if can_retry
                    else 1
                )
                dashboard_state.error = (
                    None
                    if can_retry
                    else (
                        "Planner retry budget exhausted "
                        f"(attempt={format_attempt_counter(current_attempt, planner_retry_limit)}, "
                        f"failure_kind={failure_kind}, reason={reason})."
                    )
                )
                dashboard_state.route_progress = (0, 1)
                dashboard_state.entry_status = ""
                dashboard_state.last_output_age_s = None
                dashboard_state.ego_route_scores = {}
                dashboard_state.ego_status = {}
                dashboard_state.carla_up = None
                dashboard_state.tm_up = None
            replacement_note = ""
            if replacement_slot.ports != completed.ports:
                replacement_note = (
                    " Reassigned fresh ports "
                    f"{replacement_slot.ports.port}/{replacement_slot.ports.tm_port}."
                )
            gpu_note = ""
            if completed.gpu is not None:
                gpu_note = f" De-prioritizing GPU {completed.gpu} for the next launch."
            if not can_retry:
                if not dashboard_enabled:
                    print(
                        "[ERROR] Planner retry budget exhausted: "
                        f"run_id={run_id} attempt={format_attempt_counter(current_attempt, planner_retry_limit)} "
                        f"failure_kind={failure_kind} reason={reason}"
                    )
                return
            if not dashboard_enabled:
                print(
                    "[WARN] Requeueing planner "
                    f"{completed.planner_request.run_id} after runtime failure ({reason})."
                    + replacement_note
                    + gpu_note
                )

        def _planner_runner_main(
            planner_request: PlannerRequest,
            slot: PlannerSlot,
            launch_attempt: int,
        ) -> None:
            runner: MonitoredSubprocessRunner | None = None
            outcome: PlannerThreadResult | None = None
            dashboard_state = planner_dashboard_states[planner_request.run_id]
            try:
                replacements: List[str] = [
                    "--planner",
                    planner_request.name,
                    "--port",
                    str(slot.ports.port),
                    "--traffic-manager-port",
                    str(slot.ports.tm_port),
                    "--results-tag",
                    f"{planner_base_results_tag}/{planner_request.run_id}",
                    "--scenario-retry-limit",
                    str(planner_child_retry_budget),
                ]
                value_options_to_remove = [
                    "--planner",
                    "--gpu",
                    "--gpus",
                    "--port",
                    "--traffic-manager-port",
                    "--results-tag",
                    "--scenario-retry-limit",
                    "--scenario-shard",
                    "--colmdriver-vllm-env",
                    "--colmdriver-vllm-startup-timeout",
                    "--carla-forensics-log-file",
                    "--carla-forensics-server-log-file",
                ]
                flag_options_to_remove: List[str] = ["--start-colmdriver-vllm", "--max-multiprocess"]
                # Defensive guard: child planners should only inherit dry-run
                # when the parent invocation explicitly requested it.
                if not args.dry_run:
                    flag_options_to_remove.append("--dry-run")
                if planner_routes_dir is not None:
                    replacements.extend(
                        [
                            "--routes-dir",
                            str(planner_routes_dir),
                            "--scenario-name",
                            str(planner_scenario_name),
                        ]
                    )
                    value_options_to_remove.extend(
                        [
                            "--zip",
                            "--routes-dir",
                            "--output-root",
                            "--scenario-name",
                        ]
                    )
                if args.overwrite:
                    flag_options_to_remove.append("--overwrite")
                # Pass scenario shard filter to the child if set.
                if planner_request.scenario_shard is not None:
                    replacements.extend([
                        "--scenario-shard",
                        str(planner_request.scenario_shard),
                    ])

                planner_forensics_log = (
                    add_suffix_to_path(forensics_log_path, planner_request.run_id or planner_request.name)
                    if forensics_log_path is not None
                    else None
                )
                planner_server_log = (
                    add_suffix_to_path(
                        forensics_server_log_path,
                        planner_request.run_id or planner_request.name,
                    )
                    if forensics_server_log_path is not None
                    else None
                )
                if planner_forensics_log is not None:
                    replacements.extend(
                        [
                            "--carla-forensics-log-file",
                            str(planner_forensics_log),
                        ]
                    )
                if planner_server_log is not None:
                    replacements.extend(
                        [
                            "--carla-forensics-server-log-file",
                            str(planner_server_log),
                        ]
                    )
                if args.carla_rootcause:
                    _planner_rc_root = str(
                        planner_results_root / planner_request.run_id
                    )
                    replacements.extend(
                        [
                            "--carla-rootcause-log-root",
                            _planner_rc_root,
                        ]
                    )
                    if "--carla-rootcause-log-root" not in value_options_to_remove:
                        value_options_to_remove.append("--carla-rootcause-log-root")

                child_argv = build_child_argv(
                    base_argv,
                    replacements=replacements,
                    value_options_to_remove=value_options_to_remove,
                    flag_options_to_remove=flag_options_to_remove,
                )
                if planner_request.conda_env:
                    cmd = [
                        "conda",
                        "run",
                        "--no-capture-output",
                        "-n",
                        planner_request.conda_env,
                        "python",
                        str(Path(__file__).resolve()),
                        *child_argv,
                    ]
                else:
                    cmd = [sys.executable, str(Path(__file__).resolve()), *child_argv]

                child_env = runtime_env.copy()
                if planner_request.conda_env:
                    child_env = _build_conda_launch_env(child_env)
                if slot.gpu is not None:
                    child_env["CUDA_VISIBLE_DEVICES"] = str(slot.gpu)
                child_env[DASHBOARD_STATUS_ENV] = str(dashboard_state.status_path)
                planner_child_timeout_s = None
                if (
                    not args.dry_run
                    and args.carla_crash_multiplier > 0
                    and planner_timeout_routes_dir is not None
                ):
                    planner_child_timeout_s = estimate_planner_child_timeout(
                        args,
                        planner_timeout_routes_dir,
                        planner_name=planner_request.name,
                        retry_limit=planner_child_retry_budget,
                    )
                    # Scale timeout for sharded planners: they run fewer scenarios.
                    if (
                        planner_child_timeout_s is not None
                        and planner_request.scenario_shard is not None
                        and dashboard_scenario_entries
                    ):
                        _shard_entries = _per_planner_scenario_entries.get(
                            planner_request.run_id, dashboard_scenario_entries
                        )
                        _shard_ratio = len(_shard_entries) / max(1, len(dashboard_scenario_entries))
                        if _shard_ratio < 1.0:
                            planner_child_timeout_s = max(
                                120.0,
                                planner_child_timeout_s * _shard_ratio,
                            )

                launch_message = (
                    "[INFO] Launching planner "
                    f"{planner_request.name} (run_id={planner_request.run_id}) "
                    f"launch_attempt={launch_attempt} "
                    f"on slot={slot.slot_index} gpu={slot.gpu or '<inherit>'} "
                    f"ports={slot.ports.port}/{slot.ports.tm_port}"
                    + (
                        f" conda_env={planner_request.conda_env}"
                        if planner_request.conda_env
                        else ""
                    )
                    + (
                        f" watchdog={planner_child_timeout_s:.1f}s"
                        if planner_child_timeout_s is not None
                        else ""
                    )
                )
                if not dashboard_enabled:
                    print(launch_message)
                _append_text_log_line(dashboard_state.log_path, launch_message)
                with planner_state_lock:
                    dashboard_state.phase = "running"
                    dashboard_state.slot_index = slot.slot_index
                    dashboard_state.gpu = slot.gpu
                    dashboard_state.ports = slot.ports
                    dashboard_state.start_time = time.monotonic()
                    dashboard_state.error = None
                    dashboard_state.returncode = None
                    dashboard_state.current_attempt = launch_attempt
                    dashboard_state.attempt_limit = planner_retry_limit
                _eval_env_python = child_env.get("CONDA_PYTHON_EXE") or cmd[0]
                _eval_env_path_head = os.pathsep.join(
                    child_env.get("PATH", "").split(os.pathsep)[:5]
                )
                _eval_env_msg = (
                    f"[EVAL ENV] python={_eval_env_python!r} "
                    f"PATH(first5)={_eval_env_path_head!r}"
                )
                if not dashboard_enabled:
                    print(_eval_env_msg)
                _append_text_log_line(dashboard_state.log_path, _eval_env_msg)
                gate_ready = wait_for_carla_teardown_gate(
                    host=str(args.host),
                    world_port=int(slot.ports.port),
                    tm_port=int(slot.ports.tm_port),
                    context=(
                        f"planner_launch:{planner_request.run_id}:"
                        f"slot={slot.slot_index}:attempt={launch_attempt}"
                    ),
                    quiet=dashboard_enabled,
                )
                if not gate_ready:
                    raise RuntimeError(
                        "Evaluator teardown gate timed out before planner launch "
                        f"(run_id={planner_request.run_id}, slot={slot.slot_index})."
                    )
                runner = MonitoredSubprocessRunner(
                    cmd,
                    env=child_env,
                    cwd=repo_root,
                    timeout_s=planner_child_timeout_s,
                    stall_output_timeout_s=float(args.scenario_stall_output_timeout),
                    output_prefix=f"[planner:{planner_request.run_id}] ",
                    show_carla_diag=bool(args.show_carla_diag),
                    emit_stdout=not dashboard_enabled,
                    log_path=dashboard_state.log_path,
                )
                with planner_state_lock:
                    dashboard_state.runner = runner
                with active_runners_lock:
                    active_runners.append(runner)
                result = runner.run()
                _planner_entries = _per_planner_scenario_entries.get(
                    planner_request.run_id, dashboard_scenario_entries
                )
                validation_error = validate_planner_child_results(
                    results_dir=dashboard_state.results_dir,
                    ego_count=planner_validation_ego_count,
                    scenario_entries=_planner_entries,
                    default_results_subdir=dashboard_state.default_results_subdir,
                )
                if result.returncode != 0 and validation_error is None:
                    _append_text_log_line(
                        dashboard_state.log_path,
                        "[INFO] Planner child exited nonzero after writing terminal route outcomes; "
                        f"accepting results (exit_code={result.returncode}).",
                    )
                with planner_state_lock:
                    dashboard_state.returncode = 0 if validation_error is None else 1
                    dashboard_state.end_time = time.monotonic()
                    dashboard_state.phase = "finished" if validation_error is None else "failed"
                    if validation_error is not None:
                        dashboard_state.error = validation_error
                if validation_error is not None:
                    outcome = PlannerThreadResult(
                        planner_request=planner_request,
                        slot_index=slot.slot_index,
                        gpu=slot.gpu,
                        ports=slot.ports,
                        error=validation_error,
                        failure_kind=classify_failure_text_kind(validation_error),
                    )
                else:
                    outcome = PlannerThreadResult(
                        planner_request=planner_request,
                        slot_index=slot.slot_index,
                        gpu=slot.gpu,
                        ports=slot.ports,
                        returncode=0 if validation_error is None else result.returncode,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                with planner_state_lock:
                    dashboard_state.phase = "failed"
                    dashboard_state.error = f"{type(exc).__name__}: {exc}"
                    dashboard_state.end_time = time.monotonic()
                outcome = PlannerThreadResult(
                    planner_request=planner_request,
                    slot_index=slot.slot_index,
                    gpu=slot.gpu,
                    ports=slot.ports,
                    error=f"{type(exc).__name__}: {exc}",
                    failure_kind=classify_failure_text_kind(f"{type(exc).__name__}: {exc}"),
                )
            finally:
                if runner is not None:
                    with active_runners_lock:
                        if runner in active_runners:
                            active_runners.remove(runner)
                    with planner_state_lock:
                        dashboard_state.runner = runner
                if outcome is None:
                    outcome = PlannerThreadResult(
                        planner_request=planner_request,
                        slot_index=slot.slot_index,
                        gpu=slot.gpu,
                        ports=slot.ports,
                        error="Planner thread exited without a result.",
                        failure_kind="generic",
                    )
                completion_queue.put(outcome)

        def _maybe_launch_planner(
            planner_request: PlannerRequest,
            slot_template: PlannerSlot,
            gpu_state: GPUPlannerState,
        ) -> None:
            nonlocal active_non_colmdriver_threads
            planner_relaunch_counts[planner_request.run_id] += 1
            launch_attempt = planner_relaunch_counts[planner_request.run_id]
            slot = PlannerSlot(
                slot_index=slot_template.slot_index,
                gpu=gpu_state.gpu,
                ports=slot_template.ports,
            )
            thread = threading.Thread(
                target=_planner_runner_main,
                args=(planner_request, slot, launch_attempt),
                daemon=True,
            )
            active_threads[slot.slot_index] = thread
            gpu_state.active_count += 1
            if planner_request.run_id not in deferred_colmdriver_run_ids:
                active_non_colmdriver_threads += 1
            with planner_state_lock:
                state = planner_dashboard_states[planner_request.run_id]
                state.phase = "launching"
                state.slot_index = slot.slot_index
                state.gpu = gpu_state.gpu
                state.ports = slot_template.ports
                state.current_attempt = launch_attempt
                state.attempt_limit = planner_retry_limit
            thread.start()

        # Track active reservations and assigned GPUs per run_id so that the
        # bin-packer can release them on completion.
        launch_reservations: Dict[str, Tuple[str | None, int]] = {}

        def _release_reservation(run_id: str) -> None:
            assignment = launch_reservations.pop(run_id, None)
            if assignment is None:
                return
            gpu_id, reserved_mib = assignment
            state = gpu_states.get(gpu_id)
            if state is None:
                return
            state.reserved_mib = max(0, int(state.reserved_mib) - int(reserved_mib))
            state.active_run_ids.discard(run_id)

        def _commit_reservation(run_id: str, gpu_id: str | None, reserved_mib: int) -> None:
            launch_reservations[run_id] = (gpu_id, int(reserved_mib))
            state = gpu_states.get(gpu_id)
            if state is not None:
                state.reserved_mib += int(reserved_mib)
                state.active_run_ids.add(run_id)

        def _launch_decisions_vram_aware() -> bool:
            """Run one tick of the VRAM-aware bin-packer. Returns True on launch."""
            nonlocal active_non_colmdriver_threads
            if not slot_pool:
                return False
            pending = list(planner_pending) + list(planner_pending_deferred)
            if not pending:
                return False
            decay_stale_oom_penalties(
                gpu_states,
                decay_seconds=float(args.oom_backoff_decay_seconds),
                cap_limit=max(1, int(args.auto_workers_max_per_gpu)),
                tick_logger=scheduler_tick_logger,
            )
            decisions, rejections = bin_pack_launch_plan(
                pending_requests=pending,
                gpu_states=gpu_states,
                gpu_monitor=gpu_monitor,
                vram_estimator=vram_estimator,
                carla_fixed_mib=int(args.carla_vram_estimate_mib),
                safety_buffer_mib=int(args.vram_safety_buffer_mib),
                min_free_mib=int(args.auto_workers_min_free_mib),
                active_non_colmdriver_threads=active_non_colmdriver_threads,
                deferred_run_ids=deferred_colmdriver_run_ids,
                managed_carla_single_worker_mode=managed_carla_single_worker_mode,
                allow_multi_carla=bool(args.allow_multiple_carlas_per_gpu),
                start_carla=bool(args.start_carla),
                tick_logger=scheduler_tick_logger,
            )
            launched_any = False
            for planner_request, gpu_id, reserved_mib in decisions:
                if not slot_pool:
                    break
                # Remove from whichever queue it lives in.
                if planner_request in planner_pending:
                    planner_pending.remove(planner_request)
                elif planner_request in planner_pending_deferred:
                    if active_non_colmdriver_threads > 0:
                        # Still gated; skip.
                        continue
                    planner_pending_deferred.remove(planner_request)
                else:
                    continue
                if (
                    defer_colmdriver_vllm_phase
                    and planner_request.run_id in deferred_colmdriver_run_ids
                ):
                    _ensure_colmdriver_vllm_started(
                        trigger=f"planner:{planner_request.run_id}"
                    )
                slot_template = slot_pool.popleft()
                gpu_state = gpu_states[gpu_id]
                _commit_reservation(planner_request.run_id, gpu_id, reserved_mib)
                if gpu_monitor is not None and gpu_id is not None:
                    gpu_monitor.reset_baseline(str(gpu_id))
                if not dashboard_enabled:
                    _gpu_stats = gpu_monitor.stats_for(gpu_id) if gpu_monitor and gpu_id else None
                    print(
                        f"[SCHED] Launching {planner_request.name} "
                        f"(run_id={planner_request.run_id}) on GPU {gpu_id} "
                        f"reserved={reserved_mib}MiB "
                        f"gpu_free={_gpu_stats.free_mib if _gpu_stats else '?'}MiB "
                        f"active_on_gpu={gpu_state.active_count + 1}"
                    )
                _sched_log(
                    f"LAUNCH {planner_request.name} run_id={planner_request.run_id} "
                    f"gpu={gpu_id} reserved={reserved_mib}MiB "
                    f"active_on_gpu={gpu_state.active_count + 1} "
                    f"slots_free={len(slot_pool)}"
                )
                _maybe_launch_planner(planner_request, slot_template, gpu_state)
                launched_any = True
            if not launched_any and rejections and has_launchable_planner_requests():
                # Log rejection batch so operators can diagnose stalls.
                if scheduler_tick_logger.enabled:
                    scheduler_tick_logger.log(
                        "launch_rejected_batch",
                        reject_count=len(rejections),
                        first_rejection=rejections[0],
                    )
                # Also print a console summary periodically so stalls are visible
                # even without the tick log file.
                for rej in rejections[:3]:
                    gpu_reasons = rej.get("gpu_rejections", [])
                    summary_parts = [f"run_id={rej.get('run_id')} planner={rej.get('planner')}"]
                    for gr in gpu_reasons[:2]:
                        summary_parts.append(
                            f"gpu={gr.get('gpu')} reason={gr.get('reason')} "
                            f"free={gr.get('effective_free_mib', gr.get('free_mib', '?'))}MiB "
                            f"cost={gr.get('planner_cost_mib', '?')}MiB "
                            f"floor={gr.get('required_floor_mib', '?')}MiB"
                        )
                    if not dashboard_enabled:
                        print(f"[SCHED] Launch blocked: {'; '.join(summary_parts)}")
                    _sched_log(f"BLOCKED {'; '.join(summary_parts)}")
            return launched_any

        def _log_scheduler_snapshot(reason: str) -> None:
            if not scheduler_tick_logger.enabled:
                return
            per_gpu: Dict[str, Any] = {}
            for gpu_id, state in gpu_states.items():
                stats = None
                if gpu_monitor is not None and gpu_id is not None:
                    stats = gpu_monitor.stats_for(str(gpu_id))
                elif state.last_stats is not None:
                    stats = state.last_stats
                per_gpu[str(gpu_id)] = {
                    "active": int(state.active_count),
                    "reserved_mib": int(state.reserved_mib),
                    "max_concurrency": int(state.max_concurrency),
                    "oom_penalty_mib": int(state.oom_penalty_mib),
                    "free_mib": int(stats.free_mib) if stats is not None else None,
                    "used_mib": int(stats.used_mib) if stats is not None else None,
                    "total_mib": int(stats.total_mib) if stats is not None else None,
                    "run_ids": sorted(state.active_run_ids),
                }
            scheduler_tick_logger.log(
                "scheduler_tick",
                reason=reason,
                pending=len(planner_pending),
                pending_deferred=len(planner_pending_deferred),
                active_threads=len(active_threads),
                slot_pool=len(slot_pool),
                active_non_colmdriver_threads=int(active_non_colmdriver_threads),
                per_gpu=per_gpu,
            )

        try:
            try:
                planner_dashboard.start()
                last_snapshot_log = 0.0
                while has_pending_planner_requests() or active_threads:
                    # -------------------- DISPATCH PHASE --------------------
                    if vram_aware_scheduler:
                        launch_made = _launch_decisions_vram_aware()
                    else:
                        # Legacy path retained for fallback / debugging.
                        launch_made = False
                        for gpu_key in scheduler_gpu_ids:
                            gpu_state = gpu_states[gpu_key]
                            if (
                                not has_launchable_planner_requests()
                                or not slot_pool
                                or gpu_state.active_count >= gpu_state.launch_capacity
                            ):
                                continue
                            planner_request = pop_next_pending_planner()
                            if planner_request is None:
                                continue
                            if (
                                defer_colmdriver_vllm_phase
                                and planner_request.run_id in deferred_colmdriver_run_ids
                            ):
                                _ensure_colmdriver_vllm_started(
                                    trigger=f"planner:{planner_request.run_id}"
                                )
                            slot_template = slot_pool.popleft()
                            _maybe_launch_planner(planner_request, slot_template, gpu_state)
                            launch_made = True
                            # No inline refresh: we rely on the background
                            # monitor + max_concurrency cap.
                            gpu_state.launch_capacity = max(
                                gpu_state.launch_capacity,
                                gpu_state.active_count + 1,
                            )
                            gpu_state.launch_capacity = min(
                                gpu_state.launch_capacity,
                                max(1, int(gpu_state.max_concurrency)),
                            )

                    if launch_made and has_launchable_planner_requests() and slot_pool:
                        # Try to pack more on the same tick if anything is left.
                        continue

                    # ----- FORCED-LAUNCH FALLBACK (queue stalled, no active) ----
                    if (
                        not launch_made
                        and has_launchable_planner_requests()
                        and not active_threads
                    ):
                        forced_state = choose_force_launch_gpu(gpu_states)
                        forced_state.launch_capacity = max(
                            forced_state.launch_capacity,
                            forced_state.active_count + 1,
                        )
                        forced_state.launch_capacity = min(
                            forced_state.launch_capacity,
                            max(1, int(forced_state.max_concurrency)),
                        )
                        # Forced launches bypass the VRAM floor, but still
                        # reserve what the estimator believes they'll use.
                        if has_launchable_planner_requests() and slot_pool:
                            planner_request = pop_next_pending_planner()
                            if planner_request is not None:
                                if (
                                    defer_colmdriver_vllm_phase
                                    and planner_request.run_id in deferred_colmdriver_run_ids
                                ):
                                    _ensure_colmdriver_vllm_started(
                                        trigger=f"planner:{planner_request.run_id}"
                                    )
                                slot_template = slot_pool.popleft()
                                est = vram_estimator.estimate(planner_request.name)
                                if args.start_carla:
                                    est += int(args.carla_vram_estimate_mib)
                                _commit_reservation(
                                    planner_request.run_id, forced_state.gpu, est,
                                )
                                if gpu_monitor is not None and forced_state.gpu is not None:
                                    gpu_monitor.reset_baseline(str(forced_state.gpu))
                                _maybe_launch_planner(planner_request, slot_template, forced_state)
                                if scheduler_tick_logger.enabled:
                                    scheduler_tick_logger.log(
                                        "forced_launch",
                                        run_id=planner_request.run_id,
                                        gpu=str(forced_state.gpu) if forced_state.gpu is not None else None,
                                        reserved_mib=est,
                                    )
                                if not dashboard_enabled:
                                    print(
                                        "[WARN] Planner queue was stalled by the current GPU threshold. "
                                        f"Forced launch on GPU {forced_state.gpu}."
                                    )
                        continue

                    if not active_threads:
                        continue

                    # -------------------- COMPLETION PHASE ------------------
                    try:
                        completed = completion_queue.get(
                            timeout=float(args.scheduler_tick_interval),
                        )
                    except queue.Empty:
                        # Periodically log a snapshot so stalls are visible.
                        now_mono = time.monotonic()
                        if now_mono - last_snapshot_log > 10.0:
                            _log_scheduler_snapshot("idle_tick")
                            last_snapshot_log = now_mono
                        continue

                    completed_thread = active_threads.pop(completed.slot_index, None)
                    if completed_thread is not None:
                        completed_thread.join(timeout=0.1)

                    state = gpu_states[completed.gpu]
                    state.active_count = max(0, state.active_count - 1)
                    if completed.planner_request.run_id not in deferred_colmdriver_run_ids:
                        active_non_colmdriver_threads = max(
                            0,
                            int(active_non_colmdriver_threads) - 1,
                        )

                    # Merge a peak-VRAM observation back into the estimator
                    # cache for next time.
                    if (
                        gpu_monitor is not None
                        and completed.gpu is not None
                    ):
                        observed_peak = gpu_monitor.consume_peak_delta(str(completed.gpu))
                        if observed_peak > 0:
                            vram_estimator.observe_peak(
                                completed.planner_request.name,
                                observed_peak,
                            )
                            if scheduler_tick_logger.enabled:
                                scheduler_tick_logger.log(
                                    "vram_peak_observed",
                                    run_id=completed.planner_request.run_id,
                                    planner=completed.planner_request.name,
                                    peak_mib=int(observed_peak),
                                    new_estimate_mib=int(
                                        vram_estimator.estimate(
                                            completed.planner_request.name
                                        )
                                    ),
                                )

                    _release_reservation(completed.planner_request.run_id)
                    state.launch_capacity = max(1, state.active_count + 1)
                    state.launch_capacity = min(
                        state.launch_capacity,
                        max(1, int(state.max_concurrency)),
                    )
                    if not dashboard_enabled:
                        _gpu_stats = gpu_monitor.stats_for(completed.gpu) if gpu_monitor and completed.gpu else None
                        _status = "OK" if (completed.error is None and completed.returncode == 0) else "FAIL"
                        print(
                            f"[SCHED] Completed {completed.planner_request.name} "
                            f"(run_id={completed.planner_request.run_id}) [{_status}] "
                            f"gpu={completed.gpu} "
                            f"gpu_free={_gpu_stats.free_mib if _gpu_stats else '?'}MiB "
                            f"active_on_gpu={state.active_count} "
                            f"pending={len(planner_pending)+len(planner_pending_deferred)} "
                            f"slots={len(slot_pool)}"
                        )
                    _sched_log(
                        f"DONE {completed.planner_request.name} "
                        f"run_id={completed.planner_request.run_id} "
                        f"{'OK' if (completed.error is None and completed.returncode == 0) else 'FAIL'} "
                        f"gpu={completed.gpu} active_on_gpu={state.active_count} "
                        f"pending={len(planner_pending)+len(planner_pending_deferred)} "
                        f"slots={len(slot_pool)}"
                    )
                    _log_scheduler_snapshot("completion")

                    if completed.error is not None or completed.returncode != 0:
                        dashboard_state = planner_dashboard_states[completed.planner_request.run_id]
                        reason = (
                            str(completed.error)
                            if completed.error is not None
                            else f"exit_code={completed.returncode}"
                        )
                        failure_kind = infer_planner_thread_failure_kind(
                            completed,
                            dashboard_state=dashboard_state,
                        )
                        # Raise the planner's VRAM floor so the bin-packer
                        # will steer it onto a bigger GPU next time rather
                        # than repeating the OOM.
                        if failure_kind in {"cuda_oom", "cuda_unavailable"}:
                            bump_mib = max(
                                1024,
                                int(args.oom_free_mib_penalty_step),
                            )
                            vram_estimator.bump_after_oom(
                                completed.planner_request.name,
                                bump_mib=bump_mib,
                            )
                            if scheduler_tick_logger.enabled:
                                scheduler_tick_logger.log(
                                    "oom_bump_estimate",
                                    run_id=completed.planner_request.run_id,
                                    planner=completed.planner_request.name,
                                    bump_mib=int(bump_mib),
                                    new_estimate_mib=int(
                                        vram_estimator.estimate(
                                            completed.planner_request.name
                                        )
                                    ),
                                )
                        _queue_planner_retry(
                            completed,
                            reason=reason,
                            failure_kind=failure_kind,
                        )
                    else:
                        apply_gpu_success_recovery(
                            state,
                            args=args,
                        )
                        slot_pool.append(
                            PlannerSlot(
                                slot_index=completed.slot_index,
                                gpu=None,
                                ports=completed.ports,
                            )
                        )
                        if not dashboard_enabled:
                            print(
                                "[INFO] Planner "
                                f"{completed.planner_request.run_id} finished successfully."
                            )
                        append_retry_event(
                            planner_retry_log_path,
                            scope="multi_planner_parent",
                            attempt=int(planner_relaunch_counts.get(completed.planner_request.run_id, 0)),
                            retry_limit=planner_retry_limit,
                            failure_kind="none",
                            outcome="success",
                            details="planner_completed",
                            planner_run_id=completed.planner_request.run_id,
                            gpu=completed.gpu,
                            ports=completed.ports,
                        )
            except KeyboardInterrupt:
                print("[INFO] Interrupt received. Terminating active planner runs...")
                with active_runners_lock:
                    active_snapshot = list(active_runners)
                for runner in active_snapshot:
                    runner.terminate("[INFO] Stopping planner child process.")
                for thread in active_threads.values():
                    thread.join(timeout=5.0)

                # Mark in-flight routes as interrupted so they are clearly
                # identifiable as partial results rather than silent failures.
                interrupted_run_ids: List[str] = []
                with planner_state_lock:
                    for run_id, state in planner_dashboard_states.items():
                        if state.phase in ("launching", "running"):
                            state.phase = "interrupted"
                            state.end_time = time.monotonic()
                            state.error = "Evaluator interrupted by user (SIGINT/Ctrl+C)."
                            interrupted_run_ids.append(run_id)
                for run_id in interrupted_run_ids:
                    state = planner_dashboard_states[run_id]
                    progress = state.route_progress
                    try:
                        update_dashboard_status(
                            state.status_path,
                            phase="interrupted",
                            error=state.error,
                            route_progress_current=progress[0],
                            route_progress_total=progress[1],
                            interrupted_at=dt.datetime.now().isoformat(timespec="seconds"),
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    print(
                        f"[INFO] Marked {run_id} as interrupted "
                        f"(progress: {progress[0]}/{progress[1]} routes)."
                    )
                if interrupted_run_ids:
                    print(
                        f"[INFO] {len(interrupted_run_ids)} route(s) marked as interrupted (partial)."
                    )
                raise
        finally:
            # Safety-net: mark any routes still in-flight (covers SIGTERM /
            # SystemExit / unexpected exceptions not caught above).
            with planner_state_lock:
                _still_inflight = [
                    (rid, st) for rid, st in planner_dashboard_states.items()
                    if st.phase in ("launching", "running")
                ]
            for rid, st in _still_inflight:
                with planner_state_lock:
                    st.phase = "interrupted"
                    st.end_time = time.monotonic()
                    if not st.error:
                        st.error = "Evaluator terminated unexpectedly."
                progress = st.route_progress
                try:
                    update_dashboard_status(
                        st.status_path,
                        phase="interrupted",
                        error=st.error,
                        route_progress_current=progress[0],
                        route_progress_total=progress[1],
                        interrupted_at=dt.datetime.now().isoformat(timespec="seconds"),
                    )
                except Exception:  # noqa: BLE001
                    pass
                print(
                    f"[INFO] Marked {rid} as interrupted "
                    f"(progress: {progress[0]}/{progress[1]} routes)."
                )

            planner_dashboard.stop()
            if gpu_monitor is not None:
                try:
                    gpu_monitor.stop()
                    gpu_monitor.join(timeout=2.0)
                except Exception:  # pylint: disable=broad-except
                    pass
            if scheduler_tick_logger.enabled:
                scheduler_tick_logger.log(
                    "scheduler_finished",
                    vram_estimates=vram_estimator.snapshot(),
                )
            if colmdriver_vllm_manager is not None:
                colmdriver_vllm_manager.stop()
            _active_colmdriver_vllm_manager = None
            wall_s = time.monotonic() - run_start_monotonic
            for line in format_performance_summary_lines(wall_clock_s=wall_s):
                print(line)
            write_performance_report(
                planner_perf_report_path,
                wall_clock_s=wall_s,
                context={
                    "mode": "multi_planner",
                    "planner_count": len(planner_requests),
                    "results_root": str(planner_results_root),
                    "vram_estimates_final": vram_estimator.snapshot(),
                },
            )
        interrupted_planners = [
            run_id
            for run_id, state in planner_dashboard_states.items()
            if state.phase == "interrupted"
        ]
        failed_planners = [
            run_id
            for run_id, state in planner_dashboard_states.items()
            if state.phase == "failed" or (state.returncode is not None and int(state.returncode) != 0)
        ]
        if interrupted_planners:
            print(
                f"[WARN] {len(interrupted_planners)} planner(s) interrupted (partial results): "
                + ", ".join(sorted(interrupted_planners))
            )
        if failed_planners:
            raise RuntimeError(
                "One or more planner workers failed after retry exhaustion: "
                + ", ".join(sorted(failed_planners))
            )
        return

    scenario_summary = None
    routes_dir: Path
    scenario_name = args.scenario_name
    manifest_path: Path | None = None
    actors_manifest: Dict[str, List[Dict[str, str]]] | None = None
    fake_ego_temp_dir: tempfile.TemporaryDirectory | None = None
    fake_ego_views: List[Tuple[int, str]] = []

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

        # Check if routes_dir contains multiple scenario subfolders
        scenario_subfolders = detect_scenario_subfolders(routes_dir, args.routes_leaf)
        # Apply scenario shard filter if specified.
        if scenario_subfolders and getattr(args, "_scenario_shard_parsed", None) is not None:
            _shard_idx, _shard_cnt = args._scenario_shard_parsed
            _all_count = len(scenario_subfolders)
            scenario_subfolders = [
                (sf_path, sf_name)
                for i, (sf_path, sf_name) in enumerate(scenario_subfolders)
                if i % _shard_cnt == _shard_idx
            ]
            print(
                f"[INFO] Scenario shard {_shard_idx}/{_shard_cnt}: "
                f"running {len(scenario_subfolders)} of {_all_count} scenarios."
            )
        if scenario_subfolders:
            base_results_tag = infer_base_results_tag(args)
            multi_scenario_results_root = (
                repo_root / "results" / "results_driving_custom" / base_results_tag
            )
            multi_scenario_retry_log_path = multi_scenario_results_root / "_retry_events.jsonl"
            multi_scenario_perf_report_path: Path | None = None
            if args.perf_report_file:
                perf_candidate = Path(str(args.perf_report_file))
                multi_scenario_perf_report_path = (
                    perf_candidate
                    if perf_candidate.is_absolute()
                    else multi_scenario_results_root / perf_candidate
                )
            print(f"Detected {len(scenario_subfolders)} scenario subfolders in {routes_dir}:")
            for sf_path, sf_name in scenario_subfolders:
                print(f"  - {sf_name} ({sf_path})")
            print("\nRunning each scenario sequentially...\n")
            completed_scenario_count = 0
            update_dashboard_status(
                dashboard_status_path,
                phase="running",
                scenario_mode="multi",
                scenario_total=len(scenario_subfolders),
                scenario_index=0,
                completed_scenarios=0,
                scenario_name="",
                scenario_results_subdir="",
                current_attempt=0,
                attempt_limit=retry_limit_for_status(args.scenario_retry_limit),
            )

            carla_manager = None
            if args.start_carla and not args.dry_run:
                _scenario_rootcause_config = carla_rootcause_config
                if _scenario_rootcause_config is not None:
                    _rc_results_base = (
                        repo_root / "results" / "results_driving_custom" / base_results_tag
                    )
                    _rc_results_base.mkdir(parents=True, exist_ok=True)
                    _scenario_rootcause_config = _dc_replace(
                        _scenario_rootcause_config, log_root=_rc_results_base,
                    )
                carla_manager = CarlaProcessManager(
                    carla_root=carla_root,
                    host=str(args.host),
                    port=args.port,
                    tm_port_offset=tm_port_offset,
                    extra_args=args.carla_arg,
                    env=runtime_env.copy(),
                    port_tries=args.carla_port_tries,
                    port_step=args.carla_port_step,
                    server_log_path=(
                        forensics_server_log_path
                        if args.carla_forensics and not args.carla_rootcause
                        else None
                    ),
                    rootcause_config=_scenario_rootcause_config,
                )
                _active_carla_manager = carla_manager
                _install_signal_handlers()
                try:
                    selected_bundle = carla_manager.start()
                except Exception as exc:
                    append_retry_event(
                        multi_scenario_retry_log_path,
                        scope="multi_scenario_launcher",
                        attempt=1,
                        retry_limit=normalize_retry_limit(args.scenario_retry_limit),
                        failure_kind="carla_startup_failure",
                        outcome="retry_exhausted",
                        details=f"{type(exc).__name__}: {exc}",
                        scenario_name="",
                        gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                        ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                    )
                    wall_s = time.monotonic() - run_start_monotonic
                    for line in format_performance_summary_lines(wall_clock_s=wall_s):
                        print(line)
                    write_performance_report(
                        multi_scenario_perf_report_path,
                        wall_clock_s=wall_s,
                        context={
                            "mode": "multi_scenario",
                            "results_root": str(multi_scenario_results_root),
                            "finalized_in": "carla_start_failure",
                        },
                    )
                    raise
                set_effective_ports(
                    args,
                    world_port=selected_bundle.port,
                    tm_port_offset=tm_port_offset,
                )
                sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                wait_for_carla_post_start_buffer(args.carla_post_start_buffer)

            try:
                # Run each subfolder as a separate scenario
                for i, (subfolder, sub_name) in enumerate(scenario_subfolders, 1):
                    print(f"\n{'='*60}")
                    print(f"SCENARIO {i}/{len(scenario_subfolders)}: {sub_name}")
                    print(f"{'='*60}")
                    scenario_results_subdir = combine_results_subdir(args.results_subdir, sub_name)
                    effective_results_tag = f"{base_results_tag}/{scenario_results_subdir}"
                    scenario_result_roots = planned_result_roots(
                        repo_root,
                        effective_results_tag,
                        args,
                    )
                    timeout_seconds = None
                    child_retry_budget = planner_child_retry_limit(args.scenario_retry_limit)
                    subfolder_ego_count = max(
                        1,
                        detect_ego_routes(
                            subfolder,
                            replay_mode=(args.planner == "log-replay"),
                        ),
                    )
                    # Skip scenarios that already have terminal outcomes from a
                    # previous (possibly crashed) run.  This avoids re-running
                    # scenarios that completed successfully before the parent
                    # child was retried.
                    if result_roots_have_terminal_outcomes(
                        scenario_result_roots,
                        ego_count=subfolder_ego_count,
                    ):
                        print(
                            f"[INFO] Scenario {sub_name} already completed "
                            f"(terminal outcomes exist) — skipping."
                        )
                        completed_scenario_count += 1
                        update_dashboard_status(
                            dashboard_status_path,
                            phase="running",
                            scenario_mode="multi",
                            scenario_total=len(scenario_subfolders),
                            scenario_index=i,
                            completed_scenarios=completed_scenario_count,
                            scenario_name=sub_name,
                            scenario_results_subdir=scenario_results_subdir,
                            current_attempt=0,
                            attempt_limit=normalize_retry_limit(args.scenario_retry_limit),
                        )
                        continue
                    if not args.dry_run and args.carla_crash_multiplier > 0:
                        timeout_seconds = estimate_scenario_timeout(
                            args,
                            subfolder,
                            planner_name=args.planner,
                        )
                        print(
                            f"[INFO] Scenario timeout watchdog: {timeout_seconds:.1f}s "
                            f"(stall quiet window {float(args.scenario_stall_output_timeout):.1f}s)"
                        )

                    max_retries = normalize_retry_limit(args.scenario_retry_limit)
                    attempt = 0
                    scenario_completed = False
                    # FD tracker checkpoint: before scenario execution
                    try:
                        _fdt.checkpoint(f"scenario_{i:03d}_{sub_name}_start")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    while should_continue_retrying(attempt, max_retries) and not scenario_completed:
                        attempt += 1
                        update_dashboard_status(
                            dashboard_status_path,
                            phase="running",
                            scenario_mode="multi",
                            scenario_total=len(scenario_subfolders),
                            scenario_index=i,
                            completed_scenarios=completed_scenario_count,
                            scenario_name=sub_name,
                            scenario_results_subdir=scenario_results_subdir,
                            current_attempt=attempt,
                            attempt_limit=max_retries,
                        )
                        if carla_manager and not carla_manager.is_running():
                            if args.carla_rootcause:
                                _print_failure_banner(
                                    f"CARLA process exited unexpectedly before scenario {sub_name!r} (attempt {attempt}).",
                                    "reason=carla_crash, --carla-rootcause enabled -- aborting to preserve logs.",
                                )
                                carla_manager.plan_logdir_outcome("FAILED_carla_crash")
                                raise SystemExit(1)
                            _print_failure_banner(
                                f"CARLA process is not running before scenario {sub_name!r} (attempt {attempt}).",
                                "reason=carla_crash -- restarting CARLA.",
                            )
                            selected_bundle = carla_manager.restart(args.carla_restart_wait, failure_reason="carla_crash")
                            set_effective_ports(
                                args,
                                world_port=selected_bundle.port,
                                tm_port_offset=tm_port_offset,
                            )
                            sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                            wait_for_carla_post_start_buffer(args.carla_post_start_buffer)

                        scenario_flag_options_to_remove = [
                            "--start-carla",
                            "--start-colmdriver-vllm",
                            "--carla-rootcause",
                        ]
                        # Defensive guard: scenario children should only inherit
                        # dry-run when the parent invocation requested it.
                        if not args.dry_run:
                            scenario_flag_options_to_remove.append("--dry-run")

                        child_argv = build_child_argv(
                            base_argv,
                            replacements=[
                                "--routes-dir",
                                str(subfolder),
                                "--results-tag",
                                base_results_tag,
                                "--results-subdir",
                                scenario_results_subdir,
                                "--scenario-name",
                                sub_name,
                                "--port",
                                str(args.port),
                                "--traffic-manager-port",
                                str(args.tm_port),
                                "--scenario-retry-limit",
                                str(child_retry_budget),
                            ],
                            value_options_to_remove=[
                                "--zip",
                                "--routes-dir",
                                "--results-tag",
                                "--results-subdir",
                                "--scenario-name",
                                "--port",
                                "--traffic-manager-port",
                                "--scenario-retry-limit",
                                "--colmdriver-vllm-env",
                                "--colmdriver-vllm-startup-timeout",
                                "--carla-forensics-server-log-file",
                                "--carla-rootcause-script",
                                "--carla-rootcause-mode",
                                "--carla-rootcause-diag-interval",
                                "--carla-rootcause-startup-timeout",
                                "--carla-rootcause-core-wait-seconds",
                                "--carla-rootcause-log-root",
                            ],
                            flag_options_to_remove=scenario_flag_options_to_remove,
                        )
                        new_cmd = [sys.executable, str(Path(__file__).resolve()), *child_argv]
                        gate_ready = wait_for_carla_teardown_gate(
                            host=str(args.host),
                            world_port=int(args.port),
                            tm_port=int(args.tm_port),
                            context=(
                                f"scenario_child_launch:{sub_name}:"
                                f"attempt={attempt}/{max_retries if max_retries is not None else 'inf'}"
                            ),
                        )
                        if not gate_ready:
                            synthetic_gate_result = MonitoredRunResult(
                                returncode=124,
                                stalled=False,
                                elapsed_s=0.0,
                                last_output_age_s=0.0,
                                output_tail=[
                                    "Evaluator teardown gate timed out before scenario child launch."
                                ],
                                terminated_reason="teardown_gate_timeout",
                            )
                            failure_kind = "teardown_gate_timeout"
                            if should_retry_failure(attempt, max_retries, failure_kind):
                                append_retry_event(
                                    multi_scenario_retry_log_path,
                                    scope="multi_scenario_child",
                                    attempt=attempt,
                                    retry_limit=max_retries,
                                    failure_kind=failure_kind,
                                    outcome="retry_queued",
                                    details="teardown_gate_timeout_before_child_launch",
                                    scenario_name=sub_name,
                                    gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                    ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                                )
                                _print_failure_banner(
                                    f"Scenario {sub_name!r} launch blocked by teardown gate on attempt "
                                    f"{format_attempt_counter(attempt, max_retries)} -- will retry.",
                                    _describe_failure_detail(
                                        failure_kind,
                                        synthetic_gate_result,
                                        carla_manager,
                                        stage="teardown_gate",
                                        progress_summary=None,
                                    ),
                                )
                                delete_result_roots(scenario_result_roots)
                                if carla_manager:
                                    print(
                                        "[INFO] Restarting CARLA before retrying the scenario "
                                        "after teardown-gate timeout."
                                    )
                                    selected_bundle = carla_manager.restart(
                                        args.carla_restart_wait,
                                        failure_reason=failure_kind,
                                    )
                                    set_effective_ports(
                                        args,
                                        world_port=selected_bundle.port,
                                        tm_port_offset=tm_port_offset,
                                    )
                                    sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                                    wait_for_carla_post_start_buffer(args.carla_post_start_buffer)
                                continue
                            _print_failure_banner(
                                f"Scenario {sub_name!r} launch blocked by teardown gate (non-retryable).",
                                _describe_failure_detail(
                                    failure_kind,
                                    synthetic_gate_result,
                                    carla_manager,
                                    stage="teardown_gate",
                                    progress_summary=None,
                                ),
                            )
                            append_retry_event(
                                multi_scenario_retry_log_path,
                                scope="multi_scenario_child",
                                attempt=attempt,
                                retry_limit=max_retries,
                                failure_kind=failure_kind,
                                outcome="retry_exhausted",
                                details="teardown_gate_timeout_before_child_launch",
                                scenario_name=sub_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                            )
                            raise RuntimeError(
                                "Evaluator teardown gate timed out before scenario child launch "
                                f"(scenario={sub_name}, attempt={attempt})."
                            )
                        _grandchild_env = runtime_env.copy()
                        _grandchild_env.pop(DASHBOARD_STATUS_ENV, None)
                        result = MonitoredSubprocessRunner(
                            new_cmd,
                            env=_grandchild_env,
                            cwd=repo_root,
                            timeout_s=timeout_seconds,
                            stall_output_timeout_s=float(args.scenario_stall_output_timeout),
                            output_prefix=f"[scenario:{sub_name}] ",
                            show_carla_diag=bool(args.show_carla_diag),
                        ).run()
                        if args.dry_run and result.returncode == 0:
                            scenario_completed = True
                            completed_scenario_count += 1
                            append_retry_event(
                                multi_scenario_retry_log_path,
                                scope="multi_scenario_child",
                                attempt=attempt,
                                retry_limit=max_retries,
                                failure_kind="none",
                                outcome="success",
                                details="dry_run_child_success",
                                scenario_name=sub_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                            )
                            update_dashboard_status(
                                dashboard_status_path,
                                phase="running",
                                scenario_mode="multi",
                                scenario_total=len(scenario_subfolders),
                                scenario_index=i,
                                completed_scenarios=completed_scenario_count,
                                scenario_name=sub_name,
                                scenario_results_subdir=scenario_results_subdir,
                                current_attempt=attempt,
                                attempt_limit=max_retries,
                            )
                            break
                        if result_roots_have_terminal_outcomes(
                            scenario_result_roots,
                            ego_count=subfolder_ego_count,
                        ):
                            if result.returncode != 0:
                                print(
                                    "[WARN] Scenario "
                                    f"{sub_name} wrote terminal baselines despite child exit_code="
                                    f"{result.returncode}; accepting results."
                                )
                            scenario_completed = True
                            completed_scenario_count += 1
                            append_retry_event(
                                multi_scenario_retry_log_path,
                                scope="multi_scenario_child",
                                attempt=attempt,
                                retry_limit=max_retries,
                                failure_kind="none",
                                outcome="success",
                                details="terminal_outcomes_written",
                                scenario_name=sub_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                            )
                            update_dashboard_status(
                                dashboard_status_path,
                                phase="running",
                                scenario_mode="multi",
                                scenario_total=len(scenario_subfolders),
                                scenario_index=i,
                                completed_scenarios=completed_scenario_count,
                                scenario_name=sub_name,
                                scenario_results_subdir=scenario_results_subdir,
                                current_attempt=attempt,
                                attempt_limit=max_retries,
                            )
                            break

                        progress_summaries = [
                            collect_result_root_progress(
                                result_root,
                                ego_count=subfolder_ego_count,
                            )
                            for result_root in scenario_result_roots
                        ]
                        if result.returncode != 0 and args.accept_stalled_near_end_failures:
                            stalled_accepts: List[Tuple[Path, str]] = []
                            for result_root, summary in zip(scenario_result_roots, progress_summaries):
                                accept_stalled, accept_detail = should_accept_stalled_near_end_failure(
                                    result=result,
                                    result_root=result_root,
                                    summary=summary,
                                    min_route_score=float(args.stalled_near_end_min_route_score),
                                    require_stall_evidence=bool(args.stalled_near_end_require_stall_evidence),
                                )
                                if accept_stalled:
                                    stalled_accepts.append((result_root, accept_detail))
                            if stalled_accepts and len(stalled_accepts) == len(progress_summaries):
                                accepted_msg = "; ".join(
                                    f"{root}: {detail}" for root, detail in stalled_accepts
                                )
                                print(
                                    "[WARN] Scenario "
                                    f"{sub_name} exited non-zero but matches stalled-near-end "
                                    f"acceptance policy; keeping results ({accepted_msg})."
                                )
                                scenario_completed = True
                                completed_scenario_count += 1
                                append_retry_event(
                                    multi_scenario_retry_log_path,
                                    scope="multi_scenario_child",
                                    attempt=attempt,
                                    retry_limit=max_retries,
                                    failure_kind="accepted_stalled_near_end",
                                    outcome="success",
                                    details=accepted_msg,
                                    scenario_name=sub_name,
                                    gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                    ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                                )
                                if carla_manager and args.carla_rootcause:
                                    carla_manager.plan_logdir_outcome("ACCEPTED_stalled_near_end")
                                update_dashboard_status(
                                    dashboard_status_path,
                                    phase="running",
                                    scenario_mode="multi",
                                    scenario_total=len(scenario_subfolders),
                                    scenario_index=i,
                                    completed_scenarios=completed_scenario_count,
                                    scenario_name=sub_name,
                                    scenario_results_subdir=scenario_results_subdir,
                                    current_attempt=attempt,
                                    attempt_limit=max_retries,
                                )
                                break
                        progress_summary = next(
                            (
                                summary
                                for summary in progress_summaries
                                if did_scenario_record_runtime_failure(summary)
                            ),
                            None,
                        )
                        if progress_summary is None:
                            progress_summary = next(
                                (
                                    summary
                                    for summary in progress_summaries
                                    if summary.route_progress[0] > 0
                                ),
                                progress_summaries[0] if progress_summaries else None,
                            )
                        any_runtime_failure = any(
                            did_scenario_record_runtime_failure(summary)
                            for summary in progress_summaries
                        )
                        any_route_progress = any(
                            summary.route_progress[0] > 0 for summary in progress_summaries
                        )
                        if not any_route_progress:
                            failure_kind = (
                                "no_route_execution"
                                if any(summary.started for summary in progress_summaries)
                                else "no_results_written"
                            )
                        else:
                            failure_kind = classify_scenario_failure(
                                result,
                                host=str(args.host),
                                world_port=int(args.port),
                                tm_port=int(args.tm_port),
                                carla_manager=carla_manager,
                                progress_summary=progress_summary,
                            )
                        if should_retry_failure(attempt, max_retries, failure_kind):
                            append_retry_event(
                                multi_scenario_retry_log_path,
                                scope="multi_scenario_child",
                                attempt=attempt,
                                retry_limit=max_retries,
                                failure_kind=failure_kind,
                                outcome="retry_queued",
                                details="child_failed_retrying",
                                scenario_name=sub_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                            )
                            detail = _describe_failure_detail(
                                failure_kind,
                                result,
                                carla_manager,
                                stage="evaluator_exit",
                                progress_summary=progress_summary,
                            )
                            _print_failure_banner(
                                f"Scenario {sub_name!r} failed on attempt "
                                f"{format_attempt_counter(attempt, max_retries)} -- will retry.",
                                detail,
                            )
                            delete_result_roots(scenario_result_roots)
                            # GPU rotation: if the failure is CUDA-related, try a better GPU.
                            # Note: in multi-planner mode the child has planner_gpu_requests=[]
                            # because --gpus is stripped from child argv.  Fall back to the GPU
                            # known from CUDA_VISIBLE_DEVICES so Tier-2 system-wide scan fires.
                            current_gpu = runtime_env.get("CUDA_VISIBLE_DEVICES")
                            _gpu_pool = planner_gpu_requests or ([current_gpu] if current_gpu else [])
                            if failure_kind in ("cuda_oom", "cuda_unavailable") and _gpu_pool:
                                fallback_gpu = _pick_fallback_gpu(current_gpu, _gpu_pool)
                                if fallback_gpu is not None and fallback_gpu != current_gpu:
                                    print(
                                        f"[INFO] GPU rotation: switching from GPU {current_gpu!r} "
                                        f"to GPU {fallback_gpu!r} due to {failure_kind}."
                                    )
                                    runtime_env["CUDA_VISIBLE_DEVICES"] = fallback_gpu
                                else:
                                    print(
                                        f"[WARN] GPU rotation: no better GPU found for {failure_kind} "
                                        f"(current={current_gpu!r}); retrying on same GPU."
                                    )
                            if carla_manager:
                                if args.carla_rootcause and failure_kind in ("carla_crash", "carla_unreachable"):
                                    carla_manager.plan_logdir_outcome(f"FAILED_{failure_kind}")
                                    print(
                                        f"[ERROR] CARLA failure ({failure_kind}) detected and --carla-rootcause is enabled. "
                                        "Aborting to preserve logs for analysis."
                                    )
                                    raise SystemExit(1)
                                print(
                                    "[INFO] Restarting CARLA before retrying the scenario "
                                    "to rule out simulator-side failure."
                                )
                                selected_bundle = carla_manager.restart(args.carla_restart_wait, failure_reason=failure_kind)
                                set_effective_ports(
                                    args,
                                    world_port=selected_bundle.port,
                                    tm_port_offset=tm_port_offset,
                                )
                                sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                                wait_for_carla_post_start_buffer(args.carla_post_start_buffer)
                            elif should_continue_retrying(attempt, max_retries):
                                print(
                                    "[WARN] --start-carla is disabled; retrying the scenario without "
                                    "managed CARLA restart."
                                )
                            continue

                        _detail = _describe_failure_detail(
                            failure_kind,
                            result,
                            carla_manager,
                            stage="evaluator_exit",
                            progress_summary=progress_summary,
                        )
                        _print_failure_banner(
                            f"Scenario {sub_name!r} failed with exit code {result.returncode} (no more retries).",
                            _detail,
                        )
                        append_retry_event(
                            multi_scenario_retry_log_path,
                            scope="multi_scenario_child",
                            attempt=attempt,
                            retry_limit=max_retries,
                            failure_kind=failure_kind,
                            outcome="retry_exhausted",
                            details="child_failed_no_more_retries",
                            scenario_name=sub_name,
                            gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                            ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                        )
                        break

                    # FD tracker checkpoint: after scenario execution
                    try:
                        _fdt.checkpoint(f"scenario_{i:03d}_{sub_name}_end")  # type: ignore[union-attr]
                    except Exception:
                        pass
                    if not scenario_completed:
                        append_retry_event(
                            multi_scenario_retry_log_path,
                            scope="multi_scenario_child",
                            attempt=attempt,
                            retry_limit=max_retries,
                            failure_kind="retry_limit_exceeded",
                            outcome="retry_exhausted",
                            details="scenario_not_completed_after_attempts",
                            scenario_name=sub_name,
                            gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                            ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
                        )
                        print(
                            f"[WARNING] Scenario {sub_name} did not complete after "
                            f"{attempt} attempt(s). Continuing to the next scenario."
                        )
            finally:
                if carla_manager:
                    # If no failure outcome was already scheduled, this was a clean session end.
                    if args.carla_rootcause and carla_manager._pending_logdir_outcome is None:
                        carla_manager.plan_logdir_outcome("SUCCESS")
                    carla_manager.stop()
                _active_carla_manager = None
                if colmdriver_vllm_manager is not None:
                    colmdriver_vllm_manager.stop()
                _active_colmdriver_vllm_manager = None

            print(f"\n{'='*60}")
            print(f"Completed all {len(scenario_subfolders)} scenarios")
            print(f"{'='*60}")
            final_phase = (
                "finished"
                if int(completed_scenario_count) >= int(len(scenario_subfolders))
                else "failed"
            )
            update_dashboard_status(
                dashboard_status_path,
                phase=final_phase,
                scenario_mode="multi",
                scenario_total=len(scenario_subfolders),
                scenario_index=len(scenario_subfolders),
                completed_scenarios=completed_scenario_count,
                scenario_name=(
                    scenario_subfolders[-1][1] if scenario_subfolders else ""
                ),
                scenario_results_subdir=(
                    combine_results_subdir(args.results_subdir, scenario_subfolders[-1][1])
                    if scenario_subfolders
                    else str(args.results_subdir or "")
                ),
                current_attempt=0,
                attempt_limit=retry_limit_for_status(args.scenario_retry_limit),
            )
            wall_s = time.monotonic() - run_start_monotonic
            for line in format_performance_summary_lines(wall_clock_s=wall_s):
                print(line)
            write_performance_report(
                multi_scenario_perf_report_path,
                wall_clock_s=wall_s,
                context={
                    "mode": "multi_scenario",
                    "scenario_total": len(scenario_subfolders),
                    "completed_scenarios": completed_scenario_count,
                    "results_root": str(multi_scenario_results_root),
                },
            )
            return

        auto_ego_num = detect_ego_routes(
            routes_dir,
            replay_mode=(args.planner == "log-replay"),
        )
        if auto_ego_num == 0:
            raise RuntimeError(
                f"No ego routes located under {routes_dir}. "
                "Ensure your XML files include role=\"ego\"."
            )
        manifest_path, actors_manifest = read_manifest(routes_dir)

    if auto_ego_num == 0:
        raise RuntimeError(
            f"No ego routes located under {routes_dir}. "
            "At least one role=\"ego\" XML is required as source trajectory input."
        )

    # Determine control mode for fake_ego actors
    # Default to "replay" for --npc-only-fake-ego mode since we want to replay recorded trajectories
    if args.fake_ego_control_mode:
        effective_fake_ego_control_mode = str(args.fake_ego_control_mode).strip().lower()
    elif args.npc_only_fake_ego:
        # When using npc-only-fake-ego, default to replay mode for accurate trajectory following
        effective_fake_ego_control_mode = "replay"
    else:
        effective_fake_ego_control_mode = str(args.custom_actor_control_mode).strip().lower()

    if args.npc_only_fake_ego:
        if args.ego_num not in (None, 0):
            print(
                "[WARN] --npc-only-fake-ego forces --ego-num 0. "
                f"Ignoring explicit --ego-num {args.ego_num}."
            )
        (
            routes_dir,
            fake_ego_temp_dir,
            fake_ego_views,
            manifest_path,
        ) = prepare_npc_only_fake_ego_routes(
            routes_dir=routes_dir,
            fake_ego_control_mode=str(effective_fake_ego_control_mode),
            replay_mode=(args.planner == "log-replay"),
        )
        manifest_path, actors_manifest = read_manifest(routes_dir)
        ego_num = 0
    else:
        ego_num = auto_ego_num if args.ego_num is None else int(args.ego_num)
        if ego_num <= 0:
            raise RuntimeError(
                "--ego-num must be >= 1 unless --npc-only-fake-ego is enabled."
            )

    scenario_name = scenario_name or routes_dir.name
    results_tag = args.results_tag or scenario_name
    if args.results_subdir:
        results_tag = f"{results_tag}/{args.results_subdir}"
    single_results_root = repo_root / "results" / "results_driving_custom" / results_tag
    single_retry_log_path = single_results_root / "_retry_events.jsonl"
    single_perf_report_path: Path | None = None
    if args.perf_report_file:
        perf_candidate = Path(str(args.perf_report_file))
        single_perf_report_path = (
            perf_candidate
            if perf_candidate.is_absolute()
            else single_results_root / perf_candidate
        )
    update_dashboard_status(
        dashboard_status_path,
        phase="running",
        scenario_mode="single",
        scenario_total=1,
        scenario_index=1,
        completed_scenarios=0,
        scenario_name=scenario_name,
        scenario_results_subdir=str(args.results_subdir or ""),
        current_attempt=0,
        attempt_limit=retry_limit_for_status(args.scenario_retry_limit),
    )

    variants: list[VariantSpec] = []
    if args.weather:
        weather_key = args.weather.lower()
        if weather_key == "night":
            weather_override = NIGHT_WEATHER
            desc = "Night-time lighting (sun altitude override)."
        elif weather_key == "overcast":
            weather_override = OVERCAST_WEATHER
            desc = "Overcast daylight (reduced sun + heavy cloud cover)."
        elif weather_key == "cloudy":
            weather_override = CLOUDY_WEATHER
            desc = "Daylight with heavy cloud cover."
        elif weather_key == "clear":
            weather_override = None
            desc = "Default daylight / clear run."
        else:
            weather_override = None
            desc = "Default daylight / clear run."
        variants.append(
            VariantSpec(
                label=weather_key,
                suffix=weather_key,
                description=desc,
                scenario_parameter=args.scenario_parameter,
                weather=weather_override,
            )
        )
    else:
        variants.append(
            VariantSpec(
                label="baseline",
                suffix=None,
                description="Default daylight / clear run.",
                scenario_parameter=args.scenario_parameter,
                weather=None,
            )
        )
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

    leaderboard_root = repo_root / "simulation" / "leaderboard"
    scenario_runner_root = repo_root / "simulation" / "scenario_runner"
    data_root = repo_root / "simulation" / "assets" / "v2xverse_debug"

    env_template = runtime_env.copy()
    env_template["CARLA_ROOT"] = str(carla_root)
    env_template["LEADERBOARD_ROOT"] = str(leaderboard_root)
    env_template["SCENARIO_RUNNER_ROOT"] = str(scenario_runner_root)
    env_template["DATA_ROOT"] = str(data_root)
    env_template.setdefault("COLMDRIVER_OFFLINE", "0")

    append_pythonpath(env_template, repo_root)
    append_pythonpath(env_template, scenario_runner_root)
    append_pythonpath(env_template, leaderboard_root)
    append_pythonpath(env_template, leaderboard_root / "team_code")
    append_pythonpath(env_template, carla_root / "PythonAPI")
    append_pythonpath(env_template, carla_root / "PythonAPI" / "carla")
    append_pythonpath(env_template, find_carla_egg(carla_root))

    tm_port = set_effective_ports(
        args,
        world_port=int(args.port),
        tm_port_offset=tm_port_offset,
    ).tm_port

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
    if args.npc_only_fake_ego:
        print("Mode: npc-only fake-ego (no real egos)")
        print("Fake ego control mode:", str(effective_fake_ego_control_mode))
        if fake_ego_views:
            view_desc = ", ".join(f"{name}->rgb_front_{idx}" for idx, name in fake_ego_views)
            print("Fake ego camera views:", view_desc)
    if scenario_summary is not None:
        print("Actors discovered:")
        for path, role, route_id, town in scenario_summary["routes"]:
            try:
                rel = path.relative_to(routes_dir)
            except ValueError:
                rel = path
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

    carla_manager = None
    crosswalk_worker: UCLAV2CrosswalkWorker | None = None
    if args.start_carla and not args.dry_run:
        _single_rootcause_config = carla_rootcause_config
        if _single_rootcause_config is not None:
            _rc_results_base = (
                repo_root / "results" / "results_driving_custom" / results_tag
            )
            _rc_results_base.mkdir(parents=True, exist_ok=True)
            _single_rootcause_config = _dc_replace(
                _single_rootcause_config, log_root=_rc_results_base,
            )
        carla_manager = CarlaProcessManager(
            carla_root=carla_root,
            host=str(args.host),
            port=args.port,
            tm_port_offset=tm_port_offset,
            extra_args=args.carla_arg,
            env=runtime_env.copy(),
            port_tries=args.carla_port_tries,
            port_step=args.carla_port_step,
            server_log_path=(
                forensics_server_log_path
                if args.carla_forensics and not args.carla_rootcause
                else None
            ),
            rootcause_config=_single_rootcause_config,
        )
        _active_carla_manager = carla_manager
        _install_signal_handlers()
        try:
            selected_bundle = carla_manager.start()
        except Exception as exc:
            append_retry_event(
                single_retry_log_path,
                scope="single_scenario_launcher",
                attempt=1,
                retry_limit=normalize_retry_limit(args.scenario_retry_limit),
                failure_kind="carla_startup_failure",
                outcome="retry_exhausted",
                details=f"{type(exc).__name__}: {exc}",
                scenario_name=scenario_name,
                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                ports=PortBundle(port=int(args.port), tm_port=int(args.tm_port)),
            )
            wall_s = time.monotonic() - run_start_monotonic
            for line in format_performance_summary_lines(wall_clock_s=wall_s):
                print(line)
            write_performance_report(
                single_perf_report_path,
                wall_clock_s=wall_s,
                context={
                    "mode": "single_scenario",
                    "scenario_name": str(scenario_name),
                    "results_root": str(single_results_root),
                    "finalized_in": "carla_start_failure",
                },
            )
            raise
        tm_port = set_effective_ports(
            args,
            world_port=selected_bundle.port,
            tm_port_offset=tm_port_offset,
        ).tm_port
        sync_connection_event_env(runtime_env, carla_manager=carla_manager)
        wait_for_carla_post_start_buffer(args.carla_post_start_buffer)
    elif not args.dry_run:
        if not is_port_open(str(args.host), int(args.port), timeout=1.0):
            raise RuntimeError(
                "External CARLA world port is unreachable while --start-carla is disabled: "
                f"host={args.host} port={int(args.port)}. "
                "Start CARLA on this port or rerun with --start-carla."
            )

    if args.inject_ucla_v2_crosswalk and not args.dry_run:
        crosswalk_worker = UCLAV2CrosswalkWorker(
            repo_root=repo_root,
            carla_root=carla_root,
            host=str(args.host),
            port=int(args.port),
            cache_file=(repo_root / str(args.ucla_v2_crosswalk_cache_file)).resolve(),
            trim_outermost_per_side=int(args.ucla_v2_crosswalk_trim_outermost_per_side),
            high_z_prune_count=int(args.ucla_v2_crosswalk_high_z_prune_count),
            z_mode=str(args.ucla_v2_crosswalk_z_mode),
            poll_sec=float(args.ucla_v2_crosswalk_poll_sec),
        )
        _active_crosswalk_worker = crosswalk_worker
        crosswalk_worker.start()
        print(
            "[INFO] UCLA v2 crosswalk worker enabled: "
            f"z_mode={args.ucla_v2_crosswalk_z_mode} "
            f"trim={int(args.ucla_v2_crosswalk_trim_outermost_per_side)} "
            f"prune={int(args.ucla_v2_crosswalk_high_z_prune_count)} "
            f"cache={args.ucla_v2_crosswalk_cache_file} "
            "(cache + live fallback)"
        )

    def _restart_crosswalk_worker() -> None:
        global _active_crosswalk_worker
        nonlocal crosswalk_worker
        if crosswalk_worker is None:
            return
        crosswalk_worker.stop()
        crosswalk_worker = UCLAV2CrosswalkWorker(
            repo_root=repo_root,
            carla_root=carla_root,
            host=str(args.host),
            port=int(args.port),
            cache_file=(repo_root / str(args.ucla_v2_crosswalk_cache_file)).resolve(),
            trim_outermost_per_side=int(args.ucla_v2_crosswalk_trim_outermost_per_side),
            high_z_prune_count=int(args.ucla_v2_crosswalk_high_z_prune_count),
            z_mode=str(args.ucla_v2_crosswalk_z_mode),
            poll_sec=float(args.ucla_v2_crosswalk_poll_sec),
        )
        _active_crosswalk_worker = crosswalk_worker
        crosswalk_worker.start()

    scenario_result_roots = planned_result_roots(repo_root, results_tag, args)
    variant_timeout_seconds = None
    if not args.dry_run and args.carla_crash_multiplier > 0:
        variant_timeout_seconds = estimate_scenario_timeout(
            args,
            routes_dir,
            planner_name=args.planner,
            run_matrix_size=1,
        )
        print(
            "[INFO] Variant timeout watchdog: "
            f"{variant_timeout_seconds:.1f}s "
            f"(stall quiet window {float(args.scenario_stall_output_timeout):.1f}s)"
        )
    scenario_retry_limit = normalize_retry_limit(args.scenario_retry_limit)
    single_perf_report_written = False

    try:
        scenario_attempt = 0
        scenario_completed = False
        while should_continue_retrying(scenario_attempt, scenario_retry_limit) and not scenario_completed:
            scenario_attempt += 1
            update_dashboard_status(
                dashboard_status_path,
                phase="running",
                scenario_mode="single",
                scenario_total=1,
                scenario_index=1,
                completed_scenarios=0,
                scenario_name=scenario_name,
                scenario_results_subdir=str(args.results_subdir or ""),
                current_attempt=scenario_attempt,
                attempt_limit=scenario_retry_limit,
            )
            retry_scenario = False
            retry_reason = ""
            if scenario_retry_limit is None or scenario_retry_limit > 1:
                print(
                    "[INFO] Scenario attempt "
                    f"{format_attempt_counter(scenario_attempt, scenario_retry_limit)}"
                )
            for spec, negotiation in run_matrix:
                if carla_manager and not carla_manager.is_running():
                    if args.carla_rootcause:
                        _print_failure_banner(
                            f"CARLA process exited unexpectedly before scenario attempt {scenario_attempt}.",
                            "reason=carla_crash, --carla-rootcause enabled -- aborting to preserve logs.",
                        )
                        carla_manager.plan_logdir_outcome("FAILED_carla_crash")
                        raise SystemExit(1)
                    _print_failure_banner(
                        f"CARLA process is not running before attempt {scenario_attempt}.",
                        "reason=carla_crash -- restarting CARLA.",
                    )
                    selected_bundle = carla_manager.restart(args.carla_restart_wait, failure_reason="carla_crash")
                    tm_port = set_effective_ports(
                        args,
                        world_port=selected_bundle.port,
                        tm_port_offset=tm_port_offset,
                    ).tm_port
                    sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                    _restart_crosswalk_worker()
                    wait_for_carla_post_start_buffer(args.carla_post_start_buffer)

                variant_routes_dir, temp_dir = clone_routes_with_weather_override(routes_dir, spec.weather)
                try:
                    if args.align_ego_routes:
                        if temp_dir is None:
                            temp_dir = tempfile.TemporaryDirectory(prefix=f"{routes_dir.name}_align_")
                            clone_root = Path(temp_dir.name) / routes_dir.name
                            shutil.copytree(routes_dir, clone_root)
                            variant_routes_dir = clone_root

                        align_port = args.align_ego_port or args.port
                        town = None
                        for xml_path in variant_routes_dir.glob("*.xml"):
                            try:
                                _, xml_town, role = parse_route_metadata(xml_path.read_bytes())
                            except Exception:
                                continue
                            if role == "ego":
                                town = xml_town
                                break
                        if town:
                            align_ego_routes_in_directory(
                                routes_dir=variant_routes_dir,
                                town=town,
                                carla_host=str(args.align_ego_host),
                                carla_port=int(align_port),
                                sampling_resolution=float(args.align_ego_sampling_resolution),
                            )
                        else:
                            print("[WARN] Ego route alignment requested, but no ego XMLs found to infer town.")

                    sync_connection_event_env(env_template, carla_manager=carla_manager)
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
                        behavior_path = variant_manifest.parent / "actors_behavior.json"
                        if behavior_path.exists():
                            env["CUSTOM_ACTOR_BEHAVIORS"] = str(behavior_path)
                        else:
                            env.pop("CUSTOM_ACTOR_BEHAVIORS", None)
                    else:
                        env.pop("CUSTOM_ACTOR_MANIFEST", None)
                        env.pop("CUSTOM_ACTOR_ROOT", None)
                        env.pop("CUSTOM_ACTOR_BEHAVIORS", None)

                    if args.npc_only_fake_ego and fake_ego_views:
                        env["CUSTOM_FAKE_EGO_CAMERA_NAMES"] = ",".join(
                            name for _, name in fake_ego_views
                        )
                        env["CUSTOM_FAKE_EGO_CAMERA_IDS"] = ",".join(
                            str(idx) for idx, _ in fake_ego_views
                        )
                    else:
                        env.pop("CUSTOM_FAKE_EGO_CAMERA_NAMES", None)
                        env.pop("CUSTOM_FAKE_EGO_CAMERA_IDS", None)

                    if negotiation.disable_comm:
                        env["COLMDRIVER_DISABLE_NEGOTIATION"] = "1"
                    else:
                        env.pop("COLMDRIVER_DISABLE_NEGOTIATION", None)

                    # Set LLM/VLM ports for CoLMDriver
                    env["COLMDRIVER_LLM_PORT"] = str(args.llm_port)
                    env["COLMDRIVER_VLM_PORT"] = str(args.vlm_port)
                    if args.debug:
                        env.setdefault("CUSTOM_LOG_REPLAY_DEBUG", "1")
                        env.setdefault("CUSTOM_LOG_REPLAY_DEBUG_INTERVAL", "2.0")

                    _planner_label = getattr(args, "planner", "") or Path(str(agent_path)).stem
                    suffix_parts: list[str] = []
                    if spec.suffix:
                        suffix_parts.append(spec.suffix)
                    if negotiation.suffix:
                        suffix_parts.append(negotiation.suffix)
                    variant_suffix = "_".join(suffix_parts) if suffix_parts else None
                    variant_results_tag = results_tag if not variant_suffix else f"{results_tag}_{variant_suffix}"
                    result_root = repo_root / "results" / "results_driving_custom" / variant_results_tag
                    checkpoint_endpoint = result_root / "results.json"
                    checkpoint_endpoint.parent.mkdir(parents=True, exist_ok=True)
                    save_path = result_root / "image"
                    save_path.mkdir(parents=True, exist_ok=True)
                    sync_child_evaluator_context_env(
                        env,
                        planner_name=_planner_label,
                        world_port=int(args.port),
                        tm_port=int(tm_port),
                        route_id=str(getattr(spec, "route_id", "") or ""),
                        gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        routes_dir=variant_routes_dir,
                        result_root=result_root,
                        save_path=save_path,
                        checkpoint_endpoint=checkpoint_endpoint,
                    )
                    if args.capture_actor_images:
                        actor_image_dir = result_root / "actor_images"
                        actor_image_dir.mkdir(parents=True, exist_ok=True)
                        env["CUSTOM_ACTOR_IMAGE_DIR"] = str(actor_image_dir)
                    else:
                        env.pop("CUSTOM_ACTOR_IMAGE_DIR", None)
                    if args.normalize_actor_z:
                        env["CUSTOM_ACTOR_NORMALIZE_Z"] = "1"
                    else:
                        env.pop("CUSTOM_ACTOR_NORMALIZE_Z", None)
                    env["CUSTOM_ACTOR_SPAWN_XY_OFFSETS"] = str(args.spawn_retry_xy_offsets)
                    env["CUSTOM_ACTOR_SPAWN_Z_OFFSETS"] = str(args.spawn_retry_z_offsets)
                    env["CUSTOM_ACTOR_SPAWN_MAX_ATTEMPTS"] = str(max(0, int(args.spawn_retry_max_attempts)))
                    if args.image_save_interval is not None:
                        env["TCP_SAVE_INTERVAL"] = str(args.image_save_interval)
                    if args.capture_every_sensor_frame:
                        env["TCP_CAPTURE_SENSOR_FRAMES"] = "1"
                    else:
                        env.pop("TCP_CAPTURE_SENSOR_FRAMES", None)
                    if args.capture_logreplay_images:
                        env["TCP_CAPTURE_LOGREPLAY_IMAGES"] = "1"
                    else:
                        env.pop("TCP_CAPTURE_LOGREPLAY_IMAGES", None)
                    if args.grp_postprocess_mode is not None:
                        mode = str(args.grp_postprocess_mode).strip().lower()
                        if mode == "none":
                            env["CARLA_GRP_PP_ENABLE"] = "0"
                        else:
                            env["CARLA_GRP_PP_ENABLE"] = "1"
                            env["CARLA_GRP_PP_MODE"] = mode
                    if args.grp_postprocess_ignore_endpoints is not None:
                        env["CARLA_GRP_PP_IGNORE_ENDPOINTS"] = (
                            "1" if args.grp_postprocess_ignore_endpoints else "0"
                        )
                    if args.disable_stall_detection:
                        env["CUSTOM_DISABLE_STALL_DETECTION"] = "1"
                    else:
                        env.pop("CUSTOM_DISABLE_STALL_DETECTION", None)
                    overhead_env_keys = [
                        "CUSTOM_OVERHEAD_CAPTURE_PER_TICK",
                        "CUSTOM_OVERHEAD_SAVE_EVERY_N",
                        "CUSTOM_OVERHEAD_DRAW_GRP_BOXES",
                        "CUSTOM_OVERHEAD_OUTPUT_SUBDIR",
                        "CUSTOM_OVERHEAD_IMAGE_WIDTH",
                        "CUSTOM_OVERHEAD_IMAGE_HEIGHT",
                        "CUSTOM_OVERHEAD_IMAGE_FOV",
                        "CUSTOM_OVERHEAD_IMAGE_MARGIN_M",
                        "CUSTOM_OVERHEAD_IMAGE_Z_PADDING",
                        "CUSTOM_OVERHEAD_MIN_CAMERA_Z",
                        "CUSTOM_OVERHEAD_BOX_SIZE_XY",
                        "CUSTOM_OVERHEAD_BOX_HEIGHT",
                        "CUSTOM_OVERHEAD_BOX_Z_OFFSET",
                        "CUSTOM_OVERHEAD_BOX_THICKNESS",
                        "CUSTOM_OVERHEAD_BOX_LIFE_TIME",
                        "CUSTOM_OVERHEAD_BOX_COLOR_REGULAR_RGB",
                        "CUSTOM_OVERHEAD_BOX_COLOR_CORRECTED_RGB",
                        "CUSTOM_OVERHEAD_BOX_COLOR_SANITIZED_RGB",
                    ]
                    for key in overhead_env_keys:
                        env.pop(key, None)
                    if args.rgb_width is not None:
                        env["TCP_RGB_WIDTH"] = str(args.rgb_width)
                    if args.rgb_height is not None:
                        env["TCP_RGB_HEIGHT"] = str(args.rgb_height)
                    if args.rgb_fov is not None:
                        env["TCP_RGB_FOV"] = str(args.rgb_fov)
                    if args.rgb_x is not None:
                        env["TCP_RGB_X"] = str(args.rgb_x)
                    if args.rgb_y is not None:
                        env["TCP_RGB_Y"] = str(args.rgb_y)
                    if args.rgb_z is not None:
                        env["TCP_RGB_Z"] = str(args.rgb_z)
                    if args.rgb_roll is not None:
                        env["TCP_RGB_ROLL"] = str(args.rgb_roll)
                    if args.rgb_pitch is not None:
                        env["TCP_RGB_PITCH"] = str(args.rgb_pitch)
                    if args.rgb_yaw is not None:
                        env["TCP_RGB_YAW"] = str(args.rgb_yaw)
                    if args.logreplay_rgb_width is not None:
                        env["TCP_LOGREPLAY_RGB_WIDTH"] = str(args.logreplay_rgb_width)
                    if args.logreplay_rgb_height is not None:
                        env["TCP_LOGREPLAY_RGB_HEIGHT"] = str(args.logreplay_rgb_height)
                    if args.logreplay_rgb_fov is not None:
                        env["TCP_LOGREPLAY_RGB_FOV"] = str(args.logreplay_rgb_fov)
                    if args.logreplay_rgb_x is not None:
                        env["TCP_LOGREPLAY_RGB_X"] = str(args.logreplay_rgb_x)
                    if args.logreplay_rgb_y is not None:
                        env["TCP_LOGREPLAY_RGB_Y"] = str(args.logreplay_rgb_y)
                    if args.logreplay_rgb_z is not None:
                        env["TCP_LOGREPLAY_RGB_Z"] = str(args.logreplay_rgb_z)
                    if args.logreplay_rgb_roll is not None:
                        env["TCP_LOGREPLAY_RGB_ROLL"] = str(args.logreplay_rgb_roll)
                    if args.logreplay_rgb_pitch is not None:
                        env["TCP_LOGREPLAY_RGB_PITCH"] = str(args.logreplay_rgb_pitch)
                    if args.logreplay_rgb_yaw is not None:
                        env["TCP_LOGREPLAY_RGB_YAW"] = str(args.logreplay_rgb_yaw)
                    if args.bev_width is not None:
                        env["TCP_BEV_WIDTH"] = str(args.bev_width)
                    if args.bev_height is not None:
                        env["TCP_BEV_HEIGHT"] = str(args.bev_height)
                    if args.bev_fov is not None:
                        env["TCP_BEV_FOV"] = str(args.bev_fov)
                    actor_replay_enabled = bool(
                        args.log_replay_actors or args.custom_actor_control_mode == "replay"
                    )
                    if args.planner == "log-replay":
                        env["CUSTOM_EGO_LOG_REPLAY"] = "1"
                    else:
                        env.pop("CUSTOM_EGO_LOG_REPLAY", None)
                    if actor_replay_enabled:
                        env["CUSTOM_ACTOR_LOG_REPLAY"] = "1"
                    else:
                        env.pop("CUSTOM_ACTOR_LOG_REPLAY", None)
                    # When custom actors are explicitly in replay mode, preserve authored
                    # XML pose exactly and avoid per-tick runtime ground pose rewrites.
                    if str(args.custom_actor_control_mode).strip().lower() == "replay":
                        env["CUSTOM_ACTOR_REPLAY_DISABLE_RUNTIME_GROUND_POSE"] = "1"
                    else:
                        env.pop("CUSTOM_ACTOR_REPLAY_DISABLE_RUNTIME_GROUND_POSE", None)
                    if args.smooth_log_replay_vehicles is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_VEHICLES"] = (
                            "1" if args.smooth_log_replay_vehicles else "0"
                        )
                    if args.log_replay_smooth_min_cutoff is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_MIN_CUTOFF"] = str(
                            float(args.log_replay_smooth_min_cutoff)
                        )
                    if args.log_replay_smooth_beta is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_BETA"] = str(float(args.log_replay_smooth_beta))
                    if args.log_replay_smooth_yaw_min_cutoff is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_YAW_MIN_CUTOFF"] = str(
                            float(args.log_replay_smooth_yaw_min_cutoff)
                        )
                    if args.log_replay_smooth_yaw_beta is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_YAW_BETA"] = str(float(args.log_replay_smooth_yaw_beta))
                    if args.log_replay_smooth_z_min_cutoff is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_Z_MIN_CUTOFF"] = str(
                            float(args.log_replay_smooth_z_min_cutoff)
                        )
                    if args.log_replay_smooth_z_beta is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_Z_BETA"] = str(float(args.log_replay_smooth_z_beta))
                    if args.log_replay_smooth_d_cutoff is not None:
                        env["CUSTOM_LOG_REPLAY_SMOOTH_D_CUTOFF"] = str(float(args.log_replay_smooth_d_cutoff))
                    if args.log_replay_lateral_damping is not None:
                        env["CUSTOM_LOG_REPLAY_LATERAL_DAMPING"] = str(
                            min(1.0, max(0.0, float(args.log_replay_lateral_damping)))
                        )
                    if args.log_replay_lateral_passes is not None:
                        env["CUSTOM_LOG_REPLAY_LATERAL_PASSES"] = str(
                            max(1, int(args.log_replay_lateral_passes))
                        )
                    if args.log_replay_lateral_max_correction is not None:
                        env["CUSTOM_LOG_REPLAY_LATERAL_MAX_CORRECTION"] = str(
                            max(0.0, float(args.log_replay_lateral_max_correction))
                        )
                    if args.log_replay_lateral_turn_keep is not None:
                        env["CUSTOM_LOG_REPLAY_LATERAL_TURN_KEEP"] = str(
                            min(1.0, max(0.0, float(args.log_replay_lateral_turn_keep)))
                        )
                    if args.log_replay_lateral_turn_angle_deg is not None:
                        env["CUSTOM_LOG_REPLAY_LATERAL_TURN_ANGLE_DEG"] = str(
                            max(0.0, float(args.log_replay_lateral_turn_angle_deg))
                        )
                    if args.log_replay_time_lead is not None:
                        env["CUSTOM_LOG_REPLAY_TIME_LEAD"] = str(
                            min(0.25, max(0.0, float(args.log_replay_time_lead)))
                        )
                    if args.stabilize_static_log_replay_vehicles is not None:
                        env["CUSTOM_LOG_REPLAY_STABILIZE_STATIC_VEHICLES"] = (
                            "1" if args.stabilize_static_log_replay_vehicles else "0"
                        )
                    if args.static_log_replay_total_disp is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_TOTAL_DISP"] = str(
                            max(0.0, float(args.static_log_replay_total_disp))
                        )
                    if args.static_log_replay_total_path is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_TOTAL_PATH"] = str(
                            max(0.0, float(args.static_log_replay_total_path))
                        )
                    if args.static_log_replay_window_min_duration is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MIN_DURATION"] = str(
                            max(0.0, float(args.static_log_replay_window_min_duration))
                        )
                    if args.static_log_replay_window_max_disp is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_DISP"] = str(
                            max(0.0, float(args.static_log_replay_window_max_disp))
                        )
                    if args.static_log_replay_window_max_speed is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_SPEED"] = str(
                            max(0.0, float(args.static_log_replay_window_max_speed))
                        )
                    if args.static_log_replay_window_max_yaw_delta is not None:
                        env["CUSTOM_LOG_REPLAY_STATIC_WINDOW_MAX_YAW_DELTA"] = str(
                            max(0.0, float(args.static_log_replay_window_max_yaw_delta))
                        )
                    if args.animate_walkers is not None:
                        env["CUSTOM_LOG_REPLAY_ANIMATE_WALKERS"] = "1" if args.animate_walkers else "0"
                    if args.walker_max_speed is not None:
                        env["CUSTOM_LOG_REPLAY_WALKER_MAX_SPEED"] = str(max(0.1, float(args.walker_max_speed)))
                    if args.walker_teleport_distance is not None:
                        env["CUSTOM_LOG_REPLAY_WALKER_TELEPORT_DIST"] = str(
                            max(0.1, float(args.walker_teleport_distance))
                        )
                    if args.walker_ground_lift is not None:
                        env["CUSTOM_WALKER_GROUND_LIFT"] = str(float(args.walker_ground_lift))
                    if args.normalize_ego_z:
                        env["CUSTOM_EGO_NORMALIZE_Z"] = "1"
                    else:
                        env.pop("CUSTOM_EGO_NORMALIZE_Z", None)
                    # Always enable per-tick vehicle tilt alignment so vehicles
                    # pitch/roll match the road surface (4-corner ground raycast).
                    env["CUSTOM_EGO_GROUND_ALIGN_TILT"] = "1"
                    effective_recovery_mode = str(args.recovery_mode or "off").strip().lower()
                    if args.disable_checkpoint_recovery:
                        effective_recovery_mode = "off"
                    env["CARLA_RECOVERY_MODE"] = effective_recovery_mode
                    env["CARLA_CHECKPOINT_RECOVERY_ENABLE"] = (
                        "0" if effective_recovery_mode == "off" else "1"
                    )
                    env["CARLA_RECOVERY_STRICT_RECORD_ACCOUNTING"] = (
                        "1" if effective_recovery_mode == "publication_safe" else "0"
                    )
                    env["CARLA_CHECKPOINT_INTERVAL_FRAMES"] = str(
                        max(1, int(args.checkpoint_interval_frames))
                    )
                    env["CARLA_CHECKPOINT_MAX_RECOVERIES"] = str(
                        max(0, int(args.checkpoint_max_recoveries))
                    )
                    env["CARLA_CHECKPOINT_MAX_COUNT"] = str(
                        max(0, int(args.checkpoint_max_count))
                    )
                    env["CARLA_CHECKPOINT_NEARBY_RADIUS_M"] = str(
                        max(5.0, float(args.checkpoint_nearby_radius_m))
                    )
                    env["CARLA_CHECKPOINT_TL_RADIUS_M"] = str(
                        max(5.0, float(args.checkpoint_tl_radius_m))
                    )
                    env["CARLA_CHECKPOINT_ACTOR_MATCH_M"] = str(
                        max(1.0, float(args.checkpoint_actor_match_m))
                    )
                    env["CARLA_PLANNER_REPLAY_WINDOW"] = str(
                        max(1, int(args.planner_replay_window))
                    )
                    if args.checkpoint_restart_cmd:
                        env["CARLA_RECOVERY_RESTART_CMD"] = str(args.checkpoint_restart_cmd)
                    else:
                        env.pop("CARLA_RECOVERY_RESTART_CMD", None)

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
                        "--host",
                        str(args.host),
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
                    if args.route_plots_only:
                        cmd.append("--route-plots-only")

                    variant_name = spec.suffix or "baseline"
                    if negotiation.disable_comm:
                        variant_name = f"{variant_name} (no negotiation)"
                    print(f"\n=== Running variant: {variant_name} ===")
                    print("Scenario parameter:", scenario_parameter_path)
                    print(
                        "Custom actor control mode:",
                        "replay" if actor_replay_enabled else "policy",
                    )
                    if spec.weather:
                        print("Weather override applied via:", variant_routes_dir)
                    print("Results will be stored under:", result_root)
                    print("Image saving:", save_path)
                    print("Command:")
                    print("  " + " ".join(cmd))

                    if args.dry_run:
                        continue

                    if args.carla_forensics:
                        forensic_env_keys = [
                            "CARLA_STREAM_DEBUG",
                            "CARLA_STREAM_DEBUG_SUMMARY_EVERY",
                            "MALLOC_CHECK_",
                            "MALLOC_PERTURB_",
                        ]
                        forensic_payload = {
                            "carla_forensics": True,
                            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                            "start_carla": bool(args.start_carla),
                            "carla_args": list(args.carla_arg),
                            "forensics_log_file": str(forensics_log_path) if forensics_log_path else "",
                            "forensics_server_log_file": (
                                str(forensics_server_log_path) if forensics_server_log_path else ""
                            ),
                            "command": cmd,
                            "env": {k: env.get(k, "") for k in forensic_env_keys},
                        }
                        try:
                            (result_root / "carla_forensics_config.json").write_text(
                                json.dumps(forensic_payload, indent=2),
                                encoding="utf-8",
                            )
                        except Exception as exc:  # pylint: disable=broad-except
                            print(f"[WARN] Failed to write carla_forensics_config.json: {exc}")

                    runner_ref: list[MonitoredSubprocessRunner | None] = [None]
                    _tick_hb_path = result_root / "tick_heartbeat"
                    env["CARLA_TICK_HEARTBEAT_FILE"] = str(_tick_hb_path)
                    child_recovery_enabled = str(
                        env.get("CARLA_CHECKPOINT_RECOVERY_ENABLE", "0")
                    ).strip().lower() in ("1", "true", "yes", "on")
                    carla_down_checker = None
                    if (
                        not child_recovery_enabled
                        or args.launcher_carladown_abort_with_child_recovery
                    ):
                        carla_down_checker = build_carla_down_abort_checker(
                            host=str(args.host),
                            world_port=int(args.port),
                            carla_manager=carla_manager,
                        )
                    health_check = combine_health_checks(
                        carla_down_checker,
                        build_startup_stall_abort_checker(
                            result_root=result_root,
                            ego_count=ego_num,
                            runner_provider=lambda: runner_ref[0],
                            quiet_timeout_s=float(args.scenario_startup_quiet_timeout),
                        ),
                        build_tick_heartbeat_checker(_tick_hb_path),
                    )
                    runner = MonitoredSubprocessRunner(
                        cmd,
                        env=env,
                        cwd=repo_root,
                        timeout_s=variant_timeout_seconds,
                        stall_output_timeout_s=float(args.scenario_stall_output_timeout),
                        output_prefix=f"[variant:{sanitize_run_id(variant_name)}] ",
                        show_carla_diag=bool(args.show_carla_diag),
                        health_check=health_check,
                    )
                    runner_ref[0] = runner
                    gate_ready = wait_for_carla_teardown_gate(
                        host=str(args.host),
                        world_port=int(args.port),
                        tm_port=int(tm_port),
                        context=(
                            f"variant_launch:{variant_name}:"
                            f"attempt={scenario_attempt}/{scenario_retry_limit if scenario_retry_limit is not None else 'inf'}"
                        ),
                    )
                    if not gate_ready:
                        retry_scenario = True
                        retry_reason = "teardown_gate_timeout"
                        print(
                            "[WARN] Evaluator teardown gate timed out before variant launch; "
                            "marking scenario for retry."
                        )
                        break
                    result = runner.run()
                    if result.returncode == 0:
                        soft_failure_kind = classify_retryable_soft_failure(
                            result,
                            result_root=result_root,
                            ego_count=ego_num,
                            host=str(args.host),
                            world_port=int(args.port),
                            tm_port=int(tm_port),
                            carla_manager=carla_manager,
                        )
                        if soft_failure_kind is not None:
                            progress_summary = collect_result_root_progress(
                                result_root,
                                ego_count=ego_num,
                            )
                            if should_retry_failure(
                                scenario_attempt,
                                scenario_retry_limit,
                                soft_failure_kind,
                            ):
                                retry_scenario = True
                                retry_reason = soft_failure_kind
                                break
                            _print_failure_banner(
                                f"Scenario {scenario_name!r} failed (soft failure, non-retryable).",
                                _describe_failure_detail(
                                    soft_failure_kind,
                                    result,
                                    carla_manager,
                                    stage="soft_failure",
                                    progress_summary=progress_summary,
                                ),
                            )
                            if carla_manager and args.carla_rootcause:
                                carla_manager.plan_logdir_outcome(f"FAILED_{soft_failure_kind}")
                            update_dashboard_status(
                                dashboard_status_path,
                                phase="failed",
                                scenario_mode="single",
                                scenario_total=1,
                                scenario_index=1,
                                completed_scenarios=0,
                                scenario_name=scenario_name,
                                scenario_results_subdir=str(args.results_subdir or ""),
                                current_attempt=scenario_attempt,
                                attempt_limit=scenario_retry_limit,
                            )
                            append_retry_event(
                                single_retry_log_path,
                                scope="single_scenario",
                                attempt=scenario_attempt,
                                retry_limit=scenario_retry_limit,
                                failure_kind=soft_failure_kind,
                                outcome="retry_exhausted",
                                details="soft_failure_non_retryable",
                                scenario_name=scenario_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
                            )
                            raise RuntimeError(
                                "Scenario failed after evaluator exit "
                                f"(reason={soft_failure_kind})."
                            )
                        if (
                            not args.route_plots_only
                            and not result_root_has_terminal_outcomes(
                                result_root,
                                ego_count=ego_num,
                            )
                        ):
                            progress_summary = collect_result_root_progress(
                                result_root,
                                ego_count=ego_num,
                            )
                            missing_results_kind = classify_missing_terminal_results(progress_summary)
                            if should_retry_failure(
                                scenario_attempt,
                                scenario_retry_limit,
                                missing_results_kind,
                            ):
                                retry_scenario = True
                                retry_reason = missing_results_kind
                                break
                            _print_failure_banner(
                                (
                                    f"Scenario {scenario_name!r} exited cleanly but produced no terminal route "
                                    f"outcomes (stage=result_validation, classification={missing_results_kind})."
                                ),
                                _describe_failure_detail(
                                    missing_results_kind,
                                    result,
                                    carla_manager,
                                    stage="result_validation",
                                    progress_summary=progress_summary,
                                ),
                            )
                            update_dashboard_status(
                                dashboard_status_path,
                                phase="failed",
                                scenario_mode="single",
                                scenario_total=1,
                                scenario_index=1,
                                completed_scenarios=0,
                                scenario_name=scenario_name,
                                scenario_results_subdir=str(args.results_subdir or ""),
                                current_attempt=scenario_attempt,
                                attempt_limit=scenario_retry_limit,
                            )
                            detail_reason = missing_results_kind
                            append_retry_event(
                                single_retry_log_path,
                                scope="single_scenario",
                                attempt=scenario_attempt,
                                retry_limit=scenario_retry_limit,
                                failure_kind=detail_reason,
                                outcome="retry_exhausted",
                                details="missing_terminal_outcomes_non_retryable",
                                scenario_name=scenario_name,
                                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                                ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
                            )
                            raise RuntimeError(
                                "Scenario exited without terminal route outcomes "
                                f"(reason={detail_reason})."
                            )
                    if result.returncode != 0:
                        progress_summary = collect_result_root_progress(
                            result_root,
                            ego_count=ego_num,
                        )
                        if has_only_terminal_route_outcomes(progress_summary):
                            print(
                                "[WARN] Variant "
                                f"{variant_name} wrote terminal route outcomes despite "
                                f"exit_code={result.returncode}; accepting results."
                            )
                            continue
                        if args.accept_stalled_near_end_failures:
                            accept_stalled, accept_detail = should_accept_stalled_near_end_failure(
                                result=result,
                                result_root=result_root,
                                summary=progress_summary,
                                min_route_score=float(args.stalled_near_end_min_route_score),
                                require_stall_evidence=bool(args.stalled_near_end_require_stall_evidence),
                            )
                            if accept_stalled:
                                print(
                                    "[WARN] Variant "
                                    f"{variant_name} exited non-zero but matches "
                                    f"stalled-near-end acceptance policy; keeping results ({accept_detail})."
                                )
                                if carla_manager and args.carla_rootcause:
                                    carla_manager.plan_logdir_outcome("ACCEPTED_stalled_near_end")
                                continue
                        failure_kind = classify_scenario_failure(
                            result,
                            host=str(args.host),
                            world_port=int(args.port),
                            tm_port=int(tm_port),
                            carla_manager=carla_manager,
                            progress_summary=progress_summary,
                        )
                        if should_retry_failure(
                            scenario_attempt,
                            scenario_retry_limit,
                            failure_kind,
                        ):
                            retry_scenario = True
                            retry_reason = failure_kind
                            break
                        _print_failure_banner(
                            f"Scenario {scenario_name!r} evaluator exited non-zero (non-retryable).",
                            _describe_failure_detail(
                                failure_kind,
                                result,
                                carla_manager,
                                stage="evaluator_exit",
                                progress_summary=progress_summary,
                            ),
                        )
                        if carla_manager and args.carla_rootcause:
                            carla_manager.plan_logdir_outcome(f"FAILED_{failure_kind}")
                        update_dashboard_status(
                            dashboard_status_path,
                            phase="failed",
                            scenario_mode="single",
                            scenario_total=1,
                            scenario_index=1,
                            completed_scenarios=0,
                            scenario_name=scenario_name,
                            scenario_results_subdir=str(args.results_subdir or ""),
                            current_attempt=scenario_attempt,
                            attempt_limit=scenario_retry_limit,
                        )
                        append_retry_event(
                            single_retry_log_path,
                            scope="single_scenario",
                            attempt=scenario_attempt,
                            retry_limit=scenario_retry_limit,
                            failure_kind=failure_kind,
                            outcome="retry_exhausted",
                            details=f"evaluator_exit_non_retryable_rc={result.returncode}",
                            scenario_name=scenario_name,
                            gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                            ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
                        )
                        raise RuntimeError(
                            "Scenario evaluator exited nonzero "
                            f"(reason={failure_kind}, returncode={result.returncode})."
                        )
                    print(
                        f"[INFO] Variant {variant_name} completed in {result.elapsed_s:.1f}s."
                    )
                finally:
                    if temp_dir is not None:
                        temp_dir.cleanup()

            if retry_scenario:
                append_retry_event(
                    single_retry_log_path,
                    scope="single_scenario",
                    attempt=scenario_attempt,
                    retry_limit=scenario_retry_limit,
                    failure_kind=retry_reason or "generic",
                    outcome="retry_queued",
                    details="scenario_retry_requested",
                    scenario_name=scenario_name,
                    gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                    ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
                )
                _print_failure_banner(
                    f"Scenario {scenario_name!r} failed -- "
                    f"attempt {format_attempt_counter(scenario_attempt, scenario_retry_limit)}, "
                    f"reason={retry_reason}.",
                    "Cleaning result folders and restarting.",
                )
                delete_result_roots(scenario_result_roots)
                # GPU rotation: if the failure is CUDA-related, try a better GPU.
                # Note: in multi-planner mode the child has planner_gpu_requests=[] because
                # --gpus is stripped from child argv.  Fall back to the GPU known from
                # CUDA_VISIBLE_DEVICES so Tier-2 system-wide scan fires.
                current_gpu = runtime_env.get("CUDA_VISIBLE_DEVICES")
                _gpu_pool = planner_gpu_requests or ([current_gpu] if current_gpu else [])
                if retry_reason in ("cuda_oom", "cuda_unavailable") and _gpu_pool:
                    fallback_gpu = _pick_fallback_gpu(current_gpu, _gpu_pool)
                    if fallback_gpu is not None and fallback_gpu != current_gpu:
                        print(
                            f"[INFO] GPU rotation: switching from GPU {current_gpu!r} "
                            f"to GPU {fallback_gpu!r} due to {retry_reason}."
                        )
                        runtime_env["CUDA_VISIBLE_DEVICES"] = fallback_gpu
                    else:
                        print(
                            f"[WARN] GPU rotation: no better GPU found for {retry_reason} "
                            f"(current={current_gpu!r}); retrying on same GPU."
                        )
                if carla_manager:
                    if args.carla_rootcause and retry_reason in ("carla_crash", "carla_unreachable"):
                        carla_manager.plan_logdir_outcome(f"FAILED_{retry_reason}")
                        print(
                            f"[ERROR] CARLA failure ({retry_reason}) detected and --carla-rootcause is enabled. "
                            "Aborting to preserve logs for analysis."
                        )
                        raise SystemExit(1)
                    print(
                        "[INFO] Restarting CARLA before retrying the scenario "
                        "to rule out simulator-side failure."
                    )
                    selected_bundle = carla_manager.restart(args.carla_restart_wait, failure_reason=retry_reason)
                    tm_port = set_effective_ports(
                        args,
                        world_port=selected_bundle.port,
                        tm_port_offset=tm_port_offset,
                    ).tm_port
                    sync_connection_event_env(runtime_env, carla_manager=carla_manager)
                    _restart_crosswalk_worker()
                    wait_for_carla_post_start_buffer(args.carla_post_start_buffer)
                elif should_continue_retrying(scenario_attempt, scenario_retry_limit):
                    print(
                        "[WARN] --start-carla is disabled; retrying the scenario without "
                        "managed CARLA restart."
                    )
                continue

            scenario_completed = True

        if not scenario_completed:
            update_dashboard_status(
                dashboard_status_path,
                phase="failed",
                scenario_mode="single",
                scenario_total=1,
                scenario_index=1,
                completed_scenarios=0,
                scenario_name=scenario_name,
                scenario_results_subdir=str(args.results_subdir or ""),
                current_attempt=scenario_attempt,
                attempt_limit=scenario_retry_limit,
            )
            append_retry_event(
                single_retry_log_path,
                scope="single_scenario",
                attempt=scenario_attempt,
                retry_limit=scenario_retry_limit,
                failure_kind="retry_limit_exceeded",
                outcome="retry_exhausted",
                details="scenario_retry_limit_exceeded",
                scenario_name=scenario_name,
                gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
                ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
            )
            retry_limit_label = (
                str(scenario_retry_limit)
                if scenario_retry_limit is not None
                else "unlimited"
            )
            raise RuntimeError(
                f"Scenario '{results_tag}' exceeded the retry limit ({retry_limit_label})."
            )
        append_retry_event(
            single_retry_log_path,
            scope="single_scenario",
            attempt=scenario_attempt,
            retry_limit=scenario_retry_limit,
            failure_kind="none",
            outcome="success",
            details="scenario_completed",
            scenario_name=scenario_name,
            gpu=runtime_env.get("CUDA_VISIBLE_DEVICES"),
            ports=PortBundle(port=int(args.port), tm_port=int(tm_port)),
        )
        update_dashboard_status(
            dashboard_status_path,
            phase="finished",
            scenario_mode="single",
            scenario_total=1,
            scenario_index=1,
            completed_scenarios=1,
            scenario_name=scenario_name,
            scenario_results_subdir=str(args.results_subdir or ""),
            current_attempt=scenario_attempt,
            attempt_limit=scenario_retry_limit,
        )
        wall_s = time.monotonic() - run_start_monotonic
        for line in format_performance_summary_lines(wall_clock_s=wall_s):
            print(line)
        write_performance_report(
            single_perf_report_path,
            wall_clock_s=wall_s,
            context={
                "mode": "single_scenario",
                "scenario_name": str(scenario_name),
                "results_root": str(single_results_root),
            },
        )
        single_perf_report_written = True
    finally:
        if crosswalk_worker:
            crosswalk_worker.stop()
        _active_crosswalk_worker = None
        if carla_manager:
            # If no failure outcome was already scheduled, annotate as SUCCESS (clean session end).
            if args.carla_rootcause and carla_manager._pending_logdir_outcome is None:
                carla_manager.plan_logdir_outcome("SUCCESS")
            carla_manager.stop()
        _active_carla_manager = None
        if colmdriver_vllm_manager is not None:
            colmdriver_vllm_manager.stop()
        _active_colmdriver_vllm_manager = None
        if fake_ego_temp_dir is not None:
            fake_ego_temp_dir.cleanup()
        if not single_perf_report_written:
            wall_s = time.monotonic() - run_start_monotonic
            for line in format_performance_summary_lines(wall_clock_s=wall_s):
                print(line)
            write_performance_report(
                single_perf_report_path,
                wall_clock_s=wall_s,
                context={
                    "mode": "single_scenario",
                    "scenario_name": str(scenario_name),
                    "results_root": str(single_results_root),
                    "finalized_in": "finally",
                },
            )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_process_exception(exc, process_name=RUN_CUSTOM_EVAL_PROCESS_NAME, where="__main__")
        raise
