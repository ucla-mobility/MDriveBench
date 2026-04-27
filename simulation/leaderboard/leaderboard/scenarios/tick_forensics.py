"""
Comprehensive tick-rate and watchdog forensics for scenario_manager.

Designed to give root-cause clarity on the three watchdog-triggered failures:
  - reason=stalled               (parent stdout watchdog, 15s silence)
  - reason=no_route_execution    (parent tick-heartbeat watchdog, 300s)
  - reason=carla_unreachable     (parent CARLA RPC health-check)

Plus distinguishing "watchdog false positive" (slow but progressing) from
"true freeze" (nothing happening) from "CARLA-side hang" (server unresponsive).

Output: every line is prefixed with `[TICK_FORENSICS]` so it greps cleanly out
of the noisy planner logs. All probes are env-gated; default behavior is
unchanged.

Env vars:
  CARLA_TICK_FORENSICS=1                 master switch (required for any output)
  CARLA_TICK_FORENSICS_SNAPSHOT_S=30     periodic state snapshot interval
  CARLA_TICK_FORENSICS_SLOW_TICK_S=1.0   per-tick "slow" threshold (log SLOW_TICK)
  CARLA_RPC_PROBE_INTERVAL_S=5           CARLA RPC liveness probe interval
  CARLA_RPC_PROBE_TIMEOUT_S=10           per-probe RPC timeout
  CARLA_TICK_FORENSICS_TRACE_EVERY=0     log EVERY tick's phases (heavy; 0=off)

Event types emitted (all under [TICK_FORENSICS] prefix):
  INIT                  forensics started, includes config
  TICK                  per-tick phase breakdown (only if trace_every > 0)
  SLOW_TICK             a single tick exceeded the slow threshold
  SNAPSHOT              periodic rolling state (last 20 ticks aggregate)
  CARLA_RPC_OK          successful RPC probe (only if slow)
  CARLA_RPC_SLOW        RPC probe RTT exceeded threshold
  CARLA_RPC_FAILED      RPC probe raised an exception
  STALL_WARNING_60S     ticks haven't completed in 60s (early warning)
  STALL_WARNING_120S    ticks haven't completed in 120s (mid warning)
  STALL_APPROACHING_KILL ticks haven't completed in ~270s (pre-kill warning)
  STALL_THREAD_DUMP     stack trace of every Python thread at stall time
  TICK_RECOVERED        ticks resumed after a stall warning
  STOP                  forensics stopped
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from collections import deque
from typing import Optional

_LOG_PREFIX = "[TICK_FORENSICS]"

# Import-time env evaluation. Cheap and avoids per-tick env lookups.
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if not v:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if not v:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _emit(event: str, instance_id, **fields) -> None:
    """Emit a structured forensics line. instance_id is included on every line."""
    parts = [f"instance={instance_id}"]
    for k, v in fields.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        elif isinstance(v, str) and (" " in v or "=" in v):
            # Avoid breaking the grep-friendly key=value format
            v_clean = v.replace("\n", "\\n")[:500]
            parts.append(f"{k}={v_clean!r}")
        else:
            parts.append(f"{k}={v}")
    print(f"{_LOG_PREFIX} {event} " + " ".join(parts), flush=True)


class CarlaRpcProbe:
    """
    Daemon thread that periodically calls a trivial CARLA RPC and times it.

    Critical for distinguishing CARLA-side hang from Python-side hang. If the
    tick loop is blocked but RPC probes succeed, CARLA is fine and the
    blockage is in the agent or scenario_manager. If RPC probes also fail,
    CARLA itself is hung (e.g., server-side render queue saturation).
    """

    def __init__(self, world, *, instance_id, interval_s=5.0, timeout_s=10.0, slow_rtt_ms=1000):
        self._world = world
        self._instance_id = instance_id
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._slow_rtt_ms = slow_rtt_ms
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Rolling stats (lock-protected)
        self._lock = threading.Lock()
        self._stats = {
            "probes_total": 0,
            "probes_ok": 0,
            "probes_slow": 0,
            "probes_failed": 0,
            "consecutive_failures": 0,
            "last_rtt_ms": None,
            "last_probe_at": None,
            "last_failure_at": None,
            "last_failure_msg": None,
            "max_rtt_ms": 0.0,
        }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"carla_rpc_probe_{self._instance_id}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            t0 = time.time()
            try:
                # get_settings is the cheapest RPC that bypasses the tick
                # synchronization machinery -- it goes straight to the server's
                # episode settings cache. If THIS hangs, CARLA itself is dead.
                _ = self._world.get_settings()
                rtt_ms = (time.time() - t0) * 1000.0
                with self._lock:
                    self._stats["probes_total"] += 1
                    self._stats["probes_ok"] += 1
                    self._stats["consecutive_failures"] = 0
                    self._stats["last_rtt_ms"] = rtt_ms
                    self._stats["last_probe_at"] = time.time()
                    if rtt_ms > self._stats["max_rtt_ms"]:
                        self._stats["max_rtt_ms"] = rtt_ms
                    if rtt_ms >= self._slow_rtt_ms:
                        self._stats["probes_slow"] += 1
                if rtt_ms >= self._slow_rtt_ms:
                    _emit("CARLA_RPC_SLOW", self._instance_id, rtt_ms=rtt_ms,
                          slow_threshold_ms=self._slow_rtt_ms)
            except Exception as e:
                rtt_ms = (time.time() - t0) * 1000.0
                with self._lock:
                    self._stats["probes_total"] += 1
                    self._stats["probes_failed"] += 1
                    self._stats["consecutive_failures"] += 1
                    self._stats["last_probe_at"] = time.time()
                    self._stats["last_failure_at"] = time.time()
                    self._stats["last_failure_msg"] = f"{type(e).__name__}: {str(e)[:200]}"
                _emit("CARLA_RPC_FAILED", self._instance_id, rtt_ms=rtt_ms,
                      consecutive=self._stats["consecutive_failures"],
                      err_type=type(e).__name__, err_msg=str(e)[:200])


class StallDetector:
    """
    Daemon thread that detects when scenario ticks have stopped progressing
    AND emits warnings BEFORE the parent's watchdog SIGKILL fires.

    Parent's tick-heartbeat watchdog fires at 300s. We emit warnings at
    60s/120s/200s/270s so the log captures the pre-kill state (since SIGKILL
    gives no chance to write final state).

    At 200s+ we dump every Python thread's stack -- this is the SINGLE most
    important data point for distinguishing where the stall is happening.
    """

    def __init__(self, forensics, *, instance_id, check_interval_s=10.0):
        self._forensics = forensics
        self._instance_id = instance_id
        self._check_interval_s = check_interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired_thresholds: set = set()
        # Match the structure: each tuple is (threshold_s, event_name, dump_threads)
        self._stages = [
            (60.0, "STALL_WARNING_60S", False),
            (120.0, "STALL_WARNING_120S", False),
            (200.0, "STALL_WARNING_200S", True),    # dump thread stacks
            (270.0, "STALL_APPROACHING_KILL", True),  # dump again right before kill
        ]

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"stall_detector_{self._instance_id}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        """Called when a tick completes after a stall warning fired -- so we
        log recovery and re-arm the warnings for any future stall."""
        if self._fired_thresholds:
            _emit("TICK_RECOVERED", self._instance_id,
                  fired_warnings=sorted(self._fired_thresholds))
            self._fired_thresholds = set()

    def _loop(self) -> None:
        while not self._stop_event.wait(self._check_interval_s):
            last_tick = self._forensics.last_tick_completed_at
            if last_tick is None:
                continue
            age_s = time.time() - last_tick
            current_phase = self._forensics.current_phase
            current_tick_age_s = self._forensics.current_tick_age_s
            # Use whichever is older: time since last completed tick OR time
            # current tick has been in progress. Either signals a stall.
            stall_age_s = max(age_s, current_tick_age_s or 0.0)

            for threshold_s, event_name, dump_threads in self._stages:
                if stall_age_s >= threshold_s and threshold_s not in self._fired_thresholds:
                    self._fired_thresholds.add(threshold_s)
                    self._emit_stall(stall_age_s, threshold_s, event_name,
                                     current_phase, dump_threads)

    def _emit_stall(self, stall_age_s, threshold_s, event_name, current_phase, dump_threads):
        rpc = self._forensics.rpc_probe.get_stats() if self._forensics.rpc_probe else {}
        last_frame = self._forensics.last_frame
        recent_avg = self._forensics.recent_tick_avg_s()

        _emit(event_name, self._instance_id,
              threshold_s=threshold_s,
              stall_age_s=stall_age_s,
              last_completed_frame=last_frame,
              current_phase=current_phase,
              recent_avg_tick_s=recent_avg,
              rpc_last_rtt_ms=rpc.get("last_rtt_ms", -1),
              rpc_consecutive_failures=rpc.get("consecutive_failures", 0),
              rpc_total_failures=rpc.get("probes_failed", 0))

        if dump_threads:
            self._dump_all_threads()

    def _dump_all_threads(self) -> None:
        """Dump stack of every Python thread. This catches WHERE in the
        Python code execution is blocked (world.tick? agent? sensor wait?).
        Emitted as multiple lines so each thread is independently grep-able."""
        try:
            frames = sys._current_frames()
        except Exception:
            return
        for tid, frame in frames.items():
            try:
                # Find the thread name
                tname = "?"
                for t in threading.enumerate():
                    if t.ident == tid:
                        tname = t.name
                        break
                stack = traceback.format_stack(frame)
                # Last 8 frames is usually enough to identify the blocking call
                stack_str = " >> ".join(
                    line.strip().replace("\n", " | ") for line in stack[-8:]
                )
                _emit("STALL_THREAD_STACK", self._instance_id,
                      thread_id=tid, thread_name=tname,
                      stack=stack_str[:1500])
            except Exception:
                continue


class TickForensics:
    """
    Records per-tick wallclock phase timings, maintains rolling state,
    emits periodic snapshots, owns the RPC probe and stall detector.

    Thread-safe: tick recording runs on the main scenario_manager thread;
    snapshot emission and stall detection run on background threads. All
    shared state is protected by self._lock.
    """

    _instance_counter = 0

    def __init__(self, world, *, history_size=200):
        TickForensics._instance_counter += 1
        self.instance_id = TickForensics._instance_counter

        self.enabled = _env_bool("CARLA_TICK_FORENSICS", default=False)
        if not self.enabled:
            self.rpc_probe = None
            self.stall_detector = None
            return

        # Config (defaults tuned for low log volume — only emit on real signals)
        self._snapshot_interval_s = _env_float("CARLA_TICK_FORENSICS_SNAPSHOT_S", 120.0)  # was 30s
        self._slow_tick_threshold_s = _env_float("CARLA_TICK_FORENSICS_SLOW_TICK_S", 5.0)  # was 1.0
        self._trace_every_n = _env_int("CARLA_TICK_FORENSICS_TRACE_EVERY", 0)
        self._rpc_interval_s = _env_float("CARLA_RPC_PROBE_INTERVAL_S", 5.0)
        self._rpc_timeout_s = _env_float("CARLA_RPC_PROBE_TIMEOUT_S", 10.0)
        self._slow_rpc_rtt_ms = _env_float("CARLA_RPC_PROBE_SLOW_MS", 1000.0)

        # State (lock-protected)
        self._lock = threading.Lock()
        self._tick_history: deque = deque(maxlen=history_size)
        self._tick_count = 0
        self._slow_tick_count = 0
        self.last_tick_completed_at: Optional[float] = None
        self.last_frame: Optional[int] = None

        # Currently-in-progress tick state (read by stall detector)
        self._current_tick_started_at: Optional[float] = None
        self._current_tick_phase_started_at: Optional[float] = None
        self.current_phase: str = "not_started"
        self._current_tick_phases: dict = {}
        self._current_tick_frame: Optional[int] = None

        # Periodic snapshot
        self._last_snapshot_at = time.time()
        self._snapshot_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Sub-systems
        self.rpc_probe = CarlaRpcProbe(
            world,
            instance_id=self.instance_id,
            interval_s=self._rpc_interval_s,
            timeout_s=self._rpc_timeout_s,
            slow_rtt_ms=self._slow_rpc_rtt_ms,
        )
        self.stall_detector = StallDetector(self, instance_id=self.instance_id)

        self._init_at = time.time()
        _emit("INIT", self.instance_id,
              snapshot_s=self._snapshot_interval_s,
              slow_tick_s=self._slow_tick_threshold_s,
              trace_every=self._trace_every_n,
              rpc_interval_s=self._rpc_interval_s,
              rpc_slow_ms=self._slow_rpc_rtt_ms)

    @property
    def current_tick_age_s(self) -> Optional[float]:
        if self._current_tick_started_at is None:
            return None
        return time.time() - self._current_tick_started_at

    def start(self) -> None:
        if not self.enabled:
            return
        self.rpc_probe.start()
        self.stall_detector.start()
        # Periodic snapshot thread
        self._snapshot_thread = threading.Thread(
            target=self._snapshot_loop, daemon=True,
            name=f"tick_forensics_snapshot_{self.instance_id}"
        )
        self._snapshot_thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        # Final snapshot before stopping
        self._emit_snapshot(label="FINAL")
        self._stop_event.set()
        if self.rpc_probe:
            self.rpc_probe.stop()
        if self.stall_detector:
            self.stall_detector.stop()
        _emit("STOP", self.instance_id, total_ticks=self._tick_count,
              slow_ticks=self._slow_tick_count,
              uptime_s=time.time() - self._init_at)

    def begin_tick(self, frame_id) -> None:
        """Called at the START of _tick_scenario when timestamp progressed."""
        if not self.enabled:
            return
        with self._lock:
            self._current_tick_started_at = time.time()
            self._current_tick_phase_started_at = self._current_tick_started_at
            self._current_tick_frame = frame_id
            self._current_tick_phases = {}
            self.current_phase = "begin"

    def begin_phase(self, name: str) -> None:
        """Called at the start of each named phase within _tick_scenario."""
        if not self.enabled:
            return
        with self._lock:
            self._current_tick_phase_started_at = time.time()
            self.current_phase = name

    def end_phase(self, name: str) -> None:
        """Called at the end of each named phase. Records phase duration."""
        if not self.enabled or self._current_tick_phase_started_at is None:
            return
        with self._lock:
            dur = time.time() - self._current_tick_phase_started_at
            self._current_tick_phases[name] = dur
            self.current_phase = "between_phases"
            self._current_tick_phase_started_at = None

    def end_tick(self) -> None:
        """Called at the END of _tick_scenario (after world.tick())."""
        if not self.enabled or self._current_tick_started_at is None:
            return
        now = time.time()
        with self._lock:
            total_s = now - self._current_tick_started_at
            phases = dict(self._current_tick_phases)
            frame = self._current_tick_frame
            self._tick_count += 1
            self._tick_history.append({
                "frame": frame, "total_s": total_s, "phases": phases, "at": now
            })
            self.last_tick_completed_at = now
            self.last_frame = frame
            self.current_phase = "between_ticks"
            self._current_tick_started_at = None
            self._current_tick_phase_started_at = None
            is_slow = total_s >= self._slow_tick_threshold_s
            if is_slow:
                self._slow_tick_count += 1

        # Per-tick logging (outside lock)
        if is_slow:
            phase_str = " ".join(f"{k}={v:.3f}s" for k, v in sorted(phases.items()))
            _emit("SLOW_TICK", self.instance_id, frame=frame, total_s=total_s,
                  threshold_s=self._slow_tick_threshold_s, phases=phase_str)

        if self._trace_every_n > 0 and (self._tick_count % self._trace_every_n) == 0:
            phase_str = " ".join(f"{k}={v:.3f}s" for k, v in sorted(phases.items()))
            _emit("TICK", self.instance_id, frame=frame, total_s=total_s,
                  phases=phase_str)

        # Tell stall detector to clear any fired warnings -- this tick succeeded
        if self.stall_detector:
            self.stall_detector.reset()

    def recent_tick_avg_s(self, n: int = 20) -> Optional[float]:
        with self._lock:
            if not self._tick_history:
                return None
            recent = list(self._tick_history)[-n:]
        return sum(t["total_s"] for t in recent) / len(recent)

    def _snapshot_loop(self) -> None:
        while not self._stop_event.wait(self._snapshot_interval_s):
            self._emit_snapshot()

    def _emit_snapshot(self, label: str = "SNAPSHOT") -> None:
        with self._lock:
            if not self._tick_history:
                _emit(label, self.instance_id, status="no_ticks_yet",
                      uptime_s=time.time() - self._init_at)
                return
            recent = list(self._tick_history)[-20:]
            tick_count = self._tick_count
            slow_count = self._slow_tick_count
            last_frame = self.last_frame
            last_tick_at = self.last_tick_completed_at
            current_phase = self.current_phase

        avg_total = sum(t["total_s"] for t in recent) / len(recent)
        min_total = min(t["total_s"] for t in recent)
        max_total = max(t["total_s"] for t in recent)
        # Sim rate: assume 0.05s sim time per tick (20Hz default)
        sim_rate_hz = 1.0 / avg_total if avg_total > 0 else 0.0
        slowdown_vs_realtime = avg_total / 0.05 if avg_total > 0 else 0.0
        # Per-phase aggregate
        phase_keys = set()
        for t in recent:
            phase_keys.update(t["phases"].keys())
        phase_avgs = {
            k: sum(t["phases"].get(k, 0) for t in recent) / len(recent)
            for k in phase_keys
        }
        phase_str = " ".join(f"{k}={v:.3f}s" for k, v in sorted(phase_avgs.items()))

        rpc = self.rpc_probe.get_stats() if self.rpc_probe else {}
        age_since_last_tick = (time.time() - last_tick_at) if last_tick_at else None

        _emit(label, self.instance_id,
              ticks_total=tick_count, ticks_slow=slow_count,
              last_frame=last_frame,
              last20_avg_s=avg_total, last20_min_s=min_total, last20_max_s=max_total,
              sim_rate_hz=sim_rate_hz,
              slowdown_vs_realtime_x=slowdown_vs_realtime,
              age_since_last_tick_s=age_since_last_tick if age_since_last_tick else -1,
              current_phase=current_phase,
              phases_avg=phase_str,
              rpc_probes=rpc.get("probes_total", 0),
              rpc_failures=rpc.get("probes_failed", 0),
              rpc_consecutive_failures=rpc.get("consecutive_failures", 0),
              rpc_last_rtt_ms=rpc.get("last_rtt_ms", -1) if rpc.get("last_rtt_ms") else -1,
              rpc_max_rtt_ms=rpc.get("max_rtt_ms", 0))


# Convenience: a no-op TickForensics for use when forensics is disabled. This
# avoids littering scenario_manager.py with `if forensics: ...` checks.
class _NoOpForensics:
    enabled = False
    rpc_probe = None
    stall_detector = None
    last_tick_completed_at = None
    last_frame = None
    current_phase = "disabled"
    current_tick_age_s = None
    instance_id = 0

    def start(self): pass
    def stop(self): pass
    def begin_tick(self, frame_id): pass
    def begin_phase(self, name): pass
    def end_phase(self, name): pass
    def end_tick(self): pass
    def recent_tick_avg_s(self, n=20): return None


def make_forensics(world) -> TickForensics:
    """Factory: returns enabled TickForensics or a no-op stub depending on env."""
    f = TickForensics(world)
    if not f.enabled:
        return _NoOpForensics()
    return f
