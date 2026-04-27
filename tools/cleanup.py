#!/usr/bin/env python3
"""GPU-scoped process cleanup tool for CoLMDriver baseline runs.

Lets you nuke all CARLAs / leaderboard subprocesses / per-scenario wrappers
bound to a SPECIFIC subset of GPUs without touching anything on other GPUs.
Use this when one of your masters has cascaded but a different master on
different GPUs is still healthy and you don't want to disturb it.

Default is dry-run — pass ``--confirm-kill`` to actually send signals.

Usage examples:
    # Dry run: see what's on GPU 3 and 7
    python tools/cleanup.py --gpu 3,7

    # Actually clean GPU 3 and 7 (CARLAs + their per-scenario wrappers)
    python tools/cleanup.py --gpu 3,7 --confirm-kill

    # Also remove orphaned _partial_* result dirs
    python tools/cleanup.py --gpu 3,7 --confirm-kill --confirm-dirs

Safety properties:
  * Only kills processes that nvidia-smi reports as USING the target GPUs
    (compute apps). This naturally excludes CARLAs/leaderboards on other GPUs.
  * Walks UP from each compute PID to find its per-scenario wrapper (the
    intermediate ``run_custom_eval.py`` instance with ``--no-start-carla``)
    and kills the wrapper too, so the master cleanly observes a child exit.
  * NEVER kills the top-level master (the one with ``--scenario-pool`` in its
    argv) — masters typically span multiple GPUs and killing one master
    would orphan its other-GPU CARLAs too.
  * SIGINT first with ``--grace-s`` (default 30s) for clean teardown that
    writes per-ego results.json. SIGKILL only as escalation for stragglers.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal as _sig
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ─── nvidia-smi helpers ────────────────────────────────────────────

def gpu_uuid_by_index() -> Dict[int, str]:
    """Return ``{cuda_idx: uuid}`` mapping. UUID prefix stripped of ``GPU-``."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
    except Exception as exc:
        print(f"[cleanup] nvidia-smi query failed: {exc}", file=sys.stderr)
        return {}
    out_map: Dict[int, str] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                idx = int(parts[0])
                uuid = parts[1].lstrip("GPU-").lower()
                out_map[idx] = uuid
            except ValueError:
                continue
    return out_map


def compute_pids_per_gpu() -> Dict[str, List[Tuple[int, int]]]:
    """Return ``{gpu_uuid: [(pid, used_mib), ...]}`` for compute apps."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception as exc:
        print(f"[cleanup] nvidia-smi compute-apps query failed: {exc}", file=sys.stderr)
        return {}
    by_uuid: Dict[str, List[Tuple[int, int]]] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            uuid = parts[0].lstrip("GPU-").lower()
            pid = int(parts[1])
            mib = int(parts[2])
        except ValueError:
            continue
        by_uuid.setdefault(uuid, []).append((pid, mib))
    return by_uuid


# ─── /proc helpers ─────────────────────────────────────────────────

def proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode("utf-8", errors="replace").replace("\0", " ").strip()
    except Exception:
        return ""


def proc_parent(pid: int) -> int | None:
    """Return PPID of pid, or None if process is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
            # field 4 is PPID; but PID may contain space in comm so handle correctly
            # comm is parts[1] possibly with embedded spaces -- use rfind of ')' on full line
        with open(f"/proc/{pid}/stat") as f:
            line = f.read()
        rparen = line.rfind(")")
        if rparen < 0:
            return None
        rest = line[rparen + 1 :].strip().split()
        # rest[0] is state, rest[1] is ppid
        return int(rest[1])
    except Exception:
        return None


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def find_per_scenario_wrapper(leaf_pid: int) -> int | None:
    """Walk up from leaf_pid until we find the per-scenario wrapper —
    a ``run_custom_eval.py`` Python process WITHOUT ``--scenario-pool``
    in argv. Returns its PID or None if not found.

    This wrapper is the parent of leaderboard_evaluator_parameter, and
    its parent is the master. Killing the wrapper cleanly observes from
    the master's POV without disturbing the master itself.
    """
    cur = leaf_pid
    for _ in range(20):  # ascend up to 20 levels (paranoia cap)
        ppid = proc_parent(cur)
        if ppid is None or ppid <= 1:
            return None
        ppid_cmd = proc_cmdline(ppid)
        # Stop if we've reached a master (DON'T kill master)
        if "--scenario-pool" in ppid_cmd:
            # cur is the per-scenario wrapper (or close to it)
            return cur if cur != leaf_pid else None
        # Match a run_custom_eval.py invocation that's clearly per-scenario
        if "run_custom_eval.py" in ppid_cmd and "--scenario-pool" not in ppid_cmd:
            return ppid
        cur = ppid
    return None


# ─── kill orchestration ───────────────────────────────────────────

def signal_pids(
    pids: Set[int], sig: int, label: str
) -> int:
    sent = 0
    for pid in pids:
        try:
            os.kill(pid, sig)
            sent += 1
        except (OSError, ProcessLookupError):
            pass
    print(f"  [{label}] delivered to {sent}/{len(pids)} PID(s)")
    return sent


def survivors(pids: Set[int]) -> Set[int]:
    return {pid for pid in pids if is_alive(pid)}


# ─── partial dir cleanup ──────────────────────────────────────────

def find_partial_dirs(results_root: Path) -> List[Path]:
    if not results_root.exists():
        return []
    return [p for p in results_root.rglob("*_partial_*") if p.is_dir()]


# ─── main ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="GPU-scoped process cleanup for CoLMDriver baseline runs",
    )
    ap.add_argument(
        "--gpu",
        required=True,
        help="Comma-separated CUDA GPU indices to clean (e.g. '3,7')",
    )
    ap.add_argument(
        "--confirm-kill",
        action="store_true",
        help="Actually send signals (default: dry-run)",
    )
    ap.add_argument(
        "--confirm-dirs",
        action="store_true",
        help="Also remove orphaned _partial_* result dirs",
    )
    ap.add_argument(
        "--results-root",
        default="/data2/marco/CoLMDriver/results/results_driving_custom",
        help="Where to look for _partial_ dirs (only used with --confirm-dirs)",
    )
    ap.add_argument(
        "--grace-s",
        type=float,
        default=30.0,
        help="Seconds between SIGINT and SIGKILL escalation (default 30)",
    )
    ap.add_argument(
        "--include-wrappers",
        action="store_true",
        default=True,
        help="Also kill per-scenario wrappers above leaf compute PIDs (default on)",
    )
    args = ap.parse_args()

    try:
        target_gpus = {int(x.strip()) for x in args.gpu.split(",") if x.strip()}
    except ValueError:
        print(f"[cleanup] bad --gpu value: {args.gpu!r}", file=sys.stderr)
        return 2
    if not target_gpus:
        print("[cleanup] --gpu produced empty set", file=sys.stderr)
        return 2

    print(f"[cleanup] target CUDA GPUs: {sorted(target_gpus)}")
    print(f"[cleanup] mode: {'KILL' if args.confirm_kill else 'DRY-RUN'}")

    # 1. Resolve target UUIDs
    uuid_map = gpu_uuid_by_index()
    target_uuids = {uuid_map[g] for g in target_gpus if g in uuid_map}
    if not target_uuids:
        print("[cleanup] no GPU UUIDs found for target indices. Exiting.")
        return 1
    print(f"[cleanup] target GPU UUIDs: {sorted(target_uuids)}")

    # 2. Find compute PIDs on target GPUs
    by_uuid = compute_pids_per_gpu()
    leaf_pids: Set[int] = set()
    for uuid in target_uuids:
        for pid, mib in by_uuid.get(uuid, []):
            leaf_pids.add(pid)
    print(f"\n[cleanup] {len(leaf_pids)} compute PID(s) on target GPUs:")
    for pid in sorted(leaf_pids):
        cmd = proc_cmdline(pid)[:140]
        print(f"  pid={pid:>7}  cmd={cmd}")

    if not leaf_pids:
        print("[cleanup] nothing to do.")
        return 0

    # 3. Walk up to find per-scenario wrappers (so we kill the whole subtree
    #    cleanly and the master sees a wrapper exit)
    wrapper_pids: Set[int] = set()
    if args.include_wrappers:
        for leaf in leaf_pids:
            w = find_per_scenario_wrapper(leaf)
            if w is not None:
                wrapper_pids.add(w)
        print(
            f"\n[cleanup] {len(wrapper_pids)} per-scenario wrapper(s) cover "
            f"these compute PIDs:"
        )
        for pid in sorted(wrapper_pids):
            cmd = proc_cmdline(pid)[:140]
            print(f"  pid={pid:>7}  cmd={cmd}")

    # 4. Defensive check: refuse to kill any PID whose cmdline contains
    #    ``--scenario-pool`` (= a master). This is a belt-and-braces guard
    #    in case the wrapper-detection logic above misidentifies something.
    all_kill = leaf_pids | wrapper_pids
    masters_in_kill_list = {
        pid for pid in all_kill if "--scenario-pool" in proc_cmdline(pid)
    }
    if masters_in_kill_list:
        print(
            f"\n[cleanup] REFUSING TO KILL: {len(masters_in_kill_list)} master "
            "process(es) ended up in the kill list:",
            file=sys.stderr,
        )
        for pid in masters_in_kill_list:
            print(f"  pid={pid}  cmd={proc_cmdline(pid)[:140]}", file=sys.stderr)
        print(
            "[cleanup] this would disturb other GPUs. Aborting. Use --gpu set "
            "that doesn't span a master, or kill the master manually.",
            file=sys.stderr,
        )
        return 3
    all_kill -= masters_in_kill_list

    # 5. Show summary
    print(f"\n[cleanup] total PID(s) to signal: {len(all_kill)}")
    if not args.confirm_kill:
        print(
            "\n[cleanup] DRY-RUN — no signals sent. Re-run with --confirm-kill "
            "to actually clean up."
        )
        return 0

    # 6. SIGINT phase
    print(f"\n[cleanup] phase 1: SIGINT (grace={args.grace_s:.0f}s) ...")
    signal_pids(all_kill, _sig.SIGINT, "SIGINT")
    time.sleep(args.grace_s)

    # 7. SIGKILL phase
    still = survivors(all_kill)
    if still:
        print(f"\n[cleanup] phase 2: SIGKILL to {len(still)} survivor(s) ...")
        signal_pids(still, _sig.SIGKILL, "SIGKILL")
        time.sleep(5.0)
        still2 = survivors(all_kill)
        if still2:
            print(
                f"\n[cleanup] WARNING: {len(still2)} PID(s) still alive after "
                "SIGKILL — likely in D state (uninterruptible kernel sleep). "
                "May require system reboot to clear.",
                file=sys.stderr,
            )
            for pid in still2:
                print(f"  pid={pid}  cmd={proc_cmdline(pid)[:140]}", file=sys.stderr)
    else:
        print("[cleanup] all PIDs exited gracefully on SIGINT.")

    # 8. Verify VRAM dropped on our GPUs
    print("\n[cleanup] post-kill GPU state:")
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                marker = " <-- target" if idx in target_gpus else ""
                print(f"  GPU {idx}: used={parts[1]} MiB  free={parts[2]} MiB{marker}")
    except Exception as exc:
        print(f"  (post-kill nvidia-smi query failed: {exc})")

    # 9. Optionally clean up orphaned _partial_ dirs
    if args.confirm_dirs:
        results_root = Path(args.results_root)
        partials = find_partial_dirs(results_root)
        print(f"\n[cleanup] phase 3: removing {len(partials)} _partial_* dirs under {results_root}")
        for d in partials:
            try:
                shutil.rmtree(d)
                print(f"  removed: {d}")
            except Exception as exc:
                print(f"  failed to remove {d}: {exc}", file=sys.stderr)

    print("\n[cleanup] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
