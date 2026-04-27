"""Soak runner for the ``--scenario-pool`` architecture.

Launches N iterations of a small ``run_custom_eval.py`` invocation
(--scenario-pool on, a handful of planners, a handful of scenarios) and
reports pass/fail counts and wall time per iteration.  Intended for use
before flipping ``--scenario-pool`` to the default path.

Usage
-----
  python -m tools.soak_scenario_pool \\
      --routes-dir /path/to/routes_parent \\
      --planners rule-based,autoagent \\
      --gpus 0,1 \\
      --iterations 3

The runner does NOT validate result contents; it only checks exit codes
and surfaces timing.  For end-to-end correctness, diff the resulting
``results/results_driving_custom/*/`` directories against a known-good
baseline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_one(
    *,
    routes_dir: Path,
    planners: list[str],
    gpus: list[int],
    results_tag: str,
    extra_args: list[str],
) -> tuple[int, float]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_custom_eval.py"),
        "--scenario-pool",
        "--routes-dir", str(routes_dir),
        "--results-tag", results_tag,
        "--gpus", ",".join(str(g) for g in gpus),
    ]
    for p in planners:
        cmd.extend(["--planner", p])
    cmd.extend(extra_args)

    print(f"[soak] launching: {' '.join(cmd)}")
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return (proc.returncode, time.monotonic() - start)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-dir", required=True, type=Path)
    parser.add_argument(
        "--planners",
        default="rule-based",
        help="Comma-separated planner list.",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of independent soak iterations to run.",
    )
    parser.add_argument(
        "--results-tag-prefix",
        default="_soak_scenario_pool",
        help="Results tag prefix; iteration index is appended.",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Remaining arguments are forwarded verbatim to run_custom_eval.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    routes_dir = args.routes_dir.expanduser().resolve()
    if not routes_dir.is_dir():
        print(f"[soak] ERROR: routes-dir does not exist: {routes_dir}")
        return 2

    planners = [p.strip() for p in args.planners.split(",") if p.strip()]
    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    iterations = max(1, int(args.iterations))
    extra_args = list(args.extra or [])
    # argparse REMAINDER swallows a leading '--'; strip it.
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    results: list[tuple[int, int, float]] = []  # (iter_idx, rc, elapsed_s)
    for i in range(1, iterations + 1):
        tag = f"{args.results_tag_prefix}/iter_{i:02d}"
        rc, elapsed = _run_one(
            routes_dir=routes_dir,
            planners=planners,
            gpus=gpus,
            results_tag=tag,
            extra_args=extra_args,
        )
        results.append((i, rc, elapsed))
        outcome = "PASS" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[soak] iter {i}/{iterations}: {outcome} in {elapsed:.1f}s")

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total_elapsed = sum(e for _, _, e in results)
    print()
    print("=" * 60)
    print(f"Soak summary: {passed}/{iterations} iterations passed")
    print(f"Total wall time: {total_elapsed:.1f}s")
    print(f"Mean per iteration: {total_elapsed / iterations:.1f}s")
    print("=" * 60)
    return 0 if passed == iterations else 1


if __name__ == "__main__":
    sys.exit(main())
