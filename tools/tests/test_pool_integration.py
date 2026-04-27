"""Phase 3 unit tests for :mod:`tools._pool_integration`.

Focus: argv construction and outcome classification (pure logic, no
subprocess launches).  End-to-end ``run_scenario_pool`` is validated by
live ``--scenario-pool`` runs.

Run with:  python -m unittest tools.tests.test_pool_integration
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import io


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# run_custom_eval has unqualified imports (``from setup_scenario_from_zip
# import ...``) that assume tools/ is on sys.path -- match that here.
_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


# Defer tools._pool_integration import until a test; importing it triggers
# lazy imports on tools.run_custom_eval, which is fine but slow.


def _evict_fake_rce_module() -> None:
    """Remove any fake ``tools.run_custom_eval`` left over by sibling tests.

    ``tools/tests/test_carla_pool.py`` installs a stub module into
    ``sys.modules['tools.run_custom_eval']`` to avoid the cost of importing
    the real 13k-line file.  When we also run, we need the real module
    because ``build_pool_grandchild_argv`` lazy-imports ``build_child_argv``.
    """
    mod = sys.modules.get("tools.run_custom_eval")
    if mod is not None and not hasattr(mod, "build_child_argv"):
        del sys.modules["tools.run_custom_eval"]


class _Job:
    """Minimal ScenarioJob-like stand-in."""

    def __init__(self, *, planner_name="myplanner", scenario_dir="/tmp/sc_a",
                 results_tag="results/myplanner"):
        self.planner_name = planner_name
        self.scenario_dir = Path(scenario_dir)
        self.results_tag = results_tag


class _Lease:
    def __init__(self, *, world_port=2000, tm_port=2002):
        self.world_port = world_port
        self.tm_port = tm_port


class BuildScenarioJobListTests(unittest.TestCase):
    def test_carries_conda_env_in_spawn_kwargs(self):
        from tools._pool_integration import build_scenario_job_list

        class _Estimator:
            @staticmethod
            def estimate(_planner_name):
                return 4096

        planner = type(
            "Planner",
            (),
            {"name": "tcp", "run_id": "tcp", "conda_env": "colmdrivermarco2"},
        )()
        jobs = build_scenario_job_list(
            planner_requests=[planner],
            scenario_entries=[(Path("/tmp/scenario_1"), "scenario_1")],
            results_root=Path("/tmp/results"),
            results_tag_base="smoke_pool_v1",
            retry_budget=1,
            vram_estimator=_Estimator(),
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].spawn_kwargs.get("conda_env"), "colmdrivermarco2")


class BuildPoolGrandchildArgvTests(unittest.TestCase):
    def setUp(self):
        _evict_fake_rce_module()

    def test_injects_required_replacements(self):
        from tools._pool_integration import build_pool_grandchild_argv

        base_argv = ["--start-carla", "--routes-dir", "/tmp/original"]
        argv = build_pool_grandchild_argv(
            base_argv=base_argv,
            job=_Job(planner_name="plan-x", scenario_dir="/tmp/sc_42",
                     results_tag="run/plan-x"),
            lease=_Lease(world_port=2100, tm_port=2102),
        )
        self.assertIn("--planner", argv)
        self.assertIn("plan-x", argv)
        self.assertIn("--no-start-carla", argv)
        # --start-carla flag must be stripped from the base argv.
        self.assertNotIn("--start-carla", argv)
        # --routes-dir must point to the job's scenario dir, not the original.
        rd_idx = argv.index("--routes-dir")
        self.assertEqual(argv[rd_idx + 1], "/tmp/sc_42")
        # --port / --traffic-manager-port reflect the lease.
        p_idx = argv.index("--port")
        self.assertEqual(argv[p_idx + 1], "2100")
        tm_idx = argv.index("--traffic-manager-port")
        self.assertEqual(argv[tm_idx + 1], "2102")

    def test_strips_scenario_pool_flags_from_base(self):
        from tools._pool_integration import build_pool_grandchild_argv

        base_argv = [
            "--scenario-pool",
            "--scenario-pool-no-reuse",
            "--start-carla",
            "--planner", "old-planner",
            "--gpus", "0,1",
            "--port", "2000",
        ]
        argv = build_pool_grandchild_argv(
            base_argv=base_argv,
            job=_Job(planner_name="new-planner"),
            lease=_Lease(),
        )
        # Grandchild must not itself trigger pool logic.
        self.assertNotIn("--scenario-pool", argv)
        self.assertNotIn("--scenario-pool-no-reuse", argv)
        self.assertNotIn("--gpus", argv)
        # Only the replacement --planner should remain.
        planner_indices = [i for i, tok in enumerate(argv) if tok == "--planner"]
        self.assertEqual(len(planner_indices), 1)
        self.assertEqual(argv[planner_indices[0] + 1], "new-planner")

    def test_overwrite_propagated(self):
        from tools._pool_integration import build_pool_grandchild_argv

        argv = build_pool_grandchild_argv(
            base_argv=[],
            job=_Job(),
            lease=_Lease(),
            overwrite=True,
        )
        self.assertIn("--overwrite", argv)

    def test_no_overwrite_by_default(self):
        from tools._pool_integration import build_pool_grandchild_argv

        argv = build_pool_grandchild_argv(
            base_argv=[],
            job=_Job(),
            lease=_Lease(),
        )
        self.assertNotIn("--overwrite", argv)

    def test_results_subdir_propagated(self):
        from tools._pool_integration import build_pool_grandchild_argv

        argv = build_pool_grandchild_argv(
            base_argv=[],
            job=_Job(),
            lease=_Lease(),
            results_subdir="scenario_a",
        )
        idx = argv.index("--results-subdir")
        self.assertEqual(argv[idx + 1], "scenario_a")


class PoolStatusReporterTests(unittest.TestCase):
    def test_eval_done_tail_prefers_inner_exception_over_conda_wrapper(self):
        from tools._pool_integration import _PoolStatusReporter

        reporter = _PoolStatusReporter(total_jobs=1)
        result = SimpleNamespace(
            job=SimpleNamespace(planner_name="tcp", scenario_name="3"),
            status=SimpleNamespace(value="soft_failure"),
            lease_gpu_id=1,
            duration_s=3.3,
            stderr_tail="\n".join(
                [
                    "Traceback (most recent call last):",
                    "RuntimeError: Missing ego route XML for ego_id=5. Discovered keys: ['0', '1', '2', '3', '4', '49']",
                    "ERROR conda.cli.main_run:execute(125): `conda run python tools/run_custom_eval.py ...`",
                ]
            ),
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            reporter.on_eval_finish(result=result)
        line = buf.getvalue().strip()
        self.assertIn("tail=RuntimeError: Missing ego route XML for ego_id=5.", line)
        self.assertNotIn("conda.cli.main_run", line)


class ClassifyGrandchildOutcomeTests(unittest.TestCase):
    def test_returncode_0_is_success(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=0,
            output_tail=[],
            result_root=Path("/nonexistent"),
        )
        self.assertEqual(status, "success")
        self.assertEqual(release, "clean")

    def test_oom_marker_in_tail(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["some line", "torch.cuda.OutOfMemoryError: CUDA out of memory."],
            result_root=Path("/nonexistent"),
        )
        self.assertEqual(status, "oom")
        self.assertEqual(release, "crashed")

    def test_carla_crash_marker_in_tail(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["rpc::timeout while reading response"],
            result_root=Path("/nonexistent"),
        )
        self.assertEqual(status, "carla_crash")
        self.assertEqual(release, "crashed")

    def test_carla_unreachable_marker_in_tail(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=[
                "RuntimeError: Scenario evaluator exited nonzero "
                "(reason=carla_unreachable, returncode=-9)."
            ],
            result_root=Path("/nonexistent"),
        )
        self.assertEqual(status, "carla_crash")
        self.assertEqual(release, "crashed")

    def test_default_soft_failure(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["ValueError: bad arg"],
            result_root=Path("/nonexistent"),
        )
        self.assertEqual(status, "soft_failure")
        self.assertEqual(release, "clean")

    def test_sentinel_oom_classification(self):
        from tools._pool_integration import classify_grandchild_outcome

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_run_status.json").write_text(
                json.dumps({"status": "retry_exhausted", "failure_kind": "cuda_oom"})
            )
            status, release, _reason, _detail = classify_grandchild_outcome(
                returncode=1,
                output_tail=[],
                result_root=root,
            )
            self.assertEqual(status, "oom")
            self.assertEqual(release, "crashed")

    def test_sentinel_carla_classification(self):
        from tools._pool_integration import classify_grandchild_outcome

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_run_status.json").write_text(
                json.dumps({"status": "retry_exhausted", "failure_kind": "carla_startup_failure"})
            )
            status, _, _reason, _detail = classify_grandchild_outcome(
                returncode=1,
                output_tail=[],
                result_root=root,
            )
            self.assertEqual(status, "carla_crash")


class RetryBudgetTests(unittest.TestCase):
    """The pool scheduler owns retries now (the grandchild is always run
    with --scenario-retry-limit 1).  ``classify_grandchild_outcome`` must
    return ``retry_queued`` on transient failures when budget remains, so
    the scheduler can re-enqueue the job instead of burying it as partial.
    """

    def test_transient_carla_crash_is_retried(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["rpc::timeout while reading response"],
            result_root=Path("/nonexistent"),
            retry_attempt=0,
            retry_budget=1,
        )
        self.assertEqual(status, "retry_queued")
        self.assertEqual(release, "crashed")

    def test_exhausted_retries_fall_through_to_terminal(self):
        from tools._pool_integration import classify_grandchild_outcome

        status, _, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["rpc::timeout while reading response"],
            result_root=Path("/nonexistent"),
            retry_attempt=1,
            retry_budget=1,
        )
        self.assertEqual(status, "carla_crash")

    def test_unclassified_sigkill_retried_once(self):
        from tools._pool_integration import classify_grandchild_outcome

        # Mirrors the live-log pattern of reason=no_route_execution,
        # returncode=-9 after a CARLA heartbeat freeze: no explicit marker
        # in tail, but almost always recovers on a fresh CARLA.
        status, release, _reason, _detail = classify_grandchild_outcome(
            returncode=-9,
            output_tail=["heartbeat has not been updated for 301s"],
            result_root=Path("/nonexistent"),
            retry_attempt=0,
            retry_budget=1,
        )
        self.assertEqual(status, "retry_queued")
        self.assertEqual(release, "crashed")

    def test_oom_is_not_retried_at_this_layer(self):
        from tools._pool_integration import classify_grandchild_outcome

        # OOMs are throttled by the scheduler's multiplier, not re-enqueued
        # -- otherwise we'd just repeat the failure.
        status, _, _reason, _detail = classify_grandchild_outcome(
            returncode=1,
            output_tail=["CUDA out of memory"],
            result_root=Path("/nonexistent"),
            retry_attempt=0,
            retry_budget=5,
        )
        self.assertEqual(status, "oom")

    def test_sentinel_sensor_failure_retried(self):
        from tools._pool_integration import classify_grandchild_outcome

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_run_status.json").write_text(
                json.dumps(
                    {"status": "retry_exhausted", "failure_kind": "sensor_failure"}
                )
            )
            status, release, _reason, _detail = classify_grandchild_outcome(
                returncode=1,
                output_tail=[],
                result_root=root,
                retry_attempt=0,
                retry_budget=2,
            )
            self.assertEqual(status, "retry_queued")
            self.assertEqual(release, "clean")


if __name__ == "__main__":
    unittest.main()
