"""Unit tests for :mod:`tools._scheduler`.

Focus: pure decision logic -- weighted RR, OOM backoff, VRAM budget
calculation, and the three-tier dispatch priority tree.  The full
``run()`` loop is not exercised here (it is threaded and time-dependent);
its orchestration is covered by end-to-end ``--scenario-pool`` runs.

Run with:  python -m pytest tools/tests/test_scheduler.py -q
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools import _scheduler as sch  # noqa: E402


# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------


class _FakeStats:
    def __init__(self, free_mib: int, used_mib: int = 0):
        self.free_mib = free_mib
        self.used_mib = used_mib


class _FakeGpuMonitor:
    def __init__(
        self,
        free_per_gpu: dict[int, int],
        *,
        used_per_gpu: dict[int, int] | None = None,
        fresh: bool = True,
    ):
        self._free = dict(free_per_gpu)
        self._used = dict(used_per_gpu or {})
        self._fresh = fresh

    def has_fresh_snapshot(self, max_age_s: float) -> bool:
        return self._fresh

    def stats_for(self, gpu_id_str: str):
        try:
            gid = int(gpu_id_str)
        except Exception:
            return None
        if gid not in self._free:
            return None
        return _FakeStats(self._free[gid], self._used.get(gid, 0))

    def set_free(self, gpu_id: int, free_mib: int) -> None:
        self._free[gpu_id] = free_mib

    def set_used(self, gpu_id: int, used_mib: int) -> None:
        self._used[gpu_id] = used_mib

    def set_fresh(self, fresh: bool) -> None:
        self._fresh = fresh


class _FakePoolInstance:
    def __init__(self, instance_id: str, gpu_id: int, current_town: str | None):
        self.instance_id = instance_id
        self.gpu_id = gpu_id
        self.current_town = current_town


class _FakeLease:
    def __init__(self, instance_id: str, generation: int = 1):
        self.instance_id = instance_id
        self.generation = generation


class _FakeCarlaPool:
    """Minimal pool stand-in driven by test-controlled ``instances`` list."""

    def __init__(self):
        self.instances: list[_FakePoolInstance] = []
        self.started: list[int] = []
        self.released: list[tuple[str, str]] = []

    def find_available_instance(self, *, town: str | None = None, gpu_id: int | None = None):
        for inst in self.instances:
            if gpu_id is not None and inst.gpu_id != gpu_id:
                continue
            if town is not None and inst.current_town != town:
                continue
            return inst
        if town is not None:
            # Fall back to any-gpu any-town search with matching gpu only.
            return None
        return None

    def acquire_lease(self, instance_id: str, *, town: str):
        for inst in self.instances:
            if inst.instance_id == instance_id:
                inst.current_town = town
                return _FakeLease(instance_id)
        return None

    def release_lease(self, lease, *, status: str):
        self.released.append((lease.instance_id, status))

    def start_instance(self, *, gpu_id: int):
        self.started.append(gpu_id)
        inst = _FakePoolInstance(
            instance_id=f"inst-{len(self.started)}",
            gpu_id=gpu_id,
            current_town=None,
        )
        self.instances.append(inst)
        return inst


def _make_scheduler(
    *,
    pool=None,
    monitor=None,
    gpu_ids=(0,),
    carla_vram=6000,
    safety_buffer=1500,
) -> sch.ScenarioScheduler:
    pool = pool or _FakeCarlaPool()
    monitor = monitor or _FakeGpuMonitor({g: 24000 for g in gpu_ids})

    estimator = mock.MagicMock()
    estimator.estimate.return_value = 4000

    return sch.ScenarioScheduler(
        carla_pool=pool,
        gpu_monitor=monitor,
        vram_estimator=estimator,
        spawn_grandchild=lambda **kw: None,
        on_result=lambda r: None,
        gpu_ids=list(gpu_ids),
        carla_vram_estimate_mib=carla_vram,
        vram_safety_buffer_mib=safety_buffer,
        dispatch_tick_s=0.01,
    )


def _make_job(
    *,
    planner: str = "p1",
    scenario: str = "sc",
    town: str = "Town01",
    vram_mib: int = 4000,
) -> sch.ScenarioJob:
    return sch.ScenarioJob(
        job_id=f"{planner}:{scenario}",
        planner_run_id=planner,
        planner_name=planner,
        scenario_name=scenario,
        scenario_dir=Path("/tmp/fake"),
        results_tag=f"results/{planner}",
        town=town,
        vram_estimate_mib=vram_mib,
        retry_budget=3,
    )


# ----------------------------------------------------------------------
# Weighted round-robin
# ----------------------------------------------------------------------


class WeightedRRTests(unittest.TestCase):
    def test_empty_queues_return_none(self):
        s = _make_scheduler()
        self.assertIsNone(s._next_job_weighted_rr())

    def test_single_planner_returns_in_fifo_order(self):
        s = _make_scheduler()
        s.register_planner("p1", weight=1.0)
        s.enqueue(_make_job(planner="p1", scenario="a"))
        s.enqueue(_make_job(planner="p1", scenario="b"))
        first = s._next_job_weighted_rr()
        second = s._next_job_weighted_rr()
        self.assertEqual(first.scenario_name, "a")
        self.assertEqual(second.scenario_name, "b")

    def test_higher_weight_dispatched_more_often(self):
        s = _make_scheduler()
        s.register_planner("fast", weight=4.0)   # 1/w = 0.25
        s.register_planner("slow", weight=1.0)   # 1/w = 1.0
        for i in range(50):
            s.enqueue(_make_job(planner="fast", scenario=f"f{i}"))
            s.enqueue(_make_job(planner="slow", scenario=f"s{i}"))
        # Sample only the first 20 picks; WFQ bias shows in this window,
        # beyond it both queues drain and totals converge.
        picks: list[str] = []
        for _ in range(20):
            j = s._next_job_weighted_rr()
            self.assertIsNotNone(j)
            picks.append(j.planner_name)
        fast_count = picks.count("fast")
        slow_count = picks.count("slow")
        self.assertGreater(fast_count, slow_count)

    def test_enqueue_unregistered_planner_raises(self):
        s = _make_scheduler()
        with self.assertRaises(KeyError):
            s.enqueue(_make_job(planner="ghost"))


# ----------------------------------------------------------------------
# VRAM budget calculation
# ----------------------------------------------------------------------


class EffectiveBudgetTests(unittest.TestCase):
    def test_stale_snapshot_returns_none(self):
        mon = _FakeGpuMonitor({0: 24000}, fresh=False)
        s = _make_scheduler(monitor=mon, gpu_ids=(0,))
        self.assertIsNone(s._effective_budget_mib(0))

    def test_missing_gpu_returns_none(self):
        mon = _FakeGpuMonitor({0: 24000})
        s = _make_scheduler(monitor=mon, gpu_ids=(0, 1))
        # GPU 1 is not in the fake monitor's dict -> stats_for returns None.
        self.assertIsNone(s._effective_budget_mib(1))

    def test_deducts_safety_buffer(self):
        mon = _FakeGpuMonitor({0: 10000})
        s = _make_scheduler(monitor=mon, gpu_ids=(0,), safety_buffer=1500)
        self.assertEqual(s._effective_budget_mib(0), 10000 - 1500)

    def test_deducts_reserved(self):
        mon = _FakeGpuMonitor({0: 10000}, used_per_gpu={0: 0})
        s = _make_scheduler(monitor=mon, gpu_ids=(0,), safety_buffer=1000)
        s._reserve_gpu_vram(0, 4000)
        self.assertEqual(s._effective_budget_mib(0), 10000 - 1000 - 4000)

    def test_only_deducts_unobserved_reserved(self):
        mon = _FakeGpuMonitor({0: 10000}, used_per_gpu={0: 1000})
        s = _make_scheduler(monitor=mon, gpu_ids=(0,), safety_buffer=1000)
        s._reserve_gpu_vram(0, 4000)
        mon.set_used(0, 5000)
        self.assertEqual(s._effective_budget_mib(0), 10000 - 1000)

    def test_applies_oom_multiplier(self):
        mon = _FakeGpuMonitor({0: 10000})
        s = _make_scheduler(monitor=mon, gpu_ids=(0,), safety_buffer=0)
        s._gpu_oom_multiplier[0] = 0.5
        self.assertEqual(s._effective_budget_mib(0), 5000)

    def test_negative_budget_floored_at_zero(self):
        mon = _FakeGpuMonitor({0: 1000})
        s = _make_scheduler(monitor=mon, gpu_ids=(0,), safety_buffer=2000)
        self.assertEqual(s._effective_budget_mib(0), 0)


# ----------------------------------------------------------------------
# OOM backoff
# ----------------------------------------------------------------------


class OomBackoffTests(unittest.TestCase):
    def test_record_oom_reduces_multiplier(self):
        s = _make_scheduler()
        s.record_oom(0)
        self.assertLess(s._gpu_oom_multiplier[0], 1.0)

    def test_record_oom_floor_at_0_3(self):
        s = _make_scheduler()
        for _ in range(100):
            s.record_oom(0)
        self.assertGreaterEqual(s._gpu_oom_multiplier[0], 0.3 - 1e-9)

    def test_record_success_resets_counter_on_oom(self):
        s = _make_scheduler()
        s.record_success(0)
        s.record_success(0)
        self.assertEqual(s._gpu_successes_since_oom[0], 2)
        s.record_oom(0)
        self.assertEqual(s._gpu_successes_since_oom[0], 0)

    def test_repeated_success_recovers_after_threshold(self):
        s = _make_scheduler()
        s.record_oom(0)
        reduced = s._gpu_oom_multiplier[0]
        self.assertLess(reduced, 1.0)
        # oom_backoff_recover_count defaults to 5 -> 5 successes raise the mul.
        for _ in range(5):
            s.record_success(0)
        self.assertGreater(s._gpu_oom_multiplier[0], reduced)


# ----------------------------------------------------------------------
# Dispatch decision tree
# ----------------------------------------------------------------------


class PickGpuAndDecideCarlaTests(unittest.TestCase):
    def test_no_fresh_snapshot_returns_none(self):
        mon = _FakeGpuMonitor({0: 24000}, fresh=False)
        s = _make_scheduler(monitor=mon)
        s.register_planner("p1")
        self.assertIsNone(s._pick_gpu_and_decide_carla(_make_job()))

    def test_priority1_same_town_warm_instance(self):
        pool = _FakeCarlaPool()
        pool.instances.append(_FakePoolInstance("i1", gpu_id=0, current_town="Town01"))
        mon = _FakeGpuMonitor({0: 10000})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0)
        s.register_planner("p1")
        pick = s._pick_gpu_and_decide_carla(_make_job(town="Town01", vram_mib=4000))
        self.assertEqual(pick, (0, False))

    def test_priority2_any_town_warm_instance(self):
        pool = _FakeCarlaPool()
        pool.instances.append(_FakePoolInstance("i1", gpu_id=0, current_town="Town05"))
        mon = _FakeGpuMonitor({0: 10000})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0)
        s.register_planner("p1")
        pick = s._pick_gpu_and_decide_carla(_make_job(town="Town01", vram_mib=4000))
        self.assertEqual(pick, (0, False))

    def test_priority3_start_new_carla(self):
        # No warm instance; enough headroom for planner+CARLA.
        pool = _FakeCarlaPool()
        mon = _FakeGpuMonitor({0: 20000})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0, carla_vram=6000)
        s.register_planner("p1")
        pick = s._pick_gpu_and_decide_carla(_make_job(vram_mib=4000))
        self.assertEqual(pick, (0, True))

    def test_no_capacity_returns_none(self):
        pool = _FakeCarlaPool()
        # 2000 MiB free -> not enough for 4000 planner alone.
        mon = _FakeGpuMonitor({0: 2000})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0, carla_vram=6000)
        s.register_planner("p1")
        self.assertIsNone(s._pick_gpu_and_decide_carla(_make_job(vram_mib=4000)))

    def test_prefers_warm_over_new_even_when_both_fit(self):
        pool = _FakeCarlaPool()
        pool.instances.append(_FakePoolInstance("i1", gpu_id=0, current_town="Town01"))
        mon = _FakeGpuMonitor({0: 40000})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0, carla_vram=6000)
        s.register_planner("p1")
        pick = s._pick_gpu_and_decide_carla(_make_job(town="Town01", vram_mib=4000))
        # Warm reuse path -> need_new_carla = False.
        self.assertEqual(pick[1], False)

    def test_picks_gpu_with_highest_headroom(self):
        pool = _FakeCarlaPool()
        mon = _FakeGpuMonitor({0: 8000, 1: 20000})
        s = _make_scheduler(pool=pool, monitor=mon, gpu_ids=(0, 1), safety_buffer=0, carla_vram=6000)
        s.register_planner("p1")
        pick = s._pick_gpu_and_decide_carla(_make_job(vram_mib=4000))
        self.assertEqual(pick, (1, True))

    def test_reservation_blocks_second_dispatch(self):
        pool = _FakeCarlaPool()
        mon = _FakeGpuMonitor({0: 12000}, used_per_gpu={0: 0})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=0, carla_vram=6000)
        s.register_planner("p1")
        # First dispatch: 4000 planner + 6000 CARLA = 10000 fits in 12000.
        first = s._pick_gpu_and_decide_carla(_make_job(vram_mib=4000))
        self.assertIsNotNone(first)
        # Simulate the main loop reserving.
        s._reserve_gpu_vram(0, 10000)
        # Second dispatch should be refused (only 2000 free now).
        second = s._pick_gpu_and_decide_carla(_make_job(vram_mib=4000))
        self.assertIsNone(second)

    def test_observed_growth_stops_double_counting_prior_launch(self):
        pool = _FakeCarlaPool()
        mon = _FakeGpuMonitor({0: 12000}, used_per_gpu={0: 0})
        s = _make_scheduler(pool=pool, monitor=mon, safety_buffer=1000, carla_vram=6000)
        s.register_planner("p1")
        s._reserve_gpu_vram(0, 10000)
        mon.set_used(0, 10000)
        pick = s._pick_gpu_and_decide_carla(_make_job(vram_mib=3500))
        self.assertEqual(pick, (0, True))


# ----------------------------------------------------------------------
# VRAM cost calculation
# ----------------------------------------------------------------------


class VramCostTests(unittest.TestCase):
    def test_cost_includes_carla_when_new(self):
        s = _make_scheduler(carla_vram=6000)
        cost = s._vram_cost_for_job(_make_job(vram_mib=4000), need_new_carla=True)
        self.assertEqual(cost, 10000)

    def test_cost_excludes_carla_when_warm(self):
        s = _make_scheduler(carla_vram=6000)
        cost = s._vram_cost_for_job(_make_job(vram_mib=4000), need_new_carla=False)
        self.assertEqual(cost, 4000)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class MaxLeasesPerGpuTests(unittest.TestCase):
    """Per-GPU concurrency cap must exclude saturated GPUs from dispatch
    even when their VRAM budget is still positive.  The live system wedged
    when 6+ warm CARLAs on one GPU caused sim-speed collapse + heartbeat
    kills, despite VRAM alone being available."""

    def test_cap_excludes_saturated_gpu(self):
        pool = _FakeCarlaPool()
        # Two warm CARLAs on gpu=0, empty on gpu=1.
        pool.instances = [
            _FakePoolInstance("inst-a", 0, "Town01"),
            _FakePoolInstance("inst-b", 0, "Town01"),
        ]
        monitor = _FakeGpuMonitor({0: 24000, 1: 24000})
        s = _make_scheduler(pool=pool, monitor=monitor, gpu_ids=(0, 1))
        # Artificially inflate gpu=0's active-lease counter to the cap.
        s._max_leases_per_gpu = 2
        s._gpu_active_leases[0] = 2
        job = _make_job(vram_mib=4000)
        placement = s._pick_gpu_and_decide_carla(job)
        # Must NOT dispatch to gpu=0 (capped) — only gpu=1 is eligible.
        self.assertIsNotNone(placement)
        self.assertEqual(placement[0], 1)

    def test_disabled_cap_allows_overscheduling(self):
        pool = _FakeCarlaPool()
        pool.instances = [_FakePoolInstance("inst-a", 0, "Town01")]
        monitor = _FakeGpuMonitor({0: 24000})
        s = _make_scheduler(pool=pool, monitor=monitor, gpu_ids=(0,))
        s._max_leases_per_gpu = None  # disabled
        s._gpu_active_leases[0] = 99
        job = _make_job(vram_mib=4000)
        placement = s._pick_gpu_and_decide_carla(job)
        self.assertIsNotNone(placement)
        self.assertEqual(placement[0], 0)


class OomIdleRecoveryTests(unittest.TestCase):
    """After all small-VRAM planners drain, a knocked-down GPU must not
    stay wedged at the multiplier floor: the idle-recovery probe nudges
    it back toward 1.0 so larger planners can be admitted again.
    Prevents the observed `running=0 with 736 queued` deadlock."""

    def test_idle_recovery_restores_multiplier_when_gpu_is_idle(self):
        monitor = _FakeGpuMonitor({0: 20000})
        s = _make_scheduler(monitor=monitor, gpu_ids=(0,))
        # Simulate a knocked-down GPU with nothing running.
        s._gpu_oom_multiplier[0] = 0.33
        s._gpu_active_leases[0] = 0
        import time as _time
        s._gpu_last_oom_monotonic[0] = _time.monotonic() - 1000.0
        s._oom_multiplier_recovery_idle_s = 10.0
        # Calling _effective_budget_mib should trigger the probe.
        _ = s._effective_budget_mib(0)
        self.assertGreater(s._gpu_oom_multiplier[0], 0.33)

    def test_idle_recovery_skipped_when_gpu_is_active(self):
        monitor = _FakeGpuMonitor({0: 20000})
        s = _make_scheduler(monitor=monitor, gpu_ids=(0,))
        s._gpu_oom_multiplier[0] = 0.33
        s._gpu_active_leases[0] = 1  # still running
        import time as _time
        s._gpu_last_oom_monotonic[0] = _time.monotonic() - 1000.0
        s._oom_multiplier_recovery_idle_s = 10.0
        _ = s._effective_budget_mib(0)
        self.assertEqual(s._gpu_oom_multiplier[0], 0.33)


class ActiveLeaseCountingTests(unittest.TestCase):
    def test_reserve_and_release_track_lease_count(self):
        monitor = _FakeGpuMonitor({0: 24000})
        s = _make_scheduler(monitor=monitor, gpu_ids=(0,))
        self.assertEqual(s._gpu_active_leases[0], 0)
        s._reserve_gpu_vram(0, 4000)
        self.assertEqual(s._gpu_active_leases[0], 1)
        s._reserve_gpu_vram(0, 4000)
        self.assertEqual(s._gpu_active_leases[0], 2)
        s._release_gpu_vram(0, 4000)
        self.assertEqual(s._gpu_active_leases[0], 1)
        s._release_gpu_vram(0, 4000)
        self.assertEqual(s._gpu_active_leases[0], 0)


class ComputePlannerWeightTests(unittest.TestCase):
    def test_faster_planner_gets_higher_weight(self):
        fast = sch.compute_planner_weight(60.0)
        slow = sch.compute_planner_weight(300.0)
        self.assertGreater(fast, slow)

    def test_floor_protects_against_tiny_mean(self):
        # mean=1s would yield weight=1.0, floor=60 yields weight=1/60.
        w = sch.compute_planner_weight(1.0, floor=60.0)
        self.assertAlmostEqual(w, 1.0 / 60.0)


if __name__ == "__main__":
    unittest.main()
