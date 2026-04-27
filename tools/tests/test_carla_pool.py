"""Phase 1 smoke tests for :mod:`tools._carla_pool`.

These tests exercise the pool's pure logic -- port-band picking, state
transitions, lease accounting, collision avoidance -- without launching an
actual CARLA process.  Real end-to-end validation of CARLA startup is
performed via ``--scenario-pool`` in live runs.

Run with:  python -m pytest tools/tests/test_carla_pool.py -q
(or simply ``python tools/tests/test_carla_pool.py`` which calls unittest.main)
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock


# Make `tools` importable when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tools import _carla_pool as cp  # noqa: E402


class _FakeManager:
    """Stand-in for :class:`tools.run_custom_eval.CarlaProcessManager`."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.raise_on_start = False
        self.port = kwargs["port"]

    def start(self):
        if self.raise_on_start:
            raise RuntimeError("fake start failure")
        self.started = True
        return types.SimpleNamespace(port=self.kwargs["port"], tm_port=self.kwargs["port"] + 2)

    def stop(self, timeout=15.0):
        self.stopped = True


def _install_fake_rce(
    *,
    ports_available_fn=lambda host, ports: True,
    manager_cls=_FakeManager,
):
    """Install a fake ``run_custom_eval`` module for the pool's lazy import.

    Returns the fake module so a test can inspect its interactions.
    """
    fake = types.ModuleType("tools.run_custom_eval")
    fake.CarlaProcessManager = manager_cls
    fake.carla_service_ports = lambda port, offset: [port, port + offset]
    fake.are_ports_available = ports_available_fn
    fake.wait_for_port_close = lambda host, port, timeout: True
    fake.wait_for_carla_post_start_buffer = lambda buf: None
    # Lease-time RPC liveness probe (added for the pool health-check
    # hardening); the fake returns True so the test path behaves the
    # same as before the probe was wired in.
    fake.wait_for_carla_rpc_ready = lambda host, port, timeout=5.0, quiet=True: True
    sys.modules["tools.run_custom_eval"] = fake
    # Also install as an attribute of the ``tools`` package so
    # ``import tools.run_custom_eval as _m`` in the pool's lazy helper
    # picks up the stub even if a sibling test already imported the real
    # module earlier in the pytest session.
    try:
        import tools as _tools_pkg  # type: ignore[import]
        setattr(_tools_pkg, "run_custom_eval", fake)
    except Exception:
        pass
    return fake


class DefaultPortBandTests(unittest.TestCase):
    def test_layout_per_gpu(self):
        band0 = cp.default_port_band(0)
        band1 = cp.default_port_band(1)
        self.assertEqual(band0[0], (2000, 2002))
        # Adjacent world ports differ by the stride (default 4).
        self.assertEqual(band0[1][0] - band0[0][0], 4)
        # GPU N's band starts per_gpu_span after GPU N-1's base (default 512).
        self.assertEqual(band1[0][0] - band0[0][0], 512)
        # Each entry is (world_port, world_port+2).
        for world_port, tm_port in band0:
            self.assertEqual(tm_port, world_port + 2)
        # Slot count matches the documented default.
        self.assertEqual(len(band0), 64)

    def test_bands_do_not_overlap(self):
        band0_ports = {p for pair in cp.default_port_band(0) for p in pair}
        band1_ports = {p for pair in cp.default_port_band(1) for p in pair}
        self.assertFalse(band0_ports & band1_ports)


class FindAvailableInstanceTests(unittest.TestCase):
    def test_empty_pool_returns_none(self):
        _install_fake_rce()
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"))
        self.assertIsNone(pool.find_available_instance())

    def test_clean_same_town_preferred(self):
        _install_fake_rce()
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"))
        # Manually inject instances to bypass start_instance.
        a = cp.PoolInstance(
            instance_id="a", gpu_id=0, host="127.0.0.1",
            world_port=2000, tm_port=2002, manager=None,
            state=cp.InstanceState.CLEAN, current_town="Town05",
        )
        b = cp.PoolInstance(
            instance_id="b", gpu_id=0, host="127.0.0.1",
            world_port=2004, tm_port=2006, manager=None,
            state=cp.InstanceState.CLEAN, current_town="Town01",
        )
        pool._instances["a"] = a
        pool._instances["b"] = b
        pick = pool.find_available_instance(town="Town01")
        self.assertEqual(pick.instance_id, "b")

    def test_leased_instances_skipped(self):
        _install_fake_rce()
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"))
        a = cp.PoolInstance(
            instance_id="a", gpu_id=0, host="127.0.0.1",
            world_port=2000, tm_port=2002, manager=None,
            state=cp.InstanceState.LEASED, current_town="Town01",
        )
        pool._instances["a"] = a
        self.assertIsNone(pool.find_available_instance(town="Town01"))


class LeaseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.fake = _install_fake_rce()
        self.pool = cp.CarlaPool(carla_root=Path("/nonexistent"))

    def test_start_instance_happy_path(self):
        inst = self.pool.start_instance(gpu_id=0)
        self.assertIsNotNone(inst)
        self.assertEqual(inst.gpu_id, 0)
        self.assertEqual(inst.state, cp.InstanceState.IDLE)
        self.assertTrue(inst.manager.started)
        self.assertEqual(inst.manager.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_start_instance_no_free_port(self):
        _install_fake_rce(ports_available_fn=lambda host, ports: False)
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"))
        self.assertIsNone(pool.start_instance(gpu_id=0))

    def test_port_collision_avoided_within_same_pool(self):
        """Two start_instance calls on the same GPU must pick different ports."""
        a = self.pool.start_instance(gpu_id=0)
        b = self.pool.start_instance(gpu_id=0)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a.world_port, b.world_port)

    def test_start_failure_propagates_as_none(self):
        class FailingManager(_FakeManager):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.raise_on_start = True

        _install_fake_rce(manager_cls=FailingManager)
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"))
        self.assertIsNone(pool.start_instance(gpu_id=0))
        # Failed instance must not linger.
        self.assertEqual(pool.list_instances(), [])

    def test_acquire_release_clean_keeps_instance(self):
        inst = self.pool.start_instance(gpu_id=0)
        lease = self.pool.acquire_lease(inst.instance_id, town="Town01")
        self.assertIsNotNone(lease)
        self.assertEqual(lease.current_town, "Town01")
        self.assertEqual(inst.state, cp.InstanceState.LEASED)
        self.pool.release_lease(lease, status="clean")
        self.assertEqual(inst.state, cp.InstanceState.CLEAN)
        self.assertEqual(inst.scenarios_served, 1)
        # Second acquire must succeed against the same instance.
        lease2 = self.pool.acquire_lease(inst.instance_id, town="Town01")
        self.assertIsNotNone(lease2)
        self.assertNotEqual(lease.lease_id, lease2.lease_id)

    def test_acquire_rejects_leased_instance(self):
        inst = self.pool.start_instance(gpu_id=0)
        self.pool.acquire_lease(inst.instance_id, town="Town01")
        # Second acquire on the same LEASED instance must fail.
        self.assertIsNone(self.pool.acquire_lease(inst.instance_id, town="Town01"))

    def test_release_crashed_removes_instance(self):
        inst = self.pool.start_instance(gpu_id=0)
        lease = self.pool.acquire_lease(inst.instance_id, town="Town01")
        self.pool.release_lease(lease, status="crashed")
        # stop_instance should have been called.
        self.assertTrue(inst.manager.stopped)
        self.assertNotIn(inst.instance_id, [i.instance_id for i in self.pool.list_instances()])

    def test_double_release_is_ignored(self):
        inst = self.pool.start_instance(gpu_id=0)
        lease = self.pool.acquire_lease(inst.instance_id, town="Town01")
        self.pool.release_lease(lease, status="clean")
        # Second release with the same lease -- pool must log+ignore.
        self.pool.release_lease(lease, status="clean")
        # State must remain CLEAN.
        self.assertEqual(inst.state, cp.InstanceState.CLEAN)

    def test_no_reuse_mode_recycles_on_clean_release(self):
        """In no_reuse mode, even a clean release should kill the instance."""
        _install_fake_rce()
        pool = cp.CarlaPool(carla_root=Path("/nonexistent"), no_reuse=True)
        inst = pool.start_instance(gpu_id=0)
        lease = pool.acquire_lease(inst.instance_id, town="Town01")
        pool.release_lease(lease, status="clean")
        # Instance must be gone (stopped + removed from pool).
        self.assertTrue(inst.manager.stopped)
        self.assertNotIn(inst.instance_id, [i.instance_id for i in pool.list_instances()])

    def test_shutdown_stops_all_instances(self):
        a = self.pool.start_instance(gpu_id=0)
        b = self.pool.start_instance(gpu_id=0)
        self.pool.shutdown()
        self.assertTrue(a.manager.stopped)
        self.assertTrue(b.manager.stopped)
        self.assertEqual(self.pool.list_instances(), [])

    def test_status_callback_emits_lifecycle_events(self):
        events = []

        def _cb(event, payload):
            events.append((event, payload))

        _install_fake_rce()
        pool = cp.CarlaPool(
            carla_root=Path("/nonexistent"),
            status_callback=_cb,
        )
        inst = pool.start_instance(gpu_id=0)
        lease = pool.acquire_lease(inst.instance_id, town="Town01")
        pool.release_lease(lease, status="clean")
        pool.shutdown()

        event_names = [name for name, _payload in events]
        self.assertIn("carla_start_begin", event_names)
        self.assertIn("carla_start_ready", event_names)
        self.assertIn("carla_lease_acquired", event_names)
        self.assertIn("carla_lease_released", event_names)
        self.assertIn("carla_stopped", event_names)


if __name__ == "__main__":
    unittest.main()
