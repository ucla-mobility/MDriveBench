from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))


def _evict_fake_rce_module() -> None:
    mod = sys.modules.get("tools.run_custom_eval")
    if mod is not None and not hasattr(mod, "build_child_argv"):
        del sys.modules["tools.run_custom_eval"]


class _FakeProc:
    def __init__(self, pid: int = 123, returncode: int | None = None):
        self.pid = pid
        self._returncode = returncode

    def poll(self) -> int | None:
        return self._returncode


class CarlaGpuBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        _evict_fake_rce_module()
        from tools import run_custom_eval as rce

        self.rce = rce

    def test_parse_vulkaninfo_summary_adapter_uuids_skips_cpu_adapter(self):
        parsed = self.rce._parse_vulkaninfo_summary_adapter_uuids(
            "\n".join(
                [
                    "GPU0:",
                    "\tvendorID = 0x10de",
                    "\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                    "\tdeviceUUID = 7f8fbf37-4305-8840-a3aa-f1b1488b535f",
                    "GPU1:",
                    "\tvendorID = 0x10005",
                    "\tdeviceType = PHYSICAL_DEVICE_TYPE_CPU",
                    "\tdeviceUUID = 6d657361-3233-2e32-2e31-2d3175627500",
                    "GPU2:",
                    "\tvendorID = 0x10de",
                    "\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                    "\tdeviceUUID = e2f2a08f-6146-9a37-5db8-190f3d503188",
                ]
            )
        )
        self.assertEqual(
            parsed,
            {
                0: "7f8fbf37-4305-8840-a3aa-f1b1488b535f",
                2: "e2f2a08f-6146-9a37-5db8-190f3d503188",
            },
        )

    def test_resolve_carla_launch_args_uses_vulkan_adapter_index(self):
        with mock.patch.object(
            self.rce,
            "_query_vulkan_adapter_index_by_gpu_index",
            return_value={"3": 0},
        ):
            resolved = self.rce._resolve_carla_launch_args(
                ["-quality-level=Low"],
                {"CUDA_VISIBLE_DEVICES": "3"},
            )
        self.assertIn("-graphicsadapter=0", resolved)
        self.assertIn("-RenderOffScreen", resolved)

    def test_wait_for_carla_gpu_binding_succeeds_when_expected_gpu_appears(self):
        proc = _FakeProc(pid=123)
        with (
            mock.patch.object(
                self.rce,
                "_resolve_physical_gpu_index_from_env",
                return_value="3",
            ),
            mock.patch.object(
                self.rce,
                "_candidate_carla_gpu_pids",
                return_value=[123],
            ),
            mock.patch.object(
                self.rce,
                "_query_gpu_bindings_by_pid",
                side_effect=[
                    {123: [("0", 6)]},
                    {123: [("3", 1024)]},
                ],
            ),
            mock.patch.object(self.rce.time, "sleep", return_value=None),
        ):
            ok, reason = self.rce.wait_for_carla_gpu_binding(
                proc,
                {"CUDA_VISIBLE_DEVICES": "3"},
                timeout=2.0,
                quiet=True,
            )
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_wait_for_carla_gpu_binding_reports_mismatch(self):
        proc = _FakeProc(pid=123)
        monotonic_values = iter([0.0, 0.0, 0.6, 1.2, 1.8, 2.4])
        with (
            mock.patch.object(
                self.rce,
                "_resolve_physical_gpu_index_from_env",
                return_value="3",
            ),
            mock.patch.object(
                self.rce,
                "_candidate_carla_gpu_pids",
                return_value=[123],
            ),
            mock.patch.object(
                self.rce,
                "_query_gpu_bindings_by_pid",
                return_value={123: [("0", 1536)]},
            ),
            mock.patch.object(self.rce.time, "sleep", return_value=None),
            mock.patch.object(
                self.rce.time,
                "monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
        ):
            ok, reason = self.rce.wait_for_carla_gpu_binding(
                proc,
                {"CUDA_VISIBLE_DEVICES": "3"},
                timeout=2.0,
                quiet=True,
            )
        self.assertFalse(ok)
        self.assertIn("GPU 0", str(reason))
        self.assertIn("requested GPU 3", str(reason))


if __name__ == "__main__":
    unittest.main()
