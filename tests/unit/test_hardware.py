"""
What the machine reports, and what a thread pool should believe.

Every branch here is one CI cannot reach on the machine it runs on — an arm64
Mac's performance-core split, a cgroup quota — which is exactly why the numbers
are worth pinning: a pool sized from the wrong count is not an error anywhere,
it is a process that is quietly slower than the hardware it paid for.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dsrag.utils import hardware


class TestIsAppleSilicon(unittest.TestCase):
    def test_true_on_an_arm64_mac(self):
        with mock.patch("platform.system", return_value="Darwin"), mock.patch(
            "platform.machine", return_value="arm64"
        ):
            self.assertTrue(hardware.is_apple_silicon())

    def test_false_on_an_intel_mac(self):
        with mock.patch("platform.system", return_value="Darwin"), mock.patch(
            "platform.machine", return_value="x86_64"
        ):
            self.assertFalse(hardware.is_apple_silicon())

    def test_false_in_a_linux_container_on_an_arm64_mac(self):
        """
        The container is the case that matters most and reads as Linux.

        A Docker container on an Apple Silicon host has no access to the host's
        GPU or Neural Engine, so a provider chosen from the HOST's hardware
        would be chosen for accelerators this process cannot reach.
        """
        with mock.patch("platform.system", return_value="Linux"), mock.patch(
            "platform.machine", return_value="aarch64"
        ):
            self.assertFalse(hardware.is_apple_silicon())


class TestPerformanceCores(unittest.TestCase):
    def test_performance_cores_only_on_apple_silicon(self):
        """
        Efficiency cores set the pace of a parallel-for rather than adding to
        it, so a 6P/6E machine wants six threads and not twelve.
        """
        with mock.patch.object(
            hardware, "is_apple_silicon", return_value=True
        ), mock.patch.object(hardware, "_sysctl", return_value=6):
            self.assertEqual(hardware.performance_cores(), 6)

    def test_falls_back_when_sysctl_says_nothing(self):
        with mock.patch.object(
            hardware, "is_apple_silicon", return_value=True
        ), mock.patch.object(hardware, "_sysctl", return_value=0), mock.patch.object(
            hardware, "usable_cores", return_value=3
        ):
            self.assertEqual(hardware.performance_cores(), 3)

    def test_usable_cores_elsewhere(self):
        with mock.patch.object(
            hardware, "is_apple_silicon", return_value=False
        ), mock.patch.object(hardware, "usable_cores", return_value=4):
            self.assertEqual(hardware.performance_cores(), 4)

    def test_a_real_machine_reports_at_least_one(self):
        self.assertGreaterEqual(hardware.performance_cores(), 1)


class TestUsableCores(unittest.TestCase):
    def test_the_cgroup_quota_wins_over_the_host_count(self):
        """
        The container case: sixteen cores on the box, two admitted to this
        process, and onnxruntime's own default would size for sixteen.
        """
        with mock.patch("os.cpu_count", return_value=16), mock.patch.object(
            hardware, "_cgroup_quota", return_value=2
        ):
            self.assertEqual(hardware.usable_cores(), 2)

    def test_no_quota_leaves_the_machine_count(self):
        with mock.patch("os.cpu_count", return_value=8), mock.patch.object(
            hardware, "_cgroup_quota", return_value=0
        ), mock.patch.object(os, "sched_getaffinity", lambda _pid: set(range(8))):
            self.assertEqual(hardware.usable_cores(), 8)

    def test_never_zero(self):
        with mock.patch("os.cpu_count", return_value=None), mock.patch.object(
            hardware, "_cgroup_quota", return_value=0
        ), mock.patch.object(os, "sched_getaffinity", lambda _pid: set()):
            self.assertEqual(hardware.usable_cores(), 1)


class TestCgroupQuota(unittest.TestCase):
    def test_reads_cgroup_v2(self):
        with mock.patch.object(hardware, "_read_text", side_effect=["150000 100000"]):
            # 1.5 cores rounds UP: a pool of one would leave half a core idle.
            self.assertEqual(hardware._cgroup_quota(), 2)

    def test_unlimited_v2_is_no_limit(self):
        with mock.patch.object(hardware, "_read_text", side_effect=["max 100000", "", ""]):
            self.assertEqual(hardware._cgroup_quota(), 0)

    def test_falls_back_to_cgroup_v1(self):
        with mock.patch.object(hardware, "_read_text", side_effect=["", "400000", "100000"]):
            self.assertEqual(hardware._cgroup_quota(), 4)

    def test_v1_without_a_limit_is_no_limit(self):
        with mock.patch.object(hardware, "_read_text", side_effect=["", "-1", "100000"]):
            self.assertEqual(hardware._cgroup_quota(), 0)


if __name__ == "__main__":
    unittest.main()
