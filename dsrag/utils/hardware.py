"""
What the machine underneath actually offers an on-device model.

Two numbers matter to a local cross-encoder, and the runtime's defaults are
wrong about both on the hardware this is most often run on:

  - HOW MANY THREADS. onnxruntime sizes its thread pool from the core count,
    and an Apple Silicon core count includes efficiency cores. A parallel-for
    across P and E cores together finishes when the SLOWEST thread finishes, so
    the E-cores do not add throughput to a single small graph — they set its
    pace. Counting performance cores only is measurably faster for one
    latency-bound forward pass, which is what reranking is.

  - HOW MANY CORES ARE REALLY THERE. In a container the runtime still reads the
    host's cores while the cgroup admits a fraction of them, so the pool is
    sized for a machine the process cannot use and the threads spend their time
    contending. The scheduler affinity and the cgroup quota are the honest
    numbers, and the smaller of the two is the one to believe.
"""
import os
import platform
import subprocess


def is_apple_silicon() -> bool:
    """True on an arm64 Mac — NOT inside a Linux container running on one."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def performance_cores() -> int:
    """
    Threads worth giving one forward pass. Never less than 1.

    On Apple Silicon this is the performance-core count; everywhere else it is
    the count of cores this process may actually run on.
    """
    if is_apple_silicon():
        cores = _sysctl("hw.perflevel0.logicalcpu")
        if cores:
            return cores
    return usable_cores()


def usable_cores() -> int:
    """
    Cores this process may run on, honouring affinity and the cgroup quota.

    `os.cpu_count()` reports the machine. A container limited to two CPUs on a
    sixteen-core host still reports sixteen, and every pool sized from it
    oversubscribes by a factor of eight.
    """
    counts = [os.cpu_count() or 1]

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        counts.append(len(affinity(0)))

    quota = _cgroup_quota()
    if quota:
        counts.append(quota)

    return max(1, min(counts))


def _sysctl(name: str) -> int:
    """One sysctl integer, or 0 when it cannot be read."""
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if completed.returncode != 0:
        return 0
    try:
        return max(0, int(completed.stdout.strip()))
    except ValueError:
        return 0


def _cgroup_quota() -> int:
    """
    Whole cores the cgroup admits, rounded up, or 0 when there is no limit.

    Both cgroup versions, because a v1 host is still what a good deal of
    on-premise Kubernetes runs.
    """
    v2 = _read_text("/sys/fs/cgroup/cpu.max")
    if v2:
        parts = v2.split()
        if len(parts) == 2 and parts[0] != "max":
            return _cores_from(parts[0], parts[1])

    quota = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and not quota.startswith("-"):
        return _cores_from(quota, period)

    return 0


def _cores_from(quota: str, period: str) -> int:
    try:
        allowance = int(quota) / int(period)
    except (ValueError, ZeroDivisionError):
        return 0
    return max(1, int(allowance + 0.999)) if allowance > 0 else 0


def _read_text(path: str) -> str:
    try:
        with open(path, "r") as handle:
            return handle.read().strip()
    except OSError:
        return ""
