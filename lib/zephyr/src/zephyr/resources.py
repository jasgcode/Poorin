# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The memory and CPU limits visible to this process.

Salvaged from upstream Marin's ``iris.env_resources`` (git tag ``upstream-baseline``) without its
task-resource environment variable. Limits come from cgroup v2, then cgroup v1, then
``/proc/meminfo`` and ``os.cpu_count()``
with a warning, so a container's limits win over the host's.
"""

import dataclasses
import logging
import os

logger = logging.getLogger(__name__)

# cgroup paths in preference order (v2 first, then v1).
_CGROUP_MEMORY_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)
_CGROUP_CPU_QUOTA_PATH = "/sys/fs/cgroup/cpu.max"  # cgroup v2: "quota period"
_CGROUP_CPU_QUOTA_V1_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_CPU_PERIOD_V1_PATH = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


@dataclasses.dataclass(frozen=True)
class HostResources:
    """Memory and CPU limits for this process; build one with :meth:`from_environment`."""

    memory_bytes: int
    cpu_cores: float

    @staticmethod
    def from_environment() -> "HostResources":
        """Read the limits from cgroups, falling back to ``/proc/meminfo`` and ``os.cpu_count()``."""
        result = HostResources(memory_bytes=_resolve_memory(), cpu_cores=_resolve_cpu())
        logger.info("HostResources: %s", result)
        return result


def _read_cgroup_file(path: str) -> str | None:
    try:
        with open(path) as f:
            value = f.read().strip()
        if value and value != "max":
            return value
    except (FileNotFoundError, PermissionError):
        pass
    return None


def _read_proc_meminfo_total() -> int | None:
    """Read total memory from ``/proc/meminfo`` (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB → bytes
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        pass
    return None


def _resolve_memory() -> int:
    """Memory limit in bytes: cgroup, then ``/proc/meminfo``, then 0 for unknown."""
    for path in _CGROUP_MEMORY_PATHS:
        value = _read_cgroup_file(path)
        if value is not None:
            return int(value)

    total = _read_proc_meminfo_total()
    if total is not None:
        logger.warning("No cgroup memory limit found, falling back to /proc/meminfo: %d bytes", total)
        return total

    logger.warning("Cannot determine memory limit from cgroups or /proc/meminfo")
    return 0


def _resolve_cpu() -> float:
    """CPU limit in cores: cgroup v2, then cgroup v1, then ``os.cpu_count()``."""
    # cgroup v2: "quota period" (e.g., "200000 100000" means 2 cores)
    value = _read_cgroup_file(_CGROUP_CPU_QUOTA_PATH)
    if value is not None:
        parts = value.split()
        if len(parts) == 2 and parts[0] != "max":
            return int(parts[0]) / int(parts[1])

    # cgroup v1
    quota = _read_cgroup_file(_CGROUP_CPU_QUOTA_V1_PATH)
    period = _read_cgroup_file(_CGROUP_CPU_PERIOD_V1_PATH)
    if quota is not None and period is not None and int(quota) > 0:
        return int(quota) / int(period)

    cpus = float(os.cpu_count() or 1)
    logger.warning("No cgroup CPU limit found, falling back to os.cpu_count(): %.1f cores", cpus)
    return cpus
