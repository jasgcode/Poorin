# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The memory limit visible to this process.

Salvaged from upstream Marin's ``iris.env_resources`` (git tag ``upstream-baseline``). The limit
comes from cgroup v2, then cgroup v1, then the host's total memory with a warning, so a
container's limit wins over the host's.
"""

import functools
import logging

import psutil

logger = logging.getLogger(__name__)

# cgroup paths in preference order (v2 first, then v1).
_CGROUP_MEMORY_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)


@functools.cache
def memory_limit_bytes() -> int:
    """This process's memory limit in bytes, resolved once: the cgroup's, else the host's total."""
    for path in _CGROUP_MEMORY_PATHS:
        value = _read_cgroup_file(path)
        if value is not None:
            return int(value)
    total = psutil.virtual_memory().total
    logger.warning("No cgroup memory limit found, using the host's total memory: %d bytes", total)
    return total


def _read_cgroup_file(path: str) -> str | None:
    try:
        with open(path) as f:
            value = f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None
    return value if value and value != "max" else None
