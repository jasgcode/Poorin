# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections.abc import Callable
from typing import TypeVar

from fray.cluster import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")

# Runtime-tuning env vars forwarded from the dispatcher to the train tasks, for
# a backend whose tasks don't inherit the submitter's shell. LocalClient runs the
# task in-process and ignores the forwarded environment.
# JAX_PLATFORMS is excluded: the dispatcher runs CPU-only and its value must
# not leak onto accelerator tasks.
_FORWARDED_ENV_PREFIXES = ("XLA_", "LIBTPU_INIT_ARGS", "NCCL_", "JAX_", "MALLOC_")
_FORWARDED_ENV_NAMES = ("LD_PRELOAD",)
_FORWARDED_ENV_EXCLUDE = ("JAX_PLATFORMS",)


def _forwarded_env_vars() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if (k in _FORWARDED_ENV_NAMES or k.startswith(_FORWARDED_ENV_PREFIXES)) and k not in _FORWARDED_ENV_EXCLUDE
    }


def _safe_job_suffix(run_id: str) -> str:
    """Sanitize run IDs into Fray-safe job-name suffixes."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id)


def dispatch_grug_training_run(
    *,
    run_id: str,
    config: ConfigT,
    local_entrypoint: Callable[[ConfigT], None],
    resources: ResourceConfig,
    max_retries_failure: int = 3,
    max_task_failures: int = 10,
    processes_per_task: int = 1,
    priority: int = 0,
    extras: list[str] | None = None,
) -> None:
    """Submit a grug train entrypoint through Fray and wait for completion.

    ``priority`` 0 leaves the backend's default priority.

    ``max_retries_failure`` is the per-task retry budget and ``max_task_failures`` is the
    cumulative one. The job fails when either is exhausted, so raise the two together: a large
    per-task budget under a small cumulative one still ends the job at the cumulative limit.

    ``extras`` overrides the dependency extras inferred from the accelerator type.
    """
    safe_run_id = _safe_job_suffix(run_id)
    env_vars = resolve_training_env(base_env=_forwarded_env_vars(), resources=resources)
    request = JobRequest(
        name=f"grug-train-{safe_run_id}",
        entrypoint=Entrypoint.from_callable(local_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(
            env_vars=env_vars,
            extras=extras_for_resources(resources) if extras is None else extras,
        ),
        max_retries_failure=max_retries_failure,
        max_task_failures=max_task_failures,
        processes_per_task=processes_per_task,
        priority=priority,
    )
    logger.info("Dispatching grug training via Fray: %s", request.name)
    job = current_client().submit(request)
    job.wait(raise_on_failure=True)
