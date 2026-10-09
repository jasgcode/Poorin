# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What exists at an artifact's output path: nothing, a started build, a finished one, or one
built from another recipe.

The plan that ``poorin.experiment.cli`` prints and the drift check in
:func:`poorin.execution.lazy.run` both read an artifact's state here, so they agree on what a run
will do. Handles are typed by a protocol so that ``lazy`` can import this module.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rigging.filesystem.cluster_config import poorin_prefix
from rigging.filesystem.storage_path import prefix_join

from poorin.execution.artifact import is_mutable_version, recorded_recipe
from poorin.execution.fingerprint import describe_drift, fingerprint_hash
from poorin.execution.step_status import STATUS_SUCCESS, StatusFile


class Handle(Protocol):
    """The parts of an ``ArtifactStep`` the state check reads."""

    name: str
    version: str
    override_path: str | None

    def fingerprint_payload(self) -> str: ...


class CacheState(StrEnum):
    PINNED = "pinned"
    """Points at existing data through ``override_path``; not checked."""
    MUTABLE = "mutable"
    """A dev version: always builds, and a resumed one continues its checkpoints."""
    NOT_BUILT = "not built"
    STARTED = "started"
    """A calendar version whose build started but did not succeed: the build restarts."""
    BUILT = "built"
    DRIFTED = "built from another recipe"


@dataclass(frozen=True)
class HandleState:
    handle: Handle
    fingerprint: str
    state: CacheState
    recorded_fingerprint: str | None
    change: str
    """The field-level diff from :func:`describe_drift`; empty unless ``DRIFTED``."""

    @property
    def label(self) -> str:
        return f"{self.handle.name}@{self.handle.version}"


def record_path(handle: Handle) -> str:
    """Where ``handle``'s status and record live: ``{prefix}/{name}/{version}``, an adopted handle
    included (its data stays at the source; its record does not)."""
    return prefix_join(poorin_prefix(), f"{handle.name}/{handle.version}")


def handle_state(handle: Handle) -> HandleState:
    """``handle``'s state, from the recipe and status recorded at :func:`record_path`."""
    payload = handle.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)
    if handle.override_path is not None:
        return HandleState(handle, fingerprint, CacheState.PINNED, None, "")
    path = record_path(handle)
    recorded = recorded_recipe(path)
    if recorded is not None and recorded.fingerprint != fingerprint:
        change = describe_drift(recorded.payload, payload)
        return HandleState(handle, fingerprint, CacheState.DRIFTED, recorded.fingerprint, change)
    recorded_fingerprint = recorded.fingerprint if recorded is not None else None
    if is_mutable_version(handle.version):
        return HandleState(handle, fingerprint, CacheState.MUTABLE, recorded_fingerprint, "")
    if StatusFile(path, worker_id="plan").status == STATUS_SUCCESS:
        return HandleState(handle, fingerprint, CacheState.BUILT, recorded_fingerprint, "")
    if recorded is not None:
        return HandleState(handle, fingerprint, CacheState.STARTED, recorded_fingerprint, "")
    return HandleState(handle, fingerprint, CacheState.NOT_BUILT, None, "")
