# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordination for sweeps: which target the next training round runs.

Marin ran sweeps as multi-host TPU gangs, with a leader actor on Iris task 0 handing each round's
target to the followers. Poorin runs one host, so every process is the leader of a gang of one
and runs the loop alone; :class:`SweepCoordinator` remains the seam a multi-host dispatcher would
implement.
"""

import logging
from enum import StrEnum
from typing import Protocol

logger = logging.getLogger(__name__)

# Actor name the leader registers and followers resolve.
_LEADER_ACTOR = "sweep_leader"
# Named port the multi-host sweep job must allocate for the leader's server.
_ACTOR_PORT = "actor"


class GangRole(StrEnum):
    """This task's part in a sweep gang."""

    LEADER = "leader"
    """Task 0 (or any single-host worker): races ``step_lock`` and announces rounds."""

    FOLLOWER = "follower"
    """Task 1..N-1 of a multi-host gang: fetches the leader's rounds and mirrors them."""


class SweepCoordinator(Protocol):
    """Leader/follower channel for a sweep gang.

    The leader calls :meth:`publish` then :meth:`wait_for_followers`; followers
    call :meth:`receive`. :meth:`close` releases resources and must be safe to
    call after an error (so it never blocks on followers).
    """

    @property
    def role(self) -> GangRole: ...

    def publish(self, seq: int, target_id: str | None) -> None:
        """Announce round ``seq``. ``target_id is None`` is the terminal stop."""
        ...

    def receive(self, seq: int) -> str | None:
        """Fetch round ``seq``; return its ``target_id`` or ``None`` for stop."""
        ...

    def wait_for_followers(self, seq: int) -> None:
        """Block until every follower has consumed round ``seq`` (leader only)."""
        ...

    def close(self) -> None:
        """Release resources. Safe to call after an error."""
        ...


class SoloLeader:
    """Leader of a gang of one: no actor, announcements go nowhere."""

    @property
    def role(self) -> GangRole:
        return GangRole.LEADER

    def publish(self, seq: int, target_id: str | None) -> None:
        return None

    def receive(self, seq: int) -> str | None:
        raise AssertionError("SoloLeader.receive must never be called")

    def wait_for_followers(self, seq: int) -> None:
        return None

    def close(self) -> None:
        return None


def gang_coordinator() -> SweepCoordinator:
    """The coordinator for this process: always a :class:`SoloLeader` on one host."""
    return SoloLeader()
