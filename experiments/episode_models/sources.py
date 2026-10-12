# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The example episode datasets a launcher can name with ``--dataset``, and the fields each template reads."""

from collections.abc import Callable
from dataclasses import dataclass

from poorin.episodes.build import EpisodeDataset
from poorin.execution.lazy import ArtifactStep

from experiments.datasets.episodes.libero import libero_spatial
from experiments.datasets.episodes.ssv2 import ssv2_subset
from experiments.datasets.episodes.vpt import vpt_subset


@dataclass(frozen=True)
class EpisodeSource:
    dataset: Callable[[], ArtifactStep[EpisodeDataset]]
    """Builds the handle only for the chosen source: SSv2's reads a local tree and VPT's the network."""
    video_field: str
    action_field: str | None
    task_key: str | None


SOURCES: dict[str, EpisodeSource] = {
    "libero-spatial": EpisodeSource(
        libero_spatial, video_field="agentview", action_field="action", task_key="instruction"
    ),
    "vpt-10": EpisodeSource(vpt_subset, video_field="video", action_field="action", task_key=None),
    "ssv2-500": EpisodeSource(ssv2_subset, video_field="video", action_field=None, task_key=None),
}
