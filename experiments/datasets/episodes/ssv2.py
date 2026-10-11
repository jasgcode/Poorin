# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Something-Something v2 clips as episodes without actions: video, and the label as metadata."""

import functools
import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, VideoEncoding
from poorin.episodes.video_files import load_video
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

SSV2_ROOT = "/media/johnnyas/D/research-storage/ssv2"
VIDEO_DIR = "20bn-something-something-v2"
LABELS = "train.json"
STEP_RATE = 12.0  # SSv2's frame rate
SIZE = (128, 128)
SUBSET_CLIPS = 500
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={"video": FieldSpec(FieldKind.VIDEO, "uint8", (*SIZE, 3), VideoEncoding(keyframe_interval=16))},
)


@functools.cache
def _labels(root: str) -> dict[str, str]:
    with open(os.path.join(root, LABELS)) as f:
        return {item["id"]: item["label"] for item in json.load(f)}


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One SSv2 clip id."""
    root = options["root"]
    video = load_video(os.path.join(root, VIDEO_DIR, f"{source}.webm"), size=SIZE, step_rate=STEP_RATE)
    yield Episode(id=source, fields={"video": video}, metadata={"label": _labels(root)[source]})


def ssv2_subset(root: str = SSV2_ROOT) -> ArtifactStep[EpisodeDataset]:
    ids = sorted(_labels(root))[:SUBSET_CLIPS]
    return episodes(
        "examples/ssv2-500",
        sources=ids,
        reader="experiments.datasets.episodes.ssv2:read",
        reader_options={"root": root},
        schema=SCHEMA,
        num_shards=16,
        version="2026.10.11",
    )
