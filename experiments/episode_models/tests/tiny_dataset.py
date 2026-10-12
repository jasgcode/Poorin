# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A tiny episode dataset for the templates' tests: 16x16 video, a 3-dim action whose last dimension
never varies, and one of two instructions per episode."""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from poorin.episodes.build import EpisodeBuildConfig, EpisodeDataset, build_episodes
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding

SCHEMA = DatasetSchema(
    step_rate=10.0,
    fields={
        "video": FieldSpec(FieldKind.VIDEO, "uint8", (16, 16, 3), VideoEncoding(keyframe_interval=4)),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (3,), RawEncoding()),
    },
)
READER = "experiments.episode_models.tests.tiny_dataset:read"
INSTRUCTIONS = ("pour", "stack")


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """A source is ``<index>:<length>``."""
    index, length = (int(part) for part in source.split(":"))
    steps = np.arange(length)
    frames = np.broadcast_to((steps * 8 + index * 16).astype(np.uint8)[:, None, None, None], (length, 16, 16, 3))
    actions = np.stack([np.sin(steps + index), np.cos(steps), np.full(length, 0.5)], axis=1).astype(np.float32)
    yield Episode(
        id=f"e{index}",
        fields={"video": frames.copy(), "action": actions},
        metadata={"instruction": INSTRUCTIONS[index % 2]},
    )


def build_tiny(tmp_path, episodes: int = 10, length: int = 16) -> EpisodeDataset:
    config = EpisodeBuildConfig(
        output_path=str(tmp_path / "tiny"),
        sources=tuple(f"{i}:{length}" for i in range(episodes)),
        reader=READER,
        reader_options={},
        schema=SCHEMA,
        num_shards=2,
    )
    build_episodes(config)
    return EpisodeDataset.raw_load(config.output_path)
