# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an episode dataset: read source units into episodes, encode their fields, and write
ArrayRecord shards, ``episodes.parquet``, and ``dataset.json``.

The build is one Zephyr stage. Each shard reads its group of source units, so the large decoded
arrays never pass between stages.
"""

import importlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import msgpack
import numpy as np
import pandas as pd
from array_record.python.array_record_module import ArrayRecordWriter
from fray.types import ResourceConfig
from rigging.filesystem.factory import open_url
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo

from poorin.episodes.encodings import encode_field
from poorin.episodes.schema import DatasetSchema, Episode, validate_episode
from poorin.execution.artifact import Artifact

SCHEMA_FILENAME = "dataset.json"
INDEX_FILENAME = "episodes.parquet"
SHARD_PATTERN = "shard-{:05d}.array_record"
# One record per group: every record is read on its own, by index.
ARRAY_RECORD_OPTIONS = "group_size:1"
WORKER_RESOURCES = ResourceConfig(cpu=1, ram="16g")

Reader = Callable[[str, Mapping[str, Any]], Iterator[Episode]]


@dataclass(frozen=True)
class EpisodeBuildConfig:
    output_path: str
    sources: tuple[str, ...]
    """Source units, handed to the reader one at a time, in this order."""
    reader: str
    """``module.path:function``, a :data:`Reader`. A string, so the fingerprint can hash it."""
    reader_options: Mapping[str, Any]
    schema: DatasetSchema
    num_shards: int


class EpisodeDataset(Artifact):
    """A built episode dataset. Reads ``dataset.json`` and ``episodes.parquet`` from its path, so a
    pinned dataset with no record reads the same way."""

    @cached_property
    def schema(self) -> DatasetSchema:
        with open_url(os.path.join(self.path, SCHEMA_FILENAME)) as f:
            return DatasetSchema.from_json(json.load(f))

    @cached_property
    def index(self) -> pd.DataFrame:
        with open_url(os.path.join(self.path, INDEX_FILENAME), "rb") as f:
            return pd.read_parquet(f)

    @property
    def shard_paths(self) -> list[str]:
        count = int(self.index["shard"].max()) + 1 if len(self.index) else 0
        return [os.path.join(self.path, SHARD_PATTERN.format(shard)) for shard in range(count)]


def load_reader(path: str) -> Reader:
    module, _, name = path.partition(":")
    if not name:
        raise ValueError(f"a reader is 'module.path:function', not {path!r}")
    return getattr(importlib.import_module(module), name)


def build_episodes(config: EpisodeBuildConfig) -> None:
    groups = [list(group) for group in np.array_split(np.asarray(config.sources, dtype=object), config.num_shards)]
    groups = [group for group in groups if group]
    os.makedirs(config.output_path, exist_ok=True)

    def write(units: Iterator[list[str]], shard: ShardInfo) -> Iterator[list[dict]]:
        yield _write_shard(config, shard.shard_idx, [unit for group in units for unit in group])

    ctx = ZephyrContext(name="episodes", max_workers=min(len(groups), os.cpu_count() or 1), resources=WORKER_RESOURCES)
    rows = [row for shard_rows in ctx.execute(Dataset.from_list(groups).map_shard(write)).results for row in shard_rows]
    _write_index(config, rows)


def _write_shard(config: EpisodeBuildConfig, shard: int, units: list[str]) -> list[dict]:
    reader = load_reader(config.reader)
    schema = config.schema
    rows: list[dict] = []
    writer = ArrayRecordWriter(os.path.join(config.output_path, SHARD_PATTERN.format(shard)), ARRAY_RECORD_OPTIONS)
    try:
        for unit in units:
            for episode in reader(unit, config.reader_options):
                length = validate_episode(schema, episode)
                record = {
                    name: encode_field(spec, schema.step_rate, episode.fields[name])
                    for name, spec in schema.fields.items()
                }
                writer.write(msgpack.packb(record))
                rows.append(
                    {
                        "episode": episode.id,
                        "shard": shard,
                        "record": len(rows),
                        "length": length,
                        "metadata": json.dumps(dict(episode.metadata), sort_keys=True),
                    }
                )
    finally:
        writer.close()
    return rows


def _write_index(config: EpisodeBuildConfig, rows: list[dict]) -> None:
    index = pd.DataFrame(rows, columns=["episode", "shard", "record", "length", "metadata"])
    duplicated = index["episode"][index["episode"].duplicated()]
    if len(duplicated):
        raise ValueError(f"episode {duplicated.iloc[0]!r} was read more than once; episode ids must be unique")
    index = index.sort_values(["shard", "record"], ignore_index=True)
    with open_url(os.path.join(config.output_path, INDEX_FILENAME), "wb") as f:
        index.to_parquet(f, index=False)
    with open_url(os.path.join(config.output_path, SCHEMA_FILENAME), "w") as f:
        json.dump(config.schema.to_json(), f, indent=2, sort_keys=True)
