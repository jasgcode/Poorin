# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read an episode dataset as clips through Grain.

:func:`clip_dataset` is a finite, unshuffled ``grain.MapDataset`` of clips; a training script
shuffles, repeats, and batches it. Everything up to batching has random access, so batch N is a
pure function of (seed, N): :func:`batch_iterator` resumes at a step by slicing, and decodes in
worker processes. :func:`to_mesh` places a batch on the mesh.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import grain
import jax
import msgpack
import numpy as np
from absl import flags
from array_record.python.array_record_module import ArrayRecordReader
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.encodings import decode_field
from poorin.episodes.schema import DatasetSchema

# Numeric dtype kinds placed on the mesh: bool, signed, unsigned, float. Text stays on the host.
DEVICE_DTYPE_KINDS = "biuf"


@dataclass(frozen=True)
class ClipSpec:
    length: int
    """Steps in a clip."""
    stride: int
    """Steps between the starts of consecutive clips in an episode."""
    skip: int
    """Steps between consecutive steps of a clip; 1 takes every step."""

    def __post_init__(self) -> None:
        too_small = {name: value for name, value in vars(self).items() if value < 1}
        if too_small:
            raise ValueError(f"a clip's length, stride, and skip are at least 1 step, not {too_small}")

    @property
    def span(self) -> int:
        return (self.length - 1) * self.skip + 1


def clip_starts(lengths: np.ndarray, clip: ClipSpec) -> tuple[np.ndarray, np.ndarray]:
    """Every clip as (episode row, start step). A clip never crosses its episode's end, and an
    episode shorter than one clip has none."""
    counts = np.maximum(0, (lengths - clip.span) // clip.stride + 1)
    rows = np.repeat(np.arange(len(lengths)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return rows, offsets * clip.stride


def read_record(dataset: EpisodeDataset, shard: int, record: int) -> dict[str, bytes]:
    reader = ArrayRecordReader(dataset.shard_paths[shard])
    try:
        return msgpack.unpackb(reader.read([record])[0])
    finally:
        reader.close()


class _ClipSource:
    """A Grain source of clip locations: (shard, record, episode length, start, episode row)."""

    def __init__(self, dataset: EpisodeDataset, clip: ClipSpec, rows: Sequence[int] | None):
        index = dataset.index
        selected = np.arange(len(index)) if rows is None else np.asarray(rows, dtype=np.int64)
        lengths = index["length"].to_numpy()
        positions, self._starts = clip_starts(lengths[selected], clip)
        self._rows = selected[positions]
        self._shards = index["shard"].to_numpy()
        self._records = index["record"].to_numpy()
        self._lengths = lengths

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> tuple[int, int, int, int, int]:
        row = int(self._rows[i])
        return int(self._shards[row]), int(self._records[row]), int(self._lengths[row]), int(self._starts[i]), row


class _ClipReader:
    """Decodes one clip. Picklable for Grain's worker processes: readers open lazily in each worker."""

    def __init__(self, dataset: EpisodeDataset, clip: ClipSpec, fields: Sequence[str], metadata: Sequence[str]):
        self._shard_paths = dataset.shard_paths
        self._schema: DatasetSchema = dataset.schema
        self._clip = clip
        self._fields = tuple(fields)
        self._metadata_keys = tuple(metadata)
        self._metadata = [json.loads(value) for value in dataset.index["metadata"]] if metadata else []
        self._readers: dict[int, ArrayRecordReader] = {}
        unknown = set(self._fields) - set(self._schema.fields)
        if unknown:
            raise ValueError(f"fields {sorted(unknown)} are not in the dataset; it has {sorted(self._schema.fields)}")

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_readers": {}}

    def __call__(self, location: tuple[int, int, int, int, int]) -> dict[str, np.ndarray]:
        shard, record, length, start, row = location
        if shard not in self._readers:
            self._readers[shard] = ArrayRecordReader(self._shard_paths[shard])
        data = msgpack.unpackb(self._readers[shard].read([record])[0])
        steps = range(start, start + self._clip.span, self._clip.skip)
        clip = {
            name: decode_field(self._schema.fields[name], self._schema.step_rate, data[name], length, steps)
            for name in self._fields
        }
        for key in self._metadata_keys:
            clip[key] = np.asarray(self._metadata[row][key])
        return clip


def clip_dataset(
    dataset: EpisodeDataset,
    *,
    clip: ClipSpec,
    fields: Sequence[str],
    metadata: Sequence[str] = (),
    rows: Sequence[int] | None = None,
) -> grain.MapDataset:
    """Every clip of ``dataset``, in index order, as ``{field: [clip.length, *shape]}`` plus the
    requested per-episode ``metadata`` values. Only ``fields`` are decoded. ``rows`` (positions in
    ``dataset.index``) restricts the clips to those episodes, for a held-out split."""
    source = _ClipSource(dataset, clip, rows)
    if not len(source):
        lengths = dataset.index["length"].to_numpy()
        if rows is not None:
            lengths = lengths[np.asarray(rows, dtype=np.int64)]
        longest = int(lengths.max()) if len(lengths) else 0
        raise ValueError(f"no clip of {clip.span} steps fits in these episodes: the longest episode has {longest}")
    return grain.MapDataset.source(source).map(_ClipReader(dataset, clip, fields, metadata))


def batch_iterator(clips: grain.MapDataset, *, batch_size: int, start_step: int, num_workers: int) -> grain.IterDataset:
    """Batches from ``start_step`` on, decoded in ``num_workers`` processes. ``clips`` is shuffled
    and repeated by the caller; resuming at a step reads the batches an uninterrupted run would."""
    batched = clips.batch(batch_size, drop_remainder=True)
    options = grain.MultiprocessingOptions(num_workers=num_workers)
    # Grain reads one of its absl flags when it starts workers, and a program that is not an absl
    # app never parses flags; Grain's defaults apply.
    if not flags.FLAGS.is_parsed():
        flags.FLAGS.mark_as_parsed()
    return batched[start_step:].to_iter_dataset().mp_prefetch(options)


def to_mesh(batch: Mapping[str, np.ndarray], mesh: Mesh, axes: tuple[str, ...]) -> dict[str, jax.Array | np.ndarray]:
    """Numeric fields as ``jax.Array``s sharded on their leading (batch) dimension over ``axes``;
    text fields stay numpy arrays on the host."""
    sharding = NamedSharding(mesh, PartitionSpec(axes))
    return {
        name: jax.make_array_from_process_local_data(sharding, values)
        if values.dtype.kind in DEVICE_DTYPE_KINDS
        else values
        for name, values in batch.items()
    }
