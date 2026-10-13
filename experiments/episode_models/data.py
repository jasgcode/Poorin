# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Episode datasets as model batches: clips of history and future steps, an episode-level held-out
split, action normalization by percentiles, and task indices from instructions.

A batch at step N is a pure function of (seed, N), as in :mod:`poorin.episodes.read`; nothing random
happens here, so the training loop draws all noise from its own key.
"""

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import grain
import jax
import numpy as np
import pandas as pd
from jax.sharding import Mesh
from poorin.episodes.build import EpisodeDataset
from poorin.episodes.encodings import decode_field
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset, read_record, to_mesh
from poorin.episodes.schema import FieldKind
from poorin.execution.artifact import Artifact
from poorin.execution.lazy import OUT, ArtifactStep, apply

# Data parallel only: parameters are replicated and the batch is split over ``data``.
BATCH_AXES = ("data",)
ACTION_STATS_FILENAME = "action_stats.json"
LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0


@dataclass(frozen=True)
class ClipConfig:
    history: int = 4
    """Clean steps the model conditions on."""
    horizon: int = 8
    """Future steps the model predicts."""
    stride: int = 4
    """Steps between the starts of consecutive clips in an episode."""
    skip: int = 1
    """Steps between consecutive steps of a clip."""

    def __post_init__(self) -> None:
        if self.history < 1 or self.horizon < 1:
            raise ValueError(f"a clip needs at least one history and one future step, not {self}")

    @property
    def length(self) -> int:
        return self.history + self.horizon

    @property
    def spec(self) -> ClipSpec:
        return ClipSpec(length=self.length, stride=self.stride, skip=self.skip)


@dataclass(frozen=True)
class EpisodeDataConfig:
    video_field: str
    action_field: str | None
    """None for a family that reads no actions."""
    task_key: str | None
    """The episode metadata key holding its instruction; None for a family without task conditioning."""
    clip: ClipConfig = ClipConfig()
    eval_percent: int = 5
    num_workers: int = 16


@dataclass(frozen=True)
class EpisodeDims:
    """Sizes the dataset decides, so a model config holds only choices."""

    frame_shape: tuple[int, int, int]
    action_dim: int | None
    num_tasks: int | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeBatch:
    frames: jax.Array
    """``uint8 [B, history + horizon, H, W, C]``."""
    actions: jax.Array | None
    """``float32 [B, history + horizon, A]``, normalized to ``[-1, 1]``."""
    task: jax.Array | None
    """``int32 [B]``, an index into the dataset's sorted instructions."""


def held_out(episode_id: str, eval_percent: int) -> bool:
    """Whether an episode is held out. BLAKE2b, unlike Python's salted ``hash``, agrees across processes."""
    digest = hashlib.blake2b(episode_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % 100 < eval_percent


def split_rows(index: pd.DataFrame, eval_percent: int) -> tuple[np.ndarray, np.ndarray]:
    """The (training, held-out) episode rows. Splitting by episode keeps held-out clips from sharing
    frames with training clips."""
    held = np.array([held_out(str(episode), eval_percent) for episode in index["episode"]], dtype=bool)
    train, evaluation = np.flatnonzero(~held), np.flatnonzero(held)
    if not len(evaluation):
        raise ValueError(f"no held-out episodes at eval_percent={eval_percent} among {len(index)}; raise it")
    if not len(train):
        raise ValueError(f"no training episodes at eval_percent={eval_percent} among {len(index)}; lower it")
    return train, evaluation


def task_vocabulary(index: pd.DataFrame, task_key: str) -> tuple[str, ...]:
    return tuple(sorted({str(json.loads(value)[task_key]) for value in index["metadata"]}))


def dataset_dims(dataset: EpisodeDataset, config: EpisodeDataConfig) -> EpisodeDims:
    fields = dataset.schema.fields
    wanted = [name for name in (config.video_field, config.action_field) if name is not None]
    missing = [name for name in wanted if name not in fields]
    if missing:
        raise ValueError(f"the dataset has no field {missing}; it has {sorted(fields)}")
    video = fields[config.video_field]
    if video.kind != FieldKind.VIDEO:
        raise ValueError(f"{config.video_field} is a {video.kind} field, not video")
    action_dim = None
    if config.action_field is not None:
        action = fields[config.action_field]
        if action.kind != FieldKind.ARRAY or len(action.shape) != 1:
            raise ValueError(f"{config.action_field} must be a per-step vector, not {action.kind} {action.shape}")
        action_dim = action.shape[0]
    num_tasks = None if config.task_key is None else len(task_vocabulary(dataset.index, config.task_key))
    height, width, channels = video.shape
    return EpisodeDims(frame_shape=(height, width, channels), action_dim=action_dim, num_tasks=num_tasks)


@dataclass(frozen=True)
class ActionStats:
    low: np.ndarray
    high: np.ndarray


def compute_action_stats(dataset_path: str, field: str, output_path: str) -> None:
    """Write the 1st and 99th percentiles of each dimension of ``field`` over every step of the dataset."""
    dataset = EpisodeDataset.raw_load(dataset_path)
    spec, step_rate = dataset.schema.fields[field], dataset.schema.step_rate
    values = []
    for row in dataset.index.itertuples():
        data = read_record(dataset, int(row.shard), int(row.record))[field]
        values.append(decode_field(spec, step_rate, data, int(row.length), range(int(row.length))))
    steps = np.concatenate(values).astype(np.float64)
    stats = {
        "field": field,
        "low": np.percentile(steps, LOW_PERCENTILE, axis=0).tolist(),
        "high": np.percentile(steps, HIGH_PERCENTILE, axis=0).tolist(),
    }
    # Episode datasets are local (ArrayRecord opens local paths only), and so are their stats.
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, ACTION_STATS_FILENAME), "w") as f:
        json.dump(stats, f)


def read_action_stats(path: str) -> ActionStats:
    with open(os.path.join(path, ACTION_STATS_FILENAME)) as f:
        stats = json.load(f)
    return ActionStats(low=np.asarray(stats["low"]), high=np.asarray(stats["high"]))


def action_stats(dataset: ArtifactStep[EpisodeDataset], field: str) -> ArtifactStep[Artifact]:
    """The percentiles of one action field, versioned with its dataset, so each dataset version
    computes them once."""
    return apply(
        f"episode-stats/{dataset.name}/{field}",
        compute_action_stats,
        version=dataset.version,
        dataset_path=dataset,
        field=field,
        output_path=OUT,
    )


def normalize_actions(actions: np.ndarray, stats: ActionStats) -> np.ndarray:
    """Map each dimension's percentiles to -1 and 1, then clip. A dimension that never varies maps to 0."""
    span = stats.high - stats.low
    varies = span > 0
    scaled = np.where(varies, 2 * (actions - stats.low) / np.where(varies, span, 1) - 1, 0)
    return np.clip(scaled, -1, 1).astype(np.float32)


class _ToExample:
    """A clip to the model's fields: ``frames``, normalized ``actions``, and the ``task`` index.
    Picklable for Grain's worker processes."""

    def __init__(self, config: EpisodeDataConfig, stats: ActionStats | None, tasks: Sequence[str] | None):
        self._config = config
        self._stats = stats
        self._task_index = {task: i for i, task in enumerate(tasks)} if tasks is not None else None

    def __call__(self, clip: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        example = {"frames": clip[self._config.video_field]}
        if self._config.action_field is not None:
            if self._stats is None:
                raise ValueError("actions are read, so their stats are required")
            example["actions"] = normalize_actions(clip[self._config.action_field], self._stats)
        if self._config.task_key is not None and self._task_index is not None:
            example["task"] = np.asarray(self._task_index[str(clip[self._config.task_key])], dtype=np.int32)
        return example


def example_dataset(
    dataset: EpisodeDataset,
    config: EpisodeDataConfig,
    rows: np.ndarray,
    stats: ActionStats | None,
    tasks: Sequence[str] | None,
) -> grain.MapDataset:
    """The clips of episode ``rows``, in index order, as model examples."""
    fields = [config.video_field] + ([config.action_field] if config.action_field is not None else [])
    metadata = [config.task_key] if config.task_key is not None else []
    clips = clip_dataset(dataset, clip=config.clip.spec, fields=fields, metadata=metadata, rows=rows)
    return clips.map(_ToExample(config, stats, tasks))


def train_batches(
    examples: grain.MapDataset, *, batch_size: int, seed: int, start_step: int, num_workers: int
) -> Iterator[dict[str, np.ndarray]]:
    shuffled = examples.shuffle(seed=seed).repeat(None)
    return iter(batch_iterator(shuffled, batch_size=batch_size, start_step=start_step, num_workers=num_workers))


def eval_batches(examples: grain.MapDataset, *, batch_size: int, count: int) -> list[dict[str, np.ndarray]]:
    """``count`` batches of held-out clips spread evenly over the split in index order, so every
    held-out episode contributes in proportion to its length: the same clips every evaluation."""
    wanted = batch_size * count
    if len(examples) < wanted:
        raise ValueError(f"the held-out split has {len(examples)} clips; {count} batches of {batch_size} need {wanted}")
    picked = [cast(dict[str, np.ndarray], examples[int(i)]) for i in np.linspace(0, len(examples) - 1, wanted).round()]
    return [
        {name: np.stack([clip[name] for clip in picked[start : start + batch_size]]) for name in picked[0]}
        for start in range(0, wanted, batch_size)
    ]


def data_mesh() -> Mesh:
    """Every device on one ``data`` axis, with automatic sharding: JAX propagates the batch's
    sharding through reshapes such as patching, which explicit sharding refuses to infer."""
    return Mesh(np.array(jax.devices()), BATCH_AXES)


def to_batch(arrays: Mapping[str, np.ndarray], mesh: Mesh) -> EpisodeBatch:
    placed = to_mesh(arrays, mesh, BATCH_AXES)
    return EpisodeBatch(frames=placed["frames"], actions=placed.get("actions"), task=placed.get("task"))
