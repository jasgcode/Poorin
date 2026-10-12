# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import numpy as np
import pytest

from experiments.episode_models.data import (
    ActionStats,
    ClipConfig,
    EpisodeDataConfig,
    EpisodeDims,
    compute_action_stats,
    dataset_dims,
    eval_batches,
    example_dataset,
    held_out,
    normalize_actions,
    read_action_stats,
    split_rows,
    task_vocabulary,
)
from experiments.episode_models.tests.tiny_dataset import build_tiny

SHORT_CLIP = ClipConfig(history=2, horizon=2, stride=4, skip=1)


def test_the_split_is_stable_and_disjoint(tmp_path):
    dataset = build_tiny(tmp_path)
    train, held = split_rows(dataset.index, eval_percent=50)
    assert set(train).isdisjoint(held) and len(train) + len(held) == len(dataset.index)
    # The hash is BLAKE2b, not Python's salted hash, so another process agrees.
    script = "from experiments.episode_models.data import held_out; print(held_out('e3', 50))"
    other = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert other.stdout.strip() == str(held_out("e3", 50))


def test_split_refuses_an_empty_side(tmp_path):
    dataset = build_tiny(tmp_path)
    with pytest.raises(ValueError, match="no held-out episodes"):
        split_rows(dataset.index, eval_percent=0)
    with pytest.raises(ValueError, match="no training episodes"):
        split_rows(dataset.index, eval_percent=100)


def test_normalization_maps_the_percentiles_to_the_unit_interval():
    stats = ActionStats(low=np.array([0.0, -2.0]), high=np.array([10.0, 2.0]))
    out = normalize_actions(np.array([[0.0, -2.0], [10.0, 2.0], [5.0, 0.0], [20.0, -9.0]], np.float32), stats)
    np.testing.assert_allclose(out, [[-1, -1], [1, 1], [0, 0], [1, -1]])


def test_a_constant_action_dimension_normalizes_to_finite_values(tmp_path):
    dataset = build_tiny(tmp_path)
    compute_action_stats(dataset.path, "action", str(tmp_path / "stats"))
    stats = read_action_stats(str(tmp_path / "stats"))
    assert stats.low[2] == stats.high[2] == 0.5
    assert np.isfinite(normalize_actions(np.ones((4, 3), np.float32), stats)).all()


def test_examples_carry_frames_normalized_actions_and_task_indices(tmp_path):
    dataset = build_tiny(tmp_path)
    compute_action_stats(dataset.path, "action", str(tmp_path / "stats"))
    config = EpisodeDataConfig(
        video_field="video", action_field="action", task_key="instruction", clip=SHORT_CLIP, num_workers=0
    )
    tasks = task_vocabulary(dataset.index, "instruction")
    rows = np.arange(len(dataset.index))
    examples = example_dataset(dataset, config, rows, read_action_stats(str(tmp_path / "stats")), tasks)
    first, last = examples[0], examples[len(examples) - 1]
    assert first["frames"].shape == (4, 16, 16, 3) and first["frames"].dtype == np.uint8
    assert first["actions"].shape == (4, 3) and np.abs(first["actions"]).max() <= 1
    assert tasks == ("pour", "stack")
    # Episode e0 is "pour" and e9 is "stack"; clips come in index order.
    assert (int(first["task"]), int(last["task"])) == (0, 1)
    assert dataset_dims(dataset, config) == EpisodeDims(frame_shape=(16, 16, 3), action_dim=3, num_tasks=2)


def test_dims_refuse_a_field_the_dataset_lacks(tmp_path):
    dataset = build_tiny(tmp_path)
    config = EpisodeDataConfig(video_field="agentview", action_field=None, task_key=None, clip=SHORT_CLIP)
    with pytest.raises(ValueError, match="agentview"):
        dataset_dims(dataset, config)


def test_eval_batches_refuse_a_split_smaller_than_a_batch(tmp_path):
    dataset = build_tiny(tmp_path, episodes=2)
    config = EpisodeDataConfig(video_field="video", action_field=None, task_key=None, clip=SHORT_CLIP, num_workers=0)
    examples = example_dataset(dataset, config, np.array([0]), None, None)
    with pytest.raises(ValueError, match="fewer clips than one batch"):
        eval_batches(examples, batch_size=64, count=1)
