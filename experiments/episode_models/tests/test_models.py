# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.episode_models.data import ClipConfig, EpisodeBatch, EpisodeDims
from experiments.episode_models.layers import patchify, to_unit
from experiments.episode_models.wam.model import WamConfig

CONFIGS = {"wam": WamConfig}
CLIP = ClipConfig(history=2, horizon=2, stride=1, skip=1)
BATCH = 2


def tiny_setup(family: str, hidden_dim: int = 64, num_layers: int = 1):
    config = CONFIGS[family](hidden_dim=hidden_dim, num_layers=num_layers, patch_size=8)
    uses_actions = "action" in config.streams
    dims = EpisodeDims(
        frame_shape=(16, 16, 3),
        action_dim=3 if uses_actions else None,
        num_tasks=2 if config.conditions_on_task else None,
    )
    frames_key, actions_key = jax.random.split(jax.random.key(7))
    batch = EpisodeBatch(
        frames=jax.random.randint(frames_key, (BATCH, CLIP.length, 16, 16, 3), 0, 256).astype(jnp.uint8),
        actions=jax.random.uniform(actions_key, (BATCH, CLIP.length, 3), minval=-1, maxval=1) if uses_actions else None,
        task=jnp.array([0, 1], jnp.int32) if config.conditions_on_task else None,
    )
    return config, dims, CLIP, batch


@pytest.fixture(params=sorted(CONFIGS))
def family(request) -> str:
    return request.param


def test_the_analytic_flops_match_xla_within_ten_percent(family):
    config, dims, clip, batch = tiny_setup(family, hidden_dim=256, num_layers=2)
    model = config.init(dims, clip, key=jax.random.key(0))
    times = {s: jnp.full((BATCH,), 0.5) for s in config.streams}
    compiled = jax.jit(lambda m, b: m.stream_losses(b, jax.random.key(1), times)).lower(model, batch).compile()
    xla = compiled.cost_analysis()["flops"]
    analytic = config.flops_per_example(clip, dims) * BATCH
    assert abs(analytic - xla) / xla < 0.10, (analytic, xla)


def test_the_sampler_returns_the_future_of_each_stream(family):
    config, dims, clip, batch = tiny_setup(family)
    model = config.init(dims, clip, key=jax.random.key(0))
    out = model.sample(batch, jax.random.key(2), num_steps=3)
    again = model.sample(batch, jax.random.key(2), num_steps=3)
    assert set(out) == set(config.streams)
    if "video" in out:
        assert out["video"].shape == (BATCH, clip.horizon, 16, 16, 3)
    if "action" in out:
        assert out["action"].shape == (BATCH, clip.horizon, 3)
    jax.tree.map(np.testing.assert_array_equal, out, again)


def test_init_refuses_frames_the_patch_does_not_divide(family):
    config, dims, clip, _ = tiny_setup(family)
    with pytest.raises(ValueError, match="patch_size 5"):
        dataclasses.replace(config, patch_size=5).init(dims, clip, key=jax.random.key(0))


def test_a_later_future_frame_changes_no_earlier_prediction():
    config, dims, clip, batch = tiny_setup("wam")
    model = config.init(dims, clip, key=jax.random.key(0))
    patches = patchify(to_unit(batch.frames, jnp.float32), 8)
    t = jnp.full((BATCH,), 0.3)
    frames, actions = model._predict(patches, batch.actions, batch.task, t, t)
    changed_frames, changed_actions = model._predict(patches.at[:, -1].add(1.0), batch.actions, batch.task, t, t)
    np.testing.assert_allclose(frames[:, 0], changed_frames[:, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actions[:, 0], changed_actions[:, 0], rtol=1e-5, atol=1e-5)
    assert not np.allclose(frames[:, 1], changed_frames[:, 1])
