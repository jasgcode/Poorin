# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np

from experiments.episode_models.flow import (
    EVAL_TIME_COUNT,
    euler_sample,
    eval_times,
    interpolate,
    sample_times,
    velocity_target,
)


def test_the_path_runs_from_noise_to_data():
    data, noise = jnp.full((2, 3), 5.0), jnp.full((2, 3), -1.0)
    np.testing.assert_allclose(interpolate(data, noise, jnp.zeros(2)), noise)
    np.testing.assert_allclose(interpolate(data, noise, jnp.ones(2)), data)
    # A step of dt along the target velocity moves the path by dt.
    t = jnp.array([0.25, 0.5])
    moved = interpolate(data, noise, t) + 0.25 * velocity_target(data, noise)
    np.testing.assert_allclose(moved, interpolate(data, noise, t + 0.25), rtol=1e-6)


def test_euler_with_the_true_velocity_reaches_the_data():
    data = {"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.ones((2, 1))}
    noise = jax.tree.map(jnp.zeros_like, data)
    out = euler_sample(lambda x, t: jax.tree.map(lambda d, n: d - n, data, noise), noise, num_steps=4)
    jax.tree.map(lambda o, d: np.testing.assert_allclose(o, d, rtol=1e-6), out, data)


def test_sampled_times_lie_inside_the_unit_interval_and_eval_times_are_fixed():
    t = sample_times(jax.random.key(0), 4096)
    assert float(t.min()) > 0 and float(t.max()) < 1
    assert abs(float(jnp.median(t)) - 0.5) < 0.03
    np.testing.assert_allclose(eval_times(EVAL_TIME_COUNT), (np.arange(8) + 0.5) / 8)
