# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flow matching on the rectified-flow path, with ``t = 1`` clean: ``x_t = t * x + (1 - t) * noise``.

A model predicts the velocity ``x - noise``. Training draws times from a logit-normal distribution;
a held-out loss averages over a fixed grid of times, so it is deterministic for given parameters.
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

LOGIT_NORMAL_SCALE = 1.0
EVAL_TIME_COUNT = 8


def interpolate(data: Array, noise: Array, t: Float[Array, "B"]) -> Array:
    t = t.reshape(t.shape + (1,) * (data.ndim - t.ndim)).astype(data.dtype)
    return t * data + (1 - t) * noise


def velocity_target(data: Array, noise: Array) -> Array:
    return data - noise


def sample_times(key: PRNGKeyArray, batch_size: int) -> Float[Array, "B"]:
    """Logit-normal flow times, which weight the middle of the path (as in SD3)."""
    return jax.nn.sigmoid(LOGIT_NORMAL_SCALE * jax.random.normal(key, (batch_size,)))


def eval_times() -> np.ndarray:
    """The grid a held-out loss averages over: the midpoints of eight equal intervals of [0, 1]."""
    return (np.arange(EVAL_TIME_COUNT) + 0.5) / EVAL_TIME_COUNT


def euler_sample(velocity: Callable[[Any, Float[Array, "B"]], Any], noise: Any, num_steps: int) -> Any:
    """Integrate ``velocity(x, t)`` over the pytree ``x`` from ``t = 0`` (``noise``) to ``t = 1`` in
    ``num_steps`` equal Euler steps."""
    dt = 1.0 / num_steps
    batch_size = jax.tree.leaves(noise)[0].shape[0]

    def step(x, i):
        v = velocity(x, jnp.full((batch_size,), i * dt, jnp.float32))
        return jax.tree.map(lambda a, b: a + dt * b.astype(a.dtype), x, v), None

    out, _ = jax.lax.scan(step, noise, jnp.arange(num_steps))
    return out
