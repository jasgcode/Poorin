# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import RotaryConfig, apply_rotary_embedding

from experiments.episode_models.layers import Trunk, TrunkConfig, block_mask, patchify, rotary, unpatchify


def test_rotary_at_consecutive_positions_matches_grug():
    q = jax.random.normal(jax.random.key(0), (2, 5, 3, 64))
    expected, _ = apply_rotary_embedding(q, q, seq_len=5, head_dim=64, rope=RotaryConfig())
    np.testing.assert_allclose(rotary(q, jnp.arange(5)), expected, rtol=1e-5, atol=1e-5)


def test_tokens_that_share_a_position_get_the_same_rotation():
    x = jnp.ones((1, 3, 1, 64))
    out = rotary(x, jnp.array([2, 2, 3]))
    np.testing.assert_array_equal(out[:, 0], out[:, 1])
    assert not np.allclose(out[:, 1], out[:, 2])


def test_block_mask_sees_its_block_and_earlier_ones():
    mask = block_mask(np.array([0, 1, 1, 2]))
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, True, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_patchify_round_trips():
    frames = jax.random.normal(jax.random.key(1), (2, 3, 16, 24, 3))
    patches = patchify(frames, 8)
    assert patches.shape == (2, 3, 6, 8 * 8 * 3)
    np.testing.assert_array_equal(unpatchify(patches, 8, 16, 24), frames)


def test_a_trunk_token_is_unaffected_by_later_blocks():
    trunk = Trunk.init(TrunkConfig(hidden_dim=64, num_layers=2), key=jax.random.key(2))
    x = jax.random.normal(jax.random.key(3), (1, 4, 64))
    blocks = np.array([0, 1, 1, 2])
    mask = jnp.asarray(block_mask(blocks))
    out = trunk(x, jnp.asarray(blocks), mask)
    changed = trunk(x.at[:, 3].set(7.0), jnp.asarray(blocks), mask)
    np.testing.assert_allclose(out[:, :3], changed[:, :3], rtol=1e-5, atol=1e-5)
    assert not np.allclose(out[:, 3], changed[:, 3])


def test_hidden_dim_must_be_a_multiple_of_the_head_dim():
    with pytest.raises(ValueError, match="multiple of 64"):
        TrunkConfig(hidden_dim=100, num_layers=1)
