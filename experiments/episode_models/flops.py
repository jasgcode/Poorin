# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analytic forward FLOPs for the episode models' parts. Each family sums them per example; training
costs three times its forward pass."""

from levanter.utils.flop_utils import lm_flops_per_token

from experiments.episode_models.layers import TIME_FEATURES, TrunkConfig


def trunk_flops_per_token(trunk: TrunkConfig, seq_len: int) -> float:
    """The trunk's FLOPs per token over a ``seq_len`` sequence, by Levanter's language-model estimate
    with no vocabulary; the MLP is not gated."""
    return lm_flops_per_token(
        hidden_dim=trunk.hidden_dim,
        intermediate_dim=trunk.intermediate_dim,
        num_layers=trunk.num_layers,
        num_kv_heads=trunk.num_heads,
        num_heads=trunk.num_heads,
        seq_len=seq_len,
        vocab_size=0,
        glu=False,
    )


def linear_flops(in_dim: int, out_dim: int) -> float:
    return 2.0 * in_dim * out_dim


def time_embedding_flops(hidden_dim: int) -> float:
    return linear_flops(TIME_FEATURES, hidden_dim) + linear_flops(hidden_dim, hidden_dim)
