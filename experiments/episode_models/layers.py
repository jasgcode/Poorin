# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Layers shared by the episode models: grug base's transformer block, a rotary embedding at
explicit positions, the block mask, frame patching, and the flow-time embedding.

The block follows ``experiments/grug/base/model.py`` (RMSNorm, attention, RMSNorm, ReLU MLP) with
raw parameter arrays. Parameters are replicated; only the batch is sharded.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from levanter.grug.attention import attention

HEAD_DIM = 64
MLP_RATIO = 4
INIT_STD = 0.02
NORM_EPS = 1e-5
ROPE_THETA = 10_000.0
TIME_FEATURES = 256
# Flow times lie in [0, 1]; stretching them lets the sinusoidal features resolve small steps.
TIME_SCALE = 1000.0


@dataclass(frozen=True)
class TrunkConfig:
    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        if self.hidden_dim % HEAD_DIM:
            raise ValueError(f"hidden_dim must be a multiple of {HEAD_DIM}, not {self.hidden_dim}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, not {self.num_layers}")

    @property
    def num_heads(self) -> int:
        return self.hidden_dim // HEAD_DIM

    @property
    def intermediate_dim(self) -> int:
        return MLP_RATIO * self.hidden_dim


def init_weight(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return INIT_STD * jax.random.normal(key, shape, jnp.float32)


class Linear(eqx.Module):
    weight: Array
    bias: Array

    @staticmethod
    def init(in_dim: int, out_dim: int, *, key: PRNGKeyArray) -> "Linear":
        return Linear(weight=init_weight(key, (in_dim, out_dim)), bias=jnp.zeros((out_dim,), jnp.float32))

    def __call__(self, x: Float[Array, "... I"]) -> Float[Array, "... O"]:
        return jnp.einsum("...i,io->...o", x, self.weight) + self.bias


class RMSNorm(eqx.Module):
    weight: Array

    @staticmethod
    def init(dim: int) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), jnp.float32))

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + NORM_EPS)
        return (x * self.weight.astype(jnp.float32)).astype(dtype)


def rotary(x: Float[Array, "B S H K"], positions: Int[Array, "S"]) -> Float[Array, "B S H K"]:
    """Rotary embedding at explicit positions, in grug's split-half convention. Tokens that share a
    position (the patches of one frame) get the same rotation."""
    half = x.shape[-1] // 2
    inv_freq = ROPE_THETA ** (-jnp.arange(half, dtype=jnp.float32) / half)
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos, sin = jnp.cos(angles)[None, :, None, :], jnp.sin(angles)[None, :, None, :]
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def block_mask(blocks: np.ndarray) -> np.ndarray:
    """``[S, S]``: query ``i`` sees key ``j`` when ``blocks[j] <= blocks[i]``. Tokens of one block see
    each other; a block never sees a later one."""
    return blocks[None, :] <= blocks[:, None]


class Attention(eqx.Module):
    w_q: Array
    w_k: Array
    w_v: Array
    w_o: Array
    num_heads: int = eqx.field(static=True)

    @staticmethod
    def init(config: TrunkConfig, *, key: PRNGKeyArray) -> "Attention":
        k_q, k_k, k_v, k_o = jax.random.split(key, 4)
        d = config.hidden_dim
        return Attention(
            w_q=init_weight(k_q, (d, d)),
            w_k=init_weight(k_k, (d, d)),
            w_v=init_weight(k_v, (d, d)),
            w_o=init_weight(k_o, (d, d)),
            num_heads=config.num_heads,
        )

    def __call__(
        self, x: Float[Array, "B S D"], positions: Int[Array, "S"], mask: Bool[Array, "S S"]
    ) -> Float[Array, "B S D"]:
        batch, seq, _ = x.shape

        def heads(w: Array) -> Array:
            return jnp.einsum("bsd,de->bse", x, w).reshape(batch, seq, self.num_heads, HEAD_DIM)

        q, k, v = rotary(heads(self.w_q), positions), rotary(heads(self.w_k), positions), heads(self.w_v)
        # A block mask is neither causal nor segmented, so FlashAttention 4 cannot take it.
        out = attention(q, k, v, jnp.broadcast_to(mask, (batch, seq, seq)), implementation="reference")
        return jnp.einsum("bse,ed->bsd", out.reshape(batch, seq, -1), self.w_o)


class MLP(eqx.Module):
    up: Array
    down: Array

    @staticmethod
    def init(config: TrunkConfig, *, key: PRNGKeyArray) -> "MLP":
        k_up, k_down = jax.random.split(key)
        return MLP(
            up=init_weight(k_up, (config.hidden_dim, config.intermediate_dim)),
            down=init_weight(k_down, (config.intermediate_dim, config.hidden_dim)),
        )

    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        return jnp.einsum("bsm,md->bsd", jax.nn.relu(jnp.einsum("bsd,dm->bsm", x, self.up)), self.down)


class Block(eqx.Module):
    attention_norm: RMSNorm
    attention: Attention
    mlp_norm: RMSNorm
    mlp: MLP

    @staticmethod
    def init(config: TrunkConfig, *, key: PRNGKeyArray) -> "Block":
        k_attention, k_mlp = jax.random.split(key)
        return Block(
            attention_norm=RMSNorm.init(config.hidden_dim),
            attention=Attention.init(config, key=k_attention),
            mlp_norm=RMSNorm.init(config.hidden_dim),
            mlp=MLP.init(config, key=k_mlp),
        )

    def __call__(
        self, x: Float[Array, "B S D"], positions: Int[Array, "S"], mask: Bool[Array, "S S"]
    ) -> Float[Array, "B S D"]:
        x = x + self.attention(self.attention_norm(x), positions, mask)
        return x + self.mlp(self.mlp_norm(x))


class Trunk(eqx.Module):
    blocks: tuple[Block, ...]
    final_norm: RMSNorm

    @staticmethod
    def init(config: TrunkConfig, *, key: PRNGKeyArray) -> "Trunk":
        keys = jax.random.split(key, config.num_layers)
        return Trunk(blocks=tuple(Block.init(config, key=k) for k in keys), final_norm=RMSNorm.init(config.hidden_dim))

    def __call__(
        self, x: Float[Array, "B S D"], positions: Int[Array, "S"], mask: Bool[Array, "S S"]
    ) -> Float[Array, "B S D"]:
        for block in self.blocks:
            x = block(x, positions, mask)
        return self.final_norm(x)


class TimeEmbedding(eqx.Module):
    """A flow time to a hidden-size vector: sinusoidal features, then a two-layer SiLU MLP."""

    w_in: Linear
    w_out: Linear

    @staticmethod
    def init(hidden_dim: int, *, key: PRNGKeyArray) -> "TimeEmbedding":
        k_in, k_out = jax.random.split(key)
        return TimeEmbedding(
            w_in=Linear.init(TIME_FEATURES, hidden_dim, key=k_in), w_out=Linear.init(hidden_dim, hidden_dim, key=k_out)
        )

    def __call__(self, t: Float[Array, "B"]) -> Float[Array, "B D"]:
        half = TIME_FEATURES // 2
        freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = TIME_SCALE * t.astype(jnp.float32)[:, None] * freqs[None, :]
        features = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1).astype(self.w_in.weight.dtype)
        return self.w_out(jax.nn.silu(self.w_in(features)))


def patchify(frames: Float[Array, "B T H W C"], patch: int) -> Float[Array, "B T N P"]:
    return rearrange(frames, "b t (h p) (w q) c -> b t (h w) (p q c)", p=patch, q=patch)


def unpatchify(patches: Float[Array, "B T N P"], patch: int, height: int, width: int) -> Float[Array, "B T H W C"]:
    return rearrange(
        patches, "b t (h w) (p q c) -> b t (h p) (w q) c", p=patch, q=patch, h=height // patch, w=width // patch
    )


def to_unit(frames: Array, dtype: jnp.dtype) -> Array:
    """``uint8`` pixels to ``[-1, 1]`` in ``dtype``."""
    return frames.astype(dtype) / 127.5 - 1
