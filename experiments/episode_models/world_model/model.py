# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Video world model: the WAM (``wam/model.py``) without actions or tasks. It denoises future frames
conditioned on clean history frames.

Each timestep is one block of its frame's patches; blocks see themselves and earlier blocks, and all
patches of a frame share one rotary position.
"""

from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PRNGKeyArray

from experiments.episode_models.data import ClipConfig, EpisodeBatch, EpisodeDims
from experiments.episode_models.flops import linear_flops, time_embedding_flops, trunk_flops_per_token
from experiments.episode_models.flow import euler_sample, interpolate, velocity_loss, velocity_target
from experiments.episode_models.layers import (
    Linear,
    TimeEmbedding,
    Trunk,
    TrunkConfig,
    block_mask,
    init_weight,
    num_patches,
    patch_dim,
    patchify,
    to_unit,
    unpatchify,
)


@dataclass(frozen=True)
class WorldModelConfig:
    hidden_dim: int = 384
    num_layers: int = 6
    patch_size: int = 16

    streams: ClassVar[tuple[str, ...]] = ("video",)
    conditions_on_task: ClassVar[bool] = False
    needs_actions: ClassVar[bool] = False

    @property
    def trunk(self) -> TrunkConfig:
        return TrunkConfig(self.hidden_dim, self.num_layers)

    @property
    def stream_weights(self) -> dict[str, float]:
        return {"video": 1.0}

    def init(self, dims: EpisodeDims, clip: ClipConfig, *, key: PRNGKeyArray) -> "WorldModel":
        patches, pixels, d = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size), self.hidden_dim
        keys = jax.random.split(key, 5)
        return WorldModel(
            patch_embed=Linear.init(pixels, d, key=keys[0]),
            patch_position=init_weight(keys[1], (patches, d)),
            video_time=TimeEmbedding.init(d, key=keys[2]),
            trunk=Trunk.init(self.trunk, key=keys[3]),
            frame_head=Linear.init(d, pixels, key=keys[4]),
            patch_size=self.patch_size,
            history=clip.history,
        )

    def tokens_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> int:
        return clip.length * num_patches(dims.frame_shape, self.patch_size)

    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float:
        """Forward FLOPs: the trunk, patch embeddings for every step, the head for future steps, and two
        time embeddings (the flow time and the clean time of history)."""
        n, p = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size)
        d, seq = self.hidden_dim, self.tokens_per_example(clip, dims)
        return (
            seq * trunk_flops_per_token(self.trunk, seq)
            + clip.length * n * linear_flops(p, d)
            + clip.horizon * n * linear_flops(d, p)
            + 2 * time_embedding_flops(d)
        )


class WorldModel(eqx.Module):
    patch_embed: Linear
    patch_position: Array
    video_time: TimeEmbedding
    trunk: Trunk
    frame_head: Linear
    patch_size: int = eqx.field(static=True)
    history: int = eqx.field(static=True)

    def _predict(self, patches: Float[Array, "B T N P"], t_video: Float[Array, "B"]) -> Float[Array, "B H N P"]:
        """Velocities of the future frames, from clean history frames and noisy future ones."""
        batch, steps, n, _ = patches.shape
        future = (jnp.arange(steps) >= self.history)[None, :, None]
        clean = jnp.ones((batch,), jnp.float32)
        video_time = jnp.where(future, self.video_time(t_video)[:, None], self.video_time(clean)[:, None])
        tokens = (self.patch_embed(patches) + self.patch_position + video_time[:, :, None]).reshape(batch, steps * n, -1)
        layout = np.repeat(np.arange(1, steps + 1), n)
        hidden = self.trunk(tokens, jnp.asarray(layout), jnp.asarray(block_mask(layout)))
        return self.frame_head(hidden.reshape(batch, steps, n, -1)[:, self.history :])

    def _patches(self, batch: EpisodeBatch) -> Array:
        return patchify(to_unit(batch.frames, self.patch_embed.weight.dtype), self.patch_size)

    def stream_losses(
        self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Float[Array, "B"]]
    ) -> dict[str, Float[Array, ""]]:
        patches = self._patches(batch)
        c = self.history
        noise = jax.random.normal(key, patches[:, c:].shape, patches.dtype)
        noisy = jnp.concatenate([patches[:, :c], interpolate(patches[:, c:], noise, times["video"])], axis=1)
        return {"video": velocity_loss(self._predict(noisy, times["video"]), velocity_target(patches[:, c:], noise))}

    def sample(self, batch: EpisodeBatch, key: PRNGKeyArray, num_steps: int) -> dict[str, Array]:
        """The future frames, in ``[-1, 1]``, integrated from noise."""
        patches = self._patches(batch)
        c = self.history
        noise = jax.random.normal(key, patches[:, c:].shape, patches.dtype)

        def velocity(future: Array, t: Float[Array, "B"]) -> Array:
            return self._predict(jnp.concatenate([patches[:, :c], future], axis=1), t)

        future = euler_sample(velocity, noise, num_steps)
        return {"video": unpatchify(future, self.patch_size, batch.frames.shape[2], batch.frames.shape[3])}
