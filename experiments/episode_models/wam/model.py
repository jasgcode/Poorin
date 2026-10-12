# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""World action model: denoises future frames and future actions together, each at its own flow
time, conditioned on clean history steps and a task.

Each timestep is one block of tokens: its frame's patches, then its action. A leading task token
comes first when the dataset has tasks. Blocks see themselves and earlier blocks, and all tokens
of a timestep share one rotary position. Because the two streams have independent flow times, one
model serves as a world model (actions at pure noise), a forward model (clean actions), or an
inverse model (clean future frames).
"""

from dataclasses import dataclass
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PRNGKeyArray

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
class WamConfig:
    hidden_dim: int = 384
    num_layers: int = 6
    patch_size: int = 16
    action_loss_weight: float = 1.0

    streams: ClassVar[tuple[str, ...]] = ("video", "action")
    conditions_on_task: ClassVar[bool] = True
    needs_actions: ClassVar[bool] = True

    @property
    def trunk(self) -> TrunkConfig:
        return TrunkConfig(self.hidden_dim, self.num_layers)

    @property
    def stream_weights(self) -> dict[str, float]:
        return {"video": 1.0, "action": self.action_loss_weight}

    def init(self, dims: EpisodeDims, clip: ClipConfig, *, key: PRNGKeyArray) -> "Wam":
        if dims.action_dim is None:
            raise ValueError("the WAM predicts actions, so its dataset needs an action field")
        patches, pixels, d = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size), self.hidden_dim
        keys = jax.random.split(key, 9)
        return Wam(
            patch_embed=Linear.init(pixels, d, key=keys[0]),
            patch_position=init_weight(keys[1], (patches, d)),
            action_embed=Linear.init(dims.action_dim, d, key=keys[2]),
            task_embed=None if dims.num_tasks is None else init_weight(keys[3], (dims.num_tasks, d)),
            video_time=TimeEmbedding.init(d, key=keys[4]),
            action_time=TimeEmbedding.init(d, key=keys[5]),
            trunk=Trunk.init(self.trunk, key=keys[6]),
            frame_head=Linear.init(d, pixels, key=keys[7]),
            action_head=Linear.init(d, dims.action_dim, key=keys[8]),
            patch_size=self.patch_size,
            history=clip.history,
        )

    def tokens_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> int:
        return int(dims.num_tasks is not None) + clip.length * (num_patches(dims.frame_shape, self.patch_size) + 1)

    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float:
        """Forward FLOPs: the trunk, patch and action embeddings for every step, heads for future steps,
        and four time embeddings (each stream's flow time and its clean time for history)."""
        if dims.action_dim is None:
            raise ValueError("the WAM needs an action dimension")
        n, p = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size)
        a, d, seq = dims.action_dim, self.hidden_dim, self.tokens_per_example(clip, dims)
        return (
            seq * trunk_flops_per_token(self.trunk, seq)
            + clip.length * (n * linear_flops(p, d) + linear_flops(a, d))
            + clip.horizon * (n * linear_flops(d, p) + linear_flops(d, a))
            + 4 * time_embedding_flops(d)
        )


class Wam(eqx.Module):
    patch_embed: Linear
    patch_position: Array
    action_embed: Linear
    task_embed: Array | None
    video_time: TimeEmbedding
    action_time: TimeEmbedding
    trunk: Trunk
    frame_head: Linear
    action_head: Linear
    patch_size: int = eqx.field(static=True)
    history: int = eqx.field(static=True)

    def layout(self, steps: int, patches: int) -> np.ndarray:
        """Each token's timestep, which is both its rotary position and its mask block: the task token
        (if any) is 0, and timestep ``t`` (from 1) has ``patches + 1`` tokens."""
        per_step = np.repeat(np.arange(1, steps + 1), patches + 1)
        return per_step if self.task_embed is None else np.concatenate([[0], per_step])

    def _predict(
        self,
        patches: Float[Array, "B T N P"],
        actions: Float[Array, "B T A"],
        task: Int[Array, "B"] | None,
        t_video: Float[Array, "B"],
        t_action: Float[Array, "B"],
    ) -> tuple[Float[Array, "B H N P"], Float[Array, "B H A"]]:
        """Velocities of the future steps, from clean history steps and noisy future ones."""
        batch, steps, n, _ = patches.shape
        future = (jnp.arange(steps) >= self.history)[None, :, None]
        clean = jnp.ones((batch,), jnp.float32)
        video_time = jnp.where(future, self.video_time(t_video)[:, None], self.video_time(clean)[:, None])
        action_time = jnp.where(future, self.action_time(t_action)[:, None], self.action_time(clean)[:, None])
        frames = self.patch_embed(patches) + self.patch_position + video_time[:, :, None]
        acts = self.action_embed(actions) + action_time
        tokens = jnp.concatenate([frames, acts[:, :, None]], axis=2).reshape(batch, steps * (n + 1), -1)
        if self.task_embed is not None:
            if task is None:
                raise ValueError("this WAM conditions on tasks, but the batch has none")
            tokens = jnp.concatenate([self.task_embed[task][:, None], tokens], axis=1)
        layout = self.layout(steps, n)
        hidden = self.trunk(tokens, jnp.asarray(layout), jnp.asarray(block_mask(layout)))
        hidden = hidden[:, int(self.task_embed is not None) :].reshape(batch, steps, n + 1, -1)[:, self.history :]
        return self.frame_head(hidden[:, :, :n]), self.action_head(hidden[:, :, n])

    def _inputs(self, batch: EpisodeBatch) -> tuple[Array, Array]:
        if batch.actions is None:
            raise ValueError("the WAM needs actions in its batch")
        dtype = self.patch_embed.weight.dtype
        return patchify(to_unit(batch.frames, dtype), self.patch_size), batch.actions.astype(dtype)

    def stream_losses(
        self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Float[Array, "B"]]
    ) -> dict[str, Float[Array, ""]]:
        patches, actions = self._inputs(batch)
        c = self.history
        video_key, action_key = jax.random.split(key)
        video_noise = jax.random.normal(video_key, patches[:, c:].shape, patches.dtype)
        action_noise = jax.random.normal(action_key, actions[:, c:].shape, actions.dtype)
        noisy_patches = jnp.concatenate([patches[:, :c], interpolate(patches[:, c:], video_noise, times["video"])], axis=1)
        noisy_actions = jnp.concatenate([actions[:, :c], interpolate(actions[:, c:], action_noise, times["action"])], axis=1)
        v_frames, v_actions = self._predict(noisy_patches, noisy_actions, batch.task, times["video"], times["action"])
        return {
            "video": velocity_loss(v_frames, velocity_target(patches[:, c:], video_noise)),
            "action": velocity_loss(v_actions, velocity_target(actions[:, c:], action_noise)),
        }

    def sample(self, batch: EpisodeBatch, key: PRNGKeyArray, num_steps: int) -> dict[str, Array]:
        """The future frames (in ``[-1, 1]``) and normalized actions, integrated jointly from noise."""
        patches, actions = self._inputs(batch)
        c = self.history
        video_key, action_key = jax.random.split(key)
        noise = (
            jax.random.normal(video_key, patches[:, c:].shape, patches.dtype),
            jax.random.normal(action_key, actions[:, c:].shape, actions.dtype),
        )

        def velocity(future: tuple[Array, Array], t: Float[Array, "B"]) -> tuple[Array, Array]:
            future_patches, future_actions = future
            return self._predict(
                jnp.concatenate([patches[:, :c], future_patches], axis=1),
                jnp.concatenate([actions[:, :c], future_actions], axis=1),
                batch.task,
                t,
                t,
            )

        future_patches, future_actions = euler_sample(velocity, noise, num_steps)
        height, width = batch.frames.shape[2], batch.frames.shape[3]
        return {"video": unpatchify(future_patches, self.patch_size, height, width), "action": future_actions}
