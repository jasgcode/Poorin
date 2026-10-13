# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""World action model: denoises future frames and future actions together, each at its own flow
time, conditioned on clean history steps and a task.

Each timestep is one block of tokens: its frame's patches, then the action taken after that frame.
The last timestep has no action, since that action's result falls past the clip. A leading task
token comes first when the dataset has tasks. Blocks see themselves and earlier blocks, and all
tokens of a timestep share one rotary position. The action stream starts with the action after the
last history frame, so the first action a policy executes is predicted, not given.

Because the two streams have independent flow times, one model serves as a world model (actions at
pure noise) or a forward model (clean actions).
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
        patches, pixels, d = (
            num_patches(dims.frame_shape, self.patch_size),
            patch_dim(dims.frame_shape, self.patch_size),
            self.hidden_dim,
        )
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
        return (
            int(dims.num_tasks is not None)
            + clip.length * num_patches(dims.frame_shape, self.patch_size)
            + clip.length
            - 1
        )

    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float:
        """Forward FLOPs: the trunk, patch embeddings for every step and action embeddings for all but
        the last, heads for future steps, and four time embeddings (each stream's flow time and its
        clean time for history)."""
        if dims.action_dim is None:
            raise ValueError("the WAM needs an action dimension")
        n, p = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size)
        a, d, seq = dims.action_dim, self.hidden_dim, self.tokens_per_example(clip, dims)
        return (
            seq * trunk_flops_per_token(self.trunk, seq)
            + clip.length * n * linear_flops(p, d)
            + (clip.length - 1) * linear_flops(a, d)
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
        (if any) is 0, timestep ``t`` (from 1) has ``patches + 1`` tokens, and the last has
        ``patches``."""
        per_step = np.concatenate([np.repeat(np.arange(1, steps), patches + 1), np.full(patches, steps)])
        return per_step if self.task_embed is None else np.concatenate([[0], per_step])

    def _predict(
        self,
        patches: Float[Array, "B T N P"],
        actions: Float[Array, "B S A"],
        task: Int[Array, " B"] | None,
        t_video: Float[Array, " B"],
        t_action: Float[Array, " B"],
    ) -> tuple[Float[Array, "B H N P"], Float[Array, "B H A"]]:
        """Velocities of the future frames and actions, from clean history and noisy future ones.
        ``actions`` holds one action per step but the last (``S = T - 1``)."""
        # Noise, the noisy path, and targets stay in float32; only the model's input is cast.
        dtype = self.patch_embed.weight.dtype
        patches, actions = patches.astype(dtype), actions.astype(dtype)
        batch, steps, n, _ = patches.shape
        paired = (steps - 1) * (n + 1)
        clean = jnp.ones((batch,), jnp.float32)
        future_frames = (jnp.arange(steps) >= self.history)[None, :, None]
        future_actions = (jnp.arange(steps - 1) >= self.history - 1)[None, :, None]
        video_time = jnp.where(future_frames, self.video_time(t_video)[:, None], self.video_time(clean)[:, None])
        action_time = jnp.where(future_actions, self.action_time(t_action)[:, None], self.action_time(clean)[:, None])
        frames = self.patch_embed(patches) + self.patch_position + video_time[:, :, None]
        acts = self.action_embed(actions) + action_time
        steps_with_actions = jnp.concatenate([frames[:, :-1], acts[:, :, None]], axis=2).reshape(batch, paired, -1)
        tokens = jnp.concatenate([steps_with_actions, frames[:, -1]], axis=1)
        if self.task_embed is not None:
            if task is None:
                raise ValueError("this WAM conditions on tasks, but the batch has none")
            tokens = jnp.concatenate([self.task_embed[task][:, None], tokens], axis=1)
        layout = self.layout(steps, n)
        hidden = self.trunk(tokens, jnp.asarray(layout), jnp.asarray(block_mask(layout)))
        hidden = hidden[:, int(self.task_embed is not None) :]
        with_actions = hidden[:, :paired].reshape(batch, steps - 1, n + 1, -1)
        frame_hidden = jnp.concatenate([with_actions[:, :, :n], hidden[:, None, paired:]], axis=1)
        return (
            self.frame_head(frame_hidden[:, self.history :]),
            self.action_head(with_actions[:, self.history - 1 :, n]),
        )

    def _inputs(self, batch: EpisodeBatch) -> tuple[Array, Array]:
        if batch.actions is None:
            raise ValueError("the WAM needs actions in its batch")
        assert batch.actions.shape[1] == batch.frames.shape[1] > self.history, (batch.actions.shape, batch.frames.shape)
        # The clip's last action follows its last frame, so its result is past the clip.
        return patchify(to_unit(batch.frames, jnp.float32), self.patch_size), batch.actions[:, :-1]

    def stream_losses(
        self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Float[Array, " B"]]
    ) -> dict[str, Float[Array, ""]]:
        patches, actions = self._inputs(batch)
        c = self.history
        video_key, action_key = jax.random.split(key)
        video_noise = jax.random.normal(video_key, patches[:, c:].shape, patches.dtype)
        action_noise = jax.random.normal(action_key, actions[:, c - 1 :].shape, actions.dtype)
        noisy_patches = jnp.concatenate(
            [patches[:, :c], interpolate(patches[:, c:], video_noise, times["video"])], axis=1
        )
        noisy_actions = jnp.concatenate(
            [actions[:, : c - 1], interpolate(actions[:, c - 1 :], action_noise, times["action"])], axis=1
        )
        v_frames, v_actions = self._predict(noisy_patches, noisy_actions, batch.task, times["video"], times["action"])
        return {
            "video": velocity_loss(v_frames, velocity_target(patches[:, c:], video_noise)),
            "action": velocity_loss(v_actions, velocity_target(actions[:, c - 1 :], action_noise)),
        }

    def sample(self, batch: EpisodeBatch, key: PRNGKeyArray, num_steps: int) -> dict[str, Array]:
        """The future frames (in ``[-1, 1]``) and normalized actions, integrated jointly from noise."""
        patches, actions = self._inputs(batch)
        c = self.history
        video_key, action_key = jax.random.split(key)
        noise = (
            jax.random.normal(video_key, patches[:, c:].shape, patches.dtype),
            jax.random.normal(action_key, actions[:, c - 1 :].shape, actions.dtype),
        )

        def velocity(future: tuple[Array, Array], t: Float[Array, " B"]) -> tuple[Array, Array]:
            future_patches, future_actions = future
            return self._predict(
                jnp.concatenate([patches[:, :c], future_patches], axis=1),
                jnp.concatenate([actions[:, : c - 1], future_actions], axis=1),
                batch.task,
                t,
                t,
            )

        future_patches, future_actions = euler_sample(velocity, noise, num_steps)
        height, width = batch.frames.shape[2], batch.frames.shape[3]
        return {"video": unpatchify(future_patches, self.patch_size, height, width), "action": future_actions}
