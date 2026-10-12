# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vision-language-action policy: the WAM (``wam/model.py``) without future frames. It denoises a
chunk of future actions conditioned on clean history frames and a task.

History frames are blocks by timestep, as in the WAM. The action chunk is one block after them, so
every action token sees all history and every other action, as in pi0's action expert. The chunk
starts with the action taken after the last history frame, and each action shares its frame's
rotary position. The VLA reads no past actions.
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
)


@dataclass(frozen=True)
class VlaConfig:
    hidden_dim: int = 384
    num_layers: int = 6
    patch_size: int = 16

    streams: ClassVar[tuple[str, ...]] = ("action",)
    conditions_on_task: ClassVar[bool] = True
    needs_actions: ClassVar[bool] = True

    @property
    def trunk(self) -> TrunkConfig:
        return TrunkConfig(self.hidden_dim, self.num_layers)

    @property
    def stream_weights(self) -> dict[str, float]:
        return {"action": 1.0}

    def init(self, dims: EpisodeDims, clip: ClipConfig, *, key: PRNGKeyArray) -> "Vla":
        if dims.action_dim is None:
            raise ValueError("the VLA predicts actions, so its dataset needs an action field")
        patches, pixels, d = (
            num_patches(dims.frame_shape, self.patch_size),
            patch_dim(dims.frame_shape, self.patch_size),
            self.hidden_dim,
        )
        keys = jax.random.split(key, 7)
        return Vla(
            patch_embed=Linear.init(pixels, d, key=keys[0]),
            patch_position=init_weight(keys[1], (patches, d)),
            action_embed=Linear.init(dims.action_dim, d, key=keys[2]),
            task_embed=None if dims.num_tasks is None else init_weight(keys[3], (dims.num_tasks, d)),
            action_time=TimeEmbedding.init(d, key=keys[4]),
            trunk=Trunk.init(self.trunk, key=keys[5]),
            action_head=Linear.init(d, dims.action_dim, key=keys[6]),
            patch_size=self.patch_size,
            history=clip.history,
        )

    def tokens_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> int:
        return (
            int(dims.num_tasks is not None)
            + clip.history * num_patches(dims.frame_shape, self.patch_size)
            + clip.horizon
        )

    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float:
        """Forward FLOPs: the trunk, patch embeddings for history steps, action embeddings and the head
        for future steps, and one time embedding."""
        if dims.action_dim is None:
            raise ValueError("the VLA needs an action dimension")
        n, p = num_patches(dims.frame_shape, self.patch_size), patch_dim(dims.frame_shape, self.patch_size)
        a, d, seq = dims.action_dim, self.hidden_dim, self.tokens_per_example(clip, dims)
        return (
            seq * trunk_flops_per_token(self.trunk, seq)
            + clip.history * n * linear_flops(p, d)
            + clip.horizon * (linear_flops(a, d) + linear_flops(d, a))
            + time_embedding_flops(d)
        )


class Vla(eqx.Module):
    patch_embed: Linear
    patch_position: Array
    action_embed: Linear
    task_embed: Array | None
    action_time: TimeEmbedding
    trunk: Trunk
    action_head: Linear
    patch_size: int = eqx.field(static=True)
    history: int = eqx.field(static=True)

    def layout(self, patches: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Each token's (rotary position, mask block): the task token (if any) at 0, history frame
        ``t`` (from 1) at ``t``, and the action taken after frame ``t`` at position ``t``, in one
        shared block after the history."""
        frames = np.repeat(np.arange(1, self.history + 1), patches)
        action_positions = np.arange(self.history, self.history + horizon)
        action_blocks = np.full(horizon, self.history + 1)
        lead = np.zeros(int(self.task_embed is not None), dtype=np.int64)
        return (
            np.concatenate([lead, frames, action_positions]),
            np.concatenate([lead, frames, action_blocks]),
        )

    def _predict(
        self,
        history_patches: Float[Array, "B C N P"],
        actions: Float[Array, "B H A"],
        task: Int[Array, " B"] | None,
        t_action: Float[Array, " B"],
    ) -> Float[Array, "B H A"]:
        """Velocities of the action chunk, from clean history frames and noisy actions."""
        batch, _, n, _ = history_patches.shape
        horizon = actions.shape[1]
        frames = (self.patch_embed(history_patches) + self.patch_position).reshape(batch, self.history * n, -1)
        acts = self.action_embed(actions) + self.action_time(t_action)[:, None]
        tokens = jnp.concatenate([frames, acts], axis=1)
        if self.task_embed is not None:
            if task is None:
                raise ValueError("this VLA conditions on tasks, but the batch has none")
            tokens = jnp.concatenate([self.task_embed[task][:, None], tokens], axis=1)
        positions, blocks = self.layout(n, horizon)
        hidden = self.trunk(tokens, jnp.asarray(positions), jnp.asarray(block_mask(blocks)))
        return self.action_head(hidden[:, -horizon:])

    def _inputs(self, batch: EpisodeBatch) -> tuple[Array, Array]:
        if batch.actions is None:
            raise ValueError("the VLA needs actions in its batch")
        dtype = self.patch_embed.weight.dtype
        history = patchify(to_unit(batch.frames[:, : self.history], dtype), self.patch_size)
        # The actions taken after frames history - 1 through length - 2; the clip's last action is unused.
        return history, batch.actions[:, self.history - 1 : -1].astype(dtype)

    def stream_losses(
        self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Float[Array, " B"]]
    ) -> dict[str, Float[Array, ""]]:
        history, actions = self._inputs(batch)
        noise = jax.random.normal(key, actions.shape, actions.dtype)
        predicted = self._predict(history, interpolate(actions, noise, times["action"]), batch.task, times["action"])
        return {"action": velocity_loss(predicted, velocity_target(actions, noise))}

    def sample(self, batch: EpisodeBatch, key: PRNGKeyArray, num_steps: int) -> dict[str, Array]:
        """The normalized action chunk, integrated from noise."""
        history, actions = self._inputs(batch)
        noise = jax.random.normal(key, actions.shape, actions.dtype)

        def velocity(future: Array, t: Float[Array, " B"]) -> Array:
            return self._predict(history, future, batch.task, t)

        return {"action": euler_sample(velocity, noise, num_steps)}
