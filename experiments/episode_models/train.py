# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The training loop every episode model shares.

A family supplies a config that builds its model from the dataset's sizes, and a model that returns
one flow-matching loss per stream at given flow times. The loop samples the times, trains, evaluates
the held-out loss on a fixed time grid, checkpoints, and resumes. It follows grug base's loop
(``experiments/grug/base/train.py``): Levanter's tracker, checkpointer, and optimizers, grug's
checkpoint restore, and grug's Fray dispatch.
"""

import dataclasses
import functools
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, no_type_check

import jax
import jax.numpy as jnp
import jmp
import levanter.callbacks as callbacks
import levanter.tracker
import numpy as np
import optax
from fray.types import ResourceConfig
from haliax.partitioning import set_mesh
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, PRNGKeyArray
from levanter.callbacks.state_adapter import StateCallbackRunner
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import parameter_count
from levanter.utils.logging import LoadingTimeTrackerIterator
from poorin.episodes.build import EpisodeDataset

from experiments.episode_models.data import (
    ClipConfig,
    EpisodeBatch,
    EpisodeDataConfig,
    EpisodeDims,
    data_mesh,
    dataset_dims,
    eval_batches,
    example_dataset,
    read_action_stats,
    split_rows,
    task_vocabulary,
    to_batch,
    train_batches,
)
from experiments.episode_models.flow import eval_times, sample_times
from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run

logger = logging.getLogger(__name__)

# The held-out loss draws its noise from this fixed key, whatever the run's seed, so runs compare.
EVAL_SEED = 0
# Training costs a forward and a backward pass, about three forward passes.
TRAIN_FLOPS_PER_FORWARD = 3


class EpisodeModel(Protocol):
    def stream_losses(
        self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Float[Array, " B"]]
    ) -> dict[str, Float[Array, ""]]: ...


class EpisodeModelConfig(Protocol):
    @property
    def streams(self) -> tuple[str, ...]: ...

    @property
    def conditions_on_task(self) -> bool: ...

    @property
    def needs_actions(self) -> bool: ...

    @property
    def stream_weights(self) -> dict[str, float]: ...

    def init(self, dims: EpisodeDims, clip: ClipConfig, *, key: PRNGKeyArray) -> EpisodeModel: ...

    def tokens_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> int: ...

    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float: ...


@dataclass(frozen=True)
class EpisodeTrainConfig:
    steps: int = 2000
    batch_size: int = 32
    seed: int = 0
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"
    optimizer: OptimizerConfig = field(
        default_factory=lambda: AdamConfig(
            learning_rate=1e-3, weight_decay=0.01, warmup=0.05, lr_schedule="cosine", min_lr_ratio=0.1
        )
    )
    steps_per_eval: int = 250
    eval_batches: int = 4
    eval_batch_size: int = 32
    """Separate from ``batch_size``, so a batch-size sweep evaluates every trial on the same clips."""
    checkpoint_every: int = 500
    """Steps between permanent checkpoints; time-based temporary ones come from the checkpointer."""
    log_every: int = 1


@dataclass(frozen=True)
class EpisodeTrainingRun:
    """Everything one training run reads. Launchers build it; tests build it directly."""

    model: EpisodeModelConfig
    dataset_path: str
    action_stats_path: str | None
    data: EpisodeDataConfig
    train: EpisodeTrainConfig
    trainer: TrainerConfig
    resources: ResourceConfig


# A JAX pytree: JAX rebuilds it with stand-in leaves, so its constructor is exempt from the tests'
# runtime type checks. ``ema_params`` is always None; grug's checkpoint restore reads the field.
@no_type_check
@register_dataclass
@dataclass(frozen=True)
class EpisodeTrainState:
    step: jax.Array
    params: Any
    opt_state: optax.OptState
    ema_params: None


def _make_train_step(
    optimizer: optax.GradientTransformation, mp: jmp.Policy, streams: Sequence[str], weights: Mapping[str, float]
):
    @functools.partial(jax.jit, donate_argnums=(0,))
    def train_step(state: EpisodeTrainState, batch: EpisodeBatch, key: PRNGKeyArray):
        time_key, noise_key = jax.random.split(key)
        size = batch.frames.shape[0]
        times = {s: sample_times(jax.random.fold_in(time_key, i), size) for i, s in enumerate(streams)}

        def loss_fn(params):
            losses = mp.cast_to_compute(params).stream_losses(batch, noise_key, times)
            return sum(weights[s] * losses[s] for s in streams), losses

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        next_state = dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state)
        return next_state, {"train/loss": loss, **{f"train/{s}_loss": losses[s] for s in streams}}

    return train_step


def _make_evaluate(mp: jmp.Policy, streams: Sequence[str], weights: Mapping[str, float]):
    """The held-out loss: each stream's loss averaged over the fixed time grid and the held-out
    batches, with noise from a fixed key. Deterministic for given parameters."""
    grid = jnp.asarray(eval_times(), jnp.float32)

    @jax.jit
    def evaluate_batch(params, batch: EpisodeBatch, key: PRNGKeyArray) -> dict[str, Array]:
        model = mp.cast_to_compute(params)
        size = batch.frames.shape[0]
        losses = jax.lax.map(lambda t: model.stream_losses(batch, key, {s: jnp.full((size,), t) for s in streams}), grid)
        return {s: losses[s].mean() for s in streams}

    def evaluate(params, batches: Sequence[EpisodeBatch]) -> dict[str, float]:
        per_batch = [
            evaluate_batch(params, batch, jax.random.fold_in(jax.random.key(EVAL_SEED), i))
            for i, batch in enumerate(batches)
        ]
        means = {s: float(np.mean([float(result[s]) for result in per_batch])) for s in streams}
        return {
            "eval/loss": sum(weights[s] * means[s] for s in streams),
            **{f"eval/{s}_loss": means[s] for s in streams},
        }

    return evaluate


def _train_locally(run: EpisodeTrainingRun) -> None:
    trainer = run.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(run)
    run_id = trainer.id
    if run_id is None:
        raise ValueError("trainer.id was not initialized")

    dataset = EpisodeDataset.raw_load(run.dataset_path)
    dims = dataset_dims(dataset, run.data)
    stats = read_action_stats(run.action_stats_path) if run.action_stats_path is not None else None
    tasks = task_vocabulary(dataset.index, run.data.task_key) if run.data.task_key is not None else None
    train_rows, eval_rows = split_rows(dataset.index, run.data.eval_percent)
    train_examples = example_dataset(dataset, run.data, train_rows, stats, tasks)
    eval_examples = example_dataset(dataset, run.data, eval_rows, stats, tasks)

    model_config, train = run.model, run.train
    streams, weights = model_config.streams, model_config.stream_weights
    optimizer = train.optimizer.build(train.steps)
    mp = jmp.get_policy(train.mp)
    model_key, noise_key = jax.random.split(jax.random.key(train.seed))
    tokens_per_example = model_config.tokens_per_example(run.data.clip, dims)
    train_flops_per_example = TRAIN_FLOPS_PER_FORWARD * model_config.flops_per_example(run.data.clip, dims)

    mesh = data_mesh()
    checkpointer = trainer.checkpointer.create(run_id)
    with set_mesh(mesh):

        @jax.jit
        def init_state(key: PRNGKeyArray) -> EpisodeTrainState:
            params = mp.cast_to_param(model_config.init(dims, run.data.clip, key=key))
            return EpisodeTrainState(
                step=jnp.array(0, jnp.int32), params=params, opt_state=optimizer.init(params), ema_params=None
            )

        state = restore_grug_state_from_checkpoint(
            init_state(model_key),
            checkpoint_search_paths=trainer.checkpoint_search_paths(run_id),
            load_checkpoint_setting=trainer.load_checkpoint,
            mesh=mesh,
            allow_partial=trainer.allow_partial_checkpoint,
        )
        levanter.tracker.log_summary(
            {
                "parameter_count": parameter_count(state.params),
                "throughput/flops_per_example_analytic": train_flops_per_example,
                "throughput/tokens_per_example": tokens_per_example,
            }
        )

        held_out = [
            to_batch(b, mesh)
            for b in eval_batches(eval_examples, batch_size=train.eval_batch_size, count=train.eval_batches)
        ]
        train_step = _make_train_step(optimizer, mp, streams, weights)
        evaluate = _make_evaluate(mp, streams, weights)

        state_callbacks = StateCallbackRunner[EpisodeTrainState](
            step_getter=lambda s: s.step,
            model_getter=lambda s: s.params,
            eval_model_getter=lambda s: s.params,
            opt_state_getter=lambda s: s.opt_state,
        )
        log_every = max(1, train.log_every)
        state_callbacks.add_hook(
            callbacks.log_performance_stats(tokens_per_example, train.batch_size, train_flops_per_example),
            every=log_every,
        )
        state_callbacks.add_hook(callbacks.pbar_logger(total=train.steps), every=log_every)
        state_callbacks.add_hook(callbacks.log_step_info(train.steps), every=log_every)

        batches = LoadingTimeTrackerIterator(
            train_batches(
                train_examples,
                batch_size=train.batch_size,
                seed=train.seed,
                start_step=int(state.step),
                num_workers=run.data.num_workers,
            )
        )
        last_loss: float | jax.Array = 0.0
        last_duration = 0.0
        while int(state.step) < train.steps:
            batch = to_batch(next(batches), mesh)
            step = int(state.step)
            started = time.perf_counter()
            state, metrics = train_step(state, batch, jax.random.fold_in(noise_key, step))
            jax.block_until_ready(metrics["train/loss"])
            last_loss, last_duration = metrics["train/loss"], time.perf_counter() - started
            state_callbacks.run(state, loss=last_loss, step_duration=last_duration)
            levanter.tracker.log(
                {
                    **{k: v for k, v in metrics.items() if k != "train/loss"},
                    "throughput/loading_time": batches.this_load_time,
                },
                step=step,
            )
            done = step + 1
            if done % train.steps_per_eval == 0 or done == train.steps:
                levanter.tracker.log(evaluate(state.params, held_out), step=step)
            if checkpointer is not None:
                checkpointer.on_step(tree=state, step=done)

        state_callbacks.run(state, loss=last_loss, step_duration=last_duration, force=True)
        if checkpointer is not None:
            checkpointer.on_step(tree=state, step=int(state.step), force=True)
            checkpointer.wait_until_finished()

    levanter.tracker.current_tracker().finish()


def run_episode_training(run: EpisodeTrainingRun) -> None:
    """Train ``run`` through a Fray job, which holds the process's training lock."""
    if run.trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching training")
    dispatch_grug_training_run(
        run_id=run.trainer.id, config=run, local_entrypoint=_train_locally, resources=run.resources
    )
