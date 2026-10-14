# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluating episode-model runs (D114): their checkpoint type and the held-out loss suite.

An episode trial records :class:`EpisodeCheckpoint`, so ``python -m poorin.experiment.evaluate``
finds the suites below. :class:`HeldOutLoss` recomputes the training loop's held-out loss from a
permanent checkpoint with the loop's own function. Its defaults are the training defaults, so with
them it reproduces the ``eval/*`` values the run logged at that step (logged at step ``N - 1`` for
the checkpoint ``step-N``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, cast

import jax
import jmp
from haliax.partitioning import set_mesh
from poorin.evaluation.contract import (
    EvalResult,
    EvalRuntime,
    FinishedRun,
    Suite,
    checkpoint_path,
    recorded,
    recorded_value,
)
from poorin.execution.artifact import Artifact
from poorin.execution.fingerprint import CLASS_KEY

from experiments.episode_models.data import EpisodeDataConfig, data_mesh, eval_batches, to_batch
from experiments.episode_models.flow import EVAL_TIME_COUNT
from experiments.episode_models.train import EVAL_SEED, EpisodeModelConfig, held_out_loss, load_examples
from experiments.episode_models.vla.model import VlaConfig
from experiments.episode_models.wam.model import WamConfig
from experiments.episode_models.world_model.model import WorldModelConfig
from experiments.grug.checkpointing import load_grug_checkpoint

# A family's model config, by the class name its record carries.
MODEL_CONFIGS: dict[str, type] = {cls.__name__: cls for cls in (WamConfig, WorldModelConfig, VlaConfig)}


@dataclass(frozen=True)
class RecordedEpisodeRun:
    """What the held-out loss reads from a run's recipe."""

    model: EpisodeModelConfig
    data: EpisodeDataConfig
    mp: str
    dataset_path: str
    action_stats_path: str | None


def recorded_episode_run(run: FinishedRun) -> RecordedEpisodeRun:
    family = cast(str, recorded_value(run, f"model.{CLASS_KEY}"))
    if family not in MODEL_CONFIGS:
        raise ValueError(f"{run.label} records a {family} model; the episode families are {sorted(MODEL_CONFIGS)}")
    return RecordedEpisodeRun(
        model=recorded(run, "model", MODEL_CONFIGS[family]),
        data=recorded(run, "data", EpisodeDataConfig),
        mp=recorded(run, "train.mp", str),
        dataset_path=recorded(run, "run.dataset_path", str),
        action_stats_path=cast(str | None, recorded_value(run, "run.action_stats_path")),
    )


def held_out_metrics(run: RecordedEpisodeRun, checkpoint: str, suite: "HeldOutLoss") -> dict[str, float]:
    """The held-out loss of the parameters at ``checkpoint``, by the training loop's own function."""
    examples = load_examples(run.dataset_path, run.action_stats_path, run.data)
    mp = jmp.get_policy(run.mp)
    mesh = data_mesh()
    with set_mesh(mesh):
        exemplar = jax.jit(lambda key: mp.cast_to_param(run.model.init(examples.dims, run.data.clip, key=key)))(
            jax.random.key(0)
        )
        loaded = load_grug_checkpoint(state={"params": exemplar}, candidate=checkpoint, mesh=mesh, allow_partial=False)
        batches = [
            to_batch(batch, mesh)
            for batch in eval_batches(
                examples.held_out, batch_size=suite.batch_size, count=suite.clips // suite.batch_size
            )
        ]
        evaluate = held_out_loss(
            mp, run.model.streams, run.model.stream_weights, flow_times=suite.flow_times, seed=suite.seed
        )
        return evaluate(loaded["params"], batches)


@dataclass(frozen=True)
class HeldOutLoss:
    """The training loop's held-out loss at one checkpoint.

    ``clips`` held-out clips spread evenly over the split, in batches of ``batch_size``, each stream's
    loss averaged over a grid of ``flow_times`` flow times. Batch ``i`` draws its noise from
    ``fold_in(key(seed), i)``, so the batch size is part of the recipe.
    """

    name: ClassVar[str] = "held-out"
    clips: int = 128
    batch_size: int = 32
    flow_times: int = EVAL_TIME_COUNT
    seed: int = EVAL_SEED

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.clips < self.batch_size or self.clips % self.batch_size:
            raise ValueError(f"clips must be a positive multiple of batch_size, not {self.clips} and {self.batch_size}")
        if self.flow_times < 1:
            raise ValueError(f"flow_times must be at least 1, not {self.flow_times}")

    def check(self, run: FinishedRun, runtime: EvalRuntime) -> None:
        self._checked_run(run, runtime)

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        metrics = held_out_metrics(self._checked_run(run, runtime), checkpoint_path(run, step), self)
        return EvalResult(run=run.label, step=step, suite=self.name, metrics=metrics, examples=self.clips)

    def _checked_run(self, run: FinishedRun, runtime: EvalRuntime) -> RecordedEpisodeRun:
        if runtime.batch_size is not None:
            raise ValueError(
                "the episode held-out loss draws its noise per batch, so its batch size is part of the "
                "recipe: pass --set suite.batch_size=N, not --batch-size"
            )
        return recorded_episode_run(run)


class EpisodeCheckpoint(Artifact):
    """A finished episode-model training run: a WAM, a world model, or a VLA."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]:
        return {HeldOutLoss.name: HeldOutLoss}
