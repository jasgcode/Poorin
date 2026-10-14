# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluating grug base runs (D114): their checkpoint type and the held-out loss suite.

A grug base trial records :class:`GrugCheckpoint`. :class:`HeldOutLoss` scores a permanent
checkpoint on each of the run's validation sets with the evaluator training uses, and reports the
keys training logs without its timing keys: ``eval/loss``, ``eval/bpb``, ``eval/<set>/loss``, and
``eval/<set>/bpb``, with ``eval/macro_*`` when there are several sets, under ``eval/ema/`` for EMA
weights.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast, no_type_check

import jax
import jmp
from haliax.partitioning import set_mesh
from levanter.data.text.datasets import LmDataConfig
from levanter.eval import eval_model
from levanter.grug.sharding import compact_grug_mesh
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

from experiments.grug.base.model import GrugModelConfig, Transformer
from experiments.grug.base.train import GrugEvalConfig, tagged_evaluator
from experiments.grug.checkpointing import load_grug_checkpoint

# Training's evaluation settings: the prefix its keys log under, the batch sharding, and bpb.
TRAINING_EVAL = GrugEvalConfig()
TIMING_KEYS = ("loading_time", "total_time")
# The device batch size for a run that recorded none.
DEFAULT_BATCH_SIZE = 16


class Weights(StrEnum):
    EMA = "ema"
    CURRENT = "current"


@dataclass(frozen=True)
class RecordedGrugRun:
    """What the held-out loss reads from a run's recipe."""

    model: GrugModelConfig
    data: LmDataConfig
    mp: str
    ema: bool
    """Whether the run kept EMA weights."""
    eval_batch_size: int | None


def recorded_grug_run(run: FinishedRun) -> RecordedGrugRun:
    return RecordedGrugRun(
        model=recorded(run, "model", GrugModelConfig),
        data=recorded(run, "data", LmDataConfig),
        mp=recorded(run, "mp", str),
        ema=recorded_value(run, "ema_beta") is not None,
        eval_batch_size=cast(int | None, recorded_value(run, "eval_batch_size")),
    )


def held_out_metrics(
    run: RecordedGrugRun, checkpoint: str, suite: "HeldOutLoss", *, batch_size: int, compute_bpb: bool
) -> tuple[dict[str, float], int]:
    """The loss and bpb of the weights at ``checkpoint`` on each validation set, and the sequences scored."""
    ema = suite.weights is Weights.EMA and run.ema
    prefix = f"{TRAINING_EVAL.prefix}/ema" if ema else TRAINING_EVAL.prefix
    weights_field = "ema_params" if ema else "params"
    mp = jmp.get_policy(run.mp)
    mesh = compact_grug_mesh()
    with set_mesh(mesh):
        evaluator = tagged_evaluator(
            data_config=run.data,
            max_seq_len=run.model.max_seq_len,
            mesh=mesh,
            batch_size=batch_size,
            batch_pspec=TRAINING_EVAL.eval_batch_pspec,
            max_examples_per_dataset=suite.max_sequences,
            compute_bpb=compute_bpb,
        )
        if evaluator is None:
            raise ValueError(f"{checkpoint}: the run's data has no validation set")
        exemplar = jax.jit(lambda key: mp.cast_to_param(Transformer.init(run.model, key=key)))(jax.random.key(0))
        loaded = load_grug_checkpoint(
            state={weights_field: exemplar}, candidate=checkpoint, mesh=mesh, allow_partial=False
        )
        log = eval_model(evaluator, loaded[weights_field], prefix=prefix)
    metrics = {key: float(value) for key, value in log.items() if not key.endswith(TIMING_KEYS)}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError(f"the held-out loss at {checkpoint} is not finite: {metrics}")
    return metrics, len(evaluator.dataset.as_sync_dataset())


@dataclass(frozen=True)
class HeldOutLoss:
    """Loss and bits per byte on each of the run's validation sets at one checkpoint."""

    name: ClassVar[str] = "held-out"
    max_sequences: int | None = None
    """Sequences scored per validation set; ``None`` scores each whole set. Training caps each set at
    ``max_eval_batches * eval_batch_size``, so reproducing a run passes that cap."""
    weights: Weights = Weights.EMA
    """``ema`` falls back to the current weights for a run that kept no EMA."""

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        recorded_run = recorded_grug_run(run)
        batch_size = runtime.batch_size or recorded_run.eval_batch_size or DEFAULT_BATCH_SIZE
        metrics, examples = held_out_metrics(
            recorded_run, checkpoint_path(run, step), self, batch_size=batch_size, compute_bpb=TRAINING_EVAL.compute_bpb
        )
        return EvalResult(run=run.label, step=step, suite=self.name, metrics=metrics, examples=examples)


# Suite is a protocol with a data member, which issubclass cannot check, so the tests' beartype hook
# would reject the return annotation of suites().
@no_type_check
class GrugCheckpoint(Artifact):
    """A finished grug base training run."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]:
        return {HeldOutLoss.name: HeldOutLoss}
