# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""grug-base trial run, authored as a lazy artifact.

The run is a function that returns a typed :class:`Checkpoint` handle addressed by an
explicit ``name@version``. The model, optimizer, data mixture, token budget, and evals
are stated inline; the output path is ``ctx.output_path`` and the GPU is a run-arg, so neither
bears on the artifact's identity.

The grug training mechanism (``GrugBaseLaunchConfig`` + ``build_grug_run_config`` ->
``run_grug``) is grug-specific compute and is kept as-is: the recipe's ``fn`` builds the
Levanter trainer (its checkpointer paths derive from ``output_path``) and dispatches the
training job to Fray. Only the data/validation wiring around it is assembled lazily.

This launcher is Poorin's R2 prune root: it must train locally on one RTX 5090. Diagnostics
(gradient watch, backward flow) are off; enable them per run by setting their intervals.
"""

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from typing import no_type_check

import click
import jmp
from fray.types import ResourceConfig
from levanter.analysis.backward_flow import BackwardFlowConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from poorin.execution.build_context import resolve_version
from poorin.execution.lazy import ArtifactStep, StepContext
from poorin.experiment.cli import build_options
from poorin.experiment.data import mixture
from poorin.experiment.namespacing import user_namespaced_name
from poorin.experiment.overrides import Overrides
from poorin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.fineweb_edu import fineweb_edu_10m_dataset
from experiments.grug.base.model import GrugModelConfig
from experiments.grug.base.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# One local RTX 5090. A run-arg, not part of the config's identity: the same checkpoint can be
# rebuilt on other hardware. The local backend ignores these values; they describe the machine.
_TRAIN_RESOURCES = ResourceConfig.with_gpu("RTX5090", count=1, cpu=16, disk="128G", ram="96G")

_TRAIN_DATA = fineweb_edu_10m_dataset()

GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

# GRUG_130M_MODEL at 1024 tokens, which fits batch 16 on one RTX 5090 with room to spare.
_LOCAL_MODEL = dataclasses.replace(GRUG_130M_MODEL, max_seq_len=1024)


# Fields pulled from the step context hold placeholder strings such as "<train_resources>" while the
# fingerprint is computed, which the tests' beartype hook would reject.
@no_type_check
@dataclass(frozen=True)
class GrugBaseLaunchConfig:
    """Last-mile run config for the base grug template.

    Keep this as the main entry point for day-to-day edits (model/data/optimizer/trainer/eval knobs).

    The trainer and eval knobs are flat scalars rather than nested
    ``GrugTrainerConfig`` / ``GrugEvalConfig`` objects: those carry
    ``jax.sharding.PartitionSpec`` batch-sharding defaults, which have no canonical form, so
    the artifact fingerprint raises on them. ``build_grug_run_config`` reconstitutes them (with
    their default pspecs) at run time, keeping the sharding plumbing out of the
    artifact's identity.

    Each enabled diagnostic is a static flag on the jitted step, so it adds one compiled step
    variant (about 90 s each for the 130M model on an RTX 5090).
    """

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    finelog_url: str | None  # Finelog server that receives the run's metrics; None publishes none.
    optimizer: OptimizerConfig
    watch_interval: int  # gradient/parameter statistics every N steps; 0 disables them.
    backward_flow_interval: int  # backward-flow diagnostic every N steps; 0 disables it.
    z_loss_weight: float = 1e-4
    ema_beta: float | None = None  # EMA coefficient for eval/checkpoint model; None disables EMA.
    log_every: int = 1
    loss_implementation: str | tuple[str, ...] | None = None  # cross-entropy kernel; None uses the trainer default.
    eval_batch_size: int | None = 512  # None disables perplexity eval.
    steps_per_eval: int = 1000
    max_eval_batches: int = 8
    eval_current: bool = True
    eval_ema: bool = False


def _trackers(launch: GrugBaseLaunchConfig) -> tuple[TrackerConfig, ...]:
    tracker = launch.tracker
    if isinstance(tracker, WandbConfig):
        # replicate_path mirrors the run's metrics next to its output so they outlive the job.
        tracker = dataclasses.replace(tracker, name=launch.run_id, replicate_path=launch.output_path)
    return tracker, TelemetryConfig(finelog_url=launch.finelog_url)


def build_grug_run_config(launch: GrugBaseLaunchConfig) -> GrugRunConfig:
    """Map launch knobs onto the trainer's full ``GrugRunConfig``.

    The checkpointer's ``base_path`` and ``temporary_base_path`` derive from
    ``launch.output_path``, so a run resolves its checkpoint locations under the
    region its output path lives in.
    """
    output_path = launch.output_path
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=launch.seed,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(launch.mp),
        tracker=_trackers(launch),
        use_explicit_mesh_axes=True,
        watch=WatchConfig(interval=launch.watch_interval),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=10), keep=None),
            output_path,
        ),
    )

    grug_trainer = GrugTrainerConfig(
        trainer=trainer,
        z_loss_weight=launch.z_loss_weight,
        ema_beta=launch.ema_beta,
        log_every=launch.log_every,
        loss_implementation=launch.loss_implementation,
        backward_flow=BackwardFlowConfig(interval=launch.backward_flow_interval),
    )

    eval_config = (
        GrugEvalConfig(
            eval_batch_size=launch.eval_batch_size,
            steps_per_eval=launch.steps_per_eval,
            max_eval_batches=launch.max_eval_batches,
            eval_current=launch.eval_current,
            eval_ema=launch.eval_ema,
        )
        if launch.eval_batch_size is not None
        else None
    )

    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        output_path=output_path,
        optimizer=launch.optimizer,
        trainer=grug_trainer,
        eval=eval_config,
    )


def run_grug_base_trial(config: GrugBaseLaunchConfig) -> None:
    """Build the full grug run config and dispatch the training job to Fray.

    Runs inline on the launcher; ``run_grug`` submits the job and blocks until it
    completes.
    """
    run_grug(build_grug_run_config(config))


def grug_base_trial(
    *, overrides: Overrides, version: str | None = None, finelog_url: str | None = None
) -> ArtifactStep[LevanterCheckpoint]:
    """The base grug trial on 10M FineWeb-Edu tokens, on one local GPU, as a lazy checkpoint.

    ``overrides`` (from ``--set`` and ``--sweep``) apply to the ``GrugBaseLaunchConfig``; they enter
    the checkpoint's fingerprint and name, and its run ID. ``finelog_url`` names a Finelog server
    for the run's metrics. It is a runtime argument, so it does not change the checkpoint's identity.
    """
    base_name = "grug/base-trial"
    version = resolve_version(base_name, version)  # --version-of grug/base-trial=... reaches every trial
    train = {_TRAIN_DATA: 1.0}
    run_id = overrides.run_id(f"grug-base-{version}")

    def build_config(ctx: StepContext) -> GrugBaseLaunchConfig:
        return overrides.apply(
            ctx,
            GrugBaseLaunchConfig(
                model=_LOCAL_MODEL,
                data=mixture(ctx, train),
                output_path=ctx.output_path,
                run_id=ctx.runtime_arg("run_id"),
                resources=ctx.runtime_arg("train_resources"),
                steps=50,
                batch_size=16,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=ctx.runtime_arg("tracker"),
                finelog_url=ctx.runtime_arg("finelog_url"),
                optimizer=AdamConfig(
                    learning_rate=3e-3,
                    weight_decay=0.1,
                    lr_schedule="cosine",
                    decay=0.2,
                    min_lr_ratio=0.1,
                    warmup=10,
                ),
                watch_interval=0,
                backward_flow_interval=0,
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=ctx.runtime_arg("log_every"),
                eval_batch_size=None,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(overrides.named(base_name), version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_base_trial,
        build_config=build_config,
        deps=tuple(train),
        runtime_args={
            "train_resources": _TRAIN_RESOURCES,
            "finelog_url": finelog_url,
            "run_id": run_id,
            "tracker": WandbConfig(project="poorin", tags=["grug", "template"], group="grug-base-trial", name=None),
            "log_every": 1,
        },
    )


@click.command()
@click.option("--finelog-url", default=None, help="Finelog server that receives the run's metrics.")
@build_options
def main(finelog_url: str | None, overrides: Overrides) -> ArtifactStep[LevanterCheckpoint]:
    return grug_base_trial(finelog_url=finelog_url, overrides=overrides)


if __name__ == "__main__":
    main()
