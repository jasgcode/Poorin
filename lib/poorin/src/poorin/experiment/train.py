# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A generic language-model training assembler.

:func:`train_lm` turns the *meaningful* decisions of a training run — the model, the
optimizer, the data, the token budget, the regularization, the evals — into a lazy
``ArtifactStep[LevanterCheckpoint]``. Every one of those is a required argument:
the helper defaults none of them, so reading the call shows the whole experiment. What
it *does* own is the mechanical training plumbing that is identical across runs and
carries no experiment meaning: the data-parallel mesh and token ``compute_mapping``,
the rolling resumption checkpointer, the eval-harness wiring, the WandB replication
path, and the Fray dispatch of the training job. That split is this design's
identity-vs-execution line — *what is computed* is the caller's, *how/where it runs* is
the library's.

This is deliberately **not** a ``default_train``: it bakes in no optimizer, no mixture,
no default eval suite, no learning rate. It only removes boilerplate.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.types import GpuConfig, ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.eval_harness_config import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.lm_model import LmConfig
from levanter.optim.config import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from poorin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from poorin.execution.artifact import is_mutable_version
from poorin.execution.build_context import resolve_version
from poorin.execution.lazy import ArtifactStep, StepContext
from poorin.execution.remote import remote
from poorin.experiment.data import mixture
from poorin.experiment.namespacing import user_namespaced_name
from poorin.processing.tokenize.tokenize import TokenizedCache
from poorin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

# Compute in bf16, keep master params and optimizer state in f32. The universal marin
# precision policy; it bears identity (it changes numerics), so overriding it is a
# deliberate experiment, but it is the same across essentially every marin LM run.
POORIN_PRECISION = "p=f32,c=bfloat16"

# The marin token axis maps onto the data-parallel mesh. This is hardware plumbing, not
# an experiment choice: it says nothing about what is computed, only how the sequence
# axis is laid out across the pod.
_TOKEN_AXES = (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA)

# Rolling resumption checkpoint cadence. Operational (it governs recovery, not the
# trained model), so it is not an experiment knob.
_RESUMPTION_INTERVAL = timedelta(minutes=10)


@dataclass(frozen=True)
class EvalSuite:
    """A set of harness tasks plus the step interval at which to run them."""

    tasks: tuple[EvalTaskConfig, ...]
    every: int


def _poorin_mesh(tensor_parallel_size: int) -> MeshConfig:
    """The standard marin training mesh: data parallel, optional tensor sharding.

    ``model`` is the tensor-parallel width (1 = no sharding); ``data`` absorbs the rest
    of the pod. The token axes ride the replica/data axes the marin path expects.
    """
    return MeshConfig(
        axes={"replica": 1, "data": -1, "model": tensor_parallel_size},
        compute_mapping={"token": _TOKEN_AXES, "token_repeat": _TOKEN_AXES},
    )


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch the assembled config as its own Fray training job."""
    # GPU: resolve the env now so XLA_FLAGS is in the pod environment before the
    # worker imports JAX. TPU/CPU resolve in-worker, where the WANDB_API_KEY the
    # TPU path requires is present (the GPU path skips that check).
    env_vars = (
        resolve_training_env(pod_config.env_vars, pod_config.resources)
        if isinstance(pod_config.resources.device, GpuConfig)
        else {}
    )
    remote(run_levanter_train_lm, resources=pod_config.resources, env_vars=env_vars)(pod_config)


def _run_tracker(ctx: StepContext) -> TrackerConfig:
    """The W&B tracker from the runtime args, mirroring its metrics next to the run's output."""
    tracker = ctx.runtime_arg("tracker")
    return tracker if ctx.is_fingerprint else dataclasses.replace(tracker, replicate_path=ctx.output_path)


def train_lm(
    *,
    name: str,
    model: LmConfig,
    optimizer: OptimizerConfig,
    datasets: Mapping[ArtifactStep[TokenizedCache], float],
    batch_size: int,
    seq_len: int,
    num_train_steps: int | None = None,
    num_train_epochs: int | None = None,
    z_loss_weight: float | None,
    evals: EvalSuite | None,
    resources: ResourceConfig,
    version: str | None = None,
    validation: Sequence[ArtifactStep[TokenizedCache]] = (),
    init_from: ArtifactStep[LevanterCheckpoint] | None = None,
    mp: str = POORIN_PRECISION,
    tensor_parallel_size: int = 1,
    steps_per_eval: int = 1000,
    wandb_project: str = "poorin",
    wandb_group: str | None = None,
    run_id: str | None = None,
    tags: Sequence[str] = (),
    env_vars: dict[str, str] | None = None,
) -> ArtifactStep[LevanterCheckpoint]:
    """Assemble a language-model training run as an ``ArtifactStep[LevanterCheckpoint]``.

    The required arguments are the run's identity-bearing decisions; the helper defaults
    none of them. ``datasets`` maps each tokenized-dataset handle to its mixture weight,
    and ``validation`` lists handles to add at weight 0; ``train_lm`` assembles the
    :func:`~poorin.experiment.data.mixture` internally and derives the step's deps from
    those handles, so they materialize first and the data config cannot desync from the
    dependencies. ``evals=None`` opts out of harness evals explicitly — there is no
    implicit default suite.

    The remaining parameters are execution choices that do not define the experiment:
    ``mp`` (the standard marin precision, identity-bearing but universal),
    ``tensor_parallel_size`` (model sharding width), and eval/checkpoint cadence. Tracker
    metadata (project, tags, group, run ID) and ``resources`` (the device the job asks for) are
    runtime args, so they never enter the checkpoint's fingerprint; ``LocalClient`` runs every
    job on this machine.
    ``init_from`` chains this run onto another checkpoint (it becomes a dep and seeds
    ``initialize_from_checkpoint_path``).

    Training length is set by exactly one of ``num_train_steps`` or ``num_train_epochs`` (setting
    both, or neither, is an error). ``num_train_epochs`` is a thin convenience over
    ``num_train_steps``: it resolves to ``ceil(num_train_epochs * num_train_tokens / (seq_len *
    batch_size))`` from the training cache's token total, which counts packed sequences rather than
    raw documents — the correct measure for packed SFT, where a hand-computed step count over raw
    documents can be several times too long. Epoch semantics are only defined for a single training
    dataset; pass ``num_train_steps`` directly for a mixture.

    A mutable (``dev``) ``version`` namespaces the checkpoint per user — its name becomes
    ``users/{username}/{name}`` so concurrent authors of the same experiment do not clobber each
    other; a fixed (calendar) ``version`` keeps the shared name. ``version`` defers to the ambient
    :class:`~poorin.execution.build_context.BuildContext` when omitted (resolved by ``name`` before
    the per-user namespacing), so a driver can set it once for the whole run via
    :mod:`poorin.experiment.cli`.
    """
    version = resolve_version(name, version)
    # A mutable run gets a fresh output path per invocation (poorin.experiment.cli.run_version);
    # suffix its tracker ID the same way so reruns do not log into the previous W&B run.
    tracker_id = f"{run_id}-{version}" if run_id is not None and is_mutable_version(version) else run_id
    if (num_train_steps is None) == (num_train_epochs is None):
        raise ValueError("Exactly one of num_train_steps or num_train_epochs must be set.")
    if num_train_epochs is not None:
        if num_train_epochs < 1:
            raise ValueError(f"num_train_epochs must be >= 1, got {num_train_epochs}")
        if len(datasets) != 1:
            raise ValueError(
                "num_train_epochs is only defined for a single training dataset; got "
                f"{len(datasets)}. Pass num_train_steps directly for a mixture."
            )

    harness = (
        LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(evals.tasks))) if evals is not None else None
    )
    all_deps = (*datasets, *validation, *((init_from,) if init_from is not None else ()))

    def resolve_num_train_steps(ctx: StepContext) -> int:
        """Steps for one full pass (times ``num_train_epochs``) over the training tokens.

        Dividing the split's token total by ``seq_len * batch_size`` counts packed sequences, not
        raw documents. At fingerprint time the token total is unavailable (the cache is not built),
        so we fall back to ``num_train_epochs`` as a placeholder that still keeps the epoch count in
        the artifact identity; the real step count is read from the cache at run time.
        """
        if num_train_steps is not None:
            return num_train_steps
        assert num_train_epochs is not None  # guaranteed by the mutual-exclusivity check above
        if ctx.is_fingerprint:
            return num_train_epochs
        (train_dataset,) = tuple(datasets)
        num_train_tokens = ctx.resolved(train_dataset).num_train_tokens
        return math.ceil(num_train_epochs * num_train_tokens / (seq_len * batch_size))

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        init_path = (
            LevanterCheckpoint(path=ctx.artifact_path(init_from)).checkpoint_dir if init_from is not None else None
        )
        inner = TrainLmConfig(
            data=mixture(ctx, datasets, validation=validation),
            trainer=TrainerConfig(
                id=ctx.runtime_arg("run_id"),
                tracker=_run_tracker(ctx),
                mp=jmp.get_policy(mp),
                train_batch_size=batch_size,
                per_device_parallelism=-1,
                num_train_steps=resolve_num_train_steps(ctx),
                steps_per_eval=steps_per_eval,
                checkpointer=CheckpointerConfig(save_interval=_RESUMPTION_INTERVAL, keep=[]),
                mesh=_poorin_mesh(tensor_parallel_size),
                per_device_eval_parallelism=-1,
                allow_nondivisible_batch_size=True,
            ),
            model=model,
            optimizer=optimizer,
            z_loss_weight=z_loss_weight,
            train_seq_len=seq_len,
            initialize_from_checkpoint_path=init_path,
            eval_harness=harness,
            eval_harness_steps=evals.every if evals is not None else None,
            adapter=NoAdaptorConfig(),
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            env_vars=env_vars,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=all_deps,
        runtime_args={
            "train_resources": resources,
            "run_id": tracker_id,
            "tracker": WandbConfig(project=wandb_project, name=tracker_id, tags=[*tags], group=wandb_group),
        },
    )
