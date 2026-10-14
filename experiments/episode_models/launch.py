# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an episode model's training trial as a lazy checkpoint artifact.

A family's ``launch.py`` passes its model config and training defaults to :func:`episode_trial`,
which names the trial ``episode/<family>-<dataset>``, makes the dataset (and its action stats) the
trial's dependencies, and pulls the output path, run id, GPU, and tracker from the step context, so
they never change the trial's identity. ``--set`` and ``--sweep`` reach any field of
:class:`EpisodeLaunchConfig` by dotted path: ``model.hidden_dim``, ``train.steps``,
``data.clip.horizon``.
"""

import dataclasses
from dataclasses import dataclass
from datetime import timedelta
from typing import no_type_check

import jmp
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from poorin.execution.build_context import resolve_version
from poorin.execution.lazy import ArtifactStep, StepContext
from poorin.experiment.namespacing import user_namespaced_name
from poorin.experiment.overrides import Overrides
from poorin.training.training import resolve_checkpointer_output_path

from experiments.episode_models.data import EpisodeDataConfig, action_stats
from experiments.episode_models.evaluation import EpisodeCheckpoint
from experiments.episode_models.sources import SOURCES
from experiments.episode_models.train import (
    EpisodeModelConfig,
    EpisodeTrainConfig,
    EpisodeTrainingRun,
    run_episode_training,
)

# Execution choices: runtime args, so changing one never changes a trial's identity.
# One local RTX 5090.
TRAIN_RESOURCES = ResourceConfig.with_gpu("RTX5090", count=1, cpu=16, disk="128G", ram="96G")
# Each loader process holds about 0.9 GB with poorin.episodes.read's bounded read-ahead.
NUM_WORKERS = 16
LOG_EVERY = 1
# Steps between permanent checkpoints; time-based temporary ones come every interval.
CHECKPOINT_EVERY = 500
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=10)


# Fields pulled from the step context hold placeholder strings while the fingerprint is computed,
# which the tests' beartype hook would reject.
@no_type_check
@dataclass(frozen=True)
class EpisodeRunConfig:
    """Where and how a trial runs, all from the step context. The dataset and stats paths enter the
    trial's identity only as their dependencies' labels; the other fields never do, and ``--set``
    refuses all of them."""

    output_path: str
    run_id: str
    resources: ResourceConfig
    tracker: TrackerConfig
    finelog_url: str | None
    dataset_path: str
    action_stats_path: str | None
    num_workers: int
    log_every: int
    checkpoint_every: int


@no_type_check
@dataclass(frozen=True)
class EpisodeLaunchConfig:
    model: EpisodeModelConfig
    data: EpisodeDataConfig
    train: EpisodeTrainConfig
    run: EpisodeRunConfig


def _trackers(launch: EpisodeLaunchConfig) -> tuple[TrackerConfig, ...]:
    tracker = launch.run.tracker
    if isinstance(tracker, WandbConfig):
        # replicate_path mirrors the run's metrics next to its output so they outlive the job.
        tracker = dataclasses.replace(tracker, name=launch.run.run_id, replicate_path=launch.run.output_path)
    return tracker, TelemetryConfig(finelog_url=launch.run.finelog_url)


def build_training_run(launch: EpisodeLaunchConfig) -> EpisodeTrainingRun:
    """Map a launch config onto the training loop's run: Levanter's trainer config with checkpoints
    under the trial's output path, a permanent one every ``run.checkpoint_every`` steps."""
    trainer = TrainerConfig(
        id=launch.run.run_id,
        seed=launch.train.seed,
        train_batch_size=launch.train.batch_size,
        num_train_steps=launch.train.steps,
        mp=jmp.get_policy(launch.train.mp),
        tracker=_trackers(launch),
        require_accelerator=True,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL, keep=[dict(every=launch.run.checkpoint_every)]
            ),
            launch.run.output_path,
        ),
    )
    return EpisodeTrainingRun(
        model=launch.model,
        dataset_path=launch.run.dataset_path,
        action_stats_path=launch.run.action_stats_path,
        data=launch.data,
        train=launch.train,
        trainer=trainer,
        resources=launch.run.resources,
        num_workers=launch.run.num_workers,
        log_every=launch.run.log_every,
    )


def run_episode_trial(launch: EpisodeLaunchConfig) -> None:
    run_episode_training(build_training_run(launch))


def episode_trial(
    *,
    family: str,
    model: EpisodeModelConfig,
    dataset: str,
    train: EpisodeTrainConfig,
    overrides: Overrides,
    version: str | None,
    finelog_url: str | None,
) -> ArtifactStep[EpisodeCheckpoint]:
    """A family's trial on the example dataset named ``dataset`` (a key of ``SOURCES``)."""
    if dataset not in SOURCES:
        raise ValueError(f"unknown dataset {dataset!r}; the sources are {sorted(SOURCES)}")
    source = SOURCES[dataset]
    if model.needs_actions and source.action_field is None:
        raise ValueError(f"{dataset} has no action field, which the {family} template needs")
    action_field = source.action_field if model.needs_actions else None
    data = EpisodeDataConfig(
        video_field=source.video_field,
        action_field=action_field,
        task_key=source.task_key if model.conditions_on_task else None,
    )
    dataset_handle = source.dataset()
    stats_handle = action_stats(dataset_handle, action_field) if action_field is not None else None
    deps = (dataset_handle,) if stats_handle is None else (dataset_handle, stats_handle)

    base_name = f"episode/{family}-{dataset}"
    version = resolve_version(base_name, version)
    run_id = overrides.run_id(f"{family}-{dataset}-{version}")

    def build_config(ctx: StepContext) -> EpisodeLaunchConfig:
        run = EpisodeRunConfig(
            output_path=ctx.output_path,
            run_id=ctx.runtime_arg("run_id"),
            resources=ctx.runtime_arg("train_resources"),
            tracker=ctx.runtime_arg("tracker"),
            finelog_url=ctx.runtime_arg("finelog_url"),
            dataset_path=ctx.artifact_path(dataset_handle),
            action_stats_path=ctx.artifact_path(stats_handle) if stats_handle is not None else None,
            num_workers=ctx.runtime_arg("num_workers"),
            log_every=ctx.runtime_arg("log_every"),
            checkpoint_every=ctx.runtime_arg("checkpoint_every"),
        )
        return overrides.apply(ctx, EpisodeLaunchConfig(model=model, data=data, train=train, run=run))

    return ArtifactStep(
        name=user_namespaced_name(overrides.named(base_name), version),
        version=version,
        artifact_type=EpisodeCheckpoint,
        run=run_episode_trial,
        build_config=build_config,
        deps=deps,
        runtime_args={
            "train_resources": TRAIN_RESOURCES,
            "finelog_url": finelog_url,
            "run_id": run_id,
            "tracker": WandbConfig(project="poorin", tags=["episode", family], group=f"episode-{family}", name=None),
            "num_workers": NUM_WORKERS,
            "log_every": LOG_EVERY,
            "checkpoint_every": CHECKPOINT_EVERY,
        },
    )
