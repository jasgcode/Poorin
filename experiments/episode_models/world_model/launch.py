# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the video world model: future frames.

    uv run --no-sync python experiments/episode_models/world_model/launch.py --version dev --run
    uv run --no-sync python experiments/episode_models/world_model/launch.py --version dev \
        --sweep 'model.hidden_dim=[128, 256, 384]' --set train.steps=1000 --run

Without ``--run`` the plan is printed and nothing runs. ``--dataset`` picks an example episode
dataset; ``--set PATH=VALUE`` overrides any field of the launch config.
"""

import click
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.cli import build_options
from poorin.experiment.overrides import Overrides
from poorin.training.training import LevanterCheckpoint

from experiments.episode_models.launch import episode_trial
from experiments.episode_models.sources import SOURCES
from experiments.episode_models.train import EpisodeTrainConfig
from experiments.episode_models.world_model.model import WorldModelConfig

DEFAULT_DATASET = "ssv2-500"


def world_model_trial(
    *, dataset: str, overrides: Overrides, version: str | None = None, finelog_url: str | None = None
) -> ArtifactStep[LevanterCheckpoint]:
    return episode_trial(
        family="world-model",
        model=WorldModelConfig(),
        dataset=dataset,
        train=EpisodeTrainConfig(),
        overrides=overrides,
        version=version,
        finelog_url=finelog_url,
    )


@click.command()
@click.option("--dataset", type=click.Choice(sorted(SOURCES)), default=DEFAULT_DATASET, show_default=True)
@click.option("--finelog-url", default=None, help="Finelog server that receives the run's metrics.")
@build_options
def main(dataset: str, finelog_url: str | None, overrides: Overrides) -> ArtifactStep[LevanterCheckpoint]:
    return world_model_trial(dataset=dataset, overrides=overrides, finelog_url=finelog_url)


if __name__ == "__main__":
    main()
