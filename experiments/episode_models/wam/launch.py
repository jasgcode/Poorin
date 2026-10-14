# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the WAM: future frames and actions, denoised together.

    uv run --no-sync python experiments/episode_models/wam/launch.py --version dev --run
    uv run --no-sync python experiments/episode_models/wam/launch.py --version dev \
        --sweep 'model.hidden_dim=[128, 256, 384]' --set train.steps=1000 --run

Without ``--run`` the plan is printed and nothing runs. ``--dataset`` picks an example episode
dataset; ``--set PATH=VALUE`` overrides any field of the launch config.
"""

import click
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.cli import build_options
from poorin.experiment.overrides import Overrides

from experiments.episode_models.evaluation import EpisodeCheckpoint
from experiments.episode_models.launch import episode_trial
from experiments.episode_models.sources import SOURCES
from experiments.episode_models.train import EpisodeTrainConfig
from experiments.episode_models.wam.model import WamConfig

DEFAULT_DATASET = "libero-spatial"


def wam_trial(
    *, dataset: str, overrides: Overrides, version: str | None = None, finelog_url: str | None = None
) -> ArtifactStep[EpisodeCheckpoint]:
    return episode_trial(
        family="wam",
        model=WamConfig(),
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
def main(dataset: str, finelog_url: str | None, overrides: Overrides) -> ArtifactStep[EpisodeCheckpoint]:
    return wam_trial(dataset=dataset, overrides=overrides, finelog_url=finelog_url)


if __name__ == "__main__":
    main()
