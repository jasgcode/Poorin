# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import json
import logging
import uuid
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.distributed import DistributedConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig

from experiments.episode_models.data import ClipConfig, EpisodeDataConfig, compute_action_stats
from experiments.episode_models.tests.tiny_dataset import build_tiny
from experiments.episode_models.train import EpisodeTrainConfig, EpisodeTrainingRun, run_episode_training
from experiments.episode_models.vla.model import VlaConfig
from experiments.episode_models.wam.model import WamConfig
from experiments.episode_models.world_model.model import WorldModelConfig

FAMILIES = {"wam": WamConfig, "world_model": WorldModelConfig, "vla": VlaConfig}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> tuple[str, str]:
    """The tiny dataset's path and its action stats' path, built once for the module."""
    root = tmp_path_factory.mktemp("data")
    dataset = build_tiny(root)
    compute_action_stats(dataset.path, "action", str(root / "stats"))
    return dataset.path, str(root / "stats")


@contextlib.contextmanager
def json_records() -> Iterator[tuple[str, StringIO]]:
    name = f"test_episode_models_{uuid.uuid4().hex}"
    stream = StringIO()
    logger = logging.getLogger(name)
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield name, stream
    finally:
        logger.removeHandler(handler)


def tiny_run(run_dir: Path, family: str, steps: int, tiny: tuple[str, str], logger_name: str) -> EpisodeTrainingRun:
    dataset_path, stats_path = tiny
    config = FAMILIES[family](hidden_dim=64, num_layers=1, patch_size=8)
    reads_actions = "action" in config.streams or family == "wam"
    data = EpisodeDataConfig(
        video_field="video",
        action_field="action" if reads_actions else None,
        task_key="instruction" if config.conditions_on_task else None,
        clip=ClipConfig(history=2, horizon=2, stride=2, skip=1),
        eval_percent=50,
        num_workers=0,
    )
    train = EpisodeTrainConfig(
        steps=steps,
        batch_size=2,
        seed=0,
        mp="params=float32,compute=float32,output=float32",
        optimizer=AdamConfig(learning_rate=1e-3, lr_schedule="constant", warmup=0.0, weight_decay=0.0),
        steps_per_eval=steps,
        eval_batches=1,
        eval_batch_size=2,
        checkpoint_every=1000,
    )
    trainer = TrainerConfig(
        id=f"test-{family}",
        num_train_steps=steps,
        train_batch_size=2,
        tracker=JsonLoggerConfig(logger_name=logger_name),
        require_accelerator=False,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_dir=run_dir / "logs",
        checkpointer=CheckpointerConfig(base_path=str(run_dir / "checkpoints")),
    )
    return EpisodeTrainingRun(
        model=config,
        dataset_path=dataset_path,
        action_stats_path=stats_path if reads_actions else None,
        data=data,
        train=train,
        trainer=trainer,
        resources=ResourceConfig.with_cpu(),
    )


def train_tiny(run_dir: Path, family: str, steps: int, tiny: tuple[str, str]) -> list[dict]:
    with json_records() as (name, stream):
        run_episode_training(tiny_run(run_dir, family, steps, tiny, name))
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def losses_by_step(records: list[dict]) -> dict[int, float]:
    return {
        r["step"]: r["metrics"]["train/loss"]
        for r in records
        if r["event"] == "log" and "train/loss" in r["metrics"] and r["step"] is not None
    }


def finish_summary(records: list[dict]) -> dict:
    (finish,) = [r for r in records if r["event"] == "finish"]
    return finish["summary"]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_trains_and_logs_the_sweep_contract(tmp_path, tiny, family):
    summary = finish_summary(train_tiny(tmp_path, family, steps=3, tiny=tiny))
    for key in [
        "train/loss",
        "eval/loss",
        "parameter_count",
        "throughput/flops_per_example_analytic",
        "throughput/total_tokens",
    ]:
        assert key in summary, key
    for stream in FAMILIES[family].streams:
        assert f"eval/{stream}_loss" in summary
        assert f"train/{stream}_loss" in summary
    assert np.isfinite(summary["eval/loss"])


def test_a_resumed_run_reports_the_losses_of_an_uninterrupted_one(tmp_path, tiny):
    uninterrupted = losses_by_step(train_tiny(tmp_path / "a", "wam", steps=6, tiny=tiny))
    train_tiny(tmp_path / "b", "wam", steps=4, tiny=tiny)
    resumed = losses_by_step(train_tiny(tmp_path / "b", "wam", steps=6, tiny=tiny))
    assert sorted(resumed) == [4, 5]
    for step in (4, 5):
        np.testing.assert_allclose(resumed[step], uninterrupted[step], rtol=1e-6)


def test_the_held_out_loss_is_the_same_on_every_evaluation(tmp_path, tiny):
    # A learning rate of 0 keeps the parameters fixed, so the two evaluations see the same model.
    frozen = AdamConfig(learning_rate=0.0, lr_schedule="constant", warmup=0.0)
    with json_records() as (name, stream):
        run = tiny_run(tmp_path, "wam", steps=2, tiny=tiny, logger_name=name)
        run = dataclasses.replace(run, train=dataclasses.replace(run.train, steps_per_eval=1, optimizer=frozen))
        run_episode_training(run)
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    evals = [r["metrics"]["eval/loss"] for r in records if r["event"] == "log" and "eval/loss" in r["metrics"]]
    assert len(evals) == 2 and evals[0] == evals[1]
