# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The episode templates' held-out loss suite reproduces the held-out loss training logged."""

import dataclasses
import json
from pathlib import Path

import pytest
from fray.types import ResourceConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from poorin.evaluation.contract import EvalRuntime, FinishedRun
from poorin.execution.artifact import ArtifactRecord, result_type_name
from poorin.execution.fingerprint import canonical_json

from experiments.episode_models.evaluation import (
    EpisodeCheckpoint,
    HeldOutLoss,
    RecordedEpisodeRun,
    held_out_metrics,
)
from experiments.episode_models.launch import EpisodeLaunchConfig, EpisodeRunConfig
from experiments.episode_models.tests.test_train import FAMILIES, json_records, tiny_run
from experiments.episode_models.train import EpisodeTrainingRun, run_episode_training

STEPS = 2
# The tiny runs evaluate one batch of two clips (tiny_run: eval_batches=1, eval_batch_size=2).
TINY_SUITE = HeldOutLoss(clips=2, batch_size=2)


def _train(run_dir: Path, family: str, tiny: tuple[str, str]) -> tuple[EpisodeTrainingRun, dict[str, float]]:
    """Train ``family`` for ``STEPS`` with its checkpoints at ``run_dir/checkpoints/step-N``; the run and
    the held-out loss it logged last."""
    with json_records() as (name, stream):
        run = tiny_run(run_dir, family, STEPS, tiny, name)
        checkpointer = dataclasses.replace(run.trainer.checkpointer, append_run_id_to_base_path=False)
        run = dataclasses.replace(run, trainer=dataclasses.replace(run.trainer, checkpointer=checkpointer))
        run_episode_training(run)
    records = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    (*_, logged) = [r["metrics"] for r in records if r["event"] == "log" and "eval/loss" in r["metrics"]]
    return run, logged


def _recorded(run: EpisodeTrainingRun) -> RecordedEpisodeRun:
    return RecordedEpisodeRun(
        model=run.model,
        data=run.data,
        mp=run.train.mp,
        dataset_path=run.dataset_path,
        action_stats_path=run.action_stats_path,
    )


def _finished(run: EpisodeTrainingRun, output: Path, *, stale_data_field: bool = False) -> FinishedRun:
    launch = EpisodeLaunchConfig(
        model=run.model,
        data=run.data,
        train=run.train,
        run=EpisodeRunConfig(
            output_path=str(output),
            run_id="test",
            resources=ResourceConfig.with_cpu(),
            tracker=JsonLoggerConfig(),
            finelog_url=None,
            dataset_path=run.dataset_path,
            action_stats_path=run.action_stats_path,
            num_workers=0,
            log_every=1,
            checkpoint_every=STEPS,
        ),
    )
    config = json.loads(canonical_json(launch))
    if stale_data_field:
        config["data"]["num_workers"] = 16  # as D113's acceptance runs recorded it
    record = ArtifactRecord(
        name="episode/tiny",
        version="2026.10.13",
        result_type=result_type_name(EpisodeCheckpoint),
        output_path=str(output),
        config=config,
    )
    return FinishedRun("episode/tiny@2026.10.13", str(output), record)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_held_out_loss_of_a_checkpoint_reproduces_the_logged_one(tmp_path, tiny, family):
    run, logged = _train(tmp_path, family, tiny)
    metrics = held_out_metrics(_recorded(run), str(tmp_path / "checkpoints" / f"step-{STEPS}"), TINY_SUITE)
    assert metrics == pytest.approx(logged, rel=1e-5)


def test_the_suite_reads_the_run_from_its_record(tmp_path, tiny):
    run, logged = _train(tmp_path, "wam", tiny)
    result = TINY_SUITE.evaluate(_finished(run, tmp_path), STEPS, str(tmp_path / "eval"), EvalRuntime(None))
    assert result.metrics == pytest.approx(logged, rel=1e-5)
    assert (result.run, result.step, result.suite, result.examples) == ("episode/tiny@2026.10.13", STEPS, "held-out", 2)


def test_a_record_with_a_field_the_code_dropped_names_the_migration(tmp_path, tiny):
    run = tiny_run(tmp_path, "wam", STEPS, tiny, "unused")
    finished = _finished(run, tmp_path, stale_data_field=True)
    with pytest.raises(ValueError, match=r"--drop data\.num_workers$"):
        TINY_SUITE.check(finished, EvalRuntime(None))


def test_the_suite_refuses_a_device_batch_size(tmp_path, tiny):
    run = tiny_run(tmp_path, "wam", STEPS, tiny, "unused")
    with pytest.raises(ValueError, match=r"suite\.batch_size"):
        TINY_SUITE.check(_finished(run, tmp_path), EvalRuntime(8))
