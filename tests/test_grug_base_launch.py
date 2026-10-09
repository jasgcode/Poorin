# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from fray.types import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from poorin.execution.artifact import ArtifactRecord, write_record
from poorin.execution.lazy import materialized_config
from poorin.processing.tokenize.tokenize import TokenizedCache

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, build_grug_run_config, grug_base_trial


def test_the_trial_publishes_its_metrics_to_the_finelog_url_it_is_given(tmp_path):
    launch = GrugBaseLaunchConfig(
        model=GRUG_130M_MODEL,
        data=LmDataConfig(),
        output_path=str(tmp_path),
        run_id="grug-base-test",
        resources=ResourceConfig.with_cpu(),
        steps=1,
        batch_size=1,
        seed=0,
        mp="params=float32,compute=float32,output=float32",
        tracker=WandbConfig(project="poorin"),
        finelog_url="http://127.0.0.1:9",
        optimizer=AdamConfig(),
        watch_interval=0,
        backward_flow_interval=0,
    )

    trackers = build_grug_run_config(launch).trainer.trainer.tracker

    assert trackers == (
        WandbConfig(project="poorin", name="grug-base-test", replicate_path=str(tmp_path)),
        TelemetryConfig("http://127.0.0.1:9"),
    )


def test_tracker_settings_do_not_change_the_trials_fingerprint():
    trial = grug_base_trial(version="2026.10.09")
    retagged = dataclasses.replace(
        trial,
        runtime_args={**trial.runtime_args, "tracker": WandbConfig(project="other", tags=["x"]), "log_every": 10},
    )
    assert retagged.fingerprint() == trial.fingerprint()


def test_the_trials_tracker_mirrors_its_metrics_next_to_the_output(tmp_path):
    trial = grug_base_trial(version="2026.10.09")
    # The minimal record a built dataset leaves, so mixture() can read its tokenizer offline.
    for dep in trial.deps:
        write_record(
            ArtifactRecord(
                name=dep.name,
                version=dep.version,
                output_path=dep.path(str(tmp_path)),
                result_type=f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}",
                config={"tokenizer": "marin-community/marin-tokenizer", "format": {"text_key": "text"}},
            )
        )
    launch = materialized_config(trial, str(tmp_path))
    wandb, _ = build_grug_run_config(launch).trainer.trainer.tracker
    assert wandb.replicate_path == trial.path(str(tmp_path))
