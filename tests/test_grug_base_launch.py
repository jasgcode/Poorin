# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import AdamConfig
from levanter.tracker.telemetry import TelemetryConfig
from levanter.tracker.wandb import WandbConfig
from poorin_fray.cluster import ResourceConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, build_grug_run_config


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

    assert trackers == (WandbConfig(project="poorin", name="grug-base-test"), TelemetryConfig("http://127.0.0.1:9"))
