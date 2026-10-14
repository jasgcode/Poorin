# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug base's held-out loss suite reproduces the evaluation training logged."""

import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import pytest
from fray.types import ResourceConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DatasetComponent, DirectDatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.examples import GrugLmExample
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from poorin.evaluation.contract import FinishedRun
from poorin.execution.artifact import ArtifactRecord
from poorin.execution.fingerprint import canonical_json

from experiments.grug.base import train as train_module
from experiments.grug.base.evaluation import HeldOutLoss, RecordedGrugRun, Weights, held_out_metrics, recorded_grug_run
from experiments.grug.base.launch import GrugBaseLaunchConfig
from experiments.grug.base.model import GrugModelConfig
from tests.test_grug_variant_contracts import json_tracker_logger, json_tracker_records, tiny_grug_base_run_config

RUN_ID = "test-grug-eval"
# The tiny run's TrainerConfig keeps Levanter's default policy, f32.
F32 = "params=float32,compute=float32,output=float32"


def _train(tmp_path: Path, *, ema_beta: float | None = None) -> tuple[train_module.GrugRunConfig, dict[str, float], str]:
    """A one-step tiny grug run; its config, the evaluation it logged, and its last checkpoint."""
    with json_tracker_logger() as (logger_name, stream):
        config = tiny_grug_base_run_config(tmp_path / "run", run_id=RUN_ID, logger_name=logger_name)
        config = dataclasses.replace(config, trainer=dataclasses.replace(config.trainer, ema_beta=ema_beta))
        train_module.run_grug(config)
    *_, logged = [
        r["metrics"] for r in json_tracker_records(stream) if r["event"] == "log" and "eval/loss" in r["metrics"]
    ]
    return config, logged, str(tmp_path / "run" / "checkpoints" / RUN_ID / "step-1")


def test_the_held_out_loss_of_a_checkpoint_reproduces_the_logged_one(tmp_path):
    config, logged, checkpoint = _train(tmp_path)
    recorded = RecordedGrugRun(model=config.model, data=config.data, mp=F32, ema=False, eval_batch_size=None)
    metrics, examples = held_out_metrics(
        recorded, checkpoint, HeldOutLoss(max_sequences=1, weights=Weights.CURRENT), batch_size=1, compute_bpb=False
    )
    expected = {k: v for k, v in logged.items() if not k.endswith(("loading_time", "total_time"))}
    assert metrics == pytest.approx(expected, rel=1e-5)
    assert examples == 1


def test_a_suite_built_with_the_string_ema_scores_the_ema_weights(tmp_path):
    config, _, checkpoint = _train(tmp_path, ema_beta=0.5)
    recorded = RecordedGrugRun(model=config.model, data=config.data, mp=F32, ema=True, eval_batch_size=None)
    suite = HeldOutLoss(max_sequences=1)
    # What HeldOutLoss(weights="ema") holds outside the tests, whose type-check hook refuses the str.
    object.__setattr__(suite, "weights", "ema")
    metrics, _ = held_out_metrics(recorded, checkpoint, suite, batch_size=1, compute_bpb=False)
    assert metrics and all(key.startswith("eval/ema/") for key in metrics)


def test_the_held_out_loss_does_not_depend_on_the_device_batch_size(tmp_path):
    config, _, checkpoint = _train(tmp_path)
    held_out = [GrugLmExample.causal((jnp.arange(32, dtype=jnp.int32) + i) % 128) for i in range(4)]
    data = LmDataConfig(
        components={"direct": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(held_out)})},
        vocab_size=128,
        tokenizer="passthrough",
    )
    recorded = RecordedGrugRun(model=config.model, data=data, mp=F32, ema=False, eval_batch_size=None)
    suite = HeldOutLoss(weights=Weights.CURRENT)
    losses = [
        held_out_metrics(recorded, checkpoint, suite, batch_size=size, compute_bpb=False)[0]["eval/loss"]
        for size in (1, 2, 4)
    ]
    assert losses == pytest.approx([losses[0]] * 3, rel=1e-6)


def test_the_suite_reads_the_run_from_its_record():
    text = TextLmDatasetFormat()
    data = LmDataConfig(
        components={
            "wikitext": DatasetComponent(
                source=UrlDatasetSourceConfig(cache_dir="/c/wikitext", format=text), cache_dir="/c/wikitext", format=text
            )
        },
        train_weights={"wikitext": 0.0},
        tokenizer="marin-community/marin-tokenizer",
    )
    model = GrugModelConfig(
        vocab_size=128, hidden_dim=32, intermediate_dim=64, num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=32
    )
    launch = GrugBaseLaunchConfig(
        model=model,
        data=data,
        output_path="/runs/grug",
        run_id="r",
        resources=ResourceConfig.with_cpu(),
        steps=1,
        batch_size=1,
        seed=0,
        mp=F32,
        tracker=JsonLoggerConfig(),
        finelog_url=None,
        optimizer=AdamConfig(),
        watch_interval=0,
        backward_flow_interval=0,
        ema_beta=0.999,
        eval_batch_size=16,
    )
    record = ArtifactRecord(name="grug/base-trial", version="2026.10.13", config=json.loads(canonical_json(launch)))
    recorded = recorded_grug_run(FinishedRun("grug/base-trial@2026.10.13", "/runs/grug", record))
    assert (recorded.model, recorded.mp, recorded.ema, recorded.eval_batch_size) == (model, F32, True, 16)
    assert json.loads(canonical_json(recorded.data)) == json.loads(canonical_json(data))
