# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from poorin.experiment.data import episodes
from poorin.experiment.overrides import Override, Overrides

from experiments.episode_models import sources
from experiments.episode_models.sources import EpisodeSource
from experiments.episode_models.tests.tiny_dataset import READER, SCHEMA
from experiments.episode_models.vla.launch import vla_trial
from experiments.episode_models.wam.launch import wam_trial
from experiments.episode_models.world_model.launch import world_model_trial

VERSION = "2026.10.11"


@pytest.fixture
def offline_vpt(monkeypatch):
    """VPT's real handle reads its episode index from the network; this one has VPT's fields."""
    handle = episodes("examples/vpt-10", sources=["0:16"], reader=READER, schema=SCHEMA, num_shards=1, version=VERSION)
    monkeypatch.setitem(
        sources.SOURCES,
        "vpt-10",
        EpisodeSource(lambda: handle, video_field="video", action_field="action", task_key=None),
    )


def payload(trial) -> dict:
    return json.loads(trial.fingerprint_payload())


def test_a_family_refuses_a_source_without_its_fields():
    with pytest.raises(ValueError, match="ssv2-500 has no action field"):
        vla_trial(dataset="ssv2-500", overrides=Overrides(), version=VERSION)


def test_a_set_reaches_the_model_and_names_the_trial():
    trial = wam_trial(
        dataset="libero-spatial", overrides=Overrides((Override.parse("model.hidden_dim=512"),)), version=VERSION
    )
    assert trial.name == "episode/wam-libero-spatial/model.hidden_dim=512"
    assert payload(trial)["model"]["hidden_dim"] == 512


def test_the_dataset_and_its_action_stats_are_dependencies():
    trial = wam_trial(dataset="libero-spatial", overrides=Overrides(), version=VERSION)
    assert [dep.label for dep in trial.deps] == [
        f"examples/libero-spatial@{VERSION}",
        f"episode-stats/examples/libero-spatial/action@{VERSION}",
    ]
    data = payload(trial)["data"]
    assert (data["video_field"], data["action_field"], data["task_key"]) == ("agentview", "action", "instruction")


def test_the_world_model_reads_no_actions_even_from_a_source_with_them(offline_vpt):
    trial = world_model_trial(dataset="vpt-10", overrides=Overrides(), version=VERSION)
    assert [dep.name for dep in trial.deps] == ["examples/vpt-10"]
    assert payload(trial)["data"]["action_field"] is None


def test_the_run_fields_stay_out_of_the_fingerprint():
    trial = wam_trial(dataset="libero-spatial", overrides=Overrides(), version=VERSION)
    with pytest.raises(Exception, match="step context"):
        wam_trial(
            dataset="libero-spatial", overrides=Overrides((Override.parse("run.run_id=mine"),)), version=VERSION
        ).fingerprint()
    assert payload(trial)["run"]["output_path"] == "<output_path>"
