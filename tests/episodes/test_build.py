# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pytest
from poorin.episodes.build import EpisodeBuildConfig, EpisodeDataset, build_episodes
from poorin.episodes.encodings import decode_field
from poorin.episodes.read import read_record
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.execution.lazy import resolve
from poorin.experiment.data import episodes

STEP_RATE = 10.0
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "video": FieldSpec(FieldKind.VIDEO, "uint8", (16, 16, 3), VideoEncoding(keyframe_interval=4)),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (2,), RawEncoding()),
    },
)
READER = "tests.episodes.test_build:synthetic_reader"


def synthetic_episode(episode_id: str, length: int) -> Episode:
    seed = int.from_bytes(episode_id.encode(), "little") % 2**32
    rng = np.random.default_rng(seed)
    video = np.broadcast_to(np.arange(length, dtype=np.uint8)[:, None, None, None] * 8, (length, 16, 16, 3)).copy()
    action = rng.normal(size=(length, 2)).astype(np.float32)
    return Episode(id=episode_id, fields={"video": video, "action": action}, metadata={"task": f"task {episode_id}"})


def synthetic_reader(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """Each source names its episodes as ``id:length,id:length``; ``options['broken']`` breaks one."""
    for item in source.split(","):
        episode_id, length = item.split(":")
        episode = synthetic_episode(episode_id, int(length))
        if options.get("broken") == episode_id:
            wrong_action = np.zeros((int(length), 3), np.float32)
            episode = Episode(id=episode_id, fields={**episode.fields, "action": wrong_action}, metadata={})
        yield episode


def error_messages(error: BaseException | None) -> list[str]:
    """The messages along an exception's cause chain: Zephyr may wrap a worker's error."""
    messages = []
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    return messages


def build(tmp_path, sources, **options) -> EpisodeDataset:
    config = EpisodeBuildConfig(
        output_path=str(tmp_path / "ds"),
        sources=tuple(sources),
        reader=READER,
        reader_options=options,
        schema=SCHEMA,
        num_shards=2,
    )
    build_episodes(config)
    return EpisodeDataset.raw_load(config.output_path)


def test_build_writes_schema_index_and_records(tmp_path):
    dataset = build(tmp_path, ["a:12,b:7", "c:20"])

    assert dataset.schema == SCHEMA
    assert json.loads((tmp_path / "ds" / "dataset.json").read_text()) == SCHEMA.to_json()
    index = dataset.index
    assert sorted(index["episode"]) == ["a", "b", "c"]
    assert dict(zip(index["episode"], index["length"], strict=True)) == {"a": 12, "b": 7, "c": 20}
    assert json.loads(index.set_index("episode").loc["c", "metadata"]) == {"task": "task c"}
    assert len(dataset.shard_paths) == 2

    row = index.set_index("episode").loc["c"]
    record = read_record(dataset, int(row["shard"]), int(row["record"]))
    expected = synthetic_episode("c", 20)
    np.testing.assert_array_equal(
        decode_field(SCHEMA.fields["action"], STEP_RATE, record["action"], 20, range(20)), expected.fields["action"]
    )


def test_an_episode_that_breaks_the_schema_fails_the_build(tmp_path):
    with pytest.raises(Exception) as raised:
        build(tmp_path, ["a:12,b:7"], broken="b")
    assert any(re.search(r"episode 'b'.*field 'action'", message) for message in error_messages(raised.value))


def test_a_duplicate_episode_id_fails_the_build(tmp_path):
    with pytest.raises(ValueError, match=r"episode 'a'.*more than once"):
        build(tmp_path, ["a:12", "a:5"])


def test_the_handle_builds_and_its_fingerprint_follows_the_schema(tmp_path):
    handle = episodes(
        "synthetic", sources=["a:12", "b:9"], reader=READER, schema=SCHEMA, num_shards=1, version="2026.10.11"
    )
    smaller = DatasetSchema(
        step_rate=STEP_RATE,
        fields={
            **SCHEMA.fields,
            "video": FieldSpec(FieldKind.VIDEO, "uint8", (8, 8, 3), VideoEncoding(keyframe_interval=4)),
        },
    )
    resized = episodes(
        "synthetic", sources=["a:12", "b:9"], reader=READER, schema=smaller, num_shards=1, version="2026.10.11"
    )

    assert handle.fingerprint() != resized.fingerprint()
    dataset = resolve(handle)
    assert sorted(dataset.index["episode"]) == ["a", "b"]


def test_an_object_store_output_path_is_refused_before_anything_is_written(tmp_path):
    config = EpisodeBuildConfig(
        output_path="gs://bucket/ds",
        sources=("a:12",),
        reader=READER,
        reader_options={},
        schema=SCHEMA,
        num_shards=1,
    )
    with pytest.raises(ValueError, match=r"local path.*gs://bucket/ds"):
        build_episodes(config)


def test_an_object_store_dataset_is_refused_when_read():
    with pytest.raises(ValueError, match=r"local path.*gs://bucket/ds"):
        list(EpisodeDataset(path="gs://bucket/ds").shard_paths)
