# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import grain
import jax
import numpy as np
import pytest
from jax.sharding import Mesh
from poorin.episodes.build import EpisodeBuildConfig, EpisodeDataset, build_episodes
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset, clip_starts, to_mesh
from poorin.episodes.schema import DatasetSchema, Episode

from tests.episodes.test_build import READER, SCHEMA, synthetic_episode

VIDEO_ONLY = DatasetSchema(step_rate=10.0, fields={"video": SCHEMA.fields["video"]})
VIDEO_ONLY_READER = "tests.episodes.test_read:video_only_reader"


def video_only_reader(source, options):
    for item in source.split(","):
        episode_id, length = item.split(":")
        video = synthetic_episode(episode_id, int(length)).fields["video"]
        yield Episode(id=episode_id, fields={"video": video}, metadata={})


def build(tmp_path, name, sources, schema=SCHEMA, reader=READER) -> EpisodeDataset:
    config = EpisodeBuildConfig(
        output_path=str(tmp_path / name),
        sources=tuple(sources),
        reader=reader,
        reader_options={},
        schema=schema,
        num_shards=2,
    )
    build_episodes(config)
    return EpisodeDataset.raw_load(config.output_path)


def test_clip_starts_stay_inside_episodes():
    clip = ClipSpec(length=4, stride=2, skip=2)  # spans 7 steps
    rows, starts = clip_starts(np.array([10, 3, 7]), clip)
    # Episode 0 (10 steps): starts 0 and 2 fit (2 + 7 = 9 <= 10). Episode 1 is shorter than a clip.
    # Episode 2 (7 steps): start 0 only.
    assert rows.tolist() == [0, 0, 2]
    assert starts.tolist() == [0, 2, 0]


def test_a_clip_holds_the_requested_steps_of_every_selected_field(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12,b:7", "c:20"])
    clips = clip_dataset(
        dataset, clip=ClipSpec(length=5, stride=1, skip=3), fields=["video", "action"], metadata=["task"]
    )
    index = dataset.index
    expected_clips = sum(max(0, (n - 13) // 1 + 1) for n in index["length"])  # span is 13
    assert len(clips) == expected_clips

    last = clips[len(clips) - 1]  # the last valid start of the last episode, "c" (20 steps): start 7
    episode = synthetic_episode("c", 20)
    np.testing.assert_array_equal(last["action"], episode.fields["action"][7:20:3])
    # H.264 is lossy, so a pixel may be off by a little; selected steps differ by 24 (8 per step, skip 3).
    np.testing.assert_allclose(
        last["video"][:, 0, 0, 0].astype(int), episode.fields["video"][7:20:3, 0, 0, 0].astype(int), atol=3
    )
    assert str(last["task"]) == "task c"


def test_unselected_fields_are_not_returned(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12"])
    clip = clip_dataset(dataset, clip=ClipSpec(length=4, stride=4, skip=1), fields=["action"])[0]
    assert set(clip) == {"action"}


def test_resuming_at_a_batch_reads_what_an_uninterrupted_run_reads(tmp_path):
    dataset = build(tmp_path, "ds", ["a:30,b:25", "c:40"])
    clips = clip_dataset(dataset, clip=ClipSpec(length=4, stride=1, skip=1), fields=["action"])
    shuffled = clips.shuffle(seed=7).repeat(None)

    uninterrupted = shuffled.batch(8)
    resumed = batch_iterator(shuffled, batch_size=8, start_step=5, num_workers=2)
    iterator = iter(resumed)
    try:
        for step in range(5, 9):
            np.testing.assert_array_equal(next(iterator)["action"], uninterrupted[step]["action"])
    finally:
        iterator.close()

    other_seed = clips.shuffle(seed=8).repeat(None).batch(8)
    assert not np.array_equal(other_seed[0]["action"], uninterrupted[0]["action"])


def test_each_epoch_has_its_own_order(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12"])
    clips = clip_dataset(dataset, clip=ClipSpec(length=2, stride=1, skip=1), fields=["action"])
    shuffled = clips.shuffle(seed=0).repeat(2)
    first = [shuffled[i]["action"].tobytes() for i in range(len(clips))]
    second = [shuffled[len(clips) + i]["action"].tobytes() for i in range(len(clips))]
    assert sorted(first) == sorted(second)
    assert first != second


def test_a_video_only_dataset_mixes_with_one_that_has_actions(tmp_path):
    robot = build(tmp_path, "robot", ["a:12"])
    video = build(tmp_path, "video", ["v:12"], schema=VIDEO_ONLY, reader=VIDEO_ONLY_READER)
    clip = ClipSpec(length=4, stride=4, skip=1)
    video_clips = clip_dataset(video, clip=clip, fields=["video"])
    assert set(video_clips[0]) == {"video"}

    robot_frames = clip_dataset(robot, clip=clip, fields=["video"])
    mixed = grain.MapDataset.mix([robot_frames, video_clips], weights=[1, 1])
    assert {mixed[i]["video"].shape for i in range(4)} == {(4, 16, 16, 3)}


def test_to_mesh_shards_numeric_fields_and_passes_text_through():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    batch = {"action": np.zeros((4, 3, 2), np.float32), "task": np.array(["a", "b", "c", "d"])}
    placed = to_mesh(batch, mesh, ("data",))
    assert isinstance(placed["action"], jax.Array)
    assert placed["action"].sharding.spec == jax.sharding.PartitionSpec(("data",))
    np.testing.assert_array_equal(placed["task"], batch["task"])


@pytest.mark.parametrize("field", ["length", "stride", "skip"])
def test_a_clip_spec_below_one_step_is_refused(field):
    values = {"length": 4, "stride": 1, "skip": 1, field: 0}
    with pytest.raises(ValueError, match=field):
        ClipSpec(**values)


def test_a_dataset_with_no_clip_this_long_is_refused(tmp_path):
    dataset = build(tmp_path, "ds", ["a:12,b:7"])
    with pytest.raises(ValueError, match=r"no clip.*13 steps.*longest episode has 12"):
        clip_dataset(dataset, clip=ClipSpec(length=7, stride=1, skip=2), fields=["action"])
