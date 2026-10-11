# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from poorin.episodes.encodings import decode_field, encode_field
from poorin.episodes.schema import (
    DatasetSchema,
    Episode,
    FieldKind,
    FieldSpec,
    RawEncoding,
    VideoEncoding,
    validate_episode,
)

KEYFRAME_INTERVAL = 16


def moving_pattern(length: int, height: int = 64, width: int = 64) -> np.ndarray:
    """Frames whose content changes every step, so a frame from the wrong step is detectable."""
    t = np.arange(length)[:, None, None]
    y = np.arange(height)[None, :, None]
    x = np.arange(width)[None, None, :]
    channels = np.broadcast_arrays((x * 4 + t * 3) % 256, (y * 4 + t * 5) % 256, ((x + y) * 2) % 256)
    return np.stack(channels, axis=-1).astype(np.uint8)


def video_spec(height: int = 64, width: int = 64) -> FieldSpec:
    return FieldSpec(
        kind=FieldKind.VIDEO,
        dtype="uint8",
        shape=(height, width, 3),
        encoding=VideoEncoding(keyframe_interval=KEYFRAME_INTERVAL),
    )


def test_raw_field_round_trips_exactly():
    spec = FieldSpec(kind=FieldKind.ARRAY, dtype="float32", shape=(7,), encoding=RawEncoding())
    values = np.random.default_rng(0).normal(size=(40, 7)).astype(np.float32)
    data = encode_field(spec, 20.0, values)
    np.testing.assert_array_equal(decode_field(spec, 20.0, data, 40, range(0, 40)), values)
    np.testing.assert_array_equal(decode_field(spec, 20.0, data, 40, range(5, 30, 3)), values[5:30:3])


@pytest.mark.parametrize("step_rate", [20.0, 12.0, 29.97])
def test_video_clip_equals_the_same_steps_of_a_full_decode(step_rate):
    frames = moving_pattern(256)
    spec = video_spec()
    data = encode_field(spec, step_rate, frames)
    full = decode_field(spec, step_rate, data, 256, range(0, 256))
    for start in (0, 1, KEYFRAME_INTERVAL - 1, KEYFRAME_INTERVAL, KEYFRAME_INTERVAL + 1, 100, 240):
        clip = decode_field(spec, step_rate, data, 256, range(start, start + 16))
        np.testing.assert_array_equal(clip, full[start : start + 16])
    skipped = decode_field(spec, step_rate, data, 256, range(37, 83, 3))
    np.testing.assert_array_equal(skipped, full[37:83:3])


def test_video_decode_stays_close_to_the_original():
    frames = moving_pattern(64)
    spec = video_spec()
    decoded = decode_field(spec, 20.0, encode_field(spec, 20.0, frames), 64, range(0, 64))
    mse = np.mean((decoded.astype(np.float64) - frames) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    # yuv420p halves chroma resolution; this high-frequency color pattern measured 26.9 dB at crf 18.
    assert psnr > 24


def test_schema_round_trips_through_json():
    schema = DatasetSchema(
        step_rate=20.0,
        fields={"video": video_spec(), "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding())},
    )
    assert DatasetSchema.from_json(schema.to_json()) == schema


def test_an_episode_that_breaks_the_schema_names_the_episode_and_field():
    schema = DatasetSchema(
        step_rate=20.0,
        fields={"video": video_spec(), "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding())},
    )
    good = {"video": moving_pattern(10), "action": np.zeros((10, 7), np.float32)}
    assert validate_episode(schema, Episode(id="ok", fields=good, metadata={})) == 10

    wrong_shape = {**good, "action": np.zeros((10, 6), np.float32)}
    with pytest.raises(ValueError, match=r"episode 'e1'.*field 'action'"):
        validate_episode(schema, Episode(id="e1", fields=wrong_shape, metadata={}))
    missing = {"video": good["video"]}
    with pytest.raises(ValueError, match=r"episode 'e2'.*field 'action'"):
        validate_episode(schema, Episode(id="e2", fields=missing, metadata={}))
    unequal = {**good, "action": np.zeros((9, 7), np.float32)}
    with pytest.raises(ValueError, match=r"episode 'e3'.*field 'action'"):
        validate_episode(schema, Episode(id="e3", fields=unequal, metadata={}))


def test_a_video_encoding_with_a_non_video_codec_names_the_codec():
    with pytest.raises(ValueError, match=r"'aac' is not a video codec"):
        VideoEncoding(codec="aac")
