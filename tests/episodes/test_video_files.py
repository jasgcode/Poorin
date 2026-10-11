# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import av
import numpy as np
from poorin.episodes.video_files import load_video

SOURCE_RATE = 30
SOURCE_FRAMES = 60


def write_source_video(path, height: int, width: int) -> None:
    """A source video whose red channel encodes the frame number, so resampling is checkable."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=SOURCE_RATE)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"bf": "0", "crf": "0"}
        for index in range(SOURCE_FRAMES):
            frame = np.zeros((height, width, 3), np.uint8)
            frame[..., 0] = index * 4
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = index
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())


def test_load_video_resizes_crops_and_resamples(tmp_path):
    path = tmp_path / "source.mp4"
    write_source_video(path, height=90, width=160)

    frames = load_video(str(path), size=(32, 32), step_rate=10.0)

    # 60 frames at 30 fps is 2 seconds, so 20 steps at 10 Hz.
    assert frames.shape == (20, 32, 32, 3)
    assert frames.dtype == np.uint8
    # Step i is the source frame nearest i / 10 s, which is frame 3 * i.
    red = frames[:, 16, 16, 0].astype(int)
    np.testing.assert_allclose(red, np.arange(20) * 3 * 4, atol=6)
