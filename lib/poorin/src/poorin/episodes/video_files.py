# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode a video file to frames at a dataset's size and step rate, for source readers."""

import av
import numpy as np

FRAME_FORMAT = "rgb24"


def load_video(path: str, *, size: tuple[int, int], step_rate: float) -> np.ndarray:
    """Every step of the video at ``path`` as ``uint8 [T, height, width, 3]``.

    Frames are scaled so the shorter side matches and then center-cropped, and resampled to
    ``step_rate`` by taking the source frame nearest each step's time.
    """
    height, width = size
    times: list[float] = []
    frames: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        scale = max(height / stream.height, width / stream.width)
        scaled_height, scaled_width = round(stream.height * scale), round(stream.width * scale)
        for frame in container.decode(stream):
            assert frame.time is not None
            times.append(frame.time)
            scaled = frame.reformat(width=scaled_width, height=scaled_height, format=FRAME_FORMAT).to_ndarray()
            top, left = (scaled_height - height) // 2, (scaled_width - width) // 2
            frames.append(scaled[top : top + height, left : left + width])
    source_times = np.asarray(times) - times[0]
    duration = source_times[-1] + (source_times[1] - source_times[0] if len(times) > 1 else 0.0)
    step_times = np.arange(round(duration * step_rate)) / step_rate
    nearest = np.abs(source_times[None, :] - step_times[:, None]).argmin(axis=1)
    return np.stack([frames[index] for index in nearest])
