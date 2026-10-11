# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encode a field's ``[length, *shape]`` array to bytes, and decode any steps of it back.

A new encoding is one class in :mod:`poorin.episodes.schema` and one branch in each function here.
"""

import io
from fractions import Fraction

import av
import numpy as np

from poorin.episodes.schema import FieldSpec, RawEncoding, VideoEncoding

VIDEO_CONTAINER = "mp4"
VIDEO_PIXEL_FORMAT = "yuv420p"
FRAME_FORMAT = "rgb24"
# Keeps a rate such as 29.97 exact as 30000/1001.
MAX_RATE_DENOMINATOR = 1001


def encode_field(spec: FieldSpec, step_rate: float, values: np.ndarray) -> bytes:
    match spec.encoding:
        case RawEncoding():
            return np.ascontiguousarray(values, dtype=spec.dtype).tobytes()
        case VideoEncoding() as encoding:
            return _encode_video(encoding, step_rate, values)


def decode_field(spec: FieldSpec, step_rate: float, data: bytes, length: int, steps: range) -> np.ndarray:
    """Steps ``steps`` of a field stored as ``data``, as ``[len(steps), *spec.shape]``."""
    match spec.encoding:
        case RawEncoding():
            return np.frombuffer(data, dtype=spec.dtype).reshape(length, *spec.shape)[
                steps.start : steps.stop : steps.step
            ]
        case VideoEncoding():
            return _decode_video(step_rate, data, steps)


def _encode_video(encoding: VideoEncoding, step_rate: float, frames: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=VIDEO_CONTAINER) as container:
        stream = container.add_stream(encoding.codec, rate=Fraction(step_rate).limit_denominator(MAX_RATE_DENOMINATOR))
        assert isinstance(stream, av.VideoStream)  # VideoEncoding checks its codec is a video codec
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = VIDEO_PIXEL_FORMAT
        interval = str(encoding.keyframe_interval)
        # No B-frames and no scene-cut keyframes: frame i decodes at pts i and keyframes fall exactly
        # every keyframe_interval frames, which the seek in _decode_video relies on.
        stream.options = {
            "g": interval,
            "keyint_min": interval,
            "sc_threshold": "0",
            "bf": "0",
            "crf": str(encoding.crf),
        }
        for index, frame in enumerate(frames):
            video_frame = av.VideoFrame.from_ndarray(frame, format=FRAME_FORMAT)
            video_frame.pts = index
            container.mux(stream.encode(video_frame))
        container.mux(stream.encode())
    return buffer.getvalue()


def _decode_video(step_rate: float, data: bytes, steps: range) -> np.ndarray:
    wanted = set(steps)
    decoded: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(data), mode="r", format=VIDEO_CONTAINER) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        container.seek(int(steps.start / step_rate / time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            assert frame.pts is not None
            index = round(float(frame.pts * time_base) * step_rate)
            if index >= steps.stop:
                break
            if index in wanted:
                decoded[index] = frame.to_ndarray(format=FRAME_FORMAT)
    return np.stack([decoded[index] for index in steps])
