# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The episode format's schema: fields, their encodings, and one step clock per dataset.

An episode is a set of named per-step arrays. A plain video is an episode with no action field.
Each field declares its per-step shape and dtype and how it is stored; ``dataset.json`` is
:meth:`DatasetSchema.to_json`.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import av
import numpy as np

RAW_ENCODING_NAME = "raw"
VIDEO_ENCODING_NAME = "video"


class FieldKind(StrEnum):
    VIDEO = "video"
    ARRAY = "array"


@dataclass(frozen=True)
class RawEncoding:
    """The array's bytes, C order: actions, proprioception, latents."""


@dataclass(frozen=True)
class VideoEncoding:
    """One compressed stream per episode with a keyframe every ``keyframe_interval`` steps, so a clip
    decodes from the nearest keyframe instead of the episode's start."""

    codec: str = "libx264"
    keyframe_interval: int = 16
    crf: int = 18

    def __post_init__(self) -> None:
        if av.Codec(self.codec, "w").type != "video":
            raise ValueError(f"{self.codec!r} is not a video codec")


FieldEncoding = RawEncoding | VideoEncoding


@dataclass(frozen=True)
class FieldSpec:
    kind: FieldKind
    dtype: str
    shape: tuple[int, ...]
    """Per step; an episode of length T stores ``[T, *shape]``."""
    encoding: FieldEncoding

    def __post_init__(self) -> None:
        if isinstance(self.encoding, VideoEncoding):
            if self.dtype != "uint8" or len(self.shape) != 3 or self.shape[2] != 3:
                raise ValueError(f"a video-encoded field is uint8 [H, W, 3], not {self.dtype} {list(self.shape)}")
            if self.shape[0] % 2 or self.shape[1] % 2:
                raise ValueError(f"yuv420p video needs an even height and width, not {list(self.shape[:2])}")


@dataclass(frozen=True)
class DatasetSchema:
    step_rate: float
    """Steps per second, shared by every field."""
    fields: Mapping[str, FieldSpec]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a dataset schema needs at least one field")

    def to_json(self) -> dict:
        return {"step_rate": self.step_rate, "fields": {name: _field_json(spec) for name, spec in self.fields.items()}}

    @classmethod
    def from_json(cls, data: dict) -> "DatasetSchema":
        return cls(
            step_rate=float(data["step_rate"]),
            fields={name: _field_from_json(spec) for name, spec in data["fields"].items()},
        )


@dataclass(frozen=True)
class Episode:
    id: str
    fields: Mapping[str, np.ndarray]
    """Field name to ``[length, *shape]``; every field has the same length."""
    metadata: Mapping[str, Any]
    """Per-episode values, stored in the index as JSON: a language instruction, an embodiment, a source."""


def validate_episode(schema: DatasetSchema, episode: Episode) -> int:
    """The episode's length, after checking it against ``schema``. Raises ``ValueError`` naming the
    episode and field on a missing or extra field, a wrong dtype or per-step shape, or unequal lengths."""
    if set(episode.fields) != set(schema.fields):
        field = sorted(set(episode.fields) ^ set(schema.fields))[0]
        raise ValueError(f"episode {episode.id!r}: field {field!r} is in only one of the episode and the schema")
    for name, spec in schema.fields.items():
        values = episode.fields[name]
        if not isinstance(values, np.ndarray) or values.ndim == 0:
            raise ValueError(
                f"episode {episode.id!r}: field {name!r} is a {type(values).__name__}, not a [length, *shape] array"
            )
        if values.dtype != np.dtype(spec.dtype) or values.shape[1:] != spec.shape:
            raise ValueError(
                f"episode {episode.id!r}: field {name!r} is {values.dtype} {list(values.shape[1:])} per step, "
                f"the schema says {spec.dtype} {list(spec.shape)}"
            )
    lengths = {name: len(episode.fields[name]) for name in schema.fields}
    if len(set(lengths.values())) > 1:
        described = ", ".join(f"field {name!r} has {length} steps" for name, length in lengths.items())
        raise ValueError(f"episode {episode.id!r}: fields differ in length: {described}")
    return next(iter(lengths.values()))


def _field_json(spec: FieldSpec) -> dict:
    match spec.encoding:
        case RawEncoding():
            encoding: dict = {"name": RAW_ENCODING_NAME}
        case VideoEncoding(codec=codec, keyframe_interval=interval, crf=crf):
            encoding = {"name": VIDEO_ENCODING_NAME, "codec": codec, "keyframe_interval": interval, "crf": crf}
    return {"kind": spec.kind.value, "dtype": spec.dtype, "shape": list(spec.shape), "encoding": encoding}


def _field_from_json(data: dict) -> FieldSpec:
    encoding = dict(data["encoding"])
    name = encoding.pop("name")
    if name == RAW_ENCODING_NAME:
        parsed: FieldEncoding = RawEncoding(**encoding)
    elif name == VIDEO_ENCODING_NAME:
        parsed = VideoEncoding(**encoding)
    else:
        raise ValueError(f"unknown field encoding {name!r}")
    return FieldSpec(kind=FieldKind(data["kind"]), dtype=data["dtype"], shape=tuple(data["shape"]), encoding=parsed)
