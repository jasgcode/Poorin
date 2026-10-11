# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A few Minecraft VPT contractor episodes: video with keyboard and mouse actions."""

import json
import os
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.episodes.video_files import load_video
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

BASE_URL = "https://openaipublic.blob.core.windows.net/minecraft-rl/"
# A fixed snapshot of OpenAI's contractor-data index (71,868 episodes); the first episodes are the subset.
INDEX_URL = BASE_URL + "snapshots/all_6xx_Jun_29.json"
SUBSET_EPISODES = 10
STEP_RATE = 20.0  # Minecraft's tick rate in the recordings
SIZE = (128, 128)
KEYS = (
    "key.keyboard.w",
    "key.keyboard.a",
    "key.keyboard.s",
    "key.keyboard.d",
    "key.keyboard.space",
    "key.keyboard.left.shift",
    "key.keyboard.left.control",
    "key.keyboard.e",
    "key.keyboard.q",
    *(f"key.keyboard.{n}" for n in range(1, 10)),
)
MOUSE_BUTTONS = 3
# Keys pressed (multi-hot), mouse dx and dy, mouse buttons held (multi-hot).
ACTION_SIZE = len(KEYS) + 2 + MOUSE_BUTTONS
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "video": FieldSpec(FieldKind.VIDEO, "uint8", (*SIZE, 3), VideoEncoding(keyframe_interval=16)),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (ACTION_SIZE,), RawEncoding()),
    },
)


def _action(step: dict) -> np.ndarray:
    action = np.zeros(ACTION_SIZE, np.float32)
    pressed = set(step["keyboard"]["keys"])
    action[: len(KEYS)] = [key in pressed for key in KEYS]
    action[len(KEYS)] = step["mouse"]["dx"]
    action[len(KEYS) + 1] = step["mouse"]["dy"]
    for button in step["mouse"]["buttons"]:
        if button < MOUSE_BUTTONS:
            action[len(KEYS) + 2 + button] = 1.0
    return action


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One episode's relative path without extension, e.g. ``data/6.10/Player129-...-171344``."""
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "episode.mp4")
        urllib.request.urlretrieve(options["base_url"] + source + ".mp4", video_path)
        video = load_video(video_path, size=SIZE, step_rate=STEP_RATE)
    with urllib.request.urlopen(options["base_url"] + source + ".jsonl") as response:
        steps = [json.loads(line) for line in response.read().decode().splitlines() if line]
    length = min(len(video), len(steps))
    actions = np.stack([_action(step) for step in steps[:length]])
    yield Episode(id=source, fields={"video": video[:length], "action": actions}, metadata={"source": source})


def vpt_subset() -> ArtifactStep[EpisodeDataset]:
    with urllib.request.urlopen(INDEX_URL) as response:
        relpaths = json.load(response)["relpaths"]
    sources = [path.removesuffix(".mp4") for path in relpaths[:SUBSET_EPISODES]]
    return episodes(
        "examples/vpt-10",
        sources=sources,
        reader="experiments.datasets.episodes.vpt:read",
        reader_options={"base_url": BASE_URL},
        schema=SCHEMA,
        num_shards=SUBSET_EPISODES,
        version="2026.10.11",
    )
