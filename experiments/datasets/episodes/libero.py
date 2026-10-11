# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LIBERO-Spatial demonstrations as episodes: two cameras, joint and gripper state, actions, and the
instruction as metadata."""

from collections.abc import Iterator, Mapping
from typing import Any

import h5py
import numpy as np
from huggingface_hub import hf_hub_download
from poorin.episodes.build import EpisodeDataset
from poorin.episodes.schema import DatasetSchema, Episode, FieldKind, FieldSpec, RawEncoding, VideoEncoding
from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import episodes

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SUITE = "libero_spatial"
TASKS = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
)
STEP_RATE = 20.0  # LIBERO's control rate
CAMERA = FieldSpec(FieldKind.VIDEO, "uint8", (128, 128, 3), VideoEncoding(keyframe_interval=16))
SCHEMA = DatasetSchema(
    step_rate=STEP_RATE,
    fields={
        "agentview": CAMERA,
        "wrist": CAMERA,
        "joint_state": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding()),
        "gripper_state": FieldSpec(FieldKind.ARRAY, "float32", (2,), RawEncoding()),
        "action": FieldSpec(FieldKind.ARRAY, "float32", (7,), RawEncoding()),
    },
)


def read(source: str, options: Mapping[str, Any]) -> Iterator[Episode]:
    """One task's HDF5 file: every demonstration in it."""
    path = hf_hub_download(REPO_ID, f"{SUITE}/{source}_demo.hdf5", repo_type="dataset", revision=options["revision"])
    instruction = source.replace("_", " ")
    with h5py.File(path, "r") as f:
        for demo in sorted(f["data"], key=lambda name: int(name.split("_")[1])):
            group = f["data"][demo]
            obs = group["obs"]
            yield Episode(
                id=f"{source}/{demo}",
                fields={
                    # LIBERO renders upside down; rotate 180 degrees. h5py slices only with positive
                    # steps, so read the frames before flipping them.
                    "agentview": np.ascontiguousarray(obs["agentview_rgb"][:][:, ::-1, ::-1]),
                    "wrist": np.ascontiguousarray(obs["eye_in_hand_rgb"][:][:, ::-1, ::-1]),
                    "joint_state": obs["joint_states"][:].astype(np.float32),
                    "gripper_state": obs["gripper_states"][:].astype(np.float32),
                    "action": group["actions"][:].astype(np.float32),
                },
                metadata={"instruction": instruction, "suite": SUITE},
            )


def libero_spatial() -> ArtifactStep[EpisodeDataset]:
    return episodes(
        "examples/libero-spatial",
        sources=list(TASKS),
        reader="experiments.datasets.episodes.libero:read",
        reader_options={"revision": REVISION},
        schema=SCHEMA,
        num_shards=len(TASKS),
        version="2026.10.11",
    )
