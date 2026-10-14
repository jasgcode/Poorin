# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.episode_models.data import compute_action_stats
from experiments.episode_models.tests.tiny_dataset import build_tiny


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> tuple[str, str]:
    """The tiny dataset's path and its action stats' path, built once for the module."""
    root = tmp_path_factory.mktemp("data")
    dataset = build_tiny(root)
    compute_action_stats(dataset.path, "action", 1.0, 99.0, str(root / "stats"))
    return dataset.path, str(root / "stats")
