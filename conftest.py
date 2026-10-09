# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session setup for the repository-root test suite (``tests/`` and ``experiments/``).

The root suite was written against Marin's multi-region cluster layout. Poorin's default
cluster, ``config/local.yaml``, declares no buckets, so the suite runs against Marin's configs,
kept as fixtures in ``config/testing/``. The setting starts at configuration time because some
experiment modules resolve temp-bucket paths when they are imported, before any fixture runs.
"""

from pathlib import Path

import pytest
import rigging.filesystem.cluster_config as cluster_config
from jaxtyping import install_import_hook

# Check grug's jaxtyping annotations at run time: every function and dataclass in these packages is
# wrapped with beartype as it is imported, so a wrong shape or dtype fails the test that hit it. The
# hook must be installed before any test imports the packages.
install_import_hook(["experiments.grug", "levanter.grug"], "beartype.beartype")

# Absolute, because a relative search dir resolves from the working directory, which a test may change.
POORIN_FIXTURE_CLUSTERS = str(Path(__file__).parent / "config" / "testing")
_poorin_layout = pytest.MonkeyPatch()


def pytest_configure(config: pytest.Config) -> None:
    _poorin_layout.setattr(cluster_config, "POORIN_CLUSTER_CONFIG_DIRS", (POORIN_FIXTURE_CLUSTERS,))
    _poorin_layout.setenv("POORIN_CLUSTER", "marin")
    cluster_config.reset_data_config_cache()


def pytest_unconfigure(config: pytest.Config) -> None:
    _poorin_layout.undo()
    cluster_config.reset_data_config_cache()
