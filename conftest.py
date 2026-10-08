# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session setup for the repository-root test suite (``tests/`` and ``experiments/``).

The root suite was written against Marin's multi-region cluster layout. Poorin's default
cluster, ``config/local.yaml``, declares no buckets, so the suite runs against Marin's configs,
kept as fixtures in ``config/testing/``. The setting starts at configuration time because some
experiment modules resolve temp-bucket paths when they are imported, before any fixture runs.
"""

import pytest
import rigging.filesystem.cluster_config as cluster_config

MARIN_FIXTURE_CLUSTERS = "config/testing"
_marin_layout = pytest.MonkeyPatch()


def pytest_configure(config: pytest.Config) -> None:
    _marin_layout.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (MARIN_FIXTURE_CLUSTERS,))
    _marin_layout.setenv("MARIN_CLUSTER", "marin")
    cluster_config.reset_data_config_cache()


def pytest_unconfigure(config: pytest.Config) -> None:
    _marin_layout.undo()
    cluster_config.reset_data_config_cache()
