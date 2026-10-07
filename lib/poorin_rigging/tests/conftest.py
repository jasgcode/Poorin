# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import poorin_rigging.filesystem.cluster_config as cluster_config
import pytest
from poorin_rigging.config_discovery import find_project_root


@pytest.fixture(autouse=True)
def _hermetic_cluster_config(monkeypatch):
    """Resolve clusters from Marin's multi-region configs, kept as fixtures in ``config/testing``.

    Rigging's tests exercise region routing, temp buckets, and cross-region guards, which need
    Marin's buckets; Poorin's default ``config/local.yaml`` declares none. Replacing the search
    dirs also keeps a per-user ``~/.config/marin/clusters`` override from changing results.
    """
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", ("config/testing",))
    monkeypatch.setenv("MARIN_CLUSTER", "marin")
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
    yield
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
