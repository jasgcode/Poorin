# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Where launched jobs' extra environment variables come from."""

from pathlib import Path

import pytest
import yaml
from draccus.utils import DecodingError

from levanter.infra.cli_helpers import load_config


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(work)
    return home


def test_the_working_directory_file_overlays_the_users_and_the_accelerator_variant_wins(home):
    _write(
        home / ".config/poorin/config.yaml",
        {"env": {"A": "user", "B": "user"}, "accel_env": {"v5litepod": {"C": "family"}}},
    )
    _write(Path(".levanter.yaml"), {"env": {"B": "local"}, "accel_env": {"v5litepod-8": {"C": "variant"}}})

    assert load_config().env_for_accel("v5litepod-8") == {"A": "user", "B": "local", "C": "variant"}


def test_a_config_with_a_removed_cluster_field_fails_to_load(home):
    # Marin-era files set TPU and Docker fields that CliConfig no longer has; they fail loudly
    # rather than being ignored.
    _write(home / ".config/poorin/config.yaml", {"tpu": "v5litepod-8"})

    with pytest.raises(DecodingError, match="tpu"):
        load_config()
