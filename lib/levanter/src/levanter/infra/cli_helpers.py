# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import yaml


USER_CONFIG_PATH = Path(".config/poorin/config.yaml")
LOCAL_CONFIG_PATH = Path(".levanter.yaml")


@dataclass(frozen=True)
class CliConfig:
    """Extra environment variables for launched training jobs, read by :func:`load_config`."""

    env: dict[str, str] = dataclasses.field(default_factory=dict)

    accel_env: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    """
    Environment variables specific to a type of accelerator. The keys are the accelerator type (e.g. v5litepod-256) or
    generation (e.g. v5litepod), with priority given to the more specific key. The values are dictionaries of environment
    variables to set. These take priority over the general `env` field.
    """

    def env_for_accel(self, accel_type: str) -> dict[str, str]:
        base_env = self.env.copy()

        if "-" in accel_type:
            base_env.update(self.accel_env.get(accel_type.split("-")[0], {}))

        if accel_type in self.accel_env:
            base_env.update(self.accel_env[accel_type])

        return base_env


def load_config() -> CliConfig:
    """The user's ``~/.config/poorin/config.yaml``, overlaid by ``.levanter.yaml`` in the working directory.

    Either file may be missing. A key :class:`CliConfig` does not define is an error.
    """
    config_dict = _load_yaml_mapping(Path.home() / USER_CONFIG_PATH)

    if LOCAL_CONFIG_PATH.exists():
        config_dict = _merge_config_dicts(config_dict, _load_yaml_mapping(LOCAL_CONFIG_PATH))

    return draccus.decode(CliConfig, config_dict)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return data


def _merge_config_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_git_commit():
    """Get the current git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def default_run_id():
    """Generate a run ID for wandb and continuation.

    Wandb expects a base36 encoded ID of exactly 8 lowercase characters
    or it won't generate a display name."""
    rng_bytes = os.urandom(16)
    run_id = base64.b32encode(rng_bytes)[:8].lower()
    run_id = run_id.decode("utf-8")
    assert len(run_id) == 8
    for char in run_id:
        assert char in "abcdefghijklmnopqrstuvwxyz0123456789"
    return run_id


def find_repo_root(start: Path | None = None) -> Path:
    """Return the nearest parent containing a .git directory."""
    here = (start or Path(__file__)).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists():
            return p
    return Path.cwd()
