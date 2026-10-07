# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.nvrtc import nvrtc_builtins_path


def test_finds_the_builtins_library_of_the_installed_nvrtc_version(tmp_path):
    (tmp_path / "libnvrtc.so.13").touch()
    (tmp_path / "libnvrtc-builtins.so.13.4").touch()

    assert nvrtc_builtins_path(tmp_path) == tmp_path / "libnvrtc-builtins.so.13.4"


def test_a_missing_builtins_library_is_an_error(tmp_path):
    (tmp_path / "libnvrtc.so.13").touch()

    with pytest.raises(FileNotFoundError, match="libnvrtc-builtins"):
        nvrtc_builtins_path(tmp_path)
