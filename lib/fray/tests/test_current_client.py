# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for current_client() and set_current_client()."""

import pytest
from fray.current_client import current_client, set_current_client
from fray.local_backend import LocalClient


def test_default_returns_local_client():
    client = current_client()
    assert isinstance(client, LocalClient)


def test_set_current_client_context_manager():
    explicit = LocalClient(max_threads=2)
    with set_current_client(explicit) as c:
        assert c is explicit
        assert current_client() is explicit
    assert current_client() is not explicit


def test_set_current_client_restores_on_exception():
    explicit = LocalClient(max_threads=2)
    with pytest.raises(RuntimeError):
        with set_current_client(explicit):
            raise RuntimeError("boom")
    assert current_client() is not explicit
