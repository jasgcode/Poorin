# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serving the coordinator's dashboard on localhost."""

import json
import socket
import urllib.error
import urllib.request

import pytest
from fray.types import ResourceConfig
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from zephyr.context import ZephyrContext
from zephyr.dashboard.server import serve_dashboard
from zephyr.dataset import Dataset


def _app() -> Starlette:
    def ok(_request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Starlette(routes=[Route("/", ok)])


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode()


def test_the_dashboard_is_served_on_localhost_until_closed():
    port = _free_port()
    with serve_dashboard(_app(), port) as server:
        assert server.url == f"http://127.0.0.1:{port}/"
        assert _get(server.url) == "ok"
    with pytest.raises(urllib.error.URLError):
        _get(server.url)


def test_a_taken_port_moves_the_dashboard_to_a_free_one():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with serve_dashboard(_app(), port) as server:
            assert server.url != f"http://127.0.0.1:{port}/"
            assert _get(server.url) == "ok"


def test_a_context_serves_its_pipelines_dashboard_while_it_is_open(local_client, tmp_path):
    port = _free_port()
    ctx = ZephyrContext(
        client=local_client,
        max_workers=1,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="dashboard-test",
        dashboard_port=port,
    )
    with ctx:
        assert sorted(ctx.execute(Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)).results) == [2, 4, 6]
        # The coordinator's own JSON API and page; a finished pipeline leaves the pipeline list.
        assert "pipelines" in json.loads(_get(f"http://127.0.0.1:{port}/api/pipelines"))
        assert "<html" in _get(f"http://127.0.0.1:{port}/").lower()
    with pytest.raises(urllib.error.URLError):
        _get(f"http://127.0.0.1:{port}/")
