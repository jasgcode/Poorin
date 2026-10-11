# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve a coordinator's dashboard on localhost, for a reverse proxy such as ``tailscale serve``.

The server binds ``127.0.0.1`` only, so the dashboard is reachable from other machines solely
through a proxy the owner sets up, and runs uvicorn on a daemon thread of the coordinator's
process.
"""

import errno
import logging
import socket
import threading
from dataclasses import dataclass
from types import TracebackType

import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

DASHBOARD_HOST = "127.0.0.1"
# How long serve_dashboard waits for uvicorn to accept connections.
_STARTUP_TIMEOUT = 10.0


@dataclass
class DashboardServer:
    """A running dashboard server; ``close`` stops it and frees its port."""

    url: str
    _server: uvicorn.Server
    _thread: threading.Thread

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=_STARTUP_TIMEOUT)

    def __enter__(self) -> "DashboardServer":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()


def serve_dashboard(app: ASGIApp, port: int) -> DashboardServer:
    """Serve ``app`` at ``http://127.0.0.1:<port>/`` on a background thread.

    When ``port`` is taken, for example by another pipeline's dashboard, the server binds a free
    port the OS picks instead and logs a warning; the URL it logs is where the dashboard is.
    """
    sock = _bind(port)
    bound_port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True, name=f"zephyr-dashboard-{bound_port}"
    )
    thread.start()
    ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
        lambda: server.started or not thread.is_alive(), timeout=Duration.from_seconds(_STARTUP_TIMEOUT)
    )
    if not server.started:
        server.should_exit = True
        sock.close()
        raise RuntimeError(f"the Zephyr dashboard server did not start on {DASHBOARD_HOST}:{bound_port}")
    url = f"http://{DASHBOARD_HOST}:{bound_port}/"
    if bound_port != port:
        logger.warning("Port %d is in use; the Zephyr dashboard is at %s", port, url)
    else:
        logger.info("Zephyr dashboard: %s", url)
    return DashboardServer(url, server, thread)


def _bind(port: int) -> socket.socket:
    """A socket bound to ``port`` on localhost, or to a free port when ``port`` is taken."""
    try:
        return _bound_socket(port)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
    return _bound_socket(0)


def _bound_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Rebinding a port a just-closed dashboard left in TIME_WAIT; on Linux this still refuses a
    # port another socket is listening on.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((DASHBOARD_HOST, port))
    except OSError:
        sock.close()
        raise
    return sock
