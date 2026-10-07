# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve the active Fray client."""

import contextlib
import contextvars
import logging
from collections.abc import Generator

from poorin_fray.client import Client
from poorin_fray.local_backend import LocalClient

logger = logging.getLogger(__name__)

_current_client_var: contextvars.ContextVar[Client | None] = contextvars.ContextVar("_current_client_var", default=None)


def current_client() -> Client:
    """Return the client set with :func:`set_current_client`, otherwise a ``LocalClient``."""
    client = _current_client_var.get()
    if client is not None:
        logger.info("current_client: using explicitly set client")
        return client
    logger.info("current_client: using LocalClient")
    return LocalClient()


@contextlib.contextmanager
def set_current_client(client: Client) -> Generator[Client, None, None]:
    """Context manager that sets the current client and restores on exit."""
    token = _current_client_var.set(client)
    try:
        yield client
    finally:
        _current_client_var.reset(token)
