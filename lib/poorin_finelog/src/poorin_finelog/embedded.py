# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process finelog server, backed by the native Rust extension.

``EmbeddedServer`` boots the same axum app the ``finelog-server`` binary serves,
on an owned tokio runtime bound to a local port. Callers then talk to it over
the normal RPC contract (:class:`poorin_finelog.client.LogClient` / proxies), so there
is exactly one server implementation.

The extension ships as the top-level ``finelog_server`` module in the separate
``marin-finelog-server`` wheel. The pure ``poorin-finelog`` client does not
depend on it; consumers that need the in-process server depend on it
explicitly. Its Rust source is not in this tree: it lives under
``lib/poorin_finelog/rust`` at git tag ``upstream-baseline``. If the extension is
missing, ``EmbeddedServer`` is ``None`` and :func:`require_embedded_server`
raises an actionable error; :func:`is_available` lets callers probe without
importing.
"""

try:
    from finelog_server import EmbeddedServer
except ImportError:
    EmbeddedServer = None  # type: ignore[assignment,misc]

_INSTALL_MSG = (
    "finelog native server extension (finelog_server) is not available. It ships "
    "in the separate marin-finelog-server wheel; install it (e.g. 'pip install "
    "marin-finelog-server'), or build it from the Rust source under "
    "lib/poorin_finelog/rust at git tag upstream-baseline."
)


def is_available() -> bool:
    """Whether the native in-process server extension is importable."""
    return EmbeddedServer is not None


def require_embedded_server() -> type:
    """Return :class:`EmbeddedServer`, raising a clear error if it is missing."""
    if EmbeddedServer is None:
        raise ImportError(_INSTALL_MSG)
    return EmbeddedServer
