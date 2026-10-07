# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX's persistent compilation cache, on by default.

Salvaged from Marin's Iris (``iris.jax.compile_cache``) without the cluster parts: the node-local
scratch mount and the object-store mirror of XLA's per-fusion autotune cache.
"""

import logging
import os
from pathlib import Path

import jax

logger = logging.getLogger(__name__)

DEFAULT_COMPILATION_CACHE_DIR = Path.home() / ".cache" / "poorin" / "jax"


def configure_compilation_cache(cache_dir: str | None) -> str:
    """Enable JAX's persistent compilation cache and return its directory.

    ``cache_dir`` (``TrainerConfig.jax_compilation_cache_dir``) wins, then the
    ``JAX_COMPILATION_CACHE_DIR`` environment variable, then ``~/.cache/poorin/jax``.
    """
    resolved = cache_dir or os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(DEFAULT_COMPILATION_CACHE_DIR)
    jax.config.update("jax_compilation_cache_dir", resolved)
    logger.info("JAX compilation cache: %s", resolved)
    return resolved
