# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Make NVRTC's builtins library loadable for cuDNN on CUDA 13.

cuDNN compiles attention kernels at run time with NVRTC 13, which loads
``libnvrtc-builtins.so.13.0`` by soname. JAX's CUDA 13 wheels install it under
``site-packages/nvidia/cu13/lib``, which is not on the loader's search path, so without help cuDNN
fails with "No valid execution plans built". Loading the library once with ``RTLD_GLOBAL`` makes
the soname resolvable for every later ``dlopen``.
"""

import ctypes
import logging
import sysconfig
from pathlib import Path

import jax

logger = logging.getLogger(__name__)

NVRTC_BUILTINS = "libnvrtc-builtins.so.13.0"


def nvrtc_builtins_path() -> Path:
    """Where JAX's CUDA 13 wheels install NVRTC's builtins library."""
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib" / NVRTC_BUILTINS


def preload_nvrtc_builtins() -> None:
    """Load NVRTC's builtins globally when JAX runs on a GPU; do nothing on CPU or TPU."""
    if jax.default_backend() != "gpu":
        return
    path = nvrtc_builtins_path()
    if not path.is_file():
        raise FileNotFoundError(f"cuDNN on CUDA 13 needs {path}; install JAX with its CUDA 13 wheels")
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    logger.info("Preloaded %s", path)
