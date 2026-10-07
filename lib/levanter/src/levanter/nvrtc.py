# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Make NVRTC's builtins library loadable for cuDNN on CUDA 13.

cuDNN compiles attention kernels at run time with NVRTC 13, which loads its builtins library,
``libnvrtc-builtins.so.13.<minor>``, by a soname that carries NVRTC's own version. JAX's CUDA 13
wheels install it under ``site-packages/nvidia/cu13/lib``, which is not on the loader's search
path, so without help cuDNN fails with "No valid execution plans built". Loading the library once
with ``RTLD_GLOBAL`` makes the soname resolvable for every later ``dlopen``.
"""

import ctypes
import logging
import sysconfig
from pathlib import Path

import jax

logger = logging.getLogger(__name__)

NVRTC_BUILTINS_PREFIX = "libnvrtc-builtins.so.13."


def cuda_lib_dir() -> Path:
    """Where JAX's CUDA 13 wheels install their shared libraries."""
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"


def nvrtc_builtins_path(lib_dir: Path) -> Path:
    """The one NVRTC builtins library in ``lib_dir``, whatever NVRTC 13 minor version installed it."""
    found = sorted(lib_dir.glob(f"{NVRTC_BUILTINS_PREFIX}*"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"cuDNN on CUDA 13 needs exactly one {NVRTC_BUILTINS_PREFIX}* in {lib_dir}, found "
            f"{[path.name for path in found]}; install JAX with its CUDA 13 wheels"
        )
    return found[0]


def preload_nvrtc_builtins() -> None:
    """Load NVRTC's builtins globally when JAX runs on a GPU; do nothing on CPU or TPU."""
    if jax.default_backend() != "gpu":
        return
    path = nvrtc_builtins_path(cuda_lib_dir())
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    logger.info("Preloaded %s", path)
