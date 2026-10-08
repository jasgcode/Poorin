# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
import shutil
import tempfile
import urllib.parse
from typing import Optional

import fsspec
import jax
from fsspec import AbstractFileSystem
from haliax.jax_utils import sync_global_devices

logger = logging.getLogger(__name__)

_sync_count = 0


@contextlib.contextmanager
def temp_dir_before_upload(path, process_should_upload: Optional[bool] = None, sync_on_exit: bool = True):
    """
    Creates a temp_dir, yields it, then uploads it to the given path or url on exit using fsspec.
    If it's a local path instead of a url, it just yields the path.

    :param process_should_upload: If None, only upload if jax.process_index() == 0. Otherwise, upload if this is True.
    :param sync_on_exit: Synchronize processes after uploading. Disable only when the caller synchronizes after
        all uploads; worker threads cannot enter a multi-host collective.
    """

    if process_should_upload is None:
        process_should_upload = jax.process_index() == 0

    def _is_url_like(path):
        return urllib.parse.urlparse(path).scheme != ""

    try:
        tmpdir: Optional[str] = None
        if _is_url_like(path):
            tmpdir = tempfile.mkdtemp()
            local_path = tmpdir
        else:
            local_path = path

        yield local_path

        if tmpdir is not None:  # we're uploading to GCS or similar
            if process_should_upload:
                logger.info(f"Copying from temp dir to {path}")
                fs: AbstractFileSystem = fsspec.core.get_fs_token_paths(path, mode="wb")[0]
                fs.put(os.path.join(local_path, "*"), path, recursive=True)
                logger.info(f"Finished copying to {path}")
            elif sync_on_exit:
                logger.info(f"Waiting for process 0 to finish saving checkpoint to {path}")

            if sync_on_exit:
                global _sync_count
                sync_global_devices(f"upload? {path}{_sync_count}")
                _sync_count += 1
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
