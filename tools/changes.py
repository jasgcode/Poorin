# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The uncommitted change that Poorin's tools check."""

import subprocess
from pathlib import Path


def changed_files(repo_root: Path, base: str = "HEAD") -> list[str]:
    """Tracked paths that differ from where ``HEAD`` left ``base``, plus untracked files:
    repository-relative, sorted.

    With the default ``base``, that is the uncommitted change. With a branch such as ``main``, it
    adds the commits made since the current branch left it, but not ones made on ``main`` after.
    Deleted paths are included, and a rename counts as a deletion plus an addition. Callers that
    read files skip the paths that no longer exist.
    """
    [fork_point] = _git(repo_root, "merge-base", base, "HEAD")
    tracked = _git(repo_root, "diff", "--name-only", "--no-renames", fork_point)
    untracked = _git(repo_root, "ls-files", "--others", "--exclude-standard")
    return sorted({*tracked, *untracked})


def staged_files(repo_root: Path) -> list[str]:
    """Paths whose staged content differs from ``HEAD``: what the next commit holds, sorted.

    Unstaged edits and untracked files are left out. Deletions and renames count as in
    :func:`changed_files`.
    """
    return sorted(_git(repo_root, "diff", "--cached", "--name-only", "--no-renames", "HEAD"))


def tracked_files(repo_root: Path) -> list[str]:
    """Every file Git tracks, sorted."""
    return sorted(_git(repo_root, "ls-files"))


def _git(repo_root: Path, *args: str) -> list[str]:
    out = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]
