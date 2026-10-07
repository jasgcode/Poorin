# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

from tools.changes import changed_files, staged_files


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo_with(tmp_path, *names):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    for name in names:
        (tmp_path / name).write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")


def test_changed_files_include_edits_deletions_both_sides_of_a_rename_and_untracked_files(tmp_path):
    _repo_with(tmp_path, "kept.py", "edited.py", "removed.py", "renamed.py")
    (tmp_path / "edited.py").write_text("x = 2\n")
    (tmp_path / "removed.py").unlink()
    _git(tmp_path, "mv", "renamed.py", "moved.py")
    (tmp_path / "new.py").write_text("y = 1\n")

    # A renamed module's old name must count as changed: tests that still import it are affected.
    assert changed_files(tmp_path) == ["edited.py", "moved.py", "new.py", "removed.py", "renamed.py"]


def test_staged_files_leave_out_unstaged_edits_and_untracked_files(tmp_path):
    # The Git hook lints only what the commit holds, so scratch files are never rewritten.
    _repo_with(tmp_path, "staged.py", "unstaged.py", "renamed.py")
    (tmp_path / "staged.py").write_text("x = 2\n")
    _git(tmp_path, "add", "staged.py")
    _git(tmp_path, "mv", "renamed.py", "moved.py")
    (tmp_path / "unstaged.py").write_text("x = 2\n")
    (tmp_path / "scratch.ipynb").write_text("{}\n")

    assert staged_files(tmp_path) == ["moved.py", "renamed.py", "staged.py"]
