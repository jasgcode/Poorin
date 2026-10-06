# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

from tools.changes import changed_files


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_changed_files_include_edits_deletions_both_sides_of_a_rename_and_untracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    for name in ("kept.py", "edited.py", "removed.py", "renamed.py"):
        (tmp_path / name).write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "edited.py").write_text("x = 2\n")
    (tmp_path / "removed.py").unlink()
    _git(tmp_path, "mv", "renamed.py", "moved.py")
    (tmp_path / "new.py").write_text("y = 1\n")

    # A renamed module's old name must count as changed: tests that still import it are affected.
    assert changed_files(tmp_path) == ["edited.py", "moved.py", "new.py", "removed.py", "renamed.py"]
