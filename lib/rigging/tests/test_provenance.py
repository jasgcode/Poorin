# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import subprocess
from pathlib import Path

import pytest
from rigging.provenance import Provenance, current_launch, launch_provenance, launched_as, username_segment


def _clean(**fields) -> Provenance:
    base = dict(tree_hash="aaaa", base_commit="a0a406ff4", dirty=False, branch="main", built_by="power")
    return Provenance(**{**base, **fields})


def test_a_clean_label_names_the_branch_commit_and_subject():
    assert _clean(commit_subject="Trim the devices").label == "main@a0a406f  Trim the devices"


def test_a_note_replaces_the_subject():
    assert _clean(commit_subject="Trim the devices", note="try rmsnorm").label == "main@a0a406f  try rmsnorm"


def test_a_dirty_label_names_up_to_three_files_and_the_snapshot():
    p = _clean(
        dirty=True,
        uncommitted=(" M experiments/llama.py", "?? experiments/grug/moe/model.py", " M docs/a.md", " M docs/b.md"),
        snapshot_commit="5c1d2e9aa0",
    )
    assert p.label == "main@a0a406f + 4 uncommitted: llama.py, model.py, a.md, …  (snapshot 5c1d2e9)"


def test_a_dirty_label_with_a_note_counts_the_files():
    p = _clean(
        dirty=True, uncommitted=(" M a.py", " M b.py", " M c.py"), note="try rmsnorm", snapshot_commit="5c1d2e9aa0"
    )
    assert p.label == "main@a0a406f + 3 uncommitted  try rmsnorm  (snapshot 5c1d2e9)"


def test_a_label_stays_on_one_line_and_cuts_long_text():
    p = _clean(note="first line\nsecond line " + "x" * 80)
    assert "\n" not in p.label
    text = p.label.removeprefix("main@a0a406f  ")
    assert len(text) == 60 and text.endswith("…") and text.startswith("first line second line")


def test_a_detached_head_and_a_missing_checkout_have_labels():
    assert _clean(branch=None, commit_subject="s").label == "detached@a0a406f  s"
    assert _clean(base_commit="").label == "no git checkout"


def test_a_record_from_before_labels_still_loads_and_labels():
    old = '{"tree_hash": "aaaa", "base_commit": "a0a406ff4", "dirty": true, "branch": "main", "built_by": "power"}'
    p = Provenance.from_json(old)
    assert (p.commit_subject, p.uncommitted, p.snapshot_commit, p.note) == ("", (), None, None)
    assert p.label == "main@a0a406f + uncommitted changes"


def test_str_is_the_label():
    p = _clean(commit_subject="s")
    assert str(p) == p.label


def test_json_round_trip():
    p = Provenance(tree_hash="aaaa", base_commit="bbbb", dirty=True, branch=None, built_by="power")
    assert Provenance.from_json(p.to_json()) == p


def test_username_segment_sanitizes_email_to_a_clean_distinct_segment(monkeypatch):
    # Drops the domain but keeps the whole local name, so the segment is path-safe (no '@'/'.')
    # and two users who share a first name don't collapse onto one namespace.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: "Russell.Power@gmail.com")
    assert username_segment() == "russell-power"


def test_username_segment_raises_when_unresolvable(monkeypatch):
    # Fail-fast is the contract: a username that does not resolve must not silently namespace
    # artifacts under a shared bucket.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: None)
    with pytest.raises(RuntimeError):
        username_segment()


def test_username_segment_warns_when_resolving_to_machine_login(monkeypatch, caplog):
    # A machine login owning a per-user namespace means a shared or service account launched
    # the run; the segment is still returned, but the warning says so.
    monkeypatch.setattr("rigging.provenance._getuser", lambda: "root")
    with caplog.at_level(logging.WARNING, logger="rigging.provenance"):
        assert username_segment() == "root"
    assert [r.levelname for r in caplog.records] == ["WARNING"]


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path: Path) -> Path:
    _run(["git", "init", "-b", "main"], tmp_path)
    _run(["git", "config", "user.email", "t@example.com"], tmp_path)
    _run(["git", "config", "user.name", "tester"], tmp_path)
    (tmp_path / "f.txt").write_text("hello\n")
    _run(["git", "add", "f.txt"], tmp_path)
    _run(["git", "commit", "-m", "init"], tmp_path)
    return tmp_path


def test_from_git_clean(tmp_path):
    repo = _init_repo(tmp_path)
    p = Provenance.from_git(repo)
    assert p.dirty is False
    assert p.branch == "main"
    assert p.tree_hash and p.base_commit
    # A clean tree's hash is HEAD's tree; dedup_key is that tree hash.
    head_tree = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD^{tree}"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert p.dedup_key == head_tree


def test_from_git_dirty_changes_tree_not_base(tmp_path):
    repo = _init_repo(tmp_path)
    clean = Provenance.from_git(repo)
    (repo / "f.txt").write_text("changed\n")
    dirty = Provenance.from_git(repo)
    assert dirty.dirty is True
    assert dirty.base_commit == clean.base_commit
    assert dirty.tree_hash != clean.tree_hash


def test_git_provenance_untracked_file_is_dirty(tmp_path):
    repo = _init_repo(tmp_path)
    clean = Provenance.from_git(repo)
    (repo / "untracked.txt").write_text("new\n")
    strict = Provenance.from_git(repo)
    captured = Provenance.capture(repo)
    assert strict.dirty is True
    assert captured.dirty is True
    assert strict.tree_hash == clean.tree_hash
    assert captured.tree_hash == clean.tree_hash


def test_capture_is_best_effort_outside_a_repo(tmp_path):
    # from_git is strict (the tree hash is the point); capture degrades instead of raising,
    # still recording the launch context that has nothing to do with git.
    with pytest.raises(RuntimeError):
        Provenance.from_git(tmp_path)
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_capture_tolerates_missing_git_binary(tmp_path, monkeypatch):
    # A bundle image may lack git entirely: subprocess raises FileNotFoundError instead of
    # returning a nonzero exit. capture degrades like being outside a checkout.
    monkeypatch.setenv("PATH", "")
    p = Provenance.capture(tmp_path)
    assert p.tree_hash == ""
    assert p.base_commit == ""
    assert p.dirty is False
    assert p.command_line  # argv, captured regardless of git
    assert p.created_at  # timestamp, captured regardless of git


def test_json_round_trip_preserves_run_fields():
    p = Provenance(
        tree_hash="aaaa",
        base_commit="bbbb",
        dirty=False,
        branch="main",
        built_by="power",
        git_remote="git@github.com:o/r.git",
        created_at="2026-06-30T00:00:00",
        command_line=("python", "-m", "experiments.foo"),
    )
    assert Provenance.from_json(p.to_json()) == p


def test_capture_records_the_subject_time_and_uncommitted_paths(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "f.txt").write_text("changed\n")
    (repo / "new file é.txt").write_text("new\n")
    _run(["git", "mv", "f.txt", "g.txt"], repo)
    p = Provenance.capture(repo)
    assert p.commit_subject == "init"
    assert p.commit_time  # HEAD's committer date, ISO 8601
    paths = sorted(entry[3:] for entry in p.uncommitted)
    assert paths == ["g.txt", "new file é.txt"]  # the rename once, under its new path; no quoting
    assert p.dirty is True


def test_capture_of_a_clean_checkout_has_no_uncommitted_paths(tmp_path):
    p = Provenance.capture(_init_repo(tmp_path))
    assert (p.dirty, p.uncommitted) == (False, ())


def test_json_round_trip_preserves_the_code_fields():
    p = _clean(
        dirty=True,
        commit_subject="s",
        commit_time="2026-10-11T09:50:00-05:00",
        uncommitted=(" M a.py",),
        snapshot_commit="5c1d2e9aa0",
        note="n",
    )
    assert Provenance.from_json(p.to_json()) == p


def test_launched_as_installs_a_launch_until_the_block_exits():
    launch = _clean(note="installed")
    assert current_launch() is None
    with launched_as(launch):
        assert current_launch() is launch
        assert launch_provenance() is launch
    assert current_launch() is None
