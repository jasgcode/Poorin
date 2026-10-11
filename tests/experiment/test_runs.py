# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Listing runs under the prefix and showing the code behind one (poorin.experiment.runs)."""

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from poorin.execution.lazy import OUT, apply, run
from poorin.experiment.runs import main
from rigging.provenance import Provenance, save_snapshot


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "checkout"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "model.py").write_text("v1\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "Add the model")
    monkeypatch.chdir(path)
    return path


@pytest.fixture
def prefix(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "prefix"
    monkeypatch.setenv("POORIN_PREFIX", str(path))
    return path


def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _fail(out):
    raise RuntimeError("boom")


def _build(name, provenance, version="2026.07.16", **inputs):
    handle = apply(name, _write, version=version, out=OUT, **inputs)
    run(handle, provenance=provenance)
    return handle


def _at(provenance, when):
    return dataclasses.replace(provenance, created_at=when)


def test_list_shows_runs_newest_first_with_their_code(checkout, prefix):
    clean = Provenance.capture()
    _build("data/old", _at(clean, "2026-10-11T09:00:00"))
    _build("data/new", _at(dataclasses.replace(clean, note="second try"), "2026-10-11T10:00:00"))
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("2026-10-11 10:00  built") and "data/new@2026.07.16" in lines[0]
    assert lines[0].endswith("second try")
    assert "data/old@2026.07.16" in lines[1] and lines[1].endswith(f"main@{clean.base_commit[:7]}  Add the model")


def test_list_filters_by_name_and_limits_the_count(checkout, prefix):
    clean = Provenance.capture()
    for i in range(3):
        _build(f"data/toy{i}", _at(clean, f"2026-10-11T09:0{i}:00"))
    _build("models/toy", _at(clean, "2026-10-11T09:05:00"))
    assert len(CliRunner().invoke(main, ["list", "data/"]).output.splitlines()) == 3
    assert len(CliRunner().invoke(main, ["list", "--limit", "2"]).output.splitlines()) == 2


def test_list_shows_a_failed_build_without_code(checkout, prefix):
    with pytest.raises(RuntimeError):
        run(apply("data/broken", _fail, version="2026.07.16", out=OUT), provenance=Provenance.capture())
    result = CliRunner().invoke(main, ["list"])
    assert "failed" in result.output and "data/broken@2026.07.16" in result.output
    assert result.output.rstrip().endswith("-")


def test_show_describes_a_run_from_a_snapshot(checkout, prefix):
    (checkout / "model.py").write_text("v2\n")
    launch = Provenance.capture()
    launch = dataclasses.replace(launch, note="try v2", snapshot_commit=save_snapshot("try v2"))
    _build("data/toy", launch, version="dev")
    _git(checkout, "commit", "-am", "Use v2")  # main moves one commit past the run
    result = CliRunner().invoke(main, ["show", "data/toy@dev"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "data/toy@dev   built" in out
    assert "Note         try v2" in out
    assert f"Commit       {launch.base_commit} on main" in out and "Add the model" in out
    assert "main has moved 1 commit past it" in out
    assert "M   model.py" in out and "+1 -1" in out
    assert f"git diff {launch.base_commit} {launch.snapshot_commit[:7]}" in out
    assert f"git worktree add ../run-{launch.snapshot_commit[:7]} {launch.snapshot_commit[:7]}" in out


def test_show_reads_a_record_from_before_subjects_were_recorded(checkout, prefix):
    old = dataclasses.replace(Provenance.capture(), commit_subject="", commit_time="")
    _build("data/toy", old)
    result = CliRunner().invoke(main, ["show", str(prefix / "data" / "toy" / "2026.07.16")])
    assert result.exit_code == 0, result.output
    assert "Add the model" in result.output  # looked up with git log
    assert "main has not moved since" in result.output


def test_show_rejects_an_unknown_run(checkout, prefix):
    result = CliRunner().invoke(main, ["show", "data/nothing@2026.07.16"])
    assert result.exit_code != 0
    assert "no run at" in result.output
