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


def _labels(output: str) -> list[str]:
    """The ``name@version`` column of ``list``'s lines, top to bottom."""
    return [line.split()[3] for line in output.splitlines()]


def test_list_shows_runs_newest_first_with_their_code(checkout, prefix):
    clean = Provenance.capture()
    # Built d, a, c, b. Newest first is c, a, d, b: not alphabetical, not the build order, not its reverse.
    _build("data/d", _at(clean, "2026-10-11T09:15:00"))
    _build("data/a", _at(clean, "2026-10-11T09:30:00"))
    _build("data/c", _at(dataclasses.replace(clean, note="second try"), "2026-10-11T10:00:00"))
    _build("data/b", _at(clean, "2026-10-11T09:00:00"))
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0, result.output
    assert _labels(result.output) == [f"data/{name}@2026.07.16" for name in "cadb"]
    lines = result.output.splitlines()
    assert lines[0].startswith("2026-10-11 10:00  built") and lines[0].endswith("second try")
    assert lines[3].startswith("2026-10-11 09:00  built")
    assert lines[3].endswith(f"main@{clean.base_commit[:7]}  Add the model")


def test_list_filters_by_name(checkout, prefix):
    clean = Provenance.capture()
    for i in range(3):
        _build(f"data/toy{i}", _at(clean, f"2026-10-11T09:0{i}:00"))
    _build("models/toy", _at(clean, "2026-10-11T09:05:00"))
    result = CliRunner().invoke(main, ["list", "data/"])
    assert sorted(_labels(result.output)) == [f"data/toy{i}@2026.07.16" for i in range(3)]


def test_list_limit_keeps_the_newest_runs(checkout, prefix):
    clean = Provenance.capture()
    _build("data/toy0", _at(clean, "2026-10-11T09:00:00"))
    _build("data/toy1", _at(clean, "2026-10-11T09:02:00"))
    _build("data/toy2", _at(clean, "2026-10-11T09:01:00"))
    _build("models/toy", _at(clean, "2026-10-11T09:05:00"))
    result = CliRunner().invoke(main, ["list", "--limit", "2"])
    assert _labels(result.output) == ["models/toy@2026.07.16", "data/toy1@2026.07.16"]


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
