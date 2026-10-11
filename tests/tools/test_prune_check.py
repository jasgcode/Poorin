# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.prune_check."""

import csv
import os
import subprocess
from dataclasses import replace

import pytest
from finelog.embedded import require_embedded_server
from levanter.tracker.finelog_metrics import LevanterMetricsWriter
from poorin.execution.lazy import OUT, apply, run
from rigging.provenance import SNAPSHOT_REF_PREFIX, Provenance, save_snapshot

from tools.prune_check import (
    METRIC_COLUMNS,
    ImportProfile,
    append_metrics_row,
    levanter_metric_rows,
    parse_importtime,
    remove_new_entries,
    remove_run_outputs,
    remove_run_snapshots,
    snapshot_entries,
    snapshot_refs,
    train_roots,
)
from tools.reachability import REPO_ROOT, RunnableRoot

IMPORTTIME_STDERR = """\
import time: self [us] | cumulative | imported package
import time:       100 |        100 |   _io
import time:        50 |        150 | encodings
I20261005 09:00:00 140000 some.logger Message that is not an import record
import time:        10 |         10 |     child
import time:        20 |         30 |   parent
import time:         5 |        200 | toplevel
"""


def test_parse_importtime_counts_modules_and_sums_top_level_time_ignoring_log_lines():
    # Top-level records carry one leading space; nested ones carry two more per level.
    assert parse_importtime(IMPORTTIME_STDERR) == ImportProfile(modules=5, seconds=pytest.approx(350e-6))


def _row(step: str) -> dict[str, object]:
    row: dict[str, object] = {column: 0 for column in METRIC_COLUMNS}
    row.update(phase="P0", step=step)
    return row


def test_append_metrics_row_writes_the_header_once(tmp_path):
    path = tmp_path / "docs" / "metrics.csv"
    append_metrics_row(path, _row("first"))
    append_metrics_row(path, _row("second"))
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [row["step"] for row in rows] == ["first", "second"]
    assert tuple(rows[0]) == METRIC_COLUMNS


def test_remove_run_outputs_deletes_user_runs_and_keeps_shared_caches(tmp_path):
    (tmp_path / "users" / "someone" / "checkpoints" / "run").mkdir(parents=True)
    (tmp_path / "wikitext" / "2026.06.28").mkdir(parents=True)
    (tmp_path / "tmp" / "levanter_kernel_autotune").mkdir(parents=True)
    remove_run_outputs(tmp_path)
    assert not (tmp_path / "users").exists()
    assert (tmp_path / "wikitext" / "2026.06.28").is_dir()
    assert (tmp_path / "tmp" / "levanter_kernel_autotune").is_dir()


def test_remove_new_entries_deletes_only_what_appeared_after_the_snapshot(tmp_path):
    logs, wandb = tmp_path / "logs", tmp_path / "wandb"
    (logs / "old-run").mkdir(parents=True)
    before = snapshot_entries([logs, wandb])
    (logs / "new-run" / "artifacts").mkdir(parents=True)
    (wandb / "offline-run-1").mkdir(parents=True)
    (wandb / "debug.log").write_text("x")
    remove_new_entries(before)
    assert [p.name for p in logs.iterdir()] == ["old-run"]
    assert list(wandb.iterdir()) == []


def test_train_roots_removes_run_outputs_when_a_run_fails(tmp_path):
    prefix = tmp_path / "prefix"
    (prefix / "users" / "someone" / "grug").mkdir(parents=True)
    failing = RunnableRoot("R", "tools.no_such_module", ())
    with pytest.raises(subprocess.CalledProcessError):
        train_roots([failing], dict(os.environ), prefix=prefix, log_dirs=[], repo=REPO_ROOT)
    assert not (prefix / "users").exists()


def _git(repo, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _make_dir(out):
    os.makedirs(out, exist_ok=True)


def test_remove_run_snapshots_deletes_only_the_refs_the_checks_runs_created(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-b", "main"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        _git(repo, *args)
    (repo / "f.txt").write_text("v1\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "init")
    (repo / "f.txt").write_text("v2\n")
    older = save_snapshot("existed before the check", repo)
    before = snapshot_refs(repo)
    check_run = save_snapshot("the check's own run", repo)
    kept_input = save_snapshot("a dataset the check keeps", repo)
    elsewhere = save_snapshot("another launch during the check", repo)
    prefix = tmp_path / "prefix"
    monkeypatch.setenv("POORIN_PREFIX", str(prefix))
    launch = Provenance.capture(repo)
    for name, snapshot in (
        ("users/t/checkpoints/toy", check_run),
        ("users/t/checkpoints/old", older),
        ("data/toy", kept_input),
    ):
        run(apply(name, _make_dir, version="2026.07.16", out=OUT), provenance=replace(launch, snapshot_commit=snapshot))

    remove_run_snapshots(prefix, before, repo)

    left = {ref.removeprefix(SNAPSHOT_REF_PREFIX) for ref in snapshot_refs(repo)}
    # The run output the check removes loses its ref. A ref that existed before the check, one a
    # kept output names, and one no output of the check names all stay.
    assert left == {older, kept_input, elsewhere}


def test_levanter_metric_rows_counts_the_rows_a_levanter_run_wrote(tmp_path):
    with require_embedded_server()(log_dir=str(tmp_path)) as server:
        assert levanter_metric_rows(server.address) == 0
        writer = LevanterMetricsWriter.connect(server.address, "grug-base-test", 0)
        writer.scalar("train/loss", 1.0, step=1)
        writer.scalar("train/loss", 0.5, step=2)
        writer.close()
        assert levanter_metric_rows(server.address) == 2
