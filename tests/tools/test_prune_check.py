# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the pure parts of tools.prune_check."""

import csv
import subprocess

import pytest

from tools.prune_check import (
    METRIC_COLUMNS,
    ImportProfile,
    PytestInvocation,
    append_metrics_row,
    changed_files,
    parse_importtime,
    pytest_invocations,
    remove_run_outputs,
)

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


def test_append_metrics_row_rejects_a_row_with_missing_columns(tmp_path):
    with pytest.raises(ValueError):
        append_metrics_row(tmp_path / "metrics.csv", {"phase": "P0"})


def test_remove_run_outputs_deletes_user_runs_and_keeps_shared_caches(tmp_path):
    (tmp_path / "users" / "someone" / "checkpoints" / "run").mkdir(parents=True)
    (tmp_path / "wikitext" / "2026.06.28").mkdir(parents=True)
    (tmp_path / "tmp" / "levanter_kernel_autotune").mkdir(parents=True)
    remove_run_outputs(tmp_path)
    assert not (tmp_path / "users").exists()
    assert (tmp_path / "wikitext" / "2026.06.28").is_dir()
    assert (tmp_path / "tmp" / "levanter_kernel_autotune").is_dir()


def test_pytest_invocations_run_each_package_separately_and_give_haliax_eight_cpu_devices():
    # One invocation per package: packages' tests/conftest.py modules collide in a single run.
    legs = [
        ("marin-levanter", "lib/levanter/tests/a.py"),
        ("marin-haliax", "lib/haliax/tests/h.py"),
        ("marin-levanter", "lib/levanter/tests/b.py lib/levanter/tests/a.py"),
    ]
    assert pytest_invocations(legs) == [
        PytestInvocation(paths=("lib/haliax/tests/h.py",), env={"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}),
        PytestInvocation(paths=("lib/levanter/tests/a.py", "lib/levanter/tests/b.py"), env={"JAX_PLATFORMS": "cpu"}),
    ]


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_changed_files_are_the_uncommitted_change_including_untracked_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    for name in ("kept.py", "edited.py", "removed.py"):
        (tmp_path / name).write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "edited.py").write_text("x = 2\n")
    (tmp_path / "removed.py").unlink()
    (tmp_path / "new.py").write_text("y = 1\n")
    assert changed_files(tmp_path) == ["edited.py", "new.py", "removed.py"]
