# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Poorin prune invariant and append one row of measurements.

Run from the repository root before every prune commit:

    uv run --no-sync python -m tools.prune_check --phase P0 --step "short label"

Checks run in order and the first failure stops the run, so nothing is recorded:

1. Every kept root imports, except ``static_only`` roots.
2. ``ty check`` passes.
3. ``tools.lint`` passes on the uncommitted change.
4. The unit tests affected by the uncommitted change pass, as ``tools.affected_tests`` selects
   and runs them: one CPU pytest process per suite.
5. Each runnable root trains to completion twice. The second run, with warm caches, is timed.
6. R2's metric rows arrived in Finelog. A local Finelog server runs in a temporary directory for
   the duration of the runs, and R2 receives its address with ``--finelog-url``.

``--fast`` skips checks 2 to 4 while the tree is being stripped; a cleanup pass runs them all
afterwards (D73).

Runnable roots write under ``~/.cache/poorin/prune-check``. After the runs, even failed ones, the
per-user run outputs there and the new entries in the repository's ``logs/`` and ``wandb/`` are
deleted; tokenized datasets and kernel autotune caches are kept so later checks stay warm. The
roots run with ``--allow-uncommitted``, so on an uncommitted change each saves a snapshot commit
(D107); the snapshot refs those removed outputs name are deleted with them (D108).
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from levanter.tracker.finelog_metrics import LEVANTER_METRICS_NAMESPACE
from poorin.experiment.runs import collect_runs
from rigging.provenance import SNAPSHOT_REF_PREFIX

from tools import affected_tests
from tools.lint import TY
from tools.reachability import REPO_ROOT, ROOTS_FILE, RunnableRoot, expand, load_roots, summarize, workspace_modules

METRICS_FILE = REPO_ROOT / "docs" / "poorin" / "prune-metrics.csv"
CACHE_HOME = Path.home() / ".cache" / "poorin"
POORIN_PREFIX = CACHE_HOME / "prune-check"
# Levanter writes run logs under ./logs and W&B under ./wandb; runnable roots run from REPO_ROOT.
RUN_LOG_DIRS = (REPO_ROOT / "logs", REPO_ROOT / "wandb")
# The runnable root whose metrics must reach Finelog (spec invariant item 5).
FINELOG_ROOT = "R2"

METRIC_COLUMNS = (
    "date",
    "phase",
    "step",
    "first_party_lines",
    "reachable_lines",
    "reachable_modules",
    "r1_imported_modules",
    "r1_import_time",
    "r1_warm_wall_time",
    "r2_warm_wall_time",
    "venv_size",
)


@dataclass(frozen=True)
class ImportProfile:
    modules: int
    seconds: float


def parse_importtime(stderr: str) -> ImportProfile:
    """Summarize ``python -X importtime`` output: modules imported and total time.

    Total time is the sum of cumulative times of top-level imports. Lines that are not import
    records, such as log lines that share stderr, are ignored.
    """
    modules = 0
    total_us = 0
    for line in stderr.splitlines():
        if not line.startswith("import time:"):
            continue
        fields = line.removeprefix("import time:").split("|")
        if len(fields) != 3 or not fields[0].strip().isdigit():
            continue  # the header row
        modules += 1
        name = fields[2]
        depth = (len(name) - len(name.lstrip(" ")) - 1) // 2
        if depth == 0:
            total_us += int(fields[1])
    return ImportProfile(modules=modules, seconds=total_us / 1e6)


def append_metrics_row(path: Path, row: Mapping[str, object]) -> None:
    """Append ``row`` to the metrics CSV, writing the header if the file is new."""
    if set(row) != set(METRIC_COLUMNS):
        raise ValueError(f"metrics row columns {sorted(row)} != {sorted(METRIC_COLUMNS)}")
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def remove_run_outputs(prefix: Path) -> None:
    """Delete per-user run outputs under ``prefix``. Shared datasets and caches stay."""
    users = prefix / "users"
    if users.exists():
        shutil.rmtree(users)


def snapshot_refs(repo: Path) -> set[str]:
    """The snapshot refs (``refs/poorin/snapshots/...``) the repository at ``repo`` holds now."""
    listing = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname)", SNAPSHOT_REF_PREFIX],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return set(listing.split())


def remove_run_snapshots(prefix: Path, before: set[str], repo: Path) -> None:
    """Delete the snapshot refs the check's own runs created, before their outputs are removed.

    A ref goes when it appeared after ``before`` was taken, a per-user run output under ``prefix``
    names its commit, and no output the check keeps (a tokenized dataset) names it. A run of the
    user's that snapshots the identical working tree during the check shares that commit; its ref
    goes too, and the commit stays in Git's object store until garbage collection.
    """
    users = prefix / "users"
    removed: set[str] = set()
    kept: set[str] = set()
    for run in collect_runs(str(prefix)):
        provenance = run.record.provenance if run.record is not None else None
        if provenance is None or provenance.snapshot_commit is None:
            continue
        (removed if Path(run.path).is_relative_to(users) else kept).add(provenance.snapshot_commit)
    for ref in sorted(snapshot_refs(repo) - before):
        if ref.removeprefix(SNAPSHOT_REF_PREFIX) in removed - kept:
            subprocess.run(["git", "update-ref", "-d", ref], cwd=repo, check=True)


def snapshot_entries(dirs: Iterable[Path]) -> dict[Path, set[str]]:
    """The entries each directory holds now; a missing directory has none."""
    return {directory: set(os.listdir(directory)) if directory.is_dir() else set() for directory in dirs}


def remove_new_entries(before: Mapping[Path, set[str]]) -> None:
    """Delete the entries that appeared in each directory after ``before`` was taken."""
    for directory, names in before.items():
        if not directory.is_dir():
            continue
        for name in set(os.listdir(directory)) - names:
            path = directory / name
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def run_env() -> dict[str, str]:
    """Environment for runnable roots. Levanter enables the compilation cache and preloads NVRTC itself."""
    env = dict(os.environ)
    env.update(POORIN_PREFIX=str(POORIN_PREFIX), WANDB_MODE="offline")
    return env


def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print(f"$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def import_kept_roots(kept: Sequence[str], static_only: Sequence[str], modules: Sequence[str]) -> None:
    targets = sorted(expand(kept, modules) - expand(static_only, modules))
    code = "import importlib, sys\nfor name in sys.argv[1:]:\n    importlib.import_module(name)\n"
    _run([sys.executable, "-c", code, *targets], env={**os.environ, "JAX_PLATFORMS": "cpu"})


def train(root: RunnableRoot, env: Mapping[str, str]) -> float:
    start = time.perf_counter()
    _run([sys.executable, "-m", root.module, *root.args], env=env)
    return time.perf_counter() - start


def train_roots(
    runnable: Sequence[RunnableRoot],
    env: Mapping[str, str],
    *,
    prefix: Path,
    log_dirs: Sequence[Path],
    repo: Path,
) -> dict[str, float]:
    """Train each root twice and time the second, warm run.

    Run outputs, and the snapshot refs they name, are removed even when a run fails, so a failed
    check does not leave a checkpoint behind.
    """
    before = snapshot_entries(log_dirs)
    refs_before = snapshot_refs(repo)
    warm: dict[str, float] = {}
    try:
        for root in runnable:
            train(root, env)  # first run fills the compile and autotune caches
            warm[root.name] = train(root, env)
    finally:
        try:
            remove_run_snapshots(prefix, refs_before, repo)
        finally:
            remove_run_outputs(prefix)
            remove_new_entries(before)
    return warm


def with_finelog_url(runnable: Sequence[RunnableRoot], url: str) -> list[RunnableRoot]:
    """The runnable roots, with ``FINELOG_ROOT`` told to publish its metrics to ``url``."""
    return [
        replace(root, args=(*root.args, "--finelog-url", url)) if root.name == FINELOG_ROOT else root
        for root in runnable
    ]


def levanter_metric_rows(url: str) -> int:
    """Rows in the Levanter metrics table of the Finelog server at ``url``; every run shares it."""
    client = LogClient.connect(url)
    try:
        namespaces = client.list_namespaces()
    finally:
        client.close()
    return next((ns.row_count for ns in namespaces if ns.namespace == LEVANTER_METRICS_NAMESPACE), 0)


def profile_import(module: str) -> ImportProfile:
    result = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", f"import {module}"],
        cwd=REPO_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_importtime(result.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Poorin prune invariant.")
    parser.add_argument("--phase", required=True, help="Prune phase, for example P0.")
    parser.add_argument("--step", required=True, help="Short label for the change being checked.")
    parser.add_argument("--no-record", action="store_true", help="Run the checks without appending a row.")
    parser.add_argument("--fast", action="store_true", help="Skip ty, lint, and tests; import and train only.")
    args = parser.parse_args(argv)

    roots = load_roots(ROOTS_FILE)
    modules = sorted(workspace_modules(REPO_ROOT))
    import_kept_roots(roots.kept, roots.static_only, modules)
    if not args.fast:
        _run(list(TY))
        _run([sys.executable, "-m", "tools.lint"])
        affected_tests.main([])

    with tempfile.TemporaryDirectory() as finelog_dir, require_embedded_server()(log_dir=finelog_dir) as finelog:
        runnable = with_finelog_url(roots.runnable, finelog.address)
        warm = train_roots(runnable, run_env(), prefix=POORIN_PREFIX, log_dirs=RUN_LOG_DIRS, repo=REPO_ROOT)
        metric_rows = levanter_metric_rows(finelog.address)
    if metric_rows == 0:
        raise RuntimeError(f"{FINELOG_ROOT} trained but wrote no metric rows to Finelog")
    print(f"Finelog received {metric_rows} Levanter metric rows", flush=True)

    r1 = next(root for root in roots.runnable if root.name == "R1")
    profile = profile_import(r1.module)
    summary = summarize(REPO_ROOT, roots)
    row = {
        "date": date.today().isoformat(),
        "phase": args.phase,
        "step": args.step,
        "first_party_lines": summary.first_party_lines,
        "reachable_lines": summary.reachable_lines,
        "reachable_modules": summary.reachable_modules,
        "r1_imported_modules": profile.modules,
        "r1_import_time": round(profile.seconds, 2),
        "r1_warm_wall_time": round(warm["R1"], 1),
        "r2_warm_wall_time": round(warm["R2"], 1),
        "venv_size": directory_size(REPO_ROOT / ".venv"),
    }
    print(row, flush=True)
    if not args.no_record:
        append_metrics_row(METRICS_FILE, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
