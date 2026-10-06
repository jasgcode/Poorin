# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Poorin prune invariant and append one row of measurements.

Run from the repository root before every prune commit:

    uv run --no-sync python -m tools.prune_check --phase P0 --step "short label"

Checks run in order and the first failure stops the run, so nothing is recorded:

1. Every kept root imports, except ``static_only`` roots.
2. ``pyrefly check`` passes.
3. ``./infra/pre-commit.py`` passes on every changed file. The files are passed by path because
   ``--changed-files`` skips untracked files.
4. The unit tests affected by the uncommitted change pass, as ``tools.affected_tests`` selects
   and runs them: one CPU pytest process per suite.
5. Each runnable root trains to completion twice. The second run, with warm caches, is timed.

Runnable roots write under ``~/.cache/poorin/prune-check``. After the runs, even failed ones, the
per-user run outputs there and the new entries in the repository's ``logs/`` and ``wandb/`` are
deleted; tokenized datasets and kernel autotune caches are kept so later checks stay warm.
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tools import affected_tests
from tools.changes import changed_files
from tools.reachability import ROOTS_FILE, RunnableRoot, expand, load_roots, summarize, workspace_modules

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = REPO_ROOT / "docs" / "poorin" / "prune-metrics.csv"
CACHE_HOME = Path.home() / ".cache" / "poorin"
MARIN_PREFIX = CACHE_HOME / "prune-check"
JAX_CACHE_DIR = CACHE_HOME / "jax"
# Levanter writes run logs under ./logs and W&B under ./wandb; runnable roots run from REPO_ROOT.
RUN_LOG_DIRS = (REPO_ROOT / "logs", REPO_ROOT / "wandb")

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
    """Environment for runnable roots. P5a moves the cache and NVRTC fixes into Levanter."""
    nvidia_lib = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13" / "lib"
    env = dict(os.environ)
    env.update(
        MARIN_PREFIX=str(MARIN_PREFIX),
        JAX_COMPILATION_CACHE_DIR=str(JAX_CACHE_DIR),
        WANDB_MODE="offline",
        LD_LIBRARY_PATH=os.pathsep.join(p for p in (str(nvidia_lib), env.get("LD_LIBRARY_PATH", "")) if p),
    )
    return env


def _run(cmd: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    print(f"$ {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def lint_changed_files() -> None:
    paths = [path for path in changed_files(REPO_ROOT) if (REPO_ROOT / path).is_file()]
    if paths:
        _run(["./infra/pre-commit.py", *paths])


def run_affected_tests() -> None:
    selections = affected_tests.select(changed_files(REPO_ROOT), REPO_ROOT)
    affected_tests.run(affected_tests.pytest_runs(selections), REPO_ROOT)


def import_kept_roots(kept: Sequence[str], static_only: Sequence[str], modules: Sequence[str]) -> None:
    targets = sorted(expand(kept, modules) - expand(static_only, modules))
    code = "import importlib, sys\nfor name in sys.argv[1:]:\n    importlib.import_module(name)\n"
    _run([sys.executable, "-c", code, *targets], env={**os.environ, "JAX_PLATFORMS": "cpu"})


def train(root: RunnableRoot, env: Mapping[str, str]) -> float:
    start = time.perf_counter()
    _run([sys.executable, "-m", root.module, *root.args], env=env)
    return time.perf_counter() - start


def train_roots(
    runnable: Sequence[RunnableRoot], env: Mapping[str, str], *, prefix: Path, log_dirs: Sequence[Path]
) -> dict[str, float]:
    """Train each root twice and time the second, warm run.

    Run outputs are removed even when a run fails, so a failed check does not leave a checkpoint
    behind.
    """
    before = snapshot_entries(log_dirs)
    warm: dict[str, float] = {}
    try:
        for root in runnable:
            train(root, env)  # first run fills the compile and autotune caches
            warm[root.name] = train(root, env)
    finally:
        remove_run_outputs(prefix)
        remove_new_entries(before)
    return warm


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
    args = parser.parse_args(argv)

    roots = load_roots(ROOTS_FILE)
    modules = sorted(workspace_modules(REPO_ROOT))
    import_kept_roots(roots.kept, roots.static_only, modules)
    _run(["uv", "run", "--no-sync", "pyrefly", "check"])
    lint_changed_files()
    run_affected_tests()

    warm = train_roots(roots.runnable, run_env(), prefix=MARIN_PREFIX, log_dirs=RUN_LOG_DIRS)

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
