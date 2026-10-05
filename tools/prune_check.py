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
4. The unit tests affected by the branch and working tree pass. Selection comes from
   ``infra/ci/select_tests.py``; pytest runs with ``--no-sync`` in the existing GPU venv and with
   ``JAX_PLATFORMS=cpu``, as in Marin's CI. ``infra/ci/run_tests.py`` is not used because it
   re-syncs ``.venv`` with CPU-only JAX.
5. Each runnable root trains to completion twice. The second run, with warm caches, is timed.

Runnable roots write under ``~/.cache/poorin/prune-check``. Per-user run outputs are deleted after
the runs; tokenized datasets and kernel autotune caches are kept so later checks stay warm.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from infra.ci.select_tests import select_local_tests
from tools.reachability import ROOTS_FILE, RunnableRoot, expand, load_roots, summarize, workspace_modules

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_FILE = REPO_ROOT / "docs" / "poorin" / "prune-metrics.csv"
CACHE_HOME = Path.home() / ".cache" / "poorin"
MARIN_PREFIX = CACHE_HOME / "prune-check"
JAX_CACHE_DIR = CACHE_HOME / "jax"
BASE_REF = "main"

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


def _changed_files() -> list[str]:
    """Paths changed since the merge base with ``main``, including uncommitted and untracked files."""

    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if line]

    merge_base = git("merge-base", BASE_REF, "HEAD")[0]
    return sorted({*git("diff", "--name-only", merge_base), *git("ls-files", "--others", "--exclude-standard")})


def lint_changed_files() -> None:
    paths = [path for path in _changed_files() if (REPO_ROOT / path).is_file()]
    if paths:
        _run(["./infra/pre-commit.py", *paths])


def run_affected_tests() -> None:
    selection = select_local_tests(_changed_files(), REPO_ROOT)
    paths = sorted({path for leg in selection.matrix for path in shlex.split(leg.test_paths)})
    if not paths:
        print("no affected tests", flush=True)
        return
    # CPU, as in CI: xdist starts many workers, and each would claim most of the GPU's memory.
    _run(["uv", "run", "--no-sync", "pytest", *paths], env={**os.environ, "JAX_PLATFORMS": "cpu"})


def import_kept_roots(kept: Sequence[str], static_only: Sequence[str], modules: Sequence[str]) -> None:
    targets = sorted(expand(kept, modules) - expand(static_only, modules))
    code = "import importlib, sys\nfor name in sys.argv[1:]:\n    importlib.import_module(name)\n"
    _run([sys.executable, "-c", code, *targets], env={**os.environ, "JAX_PLATFORMS": "cpu"})


def train(root: RunnableRoot, env: Mapping[str, str]) -> float:
    start = time.perf_counter()
    _run([sys.executable, "-m", root.module, *root.args], env=env)
    return time.perf_counter() - start


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

    env = run_env()
    warm: dict[str, float] = {}
    for root in roots.runnable:
        train(root, env)  # first run fills the compile and autotune caches
        warm[root.name] = train(root, env)
    remove_run_outputs(MARIN_PREFIX)

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
