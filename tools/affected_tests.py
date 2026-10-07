# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and run the unit tests affected by a change.

A change is a list of repository-relative paths; by default, the uncommitted change from
``tools.changes.changed_files``. Tests run in suites, one per pytest configuration: the
repository root (``tests/`` and ``experiments/``) and every workspace library with a
``lib/<name>/tests`` directory.

A test file is selected when it changed, or when it, a suite-local module it imports, or a
``conftest.py`` above it imports a changed module or a module that imports one transitively.
Imports are read statically, including imports inside functions. A deleted module keeps its name,
so tests that still import it are selected and fail. Some changes select whole suites: the root
``pyproject.toml``, ``uv.lock``, or anything under ``config/`` selects every suite, and a library's
``pyproject.toml`` or a non-Python file under ``tests/`` or ``lib/<name>/tests`` selects that
suite.

Each suite runs in its own pytest process, because the suites' ``conftest.py`` modules collide in
one process. Tests run on CPU in UTC, as in Marin's CI, with a complete marker expression that
leaves out slow, cluster, Docker, manual, and Torch tests.

Usage (from the repository root):
    uv run --no-sync python -m tools.affected_tests
"""

import argparse
import os
import shlex
import subprocess
import tomllib
from collections import defaultdict, deque
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.changes import changed_files
from tools.reachability import REPO_ROOT, TOP_LEVEL_PACKAGES, imported_modules, is_test_file, workspace_modules

WORKSPACE_MANIFESTS = frozenset({"pyproject.toml", "uv.lock"})
# Data that tests in several suites read and no module imports: the cluster configs and their test
# fixtures. A change under one of these directories runs every suite.
SHARED_DATA_DIRS = ("config",)
# Test files that load modules by computed name, with the package they load from. No import graph
# sees those edges, so a change to any module in the package selects the test.
DYNAMIC_IMPORTERS = {"tests/test_grug_variant_contracts.py": "experiments.grug"}
ROOT_TEST_DIRS = ("tests", "experiments")
# experiments/ is a source tree that also holds tests: its data files select no tests.
SOURCE_TREES = frozenset({"experiments"})
LOCAL_SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual and not torch"
)
TEST_ENV = {"JAX_PLATFORMS": "cpu", "TZ": "UTC"}
# Haliax's sharding tests expect eight CPU devices, as in Marin's CI.
SUITE_ENV = {"lib/haliax": {"JAX_NUM_CPU_DEVICES": "8"}}
# Tests that fail on this workstation because of the host, not the code. Each entry goes away with
# the code it tests.
HOST_EXCLUSIONS = (
    # tensorboardX is in Levanter's test dependency group, which the GPU environment does not
    # install. P7 deletes the TensorBoard tracker.
    "--ignore=lib/levanter/tests/test_tensorboard.py",
    # Iris's hardware probe expects a host without a GPU. P5 deletes Iris.
    "--deselect=lib/iris/tests/cluster/worker/test_env_probe.py::test_hardware_probe_builds_valid_metadata",
    "--deselect=lib/iris/tests/cluster/worker/test_env_probe.py"
    "::test_hardware_probe_ignores_tpu_env_vars_without_metadata",
    # Verifyit's Go grader fails with this workstation's Go toolchain. P3 deletes Verifyit (D69).
    "--deselect=lib/verifyit/tests/test_gotest.py::test_gotest_real_go_module_passes",
    "--deselect=lib/verifyit/tests/test_gotest.py::test_gotest_undiscovered_test_function_cannot_leave_positive_reward",
)


@dataclass(frozen=True)
class Suite:
    """One pytest configuration. ``config_dir`` holds its ``pyproject.toml``; "" is the root."""

    config_dir: str
    test_dirs: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    """What one suite runs: test files, or its test directories when the whole suite runs."""

    suite: Suite
    paths: tuple[str, ...]


@dataclass(frozen=True)
class PytestRun:
    args: tuple[str, ...]
    env: dict[str, str]


def suites(repo_root: Path) -> list[Suite]:
    """The root suite, then one suite per workspace library that has a ``tests`` directory."""
    members = tomllib.loads((repo_root / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]
    root = Suite("", tuple(name for name in ROOT_TEST_DIRS if (repo_root / name).is_dir()))
    libraries = [
        Suite(member, (f"{member}/tests",)) for member in sorted(members) if (repo_root / member / "tests").is_dir()
    ]
    return [root, *libraries]


def select(changed: Sequence[str], repo_root: Path) -> list[Selection]:
    """The tests each suite must run for ``changed``, a list of repository-relative paths."""
    all_suites = suites(repo_root)
    paths = [PurePosixPath(path) for path in changed]
    if WORKSPACE_MANIFESTS & set(changed) or any(path.is_relative_to(d) for path in paths for d in SHARED_DATA_DIRS):
        return [Selection(suite, suite.test_dirs) for suite in all_suites]
    changed_modules = {name for path in paths if (name := _source_module(path)) is not None}
    modules = workspace_modules(repo_root)
    known = set(modules) | changed_modules
    affected = _with_importers(changed_modules, modules, known)
    selections = []
    for suite in all_suites:
        if _changes_whole_suite(suite, paths):
            selected = suite.test_dirs
        else:
            selected = _affected_tests(suite, paths, affected, known, repo_root)
        if selected:
            selections.append(Selection(suite, selected))
    return selections


def pytest_runs(selections: Iterable[Selection], exclusions: Sequence[str]) -> list[PytestRun]:
    """One CPU pytest invocation per selected suite, skipping the tests ``exclusions`` name.

    ``exclusions`` are ``--ignore`` and ``--deselect`` options with repository-relative paths, such
    as ``HOST_EXCLUSIONS``; each suite gets the ones under its test directories. pytest collects a
    file passed by name even when ``--ignore`` names it, so ignored paths also leave the selection,
    and a suite left with nothing selected does not run.
    """
    runs = []
    for selection in selections:
        suite = selection.suite
        own_prefixes = tuple(f"{test_dir}/" for test_dir in suite.test_dirs)
        own = [option for option in exclusions if option.split("=", 1)[1].startswith(own_prefixes)]
        ignored = tuple(option.removeprefix("--ignore=") for option in own if option.startswith("--ignore="))
        paths = tuple(path for path in selection.paths if not _under_any(path, ignored))
        if not paths:
            continue
        options = [_suite_relative(option, suite) for option in own]
        runs.append(
            PytestRun(
                args=("uv", "run", "--no-sync", "pytest", "-m", LOCAL_SAFE_MARKERS, *options, *paths),
                env={**TEST_ENV, **SUITE_ENV.get(suite.config_dir, {})},
            )
        )
    return runs


def _under_any(path: str, roots: Sequence[str]) -> bool:
    """Whether ``path`` is one of ``roots`` or lies under one."""
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _suite_relative(option: str, suite: Suite) -> str:
    """``option`` as pytest reads it in ``suite``'s run from the repository root.

    ``--deselect`` takes a node ID, which pytest makes relative to the suite's configuration
    directory; ``--ignore`` takes a path relative to the working directory.
    """
    flag, path = option.split("=", 1)
    if flag != "--deselect" or not suite.config_dir:
        return option
    return f"{flag}={path.removeprefix(f'{suite.config_dir}/')}"


def run(runs: Sequence[PytestRun], repo_root: Path) -> None:
    """Run each pytest in turn. A failing run raises ``subprocess.CalledProcessError``."""
    if not runs:
        print("no affected tests", flush=True)
        return
    for pytest_run in runs:
        print(f"$ {shlex.join(pytest_run.args)}", flush=True)
        subprocess.run(pytest_run.args, cwd=repo_root, env={**os.environ, **pytest_run.env}, check=True)


def _source_module(path: PurePosixPath) -> str | None:
    """The module name of a first-party source file, whether or not the file still exists."""
    if path.suffix != ".py" or is_test_file(Path(path)):
        return None
    parts = list(path.with_suffix("").parts)
    if len(parts) > 3 and parts[0] == "lib" and parts[2] == "src":
        parts = parts[3:]
    elif parts[0] not in TOP_LEVEL_PACKAGES:
        return None
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _with_importers(seeds: Collection[str], modules: Mapping[str, Path], known: Collection[str]) -> set[str]:
    """``seeds`` plus every workspace module that imports one of them, directly or transitively."""
    if not seeds:
        return set()
    importers: dict[str, set[str]] = defaultdict(set)
    for name, path in modules.items():
        for dependency in imported_modules(path, name, known, eager=False):
            importers[dependency].add(name)
    affected = set(seeds)
    queue = deque(seeds)
    while queue:
        for importer in importers[queue.popleft()] - affected:
            affected.add(importer)
            queue.append(importer)
    return affected


def _changes_whole_suite(suite: Suite, paths: Iterable[PurePosixPath]) -> bool:
    """A library's manifest, or a data file in a test-only directory, can affect any of its tests."""
    manifest = PurePosixPath(suite.config_dir, "pyproject.toml")
    data_dirs = [test_dir for test_dir in suite.test_dirs if test_dir not in SOURCE_TREES]
    return any(
        (bool(suite.config_dir) and path == manifest)
        or (path.suffix != ".py" and any(path.is_relative_to(data_dir) for data_dir in data_dirs))
        for path in paths
    )


def _affected_tests(
    suite: Suite, changed: Sequence[PurePosixPath], affected: set[str], known: set[str], repo_root: Path
) -> tuple[str, ...]:
    changed_local = {path for path in changed if _is_local(path, suite)}
    if not affected and not changed_local:
        return ()
    files = _local_files(suite, repo_root)
    present = {*files, *changed_local}  # deleted local files keep their names
    by_name: dict[str, set[PurePosixPath]] = defaultdict(set)
    for path in present:
        for name in _local_names(path, suite):
            by_name[name].add(path)
    local_imports: dict[PurePosixPath, set[PurePosixPath]] = {}
    reaches_affected: set[PurePosixPath] = set()
    for path in files:
        module = _dotted(path, "")
        names = imported_modules(repo_root / path, module, by_name, eager=False)
        local_imports[path] = {target for name in names for target in by_name[name]} - {path}
        if imported_modules(repo_root / path, module, known, eager=False) & affected:
            reaches_affected.add(path)
        package = DYNAMIC_IMPORTERS.get(str(path))
        if package is not None and any(name == package or name.startswith(f"{package}.") for name in affected):
            reaches_affected.add(path)
    hits = changed_local | reaches_affected
    return tuple(
        sorted(
            str(path)
            for path in files
            if _collected_by_name(path.name) and _reached(path, suite, local_imports, present) & hits
        )
    )


def _config_conftest(suite: Suite) -> PurePosixPath:
    return PurePosixPath(suite.config_dir or ".", "conftest.py")


def _local_files(suite: Suite, repo_root: Path) -> list[PurePosixPath]:
    """The suite's Python files that are not workspace source modules: tests, helpers, conftests."""
    found = []
    for test_dir in suite.test_dirs:
        for file in sorted((repo_root / test_dir).rglob("*.py")):
            path = PurePosixPath(file.relative_to(repo_root).as_posix())
            if "__pycache__" not in path.parts and _source_module(path) is None:
                found.append(path)
    if (repo_root / _config_conftest(suite)).is_file():
        found.append(_config_conftest(suite))
    return found


def _is_local(path: PurePosixPath, suite: Suite) -> bool:
    """Whether ``path`` is a test, helper, or conftest of ``suite``, existing or deleted."""
    if path.suffix != ".py" or _source_module(path) is not None:
        return False
    return path == _config_conftest(suite) or any(path.is_relative_to(test_dir) for test_dir in suite.test_dirs)


def _dotted(path: PurePosixPath, root: str) -> str:
    parts = list((path.relative_to(root) if root else path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _local_names(path: PurePosixPath, suite: Suite) -> set[str]:
    """Names a suite-local file is imported by: relative to the repository root, the suite's
    directory, and its test directory. Each is on ``sys.path`` in some suite."""
    roots = {"", suite.config_dir, *(test_dir for test_dir in suite.test_dirs if path.is_relative_to(test_dir))}
    return {name for root in roots if (name := _dotted(path, root))}


def _conftests_above(path: PurePosixPath, suite: Suite) -> list[PurePosixPath]:
    """``conftest.py`` in the test's directory and each parent, up to the suite's directory."""
    stop = PurePosixPath(suite.config_dir or ".")
    directory = path.parent
    found = [directory / "conftest.py"]
    while directory not in (stop, PurePosixPath(".")):
        directory = directory.parent
        found.append(directory / "conftest.py")
    return found


def _reached(
    test: PurePosixPath,
    suite: Suite,
    local_imports: Mapping[PurePosixPath, set[PurePosixPath]],
    present: Collection[PurePosixPath],
) -> set[PurePosixPath]:
    """The test file, the conftests above it, and the suite-local files they import, transitively."""
    queue = deque([test, *(conftest for conftest in _conftests_above(test, suite) if conftest in present)])
    seen: set[PurePosixPath] = set()
    while queue:
        path = queue.popleft()
        if path not in seen:
            seen.add(path)
            queue.extend(local_imports.get(path, ()))
    return seen


def _collected_by_name(name: str) -> bool:
    """Whether pytest collects a file by its name (the default ``python_files`` patterns)."""
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="Run the unit tests affected by the uncommitted change.").parse_args(argv)
    run(pytest_runs(select(changed_files(REPO_ROOT), REPO_ROOT), HOST_EXCLUSIONS), REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
