# P1 Fleet Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete Marin's fleet operations (`infra/`, `scripts/`, `docker/`, `etc/`, the GitHub
configuration, the ferries, Marin's agent notes and organization skills), after moving the two
things Poorin still needs from them (affected-test selection and the lint entry point) into
`tools/`; switch to ty for type checking and ruff for formatting; make the local machine the
default cluster; and replace the kept roots with the workbench tools.

**Architecture:** Two new stdlib-plus-PyYAML tools join P0's `tools/` package:
`tools/affected_tests.py` selects tests from the import graph that `tools/reachability.py` already
builds, and `tools/lint.py` runs the checks Marin's `infra/pre-commit.py` ran, at the same tool
versions. A third small module, `tools/changes.py`, defines "the uncommitted change" for both.
Once nothing imports `infra/` or `scripts/`, each is deleted in one commit with its tests, docs,
workspace entries, and tool configuration. Marin's two cluster configs become test fixtures in
`config/testing/`, and `config/local.yaml` becomes the default.

**Tech Stack:** Python 3.12, `uv`, pytest with xdist, PyYAML, and through `uvx`: ruff 0.14.3,
ty 0.0.84 (replacing pyrefly 1.0.0 in Task 3), and black 25.9.0 (until ruff format replaces it in
Task 11).

**Spec:** `docs/poorin/specs/2026-10-04-core-prune-design.md`: sections "Pass 1: deletion order"
(row P1), "Lint replacement (P1)", "Agent instructions and documentation", "Invariant", and
"Risks". Its amendment `docs/poorin/specs/2026-10-06-workbench-and-roots-design.md` replaces the
kept roots. Decisions D49 (ty) and D50 (ruff format) in `docs/poorin/decisions.md` amend "Lint
replacement (P1)". Read all three before starting.

## Global Constraints

- Work on branch `prune/p1-fleet-ops`, cut from local `main` at `75ea55308`. Do not push.
- Run every project command with `uv run --no-sync ...` from the repository root. After any
  `uv lock`, resync with `uv sync --locked --all-packages --extra=gpu` (this reproduces the
  current venv; a dry run at `75ea55308` showed no package changes) and confirm the GPU backend
  with `uv run --no-sync python -c "import jax; print(jax.default_backend())"` (expected: `gpu`).
- Every commit runs `uv run --no-sync python -m tools.prune_check --phase P1 --step "<label>"`
  before committing, and commits the metrics row it appends to `docs/poorin/prune-metrics.csv`
  together with the change. A check takes about 5 minutes. A commit that changes the root
  `pyproject.toml` or `uv.lock` runs every test suite, about 12 more minutes.
- Lint with `./infra/pre-commit.py <paths>` until Task 2 lands, then with
  `uv run --no-sync python -m tools.lint`. `prune_check` runs the current one.
- New Python files start with `# Copyright The Marin Authors` and
  `# SPDX-License-Identifier: Apache-2.0`.
- Tests run with `JAX_PLATFORMS=cpu`. Run a single test file in-process with `-n 0`. Never pass
  a partial `-m` marker expression.
- Follow `TESTING.md`: test behavior, no tautologies, no mocks of internal functions, no
  `time.sleep`.
- Commit messages: imperative title of at most 72 characters with a `[scope]` prefix; the body
  says what the deleted code did for Marin and why Poorin does not need it; no attribution
  trailers.
- Each deletion commit removes the code, its tests, its docs, its workspace entry, and its tool
  configuration. Before deleting a module, search the surviving tree for its dotted name as a
  string.
- Before any `rm -rf` of a deleted directory, list what git does not track there
  (`git status --short --ignored <dir>`); expected: only `__pycache__` directories.
- Each dependency change is its own commit, and its `uv.lock` diff is checked for versions that
  moved without being asked (Tasks 7 and 8 give the check).
- `docs/poorin/decisions.md` is the decision log. When a task lands a decision listed there as
  Planned (D37–D43 and D47–D50), set its status to Adopted and add the commit, in that task's commit. Every
  ruling made during execution, and every final-review finding fixed or set aside, gets a new
  entry in the commit it governs.

## Measured before planning (at `75ea55308`)

- Every suite, local-safe markers, on this workstation: rigging 7 s, fray 8 s, zephyr 108 s,
  finestore 8 s, finelog 4 s, haliax 31 s, levanter 250 s, root (`tests`, `experiments`) 90 s,
  iris 61 s, ducky 4 s, dupekit 6 s, verifyit 161 s. Total about 12 minutes.
- Failures at baseline, none caused by a code defect in the tested library:
  - `lib/finelog/tests/test_telemetry_layout_bench.py`: assumes UTC; passes with `TZ=UTC`.
  - `lib/levanter/tests/test_tensorboard.py`: `tensorboardX` is in Levanter's `test` dependency
    group, which `--extra=gpu` does not install. Installing every test group would add 28
    packages, Gradio among them, for code P7 deletes.
  - Two `lib/iris/tests/cluster/worker/test_env_probe.py` tests expect a host without a GPU.
  - Two `lib/verifyit/tests/test_gotest.py` tests fail with this workstation's Go toolchain.
  - Four `tests/infra/ci/test_package_release.py` tests read `.github/workflows`, which P0
    deleted; P0's test selection never ran them. Task 7 deletes them.
  - `lib/shellbox` and `lib/taskcompendium` tests cannot import their packages: neither is a
    workspace member. P2 deletes both.
- `./infra/pre-commit.py --all-files` passes. Its patterns are matched with `fnmatch`, so
  `**/*` never matched a root-level file: `CLAUDE.md` and `.pyrefly-baseline.json` lack a final
  newline and were never checked.
- With `MARIN_CLUSTER=local` and no Marin cluster configs, 23 rigging tests and 12 root-suite
  tests fail; all assert Marin's multi-region bucket layout. Two of the root failures come from
  experiment modules that compute temp-bucket paths at import time.

## Decisions this plan makes

| Decision | Reason |
|---|---|
| Test selection moves to `tools/affected_tests.py`, built on `tools.reachability`'s import graph. | `prune_check` used 1 of `infra/ci/select_tests.py`'s 1,140 lines of CI-matrix logic. The reachability graph already parses every import, including imports inside functions, so lazily imported code still selects its tests. |
| Suites come from the workspace members in the root `pyproject.toml`. | Shellbox and Taskcompendium are not members and their tests cannot import. |
| `tools/affected_tests.py` keeps a commented list of host-specific exclusions, each naming the phase that deletes its code. | The alternative, installing every test group, grows the venv by 28 packages for code that P2, P5, and P7 delete. |
| Tests run with `TZ=UTC`. | Marin's CI runs in UTC and one Finelog test depends on it. |
| The lint entry point is `tools/lint.py`, run as `python -m tools.lint`, not a script in the repository root. | `tools/` is the root-level package for Poorin's own tooling: it is type-checked, its tests run in the root suite, and it shares `changed_files` with the test selector. |
| `tools/lint.py` lets ruff find each library's configuration itself and passes black an explicit `--config` per library. | Ruff resolves the nearest `pyproject.toml` per file, so Marin's explicit `--config lib/marin/pyproject.toml` was redundant. Black uses one configuration per invocation, and Levanter (119 columns), Haliax (119, preview), and the root (121, preview) differ. |
| `tools/lint.py` checks root-level files and validates every `.json` file. | Marin's patterns skipped root files by accident. Marin's JSON check only covered Grafana dashboards, which Task 6 deletes. Every tracked JSON file outside `infra/` parses today. |
| `tools.*` becomes a kept root. | `tools/` is counted in `first_party_lines` but was unreachable, so P7's unreachable sweep would have listed it. `reachable_lines` rises by the size of `tools/` in Task 1. |
| Delete the `Makefile`. | Every target wraps a tool this phase deletes, or sets up Marin's GCP project. |
| Delete `.github/` entirely, but keep `.dockerignore`. | `.github/` holds composite actions for the deleted workflows, CodeQL and Dependabot configuration, and PR templates. Levanter's TPU tutorial builds Docker images that respect `.dockerignore`; sub-project 2 decides on it. |
| Delete Marin's ferries in this phase. | Ferries are Marin's scheduled canary runs. No root reaches them, and three import `infra.ci.run_status`. |
| Delete `.agents/agents`, `.agents/docs`, `.agents/logbooks`, `.agents/ops`, and `.agents/projects`. | 151 files of Marin design notes, logbooks, and an ops persona for systems Poorin deletes. They remain at `upstream-baseline`. |
| Marin's cluster configs move to `config/testing/`. The rigging and root suites resolve clusters from there. | Cluster discovery reads only the top level of each config directory, so `config/testing/` is invisible to the default search. Those suites test region routing, temp buckets, and cross-region guards against Marin's layout. |
| The root suite pins the Marin layout in `pytest_configure`, not a fixture. | Two experiment modules call `marin_temp_bucket` at import time, before any fixture runs. |
| ty replaces pyrefly (D49), right after the linter is ported. Existing errors get `# ty: ignore[rule]` markers for Pass 2 to fix. | The owner's choice of Astral's toolchain. Swapping right after the port keeps every later commit under the new checker. |
| ruff format replaces black (D50), after the deletions. One commit reformats every tracked Python file; a second lists it in `.git-blame-ignore-revs`. | One tool for linting and formatting. Reformatting after Tasks 7 and 9 skips `infra/` and `scripts/`. |
| The kept roots are replaced by the workbench tools (D47, D48) in their own commit. | The amendment spec. Its own commit gives the drop in `reachable_lines` its own metrics row. |

## Review Focus

1. **A deleted test package** (for example `tests/ci/`, including its `__init__.py`): the
   selector must select nothing for it and must not run the whole root suite. Test in Task 1.
2. **A module imported only inside a function:** changing it must still select its tests. Test
   in Task 1.
3. **A renamed Python file:** both the old and the new path count as changed, so tests that still
   import the old name are selected. Test in Task 1.
4. **Binary or non-UTF-8 files in the change:** lint skips the text checks for them instead of
   crashing. Test in Task 2.
5. **`--fix` in a Git hook:** a run that repaired a file still fails, so the repaired file is
   reviewed before it is committed. Test in Task 2.

---

### Task 1: Select affected tests without `infra/ci`

**Files:**
- Create: `tools/changes.py`
- Create: `tools/affected_tests.py`
- Create: `tests/tools/test_changes.py`
- Create: `tests/tools/test_affected_tests.py`
- Modify: `tools/prune_check.py`
- Modify: `tests/tools/test_prune_check.py`
- Modify: `tools/roots.toml`

**Interfaces:**
- Consumes (from P0's `tools.reachability`): `REPO_ROOT`, `TOP_LEVEL_PACKAGES`,
  `imported_modules(path, module, known, *, eager) -> set[str]`, `is_test_file(rel_path) -> bool`,
  `workspace_modules(repo_root) -> dict[str, Path]`.
- Produces:
  - `tools.changes.changed_files(repo_root: Path) -> list[str]`
  - `tools.affected_tests.Suite(config_dir: str, test_dirs: tuple[str, ...])`
  - `tools.affected_tests.Selection(suite: Suite, paths: tuple[str, ...])`
  - `tools.affected_tests.PytestRun(args: tuple[str, ...], env: dict[str, str])`
  - `tools.affected_tests.suites(repo_root: Path) -> list[Suite]`
  - `tools.affected_tests.select(changed: Sequence[str], repo_root: Path) -> list[Selection]`
  - `tools.affected_tests.pytest_runs(selections: Iterable[Selection]) -> list[PytestRun]`
  - `tools.affected_tests.run(runs: Sequence[PytestRun], repo_root: Path) -> None` (raises
    `subprocess.CalledProcessError` at the first failing suite)
  - `tools.affected_tests.LOCAL_SAFE_MARKERS: str`
  - CLI: `python -m tools.affected_tests`

- [ ] **Step 1: Create the branch**

```bash
git checkout -b prune/p1-fleet-ops main
git log --oneline -1
```

Expected: `75ea55308 [poorin] Fix prune_check's own test selection and run cleanup`.

- [ ] **Step 2: Write the failing test for `changed_files`**

Create `tests/tools/test_changes.py`:

```python
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
```

- [ ] **Step 3: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_changes.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.changes'`.

- [ ] **Step 4: Implement `tools/changes.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The uncommitted change that Poorin's tools check."""

import subprocess
from pathlib import Path


def changed_files(repo_root: Path) -> list[str]:
    """Tracked paths that differ from ``HEAD``, plus untracked files: repository-relative, sorted.

    Deleted paths are included, and a rename counts as a deletion plus an addition. Callers that
    read files skip the paths that no longer exist.
    """

    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], cwd=repo_root, check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if line]

    return sorted({*git("diff", "--name-only", "--no-renames", "HEAD"), *git("ls-files", "--others", "--exclude-standard")})
```

- [ ] **Step 5: Run it to verify it passes**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_changes.py -n 0 -q`
Expected: `1 passed`.

- [ ] **Step 6: Write the failing selector tests**

Create `tests/tools/test_affected_tests.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.affected_tests on a small synthetic workspace."""

from pathlib import Path

import pytest

from tools.affected_tests import LOCAL_SAFE_MARKERS, Selection, Suite, pytest_runs, select

FILES = {
    "pyproject.toml": '[tool.uv.workspace]\nmembers = ["lib/alpha", "lib/beta"]\n',
    "uv.lock": "",
    "lib/alpha/pyproject.toml": "",
    "lib/alpha/src/alpha/__init__.py": "",
    "lib/alpha/src/alpha/core.py": "from alpha import util\n",
    "lib/alpha/src/alpha/util.py": "X = 1\n",
    "lib/alpha/src/alpha/other.py": "Y = 1\n",
    "lib/alpha/tests/conftest.py": "",
    "lib/alpha/tests/helpers.py": "import alpha.other\n\nTHING = 1\n",
    "lib/alpha/tests/test_core.py": "import alpha.core\n",
    "lib/alpha/tests/test_other.py": "from helpers import THING\n",  # pythonpath = ["tests"]
    "lib/alpha/tests/test_relative.py": "from .helpers import THING\n",
    "lib/alpha/tests/data/sample.txt": "x\n",
    "lib/alpha/tests/nested/conftest.py": "",
    "lib/alpha/tests/nested/test_nested.py": "",
    "lib/beta/pyproject.toml": "",
    "lib/beta/src/beta/__init__.py": "",
    "lib/beta/src/beta/api.py": "def load():\n    import alpha.util\n\n    return alpha.util\n",
    "lib/beta/tests/test_api.py": "from beta import api\n",
    "lib/gamma/tests/test_gamma.py": "import alpha.util\n",  # not a workspace member
    "experiments/__init__.py": "",
    "experiments/README.md": "",
    "experiments/run.py": "import alpha.core\n",
    "experiments/test_run.py": "import experiments.run\n",
    "tests/__init__.py": "",
    "tests/support.py": "import beta.api\n\nload = beta.api.load\n",
    "tests/test_root.py": "from tests.support import load\n",
    "docs/index.md": "",
}
ROOT = Suite("", ("tests", "experiments"))
ALPHA = Suite("lib/alpha", ("lib/alpha/tests",))
BETA = Suite("lib/beta", ("lib/beta/tests",))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_changing_a_module_selects_tests_that_import_it_directly_lazily_or_through_other_modules(repo):
    # core imports util; beta.api imports it inside a function; experiments.run imports core;
    # tests/support.py (a root-suite helper) imports beta.api.
    assert select(["lib/alpha/src/alpha/util.py"], repo) == [
        Selection(ROOT, ("experiments/test_run.py", "tests/test_root.py")),
        Selection(ALPHA, ("lib/alpha/tests/test_core.py",)),
        Selection(BETA, ("lib/beta/tests/test_api.py",)),
    ]


def test_tests_reach_changed_modules_through_suite_local_helpers(repo):
    # helpers.py is imported by its pythonpath name in one test and relatively in the other.
    assert select(["lib/alpha/src/alpha/other.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_a_changed_helper_selects_the_tests_that_import_it(repo):
    assert select(["lib/alpha/tests/helpers.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_deleting_a_module_selects_the_tests_that_still_import_it(repo):
    (repo / "lib/alpha/src/alpha/other.py").unlink()
    assert select(["lib/alpha/src/alpha/other.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_deleting_a_test_package_selects_nothing(repo):
    assert select(["tests/gone/__init__.py", "tests/gone/test_gone.py"], repo) == []


def test_a_changed_conftest_selects_the_tests_below_it(repo):
    assert select(["lib/alpha/tests/nested/conftest.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/nested/test_nested.py",)),
    ]
    assert select(["lib/alpha/tests/conftest.py"], repo) == [
        Selection(
            ALPHA,
            (
                "lib/alpha/tests/nested/test_nested.py",
                "lib/alpha/tests/test_core.py",
                "lib/alpha/tests/test_other.py",
                "lib/alpha/tests/test_relative.py",
            ),
        ),
    ]


def test_workspace_manifests_select_every_member_suite_and_a_library_manifest_its_own(repo):
    assert select(["uv.lock"], repo) == [
        Selection(ROOT, ROOT.test_dirs),
        Selection(ALPHA, ALPHA.test_dirs),
        Selection(BETA, BETA.test_dirs),
    ]
    assert select(["lib/beta/pyproject.toml"], repo) == [Selection(BETA, BETA.test_dirs)]


def test_test_data_selects_its_suite_but_experiment_files_and_docs_select_nothing(repo):
    assert select(["lib/alpha/tests/data/sample.txt"], repo) == [Selection(ALPHA, ALPHA.test_dirs)]
    assert select(["experiments/README.md", "docs/index.md"], repo) == []


def test_each_suite_runs_on_cpu_in_utc_with_the_complete_local_marker_expression():
    haliax = Suite("lib/haliax", ("lib/haliax/tests",))
    root_run, haliax_run = pytest_runs([Selection(ROOT, ("tests/test_root.py",)), Selection(haliax, haliax.test_dirs)])
    assert root_run.env == {"JAX_PLATFORMS": "cpu", "TZ": "UTC"}
    assert haliax_run.env == {"JAX_PLATFORMS": "cpu", "TZ": "UTC", "JAX_NUM_CPU_DEVICES": "8"}
    for run, paths in ((root_run, ("tests/test_root.py",)), (haliax_run, ("lib/haliax/tests",))):
        assert run.args[run.args.index("-m") + 1] == LOCAL_SAFE_MARKERS
        assert run.args[-len(paths) :] == paths


def test_host_exclusions_apply_only_to_their_own_suite():
    levanter = Suite("lib/levanter", ("lib/levanter/tests",))
    root_run, levanter_run = pytest_runs([Selection(ROOT, ROOT.test_dirs), Selection(levanter, levanter.test_dirs)])
    assert "--ignore=lib/levanter/tests/test_tensorboard.py" in levanter_run.args
    assert not any(arg.startswith(("--ignore", "--deselect")) for arg in root_run.args)
```

- [ ] **Step 7: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_affected_tests.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.affected_tests'`.

- [ ] **Step 8: Implement `tools/affected_tests.py`**

```python
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
``pyproject.toml`` or ``uv.lock`` selects every suite, and a library's ``pyproject.toml`` or a
non-Python file under ``tests/`` or ``lib/<name>/tests`` selects that suite.

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
    # Verifyit's Go grader fails with this workstation's Go toolchain. P2 deletes Verifyit.
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
    if WORKSPACE_MANIFESTS & set(changed):
        return [Selection(suite, suite.test_dirs) for suite in all_suites]
    paths = [PurePosixPath(path) for path in changed]
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


def pytest_runs(selections: Iterable[Selection]) -> list[PytestRun]:
    """One CPU pytest invocation per selected suite."""
    runs = []
    for selection in selections:
        suite = selection.suite
        own_prefixes = tuple(f"{test_dir}/" for test_dir in suite.test_dirs)
        exclusions = [option for option in HOST_EXCLUSIONS if option.split("=", 1)[1].startswith(own_prefixes)]
        runs.append(
            PytestRun(
                args=("uv", "run", "--no-sync", "pytest", "-m", LOCAL_SAFE_MARKERS, *exclusions, *selection.paths),
                env={**TEST_ENV, **SUITE_ENV.get(suite.config_dir, {})},
            )
        )
    return runs


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
    run(pytest_runs(select(changed_files(REPO_ROOT), REPO_ROOT)), REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 9: Run the selector tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_affected_tests.py tests/tools/test_changes.py -n 0 -q`
Expected: `11 passed`.

- [ ] **Step 10: Switch `prune_check` to the new modules**

In `tools/prune_check.py`:

1. Replace the import block line `from infra.ci.select_tests import select_local_tests` with
   `from tools import affected_tests` and add `from tools.changes import changed_files`
   (ruff's isort orders them).
2. Delete `HALIAX_PACKAGE`, `HALIAX_CPU_DEVICES`, `TOOLS_TEST_LEG`, the `PytestInvocation`
   dataclass, and the functions `changed_files`, `pytest_invocations`, and `affected_legs`.
3. Replace `run_affected_tests` with:

```python
def run_affected_tests() -> None:
    selections = affected_tests.select(changed_files(REPO_ROOT), REPO_ROOT)
    affected_tests.run(affected_tests.pytest_runs(selections), REPO_ROOT)
```

4. Replace docstring item 4 with:

```
4. The unit tests affected by the uncommitted change pass, as ``tools.affected_tests`` selects
   and runs them: one CPU pytest process per suite.
```

In `tests/tools/test_prune_check.py`, remove `PytestInvocation`, `affected_legs`,
`changed_files`, and `pytest_invocations` from the import, and delete
`test_pytest_invocations_run_each_package_separately_and_give_haliax_eight_cpu_devices`,
`_git`, `test_changed_files_are_the_uncommitted_change_including_untracked_files` (moved to
`tests/tools/test_changes.py`), and `test_affected_legs_add_the_tools_tests_for_changes_under_tools`
(a change under `tools/` now selects `tests/tools` through the import graph). Remove `subprocess`
from the imports if nothing else uses it.

In `tools/roots.toml`, add `"tools.*",` as the last entry of `[kept] modules`, with the comment
`# Poorin's own tooling: the prune must not delete it.` above it.

- [ ] **Step 11: Run the tools tests and confirm nothing in `tools/` imports `infra`**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools -n 0 -q
grep -rn "infra" tools/
```

Expected: all pass (16 from P0 minus 3 deleted, plus 11 new: 24). The grep prints only the
`prune_check.py` docstring lines that describe `./infra/pre-commit.py`; Task 2 replaces them.

- [ ] **Step 12: Check the selector on the real change**

Run: `uv run --no-sync python -c "from tools.affected_tests import select; from tools.changes import changed_files; from tools.reachability import REPO_ROOT; print(select(changed_files(REPO_ROOT), REPO_ROOT))"`

Expected: one root-suite selection with exactly `tests/tools/test_affected_tests.py`,
`tests/tools/test_changes.py`, and `tests/tools/test_prune_check.py`. The changed modules are
`tools.changes`, `tools.affected_tests`, and `tools.prune_check`; `test_reachability.py` imports
only the unchanged `tools.reachability`, and `roots.toml` is not Python. Any other test means
the selector over-reaches; investigate before continuing.

- [ ] **Step 13: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Select affected tests without infra/ci"`
Expected: exit 0 and one new row in `docs/poorin/prune-metrics.csv`; `reachable_lines` and
`reachable_modules` rise by the size of `tools/`.

- [ ] **Step 14: Commit**

This commit also carries this plan, `docs/poorin/decisions.md`, and the amendment spec
`docs/poorin/specs/2026-10-06-workbench-and-roots-design.md`, which are uncommitted when the branch
is cut. Set D37 and D40 to Adopted first.

```bash
git add tools tests/tools docs/poorin/prune-metrics.csv docs/poorin/plans/2026-10-05-p1-fleet-operations.md \
  docs/poorin/decisions.md docs/poorin/specs/2026-10-06-workbench-and-roots-design.md
git commit -F - <<'EOF'
[poorin] Select affected tests from the reachability import graph

prune_check chose tests through infra/ci/select_tests.py, which P1
deletes. That module builds GitHub Actions matrices; prune_check used
one entry point of its 1,140 lines. tools/affected_tests.py selects
from the import graph tools/reachability.py already builds, so tests
of lazily imported modules, of deleted modules, and of test helpers
are selected too.

Each suite still runs in its own pytest process, now also with
TZ=UTC (one Finelog test assumes it), a complete local-safe marker
expression, and a commented list of tests that fail only because of
this workstation. Suites come from the workspace members, so
Shellbox and Taskcompendium, which are not members, are skipped.

tools/changes.py defines the uncommitted change for both tools and
lists both sides of a rename. tools.* becomes a kept root so the
prune never treats the tooling as unreachable.
EOF
```

---

### Task 2: Lint entry point in `tools/`

**Files:**
- Create: `tools/lint.py`
- Create: `tests/tools/test_lint.py`
- Modify: `tools/prune_check.py`
- Modify: `CLAUDE.md`, `.pyrefly-baseline.json` (final newline only, written by `--fix`)

**Interfaces:**
- Consumes: `tools.changes.changed_files`, `tools.reachability.REPO_ROOT`.
- Produces:
  - `tools.lint.lint(paths: Sequence[str], *, fix: bool, repo_root: Path = REPO_ROOT) -> list[str]`
    (problem lines; empty means clean)
  - `tools.lint.file_problems(repo_root: Path, rel: str, fix: bool) -> list[str]`
  - `tools.lint.python_style(path: str) -> PythonStyle` with fields `prefix`, `black_config`,
    `header`
  - `tools.lint.is_excluded(path: str) -> bool`
  - CLI: `python -m tools.lint [--fix] [--all-files] [PATH ...]`; exit 1 when any problem was
    found

- [ ] **Step 1: Write the failing tests**

Create `tests/tools/test_lint.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.lint's file checks. Ruff, black, and pyrefly are not run here."""

import json
from pathlib import Path

import pytest

from tools.lint import file_problems, is_excluded, lint, python_style

MARIN = "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n"
HALIAX = "# Copyright The Levanter Authors\n#\n# SPDX-License-Identifier: Apache-2.0\n"


def write(root: Path, rel: str, content: str | bytes) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return rel


def test_a_missing_license_header_is_reported_and_inserted_after_the_shebang(tmp_path):
    rel = write(tmp_path, "tools/run.py", "#!/usr/bin/env python\nprint('hi')\n")
    assert file_problems(tmp_path, rel, fix=True) == ["missing license header"]
    assert (tmp_path / rel).read_text() == "#!/usr/bin/env python\n" + MARIN + "\nprint('hi')\n"
    assert file_problems(tmp_path, rel, fix=False) == []


def test_each_library_requires_its_own_license_header(tmp_path):
    haliax = write(tmp_path, "lib/haliax/src/haliax/core.py", HALIAX + "X = 1\n")
    marin_header_in_haliax = write(tmp_path, "lib/haliax/src/haliax/other.py", MARIN + "X = 1\n")
    assert file_problems(tmp_path, haliax, fix=False) == []
    assert file_problems(tmp_path, marin_header_in_haliax, fix=False) == ["missing license header"]


def test_black_uses_each_library_configuration():
    assert python_style("lib/levanter/src/levanter/main/train_lm.py").black_config == "lib/levanter/pyproject.toml"
    assert python_style("lib/haliax/tests/test_axis.py").black_config == "lib/haliax/pyproject.toml"
    assert python_style("lib/marin/src/marin/experiment/cli.py").black_config == "pyproject.toml"
    assert python_style("tools/lint.py").black_config == "pyproject.toml"


def test_trailing_whitespace_and_a_missing_final_newline_are_fixed(tmp_path):
    rel = write(tmp_path, "notes.md", "title  \nbody\t\nend")
    assert file_problems(tmp_path, rel, fix=True) == ["trailing whitespace", "no final newline"]
    assert (tmp_path / rel).read_text() == "title\nbody\nend\n"


def test_merge_conflict_markers_are_reported(tmp_path):
    rel = write(tmp_path, "notes.md", f"{'<' * 7} HEAD\nours\n{'=' * 7}\ntheirs\n{'>' * 7} branch\n")
    assert file_problems(tmp_path, rel, fix=False) == ["merge-conflict marker"]


@pytest.mark.parametrize(
    "rel, content",
    [("bad.py", MARIN + "def f(:\n"), ("bad.toml", "a =\n"), ("bad.yaml", "a: [\n"), ("bad.json", "{\n")],
)
def test_files_that_do_not_parse_are_reported(tmp_path, rel, content):
    write(tmp_path, rel, content)
    [problem] = file_problems(tmp_path, rel, fix=False)
    assert problem.startswith("does not parse")


def test_levanter_yaml_include_tags_parse(tmp_path):
    rel = write(tmp_path, "lib/levanter/config/run.yaml", "data: !include data/openwebtext.yaml\n")
    assert file_problems(tmp_path, rel, fix=False) == []


def test_notebook_outputs_are_reported_and_cleared(tmp_path):
    cell = {"cell_type": "code", "source": ["1 + 1"], "metadata": {}, "execution_count": 3, "outputs": [{"x": 1}]}
    notebook = {"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    rel = write(tmp_path, "docs/demo.ipynb", json.dumps(notebook) + "\n")
    assert file_problems(tmp_path, rel, fix=True) == ["notebook has outputs"]
    [cleared] = json.loads((tmp_path / rel).read_text())["cells"]
    assert cleared["outputs"] == [] and cleared["execution_count"] is None


def test_large_files_are_reported(tmp_path):
    rel = write(tmp_path, "data.txt", "x\n" * 256_001)
    assert file_problems(tmp_path, rel, fix=False) == ["larger than 500 KB"]


def test_binary_files_skip_the_text_checks(tmp_path):
    rel = write(tmp_path, "model.bin", b"\xff\xfe trailing \nno final newline")
    assert file_problems(tmp_path, rel, fix=False) == []


def test_lock_files_and_images_are_excluded_at_any_depth():
    assert is_excluded("uv.lock")
    assert is_excluded("docs/images/plot.png")
    assert not is_excluded("pyproject.toml")


def test_a_run_that_fixed_a_file_still_fails_and_the_next_run_passes(tmp_path):
    rel = write(tmp_path, "notes.md", "x \n")
    assert lint([rel], fix=True, repo_root=tmp_path) == ["notes.md: trailing whitespace"]
    assert lint([rel], fix=False, repo_root=tmp_path) == []


def test_paths_that_no_longer_exist_are_skipped(tmp_path):
    assert lint(["deleted.py"], fix=False, repo_root=tmp_path) == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_lint.py -n 0 -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.lint'`.

- [ ] **Step 3: Implement `tools/lint.py`**

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lint and format Poorin's files. Replaces Marin's ``infra/pre-commit.py``.

Usage (from the repository root):
    uv run --no-sync python -m tools.lint                 # the uncommitted change
    uv run --no-sync python -m tools.lint --fix           # and fix what can be fixed
    uv run --no-sync python -m tools.lint --all-files     # every tracked file
    uv run --no-sync python -m tools.lint PATH ...        # these files

Every file is checked for size, merge-conflict markers, trailing whitespace, and a final newline.
Python, TOML, YAML, and JSON files must parse, and notebooks must have no outputs. Python files
must start with their library's license header and pass ruff and black, and pyrefly checks the
project when any Python file is linted. Ruff and black run at the versions Marin pinned, with
each library's own configuration, so switching from Marin's script reformats nothing.

A check fails when it finds a problem, even when ``--fix`` repaired it, so a fixed file is
reviewed before it is committed.
"""

import argparse
import ast
import fnmatch
import json
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tools.changes import changed_files
from tools.reachability import REPO_ROOT

RUFF = ("uvx", "ruff@0.14.3", "check")
BLACK = ("uvx", "black@25.9.0")
PYREFLY = ("uv", "run", "--no-sync", "pyrefly", "check")
MAX_FILE_BYTES = 500 * 1024
# Built at runtime so this file does not contain the markers it looks for.
CONFLICT_MARKERS = (b"<" * 7, b">" * 7)
EXCLUDED = (
    ".git/**",
    ".github/**",
    "tests/snapshots/**",
    "**/*_connect.py",
    "**/*_pb2.py",
    "**/*.gz",
    "**/*.pb",
    "**/*.index",
    "**/*.ico",
    "**/*.npy",
    "**/*.lock",
    "**/*.png",
    "**/*.jpg",
    "**/*.html",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.mov",
    "**/*.mp4",
    "**/*.data-*",
    "**/package-lock.json",
    "**/*-template.yaml",
)
MARIN_HEADER = ("Copyright The Marin Authors", "SPDX-License-Identifier: Apache-2.0")
LEVANTER_HEADER = ("Copyright The Levanter Authors", "SPDX-License-Identifier: Apache-2.0")
HALIAX_HEADER = ("Copyright The Levanter Authors", "", "SPDX-License-Identifier: Apache-2.0")


@dataclass(frozen=True)
class PythonStyle:
    """How Python under ``prefix`` is formatted: black's configuration and the license header."""

    prefix: str
    black_config: str
    header: tuple[str, ...]


# Levanter and Haliax keep the formatting and headers they had as separate projects.
PYTHON_STYLES = (
    PythonStyle("lib/levanter/", "lib/levanter/pyproject.toml", LEVANTER_HEADER),
    PythonStyle("lib/haliax/", "lib/haliax/pyproject.toml", HALIAX_HEADER),
    PythonStyle("", "pyproject.toml", MARIN_HEADER),
)


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that accepts Levanter's ``!include`` tag without following it."""


def _include(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    return str(loader.construct_scalar(node))


_ConfigLoader.add_constructor("!include", _include)


def python_style(path: str) -> PythonStyle:
    """The style for a repository-relative Python path."""
    return next(style for style in PYTHON_STYLES if path.startswith(style.prefix))


def is_excluded(path: str) -> bool:
    """Whether a repository-relative path is skipped. A leading ``**/`` also matches no directory."""
    return any(
        fnmatch.fnmatch(path, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]))
        for pattern in EXCLUDED
    )


def file_problems(repo_root: Path, rel: str, fix: bool) -> list[str]:
    """Problems in one file. With ``fix``, the fixable ones are repaired and still reported."""
    path = repo_root / rel
    data = path.read_bytes()
    problems = []
    if len(data) > MAX_FILE_BYTES:
        problems.append(f"larger than {MAX_FILE_BYTES // 1024} KB")
    if any(marker in data for marker in CONFLICT_MARKERS):
        problems.append("merge-conflict marker")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return problems  # binary: only the byte-level checks apply
    fixed = text
    if any(line.rstrip("\r").endswith((" ", "\t")) for line in fixed.split("\n")):
        problems.append("trailing whitespace")
        fixed = "\n".join(line.rstrip() for line in fixed.split("\n"))
    if fixed and not fixed.endswith("\n"):
        problems.append("no final newline")
        fixed += "\n"
    syntax_error = _syntax_error(path, fixed)
    if syntax_error is not None:
        problems.append(f"does not parse: {syntax_error}")
    elif path.suffix == ".ipynb":
        notebook = json.loads(fixed)
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in code_cells):
            problems.append("notebook has outputs")
            for cell in code_cells:
                cell["outputs"] = []
                cell["execution_count"] = None
            fixed = json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if path.suffix == ".py":
        with_header = _with_header(fixed, python_style(rel).header)
        if with_header is not None:
            problems.append("missing license header")
            fixed = with_header
    if fix and fixed != text:
        path.write_text(fixed, encoding="utf-8")
    return problems


def _syntax_error(path: Path, text: str) -> str | None:
    try:
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in (".yaml", ".yml"):
            yaml.load(text, Loader=_ConfigLoader)
        elif path.suffix in (".json", ".ipynb"):
            json.loads(text)
    except (SyntaxError, ValueError, yaml.YAMLError) as error:  # TOML and JSON errors are ValueErrors
        return f"{type(error).__name__}: {error}".splitlines()[0]
    return None


def _with_header(text: str, header: Sequence[str]) -> str | None:
    """``text`` with the license header inserted after any shebang, or None if it has the header.

    The header counts as present when it appears in the leading comment block.
    """
    lines = text.split("\n")
    shebang = lines.pop(0) if text.startswith("#!") else None
    comments = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            if stripped:
                break
            continue
        comments.append(stripped[2:] if stripped.startswith("# ") else stripped[1:])
    if "\n".join(header) in "\n".join(comments):
        return None
    block = "\n".join(f"# {line}" if line else "#" for line in header) + "\n"
    body = "\n".join(lines)
    with_header = f"{block}\n{body}" if body else block
    return with_header if shebang is None else f"{shebang}\n{with_header}"


def _tool_output(command: Sequence[str], repo_root: Path) -> str | None:
    """None when ``command`` succeeds, else its combined output."""
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    return None if result.returncode == 0 else (result.stdout + result.stderr).strip()


def python_tool_problems(paths: Sequence[str], fix: bool, repo_root: Path) -> list[str]:
    """Ruff over every file, black once per library style, and pyrefly over the project."""
    if not paths:
        return []
    problems = []
    ruff = _tool_output([*RUFF, *(("--fix", "--exit-non-zero-on-fix") if fix else ()), *paths], repo_root)
    if ruff is not None:
        problems.append(f"ruff:\n{ruff}")
    for style in PYTHON_STYLES:
        members = [path for path in paths if python_style(path) == style]
        if not members:
            continue
        config = ("--config", style.black_config)
        black = _tool_output([*BLACK, "--check", *config, *members], repo_root)
        if black is not None:
            problems.append(f"black ({style.black_config}):\n{black}")
            if fix:
                _tool_output([*BLACK, *config, *members], repo_root)
    pyrefly = _tool_output(PYREFLY, repo_root)
    if pyrefly is not None:
        problems.append(f"pyrefly:\n{pyrefly}")
    return problems


def lint(paths: Sequence[str], *, fix: bool, repo_root: Path = REPO_ROOT) -> list[str]:
    """Problems in ``paths`` (repository-relative). Missing and excluded paths are skipped."""
    files = sorted({path for path in paths if (repo_root / path).is_file() and not is_excluded(path)})
    problems = [f"{rel}: {problem}" for rel in files for problem in file_problems(repo_root, rel, fix)]
    return problems + python_tool_problems([rel for rel in files if rel.endswith(".py")], fix, repo_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint and format Poorin's files.")
    parser.add_argument("--fix", action="store_true", help="Fix what can be fixed; the run still fails.")
    parser.add_argument("--all-files", action="store_true", help="Lint every tracked file.")
    parser.add_argument("paths", nargs="*", help="Files to lint, relative to the repository root.")
    args = parser.parse_args(argv)
    if args.all_files:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        paths = tracked.stdout.splitlines()
    else:
        paths = args.paths or changed_files(REPO_ROOT)
    problems = lint(paths, fix=args.fix)
    for problem in problems:
        print(problem)
    print("FAILED" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_lint.py -n 0 -q`
Expected: `16 passed` (13 test functions; the parse test has 4 cases).

- [ ] **Step 5: Prove the switch reformats nothing**

```bash
uv run --no-sync python -m tools.lint --all-files
```

Expected: `FAILED` with exactly two problems, both root-level files that Marin's patterns never
reached: `.pyrefly-baseline.json: no final newline` and `CLAUDE.md: no final newline`. Any ruff,
black, or pyrefly problem means a configuration does not match Marin's; fix the routing, not the
files.

```bash
uv run --no-sync python -m tools.lint --all-files --fix
uv run --no-sync python -m tools.lint --all-files
git status --short
```

Expected: the second run prints `OK`; `git status` shows only `.pyrefly-baseline.json`,
`CLAUDE.md`, and this task's new and modified files.

- [ ] **Step 6: Switch `prune_check`'s lint step**

In `tools/prune_check.py`, delete `lint_changed_files`, and in `main` replace the
`lint_changed_files()` call with `_run([sys.executable, "-m", "tools.lint"])`. Replace docstring
item 3 with:

```
3. ``tools.lint`` passes on the uncommitted change.
```

Run: `grep -n "infra" tools/*.py`
Expected: no output.

- [ ] **Step 7: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Lint with tools/lint.py"`
Expected: exit 0; one new metrics row.

- [ ] **Step 8: Commit**

```bash
git add tools tests/tools CLAUDE.md .pyrefly-baseline.json docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Replace infra/pre-commit.py with tools/lint.py

infra/pre-commit.py, which P1 deletes, also ran an agent lint review,
checked skill metadata, and checked pre-commit commands quoted in
Markdown. tools/lint.py keeps the checks Poorin needs at the same
versions: ruff 0.14.3, black 25.9.0 with each library's configuration,
pyrefly, license headers, file size, Python syntax, merge-conflict
markers, TOML, YAML, and JSON validity, trailing whitespace, a final
newline, and cleared notebooks. A full run over every tracked file
changes no Python file.

Marin matched its patterns with fnmatch, so "**/*" never matched a
file at the repository root. Those files are now checked, and the two
that lacked a final newline get one.
EOF
```

---

### Task 3: Type-check with ty instead of pyrefly

**Files:**
- Modify: `pyproject.toml` (the pyrefly block, from `# Pyrefly type checker configuration` up to
  `[tool.hatch.metadata]`, becomes a `[tool.ty]` block)
- Modify: `tools/lint.py`, `tools/prune_check.py`
- Modify: the Python files that carry `# pyrefly: ignore` comments (97 in all, 19 on their own
  line) and the files that get `# ty: ignore[rule]` markers
- Delete: `.pyrefly-baseline.json`

**Interfaces:**
- Consumes: `tools.lint.python_tool_problems` from Task 2.
- Produces: `tools.lint.TY: tuple[str, ...] = ("uvx", "ty@0.0.84", "check")`, which
  `tools/prune_check.py` runs in place of `pyrefly check`.

Measured before planning, with pyrefly's disabled categories mirrored: ty reports 62 errors in
code that survives the prune, 24 in code P7 deletes, 47 unresolved optional imports, and 232
warnings (161 of them unused `# type: ignore` comments). Warnings do not fail ty.

- [ ] **Step 1: Replace pyrefly's configuration with ty's**

In `pyproject.toml`, replace everything from the line `# Pyrefly type checker configuration` up
to (not including) `[tool.hatch.metadata]` with:

```toml
# ty type checker configuration (D49). ty runs through uvx at the version pinned in
# tools/lint.py. It checks the libraries that survive the prune and tools/; Iris, Ducky, and
# Verifyit are deleted in P2 to P5 and stay unchecked.
[tool.ty.environment]
python = ".venv"
python-version = "3.12"

[tool.ty.src]
include = [
    "lib/marin/src",
    "lib/levanter/src",
    "lib/haliax/src",
    "lib/fray/src",
    "lib/rigging/src",
    "lib/zephyr/src",
    "lib/finestore/src",
    "tools",
]

[tool.ty.rules]
# Mostly third-party stub noise (JAX, Equinox, NumPy arrays); pyrefly's configuration turned
# off the same categories (bad-argument-type, missing-attribute, no-matching-overload).
invalid-argument-type = "ignore"
unresolved-attribute = "ignore"
no-matching-overload = "ignore"

[tool.ty.analysis]
# Optional dependencies that the GPU environment does not install.
allowed-unresolved-imports = [
    "cuda", "cuda.**",
    "cutlass", "cutlass.**",
    "harbor", "harbor.**",
    "jaxlib.**",
    "jaxpp", "jaxpp.**",
    "librosa", "librosa.**",
    "lm_eval", "lm_eval.**",
    "resiliparse", "resiliparse.**",
    "soundfile", "soundfile.**",
    "tensorboardX", "tensorboardX.**",
    "trackio", "trackio.**",
    "transformer_engine", "transformer_engine.**",
    "xprof", "xprof.**",
]
```

Then `git rm -q .pyrefly-baseline.json`.

Run: `uvx ty@0.0.84 check 2>&1 | tail -3`
Expected: ty reads the configuration (no unknown-key error) and reports about 86 errors, none of
them `unresolved-import`. If ty rejects a key, look up the 0.0.84 name with `uvx ty@0.0.84 check
--help` and record a ruling.

- [ ] **Step 2: Remove the pyrefly comments**

```bash
uv run --no-sync python - <<'EOF'
import re
import subprocess

files = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True, check=True).stdout.split()
own_line = re.compile(r"^\s*# pyrefly: ignore.*\n", re.MULTILINE)
trailing = re.compile(r"\s*# pyrefly: ignore(\[[^\]]*\])?")
changed = 0
for path in files:
    text = open(path, encoding="utf-8").read()
    new = trailing.sub("", own_line.sub("", text))
    if new != text:
        open(path, "w", encoding="utf-8").write(new)
        changed += 1
print(changed, "files changed")
EOF
git grep -n "pyrefly" -- '*.py' | head
```

Expected: about 40 files changed; `git grep` prints nothing.

- [ ] **Step 3: Mark the existing errors**

```bash
uvx ty@0.0.84 check --add-ignore
uvx ty@0.0.84 check > /dev/null 2>&1; echo "exit=$?"
git grep -c "ty: ignore" -- '*.py' | awk -F: '{s+=$2} END {print s, "markers"}'
```

Expected: the second run exits 0 with warnings only; roughly 90 to 110 markers (the 86 errors,
plus sites the removed pyrefly comments used to hide).

- [ ] **Step 4: Switch the linter and the invariant to ty**

In `tools/lint.py`, replace `PYREFLY = ("uv", "run", "--no-sync", "pyrefly", "check")` with
`TY = ("uvx", "ty@0.0.84", "check")`, and in `python_tool_problems` replace the pyrefly call with:

```python
    ty = _tool_output(TY, repo_root)
    if ty is not None:
        problems.append(f"ty:\n{ty}")
```

Update the module docstring: "pyrefly checks the project" becomes "ty checks the project", and
"the pyrefly installed in `.venv`" becomes "ty 0.0.84". Update `python_tool_problems`'s docstring
to "Ruff over every file, black once per library style, and ty over the project."

In `tools/prune_check.py`, add `from tools.lint import TY`, replace
`_run(["uv", "run", "--no-sync", "pyrefly", "check"])` with `_run(list(TY))`, and change docstring
item 2 to "``ty check`` passes."

Run: `git grep -n "pyrefly" -- tools/`
Expected: no output.

- [ ] **Step 5: Lint the change**

Run: `uv run --no-sync python -m tools.lint`
Expected: `OK`. If ruff reports E501 on a line that ty's marker made too long, keep the marker
and add `noqa: E501` to the same comment (`# ty: ignore[rule]  # noqa: E501`), and record a
ruling.

- [ ] **Step 6: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Type-check with ty"`
Expected: exit 0. `pyproject.toml` changed, so every suite runs.

- [ ] **Step 7: Commit**

Set D49 to Adopted in `docs/poorin/decisions.md`, with "Landed:" naming this commit's title.

```bash
git add -A pyproject.toml .pyrefly-baseline.json tools lib experiments docs/poorin/decisions.md \
  docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Type-check with ty instead of pyrefly

ty, Astral's type checker, replaces pyrefly, pinned at 0.0.84 and run
through uvx like ruff. It checks the libraries that survive the prune and
tools/. The categories pyrefly's configuration turned off as stub noise
(argument types, attributes, overloads) are off in ty too, and pyrefly's
optional-import list becomes allowed-unresolved-imports.

The errors ty reports at the switch carry ty: ignore markers, so every
later commit is checked from a clean baseline; Pass 2 fixes the ones in
surviving code as it reads each module. The pyrefly ignore comments and
the empty pyrefly baseline are gone.
EOF
```

---

### Task 4: Point agent instructions and contributor docs at Poorin's tools

**Files:**
- Modify: `AGENTS.md` (rewrite)
- Modify: `TESTING.md`, `lib/marin/AGENTS.md`, `lib/haliax/AGENTS.md`, `lib/zephyr/AGENTS.md`,
  `lib/levanter/CONTRIBUTING.md`, `lib/haliax/CONTRIBUTING.md`,
  `docs/dev-guide/contributing.md`, `docs/dev-guide/coding-standards.md`,
  `docs/dev-guide/building-docs.md`, `docs/tutorials/add-chat-dataset.md`

**Interfaces:**
- Consumes: the CLIs from Tasks 1 to 3.
- Produces: `AGENTS.md` that later tasks and sessions follow.

- [ ] **Step 1: Rewrite `AGENTS.md`**

Write `AGENTS.md` with the content below. The sections from "Code Style" through "LLM-Generated
Code Pitfalls", and "Testing", are copied unchanged from the current file. It drops Marin's
operational guides, Pulumi, Slack and GitHub request handling, Echo, PR labels, the `🤖`
comment prefix, access-control commit rules, grant rules, and the `commit` skill.

````markdown
# Agent Guidelines for Poorin

Poorin is a hard fork of [Marin](https://github.com/marin-community/marin) for one researcher
training LLMs, video world models, and robot policies (VLAs and WAMs) on a local RTX 5090, with
occasional cloud runs. Upstream Marin is a reference only; nothing is merged from it. The tag
`upstream-baseline` marks the tree the fork started from.

Read `docs/poorin/` first: `specs/` holds the approved designs, `plans/` the implementation
plans, `decisions.md` why each change was made, and `prune-metrics.csv` one row of
measurements per prune commit.

Library manuals:

- `lib/levanter/AGENTS.md` — Levanter (JAX training library)
- `lib/marin/AGENTS.md` — Marin (experiment executor and pipelines)
- `lib/zephyr/AGENTS.md` — Zephyr (dataset processing)
- `lib/fray/AGENTS.md` — Fray (job dispatch)
- `lib/haliax/AGENTS.md` — Haliax (named tensors)

## Workflow Playbooks

Skills are task-focused playbooks in `.agents/skills/` (also `.claude/skills/`). Before starting
a non-trivial task, check whether a skill matches; if one does, follow it.

## Development

```bash
# Lint and format the uncommitted change (--all-files for the whole tree)
uv run --no-sync python -m tools.lint --fix

# Type checking (tools.lint also runs it when Python files change)
uvx ty@0.0.84 check

# Unit tests affected by the uncommitted change
uv run --no-sync python -m tools.affected_tests

# During the prune: every check, both training roots, and one metrics row
uv run --no-sync python -m tools.prune_check --phase P<n> --step "<short label>"
```

- Run every project command with `uv run --no-sync`. A syncing `uv run` can replace the CUDA
  build of JAX with the CPU build. Restore the GPU environment with
  `uv sync --locked --all-packages --extra=gpu`.
- Unit tests run on CPU: `tools.affected_tests` sets `JAX_PLATFORMS=cpu`. Run one test file
  in-process with `-n 0`; run GPU tests the same way, without `JAX_PLATFORMS=cpu`.
- Do not pass a partial marker expression such as `-m "not slow"`: it replaces the default
  expression and selects cluster and Docker tests. `tools.affected_tests` passes a complete one.
- Python >= 3.12.
- Dependencies may be upgraded freely. Each dependency change is its own commit, and its
  `uv.lock` diff is checked for versions that moved without being asked.
- Avoid large transfers between cloud regions or from the open internet; storage egress costs
  money.

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- Commit only when asked. Never push unless asked.
- Commit titles are imperative, at most 72 characters, with a `[scope]` prefix such as
  `[poorin]`, `[grug]`, or `[levanter]`. The body states what changed and why. Follow
  `writing-style` for commit messages, documentation, and blog prose.
- NEVER credit yourself in commit messages: no `Co-Authored-By` trailer, no "Generated with …"
  line, no emoji attribution, even if a tool default suggests one.

## Code Style

- All imports at the top of the file. No local imports except to break circular dependencies or guard optional deps. No `TYPE_CHECKING` guards — fix cycles structurally via protocols.
- Prefer top-level functions over classes when code does not mutate shared state. Reduce deep inheritance hierarchies.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings. Skip docstrings on trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse. Start simple; abstract only under real pressure.
- Delete dead code: unused parameters, stale options, old experiments.
- Top-level constants for magic strings/numbers.
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle.

## Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed in this codebase). No abbreviations like `exe` — use `exec` or full words.

## Types & Data Structures

- Dataclass/namedtuple over raw dicts. `StrEnum` over string keys.
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.

## Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass.

## API Design

- Accept only what's necessary. Replace boolean flags with meaningful parameters (e.g., `num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior (`NativeVllm` / `DockerVllm`, not `Vllm(docker=True)`).
- Normalize inputs to a standard format once at the boundary, not throughout.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.
- Assert liberally; prefer `raise ValueError` over silent fallbacks.

## Documentation

- Keep MkDocs content in sync with code. Use Markdown and mkdocs-style links.
- Write docs that stand alone without conversational context.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add compatibility shims if the user explicitly requests it.

## Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

## LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:
- Over-protective try/except and defensive None checks
- Tautological and guard-mirroring tests (type exists, constant has value,
  nonpositive scalar raises without a regression or public compatibility
  contract)
- Verbose/redundant docstrings and `__all__` in `__init__.py`
- Boolean dispatch instead of separate classes
- Environment variables instead of explicit parameters

## Planning

- Produce a detailed plan, with code snippets where they clarify the implementation, for
  non-trivial changes. Resolve context from the repository and `docs/poorin/` first; ask only
  when a missing decision would materially change the implementation.
- Approved designs live in `docs/poorin/specs/`, implementation plans in `docs/poorin/plans/`.
- Record each decision that chooses between options, changes scope, or departs from a plan in
  `docs/poorin/decisions.md`, in the same commit as the change.
- In answer mode, investigate and reply directly. Do not manufacture a plan or repository
  artifact.

## Code Reuse

Before writing any utility function, helper, or data structure:
1. Search the codebase for existing implementations
2. Check the libraries: `lib/marin/src/marin/`, `lib/levanter/src/levanter/`, `lib/rigging/src/rigging/`
3. Check `pyproject.toml` for available third-party packages before adding new ones

If a suitable implementation exists, use it. Do not create parallel implementations.

Dependency direction: `haliax` → {`levanter`, `zephyr`} → `marin`. Each layer may only import
from layers to its left. Never introduce reverse dependencies (e.g., levanter importing from
marin).

## Testing

Read `TESTING.md` before writing or reviewing tests. It is the root testing
policy for behavior-focused tests, slop-test rejection, mocks/fakes, timing,
numerical tolerances, and pytest style.

Before touching tests under `lib/*`, also read the nearest module `AGENTS.md`
and any module `TESTING.md` it references. Module docs define local commands,
markers, fakes, mocks, optional dependencies, and integration-test boundaries.

Always fix tests you broke. Do not relax tolerances or hack around failures.
````

Check that the unchanged sections match the old file exactly:
`diff <(git show HEAD:AGENTS.md | sed -n '/^## Code Style/,/^## Planning/p' | sed '$d') <(sed -n '/^## Code Style/,/^## Planning/p' AGENTS.md | sed '$d')`
→ the only difference is the deleted `## Agent Artifacts` section (Echo and ops-log rules).

- [ ] **Step 2: Update `TESTING.md`**

1. In the first paragraph, replace `This is Marin's project-wide testing policy.` with
   `This is the project-wide testing policy.`
2. Replace the "Running Tests Locally" opening and command block:

````markdown
Run a narrow test while editing, then run all safe tests affected by the uncommitted change:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest <test path> -n 0
uv run --no-sync python -m tools.affected_tests
```
````

3. Replace the whole "## Adding a New Library" section (its two paragraphs) with:

```markdown
## Adding a New Library

A new `lib/<name>/` package gets its own test suite when it is a workspace member in the root
`pyproject.toml` and has a `lib/<name>/tests` directory. `tools/affected_tests.py` selects its
tests from the import graph; nothing else needs registering. A change to the root
`pyproject.toml` or `uv.lock` runs every suite.
```

- [ ] **Step 3: Update the library manuals**

`lib/marin/AGENTS.md`, the "## Development" code block becomes:

```bash
# Root suite (tests/ and experiments/), on CPU
JAX_PLATFORMS=cpu uv run --no-sync pytest tests experiments

# Lint
uv run --no-sync python -m tools.lint --all-files --fix
```

`lib/haliax/AGENTS.md`, replace the three bullets starting "**Formatting and Linting**",
"**Typing**" (two lines), and "**Run `./infra/pre-commit.py --all-files`**" with:

```markdown
* **Formatting and Linting**: run `uv run --no-sync python -m tools.lint` from the repository root; it runs the linter, the formatter, the license-header check, and the type checker.
* **Typing**: the code base uses `ty` for static type checking; its configuration is in the root `pyproject.toml`.
* **Run `uv run --no-sync python -m tools.lint --all-files`** before committing.
```

`lib/zephyr/AGENTS.md`, under "## Key Docs": change `(also used by skills: \`debug\`,
\`babysit-zephyr\`)` to `(also used by the \`debug\` skill)`, and delete the three-line "Echo —"
bullet.

- [ ] **Step 4: Update the contributor guides**

`lib/levanter/CONTRIBUTING.md` and `lib/haliax/CONTRIBUTING.md`, in both files:

| Old | New |
|---|---|
| `` `infra/pre-commit.py` checks: `` | `` `tools/lint.py` checks: `` |
| `./infra/pre-commit.py --fix` (inside the hook heredoc) | `uv run --no-sync python -m tools.lint --fix` |
| `    ./infra/pre-commit.py --all-files` | `    uv run --no-sync python -m tools.lint --all-files` |

`docs/dev-guide/contributing.md`:

1. Replace "then install
   the development dependencies and the Git hook that runs `infra/pre-commit.py`:" and its code
   block with:

````markdown
then install the Git hook that runs `tools/lint.py` before each commit:

```bash
cat <<'EOF' > .git/hooks/pre-commit
#!/bin/sh
set -e
cd "$(git rev-parse --show-toplevel)"
uv run --no-sync python -m tools.lint --fix
EOF
chmod +x .git/hooks/pre-commit
```
````

2. Replace the two sentences under "### Linting" with: `The Git hook configured above lints the
   uncommitted change before each commit. Lint every file with
   \`uv run --no-sync python -m tools.lint --all-files --fix\`.`
3. Replace the "### Testing" code block with:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest <relevant test paths> -n 0
uv run --no-sync python -m tools.affected_tests
```

4. In "### Opening a pull request", item 1 becomes `Run \`uv run --no-sync python -m tools.lint
   --all-files --fix\`.`, item 2 becomes `Run \`uv run --no-sync python -m tools.affected_tests\`.`,
   delete item 3 (`check_docs_source_links.py`), and renumber items 4–8 as 3–7.

`docs/dev-guide/coding-standards.md`: `License headers are managed by pre-commit.py. See below.`
becomes `License headers are checked by \`tools/lint.py\`. See below.`, and
`Enforced automatically by \`./infra/pre-commit.py --all-files --fix\`.` becomes
`Enforced automatically by \`uv run --no-sync python -m tools.lint --all-files --fix\`.`

`docs/dev-guide/building-docs.md`: delete validation step 2 (the
`infra/check_docs_source_links.py` block) and renumber step 3 as 2; contributing step 4 becomes
`4. Run \`uv run --no-sync python -m tools.lint --all-files --fix\``; delete the "Broken Links"
bullet that runs `infra/check_docs_source_links.py`.

`docs/tutorials/add-chat-dataset.md`, the command block becomes:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/datakit/download tests/datakit/test_chat.py tests/datakit/test_chat_normalize.py
uv run --no-sync python -m tools.affected_tests
uv run --no-sync python -m tools.lint --fix
```

- [ ] **Step 5: Verify no instruction points at the old tools**

```bash
grep -rnE "infra/pre-commit|pre-commit\.py|run_tests\.py|select_tests\.py|check_docs_source_links|consult-echo|write-ops-log" \
  AGENTS.md TESTING.md lib/marin/AGENTS.md lib/haliax/AGENTS.md lib/zephyr/AGENTS.md lib/levanter/AGENTS.md \
  lib/fray/AGENTS.md lib/levanter/CONTRIBUTING.md lib/haliax/CONTRIBUTING.md docs/dev-guide docs/tutorials
```

Expected: no output.

- [ ] **Step 6: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Point agent instructions at Poorin's tools"`
Expected: exit 0; `no affected tests` (only Markdown changed).

- [ ] **Step 7: Commit**

```bash
git add AGENTS.md TESTING.md lib/marin/AGENTS.md lib/haliax/AGENTS.md lib/zephyr/AGENTS.md \
  lib/levanter/CONTRIBUTING.md lib/haliax/CONTRIBUTING.md docs/dev-guide docs/tutorials/add-chat-dataset.md \
  docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Rewrite AGENTS.md for Poorin and point docs at tools/

AGENTS.md told agents to use Marin's operations: Iris and Zephyr runbooks,
Pulumi, Echo prior-work search, PR labels, comment prefixes, grant rules,
and the infra/ lint and test runners. Poorin's version keeps the code,
naming, typing, error-handling, testing, and comment rules unchanged,
replaces the commands with tools.lint, tools.affected_tests, and
tools.prune_check, and points at docs/poorin for the design.

TESTING.md, the library manuals, and the contributor guides get the same
command changes. The docs link checker lived in infra/ and is dropped.
EOF
```

---

### Task 5: Prune skills and Marin's agent notes

**Files:**
- Delete: 29 skills in `.agents/skills/` (table below), `.agents/agents/`, `.agents/docs/`,
  `.agents/logbooks/`, `.agents/ops/`, `.agents/projects/`
- Modify: `.agents/skills/debug/SKILL.md`, `.agents/skills/change-grug/SKILL.md`,
  `.agents/skills/noslop/SKILL.md`, `.agents/skills/writing-style/issues.md`,
  `.agents/skills/write-pipeline/SKILL.md`

**Interfaces:** none.

Skill verdicts:

| Skill | Verdict | Reason |
|---|---|---|
| `add-dataset` | Keep | Registers Hugging Face datasets in `experiments/datasets`, which fast-track reaches |
| `add-pallas-kernel` | Keep | Kernel work; its benchmark paths are Levanter's own `lib/levanter/scripts` |
| `change-grug` | Keep, edit | Grug is the model template; replace the deleted diff script and lint command |
| `debug` | Keep, rewrite | Drop Echo, the ops log, and the Iris runbooks |
| `noslop` | Keep, edit | Test and prose cleanup; replace the lint commands |
| `profile-training` | Keep | Profiling JAX training |
| `update-docs` | Keep | Keeping docs in sync with behavior |
| `upload-hf-model` | Keep | Publishing weights (sub-project 8) |
| `use-fsutil` | Keep | Rigging's storage CLI |
| `write-pipeline` | Keep, edit | The executor's pipeline patterns; drop the `commit` skill reference |
| `write-tests` | Keep | Test policy |
| `writing-style` | Keep, edit | Commit, doc, and blog prose; drop the `file-issue` reference |
| `ab-test-zephyr`, `babysit-zephyr` | Delete | Zephyr benchmarks and monitoring on Iris |
| `add-grant`, `review-grant`, `trace-pulumi-diff` | Delete | Marin's GCP IAM and Pulumi |
| `archive-experiments`, `organize-experiments`, `scrub-experiment-issue-tldrs` | Delete | Marin's experiment issues and report index |
| `commit`, `lint-review`, `review-pr`, `file-issue`, `fix-issue` | Delete | Marin's PR and issue process; commit rules live in `AGENTS.md` |
| `consult-echo`, `write-ops-log`, `research`, `scrub-docs-code-parity`, `scrub-reflection-self-improvement` | Delete | Echo and scheduled scrubs |
| `deploy-hero-change`, `manage-hero-run`, `run-ferries`, `triage-canary` | Delete | Production runs and ferries |
| `launch-rl` | Delete | Marin's SkyRL integration (P2) |
| `marina-applet`, `onboard-marin`, `query-finelog`, `refresh-fork`, `scan-logs`, `use-iris` | Delete | Marina, Marin onboarding, Iris telemetry, external fork pins, the deleted log scanner, Iris |

- [ ] **Step 1: Delete the skills and notes**

```bash
cd .agents/skills
git rm -r -q ab-test-zephyr add-grant archive-experiments babysit-zephyr commit consult-echo \
  deploy-hero-change file-issue fix-issue launch-rl lint-review manage-hero-run marina-applet \
  onboard-marin organize-experiments query-finelog refresh-fork research review-grant review-pr \
  run-ferries scan-logs scrub-docs-code-parity scrub-experiment-issue-tldrs \
  scrub-reflection-self-improvement trace-pulumi-diff triage-canary use-iris write-ops-log
cd ../..
git rm -r -q .agents/agents .agents/docs .agents/logbooks .agents/ops .agents/projects
ls .agents .agents/skills
```

Expected: `.agents` holds only `skills`; `.agents/skills` lists the 12 kept skills.
`.claude/agents` is now a dangling symlink to `../.agents/agents`; remove it with
`git rm -q .claude/agents`.

- [ ] **Step 2: Rewrite `.agents/skills/debug/SKILL.md`**

```markdown
---
name: debug
description: Diagnose a stated code, JAX, Levanter, Marin, Zephyr, or GPU fault or startup/performance regression; do not activate for ordinary implementation or optimization without a symptom.
---

# Debug

Keep working notes in the active task. Do not add repository debug-log files.

## GPU faults

For a fault on the local RTX 5090 (cuDNN attention, the compilation cache, attention backends,
missing MFU), read `docs/poorin/spike-0-rtx5090.md` first: it records what is known to fail on
sm_120 and the workaround for each.

## Pipeline faults

For a slow Zephyr pipeline, stragglers, data skew, or worker failures, read
`lib/zephyr/AGENTS.md`, then `lib/zephyr/OPS.md` → Diagnostic Patterns, Observability.

## Code bugs

For code bugs, reproduce the failure, identify the smallest falsifiable
hypothesis, change one cause at a time, and test the behavior that failed. Let
exceptions propagate unless added context changes the diagnosis.
```

- [ ] **Step 3: Edit the other kept skills**

`.agents/skills/change-grug/SKILL.md`, section "### 3) Record the experiment": replace the
text from "and diff link (prefer the CI-posted PR comment link; fallback to local report path)."
through the end of the `grug_dir_diff.py` code block with:

````markdown
and how to see its diff against the origin.

To review a variant against its origin:

```bash
git diff --no-index --stat experiments/grug/base experiments/grug/<variant>
git diff --no-index experiments/grug/base experiments/grug/<variant>
```
````

In "### 6) Validate", the code block becomes:

```bash
uv run --no-sync python -m tools.lint --all-files
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_grug_variant_contracts.py -n 0
```

`.agents/skills/noslop/SKILL.md`: `uv run python .agents/skills/noslop/scripts/scan_diff.py
"$BASE"` becomes `uv run --no-sync python .agents/skills/noslop/scripts/scan_diff.py "$BASE"`.
In "## Fix And Verify", replace the code block and the paragraph after it with:

````markdown
```bash
uv run --no-sync python -m tools.lint --fix
```

Do not claim a clean pass while known findings remain.
````

`.agents/skills/writing-style/issues.md`: delete the two-line "See also the
[file-issue](../file-issue/SKILL.md) skill ..." paragraph.

`.agents/skills/write-pipeline/SKILL.md`: delete the sentence `Use the \`commit\` skill if asked
to publish the result.`

- [ ] **Step 4: Verify kept skills reference no deleted skill or tool**

```bash
grep -rnE "\`(ab-test-zephyr|add-grant|archive-experiments|babysit-zephyr|commit|consult-echo|deploy-hero-change|file-issue|fix-issue|launch-rl|lint-review|manage-hero-run|marina-applet|onboard-marin|organize-experiments|query-finelog|refresh-fork|research|review-grant|review-pr|run-ferries|scan-logs|trace-pulumi-diff|triage-canary|use-iris|write-ops-log)\`|\.\./(file-issue|commit|consult-echo)/|infra/pre-commit|scripts/ci/" .agents/skills
```

Expected: no output.

- [ ] **Step 5: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Prune skills and Marin's agent notes"`
Expected: exit 0; `no affected tests` (`.agents/` is outside every source tree and test
directory).

- [ ] **Step 6: Commit**

```bash
git add -A .agents .claude docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Delete Marin's organization skills and agent notes

Twenty-nine skills served Marin's organization or systems Poorin
deletes: Iris, Echo, Marina, Pulumi grants, hero runs, ferries, canary
triage, scheduled scrubs, SkyRL launches, and Marin's PR and issue
process. .agents/projects, docs, logbooks, ops, and the ops-steward
agent held Marin's design notes and incident history for the same
systems. All remain at upstream-baseline.

Twelve general skills stay. debug, change-grug, noslop, writing-style,
and write-pipeline lose their references to deleted skills and tools;
debug now points GPU faults at the spike 0 note.
EOF
```

---

### Task 6: Delete Marin's ferries

**Files:**
- Delete: `experiments/ferries/`, `docs/experiments/daily-ferry-log.md`
- Modify: `mkdocs.yml`, `docs/recipes/add_scaling_heuristic.md`

- [ ] **Step 1: Confirm nothing that survives loads a ferry by name**

```bash
for m in canary_ferry daily datakit_ferry datakit_nemotron_ferry datakit_tier2_skewed_ferry grug_multislice_smoke; do
  uv run --no-sync python -m tools.reachability --why experiments.ferries.$m
done
grep -rn "experiments\.ferries\|experiments/ferries" --include=*.py --include=*.toml --include=*.yaml lib experiments tests tools
```

Expected: `unreachable` six times. The grep prints only files inside `experiments/ferries/` and
the docstring in `experiments/grug/moe/launch_cw_scale.py`, an unreachable experiment that P7
deletes.

- [ ] **Step 2: Delete**

```bash
git status --short --ignored experiments/ferries
git rm -r -q experiments/ferries docs/experiments/daily-ferry-log.md
rm -rf experiments/ferries
```

Expected from the status line: only `experiments/ferries/__pycache__/`.

- [ ] **Step 3: Remove the docs references**

`mkdocs.yml`: delete the nav line `      - Daily Ferry Log: experiments/daily-ferry-log.md`.

`docs/recipes/add_scaling_heuristic.md`:
- delete the bullet `- **Canary promotion** in \`experiments/ferries/canary_ferry.py\`.`;
- in "Key reference files", delete the line `experiments/ferries/canary_ferry.py            # daily canary`
  and the blank line before it;
- delete the section "### 6) Promote to canary" (its heading and paragraph);
- in "Definition of Done", delete `- Canary promoted in \`experiments/ferries/canary_ferry.py\`.`

Run: `grep -rn "ferr" docs/recipes/add_scaling_heuristic.md mkdocs.yml`
Expected: no output.

- [ ] **Step 4: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Delete Marin's ferries"`
Expected: exit 0; `first_party_lines` drops by the size of `experiments/ferries`;
`reachable_lines` unchanged.

- [ ] **Step 5: Commit**

```bash
git add -A experiments/ferries docs/experiments mkdocs.yml docs/recipes/add_scaling_heuristic.md \
  docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Delete Marin's ferries

Ferries were Marin's scheduled canary and data-pipeline runs on its
clusters, with a daily log and CI checks of their metrics. No root
reaches them, three of them import infra/ci, and the one Poorin
launcher they shaped, grug's base, became local-first in P0. The
scaling-heuristic recipe no longer ends by promoting the result to
the canary.
EOF
```

---

### Task 7: Delete `infra/`

**Files:**
- Delete: `infra/`, `tests/infra/`, `tests/codehealth/`, `Makefile`,
  `docs/design/marin-service-architecture.md`, `docs/dev-guide/guidelines-internal.md`,
  `docs/ops/`, `docs/tutorials/storage-bucket.md`
- Modify: `pyproject.toml`, `uv.lock`, `mkdocs.yml`, `.gitignore`, `docs/index.md`,
  `docs/tutorials/installation.md`, `lib/levanter/src/levanter/tracker/telemetry.py`

`infra/` held: the Pulumi projects for Marin's buckets, IAM, and services (`pulumi`, `buckets`,
`cloudsql`); the `marin-deploy` CLI and deployments for Ducky and Finelog; Grafana dashboards and
alerts; Marina, the web host for Echo and the evaluation dashboard; the code-review workbench
(`codehealth`), the agent lint catalog (`lint`, `linter.py`), and agent runner profiles (`loom`);
TPU CI runners; the xprof gateway; the CI test selector and runner; the lint entry point; and
small scripts for the container registry, wheel deployment, docs link checking, and MkDocs.

- [ ] **Step 1: Confirm nothing that survives imports `infra`**

```bash
grep -rnE "^\s*(from|import) (infra|marin_deploy|iac|marina)\b" --include=*.py lib experiments tests tools \
  | grep -v -E "^tests/(infra|codehealth)/"
```

Expected: no output.

- [ ] **Step 2: Record the lock and delete**

```bash
mkdir -p .superpowers/sdd/2026-10-05-p1-fleet-operations
git show HEAD:uv.lock > .superpowers/sdd/2026-10-05-p1-fleet-operations/uv.lock.before
git status --short --ignored infra | grep -v __pycache__
git rm -r -q infra tests/infra tests/codehealth Makefile \
  docs/design/marin-service-architecture.md docs/dev-guide/guidelines-internal.md docs/ops \
  docs/tutorials/storage-bucket.md
rm -rf infra tests/infra tests/codehealth
```

Expected from the status line: no output (only `__pycache__` directories are untracked). If
anything else is listed, stop and report it.

- [ ] **Step 3: Edit `pyproject.toml`**

1. In `[project] dependencies`, delete `"marin-deploy",` and the three lines
   `# marin-marina is the multi-app web kernel (infra/marina), ...`, `# \`uv run marina dev\`
   serves ...`, `"marin-marina",`.
2. In `[tool.uv.workspace] members`, delete `"infra/pulumi",`, `"infra/buckets",`,
   `"infra/deploy",`, `"infra/marina",`.
3. In `[tool.uv.sources]`, delete `marin-deploy = { workspace = true }`,
   `marin-marina = { workspace = true }`, `marin-iac = { workspace = true }`.
4. In `[tool.ruff.lint.isort]`, delete the three comment lines about `infra/tpu-ci` and change
   `known-third-party = ["wandb", "config", "marin"]` to `known-third-party = ["wandb", "marin"]`.
The type checker's configuration needs no edit: Task 3 replaced pyrefly's, and ty's
`[tool.ty.src] include` never listed `infra/`.

- [ ] **Step 4: Edit the other files**

- `mkdocs.yml`: delete the `hooks:` key and its `- infra/mkdocs_hooks.py` item (the hook only
  lowered a mkdocstrings inventory warning to info); delete the nav lines
  `      - Prepare a Storage Bucket: tutorials/storage-bucket.md` and
  `      - Internal Onboarding: dev-guide/guidelines-internal.md`.
- `docs/index.md`: delete `- [Internal Onboarding](dev-guide/guidelines-internal.md)`.
- `docs/tutorials/installation.md`: delete the section `## Rust Crates (dupekit)`, from its
  heading up to (not including) `## Trying it Out`. It documented the `make rust-*` targets,
  which wrapped `scripts/rust_mode.py`; Dupekit and Finelog's Rust server stay pre-built wheels.
- `.gitignore`: delete `# Echo's judged search corpus is source, not a run artifact.` and
  `!infra/marina/apps/echo/benchmarks/*.jsonl`.
- `lib/levanter/src/levanter/tracker/telemetry.py`: in the comment above
  `_PHASE_HEARTBEAT_SECONDS`, delete the sentence `The reader's window lives in
  infra/grafana/src/training_stalls.py.`

- [ ] **Step 5: Relock, check the lock diff, and resync**

```bash
uv lock
uv run --no-sync python - <<'EOF'
import tomllib
from collections import Counter

def packages(path):
    with open(path, "rb") as f:
        return Counter((p["name"], p.get("version")) for p in tomllib.load(f)["package"])

before = packages(".superpowers/sdd/2026-10-05-p1-fleet-operations/uv.lock.before")
after = packages("uv.lock")
print("removed:", sorted((before - after).elements()))
print("added:", sorted((after - before).elements()))
EOF
uv sync --locked --all-packages --extra=gpu
uv run --no-sync python -c "import jax; print(jax.default_backend())"
```

Expected: `removed` lists `marin-buckets`, `marin-deploy`, `marin-iac`, `marin-marina`, and
packages only they needed (for example Pulumi's); `added` is `[]`; the backend is `gpu`. A
version change in `added`/`removed` pairs for the same name is a moved version: stop and report.

- [ ] **Step 6: Search for leftover references**

The system `grep` is ugrep, which prints paths without a leading `./`, so the filters below
accept both forms.

```bash
grep -rnE "infra/|\binfra\." --include=*.py --include=*.toml --include=*.yml --include=*.yaml --include=*.md \
  . 2>/dev/null | grep -v -E "^(\./)?(\.venv|scripts|docs/poorin|lib/(iris|finelog|ducky|dupekit|verifyit|shellbox|taskcompendium)|\.superpowers|\.remember)/" \
  | grep -v "levanter\.infra\|levanter/infra"
```

Expected, and left for later phases:
- `experiments/evaluation/README.md` (Evaldash's results database; P3 rewrites the evaluation
  launchers), `experiments/post_training/`, `experiments/rl_data_reviews/`, and
  `docs/references/rl-data-atlas.md` (P2);
- `.github/actions/pulumi-preview/` (Task 10) and `lib/rigging/src/rigging/filesystem/cluster_config.py`
  plus `config/coreweave.yaml` (Task 12);
- `lib/levanter/docs/` and `lib/levanter/config/` lines for `infra/launch.py`,
  `infra/babysit-tpu-vm`, and `infra/spin-up-vm.sh`: Levanter's own scripts from before it moved
  into Marin, not this directory (P7).

Fix anything else.

- [ ] **Step 7: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Delete infra/"`
Expected: exit 0. `pyproject.toml` and `uv.lock` changed, so every suite runs (about 12
minutes). `venv_size` drops.

- [ ] **Step 8: Commit**

```bash
git add -A infra tests/infra tests/codehealth Makefile docs mkdocs.yml \
  pyproject.toml uv.lock .gitignore lib/levanter/src/levanter/tracker/telemetry.py
git commit -F - <<'EOF'
[poorin] Delete infra/

infra/ ran Marin's fleet: Pulumi projects for buckets, IAM, and
services; the marin-deploy CLI and Ducky and Finelog deployments;
Grafana dashboards and alerts; Marina, the web host for Echo and the
evaluation dashboard; the code-review workbench, agent lint catalog,
and agent runner profiles; TPU CI runners; the xprof gateway; and the
CI test selector, test runner, and lint entry point. Poorin runs on
one workstation, and tools/ now holds the test selector and linter.

Its four workspace packages leave the workspace and the lock. The
tests for infra/ and codehealth, the Makefile (every target wrapped a
deleted tool or set up Marin's GCP project), the MkDocs hook, and the
pages describing Marin's services,
alerts, internal onboarding, and bucket setup go with it. The tests in
tests/infra/ci/test_package_release.py had failed since P0 deleted the
workflows they read.
EOF
```

---

### Task 8: Drop dependencies nothing uses

**Files:**
- Modify: `pyproject.toml`, `lib/marin/pyproject.toml`, `uv.lock`

The root project and `marin-core`'s `dev` group carry Codehealth's database clients.
`marin-core`'s `lint` group carries black and pyrefly, which no tool runs from the venv any more:
`tools/lint.py` runs black through `uvx`, and Task 3 replaced pyrefly with ty.

- [ ] **Step 1: Confirm nothing imports them**

```bash
grep -rnE "^\s*(from|import) (sqlalchemy|pg8000|google\.cloud\.sql|black|pyrefly)\b" --include=*.py lib experiments tests tools \
  | grep -v -E "^lib/(iris|ducky|verifyit|shellbox|taskcompendium)/"
```

Expected: no output.

- [ ] **Step 2: Edit and relock**

- `pyproject.toml`, `[project] dependencies`: delete the comment
  `# Codehealth's weekly review workbench persists into the shared metadata database.`,
  `"cloud-sql-python-connector[pg8000]>=1.9",`, and `"sqlalchemy>=2",`.
- `lib/marin/pyproject.toml`: delete `"cloud-sql-python-connector[pg8000]>=1.9",` from the `dev`
  group, and `"black>=26.3.1",` and `"pyrefly>=1.0.0,<1.1.0",` from the `lint` group.

```bash
git show HEAD:uv.lock > .superpowers/sdd/2026-10-05-p1-fleet-operations/uv.lock.before
uv lock
```

Run the lock comparison script from Task 7 Step 5, then
`uv sync --locked --all-packages --extra=gpu` and the backend check.

Expected: `removed` lists `cloud-sql-python-connector`, `pg8000`, `black`, `pyrefly`, and
packages only they needed, plus `sqlalchemy` only if no other package requires it; `added` is
`[]`; the backend is `gpu`.

- [ ] **Step 3: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Drop dependencies nothing uses"`
Expected: exit 0; every suite runs.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml lib/marin/pyproject.toml uv.lock docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Drop dependencies nothing uses

The root project and marin-core's dev group depended on the Cloud SQL
connector and SQLAlchemy so infra/codehealth could write review statistics
to Marin's metadata database. marin-core's lint group installed black and
pyrefly, but tools/lint.py runs black through uvx and ty replaced pyrefly.
Nothing that remains imports any of them.
EOF
```

---

### Task 9: Delete `scripts/`

**Files:**
- Delete: `scripts/`, `tests/ci/`, `tests/iris/test_rollout_controllers.py`,
  `tests/test_coreweave_storage_usage.py`, `tests/test_validate_canary_metrics.py`,
  `tests/test_grug_dir_diff.py`, `tests/test_grug_variant_diff_ci.py`,
  `docs/dev-guide/releasing.md`, `docs/tutorials/publish-analysis-site.md`,
  `docs/reports/agent-moe-experiments.md`, `docs/reports/data/`
- Modify: `pyproject.toml`, `.gitignore`, `mkdocs.yml`, `docs/reports/index.md`,
  `docs/tutorials/cloud-gpu.md`, `experiments/grug/README.md`

`scripts/` held: CI helpers (dependency updates, canary metric validation, package releases, grug
variant diffs, Iris monitoring), the cost manager, Iris cluster and rollout scripts, storage and
publishing operations, report rendering, ferry analysis, speedrun and training launch helpers, the
Rust build-mode switch, and one-off debugging scripts.

- [ ] **Step 1: Confirm only the tests being deleted import `scripts`**

```bash
grep -rnE "^\s*(from|import) scripts\b" --include=*.py lib experiments tests tools
```

Expected: only the six test files listed above.

- [ ] **Step 2: Delete**

```bash
git status --short --ignored scripts | grep -v __pycache__
git rm -r -q scripts tests/ci tests/iris/test_rollout_controllers.py tests/test_coreweave_storage_usage.py \
  tests/test_validate_canary_metrics.py tests/test_grug_dir_diff.py tests/test_grug_variant_diff_ci.py \
  docs/dev-guide/releasing.md docs/tutorials/publish-analysis-site.md docs/reports/agent-moe-experiments.md \
  docs/reports/data
rm -rf scripts tests/ci tests/iris
```

Expected from the status line: no output.

- [ ] **Step 3: Edit `pyproject.toml`**

1. In `[project] dependencies`, the comment lines
   `# marin-iris, which depends on it explicitly. \`scripts/rust_mode.py dev\` swaps` and
   `# marin-finelog-server to a local source build.` become one line,
   `# marin-iris, which depends on it explicitly.`; the lines
   `# as marin-dupekit's runtime dependency. \`scripts/rust_mode.py dev\` swaps` and
   `# marin-dupekit-native to a local source build.` become
   `# as marin-dupekit's runtime dependency.`
2. In `[tool.uv.sources]`, delete `# ### BEGIN RUST-DEV SOURCES ###` and
   `# ### END RUST-DEV SOURCES ###`.
3. In `[tool.black]`, delete `# Note :: Grow more strict over time!` and the
   `extend-exclude = """ ( scripts/ ) """` block.
4. In `[tool.ruff]`, delete the four comment lines starting `# Note :: Grow more strict over
   time!` and the `extend-exclude = ["scripts/"]` and `extend-include = [...]` lines.

The type checker's configuration needs no edit: ty's `[tool.ty.src] include` (Task 3) never
listed `scripts/`.

- [ ] **Step 4: Edit the other files**

- `.gitignore`: delete `!/scripts/web/output/`.
- `mkdocs.yml`: delete the nav lines `      - Publishing an Analysis Site: tutorials/publish-analysis-site.md`,
  `      - Agent MoE Experiment Digest: reports/agent-moe-experiments.md`, and
  `      - Releasing to PyPI: dev-guide/releasing.md`.
- `docs/reports/index.md`: delete everything from the line `## Curated experiment series` up to
  (not including) `## Marin 8B Base`: both sections pointed at deleted pages.
- `docs/tutorials/cloud-gpu.md`: delete the sentences `Use \`scripts/iris/dev_gpu.py\` for short
  interactive tests. Submit a full Iris job after the code works on the reserved node.` (P5
  deletes the page with Iris).
- `experiments/grug/README.md`: replace the section `## Visual diff for template variants`, from
  its heading up to (not including) `## Common edit points`, with:

````markdown
## Diffing a variant against base

When template-copying `experiments/grug/base/` to a new variant, review the changes with git:

```bash
git diff --no-index --stat experiments/grug/base experiments/grug/<variant>
git diff --no-index experiments/grug/base experiments/grug/<variant>
```
````

- [ ] **Step 5: Search for leftover references**

```bash
grep -rnE "(^|[\` (\"'])scripts/" --include=*.py --include=*.toml --include=*.yml --include=*.yaml --include=*.md \
  . 2>/dev/null | grep -v -E "^(\./)?(\.venv|docs/poorin|lib/(iris|finelog|ducky|dupekit|verifyit|shellbox|taskcompendium)|\.superpowers|\.remember)/" \
  | grep -v -E "lib/levanter/scripts|experiments/[a-z_/]*scripts/|\.agents/skills/noslop/scripts"
```

Expected, and left alone:
- `experiments/datakit/README.md`, which describes `experiments/datakit/scripts/`;
- `tests/test_inspect_data_region.py`, whose docstring records where its helpers came from;
- `lib/levanter/pyproject.toml` and `lib/levanter/docs/`, which mean Levanter's own
  `lib/levanter/scripts/`;
- `experiments/post_training/` (P2).

Fix anything else.

- [ ] **Step 6: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Delete scripts/"`
Expected: exit 0; every suite runs because `pyproject.toml` changed.

- [ ] **Step 7: Commit**

```bash
git add -A scripts tests pyproject.toml .gitignore mkdocs.yml docs experiments/grug/README.md
git commit -F - <<'EOF'
[poorin] Delete scripts/

scripts/ held Marin's CI helpers (dependency updates, canary metric
checks, package releases, grug variant diffs, Iris monitoring), the
cost manager, Iris cluster and rollout scripts, storage and publishing
operations, report rendering, ferry analysis, the Rust build-mode
switch, and one-off debugging scripts. Nothing that survives imports
them.

Their tests, the lint and type-check configuration that excluded or
included them, and the pages about releasing to PyPI, publishing
analysis sites, and the generated agent MoE report go with them.
EOF
```

---

### Task 10: Delete `docker/`, `etc/`, and the GitHub configuration

**Files:**
- Delete: `docker/`, `etc/`, `lib/levanter/etc/`, `lib/haliax/etc/`, `.github/`,
  `docs/dev-guide/forking-policy.md`
- Modify: `tools/lint.py`, `.gitignore`, `mkdocs.yml`, `lib/marin/pyproject.toml`

- [ ] **Step 1: Confirm what reads them**

```bash
grep -rIn "docker/marin\|license_header\.txt\|forking-policy" --exclude-dir=.git --exclude-dir=.venv . \
  | grep -v -E "^(\./)?(lib/iris|docs/poorin|\.superpowers|\.remember)/"
```

Expected: `mkdocs.yml`'s nav entry for the forking policy and a comment in
`lib/marin/pyproject.toml` that cites it. `tools/lint.py` holds the three license headers inline
since Task 2, so the `license_header.txt` files have no reader. `.dockerignore` stays: Levanter's
TPU tutorial builds images that respect it.

- [ ] **Step 2: Delete and edit**

```bash
git rm -r -q docker etc lib/levanter/etc lib/haliax/etc .github docs/dev-guide/forking-policy.md
```

- `tools/lint.py`: delete `".github/**",` from `EXCLUDED`.
- `.gitignore`: delete `.github/.tmp/`.
- `mkdocs.yml`: delete the nav line `      - Forking Policy: dev-guide/forking-policy.md`.
- `lib/marin/pyproject.toml`: the comment
  `# Marin's patched ragged all-to-all PJRT plugin, promoted from marin-community/xla` /
  `# (docs/dev-guide/forking-policy.md). Non-aarch64 resolution falls back to PyPI.` becomes
  `# Marin's patched ragged all-to-all PJRT plugin, promoted from marin-community/xla.` /
  `# Non-aarch64 resolution falls back to PyPI.`

- [ ] **Step 3: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Delete docker/, etc/, and .github/"`
Expected: exit 0; `tests/tools/test_lint.py` runs because `tools/lint.py` changed.

- [ ] **Step 4: Commit**

```bash
git add -A docker etc lib/levanter/etc lib/haliax/etc .github docs/dev-guide/forking-policy.md \
  tools/lint.py .gitignore mkdocs.yml lib/marin/pyproject.toml docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Delete docker/, etc/, and the GitHub configuration

docker/ built Marin's TPU CI image and patched vLLM for it. The three
license_header.txt files were read only by infra/pre-commit.py;
tools/lint.py holds the headers. .github/ kept composite actions for
the workflows P0 deleted, CodeQL and Dependabot configuration, and
Marin's PR templates. The forking-policy page described the weekly
workflow that refreshed Marin's external fork pins.

.dockerignore stays: Levanter's TPU launcher builds images that
respect it, and sub-project 2 decides how TPU jobs are packaged.
EOF
```

---

### Task 11: Format with ruff instead of black

**Files:**
- Modify: `tools/lint.py`, `tests/tools/test_lint.py`
- Modify: `pyproject.toml`, `lib/levanter/pyproject.toml`, `lib/haliax/pyproject.toml`
- Modify: every tracked Python file `ruff format` changes (about 205 of 2,297, measured before
  Task 3's markers)
- Create: `.git-blame-ignore-revs`

**Interfaces:**
- Consumes: `tools.lint` from Tasks 2 and 3.
- Produces: `tools.lint.RUFF_FORMAT = ("uvx", "ruff@0.14.3", "format")` and
  `tools.lint.license_header(path: str) -> tuple[str, ...]`, replacing `PythonStyle` and
  `python_style`. Ruff finds each file's line length itself, so nothing routes formatter
  configuration any more.

This task makes two commits: the reformat, then `.git-blame-ignore-revs` naming it.

- [ ] **Step 1: Update the tests**

In `tests/tools/test_lint.py`, delete `test_black_uses_each_library_configuration` and remove
`python_style` from the import. The license-header routing it used stays covered by
`test_each_library_requires_its_own_license_header`. Formatting is checked on the real tree in
Step 5, since the unit tests never run external tools.

- [ ] **Step 2: Switch `tools/lint.py` to ruff format**

Replace `BLACK = ("uvx", "black@25.9.0")` with `RUFF_FORMAT = ("uvx", "ruff@0.14.3", "format")`.
Replace the `PythonStyle` dataclass, `PYTHON_STYLES`, and `python_style` with:

```python
# Levanter and Haliax keep the license headers they had as separate projects.
LICENSE_HEADERS = (
    ("lib/levanter/", LEVANTER_HEADER),
    ("lib/haliax/", HALIAX_HEADER),
    ("", MARIN_HEADER),
)


def license_header(path: str) -> tuple[str, ...]:
    """The license header a repository-relative Python path must start with."""
    return next(header for prefix, header in LICENSE_HEADERS if path.startswith(prefix))
```

In `file_problems`, `python_style(rel).header` becomes `license_header(rel)`. Replace the black
loop in `python_tool_problems` with:

```python
    formatting = _tool_output([*RUFF_FORMAT, "--check", *paths], repo_root)
    if formatting is not None:
        problems.append(f"ruff format:\n{formatting}")
        if fix:
            _tool_output([*RUFF_FORMAT, *paths], repo_root)
```

and its docstring with "Ruff's linter and formatter over every file, each file under its nearest
configuration, and ty over the project." In the module docstring, "pass ruff and black" becomes
"pass ruff's linter and formatter", and the sentence about versions becomes "Ruff runs at the
version Marin pinned (0.14.3) with each library's own configuration."

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools/test_lint.py -n 0 -q`
Expected: `15 passed`.

- [ ] **Step 3: Move the formatter configuration to ruff**

- `pyproject.toml`: delete the `[tool.black]` table (`line-length = 121`,
  `target-version = ["py312"]`, `preview = true`).
- `lib/levanter/pyproject.toml`: delete the `[tool.black]` table. Its `[tool.ruff]` already sets
  `line-length = 119`.
- `lib/haliax/pyproject.toml`: delete the `[tool.black]` and `[tool.isort]` tables (`isort`'s
  `profile = "black"` configured a tool nothing runs), and add above `[tool.ruff.lint]`:

```toml
[tool.ruff]
line-length = 119
target-version = "py312"
```

Without it, ruff would format Haliax at its default 88 columns.

- [ ] **Step 4: Reformat**

```bash
git ls-files '*.py' | xargs uvx ruff@0.14.3 format
git diff --stat | tail -1
```

Expected: ruff reports about 205 files reformatted. `ruff format` refuses to write a file whose
reformatted syntax tree differs from the original, so this changes formatting only.

- [ ] **Step 5: Check the whole tree**

```bash
uv run --no-sync python -m tools.lint --all-files
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/tools -n 0 -q
```

Expected: `OK`; all tools tests pass.

- [ ] **Step 6: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Format with ruff"`
Expected: exit 0. `pyproject.toml` changed, so every suite runs.

- [ ] **Step 7: Commit the reformat**

Set D50 to Adopted in `docs/poorin/decisions.md`, with "Landed:" naming this commit's title.

```bash
git add -A
git status --short | grep -v -E "\.py$|pyproject\.toml$|decisions\.md$|prune-metrics\.csv$"
```

Expected from the second command: no output (the commit holds only Python files, the three
manifests, the decision log, and the metrics row).

```bash
git commit -F - <<'EOF'
[poorin] Format with ruff instead of black

ruff format replaces black at the ruff version the linter already pins,
so one tool lints and formats. Each library keeps its line length: 121
for the root, 119 for Levanter and Haliax. This commit is formatting
only; ruff checks that every reformatted file parses to the same syntax
tree. black's preview style is not reproduced, which accounts for most
of the changed files.
EOF
```

- [ ] **Step 8: Skip the reformat in `git blame`**

```bash
printf '# ruff format replaces black (D50)\n%s\n' "$(git rev-parse HEAD)" > .git-blame-ignore-revs
uv run --no-sync python -m tools.prune_check --phase P1 --step "Skip the reformat in git blame"
git add .git-blame-ignore-revs docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Skip the ruff reformat in git blame

GitHub reads .git-blame-ignore-revs on its own. Locally, run
`git config blame.ignoreRevsFile .git-blame-ignore-revs` once.
EOF
```

Expected: `prune_check` exits 0 with `no affected tests`.

---

### Task 12: Make the local machine the default cluster

**Files:**
- Create: `config/local.yaml`, `conftest.py` (repository root)
- Move: `config/marin.yaml` → `config/testing/marin.yaml`, `config/coreweave.yaml` →
  `config/testing/coreweave.yaml`
- Modify: `lib/rigging/src/rigging/filesystem/cluster_config.py`, `lib/rigging/tests/conftest.py`,
  `lib/rigging/tests/test_cluster_config.py`, `lib/rigging/pyproject.toml`

**Interfaces:**
- Produces: `rigging.filesystem.cluster_config._DEFAULT_CLUSTER == "local"`; the fixture
  directory `config/testing/` with clusters `marin` and `coreweave`.

- [ ] **Step 1: Write the failing test**

In `lib/rigging/tests/test_cluster_config.py`, add `data_buckets` to the
`rigging.filesystem.cluster_config` import, add `from rigging.config_discovery import
resolve_cluster_config`, and replace `test_default_loads_marin_config_not_fallback` with:

```python
def test_default_cluster_is_the_local_config_with_no_buckets(monkeypatch):
    """No override and no MARIN_CLUSTER -> config/local.yaml: one machine, no region buckets."""
    monkeypatch.setattr(fs, "MARIN_CLUSTER_CONFIG_DIRS", ("config",))
    monkeypatch.delenv("MARIN_CLUSTER", raising=False)
    reset_data_config_cache()

    assert resolve_cluster_config("local", ("config",)).name == "local.yaml"
    assert data_config() is load_cluster_config("local")
    assert data_config().region_buckets == {}
    assert data_buckets() == {}  # Marin's buckets are fixtures, outside the default search
```

Rename `test_committed_cluster_configs_load` to `test_marin_fixture_cluster_configs_load` and
change its docstring to `"""Marin's fixture configs parse and are the expected kind."""`.

- [ ] **Step 2: Run it to verify it fails**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest lib/rigging/tests/test_cluster_config.py -n 0 -q`
Expected: FAIL in `test_default_cluster_is_the_local_config_with_no_buckets` with
`FileNotFoundError: No config file found for cluster 'local'`.

- [ ] **Step 3: Move Marin's configs and add the local one**

```bash
mkdir -p config/testing
git mv config/marin.yaml config/testing/marin.yaml
git mv config/coreweave.yaml config/testing/coreweave.yaml
```

Create `config/local.yaml`:

```yaml
# Poorin's default cluster: one machine with local disk. Outputs go under MARIN_PREFIX, or
# /tmp/marin when it is unset. No region buckets are declared, so no path is treated as a
# region-local cloud bucket. Marin's multi-region clusters live in config/testing/ as fixtures
# for the tests of region routing.
data:
  region_buckets: {}
```

- [ ] **Step 4: Change the default and its documentation in `cluster_config.py`**

- `_DEFAULT_CLUSTER = "marin"` → `_DEFAULT_CLUSTER = "local"`.
- Module docstring: `(default\n\`\`marin\`\`)` → `(default\n\`\`local\`\`)`, and delete the
  sentence `Lifecycle rules\non every configured data bucket are managed by \`\`infra/buckets\`\`.`
- The comment above `PER_USER_CLUSTER_CONFIG_DIR` becomes:

```python
# A per-user override dir for connecting to a live cluster (e.g. on a dev VM). It takes
# priority over the repository's config/ directory, so tests replace the search dirs to stay
# independent of the host.
```

- The comment above `_DEFAULT_CLUSTER` becomes:

```python
# The default cluster is config/local.yaml: one machine and no buckets, so outputs go under
# MARIN_PREFIX, or _DEFAULT_LOCAL_PREFIX when it is unset. This in-code config is the fallback
# when no config file is discoverable, e.g. an installed package outside a checkout.
```

- `data_config` docstring: `(default \`\`marin\`\`)` → `(default \`\`local\`\`)`.
- `load_cluster_config` docstring: `> \`\`marin\`\`. A` → `> \`\`local\`\`. A`, and `When the
  default \`\`marin\`\` config` → `When the default \`\`local\`\` config`.
- `_s3_bucket_from_prefix` docstring: `(the R2/CoreWeave buckets\n    with lifecycle rules managed
  by \`\`infra/buckets\`\`)` → `(the R2/CoreWeave buckets, which carry\n    \`\`tmp/ttl=Nd/\`\`
  lifecycle rules)`.
- The comment `# R2 and CoreWeave temp lives at the bucket root so the \`tmp/ttl=Nd/\`\n    #
  lifecycle prefix managed by infra/buckets applies.` → `# R2 and CoreWeave temp lives at the
  bucket root so the bucket's \`tmp/ttl=Nd/\`\n    # lifecycle rules apply.`

`lib/rigging/pyproject.toml`: in the `force-include` comment, `can still resolve
config/marin.yaml` → `can still resolve config/local.yaml`.

- [ ] **Step 5: Pin the rigging and root suites to Marin's layout**

`lib/rigging/tests/conftest.py`, replace `_hermetic_cluster_config` with:

```python
@pytest.fixture(autouse=True)
def _hermetic_cluster_config(monkeypatch):
    """Resolve clusters from Marin's multi-region configs, kept as fixtures in ``config/testing``.

    Rigging's tests exercise region routing, temp buckets, and cross-region guards, which need
    Marin's buckets; Poorin's default ``config/local.yaml`` declares none. Replacing the search
    dirs also keeps a per-user ``~/.config/marin/clusters`` override from changing results.
    """
    monkeypatch.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", ("config/testing",))
    monkeypatch.setenv("MARIN_CLUSTER", "marin")
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
    yield
    find_project_root.cache_clear()
    cluster_config.reset_data_config_cache()
```

Create `conftest.py` at the repository root (pytest loads it for the root suite only: each
library suite's `pyproject.toml` sets the conftest cut-off to its own directory):

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session setup for the repository-root test suite (``tests/`` and ``experiments/``).

The root suite was written against Marin's multi-region cluster layout. Poorin's default
cluster, ``config/local.yaml``, declares no buckets, so the suite runs against Marin's configs,
kept as fixtures in ``config/testing/``. The setting starts at configuration time because some
experiment modules resolve temp-bucket paths when they are imported, before any fixture runs.
"""

import pytest
import rigging.filesystem.cluster_config as cluster_config

MARIN_FIXTURE_CLUSTERS = "config/testing"
_marin_layout = pytest.MonkeyPatch()


def pytest_configure(config: pytest.Config) -> None:
    _marin_layout.setattr(cluster_config, "MARIN_CLUSTER_CONFIG_DIRS", (MARIN_FIXTURE_CLUSTERS,))
    _marin_layout.setenv("MARIN_CLUSTER", "marin")
    cluster_config.reset_data_config_cache()


def pytest_unconfigure(config: pytest.Config) -> None:
    _marin_layout.undo()
    cluster_config.reset_data_config_cache()
```

- [ ] **Step 6: Run the rigging and root suites**

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest lib/rigging/tests -q
JAX_PLATFORMS=cpu TZ=UTC uv run --no-sync pytest tests experiments -q
```

Expected: both pass. A probe before planning showed the rigging suite fully passing and the
root suite failing only on tests Tasks 7 and 9 delete.

- [ ] **Step 7: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Make the local machine the default cluster"`
Expected: exit 0. `rigging.filesystem.cluster_config` is imported almost everywhere, so most
suites run. Both training roots run with `MARIN_PREFIX` set, so their outputs do not move.

- [ ] **Step 8: Commit**

```bash
git add config conftest.py lib/rigging docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[rigging] Make the local machine the default cluster

With MARIN_CLUSTER unset, rigging loaded config/marin.yaml: Marin's
seven regional GCS buckets. config/local.yaml now declares one machine
and no buckets, so outputs go under MARIN_PREFIX or /tmp/marin, and no
path is mistaken for a region-local cloud bucket.

Marin's marin and coreweave configs move to config/testing/, which the
default search does not read. The rigging and root test suites, written
against Marin's multi-region layout for region routing, temp buckets,
and cross-region guards, now resolve clusters from there. The root
suite sets this at configuration time because two experiment modules
compute temp-bucket paths when they are imported.
EOF
```

---

### Task 13: Replace the kept roots with the workbench tools

**Files:**
- Modify: `tools/roots.toml`, `docs/poorin/decisions.md`

**Interfaces:**
- Consumes: `tools.reachability` (P0) and the `"tools.*"` root added in Task 1.
- Produces: the kept roots that P2 to P8 and Pass 2 prune against, as listed in
  `docs/poorin/specs/2026-10-06-workbench-and-roots-design.md`.

- [ ] **Step 1: Replace the `[kept]` table**

In `tools/roots.toml`, replace everything from `[kept]` up to (not including) `[deleted]` with:

```toml
[kept]
modules = [
    # Workbench: data preparation
    "marin.experiment.data",
    "marin.processing.tokenize.*",
    "marin.datakit.download.huggingface",
    "marin.datakit.download.nemotron_v1",
    "marin.datakit.canonical.fineweb_edu",
    "marin.datakit.normalize",
    "marin.datakit.decon",
    "marin.processing.classification.consolidate",
    "marin.processing.classification.deduplication.exact",
    "marin.processing.classification.deduplication.fuzzy_minhash",
    "marin.processing.classification.deduplication.fuzzy_dups",
    "marin.processing.classification.deduplication.verify_fuzzy_dups",
    "marin.experiment.cli",
    "marin.experiment.sweep",
    # Dataset recipes: the Delphi mix and its validation sets
    "experiments.datasets.nemotron",
    "experiments.datasets.starcoder",
    "experiments.datasets.proofpile",
    "experiments.datasets.paloma",
    "experiments.datasets.uncheatable",
    # Workbench: evaluation
    "levanter.main.eval_lm",
    "levanter.eval_harness",
    "levanter.main.perplexity_gap",
    "marin.evaluation.lm_eval",
    "marin.evaluation.records",
    "marin.evaluation.eval_stats",
    "marin.evaluation.eval_measurements",
    "marin.evaluation.metric_selection",
    "marin.evaluation.trace_labeled_eval",
    "marin.evaluation.rollouts",
    "marin.evaluation.model_config",
    "marin.evaluation.model_identity",
    "marin.evaluation.eval_env",
    "experiments.evals.task_configs",
    # Workbench: scaling laws and the Delphi recipe
    "marin.scaling_laws",
    "marin.scaling_laws.*",
    "experiments.references.completed_adamh",
    "experiments.grug.fast_track.heuristic",
    # Light post-training kit (D9)
    "levanter.main.train_dpo",
    "levanter.main.lora_lm",
    "levanter.main.lora_dpo",
    "levanter.main.sample_lm",
    # Plugin registry members: draccus finds these by name, so static imports do not reach them.
    "levanter.optim.*",
    "levanter.data.text.*",
    "levanter.tracker.tracker",
    "levanter.tracker.wandb",
    "levanter.tracker.json_logger",
    "levanter.tracker.json_file",
    "levanter.tracker.telemetry",
    "levanter.models.llama",
    # Poorin's own tooling: the prune must not delete it.
    "tools.*",
]
# No root needs to be importable only inside another tool's runtime any more.
static_only = []

```

- [ ] **Step 2: Measure, and check that the copy-worthy code stays reachable**

```bash
uv run --no-sync python -m tools.reachability
for m in levanter.kernels.pallas.fused_cross_entropy_loss.xla levanter.kernels.pallas.splash_attention \
         levanter.analysis.backward_flow levanter.data._prp levanter.utils.flop_utils \
         experiments.grug.fast_track.adamh dupekit; do
  printf '%s: ' "$m"; uv run --no-sync python -m tools.reachability --why "$m" | head -1
done
```

Expected: no root pattern fails to match; reachable lines about 12k below the previous metrics
row (the spec measured 138,099 to 126,306 before `tools.*` and P1's other changes); no module
prints `unreachable`.

- [ ] **Step 3: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Keep the workbench tools as roots"`
Expected: exit 0. The import step now imports the data, dedup, and decontamination roots, which
proves they import on this machine; `no affected tests`, since `roots.toml` is not Python.

- [ ] **Step 4: Commit**

Set D47 and D48 to Adopted in `docs/poorin/decisions.md`, with "Landed:" naming this commit's
title.

```bash
git add tools/roots.toml docs/poorin/decisions.md docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Keep the workbench tools as the prune's roots

Projects use Poorin as a workbench only (D46), so the kept roots become
the Marin utilities a solo researcher runs: tokenization, the Hugging Face
and Nemotron-CC downloaders, normalize, dedup and decontamination,
Levanter's BPB evaluation, lm-eval harness, and perplexity gap, the
scaling-law fits, the Delphi recipe and dataset recipes, and the
post-training kit (D47).

Harbor and Evalchemy, the MoE scaling ladder, the reference scaling
suite, Levanter's OpenAI server, and the tutorial sweep leave the roots
(D48); later phases delete what they kept reachable. See
docs/poorin/specs/2026-10-06-workbench-and-roots-design.md.
EOF
```

---

### Task 14: Fix stale docs from P0 and record P1 in the spec

**Files:**
- Modify: `experiments/grug/README.md`, `docs/poorin/specs/2026-10-04-core-prune-design.md`

- [ ] **Step 1: Edit**

`experiments/grug/README.md`, "## Environment variables you will likely use": replace the
`GRUG_RUN_ID` and `FERRY_DATE` bullets with:

```markdown
- The run ID is `grug-base-<version>`. `--version dev` gives each invocation a fresh timestamped version; the launcher prints it, and `--resume <version>` continues that run.
```

In "## Where outputs show up", replace `Base Grug samples every 50 steps by default; set
\`trainer.backward_flow.interval=0\` to disable.` with `Base Grug disables it by default
(\`backward_flow_interval=0\` in \`base/launch.py\`); set a positive interval to sample it.`

`docs/poorin/specs/2026-10-04-core-prune-design.md`:
- line 3: `Implementation plan: not written yet.` → `Implementation plans: one per phase in
  \`docs/poorin/plans/\`; P0 and P1 are done. Amended by
  \`2026-10-06-workbench-and-roots-design.md\`.`
- Libraries table, `dupekit` row: `Deleted in P4` → `Kept: normalize, dedup, and
  decontamination import it (amendment of 2026-10-06)`.
- Directly under `## Roots`, add: `Amended 2026-10-06: the kept roots are listed in
  [2026-10-06-workbench-and-roots-design.md](2026-10-06-workbench-and-roots-design.md), which
  replaces the "Kept roots" table below.`
- Invariant, item 4: `` `uv run pyrefly check` passes, and the lint entry point passes:
  `./infra/pre-commit.py` until P1, the replacement after P1. `` → `` The type checker passes
  (pyrefly until P1, ty after), and the lint entry point passes: `./infra/pre-commit.py` until
  P1, `tools/lint.py` after. ``
- P0 row: `(38 workflows; ...` → `(37 workflows and a README; ...`
- P1 row: append ` P1 also deleted the ferries, the GitHub configuration, the Makefile, and
  Marin's agent notes; replaced pyrefly with ty and black with ruff format (D49, D50); moved
  Marin's cluster configs to \`config/testing/\` as test fixtures; and replaced the kept roots
  (D47, D48). See \`docs/poorin/plans/2026-10-05-p1-fleet-operations.md\`.` to the Content cell.
- P4 row: `` `lib/ducky`, `lib/dupekit` `` → `` `lib/ducky` (Dupekit stays; see the amendment) ``.
- Under `### Lint replacement (P1)`, add a final paragraph: `Amended during P1: after the port,
  ty replaced pyrefly (D49) and ruff format replaced black (D50).`

- [ ] **Step 2: Run the invariant**

Run: `uv run --no-sync python -m tools.prune_check --phase P1 --step "Fix stale P0 docs"`
Expected: exit 0; `no affected tests`.

- [ ] **Step 3: Commit**

```bash
git add experiments/grug/README.md docs/poorin/specs/2026-10-04-core-prune-design.md docs/poorin/prune-metrics.csv
git commit -F - <<'EOF'
[poorin] Update the grug README and the prune spec after P1

The grug README still described GRUG_RUN_ID, FERRY_DATE, and backward
flow every 50 steps, which P0 removed. The spec's status line and
workflow count were out of date. The spec now points at the 2026-10-06
amendment for its roots, keeps Dupekit, names ty and tools/lint.py in
the invariant, and its P1 row lists what P1 did beyond the original
scope.
EOF
```

---

## After the last task

Run the final whole-branch review (superpowers:executing-plans, "Final Review"), then
superpowers:finishing-a-development-branch. Expected end state: no file under `infra/`,
`scripts/`, `docker/`, `etc/`, or `.github/`; `tools/` holds `reachability.py`,
`prune_check.py`, `affected_tests.py`, `lint.py`, and `changes.py`; fifteen P1 rows in
`docs/poorin/prune-metrics.csv`; nothing pushed.
