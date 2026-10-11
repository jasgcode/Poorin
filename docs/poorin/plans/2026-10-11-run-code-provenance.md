# Code Provenance for Experiment Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the code behind every experiment run recoverable and readable: refuse calendar builds
from uncommitted code, snapshot uncommitted code for dev runs, label each run's code in one line,
and list or show runs from the records on disk.

**Architecture:** `rigging.provenance` gains the record fields, the one-line label, the snapshot
commit, and a process-wide launch provenance. `poorin.execution` exposes the runner's schedule
(`steps_to_build`) and lets `run()` take a provenance. A new `poorin.experiment.code_check` decides
whether a run may build from the working tree; the experiment command line applies it, saves the
snapshot, and prints the `Code:` lines. Levanter's W&B tracker records the label. A new
`poorin.experiment.runs` command reads `.artifact.json` records to list runs and show one run's code.

**Tech Stack:** Python 3.12, Git (plumbing: `read-tree`, `write-tree`, `commit-tree`,
`update-ref`), Click, fsspec through `rigging.filesystem`, pytest.

**Spec:** `docs/poorin/specs/2026-10-11-run-code-provenance-design.md` (approved 2026-10-11).

## Global Constraints

- Work on the branch `feature/run-code-provenance`, cut from `main`. Never commit to `main`.
- Run every project command with `uv run --no-sync`. Never run a syncing `uv run`.
- Unit tests run on CPU: `JAX_PLATFORMS=cpu uv run --no-sync pytest <file> -n 0`.
- Before each commit: `uv run --no-sync python -m tools.lint --fix`, then the full check
  `uv run --no-sync python -m tools.prune_check --phase D107 --step "<the commit title>"`. The
  full check picks tests from the **uncommitted** change, so run it before committing and do not
  edit the tree while it runs. Commit the row it appends to `docs/poorin/prune-metrics.csv` with
  the task.
- ty must pass with warnings as errors: `uvx ty@0.0.84 check --error-on-warning` (`tools.lint` runs it).
- Lines are at most 121 characters.
- Commit titles: imperative, at most 72 characters, a `[scope]` prefix (`[rigging]`, `[poorin]`,
  `[levanter]`). The body says what changed and why. No `Co-Authored-By` trailer and no
  "Generated with" line.
- All imports at the top of the file. No `TYPE_CHECKING` guards. Top-level constants for magic
  strings and numbers. No `*_utils.py` modules.
- Tests follow `TESTING.md`: test behavior through public interfaces, use real Git repositories in
  `tmp_path` (no mocking of Git), and no tautological tests.
- The snapshot ref namespace is exactly `refs/poorin/snapshots/<full commit hash>`.
- The refusal's last line is exactly: `Commit the changes and run again, run a dev version, or pass
  --allow-uncommitted to build from a snapshot.`
- Label formats (spec, "The one-line label"):
  `main@a0a406f  <subject or note>`, `main@a0a406f + 3 uncommitted: a.py, b.py, c.py  (snapshot 5c1d2e9)`,
  `main@a0a406f + 3 uncommitted  <note>  (snapshot 5c1d2e9)`, `detached@a0a406f  <subject>`,
  `no git checkout`. Text is cut with `…` at 60 characters; at most three file names, then `, …`.
- Records written before this change must still load and label.

## Review Focus

1. Paths with spaces, renames, and non-ASCII names: the uncommitted list must hold the real
   repository-relative path, unquoted, with a rename listed once under its new path (Task 1 test).
2. Running from a subdirectory of the checkout, such as `experiments/`: the snapshot must hold
   changes anywhere in the tree (Task 2 test).
3. Running from a linked `git worktree`: the snapshot ref must be visible from the main checkout,
   since refs are shared (Task 2 test).
4. Two launches snapshotting the same dirty state at once (two terminals): both must succeed and
   agree on one commit (Task 2 test).
5. A `--note` with newlines or a long note: the label must stay on one line and be cut with `…`
   (Task 1 test).

---

### Task 1: Provenance fields, the one-line label, and the launch provenance

**Files:**
- Modify: `lib/rigging/src/rigging/provenance.py`
- Test: `lib/rigging/tests/test_provenance.py`

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `Provenance` fields, all defaulted: `commit_subject: str = ""`, `commit_time: str = ""`,
    `uncommitted: tuple[str, ...] = ()` (entries `"XY path"`, two status characters, a space, the
    path), `snapshot_commit: str | None = None`, `note: str | None = None`.
  - `Provenance.label -> str` (property), `Provenance.change_summary -> str` (property);
    `str(provenance)` returns `label`.
  - `read_git(args: Sequence[str], repo_dir: Path | str | None = None) -> str`: raw stdout, `""`
    when Git fails or is missing.
  - `launched_as(provenance: Provenance) -> ContextManager[Provenance]`,
    `current_launch() -> Provenance | None`, and `launch_provenance()` now returning the installed
    launch when there is one.

- [ ] **Step 1: Write the failing tests**

Replace the three `test_str_*` tests at the top of `lib/rigging/tests/test_provenance.py` with the
label tests below, and add the capture, round-trip, and launch tests after `test_json_round_trip_preserves_run_fields`.
Add `import dataclasses` and change the rigging import to
`from rigging.provenance import Provenance, current_launch, launch_provenance, launched_as, username_segment`.

```python
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
    p = _clean(dirty=True, uncommitted=(" M a.py", " M b.py", " M c.py"), note="try rmsnorm", snapshot_commit="5c1d2e9aa0")
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
```

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest lib/rigging/tests/test_provenance.py -n 0 -q`
Expected: FAIL (ImportError on `current_launch`, then missing fields and `label`).

- [ ] **Step 3: Implement**

In `lib/rigging/src/rigging/provenance.py`:

Update the imports:

```python
import contextlib
import dataclasses
import functools
import getpass
import json
import logging
import re
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
```

Add the constants below `logger`:

```python
# The longest note or commit subject a label shows before cutting it with "…".
LABEL_TEXT_WIDTH = 60
# The most uncommitted files a label names; more end in ", …".
LABEL_FILES = 3
```

Add the five fields after `command_line` and document them in the class docstring's
`Attributes:` list:

```python
    commit_subject: str = ""
    commit_time: str = ""
    uncommitted: tuple[str, ...] = ()
    snapshot_commit: str | None = None
    note: str | None = None
```

```text
        commit_subject: First line of the HEAD commit's message, or "" if uncaptured.
        commit_time: HEAD's committer date, ISO 8601, or "" if uncaptured.
        uncommitted: The working tree's changes as ``"XY path"`` entries (``git status
            --porcelain`` codes), untracked files included; a rename is listed under its new path.
        snapshot_commit: The commit that saved the uncommitted changes (``save_snapshot``), if any.
        note: The launch's own description of what the run tries, if given.
```

Replace the body of `capture` (keep its docstring, adding one sentence: "It also records HEAD's
subject and date and the uncommitted paths."):

```python
        cwd = str(repo_dir) if repo_dir is not None else None

        def g(*args: str) -> str:
            return read_git(args, cwd).strip()

        stash = g("stash", "create")
        ref = stash or "HEAD"
        uncommitted = _status_entries(read_git(["status", "--porcelain", "-z", "--untracked-files=all"], cwd))
        commit_time, _, commit_subject = g("log", "-1", "--format=%cI%n%s").partition("\n")
        return cls(
            tree_hash=g("rev-parse", "--short", f"{ref}^{{tree}}"),
            base_commit=g("rev-parse", "--short", "HEAD"),
            dirty=bool(uncommitted),
            branch=g("symbolic-ref", "--short", "-q", "HEAD") or None,
            built_by=_getuser(),
            git_remote=g("remote", "get-url", "origin") or None,
            created_at=datetime.now().isoformat(),
            command_line=tuple(sys.argv),
            commit_subject=commit_subject,
            commit_time=commit_time,
            uncommitted=uncommitted,
        )
```

Extend `from_json`'s constructor call:

```python
            commit_subject=d.get("commit_subject", ""),
            commit_time=d.get("commit_time", ""),
            uncommitted=tuple(d.get("uncommitted", ())),
            snapshot_commit=d.get("snapshot_commit"),
            note=d.get("note"),
```

Replace `__str__` with the label and its summary:

```python
    @property
    def change_summary(self) -> str:
        """The uncommitted files in a few words, such as ``3 uncommitted: llama.py, model.py, notes.md``."""
        if not self.uncommitted:
            return "uncommitted changes"
        names = [PurePosixPath(entry[3:]).name for entry in self.uncommitted]
        shown = ", ".join(names[:LABEL_FILES]) + (", …" if len(names) > LABEL_FILES else "")
        return f"{len(names)} uncommitted: {shown}"

    @property
    def label(self) -> str:
        """The code in one line: branch, short commit, and the subject, the note, or the changes."""
        if not self.base_commit:
            return "no git checkout"
        head = f"{self.branch or 'detached'}@{self.base_commit[:7]}"
        if not self.dirty:
            return f"{head}  {_one_line(self.note or self.commit_subject)}".rstrip()
        if self.note:
            changes = f"+ {len(self.uncommitted)} uncommitted  {_one_line(self.note)}"
        else:
            changes = f"+ {self.change_summary}"
        snapshot = f"  (snapshot {self.snapshot_commit[:7]})" if self.snapshot_commit else ""
        return f"{head} {changes}{snapshot}"

    def __str__(self) -> str:
        return self.label
```

Add these module functions above `_capture_once`, and replace `launch_provenance`:

```python
def read_git(args: Sequence[str], repo_dir: Path | str | None = None) -> str:
    """Raw stdout of ``git args`` run in ``repo_dir``; "" when Git fails or is not installed."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=repo_dir)
    except FileNotFoundError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def _status_entries(porcelain_z: str) -> tuple[str, ...]:
    """``git status --porcelain -z`` output as ``"XY path"`` entries.

    ``-z`` leaves paths unquoted. A rename or copy is followed by its source path, which is skipped,
    so it is listed once under its new path.
    """
    entries: list[str] = []
    fields = iter(porcelain_z.split("\0"))
    for field in fields:
        if not field:
            continue
        entries.append(field)
        if "R" in field[:2] or "C" in field[:2]:
            next(fields, None)
    return tuple(entries)


def _one_line(text: str) -> str:
    """``text`` on one line, cut with "…" at the label's width."""
    flat = " ".join(text.split())
    return flat if len(flat) <= LABEL_TEXT_WIDTH else flat[: LABEL_TEXT_WIDTH - 1] + "…"
```

```python
# The provenance an experiment command installed for its launch, innermost last.
_launches: list[Provenance] = []


@contextlib.contextmanager
def launched_as(provenance: Provenance) -> Iterator[Provenance]:
    """Make ``provenance`` this process's launch provenance until the block exits.

    The experiment command line installs the provenance it captured, snapshot and note included, so
    the artifact records and the W&B tracker of the jobs it runs, which are threads of this
    process, record the same launch.
    """
    with _capture_lock:
        _launches.append(provenance)
    try:
        yield provenance
    finally:
        with _capture_lock:
            _launches.remove(provenance)


def current_launch() -> Provenance | None:
    """The provenance installed with :func:`launched_as`, or ``None`` outside one."""
    with _capture_lock:
        return _launches[-1] if _launches else None


def launch_provenance() -> Provenance:
    """The current launch's provenance: the installed one, else captured once per process (thread-safe)."""
    with _capture_lock:
        return _launches[-1] if _launches else _capture_once()
```

`to_json` stays `json.dumps(dataclasses.asdict(self))`: tuples become lists, which `from_json`
turns back into tuples.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest lib/rigging/tests/test_provenance.py -n 0 -q`
Expected: PASS. Also run `grep -rn "str(.*provenance\|{provenance}" lib/*/src` and confirm no
caller relied on the old `__str__` format.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Record each run's commit subject and changes"
git add lib/rigging docs/poorin/prune-metrics.csv
git commit -m "[rigging] Record each run's commit subject and changes" -m "Provenance gains the commit's subject and date, the uncommitted paths, a snapshot commit, and a note, with a one-line label that names the code without a bare hash. launched_as installs a launch's provenance for the jobs it runs. Spec: specs/2026-10-11-run-code-provenance-design.md."
```

---

### Task 2: The snapshot commit

**Files:**
- Modify: `lib/rigging/src/rigging/provenance.py`
- Test: `lib/rigging/tests/test_provenance.py`

**Interfaces:**
- Consumes: `read_git` (Task 1).
- Produces: `SNAPSHOT_REF_PREFIX = "refs/poorin/snapshots/"` and
  `save_snapshot(message: str, repo_dir: Path | None = None) -> str` (the full commit hash).
  `_git` gains a keyword `env: Mapping[str, str] | None = None`.

- [ ] **Step 1: Write the failing tests**

Add to `lib/rigging/tests/test_provenance.py` (import `os` and
`from concurrent.futures import ThreadPoolExecutor`; add `SNAPSHOT_REF_PREFIX, save_snapshot` to the
rigging import):

```python
def _out(args: list[str], cwd: Path) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _dirty_repo(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path)
    (repo / ".gitignore").write_text("ignored/\n")
    (repo / "gone.txt").write_text("bye\n")
    _run(["git", "add", ".gitignore", "gone.txt"], repo)
    _run(["git", "commit", "-m", "more"], repo)
    (repo / "f.txt").write_text("edited\n")  # tracked edit
    (repo / "staged.txt").write_text("staged\n")
    _run(["git", "add", "staged.txt"], repo)  # staged new file
    (repo / "gone.txt").unlink()  # deletion
    (repo / "sub").mkdir()
    (repo / "sub" / "new.txt").write_text("untracked\n")  # untracked file
    (repo / "ignored").mkdir()
    (repo / "ignored" / "big.bin").write_text("data\n")  # ignored
    return repo


def test_a_snapshot_holds_every_change_but_ignored_files(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo)
    files = set(_out(["git", "ls-tree", "-r", "--name-only", commit], repo).split())
    assert files == {".gitignore", "f.txt", "staged.txt", "sub/new.txt"}
    assert _out(["git", "show", f"{commit}:f.txt"], repo) == "edited\n"
    assert _out(["git", "log", "-1", "--format=%s", commit], repo).strip() == "try it"
    assert _out(["git", "rev-parse", f"{commit}^"], repo) == _out(["git", "rev-parse", "HEAD"], repo)


def test_a_snapshot_leaves_the_branch_index_and_files_alone(tmp_path):
    repo = _dirty_repo(tmp_path)
    before = [_out(args, repo) for args in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"],
                                            ["git", "diff", "--cached"], ["git", "stash", "list"])]
    save_snapshot("try it", repo)
    after = [_out(args, repo) for args in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"],
                                           ["git", "diff", "--cached"], ["git", "stash", "list"])]
    assert after == before
    assert (repo / "f.txt").read_text() == "edited\n"


def test_a_snapshot_is_kept_by_a_ref(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo)
    assert _out(["git", "rev-parse", f"{SNAPSHOT_REF_PREFIX}{commit}"], repo).strip() == commit


def test_one_state_and_message_give_one_snapshot(tmp_path):
    repo = _dirty_repo(tmp_path)
    assert save_snapshot("try it", repo) == save_snapshot("try it", repo)
    assert save_snapshot("try it", repo) != save_snapshot("another note", repo)


def test_a_snapshot_from_a_subdirectory_holds_the_whole_tree(tmp_path):
    repo = _dirty_repo(tmp_path)
    commit = save_snapshot("try it", repo / "sub")
    assert _out(["git", "show", f"{commit}:f.txt"], repo) == "edited\n"


def test_a_snapshot_from_a_linked_worktree_is_visible_from_the_main_checkout(tmp_path):
    repo = _init_repo(tmp_path / "main")
    linked = tmp_path / "linked"
    _run(["git", "worktree", "add", "-b", "exp", str(linked)], repo)
    (linked / "f.txt").write_text("in the worktree\n")
    commit = save_snapshot("worktree run", linked)
    assert _out(["git", "show", f"{SNAPSHOT_REF_PREFIX}{commit}:f.txt"], repo) == "in the worktree\n"


def test_concurrent_snapshots_of_one_state_agree(tmp_path):
    repo = _dirty_repo(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        commits = set(pool.map(lambda _: save_snapshot("try it", repo), range(4)))
    assert len(commits) == 1


def test_a_repository_without_commits_cannot_be_snapshotted(tmp_path):
    _run(["git", "init", "-b", "main"], tmp_path)
    (tmp_path / "f.txt").write_text("x\n")
    with pytest.raises(RuntimeError):
        save_snapshot("try it", tmp_path)
```

`_init_repo(tmp_path / "main")` needs the directory to exist: change `_init_repo` to call
`tmp_path.mkdir(parents=True, exist_ok=True)` first.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-sync pytest lib/rigging/tests/test_provenance.py -n 0 -q -k snapshot`
Expected: FAIL with ImportError on `save_snapshot`.

- [ ] **Step 3: Implement**

Add `import os` and `import tempfile` to the imports, `Mapping` to the `collections.abc` import, and
`from rigging.timing import Duration, ExponentialBackoff` (rigging.timing imports nothing from
rigging, so there is no cycle). Add the constants below `LABEL_FILES`:

```python
# Snapshot commits are kept under this ref namespace, one ref per commit, so `git gc` keeps them.
SNAPSHOT_REF_PREFIX = "refs/poorin/snapshots/"
# How long a snapshot waits for a concurrent launch of the same state to finish writing its ref.
SNAPSHOT_REF_WAIT = 5.0
```

Give `_git` an environment:

```python
def _git(args: list[str], cwd: str | None, check: bool = True, env: Mapping[str, str] | None = None) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()
```

Add `save_snapshot` after `launch_provenance`:

```python
def save_snapshot(message: str, repo_dir: Path | None = None) -> str:
    """Commit the working tree, untracked files included, without touching the branch, index, or
    files, and return the commit's full hash.

    The tree is built in a temporary index, so ``.gitignore`` applies and the real index is left
    alone. The commit's parent is ``HEAD`` and its dates are ``HEAD``'s committer date, so one tree,
    parent, and message always give one commit, and repeated runs of one dirty state share it. A
    ref under :data:`SNAPSHOT_REF_PREFIX` keeps it from ``git gc``.

    Raises:
        RuntimeError: Outside a checkout, or in a repository with no commits.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    head = _git(["rev-parse", "HEAD"], cwd)
    head_date = _git(["log", "-1", "--format=%cI", "HEAD"], cwd)
    with tempfile.TemporaryDirectory(prefix="poorin-snapshot-") as tmp:
        env = {
            **os.environ,
            "GIT_INDEX_FILE": os.path.join(tmp, "index"),
            "GIT_AUTHOR_DATE": head_date,
            "GIT_COMMITTER_DATE": head_date,
        }
        _git(["read-tree", "HEAD"], cwd, env=env)
        _git(["add", "--all"], cwd, env=env)
        tree = _git(["write-tree"], cwd, env=env)
        commit = _git(["commit-tree", tree, "-p", head, "-m", message], cwd, env=env)
    ref = f"{SNAPSHOT_REF_PREFIX}{commit}"
    try:
        _git(["update-ref", ref, commit], cwd)
    except RuntimeError:
        # Another launch of the same state holds the ref's lock and is writing this same commit.
        written = ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            lambda: read_git(["rev-parse", "--verify", "--quiet", ref], cwd).strip() == commit,
            timeout=Duration.from_seconds(SNAPSHOT_REF_WAIT),
        )
        if not written:
            raise
    return commit
```

Update the module docstring's second paragraph to mention snapshots: "A dirty tree's changes can
be saved with :func:`save_snapshot`, a commit kept under ``refs/poorin/snapshots/`` that leaves the
branch and index alone."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-sync pytest lib/rigging/tests/test_provenance.py -n 0 -q`
Expected: PASS, including `test_concurrent_snapshots_of_one_state_agree`, where the launches that
lose the ref's lock wait for the winner's write.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Save uncommitted code as a snapshot commit"
git add lib/rigging docs/poorin/prune-metrics.csv
git commit -m "[rigging] Save uncommitted code as a snapshot commit" -m "save_snapshot commits the working tree, untracked files included, through a temporary index and keeps it under refs/poorin/snapshots/, leaving the branch, index, and files alone. Fixed dates make one dirty state one commit, so repeated runs share a snapshot."
```

---

### Task 3: The runner's schedule and a provenance for `run()`

**Files:**
- Modify: `lib/poorin/src/poorin/execution/step_runner.py`
- Modify: `lib/poorin/src/poorin/execution/lazy.py`
- Test: `tests/execution/test_step_runner.py`, `tests/execution/test_artifact.py`

**Interfaces:**
- Consumes: `Provenance` with `note` (Task 1).
- Produces:
  - `poorin.execution.step_runner.steps_to_build(steps: Iterable[StepSpec]) -> list[StepSpec]`
  - `poorin.execution.lazy.lower_all(handles: Sequence[ArtifactStep], provenance: Provenance) -> list[StepSpec]`
  - `poorin.execution.lazy.run(*handles, max_concurrent=8, force_run_failed=True, accept_drift=(), provenance: Provenance | None = None)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/execution/test_step_runner.py` (import `shutil`, `from rigging.provenance import Provenance`,
`from poorin.execution.lazy import OUT, apply, lower_all, run`, and `steps_to_build` from the step
runner; `_write` is a local helper):

```python
def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _graph():
    data = apply("data/toy", _write, version="2026.07.16", out=OUT)
    model = apply("models/toy", _write, version="2026.07.16", out=OUT, data=data)
    return data, model


def _names(steps):
    return [s.name for s in steps]


def test_steps_to_build_lists_the_unbuilt_graph_in_schedule_order():
    _, model = _graph()
    assert _names(steps_to_build(lower_all([model], Provenance.capture()))) == ["data/toy", "models/toy"]


def test_steps_to_build_leaves_out_built_steps_and_inputs_pruned_behind_them():
    data, model = _graph()
    run(model)
    shutil.rmtree(data.path())  # an input behind a built consumer is not rebuilt
    report = apply("reports/toy", _write, version="dev", out=OUT, model=model)
    assert _names(steps_to_build(lower_all([model], Provenance.capture()))) == []
    assert _names(steps_to_build(lower_all([report], Provenance.capture()))) == ["reports/toy"]
```

Add to `tests/execution/test_artifact.py`. Extend its `from poorin.execution.lazy import ...` line
with `OUT, apply`, import `os`, `from rigging.provenance import Provenance`, and `read_record` from
`poorin.execution.artifact` if it is not imported yet (`dataclasses` already is):

```python
def _make_dir(out):
    os.makedirs(out, exist_ok=True)


def test_run_records_the_provenance_it_is_given():
    handle = apply("data/given", _make_dir, version="2026.07.16", out=OUT)
    given = dataclasses.replace(Provenance.capture(), note="from the command line")
    run(handle, provenance=given)
    assert read_record(handle.path()).provenance.note == "from the command line"
```

`handle.path()` defaults to the current prefix. Step functions are module-level, as everywhere in
these tests: a lambda has no stable name for the fingerprint.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution/test_step_runner.py tests/execution/test_artifact.py -n 0 -q -k "steps_to_build or provenance_it_is_given"`
Expected: FAIL with ImportError on `steps_to_build` / `lower_all`.

- [ ] **Step 3: Implement**

In `step_runner.py`, add after `step_is_built`:

```python
def steps_to_build(steps: Iterable[StepSpec]) -> list[StepSpec]:
    """The steps a :meth:`StepRunner.run` of ``steps`` would launch, in schedule order.

    A built step is served from its cache, and the inputs behind it are pruned from the schedule
    (:func:`step_is_built`), so neither is listed.
    """
    seen: set[str] = set()
    pruned: set[str] = set()
    scheduled = [s for step in steps for s in _expand_unseen(step, seen, step_is_built, pruned)]
    return [s for s in scheduled if s.output_path not in pruned]
```

In `lazy.py`, add above `run` (import `Sequence` from `collections.abc` if absent):

```python
def lower_all(handles: Sequence["ArtifactStep"], provenance: Provenance) -> list[StepSpec]:
    """Lower ``handles`` into one ``StepSpec`` graph, sharing common steps and recording ``provenance``."""
    memo: dict[int, StepSpec] = {}
    return [_lower(h, provenance, memo) for h in handles]
```

and change `run`:

```python
def run(
    *handles: "ArtifactStep[T]",
    max_concurrent: int = 8,
    force_run_failed: bool = True,
    accept_drift: Collection[str] = (),
    provenance: Provenance | None = None,
) -> list[T]:
    """...existing docstring..., then add:

    ``provenance`` is the launch recorded on every artifact built, such as the one the experiment
    command line captured with its snapshot and note; omitted, it is captured from the working
    directory.
    """
    _refuse_drift(graph_states(graph_handles(list(handles))), accept_drift)
    specs = lower_all(handles, provenance if provenance is not None else Provenance.capture())
    StepRunner().run(specs, force_run_failed=force_run_failed, max_concurrent=max_concurrent)
    return [h.artifact_type.raw_load(h.path()) for h in handles]
```

(Replace "...existing docstring..." with the current docstring text unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/execution -n 0 -q`
Expected: PASS.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Expose the runner's schedule; let run() take a provenance"
git add lib/poorin tests/execution docs/poorin/prune-metrics.csv
git commit -m "[poorin] Expose the runner's schedule; let run() take a provenance" -m "steps_to_build lists the steps a run would launch, leaving out built steps and the inputs pruned behind them, so a caller can tell what a run would build before it starts. run() records a provenance it is given, such as the experiment command line's snapshot and note."
```

---

### Task 4: The code check

**Files:**
- Create: `lib/poorin/src/poorin/experiment/code_check.py`
- Modify: `docs/poorin/architecture.md` (one row)
- Test: `tests/experiment/test_code_check.py`

**Interfaces:**
- Consumes: `Provenance.label`, `Provenance.uncommitted` (Task 1); `steps_to_build`, `lower_all` (Task 3).
- Produces:
  - `CodeVerdict(StrEnum)`: `COMMITTED`, `CACHED`, `SNAPSHOT`, `REFUSED`, `UNRECORDED`.
  - `CodeCheck` (frozen dataclass): `provenance: Provenance`, `calendar: tuple[str, ...]`, `verdict: CodeVerdict`.
  - `check_code(provenance: Provenance, builds: Sequence[StepSpec], *, allow_uncommitted: bool) -> CodeCheck`
    where `builds` is `steps_to_build(...)`'s result.
  - `plan_lines(check: CodeCheck) -> list[str]` and `refusal(check: CodeCheck) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment/test_code_check.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether a run may build from the working tree's code (poorin.experiment.code_check)."""

import os
from pathlib import Path

import pytest
from poorin.execution.lazy import OUT, apply, lower_all, run
from poorin.execution.step_runner import steps_to_build
from poorin.experiment.code_check import CodeVerdict, check_code, plan_lines, refusal
from rigging.provenance import Provenance

CLEAN = Provenance(tree_hash="t", base_commit="a0a406ff4", dirty=False, branch="main", built_by="u", commit_subject="s")
DIRTY = Provenance(
    tree_hash="t", base_commit="a0a406ff4", dirty=True, branch="main", built_by="u", uncommitted=(" M model.py",)
)
NO_CHECKOUT = Provenance(tree_hash="", base_commit="", dirty=False, branch=None, built_by="u")


def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _builds(*handles):
    return steps_to_build(lower_all(list(handles), CLEAN))


def _calendar():
    return apply("data/toy", _write, version="2026.07.16", out=OUT)


def _dev(data=None):
    return apply("models/toy", _write, version="dev", out=OUT, data=data)


@pytest.mark.parametrize(
    ("provenance", "handles", "allow", "verdict"),
    [
        (CLEAN, "calendar", False, CodeVerdict.COMMITTED),
        (DIRTY, "calendar", False, CodeVerdict.REFUSED),
        (DIRTY, "calendar", True, CodeVerdict.SNAPSHOT),
        (DIRTY, "dev", False, CodeVerdict.SNAPSHOT),
        (NO_CHECKOUT, "calendar", False, CodeVerdict.REFUSED),
        (NO_CHECKOUT, "dev", False, CodeVerdict.UNRECORDED),
    ],
)
def test_the_verdict_follows_the_tree_and_what_would_build(provenance, handles, allow, verdict):
    built = _builds(_calendar()) if handles == "calendar" else _builds(_dev())
    assert check_code(provenance, built, allow_uncommitted=allow).verdict is verdict


def test_a_dev_run_over_a_built_calendar_input_may_use_uncommitted_code():
    data = _calendar()
    run(data)
    check = check_code(DIRTY, _builds(_dev(data)), allow_uncommitted=False)
    assert (check.verdict, check.calendar) == (CodeVerdict.SNAPSHOT, ())


def test_a_run_that_builds_nothing_is_cached_and_never_refused():
    data = _calendar()
    run(data)
    assert check_code(DIRTY, _builds(data), allow_uncommitted=False).verdict is CodeVerdict.CACHED


def test_a_pinned_input_is_not_a_calendar_build(tmp_path):
    pinned = apply("data/pinned", _write, version="2026.07.16", out=OUT, pin=str(tmp_path))
    check = check_code(DIRTY, _builds(_dev(pinned)), allow_uncommitted=False)
    assert check.calendar == ()


def test_the_plan_predicts_what_run_would_do():
    snapshot = check_code(DIRTY, _builds(_dev()), allow_uncommitted=False)
    assert plan_lines(snapshot) == [f"Code:   {DIRTY.label}", "        --run saves a snapshot"]
    refused = check_code(DIRTY, _builds(_calendar()), allow_uncommitted=False)
    assert refused.calendar == ("data/toy@2026.07.16",)
    assert "--run refuses: 1 calendar-versioned artifact would build" in plan_lines(refused)[1]
    assert plan_lines(check_code(CLEAN, _builds(_dev()), allow_uncommitted=False)) == [f"Code:   {CLEAN.label}"]


def test_a_refusal_names_the_artifacts_and_the_uncommitted_paths():
    message = refusal(check_code(DIRTY, _builds(_calendar()), allow_uncommitted=False))
    assert "1 calendar-versioned artifact would build from uncommitted code:" in message
    assert "  data/toy@2026.07.16" in message
    assert "Uncommitted (1): M model.py" in message
    assert message.endswith(
        "Commit the changes and run again, run a dev version, or pass --allow-uncommitted to build from a snapshot."
    )
```

If `apply(..., pin=...)` builds a step the runner schedules (a pin has no status file), the pinned
test still holds because `check_code` drops pinned steps; keep the test as written.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_code_check.py -n 0 -q`
Expected: FAIL with ModuleNotFoundError for `poorin.experiment.code_check`.

- [ ] **Step 3: Implement**

Create `lib/poorin/src/poorin/experiment/code_check.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether ``--run`` may build from the working tree's code, and the lines that report it (D107).

A calendar-versioned artifact is a result to cite, so it is built only from committed code. A run
that builds only ``dev`` artifacts may use uncommitted code, which the experiment command line saves
as a snapshot commit (:func:`rigging.provenance.save_snapshot`). ``--allow-uncommitted`` lets a
calendar artifact build from a snapshot too.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rigging.provenance import Provenance

from poorin.execution.artifact import FINGERPRINT_KEY, PINNED_KEY, VERSION_KEY, is_mutable_version
from poorin.execution.step_spec import StepSpec

# The most uncommitted paths a refusal lists.
REFUSAL_PATHS = 10
# Aligns a continuation line under the plan's "Code:   " label.
PLAN_INDENT = " " * 8
SNAPSHOT_ADVICE = (
    "Commit the changes and run again, run a dev version, or pass --allow-uncommitted to build from a snapshot."
)


class CodeVerdict(StrEnum):
    COMMITTED = "committed"
    """A clean checkout: the record names the commit."""
    CACHED = "cached"
    """Nothing would build: every artifact is served from its cache."""
    SNAPSHOT = "snapshot"
    """Uncommitted code may run, and is saved as a snapshot commit."""
    REFUSED = "refused"
    """Uncommitted or unknown code would build a calendar-versioned artifact."""
    UNRECORDED = "unrecorded"
    """No Git checkout: dev artifacts build, and no code is recorded."""


@dataclass(frozen=True)
class CodeCheck:
    provenance: Provenance
    calendar: tuple[str, ...]
    """``name@version`` of each calendar-versioned artifact the run would build."""
    verdict: CodeVerdict


def check_code(provenance: Provenance, builds: Sequence[StepSpec], *, allow_uncommitted: bool) -> CodeCheck:
    """Whether a run that launches ``builds`` (:func:`~poorin.execution.step_runner.steps_to_build`)
    may build from the code ``provenance`` describes."""
    calendar = tuple(
        f"{step.name}@{step.hash_attrs[VERSION_KEY]}"
        for step in builds
        if FINGERPRINT_KEY in step.hash_attrs
        and not step.hash_attrs.get(PINNED_KEY)
        and not is_mutable_version(step.hash_attrs[VERSION_KEY])
    )
    checkout = bool(provenance.base_commit)
    if checkout and not provenance.dirty:
        verdict = CodeVerdict.COMMITTED
    elif not builds:
        verdict = CodeVerdict.CACHED
    elif calendar and not allow_uncommitted:
        verdict = CodeVerdict.REFUSED
    else:
        verdict = CodeVerdict.SNAPSHOT if checkout else CodeVerdict.UNRECORDED
    return CodeCheck(provenance, calendar, verdict)


def _artifacts(count: int) -> str:
    return f"{count} calendar-versioned artifact{'' if count == 1 else 's'}"


def plan_lines(check: CodeCheck) -> list[str]:
    """The plan's ``Code:`` line and, for a tree that is not clean, what ``--run`` would do."""
    lines = [f"Code:   {check.provenance.label}"]
    if check.verdict is CodeVerdict.SNAPSHOT:
        lines.append(f"{PLAN_INDENT}--run saves a snapshot")
    elif check.verdict is CodeVerdict.UNRECORDED:
        lines.append(f"{PLAN_INDENT}not a Git checkout: --run records no code")
    elif check.verdict is CodeVerdict.REFUSED:
        lines.append(f"{PLAN_INDENT}--run refuses: {_artifacts(len(check.calendar))} would build from uncommitted code")
    return lines


def refusal(check: CodeCheck) -> str:
    """Why ``--run`` refuses, which artifacts it would have built, and what to do."""
    lines = [f"{_artifacts(len(check.calendar))} would build from uncommitted code:"]
    lines.extend(f"  {label}" for label in check.calendar)
    uncommitted = check.provenance.uncommitted
    if check.provenance.base_commit:
        shown = ", ".join(entry.strip() for entry in uncommitted[:REFUSAL_PATHS])
        more = f", and {len(uncommitted) - REFUSAL_PATHS} more" if len(uncommitted) > REFUSAL_PATHS else ""
        lines.append(f"Uncommitted ({len(uncommitted)}): {shown}{more}")
    else:
        lines.append("This is not a Git checkout, so the code cannot be recorded.")
    lines.append(SNAPSHOT_ADVICE)
    return "\n".join(lines)
```

`entry.strip()` turns `" M model.py"` into `"M model.py"` and keeps `"?? new.py"`.

In `docs/poorin/architecture.md`, add after the `poorin.experiment.cli` row:

```text
| `poorin.experiment.code_check` | Whether `--run` may build from the working tree's code, and the plan's `Code:` lines | R1 |
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_code_check.py tests/tools/test_architecture.py -n 0 -q`
Expected: PASS.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Decide whether a run may build from uncommitted code"
git add lib/poorin tests/experiment/test_code_check.py docs/poorin/architecture.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Decide whether a run may build from uncommitted code" -m "check_code refuses a calendar-versioned build from uncommitted or unknown code, allows a dev build to run from a snapshot, and ignores a run that builds nothing. plan_lines and refusal word the outcome for the experiment command line."
```

---

### Task 5: The command line applies the check

**Files:**
- Modify: `lib/poorin/src/poorin/experiment/cli.py`
- Modify: `tools/roots.toml` (the two runnable roots' `args`)
- Modify: `docs/poorin/decisions.md` (D107)
- Test: `tests/experiment/test_cli.py`

**Interfaces:**
- Consumes: `Provenance.capture`, `Provenance.change_summary`, `launched_as`, `save_snapshot`
  (Tasks 1–2); `lower_all`, `steps_to_build`, `run(provenance=...)` (Task 3);
  `CodeVerdict`, `check_code`, `plan_lines`, `refusal` (Task 4).
- Produces: options `--note TEXT` and `--allow-uncommitted`;
  `BuildRunner = Callable[[list[ArtifactStep], int, frozenset[str], Provenance], None]`.

- [ ] **Step 1: Write the failing tests**

In `tests/experiment/test_cli.py`, add `import subprocess` and `from pathlib import Path`, then add
this autouse fixture after the `runner` fixture. It puts every command in a clean checkout beside
the prefix, because the developer's own tree is dirty whenever tests run before a commit:

```python
def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture(autouse=True)
def checkout(tmp_path, monkeypatch) -> Path:
    """A clean Git checkout as the working directory, outside the prefix the tests write to."""
    path = tmp_path / "checkout"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "model.py").write_text("v1\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "init")
    monkeypatch.chdir(path)
    return path
```

Add these tests at the end of the file:

```python
def _marker(**_):
    return apply("scratch/marker", _write_marker, out=OUT)


def _dev_over_calendar(**_):
    data = apply("data/toy", _write_marker, version="2026.07.16", out=OUT)
    return apply("models/toy", _write_marker_with, out=OUT, payload=data)


def test_uncommitted_code_cannot_build_a_calendar_artifact(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16", "--run"])
    assert result.exit_code != 0
    assert "scratch/marker@2026.07.16" in result.output
    assert "Uncommitted (1): M model.py" in result.output
    assert not (tmp_path / "scratch" / "marker").exists()


def test_a_dev_run_on_uncommitted_code_records_a_snapshot_of_it(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--run", "--note", "try v2"])
    assert result.exit_code == 0, result.output
    [out] = (tmp_path / "scratch" / "marker").glob("*-dev")
    provenance = read_record(str(out)).provenance
    assert provenance.note == "try v2"
    assert _git(checkout, "show", f"{provenance.snapshot_commit}:model.py") == "v2"
    assert _git(checkout, "rev-parse", f"refs/poorin/snapshots/{provenance.snapshot_commit}")
    assert f"git diff {provenance.base_commit} {provenance.snapshot_commit[:9]}" in result.output
    assert _git(checkout, "status", "--porcelain") == "M model.py"  # the tree is left as it was


def test_a_dev_run_may_build_over_a_calendar_input_built_from_committed_code(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    assert runner.invoke(experiment_main(_dev_over_calendar), ["--version", "dev", "--run"]).exit_code == 0
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_dev_over_calendar), ["--version", "dev", "--run"])
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "models" / "toy").glob("*-dev"))) == 2


def test_allow_uncommitted_builds_a_calendar_artifact_from_a_snapshot(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    (checkout / "model.py").write_text("v2\n")
    args = ["--version", "2026.07.16", "--run", "--allow-uncommitted"]
    result = runner.invoke(experiment_main(_marker), args)
    assert result.exit_code == 0, result.output
    provenance = read_record(str(tmp_path / "scratch" / "marker" / "2026.07.16")).provenance
    assert provenance.dirty and provenance.snapshot_commit


def test_a_run_that_builds_nothing_is_not_refused(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    args = ["--version", "2026.07.16", "--run"]
    assert runner.invoke(experiment_main(_marker), args).exit_code == 0
    (checkout / "model.py").write_text("v2\n")
    result = runner.invoke(experiment_main(_marker), args)
    assert result.exit_code == 0, result.output


def test_a_clean_run_records_its_commit_and_no_snapshot(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    result = runner.invoke(experiment_main(_marker), ["--version", "dev", "--run"])
    assert result.exit_code == 0, result.output
    [out] = (tmp_path / "scratch" / "marker").glob("*-dev")
    provenance = read_record(str(out)).provenance
    assert (provenance.dirty, provenance.snapshot_commit) == (False, None)
    assert provenance.base_commit == _git(checkout, "rev-parse", "--short", "HEAD")
    assert f"Code:   main@{provenance.base_commit[:7]}  init" in result.output


def test_the_plan_shows_the_code_and_predicts_a_refusal(runner, tmp_path, monkeypatch, checkout):
    monkeypatch.setenv("POORIN_PREFIX", str(tmp_path))
    clean = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16"])
    assert re.search(r"^Code:   main@\w{7}  init$", clean.output, re.MULTILINE)
    (checkout / "model.py").write_text("v2\n")
    dirty = runner.invoke(experiment_main(_marker), ["--version", "2026.07.16"])
    assert dirty.exit_code == 0, dirty.output
    assert "--run refuses: 1 calendar-versioned artifact would build from uncommitted code" in dirty.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_cli.py -n 0 -q`
Expected: the new tests FAIL (no `--note` option, no refusal, no `Code:` line); the existing tests
still PASS inside the clean checkout.

- [ ] **Step 3: Implement**

In `cli.py`:

Imports (add):

```python
import dataclasses

from rigging.provenance import Provenance, launched_as, save_snapshot

from poorin.execution.lazy import ArtifactStep, graph_handles, lower_all, run
from poorin.execution.step_runner import steps_to_build
from poorin.experiment.code_check import CodeVerdict, check_code, plan_lines, refusal
```

(`poorin.execution.lazy` is already imported: extend that line rather than adding a second one.)

```python
BuildRunner = Callable[[list[ArtifactStep], int, frozenset[str], Provenance], None]
```

`_print_plan` takes the code lines and prints them after the prefix:

```python
def _print_plan(states: list[HandleState], shown: list[HandleState], code: list[str]) -> None:
    """The prefix, the code the run would use, one line per artifact (state, fingerprint,
    ``name@version``) with the diff under a drifted one, then the identity config of each artifact
    in ``shown`` as YAML."""
    click.echo(f"Prefix: {poorin_prefix()}")
    for line in code:
        click.echo(line)
    ...rest unchanged...
```

`_run_locally`:

```python
def _run_locally(
    handles: list[ArtifactStep], max_concurrent: int, accept_drift: frozenset[str], provenance: Provenance
) -> None:
    run(*handles, max_concurrent=max_concurrent, accept_drift=accept_drift, provenance=provenance)
```

Two options, placed after `--max-concurrent`:

```python
    @click.option(
        "--note",
        default=None,
        metavar="TEXT",
        help="What this run tries, in a few words. It labels the run's code in place of the commit subject.",
    )
    @click.option(
        "--allow-uncommitted",
        is_flag=True,
        help="Build calendar-versioned artifacts from uncommitted code, saved as a snapshot commit.",
    )
```

Add `note: str | None` and `allow_uncommitted: bool` to the wrapper's keyword parameters. Replace
the block from `if show_config and do_run:` to the end of the wrapper with:

```python
        if show_config and do_run:
            raise click.UsageError("--show-config prints the plan; it cannot be combined with --run")
        launch = dataclasses.replace(Provenance.capture(), note=note)
        check = check_code(launch, steps_to_build(lower_all(handles, launch)), allow_uncommitted=allow_uncommitted)
        try:
            # A run reads every artifact's recorded recipe itself (lazy.run); only the plan and a
            # resume read them here.
            if not do_run or resume_version is not None:
                states = graph_states(graph)
                returned = {f"{h.name}@{h.version}" for h in handles}
                if resume_version is not None:
                    empty = [s.label for s in states if s.label in returned and s.recorded_fingerprint is None]
                    if empty:
                        raise click.UsageError(
                            f"nothing to resume at {empty}: no run started there. "
                            "Check that the --set flags match the original run's."
                        )
                if not do_run:
                    check_accept_drift(states, accept_drift)  # as --run would, so a typo shows in the plan
                    shown = [s for s in states if s.label in returned] if show_config else []
                    _print_plan(states, shown, plan_lines(check))
                    return
            if check.verdict is CodeVerdict.REFUSED:
                raise click.ClickException(refusal(check))
            if check.verdict is CodeVerdict.SNAPSHOT:
                snapshot = save_snapshot(launch.note or launch.change_summary)
                launch = dataclasses.replace(launch, snapshot_commit=snapshot)
            click.echo(f"Code:   {launch.label}")
            with launched_as(launch):
                runner(handles, max_concurrent, frozenset(accept_drift), launch)
            if launch.snapshot_commit is not None:
                click.echo(f"See changes: git diff {launch.base_commit} {launch.snapshot_commit[:9]}")
        except (FingerprintMismatchError, DriftAcceptanceError) as e:
            raise click.ClickException(str(e)) from e
```

Module docstring: add to the option list "``--note TEXT`` (labels the run's code)" and
"``--allow-uncommitted`` (build calendar artifacts from a snapshot)", and add this paragraph after
the one on ``--version``:

```text
Before ``--run`` builds, the code is checked (:mod:`poorin.experiment.code_check`): a calendar-
versioned artifact builds only from committed code, and a run that builds only ``dev`` artifacts
saves uncommitted changes as a snapshot commit under ``refs/poorin/snapshots/``. The plan prints the
code's one-line label and what ``--run`` would do.
```

In `tools/roots.toml`, append `"--allow-uncommitted"` to both runnable roots' `args`:

```toml
args = ["--device", "rtx5090x1", "--dataset", "wikitext", "--version", "dev", "--run", "--allow-uncommitted"]
```

```toml
args = ["--version", "dev", "--run", "--allow-uncommitted"]
```

In `docs/poorin/decisions.md`, add after D106 (before the `---` that closes the log):

```markdown
### D107. Code provenance for experiment runs

- **When:** 2026-10-11. **Status:** Adopted. Spec: `specs/2026-10-11-run-code-provenance-design.md`;
  plan: `plans/2026-10-11-run-code-provenance.md`.
- **Decision:** `--run` refuses to build a calendar-versioned artifact from uncommitted code. A run
  that builds only `dev` artifacts saves its uncommitted changes, untracked files included, as a
  snapshot commit under `refs/poorin/snapshots/`, leaving the branch, index, and files alone.
  `--allow-uncommitted` builds calendar artifacts from a snapshot; the full check passes it,
  because it runs before each commit. Each run's code has a one-line label (branch, short commit,
  and the subject, the `--note`, or the changed files), shown in the plan, after the run, in W&B,
  and by `python -m poorin.experiment.runs`.
- **Rulings while planning:** a run that builds nothing saves no snapshot and is never refused; a
  repository with no commits counts as outside a checkout; one dirty state gives one snapshot
  commit, because the commit's dates are fixed to `HEAD`'s.
- **Why:** The owner, 2026-10-11: the fingerprint covers config, not code, and a record named its
  code by a hash that was lost for uncommitted work and unreadable at a glance. The owner chose
  the split by version over refusing every dirty run or never refusing.
- **Alternatives:** Refuse every `--run` on a dirty tree; never refuse and always snapshot;
  enforce the rule inside `run()`, which would break every test run on a dirty tree.
- **Cost if wrong:** A first run on an empty prefix with a dirty tree is refused, because it would
  build its calendar dataset; commit first or pass `--allow-uncommitted`. Snapshot refs stay local
  until pushed with `git push origin 'refs/poorin/*:refs/poorin/*'`.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment -n 0 -q`
Expected: PASS. Then print a real plan from the repository root and confirm the `Code:` line:
`uv run --no-sync python experiments/tutorials/train_tiny_model.py --version dev`

- [ ] **Step 5: Lint, full check, commit**

The full check's training roots now pass `--allow-uncommitted` and save a snapshot of the
uncommitted task; that is expected.

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Refuse calendar builds from uncommitted code (D107)"
git add lib/poorin tests/experiment/test_cli.py tools/roots.toml docs/poorin/decisions.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Refuse calendar builds from uncommitted code (D107)" -m "--run checks the working tree first: a calendar-versioned artifact builds only from committed code, and a dev run saves its uncommitted changes as a snapshot commit that its records name. The plan prints the code's label and what --run would do. --note labels a run; --allow-uncommitted builds calendar artifacts from a snapshot, as the full check does."
```

---

### Task 6: W&B records the code label

**Files:**
- Modify: `lib/levanter/src/levanter/tracker/wandb.py` (`_git_run_config`)
- Test: `lib/levanter/tests/test_tracker.py`

**Interfaces:**
- Consumes: `current_launch`, `launched_as`, `Provenance.label` (Task 1).
- Produces: W&B run config keys `code` (always, when the checkout is known), `code_snapshot`, and
  `code_note` (only when set).

- [ ] **Step 1: Write the failing tests**

Add to `lib/levanter/tests/test_tracker.py` (import `dataclasses` and
`from rigging.provenance import Provenance, launched_as`):

```python
def test_wandb_run_config_labels_the_launch_code(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _dirty_checkout(source)
    monkeypatch.chdir(source)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    launch = dataclasses.replace(Provenance.capture(source), snapshot_commit="5c1d2e9aa0", note="try it")

    with launched_as(launch):
        initialized = _init_with_fake_wandb(monkeypatch, WandbConfig(save_code=str(source), background=False))

    config = initialized["config"]
    assert (config["code"], config["code_snapshot"], config["code_note"]) == (launch.label, "5c1d2e9aa0", "try it")


def test_wandb_run_config_labels_the_checkout_without_a_launch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _dirty_checkout(source)
    monkeypatch.chdir(source)
    monkeypatch.delenv("GIT_COMMIT", raising=False)

    initialized = _init_with_fake_wandb(monkeypatch, WandbConfig(save_code=str(source), background=False))

    assert initialized["config"]["code"] == Provenance.capture(source).label
    assert "code_snapshot" not in initialized["config"]


def test_wandb_run_config_ignores_a_launch_from_another_commit(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _dirty_checkout(source)
    monkeypatch.chdir(source)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    other = dataclasses.replace(Provenance.capture(source), base_commit="deadbee00", note="elsewhere")

    with launched_as(other):
        initialized = _init_with_fake_wandb(monkeypatch, WandbConfig(save_code=str(source), background=False))

    assert "code_note" not in initialized["config"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest lib/levanter/tests/test_tracker.py -n 0 -q -k "code"`
Expected: FAIL with KeyError `'code'`.

- [ ] **Step 3: Implement**

Change the import to `from rigging.provenance import Provenance, current_launch` and replace the
tail of `_git_run_config` (after the two early returns):

```python
    launch = current_launch()
    if launch is not None and launch.base_commit == provenance.base_commit:
        # The experiment command line's launch adds its snapshot and note to the same checkout.
        provenance = launch
    config: dict[str, Any] = {
        "git_commit": commit or provenance.base_commit,
        "git_dirty": provenance.dirty,
        "git_tree_hash": provenance.tree_hash,
        "code": provenance.label,
    }
    if provenance.snapshot_commit is not None:
        config["code_snapshot"] = provenance.snapshot_commit
    if provenance.note is not None:
        config["code_note"] = provenance.note
    return config
```

Update the docstring's first paragraph: "Only the commit, the dirty flag, the tree hash, and the
code's one-line label (with the launch's snapshot and note, when the experiment command line
installed one for this checkout) are recorded: ..."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest lib/levanter/tests/test_tracker.py -n 0 -q`
Expected: PASS, including `test_wandb_run_ignores_the_checkout_state_when_git_commit_names_another_commit`,
whose early return is unchanged.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "Record the run's code label in W&B"
git add lib/levanter docs/poorin/prune-metrics.csv
git commit -m "[levanter] Record the run's code label in W&B" -m "The W&B run config gains code, the one-line label of the code that ran, and the launch's snapshot and note when the experiment command line installed them for the same checkout, so the runs table can show which code each run used."
```

---

### Task 7: The run list

**Files:**
- Create: `lib/poorin/src/poorin/experiment/runs.py`
- Modify: `tools/roots.toml` (kept modules), `docs/poorin/architecture.md` (one row)
- Test: `tests/experiment/test_runs.py`

**Interfaces:**
- Consumes: `Provenance`, `read_git`, `save_snapshot`, `launched_as` (Tasks 1–2);
  `run(provenance=...)` (Task 3); `read_record`, `StatusFile`.
- Produces: `python -m poorin.experiment.runs list [FILTER] [--limit N] [--prefix P]` and
  `python -m poorin.experiment.runs show NAME@VERSION|PATH [--prefix P]`; the Click group `main`.

- [ ] **Step 1: Write the failing tests**

Create `tests/experiment/test_runs.py`:

```python
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
```

The `dev` version in `test_show_describes_a_run_from_a_snapshot` is a fixed `dev` (no suffix):
`apply` takes it as given, so the output is `data/toy/dev`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_runs.py -n 0 -q`
Expected: FAIL with ModuleNotFoundError for `poorin.experiment.runs`.

- [ ] **Step 3: Implement**

Create `lib/poorin/src/poorin/experiment/runs.py`:

```python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List the runs under the prefix and show the code behind one (D107).

    uv run --no-sync python -m poorin.experiment.runs list [FILTER] [--limit N]
    uv run --no-sync python -m poorin.experiment.runs show NAME@VERSION | PATH

Both read what the runner writes beside each output, the record (``.artifact.json``) and the
status file, so they need neither W&B nor the experiment's code. Git, when the commit is in this
repository, adds what a record does not hold: how far its branch has moved since, a snapshot's line
counts, and the subject of a commit recorded before subjects were.
"""

import dataclasses
import shlex
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath

import click
from rigging.filesystem.cluster_config import poorin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.provenance import Provenance, read_git

from poorin.execution.artifact import ArtifactRecord, read_record
from poorin.execution.step_status import STATUS_DEP_FAILED, STATUS_FAILED, STATUS_SUCCESS, StatusFile

# The file beside every output the runner scheduled; it marks an artifact's directory.
STATUS_FILENAME = ".executor_status"
# Prefix directories that hold scratch space, not artifacts.
SCRATCH_DIRS = frozenset({"tmp"})
DEFAULT_LIMIT = 30
# Continuation lines of `show` start under its values.
SHOW_INDENT = " " * 15
NO_CODE = "-"


@dataclass(frozen=True)
class Run:
    """One output under the prefix."""

    label: str
    """``name@version``."""
    path: str
    state: str
    """``built``, ``started``, or ``failed``."""
    record: ArtifactRecord | None
    """``None`` until the build finishes."""
    when: datetime


def _state(output_path: str) -> str:
    status = StatusFile(output_path, worker_id="runs").status
    if status == STATUS_SUCCESS:
        return "built"
    if status in (STATUS_FAILED, STATUS_DEP_FAILED):
        return "failed"
    return "started"


def _naive_local(moment: datetime) -> datetime:
    return moment.astimezone().replace(tzinfo=None) if moment.tzinfo is not None else moment


def _output_dirs(fs, directory: str, top: bool) -> Iterator[str]:
    """Directories under ``directory`` that hold an output; an output holds no other output."""
    entries = fs.ls(directory, detail=True)
    if any(PurePosixPath(entry["name"]).name == STATUS_FILENAME for entry in entries):
        yield directory
        return
    for entry in entries:
        name = PurePosixPath(entry["name"]).name
        if entry["type"] == "directory" and not (top and name in SCRATCH_DIRS):
            yield from _output_dirs(fs, entry["name"], top=False)


def collect_runs(prefix: str) -> list[Run]:
    """Every output under ``prefix``, newest first."""
    fs, root = url_to_fs(prefix)
    if not fs.exists(root):
        return []
    runs = []
    for directory in _output_dirs(fs, root, top=True):
        relative = PurePosixPath(directory).relative_to(PurePosixPath(root))
        path = f"{prefix.rstrip('/')}/{relative}"
        record = read_record(path)
        created = record.provenance.created_at if record is not None and record.provenance is not None else ""
        when = datetime.fromisoformat(created) if created else _naive_local(fs.modified(f"{directory}/{STATUS_FILENAME}"))
        runs.append(Run(f"{relative.parent}@{relative.name}", path, _state(path), record, when))
    return sorted(runs, key=lambda r: r.when, reverse=True)


def _commit(base_commit: str) -> str:
    """The full hash of ``base_commit`` when it is in this repository, else ""."""
    return read_git(["rev-parse", "--verify", "--quiet", f"{base_commit}^{{commit}}"]).strip()


def _with_subject(provenance: Provenance) -> Provenance:
    """``provenance`` with its commit's subject looked up when the record predates subjects."""
    if provenance.commit_subject or not provenance.base_commit:
        return provenance
    full = _commit(provenance.base_commit)
    subject = read_git(["log", "-1", "--format=%s", full]).strip() if full else ""
    return dataclasses.replace(provenance, commit_subject=subject)


def _code_label(run: "Run") -> str:
    if run.record is None or run.record.provenance is None:
        return NO_CODE
    return _with_subject(run.record.provenance).label


def _time(iso: str) -> str:
    return f"{datetime.fromisoformat(iso):%Y-%m-%d %H:%M}" if iso else "unknown"


def _commit_lines(provenance: Provenance) -> list[str]:
    full = _commit(provenance.base_commit)
    when = f", {_time(provenance.commit_time)}" if provenance.commit_time else ""
    branch = provenance.branch or "a detached HEAD"
    lines = [f"  Commit       {provenance.base_commit} on {branch}{when}"]
    if provenance.commit_subject:
        lines.append(f"{SHOW_INDENT}{provenance.commit_subject}")
    if not full:
        lines.append(f"{SHOW_INDENT}not in this repository")
        return lines
    if provenance.branch:
        moved = read_git(["rev-list", "--count", f"{full}..{provenance.branch}"]).strip()
        if moved == "0":
            lines.append(f"{SHOW_INDENT}{provenance.branch} has not moved since")
        elif moved:
            noun = "commit" if moved == "1" else "commits"
            lines.append(f"{SHOW_INDENT}{provenance.branch} has moved {moved} {noun} past it")
    return lines


def _line_counts(base_commit: str, snapshot: str) -> dict[str, str]:
    counts = {}
    for row in read_git(["diff", "--numstat", base_commit, snapshot]).splitlines():
        added, removed, path = row.split("\t", 2)
        counts[path] = "binary" if added == "-" else f"+{added} -{removed}" if removed != "0" else f"+{added}"
    return counts


def _uncommitted_lines(provenance: Provenance) -> list[str]:
    snapshot = provenance.snapshot_commit
    kept = _commit(snapshot) if snapshot else ""
    files = f"{len(provenance.uncommitted)} files" if provenance.uncommitted else "changes"
    if snapshot is None:
        saved = "not saved"
    else:
        saved = f"saved as snapshot {snapshot[:7]}" + ("" if kept else " (not in this repository)")
    lines = [f"  Uncommitted  {files}, {saved}"]
    counts = _line_counts(provenance.base_commit, kept) if kept else {}
    width = max((len(entry) - 3 for entry in provenance.uncommitted), default=0)
    for entry in provenance.uncommitted:
        status, path = entry[:2].strip(), entry[3:]
        lines.append(f"{SHOW_INDENT}  {status:<3} {path:<{width}}  {counts.get(path, '')}".rstrip())
    if kept:
        short = snapshot[:7]
        lines.append(f"  See changes  git diff {provenance.base_commit} {short}")
        lines.append(f"  Restore      git worktree add ../run-{short} {short}")
    return lines


def describe(run: Run) -> list[str]:
    """Everything the record and Git say about one run's code."""
    lines = [f"{run.label}   {run.state}"]
    if run.record is None or run.record.provenance is None:
        lines.append("  Code         not recorded yet: the record is written when the build finishes")
        return lines
    provenance = _with_subject(run.record.provenance)
    lines.append(f"  Built        {_time(provenance.created_at)} by {provenance.built_by or 'unknown'}")
    if provenance.command_line:
        lines.append(f"  Command      {shlex.join(provenance.command_line)}")
    if provenance.note:
        lines.append(f"  Note         {provenance.note}")
    if not provenance.base_commit:
        lines.append("  Commit       none: not built from a Git checkout")
    else:
        lines.extend(_commit_lines(provenance))
        if provenance.dirty:
            lines.extend(_uncommitted_lines(provenance))
    if run.record.deps:
        lines.append(f"  Inputs       {', '.join(run.record.deps)}")
    return lines


@click.group()
def main() -> None:
    """List the runs under the prefix, and show the code behind one."""


@main.command("list")
@click.argument("name_filter", required=False, default="")
@click.option("--limit", type=click.IntRange(min=1), default=DEFAULT_LIMIT, show_default=True)
@click.option("--prefix", default=None, help="Read this prefix instead of POORIN_PREFIX.")
def list_runs(name_filter: str, limit: int, prefix: str | None) -> None:
    """One line per run, newest first; FILTER keeps runs whose name contains it."""
    runs = [r for r in collect_runs(prefix or poorin_prefix()) if name_filter in r.label][:limit]
    width = max((len(r.label) for r in runs), default=0)
    for r in runs:
        click.echo(f"{r.when:%Y-%m-%d %H:%M}  {r.state:<7}  {r.label:<{width}}  {_code_label(r)}")


@main.command()
@click.argument("target")
@click.option("--prefix", default=None, help="Read this prefix instead of POORIN_PREFIX.")
def show(target: str, prefix: str | None) -> None:
    """Everything about one run's code. TARGET is NAME@VERSION or the output's path."""
    if "@" in target:
        name, _, version = target.rpartition("@")
        path = f"{(prefix or poorin_prefix()).rstrip('/')}/{name}/{version}"
    else:
        path = target.rstrip("/")
    record = read_record(path)
    fs, stripped = url_to_fs(path)
    if record is None and not fs.exists(f"{stripped}/{STATUS_FILENAME}"):
        raise click.ClickException(f"no run at {path}")
    relative = PurePosixPath(path)
    label = target if "@" in target else f"{relative.parent.name}@{relative.name}"
    click.echo("\n".join(describe(Run(label, path, _state(path), record, datetime.now()))))


if __name__ == "__main__":
    main()
```

A path `TARGET` shows `<parent>@<version>` (the last two segments), which is enough to identify
it beside the full path the user typed. `when` is unused by `describe`; `datetime.now()` fills it.

In `tools/roots.toml`, add `"poorin.experiment.runs",` after `"poorin.experiment.sweep",` in the
kept modules. In `docs/poorin/architecture.md`, add after the `poorin.experiment.overrides` row:

```text
| `poorin.experiment.runs` | Lists the runs under the prefix and shows the code behind one | `poorin.experiment.runs` |
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run --no-sync pytest tests/experiment/test_runs.py tests/tools -n 0 -q`
Expected: PASS. Then run it against the real prefix:
`uv run --no-sync python -m poorin.experiment.runs list` and
`uv run --no-sync python -m poorin.experiment.runs show <a name@version it lists>`.

- [ ] **Step 5: Lint, full check, commit**

```bash
uv run --no-sync python -m tools.lint --fix
uv run --no-sync python -m tools.prune_check --phase D107 --step "List runs and show the code behind each"
git add lib/poorin tests/experiment/test_runs.py tools/roots.toml docs/poorin/architecture.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] List runs and show the code behind each" -m "python -m poorin.experiment.runs lists the outputs under the prefix with each run's one-line code label, and shows one run in full: the command, the note, the commit and how far its branch has moved, and the uncommitted files with the commands that show and restore a snapshot."
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/explanations/lazy-artifacts.md`, `AGENTS.md`, `README.md`,
  `docs/poorin/specs/2026-10-11-run-code-provenance-design.md` (status line)

**Interfaces:**
- Consumes: the behavior of Tasks 1–7. No code.

- [ ] **Step 1: Add the explanation**

In `docs/explanations/lazy-artifacts.md`, add this section after "Recipe drift":

````markdown
## Code provenance

The fingerprint covers an artifact's config, not the code that ran it. Each output's record names
the code instead: the commit, its subject, the branch, and any uncommitted changes.

Before `--run` builds anything, the command line checks the working tree, untracked files
included. A calendar-versioned artifact builds only from committed code: if one would build from a
dirty tree, the run stops and lists the artifacts and the uncommitted paths. A run that builds only
`dev` artifacts saves the uncommitted changes as a snapshot commit under `refs/poorin/snapshots/`,
leaving the branch, index, and files alone, and the record names the snapshot.
`--allow-uncommitted` builds calendar artifacts from a snapshot as well. A run that builds nothing
saves no snapshot and is never refused.

The plan prints the code's one-line label and what `--run` would do:

```text
Prefix: /data/poorin
Code:   main@a0a406f + 3 uncommitted: llama.py, model.py, notes.md
        --run saves a snapshot
```

`--note TEXT` puts your own words in the label in place of the commit subject or file list. The
label also reaches W&B as the run config's `code` field.

`python -m poorin.experiment.runs list [FILTER]` lists the runs under the prefix, newest first,
one line each. `python -m poorin.experiment.runs show NAME@VERSION` shows one run's code in full,
with `git diff` and `git worktree add` commands for its snapshot. Snapshot refs are not pushed by
`git push origin main`; `git push origin 'refs/poorin/*:refs/poorin/*'` backs them up.
````

- [ ] **Step 2: Update AGENTS.md and README.md**

In `AGENTS.md`, under `## Development`, add to the bullet list:

```markdown
- Commit before running a calendar-versioned experiment: `--run` refuses to build one from
  uncommitted code (D107). A dev run saves uncommitted code as a snapshot commit.
  `python -m poorin.experiment.runs show NAME@VERSION` shows the code behind a run.
```

In `README.md`, after the paragraph that ends "explains versions, caching, overrides, and drift.",
add:

```markdown
A calendar-versioned artifact builds only from committed code; a dev run saves uncommitted code as
a snapshot commit. `python -m poorin.experiment.runs list` shows each run with a one-line label of
its code.
```

In the spec, change the status line to:
`Status: approved design, 2026-10-11; implemented 2026-10-11 (plan
`plans/2026-10-11-run-code-provenance.md`). Decision D107.`

- [ ] **Step 3: Check the docs build and links**

Run: `uv run --no-sync python -m tools.lint` (Markdown and the architecture test run through it and
the affected tests). Expected: OK.

- [ ] **Step 4: Full check and commit**

```bash
uv run --no-sync python -m tools.prune_check --phase D107 --step "Document code provenance for runs"
git add docs/explanations/lazy-artifacts.md AGENTS.md README.md docs/poorin/specs/2026-10-11-run-code-provenance-design.md docs/poorin/prune-metrics.csv
git commit -m "[poorin] Document code provenance for runs" -m "lazy-artifacts.md explains the code check, snapshots, notes, and the run list; AGENTS.md and the README say to commit before a calendar run."
```

---

## After the plan

Not a task for an implementer: once the branch is merged, the controller updates the Poorin Field
Guide artifact (the reproducibility note, the "Will it rebuild?" explorer's code warning, and the
command sheet) to describe the gate, `--note`, and `poorin.experiment.runs`.
