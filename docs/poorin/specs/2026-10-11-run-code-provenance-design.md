# Code provenance for experiment runs

Status: approved design, 2026-10-11; implemented 2026-10-11 (plan
`plans/2026-10-11-run-code-provenance.md`). Decision D107.

## TL;DR

An artifact's fingerprint hashes its config, not the code that ran it. Every output already
records the commit it was built from and whether the working tree was dirty, but the uncommitted
code itself is lost, nothing stops a run on uncommitted code, and the record names the code by a
hash that means nothing at a glance.

This design makes the code behind every run recoverable and readable:

- `--run` refuses to build a calendar-versioned artifact from uncommitted code. A run that builds
  only `dev` artifacts goes ahead, and its uncommitted changes, untracked files included, are saved
  as a snapshot commit under `refs/poorin/snapshots/`. Branch, index, and working tree are left
  alone.
- Each run's code has a one-line label, such as `main@a0a406f  Trim the tutorial's devices to
  this workstation's two`, shown in the plan, after the run, in W&B, and in a run list.
- `--note TEXT` puts your own words in that label.
- `python -m poorin.experiment.runs` lists runs under the prefix, one line each, and shows one
  run's code in full: commit, subject, how far `main` has moved past it, the uncommitted files, and
  the commands that show and restore them.

## Background

### What is recorded today

`lazy.run()` captures a `rigging.provenance.Provenance` once per call and threads it through
lowering into every step's `.artifact.json`:

| Field | Value |
|---|---|
| `base_commit` | Short hash of `HEAD` |
| `dirty` | Whether `git status --porcelain --untracked-files=all` lists anything |
| `tree_hash` | Short tree hash of `git stash create`, or of `HEAD` on a clean tree |
| `branch`, `built_by`, `git_remote`, `created_at`, `command_line` | Launch context |

Levanter's W&B tracker captures its own provenance and records `git_commit`, `git_dirty`, and
`git_tree_hash` in the run config.

### What is missing

- **Uncommitted code is not kept.** `git stash create` writes a commit that no ref points to, so
  `git gc` deletes it after two weeks, and it leaves out untracked files. A new grug variant
  directory is untracked until committed.
- **Nothing gates a run.** A calendar-versioned result, the kind cited in writing, can be built
  from code that exists nowhere else.
- **Hashes are unreadable.** A record names its code as `8991c952f`. Matching that to an
  intent means running `git log`.

## Rules

### Which runs are refused

Before `--run` builds anything, the experiment command line reads the working tree's status,
untracked files included, in the repository that contains the working directory.

| Tree | What the run would build | Result |
|---|---|---|
| Clean | Anything | Runs. The record names the commit. |
| Dirty | At least one calendar-versioned artifact | Refused before anything runs. |
| Dirty | Only `dev` artifacts | Runs. The uncommitted changes are saved as a snapshot. |

"Would build" is the runner's own schedule: a calendar artifact whose output has no `SUCCESS`
status and is not pruned behind a built consumer. A dev run whose calendar dataset is already
built is not refused. The first run on an empty prefix is, because it would build that dataset
from uncommitted code.

The refusal lists the calendar artifacts that would build and up to ten uncommitted paths:

```text
2 calendar-versioned artifacts would build from uncommitted code:
  tinystories@2026.06.28
  checkpoints/tiny-tinystories-cpu@2026.10.11
Uncommitted (3): M experiments/llama.py, ?? experiments/grug/moe/model.py, M docs/notes.md
Commit the changes and run again, run a dev version, or pass --allow-uncommitted to build from a
snapshot.
```

`--allow-uncommitted` lets a dirty tree build calendar artifacts. The run saves a snapshot as a
dev run does, so the code stays recoverable, and the record's `dirty` flag and snapshot show that
it was used. `tools.prune_check` passes it, because the full check runs before each commit by
design, and a commit that changes a dataset's version must build that dataset during its check.

Outside a Git checkout, or without a `git` binary, the code state is unknown: calendar builds are
refused unless `--allow-uncommitted` is passed, and dev runs proceed without a snapshot.

A run that would build nothing, because every artifact is served from its cache, saves no
snapshot and is never refused.

The plan (no `--run`) applies the same check and reports what `--run` would do, so a refusal
shows before the run it would stop.

### The snapshot

A snapshot is an ordinary commit built through a temporary index:

```bash
GIT_INDEX_FILE=<tmp> git read-tree HEAD
GIT_INDEX_FILE=<tmp> git add --all          # tracked changes, deletions, untracked files
GIT_INDEX_FILE=<tmp> git write-tree         # -> tree
git commit-tree <tree> -p HEAD -m <message> # -> snapshot commit
git update-ref refs/poorin/snapshots/<commit> <commit>
```

- `git add --all` respects `.gitignore`, so data, caches, and the virtual environment stay out.
- The real index, `HEAD`, the branch, the stash, and the working tree are untouched.
- The message is the note when one is given, else a file summary (`3 uncommitted: llama.py,
  model.py, notes.md`). The author and committer dates are fixed to `HEAD`'s committer date, so
  the same tree, parent, and message always produce the same commit: repeated runs of one dirty
  state, such as the full check's four training runs, share one snapshot and one ref.
- The ref keeps the commit from `git gc`. Refs under `refs/poorin/` are not pushed by
  `git push origin main`; `git push origin 'refs/poorin/*:refs/poorin/*'` backs them up.
- A repository with no commits has nothing to snapshot against, so it counts as outside a checkout.

### The note

`--note TEXT` is optional and applies to every trial of the invocation, at any version and on a
clean or dirty tree. It is recorded in the provenance, becomes the snapshot's commit message, and
replaces the commit subject or file list in the one-line label.

## What a run records

`Provenance` gains five fields. All default to empty, so records written before this change load
unchanged.

| Field | Value |
|---|---|
| `commit_subject` | First line of `HEAD`'s message |
| `commit_time` | `HEAD`'s committer date, ISO 8601 |
| `uncommitted` | `git status --porcelain` lines, such as `M experiments/llama.py` and `?? experiments/grug/moe/model.py` |
| `snapshot_commit` | Full hash of the snapshot, or `None` |
| `note` | The `--note` text, or `None` |

Storing the subject and file list in the record lets the label render without Git, for example
when reading a prefix on another machine.

### One provenance per launch

The command line builds the launch's provenance once: capture, then snapshot and note. It passes
it to `run()` through a new keyword, `run(..., provenance=...)`, and installs it for the run with
the context manager `rigging.provenance.launched_as()`, so `launch_provenance()` (already used by
`artifact.write_artifact`) and the W&B tracker read the same one. Called without `provenance`,
`run()` captures a fresh one, as today, so a notebook that commits between runs records each
commit. The runner hook `BuildRunner` (D68) gains the `provenance` argument.

## The one-line label

`Provenance.label` renders the code in one line, at most about 100 characters; `__str__` returns
it.

```text
main@a0a406f  Trim the tutorial's devices to this workstation's two
main@a0a406f  try rmsnorm eps 1e-5
main@a0a406f + 3 uncommitted: llama.py, model.py, notes.md  (snapshot 5c1d2e9)
main@a0a406f + 3 uncommitted  try rmsnorm eps 1e-5  (snapshot 5c1d2e9)
detached@a0a406f  Trim the tutorial's devices to this workstation's two
no git checkout
```

- The subject or note is cut with `…` to fit. File summaries name at most three files by base
  name, then `, …`.
- A record from before this change has no subject. The run list looks it up with `git log` when
  the commit is in the repository, else shows the bare `main@a0a406f`.

It appears in four places:

- **The plan**, above the artifact list. A dirty tree adds what `--run` would do:

  ```text
  Prefix: /data/poorin
  Code:   main@a0a406f + 3 uncommitted: llama.py, model.py, notes.md
          --run saves a snapshot
    built        f4a8574e  tinystories@2026.06.28
    mutable      c3eb3321  users/researcher/checkpoints/tiny-tinystories-cpu@…-dev
  ```

- **After `--run`**, once, with `git diff <commit> <snapshot>` when a snapshot was saved.
- **W&B**: the tracker adds `code` (the label), `code_snapshot`, and `code_note` to the run config,
  under the same guard `_git_run_config` applies to `git_dirty` today: only when the launch's base
  commit matches the commit the tracker read.
- **The run list** (below).

## The run list: `python -m poorin.experiment.runs`

A Click group with two commands. Both read `.artifact.json` records under `poorin_prefix()`
(or `--prefix`), skipping `tmp/`, and need neither W&B nor the experiment code.

### `runs list [FILTER] [--limit N]`

One line per record, newest first by `created_at`, at most `N` (default 30). `FILTER` keeps
records whose name contains it.

```text
2026-10-11 09:44  built    checkpoints/tiny-tinystories-rtx5090x1@20261011-094416-eeb9-dev  main@a0a406f  Trim the tutorial's…
2026-10-11 09:41  built    checkpoints/tiny-tinystories-cpu@20261011-094058-aea9-dev        main@8991c95  Say that dev versions…
2026-10-11 09:40  built    tinystories@2026.06.28                                           main@8991c95  Say that dev versions…
```

The state comes from the output's status file: `built`, `started`, or `failed`.

### `runs show NAME@VERSION | PATH`

Everything about one run's code. `NAME@VERSION` resolves to `{prefix}/{name}/{version}`.

```text
checkpoints/tiny-tinystories-rtx5090x1@20261011-094416-eeb9-dev   built
  Built        2026-10-11 09:44 by researcher
  Command      experiments/tutorials/train_tiny_model.py --device rtx5090x1 --version dev --run
  Note         try rmsnorm eps 1e-5
  Commit       a0a406ff4 on main, 2026-10-11 09:50
               Trim the tutorial's devices to this workstation's two (D106)
               main has moved 4 commits past it
  Uncommitted  3 files, saved as snapshot 5c1d2e9
                 M   experiments/llama.py               +12 -3
                 ??  experiments/grug/moe/model.py      +210
                 M   docs/notes.md                      +1 -1
  See changes  git diff a0a406ff4 5c1d2e9
  Restore      git worktree add ../run-5c1d2e9 5c1d2e9
  Inputs       tinystories@2026.06.28
```

- "main has moved N commits past it" is `git rev-list --count <commit>..main`. When the commit
  is not in the repository, the line says so.
- Line counts come from `git diff --numstat <commit> <snapshot>` when the snapshot exists, else
  the record's status list is shown without counts.
- The restore command uses a worktree, so the current checkout is left alone.

## Changes by file

| File | Change |
|---|---|
| `lib/rigging/src/rigging/provenance.py` | Five fields; `capture` fills the subject, time, and status list; `label`; `save_snapshot(message, repo_dir) -> str`; `launched_as` |
| `lib/poorin/src/poorin/execution/lazy.py` | `run(..., provenance=None)` |
| `lib/poorin/src/poorin/execution/step_runner.py` | The scheduling walk exposed as a function returning the steps a run would build, shared by the runner and the gate |
| `lib/poorin/src/poorin/experiment/cli.py` | `--note`, `--allow-uncommitted`, the gate, the snapshot, the `Code:` lines; `BuildRunner` takes `provenance` |
| `lib/poorin/src/poorin/experiment/runs.py` | New: `list` and `show` |
| `lib/levanter/src/levanter/tracker/wandb.py` | `code`, `code_snapshot`, `code_note` in the run config |
| `tools/prune_check.py` | Passes `--allow-uncommitted` to the training roots |

## Tests

- **Snapshot**, in a temporary repository: tracked edits, staged edits, deletions, and untracked
  files are in the snapshot; ignored files are not; `HEAD`, the index, the working tree, and the
  stash list are unchanged; the ref exists; two snapshots of one state are one commit; a
  repository with no commits raises.
- **Label**: clean, note, dirty with snapshot, dirty with note, detached `HEAD`, no checkout, and
  truncation of a long subject and of a long file list. A record from before this change loads
  and labels.
- **Gate**, through the command line in a temporary repository with toy artifacts: a dirty tree
  refuses a calendar build and builds nothing; it allows a dev run whose calendar input is built,
  and that record carries the snapshot and the ref exists; `--allow-uncommitted` builds the
  calendar artifact and records the snapshot; a clean tree records no snapshot; a calendar
  artifact pruned behind a built consumer does not trigger the refusal; the plan prints the
  `Code:` line and the refusal it predicts; `--note` reaches the record and the label.
- **Run list**: ordering, the filter, the limit, `show` on a record with a snapshot (the diff and
  restore commands, line counts) and on one from before this change.
- **W&B config**: `code`, `code_snapshot`, and `code_note` appear when the launch provenance
  matches the tracker's commit, and are left out when it does not.
- **Existing command-line tests** that pass `--run` with a calendar version run in a clean
  temporary repository (a fixture that commits nothing and changes the working directory), since
  the developer's tree is dirty whenever tests run before a commit.

## Documents

- `docs/explanations/lazy-artifacts.md`: a "Code provenance" section covering the rule, snapshots,
  notes, and the run list.
- `AGENTS.md`: commit before running a calendar-versioned experiment; read a run's code with
  `poorin.experiment.runs show`.
- `docs/poorin/decisions.md`: D107.
- The Poorin Field Guide artifact: the reproducibility note and the command sheet.

## Out of scope

- Code in other repositories, such as a future `projects/` checkout (D52): the check reads the
  repository that contains the working directory.
- Pruning old snapshots. They are small, because Git stores each file version once; `git
  for-each-ref refs/poorin/snapshots` lists them and `git update-ref -d` deletes one.
- Large untracked files that are not ignored are copied into `.git` by the snapshot. Ignore them.
- Comparing two runs' code in one command; `git diff` between their commits or snapshots does it.
- Changing the fingerprint to cover code. A version bump stays the author's decision (lazy
  artifacts' design), and the record now says which code built each output.
