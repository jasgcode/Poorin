# Library renames: Poorin's lighter forks

Status: approved design, 2026-10-07; implemented. Plan: `plans/2026-10-07-library-renames.md`.
Decision: D85. Amended 2026-10-08 by D86: `fray`, `finelog`, and `rigging` keep their import names
and directories; only their distributions carry the `poorin-` prefix.

## Goal

The workspace libraries that Poorin cut heavily stop carrying upstream Marin's names, so code, the
lockfile, and the docs show at a glance which libraries are Poorin's lighter forks. Each renamed
library states what it lost and why.

## Scope

| Today: import / distribution / directory | After |
|---|---|
| `fray` / `marin-fray` / `lib/fray` | `poorin_fray` / `poorin-fray` / `lib/poorin_fray` |
| `finelog` / `marin-finelog` / `lib/finelog` | `poorin_finelog` / `poorin-finelog` / `lib/poorin_finelog` |
| `rigging` / `marin-rigging` / `lib/rigging` | `poorin_rigging` / `poorin-rigging` / `lib/poorin_rigging` |
| `marin` / `marin-core` / `lib/marin` | `poorin` / `poorin` / `lib/poorin` |

The workspace root project `marin-root` becomes `poorin-root`.

Unchanged: Levanter, Haliax, Zephyr, Finestore, and Dupekit (their cores are intact); environment
variables (`MARIN_PREFIX` and the other `MARIN_*`) and helper names such as `marin_prefix()` and
`marin_temp_bucket()`; `experiments/marin_tokenizer.py`; `gs://marin-*` bucket paths; upstream's
`marin-finelog-server` wheel and its `finelog_server` module; the protobuf package names
`finelog.stats` and `finelog.logging`, which are part of the wire protocol the pinned server speaks.
Renaming the variables and helpers is a separate follow-up.

## What each rename rewrites

1. The directory (`git mv lib/<old> lib/<new>`) and the package directory under its `src/`.
2. Its `pyproject.toml`: distribution name, wheel and sdist package paths, the `fsutil` console
   script (rigging), and isort's first-party name (marin).
3. Every reference in tracked files, except the historical records named below:
   - Python `import <old>` and `from <old> ...` statements;
   - dotted references `<old>.<submodule>` where `<submodule>` is a real first-level module or
     package of the library (strings, monkeypatch targets, `tools/roots.toml`, MkDocs `:::`
     directives, docs, skills). URLs and other tokens such as `marin.readthedocs` or
     `finelog.stats` do not match;
   - paths `lib/<old>` and `src/<old>`;
   - distribution names `marin-<x>` in manifests and docs, never `marin-finelog-server` or
     `marin-dupekit-native`;
   - backticked package names in Markdown, such as `` `rigging` ``.
4. The root manifest: workspace members, `[tool.uv.sources]`, ty's source paths, isort's
   third-party list, the pytest `-p` plugin path; `tools/lint.py`'s per-library ruff
   configuration path.
5. `uv.lock`, regenerated; the venv re-synced with `uv sync --locked --all-packages --extra=gpu`.

Historical records keep the names they were written with: `decisions.md` entries before D85, specs,
plans, and `prune-metrics.csv`. One note at the top of `decisions.md` points readers to the name
map in AGENTS.md.

## Cached artifacts

Artifact records (`.artifact.json`) store the artifact class by module path (`result_type`), and
loading an artifact whose record names another class raises `ArtifactTypeMismatchError`.
Fingerprints hash config values without class names, so the rename does not change them. In the
`marin` rename commit, a one-off command rewrites `"result_type": "marin.` to
`"result_type": "poorin.` in every record under the local output prefix (`~/.cache/poorin`). The
command is recorded in D85; it is not kept as code.

## Documentation

- Each renamed library's README opens with a section: it is Poorin's lighter fork of the upstream
  library, what was removed with line counts, and the decisions that record why.
- `docs/poorin/architecture.md` gains a table of forked libraries.
- AGENTS.md gains an old-to-new name map for reading upstream code and Poorin's history.

## Order and verification

One library per commit, smallest blast radius first: `fray`, `finelog`, `rigging`, then `marin`.
Each commit passes the full `tools.prune_check` (ty, lint, affected tests, both training roots,
Finelog) and leaves no import of the old name. The `marin` commit's check runs R1 and R2 against
the migrated tokenized caches.

## Rejected

- Renaming distributions only: cheap, but the user wants the fork visible at every call site.
- One commit for all four: harder to localize a failure.
- Aliases for the old names: AGENTS.md forbids backward-compatibility shims.
