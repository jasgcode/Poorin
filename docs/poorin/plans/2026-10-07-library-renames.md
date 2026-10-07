# Library Renames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `fray`, `finelog`, `rigging`, and `marin` to `poorin_fray`, `poorin_finelog`,
`poorin_rigging`, and `poorin`, and document each as Poorin's lighter fork.

**Architecture:** One library per commit. `git mv` moves the directories; a one-off rewrite script
(kept outside the repository) applies the spec's rewrite rules to tracked files; the manifest
details the script does not cover are edited by hand; `uv lock` and `uv sync` follow; the full
`prune_check` gates the commit.

**Spec:** `docs/poorin/specs/2026-10-07-library-renames-design.md`.

## Global Constraints

- Historical records keep old names: `decisions.md` entries before D85, specs, plans,
  `prune-metrics.csv`.
- Never rename `MARIN_*` environment variables, `marin_*` helpers, `marin-finelog-server`,
  `marin-dupekit*`, `finelog_server`, or the protobuf packages `finelog.stats` and
  `finelog.logging`.
- After each rename, `git grep -nE "^\s*(import|from) <old>\b"` over Python files finds nothing.
- Every project command uses `uv run --no-sync`; the only sync is
  `uv sync --locked --all-packages --extra=gpu` after each `uv lock`.

## Review Focus

1. Protobuf and Connect service names inside `finelog.rpc` (`/finelog.stats.StatsService/...`)
   must stay byte-identical, or the client cannot talk to the pinned server.
2. A string the dotted rule misses (a package-level attribute such as `fray.something` that is not
   a submodule): ty and the kept-root import step catch Python uses; the full check catches the
   rest.
3. Import order after the rename: isort sections change for `poorin*` names; `tools.lint --fix`
   re-sorts.
4. Cached tokenized datasets fail to load after the `marin` rename until their records are
   migrated: R1 and R2 load both caches.
5. Stale `__pycache__` directories left in the old `lib/<old>` paths: deleted after each move.

---

### Task 1–4: one rename each (`fray`, `finelog`, `rigging`, `marin`)

For `<old>` → `<new>` with distribution `<old_dist>` → `<new_dist>`:

- [ ] Move: `git mv lib/<old> lib/<new_dir>` then `git mv lib/<new_dir>/src/<old> lib/<new_dir>/src/<new>`;
  remove the untracked `__pycache__` directories left under `lib/<old>`.
- [ ] Rewrite: `python rename_lib.py <old> <new> <new_dir> <old_dist> <new_dist> --apply`
  (rules from the spec: import statements, `<old>.<submodule>` with real submodules only,
  `lib/<old>` and `src/<old>` paths, distribution names, backticked names in Markdown).
- [ ] Hand edits: the library's `pyproject.toml` name; the `fsutil` script target (rigging); the
  root `known-third-party` and the library's `known-first-party` (marin); the root project name
  `poorin-root` (marin task).
- [ ] `uv lock`, then `uv sync --locked --all-packages --extra=gpu`.
- [ ] `uv run --no-sync python -m tools.lint --fix`, then check no old imports remain.
- [ ] README section "Poorin's lighter fork" with what was removed (line counts against
  `upstream-baseline`) and the decisions that record it.
- [ ] marin task only: migrate artifact records under `~/.cache/poorin`
  (`"result_type": "marin.` → `"result_type": "poorin.`).
- [ ] `uv run --no-sync python -m tools.prune_check --phase Rename --step "Rename <old> to <new>"`.
- [ ] Commit with the metrics row.

### Task 5: documentation

- [ ] AGENTS.md: old-to-new name map; library manual paths.
- [ ] `architecture.md`: forked-libraries table (the module rows are rewritten by Tasks 1–4).
- [ ] `decisions.md`: a note at the top pointing to the name map.
- [ ] Commit; fast-forward `main`.
