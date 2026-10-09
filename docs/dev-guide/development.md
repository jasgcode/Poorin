# Development

Poorin has one maintainer and no CI, so the checks below are the gate for every change. The
project's conventions for code, tests, and commits are in `AGENTS.md` and `TESTING.md` at the
repository root.

## Setup

Complete the [installation guide](../tutorials/installation.md), then install the Git hook that
runs `tools/lint.py` before each commit:

```bash
cat <<'HOOK' > .git/hooks/pre-commit
#!/bin/sh
set -e
cd "$(git rev-parse --show-toplevel)"
uv run --no-sync python -m tools.lint --staged --fix
HOOK
chmod +x .git/hooks/pre-commit
```

Run every project command with `uv run --no-sync`. A syncing `uv run` installs no extras, so it
removes the CUDA build of JAX; sync the environment from the lockfile instead:
`uv sync --locked --all-packages --extra=gpu --extra=lm_eval`.

## Linting

The Git hook lints the staged change before each commit. Lint every file with
`uv run --no-sync python -m tools.lint --all-files --fix`.

## Testing

Run a narrow test while editing, then run all safe tests affected by the uncommitted
change:

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest <relevant test paths> -n 0
uv run --no-sync python -m tools.affected_tests
```

`pyproject.toml` already excludes the slow, integration, data-integration,
live-cluster, Docker, and manual markers by default. Do not pass `-m 'not slow'`:
`-m` replaces the whole default expression, so it re-selects the cluster and
Docker tests it looks like it is narrowing.

The root pytest configuration loads `poorin.pytest_timeout_guard`. Keep that package
importable in any selected test environment. The guard dumps thread state and starts a
delayed hard-kill timer when a signal-based timeout cannot stop a test; an import failure
is a test-environment error, not a product-test failure.

## Before each commit

1. Run the full check with the change uncommitted:
   `uv run --no-sync python -m tools.prune_check --phase <phase> --step "<label>"`. It runs ty,
   the linter, the tests the change affects, and both training roots on the GPU, then appends
   one row to `docs/poorin/prune-metrics.csv`; commit that row with the change. It picks tests
   from uncommitted changes, so on a clean tree it runs none.
2. If the change is docs-heavy, run `uv run --no-sync mkdocs build --strict`.
3. If the change adds or rewrites substantial prose, do a final prose-only review using
   `./.agents/skills/writing-style/SKILL.md`.
4. Write the commit message following `./.agents/skills/writing-style/pull-requests.md`: lead with
   what changes, then why, and keep the evidence and caveats a future reader needs.
5. Record each decision that chooses between options or changes scope in
   `docs/poorin/decisions.md`, in the same commit.

Before merging a branch, run the tests its commits affect with
`uv run --no-sync python -m tools.affected_tests --base main`.

## Guidelines

The [guidelines](../explanations/guidelines.md) describe the principles and practices Poorin
inherits from Marin.
