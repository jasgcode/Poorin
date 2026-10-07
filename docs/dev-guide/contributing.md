# Contributing to Marin

We welcome contributions that add missing features or fix serious issues. If you
are unsure whether a change is wanted, open an issue or ask in the
[Marin Discord](https://discord.gg/J9CTk7pqcM) before sending a PR. We are
unlikely to merge typo fixes, stylistic rewrites, or speculative refactors that
are not tied to an issue.

If you are new to Marin, ask for code review in the `#code-review` channel of
the [Marin Discord](https://discord.gg/J9CTk7pqcM). If you are unable to join
the Discord, tag a Marin maintainer in the issue or pull request.

## AI-generated contributions

We use coding agents ourselves and accept contributions made with them. We do
not accept drive-by PRs that an agent produced in one shot.

The rule of thumb: if your agent can do it in one shot, so can ours, so please
do not burden us with the PR. A throwaway agent-generated change costs us more to
review than it cost you to file. If you do contribute agent-assisted work, hold
it to the same bar as the rest of this guide: confirm the change is needed, that
it is correct, and that you can explain why it matters.

## Setup

Complete the [installation guide](../tutorials/installation.md), then install the Git hook that runs `tools/lint.py` before each commit:

```bash
cat <<'EOF' > .git/hooks/pre-commit
#!/bin/sh
set -e
cd "$(git rev-parse --show-toplevel)"
uv run --no-sync python -m tools.lint --staged --fix
EOF
chmod +x .git/hooks/pre-commit
```

### Linting

The Git hook configured above lints the staged change before each commit. Lint every file with
`uv run --no-sync python -m tools.lint --all-files --fix`.

### Testing

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

### Opening a pull request

Before opening a pull request:

1. Run `uv run --no-sync python -m tools.lint --all-files --fix`.
2. Run `uv run --no-sync python -m tools.affected_tests`.
3. If your change is docs-heavy, run `uv run mkdocs build --strict`.
4. If your change adds or rewrites substantial prose, do a final prose-only review using `./.agents/skills/writing-style/SKILL.md`. Remove generic significance framing, stock AI-writing templates, and polished filler that does not add information.
5. Follow `./.agents/skills/writing-style/pull-requests.md` for the PR title and body. The body becomes the squash-merge commit message: lead with what changes, then why, and retain the evidence and caveats a future reader needs.
6. If the work came from an issue, end the PR body with `Fixes #NNNN` or `Part of #NNNN`.
7. After pushing, verify the relevant GitHub CI checks pass before considering the PR ready for review.

## Guidelines

Please see the [guidelines](../explanations/guidelines.md) for principles and practices for Marin.
