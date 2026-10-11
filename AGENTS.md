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
- `lib/poorin/AGENTS.md` — poorin (experiment framework: artifacts, pipelines, evaluation)
- `lib/zephyr/AGENTS.md` — Zephyr (dataset processing)
- `lib/fray/AGENTS.md` — fray (job dispatch)
- `lib/finelog/AGENTS.md` — finelog (Finelog's Python client)

### Renamed libraries

Poorin renamed Marin's core package to `poorin` and gave the other libraries it cut heavily `poorin-`
distribution names, keeping their import names (D85, D86); each one's README lists what was removed.
Upstream Marin code, and Poorin's specs, plans, and decisions before D85, use the old names:

| Upstream name (import / distribution) | Poorin name |
|---|---|
| `marin` / `marin-core` | `poorin` / `poorin` |
| `fray` / `marin-fray` | `fray` / `poorin-fray` |
| `finelog` / `marin-finelog` | `finelog` / `poorin-finelog` |
| `rigging` / `marin-rigging` | `rigging` / `poorin-rigging` |

Levanter, Zephyr, Finestore, and Dupekit keep upstream's names. Haliax is not in the tree: it is
the published `marin-haliax`, pinned at the fork point (D102). The settings took Poorin
names too (D87): `MARIN_PREFIX` is `POORIN_PREFIX`, and so on. Names that point at something of
Marin's keep theirs, such as the license header and the Marin tokenizer's chat template.

## Workflow Playbooks

Skills are task-focused playbooks in `.agents/skills/` (also `.claude/skills/`). Before starting
a non-trivial task, check whether a skill matches; if one does, follow it.

## Development

```bash
# Lint and format the uncommitted change (--all-files for the whole tree)
uv run --no-sync python -m tools.lint --fix

# Type checking: errors and warnings fail (tools.lint also runs it when Python files change)
uvx ty@0.0.84 check --error-on-warning

# Unit tests affected by the uncommitted change (--all for every suite)
uv run --no-sync python -m tools.affected_tests

# During the prune: every check, both training roots, and one metrics row
uv run --no-sync python -m tools.prune_check --phase P<n> --step "<short label>"
```

- Run every project command with `uv run --no-sync`. A syncing `uv run` installs no extras, so
  it removes the CUDA build of JAX. Sync the environment explicitly from the lockfile:
  `uv sync --locked --all-packages --extra=gpu --extra=lm_eval` (the `lm_eval` extra makes the
  lm-eval harness runnable).
- Unit tests run on CPU: `tools.affected_tests` sets `JAX_PLATFORMS=cpu`. Run one test file
  in-process with `-n 0`; run GPU tests the same way, without `JAX_PLATFORMS=cpu`.
- Do not pass a partial marker expression such as `-m "not slow"`: it replaces the default
  expression and selects cluster and Docker tests. `tools.affected_tests` passes a complete one.
- Python >= 3.12.
- Dependencies may be upgraded freely. Each dependency change is its own commit, and its
  `uv.lock` diff is checked for versions that moved without being asked.
- Avoid large transfers between cloud regions or from the open internet; storage egress costs
  money.
- Commit before running a calendar-versioned experiment: `--run` refuses to build one from
  uncommitted code (D107). A dev run saves uncommitted code as a snapshot commit.
  `python -m poorin.experiment.runs show NAME@VERSION` shows the code behind a run.

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
2. Check the libraries: `lib/poorin/src/poorin/`, `lib/levanter/src/levanter/`, `lib/rigging/src/rigging/`
3. Check `pyproject.toml` for available third-party packages before adding new ones

If a suitable implementation exists, use it. Do not create parallel implementations.

Dependency direction: `haliax` → {`levanter`, `zephyr`} → `poorin`. Each layer may only import
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
