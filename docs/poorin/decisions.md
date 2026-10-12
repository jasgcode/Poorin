# Decision log

Every decision that shaped Poorin, in the order it was made: what was decided, why, what else
was considered, and what it costs if it turns out wrong. The spec
(`specs/2026-10-04-core-prune-design.md`) and the phase plans (`plans/`) hold the full designs;
commits hold the diffs. This log is the place to read why the tree looks the way it does.

Entries before D85 use upstream's library names (`marin`, `fray`, `finelog`, `rigging`); the
"Renamed libraries" table in AGENTS.md maps them to the `poorin` names.

## Adding an entry

- Add an entry when a change chooses between options, changes scope, or departs from a plan.
  That includes rulings made while executing a plan and findings from a final review that were
  fixed or deliberately left alone.
- Add it in the same commit as the change it governs. A decision made in a plan but not yet
  carried out has status **Planned**; when it lands, change the status and add the commit.
- Do not rewrite a past decision. When one is reversed, mark it **Superseded by D*n*** and add
  a new entry.
- Fields: **When** (date and phase), **Status**, **Decision**, **Why**, **Alternatives**
  (omitted when none was seriously considered), **Cost if wrong**, **Landed** (commit or
  document).

Statuses: **Adopted** (in the tree), **Planned** (in an approved or proposed plan), **Deferred**
(known, intentionally not done yet), **Superseded**.

---

## Design: 2026-10-04 to 2026-10-05

Made while writing the core prune spec, mostly in conversation with the owner. Recorded in the
spec's "Decisions" table and sub-project table.

### D1. Hard fork; upstream Marin is a reference only

- **When:** 2026-10-04, design. **Status:** Adopted.
- **Decision:** Poorin never merges from Marin. Code is deleted freely. The tag
  `upstream-baseline` marks `ab0da00df`, the tree the fork started from.
- **Why:** The goal is a repository one researcher understands end to end. Staying mergeable
  would mean keeping Marin's structure.
- **Cost if wrong:** Upstream fixes must be found and ported by hand.
- **Landed:** spec, "Decisions"; tag created in P0 Task 1.

### D2. Local RTX 5090 first; Modal and GCP TPU for larger runs

- **When:** 2026-10-04, design. **Status:** Adopted (dispatch itself is sub-project 2).
- **Decision:** Every workflow must run on the local RTX 5090. Larger jobs go to Modal or GCP
  TPU through sub-project 2.
- **Why:** Fast local iteration is the main use. Unattended long runs need a provider that keeps
  them alive.
- **Alternatives:** Colab through its CLI. Rejected for unattended runs, which Colab handles
  poorly.
- **Cost if wrong:** Sub-project 2 targets the wrong providers.
- **Landed:** spec, "Sub-project order".

### D3. Four model families; architecture search and scaling laws are the research

- **When:** 2026-10-04 to 2026-10-05, design. **Status:** Adopted.
- **Decision:** The data layer, model templates, and evaluation cover LLMs, video world models,
  VLAs, and world action models (WAMs). Online simulation data for EXPO-FT comes later
  (sub-project 6).
- **Why:** The research compares architectures and fits scaling laws across these families,
  where training recipes are not settled the way they are for LLMs.
- **Landed:** spec, "Project goals"; `1f2c1d0b5`.

### D4. Prune by reachability in two passes, then consolidate

- **When:** 2026-10-04, design. **Status:** Adopted.
- **Decision:** Pass 1 deletes everything outside the static import closure of a fixed set of
  roots, plus code deleted regardless (Iris, SkyRL glue, serving, fleet operations). Pass 2
  gives each surviving dependency a verdict (delete, make lazy, keep), one commit each.
  Sub-project 1b then consolidates packages and decides whether Haliax stays.
- **Why:** Pass 1 removes most lines mechanically. Pass 2 forces a reading of each dependency
  that survives. Consolidation is easier once the survivors are known.
- **Cost if wrong:** Static analysis misses modules loaded by name. Mitigated by listing the four
  plugin registries as roots, training two real runs before every commit, and searching for a
  module's dotted name before deleting it.
- **Landed:** spec, "Pass 1" and "Pass 2".

### D5. Delete Iris; keep two of its modules

- **When:** 2026-10-04, design. **Status:** Planned (P5).
- **Decision:** Delete Iris completely. Move `iris/jax/compile_cache.py` into Levanter and
  `iris/env_resources.py` into Zephyr.
- **Why:** Iris is a multi-tenant cluster scheduler: 84k lines of Python, 67k of tests, 3k of
  Rust, and 16k of TypeScript. One researcher needs machine provisioning, multi-host JAX setup,
  and preemption retry, which sub-project 2 provides.
- **Cost if wrong:** Multi-node GPU training would need a new coordinator discovery; the original
  stays at `upstream-baseline`.
- **Landed:** spec, "P5: Iris removal".

### D6. Delete Marin's in-repo SkyRL integration

- **When:** 2026-10-04, design. **Status:** Planned (P2).
- **Decision:** Delete `marin.rl` and `experiments/post_training`.
- **Why:** The owner maintains a customized SkyRL in a separate repository and will reconnect it
  to the dispatch from sub-project 2.
- **Landed:** spec, "Deleted regardless of reachability".

### D7. Two runnable roots; grug's base launcher becomes local-first

- **When:** 2026-10-04, design. **Status:** Adopted (P0).
- **Decision:** R1 is the tiny Levanter tutorial on wikitext. R2 is `experiments/grug/base/launch.py`
  itself, changed to train locally. No separate local launcher is added.
- **Why:** R1 exercises Levanter's trainer, Haliax models, tokenization, and the executor. R2
  exercises grug, the template new architectures are copied from; making the template itself
  local means every copy starts runnable.
- **Alternatives:** A separate spike launcher for local runs, leaving the base launcher on TPU.
- **Landed:** spec, "Runnable roots"; `80fff5ba6`.

### D8. Delete `infra/` and `scripts/` completely

- **When:** 2026-10-04, design. **Status:** Planned (P1).
- **Decision:** Delete both directories. Lint moves to a short setup Poorin owns.
- **Why:** The owner uses none of Marin's fleet tooling.
- **Landed:** spec, "Pass 1" row P1.

### D9. Keep a light LLM post-training kit

- **When:** 2026-10-04, design. **Status:** Adopted as kept roots.
- **Decision:** `train_dpo`, `lora_lm`, `lora_dpo`, and `sample_lm` stay as kept roots: they
  must import and type-check. A later sub-project makes them runnable.
- **Why:** The owner plans representation-learning experiments on models trained from scratch.
- **Landed:** spec, "Kept roots"; `tools/roots.toml`.

### D10. Keep Marin's evaluation framework; delete its fleet orchestration

- **When:** 2026-10-05, design. **Status:** Adopted as kept roots; deletions planned (P3).
  Superseded in part by D48: the Harbor and Evalchemy integrations are dropped.
- **Decision:** Keep the benchmark-harness integrations (lm-evaluation-harness, Evalchemy,
  Harbor), per-run records, the rollout table, statistics, `perplexity_gap`, and
  `trace_labeled_eval`. Delete the Iris-bound runner, serving configuration, and published-cohort
  policy.
- **Why:** The kept parts need only a model served behind an OpenAI-compatible URL. About 2k of
  the framework's 9.2k lines depend on Iris or the serving fleet.
- **Landed:** `29dfc0664`.

### D11. Delete 14 of Levanter's LLM architectures

- **When:** 2026-10-05, design. **Status:** Planned (P7).
- **Decision:** Keep Llama, Qwen, and GPT-2. Delete Mistral, Mixtral, Gemma, OLMo, OLMo 3,
  Whisper and the ASR model, Hyena, GPT-2 Hyena, ModernBERT, Apertus, Snowball, Qwen3 MoE, and
  Levanter MoE.
- **Why:** Grug is the modeling template. The deleted models remain at `upstream-baseline`.
- **Landed:** `29dfc0664`.

### D12. Evaluation splits into contracts (7a) and mechanisms (7b)

- **When:** 2026-10-05, design. **Status:** Adopted as the roadmap.
- **Decision:** Right after the prune, 7a defines one serving interface per model kind, a
  rollout schema of episodes and typed steps, and the evaluation record. 7b adds a local
  orchestrator and the simulator and video-metric suites, each with its model template.
- **Why:** Model templates (sub-project 4) are then built against an evaluation interface, and
  the scaling-law harness (sub-project 5) fits curves on one record format.
- **Landed:** `1f2c1d0b5`.

### D13. Equinox for models; pretrained Flax models through a bridge

- **When:** 2026-10-05, design. **Status:** Adopted; the bridge is sub-project 4.
- **Decision:** Models are written in Equinox, in grug's style. Pretrained Flax NNX models
  (openpi, big_vision) are used through a bridge that stores the NNX graph as a static Equinox
  field and its weights as a pytree field.
- **Why:** The training stack is already Equinox: Levanter's trainer, the optimizers, grug,
  fast-track, LoRA, the post-training kit, evaluation, and inference. Optax, Orbax, jaxtyping,
  and JAX sharding operate on Equinox pytrees. A probe trained an NNX module's weights inside an
  Equinox model with `eqx.filter_jit` and Optax (loss 1.20 to 0.00) and merged them back.
- **Alternatives:** Migrate the stack to Flax NNX, the owner's first preference.
- **Cost if wrong:** Releasing in Flax needs a port (sub-project 8).
- **Landed:** `cedd75604`.

### D14. Keep Finelog's Python client; pin the published server wheel

- **When:** 2026-10-05, design. **Status:** Planned (P6).
- **Decision:** Keep Finelog as a durable, SQL-queryable log and metrics store next to W&B. Keep
  its Python client and pin `marin-finelog-server==0.2.46.dev33324331711`. Delete its 69k lines
  of Rust source.
- **Why:** Finelog does not depend on Iris; Iris only hosted its server. A local server starts
  from a directory, and a probe wrote 300 typed rows and queried them back with SQL.
- **Cost if wrong:** If Marin stops publishing the wheel, the server must be built from
  `upstream-baseline` with `maturin`.
- **Landed:** `cedd75604`.

### D15. Release format: safetensors and a JSON config

- **When:** 2026-10-05, design. **Status:** Adopted for sub-project 8.
- **Decision:** Finished architectures are released as safetensors weights with a JSON config.
  Training checkpoints stay in Levanter's TensorStore format (Zarr v3 in OCDBT).
- **Why:** PyTorch, Flax, and other frameworks read safetensors without project code.
  TensorStore checkpoints also hold optimizer state and are written sharded and asynchronously.
- **Landed:** `d1bf49dc3`.

### D16. Dependencies may change freely, one change per commit

- **When:** 2026-10-05, design. **Status:** Adopted.
- **Decision:** `pyproject.toml` and `uv.lock` may be upgraded at any time. Each dependency
  change is its own commit, and its `uv.lock` diff is checked for versions that moved without
  being asked.
- **Why:** Marin's pins are not constraints for a hard fork, and Flax must move from 0.12.4,
  which fails to import under JAX 0.11.
- **Landed:** spec, "Risks".

### D17. Every prune commit trains both roots and records a metrics row

- **When:** 2026-10-04, design. **Status:** Adopted.
- **Decision:** Before each commit, `tools/prune_check.py` imports the kept roots, type-checks,
  lints, runs the affected tests, trains R1 and R2 twice on the 5090, and appends one row to
  `prune-metrics.csv`.
- **Why:** Every deletion stays measurable (lines, reachable lines, import cost, warm start), and
  breakage shows up in the commit that caused it.
- **Cost if wrong:** About 5 minutes per commit, and about 17 when the test selection covers every
  suite.
- **Landed:** spec, "Invariant"; `ea24a6e2f`.

---

## P0 setup: 2026-10-05

Plan: `plans/2026-10-05-p0-setup.md`. Commits `07dfeed16` to `75ea55308`, merged into local
`main` by fast-forward. Entries marked *(ruling)* were made during execution; *(review)* came
from the final review of the branch.

### D18. Delete Marin's CI workflows before anything else

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** Delete `.github/workflows` (37 workflows and a README) in the first commit.
- **Why:** The fork is public with GitHub Actions enabled, so the workflows would run against
  Poorin on the first push.
- **Landed:** `cffb19718`.

### D19. Reachability follows imports inside functions

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `tools/reachability.py` follows every import statement by default. `--eager`
  follows only statements that run at import time.
- **Why:** Lazily imported code must never look dead to Pass 1. The eager graph approximates
  startup cost for Pass 2.
- **Landed:** `1c86a081f`.

### D20. `--rank` removes one module at a time *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** Dominated weight removes exactly the ranked module; its submodules stay reachable
  through their own imports. Deleted packages in `roots.toml` still exclude their whole subtree.
- **Why:** Removing a package with its subtree credited `levanter`, `marin`, and other package
  `__init__` modules with every line below them.
- **Cost if wrong:** Pass 2's ranking would misstate what each cut saves.
- **Landed:** `1c86a081f`.

### D21. Deleted roots are not checked against existing modules *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** Patterns under `[deleted]` in `roots.toml` may name modules that no longer exist.
- **Why:** They go stale as each phase deletes its modules.
- **Cost if wrong:** A typo there leaves a module in the closure until someone notices.
- **Landed:** `1c86a081f`.

### D22. No `tests/tools/__init__.py` *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** The plan's `tests/tools/__init__.py` was dropped.
- **Why:** Under pytest's `--import-mode=importlib` it registers the test directory as a package
  named `tools`, which shadows the real `tools` package. `tests/experiment` and
  `tests/execution` have none either.
- **Landed:** `1c86a081f`.

### D23. Unit tests run on CPU without re-syncing the venv

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `prune_check` runs pytest with `uv run --no-sync` and `JAX_PLATFORMS=cpu`.
- **Why:** `infra/ci/run_tests.py` re-syncs `.venv` with CPU-only JAX, and dozens of xdist
  workers would each claim most of the GPU's memory.
- **Landed:** `ea24a6e2f`.

### D24. Lint changed files by explicit path *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted; superseded in mechanism by D39 when it lands.
- **Decision:** `prune_check` passes its own list of changed files to the linter.
- **Why:** `./infra/pre-commit.py --changed-files` skipped untracked files: it reported OK while
  ruff and black failed on new files.
- **Landed:** `ea24a6e2f`.

### D25. pyrefly checks `tools/` *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `tools/**/*.py` joins pyrefly's `project-includes`.
- **Why:** The invariant's type check would otherwise miss errors in Poorin's own tools.
- **Landed:** `ea24a6e2f`.

### D26. Select tests from the uncommitted change; one pytest per package *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted; P1 replaces the selector (D37).
- **Decision:** Tests are selected from tracked files that differ from `HEAD` plus untracked
  files, and each package's tests run in their own pytest process.
- **Why:** The first full check diffed the branch against `main`, so the branch's
  `pyproject.toml` change selected every test, and one pytest process failed at collection:
  `tests/conftest.py` and `lib/verifyit/tests/conftest.py` both load as `tests.conftest`.
- **Cost if wrong:** A change committed without running the check is never tested by a later
  check.
- **Landed:** `da0091495`.

### D27. Clean up run outputs after every check, even a failed one

- **When:** 2026-10-05, P0 (review). **Status:** Adopted.
- **Decision:** After training, `prune_check` deletes per-user run outputs and the entries its
  runs add to `logs/` and `wandb/`, inside a `finally`. Tokenized datasets and kernel caches stay.
- **Why:** Every `dev` run now writes a fresh path; a grug 130M checkpoint with optimizer state is
  about 1.5 GB, and each R1 run left about 25 MB of logs in the checkout. A failed run used to
  skip cleanup.
- **Landed:** `ea24a6e2f`, `75ea55308`.

### D28. Commit the metrics CSV despite Marin's `*.csv` ignore *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `.gitignore` gets `!docs/poorin/prune-metrics.csv`.
- **Landed:** `9d133163f`.

### D29. Execute in the main checkout on a branch *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted for P0 and P1.
- **Decision:** Plans run on a branch in the existing checkout, without a separate git worktree.
- **Why:** The CUDA venv (12 GB) and the compile and dataset caches live there; a worktree would
  need its own GPU sync.
- **Cost if wrong:** None found; the branch isolates the work from `main`.

### D30. MFU uses NVIDIA's published RTX 5090 peak

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `fray.device_flops` lists the RTX 5090 at 209.5 TFLOP/s BF16 with FP32
  accumulate, from NVIDIA's RTX Blackwell whitepaper, and maps the device kind
  "NVIDIA GeForce RTX 5090" to it. The tutorial gains an `rtx5090x1` device.
- **Why:** Runs logged "Unknown device type" and reported no MFU.
- **Alternatives:** The 222 TFLOP/s BF16 matmul rate measured in spike 0.
- **Landed:** `3de090665`.

### D31. Each `dev` run gets a fresh output path

- **When:** 2026-10-05, P0. **Status:** Adopted; superseded in part by D97 (a calendar version's
  tracker run ID carries the version too).
- **Decision:** `--version dev` resolves to `{YYYYmmdd-HHMMSS}-{4 hex}-dev` and `<label>-dev` to
  `<label>-{YYYYmmdd-HHMMSS}-{4 hex}-dev`. The launcher prints the version, `--resume <version>`
  continues that run, and `train_lm` suffixes the tracker run ID the same way. Calendar versions
  are unchanged.
- **Why:** `dev` reused the `{name}/dev` path, so Levanter resumed from the previous run's
  checkpoint; spike 0's second run logged "Resuming training from step 100".
- **Alternatives:** A code hash in the run identity.
- **Landed:** `0d07b68e3`.

### D32. `--override NAME=dev` keeps the plain `dev` version *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** Only `--version` gets the timestamp suffix.
- **Why:** The spec scopes the suffix to `--version`, and an override is an explicit choice for
  one artifact.
- **Cost if wrong:** That artifact resumes or overwrites its previous `dev` output.
- **Landed:** `0d07b68e3`.

### D33. `--resume dev` reuses exactly `dev` *(ruling)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `--resume dev` and `--resume <label>-dev` are accepted and reuse that version
  unchanged.
- **Why:** It matches the documented "reuses that exact version" contract and lets runs made
  before the change continue.
- **Landed:** `0d07b68e3`.

### D34. FA4 is grug's default GPU attention

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** On GPU, grug's attention uses FA4 (FlashAttention-4 in the CuTe DSL) for causal
  self-attention in bf16 or fp16 with head dimension 64 or 128 on compute capability 8.x to 12.x.
  Every other input keeps the reference path.
- **Why:** The reference path ran at 33 TFLOP/s on the 5090 and materialized an fp32 score matrix
  (4.3 GB at the spike's shapes). FA4 ran at 140 TFLOP/s, was already a grug implementation, and
  covers grug's causal, packed-segment, and sliding-window masks.
- **Alternatives:** cuDNN through `jax.nn.dot_product_attention` at 156 TFLOP/s. It is not a grug
  implementation, and it fails on sm_120 until NVRTC can load its builtins library (fixed in
  P5a).
- **Cost if wrong:** About 10% of attention throughput left on the table.
- **Landed:** `c6d91119e`; a GPU test pinning the FA4 lowering in `75ea55308` (review: the parity
  tests also passed on the reference path, so they could not catch a regression).

### D35. FA4's support check covers only what grug produces *(ruling, review)*

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** `_fa4_supports` does not check v's head dimension, distinct query and key segment
  IDs, or FA4's sharding constraints.
- **Why:** FA4 raises a clear error for each, and grug never produces these inputs.
- **Cost if wrong:** An unusual caller gets an error instead of a silent fallback.
- **Landed:** `c6d91119e`.

### D36. Grug's base launcher trains locally with diagnostics off

- **When:** 2026-10-05, P0. **Status:** Adopted.
- **Decision:** The base launcher trains the 130M model for 50 steps at sequence length 1024 on
  the 10M-token FineWeb-Edu cache from Hugging Face, on one RTX 5090. Gradient watch and backward
  flow become explicit launch intervals, off in the base launcher. The three other experiments
  that build the launch config pass their previous intervals (10 and 50). The run ID follows the
  resolved version.
- **Why:** The launcher trained on Marin's GCS buckets on a v5p-8 TPU. Each diagnostic is a static
  flag on the jitted step and added a compiled variant of about 90 s; with both off, steps 2 to 50
  ran in 11 s after the first compile.
- **Landed:** `80fff5ba6`.

---

## P1 fleet operations: planned 2026-10-05

Plan: `plans/2026-10-05-p1-fleet-operations.md`. Executed on branch `prune/p1-fleet-ops` on
2026-10-06; rulings made while executing it, and the fixes from its final review, are D53 to D66.

### D37. Affected tests come from the reachability import graph

- **When:** 2026-10-05, P1. **Status:** Adopted.
- **Decision:** `tools/affected_tests.py` selects tests from the graph `tools/reachability.py`
  builds, with `tools/changes.py` defining the uncommitted change (both sides of a rename count).
  Suites come from the workspace members in the root `pyproject.toml`.
- **Why:** `prune_check` used one entry point of `infra/ci/select_tests.py`'s 1,140 lines of
  GitHub Actions matrix logic, and P1 deletes it. The reachability graph includes imports inside
  functions, so tests of lazily imported modules are selected too.
- **Alternatives:** Copy the selector out of `infra/` before deleting it.
- **Cost if wrong:** A missed import edge lets a broken test pass unselected until a change to
  `pyproject.toml` or `uv.lock` runs every suite.
- **Landed:** `[poorin] Select affected tests from the reachability import graph`.

### D38. Five host-specific test exclusions; tests run in UTC

- **When:** 2026-10-05, P1. **Status:** Adopted; the exclusions are superseded by D93. The Iris and
  Verifyit deselects took effect only with D56.
- **Decision:** Skip `test_tensorboard.py` (P7 deletes the tracker), two Iris hardware-probe tests
  that expect no GPU (P5 deletes Iris), and two Verifyit Go-grader tests that fail with this
  workstation's Go toolchain (P2 deletes Verifyit). Run every test with `TZ=UTC`.
- **Why:** These fail because of the host. The alternative fix for the first, installing every
  library's test dependency group, adds 28 packages, Gradio among them, for code later phases
  delete. One Finelog test assumes UTC, as Marin's CI does.
- **Cost if wrong:** A real regression in one of those five tests goes unseen until its code is
  deleted.
- **Landed:** `[poorin] Select affected tests from the reachability import graph`.

### D39. The linter moves to `tools/lint.py`

- **When:** 2026-10-05, P1. **Status:** Adopted; superseded in part by D49 (ty replaces pyrefly)
  and D50 (ruff format replaces black). Marin's code keeps its explicit ruff linter
  configuration (`RUFF_CONFIGS` in `tools/lint.py`).
- **Decision:** `tools/lint.py` runs ruff 0.14.3 and black 25.9.0 (each library's black
  configuration), pyrefly, license headers, file size, syntax, merge-conflict markers, TOML,
  YAML, and JSON validity, trailing whitespace, final newlines, and notebook outputs. It drops
  Marin's skill-metadata check, its Markdown command check, and the agent lint review. Root-level
  files are now checked, and JSON validity covers every `.json` file.
- **Why:** The spec keeps these checks at the same versions so no file is reformatted. Marin's
  `fnmatch` patterns never matched root-level files, by accident. Marin's JSON check covered only
  Grafana dashboards, which P1 deletes.
- **Alternatives:** A script at the repository root; the `pre-commit` framework.
- **Cost if wrong:** A formatting difference would show up as churn in the first full run, which
  Task 2 checks for.
- **Landed:** `[poorin] Replace infra/pre-commit.py with tools/lint.py`.

### D40. `tools.*` becomes a kept root

- **When:** 2026-10-05, P1. **Status:** Adopted. Resolves a P0 deferred item.
- **Why:** `tools/` counts toward first-party lines but was unreachable, so P7's unreachable sweep
  would have listed it.
- **Landed:** `[poorin] Select affected tests from the reachability import graph`.

### D41. Delete more than the spec's P1 list

- **When:** 2026-10-05, P1. **Status:** Adopted, except `.agents/projects/grugformer.md`, which
  moved (D57).
- **Decision:** Also delete Marin's ferries, the whole `.github/` directory, the `Makefile`,
  `.agents/{agents,docs,logbooks,ops,projects}` (151 files of Marin's notes), 29 of 41 skills,
  the three `license_header.txt` files, and the docs pages about deleted tooling.
- **Why:** Each serves Marin's fleet or organization. No root reaches the ferries, and three of
  them import `infra`. The `Makefile` targets wrap deleted tools or set up Marin's GCP project.
  `.github/` holds actions for deleted workflows, plus Dependabot and CodeQL configuration.
- **Cost if wrong:** Anything needed again is at `upstream-baseline`.
- **Landed:** `[poorin] Delete Marin's organization skills and agent notes`, `[poorin] Delete
  Marin's ferries`, `[poorin] Delete infra/`, `[poorin] Delete scripts/`, `[poorin] Delete
  docker/, etc/, and the GitHub configuration`.

### D42. Keep `.dockerignore`

- **When:** 2026-10-05, P1. **Status:** Adopted.
- **Why:** Levanter's TPU tutorial builds Docker images that respect it. Sub-project 2 decides how
  TPU jobs are packaged.
- **Landed:** `[poorin] Delete docker/, etc/, and the GitHub configuration`.

### D43. The local machine is the default cluster; Marin's clusters become fixtures

- **When:** 2026-10-05, P1. **Status:** Adopted. One rigging test that clears the environment
  now names the `marin` cluster itself, and Iris's suite resolves clusters from
  `config/testing/` too: its CoreWeave backend reads Marin's `coreweave` config.
- **Decision:** `config/local.yaml` declares no buckets and becomes the default for
  `MARIN_CLUSTER`. Marin's `marin.yaml` and `coreweave.yaml` move to `config/testing/`, which the
  default search does not read. The rigging and root test suites resolve clusters from there; the
  root suite sets this in `pytest_configure`.
- **Why:** Rigging's tests and twelve root-suite tests assert Marin's multi-region bucket layout
  (region routing, temp buckets, cross-region guards). Two experiment modules compute temp-bucket
  paths at import time, before any fixture runs.
- **Alternatives:** Delete Marin's cluster files outright, as the spec first said, and rewrite or
  delete the tests that use them.
- **Cost if wrong:** Root-suite tests keep testing Marin's layout until later phases revisit them.
- **Landed:** `[rigging] Make the local machine the default cluster`.

---

## Poorin as a dependency: 2026-10-06

### D44. Two layers: a workbench and a library

- **When:** 2026-10-06, design. **Status:** Superseded in part by D45: the library layer is
  dropped; the workbench half stands.
- **Decision:** Poorin has two layers. The workbench runs as tools in Poorin's own environment
  and writes artifacts to disk: data preparation, the Delphi reference runs, scaling-law fits,
  and benchmark harnesses. The library is a small set of modules that research repos import.
  Research projects live in their own repositories and depend on Poorin; `looped-diffusion-lm`
  is the first.
- **Why:** Data preparation needs Zephyr, Marin's executor, and the tokenizer stack, while a
  research project's training loop needs JAX, Equinox, and Optax. Exchanging files between the
  layers keeps the two dependency sets out of one lockfile. The generic half of
  `looped-diffusion-lm`'s design (run identity, seed streams, exports, the analysis step) is
  what later projects need too.
- **Alternatives:** A monorepo where each project lives under Poorin's `experiments/`, as in
  Marin.
- **Cost if wrong:** Poorin's library needs a stable, documented interface, which a monorepo
  would not.

### D45. No library layer; Poorin is Marin's utilities

- **When:** 2026-10-06, design. **Status:** Adopted.
- **Decision:** Poorin does not grow a `poorin` package or a project kit. It stays the pruned
  set of Marin's utilities. Research projects use those utilities, and the owner builds a
  lightweight research template separately, later.
- **Why:** The owner values Marin's utilities and does not want Poorin to be the foundation a
  project is built on. Run identity, seed handling, exports, and the analysis step belong in
  the owner's template.
- **Alternatives:** A bottom-layer `poorin` package holding the generic half of
  `looped-diffusion-lm`'s design (D44), discussed the same day. Its follow-up choices (a Hydra
  adapter, Orbax in the library, launch presets in the library, a library MVP after P1) were
  dropped with it.
- **Cost if wrong:** Until the template exists, `looped-diffusion-lm` carries its own run
  identity, seeds, and exports.

### D46. Projects use Poorin as a workbench only

- **When:** 2026-10-06, design. **Status:** Superseded in part by D51 and D52: projects import
  Marin's executor and share Poorin's workspace; data still moves between them as files.
- **Decision:** A research project imports nothing from Poorin. Poorin runs as a workbench in
  its own environment and writes files the project reads: tokenized datasets, model weights,
  evaluation results, and scaling-law fits. A project that needs a small piece of Poorin's code,
  such as AdamH, copies it.
- **Why:** No coupling between the repositories: Poorin's internal structure, dependencies, and
  JAX version never constrain a project's lockfile.
- **Alternatives:** Running the workbench and also importing a few leaf modules (AdamH, the
  attention kernels, the cache reader) pinned by git commit; depending on Poorin's packages
  outright.
- **Cost if wrong:** Copied code drifts, and fixes in Poorin do not reach the project. The file
  formats become the contract between the repositories, so each needs a written definition
  and a version.

### D47. Re-derive the kept roots from the workbench survey

- **When:** 2026-10-06, design. **Status:** Adopted. With `tools.*` and P1's other changes,
  reachable code fell from 461 modules and 139,309 lines to 420 and 127,512, and every new root
  imports on the workstation.
- **Decision:** The kept roots in `tools/roots.toml` become the Marin utilities a solo researcher
  runs as tools, found by a survey of data, training, and evaluation code. Added: tokenization,
  the Hugging Face and Nemotron-CC v1 downloaders, FineWeb-Edu, normalize, exact and fuzzy
  dedup, consolidation, decontamination, Levanter's `eval_lm` and perplexity gap, the Delphi
  recipe (`completed_adamh`), the Delphi-mix dataset recipes with their Paloma and Uncheatable
  validation sets, and the lm-eval task lists. Dupekit stays, since dedup, normalize, and
  decontamination import it, so P4 deletes only Ducky. Marin's perplexity-gap wrapper, which
  asserts TPU resources and submits a Fray job, gives way to Levanter's entry point. Reachable
  code falls from 455 modules and 138,099 lines to 414 and 126,306 (measured before `tools.*`
  becomes a root).
- **Why:** With projects using Poorin as a workbench only (D46), the roots should be the
  workbench tools. The survey found these run locally under `MARIN_PREFIX`.
- **Cost if wrong:** A utility left off the list is deleted in P7; it remains at
  `upstream-baseline`.
- **Landed:** `specs/2026-10-06-workbench-and-roots-design.md`; `[poorin] Keep the workbench
  tools as the prune's roots`.

### D48. Drop four kept roots

- **When:** 2026-10-06, design. **Status:** Adopted for the roots; the deletions are planned for
  P3 and P7.
- **Decision:** Remove these kept roots. Their code becomes unreachable and is deleted later.
  - The Harbor and Evalchemy integrations, about 3k lines: their runners need Iris or Marin's
    serving sessions, and agentic benchmarks don't fit small-LM research. lm-eval, records,
    and statistics stay. Supersedes part of D10.
  - The MoE scaling ladder: fast-track's launch, train, and model scripts, grug MoE, DeepEP,
    and the short-convolution kernel, about 12k lines. Fast-track's heuristic and optimizer
    (AdamH, MuonH) stay. Supersedes the spec's Pass 2 verdict that kept grug MoE for
    dense-versus-MoE comparisons.
  - `reference_scaling_suite`: it fits 78 isoFLOP checkpoints on Marin's private buckets.
    `marin.scaling_laws` and the Delphi recipe stay.
  - Levanter's OpenAI-compatible server and the tutorial sweep example. Reference models go
    through vLLM; the inference engine stays, reachable through `sample_lm`.
- **Why:** None of them runs or matters on one workstation without Marin's infrastructure.
- **Cost if wrong:** MoE work later would start from `upstream-baseline`'s grug MoE.
- **Landed:** `[poorin] Keep the workbench tools as the prune's roots`.

### D49. ty replaces pyrefly

- **When:** 2026-10-06, P1. **Status:** Adopted. Warnings do not fail the check (D54).
- **Decision:** Type-check with ty 0.0.84, run through `uvx` at a pinned version like ruff. It
  checks the libraries that survive the prune (Marin, Levanter, Haliax, Fray, rigging, Zephyr,
  Finestore) and `tools/`; Iris, Ducky, and Verifyit, deleted in P2, P4, and P5, are left
  unchecked. The categories pyrefly's config turned off map to ty rules set to `ignore`
  (`invalid-argument-type`, `unresolved-attribute`, `no-matching-overload`), and pyrefly's
  optional-import list maps to `allowed-unresolved-imports`. The errors ty finds at the switch
  (62 in surviving code, 24 in code P7 deletes) get `# ty: ignore[rule]` markers, generated by
  `ty check --add-ignore`, and Pass 2 fixes the ones that survive as it reads each module. The
  97 `# pyrefly: ignore` comments are removed.
- **Why:** The owner's choice of Astral's toolchain (uv, ruff, ty).
- **Alternatives:** Keep pyrefly. Fix all 62 errors before continuing. Run the residual
  categories as warnings only.
- **Cost if wrong:** ty is pre-1.0 and its diagnostics change between releases, so upgrading the
  pin is its own commit with a diagnostics diff.
- **Landed:** `[poorin] Type-check with ty instead of pyrefly`.

### D50. ruff format replaces black

- **When:** 2026-10-06, P1. **Status:** Adopted. 209 files changed. Marin's code is still linted
  with its explicit ruff configuration (D39).
- **Decision:** Format with ruff 0.14.3's stable style; black and its configuration leave. Each
  library keeps its line length (121 for the root, 119 for Levanter and Haliax). One commit
  reformats every tracked Python file, so `tools.lint --all-files` keeps passing; a follow-up
  commit lists it in `.git-blame-ignore-revs`. The reformat runs after P1's deletions, so it
  skips `infra/` and `scripts/`.
- **Why:** One tool for linting and formatting.
- **Alternatives:** Keep black.
- **Cost if wrong:** About 133 files change once (measured before P1's deletions), and black's
  preview style is not reproduced exactly. Unlike black, ruff splits a line that ends in
  `# type: ignore`; one split pragma in Haliax's `axis.py` was moved back to the line ty reports.
- **Landed:** `[poorin] Format with ruff instead of black`.

### D51. Projects run on Marin's executor, not Hydra

- **When:** 2026-10-06, design. **Status:** Adopted.
- **Decision:** `looped-diffusion-lm` defines its runs as Marin executor steps instead of Hydra
  configs: typed dataclass configs, versioned output paths (`name@version`, with P0's fresh `dev`
  versions and `--resume`), fingerprints, dry runs, and `marin.experiment.sweep` for grids.
- **Why:** The owner's choice: Marin's executor already does run identity, caching, and
  versioning, which the project's design had planned to build on Hydra.
- **Alternatives:** Hydra with the project's own run keys and `runs.tsv` (its original design).
- **Cost if wrong:** Fray has no SLURM backend, so runs on NU Quest need one (sub-project 2) or
  a job script that runs the executor inside a SLURM allocation.

### D52. Projects are nested uv workspace members

- **When:** 2026-10-06, design. **Status:** Planned (after P1).
- **Decision:** A project keeps its own git repository, checked out under `Poorin/projects/` and
  ignored by Poorin's git, and is listed as a uv workspace member. It shares Poorin's lockfile and
  CUDA environment and imports the executor, Fray, and Levanter's utilities directly. Each run
  records both repositories' commits.
- **Why:** The executor brings Fray, rigging, and draccus with it; one shared workspace avoids
  carrying Poorin's package indexes and sources into a second lockfile.
- **Alternatives:** A separate repository depending on Poorin's packages pinned by commit; the
  project's code inside Poorin's `experiments/`.
- **Cost if wrong:** Poorin's lockfile carries the project's dependencies, and the project's
  imports become prune roots, so Poorin's prune has to keep them working.

---

## P1 execution: 2026-10-06

### D53. Tests that read P0's deleted workflows go before the first full check

- **When:** 2026-10-06, P1 Task 3. **Status:** Adopted.
- **Decision:** Delete `tests/infra/ci/test_package_release.py` and one `infra/pulumi` test, which
  read the `.github/workflows` files P0 deleted, in their own commit ahead of Task 7, which deletes
  the rest of `infra/`.
- **Why:** Task 3 changes the root `pyproject.toml`, so its check is the first to run every suite,
  and those tests failed on the missing files.
- **Alternatives:** Move Task 7 ahead of Task 3.
- **Cost if wrong:** None; Task 7 deletes the directories they lived in.
- **Landed:** `[poorin] Delete tests that read the workflows P0 removed`.

### D54. ty fails the check on errors only

- **When:** 2026-10-06, P1 Task 3. **Status:** Adopted.
- **Decision:** `tools.lint` and `tools.prune_check` run ty with `--exit-zero-on-warning`, and the
  markers added at the switch cover errors only: 87 `# ty: ignore[rule]` markers on 79 lines. The
  231 warnings (161 unused `type: ignore` comments, 36 redundant casts, 34 possibly missing
  submodules) are printed on every run, and Pass 2 clears them.
- **Why:** ty 0.0.84 exits non-zero on warnings, and `--add-ignore` marks warnings too: it added
  157 markers, 71 of them for warnings. D49 meant to suppress errors only.
- **Alternatives:** Mark the warnings as well; turn the three warning rules off.
- **Cost if wrong:** A new warning never fails a check, so warnings can accumulate until Pass 2.
- **Landed:** `[poorin] Type-check with ty instead of pyrefly`.

### D55. A test that times out under host load is rerun, not given a longer limit

- **When:** 2026-10-06, P1 Task 3. **Status:** Superseded by D60.
- **Decision:** When grug's `test_staged_loss_and_gradients_match_the_unsplit_model` passed the
  60-second limit while about 200 browser processes outside the checkout held the load average at
  243 on 32 cores, the check was rerun unchanged, and it passed.
- **Why:** Alone, under the same load, it took 57 seconds, and it had passed in the previous run
  of the same tree. The change being checked touched only a comment the test never reaches, and
  P7 deletes the MoE pipeline.
- **Alternatives:** `@pytest.mark.timeout(...)`, which `TESTING.md` allows for a legitimately
  long test.
- **Cost if wrong:** The check can fail on host load until P7.

### D56. Host exclusions are rewritten relative to each suite's configuration

- **When:** 2026-10-06, P1 Task 3. **Status:** Adopted.
- **Decision:** `HOST_EXCLUSIONS` keeps repository-relative paths. `pytest_runs` takes the
  exclusions explicitly and rewrites each `--deselect` relative to the suite's configuration
  directory; `--ignore` stays relative to the repository root, where pytest runs.
- **Why:** Every library under `lib/` has its own pytest configuration, which makes it pytest's
  rootdir, so node IDs inside the Iris suite start at `tests/`. D38's deselects never matched,
  and Task 1's test checked only the argument strings. The new test runs real pytest on a library
  with its own configuration.
- **Alternatives:** Write each exclusion relative to its suite by hand; run each library suite
  from its own directory, which moves every path, not only the exclusions.
- **Cost if wrong:** None known. `--ignore` still does not apply to a file passed by name (see
  Deferred).
- **Landed:** `[poorin] Deselect host-excluded tests by suite node ID`.

### D57. Grug's principles move next to the code

- **When:** 2026-10-06, P1 Task 5. **Status:** Adopted.
- **Decision:** `.agents/projects/grugformer.md`, Marin's 46-line statement of grug's rules
  (Equinox modules with explicit `init` and `__call__`, raw JAX ops instead of `eqx.nn`, explicit
  sharding, `einsum`, copy-paste variants), becomes `experiments/grug/PRINCIPLES.md` with its
  commands updated, instead of being deleted with the rest of `.agents/projects/`. The grug README
  links it there.
- **Why:** Poorin builds its models in grug style (D13), so these rules
  are design guidance for Poorin itself, not notes about Marin's systems.
- **Alternatives:** Delete it with `.agents/projects/` as planned and point the README at the
  `upstream-baseline` tag.
- **Cost if wrong:** One short file to delete.
- **Landed:** `[poorin] Delete Marin's organization skills and agent notes`.

### D58. The canary-metrics test goes with the ferries it loads

- **When:** 2026-10-06, P1 Task 7. **Status:** Adopted.
- **Decision:** Delete `tests/test_validate_canary_metrics.py` in its own commit, ahead of Task 9,
  which deletes it with `scripts/`.
- **Why:** It imports `scripts/ci/validate_canary_metrics.py`, which imports the canary ferry
  Task 6 deleted. Task 6's check did not select it: its reference search skipped `scripts/`, and
  the import graph does not read `scripts/` either. Task 7's check, which runs every suite,
  failed on it.
- **Cost if wrong:** None; Task 9 deletes it anyway. The blind spot closes when `scripts/` goes.
- **Landed:** `[poorin] Delete the canary-metrics test that loads a deleted ferry`.

### D59. Finelog's tests of Marin's deploy files go with `infra/`

- **When:** 2026-10-06, P1 Task 7. **Status:** Adopted.
- **Decision:** Delete three Finelog tests in the commit that deletes `infra/`: two that check
  `pulumi refresh` runs after a Kubernetes rollback, and one that loads Marin's cluster-capacity
  Grafana dashboard. The rest of their files stay.
- **Why:** Finelog's Pulumi deploys run from `infra/finelog` and stop with "must run from a Marin
  repository checkout" without it, so the refresh path is dead in Poorin; the dashboard JSON
  lived in `infra/grafana`. P6 deletes Finelog's `deploy/` and `benchmarks/`.
- **Alternatives:** Point the tests at a temporary Pulumi project, which keeps testing code
  Poorin cannot run; pull P6's Finelog trim into P1.
- **Cost if wrong:** None; P6 deletes the code they tested.
- **Landed:** `[poorin] Delete infra/`.

### D60. The MoE pipeline equivalence test gets a 180-second limit

- **When:** 2026-10-06, P1 Task 12. **Status:** Adopted.
- **Decision:** `test_staged_loss_and_gradients_match_the_unsplit_model` carries
  `@pytest.mark.timeout(180)`, the limit the root suite already gives its slow June model tests.
- **Why:** It timed out again in Task 12's check. The browser processes outside the checkout
  have held the load average near 250 on 32 cores for hours, and alone the test takes about 57
  seconds, so the 60-second default fails it often enough to cost a 25-minute rerun each time.
- **Alternatives:** Keep rerunning (D55); mark it `slow`, which drops it from every default run.
- **Cost if wrong:** A real hang in this test takes three minutes to report. P7 deletes the MoE
  pipeline.
- **Landed:** `[grug] Give the MoE pipeline equivalence test three minutes`.

### D61. A selected file that a host exclusion ignores leaves the selection

- **When:** 2026-10-06, P1 Task 12. **Status:** Adopted. Resolves the gap D56 deferred.
- **Decision:** `pytest_runs` drops selected paths that an `--ignore` exclusion names, and skips a
  suite left with no paths.
- **Why:** pytest collects a file passed by name even when `--ignore` names it. Task 12 changes
  rigging's cluster configuration, which Levanter's TensorBoard tracker imports, so the selector
  passed `test_tensorboard.py` by name and it failed on the missing `tensorboardX`. A suite with
  no paths left must not run: pytest would collect everything under the repository root.
- **Cost if wrong:** None known.
- **Landed:** `[poorin] Skip ignored test files that a change selects by name`.

### D62. The Git hook lints only staged files

- **When:** 2026-10-06, P1 final review. **Status:** Adopted.
- **Decision:** `tools.lint --staged` lints the paths whose staged content differs from `HEAD`
  (`tools.changes.staged_files`), and the documented pre-commit hook runs
  `tools.lint --staged --fix`. `tools.lint` with no arguments, and `prune_check`, still check the
  whole uncommitted change, untracked files included.
- **Why:** The hook ran `tools.lint --fix` over the whole uncommitted change, so committing one
  file cleared the outputs of untracked scratch notebooks, added license headers to scratch
  scripts, and failed on large untracked data files. Marin's hook checked only tracked files.
- **Alternatives:** Leave untracked files out of `changed_files`, which would hide new files from
  `prune_check`.
- **Cost if wrong:** None known.
- **Landed:** `[poorin] Lint only staged files from the Git hook`.

### D63. A change under `config/` runs every suite

- **When:** 2026-10-06, P1 final review. **Status:** Adopted.
- **Decision:** `tools.affected_tests` treats `config/` as shared test data: any change there
  selects every suite, as the root `pyproject.toml` and `uv.lock` do.
- **Why:** P1 made `config/testing/` the cluster fixtures for the rigging, Iris, and root suites,
  and rigging's default-cluster test reads `config/local.yaml`, but no module imports a YAML file,
  so a change to either selected no tests and would pass `prune_check` untested.
- **Alternatives:** A table mapping each data directory to the suites that read it.
- **Cost if wrong:** A change to `config/external/` pins also runs every suite, about 25 minutes.
- **Landed:** `[poorin] Run every suite when the cluster configs change`.

### D64. A grug change selects the variant contract test

- **When:** 2026-10-06, P1 final review. **Status:** Adopted.
- **Decision:** `tools.affected_tests` keeps a table of test files that load modules by computed
  name, with the package each loads from. A change to any module in that package selects the
  test. Its one entry maps `tests/test_grug_variant_contracts.py` to `experiments.grug`.
- **Why:** The contract test loads each grug variant with `importlib.import_module` and a computed
  name, which no static import graph sees, so editing grug's base model, the template Poorin's
  models start from (D13), selected no unit tests. Marin's selector had the same blind spot.
  `prune_check` trains R2, which covers `base` but not the other variants.
- **Alternatives:** Import the variants statically in the test, which would undo its automatic
  discovery of new variants.
- **Cost if wrong:** Every grug change runs the contract test, which compiles each variant.
- **Landed:** `[poorin] Select the grug variant contract test for any grug change`.

### D65. Lint treats a file with a NUL byte as binary

- **When:** 2026-10-06, P1 final review. **Status:** Adopted.
- **Decision:** `tools.lint` skips the text checks for a file whose first 8000 bytes contain a
  NUL, git's own binary test, as well as for a file that is not UTF-8.
- **Why:** NUL is valid UTF-8, so a small binary file could decode as text, and `--fix` would
  then strip "trailing whitespace" from it and append a newline, silently corrupting it.
- **Cost if wrong:** A text file containing a NUL skips the whitespace and newline checks.
- **Landed:** `[poorin] Treat files with a NUL byte as binary in lint`.

### D66. Stale comments and links found by the final review go now

- **When:** 2026-10-06, P1 final review. **Status:** Adopted.
- **Decision:** One commit removes what the review found stale in kept files: a duplicated line
  in the comment Task 12 rewrote in rigging's `cluster_config.py`, the `.gitignore` comment left
  without its pattern by P0, D39's status text, and links from the grug README, two kept skills,
  and the MoE hero README to deleted skills. Links in code that P2 and P7 delete stay.
- **Why:** `AGENTS.md` says to delete stale comments on discovery and keep documentation in sync.
  The review's other minor findings are deferred.
- **Cost if wrong:** None.
- **Landed:** `[poorin] Remove stale comments and links to deleted skills`.

---

## P2 RL stack: 2026-10-07

Plan: `plans/2026-10-07-p2-rl-stack.md`.

### D67. The remaining prune phases run end to end

- **When:** 2026-10-07, P2. **Status:** Adopted.
- **Decision:** The owner authorized running P2 to P8 and Pass 2 without reviewing each phase's
  plan. Each phase still gets a plan in `docs/poorin/plans/`, one branch, the invariant before
  every commit, a final review by a fresh reviewer, and a local fast-forward merge to `main`.
  Nothing is pushed. Setting up `projects/` (D52) waits until `looped-diffusion-lm` has code,
  since it has no imports to protect yet.
- **Why:** The design covers every phase explicitly; the owner asked to be notified only when a
  decision needs them.
- **Cost if wrong:** A phase the owner would have shaped differently lands on local `main`; it can
  be reverted before anything is pushed.

### D68. RL through the owner's SkyRL fork is a first-class integration

- **When:** 2026-10-07, P2. **Status:** Adopted for the roots; the integration itself is planned.
- **Decision:** Marin's SkyRL glue goes (D6), but Poorin treats RL post-training through the
  owner's SkyRL fork as first-class. The fork (`skyrl-train`, the JAX backend `skyrl-tx`,
  `skyrl-gym` environments) imports nothing from Poorin, so Poorin keeps the hooks it plugs into:
  Hugging Face export of Levanter checkpoints, during training (`hf_save_path`, through
  `levanter.compat.hf_checkpoints`) or afterwards (`levanter.main.export_lm_to_hf`); the chat
  template (`marin.datakit.chat_template`); dispatch through the executor and Fray; Levanter's
  inference engine and `sample_lm`; and evaluation records and rollouts. The first three become
  kept roots, since nothing else guarantees they survive P7 and Pass 2; the others already are
  roots or reached from them. Grug models have no Hugging Face mapping yet (only the MoE variants
  P7 deletes had one). A later sub-project designs the integration: a grug mapping, launching
  SkyRL runs from the executor, and moving checkpoints both ways.
- **Why:** The owner, 2026-10-07: delete Marin's integration directly, "but assume we have first
  class support."
- **Alternatives:** Keep Verifyit's graders for RL rewards; the fork brings its own environments
  and rewards in `skyrl-gym`.
- **Cost if wrong:** Three kept roots that keep `transformers` reachable.
- **Landed:** `[poorin] Keep the hooks a SkyRL integration needs as roots`; the export entry point
  joined after P2's final review.

### D69. Verifyit leaves with Harbor and Evalchemy in P3

- **When:** 2026-10-07, P2. **Status:** Adopted.
- **Decision:** P2 deletes the RL integration, Shellbox, and Taskcompendium, and leaves Verifyit.
  P3 deletes it right after the Harbor and Evalchemy integrations.
- **Why:** Harbor's trial driver and its root-suite test import `harbor_config`, which reaches
  the lock only through Verifyit's Harbor adapter; deleting Verifyit first broke that test.
  Verifyit is mostly benchmark adapters for Harbor, Evalchemy, and lm-eval, so it belongs with
  them.
- **Cost if wrong:** None; the spec's P2 row lists Verifyit, and P3 now carries it.

---

## P3 serving and evaluation orchestration: 2026-10-07

Plan: `plans/2026-10-07-p3-serving-and-eval-orchestration.md`.

### D70. No lockfile until P8, no all-suite runs per commit, no per-phase review

- **When:** 2026-10-07, P3. **Status:** Adopted; the lockfile clause ended in P8 (D80).
- **Decision:** `uv.lock` is deleted. Work continues in the existing environment through
  `uv run --no-sync`, which needs no lockfile; nothing syncs until P8, which resolves a fresh lock
  and re-verifies the GPU environment. A change to the root `pyproject.toml` or `uv.lock` no
  longer selects every suite; `tools.affected_tests --all` runs every suite once per phase,
  before the merge. Code, tests, and docs that existed only for Marin's CI are deleted wherever
  they turn up. A phase ends with a self-check instead of a fresh reviewer.
- **Why:** The owner, 2026-10-07: Iris, Finelog, and P8 will change dependencies heavily, so
  tracking lock diffs per commit costs more than it catches, and CI-style gating is not needed
  for one researcher.
- **Alternatives:** Keep the lock and regenerate it in P8; keep the lock without drift checks.
- **Cost if wrong:** Between phases' full runs, a commit can break a suite its imports do not
  reach, and P8's fresh resolve can move JAX or CUDA wheels, which P8 re-verifies on the 5090.
  Supersedes the spec's "Risks" row on lock regeneration and the per-phase review in D67.

### D71. P3 cuts imports from the record schema, not its types

- **When:** 2026-10-07, P3. **Status:** Adopted.
- **Decision:** `marin.evaluation.records` stops importing Harbor (it defines the commit-hash
  pattern itself) but keeps its Harbor, Evalchemy, and speculative-serving reference types.
- **Why:** `eval_measurements`, `rollouts`, and three kept test files read those types, and the
  spec says P3 "redesigns nothing, so sub-project 7a starts from Marin's framework as it is."
- **Cost if wrong:** Dead schema types until sub-project 7a.

### D72. How P3 handled tests and docs that span deleted and kept code

- **When:** 2026-10-07, P3. **Status:** Adopted.
- **Decision:** `tests/evaluation/test_model_config.py` lost its fleet and serving cases in the
  first commit, keeping only cases for `marin.evaluation.model_config`; the Iris topology case
  left with that check. `docs/explanations/evaluation.md` was rewritten once, in the first commit,
  to describe what P3 keeps. `experiments/evaluation/configs/` held only Evalchemy and Harbor
  catalogs, so each went with its harness. Verifyit moved into P3 (D69).
- **Why:** The test file imported modules from four different commits at module level, and the
  page described paths deleted across six.
- **Cost if wrong:** A few commits without tests for code deleted later in the same phase.

---

## P4 to P7 strip: 2026-10-07

Plan: `plans/2026-10-07-p4-p7-strip.md`.

### D73. Strip first, fix static checks and tests afterwards

- **When:** 2026-10-07, P4. **Status:** Adopted.
- **Decision:** P4 to P7 run on one branch. Each commit runs `tools.prune_check --fast`, which
  imports every kept root and trains R1 and R2 but skips ty, lint, and tests. After P7 a cleanup
  pass runs ty, `tools.lint --all-files`, and `tools.affected_tests --all`, and fixes what they
  report.
- **Why:** The owner, 2026-10-07: strip aggressively, ignore ty, ruff, and the CI checks for now,
  and resolve them afterwards.
- **Cost if wrong:** Type, lint, and test failures pile up and the cleanup pass is larger; the
  commits between are not individually green on those checks. The kept-root imports and the two
  training runs still gate every commit.

### D74. Telemetry publishes only when given a Finelog URL

- **When:** 2026-10-07, P5b. **Status:** Adopted.
- **Decision:** `TelemetryConfig.finelog_url` is optional. Without it the telemetry tracker, which
  the trainer appends to every run, publishes nothing; with it, process 0 writes the run's metrics
  to that server under Levanter's run ID and `jax.process_index()`, and an unreachable server is
  an error. The spec's `prune_check` Finelog check (invariant item 5) waits for the cleanup pass
  after P7.
- **Why:** The spec asked for an explicit URL that raises when it cannot connect, but the trainer
  adds telemetry to every run, so a required URL would break every local run. The Finelog check
  is new work, not stripping (D73).
- **Cost if wrong:** Local runs keep writing no Finelog metrics until a run sets the URL.

## Cleanup after the strip: 2026-10-07

### D75. Tests and configs of stripped code go; mixed files lose only the stripped cases

- **When:** 2026-10-07, cleanup (D73). **Status:** Adopted.
- **Decision:** A test file that tested only deleted code was deleted: 126 files. Three Levanter
  benchmark scripts and `scripts/wandb_tensorboard_profile.py` went with them. A file that mixed
  deleted and kept code lost only the deleted cases; for example, `test_distributed.py` keeps only
  the SLURM node-list test, and `test_finelog_metrics.py` tests `LevanterMetricsWriter.connect`.
  Levanter configs naming a deleted model type (Backpack, Gemma, Mistral, Mixtral, Whisper,
  Hyena, MPT) were deleted. So were `marin.testing`, whose only caller was a deleted cluster test,
  the docs-only remains of `experiments/grug/moe/`, and the fast-track variant's agent guide,
  which drove its deleted launcher on Iris. The fast-track README keeps its results and the files
  the Delphi recipe uses.
- **Why:** The stripped tree should pass ty, `tools.lint --all-files`, and every suite. After the
  cleanup it does: ty reports no errors, lint passes, and all nine suites pass (3,150 tests). One
  full `prune_check` run on the final tree verifies the cleanup's four commits, and its metrics
  row is committed with the last of them.
- **Cost if wrong:** A deleted test or config is recoverable from `upstream-baseline`.

### D76. Documentation describes only code Poorin has

- **When:** 2026-10-07, cleanup (D73). **Status:** Adopted.
- **Decision:** Pages about deleted subsystems were deleted: Iris actors, Finelog's server
  analytics, TPU fleet observability, perplexity-gap analysis, scaling heuristics, Snowball SFT,
  and cloud GPUs. So was Marin's project history: reports, retrospectives, the model card, and the
  system prompt. Surviving tutorials and references lost their cluster-submission sections and
  links to deleted files. Guides outside `docs/` (READMEs, `AGENTS.md` files, skills) and Python
  comments that described Iris were corrected the same way. Levanter's reference pages had been
  broken upstream since its package `__init__` files stopped re-exporting configs; their `:::`
  references now name the defining submodules. `docs/poorin/` is excluded from the site: its
  specs, plans, and decision log link into the repository and quote other files' Markdown.
- **Why:** AGENTS.md: keep MkDocs content in sync with code, and docs stand alone. The root and
  Levanter sites now build with `mkdocs build --strict`.
- **Cost if wrong:** Marin's reports and tutorials remain at `upstream-baseline`. Poorin's own
  tutorials are not written yet.

### D77. R2 gets the Finelog URL as a runtime argument

- **When:** 2026-10-07, cleanup. **Status:** Adopted.
- **Decision:** `experiments.grug.base.launch` takes `--finelog-url`. The URL is a runtime
  argument of the trial's artifact step, so it does not change the checkpoint's fingerprint. The
  launch config passes it to an explicit `TelemetryConfig` next to W&B. `prune_check` starts an
  embedded Finelog server in a temporary directory, passes its address to R2 only, and fails if
  the server's `levanter.metrics` table is empty after the runs. Every run's rows land in that one
  table, with `run_id` as a column, so a fresh server holds only R2's rows.
- **Why:** Spec invariant item 5, deferred from P5b by D74.
- **Cost if wrong:** The check confirms that rows arrived, not which metrics they hold.

### D78. The profiling package returns as a kept root

- **When:** 2026-10-07, cleanup. **Status:** Adopted.
- **Decision:** `marin.profiling` and `levanter.utils.profile_dirs`, deleted by the P7 sweeps,
  are restored with their tests, and `marin.profiling.cli` becomes a kept root. Broken scripts
  found by the same audit were deleted instead: Levanter's `infra/launch.py`,
  `infra/push_docker.py`, and `infra/cluster/`, the skip-guarded test of the deleted OpenAI
  server, an unused OpenAI stub, and `tests/snapshots/`, whose test and converters are gone.
- **Why:** The spec keeps the `profile-training` skill, which runs the package through the script
  `lib/marin/tools/profile_summary.py`. A script outside `src/` is not a module, so the
  reachability scan never saw the import. An audit of every first-party import in files outside
  `src/` found no other such case.
- **Cost if wrong:** 5,265 lines of profiling code stay without a training root that imports it.

---

## P8 workspace: 2026-10-07

Plan: `plans/2026-10-07-p8-workspace.md`.

### D79. PyTorch installs from its CPU wheel

- **When:** 2026-10-07, P8. **Status:** Adopted; corrected 2026-10-11.
- **Decision:** One root `[tool.uv.sources]` entry sends every `torch` requirement in the
  workspace to the PyTorch CPU index. Marin's per-extra CUDA 12.8 and CPU sources, its
  `pytorch-cu128` index, the root's flat CPU fallback index, and `torchvision` (no importer) go.
  Triton, which the CUDA wheel used to bring, returns to Levanter's GPU extra: QuACK imports it
  without declaring it.
- **Why:** Poorin uses PyTorch only to convert Hugging Face checkpoints and in test helpers, on
  CPU. vLLM, which needed CUDA PyTorch, is gone, and the owner's SkyRL fork has its own
  environment (D68). The CUDA wheel installed CUDA 12 copies of cuDNN, NCCL, and NVRTC beside
  JAX's CUDA 13 libraries (spec, "Risks").
- **Cost if wrong:** Code that needs PyTorch on the GPU in this environment has to bring the CUDA
  wheel back, with its CUDA 12 libraries.
- **Correction (2026-10-11):** PyTorch is not used only for checkpoint conversion and tests.
  `flash-attn-4`, `quack-kernels`, and `torch-c-dlpack-ext` in the `gpu` extra require it, and
  Levanter's FA4 backward (`levanter.grug.attention._fa4_cute_segmented_bwd`) and QuACK MoE kernel
  (`levanter.grug._moe.quack_symmetric_cute`) fail to import without it: `flash_attn.cute` and
  `quack` import it at module load. The CPU wheel satisfies them. So PyTorch cannot become an
  optional extra while the GPU environment keeps those kernels; an attempt on 2026-10-11 removed
  nothing from the synced environment.

### D80. The fresh lock resolves Linux only and takes current releases

- **When:** 2026-10-07, P8. **Status:** Adopted.
- **Decision:** `uv lock` resolves for `sys_platform == 'linux'` only (`tool.uv.environments`).
  It takes the newest release each specifier allows, so 153 packages moved from the pre-P8
  environment, among them numpy 2.5, pandas 3, pyarrow 25, the CUDA 13.4 runtime, cuDNN 9.27, and
  NCCL 2.32; the JAX wheels stay at 0.11.1. Override and constraint entries naming packages absent
  from the graph went, as did the Equinox override written for Vizier. Flax's 0.12.10 floor is a
  root constraint, not a Haliax dependency as the plan said: no workspace code imports Flax; AQT
  does. FlashAttention 4 is pinned to 4.0.0b28, the pre-P8 version: 4.0.0b34 needs a newer CuTe
  DSL than the pinned 4.6.2 and broke R2's segmented FA4 backward. NVRTC moved from 13.0 to 13.4,
  so the NVRTC preload now finds the installed builtins library instead of naming 13.0.
- **Why:** Poorin runs only on Linux, and resolving for Windows failed on FlashAttention 4's
  pre-release. Dependencies may move freely (AGENTS.md), and D70 expected P8's resolve to move
  CUDA wheels; the fresh venv's checks on the RTX 5090 are the gate.
- **Cost if wrong:** A regression from one of 153 upgrades has to be bisected by pinning. The
  full check passed on the new venv: every suite, both roots, and Finelog. The venv shrank from
  11.9 GB to 7.8 GB, and R1 imports 5,985 modules instead of 6,611.

---

## Pass 2 edge cuts: 2026-10-07

Plan: `plans/2026-10-07-pass2-edge-cuts.md`.

### D81. TPU kernel imports and backward flow stay eager

- **When:** 2026-10-07, Pass 2 Phase A. **Status:** Adopted.
- **Decision:** The spec's verdicts for two initial targets change from Make lazy to Keep. The
  splash and ragged paged attention kernels in `levanter.layers.attention` and the `pallas_tpu`
  fused cross-entropy variant stay module-level imports. So does `levanter.analysis.backward_flow`
  in grug.
- **Why:** Measured with `python -X importtime` on R1 and R2 after P8: the TPU kernel imports add
  about 2 ms, since `haliax.nn.ragged_dot` already loads the Pallas TPU package, and backward
  flow's own import is 2.3 ms; its 755 ms cumulative figure is `levanter.tracker` and W&B loading
  first through it. Grug's forward pass calls backward flow's hooks, which are no-ops while the
  interval is 0, the launcher default since P0. Making either lazy would restructure the
  attention module and the kernel registry for no measurable gain.
- **Cost if wrong:** CPU and GPU runs keep loading about 15 TPU kernel modules.

### D82. Hugging Face conversion stays eager until sub-project 4

- **When:** 2026-10-07, Pass 2 Phase A. **Status:** Adopted.
- **Decision:** `levanter.compat.hf_checkpoints` stays a module-level import of the Llama, GPT-2,
  and Qwen model modules, the adaptor package, and `train_lm`. Sub-project 4, which imports
  pretrained vision and VLA weights through this module, decides its split.
- **Why:** It costs about 0.94 s per launch (`transformers` 0.42 s, and PyTorch 0.52 s, which
  `transformers` imports because it is installed). R1 loads it at runtime anyway to save its
  Hugging Face checkpoint. A clean cut moves `HFCompatConfig` and the serialization mixins out of
  the module, but they annotate with `transformers.PretrainedConfig`: a transformers-free module
  would need a `TYPE_CHECKING` import, which AGENTS.md forbids, or weaker types on the interface
  sub-project 4 builds on.
- **Cost if wrong:** Runs that never convert checkpoints pay for `transformers` at every launch.
  Corrected 2026-10-08 (D89): for grug's R2 the saving would be about 0.4 s, not 0.9 s. FlashAttention
  4's CuTe package imports PyTorch when grug's GPU attention first runs, so PyTorch's 0.5 s is paid
  either way; only `transformers` (and, since D88, scikit-learn) would move.

### D83. Pass 2's ranked loop ends on Keep verdicts; architecture.md is checked by a test

- **When:** 2026-10-07, Pass 2. **Status:** Adopted; superseded in part by D112 (the test is
  selected per change).
- **Decision:** After the Phase A cuts, the ten modules that dominate the most reachable lines
  all get Keep verdicts: Haliax and its `core` and `nn`, grug's attention and its FA4 CuTe
  backend, the fused cross-entropy loss and its API, `zephyr.coordinator`, and
  `levanter.layers.attention`. `docs/poorin/architecture.md` gives every one of the 466 workspace
  modules a one-line purpose and the root that reaches it, and lists 34 modules that look
  vestigial on one GPU as candidates for the next pass instead of deleting them now.
  `tests/tools/test_architecture.py` fails when a module has no row or a row names a missing
  module; it runs with `tools.affected_tests --all`, not per change.
- **Why:** The spec's end condition for Pass 2. The candidates are reachable from kept roots or
  training runs, so each needs a decision about its root rather than a sweep.
- **Alternatives:** Select the test whenever any module changes, which needs a new selection rule
  in `tools.affected_tests`.
- **Cost if wrong:** A module added or deleted between full runs leaves the page stale until the
  next one.

---

## Direction after the prune: 2026-10-07

### D84. Haliax and Levanter stay as libraries; Marin's framework runs the experiments

- **When:** 2026-10-07, after Pass 2. **Status:** Adopted.
- **Decision:** Research code imports Haliax and Levanter as libraries and runs its experiments
  through Marin's framework layer: lazy versioned artifacts and their fingerprints, the step
  runner, the experiment CLI, the data pipelines, the training wrappers, evaluation records, and
  scaling-law fitting. A model variant may use Haliax named tensors or plain arrays, grug style.
  Sub-project 1b no longer decides whether Haliax stays; it shrinks to an optional merge of
  Marin's small plumbing packages (`rigging`, `fray`, `finelog`, `finestore`) if their
  boundaries get in the way. The next step is the `projects/` integration (D52), which puts
  `looped-diffusion-lm` on these libraries and the framework.
- **Why:** The owner, 2026-10-07: they had read the workspace as one Marin framework, and are
  happy to use Haliax and Levanter as libraries with Marin's framework for experimentation and
  reproducibility. Measured: grug uses Levanter as infrastructure (data, checkpointing,
  optimizers, tracking, kernels) and imports three small Haliax helpers; the libraries are
  separate packages by origin but share one lock and editable installs, so their boundaries cost
  little.
- **Alternatives:** Replace Haliax with plain arrays and checked jaxtyping annotations
  everywhere, which means reworking Levanter's data, optimizer, and checkpoint layers; merge the
  workspace into one `poorin` package.
- **Cost if wrong:** Video and action data meet Levanter's language-model-shaped data interfaces
  (`LmDataConfig`, named-array batches); sub-project 3, the data layer, decides that.

### D85. The libraries Poorin cut heavily take poorin_ names

- **When:** 2026-10-07. **Status:** Adopted; D86 restores the bare import names of `fray`,
  `finelog`, and `rigging`. Spec: `specs/2026-10-07-library-renames-design.md`.
- **Decision:** `fray`, `finelog`, `rigging`, and `marin` become `poorin_fray`, `poorin_finelog`,
  `poorin_rigging`, and `poorin`: import names, distribution names (`poorin-fray`,
  `poorin-finelog`, `poorin-rigging`, `poorin`), and directories under `lib/`. The workspace root
  becomes `poorin-root`. Each renamed library's README says what Poorin removed from it and why.
  Levanter, Haliax, Zephyr, Finestore, and Dupekit keep their names; so do the `MARIN_*`
  environment variables, `marin_*` helpers, bucket paths, upstream's `marin-finelog-server` wheel,
  and Finelog's protobuf package names. One library per commit; the owner authorized running the
  spec, plan, and commits end to end.
- **Why:** The owner, 2026-10-07: the libraries Poorin modified heavily should not carry upstream's
  names, and the docs should say explicitly that each was made lighter. Measured against
  `upstream-baseline`: Marin's code −64%, Finelog −66% plus its 75k-line Rust server, Fray −30%,
  Rigging −28%.
- **Alternatives:** Rename distributions only, keeping imports; rename every library Poorin
  touched, Levanter included; document without renaming.
- **Cost if wrong:** Reading upstream code needs the name map in AGENTS.md. Artifact records name
  their class by module path, so cached artifacts are migrated once with
  `find ~/.cache/poorin -name .artifact.json -exec sed -i 's/"result_type": "marin\./"result_type": "poorin./' {} +`.

### D86. fray, finelog, and rigging keep their import names

- **When:** 2026-10-08. **Status:** Adopted. Amends D85.
- **Decision:** The imports and directories of `poorin_fray`, `poorin_finelog`, and
  `poorin_rigging` go back to `fray`, `finelog`, and `rigging` (`lib/fray`, `lib/finelog`,
  `lib/rigging`). Their distributions stay `poorin-fray`, `poorin-finelog`, and `poorin-rigging`,
  so the lockfile still marks them as Poorin's forks, and their READMEs keep the lists of what was
  removed. Marin's core stays `poorin`.
- **Why:** The owner, 2026-10-08: the utility libraries do not need the prefix. Upstream used the
  same split, `marin-rigging` installing `rigging`.
- **Alternatives:** Bare distribution names too, which collide with unrelated PyPI projects such as
  `rigging`; return to upstream's `marin-` distribution names.
- **Cost if wrong:** None at runtime; reading code no longer shows which utility libraries Poorin
  forked, which their READMEs and the AGENTS.md map still record.

### D87. Poorin's settings take Poorin names

- **When:** 2026-10-08. **Status:** Adopted. Follows D85.
- **Decision:** Every environment variable that configures Poorin drops its `MARIN_` prefix for
  `POORIN_`: `POORIN_PREFIX`, `POORIN_TEMP_PREFIX`, `POORIN_CLUSTER`, `POORIN_CLUSTER_CONFIG_DIRS`,
  `POORIN_FINGERPRINT_STRICT`, `POORIN_I_WILL_PAY_FOR_ALL_FEES`, `POORIN_MIRROR_BUDGET_GB`,
  `POORIN_PROVENANCE`, and `POORIN_GIT_COMMIT`. The helpers follow (`poorin_prefix()`,
  `poorin_temp_bucket()`, `poorin_region()`), with the constants, tests, and private names built
  on them. The output prefix falls back to `/tmp/poorin` when unset, and the default W&B project
  is `poorin`. `MARIN_CI_DISABLE_RUNTIME_ENVS`, a CI leftover, is deleted (D70).
- **Kept:** names that point at something of Marin's: the "Copyright The Marin Authors" license
  header (`MARIN_HEADER`), the Marin tokenizer and its chat template (`marin_tokenizer`,
  `MARIN_CHAT_TEMPLATE`, `MARIN_CUSTOM_SPECIAL_TOKENS`), `gs://marin-*` bucket names, Marin's
  cluster fixtures under `config/testing/`, and W&B test data naming upstream runs.
- **Why:** The owner, 2026-10-08, after the library renames.
- **Cost if wrong:** A shell or a script that still exports `MARIN_PREFIX` is ignored and outputs
  go to `/tmp/poorin`. On 2026-10-08 nothing in the owner's shell set a `MARIN_*` variable, and the
  owner's SkyRL fork references none.

### D88. Verdicts on the workflows no training check runs

- **When:** 2026-10-08. **Status:** Adopted. Plan: `plans/2026-10-08-workflow-verdicts.md`.
- **Decision:** Of the 35k lines that only kept roots reach, the owner keeps data preparation,
  evaluation, scaling laws, profiling, post-training, the SkyRL hooks, and all optimizers. The
  five dataset recipes of the Delphi mix (Nemotron-CC, StarCoder, ProofPile, Paloma,
  uncheatable-eval) are deleted, as are the evaluation modules left from the deleted serving flow
  (`eval_env`, `model_config`, `model_identity`, `lm_eval_samples`). The `lm_eval` extra joins the
  standard sync so the harness runs; scaling laws gets its first tests; `train_dpo` and `lora_lm`
  get end-to-end smoke runs.
- **Why:** The owner, 2026-10-08, accepting the per-workflow suggestions. The recipes expect raw
  data under the prefix that is not there (Nemotron-CC is terabytes in upstream buckets); the
  evaluation leftovers describe a serving and Evalchemy flow that P3 removed; the kept workflows
  had no evidence they still run. The `lora_lm` smoke run found two bugs it had carried since
  upstream: it read a renamed setting (`max_train_length`, now `train_seq_len`) and crashed
  before training, and it never finished its tracker, so W&B and Finelog runs never closed.
- **Cost if wrong:** Reproducing Delphi's data mix needs the recipes from `upstream-baseline`. The
  `lm_eval` extra installs scikit-learn, which `transformers` imports at load time when present, so
  every launch imports about 540 more modules and spends about 0.25 s longer in imports (R1: 3.37 s
  to 3.62 s); the cut D82 defers to sub-project 4 would remove it.

### D89. Verdicts on the review modules and the Deferred table

- **When:** 2026-10-08. **Status:** Adopted; superseded in part by D90 (Zephyr's dashboard
  stays). Plan: `plans/2026-10-08-backlog.md`.
- **Decision:** The plan's tables: delete the backward-compatibility shims (`fray.cluster`,
  `levanter.main.lora_dpo`), the Iris-era and fleet pieces (Zephyr's dashboard, Rigging's NCCL
  stall probes, Levanter's recovery types, pipeline partitioning, TPU VM shutdown, and gcloud
  arguments), and two unused Finelog client modules; keep the TPU kernels, GPT-2, the Hugging Face
  model cache, `mirror://`, Finestore's migration, and fast-track's QuACK path, each for a stated
  user. Close every open Deferred row: fix the tests and tools it names, make ty fail on warnings
  now that none remain, and record the rows not worth fixing.
- **Why:** The owner, 2026-10-08: address the backlog. QuACK was flagged for not running on the
  RTX 5090; it runs only for expert-parallel meshes, which one GPU never forms, so fast-track's
  MuonH works locally.
- **Cost if wrong:** A multi-node cloud run loses NCCL stall diagnostics, which return from
  `upstream-baseline` if needed.

### D90. Keep Zephyr's dashboard

- **When:** 2026-10-08, backlog. **Status:** Adopted.
- **Decision:** Keep `zephyr.dashboard` as it is: the coordinator's status page (pipeline phase,
  stage graph, shard progress, counters, workers), its read-only JSON API on Starlette, and the
  queries over coordinator state. D89 had it deleted.
- **Why:** The owner, 2026-10-08: it may be useful for local data runs, and it is already light.
  It is one HTML page with no frontend build, about 1,000 lines in all, and Starlette is its only
  dependency.
- **Alternatives:** Serve the page with the standard library's `http.server` and drop Starlette;
  write a static snapshot file instead of serving; replace it with a fuller progress log line.
  The owner kept Starlette.
- **Cost if wrong:** About 1,000 lines and the Starlette dependency stay. Nothing serves the page
  yet: the coordinator builds the app, but nothing has mounted it since Iris was deleted (Deferred).

### D91. Local defaults that replace the Iris-era ones

- **When:** 2026-10-08, backlog. **Status:** Adopted.
- **Decision:** With Zephyr's 1,000-replica Iris cap gone, an unset worker count is one worker
  per CPU everywhere, as `LocalClient` already did. `compute_fuzzy_dups_attrs` requires
  `max_parallelism`, as its step builder already did, instead of defaulting to the cap.
  `verify_fuzzy_dups` falls back to one worker per shard, at most one per CPU. The profiler logs
  an XProf viewer link only when `XprofUploadConfig.service_url` names a service; it defaulted to
  `iris.oa.dev`.
- **Why:** On `LocalClient` a worker is a thread in the launching process, so the old defaults
  meant up to 1,000 threads for a dedup run, and a link to a service Poorin cannot reach.
- **Cost if wrong:** A cloud backend (sub-project 2) chooses its own worker counts and passes
  them explicitly.

### D92. Typed overrides, sweeps, and fail-closed drift, on draccus

- **When:** 2026-10-08, design; landed 2026-10-09. **Status:** Adopted. Spec:
  `specs/2026-10-08-experiment-overrides-design.md`.
- **Decision:** Experiments gain `--set PATH=VALUE`, `--sweep`, a readable plan with
  `--show-config`, and `--accept-drift`, built on draccus. A changed recipe under an existing
  output becomes an error, checked over the whole graph before a run, and the config class enters
  the fingerprint. The owner's rulings: `--set` reaches the whole config; an overridden trial's
  name spells its overrides when they are plain, else a digest; `--accept-drift` re-stamps the
  record; tracker settings (W&B config, run ID, log cadence) leave identity; `--override` is
  renamed `--version-of`; the experiment applies its overrides itself.
- **Why:** The owner, 2026-10-08: bring Hydra's strengths into the experiment workflow. A probe
  found four ways a stale cached output is served today, one of them silent; overrides would make
  quick recipe changes routine, and a stale checkpoint behind a published result is the costliest
  failure.
- **Alternatives:** Hydra or hydra-zen (D51 already chose the executor; OmegaConf cannot represent
  Levanter's union and registry types); tyro, chz, or fiddle (a second representation of Levanter's
  configs); the framework applying overrides implicitly; generating per-field flags with
  `draccus.parse` (needs the config before the step context exists).
- **Cost if wrong:** Every fingerprint changes once, absorbed with `--accept-drift`; about 300
  lines of code and 200 of tests.
- **Landed:** `6b92482ce`, `090f458bc`, `e71d9c065`, `c0228a683`; rulings made while building
  them are in D94.

### D93. Track runs with W&B; no host-specific test exclusions

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** Delete Levanter's TensorBoard tracker, its test, its example config, and its
  `tensorboardX` dependency; runs track with W&B (and Finelog telemetry). With that test gone, no
  host-specific exclusion remains, so `tools.affected_tests` loses the exclusion mechanism D38
  added. The `tensorboard` package stays in Levanter's `profiling` extra, as a viewer for profiles.
- **Why:** The owner, 2026-10-09: "we'll just use wandb". D38 had assumed P7 would delete the
  tracker; it did not, and its test was skipped on this workstation ever since.
- **Alternatives:** Install Levanter's test group so the test runs (28 more packages, D38).
- **Cost if wrong:** A run that wants TensorBoard scalars logs to W&B or Trackio instead; the tracker
  returns from `upstream-baseline` if needed.

### D94. Rulings while building D92

- **When:** 2026-10-09, building D92. **Status:** Adopted; superseded in part by D96 (a quoted
  override value keeps its quotes in the trial's name) and D98 (trainings run one at a time, so a
  multi-trial `--sweep` no longer needs `--max-concurrent 1`).
- **Adopted artifacts are drift-checked.** The spec exempted them with pins, but an adopted
  artifact writes a record at its canonical address, and re-adopting the same `name@version` from
  another source must be refused, as it was warned about before.
- **Two recipes for one `name@version` in a graph are refused.** Several handles can build one
  output: a sweep's trials each construct the shared dataset. The plan's check went handle by
  handle, so it listed that output once per handle, and accepting it appended the new fingerprint
  to `accepted_from` after the old one. Handles that share a `name@version` and a fingerprint now
  count as one artifact. Handles that share a `name@version` but not a fingerprint are two recipes
  claiming one output path, which the runner would build or serve once for both, so the run is
  refused before anything runs, and the error names the label and both fingerprints.
- **The runner's check skips pinned steps.** The spec exempts pins (`override_path`), which point
  at existing data, but the plan's `check_drift` compared them, so a pin to an output with another
  recipe raised an error whose `--accept-drift` advice the pre-run check then rejected. A pinned
  step carries `pinned` in its `hash_attrs`, and `check_drift` does not compare it.
- **`GrugBaseLaunchConfig` is exempt from the tests' beartype hook** (`@no_type_check`, as
  `GrugTrainState` is): while the fingerprint is computed, its runtime fields hold placeholder
  strings. Computing grug's fingerprint under the hook failed on `resources` before this change.
- **`train_lm` sets the tracker's `replicate_path` in `build_config`, not in its run function.** The
  spec sets it in the run function, from the pod config's `output_path`. `train_lm` builds the
  tracker inside `TrainerConfig`, and `ctx.output_path` is the value the pod config carries, so
  `build_config` can set it directly; the run function would have to rebuild the nested trainer
  config to replace it. At fingerprint time `_run_tracker` leaves the tracker's placeholder alone,
  so the path never enters the fingerprint. Grug follows the spec: `_trackers` sets it from
  `launch.output_path`.
- **Overlapping `--set` paths are refused.** One trial cannot set both a field and a field beneath it
  (`optimizer` and `optimizer.learning_rate`): applied in command-line order, the result would
  depend on that order while the trial's name and run ID would not. Set the node in one mapping
  value instead (`optimizer={type: muon, learning_rate: 0.02}`).
- **`train_lm` needs a `run_id` when overrides are set**, so each trial logs to its own tracker run;
  without one, Levanter names the run after the version, which every trial shares.
- **Overrides read field types from each class's annotations.** The spec decodes a value against
  the field's type from `typing.get_type_hints`, which returns no hints for a class marked
  `@no_type_check`. `GrugBaseLaunchConfig` is marked (see above), so `--set steps=200` on grug
  failed with a `KeyError`. `overrides` reads the annotations of the class and each base with
  `inspect.get_annotations`, which ignores the marker.
- **A `--sweep` string element keeps its text only when YAML reads that text back as the same
  string.** The plan kept every string element's text, which YAML parses again when the trial
  applies it, so a quoted `'yes'` or `'a: b'` reached a `str` field as `'True'` or
  `"{'a': 'b'}"`, while the trial's name came from the text as typed. Any other element, a quoted
  string of that kind included, is written back as JSON. `1e-3` stays text, so it still decodes to
  0.001 for a float field.
- **The root test suite's fixture cluster directory is absolute.** Since the pre-run drift check,
  `run` reads the prefix, and so loads the cluster config, before the runner starts.
  `test_consumer_resolves_dependency_under_relative_prefix` changes the working directory first, so
  the relative `config/testing` resolved under its temporary directory and the load failed. The
  test passed only when an earlier test in the same process had cached the config; this commit's
  tests changed that order. `conftest.py` now anchors the directory to the checkout.
- **A run of a sweep with more than one trial needs `--max-concurrent 1`.** The spec required an
  explicit `--max-concurrent` so the default of 8 would not start 8 trainings on one GPU.
  Correctness requires 1: Fray's `LocalClient` runs each trial's job as a thread of one process,
  Levanter's tracker is process-global, and outside a notebook a second `wandb.init` returns the
  W&B run the process already has open. Trials running at once log into one W&B run and can write
  `tracker_metrics.jsonl` into another trial's output, which sweep selection reads. Under `--run`,
  any other value, and no flag, is a usage error; the plan and `--show-config` need no flag.
  `--max-concurrent` accepts only positive integers, because the runner read 0 as its default.
  The rule holds until each trial gets its own tracker. **Superseded by D98**, which runs a
  process's trainings one at a time; `--max-concurrent` still accepts only positive integers.
- **An override's name spells the value YAML reads.** The plan normalized a value's text with
  Python's `int` and `float`, so `steps=010`, which YAML 1.1 decodes to 8, named its trial
  `steps=10`. The name now starts from YAML's reading: a number or a boolean is spelled as Python
  writes it, and a string that Python reads as a number is spelled as that number, so `1e-3`,
  which YAML 1.1 reads as a string, still shares a name with `0.001`. Text YAML cannot read keeps
  its spelling, and `apply` reports the error.
- **A string built from a placeholder is a context value.** `StepContext.is_context_value`
  matched only a value equal to a placeholder, so `--set` could replace a path built from one,
  such as `<output_path>/hf`. It now also matches any string that contains `<output_path>`,
  `<prefix>`, `<region>`, or a runtime argument's placeholder; a dependency reference still has to
  match whole. None of the paths that `train_lm` overwrites at run time is refused by this:
  `build_config` leaves them at their defaults, and `_prepare_training_run` sets them when the job
  starts, after the overrides apply. A `--set` on `hf_save_path`,
  `trainer.checkpointer.base_path`, `trainer.checkpointer.temporary_base_path`,
  `trainer.checkpointer.append_run_id_to_base_path`, `data.auto_build_caches`, or, for a job on
  CPU, `trainer.require_accelerator` enters the trial's name and fingerprint and never reaches the
  run. The overrides docs list them as set at run time and not overridable.

### D95. The launch config holds environment variables only, under a Poorin path

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** Levanter's `CliConfig` keeps `env` and `accel_env`, the only fields anything reads
  (training's `resolve_training_env` calls `env_for_accel`). Its 26 cluster fields (TPU, zone,
  Docker image and registry, GitHub token) and the unused `as_dict` go. The user file moves from
  `~/.config/marin/config.yaml` to `~/.config/poorin/config.yaml`, as D87 moved the other settings,
  and the deprecated `.config` fallback beside `.levanter.yaml` goes.
- **Why:** The owner approved the cleanup on 2026-10-09. The fields configured Marin's TPU and
  Docker launchers, which the prune deleted. The `.config` fallback would try to read a directory
  if one were ever present.
- **Alternatives:** Keep the fields as dead configuration; keep the Marin path.
- **Cost if wrong:** A config file that still sets a removed field fails to load, because draccus
  rejects unknown keys. No such file exists on this workstation. A file under the Marin path is
  no longer read.

### D96. A quoted override value keeps its quotes in the trial's name

- **When:** 2026-10-09, after the D92 review. **Status:** Adopted.
- **Decision:** An override's name spells a YAML string as the number Python reads from it only
  when the value was unquoted, that is, when YAML's reading equals the text. A quoted value keeps
  its text, quotes included, so its trial takes the digest name. This narrows D94's ruling that a
  string Python reads as a number is spelled as that number.
- **Why:** YAML strips the quotes, so on a `str` field `--set label='010'` decodes to the text
  `010` but was named `label=10`, the name `--set label=10` gets for the text `10`. The drift check
  refused whichever ran second, so no stale output was served, but the name misstated the value.
  `1e-3` is unquoted, so it still shares a name with `0.001`.
- **Alternatives:** Name a quoted value by the text inside the quotes. On a `str` field,
  `'0.001'` would then share the name `0.001` with an unquoted `1e-3`, which decodes to the text
  `1e-3`.
- **Cost if wrong:** A quoted number on a numeric field, such as `steps='10'`, decodes to the
  value `steps=10` gives but names a separate trial, which builds again. One misleading name
  remains: the name is computed without the field's type, so on a `str` field `1e-3` and `0.001`
  share a name while decoding to different text; the drift check refuses the second, as before.

### D97. Each calendar version of a training run gets its own tracker ID

- **When:** 2026-10-09, after the D92 review. **Status:** Adopted.
- **Decision:** `train_lm` gives its tracker run the ID `{run_id}-{version}` for every version, and
  a trial appends its overrides' digest. D31 suffixed only a `dev` version; a calendar version got
  `{run_id}`. Grug's ID already carries the version (`grug-base-{version}`).
- **Why:** Two calendar versions of one run are two outputs, but they shared a W&B run. Levanter's
  W&B tracker resumes a run by ID (`resume="allow"`) and refuses to log a step below the resumed
  run's current step, so the later version's metrics up to the earlier version's last step never
  reached W&B, and the `tracker_metrics.jsonl` it writes from the W&B run's summary could carry
  the earlier version's values. A rerun of one version keeps its ID, so a run resumed after a crash
  continues its W&B run.
- **Alternatives:** Keep `{run_id}` and leave the author to change `run_id` with each version.
- **Cost if wrong:** A calendar-versioned run started before this change and resumed after it
  opens a new W&B run, so its history splits across two. W&B run IDs and names grow by the
  version, 11 characters for `-2026.10.09`.

### D98. One training at a time in a process

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** A training holds a process-wide lock, `poorin.training.training.exclusive_training`,
  for its whole run. `run_levanter_train_lm`, `run_levanter_train_dpo`, and grug's Fray job
  (`dispatch_grug_training_run`) take it. A training that would start while another runs in the
  same process waits for it to finish. Other steps still run up to `max_concurrent` at once. The
  command line no longer requires `--max-concurrent 1` for a `--run` of a sweep with more than one
  trial (D94).
- **Why:** Fray's `LocalClient` runs each job as a thread of the process. Levanter's tracker is a
  process global: `TrainerConfig.initialize` replaces it, and grug ends by finishing
  `current_tracker()`. Outside a notebook, a second `wandb.init` returns the W&B run the process
  already has open. D94's rule, `--max-concurrent 1` for a `--run` of a `--sweep`, covered only that
  flag; a sweep built in Python and run with `run` at its default `max_concurrent=8`, or any graph
  with several training steps, still started trainings at once. Two one-step grug runs on CPU,
  started together, both logged into the tracker of the run that started second. That tracker was
  finished twice, once by each run, and the first run's tracker never finished.
  `test_grug_runs_started_together_each_log_to_their_own_tracker` reproduces it.
- **Alternatives:** A tracker per job: a `ContextVar` in place of Levanter's global, W&B runs
  created with `reinit="create_new"`, and `WandbConfig.init` writing through its own run in place of
  the module globals `wandb.summary` and `wandb.run`. Two trainings would still share the rest of
  the process: each run adds a file handler to the root logger, so each run's log file would get
  both runs' lines, and they would share `os.environ` and GPU memory. Refusing a second
  training with an error: a sweep would fail at whichever trial started second, and whether it
  failed would depend on timing. Taking the lock in `TrainerConfig.initialize` and releasing it in
  `finish`: a training that raised in between would keep it, and the next training would hang.
- **Cost if wrong:** Trainings that could share the process, such as two small CPU runs, run one
  after another. Another Levanter main run in-process, such as an evaluation that calls
  `levanter.trainer.initialize`, does not take the lock and still shares the tracker; no step runs
  such a main today.

### D99. Rigging's per-user cluster configs live under a Poorin path

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** The per-user cluster config directory moves from `~/.config/marin/clusters` to
  `~/.config/poorin/clusters`, as D95 moved the launch config.
- **Why:** The owner, 2026-10-09: the rigging cluster directory gets the rename too.
- **Alternatives:** Keep the Marin path.
- **Cost if wrong:** A cluster config under the Marin path is no longer found. Neither directory
  exists on this workstation.

### D100. Serve Zephyr's dashboard on localhost, for Tailscale

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** `ZephyrContext` gains `dashboard_port: int | None`; unset, it comes from
  `ZEPHYR_DASHBOARD_PORT`, as `max_workers` comes from `ZEPHYR_MAX_WORKERS`. With a port, the
  coordinator serves the dashboard app it already builds with uvicorn, on a daemon thread bound to
  `127.0.0.1`, and stops it on shutdown. A taken port moves the dashboard to a free one the OS
  picks, and the coordinator logs the URL. Remote access goes through `tailscale serve --bg
  <port>`, which serves the page over HTTPS on the tailnet. Zephyr depends on uvicorn.
- **Why:** The owner, 2026-10-09: the dashboard should be reachable over Tailscale. Nothing had
  served it since Iris (D90). About fifteen call sites in `poorin` build a `ZephyrContext`, so a
  field alone could not be set from the command line; the environment variable reaches all of
  them, and an explicit field still wins. Binding localhost keeps the page off other interfaces.
- **Alternatives:** Plumb the port through every pipeline; bind `0.0.0.0` (or the Tailscale
  address) directly; fail when the port is taken, which would fail a data pipeline over its
  dashboard; try the next ten ports, which a test suite with many contexts can exhaust.
- **Cost if wrong:** A second concurrent pipeline's dashboard is on a random port that `tailscale
  serve` does not proxy; its log line gives the local URL. One more dependency (uvicorn, pure
  Python, with h11).

### D101. No outside contributions; Marin's community files go

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** Poorin does not take outside contributions for now. The README describes Poorin
  instead of Marin: what the fork is and removed, its measured size since `upstream-baseline`, its
  libraries, setup, experiments, development, and credit to Marin, Levanter, and Haliax.
  `CONTRIBUTING.md` says contributions are not taken. `CODE_OF_CONDUCT.md` and Levanter's and
  Haliax's `CONTRIBUTING.md` are deleted, and their README sections say the same. The docs site's
  contributing page becomes a Development page with the maintainer's workflow, without the issue,
  pull-request, and Discord process, and the site drops Marin's Discord and repository links.
  `AUTHORS.md` and the libraries' `AUTHORS.md` and `CONTRIBUTORS.md` stay, because the license
  headers name the Marin Authors.
- **Why:** The owner, 2026-10-09: "dont have contributors for now". The files described Marin's
  community, its Discord, and its pull-request process.
- **Alternatives:** Keep the Contributor Covenant for when contributors arrive; keep the community
  files unchanged.
- **Cost if wrong:** Inviting contributors later means writing a contribution guide and code of
  conduct; Marin's are in `upstream-baseline`.

### D102. Haliax comes from PyPI, pinned at the fork point

- **When:** 2026-10-09. **Status:** Adopted.
- **Decision:** Poorin depends on the published `marin-haliax==0.2.139.dev37117265303` instead of
  carrying `lib/haliax`. That build is Marin's first nightly after the fork point, and its
  source is identical to `upstream-baseline`'s `lib/haliax/src` apart from the version string
  (checked file by file; 0.2.138 and 0.2.140 are identical too). Poorin's own Haliax commits
  changed formatting, type-checker comments and casts, and dropped `jmp`; none changed behavior.
  `jmp` returns as a Haliax requirement; Levanter already locks it. `lib/haliax` is deleted, with
  its lint header rule, its test-suite device setting, its architecture rows, and Levanter's TPU
  Dockerfiles, which bind-mounted its `pyproject.toml` and which nothing builds.
- **Why:** The owner, 2026-10-09, asked whether Levanter and Haliax could come from
  `pyproject.toml` instead of the tree. Haliax carried no Poorin change worth keeping, and its
  package depends on no Marin library. Levanter stays in the tree: its 47 commits since the fork
  remove 41,400 lines and add fork-only code (`nvrtc`, `compile_cache`), and upstream
  `marin-levanter` requires `marin-iris`, `marin-fray`, `marin-rigging`, and `marin-finelog`,
  which install the same import packages as Poorin's renamed libraries.
- **Alternatives:** Keep `lib/haliax`; depend on the standalone `haliax==1.3` (September 2026,
  older than the fork point); track Marin's latest nightly.
- **Cost if wrong:** A Haliax fix means raising the pin or vendoring it again; the nightly line
  follows Marin's main branch, so a newer pin can change behavior. `upstream-baseline` holds the
  old copy.

### D104. Grug's training data is pinned to a Hub revision

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** Grug's `fineweb-edu-10M` passes `revision="f13fbe24e962f9e20d75c3c6f08830bcfd0606b8"`
  to `pretokenized`: the head of `marin-community/fineweb-edu-pretokenized-10M`, unchanged since
  2025-05-15. The dataset keeps its version, 2026.06.28. The copy in prune_check's cache was
  downloaded in October 2026, after that commit and before any other, so it is this revision's data;
  its changed recipe (only `hf_revision`, from none to the commit) was accepted with `accept_drift`.
- **Why:** The owner, 2026-10-11. An unpinned download follows the repository's default branch, so
  a new upload would change the data under an unchanged version, and a fresh prefix would train on
  different tokens than the cache did.
- **Alternatives:** Bump the dataset version and download again (about 20 MB); leave it unpinned.
- **Cost if wrong:** Another prefix holding a copy from before the pin refuses with a diff naming
  `hf_revision` until it is accepted the same way.

### D105. A simplify pass over the fork's own code

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** Four reviews (reuse, simplification, efficiency, altitude) read the code this
  fork wrote or rewrote since `upstream-baseline`. Applied:
  - The experiment CLI computes fingerprints, which read nothing, to apply and check overrides,
    and reads each artifact's recorded recipe only for the plan or `--resume`; `run()` reads them
    itself. `--run` no longer reads every record and status twice before the runner starts.
  - `cache_state.graph_states` serves the plan and `run()`: handles that share a `name@version`
    and a recipe are one artifact, read once (a sweep no longer re-reads its shared dataset per
    trial), and two recipes for one `name@version` are refused by the plan too, not only by
    `--run`.
  - A fixed version over a `dev` dependency is refused when the handle is constructed, not when
    it is lowered, so the plan no longer lowers each handle (six Git calls apiece for provenance
    it discarded) and `run()` has no ordering constraint.
  - Smaller: `--max-concurrent` defaults to 8 directly, `--resume` reuses `--version`'s
    validation, the training lock is a plain `with`, and the tools share one module-name helper,
    one Git file listing, and `reachability.closure`; Zephyr's memory limit is resolved once per
    process through `psutil`, its unused CPU limit goes, and the dashboard waits for its server
    with `rigging.timing.ExponentialBackoff`; Levanter's unread `max_gen_toks` compatibility
    property on the harness config goes.
- **Not applied, and why:** the runner indirection (`build_options_with_runner`) is the seam D68
  keeps for launching SkyRL runs; the runner's own drift check is what the spec requires for code
  that calls `StepRunner` directly; deriving tracker run IDs from the step name and moving
  `replicate_path` into `apply_output_path` would reverse D94 and D97 rulings and change W&B IDs;
  narrowing the tests' beartype hook would drop its checks of grug's dispatch, checkpointing, and
  optimizer code; batching object-storage requests helps only a remote prefix; training each root
  once in `prune_check` would change how its metrics are measured; the eval-harness config split
  is code moved from upstream.
- **Why:** The owner asked for a `/simplify` pass of the project on 2026-10-11.
- **Cost if wrong:** A plan now fails where it used to print a graph with two recipes for one
  output; that graph always failed at `--run`.

### D106. The tutorial's devices are the two this workstation has

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** `experiments/tutorials/train_tiny_model.py` keeps the `cpu` and `rtx5090x1`
  devices. The `h100x1`, `h100x8`, `gb200x1`, `gb200x4`, `v5litepod-16`, and `v6e-4` entries go,
  with the comment on routing GPU jobs across CoreWeave regions and the `ANY_REGION` import.
- **Why:** The owner, 2026-10-11. Fray's `LocalClient` is the only backend, so those entries ran
  on this machine under a cloud device's name and batch size; the tutorial's docstring had kept
  them for a cloud backend that sub-project 2 has not built.
- **Alternatives:** Keep them until sub-project 2 adds a cloud backend.
- **Cost if wrong:** Sub-project 2 adds the cloud devices it supports back to `DEVICES`, one entry
  each.

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
- **Rulings while building:** the owner asked, 2026-10-11, for two changes to the run list. It
  resolves every commit a record names with `git rev-parse --verify --end-of-options` and passes
  only the verified hash to Git, so a crafted record cannot inject a Git option. Its `show` names
  why a run has no code: a build still running, a failed build, a finished output without a
  record, or a record written before code provenance.
- **Why:** The owner, 2026-10-11: the fingerprint covers config, not code, and a record named its
  code by a hash that was lost for uncommitted work and unreadable at a glance. The owner chose
  the split by version over refusing every dirty run or never refusing.
- **Alternatives:** Refuse every `--run` on a dirty tree; never refuse and always snapshot;
  enforce the rule inside `run()`, which would break every test run on a dirty tree.
- **Cost if wrong:** A first run on an empty prefix with a dirty tree is refused, because it would
  build its calendar dataset; commit first or pass `--allow-uncommitted`. Snapshot refs stay local
  until pushed with `git push origin 'refs/poorin/*:refs/poorin/*'`.

### D108. Follow-ups to code provenance

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** Four changes from D107's final review, and a cleanup in the full check.
  - The refusal no longer suggests a dev version. The refused run may itself be a dev run whose
    calendar input would build. Its advice is "Commit the changes and run again, or pass
    `--allow-uncommitted` to build from a snapshot." Outside a Git checkout the refusal says the
    build would happen outside a checkout, and its advice is to run from one or to pass
    `--allow-uncommitted`. This replaces the refusal text the D107 spec gives.
  - Snapshots commit as a fixed identity, `Poorin snapshot <snapshot@poorin.invalid>`. A machine
    with no Git identity, such as a fresh container, no longer fails with a raw `commit-tree` error,
    and one dirty state gives one commit on any machine. The record's `built_by` names who ran it.
  - `runs show` prints, in its `git diff` and `git worktree add` commands, 12 characters of the
    hashes Git verified, never the record's own text, and says "1 file" for one file.
  - The docs run the run list with `uv run --no-sync` and say that a snapshot holds every untracked
    file `.gitignore` does not exclude, before suggesting a push of `refs/poorin/*`.
  - `tools.prune_check` deletes the snapshot refs its own training runs created: a ref that appeared
    during the check, that a run output it removes names, and that no output it keeps names.
- **Why:** The owner asked for the review's optional follow-ups on 2026-10-11, and for the check
  to clean up after deleting the nine refs earlier checks had left. The old advice was wrong for the
  case the D107 spec gives as its example: a first dev run on an empty prefix, refused because its
  calendar dataset would build.
- **Alternatives:** Keep each snapshot's author as the user's Git identity, and turn a missing
  identity into a readable error.
- **Cost if wrong:** Snapshot commits no longer carry the user's name; the record's `built_by`
  still does. A run of the owner's that snapshots the identical working tree during a full check
  shares the check's commit and loses its ref; the commit stays in Git's object store until garbage
  collection.

### D109. One path through the experiment and execution layers

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** A second `/simplify` pass, over the experiment workflow, the execution layer, and
  `rigging.provenance`, judged by one rule from the owner: one person can understand the whole
  project. Each change removes a path no experiment takes or a second way to do one thing.
  - The step runner runs `step.fn(output_path)` and nothing else. Its own Fray submission
    (`StepSpec.resources`, the `RemoteCallable` unwrap) goes; a remote step's function submits
    itself through `poorin.execution.remote`, as every builder already did. `dry_run`,
    `force_run_failed`, `PreviousTaskFailedError`, and the never-written `DEP_FAILED` status go:
    a failed step always reruns.
  - `StepSpec` is only the lowered form of an `ArtifactStep`. It carries `version`,
    `fingerprint`, `fingerprint_payload`, and `pinned` as fields in place of the `hash_attrs`
    dict and its string keys; `hash_id`, `name_with_hash`, `output_path_prefix`,
    `writes_record`, and the plain-step record path (`write_step_record`, `StepRecordIdentity`)
    go. The plain-`StepSpec` factories in datakit and processing (download, normalize, MinHash,
    fuzzy dedup, decontamination, attribute tokenizing, store building) go with them, and only
    tests called them; the functions they wrapped stay.
  - The readers of Marin's on-disk formats go: the Ray `.executor_info` sidecar, the pre-#6649
    bare-payload record, and the JSON-lines status file. Every output under the owner's prefix
    has an `.artifact.json` and a one-token status. A started build records its recipe in
    `.artifact.started.json` in place of the Ray-shaped `.executor_info` stub, so nothing needs a
    tag to tell the stub from a record. This replaces the stub the overrides spec names.
  - Adopting existing data (`ArtifactStep.adopt`) goes; a pin is the one way to use data built
    elsewhere.
  - Launch provenance reaches a record one way: the step writes `launch_provenance()`, which
    `launched_as` sets for a command-line run. The `provenance=` parameter of `run`, `lower_all`,
    and the CLI's runner seam goes, and so do `lazy.lower` and `ArtifactStep.lower`, which captured
    their own. W&B uses the installed launch after one `git rev-parse` in place of a capture of its
    own.
  - Smaller: `BuildContext` (one field) gives way to the `VersionCodex` it held; `sweep.py` goes
    (`grid` moves into `cli.py`, and `sweep()` had only test callers); `dataset_main`,
    `hf_download`, `raw_download`, and `tokenized(raw=, glob=)` go; `train_lm` loses six
    parameters no call passed; `rigging.provenance` loses `from_git`, `dedup_key`, `from_json`, and
    `to_json`, has one strict and one lenient Git runner, and owns listing and deleting snapshot
    refs for `tools.prune_check`. The step concurrency default, the path rooting at the prefix,
    the `{prefix}/{name}/{version}` address, the `name@version` label, the status file name and a
    lock-free status read each have one definition. The plan builds each handle's config two fewer
    times, and a printed Git command shows 12 characters of a hash everywhere.
  - The tutorial and grug share one FineWeb-Edu handle, `experiments/datasets/fineweb_edu.py`, so
    the tutorial's copy gains D104's pinned revision; the two copies had given one
    `name@version` two recipes.
- **Not applied, and why:** `tree_hash` stays in the record and in W&B; dropping it changes both.
  The data and dedup functions the deleted factories wrapped stay: they are the workbench's data
  preparation (`tools/roots.toml`). `apply` stays, as the documented way to write a custom step.
  `runs list` still reads every record before filtering, since its sort needs each record's time.
- **Why:** The owner asked for a `/simplify` pass of the experiment workflow and modules on
  2026-10-11, with the goal that a human can understand the entire project, and chose to delete
  the legacy readers, keep pin over adopt, type the step's identity, delete datakit's step
  builders and `train_lm`'s unused parameters, and keep one provenance channel.
- **Cost if wrong:** An output built by Marin's Ray executor, or a build started before this change
  and not finished, no longer reads as recorded; rebuild it or write its record. Data built
  elsewhere is pinned, and a pin is not drift-checked.

### D110. An episode data layer, read through Grain

- **When:** 2026-10-11. **Status:** Adopted. Spec:
  `specs/2026-10-11-episode-data-layer-design.md`; plan: `plans/2026-10-11-episode-data-layer.md`.
- **Decision:** Sub-projects 3, 4, and 5 come next, in that order; 1b, 2, 6, 7, and 8 wait.
  Episode data (video, trajectories, latents) gets its own format: a schema, a parquet index, and
  ArrayRecord shards of msgpack records, one per episode. It reads through Grain. Text stays on
  Levanter's caches, and Levanter stays the training library. Every field of a dataset shares one
  step clock. How a field is stored is a per-field encoding (`raw` or `video` to start), not a
  property of the format.
- **Why:** The owner, 2026-10-11, has not chosen datasets and wants a foundation that does not fix
  the storage choice. Levanter's data stack assumes items that cost nothing to decode; Grain keeps
  its deterministic, resumable data order and adds decoding in worker processes. The owner's
  `open-dreamer` already uses Grain and ArrayRecord.
- **Rulings while planning:** A field's encoding is a class (`RawEncoding`, `VideoEncoding`), not an
  enum member, following AGENTS.md's "separate classes over flags"; the spec's "enum plus a pair"
  becomes "a class plus a branch". A reader delivers frames at the schema's size and step rate
  (`load_video` does this for files), and the build checks and encodes them. `episodes(...)` takes
  `num_shards`. Source units are opaque strings the reader resolves. The example readers download
  their data; VPT's subset is the first 10 episodes of the `all_6xx_Jun_29` index.
- **Rulings while building:** `VideoEncoding` rejects a codec that is not a video codec when the
  schema is built; PyAV otherwise fails inside `add_stream` with an unrelated error. Grain 0.2.18
  reads one of its absl flags when it starts worker processes and fails in any program that is not
  an absl app, so `batch_iterator` marks absl's flags parsed first; poorin declares `absl-py`.
  `tools/roots.toml` keeps `poorin.episodes.*` and the examples as workbench roots. The LIBERO
  reader reads each camera before flipping it, because h5py slices only with positive steps.
  `throughput.py` measures 500 batches after 20 warmup batches: the plan's 50 after 5 timed the
  workers' full prefetch buffers draining (4 workers read 10,647 clips/s). ty may leave ArrayRecord's
  compiled module unresolved, as it does jaxlib's.
- **Acceptance (2026-10-11):** Something-Something v2, 500 clips: 22,299 steps, 29 MB, built in 5 s.
  LIBERO-Spatial: 500 demonstrations, 62,250 steps, 125 MB, 122 s with the 6.2 GB download.
  Minecraft VPT, 10 episodes: 60,000 steps, 71 MB, 24 s with the download. Throughput on SSv2 at
  128 px, 16-step clips: 760, 1,470, 1,632, 2,219, and 2,838 clips/s at 1, 4, 8, 16, and 24 workers,
  against 1,000. The LIBERO smoke run trained on the GPU and resumed at step 100 with 100 identical
  batches. At exit, Python's resource tracker reports one shared-memory object left by Grain's
  worker prefetch.
- **Final review (2026-10-11):** The build and the reader now refuse an object-store path: ArrayRecord
  opens local paths only and wrote `gs://b/x` to `./gs:/b/x`. Episode validation names every field's
  length when they differ and refuses a value that is not a per-step array; a schema needs a field;
  `ClipSpec` refuses a length, stride, or skip below 1; `clip_dataset` refuses a clip that fits in no
  episode. The acceptance run measured only SSv2's short clips. On the VPT example's 6,000-step
  episodes (8.8 MB records) a clip costs 8.4 ms in one process, 5.7 ms of it reading the whole
  record, and throughput stays at 298, 301, and 360 clips/s at 1, 8, and 24 workers; uncompressed
  shards reach 333 to 426, and turning off the reader's readahead changes nothing. Why workers do not
  scale on long episodes is open, and keyframe-aligned records for long episodes would change the
  format; both are recorded under Deferred for the owner.
- **Alternatives:** Extend Levanter's cache and loader with field encodings; adopt LeRobot v3 as the
  on-disk format; a separate rate per field; CoinRun as an example (Procgen has no wheels above
  Python 3.10).
- **Cost if wrong:** Two data stacks persist. If Grain's worker processes cannot meet 1,000 clips
  per second, the read path changes encoding or moves decoding to the GPU.

---

### D111. Python 3.13

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** Poorin requires Python 3.13 (3.13.16 here). Every `requires-python` is `>=3.13`; the
  `<3.14` caps Marin set on poorin, fray, finelog, zephyr, and rigging are gone, and so is a stray
  `requires-python` under poorin's `[build-system]`. Ruff and ty target 3.13. Dupekit's Rust wheel
  keeps `abi3-py312`, which loads on 3.13.
- **Why:** The owner asked for 3.13 or 3.14 on 2026-10-11, and the plan moved to 3.14. On 3.14,
  `argparse.add_argument` rejects a `type=` that is not callable, and union types such as
  `int | None` are no longer callable, so draccus 0.11.6, the latest release, fails to build a parser
  for any config with an optional field (marin-community/draccus#69, open; the fix in #70 closed
  unmerged). That took out `levanter.config.main` and 49 Levanter tests. The owner chose 3.13 over a
  patched draccus fork, 2026-10-11. Moving before sub-project 3 locks its new dependencies once.
- **Versions that moved:** none. uv went from 0.8.17 to 0.13.0 (the owner approved it while trying
  3.14; 0.8.17 offered only 3.14.0rc2), and the lock's format revision went from 3 to 5.
- **Rulings while building:** `itertools.batched` in `verify_fuzzy_dups` passes `strict=False`,
  which ruff requires from 3.13 and which keeps a short final batch. Ruff rewrote nine
  `Generator[X, None, None]` annotations to `Generator[X]`. A fingerprint names a class value by its
  module and qualified name, and 3.13 moved `pathlib.Path` to `pathlib._local`, so a config holding
  the `Path` class itself (not a path, which encodes as a string) fingerprints differently than on
  3.12; `test_fingerprint` now checks the name with `datetime.timedelta`, whose module is stable.
- **Found:** `lib/levanter/tests/grug/test_fa4_cute_attention.py` fails 21 of 85 tests on the GPU, the
  same 21 on 3.12 before this change: the CuTe DSL reports `requires a code object with 4 free vars,
  not 0`, and jaxtyping rejects the launcher's arguments. Grug uses FlashAttention 4 by default on a
  GPU, and R2 trains with it in the full check, so the kernels work outside the tests. The likely
  cause is the tests' jaxtyping import hook on `levanter.grug`, which wraps the launcher the CuTe DSL
  then rejects. Recorded under Deferred.
- **Alternatives:** Python 3.14 with draccus pinned to a patched fork; staying on 3.12.
- **Cost if wrong:** Moving to 3.14 later is a second relock and environment rebuild, once draccus
  ships a fix. 3.14 also turns on Ruff's `UP037`, which unquotes forward references that jaxtyping's
  import hook then fails to resolve; that move has to keep `UP037` off.

---

### D112. Adding or deleting a module selects the architecture test

- **When:** 2026-10-11. **Status:** Adopted.
- **Decision:** `tools.affected_tests` keeps a table of test files that compare the list of
  workspace modules with a document, importing neither. Its one entry maps
  `tests/tools/test_architecture.py` to `docs/poorin/architecture.md`. A change that adds or deletes
  a module under a workspace `src/` tree, `experiments/`, or `tools/` (a rename does both), or edits
  the document, selects the test. `tools.changes.added_files` gives the added paths from Git: new
  tracked files, the new side of each rename, and untracked files. A changed source path that no
  longer exists counts as deleted. Editing a module that already exists does not select the test.
  This reverses D83's choice to run the test only with `--all`.
- **Why:** The episode data layer's commits `33905fdfe` and `a91069a06` (D110) added modules
  under `lib/poorin/src/poorin/episodes/` without rows. `tools.prune_check` runs
  `affected_tests.main([])`, which selected nothing that reads the module list, so the check
  passed while the test failed when run directly. D109's `experiments.datasets.fineweb_edu` slipped
  through the same way.
- **Alternatives:** Select the test whenever any module changes, which needs no Git status but
  starts the root suite for every source edit.
- **Cost if wrong:** One more Git call per selection. A caller of `select` that passes no `added`
  paths misses additions but still catches deletions.

---

### D113. Episode model templates: a WAM, a world model, and a VLA on one training loop

- **When:** 2026-10-11. **Status:** Adopted. Spec: `specs/2026-10-11-episode-model-templates-design.md`;
  plan: `plans/2026-10-11-episode-model-templates.md`.
- **Decision:** Sub-project 4 starts with three thin templates under `experiments/episode_models/`:
  `wam/` denoises future frames and actions together, `world_model/` and `vla/` are its model with
  the action stream or the future frames removed. All three use flow matching (rectified-flow
  path, logit-normal times, one flow time per stream), train from scratch on 128 px frames as
  16x16 patches, and share one training loop, data path, objective module, and FLOP count. A
  family owns its `model.py` and `launch.py`. Pretrained components are ported into grug-style
  modules and loaded from safetensors rather than run through a Flax bridge; this amends D13, whose
  bridge is not built. The held-out loss is the mean over a fixed grid of eight flow times with a
  fixed noise key, on an episode-level split by a stable hash of the episode id.
- **Why:** The owner, 2026-10-11, wants all three families thin before any goes deep, so the
  contract sub-project 5 sweeps (size knobs, FLOPs, one held-out loss) is settled first, and asked
  for templates that are modular, reproducible, and flexible. Flow matching is what Dreamer 4, π0,
  and UWM use, and one objective shares its sampler and evaluation. A ported module scales with the
  size knob and keeps parameter names the owner chooses for sub-project 8; a bridged Flax module is
  a fixed block that tracks the NNX API.
- **Rulings while designing:** The templates live outside `experiments/grug/`, whose contract tests
  require a language model in every variant directory. The training loop is shared rather than
  copied per family, because the owner asked for modular templates; the model file stays the
  copyable unit. The trunk copies grug base's block as it is, with a ReLU MLP four times the hidden
  width; the design conversation called it SwiGLU, which grug base does not use. Grug's
  `AttentionMask` cannot express a block-causal mask, so the templates pass a dense mask and run
  reference attention. Action percentiles are an artifact step versioned with its dataset.
- **Alternatives:** Copy grug base's training loop into each family; one template with a family
  setting; per-family objectives (MSE frames, L1 actions); discrete tokens for every family; a
  pretrained SigLIP encoder for the VLA through the Flax bridge.
- **Cost if wrong:** A family that needs its own loop forks `train.py` into its directory. The
  dense mask costs memory quadratic in sequence length once clips grow.

---

## Deferred

Known gaps left open on purpose. Each names where it gets resolved.

| Item | From | Resolved by |
|---|---|---|
| `.github/dependabot.yml` and `.github/actions/` survived P0 | P0 review | D41, P1 Task 10 |
| `prune_check` passed no marker expression, so a broad selection ran Iris's suite | P0 review | D37, P1 Task 1 |
| `tools.*` was not a root | P0 review | D40, P1 Task 1 |
| `experiments/grug/README.md` documents `GRUG_RUN_ID`, `FERRY_DATE`, and backward flow every 50 steps | P0 review | P1 Task 14 |
| `test_device_flops` asserts only that FLOPs are positive | P0 review | D89: asserts the RTX 5090's dense entries |
| The cross-attention case in `test_attention.py` is non-causal, so it never exercises the `q_len != k_len` guard in `_fa4_supports` | P0 review | D89: the case is causal with unequal lengths |
| `test_append_metrics_row_rejects_a_row_with_missing_columns` is a guard test under `TESTING.md` | P0 review | D89: deleted |
| About 125 MB of untracked `logs/` and `wandb/` entries from P0's check runs sit in the checkout | P0 | Deleted on request, 2026-10-07, spike 0's runs included |
| Set up `projects/` as a git-ignored, nested uv workspace member for `looped-diffusion-lm`, and scan `projects/*` for prune roots | D52 | Next step (D84), once `looped-diffusion-lm` has code |
| Design the SkyRL integration: a Hugging Face mapping for grug, launching the owner's SkyRL fork from the executor, and moving checkpoints both ways | D68 | A later sub-project |
| `tools.lint --all-files` reports two files under `infra/`: a TypeScript config with comments, and the old linter's own conflict-marker strings | P1 Task 2 | P1 Task 7 |
| `--ignore` skips a file found in a directory but not one passed by name, so a change that selects `test_tensorboard.py` through its imports still collects it, and it fails on `tensorboardX` | D56 | D61 |
| ty's 231 warnings | D54 | Pass 2: all cleared, ty reports none |
| Dependencies only deleted code used, such as `mcp` in `marin-core` | P3 | P8 |
| `prune_check` starts a local Finelog server and confirms R2's metric rows (spec invariant item 5); R2 needs a way to receive the URL | D74 | D77 |
| `experiments/rl_data_reviews/publish_review.py` imports the deleted `marina` package; it is unreachable and untested | P1 Task 7 | P2 |
| Links to deleted skills remain in `experiments/grug/moe/agent.md`, `experiments/grug/fast_track/agent.md`, and `experiments/post_training/cat_count_canary/README.md` | P1 review | P2, P7, D75 |
| ty's three noisy warning rules could be set per rule, so other warning rules would fail; `tools.lint` prints ty's warnings only when ty fails | P1 review | D89: ty fails on warnings; the one left, an optional base class, is suppressed with its reason |
| 26 code comments name pyrefly as the reason for a workaround | P1 review | Pass 2: workarounds ty does not need removed, the rest name ty |
| ty does not check `lib/dupekit/src`, which D47 keeps | P1 review | D89: ty checks it, and `lib/finelog/src` except protoc's generated stubs |
| A deletion-only change skips ty in `tools.lint`; `prune_check` runs it regardless | P1 review | D89: any changed Python path runs ty, deletions included |
| `tools.lint PATH` skips a path that does not exist without saying so | P1 review | D89: reported as a problem |
| The contributing guide's PR checklist still says to verify GitHub CI, and `tools.affected_tests` has no mode for a committed branch | P1 review | D89: the guide names the local checks; `tools.affected_tests --base main` covers a branch |
| Levanter's and Haliax's `AGENTS.md` and the `profile-training` skill show syncing `uv run` commands | P1 review | D89: `uv run --no-sync` there and in every Poorin doc and skill; Levanter's upstream install docs keep theirs |
| `test_ruff_checks_marin_with_its_own_configuration_as_marin_did` only mirrors the `RUFF_CONFIGS` table | P1 review | D89: deleted |
| `config/local.yaml`'s comment says outputs fall back to `/tmp/marin`; on a GCE VM, rigging falls back to `gs://marin-{region}` | P1 review | The comment says `/tmp/poorin` since D87; the GCE fallback waits for sub-project 2 |
| The ruff reformat's commit message says ruff checks each file's syntax tree; ruff has no such check, and a separate AST comparison confirmed the reformat | P1 review | Recorded here |
| Code that only made sense under Iris survives the comment cleanup (D76): Zephyr's `MAX_IRIS_WORKER_REPLICAS` and the coordinator-owned worker group, Fray fields `LocalClient` ignores (`target_cluster`, `processes_per_task`, `setup_scripts`, `sync_packages`, actor retries), grug dispatch's forwarded environment, the profiler's `iris.oa.dev` xprof URL, `MARIN_PROVENANCE`, `LogClient.connect(resolver=...)`, and `EvalRecord`, which nothing writes | Cleanup | D89: the replica cap, the `iris.oa.dev` default, `POORIN_PROVENANCE` (renamed by D87), and the resolver are deleted; Fray's remote-only fields and grug's forwarded environment stay for sub-project 2, and `EvalRecord` with the kept evaluation workflow |
| Haliax's docs fail `mkdocs build --strict` on 144 warnings, all present at `upstream-baseline`: unresolved cross-references to JAX, NumPy, and Equinox objects, and griffe docstring warnings | Cleanup | D102: Haliax and its docs left the tree |
| P1's timing columns in `prune-metrics.csv` are confounded by host load (load average about 250 from processes outside the checkout) and are not comparable with P0's | P1 | A rerun on a quiet machine |
| Nothing serves Zephyr's dashboard: the coordinator builds its Starlette app, but no server has mounted it since Iris was deleted. The owner wants it reachable over Tailscale (2026-10-09) | D90 | D100: the coordinator serves it on localhost; `tailscale serve` exposes it |
| Levanter's TensorBoard tracker is still in the tree, though D38 said P7 would delete it; its test needs `tensorboardX`, which the GPU environment does not install, so `tools.affected_tests` skips it | Backlog | D93: deleted |
| A run writes its materialized config only when it succeeds, so a failed run leaves none; it should write it when the run starts | D92 | Open |
| Grug's log file goes to `logs/` under the working directory, not into the run's output directory | D92 | Open |
| Grug's training data passes no `revision` to `pretokenized`, so its Hugging Face download follows the repository's default branch instead of a pinned revision | D92 | D104: pinned |
| `--set` has no named presets, so a bundle of overrides is retyped or kept in a shell alias | D92 | Open |
| Adaptive search (Optuna) is not supported; a study would be a Python loop over `resolve`, written when one needs it | D92 | Open |
| Fray has no SLURM backend | D92 | Open |
| `--set` is not forwarded to SkyRL runs as Hydra overrides (D68) | D92 | Open |
| `test_fa4_cute_attention.py` fails 21 tests on the GPU on both 3.12 and 3.13: the CuTe DSL reports a free-variable mismatch in the segmented forward launcher, likely because the tests' jaxtyping import hook wraps it; R2 trains with FlashAttention 4 and passes | D111 | Open |
| Python 3.14 waits on draccus parsing union-typed fields on 3.14 (marin-community/draccus#69) | D111 | Open |
| Long episodes read at about 300 to 430 clips/s whatever the worker count (VPT, 6,000 steps): profile the read path; keyframe-aligned records per long episode would change the format | D110 | Open |
