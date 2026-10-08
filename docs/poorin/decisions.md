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

- **When:** 2026-10-05, P0. **Status:** Adopted.
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

- **When:** 2026-10-05, P1. **Status:** Adopted. The Iris and Verifyit deselects took effect only
  with D56.
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

- **When:** 2026-10-07, P8. **Status:** Adopted.
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
- **Cost if wrong:** Runs that never convert checkpoints, grug's R2 among them, pay about 0.9 s at
  every launch.

### D83. Pass 2's ranked loop ends on Keep verdicts; architecture.md is checked by a test

- **When:** 2026-10-07, Pass 2. **Status:** Adopted.
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

---

## Deferred

Known gaps left open on purpose. Each names where it gets resolved.

| Item | From | Resolved by |
|---|---|---|
| `.github/dependabot.yml` and `.github/actions/` survived P0 | P0 review | D41, P1 Task 10 |
| `prune_check` passed no marker expression, so a broad selection ran Iris's suite | P0 review | D37, P1 Task 1 |
| `tools.*` was not a root | P0 review | D40, P1 Task 1 |
| `experiments/grug/README.md` documents `GRUG_RUN_ID`, `FERRY_DATE`, and backward flow every 50 steps | P0 review | P1 Task 14 |
| `test_device_flops` asserts only that FLOPs are positive | P0 review | Open |
| The cross-attention case in `test_attention.py` is non-causal, so it never exercises the `q_len != k_len` guard in `_fa4_supports` | P0 review | Open |
| `test_append_metrics_row_rejects_a_row_with_missing_columns` is a guard test under `TESTING.md` | P0 review | Open |
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
| ty's three noisy warning rules could be set per rule, so other warning rules would fail; `tools.lint` prints ty's warnings only when ty fails | P1 review | Pass 2 |
| 26 code comments name pyrefly as the reason for a workaround | P1 review | Pass 2: workarounds ty does not need removed, the rest name ty |
| ty does not check `lib/dupekit/src`, which D47 keeps | P1 review | Open |
| A deletion-only change skips ty in `tools.lint`; `prune_check` runs it regardless | P1 review | Open |
| `tools.lint PATH` skips a path that does not exist without saying so | P1 review | Open |
| The contributing guide's PR checklist still says to verify GitHub CI, and `tools.affected_tests` has no mode for a committed branch | P1 review | Open |
| Levanter's and Haliax's `AGENTS.md` and the `profile-training` skill show syncing `uv run` commands | P1 review | Open |
| `test_ruff_checks_marin_with_its_own_configuration_as_marin_did` only mirrors the `RUFF_CONFIGS` table | P1 review | Open |
| `config/local.yaml`'s comment says outputs fall back to `/tmp/marin`; on a GCE VM, rigging falls back to `gs://marin-{region}` | P1 review | Sub-project 2 |
| The ruff reformat's commit message says ruff checks each file's syntax tree; ruff has no such check, and a separate AST comparison confirmed the reformat | P1 review | Recorded here |
| Code that only made sense under Iris survives the comment cleanup (D76): Zephyr's `MAX_IRIS_WORKER_REPLICAS` and the coordinator-owned worker group, Fray fields `LocalClient` ignores (`target_cluster`, `processes_per_task`, `setup_scripts`, `sync_packages`, actor retries), grug dispatch's forwarded environment, the profiler's `iris.oa.dev` xprof URL, `MARIN_PROVENANCE`, `LogClient.connect(resolver=...)`, and `EvalRecord`, which nothing writes | Cleanup | Pass 2 |
| Haliax's docs fail `mkdocs build --strict` on 144 warnings, all present at `upstream-baseline`: unresolved cross-references to JAX, NumPy, and Equinox objects, and griffe docstring warnings | Cleanup | Open |
| P1's timing columns in `prune-metrics.csv` are confounded by host load (load average about 250 from processes outside the checkout) and are not comparable with P0's | P1 | A rerun on a quiet machine |
