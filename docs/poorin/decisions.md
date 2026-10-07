# Decision log

Every decision that shaped Poorin, in the order it was made: what was decided, why, what else
was considered, and what it costs if it turns out wrong. The spec
(`specs/2026-10-04-core-prune-design.md`) and the phase plans (`plans/`) hold the full designs;
commits hold the diffs. This log is the place to read why the tree looks the way it does.

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

Plan: `plans/2026-10-05-p1-fleet-operations.md`. Executing on branch `prune/p1-fleet-ops` since
2026-10-06; rulings made while executing it are D53 onward.

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

- **When:** 2026-10-05, P1. **Status:** Adopted. Marin's code keeps its explicit ruff configuration (see
  `PythonStyle.ruff_config`).
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
| About 125 MB of untracked `logs/` and `wandb/` entries from P0's check runs sit in the checkout | P0 | Open; delete on request |
| Set up `projects/` as a git-ignored, nested uv workspace member for `looped-diffusion-lm`, and scan `projects/*` for prune roots | D52 | The phase after P1 |
| `tools.lint --all-files` reports two files under `infra/`: a TypeScript config with comments, and the old linter's own conflict-marker strings | P1 Task 2 | P1 Task 7 |
| `--ignore` skips a file found in a directory but not one passed by name, so a change that selects `test_tensorboard.py` through its imports still collects it, and it fails on `tensorboardX` | D56 | D61 |
| ty's 231 warnings | D54 | Pass 2 |
