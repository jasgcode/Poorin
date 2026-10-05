# Sub-project 1: Core prune

Status: approved design, 2026-10-04. Implementation plan: not written yet.

## TL;DR

Poorin is a hard fork of [Marin](https://github.com/marin-community/marin) for one researcher
training LLMs, video world models, and robot policy models on a local RTX 5090, with occasional
cloud runs. This sub-project
shrinks the repository to the code that a fixed set of entry points can reach, then cuts import
edges that make unrelated code load, until every surviving module has a stated reason to exist.

The prune runs in two passes:

1. Pass 1 deletes everything outside the static import closure of the roots defined below, plus
   Iris, Marin's in-repo SkyRL integration, LLM serving and evaluation, and Marin's fleet
   operations. First-party library and experiment code (excluding tests) goes from 386k lines to
   at most 135k.
2. Pass 2 assigns a verdict to each remaining dependency, in ranked order, one commit per
   change, until only deliberate dependencies remain.

Every commit keeps two local training runs working on the 5090 and appends a row of measurements
to `docs/poorin/prune-metrics.csv`.

A follow-on sub-project, 1b, decides whether Haliax stays and consolidates the surviving
packages. It gets its own spec after Pass 2 ends.

## Context

### Project goals

- Research across four model families: LLMs, video world models, vision-language-action
  models (VLAs), and world action models (WAMs), which predict future observations and actions
  together. The work is architecture search and scaling laws, starting with video and offline
  robot datasets. Later work adds online simulation data for EXPO-FT, an RL method that
  fine-tunes a pretrained VLA with a small edit policy and a Q-function
  ([arXiv 2605.25477](https://arxiv.org/abs/2605.25477)).
- A light LLM post-training capability (SFT, DPO, LoRA, sampling) for representation-learning
  experiments on models trained from scratch.
- One evaluation framework for all four families. Marin's framework runs external benchmark
  harnesses against a model served behind an OpenAI-compatible URL and records results with
  uncertainty intervals. Its records and statistics already apply to any family; its serving
  contract and rollout schema are shaped for LLMs. Sub-project 7 generalizes them.
- A repository the owner understands end to end and documents in a technical blog. Each change
  must be small enough to explain.
- JAX and Equinox are the modeling framework. Grug's style (Equinox modules, JAX's explicit
  sharding) is the template for new models. Pretrained VLA and vision models (openpi,
  big_vision) are published in Flax; Poorin uses them through a bridge that stores an NNX
  module's graph as a static Equinox field and its weights as an ordinary pytree field.
- Finished architectures are later ported to PyTorch or Flax for release (sub-project 8).
- Durable, queryable logging next to W&B. Every run writes the same metrics to W&B and to
  Finelog, a log and metrics store that answers SQL across all runs.
- Upstream Marin serves as a reference; Poorin does not merge from it.

### Sub-project order

| # | Sub-project | Depends on |
|---|---|---|
| 0 | RTX 5090 spike (done) | |
| 1 | Core prune (this spec) | 0 |
| 1b | Consolidation: decide whether Haliax stays; if not, port R1's model path, the post-training kit, and the 12 Haliax-aware optimizer modules to plain Equinox; merge small packages | 1 |
| 2 | Compute dispatch: local, Modal, GCP TPU | 1 |
| 3 | Data layer: text, video, and trajectory datasets | 1 |
| 4 | Model templates in Equinox, grug style: LLM, video world model, VLA, WAM; the Flax bridge for pretrained components | 1, 3, 7a |
| 5 | Scaling-law harness across model families | 2, 4, 7 |
| 6 | Online simulation loop for EXPO-FT | 1, 4 |
| 7a | Evaluation contracts: one serving interface per model kind (text, policy, world model, WAM), a rollout schema of episodes and typed steps, the evaluation record | 1 |
| 7b | Evaluation mechanisms: a local orchestrator for the kept LLM harnesses, then simulator suites for policies and WAMs and video-metric suites for world models. Each family's mechanisms land with its template in 4. | 7a, 4 |
| 8 | Release ports: export finished architectures and weights to PyTorch (Hugging Face) or Flax. Release weights use safetensors with a JSON config; training checkpoints stay in Levanter's TensorStore format (Zarr v3 in OCDBT), which also holds optimizer state. `levanter.compat.hf_checkpoints`, `haliax.state_dict.to_torch_compatible_state_dict`, and `levanter.main.export_lm_to_hf` are the existing pattern. Sub-project 4 fixes parameter names and weight layouts so they map directly onto the ported modules. | 4 |

Design order after this sub-project: 7a first, so every model template in 4 is built against an
evaluation interface and sub-project 5 fits scaling curves on one record format. 1b, 2, and 3
can proceed in parallel with 7a. 4 needs 3 and 7a.

### Libraries in the workspace

| Library | Role in Marin | Fate |
|---|---|---|
| `haliax` | Named-tensor layer over JAX, built on Equinox | Kept: R1, the post-training kit, and 12 optimizer modules use it; 1b decides whether it stays |
| `levanter` | Trainer, models, data loading, checkpointing, optimizers, kernels | Kept and pruned |
| `marin` | Experiment executor, training bridge, tokenization, scaling-law analysis, evaluation framework, plus RL and inference tooling | Kept and pruned |
| `fray` | Job and resource API with a local backend and an Iris backend | Kept; Iris backend deleted |
| `zephyr` | Lazy dataset pipelines; builds Levanter's token caches | Kept |
| `rigging` | Shared utilities: filesystem and bucket configuration, timing, telemetry | Kept and pruned |
| `finestore` | Transactional Parquet store on local disk or object storage | Kept; the evaluation framework stores samples and rollouts in it |
| `iris` | Multi-tenant cluster scheduler for TPU and GPU fleets | Deleted in P5 |
| `finelog` | Log and metrics store: typed tables, SQL queries, process logs, Parquet storage on disk or object storage, forwarding between servers. Python client in the repository; server written in Rust. | Trimmed in P6 to its Python client; the server comes from the published `marin-finelog-server` wheel |
| `ducky` | DuckDB SQL service for object-store Parquet | Deleted in P4 |
| `dupekit` | Rust text-deduplication kernels | Deleted in P4 |
| `verifyit`, `shellbox`, `taskcompendium` | Grading, sandboxed shell environments, and task presentation for RL | Deleted in P2 |

### Spike 0 results that shape this design

Measured on an RTX 5090 (32 GB, sm_120, driver 580.178) at upstream commit `ab0da00df`. The
spike note is currently outside the repository; P0 adds it as
`docs/poorin/spike-0-rtx5090.md`.

| Finding | Evidence |
|---|---|
| JAX 0.11.1 runs on sm_120 | bf16 matmul at 222 TFLOP/s |
| cuDNN attention fails | `No valid execution plans built`. NVRTC 13 cannot load `libnvrtc-builtins.so.13.0`, which lives in `nvidia/cu13/lib` and is not on the loader path. Adding that directory to `LD_LIBRARY_PATH` fixes it. |
| The JAX compilation cache is never enabled for local runs | `marin.training.training` writes `JAX_COMPILATION_CACHE_DIR` only into the Fray job environment, and `LocalClient` runs jobs as threads that ignore it. With the variable set in the shell, grug 130M warm start fell from 7m08s to 1m05s. |
| `--version dev` resumes the previous run | The output path is `{name}/dev` on every invocation, so Levanter's checkpointer resumes from the last checkpoint. The second run logged `Resuming training from step 100`. |
| Grug uses slow attention on GPU | Grug's default on non-TPU backends is `reference`: 33 TFLOP/s forward and backward. `gpu_fa4_cute` reaches 140 and cuDNN 156 on the same shapes. |
| Grug compiles three step variants | `compute_watch` (every 10 steps) and `compute_backward_flow` (every 50 steps) are static arguments to the jitted step. Each variant took about 90 s to compile. |
| PyTorch loads on every run | `marin.experiment.train` → `levanter.adaptor` → `levanter.compat.hf_checkpoints` → `transformers` loads `torch` (1,006 modules) and `torchvision`. A run imports 7,080 modules in about 10 s. |
| No FLOPs entry for the 5090 | `fray.device_flops` logs `Unknown device type`, so MFU is missing. |
| Same seed, different loss | Final loss 6.621 and 6.633 on two identical grug runs. |

Grug 130M baseline: batch 16 × 1024 tokens, 146 ms per step, 112k tokens/s, about 27% MFU
against the measured 222 TFLOP/s peak.

## Decisions

| Decision | Reason |
|---|---|
| Hard fork. Delete freely. | Merging upstream is not a goal. Tag `upstream-baseline` at `ab0da00df` so the original tree stays one checkout away. |
| Delete Iris completely. Keep `iris/jax/compile_cache.py` and `iris/env_resources.py`. | Iris has 84k lines of Python source (plus 67k of tests), 3k of Rust, and 16k of TypeScript. A single researcher needs machine provisioning, multi-host JAX setup on multi-host TPU slices, and preemption retry. Sub-project 2 provides those. |
| Delete Marin's in-repo SkyRL integration. | The owner maintains a customized SkyRL in a separate repository and will reintegrate it against the new dispatch from sub-project 2. |
| Keep a light LLM post-training kit. | The owner plans representation-learning experiments on models trained from scratch. |
| Keep Marin's evaluation framework. Delete only its fleet orchestration and published-cohort policy. | The benchmark harness integrations, per-run records, normalized rollout table, and statistics need only a served endpoint. About 2k of its 9.2k lines need Iris or the serving fleet. |
| Delete 14 of Levanter's LLM architectures. Keep Llama, Qwen, and GPT-2. | Grug is the modeling template. The deleted architectures remain at `upstream-baseline`. |
| Delete `infra/` and `scripts/` completely. | The owner uses none of these tools. Lint moves to a short root-level setup in P1. |
| Keep Finelog's Python client. Depend on the published server wheel. | Finelog does not depend on Iris; Iris hosted its server. A local server starts from a directory (`EmbeddedServer(log_dir=...)`): a probe wrote 300 typed rows and queried them back with SQL. The Rust server is 69k lines and is treated as an external dependency. |
| Use Equinox for models. Reach pretrained Flax models through a bridge. | The repository's training stack is Equinox: Levanter's trainer, the optimizer collection, grug, fast-track, LoRA, the post-training kit, evaluation, and inference. Optax, Orbax, jaxtyping, and JAX sharding operate on Equinox pytrees. A probe trained a Flax NNX module's weights inside an Equinox model with `eqx.filter_jit` and Optax (loss 1.20 to 0.00), then merged them back into a live NNX module. The baseline lock pins Flax 0.12.4, which fails to import under JAX 0.11 (`jax.core.Effect` was removed); Flax 0.12.10 works. |
| Run the prune as reachability-driven Pass 1 and Pass 2, then consolidate in 1b. | Pass 1 removes most lines mechanically. Pass 2 forces a reading of each surviving dependency. Consolidation is easier once the survivors are known. |

## Roots

The roots define what survives. Code reachable from a root through static imports is kept. All
other code is deleted.

### Runnable roots

Runnable roots train on the 5090 before every commit.

| Root | Command | Exercises |
|---|---|---|
| R1 | `experiments/tutorials/train_tiny_model.py --device rtx5090x1 --dataset wikitext --version dev --run` | Levanter `train_lm`, Haliax models, tokenization through Zephyr, the executor, checkpointing |
| R2 | `experiments/grug/base/launch.py --version dev --run`, made local-first: the `marin-community/fineweb-edu-pretokenized-10M` cache from Hugging Face, one `rtx5090x1`, about 50 steps | The grug model and train loop, `levanter.grug.attention`, Fray dispatch |

P0 changes `experiments/grug/base/launch.py` itself to be local-first and removes its Nemotron
mixture and `v5p-8` defaults. No second launcher is added.

### Kept roots

Kept roots must import and pass type checking. They are not executed.

| Root | Why kept |
|---|---|
| `experiments.grug.fast_track` (`launch`, `train`, `model`, `heuristic`) | Dense and MoE scaling ladder with a compute-fitted LR, beta2, and epsilon heuristic. Its data lives on a CoreWeave bucket the owner cannot read, so it cannot run until sub-project 3 or 5. |
| `experiments.references.reference_scaling_suite` | The isoFLOP workflow: fit loss curves per compute budget, fit D\* ∝ C^α, train compute-optimal runs. |
| `experiments.tutorials.train_tiny_sweep` | The executor's sweep pattern. |
| `marin.scaling_laws`, `marin.experiment.cli` | IsoFLOP analysis and plots; the shared experiment CLI. |
| `levanter.main.train_dpo`, `levanter.main.lora_lm`, `levanter.main.lora_dpo`, `levanter.main.sample_lm` | The light post-training kit. A later sub-project makes it runnable with a small SFT → DPO → sample test on `llama_nano` (`experiments/llama.py`: 2 layers, hidden size 32). |
| `levanter.inference.openai`, `levanter.eval_harness` | An OpenAI-compatible server for Levanter models, so the benchmark harnesses can evaluate checkpoints that vLLM cannot serve; and lm-evaluation-harness multiple-choice tasks scored from JAX log-probabilities during training, with no server. |
| `marin.evaluation.evalchemy.*`, `marin.evaluation.harbor.*`, `marin.evaluation.lm_eval`, `marin.evaluation.lm_eval_samples` | Integrations with three external benchmark harnesses: lm-evaluation-harness, Evalchemy (benchmark collections built on it), and Harbor (agentic tasks in sandboxed environments). Each runs in an isolated, version-pinned `uvx` environment against a served endpoint. A benchmark that a harness already implements is added as a YAML file under `experiments/evaluation/configs/`, which is kept. |
| `marin.evaluation.records`, `rollouts`, `eval_measurements`, `eval_stats`, `metric_selection`, `evaluation_config`, `model_config`, `model_identity`, `eval_env` | One durable `record.json` per evaluation; the normalized `rollouts_v1` table; score intervals that account for ungraded items; intervals for the difference between two runs. |
| `marin.evaluation.perplexity_gap`, `marin.evaluation.trace_labeled_eval` | Per-dataset perplexity comparison between two models; loss on agent conversation traces. |
| Plugin registry members (next table) | `draccus.PluginRegistry` subclasses are found by name at config-parse time, so static imports do not reach them. |

| Registry | Discovery package | Members kept as roots |
|---|---|---|
| `OptimizerConfig` | `levanter.optim` | All modules (`levanter.optim.*`) |
| `LmDatasetFormatBase` | `levanter.data.text` | All modules (`levanter.data.text.*`): text, chat, and preference formats |
| `TrackerConfig` | `levanter.tracker` | `tracker`, `wandb`, `json_logger`, `json_file`, `telemetry` (writes to Finelog through `finelog_metrics`) |
| `LmConfig` | `levanter.models` | `llama` |

With these roots, the reachable Levanter models are `gpt2`, `linear`, `llama`, `lm_model`, `loss`,
and `qwen`. The owner chose to delete the other 14 model modules in P7: Mistral, Mixtral, Gemma,
OLMo, OLMo 3, Whisper and the ASR model, Hyena, GPT-2 Hyena, ModernBERT, Apertus, Snowball,
Qwen3 MoE, and Levanter MoE. They remain available at `upstream-baseline`.

### Deleted regardless of reachability

| Module | Edge to cut in kept code |
|---|---|
| `iris` | See P5 |
| `marin.rl`, all of `experiments/post_training` | None in roots |
| `marin.inference`, except `types.py` | `RunningModel` and `OpenAIEndpoint` move from `marin/inference/types.py` to `marin/evaluation/endpoint.py`. `evalchemy/runner.py`, `harbor/runner.py`, and `harbor/driver_config.py` stop using Iris's serving-session types (`RemoteInferenceSession`, `InferenceBackendState`) and take a `RunningModel`. `model_config.py` drops speculative-serving settings, and `resolve_tokenizer_revision` moves into it. Levanter's own inference engine (`levanter.inference`) is separate and stays. |
| `marin.evaluation` fleet orchestration: `runner.py`, `serving_config.py`, `hardware.py`, `inference_metrics.py`; published-cohort policy: `eval_policy.py`, `eval_policy_sources.py`; `marin.experiment.evaluation`; the fleet launcher in `experiments/evaluation/*.py` | Remove imports of these modules from `marin.experiment.train` and `experiments/tutorials/train_tiny_sweep.py`. The in-loop path through `levanter.eval_harness` and Levanter's validation-loss evaluation (`levanter.eval`) stay; scaling-law fits use the latter. Sub-project 7b writes a local orchestrator to replace `runner.py`. P3 removes only the Iris and fleet dependencies and redesigns nothing, so sub-project 7a starts from Marin's framework as it is. |
| `marin.mcp` | None in roots |
| `fray.iris_backend` | See P5 |

## Invariant

The owner runs `uv run python tools/prune_check.py --phase P<n> --step "<short label>"` by hand
before each commit. It fails if any check fails.

1. R1 and R2 train to completion on the 5090. With warm caches each takes about one minute.
2. Every kept root imports.
3. The default pytest selection for surviving code passes. Tests for deleted code are deleted in
   the same commit.
4. `uv run pyrefly check` passes, and the lint entry point passes: `./infra/pre-commit.py` until
   P1, the replacement after P1.
5. R2 logs to W&B in offline mode and to Finelog. `prune_check` starts a local Finelog server in
   a temporary directory for the duration of the check and confirms that R2's metric rows arrived.
   Until P5b, R2's Finelog tracker reads its run identity from Iris, so this check starts in P5b.
6. One row is appended to `docs/poorin/prune-metrics.csv` and committed with the change.

Metrics columns:

| Column | Meaning |
|---|---|
| `date`, `phase`, `step` | Identity of the row. `step` is the label passed on the command line. The commit hash is recoverable from `git log` of the CSV. |
| `first_party_lines` | Physical lines in `.py` files under `lib/*/src`, `experiments/`, and `tools/`. Test files are excluded: files under any `tests/` directory and files named `test_*.py`, `*_test.py`, or `conftest.py`. |
| `reachable_lines`, `reachable_modules` | Static closure of all roots, counted the same way |
| `r1_imported_modules`, `r1_import_time` | Modules loaded and total import time in R1, from `python -X importtime` |
| `r1_warm_wall_time`, `r2_warm_wall_time` | End-to-end time of each runnable root with warm caches |
| `venv_size` | Size of `.venv` in bytes |

### Tools added in P0

| Tool | Behavior |
|---|---|
| `tools/reachability.py` | Builds the first-party import graph. Prints the closure of the roots with per-package line counts. `--why MODULE` prints the shortest import chain from a root. `--rank` lists modules by dominated weight: the lines that become unreachable if that module alone is removed. Self-contained; it copies about 80 lines of graph logic (`workspace_modules`, `imported_names`, `resolve`) from `infra/ci/select_tests.py`, which P1 deletes. |
| `tools/prune_check.py` | Runs the invariant and appends the metrics row. |
| `tools/roots.toml` | The single list of roots, read by both tools. |

`tools/reachability.py` follows every import statement, including imports inside functions.
Lazily imported code therefore stays reachable and is never deleted by mistake. With `--eager`,
it follows only module-level imports, which approximates what loads at startup.

`tools/roots.toml` format:

```toml
[[runnable]]
name = "R1"
command = ["python", "-m", "experiments.tutorials.train_tiny_model", "--device", "rtx5090x1",
           "--dataset", "wikitext", "--version", "dev", "--run"]

[kept]
# A trailing ".*" includes every module under the package.
modules = ["marin.scaling_laws", "levanter.optim.*", "levanter.tracker.wandb"]
```

## Pass 1: deletion order

Phases run outside-in: a module is deleted only after nothing that survives imports it. Each
phase is one branch and one pull request against the fork's `main`. Each commit inside a phase
deletes one coherent unit.

The line counts in this table cover all Python a phase removes, including tests and code outside
`lib/` and `experiments/`, such as `infra/`. They therefore add up to more than the 386k
`first_party_lines` baseline. They come from a throwaway analysis during design; P0 recomputes
them with `tools/reachability.py`.

| Phase | Content | Python lines removed (approx.) |
|---|---|---|
| P0 Setup | Tag `upstream-baseline`. Delete `.github/workflows` (38 workflows; the fork is public and has Actions enabled). Add the tools and roots file. Add the `rtx5090x1` device to R1 and an RTX 5090 entry to `fray.device_flops`. Make R2 local-first. Apply the spike fixes assigned to P0 below. Add the spike note. Record the baseline metrics row. | adds ~0.5k |
| P1 Fleet operations | Delete `infra/`, `scripts/`, `docker/`, and `etc/`. Replace `infra/pre-commit.py` with a root-level lint entry point (see below). Replace `config/marin.yaml` and `config/coreweave.yaml` with `config/local.yaml`, which declares no region buckets, and change the default of `MARIN_CLUSTER` in `rigging.filesystem.cluster_config` from `marin` to `local`. `MARIN_PREFIX` keeps pointing outputs at local disk. Rewrite `AGENTS.md` and prune `.agents/skills/` (see below). | ~87k |
| P2 RL stack | `marin.rl`, `experiments/post_training`, `lib/verifyit`, `lib/shellbox`, `lib/taskcompendium` | ~47k |
| P3 LLM serving and evaluation orchestration | `marin.inference` (after moving `types.py`), the fleet orchestration and cohort-policy modules of `marin.evaluation`, `marin.experiment.evaluation`, `marin.mcp`, `eval_policy/`, the Python files of `experiments/evaluation` (its `configs/` stay); see "Deleted regardless of reachability" for the edges to cut | ~19k |
| P4 Data services | `lib/ducky`, `lib/dupekit` | ~4k |
| P5 Iris | Three steps, P5a to P5c; see the next section | ~151k (84k source, 67k tests) |
| P6 Orphans and Finelog trim | Code that only Iris kept reachable. Trim `lib/finelog` to its Python client (`client/`, `rpc/`, top-level modules, and their tests): delete `rust/` (69k lines of Rust), `deploy/`, `benchmarks/`, `dashboard/`, and its operations docs. Pin `marin-finelog-server==0.2.46.dev33324331711`, the version installed at baseline. | ~7k Python, 69k Rust |
| P7 Unreachable sweep | One commit per library: `experiments/`, `marin`, `levanter`, `zephyr`, `rigging`, `finestore`, with their tests and docs. Includes code no root reaches, such as the 14 deleted model modules. `haliax` and `fray` have almost no unreachable code left by this point. | ~70k |
| P8 Workspace | Root `pyproject.toml` dependencies, extras, and `uv` conflict declarations; `uv.lock`; upgrade Flax to 0.12.10 or later; decide whether PyTorch moves to its CPU wheel. Dependencies may be upgraded freely in any phase, each upgrade in its own commit. | 0 |

Each deletion commit removes the code, its tests, its docs, its workspace entry, and its entries
in tool configuration. The commit message states what the code did for Marin and why Poorin does
not need it.

### Lint replacement (P1)

The new entry point keeps these checks from `infra/pre-commit.py` with the same tool versions,
so no file is reformatted: ruff 0.14.3, black 25.9.0, pyrefly (`>=1.0.0,<1.1.0`, as pinned in
`lib/marin/pyproject.toml`), license headers, large files, Python syntax, merge-conflict
markers, TOML, YAML, and JSON validity, trailing whitespace, final newline, and notebooks. It
drops the Marin-specific checks for skill metadata and pre-commit invocations in Markdown, and
the LLM lint review in `infra/linter.py`.

## P5: Iris removal

### P5a: salvage

| Source | Destination | Changes |
|---|---|---|
| `iris/jax/compile_cache.py` | `levanter/compile_cache.py` | Called from `levanter.distributed` during process initialization, where Iris's JAX init used to be called. R1 and R2 both pass through it: both spike runs logged `levanter.distributed Not initializing jax.distributed`. Cache directory precedence: `TrainerConfig.jax_compilation_cache_dir`, then `JAX_COMPILATION_CACHE_DIR`, then `~/.cache/poorin/jax`. Drop the node-local scratch mount and the remote mirror of XLA's autotune cache. Delete the env-dict path in `marin.training.training`. |
| `iris/env_resources.py` | `zephyr/resources.py` | Keep cgroup v2, cgroup v1, and `/proc` detection. Drop the `IRIS_TASK_RESOURCES` branch. |

P5a also adds the NVRTC fix to the same initialization step. When JAX's backend is GPU, load
`nvidia/cu13/lib/libnvrtc-builtins.so.13.0` from the installed NVIDIA wheels with `RTLD_GLOBAL`
before the first cuDNN call. If the backend is GPU and the file is missing, raise an error that
names the expected path. On CPU and TPU backends, do nothing. Implementation must confirm that
cuDNN attention passes with the preload and without `LD_LIBRARY_PATH`.

### P5b: cut edges, one commit per library

| Library | Change |
|---|---|
| Fray | Delete `iris_backend.py` (820 lines). `current_client()` returns the explicitly set client, otherwise `LocalClient`. |
| Zephyr | Remove the Iris context checks in `context`, `coordinator`, `worker`, and `stats`. They look up Iris's log-server endpoint and otherwise use the local path. `shuffle` uses `zephyr/resources.py`. |
| Levanter | `distributed.py` keeps SLURM detection and JAX's own cluster detection, which configures `jax.distributed` on TPU VMs. It loses the Iris branch, which found the JAX coordinator through Iris's endpoint registry. Delete `megascale.py` (multi-slice TPU), `callbacks/_iris_status.py`, and `training_control.py` (an HTTP status page reached through Iris's proxy) along with its uses in `main/train_lm.py` and grug. `tracker/telemetry.py` and `tracker/finelog_metrics.py` stay; their run identity (job ID, node name, process index), which came from Iris, comes from Levanter's run ID and `jax.process_index()` instead. `TelemetryConfig` takes an explicit Finelog URL and raises if it cannot connect. Today `LevanterMetricsWriter.from_iris()` finds no server outside Iris, and `TelemetryConfig.init` catches the error and returns a tracker that silently discards metrics. |
| Marin executor | `step_status` and `step_runner` keep only the local worker ID. `sweep_coordination` keeps only the single-host path and loses the multi-host leader and follower actors. `marin.training.training` loses the env-dict cache code. Grug's dispatcher stops importing Iris's priority bands. |
| Experiments outside the roots | Any such file that imports Iris is deleted in this phase so its tests do not fail before P7. |

### P5c: delete the library

Delete `lib/iris` and its tests, its workspace membership, and `marin-iris` from the five
`pyproject.toml` files that still list it after P4: the root, `lib/fray` (optional `iris`
extra), `lib/zephyr`, `lib/levanter`, and `lib/marin`. Remove Iris references from docs and
`AGENTS.md`. Regenerate `uv.lock`.

### Deferred to sub-project 2

Fray's job API still carries Iris concepts: priority bands, regions and `ANY_REGION`, retry
budgets, and placement constraints. Sub-project 1 removes only the backend. Iris's coordinator
discovery for multi-process JAX (`iris/jax/init.py`) is gone. Single-host runs do not need it,
and JAX's own detection covers TPU VM slices. Multi-node GPU training would need a replacement;
the original remains at `upstream-baseline`.

## Pass 2: edge cuts

### Method

1. Run `tools/reachability.py --rank`.
2. Take the highest-ranked module without a verdict and assign one:
   - Delete: Poorin does not need the subsystem. Measured by `reachable_lines`.
   - Make lazy: Poorin needs the code on some runs. Move the import into the function that uses
     it, or behind a backend check. The code stays reachable, so `reachable_lines` does not
     change. Measured by `r1_imported_modules` and `r1_import_time`, and by
     `tools/reachability.py --eager`.
   - Keep: the dependency is deliberate. Record why in `docs/poorin/architecture.md`.
3. Apply the verdict, run the invariant, commit, and repeat.

Pass 2 ends when the 10 highest-ranked modules all have a Keep verdict and every surviving module
has an entry in `docs/poorin/architecture.md`. There is no line-count target.

### Initial targets

Measured on the current tree with all roots, after excluding the modules that Pass 1 deletes
regardless of reachability: 134,650 reachable lines in 439 modules.

| Dominated lines | Module | Reached through | Verdict |
|---|---|---|---|
| root | `levanter.eval_harness` | kept root; `marin.experiment.train` also imports it at module level for optional in-loop evals | Make lazy: import it only when an in-loop eval suite is configured |
| 3.5k | `levanter.grug._moe.ep_deepep`, `levanter.kernels.deepep` | grug MoE | Delete. Expert-parallel all-to-all has no use on one GPU. |
| 2.8k | `experiments.grug.moe.launch_datakit_moe_mix` | a helper that fast-track imports from another experiment | Move the helper into fast-track; cut the edge |
| 2.1k | `levanter.compat.hf_checkpoints` | `levanter.main.train_lm`; loads `transformers` and `torch` | Make lazy. Sub-project 4 needs it to import pretrained vision and VLA weights. |
| 1.9k | `levanter.main.train_dpo`, `levanter.adaptor` | `marin.training.training` imports every training mode at module level | Make lazy. Both stay reachable as kept roots; the cut stops pretraining runs from loading them. |
| 1.5k | `levanter.inference.engine`, `jit_scheduler` | the kept roots `sample_lm`, `levanter.inference.openai`, and `levanter.eval_harness` | Make lazy so training runs do not load the inference engine |
| 4.5k | Pallas variants in `levanter.kernels.pallas.fused_cross_entropy_loss`; `levanter.kernels.pallas.splash_attention` (0.2k) | Fused cross-entropy is grug's loss (`levanter.grug.loss`) and selects `batched_xla`, its pure-XLA implementation, on GPU. Splash attention is reached because `levanter.data.loader` imports the `AttentionMask` type from `levanter.layers.attention`, which imports the splash kernel at module level. | Make the TPU variants lazy behind a TPU backend check. Keep `batched_xla`. The TPU kernels are the TPU fast path for sub-project 2. |
| 1.2k | `levanter.analysis.backward_flow` | grug's train loop | Make lazy and opt-in |
| 1.8k | `levanter.tracker.telemetry` | kept root (tracker registry) | Keep. P5b removes its Iris dependency. |
| 7.6k | Zephyr (`zephyr.context`) | tokenization and cache building | Keep. Sub-project 3 decides whether Zephyr becomes the video and trajectory pipeline. |
| 5.6k | `levanter.grug.grug_moe` | fast-track | Keep. Dense versus MoE comparisons are in scope. |
| 4.1k | `levanter.grug.attention` | grug model | Keep |

## Spike fixes

| Finding | Fix | Phase |
|---|---|---|
| Compilation cache disabled locally | `levanter/compile_cache.py` (P5a). Until P5a, `prune_check` sets `JAX_COMPILATION_CACHE_DIR`. | P5a |
| cuDNN fails on sm_120 | NVRTC builtins preload (P5a). Until P5a, `prune_check` sets `LD_LIBRARY_PATH`. P8 decides whether PyTorch moves to its CPU wheel, which removes the CUDA 12 copies of cuDNN, NCCL, and NVRTC from the venv. | P5a, P8 |
| `dev` resumes old runs | A step with a mutable version (`dev` or `<label>-dev`) gets a fresh output path on each invocation: `{name}/{version}-{YYYYmmdd-HHMMSS}-{4 hex chars}`. The launcher prints the suffix. `--resume <suffix>` reuses an earlier path. Steps with calendar versions (`YYYY.MM.DD` or `YYYY.MM.DD.N`, which Marin treats as immutable) are unchanged. The tokenized datasets in the roots pin calendar versions, so their caches are reused across runs. | P0 |
| Grug's slow default attention on GPU | Make the GPU default either cuDNN (added as a grug attention implementation) or `gpu_fa4_cute`. Implementation picks the one that supports the `AttentionMask` features grug uses: causal masking, segment IDs, and sliding windows. | P0 |
| Extra step compiles from diagnostics | The local-first grug launcher disables gradient watch and backward flow by default. Pass 2 makes backward flow lazy. | P0, Pass 2 |
| PyTorch on every import | `levanter.compat.hf_checkpoints` made lazy | Pass 2 |
| No FLOPs entry | Add the RTX 5090 to `fray.device_flops` | P0 |
| Run-to-run loss noise | Not in this sub-project. Sub-project 5 measures the noise floor. | None |

## Agent instructions and documentation

- P1 rewrites `AGENTS.md` for Poorin. It keeps the code style, naming, typing, error-handling,
  testing, and comment rules. It drops Echo, Iris and Zephyr operations, hero runs, ferries,
  infrastructure grants, PR labels, and comment-prefix rules.
- P1 deletes the skills in `.agents/skills/` that serve Marin's organization or deleted systems,
  for example `use-iris`, `manage-hero-run`, `run-ferries`, `consult-echo`, `add-grant`,
  `review-grant`, `write-ops-log`, `babysit-zephyr`, `query-finelog`, `marina-applet`, and
  `launch-rl`. It keeps general ones such as `debug`, `profile-training`, `write-tests`,
  `change-grug`, `add-pallas-kernel`, and `writing-style`. The implementation plan lists every
  skill with its verdict.
- Library-level `AGENTS.md`, `OPS.md`, and `TESTING.md` files are deleted with their libraries.
- Marin's MkDocs pages are deleted with the code they describe. P7 removes the remainder.
- `docs/poorin/` holds `architecture.md` (one entry per surviving package and module: what it
  does, why it is kept, what uses it), `prune-metrics.csv`, the spike 0 note, and specs.

## Out of scope

| Item | Where it goes |
|---|---|
| Redesigning Fray's job API | Sub-project 2 |
| Deciding whether Haliax stays; porting Haliax-based code to plain Equinox | Sub-project 1b |
| The Flax bridge for pretrained models | Sub-project 4 |
| Porting finished architectures to PyTorch or Flax | Sub-project 8 |
| A persistent local Finelog service shared by all runs; forwarding from Modal and TPU runs | Sub-project 2 |
| Making the post-training kit runnable | A later sub-project |
| Evaluation contracts for all four model families; a local orchestrator; simulator and video-metric suites | Sub-projects 7a and 7b |
| Measuring run-to-run noise | Sub-project 5 |
| Video, trajectory, simulation, and VLA work | Sub-projects 3, 4, and 6 |

## Risks

| Risk | Mitigation |
|---|---|
| Static analysis misses modules loaded by name: plugin registries, `importlib`, string entry points | The four registries in this repository are listed as roots. R1 and R2 execute real runs. Before deleting a module, search the surviving tree for its dotted name as a string. |
| A kept root that cannot run decays | Kept roots must import and type-check on every commit. Sub-projects 3 and 5 give fast-track and the reference suite local data so they become runnable roots. |
| `prune_check` becomes too slow to run on every commit | Target under 10 minutes. If the test suite exceeds that, run the full suite once per phase and a fast subset per commit. |
| Marin stops publishing `marin-finelog-server`, or a later client change breaks protocol compatibility with the pinned server | The server is pinned. Its source remains at `upstream-baseline` and builds with `maturin` if a patch is needed. |
| A lock regeneration hides an unintended version change | Upgrades are allowed in any phase. Each dependency change goes in its own commit, and its `uv.lock` diff is reviewed for versions that moved without being asked. |
