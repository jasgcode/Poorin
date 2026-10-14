# Removed from Marin

Poorin is a hard fork of [Marin](https://github.com/marin-community/marin) at commit
[`ab0da00df`](https://github.com/jasgcode/Poorin/tree/ab0da00df254ca0d52648df9169ca284cb5789af),
tagged `upstream-baseline` in this repository. It keeps what one researcher needs on one local GPU
and removes the rest. This page lists what was removed, so you can decide whether Poorin fits your
work or whether you need upstream Marin.

Every path below is the path at `ab0da00df`. To read a removed file, run
`git show ab0da00df:<path>` in a clone of Poorin, or browse it on GitHub at that commit. Each item names the
decision in the [decision log](https://github.com/jasgcode/Poorin/blob/main/docs/poorin/decisions.md)
that removed it, as `Dn`.

## At a glance

| You need | Poorin | Use instead |
|---|---|---|
| A cluster scheduler, multi-host TPU or GPU jobs | Removed (Iris) | Upstream Marin |
| Docker, TPU VM, and cloud launchers | Removed | Upstream Marin |
| Reinforcement learning | Removed from the tree | An external RL framework; Poorin keeps the hooks (HF export, sampling, the executor) |
| Model serving (vLLM, an OpenAI-compatible server) | Removed | An external vLLM server |
| Mistral, Mixtral, Gemma, OLMo, Whisper, MoE, and other Levanter architectures | Removed; Llama, Qwen, and GPT-2 remain | Upstream Levanter, or a grug variant |
| Harbor and Evalchemy benchmarks, fleet evaluation | Removed; lm-eval and evaluation records remain | Upstream Marin |
| Marin's dataset catalog and the Delphi data mixes | Mostly removed; FineWeb-Edu, Hugging Face and Nemotron-CC v1 downloads remain | Upstream Marin |
| GitHub Actions CI | Removed; checks run locally | Upstream Marin's workflows |

## Size

| Measure | `ab0da00df` | Now (2026-10-13) |
|---|---|---|
| First-party Python lines (`lib/*/src`, `experiments/`, `tools/`; tests excluded) | 385,058 | 118,828 |
| First-party Python modules (tests excluded) | 1,490 | 425 |
| Tracked files | 4,440 | 997 |
| `.venv` size | 12.18 GB | 8.18 GB |

About 3,300 of today's lines are new: the episode data layer, the episode model templates, and
run provenance. The first-party line count also leaves out `infra/` (73,000 Python lines),
`scripts/` (17,500), and the Rust and TypeScript sources, all of which were removed.

## Libraries removed whole

| Library | What it was | Decision | What remains |
|---|---|---|---|
| `lib/iris` | Marin's cluster scheduler: controller, workers, actors, CLI, and dashboard | D5 | `iris/jax/compile_cache.py` is now `levanter.compile_cache`; `iris/env_resources.py` is now `zephyr.resources` |
| `lib/ducky` | DuckDB SQL over object-store Parquet, run as an Iris service | Prune P4 | Nothing |
| `lib/shellbox` | Sandboxed shell environments for RL | D6 | Nothing |
| `lib/taskcompendium` | Task definitions exported to Harbor | D6 | Nothing |
| `lib/verifyit` | Graders and benchmark adapters for Harbor, Evalchemy, and lm-eval | D69 | Nothing |
| `lib/haliax` | The in-tree named-tensor library | D102 | The `marin-haliax==0.2.139.dev37117265303` wheel from PyPI, the same source |

## Cluster, launchers, and fleet operations

- **Iris support in the kept libraries** (D5, prune P5b):
    - Fray's Iris backend (`lib/fray/src/fray/iris_backend.py`). `current_client()` returns
      `LocalClient`, which runs jobs on this machine.
    - Levanter's multi-slice TPU setup (`megascale.py`), Iris status page (`training_control.py`,
      `callbacks/_iris_status.py`), and Iris coordinator discovery in `distributed.py`. SLURM and
      JAX's own cluster detection remain.
    - Zephyr's Iris branches. An unset worker count now means one worker per CPU (D91).
    - The executor's multi-host sweep coordination (`lib/marin/src/marin/execution/sweep_coordination.py`).
- **`infra/`** (D8): Pulumi stacks for Marin's clusters, buckets, Cloud SQL, and services; the
  `marin-deploy` CLI; CI test selection; the linter; code-health telemetry; and the TPU CI VM
  manager.
- **`scripts/`** (D8): CI scripts, cloud cost reports, Iris smoke scripts, storage and Discord
  operations, GitHub issue tooling, and run-analysis parsers.
- **`docker/`, `Makefile`, `etc/license_header.txt`** (D41).
- **Levanter's Docker and Ray launchers**: `lib/levanter/docker/tpu`, `lib/levanter/infra/launch.py`,
  `push_docker.py`, and `infra/cluster` (D78, D102). The shell TPU helpers in
  `lib/levanter/infra/` remain. `CliConfig` keeps only `env` and `accel_env` (D95), and
  `TrainerConfig.shutdown_at_exit`, which deleted the TPU VM at exit, is gone (D89).
- **Pinned forks of vLLM, tpu-inference, Evalchemy, Harbor, and SkyRL**: `config/external`,
  `config/update-external.py`, `marin.external_dependencies` (prune P3).
- **Ferries**, Marin's daily and canary fleet runs: `experiments/ferries` (D41).
- **The published-cohort evaluation launcher**: `eval_policy/` (D10).

Marin's cluster configs (`config/marin.yaml`, `config/coreweave.yaml`) are test fixtures in
`config/testing/`. `config/local.yaml` is the default (D43).

## Reinforcement learning and serving

- **The in-repo SkyRL integration**: `lib/marin/src/marin/rl` (D6, D68).
- **RL experiments**: `experiments/post_training` (async RL, curricula, TaskTrove, Snowball online
  EAGLE) and `experiments/rl_data_reviews` (D6).
- **`marin.inference`** (D10): vLLM serving, Iris-hosted serving, the proxy and broker, OpenAI
  batch, tool routes, and the serving dashboard. `RunningModel` and `OpenAIEndpoint` moved to
  `poorin.evaluation.endpoint`.
- **Levanter's OpenAI-compatible server** (`levanter/inference/openai.py`, `openai_protocol.py`) and
  inference REPL (`levanter/main/inference_repl.py`) (D48). Levanter's inference engine remains.
- **Smaller pieces**: the MCP babysitter (`marin/mcp/babysitter.py`), Finestore's RL probe archive
  (`finestore/rl`), and the Snowball and MoE inference-parity test support (`marin/testing`).

Poorin keeps the hooks an external RL framework needs: Hugging Face export in both directions,
`sample_lm` and the inference engine, the chat template, evaluation records and rollouts, and the
executor (D68).

## Evaluation

- **Fleet evaluation** (D10): `marin/evaluation/runner.py`, `serving_config.py`, `hardware.py`,
  `inference_metrics.py`, the cohort policy (`eval_policy.py`, `eval_policy_sources.py`), the lazy
  evaluation steps (`marin/experiment/evaluation.py`), and the launchers in `experiments/evaluation`
  and `experiments/evals`.
- **Harbor and Evalchemy** (D48): `marin/evaluation/harbor` and `marin/evaluation/evalchemy`. Their
  record types remain in `poorin.evaluation.records` (D71).
- **Marin's perplexity-gap wrapper** (`marin/evaluation/perplexity_gap.py`, D47). Use
  `levanter.main.perplexity_gap`.
- **The log-prob visualizer** (`levanter/main/viz_logprobs.py`).

## Models and kernels

- **14 Levanter architectures** (D11), in `lib/levanter/src/levanter/models/`: Mistral, Mixtral,
  Gemma, OLMo, OLMo 3, Whisper (with ASR training and audio data), Hyena, GPT-2 Hyena, ModernBERT,
  Apertus, Snowball, Qwen3 MoE, and the generic MoE. Their configs in `lib/levanter/config/` went
  too (D75). Llama, Qwen, and GPT-2 remain.
- **Grug MoE** (D48): `levanter/grug/grug_moe.py` and `levanter/grug/_moe/`, the expert-parallel
  backends. The QuACK kernel behind fast-track's MuonH optimizer remains.
- **Kernels**: DeepEP (`kernels/deepep`), Mamba-3, SSD, and short convolution
  (`kernels/pallas/mamba3`, `ssd`, `short_conv`), and the Gated DeltaNet layer
  (`layers/gated_deltanet.py`) (D48, prune P7).
- **Pipeline parallelism**: `levanter/pipeline.py` and `mpmd_checkpoint.py` (D89).
- **GPU hang recovery**: `levanter/recovery` (D89). The progress watchdog remains.
- **TensorBoard tracking** (`levanter/tracker/tensorboard.py`, D93). Runs track with W&B.

## Data

- **Most of the datakit catalog** (prune P7): about 70 download sources under
  `marin/datakit/download/`, the source registry (`sources.py`), the ingestion manifest, and the SFT
  sources. The Hugging Face, Nemotron-CC v1, and StackExchange downloaders remain, with normalize,
  decontamination, and the chat template.
- **Format converters**: `marin/transform` (HTML, ar5iv, Wikipedia, StackExchange, conversation and
  preference data to Dolma).
- **Cluster dedup** (`marin/processing/classification/deduplication/cluster_dedup.py` and its
  siblings). Exact and fuzzy MinHash dedup and consolidation remain (D47).
- **Dataset recipes**: the Delphi-mix recipes (`experiments/datasets/nemotron.py`, `starcoder.py`,
  `proofpile.py`, `paloma.py`, `uncheatable.py`; D88) and 22 others, such as `dolma.py` and
  `dclm.py`. `experiments/datasets/fineweb_edu.py` remains.
- **Data experiments**: `experiments/datakit`, `experiments/tokenize`, `experiments/rollout_data`.

## Experiments

- **Grug MoE recipes**: `experiments/grug/moe`, `moe_hero_ep`, `moe_hero_fsdp`, `moe_pipeline`, and
  fast-track's MoE ladder (`launch.py`, `train.py`, `model.py`) (D48). Fast-track's optimizers,
  heuristic, and README with its results remain.
- **Hero runs and benchmarks**: `experiments/june_tpu_67b_a2b`, `experiments/sft`,
  `experiments/coral`, `experiments/grug/recovery`, `experiments/grug/snowball_synthetic`, and the
  FA4 tile sweep.
- **Reference runs**: the reference scaling suite, hyperparameter sweep, and training pipeline in
  `experiments/references/` (D48). `poorin.scaling_laws` and the Delphi recipe
  (`completed_adamh.py`) remain.
- **Tutorial extras**: the sweep example, hello world, the DCLM 7B reproduction, and the cloud
  devices in `train_tiny_model.py`, which now offers `cpu` and `rtx5090x1` (D106).

## Cut from kept libraries

- **The experiment and execution layers** (D105, D109): the step runner's own Fray submission,
  `dry_run`, plain-step hashing (`hash_attrs`, `name_with_hash`), readers of Marin's older on-disk
  formats, `ArtifactStep.adopt`, race-to-claim sweeps, and the HF-to-Levanter conversion step. A
  run now reads only the current artifact record.
- **Rigging**: authentication, credentials, secrets, cluster tunnels and RPC, Prometheus telemetry,
  and NCCL stall probes (prune P7, D89).
- **Finelog**: the Rust server and Vue dashboard (D14). The client remains, and the server comes from
  the `marin-finelog-server==0.2.46.dev33324331711` wheel.
- **Finestore**: `admin.py`, `fileset.py`, and `rl/`.
- **Dependencies**: CUDA PyTorch (PyTorch now comes from the CPU wheel, D79), `tensorboardX`,
  `jax-triton`, `triton`, `fasttext`, `openai`, `mcp`, Cloud SQL, and others. The lockfile resolves
  Linux only (D80).

## CI, community files, and docs

- **`.github/`**, all 37 workflows and the actions (D18). Checks run locally with
  `tools.prune_check` and `tools.affected_tests`.
- **Community files** (D101): `CODE_OF_CONDUCT.md` and the contributing guides. Poorin takes no
  outside contributions. `AUTHORS.md` and `CONTRIBUTORS.md` remain, because the license headers
  name the Marin Authors.
- **Agent notes**: Marin's `.agents/` logbooks and project notes, and 29 of its 41 skills (D41).
- **Docs for removed code** (D76): reports, model cards, Iris and Finelog designs, TPU and cloud
  tutorials, RL launching, and the Harbor integration.

## Tooling replaced

| Upstream | Poorin | Decision |
|---|---|---|
| pyrefly | ty 0.0.84; warnings fail | D49, D54, D89 |
| black | `ruff format` | D50 |
| `infra/pre-commit.py`, `infra/linter.py` | `tools/lint.py` | D39 |
| `infra/ci/select_tests.py` | `tools/affected_tests.py`, from the import graph | D37 |
| GitHub Actions | `tools/prune_check.py`, run before each commit | D17, D18 |
| Python 3.12 | Python 3.13 | D111 |

## Renames

| Marin | Poorin | Decision |
|---|---|---|
| `marin` package, `marin-core` distribution, `lib/marin` | `poorin`, `lib/poorin` | D85 |
| `fray`, `finelog`, `rigging` distributions (`marin-*`) | `poorin-fray`, `poorin-finelog`, `poorin-rigging`; import names unchanged | D86 |
| `MARIN_PREFIX`, `MARIN_TEMP_PREFIX`, and the other `MARIN_*` settings | `POORIN_*` | D87 |
| `marin_prefix()`, `marin_temp_bucket()`, `marin_region()` | `poorin_*` | D87 |
| `~/.config/marin/` | `~/.config/poorin/` | D95, D99 |
| W&B project `marin`, output fallback `/tmp/marin` | `poorin`, `/tmp/poorin` | D87 |

Artifact records written by Marin name their result type as `marin.…`. Rewrite
`"result_type": "marin.` to `"result_type": "poorin.` once to reuse them. Names that point at
something of Marin's keep it: the license header, `marin_tokenizer`, the Marin chat template, the
`gs://marin-*` buckets, and the `marin-haliax` and `marin-finelog-server` wheels.

## Kept

These remain, though a fork this size might be expected to drop them:

- **Post-training**: `levanter.main.train_dpo`, `lora_lm`, and `sample_lm`, with SFT and chat data
  formats (D9, D88).
- **Evaluation**: `poorin.evaluation` (lm-eval, records, rollouts, eval stats), `levanter.eval_harness`,
  `levanter.main.eval_lm`, and `levanter.main.perplexity_gap` (D10, D88).
- **Scaling laws**: `poorin.scaling_laws`, the Delphi recipe, and fast-track's AdamH and MuonH.
- **Data processing**: tokenization, normalize, exact and fuzzy dedup, decontamination, Dupekit,
  and Zephyr with its dashboard (D47, D90).
- **Profiling**: `poorin.profiling` (D78).
- **TPU code paths**: splash attention, the Pallas fused cross-entropy kernel, and JAX's TPU
  cluster detection (D81). Fray's remote `ResourceConfig` fields stay for later cloud dispatch.
