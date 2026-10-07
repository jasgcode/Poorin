# Amendment: Poorin as a workbench, and the roots that follow

Status: approved 2026-10-06. Superseded in part the same day by D51 and D52: projects run on
Marin's executor and live in Poorin's uv workspace, so they import Poorin's code; the files
described below remain how data moves between them. Amends
[`2026-10-04-core-prune-design.md`](2026-10-04-core-prune-design.md). Decisions D44 to D48 in
[`../decisions.md`](../decisions.md) record the choices made while writing it.

## TL;DR

- Research projects live in their own repositories and use Poorin as a workbench only. A
  project imports nothing from Poorin. Poorin runs in its own environment and writes files the
  project reads: tokenized datasets, subset index files, model weights, evaluation results, and
  scaling-law fits.
- Poorin's kept roots become the tools that workbench runs. Reachable code falls from 455
  modules and 138,099 lines to 414 modules and 126,306 lines.
- P4 keeps Dupekit. P3 also deletes the Harbor and Evalchemy integrations. The MoE scaling
  ladder leaves the kept roots, so P7 deletes it.

## Context

The first project is `looped-diffusion-lm` (`../looped-diffusion-lm`): small LMs, 10M to 500M
parameters, that read a diffusion-sampled latent plan, compared against autoregressive
baselines on a Delphi-style recipe, with data-constrained runs and scaling-law fits. Its
implementation setup already defines its own config system (Hydra with typed schemas), run
identity, seed handling, checkpoints (Orbax), and analysis step.

Three decisions on 2026-10-06 set Poorin's role:

| Decision | Content |
|---|---|
| D44 | Two layers: a workbench and a library. Superseded in part by D45. |
| D45 | No `poorin` library and no project kit. Poorin stays the pruned set of Marin's utilities; the owner writes a lightweight research template separately. |
| D46 | Projects use Poorin as a workbench only and copy small pieces of code (AdamH, the fused cross-entropy) when they need them. |

A survey of Marin's data, training, and evaluation code then sorted its utilities into tools to
run, code to copy, and code with no use outside Marin's infrastructure. It also found that
nothing in Marin handles video, images, robot trajectories, VLAs, world models, or simulators.

## What Poorin writes for a project

The files are the whole contract between the repositories. Each file kind gets a format
definition with a version number under `docs/poorin/formats/` when the workbench first
produces it.

| Output | Format | Produced by |
|---|---|---|
| Tokenized text | Levanter's cache, one directory per split: `input_ids/data` holds every document's tokens concatenated (int64, Zarr v3, LZ4-compressed); `input_ids/offsets` holds the document count at index 0 and each document's end position after it; `shard_ledger.json` records the row count and `is_finished`. Each document is BOS, its tokens, then EOS. New: a second field stores each document's source identifier (the source dataset's ID where it has one, otherwise normalize's content hash), which the tokenization step has to carry through. A manifest beside the cache records the tokenizer (path and hash), the source datasets and their versions, document and token counts, and the validation documents split into dev and test halves. | `marin.experiment.data.tokenized` through the executor, so each output has a versioned path |
| Subset index files | Nested lists of row numbers in one cache version, for example 25M ⊂ 100M ⊂ 300M ⊂ 1B unique tokens, with the seed that drew them | A new workbench step |
| Model weights | safetensors plus a JSON config (D15) | Workbench export tools |
| Evaluation results | lm-eval task results, from the JAX harness or an endpoint run, as the evaluation framework's `record.json`; BPB per dataset and perplexity-gap reports as JSON | `levanter.eval_harness`, `marin.evaluation.lm_eval`, `levanter.main.eval_lm`, `levanter.main.perplexity_gap` |
| Scaling-law fits | Input: a table of runs with columns method, budget, parameters, tokens, and loss. Output: the fit parameters and plots. | `marin.scaling_laws`, given a small entry point that reads the table |

- The tokenizer is an input. A project that needs a custom tokenizer, such as Llama 3's
  truncated to 32k merges, builds it and passes its directory; `levanter.tokenizers` accepts a
  local directory.
- Training windows are the project's business. Levanter cuts its stream into consecutive
  windows that cross document boundaries; a project's reader may do the same or keep one
  document per window.
- `marin.scaling_laws` has no bootstrap bands and no parametric L(N, D) fit. They are added when
  a project first needs them.
- Reference models that Levanter cannot load, such as Gemma 3 with its alternating local and
  global attention, are evaluated through `marin.evaluation.lm_eval` against a local vLLM
  server.

## Kept roots

This table replaces the core spec's "Kept roots" table. The runnable roots R1 and R2 are
unchanged, and `tools.*` joins in P1.

| Group | Roots |
|---|---|
| Data preparation | `marin.experiment.data`, `marin.processing.tokenize.*`, `marin.datakit.download.huggingface`, `marin.datakit.download.nemotron_v1`, `marin.datakit.canonical.fineweb_edu`, `marin.datakit.normalize`, `marin.datakit.decon`, `marin.processing.classification.consolidate`, and the dedup modules `exact`, `fuzzy_minhash`, `fuzzy_dups`, `verify_fuzzy_dups` |
| Run management | `marin.experiment.cli`, `marin.experiment.sweep` |
| Dataset recipes | `experiments.datasets.nemotron`, `starcoder`, `proofpile` (the Delphi mix), `paloma`, `uncheatable` (validation sets) |
| Evaluation | `levanter.main.eval_lm`, `levanter.eval_harness`, `levanter.main.perplexity_gap`, `experiments.evals.task_configs`, and `marin.evaluation.lm_eval`, `records`, `eval_stats`, `eval_measurements`, `metric_selection`, `rollouts`, `model_config`, `model_identity`, `eval_env`, `trace_labeled_eval` |
| Scaling laws and the Delphi recipe | `marin.scaling_laws` and its submodules, `experiments.references.completed_adamh`, `experiments.grug.fast_track.heuristic` (which keeps fast-track's AdamH and MuonH) |
| Post-training kit (D9) | `levanter.main.train_dpo`, `lora_lm`, `lora_dpo`, `sample_lm` |
| SkyRL integration hooks (D68, added in P2) | `levanter.compat.hf_checkpoints`, `levanter.main.export_lm_to_hf`, `marin.datakit.chat_template` |
| Plugin registries | `levanter.optim.*`, `levanter.data.text.*`, `levanter.tracker` (`tracker`, `wandb`, `json_logger`, `json_file`, `telemetry`), `levanter.models.llama` |

The `static_only` list becomes empty: its two entries belonged to Harbor.

The code a project would copy stays reachable through these roots and R2: the fused
cross-entropy (XLA and TPU paths), Splash attention, the attention reference, AdamH and MuonH,
the Delphi formulas, FLOP counting, the seeded data permutation, and the backward-flow
diagnostics.

## Removed roots

| Root | Why | Deleted in |
|---|---|---|
| `marin.evaluation.harbor.*`, `marin.evaluation.evalchemy.*` | Their runners need Iris or Marin's serving sessions, and agentic benchmarks don't fit small-LM research | P3, with `config/external/harbor`, `config/external/evalchemy`, and `marin.external_dependencies` |
| `experiments.grug.fast_track.launch`, `train`, `model` | The MoE scaling ladder; its heuristic and optimizer stay | P7, with grug MoE, DeepEP, and the short-convolution kernel |
| `experiments.references.reference_scaling_suite` | It fits isoFLOP checkpoints on Marin's private buckets | P7 |
| `levanter.inference.openai`, `experiments.tutorials.train_tiny_sweep` | The server serves only Levanter's own models; the sweep example only demonstrates `marin.experiment.sweep`, which stays | P7 |
| `marin.evaluation.perplexity_gap` | It asserts TPU resources and submits a Fray job; `levanter.main.perplexity_gap` does the same work locally | P7 |

## Changes to the core prune spec

- **Libraries table:** Dupekit is kept. Its Python package is a 46-line wrapper over the
  prebuilt `dupekit_native` wheel, and normalize, dedup, and decontamination import it. P4
  deletes only Ducky.
- **Pass 1, P3:** also deletes the Harbor and Evalchemy integrations, their `config/external`
  pins, and `marin.external_dependencies`. The edges the core spec listed for
  `evalchemy/runner.py`, `harbor/runner.py`, and `harbor/driver_config.py` no longer need
  cutting.
- **Pass 2, initial targets:** the DeepEP and grug MoE rows are moot: both become unreachable
  and P7 deletes them. The row about fast-track importing a helper from
  `experiments.grug.moe` is moot for the same reason.
- **Finelog (D14):** unchanged. It stays reachable through Levanter's trainer, which imports the
  telemetry tracker.

## Measurement

`tools/reachability.py`'s closure over the current and proposed roots, with the same deleted
list, before `tools.*` becomes a root:

| Roots | Modules | Lines |
|---|---|---|
| Current | 455 | 138,099 |
| Proposed | 414 | 126,306 |

The proposed roots drop 57 modules and 17,350 lines: the MoE ladder and its kernels (about
12k), Harbor and Evalchemy (3.0k), the OpenAI server (1.0k), the reference suite (0.5k), and
Marin's perplexity-gap wrapper (0.3k). They add 16 modules and 5,557 lines of data tooling.

## How it lands

P1 gains a task that replaces the kept roots in its own commit, so the drop gets its own row in
`prune-metrics.csv`. P1's final documentation task points the core spec's "Roots" section at
this document.

## Open question

Which of the core spec's later sub-projects stay with Poorin now that research lives in
separate repositories. The data layer (3), compute dispatch for workbench jobs (2), evaluation
contracts (7a), and release ports (8) are workbench concerns. The model templates (4), the
cross-family scaling harness (5), the simulation loop (6), and the per-family evaluation
mechanisms (7b) may belong to projects or to the owner's research template instead.
