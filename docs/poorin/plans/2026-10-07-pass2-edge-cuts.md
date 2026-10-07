# Pass 2 Edge Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop training runs from loading code they do not use, then give every surviving module a
verdict and an entry in `docs/poorin/architecture.md`.

**Architecture:** Each cut removes one module-level import edge that drags a subsystem into the
training roots. Cuts are structural first: move a light type (config dataclass, protocol) out of
the heavy module it lives in, so the hub imports only the light part. A function-local import is
used only where AGENTS.md allows one: an optional dependency (PyTorch, transformers, lm-eval) or a
backend-specific kernel (TPU Pallas). Then the ranked loop of the spec assigns Delete, Make lazy,
or Keep to the top modules, and `architecture.md` records why each survivor exists.

**Tech Stack:** `tools/reachability.py` (`--rank`, `--eager`), `python -X importtime`,
`tools/prune_check.py`.

**Spec:** `docs/poorin/specs/2026-10-04-core-prune-design.md`, "Pass 2: edge cuts", amended by
`2026-10-06-workbench-and-roots-design.md`.

## Global Constraints

- One cut per commit; each commit runs `tools.prune_check` in full and appends a metrics row.
- A cut changes no behavior: tests that passed keep passing without edits beyond import paths.
- Lazy imports follow AGENTS.md: only for optional dependencies or backend-specific kernels.
  `TYPE_CHECKING` guards are removed, not added.
- Measures: `r1_imported_modules` and `r1_import_time` (prune_check), and the eager reach of R1
  and R2 (`tools.reachability --eager`).

## Review Focus

1. A moved class keeps its import path somewhere a config file names it: draccus choice
   registries and YAML `type:` keys resolve by class registration, so every subclass must still
   register before parsing. The config-parsing tests and R1/R2 cover the training configs.
2. A cut that breaks a kept root's import: prune_check imports every kept root first.
3. A local import of an optional dependency that runs on every call in a hot loop: keep it at the
   entry of the code path, not inside per-step functions.
4. Pickled or serialized objects that record a class's module path (checkpoints, executor step
   configs): moved dataclasses must not appear in persisted state by module path. Executor
   fingerprints hash config values, not module paths; verify with R1's and R2's fresh runs.
5. The TPU Pallas path loses its registration: the fused cross-entropy tests that select
   `pallas_tpu` by name skip on CPU today and must still find the implementation on TPU.

---

## Phase A: the spec's initial targets

Measured 2026-10-07 after P8. R1 (`experiments.tutorials.train_tiny_model`) imports 5,985
modules in 3.34 s; `transformers` and `torch` alone take about 1 s of that.

| Task | Edge to cut (eager chain from R1) | Cut |
|---|---|---|
| A1 | `marin.experiment.train` and `levanter.main.train_lm` import `LmEvalHarnessConfig` from `levanter.eval_harness`, which loads the inference engine | Move `LmEvalHarnessConfig` (and the light configs it holds) into `levanter.eval_harness_config`; the harness module imports it from there |
| A2 | `marin.training.training` imports `levanter.main.train_dpo` for `TrainDpoOnPodConfig` | Move the DPO wrapper and its runner into `marin.training.dpo`; update callers |
| A3 | `levanter.main.train_lm` imports `levanter.analysis`, whose `__init__` re-exports backward flow and every analysis helper | Empty the re-export hub; import from the defining submodules |
| A4 | `levanter.layers.attention` imports the splash kernel at module level; `fused_cross_entropy_loss.api` imports `pallas_tpu` at module level | Import TPU kernels inside the TPU code path (backend-specific) |
| A5 | `levanter.models.{gpt2,llama,qwen}` and `train_lm` import `levanter.compat.hf_checkpoints`, which imports `transformers` (and through it `torch`) | Move the light protocol types (`HFCompatConfig`, the serialization mixin) out, or import `transformers` only in the conversion code paths; drop the `TYPE_CHECKING` guard |
| A6 | grug's `model.py`, `train.py`, and `launch.py` import `levanter.analysis.backward_flow` | Split `BackwardFlowConfig` from the capture machinery; load the machinery only when the interval is positive |

Each task:

- [ ] Write or locate the test that pins the behavior the cut must keep (config parsing, kept-root
  import, the code path that still needs the heavy module).
- [ ] Apply the cut; run the affected tests.
- [ ] Check the edge is gone: `chain.py`-style eager search from R1 and R2 (scratch helper over
  `tools.reachability.build_graph(..., eager=True)`).
- [ ] `uv run --no-sync python -m tools.prune_check --phase Pass2 --step "<cut>"`.
- [ ] Commit with the metrics row; record any choice between options in `decisions.md`.

## Phase B: the ranked loop

- [ ] `uv run --no-sync python -m tools.reachability --rank 30`.
- [ ] For each top module without a verdict: Delete, Make lazy, or Keep, per the spec. Delete and
  Make lazy follow the Phase A task steps. Keep gets its `architecture.md` entry in the same
  commit.
- [ ] Stop when the 10 highest-ranked modules all have Keep verdicts.

## Phase C: `architecture.md`

- [ ] One section per library, one row per surviving module: what it is for, which root or kept
  workflow needs it, and its verdict. Rows for Keep verdicts from Phase B link their reasons.
- [ ] A check that every module in `tools.reachability.workspace_modules` has a row.
- [ ] Commit; Pass 2 ends.
