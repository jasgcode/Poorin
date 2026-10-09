# Grug Layout and Usage

`experiments/grug/` is template-first. You edit experiment code here directly.

## Directory layout

- `base/model.py`: model config and model implementation (`init` + `__call__` + loss method).
- `base/train.py`: train loop, optimizer step, callbacks, eval/checkpoint wiring.
- `base/launch.py`: experiment config and execution entrypoint (builds a lazy `LevanterCheckpoint` + resources).

## Entry-point guide

- Start in `base/launch.py` for normal run edits.
- `GrugBaseLaunchConfig` is the user-facing knob surface (model/data/optimizer/trainer/eval/run metadata).
- `grug_base_trial(...)` returns a typed `ArtifactStep[LevanterCheckpoint]` handle addressed by `name@version`; its recipe's
  `build_config` is what the fingerprint is computed from.
- `ctx.output_path` (inside `build_config`) resolves to the current step's output root.
- `run_grug(...)` in `base/train.py` is the runtime entry point the step's `run` function (`run_grug_base_trial`) calls.
- `P` in train/model code is the usual JAX alias for `PartitionSpec`; see the JAX explicit sharding tutorial: [Explicit Sharding (JAX)](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html).

## How to use it

To change a field of `GrugBaseLaunchConfig` (steps, the optimizer, a model size), pass `--set` to
`base/launch.py` (see [Quickstart launch](#quickstart-launch)). A variant is for code changes:

1. Copy `experiments/grug/base` to a new variant directory (for example `experiments/grug/moe`).
2. Make model/training changes in that variant, not in shared trainer libraries.
3. Set run knobs in `<variant>/launch.py` (run id, data mix, optimizer, resources).
4. Launch from the variant's `launch.py` entrypoint.
5. Add or update variant-specific notes in `experiments/grug/variants.md`.

## Variant notes

Variant-specific guidance lives in `experiments/grug/variants.md`.

## Quickstart launch

The launcher takes the shared experiment CLI (`poorin.experiment.cli`): `--version` sets the
checkpoint version and is required, and `--run` builds the trial (without it the lowered plan is
printed and nothing runs). Use `--version dev` for normal experiments, diagnostics, ablations, and
sweeps. A `dev` version rebuilds on each invocation and prefixes the artifact name with
`users/<username>/`, so Grug checkpoints land under `users/<username>/grug/...`. Use a calendar
version such as `YYYY.MM.DD` only for a coordinated major production run that needs a shared,
citable checkpoint path. Calendar-versioned Grug runs use the shared `grug/...` namespace.

Print the plan (no run):

```bash
uv run --no-sync python experiments/grug/base/launch.py --version dev
```

Local run (lowers the trial and runs it through the `StepRunner`):

```bash
uv run --no-sync python experiments/grug/base/launch.py --version dev --run
```

`--set PATH=VALUE` overrides one field of `GrugBaseLaunchConfig` by dotted path, and `--sweep`
runs one trial per value. Each override enters the trial's fingerprint, its name
(`grug/base-trial/steps=200`), and its run ID. `--show-config` prints each trial's config without
running anything.

```bash
# Override fields by dotted path; values are YAML, decoded against the field's type
uv run --no-sync python experiments/grug/base/launch.py --version dev \
    --set optimizer.learning_rate=1e-3 --set steps=200 --run

# Swap a sub-config by its registered name
uv run --no-sync python experiments/grug/base/launch.py --version dev \
    --set optimizer='{type: muon, learning_rate: 0.02}' --run

# Sweep: one trial per value; several --sweep options form a grid
uv run --no-sync python experiments/grug/base/launch.py --version 2026.10.09 \
    --sweep 'optimizer.learning_rate=[1e-3, 3e-3, 1e-2]' --max-concurrent 1 --run
```

A sweep with more than one trial needs `--max-concurrent`; `1` trains the trials one after another
on one GPU. Under a calendar version, rerunning the sweep builds only the trials that are missing;
under `dev`, each invocation runs every trial.

## Diffing a variant against base

When template-copying `experiments/grug/base/` to a new variant, review the changes with git:

```bash
git diff --no-index --stat experiments/grug/base experiments/grug/<variant>
git diff --no-index experiments/grug/base experiments/grug/<variant>
```

## Common edit points

- Architecture changes: `experiments/grug/base/model.py`
- Train-loop and callback behavior: `experiments/grug/base/train.py`
- Run config, resources, and launch wiring: `experiments/grug/base/launch.py`
- Field changes: `--set` and `--sweep` on `base/launch.py`, no copy needed.
- Copy-paste variant workflow, for code changes: duplicate `experiments/grug/base/` into `experiments/grug/<variant>/` and edit there.

## Trainer knobs people ask about

- `z_loss_weight` in `GrugTrainerConfig`: weight on the logsumexp stabilization term in LM loss.
- `ema_beta` in `GrugTrainerConfig`: exponential moving average (EMA) coefficient for eval/checkpoint model; `None` disables EMA.

## Checkpoints and resume

- Checkpoints are written to `<output_path>/checkpoints` by default in `base/launch.py`.
- An own-run checkpoint resumes the complete state, including the optimizer and step, and takes precedence over
  `trainer.initialize_from`.
- `trainer.initialize_from` initializes a new phase from external model weights when no own-run checkpoint was loaded.
  The optimizer and step remain fresh. EMA starts from the loaded parameters.
- `trainer.load_checkpoint_path` is a complete-state resume, not a phase-chaining input.
- If `trainer.load_checkpoint=True` and no resume checkpoint is found, startup fails. With no resume checkpoint and no
  `trainer.initialize_from`, the run starts from scratch.

## Environment variables you will likely use

- `WANDB_API_KEY`: required for W&B logging in the default launch config.
- The run ID is `grug-base-<version>`, followed by `-<digest>` when the trial has `--set` or `--sweep` overrides. `--version dev` gives each invocation a fresh timestamped version; the launcher prints it, and `--resume <version>` continues that run.

## Where outputs show up

- Training/eval metrics: tracker backend (default W&B).
- Checkpoints: `<output_path>/checkpoints`.
- Profiler traces (if enabled): `<trainer.log_dir>/<run_id>/profiler`.
- Backward-flow artifacts: `<trainer.log_dir>/<run_id>/artifacts/backward_flow`, plus `backward_flow/dag` as native HTML media in W&B. Base Grug disables it by default (`backward_flow_interval=0` in `base/launch.py`); set a positive interval to sample it.
- Step outputs: the `ctx.output_path` root for the step.

## Logged metrics

- `train/loss`: training loss for the just-completed step.
- `global_step`: completed optimizer step index.
- `run_progress`: completed fraction of the configured run (`step / total_steps`).
- `optim/*`: optimizer hyperparameters from Optax state (for example `optim/learning_rate`).
- `throughput/duration`: step wall-clock duration (after loss is materialized).
- `throughput/examples_per_second`: examples processed per second for the current batch size.
- `throughput/tokens_per_second`: tokens processed per second.
- `throughput/total_tokens`: cumulative tokens processed so far (schedule-aware).
- `throughput/gflops_per_second`: model FLOP throughput from analytic FLOPs-per-example.
- `throughput/total_gflops`: cumulative model FLOPs (analytic).
- `throughput/mfu`: model FLOP utilization as percent of theoretical hardware FLOPs.
- `throughput/hook_time`: callback/logging overhead time after each step.
- `throughput/loading_time`: dataloader wait time for the current step.
- `throughput/flops_per_token_analytic`: analytic FLOPs per token summary value.
- `throughput/flops_per_example_analytic`: analytic FLOPs per example summary value.
- `throughput/flops_per_example`: FLOPs-per-example value used by throughput callback.
- `throughput/device_kind`: accelerator type string from JAX device info.
- `throughput/theoretical_flops_per_device`: theoretical peak FLOPs per device.
- `throughput/theoretical_flops`: theoretical peak FLOPs across all devices.
- `mixture/stage`: current data-mixture stage index.
- `mixture/weight/<dataset_name>`: effective sampling weight per dataset in the active stage.
- `eval/loss`, `eval/loading_time`, `eval/total_time`: tagged-eval loss and timing for current model.
- `eval/ema/*`: same eval metrics for EMA weights when EMA is enabled.
- `eval/macro_loss`: macro average loss across tags when multiple tags exist.
- `eval/<tag>/loss`, `eval/<tag>/micro_loss`, `eval/<tag>/macro_loss`: per-tag loss views.
- `eval/bpb`, `eval/macro_bpb`, `eval/<tag>/bpb`, `eval/<tag>/macro_bpb`: bits-per-byte metrics when tokenizer/BPB logging is enabled.
- `grad/*`, `params/*`, `updates/*`, `opt_state/*`: optional watch metrics (norms/histograms) when watch is enabled.
- `backward_flow/<scope>/*`: sampled activation and backward-gradient scale stats when `trainer.backward_flow.interval > 0`. Grug also logs `*_gradient_rms_scaled`, where gradients are multiplied by `sum(loss_weight)` to undo mean-loss scaling for the visualization.

## What should stay consistent

- Keep core training/eval metrics aligned with classic Levanter (`train/loss`, `throughput/*`, `eval/*`).
- Prefer shared helpers only for generic infrastructure; keep variant behavior local to the template.

## Variant contract (enforced by tests)

`tests/test_grug_variant_contracts.py` treats each subdirectory under `experiments/grug/` as a variant and
enforces these minimum interfaces:

- If `<variant>/model.py` exists, it must define:
  - `GrugModelConfig` constructable as `GrugModelConfig(vocab_size=...)`
  - `Transformer` with `next_token_loss(...)`
  - `debug_mesh_and_token_pspec(num_devices: int)`
- If `<variant>/train.py` exists, it must define:
  - `initial_state(model_config, *, optimizer, mp, key)` (all required)
  - `_make_train_step(...)`
  - `run_grug(...)`
- If both `model.py` and `train.py` exist, the variant must lower a one-step train path under abstract mesh via
  `eqx.filter_eval_shape`.
- Escape hatch: add `# GRUG NOVERIFY` anywhere in `<variant>/train.py` to exclude that variant from these contract
  checks.

## Further guidance

- Grug principles: [`PRINCIPLES.md`](PRINCIPLES.md)
- Change workflow: [`.agents/skills/change-grug/`](../../.agents/skills/change-grug/SKILL.md)
- Backward-flow recipe: [`/docs/recipes/add_grug_backward_flow_logging.md`](../../docs/recipes/add_grug_backward_flow_logging.md)
- HBM/OOM tuning guide: [`/docs/references/hbm-optimization.md`](../../docs/references/hbm-optimization.md)
- Lazy artifact mechanics: [`/docs/explanations/lazy-artifacts.md`](../../docs/explanations/lazy-artifacts.md)
