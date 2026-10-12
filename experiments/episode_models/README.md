# Episode model templates

Three flow-matching templates that train on episode datasets (`poorin.episodes`, D110), one per
model family besides the language model. Design: `docs/poorin/specs/2026-10-11-episode-model-templates-design.md`
(D113).

| Template | Predicts | Conditions on | Default dataset |
|---|---|---|---|
| `wam/` | Future frames and future actions, each at its own flow time | History frames and actions, the task | `libero-spatial` |
| `world_model/` | Future frames | History frames | `ssv2-500` |
| `vla/` | A chunk of future actions | History frames, the task | `libero-spatial` |

The world model and the VLA are the WAM's model with one stream deleted; compare their `model.py`
files to see what each family removes.

## Running

Every launcher takes the shared experiment CLI. Without `--run` it prints the plan.

```bash
# Print the plan, then train the WAM on LIBERO-Spatial
uv run --no-sync python experiments/episode_models/wam/launch.py --version dev
uv run --no-sync python experiments/episode_models/wam/launch.py --version dev --run

# Another example dataset
uv run --no-sync python experiments/episode_models/world_model/launch.py --dataset vpt-10 --version dev --run

# Override any field by dotted path; sweep a size knob
uv run --no-sync python experiments/episode_models/vla/launch.py --version dev \
    --set train.steps=1000 --set data.clip.horizon=16 --run
uv run --no-sync python experiments/episode_models/vla/launch.py --version dev \
    --sweep 'model.hidden_dim=[128, 256, 384]' --run

# Continue an interrupted dev run from its last checkpoint
uv run --no-sync python experiments/episode_models/wam/launch.py --resume <VERSION> --run
```

`POORIN_PREFIX` must point at the storage that holds the example datasets. The launch config has
four parts:

- `model`: the family's config. `hidden_dim` and `num_layers` are the size knobs; `head_dim` is
  64, so `hidden_dim` is a multiple of 64. `patch_size` must divide the frame size.
- `data`: the source's fields (`video_field`, `action_field`, `task_key`), the clip
  (`history`, `horizon`, `stride`, `skip`), `eval_percent`, and `num_workers`.
- `train`: `steps`, `batch_size`, `seed`, precision (`mp`), `optimizer`, `steps_per_eval`,
  `eval_batches`, `checkpoint_every`.
- `run`: the output path, run id, GPU, tracker, and dataset paths, taken from the step context.
  They are not part of the trial's identity, and `--set` refuses them.

## What every run logs

The metrics a scaling sweep reads are the same in every family:

- `parameter_count`
- `throughput/flops_per_example_analytic`: training FLOPs per clip (three forward passes), checked
  against XLA's cost analysis in the tests
- `eval/loss` and `eval/<stream>_loss`: the held-out loss, each stream's flow-matching loss averaged
  over eight fixed flow times with noise from a fixed key, on episodes held out by a hash of their
  id. It is deterministic for given parameters.
- `train/loss`, `train/<stream>_loss`, and Levanter's throughput and MFU

## Reproducibility

A trial is an artifact addressed by `name@version`. Its fingerprint covers the model, data, and
training configs and the dataset's version. Calendar versions refuse uncommitted code, and
`uv run --no-sync python -m poorin.experiment.runs show NAME@VERSION` shows the code behind a run.
Batch N is a function of the seed and N, and the flow times and noise of step N come from
`fold_in(key, N)`, so a resumed run trains on what an uninterrupted one would have.

## Adding a family

1. Copy the closest `model.py` into a new directory and edit it. A config must provide `streams`,
   `stream_weights`, `conditions_on_task`, `needs_actions`, `init(dims, clip, key=...)`,
   `tokens_per_example`, and `flops_per_example`; its model must provide
   `stream_losses(batch, key, times)` (the protocols are in `train.py`).
2. Copy a `launch.py` and point it at the new config.
3. Add the family to the `CONFIGS` and `FAMILIES` tables in `tests/test_models.py` and
   `tests/test_train.py`; the FLOP, sampler, and training tests then cover it.
4. Add the new modules to `docs/poorin/architecture.md`.

A family that needs a training step of its own (a second optimizer, a target network) copies
`train.py` into its directory rather than adding options to the shared loop.

## Tests

```bash
JAX_PLATFORMS=cpu uv run --no-sync pytest experiments/episode_models/tests -n 0
```

The tests build a 10-episode, 16x16 dataset in a temporary directory and train every family for a
few steps on the CPU.
