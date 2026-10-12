# Episode model templates

Status: approved design, 2026-10-11 (plan `plans/2026-10-11-episode-model-templates.md`).
Sub-project 4, first round. Decision D113.

## TL;DR

Sub-project 3 gave Poorin episode datasets: video and robot trajectories read as Grain clips.
Nothing trains on them yet. This round adds three thin model templates that do, one per model
family besides the language model:

- `wam/`: a world action model. It denoises future frames and future actions together, each with
  its own flow time, conditioned on clean history frames and a task.
- `world_model/`: the WAM with the action stream deleted. It denoises future frames.
- `vla/`: the WAM with future frames deleted. It denoises an action chunk from history frames
  and a task.

All three use flow matching, train from scratch, and share one training loop, one data path, one
objective module, and one FLOP count. Each family's code is a `model.py` (config, model, loss,
sampler) and a `launch.py` (defaults and the experiment-CLI entry point). A run is a lazy
artifact like grug's: `--version dev --run` trains it, `--set` and `--sweep` change any config
field, and the run's code, config, data order, and noise are reproducible from its version.

The round is done when each family trains on the RTX 5090 from an example dataset, resumes from
a checkpoint, and runs a three-size sweep that logs parameter count, FLOPs, and a held-out loss
computed the same way in every family. That is the contract sub-project 5 fits scaling curves on.

## Background

### What exists

- Grug base (`experiments/grug/base/`) is the language-model template: a Llama-style trunk in raw
  JAX arrays, Levanter's tracker, checkpointer, and optimizers, and a launch config that the
  experiment CLI versions, overrides, and sweeps. Its contract tests
  (`tests/test_grug_variant_contracts.py`) discover every directory under `experiments/grug/` with a
  `model.py` and `train.py` and require the language-model interface (`debug_mesh_and_token_pspec`,
  `GrugLmExample` batches).
- The episode layer (`poorin.episodes`, D110) reads an `EpisodeDataset` as clips:
  `clip_dataset(dataset, clip=ClipSpec(...), fields=...)`, `batch_iterator(..., start_step=...)`,
  and `to_mesh`. The batch at step N is a pure function of (seed, N).
- Example datasets (`experiments/datasets/episodes/`): LIBERO-Spatial (two 128 px cameras,
  7-dim actions, proprioception, an instruction per episode, 20 Hz), SSv2 (128 px video, 12 Hz),
  and VPT (128 px video, 23-dim actions, 20 Hz).

### Choices made while designing (2026-10-11)

| Question | Choice |
|---|---|
| Which family first | All three thin, so the shared contract is settled before any one goes deep |
| What "done" means | Trains end to end, and is sweep-ready: size knobs, FLOP accounting, one held-out loss |
| Objective | Flow matching for all three; the language model keeps cross-entropy |
| Pretrained components | Ported into grug-style modules from safetensors, not run through a Flax bridge (amends D13) |
| Structure | WAM first; the world model and the VLA are copies of its model with a stream removed |
| Shared code | One training loop, data path, objective, and FLOP count; the model is the unit a family owns |

The owner asked for templates that are modular, reproducible, and flexible: a family should be a
small set of files that plug into shared machinery, and any run should be rebuildable from its
name and version.

## Design

### Layout

```
experiments/episode_models/
  README.md          how to run a template, sweep it, and add a family
  data.py            clip and split config, EpisodeBatch, action normalization, the batch iterator
  sources.py         the example datasets a launcher can name with --dataset
  flow.py            flow-matching path, time sampling, the evaluation time grid, Euler sampling
  layers.py          trunk block, rotary embedding by position, patch and time embeddings, masks
  flops.py           analytic FLOPs per example
  train.py           the training loop, shared by every family
  launch.py          builds a family's trial as an ArtifactStep
  wam/model.py, wam/launch.py
  world_model/model.py, world_model/launch.py
  vla/model.py, vla/launch.py
```

The templates live outside `experiments/grug/` because grug's contract tests assume a language
model in every variant directory there. They follow grug's principles (raw JAX arrays, explicit
`init` and `__call__`, einsum, small configs) and reuse grug's checkpoint restore
(`experiments.grug.checkpointing`) and Fray dispatch (`experiments.grug.dispatch`).

A family's `model.py` is the file a researcher copies and edits. The shared modules are the
contract every family meets, so sub-project 5 can sweep any of them without touching its code.

### Data and batch contract

**Clips.** A clip is `history` steps followed by `horizon` steps: `ClipConfig(history=4,
horizon=8, stride=4, skip=1)`, read with `ClipSpec(length=history + horizon, stride, skip)`. The
episode layer's step clock aligns fields: `action[t]` is the action taken after observing
`frame[t]`.

**Sources.** `sources.py` maps a name to a dataset handle and the fields a template reads:

| Name | Handle | Video | Action | Task |
|---|---|---|---|---|
| `libero-spatial` | `libero_spatial()` | `agentview` | `action` | `instruction` |
| `vpt-10` | `vpt_10()` | `video` | `action` | none |
| `ssv2-500` | `ssv2_500()` | `video` | none | none |

A launcher's `--dataset NAME` picks the source; the handle becomes a dependency of the run, so
the dataset's name and version enter the run's identity. A family refuses a source that lacks a
field it needs (the VLA needs an action field; the WAM needs one too).

**Batch.** `EpisodeBatch` holds what reaches the device:

- `frames: uint8[B, history + horizon, H, W, 3]`. Frames cross to the device as `uint8` and become
  `[-1, 1]` in the compute dtype inside the jitted step.
- `actions: float32[B, history + horizon, A] | None`, normalized.
- `task: int32[B] | None`, the index of the episode's instruction in the dataset's sorted list of
  distinct instructions.

**Data-derived sizes.** `EpisodeDims(frame_shape, action_dim, num_tasks)` comes from the dataset's
schema and index at run time. A model's `init` takes its config and these dims, so the config
holds only choices and the dataset decides its own sizes.

**Action normalization.** Each action dimension maps to `[-1, 1]` by its 1st and 99th percentiles,
then clips. An `action_stats` step (`apply(...)` over the dataset handle) decodes only the action
field of every episode and writes `action_stats.json`. Its version is the dataset's version, so it
is computed once per dataset version and cached.

**Held-out split.** An episode is held out when a stable hash of its id (BLAKE2b, not Python's
salted `hash`) modulo 100 falls below `eval_percent` (default 5). Splitting by episode keeps eval
clips from sharing frames with training clips. This needs one library change:
`clip_dataset(..., rows=...)` restricts clips to the given episode rows, because Grain's
`MapDataset.filter` leaves holes that break batching.

**Order and noise.** Training clips are shuffled with the data seed and repeated; resume starts
`batch_iterator` at the restored step. Flow times and noise are drawn in JAX from
`fold_in(noise_key, step)`, never in Grain's workers. A resumed run sees the batches and the noise
an uninterrupted run would have.

### Model and token layout

**Trunk.** Grug base's block, copied: RMSNorm, attention, RMSNorm, ReLU MLP, each with a residual.
Two size knobs, `hidden_dim` and `num_layers`; `head_dim` is 64, so `num_heads = hidden_dim / 64`,
and the MLP is `4 * hidden_dim` wide. The thin default is `hidden_dim=384, num_layers=6`. Parameters
are float32 and replicated; the batch is sharded over the `data` axis. Model sharding waits for
sub-project 2.

**Tokens (WAM).** Per timestep, 64 frame tokens (16x16x3 patches of the 128 px frame, a linear map
to `hidden_dim`, plus a learned embedding per patch position) and one action token (a linear map
of the action). One task token from a learned embedding of the task index leads the sequence when
the source has tasks. With `history=4, horizon=8`: 1 + 12 x 65 = 781 tokens.

**Positions and the mask.** Each token carries a `position` (its timestep, for rotary embedding)
and a `block` (for the mask). A token sees every token whose block is at most its own. In the WAM
both equal the timestep, so the mask is block-causal by timestep and bidirectional within one. The
task token has position and block 0 and timesteps start at 1, so every token sees it. All tokens
of a timestep share a rotary position; space comes from the learned patch embedding.

Grug's `AttentionMask` supports causal, sliding-window, and segment masks, and FlashAttention 4
takes only those. A block mask needs a per-query upper bound, so the templates pass a dense
boolean mask, which grug's `attention()` runs through `reference_attention`. At about 800 tokens
this costs little; a block-causal FA4 path is later work. Grug's `apply_rotary_embedding` takes
positions `0..seq_len-1`, so `layers.py` has a rotary embedding that takes explicit positions.

**Flow time.** Each sample draws one flow time per stream it denoises (`video`, `action`). History
tokens are clean. A stream's tokens add an embedding of their flow time (sinusoidal features, then
a two-layer MLP); history tokens carry the embedding of `t = 1`.

**Heads.** A linear map from each future frame token to its patch's velocity (768 values) and
from each future action token to the action's velocity.

**World model.** The WAM without action tokens, task token, or action head: (history + horizon)
x 64 = 768 tokens, one stream (`video`).

**VLA.** The WAM without future frames or frame head. The sequence is the task token, `history`
x 64 history frame tokens (block and position = their timestep), and `horizon` action tokens
(block = `history + 1`, so they see all history and each other; position = their timestep). One
stream (`action`). This is π0's pattern of a noisy action block attending to a clean observation
prefix, without a separate action expert.

### Objective and evaluation

`flow.py` uses the rectified-flow path with `t = 1` clean: `x_t = t * x + (1 - t) * noise`, and
the model predicts the velocity `x - noise`. Training draws each stream's `t` from a logit-normal
distribution (mean 0, scale 1). A stream's loss is the mean squared velocity error over its future
tokens. The training loss is the sum over streams, each scaled by a config weight (default 1).

Each model exposes `stream_losses(batch, key, times)`, which takes the flow time of every stream
explicitly. Training calls it with sampled times. Evaluation calls it once per time on a fixed grid
of eight (`(i + 0.5) / 8`), with noise from a fixed evaluation key, over the first
`eval_batches` batches of the held-out split in index order. The held-out loss is the mean over the
grid. It is deterministic for given parameters, so it can be the y-axis of a scaling fit. The
loop logs `eval/loss` and `eval/<stream>_loss`.

Each model also has a `sample(batch, key, num_steps)` that integrates the velocity with Euler steps
from noise to the predicted future of its streams. The thin round uses it only in tests; rollout
metrics belong to sub-project 7b.

### Training loop

`train.py` runs any family whose model meets a small protocol:

```python
class EpisodeModelConfig(Protocol):
    streams: tuple[str, ...]
    def init(self, dims: EpisodeDims, *, key: PRNGKeyArray) -> "EpisodeModel": ...
    def tokens_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> int: ...
    def flops_per_example(self, clip: ClipConfig, dims: EpisodeDims) -> float: ...

class EpisodeModel(Protocol):
    def stream_losses(self, batch: EpisodeBatch, key: PRNGKeyArray, times: dict[str, Array]) -> dict[str, Array]: ...
```

The loop follows grug base's: Levanter's `TrainerConfig` initializes the tracker and checkpointer,
the optimizer comes from an `OptimizerConfig`, a `jmp` policy casts parameters for compute, the
train state is `(step, params, opt_state, ema_params)`, and grug's
`restore_grug_state_from_checkpoint` resumes it. The step is one jitted function. It logs
`train/loss`, `train/<stream>_loss`, Levanter's throughput stats with the analytic FLOPs, and the
parameter count. Training runs through grug's Fray dispatch, so it holds the process-wide training
lock like every other run.

### Launch and configuration

A family's launch config composes three shared sub-configs and its model config:

```python
@dataclass(frozen=True)
class WamLaunchConfig:
    model: WamConfig
    data: EpisodeDataConfig     # source fields, clip, eval_percent, workers, data seed
    train: EpisodeTrainConfig   # steps, batch size, seed, precision, optimizer, eval cadence
    run: EpisodeRunConfig       # output path, run id, resources, tracker, Finelog URL (pulled from the context)
```

So `--set model.hidden_dim=512`, `--set train.steps=2000`, and `--set data.clip.horizon=16` all
work, and `--sweep 'model.hidden_dim=[128, 256, 384]'` runs a size sweep. The shared
`launch.episode_trial(...)` builds the `ArtifactStep[LevanterCheckpoint]`: its dependencies are the
dataset and its action stats, its name is `episode/<family>-<dataset>` with overrides appended (`episode/wam-libero-spatial`), and its
runtime arguments are the GPU, run id, tracker, and Finelog URL. Calendar versions refuse
uncommitted code (D107), and `runs show` prints the code behind any run.

### FLOPs

`flops.py` counts forward FLOPs per example: the trunk per token from Levanter's
`lm_flops_per_token` (no vocabulary, `glu=False`, the run's sequence length) times the token count,
plus the patch embedding, frame head, action embedding, action head, and time MLPs. Training FLOPs
are three times that. A test checks the count against XLA's cost analysis of the compiled forward
pass.

## Testing and acceptance

Unit tests run on CPU under `tests/episode_models/`, on a tiny episode dataset built in the test:

- Flow matching: the path's endpoints, the velocity target, and the evaluation grid.
- Masks: a token never sees a later block; tokens of one block see each other; the VLA's action
  tokens see every history token.
- Data: the split is stable across processes and disjoint; normalization maps the 1st and 99th
  percentiles to -1 and 1; `clip_dataset(rows=...)` yields only clips of the given rows.
- FLOPs: the analytic forward count is within 10% of XLA's cost analysis, for each family.
- Training: each family trains a few steps from its launch config; a run resumed at step k
  reports the same losses after k as an uninterrupted run; the held-out loss is identical across
  two calls.
- Sampling: each family's sampler returns arrays of the future's shape, deterministic in its key.

Acceptance on the RTX 5090, recorded in D113:

1. `world_model/launch.py --dataset ssv2-500`, `wam/launch.py --dataset libero-spatial`, and
   `vla/launch.py --dataset libero-spatial`, each with `--version dev --run`: the held-out loss
   falls over training, and a run resumed from its checkpoint continues it.
2. Each family with `--sweep 'model.hidden_dim=[128, 256, 384]'`: every trial finishes and logs
   `parameter_count`, `throughput/flops_per_example`, and `eval/loss`.

## Out of scope

| Item | Where |
|---|---|
| Pretrained components (SigLIP, Gemma, π0) ported from safetensors | A later round of sub-project 4 |
| Proprioception, a second camera, language encoders | A later round of sub-project 4 |
| Tokenizer latents instead of pixels | A later round; the episode layer's encodings already allow a latent field |
| adaLN conditioning, 3D rotary embedding, diffusion forcing with per-frame times | Variants |
| A block-causal FlashAttention 4 path | Performance work, when sequences grow |
| Rollout metrics, simulator success | Sub-project 7b |
| Model sharding across GPUs | Sub-project 2 |
| VPT's long-episode read speed | D110 Deferred |
