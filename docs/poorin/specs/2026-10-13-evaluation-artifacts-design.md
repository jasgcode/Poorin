# Evaluation artifacts

Status: design for review, 2026-10-13. Sub-project 7a, first round. Decision D114.

## TL;DR

An evaluation becomes an artifact like a training run. Its inputs are a finished run and one of
its checkpoint steps; its recipe is a *suite*, a frozen dataclass whose fields define what is
measured; its output is an `EvalResult` with the metrics. The experiment layer then gives
evaluations what it gives training: a plan that shows which results exist, caching, `--set` and
`--sweep`, drift refusal, and code provenance. A checkpoint-by-benchmark matrix is a set of such
artifacts, and rerunning it builds only the missing cells.

One command runs it:

```bash
uv run --no-sync python -m poorin.experiment.evaluate \
    episode/wam-libero-spatial@accept episode/vla-libero-spatial@accept \
    --suite held-out --steps all --version 2026.10.13 --run
```

This round defines the contract for every kind of evaluation and implements one cheap,
deterministic suite, a post-hoc held-out loss, for both families: the episode templates (WAM,
world model, VLA) and grug. Benchmarks (lm-eval, LIBERO success rates, video metrics) plug into
the same contract later. The round is done when the post-hoc held-out loss of a finished run
reproduces the held-out loss the run logged during training, on the RTX 5090, for all four
templates.

## Background

### What exists

- **Artifacts.** `poorin.execution.lazy` builds lazy `ArtifactStep` handles; `apply()` makes a
  step from a function and keyword inputs, and `pin=` points a handle at existing data without
  building it. A handle's fingerprint hashes its config and its inputs' `name@version`; values
  pulled through `ctx.runtime_arg` stay out. Each finished output has an `.artifact.json` record
  holding the fingerprint, the full config with each dataclass's class name, the inputs, the
  `result_type`, the result, and code provenance (D107).
- **The experiment CLI** (`poorin.experiment.cli`) gives a launch function `--version`,
  `--version-of`, `--set`, `--sweep`, `--resume`, `--note`, the code check, drift refusal, and the
  plan printout. A dev version gets a unique suffix: `--version accept-dev` becomes
  `accept-20261012-201822-60b9-dev`, a label, a timestamp, and a 4-hex token.
- **Training outputs.** A grug or episode run writes permanent checkpoints to
  `{output}/checkpoints/step-N` and mirrors its tracker metrics to `{output}/tracker_metrics.jsonl`.
  Both record `result_type: poorin.training.training.LevanterCheckpoint`, which does not say which
  template made the run.
- **Evaluation during training.** The episode templates compute `eval/loss` and
  `eval/<stream>_loss` on a fixed set of held-out clips with a fixed flow-time grid and noise key
  (D113). Grug computes `eval/<tag>/loss` and `bpb` on each validation set through
  `build_tagged_evaluator`, with EMA or current weights.
- **Evaluation after training.** None for Poorin's own templates. Levanter's `eval_harness` loads
  only Levanter's Llama, Qwen, and GPT-2 models. `poorin.evaluation.lm_eval.run_lm_eval` runs
  lm-evaluation-harness against an OpenAI-compatible endpoint, and grug has no Hugging Face export
  to serve. `LmEvalResults(Artifact)` exists with no handle factory: Marin's
  `marin/experiment/evaluation.py` went with fleet evaluation (D10).
- **Marin's record.** `poorin.evaluation.records.EvalRunRecord` is Marin's fleet record, with its
  own provenance, serving, hardware, and Harbor and Evalchemy fields. `eval_stats` and
  `eval_measurements` read it.
- **Scaling laws.** `poorin.scaling_laws.eval_metrics_reader` reads `tracker_metrics.jsonl` and
  backfills from W&B when it is missing.
- **Exclusivity.** `poorin.training.training.exclusive_training` lets one training run at a time in
  a process (D98).

### Choices made while designing (2026-10-13)

- The first round serves both purposes: the contract covers scaling sweeps and downstream
  benchmarks, and only the held-out loss is implemented now.
- An evaluation is an ordinary artifact step. Running evaluations inside training would put the
  evaluation config in the training identity; records outside the artifact graph (Marin's design)
  lose caching and drift refusal and keep two provenance systems.
- The everyday interface is a command, with Python handles underneath, as training has
  `launch.py` over its launch function.
- Runs are named by short references; nobody types `20261012-201822-60b9`.
- Grug is in the first round, so the contract is shown to span both families.

## Design

### Layout

```text
lib/poorin/src/poorin/
  execution/recorded_config.py  # decode a recorded config value back into its dataclass
  evaluation/contract.py        # EvaluableCheckpoint, Suite, FinishedRun, EvalResult
  experiment/run_refs.py        # short run references -> full versions
  experiment/evaluate.py        # Checkpoint, finished_run, evaluate, eval_results, the CLI
experiments/datasets/
  wikitext.py                   # WikiText-2, grug base's held-out set
experiments/episode_models/
  evaluation.py                 # EpisodeCheckpoint, its loader, HeldOutLoss
experiments/grug/base/
  evaluation.py                 # GrugCheckpoint, its loader, HeldOutLoss
```

`poorin` defines the protocols and never imports a template. A template implements them next to
its model code. The dependency direction is unchanged.

### Short run references

Wherever a run is typed (`evaluate`, `--resume`, `runs show`), `NAME@REF` resolves `REF` against
the versions of `NAME` under the prefix, in the user's namespace for dev runs:

| `REF` | Resolves to |
|---|---|
| A full version | Itself |
| A 4-hex token, `60b9` | The run whose suffix carries that token; refused, listing the candidates, if several do |
| A label, `accept` | The latest run whose version is `accept-<timestamp>-<token>-dev` |
| `latest` | The latest dev run of `NAME` |

"Latest" orders by the timestamp in the version. `evaluate` considers only finished runs;
`--resume` considers every run. `--resume` takes no `NAME`, so it accepts a full version or a token,
resolved against every dev run under the prefix; a label or `latest` needs a `NAME`. References resolve once, at the CLI boundary. The plan prints the
full version, and names, fingerprints, and records store only full versions.

### Handles and identity

```python
from poorin.experiment.evaluate import Checkpoint, evaluate, finished_run
from experiments.episode_models.evaluation import HeldOutLoss

wam = finished_run("episode/wam-libero-spatial@accept")
cells = [
    evaluate(Checkpoint(wam, step), HeldOutLoss(clips=512, flow_times=8, seed=0), version="2026.10.13")
    for step in (500, 1000, 1500, 2000)
]
```

- `finished_run(ref)` resolves the reference, reads the run's record, and returns a pinned handle
  whose `artifact_type` is the record's `result_type`. It never builds or retrains. It refuses a
  run whose `.executor_status` is not `SUCCESS`.
- A training handle from the same graph is also a valid `Checkpoint.run`. It follows the existing
  rule: a calendar evaluation of a dev training handle is refused.
- **Calendar evaluations of pinned dev runs are allowed.** `ArtifactStep.__post_init__` refuses a
  fixed version over any mutable dep; it now skips pinned deps. A pinned dev run never rebuilds,
  and its permanent checkpoints do not change when it is resumed. A calendar evaluation over an
  unpinned dev handle is still refused.
- **Pinned steps never rebuild.** The step runner rebuilds every dev version, a pinned one
  included, so a pinned dev run would be handed to the runner as a build. `StepSpec.rebuilds` is
  false for a pinned step, which is served from its `SUCCESS` status like any built output.
- `evaluate(checkpoint, suite, *, version)` builds an `ArtifactStep` directly (`apply()` takes no
  runtime args):
    - Name: `eval/{run name}/{run version}/{suite name}/step-{N}`, with the run's `users/<user>/`
      namespace stripped from the run name. A dev evaluation is namespaced like a dev run.
    - Fingerprint: the run's `name@version`, the step, and the suite's class and fields. The
      dataset version and every training setting come from the run's own config.
    - Runtime args, outside the fingerprint: the device batch size when the result does not
      depend on it (grug's suite). The episode suite's batch size is a recipe field, because its
      noise is drawn per batch, and that suite refuses a device batch size. Neither suite reads
      loader workers, so `EvalRuntime` holds only the batch size.
    - Only the suite's fields can be overridden: `--set suite.FIELD`. An override joins the suite's
      segment of the name, `eval/{run}/{version}/held-out/suite.clips=256/step-N`, as a training
      trial's does.
- `eval_results(prefix, *, run=None, suite=None) -> list[EvalResult]` scans evaluation records
  and filters them, for sub-project 5 and notebooks.

### The contract

```python
class FinishedRun(NamedTuple):
    label: str                 # full name@version
    path: str                  # the run's output directory
    record: ArtifactRecord     # config, deps, provenance


class EvalResult(Artifact):
    run: str                   # full label of the evaluated run
    step: int
    suite: str
    metrics: dict[str, float]  # the keys the run logs in training: "eval/loss", "eval/frames_loss", ...
    examples: int              # clips or sequences scored
    samples_path: str | None   # a per-example file, for suites that keep one


class Suite(Protocol):
    """A frozen dataclass. Its fields are the recipe; a change to one is a new evaluation."""

    name: ClassVar[str]

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult: ...


class EvaluableCheckpoint(Protocol):
    """The artifact type a template's training run records."""

    @classmethod
    def suites(cls) -> Mapping[str, type[Suite]]: ...
```

`EvalRuntime` carries the device batch size for suites whose result does not depend on it. Each template declares
its checkpoint type as an `Artifact` subclass that implements `EvaluableCheckpoint`, and its launch
function passes it as `artifact_type`:

- `experiments.episode_models.evaluation.EpisodeCheckpoint`, with `{"held-out": HeldOutLoss}`
- `experiments.grug.base.evaluation.GrugCheckpoint`, with `{"held-out": HeldOutLoss}`

The CLI imports the class a run's record names and asks it for its suites. A template decodes its
own config from `FinishedRun.record` and loads its own checkpoint; the loader is private to the
template.

**Reading a run's recipe.** A record's `config` is the fingerprint's canonical JSON: each dataclass
carries its class under `__class__`, as its draccus choice name for a registered choice (`adam`,
`cached`) and its class name otherwise. `poorin.execution.recorded_config.decode_recorded(cls,
value)` reads a value back, walking `cls`'s field types so a choice gets draccus's `type` key only
where the declared type is a choice base. A template decodes the parts of the config it needs
(`model`, `data`), not the whole launch config: draccus cannot decode some recorded fields, such
as fray's `ResourceConfig.regions` (`Sequence[str]`). A recorded field the current code no longer
has is refused with the command that drops it (see Existing runs).

**Model-kind interfaces.** Generic benchmark suites will call a model through one interface per
model kind, which a template's loader returns. Their shape is fixed here; no code is written for
them in this round.

| Interface | Methods | First suite that uses it |
|---|---|---|
| `LanguageModel` | `loglikelihood(contexts, continuations)`, `generate(prompts, max_tokens)` | lm-eval, scored from JAX log-probabilities |
| `Policy` | `act(observation, task) -> action chunk` | LIBERO success rate |
| `WorldModel` | `predict(history_frames, actions, horizon) -> frames` | Video metrics on held-out clips |

A WAM's loader can return both a `Policy` and a `WorldModel`.

### Held-out loss suites

**Episode templates**, `experiments.episode_models.evaluation.HeldOutLoss`:

| Field | Default | Meaning |
|---|---|---|
| `clips` | 128 | Held-out clips scored, spread evenly over the held-out split (D113) |
| `batch_size` | 32 | Clips per batch. It is part of the recipe: batch `i` draws its noise from `fold_in(key(seed), i)`, so a different batch size gives each clip different noise |
| `flow_times` | 8 | Size of the fixed flow-time grid |
| `seed` | 0 | The noise key (`EVAL_SEED` in training) |

The split, clip shape, fields, task vocabulary, and action statistics come from the run's config.
The suite rebuilds the model from the config's `model` field, restores the parameters of
`checkpoints/step-N`, and calls the same evaluation function as training. The defaults equal the
training defaults (`eval_batches = 4`, `eval_batch_size = 32`, eight flow times, `EVAL_SEED = 0`),
so with them the suite measures what training logged. A run trained with other evaluation
settings is reproduced by passing them: `--set suite.clips=256 --set suite.batch_size=64`.

**Grug**, `experiments.grug.base.evaluation.HeldOutLoss`:

| Field | Default | Meaning |
|---|---|---|
| `max_sequences` | `None` | Cap on sequences per validation set; `None` scores the whole set |
| `weights` | `ema` | `ema` or `current`; `ema` falls back to current weights when the run kept no EMA |

It restores `checkpoints/step-N`, builds the evaluator from the run's data config through the same
function as training, and reports the keys training logs, without the timing keys: `eval/loss` and
`eval/bpb`, `eval/<set>/loss` and `eval/<set>/bpb` per validation set, and `eval/macro_loss` and
`eval/macro_bpb` when there are several sets. EMA weights report under `eval/ema/` instead of
`eval/`, as training logs them. The default `max_sequences` scores the whole set; training caps
each set at `max_eval_batches * eval_batch_size` sequences, so reproducing a run passes that cap.

**Grug base's held-out set.** Grug base trains on FineWeb-Edu 10M, whose cache has no validation
split, and its trial disabled evaluation. The trial gains WikiText-2's validation split as a
mixture validation component (weight 0) through `experiments.datasets.wikitext`, the same artifact
the tutorial builds, and evaluates once, at its final step (`eval_batch_size = 16`,
`max_eval_batches = 8`, up to 128 sequences of 1,024 tokens; the split has 92, all scored). The training loop forces its eval
callback on the last step, so no interval changes.

Both suites raise `FloatingPointError` on a non-finite loss.

### The command

```bash
uv run --no-sync python -m poorin.experiment.evaluate RUN@REF [RUN@REF ...] \
    --suite NAME --steps 500,1000|all --version VERSION \
    [--set suite.FIELD=VALUE] [--sweep 'suite.FIELD=[...]'] [--note TEXT] [--run]
```

- It reuses the experiment CLI's options and plan: one line per run, step, and trial, with its
  state. `--set` and `--sweep` address the suite's fields.
- `--steps all` expands, when the plan is made, to the permanent checkpoints that exist then.
- Refused before anything runs, each with a message that names the fix: an unknown or ambiguous
  reference; an unfinished run; a step with no permanent checkpoint, listing the steps that exist;
  a suite the run's checkpoint type does not offer, listing those it does; a record whose
  `result_type` is not an `EvaluableCheckpoint`.
- The plan prints, for each run, the commit it trained at and the commit evaluating it. The current
  code loads the old checkpoint; the reproduction test below catches a model-code change that
  silently changes results.
- A calendar evaluation refuses uncommitted code, and a dev evaluation saves a snapshot commit
  (D107).
- Each evaluation step holds the exclusivity lock, renamed `exclusive_accelerator`, because it now
  guards evaluations as well as training: two evaluations, or an evaluation and a training, never
  share the GPU in one process. Memory held by other programs is outside the lock and still fails
  loudly (D113's smoke OOM).

### Existing runs

Runs trained before this round record `LevanterCheckpoint`. The CLI refuses them with the
migration command, a one-time rewrite of `"result_type"` in that run's `.artifact.json` to the
template's checkpoint type, as D85's rename did:

```bash
uv run --no-sync python -m poorin.experiment.runs migrate NAME@REF --to MODULE.CLASS [--drop PATH ...]
```

`--drop` removes a recorded config field the current code moved out of the recipe. D113's
acceptance runs trained at `a25be99ae`, before `e2dddfa87` moved the loader workers out of
`EpisodeDataConfig` into a runtime arg, so their records hold `data.num_workers`, which the current
`EpisodeDataConfig` refuses. Migrating them drops it. The fingerprint and its payload stay as
recorded: they describe what ran. The three acceptance runs are migrated as part of acceptance.

### What reads the results

Sub-project 5 reads evaluation points with `eval_results`. The training curves in
`tracker_metrics.jsonl` stay as they are; changing `eval_metrics_reader` is sub-project 5's work.
Marin's `EvalRunRecord`, `eval_stats`, and `eval_measurements` stay, unused by this path, until
lm-eval moves onto the contract.

## Testing and acceptance

Unit tests, on CPU:

- Reference resolution: full version, token, ambiguous token, label (latest of several), `latest`,
  unfinished runs skipped by `evaluate`.
- Naming: `evaluate` derives the documented name, strips the user namespace, and gives two
  versions of one run different names.
- Identity: a suite field changes the fingerprint, and so does the step; grug's device batch size
  does not.
- The mutable-dep rule: a calendar evaluation over a pinned dev run builds; over an unpinned dev
  handle it is refused.
- Each refusal in the command section, through the CLI, and a `--set` outside the suite.
- Decoding: a recorded config value decodes back to the dataclass that wrote it, choices
  included; a field the class no longer has is refused, naming the migration.
- Pinning: evaluating a pinned dev run leaves its status and record untouched.
- Grug's held-out loss does not depend on the device batch size.
- `EvalResult` round-trips through the record.
- Reproduction, per template: train a tiny run for a few steps on the existing test fixtures
  (the 10-episode dataset for the three episode families, the tiny grug configuration), then
  evaluate the last checkpoint post-hoc with the training settings and assert the result equals
  the logged `eval/*` values to float tolerance.

GPU acceptance on the RTX 5090:

- Migrate the three D113 acceptance runs and reproduce their logged held-out losses at step 2000
  (WAM 0.741, world model 0.580, VLA 0.222).
- Train a short grug run and reproduce its logged `eval/*/loss`.
- Run one matrix: two runs, `--steps all`, then the same command again, which builds nothing.

Documentation: `docs/explanations/evaluation.md` gains a section on evaluation artifacts and the
command; the README's "Run something" gains one example; the decision is D114.

## Out of scope

- The lm-eval, LIBERO, and video suites, and code for the model-kind interfaces.
- Serving a model behind an endpoint.
- An `--eval` option on the template launchers that evaluates after training.
- Reading evaluation results in `poorin.scaling_laws` (sub-project 5).
- Deleting Marin's `EvalRunRecord` and its readers.
- Logging evaluation results to W&B; the record is the output.
