# Evaluation Overview

Poorin keeps Marin's evaluation framework and drops the fleet that ran it: the harness
integration, the durable per-run record, the normalized rollout table, and the statistics that
compare runs all stay. Marin's launcher, which served a model and ran Evalchemy and Harbor suites
against it, is gone. To evaluate a model after training, serve it yourself behind an
OpenAI-compatible endpoint (for example, vLLM) and point the lm-evaluation-harness integration
at it. A local orchestrator that does this for you is planned
(sub-project 7b).

## Evaluation modes

- **In-loop training evals**: [`train_lm`][poorin.experiment.train.train_lm] runs periodic
  multiple-choice evaluations through Levanter's lm-evaluation-harness integration
  (`levanter.eval_harness`, scored from JAX log-probabilities, no server) when an `EvalSuite` is
  provided, and logs to W&B. Levanter also reports validation loss on held-out sets
  (`levanter.eval`); scaling-law fits use it.
- **Post-hoc lm-eval against a served model**: `poorin.evaluation.lm_eval.run_lm_eval` takes a
  `RunningModel` (an OpenAI-compatible endpoint and its tokenizer), runs lm-evaluation-harness in
  an isolated, version-pinned `uv` environment, and persists metrics and samples.
- **Bits per byte across datasets**: `levanter.main.perplexity_gap` scores two models' per-byte
  loss on the same text slices.
- **Labeled loss on agent traces**: `poorin.evaluation.trace_labeled_eval` runs Levanter's loss
  evaluation on OpenAI-style conversation traces.

## Records

One evaluation writes one `record.json` (`poorin.evaluation.records`): what model was evaluated,
on what hardware, whether it succeeded, and the per-task metrics it produced. The record is the
source of truth for a run.

## Evaluations as artifacts

An evaluation is an artifact step like a training run: it is cached, planned, and checked for
drift, and it records the commit that built it. Its input is a finished run, which the evaluation
only reads, and one of the run's permanent checkpoint steps. Its recipe is a suite, a frozen
dataclass whose fields define what is measured. Its output is an
`EvalResult` (`poorin.evaluation.contract`): the run's `name@version`, the step, the suite, the
metrics under the keys training logs, and the number of clips or sequences scored. The design and
its rulings are in the [spec](https://github.com/jasgcode/Poorin/blob/main/docs/poorin/specs/2026-10-13-evaluation-artifacts-design.md)
and [D114](https://github.com/jasgcode/Poorin/blob/main/docs/poorin/decisions.md).

An evaluation's name is `eval/{run name}/{run version}/{suite}/step-{N}`, with the run's
`users/<you>/` namespace removed. A `--set suite.FIELD=VALUE` override joins the suite's segment,
as in `held-out/suite.clips=256`. A `dev` evaluation is namespaced like a dev run. The fingerprint
covers the run's `name@version`, the step, and the suite's class and fields. The device batch size
is a runtime argument and stays out of it. Every evaluation holds the process's accelerator lock,
so an evaluation and a training run never share the GPU in one process.

### Naming a run

`NAME@REF` names a run wherever one is typed. `NAME` may omit the `users/<you>/` namespace.
`REF` is one of:

| `REF` | Names |
|---|---|
| A full version | Itself. |
| A 4-hex token, `60b9` | The dev run whose version ends in `60b9-dev`. Several matches are an error that lists them. |
| `latest` | The newest dev run of `NAME`. |
| A label, `accept` | The newest `accept-<timestamp>-<token>-dev` run. |

Resolution tries an exact version first, then a token, then `latest`, then a label. "Newest"
orders by the timestamp in the version. `evaluate` considers only finished runs. References resolve
at the command line; names, fingerprints, and records hold only full versions.

### Running evaluations

Without `--run`, the command prints the plan: for each run, the commit it trained at, then one line
per step with its state. With `--run`, it builds the cells that are missing.

```bash
# The plan
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@latest \
    --suite held-out --steps all --version dev

# Build it
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@latest \
    --suite held-out --steps all --version dev --run
```

- `--steps` takes comma-separated steps, or `all` for every permanent checkpoint that exists when
  the plan is made. A step with no permanent checkpoint is refused, with the steps that exist.
- `--set suite.FIELD=VALUE` and `--sweep` change the suite's fields. No other field takes `--set`.
- `--batch-size N` sets the device batch size for a suite whose result does not depend on it. It is
  a runtime argument, so changing it does not make a new evaluation.
- Pass several `NAME@REF` arguments to evaluate several runs with the same suite and steps. The
  matrix is a set of artifacts, and a rerun builds only the missing cells.
- A calendar `--version` refuses uncommitted code. A pinned run is never rebuilt, so a calendar
  evaluation of a dev run is allowed.
- The command refuses, before anything runs: an unknown or ambiguous reference, an unfinished run,
  a suite the run's checkpoint type does not offer, and a run whose checkpoint type offers no suites.

### The `held-out` suites

Each template declares a checkpoint type (`EpisodeCheckpoint`, `GrugCheckpoint`) that lists its
suites. Both offer `held-out`, which recomputes the loss the training loop logs, at a permanent
checkpoint, with the loop's own evaluation function. Both raise `FloatingPointError` on a
non-finite metric.

| Suite | Field | Default | Meaning |
|---|---|---|---|
| Episode (WAM, world model, VLA) | `clips` | 128 | Held-out clips scored, spread over the split. |
| | `batch_size` | 32 | Clips per batch. Batch `i` draws its noise from `fold_in(key(seed), i)`, so the batch size is part of the recipe. The suite refuses `--batch-size`. |
| | `flow_times` | 8 | Size of the flow-time grid. |
| | `seed` | 0 | The noise key (`EVAL_SEED`). |
| Grug base | `max_sequences` | `None` | Sequences scored per validation set; `None` scores each whole set. |
| | `weights` | `ema` | `ema` or `current`. `ema` uses the current weights for a run without EMA. |

The episode suite's defaults are training's, so with them its metrics equal the `eval/*` values
the run logged. This holds for a run that kept training's default evaluation batches
(`train.eval_batches=4`, `train.eval_batch_size=32`). For a run that changed them, pass
`--set suite.clips=... --set suite.batch_size=...` to match. The suite does not read these
settings from the run's record.

Grug scores the whole validation set by default, and training caps each set at
`max_eval_batches * eval_batch_size` sequences. To reproduce a grug base trial's logged value, which
is 8 batches of 16, pass `--set suite.max_sequences=128`. The base trial evaluates on WikiText-2's
validation split once, at its final step. Grug reports the keys training logs without the timing
keys: `eval/loss`, `eval/bpb`, and `eval/<set>/loss` and `eval/<set>/bpb` per validation set, with
`eval/macro_*` when there are several sets. EMA weights report under `eval/ema/`.

A run's logged evaluation at tracker step `N - 1` is the checkpoint `step-N`: both loops log at the
index of the step just taken and checkpoint at the count of steps taken.

### Runs trained before evaluation artifacts

Earlier runs record `LevanterCheckpoint`, which does not say which template made them. Name the
checkpoint type once:

```bash
uv run --no-sync python -m poorin.experiment.runs migrate NAME@REF --to MODULE.CLASS [--drop PATH ...]
```

This rewrites `result_type` in the run's `.artifact.json`. The fingerprint stays as recorded.
`--drop PATH` removes a recorded config field that the code has since moved out of the recipe, for
example `--drop data.num_workers`. The command refuses a `--to` that is not an importable
`Artifact` class or offers no suites, a run that does not record `LevanterCheckpoint`, and a
`--drop` path the record lacks.

### Reading results

`eval_results` scans the evaluation records under a prefix. Filter by a run's full `name@version`
or by suite name:

```python
from rigging.filesystem.cluster_config import poorin_prefix

from poorin.experiment.evaluate import eval_results

for result in eval_results(poorin_prefix(), suite="held-out"):
    print(result.run, result.step, result.metrics["eval/loss"])
```

Training curves stay in each run's `tracker_metrics.jsonl`.

## Normalized rollout table

`poorin.evaluation.rollouts` builds the versioned `rollouts_v1` table from normalized sample and
step rows. It stores one ordered conversation part per row, keyed by
`(task, doc_id, trial_id, turn_id, part_id)`: `task`, `doc_id`, and `trial_id` identify one model
attempt, `turn_id` orders turns, and `part_id` orders content within a turn.

- `conversation_type` is `chat` for structured prompt messages, `completion` for a prompt string,
  and `agentic` for multi-step trajectories.
- `participant_type` is the normalized role: `system`, `user`, `assistant`, `tool`, `environment`,
  or `other`.
- `content_type` is `message`, `reasoning`, `tool_call`, or `tool_result`. Structured tool content
  is JSON in `content`; `metadata_json` holds small type-specific annotations.

## Statistics

A benchmark score is a mean over a finite item set, and a run does not always grade every item.
`poorin.evaluation.eval_stats` carries both facts as one `Measurement` and derives intervals on
read: Wilson intervals for binary scores, bounds that account for ungraded items, and intervals
for the difference between two runs. `poorin.evaluation.eval_measurements` turns records into
measurements, and `poorin.evaluation.metric_selection` picks the metric and extraction filter
presented as primary for each lm-eval task.

## Task sets

Task sets are defined in `experiments/evals/task_configs.py` as `EvalTaskConfig` tuples:
`CORE_TASKS` is the default for in-loop multiple-choice evals, `CORE_TASKS_PLUS_MMLU` adds MMLU,
and the file also holds the Open LLM Leaderboard, medical, and generation task lists. The in-loop
`EvalSuite` and post-hoc lm-eval runs draw from the same lists.

## In-loop metrics

Beyond task accuracy, the in-loop Levanter evaluator tracks these multiple-choice metrics:

1. **Bits per byte (`bpb`)**: `bpb = -log_prob / byte_length * ln(2)`
2. **Log probability (`logprob`)**: raw log probability of the correct answer.
3. **Choice log probability (`choice_logprob`)**: `log_prob_correct - log(sum(exp(log_prob_i)))`
4. **Length-normalized choice probability (`choice_prob_norm`)**:
   `exp(log_prob_correct / (byte_length_correct * ln(2))) / sum(exp(log_prob_i / (byte_length_i * ln(2))))`
