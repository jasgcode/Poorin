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
source of truth for a run. `poorin.evaluation.model_config` validates the model description a
record carries, and `poorin.evaluation.model_identity` gives two configurations of the same model
one comparison identity.

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
