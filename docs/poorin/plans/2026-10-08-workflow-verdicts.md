# Workflow Verdicts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the owner's verdicts (2026-10-08) on the kept workflows that no training check runs:
delete the dataset recipes and the evaluation leftovers of the deleted serving flow, make the
lm-eval harness runnable, give scaling laws tests, and add DPO and LoRA smoke runs.

**Decision:** D88.

## Verdicts

| Workflow | Verdict |
|---|---|
| Data preparation | Keep. Its stages already have tests that run them, and R1 runs download and tokenize; no new test. |
| Evaluation | Keep perplexity and the harness; install the `lm_eval` extra; delete `eval_env`, `model_config`, `model_identity`, and `lm_eval_samples` (its one used helper moves to `eval_measurements`). |
| Scaling laws and Delphi | Keep; add behavior tests for `isoflop_analysis`. |
| Profiling | Keep as is. |
| Post-training and SkyRL hooks | Keep; add end-to-end smoke runs of `train_dpo` and `lora_lm` on tiny models. |
| Dataset recipes | Delete the five recipes; keep `experiments/datasets/` for new dataset modules (the `add-dataset` skill writes there). |
| Optimizers | Keep all. |

## Global Constraints

- Each task is one commit; tasks that change code run `tools.prune_check` in full.
- Deleted modules leave `tools/roots.toml`, `docs/poorin/architecture.md`, and every doc and skill
  that names them in the same commit.
- Smoke runs use CPU and tiny models (seconds, not minutes) and live in Levanter's suite.

## Tasks

### Task 1: Delete the dataset recipes
- [ ] `git rm` `experiments/datasets/{nemotron,starcoder,proofpile,paloma,uncheatable}.py`.
- [ ] Remove their roots and architecture rows; rewrite `docs/tutorials/train-an-lm.md`'s data
  section, which uses `nemotron_datasets`, to tokenize a Hugging Face dataset with
  `poorin.experiment.data.tokenized`.

### Task 2: Delete the evaluation leftovers
- [ ] Move `is_scratch_artifact` from `lm_eval_samples` into `eval_measurements`.
- [ ] `git rm` `eval_env.py`, `model_config.py`, `model_identity.py`, `lm_eval_samples.py`, and
  `tests/evaluation/test_model_config.py`; drop the cases in `tests/evals/test_records.py` that
  only exercise the deleted modules.
- [ ] Roots, architecture rows, docs.

### Task 3: Make the lm-eval harness runnable
- [ ] Sync with `--extra=lm_eval` beside `--extra=gpu`; the lock already resolves the extra.
- [ ] The three harness tests that skipped without `lm_eval` now run and pass.
- [ ] AGENTS.md and `docs/tutorials/local-gpu.md` give the new sync command.

### Task 4: Scaling-law tests
- [ ] `tests/scaling_laws/test_isoflop_analysis.py`: `robust_quad_logx` recovers a known minimum;
  `fit_scaling_laws` recovers a known power law from synthetic isoflop records;
  `predict_optimal_config` picks the candidate nearest the fitted optimum;
  `round_flops_to_bucket` groups nearby budgets.

### Task 5: DPO and LoRA smoke runs
- [ ] `train_dpo.main` on a tiny GPT-2 and a few preference pairs for two steps, ending with a
  saved checkpoint.
- [ ] `lora_lm.main` from a tiny Hugging Face model saved locally, two steps, ending with a PEFT
  export.

### Task 6: Records
- [ ] D88 in `decisions.md`; architecture and AGENTS.md follow the deletions.
