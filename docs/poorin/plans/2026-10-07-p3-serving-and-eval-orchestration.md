# P3 LLM Serving and Evaluation Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete Marin's LLM serving stack (`marin.inference`), its evaluation fleet
(orchestration, published-cohort policy, launchers), the Harbor and Evalchemy integrations,
Verifyit, `marin.mcp`, and the machinery that pinned Marin's forked dependencies, while the kept
evaluation framework (lm-eval integration, records, rollouts, statistics, model configuration)
keeps working against any served OpenAI endpoint.

**Architecture:** Outside-in deletion in seven commits, ordered by the import graph among the
targets: the launchers import everything; Harbor imports the fleet modules; Evalchemy and the
fleet modules import each other, so they go together; Verifyit only after Harbor (it alone brings
`harbor_config` into the lock, D69); `marin.inference` after everything that uses it, with its two
endpoint types moved into the evaluation package first; the pin machinery last. Three edges in
kept code are cut: `marin.evaluation.records` stops importing Harbor's commit-hash pattern and
drops the Harbor and Evalchemy parts of its schema; `marin.evaluation.lm_eval` takes `RunningModel`
from `marin.evaluation.endpoint`; `marin.evaluation.model_config` drops speculative serving and the
Iris gang check and owns `resolve_tokenizer_revision`.

**Tech Stack:** Python 3.12, `uv`, pydantic (records), draccus, `tools.prune_check`.

**Spec:** `docs/poorin/specs/2026-10-04-core-prune-design.md` ("Deleted regardless of
reachability", row P3 of "Pass 1: deletion order"); amendment
`docs/poorin/specs/2026-10-06-workbench-and-roots-design.md` ("Removed roots", "Changes to the core
prune spec": P3 also deletes Harbor, Evalchemy, their `config/external` pins, and
`marin.external_dependencies`); decisions D10, D48, D68, D69. P3 "removes only the Iris and fleet
dependencies and redesigns nothing, so sub-project 7a starts from Marin's framework as it is."

## Global Constraints

- Branch `prune/p3-serving-eval` from local `main` after P2 merges. Do not push.
- `uv run --no-sync` for every project command; after `uv lock`, `uv sync --locked --all-packages
  --extra=gpu`, confirm `jax.default_backend()` is `gpu`, and diff package sets (`added` must be
  `[]`, no version may move).
- Every commit runs `uv run --no-sync python -m tools.prune_check --phase P3 --step "<label>"` and
  commits its metrics row. Lint first. Chain `git add` and `git commit` with `&&`; stage deletions
  with `git rm`.
- Before deleting a module, search the surviving tree for its dotted name and path as strings.
- A test file that imports a deleted module goes in the same commit, including files under
  `tests/cluster/` (pytest imports a file before it applies marker deselection). A test file that
  also covers kept code is split, not deleted.
- The SkyRL hooks (D68) stay: `levanter.compat.hf_checkpoints`, `marin.datakit.chat_template`,
  `levanter.inference`, `sample_lm`, evaluation records and rollouts.
- Commit titles `[poorin] ...`, no attribution trailer; each body says what the code did for Marin
  and why Poorin does not need it.

## Review Focus

1. **A surviving module loads a deleted one by string** (draccus choice registries, `importlib`,
   entry points in `pyproject.toml` scripts).
2. **Record compatibility:** after the schema cut, a `record.json` written by the kept lm-eval path
   still validates, and `tests/evals/test_records.py` and `tests/evaluation/test_eval_stats.py`
   still pass unchanged except for removed Harbor/Evalchemy cases.
3. **`model_config` behavior:** non-speculative serving configs validate exactly as before; the
   tokenizer-revision rules are unchanged by the move.
4. **Lock drift** in the Verifyit and pin-machinery commits.
5. **Docs nav** resolves after each docs deletion.

---

### Task 1: Delete the evaluation fleet launchers and the cohort policy entry point

**Delete:** `experiments/evals/{evals,federated_inference_proxy_demo,lm_eval_suite,run_base_model_evals,run_key_evals,served_qwen3}.py`;
`experiments/references/reference_training_pipeline.py`; every Python file and `migrations/`,
`serve/` under `experiments/evaluation/` (keep `experiments/evaluation/configs/` and its README
only if it still describes the configs); `eval_policy/` at the repository root; tests
`tests/evals/test_inference_proxy.py`, `tests/evaluation/test_eval_archive.py`,
`tests/evaluation/test_eval_policy_launch.py`, `tests/evaluation/test_evaluation.py`,
`tests/evaluation/test_samples_v4_migration.py`.
**Split:** `tests/evaluation/test_model_config.py`: remove the tests that need
`experiments.evaluation.fleet` or `experiments.evaluation.models`; keep the ones that exercise
`marin.evaluation.model_config` alone.
**Edit:** `tools/roots.toml` (`[deleted]` entries for `experiments.evaluation`), docs that describe
the launchers (`docs/explanations/evaluation.md`, including its sentences about SkyRL rollout rows from the deleted `marin.rl.skyrl`, and `docs/tutorials/run-lm-evals.md`: reduce to what
the kept framework does, or delete and drop from `mkdocs.yml`), any skill that names them.
Check first: `uv run --no-sync python -m tools.reachability --why <each module>` prints
`unreachable`; `experiments.evals.task_configs` (a kept root) does not import the deleted files.

### Task 2: Delete `marin.mcp`

**Delete:** `lib/marin/src/marin/mcp/`, `tests/mcp/`. **Edit:** `lib/marin/pyproject.toml`
(`marin-mcp-babysitter` script), `tools/roots.toml` (`marin.mcp`). It babysat Marin's fleet jobs
for an MCP client.

### Task 3: Delete the Harbor integration

**Delete:** `lib/marin/src/marin/evaluation/harbor/`, `tests/evaluation/test_harbor_runner.py`,
`tests/evaluation/test_harbor_trial_driver.py`, `docs/harbor-integration.md`,
`docs/debug-log-agentic-daytona.md` (if it documents only Harbor on Daytona).
**Cut in `marin.evaluation.records`:** define the full-commit pattern locally
(`FULL_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"`) instead of importing it from Harbor; delete
`HarborRef` and `EvalRef.harbor`, and any status, mechanism, or provenance field that only Harbor
runs produce. Keep fields shared with lm-eval runs.
**Edit:** `GenerationConfig` and `AgentConfig` docstrings in `model_config.py` that name Harbor;
`AgentConfig` itself stays if `ModelConfig` keeps it as experiment input (redesign nothing).

### Task 4: Delete Evalchemy and the fleet orchestration

**Delete:** `lib/marin/src/marin/evaluation/evalchemy/`, `runner.py`, `serving_config.py`,
`hardware.py`, `inference_metrics.py`, `eval_policy.py`, `eval_policy_sources.py`,
`lib/marin/src/marin/experiment/evaluation.py`; tests `tests/evals/test_evalchemy_config.py`,
`tests/evals/test_eval_result.py`, `tests/evaluation/test_eval_policy.py`,
`tests/cluster/evals/test_served_evalchemy.py`; the remaining fleet cases in
`tests/evaluation/test_model_config.py`.
**Cut in `marin.evaluation.records`:** delete `EvalchemyRef`, `EvalchemyJudgeRef`,
`EvalRef.evalchemy`, and `EVALCHEMY_INFRASTRUCTURE_ERROR` if only the Evalchemy runner used it;
keep the `evalchemy_image` validation alias only if old lm-eval records carry it.
**Edit:** `marin.evaluation.__init__` if it re-exports deleted names; `tools/roots.toml`
`[deleted]` entries for these modules.

### Task 5: Delete Verifyit

As planned in P2 Task 3 (D69): delete `lib/verifyit/`; remove `verifyit[...]` from the root
`[project]` dependencies, `lib/verifyit` from the workspace members, `verifyit` from
`[tool.uv.sources]`, the `test` group entry in `lib/marin/pyproject.toml`, and the two Verifyit
host exclusions in `tools/affected_tests.py`; relock; diff (expect Verifyit, `harbor-config`,
`reasoning-gym`, `math-verify`, and their private dependencies removed; nothing added); resync.

### Task 6: Move the endpoint types and delete `marin.inference`

**Create:** `lib/marin/src/marin/evaluation/endpoint.py` with `OpenAIEndpoint` and `RunningModel`
moved verbatim from `marin/inference/types.py` (the request, lease, and provider types belonged to
the inference broker and go with it).
**Cut:** `marin.evaluation.lm_eval` imports `RunningModel` from `marin.evaluation.endpoint`.
`marin.evaluation.model_config`: move `resolve_tokenizer_revision` in verbatim; delete
`ServeConfig.speculative` and its validation; delete the `validate_pipeline_args` call (it checked
flags against Iris's gang geometry) and keep the remaining pipeline-parallel checks.
**Delete:** `lib/marin/src/marin/inference/`, `lib/marin/src/marin/testing/inference/` (Snowball
backend-parity jobs on the serving stack); tests `tests/inference/` (except
`test_external_dependencies.py`, Task 7), `tests/cluster/vllm/`, `tests/vllm/test_llm_inference.py`;
the `marin-serve` script; `[deleted]` entries in `tools/roots.toml` for `marin.inference*`;
serving docs (`docs/tutorials/local-gpu.md` sections that launch `marin-serve`,
`docs/dev-guide/tpu_observability.md` if it covers only serving).
**Tests:** `tests/evals/test_records.py` and `tests/evaluation/test_model_config.py` (kept cases)
pass; add no test for the move itself (it is an import path).

### Task 7: Delete the pinned-fork machinery

**Delete:** `config/external/` (pins for vLLM, tpu-inference, Evalchemy, Harbor, MarinSkyRL, and
their migration descriptors), `config/update-external.py`, `lib/marin/src/marin/external_dependencies.py`,
`tests/inference/test_external_dependencies.py`. **Edit:** `tools/roots.toml`, root and
`lib/marin/pyproject.toml` entries that read `config/external` (the `exclude` glob in
`[tool.uv.workspace]`), `.gitignore`, docs. First confirm nothing that survives reads
`config/external` or imports `marin.external_dependencies`.

---

## After the last task

Final whole-branch review (fresh reviewer, most capable model), fix pass, full suite, then a local
fast-forward merge into `main`.
