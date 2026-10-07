# P8 Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trim every workspace manifest to what surviving code imports, move PyTorch to its CPU
wheel, declare Flax 0.12.10 or later, resolve a fresh `uv.lock`, and rebuild the venv on it
without losing the RTX 5090 environment.

**Architecture:** Manifests change first, by hand, while the current venv keeps every check
green. One `uv lock` then resolves the trimmed workspace, and its versions are compared with the
pre-P8 freeze. The current `.venv` is renamed aside, a fresh one is synced from the lock, and
every check runs in it. A failure restores the old venv by renaming it back.

**Tech Stack:** uv workspaces, PyTorch CPU index, JAX 0.11.1 CUDA 13 wheels.

**Spec:** `docs/poorin/specs/2026-10-04-core-prune-design.md`, P8 row and "Risks" (cuDNN on
sm_120).

## Global Constraints

- JAX, jaxlib, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` stay at 0.11.1. P8 moves no JAX
  wheel; a JAX upgrade is its own later change.
- Until the fresh venv passes, every command uses `uv run --no-sync` (D70). `uv lock` is allowed
  from Task 5; `uv sync` only in Task 6.
- The old venv is renamed, never deleted, until the owner confirms the new one.
- Each manifest change is its own commit; `uv.lock` is committed once, in Task 5.
- Decisions go in `docs/poorin/decisions.md` in the commit that makes them.

## Review Focus

1. A package removed from a manifest but imported lazily (inside a function) by kept code: the
   import audit reads only module-level `import` statements of every file, so lazy imports are
   found too, but optional-feature imports guarded by `try` may hide. The kept-root import check
   and every suite in the fresh venv are the gate.
2. The fresh resolve moving a CUDA 13 library (cuDNN, NCCL, cuBLAS) or Triton: compare against
   the freeze, and run R1 and R2, which exercise cuDNN attention through the NVRTC preload.
3. The CPU torch wheel missing an API that `levanter.compat.hf_checkpoints` uses on CPU tensors:
   the Hugging Face round-trip tests run in the fresh venv.
4. `nvrtc_builtins_path()` assumes `site-packages/nvidia/cu13/lib`: R2 fails loudly if the layout
   moves.
5. Console scripts and shebangs: a renamed venv keeps absolute paths, so it is only ever renamed
   back to `.venv`, never used from its new name.

---

### Task 1: Root manifest

**Files:** Modify `pyproject.toml`.

- [ ] Remove the `fast-track` console script (its launcher was deleted in P7).
- [ ] Remove the `pipeline` extra (JaxPP's grug variant is gone), its six `conflicts` entries,
  the `jaxpp` git source, and the `torch` source keyed on that extra.
- [ ] Remove `watchdog` and `seaborn` (no importer).
- [ ] Commit: `[poorin] Drop root dependencies of deleted code`.

### Task 2: Member manifests

**Files:** Modify `lib/marin/pyproject.toml`, `lib/levanter/pyproject.toml`,
`lib/rigging/pyproject.toml`, `lib/haliax/pyproject.toml`.

Audit (2026-10-07): no surviving file imports `mcp`, `markdownify`, `resiliparse`, `shellsim`,
`openai_harmony`, `fasttext`, `backports.zstd`, `googleapiclient`, `openai`, `fastapi`,
`uvicorn`, `prometheus_client`, `jwt`, `filelock`, `async_lru`, `lenses`, `immutabledict`,
`tokamax`, `torchvision`, `torchcodec`, `numba`, `matplotlib`, `watchdog`, or `seaborn`.

- [ ] Marin: drop `mcp`, `markdownify`, `resiliparse` (and the `marin-resiliparse` index),
  `shellsim`, `openai-harmony`, `fasttext-wheel`, `backports-zstd`, `google-api-python-client`,
  `openai`; `marin-levanter[serve]` becomes `marin-levanter`, `marin-rigging[prometheus]`
  becomes `marin-rigging`; drop the `vizier` extra and the aarch64 GB200 PJRT source.
- [ ] Levanter: drop `filelock`, `async-lru`, `lenses`, `immutabledict`; the `serve`, `deepep`,
  `kernels`, and `gcp` extras; from the test group `pytest-forked`, `soundfile`, `librosa`,
  `torchcodec`, `pytest-profiling`, `numba`; from the dev group `matplotlib`.
- [ ] Rigging: drop `pyjwt` and the `iap` and `prometheus` extras.
- [ ] Each library is one commit: `[<lib>] Drop dependencies of deleted code`.

### Task 3: PyTorch CPU wheel

**Files:** Modify `lib/marin/pyproject.toml`, `pyproject.toml`.

Surviving code uses PyTorch only for Hugging Face checkpoint conversion and test helpers, on
CPU. vLLM is gone and the owner's SkyRL fork has its own environment (D68). The CUDA build
installs a CUDA 12 copy of cuDNN, NCCL, and NVRTC beside JAX's CUDA 13 libraries.

- [ ] Every `torch` requirement resolves from the `pytorch-cpu` index on Linux x86-64;
  `torchvision` (no importer) goes from every extra and source.
- [ ] Record the decision in `decisions.md`.
- [ ] Commit: `[poorin] Install PyTorch from its CPU wheel`.

### Task 4: Flax

**Files:** Modify `lib/haliax/pyproject.toml`.

`haliax._src.fp8` imports Flax, which today arrives only through optax, orbax, safetensors, and
aqtp.

- [ ] Declare `flax>=0.12.10` in Haliax's dependencies.
- [ ] Commit: `[haliax] Declare Flax 0.12.10 or later`.

### Task 5: Fresh lock

**Files:** Create `uv.lock`; modify `pyproject.toml` overrides if needed.

- [ ] `uv lock`. Expected: resolves without conflicts.
- [ ] Compare the lock's versions with `pre-p8-freeze.txt` (workspace file): list every package
  whose version moved and every package added. JAX wheels must not move (Global Constraints);
  pin in the `gpu` extras if they do.
- [ ] Drop `override-dependencies` and `constraint-dependencies` entries naming a package absent
  from the lock, then re-lock.
- [ ] Commit `uv.lock` with the override cleanup: `[poorin] Resolve a fresh lockfile`.

### Task 6: Fresh venv

- [ ] `mv .venv .venv-pre-p8`, then `uv sync --locked --all-packages --extra=gpu`.
- [ ] GPU sanity: `uv run --no-sync python -c "import jax; print(jax.devices())"` lists the
  RTX 5090.
- [ ] Every kept root imports, every suite passes: `uv run --no-sync python -m tools.affected_tests --all`.
- [ ] `uv run --no-sync python -m tools.prune_check --phase P8 --step "Rebuild the venv on a fresh lock"`
  passes, including R1, R2, and Finelog; its row records the new `venv_size`.
- [ ] On any failure: `rm -rf .venv && mv .venv-pre-p8 .venv`, fix, retry.
- [ ] Keep `.venv-pre-p8` until the owner confirms.

### Task 7: Docs and decisions

**Files:** Modify `AGENTS.md`, `docs/poorin/decisions.md`, and any page that says there is no
lockfile.

- [ ] AGENTS.md: drop "Until P8 there is no `uv.lock`"; restore the lock-diff rule for dependency
  changes.
- [ ] decisions.md: D70's lockfile clause ends; resolve the P8 Deferred rows.
- [ ] Commit with the metrics row from Task 6.
