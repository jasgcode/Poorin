# Spike #0: Marin on an RTX 5090

Date: 2026-10-04. Machine: RTX 5090 (32 GB, sm_120), driver 580.178, 32 CPU cores, 125 GB RAM.
Repo: Poorin @ ab0da00df (upstream Marin, unmodified). All spike code lives in the session scratchpad.

## Question

Does Marin's stack train on a consumer Blackwell GPU without changes, and which parts of the
monorepo does a local training run actually touch?

## Install

`uv sync --all-packages --extra=gpu` succeeded first try: 85 s, 477 packages, 12 GB `.venv`.

The venv holds two CUDA stacks. JAX uses CUDA 13 (`nvidia/cu13/lib`); the CUDA PyTorch wheels
(`pytorch-cu128` index) bring CUDA 12 copies of cuBLAS, cuDNN, NCCL, NVRTC, and nvJitLink. Both
`nvidia-cudnn-cu12` and `nvidia-cudnn-cu13` install into the same `nvidia/cudnn/lib` directory.

Doc drift: `docs/tutorials/installation.md` still says Marin defaults to CUDA 12.9;
`docs/tutorials/local-gpu.md` says CUDA 13.

## JAX on sm_120

| Check | Result |
|---|---|
| Device detection | `NVIDIA GeForce RTX 5090`, compute capability 12.0 |
| bf16 matmul 8192³ | 221.7 TFLOP/s (about the card's spec peak) |
| `dot_product_attention(implementation="xla")` | works, 13.8 TFLOP/s (materializes T×T) |
| `dot_product_attention(implementation="cudnn")` | **fails**: `No valid execution plans built` (every head dim and mask tried) |

Root cause of the cuDNN failure: NVRTC 13 loads `libnvrtc-builtins.so.13.0` by soname at
runtime. That file lives in `nvidia/cu13/lib`, which is not on the loader search path, so the
load fails, cuDNN cannot runtime-compile its sm_120 kernels, and every execution plan is
rejected. Datacenter parts likely avoid this because cuDNN ships precompiled engines for them.
Upstream only handles the NVRTC path for vLLM inference (`marin.inference.vllm_wheel_entrypoint`).

Workaround: `LD_LIBRARY_PATH=$VENV/lib/python3.12/site-packages/nvidia/cu13/lib`. With it,
all cuDNN configurations pass.

## Training: tiny Llama tutorial

`experiments/tutorials/train_tiny_model.py --device h100x1 --dataset wikitext`, 100 steps of a
2-layer, 32-wide Llama. `LocalClient` ignores the `ResourceConfig`, so the `h100x1` entry ran
on the 5090 unchanged.

Cold run: 6m27s wall clock. Training compute was a few seconds; the rest was startup.

| Phase | Time |
|---|---|
| Tokenize 1,000 wikitext docs (Zephyr) | 3 s |
| Fused cross-entropy autotune (7 block-size candidates) | 87 s |
| Trace `train_step` (includes autotune) | 94 s |
| XLA compile + first step | 79 s |
| Compile hooks + 99 steps | 85 s |
| Eval compile + eval | 85 s |

Reruns, each under a fresh version so nothing resumes:

| Run | Caches | Wall clock | First step | Eval |
|---|---|---|---|---|
| cold | none | 6m27s | 180 s (incl. autotune) | 85 s |
| warm1 | autotune only (Marin's default locally) | 5m10s | 112 s | 87 s |
| warm2 | autotune + `JAX_COMPILATION_CACHE_DIR` set in shell, populating | 5m15s | 115 s | 89 s |
| warm3 | both caches, hitting | **1m07s** | 22 s | 12.5 s |

Setting `JAX_COMPILATION_CACHE_DIR` in the shell before launch is the single largest
iteration-speed win found. The cache holds 31 MB after two runs.

## Gotchas found

1. **`--version dev` resumes instead of retraining.** The executor reruns a `dev` step every
   time, but the output path is `{name}/dev` on every run, and Levanter's resumption
   checkpointer finds the previous checkpoint there. The second run logged
   `Resuming training from step 100` and only re-ran eval. Run identity covers the config,
   not the code, so editing model code and rerunning `dev` continues from old weights (or
   crashes on a shape change).
2. **The JAX compilation cache is never enabled locally.** Marin computes
   `JAX_COMPILATION_CACHE_DIR` and logs it, but puts it only in the job's environment dict.
   `LocalClient` runs jobs as threads in the launcher and ignores that dict, and nothing merges
   it into `os.environ` (JAX would need it before import anyway). The directory is never
   created. Levanter's `TrainerConfig.jax_compilation_cache_dir` would work but is unused.
3. **No FLOPs entry for the 5090** in `fray/device_flops.py`, so MFU is not reported
   (`Unknown device type: nvidia geforce rtx 5090`).
4. **W&B source capture refuses the repo** because the in-repo `.venv` makes the directory
   12 GB.

## What a local run imports

`python -X importtime`: 7,080 modules, about 10 s of import time per launch.

Largest packages by module count: torch 1,006, sympy 432, jax 385, scipy 357, wandb 316,
pandas 299, cutlass 194, polars 181, torchvision 179, levanter 150, transformers 137,
IPython 114, iris 69, haliax 47, marin 39, zephyr 23.

Unexpected imports and the chains that pull them in:

- torch, torchvision, sympy, torch._dynamo: `poorin.experiment.train` → `levanter.adaptor` →
  `levanter.compat.hf_checkpoints` → `transformers` (image utils pull torchvision, which pulls
  dynamo and sympy). Half of all import time.
- IPython: `levanter.compat.hf_checkpoints` → `tqdm_loggable.auto`.
- polars: `levanter.store.cache` → `deepdiff`.
- iris: `poorin.execution.step_runner` → `iris.cluster.client.job_info`.

## Grug

### Attention on sm_120

Causal attention, B=4, T=4096, H=16, D=64, bf16, forward + backward:

| Implementation | Time | Approx. TFLOP/s | Needs `cu13/lib` on the path |
|---|---|---|---|
| `grug:reference` (the GPU default) | 14.7 ms | 33 | no |
| `grug:xla_flash` | 15.7 ms | 31 | no |
| `grug:gpu_fa4_cute` (FlashAttention-4, CuTe DSL) | 3.4 ms | 140 | no |
| `jax.nn.dot_product_attention(implementation="cudnn")` | 3.1 ms | 156 | **yes** |

Grug picks `reference` on any non-TPU backend. It materializes an fp32 B×H×T×T score tensor
(4.3 GB here) and runs about 4.5× slower than FA4 or cuDNN. FA4 works on sm_120 despite
targeting Hopper and datacenter Blackwell. cuDNN is not an option in grug's attention
dispatcher.

### Training run

Throwaway launcher (`spike_grug.py`): grug's 130M base model and train loop, unchanged, on the
10M-token FineWeb-Edu cache from Hugging Face instead of Nemotron-CC on GCS. Batch 16 × 1024
tokens, 200 steps, Adam, bf16 compute, `reference` attention, offline W&B.

| Metric | Value |
|---|---|
| Steady-state step time | 146 ms |
| Throughput | 112k tokens/s, 60 TFLOP/s (analytic) |
| MFU vs. measured 222 TFLOP/s peak | about 27% |
| Loss after 200 steps (3.3M tokens) | 6.62 |
| Wall clock, cold | 7m08s, of which training was about 31 s |

Startup timeline, cold:

| Time | Event |
|---|---|
| 17:19:10 | start, fused cross-entropy autotune for grug's shapes (95 s) |
| 17:22:38 | step 1 done (first step compile) |
| 17:24:10 | step 2 done (second step-function compile, 92 s) |
| 17:25:42 | step 11 done (third step-function compile, 92 s) |
| 17:26:13 | steps 11–200 done in 31 s |

Grug's loop compiles three variants of its step, each about 90 s on this model. The cause is
two static flags on the jitted step: `compute_watch` (gradient statistics, every 10 steps) and
`compute_backward_flow` (a gradient-flow diagnostic, every 50 steps). Step 0 has both on,
step 1 has both off, step 10 has only watch on; each combination is its own executable.

Warm rerun under a fresh version (both caches populated): **1m05s** wall clock versus 7m08s
cold, 104k tokens/s.

Same seed, same data, same code: final loss 6.621 (cold) vs 6.633 (warm). GPU reductions are
not bitwise deterministic by default. Scaling-law fits need this noise floor measured, or
deterministic ops enabled (`--xla_gpu_deterministic_ops`) at some throughput cost.

## Recommendation

Stay on JAX and Marin. Nothing failed for lack of sm_120 support; every failure was
configuration or coupling. Two environment variables make the stack usable today:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/poorin/jax"
```

With both, a repeated run of the same shapes starts training in about a minute instead of
six or seven.

Input for sub-project #1 (core prune), ordered by measured impact:

1. **Apply the job environment on the local backend** and default the compilation cache on.
   Largest iteration-speed win measured (6.6× on grug).
2. **Decide run identity for dev runs.** Today `dev` silently resumes from the previous
   checkpoint. Options: fresh path per invocation, or a code hash in the identity.
3. **Cut eager imports from the training path.** Lazy-load `levanter.compat.hf_checkpoints`
   (takes torch, torchvision, transformers, IPython off every launch); remove Iris imports
   from `poorin.execution.step_runner`, `levanter.distributed`, and grug's dispatcher.
4. **Pick one CUDA stack**, or pin the NVRTC path at venv level, so cuDNN works without a
   manual `LD_LIBRARY_PATH`. Whether PyTorch stays depends on embodied tooling
   (simulators, video decoding), decided in #3 and #6.
5. **Make grug's GPU attention default FA4 or cuDNN** instead of `reference`: 4.5× faster,
   and `reference` will run out of memory at video sequence lengths.
6. **Make grug's diagnostics opt-in for iteration runs.** Each static diagnostic flag adds a
   ~90 s compile on this model.
7. Add an RTX 5090 entry to `fray/device_flops.py` so MFU is reported.
8. Measure the run-to-run noise floor before trusting small loss differences in sweeps.

## Spike artifacts

The spike scripts were throwaway and are not in the repository. `experiments/grug/base/launch.py` (P0) replaces the spike's grug launcher.

## Fixed in P0

- `--version dev` gets a fresh output path per run; `--resume <version>` continues one.
- Grug's default GPU attention is FA4 for causal self-attention.
- The local grug launcher disables gradient watch and backward flow.
- The RTX 5090 has a FLOPs entry, so MFU is reported.

The compilation cache and cuDNN fixes land in P5a; until then `tools/prune_check.py` sets
`JAX_COMPILATION_CACHE_DIR` and `LD_LIBRARY_PATH`.
