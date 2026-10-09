# Poorin architecture

Every first-party module that survived the core prune, and why it exists. Pass 2 of the prune
(spec: `specs/2026-10-04-core-prune-design.md`) ends when every module has a row here and the ten
modules that dominate the most reachable lines all have a Keep verdict.

`tests/tools/test_architecture.py` fails when a workspace module has no row or a row names a module
that no longer exists. It runs in every full test run (`tools.affected_tests --all`).

## How to read this page

- **Reached from** names the root whose import chain reaches the module: R1
  (`experiments.tutorials.train_tiny_model`, a Llama language-model run), R2
  (`experiments.grug.base.launch`, the grug transformer run), or a kept root from
  `tools/roots.toml`. "test support" marks modules that only tests import; "package" marks a
  package `__init__` whose submodules are reached.
- `uv run --no-sync python -m tools.reachability --why <module>` prints the full chain, and
  `--rank N` the modules that dominate the most reachable lines.

## Ranked verdicts

`tools.reachability --rank 10` after Pass 2's cuts (dominated lines: the lines reachable only
through that module):

| Dominated lines | Module | Verdict | Why |
|---|---|---|---|
| 4,797 | `haliax` | Keep | Named tensors under every Levanter model; sub-project 1b decides whether Haliax stays |
| 4,510 | `levanter.kernels.pallas.fused_cross_entropy_loss` | Keep | Grug's loss; `batched_xla` on GPU, `pallas_tpu` on TPU (D81) |
| 4,164 | `levanter.grug.attention` | Keep | Grug's attention implementations |
| 3,581 | `levanter.kernels.pallas.fused_cross_entropy_loss.api` | Keep | Implementation selection and block-size autotuning for the fused loss |
| 3,524 | `levanter.grug.attention._fa4_cute` | Keep | FlashAttention 4 CuTe path: 140 TFLOP/s on the RTX 5090 against 33 for reference attention |
| 2,875 | `levanter.grug.attention._fa4_cute_backend` | Keep | Segment-aware FA4 backward used by that path |
| 2,787 | `haliax.nn` | Keep | Haliax's layers (linear, embedding, normalization, scan) |
| 2,393 | `zephyr.coordinator` | Keep | Zephyr runs tokenization and cache building; sub-project 3 decides its future |
| 2,214 | `levanter.layers.attention` | Keep | Attention for Levanter's Llama, GPT-2, and Qwen models (R1) |
| 2,161 | `haliax.core` | Keep | `NamedArray` and axis arithmetic |

Pass 2's edge cuts and their measurements are in `decisions.md` (D81, D82) and
`prune-metrics.csv`. Deferred: Hugging Face conversion stays eager until sub-project 4 (D82).

## Libraries forked and made lighter

Poorin renamed Marin's core package to `poorin` and gave the other libraries it cut heavily `poorin-`
distribution names (D85, D86). Source lines against `upstream-baseline`; each library's README lists
what was removed.

| Library | Import | Distribution (upstream) | Before | Now |
|---|---|---|---|---|
| Experiment framework | `poorin` (was `marin`) | `poorin` (`marin-core`) | 65,588 | 23,602 |
| Finelog client | `finelog` | `poorin-finelog` (`marin-finelog`) | 10,023 Python, plus 74,921 Rust | 2,938 Python |
| Rigging | `rigging` | `poorin-rigging` (`marin-rigging`) | 14,465 | 8,985 |
| Fray | `fray` | `poorin-fray` (`marin-fray`) | 2,768 | 1,925 |

Deleted outright: Iris (83,889), Verifyit (7,916), Shellbox (2,675), Ducky (2,074), and
Taskcompendium (1,664). Levanter (85,297 to 62,330) keeps its name: its core is intact.

## Reviewed and kept

These modules looked vestigial on a one-GPU stack. D89 and D90 reviewed them and kept each one
for a stated user:

- `zephyr.dashboard` (with its `app` and `coordinator`): the coordinator's status page, kept for
  local data runs (D90). Nothing serves it yet (Deferred, in `decisions.md`).
- `rigging.filesystem.mirror`: Levanter stages tokenizers through `mirror://`.
- `levanter.model_cache`: it is on the Hugging Face loading path.
- `finestore.migrations.m0001_manifest`: Finestore runs it on every write.
- `experiments.grug.fast_track.grugmuon_stacked` and `levanter.grug._moe.quack_symmetric_cute`:
  fast-track's MuonH needs the stacked transform. QuACK runs only for expert-parallel meshes on
  SM90 and SM100, which one GPU never forms.
- `levanter.models.gpt2`: the tests' tiny model and Hugging Face round trips.
- TPU-only code (D81): `levanter.kernels.pallas.splash_attention`, the `pallas_tpu` fused
  cross-entropy, and `poorin.scaling_laws.tpu_utils`, which also feeds the AdamH heuristic.

## Modules

### haliax

| Module | Purpose | Reached from |
|---|---|---|
| `haliax` | Top-level NamedArray API: re-exports core ops and named wrappers of jax.numpy functions | R1 |
| `haliax.__about__` | Package version string | `haliax.__about__` |
| `haliax._src` | Package marker for internal implementations | R1 |
| `haliax._src.compile_utils` | Compile cache for jitted functions, copied from Equinox internals | R1 |
| `haliax._src.dot` | Named-axis dot product with contraction and output-sharding inference | R1 |
| `haliax._src.einsum` | Named-axis einsum over axis-name expressions | R1 |
| `haliax._src.fp8` | FP8 quantize/dequantize with delayed scaling and custom-VJP scaled dot_general | R2 |
| `haliax._src.parsing` | Parser for rearrange and einsum axis-expression strings | R1 |
| `haliax._src.rearrange` | Einops-style rearrange of NamedArray axes by name or expression | R1 |
| `haliax._src.scan` | NamedArray-aware scan, fold, and map with gradient-checkpoint policies | R1 |
| `haliax._src.state_dict` | Torch-style state-dict conversion, flattening, and save/load for module trees | R1 |
| `haliax._src.util` | Internal helpers: index_where and an identity-keyed map | R1 |
| `haliax.axis` | Axis type and axis-spec helpers: concat, union, eliminate, resize, select | R1 |
| `haliax.core` | NamedArray class and core indexing, slicing, broadcasting, and reshaping ops | R1 |
| `haliax.debug` | Module diagnostics for reused or static arrays, and sharding visualization | R1 |
| `haliax.fft` | Named-axis wrappers around jax.numpy.fft | R1 |
| `haliax.field` | equinox.field wrapper that adds axis_names metadata | R1 |
| `haliax.haxtyping` | jaxtyping-style dtype and shape annotations for NamedArray | R1 |
| `haliax.hof` | NamedArray-aware vmap | R1 |
| `haliax.jax_utils` | JAX helpers: multi-host sync, RNG splitting, checkpointing, nested scan | R2 |
| `haliax.nn` | Neural-net package; re-exports layers, activations, losses, and defines one_hot | R1 |
| `haliax.nn.activations` | Named wrappers for activation functions such as relu, gelu, silu, glu | R1 |
| `haliax.nn.array_stacked` | ArrayStacked: layer stack with a leading layer dimension in plain array leaves | R1 |
| `haliax.nn.attention` | Dot-product attention, attention masks, and ALiBi bias over named axes | R1 |
| `haliax.nn.conv` | Named-axis Conv and ConvTranspose layers | R1 |
| `haliax.nn.dropout` | Dropout function and module | R1 |
| `haliax.nn.embedding` | Embedding lookup layer | R1 |
| `haliax.nn.linear` | Linear and MoELinear layers with pluggable dot_general | R1 |
| `haliax.nn.loss` | Cross-entropy and binary cross-entropy losses with reduction helpers | R1 |
| `haliax.nn.mlp` | Multi-layer perceptron module | R1 |
| `haliax.nn.mup` | muP and standard parameterization rules for linear and embedding layers | R1 |
| `haliax.nn.normalization` | LayerNorm, RmsNorm, and named softmax, logsumexp, standardize | R1 |
| `haliax.nn.pool` | Max, min, and mean pooling over named axes | R1 |
| `haliax.nn.ragged_dot` | Grouped ragged matmul for MoE, dispatching to Megablox, Triton, or XLA kernels | R1 |
| `haliax.nn.scan` | Stacked and BlockSeq layer containers that scan or loop over blocks | R1 |
| `haliax.ops` | Named array ops: where, clip, tril/triu, pad, unique, searchsorted, bincount | R1 |
| `haliax.partitioning` | Axis-to-mesh mapping, sharding, named_jit, FSDP, and shard_map | R1 |
| `haliax.poly` | NamedArray wrappers around jax.numpy polynomial functions | R1 |
| `haliax.quantization` | FP8/INT8 dot_general ops, Linear-layer quantization, custom gradient accumulation | R2 |
| `haliax.random` | Named-axis wrappers around jax.random samplers | R1 |
| `haliax.ref` | NamedRef: jax.Ref wrapper that keeps axis metadata | R1 |
| `haliax.specialized_fns` | Named top_k selection | R1 |
| `haliax.state_dict` | Public torch-compatible state-dict export | R1 |
| `haliax.testing` | Pytest skip markers for device count and torch availability | test support |
| `haliax.tree` | jax.tree-style wrappers over haliax.tree_util | R1 |
| `haliax.tree_util` | NamedArray-aware tree map and flatten, scan-aware map, axis resize | R2 |
| `haliax.types` | Type aliases: DType, Scalar, PrecisionLike, FilterSpec, FilterTree | R2 |
| `haliax.util` | Small helpers: is_named_array, tuple coercion, safe functools.wraps | R1 |
| `haliax.wrap` | Decorators that lift unary, binary, and reduction functions to NamedArrays | R1 |

### levanter

| Module | Purpose | Reached from |
|---|---|---|
| `levanter` | Package marker; imports nothing so submodules load independently | package |
| `levanter._debug_logging` | Best-effort flush of log handlers, stdout and stderr | R2 |
| `levanter.adaptor` | AdaptorConfig choosing full fine-tuning or LoRA, with trainable filter and export | R1 |
| `levanter.adaptor.lora` | LoRA layers, parameter filtering, merging, and PEFT or merged HF checkpoint export | R1 |
| `levanter.analysis` | Package marker | R2 |
| `levanter.analysis.backward_flow` | Sampled activation and gradient-scale probes per scope, with HTML dataflow graph rendering | R2 |
| `levanter.analysis.entropy` | Token entropy and top-2 logit gap histograms, with tracker callbacks | R1 |
| `levanter.analysis.model_perplexity` | Per-document model score files: write, read, compare, and attach report metrics | `levanter.main.perplexity_gap` |
| `levanter.analysis.perplexity_gap` | Byte-span tokenization, segment scoring, and Markdown reports of perplexity gaps between models | `levanter.main.perplexity_gap` |
| `levanter.analysis.tree_stats` | Summary statistics and histograms for parameter trees, including scan-stacked layers | R2 |
| `levanter.callbacks` | Callback API re-exports, eval loss loop, mixture-weight logging hook, profiling context | R2 |
| `levanter.callbacks._core` | Callback, JitCallback, StepInfo, and training progress lifecycle events | R2 |
| `levanter.callbacks._metrics` | Per-step throughput and MFU computation, step logging, tqdm progress bar | R2 |
| `levanter.callbacks.labeled_eval` | Periodic per-label LM loss evaluation on trace-chat datasets during training | R1 |
| `levanter.callbacks.profiler` | Scheduled JAX profiler captures with XProf trace and XLA dump uploads | R2 |
| `levanter.callbacks.progress_watchdog` | Watchdog that terminates training with `EXIT_STALL` when steps or the pipeline stop progressing | R2 |
| `levanter.callbacks.state_adapter` | Runs callback hooks against arbitrary training-state types via accessors | R2 |
| `levanter.callbacks.tensorstore_callbacks` | Step hook logging TensorStore metrics when enabled by environment variable | R1 |
| `levanter.callbacks.watch` | WatchCallback logging gradient, parameter, and optimizer-state statistics | R2 |
| `levanter.checkpoint` | Checkpointer with time and step policies; TensorStore save, load, and discovery | R2 |
| `levanter.checkpoint_manifest` | Checkpoint manifest.json listing storage format and every serialized array | R2 |
| `levanter.compat` | Package marker | R1 |
| `levanter.compat.fsspec_safetensor` | Streams safetensors files over fsspec into optionally sharded JAX arrays | R1 |
| `levanter.compat.hf_checkpoints` | HFCheckpointConverter between Levanter and HF models; tokenizer loading and Hub upload | R1 |
| `levanter.compat.hf_config` | Builds HF configs from kwargs and maps RoPE parameters | R1 |
| `levanter.compile_cache` | Enables JAX's persistent compilation cache | R2 |
| `levanter.config` | draccus entrypoint wrapper accepting fsspec config paths; PartitionSpec codecs | R2 |
| `levanter.cutlass_kernel_cache` | Persistent cache of compiled CuTeDSL kernel object code across processes | R2 |
| `levanter.data` | Package marker; docstring says to import from defining submodules | R2 |
| `levanter.data._preprocessor` | BatchProcessor interface and record-batch normalization for cache writing | R1 |
| `levanter.data._prp` | Pseudo-random permutations (LCG, Feistel) for dataset shuffling | R2 |
| `levanter.data.dataset` | AsyncDataset and SyncDataset bases with mapping, slicing, permutation, block shuffling | R2 |
| `levanter.data.loader` | DataLoader batching datasets into sharded global JAX arrays on a background thread | R2 |
| `levanter.data.mixture` | Weighted MixtureDataset with staged weight schedules, and ConcatDataset | R2 |
| `levanter.data.packing` | Greedy sequence packing and per-segment loss, mainly for evaluation | R2 |
| `levanter.data.passthrough_tokenizer` | Tokenizer for pre-tokenized data that passes integer IDs written as text | R2 |
| `levanter.data.sharded_datasource` | Sharded readers for JSONL, Parquet, text, audio URLs, and HF datasets | R2 |
| `levanter.data.text` | Package marker; docstring says to import from defining submodules | R2 |
| `levanter.data.text._batch_tokenizer` | BatchTokenizer: batch text tokenization with BOS/EOS, padding, and truncation | R1 |
| `levanter.data.text.cache` | Builds or loads tokenized LM dataset caches | R2 |
| `levanter.data.text.datasets` | LmDataConfig components and token, packed, and chat LM datasets with train/val splits | R2 |
| `levanter.data.text.examples` | GrugLmExample and LabeledLmExample, with conversions to Haliax LmExample | R2 |
| `levanter.data.text.formats` | LM dataset formats (text, chat, prebuilt) and their preprocessors | R1 |
| `levanter.data.text.preference` | Preference-pair chat format, processor, and dataset for DPO | R1 |
| `levanter.data.text.trace_chat` | Agent-trace chat preprocessing with per-span loss labels for evaluation | R1 |
| `levanter.distributed` | DistributedConfig for jax.distributed initialization, with SLURM cluster discovery | R2 |
| `levanter.dpo` | DPO loss, policy/reference model pair, and cached reference log-probs | R1 |
| `levanter.eval` | Tagged and labeled LM loss evaluators and their training callbacks | R2 |
| `levanter.eval_harness` | Runs the EleutherAI LM Evaluation Harness on Levanter models | R1 |
| `levanter.eval_harness_config` | LM Eval Harness task and run configs, importable without the harness | R1 |
| `levanter.grad_accum` | Microbatched gradient accumulation wrapper and the batch-to-microbatch reshape | R2 |
| `levanter.grug` | Package marker for Grug's array-first attention, loss, and sharding primitives | R2 |
| `levanter.grug._moe` | Package marker for private Grug CuTe kernel backends | package |
| `levanter.grug._moe.quack_symmetric_cute` | QuACK symmetric GEMM for Grug Muon Newton-Schulz; SM90 and SM100 only | `levanter.grug._moe.quack_symmetric_cute` |
| `levanter.grug.attention` | Re-exports Grug attention: masks, rotary, reference, FA4/CuTe, and XLA flash | R2 |
| `levanter.grug.attention._core` | Grug attention dispatch, mask spec, rotary embedding, and reference attention | R2 |
| `levanter.grug.attention._fa4_cute` | Sharded FA4/CuTe causal self-attention wrapper with packed-segment bounds | R2 |
| `levanter.grug.attention._fa4_cute_backend` | JAX/CuTe boundary for segmented FA4 forward and backward, with custom VJP | R2 |
| `levanter.grug.attention._fa4_cute_config` | Tile and pipeline schedule configs for FA4/CuTe kernels, including SM90 and SM100 | R2 |
| `levanter.grug.attention._fa4_cute_kernels` | CuTe DSL launcher builders for segmented FA4 forward and backward kernels | R2 |
| `levanter.grug.attention._fa4_cute_segmented_bwd` | Segment-aware FA4 backward kernels for SM80 and SM120, adapted from flash-attn-4 | `levanter.grug.attention._fa4_cute_segmented_bwd` |
| `levanter.grug.attention._xla_flash` | Pure-JAX blockwise flash attention on Grug raw arrays for any XLA backend | R2 |
| `levanter.grug.loss` | Fused linear softmax cross-entropy for Grug with sharded token dimensions | R2 |
| `levanter.grug.sharding` | Grug mesh helpers: token PartitionSpecs, resharding, compact explicit mesh | R2 |
| `levanter.inference.engine` | Batch inference engine: paged-KV prefill, decode loop, sequence log-prob scoring | R1 |
| `levanter.inference.jit_scheduler` | Jit-side decode state, sequence table, and token queue for generation | R1 |
| `levanter.inference.page_table` | Paged KV-cache page allocator and per-batch page info | R1 |
| `levanter.inference.utils` | Jit-safe masked array operations and stop-sequence checks for decoding | R1 |
| `levanter.infra` | Package marker | package |
| `levanter.infra.cli_helpers` | Launch config loading, git commit, and default run ID | `levanter.infra.cli_helpers` |
| `levanter.kernels` | Package marker for custom JAX kernels | R2 |
| `levanter.kernels.pallas` | Package marker for Pallas kernels and their wrappers | R2 |
| `levanter.kernels.pallas.autotune_utils` | Sharding-aware lowering and compile helpers for kernel autotune benchmarks | R2 |
| `levanter.kernels.pallas.cost_estimate_utils` | Pallas cost-estimate helpers that set bytes accessed from kernel IO specs | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss` | Re-exports the fused cross-entropy API and tuned block-size tables | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.api` | Fused linear softmax cross-entropy dispatch across implementations, with block-size autotune cache | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.batched_xla` | GPU batched-XLA fused cross-entropy with vocab tiling and custom backward | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.config` | BlockSizes dataclass and per-device weight-tile byte budgets | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.pallas_tpu` | TPU Pallas Mosaic kernels for fused cross-entropy forward and backward | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.reference` | Reference and streaming linear softmax cross-entropy without a fused kernel | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.tuned_block_sizes` | Tuned fused cross-entropy block sizes by device kind and shape bucket | R2 |
| `levanter.kernels.pallas.fused_cross_entropy_loss.xla` | Streaming vocab-blocked cross-entropy on plain XLA with custom VJP | R2 |
| `levanter.kernels.pallas.splash_attention` | Lowering helpers for JAX TPU Splash Attention masks, segment IDs, block sizes | R1 |
| `levanter.layers` | Re-exports attention and normalization layer classes | R1 |
| `levanter.layers.attention` | Attention layers and backends (XLA, TE, Splash), gated, MLA, ragged paged attention | R1 |
| `levanter.layers.attention_mask` | Structured AttentionMask (causal, segment, sliding window) and its materialization | R1 |
| `levanter.layers.flash_attention` | Pure-JAX blockwise flash attention with custom backward on named arrays | R1 |
| `levanter.layers.kv_cache` | Paged KV cache storage and updates for paged-attention inference | R1 |
| `levanter.layers.normalization` | LayerNorm and RMSNorm config classes | R1 |
| `levanter.layers.rotary` | Rotary embedding variants and configs: default, Llama 3, YaRN | R1 |
| `levanter.layers.sampler` | Temperature-based token sampler for autoregressive generation | R1 |
| `levanter.main.eval_lm` | Script: evaluate an LM checkpoint's loss on a dataset | `levanter.main.eval_lm` |
| `levanter.main.export_hf_to_lm` | Script: import a HuggingFace model as a Levanter TensorStore checkpoint | `levanter.main.export_hf_to_lm` |
| `levanter.main.export_lm_to_hf` | Script: convert a Levanter LM checkpoint to HuggingFace format | `levanter.main.export_lm_to_hf` |
| `levanter.main.lora_lm` | Script: LoRA fine-tuning of a language model | `levanter.main.lora_lm` |
| `levanter.main.model_init` | Loads a policy model from a Levanter or HF source for train_dpo | `levanter.main.train_dpo` |
| `levanter.main.perplexity_gap` | Script: score documents with two models and report perplexity gaps | `levanter.main.perplexity_gap` |
| `levanter.main.sample_lm` | Script: sample text from a checkpoint or HF model via the inference engine | `levanter.main.sample_lm` |
| `levanter.main.train_dpo` | Script: DPO training with a separate or adapter-base reference model | `levanter.main.train_dpo` |
| `levanter.main.train_lm` | Script: Levanter LM training entrypoint and TrainLmConfig | R1 |
| `levanter.metrics` | Metric type with reduction rules for aggregating across microbatches | R2 |
| `levanter.model_cache` | Distributed-locked mirroring of HF model snapshots to a shared cache bucket | R1 |
| `levanter.model_loading` | Loads and shards an LmHeadModel from a Levanter or HF checkpoint | R1 |
| `levanter.models` | Package marker | R1 |
| `levanter.models.gpt2` | GPT-2 model with HF conversion; production use is one config default | R1 |
| `levanter.models.linear` | Marker base class identifying linear-like modules for optimizer routing | R1 |
| `levanter.models.llama` | Llama model, config, and HF conversion | R1 |
| `levanter.models.lm_model` | LM base classes: LmExample, LmConfig, LmHeadModel, and embedding resizing | R1 |
| `levanter.models.loss` | Next-token cross-entropy loss with optional fused kernel and logsumexp penalty | R1 |
| `levanter.models.qwen` | Qwen and Qwen3 models (QK-norm, sliding window) built on Llama | `experiments.references.completed_adamh` |
| `levanter.nvrtc` | Preloads NVRTC's builtins library so cuDNN works with CUDA 13 wheels | R2 |
| `levanter.optim` | Package marker; optimizer configs register lazily through a draccus plugin registry | R1 |
| `levanter.optim._block_partition` | Tensor block partitioning shared by the SOAP and Kron preconditioners | `levanter.optim._block_partition` |
| `levanter.optim.adam_mini` | Adam-mini optimizer: one second-moment scalar per parameter group | `levanter.optim.adam_mini` |
| `levanter.optim.adamh` | AdamH optimizer: Adam updates that keep linear weight norms constant | `levanter.optim.adamh` |
| `levanter.optim.adopt` | ADOPT optimizer config and transform | `levanter.optim.adopt` |
| `levanter.optim.cautious` | Cautious optimizer config and transform | `levanter.optim.cautious` |
| `levanter.optim.clip_update_norm` | Optax transform clipping update norm against its running history | R1 |
| `levanter.optim.config` | OptimizerConfig registry, AdamW and Lion configs, and learning-rate schedules | R1 |
| `levanter.optim.grugmuon` | Muon for raw (fan_in, fan_out) Grug arrays, with AdamW for embeddings | `levanter.optim.grugmuon` |
| `levanter.optim.kron` | PSGD Kron optimizer | `levanter.optim.kron` |
| `levanter.optim.mars` | MARS optimizer config and transform | `levanter.optim.mars` |
| `levanter.optim.model_averaging` | EMA model averaging variants and their configs | R2 |
| `levanter.optim.muon` | Muon optimizer: Newton-Schulz orthogonalized momentum for Haliax models | `levanter.optim.muon` |
| `levanter.optim.muonh` | MuonH optimizer: Muon updates that keep linear weight norms constant | `levanter.optim.muonh` |
| `levanter.optim.namo` | NAMO and NAMO-D optimizers: orthogonalized momentum with norm-based adaptivity | `levanter.optim.namo` |
| `levanter.optim.rmsprop` | RMSProp with momentum optimizer | `levanter.optim.rmsprop` |
| `levanter.optim.scion` | Scion optimizer config and transform | `levanter.optim.scion` |
| `levanter.optim.skipstep` | Optimizer wrapper skipping steps whose loss or gradient norm spikes | R1 |
| `levanter.optim.soap` | SOAP optimizer | `levanter.optim.soap` |
| `levanter.optim.util` | Optimizer helpers: linear-layer masks, Newton-Schulz, linear flattening, scan-aware clipping | R1 |
| `levanter.schedule` | Step-indexed value schedules and BatchSchedule for variable batch size | R1 |
| `levanter.shapes` | NamedShapeSpec and conversion to raw shapes | R2 |
| `levanter.sharding` | Reads partition specs and mesh axes from arrays and tracers | R2 |
| `levanter.store` | Package marker for on-disk cache and array stores | R1 |
| `levanter.store.cache` | TreeCache: Zephyr-built sharded dataset caches, ledgers, readers, and writers | R1 |
| `levanter.store.jagged_array` | TensorStore-backed JaggedArrayStore of variable-length arrays | R1 |
| `levanter.store.tree_store` | TreeStore: a pytree of JaggedArrayStores for batched ragged data | R1 |
| `levanter.tensorstore_serialization` | TensorStore OCDBT array save and restore with per-process write planning | R2 |
| `levanter.testing` | Package marker for shared test support | package |
| `levanter.testing.chat_templates` | Multi-tool Jinja chat template constant for tests | test support |
| `levanter.testing.cpu_devices` | Runs a test script in a fresh interpreter with a fixed CPU device count | test support |
| `levanter.testing.eight_device_checkpoints` | Checkpoint scenarios on eight CPU devices and two processes in spawned interpreters | test support |
| `levanter.testing.helpers` | Test helpers: skip markers, test meshes, toy MLP, config and tree checks | test support |
| `levanter.testing.model_configs` | Tiny Llama config for tests | test support |
| `levanter.testing.tiny_corpus` | Writes a tiny JSONL corpus and builds its cache for tests | test support |
| `levanter.testing.tokenizer` | Stages the checked-in GPT-2 tokenizer in a loadable directory | test support |
| `levanter.testing.toy_lm` | Toy LM head model and config for tests | test support |
| `levanter.tokenizers` | MarinTokenizer protocol, HF Rust tokenizer backend, chat-template rendering with masks | R1 |
| `levanter.tracker` | Re-exports Tracker classes and global logging functions | R2 |
| `levanter.tracker.background` | Runs a tracker's calls on a background thread, swallowing failures | R2 |
| `levanter.tracker.finelog_metrics` | Nonblocking writer of typed metric rows to a Finelog server | R2 |
| `levanter.tracker.helpers` | Logs optimizer hyperparameters; git root, pip freeze, and timing helpers | R2 |
| `levanter.tracker.histogram` | Histogram and summary-stat types with sharded histogram computation | R2 |
| `levanter.tracker.json_file` | Tracker accumulating metrics and writing a JSON file at finish | `levanter.tracker.json_file` |
| `levanter.tracker.json_logger` | Tracker logging metrics as JSON lines to a Python logger | R2 |
| `levanter.tracker.telemetry` | Tracker exporting training phases and snapshots to Finelog, with stall diagnostics | R2 |
| `levanter.tracker.tracker` | Tracker interface; composite, no-op, and dict trackers; TrackerConfig registry | R2 |
| `levanter.tracker.tracker_fns` | Global tracker state and log, jit_log, summary, and hyperparameter functions | R2 |
| `levanter.tracker.trackio` | Trackio tracker and config | R2 |
| `levanter.tracker.wandb` | Weights & Biases tracker and config, with artifact and run-name handling | R2 |
| `levanter.trainer` | Trainer loop: steps, hooks, checkpoint and tracker wiring; TrainerConfig and initialize | R2 |
| `levanter.trainer_state` | TrainerState pytree, trainable-parameter filtering, optimizer init, and train step | R2 |
| `levanter.utils` | Package marker | R1 |
| `levanter.utils.activation` | Activation function enum, including quick GELU and squared ReLU | R1 |
| `levanter.utils.background_iterable` | Iterates an iterable on a background thread through a queue | R2 |
| `levanter.utils.byte_budget` | Asyncio scheduler limiting in-flight host bytes against a target | R2 |
| `levanter.utils.cloud_utils` | Context manager that writes to a local temp dir, then uploads to a URL | R2 |
| `levanter.utils.datetime_utils` | Parses and encodes timedelta strings for configs | R2 |
| `levanter.utils.flop_utils` | Analytic forward FLOPs per token for LM configs | R2 |
| `levanter.utils.hardware_topology` | Device topology summary: TPU coordinates and nvidia-smi topology matrix | R2 |
| `levanter.utils.hf_utils` | UTF-8 byte length of a tokenizer token | R2 |
| `levanter.utils.jax_utils` | JAX helpers: CPU-device contexts, multihost sync and barriers, tree key paths, sharding | R1 |
| `levanter.utils.logging` | Logging setup via rigging, loading-time iterator, transformers warning suppression | R2 |
| `levanter.utils.mesh` | MeshConfig and JAX mesh construction from ICI and DCN axis sizes | R1 |
| `levanter.utils.profile_dirs` | Resolves and mirrors profiler output directories for a W&B run | `poorin.profiling.cli` |
| `levanter.utils.py_utils` | CPU core count, dataclass default init, global RNG seeding, fail-safe JSON encoder | R1 |
| `levanter.utils.stat_utils` | RunningMean accumulator for evaluation metrics | R2 |
| `levanter.utils.thread_utils` | Blocking wait on coroutines from sync code; async iterator wrapper | R1 |
| `levanter.utils.tree_utils` | Pytree helpers: inference mode, one-level flatten with keys, key-path strings | R2 |
| `levanter.utils.types` | ComputeLossFunction protocol for model loss functions | R2 |

### poorin

| Module | Purpose | Reached from |
|---|---|---|
| `poorin` | Package marker | package |
| `poorin.datakit` | Datakit package: standard Parquet format notes and `part-NNNNN-of-MMMMM` filename helper | R1 |
| `poorin.datakit.canonical` | Package marker | package |
| `poorin.datakit.canonical.fineweb_edu` | FineWeb-Edu download step from Hugging Face at a pinned revision | `poorin.datakit.canonical.fineweb_edu` |
| `poorin.datakit.chat_template` | Marin chat template string shared by tokenizer export and Datakit rendering | R1 |
| `poorin.datakit.copartitioned` | Co-partitioned input/output shard layout and source manifest for Datakit stages | `poorin.processing.classification.deduplication.fuzzy_dups` |
| `poorin.datakit.decon` | Bloom-filter n-gram decontamination of normalized data against eval text, emitting attributes | `poorin.datakit.decon` |
| `poorin.datakit.download` | Package marker | R1 |
| `poorin.datakit.download.http_session` | Retrying `requests.Session` factory for HTTP downloaders | `poorin.datakit.download.nemotron_v1` |
| `poorin.datakit.download.huggingface` | Streams a Hugging Face dataset repo to an fsspec path; download step builder | R1 |
| `poorin.datakit.download.nemotron_v1` | Nemotron-CC v1 download from Common Crawl and per-split normalize steps | `poorin.datakit.download.nemotron_v1` |
| `poorin.datakit.download.zstd_jsonl` | Streaming reader for zstd-compressed JSONL download sources | `poorin.datakit.download.nemotron_v1` |
| `poorin.datakit.normalize` | Normalizes raw JSONL/Parquet into Datakit Parquet with content ids, dedup, sorted partitions | R1 |
| `poorin.datakit.source_key` | Strips Marin bucket and prefix roots to give Datakit artifacts stable source keys | R1 |
| `poorin.evaluation` | Package marker | R1 |
| `poorin.evaluation.endpoint` | OpenAI-compatible endpoint and tokenizer types for a served model under evaluation | `poorin.evaluation.lm_eval` |
| `poorin.evaluation.eval_measurements` | Converts eval run records into statistics-engine measurements; owns harness output shapes | `poorin.evaluation.eval_measurements` |
| `poorin.evaluation.eval_stats` | Eval statistics: measurements, coverage, Wilson and attrition-widened intervals, cohort selection | `poorin.evaluation.eval_stats` |
| `poorin.evaluation.evaluation_config` | `EvalTaskConfig` lm-eval task spec, results-directory identity, Levanter task conversion | R1 |
| `poorin.evaluation.lm_eval` | Runs lm-evaluation-harness in an isolated uv env against an OpenAI-compatible endpoint | `poorin.evaluation.lm_eval` |
| `poorin.evaluation.metric_selection` | Picks the primary lm-eval metric and extraction filter for a task | `poorin.evaluation.metric_selection` |
| `poorin.evaluation.model_loading` | Loads a Levanter or Hugging Face checkpoint for Levanter-based eval entrypoints | `poorin.evaluation.trace_labeled_eval` |
| `poorin.evaluation.records` | Pydantic eval-run record schema and its `record.json` object-store read and write | `poorin.evaluation.records` |
| `poorin.evaluation.rollouts` | Builds the FineStore rollout table from archived eval samples and steps | `poorin.evaluation.rollouts` |
| `poorin.evaluation.trace_labeled_eval` | Levanter labeled-loss evaluation on OpenAI-style agent traces with resumable results | `poorin.evaluation.trace_labeled_eval` |
| `poorin.execution` | Package marker | R1 |
| `poorin.execution.artifact` | `Artifact` base type, its on-disk record, and the fingerprint drift check | R1 |
| `poorin.execution.build_context` | Ambient build context resolving deferred artifact versions through a `VersionCodex` | R2 |
| `poorin.execution.fingerprint` | Canonical JSON encoding and hashing of step configs for drift detection | R1 |
| `poorin.execution.lazy` | Lazy artifact steps: `name@version` handles, step context, lowering to the step runner | R1 |
| `poorin.execution.remote` | `@remote` decorator that submits a step function as a Fray job | R1 |
| `poorin.execution.step_runner` | DAG scheduler running `StepSpec`s with caching, locking, status writes, Fray dispatch | R1 |
| `poorin.execution.step_spec` | `StepSpec` dataclass: step name, deps, hash attributes, output path, function | R1 |
| `poorin.execution.step_status` | Per-step status files and heartbeat lock files under each output path | R1 |
| `poorin.experiment` | Package marker; docstring points to the `data`, `sweep`, and `train` submodules | R1 |
| `poorin.experiment.cli` | Click options (`--version`, `--override`, `--run`) that plan or build an experiment graph | R1 |
| `poorin.experiment.data` | Lazy dataset builders: HF and raw downloads, tokenized, pretokenized, and mixture handles | R1 |
| `poorin.experiment.namespacing` | Prefixes mutable training artifact names with the user to avoid path collisions | R2 |
| `poorin.experiment.sweep` | Grid enumeration and one trial handle per grid point for hyperparameter sweeps | `poorin.experiment.sweep` |
| `poorin.experiment.train` | `train_lm`: assembles a Levanter LM training step with mesh, checkpointer, evals | R1 |
| `poorin.processing` | Package marker | R1 |
| `poorin.processing.classification` | Package marker | `poorin.processing.classification.deduplication.exact` |
| `poorin.processing.classification.consolidate` | Filters documents or removes spans by joining co-partitioned attribute files | `poorin.processing.classification.consolidate` |
| `poorin.processing.classification.deduplication` | Package marker | `poorin.processing.classification.deduplication.exact` |
| `poorin.processing.classification.deduplication.connected_components` | Hash-to-Min connected components over Zephyr datasets for fuzzy-duplicate clustering | `poorin.processing.classification.deduplication.fuzzy_dups` |
| `poorin.processing.classification.deduplication.dedup_commons` | Shared dedup helpers: input discovery, batch loading, dup counters, W&B summary | `poorin.processing.classification.deduplication.exact` |
| `poorin.processing.classification.deduplication.exact` | Exact paragraph and document deduplication by text hash, writing duplicate attributes | `poorin.processing.classification.deduplication.exact` |
| `poorin.processing.classification.deduplication.fuzzy_dups` | LSH-graph connected components over MinHash buckets, producing fuzzy-duplicate cluster attributes | `poorin.processing.classification.deduplication.fuzzy_dups` |
| `poorin.processing.classification.deduplication.fuzzy_minhash` | Computes dupekit MinHash LSH bucket attributes per normalized Datakit shard | `poorin.processing.classification.deduplication.fuzzy_minhash` |
| `poorin.processing.classification.deduplication.fuzzy_verification` | Full-text checks for fuzzy-duplicate candidates: n-gram Jaccard, containment, line-count ratio | `poorin.processing.classification.deduplication.verify_fuzzy_dups` |
| `poorin.processing.classification.deduplication.verify_fuzzy_dups` | Verifies fuzzy-duplicate clusters against full text and writes verified duplicate attributes | `poorin.processing.classification.deduplication.verify_fuzzy_dups` |
| `poorin.processing.tokenize` | Re-exports tokenize configs, mixture config builders, and cache stats readers | R1 |
| `poorin.processing.tokenize._core` | Shared tokenization internals: file bundling, id-preserving batching, oversized-record splitting | R1 |
| `poorin.processing.tokenize.attributes` | Stage A: tokenizes normalized Datakit data into co-partitioned `input_ids` attribute Parquet | R1 |
| `poorin.processing.tokenize.cache_stats` | Reads per-split `.stats.json` document and token counts of tokenized caches | R1 |
| `poorin.processing.tokenize.data_configs` | Builds Levanter mixture data configs from tokenized sources; tokenizer vocab sizes | R1 |
| `poorin.processing.tokenize.download_pretokenized` | Downloads a pretokenized Levanter cache from Hugging Face as a dataset source | R1 |
| `poorin.processing.tokenize.store_builder` | Stage B: writes tokenized records into a sharded Levanter cache with stats | R1 |
| `poorin.processing.tokenize.tokenize` | Legacy end-to-end tokenization of files or HF datasets into a Levanter cache | R1 |
| `poorin.profiling` | Package marker for JAX and xprof profile ingestion and query tools | `poorin.profiling.cli` |
| `poorin.profiling.cli` | CLI to summarize, query, and compare JAX profile traces before and after | `poorin.profiling.cli` |
| `poorin.profiling.compare_bundle` | One-shot compare, regression-track, and report run over two profile summaries | `poorin.profiling.cli` |
| `poorin.profiling.ingest` | Downloads W&B profile artifacts and summarizes Perfetto/Chrome traces into profile summaries | `poorin.profiling.cli` |
| `poorin.profiling.publish` | Publishes profile summaries and reports as W&B artifacts | `poorin.profiling.cli` |
| `poorin.profiling.query` | Queries and diffs profile summaries: hot ops, step classes, gap contexts | `poorin.profiling.cli` |
| `poorin.profiling.report` | Markdown report generation from profile summaries | `poorin.profiling.cli` |
| `poorin.profiling.schema` | Versioned dataclass schema for profile summaries and its dict parser | `poorin.profiling.cli` |
| `poorin.profiling.semantics` | Op-name canonicalization, semantic family classification, shape signatures, FLOP proxies | `poorin.profiling.cli` |
| `poorin.profiling.trace_summary` | Engine turning trace events into a `ProfileSummary`: step times, hot ops, gaps | `poorin.profiling.cli` |
| `poorin.profiling.tracking` | Before/after profile regression thresholds and JSONL regression history | `poorin.profiling.cli` |
| `poorin.profiling.xplane` | Ingests XPlane protobuf profiles directly and through xprof tables | `poorin.profiling.cli` |
| `poorin.pytest_timeout_guard` | pytest plugin that hard-kills a test when pytest-timeout's signal handler fails | `poorin.pytest_timeout_guard` |
| `poorin.scaling_laws` | Re-exports isoFLOP analysis, scaling plots, and TPU slice helpers | `poorin.scaling_laws` |
| `poorin.scaling_laws.eval_metrics_reader` | Reads `tracker_metrics.jsonl` eval records from finished runs, with W&B backfill | `poorin.scaling_laws.eval_metrics_reader` |
| `poorin.scaling_laws.isoflop_analysis` | IsoFLOP scaling-law fitting and compute-optimal config prediction from run records | `poorin.scaling_laws.isoflop_analysis` |
| `poorin.scaling_laws.scaling_plots` | Plotly isoFLOP and scaling-fit plots, saved locally or uploaded to W&B | `poorin.scaling_laws.scaling_plots` |
| `poorin.scaling_laws.tpu_utils` | TPU v4/v5p memory specs and slice-size selection | `poorin.scaling_laws.tpu_utils` |
| `poorin.training` | Lazy re-export of LM and DPO pod training configs and run functions | R1 |
| `poorin.training.run_environment` | Training-process env defaults (git commit, HF, TPU logging) and accelerator uv extras | R1 |
| `poorin.training.training` | `LevanterCheckpoint` artifact, LM/DPO train configs, env setup, in-process Levanter launch | R1 |
| `poorin.utilities` | Package marker | R1 |
| `poorin.utilities.json_encoder` | JSON encoder for configs: dataclasses, dtypes, enums, paths, timedeltas | R1 |
| `poorin.utilities.provenance` | Writes provenance JSON sidecars for raw downloads | R1 |
| `poorin.utilities.wandb_utils` | W&B run init when an API key is set; entity and project defaults | `poorin.profiling.cli` |

### zephyr

| Module | Purpose | Reached from |
|---|---|---|
| `zephyr` | Package docstring only; directs callers to import from submodules | R1 |
| `zephyr._test_helpers` | Picklable sample dataclass for serialization round-trip tests | test support |
| `zephyr.context` | ZephyrContext: runs pipelines on a dedicated or shared worker pool | R1 |
| `zephyr.coordinator` | Coordinator actor: dispatches shard tasks to polling workers, tracks pipeline execution | R1 |
| `zephyr.counters` | User-defined pipeline- and stage-scoped counters for task code | R1 |
| `zephyr.dashboard` | Package marker for the pipeline dashboard | R1 |
| `zephyr.dashboard.app` | JSON API and standalone HTML dashboard ASGI app for pipeline status | R1 |
| `zephyr.dashboard.coordinator` | Builds dashboard responses from coordinator state | R1 |
| `zephyr.dataset` | Lazy Dataset API and logical ops: map, filter, group-by, reduce, join, write | R1 |
| `zephyr.expr` | Column expressions for filter and projection pushdown, convertible to pyarrow | R1 |
| `zephyr.input_file` | InputFileSpec describing one input file for datasets, plans, and readers | R1 |
| `zephyr.memory_budget` | Memory-budget arithmetic for shuffle flush thresholds and merge fan-in | R1 |
| `zephyr.memory_store` | Worker-hosted in-memory key-value tables partitioned like a source Dataset | R1 |
| `zephyr.parquet_scan` | Polars Parquet scan with correct addressing for S3-compatible stores | R1 |
| `zephyr.plan` | Physical planner: fuses logical ops into stages and executes a stage | R1 |
| `zephyr.readers` | Readers for JSONL, Parquet, Vortex, and zip inputs via fsspec | R1 |
| `zephyr.resources` | Detects process memory and CPU limits from cgroups or /proc | R1 |
| `zephyr.runners` | Shard execution strategies: in-process InlineRunner and per-shard SubprocessRunner | R1 |
| `zephyr.shard_keys` | Deterministic key encoding and hashing for shard routing | R1 |
| `zephyr.shard_subprocess` | Child entry point that runs one shard for SubprocessRunner | `zephyr.shard_subprocess` |
| `zephyr.shuffle` | Scatter writer and sorted-merge reader for disk-backed Parquet shuffles | R1 |
| `zephyr.stage_io` | Worker-side shard task and result types, throughput stats, stage output writers | R1 |
| `zephyr.stats` | Finelog stats schemas and writer for stage, worker, shuffle, execution metrics | R1 |
| `zephyr.testing` | Package marker for Zephyr test drivers | test support |
| `zephyr.testing.context` | Context manager yielding a small ZephyrContext for memory-store tests | test support |
| `zephyr.testing.coordinator` | Builds a test coordinator and loads a stage for coordinator tests | test support |
| `zephyr.worker` | Worker actor that polls the coordinator and runs shard tasks in threads | R1 |
| `zephyr.worker_context` | Worker context exposed to task code, with counter snapshot types | R1 |
| `zephyr.writers` | Writers for JSONL, Parquet, Vortex, and binary outputs, plus a threaded batch writer | R1 |

### rigging

| Module | Purpose | Reached from |
|---|---|---|
| `rigging` | Package marker | R1 |
| `rigging.cache` | Content hashes of files, directories, and dependencies for compiled-artifact cache keys | R2 |
| `rigging.config_discovery` | Finds the project root and cluster YAML config files | R1 |
| `rigging.filesystem` | Package marker for storage paths, cluster data config, guarded filesystem access | R1 |
| `rigging.filesystem.atomic` | Atomic local downloads and object-store write-then-rename | R2 |
| `rigging.filesystem.buckets` | Routes each bucket URL to its declared backend filesystem (GCS, CoreWeave, R2) | R1 |
| `rigging.filesystem.bulk_deletion` | Batched, throttled bulk deletion attached to S3 and GCS filesystems | R1 |
| `rigging.filesystem.cluster_config` | Cluster DataConfig: bucket and region resolution, storage prefix, temp buckets, GCS location checks | R1 |
| `rigging.filesystem.conditional_object` | Compare-and-swap reads and writes of one object on local, GCS, or S3 | R1 |
| `rigging.filesystem.cross_region` | Cross-region GCS read guard and process-wide transfer byte budget | R1 |
| `rigging.filesystem.distributed_lock` | Lease-based distributed locks on GCS, S3, local files, or fsspec | R1 |
| `rigging.filesystem.factory` | Guarded url_to_fs, open_url, filesystem with GCS region guard and S3 timeouts | R1 |
| `rigging.filesystem.hashing` | Streaming MD5 of local and object-store files | `rigging.fsutil.cli` |
| `rigging.filesystem.listing_cache` | Process-wide fsspec directory-listing cache defaults | R1 |
| `rigging.filesystem.mirror` | mirror:// filesystem copying files from other-region marin-* GCS buckets on first read | `rigging.filesystem.mirror` |
| `rigging.filesystem.paged_listing` | Paged single-level and recursive object listings for S3 and GCS | R1 |
| `rigging.filesystem.protocols` | Normalizes fsspec protocol names | R1 |
| `rigging.filesystem.s3_compat` | S3-compatible store setup: credentials, endpoints, addressing, request timeouts for CoreWeave and R2 | R1 |
| `rigging.filesystem.s3_errors` | Classifies retryable S3 and object-store transport errors | R2 |
| `rigging.filesystem.storage_path` | StoragePath value type: parsed storage URL with I/O verbs and path helpers | R1 |
| `rigging.filesystem.transfer` | Copy, move, and sync plans and execution across routed filesystems | `rigging.fsutil.cli` |
| `rigging.fsutil` | Package marker for the fsutil bucket browser and CLI | `rigging.fsutil.cli` |
| `rigging.fsutil.cli` | fsutil command line: list, read, size, copy, sync, remove across buckets | `rigging.fsutil.cli` |
| `rigging.fsutil.compression` | Compression detection for file previews | `rigging.fsutil.compression` |
| `rigging.fsutil.deletion` | Streamed parallel deletion of object-store prefixes | `rigging.fsutil.deletion` |
| `rigging.fsutil.listing` | Directory-listing entries and file previews over bucket-routed filesystems | `rigging.fsutil.listing` |
| `rigging.fsutil.parquet` | Parquet schema, summary, and row previews for fsutil | `rigging.fsutil.parquet` |
| `rigging.fsutil.render` | Renders sizes, listings, JSON tables, and file previews as text | `rigging.fsutil.render` |
| `rigging.fsutil.tui` | Curses bucket browser and file viewer for fsutil browse and show | `rigging.fsutil.tui` |
| `rigging.fsutil.usage` | Object-storage usage scans and deletion-candidate reports | `rigging.fsutil.usage` |
| `rigging.fsutil.verified_copy` | Resumable, hash-verified prefix copies across routed filesystems | `rigging.fsutil.verified_copy` |
| `rigging.log_setup` | Logging setup: level-prefix formatter, log ring buffer, fault handler | R1 |
| `rigging.provenance` | Build provenance: content tree hash plus git, user, and launch context | R1 |
| `rigging.telemetry` | Bounded, best-effort counters, gauges, histograms, and events exported to Finelog | R2 |
| `rigging.telemetry.serialization` | Bounded JSON serialization and validation for telemetry records | R2 |
| `rigging.testing` | Recording telemetry transport for tests | test support |
| `rigging.timing` | Time types and helpers: Deadline, Duration, Timer, rate limiters, backoff retry | R1 |

### fray

| Module | Purpose | Reached from |
|---|---|---|
| `fray` | Package docstring only; directs callers to import from submodules | R1 |
| `fray.actor` | Actor protocols: handles, futures, method calls, actor groups, current-actor context | R1 |
| `fray.client` | Client and JobHandle protocols for job submission, plus wait_all | R1 |
| `fray.current_client` | Gets and sets the active Fray client, defaulting to LocalClient | R1 |
| `fray.device_flops` | Peak FLOPS table per device type and dtype | R1 |
| `fray.local_backend` | LocalClient: in-process Fray backend running jobs as threads or subprocesses | R1 |
| `fray.types` | Job request, resource (CPU/GPU/TPU), environment, and entrypoint dataclasses | R1 |

### finelog

| Module | Purpose | Reached from |
|---|---|---|
| `finelog` | Package marker | R2 |
| `finelog.client` | Re-exports LogClient, errors, policy, and table-spec types | R2 |
| `finelog.client.log_client` | LogClient and Table: buffered, retried writes and queries against a Finelog server | R2 |
| `finelog.embedded` | In-process Finelog server from the optional native Rust extension | `finelog.embedded` |
| `finelog.errors` | Typed exceptions translated from Finelog ConnectRPC error codes | R2 |
| `finelog.policy` | Per-namespace storage retention policy sent at table registration | R2 |
| `finelog.rpc` | Package marker for generated proto bindings | R2 |
| `finelog.rpc.finelog_stats_connect` | Generated ConnectRPC client and server stubs for the stats service | R2 |
| `finelog.rpc.finelog_stats_pb2` | Generated protobuf messages for the stats service | R2 |
| `finelog.rpc.logging_connect` | Generated ConnectRPC client and server stubs for the log service | R2 |
| `finelog.rpc.logging_pb2` | Generated protobuf messages for the log service | R2 |
| `finelog.schema` | Table Schema and Column dataclasses with proto and Arrow conversion | R2 |
| `finelog.table_spec` | Versioned table layout, partitioning, and operating-policy declarations | R2 |
| `finelog.types` | Log-writer protocol, log-level parsing, and retryable-error check | R2 |

### finestore

| Module | Purpose | Reached from |
|---|---|---|
| `finestore` | Package docstring only: transactional Parquet tables on conditional object storage | R2 |
| `finestore.cache` | PersistentKvCache: in-memory tier over FineStore named-object byte cache | R2 |
| `finestore.commit` | Manifest snapshots and compare-and-swap commit publication | R2 |
| `finestore.compaction` | Manifest-driven compaction that merges a table's shards | R2 |
| `finestore.eval` | EvalSample and rollout schemas with typed FineStore writer and reader for eval outputs | `poorin.evaluation.rollouts` |
| `finestore.layout` | Format-v2 metadata models and object-key layout for FineStore archives | R2 |
| `finestore.migrations` | Ordered on-disk format migrations, run before writes | R2 |
| `finestore.migrations.m0001_manifest` | Upgrades listing-based v1 archives to transactional v2 manifests | R2 |
| `finestore.reader` | Snapshot reads, filtering, and deduplication over manifest-selected shards | R2 |
| `finestore.schema` | Derives pyarrow schemas from pydantic models or dataclasses | `poorin.evaluation.rollouts` |
| `finestore.shard_writer` | Streams Arrow batches into one atomically published Parquet shard | R2 |
| `finestore.store` | Transactional FineStore writer with buffered level-zero batches and flush scheduling | R2 |

### dupekit

| Module | Purpose | Reached from |
|---|---|---|
| `dupekit` | Proxy re-exporting the Rust text-dedup extension, with an ImportError stub fallback | R1 |

### experiments

| Module | Purpose | Reached from |
|---|---|---|
| `experiments` | Package marker | package |
| `experiments.datasets` | Package marker; docstring sets conventions for dataset catalog modules | package |
| `experiments.evals` | Package marker | package |
| `experiments.evals.task_configs` | Named lm-eval task lists (`EvalTaskConfig`) by suite: core, MMLU, leaderboard, multilingual | `experiments.evals.task_configs` |
| `experiments.grug` | Package marker | R2 |
| `experiments.grug.base` | Package marker | R2 |
| `experiments.grug.base.launch` | grug-base trial run as a lazy checkpoint artifact on one RTX 5090 | R2 |
| `experiments.grug.base.model` | Grug Llama-style transformer: config, attention, MLP, RMSNorm, blocks | R2 |
| `experiments.grug.base.train` | Grug training loop: configs, data loader, train step, evals, Fray dispatch | R2 |
| `experiments.grug.checkpointing` | Restores grug train state or weights from the newest loadable checkpoint | R2 |
| `experiments.grug.dispatch` | Submits a grug training entrypoint as a Fray job with forwarded runtime env | R2 |
| `experiments.grug.fast_track` | Package marker | `experiments.grug.fast_track.heuristic` |
| `experiments.grug.fast_track.adamh` | Local AdamH optimizer transform with sharding pinning before the hyperball norm | `experiments.grug.fast_track.heuristic` |
| `experiments.grug.fast_track.grugmuon_stacked` | Distributed Newton-Schulz orthogonalization for stacked grug MoE expert weights | `experiments.grug.fast_track.heuristic` |
| `experiments.grug.fast_track.heuristic` | MoE heuristic setting MuonH/Adam LR, epsilon, beta2 from token budget and batch | `experiments.grug.fast_track.heuristic` |
| `experiments.grug.fast_track.optimizer` | `GrugMoeMuonHConfig`: MuonH plus Adam with gate/router weight-decay masking | `experiments.grug.fast_track.heuristic` |
| `experiments.grug.sharding_dump` | Dumps grug parameter and optimizer-state sharding specs as JSON artifacts | R2 |
| `experiments.llama` | Llama 3 tokenizers, model size ladder (nano to 70B), and train configs | R1 |
| `experiments.marin_tokenizer` | Builds the Marin tokenizer: Llama 3 plus Olmo2-style chat template and special tokens | R1 |
| `experiments.references.completed_adamh` | CompletedAdamH heuristic deriving model, batch, and optimizer configs from a compute budget | `experiments.references.completed_adamh` |
| `experiments.simple_train_config` | `SimpleTrainConfig` dataclass of resources, batch size, steps, optimizer hyperparameters | R1 |
| `experiments.tutorials.train_tiny_model` | Tutorial: trains a tiny Llama with device and dataset chosen on the CLI | R1 |

### tools

| Module | Purpose | Reached from |
|---|---|---|
| `tools` | Package marker | package |
| `tools.affected_tests` | Selects and runs unit tests affected by a change via static import analysis | `tools.affected_tests` |
| `tools.changes` | Lists uncommitted and staged changed files via git | `tools.changes` |
| `tools.lint` | Lints and formats files: size, conflict markers, license headers, ruff, ty | `tools.lint` |
| `tools.prune_check` | Prune invariant check: roots import, ty, lint, tests, training, one metrics row | `tools.prune_check` |
| `tools.reachability` | Static first-party import graph, root closure, import chains, dominated line weights | `tools.reachability` |
