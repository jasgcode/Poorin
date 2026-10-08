# Training a Language Model

This tutorial walks through training a language model with Marin's lazy artifact API,
using a 1.4B-parameter Llama trained on a sample of FineWeb-Edu as a concrete example.

## Prerequisites

- Basic [installation](installation.md)
- A machine with GPUs set up as in [Setting up a Local GPU Environment](local-gpu.md)

## How a training script is structured

A Marin training script builds a lazy `ArtifactStep[LevanterCheckpoint]` handle, lowers it to
a runnable step graph, and passes it to `StepRunner`. Nothing executes at import time.

```python
from poorin.execution.lazy import ArtifactStep, lower
from poorin.execution.step_runner import StepRunner
from poorin.experiment.train import train_lm
from poorin.training.training import LevanterCheckpoint

def build() -> ArtifactStep[LevanterCheckpoint]:
    ...  # assemble the checkpoint handle

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

`StepRunner.run` walks the dependency graph, checks the cache for each step, and runs any
that are missing or explicitly forced. Dataset tokenization runs before training; a step
that already succeeded is skipped.

## Defining the model

Choose a Levanter model configuration. The full architecture is stated as literals so it
enters the artifact's fingerprint:

```python
from levanter.models.llama import LlamaConfig

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x, Chinchilla-optimal for ~1.4B parameters
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

llama_1_4b = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)
```

Pre-defined configurations for common sizes live in `experiments/llama.py`.

## Building the data mixture

Training data is expressed as a dict of `ArtifactStep[TokenizedCache]` handles to weights. Pass
this dict as `datasets=` to `train_lm`; it assembles the Levanter data mixture internally:

`tokenized` from `poorin.experiment.data` turns a Hugging Face dataset (or raw files under
`POORIN_PREFIX`) into a tokenized-cache handle; the first run downloads and tokenizes it, later
runs reuse the cache:

```python
from poorin.experiment.data import tokenized
from experiments.llama import llama3_tokenizer

fineweb = tokenized(
    "tokenized/fineweb-edu-sample",
    source="HuggingFaceFW/fineweb-edu",  # Hugging Face dataset id
    tokenizer=llama3_tokenizer,
    sample_count=100_000,  # documents per shard; part of the cache's identity
    version="2026.06.28",
)
weighted = {fineweb: 1.0}
```

A mixture is a dict with more than one entry; the weights need not sum to one.

## Calling `train_lm`

`train_lm` takes every identity-bearing decision as a required argument and defaults none
of them. The complete `build()` function:

```python
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from poorin.execution.lazy import ArtifactStep, lower
from poorin.execution.step_runner import StepRunner
from poorin.experiment.train import EvalSuite, train_lm
from poorin.training.training import LevanterCheckpoint
from poorin.experiment.data import tokenized
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama3_tokenizer

TRAIN_RESOURCES = ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G")

def build(*, version: str = "2026.06.28") -> ArtifactStep[LevanterCheckpoint]:
    train = tokenized(
        "tokenized/fineweb-edu-sample",
        source="HuggingFaceFW/fineweb-edu",
        tokenizer=llama3_tokenizer,
        sample_count=100_000,
        version="2026.06.28",
    )
    validation = [
        tokenized(
            "tokenized/wikitext-validation",
            source="dlwh/wikitext_2_detokenized",
            tokenizer=llama3_tokenizer,
            validation=True,
            version="2026.06.28",
        )
    ]
    weighted = {train: 1.0}

    return train_lm(
        name="checkpoints/llama_1b_fineweb_edu",
        version=version,
        model=llama_1_4b,
        optimizer=AdamConfig(
            learning_rate=3e-3,
            weight_decay=0.033,
            warmup=5000,
            min_lr_ratio=0.1,
        ),
        datasets=weighted,
        validation=validation,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=NUM_TRAIN_STEPS,
        z_loss_weight=1e-4,
        evals=EvalSuite(CORE_TASKS, every=10000),
        resources=TRAIN_RESOURCES,
        tags=["LLAMA_1B_FINEWEB_EDU"],
    )

if __name__ == "__main__":
    StepRunner().run([lower(build())])
```

### Required arguments

| Argument | Purpose |
|---|---|
| `name` | Artifact name; forms the output path `{prefix}/{name}/{version}` |
| `version` | Artifact version; bump this to produce a new run without overwriting the old one |
| `model` | Levanter `LmConfig` (architecture and hyperparameters) |
| `optimizer` | Levanter `OptimizerConfig` (learning rate, schedule, weight decay) |
| `datasets` | Dict of `ArtifactStep[TokenizedCache]` handles to weights; `train_lm` assembles the data mixture |
| `validation` | Sequence of `ArtifactStep[TokenizedCache]` handles for held-out loss tracking (optional) |
| `batch_size` | Training batch size in sequences |
| `seq_len` | Sequence length |
| `num_train_steps` | Total number of optimizer steps |
| `z_loss_weight` | Auxiliary z-loss coefficient; pass `None` to omit |
| `evals` | `EvalSuite` or `None`; pass `None` to opt out of harness evals |
| `resources` | The hardware the training job runs on (a runtime arg, excluded from fingerprint) |

`train_lm` owns the mechanical plumbing that is identical across runs: the data-parallel
mesh, the rolling resumption checkpointer, W&B metric replication, and the Fray dispatch of
the training job. None of those are experiment decisions.

## Running the experiment

Save the script (for example as `train_1b.py`) and run it on the machine that has the GPUs:

```bash
POORIN_PREFIX=local_store uv run python train_1b.py
```

Fray's local backend runs the training job on the current machine. `resources` describes
that machine; it does not request hardware from anywhere else. The script above assumes
8 H100s; on a smaller machine, reduce the model size and batch size to fit.

## Monitoring training

W&B receives metrics throughout training. The run name defaults to the `run_id` argument
(when omitted, `train_lm` derives one from the artifact name). Checkpoints are written to
`{prefix}/{name}/{version}/checkpoints/`.

## Memory pressure

If training OOMs, see [Making Things Fit in HBM](../references/hbm-optimization.md) for a
practical tuning checklist covering gradient checkpointing, activation offloading, and
tensor parallelism.

## Reference implementation

`experiments/tutorials/train_tiny_model.py` is a checked-in `train_lm` script: every
decision (model, data, optimizer, token budget, z-loss, evals) is stated inline. It trains
`llama_nano` or `llama_150m` from `experiments/llama.py`; the same structure scales up by
changing the `LlamaConfig` and token budget.
