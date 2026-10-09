# Poorin

<!--marin-intro-start-->

Poorin is a hard fork of [Marin](https://github.com/marin-community/marin) for one researcher
training language models, video world models, and robot policies (VLAs and WAMs) on a local RTX
5090, with occasional cloud runs. Upstream Marin is a reference only: nothing is merged from it,
and the tag `upstream-baseline` marks the tree the fork started from.

The fork removed what a single workstation does not use: the Iris cluster scheduler, the TPU and
Docker launchers, fleet operations, the RL and serving stacks, and Marin's experiment configs.
Marin's core package is now `poorin`. Measured by `tools.prune_check` on this machine:

| | `upstream-baseline` | Now |
|---|---:|---:|
| First-party lines of code | 385,058 | 133,536 |
| Python environment | 12.2 GB | 8.0 GB |
| Import time of the tiny-model training script | 7.3 s | 3.8 s |

<!--marin-intro-end-->

Each change and the reason for it is in [`docs/poorin/decisions.md`](docs/poorin/decisions.md),
with approved designs in [`docs/poorin/specs/`](docs/poorin/specs/), implementation plans in
[`docs/poorin/plans/`](docs/poorin/plans/), and one row of measurements per change in
[`docs/poorin/prune-metrics.csv`](docs/poorin/prune-metrics.csv).

## Libraries

| Library | Role |
|---|---|
| [`lib/poorin`](lib/poorin) | Experiment framework: lazy artifacts, the experiment command line, data pipelines, evaluation |
| [`lib/levanter`](lib/levanter) | Training library on JAX |
| [`lib/haliax`](lib/haliax) | Named tensors for JAX |
| [`lib/zephyr`](lib/zephyr) | Dataset processing, with a status dashboard per pipeline |
| [`lib/fray`](lib/fray) | Job dispatch; `LocalClient` runs jobs on this machine |
| [`lib/rigging`](lib/rigging) | Filesystems, storage paths, logging, and config shared by the others |
| [`lib/finelog`](lib/finelog) | Client for the Finelog log and metrics store |
| [`lib/finestore`](lib/finestore) | Transactional Parquet tables and named objects on object storage |
| [`lib/dupekit`](lib/dupekit) | Text de-duplication kernels in Rust |

## Setup

<!--marin-first-steps-start-->

```bash
git clone https://github.com/jasgcode/Poorin.git
cd Poorin
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
```

Run every project command with `uv run --no-sync`. A syncing `uv run` installs no extras, so it
removes the CUDA build of JAX. [Local GPU Setup](docs/tutorials/local-gpu.md) covers the RTX
5090 environment, and [First Experiment](docs/tutorials/first-experiment.md) walks through a
tiny training run.

<!--marin-first-steps-end-->

## Running experiments

An experiment is a Python function that returns lazy artifact handles: tokenized datasets,
checkpoints, evaluation results. Run without `--run`, its script prints the plan, one line per
artifact with its cache state; with `--run` it builds what is missing. `--set` overrides any
field of the experiment's config, and `--sweep` runs one trial per value:

```bash
uv run --no-sync python experiments/grug/base/launch.py --version dev --set steps=200
uv run --no-sync python experiments/grug/base/launch.py --version dev --set steps=200 --run
uv run --no-sync python experiments/grug/base/launch.py --version dev \
    --sweep 'optimizer.learning_rate=[1e-3, 3e-3]' --run
```

An existing output whose recipe changed is refused before anything runs; `--accept-drift NAME`
keeps it. [Lazy artifacts](docs/explanations/lazy-artifacts.md) explains versions, caching,
overrides, and drift.

<!--marin-example-start-->

A complete script that tokenizes [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
and trains a tiny model on the CPU:

```python
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from poorin.execution.lazy import run
from poorin.experiment.data import tokenized
from poorin.experiment.train import train_lm

from experiments.llama import llama_nano
from experiments.marin_tokenizer import marin_tokenizer

# 1. Tokenize the dataset as a lazy handle — nothing downloads yet.
tinystories_tokenized = tokenized(
    name="tokenized/tinystories",
    source="roneneldan/TinyStories",
    tokenizer=marin_tokenizer,
    version="2026.06.28",
    sample_count=1000,  # cap at 1 000 samples per shard to keep the tutorial fast
)

# 2. Train the model — depends on the tokenized dataset above.
nano_tinystories_model = train_lm(
    name="checkpoints/marin-nano-tinystories",
    version="2026.06.28",
    model=llama_nano,
    optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
    # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
    datasets={tinystories_tokenized: 1.0},
    batch_size=4,
    seq_len=llama_nano.max_seq_len,
    num_train_steps=100,
    z_loss_weight=None,
    evals=None,  # no point evaluating such a tiny model
    resources=ResourceConfig.with_cpu(),
)

if __name__ == "__main__":
    run(nano_tinystories_model)
```

The training step depends on the tokenized dataset, so `run` builds the dataset first.
[`experiments/tutorials/train_tiny_model.py`](experiments/tutorials/train_tiny_model.py) is a
fuller version, with options for the device (CPU or GPU) and the dataset (TinyStories,
WikiText, or FineWeb-Edu).

<!--marin-example-end-->

## Development

```bash
uv run --no-sync python -m tools.lint --fix        # lint and format the uncommitted change
uv run --no-sync python -m tools.affected_tests    # tests the uncommitted change affects
uv run --no-sync python -m tools.prune_check --phase <phase> --step "<label>"  # full check
```

[Development](docs/dev-guide/development.md) describes the workflow and the checks each commit
passes. [`AGENTS.md`](AGENTS.md) is the guide for coding agents, with task playbooks in
[`.agents/skills/`](.agents/skills/).

## Contributions

Poorin is a personal research codebase and does not take outside contributions for now. It is
licensed under Apache 2.0, so forks are welcome.

## Acknowledgments

Poorin is built on [Marin](https://github.com/marin-community/marin), developed by
[Stanford CRFM](https://crfm.stanford.edu/) and [Open Athena](https://openathena.ai/), and on
[Levanter](https://github.com/stanford-crfm/levanter) and
[Haliax](https://github.com/stanford-crfm/haliax). Their authors are listed in
[`AUTHORS.md`](AUTHORS.md) and in Levanter's and Haliax's `AUTHORS.md` and `CONTRIBUTORS.md`.
The code is licensed under the [Apache License 2.0](LICENSE).
