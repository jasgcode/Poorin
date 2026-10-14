# Poorin

<!--marin-intro-start-->

Poorin is a research codebase for training language models, video world models, and robot
policies (VLAs and WAMs) on one local GPU. It is a hard fork of
[Marin](https://github.com/marin-community/marin) with the cluster parts removed. Every result
gets a name, a version, and a record of the config and code that built it.

<!--marin-intro-end-->

## What is in it

- **An experiment framework** (`lib/poorin`): versioned artifacts, a command line with `--set`
  and `--sweep`, and runs that refuse to reuse a result whose config changed.
- **A language-model template** (`experiments/grug/base`): a transformer and its training loop
  in plain JAX.
- **Video and robot data** (`poorin.episodes`): one format for episodes of frames, states, and
  actions, read as clips.
- **World-model, VLA, and WAM templates** (`experiments/episode_models`): three flow-matching
  models on one training loop.
- **Training, evaluation, and scaling laws** from Levanter and Marin.

## Setup

<!--marin-first-steps-start-->

```bash
git clone https://github.com/jasgcode/Poorin.git
cd Poorin
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
```

Run every command with `uv run --no-sync`. A plain `uv run` re-syncs without the GPU extra and
removes the CUDA build of JAX. [Local GPU Setup](docs/tutorials/local-gpu.md) covers the GPU,
and [First Experiment](docs/tutorials/first-experiment.md) walks through a tiny training run.

<!--marin-first-steps-end-->

## Run something

```bash
# Print the plan, then build it
uv run --no-sync python experiments/grug/base/launch.py --version dev
uv run --no-sync python experiments/grug/base/launch.py --version dev --run

# Change a config field, or run one trial per value
uv run --no-sync python experiments/grug/base/launch.py --version dev --set steps=200 --run
uv run --no-sync python experiments/grug/base/launch.py --version dev \
    --sweep 'optimizer.learning_rate=[1e-3, 3e-3]' --run

# Train the world-action model on LIBERO-Spatial
uv run --no-sync python experiments/episode_models/wam/launch.py --version dev --run

# Score every checkpoint of a finished run on its held-out loss
uv run --no-sync python -m poorin.experiment.evaluate episode/wam-libero-spatial@latest \
    --suite held-out --steps all --version dev --run
```

## Read next

- [Lazy artifacts](docs/explanations/lazy-artifacts.md): versions, caching, overrides, and drift.
- [Episode datasets](docs/explanations/episodes.md) and the
  [episode model templates](experiments/episode_models/README.md).
- [Removed from Marin](docs/references/removed-from-marin.md): what the fork dropped, if you are
  coming from Marin.
- [Decision log](docs/poorin/decisions.md): every change and why it was made.
- [`AGENTS.md`](AGENTS.md): the guide for coding agents.

## License

Poorin is a personal research codebase and does not take outside contributions. It is licensed
under the [Apache License 2.0](LICENSE), and forks are welcome. It is built on
[Marin](https://github.com/marin-community/marin) by [Stanford CRFM](https://crfm.stanford.edu/)
and [Open Athena](https://openathena.ai/), [Levanter](https://github.com/stanford-crfm/levanter),
and [Haliax](https://github.com/stanford-crfm/haliax); their authors are in
[`AUTHORS.md`](AUTHORS.md).
