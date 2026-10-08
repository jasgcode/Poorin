# Setting up a Local GPU Environment

This guide will walk you through the steps to set up a local GPU environment for Marin.
By "local", we mean a machine that you run jobs on directly. Fray's local backend runs every
training job on the machine that launches it.

## Prerequisites

Make sure you've followed the [installation guide](installation.md) to do the basic installation.

In addition to the prerequisites from the basic installation, we have one GPU-specific system dependency:

- NVIDIA driver 580 or newer

We assume you are running Ubuntu 24.04.

## NVIDIA driver and runtime

Install an NVIDIA driver that supports CUDA 13. Verify that the driver is at least 580 and that
`nvidia-smi` reports CUDA 13.x:

```bash
nvidia-smi
```

Marin uses [JAX](https://docs.jax.dev/en/latest/index.html) as a core library. The workspace
members' `gpu` extra installs the CUDA 13 JAX runtime, including CUDA, cuDNN, and NCCL Python
wheels; the `lm_eval` extra installs the lm-eval harness for in-training and standalone
evaluation:

```bash
uv sync --locked --all-packages --extra=gpu --extra=lm_eval
```

Run commands with `uv run --no-sync` afterwards: a syncing `uv run` installs no extras, so it
removes the CUDA build of JAX.

If you install a local CUDA toolkit for custom kernels, use CUDA 13 and keep older CUDA libraries
out of `LD_LIBRARY_PATH` so they do not override the JAX wheel libraries.

See [JAX's installation guide](https://docs.jax.dev/en/latest/installation.html) for more options.

!!! tip
If you are using a DGX Spark or similar machine with unified memory, you may need to dramatically reduce the memory that XLA preallocates for itself. You can do this by setting the `XLA_PYTHON_CLIENT_MEM_FRACTION` variable, to something like 0.5:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

    You can also set this in your `.bashrc` or `.zshrc` file.
    ```bash
    echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5' >> ~/.bashrc
    ```

    For broader JAX/Levanter memory tuning (sharding, checkpointing, offloading), see [Making Things Fit in HBM](../references/hbm-optimization.md).

## Running an Experiment

Now you can run an experiment.
The unified tutorial script [`experiments/tutorials/train_tiny_model.py`](https://github.com/marin-community/marin/blob/main/experiments/tutorials/train_tiny_model.py)
accepts a `--device` flag that selects the accelerator:

```bash
export POORIN_PREFIX=local_store
export WANDB_ENTITY=...
uv run python experiments/tutorials/train_tiny_model.py \
  --device rtx5090x1 --dataset wikitext --version dev --run
```

`POORIN_PREFIX` sets the root directory for all outputs; it can be a local path or anything
fsspec supports, such as `s3://` or `gs://`.

The same script runs on CPU, GPU, and TPU — only `--device` and `--dataset` change. The GPU
device entry in `train_tiny_model.py` configures resources and batch size:

```python
from fray.types import ResourceConfig

# "rtx5090x1" entry in DEVICES (resources, batch_size)
resources = ResourceConfig.with_gpu("RTX5090", count=1, cpu=16, disk="128G", ram="96G")
batch_size = 32
```

Whereas `--device cpu` uses `ResourceConfig.with_cpu()` and a batch size of 4, `--device rtx5090x1`
uses one RTX 5090 with a larger batch. The local backend does not check the entry against the
machine, so pick the one that matches your hardware. Adding a new device is one entry in the
`DEVICES` dict — no separate file needed.

The Grug base launcher, `experiments/grug/base/launch.py`, trains a 130M-parameter Grug model for
50 steps on 10M pretokenized FineWeb-Edu tokens, on one GPU:

```bash
uv run python -m experiments.grug.base.launch --version dev --run
```
