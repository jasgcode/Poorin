# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End to end: LIBERO clips through to_mesh into a toy JAX step for 200 steps, then a resume at
step 100 that must see the same batches.

    uv run --no-sync python -m experiments.datasets.episodes.smoke_train PATH
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from poorin.episodes.build import EpisodeDataset
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset, to_mesh

STEPS = 200
RESUME_STEP = 100
BATCH_SIZE = 16
WORKERS = 8
SEED = 0
LEARNING_RATE = 1e-3


@jax.jit
def train_step(weights: jax.Array, frames: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Predict each step's action from the mean color of its frame: a toy that exercises placement."""

    def loss(w):
        features = frames.astype(jnp.float32).mean(axis=(2, 3)) / 255.0  # [B, T, 3]
        return jnp.mean((features @ w - actions) ** 2)

    value, grad = jax.value_and_grad(loss)(weights)
    return weights - LEARNING_RATE * grad, value


def run(dataset: EpisodeDataset, start_step: int, steps: int) -> list[np.ndarray]:
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    clips = clip_dataset(dataset, clip=ClipSpec(length=8, stride=4, skip=1), fields=["agentview", "action"])
    iterator = iter(
        batch_iterator(
            clips.shuffle(seed=SEED).repeat(None), batch_size=BATCH_SIZE, start_step=start_step, num_workers=WORKERS
        )
    )
    weights = jnp.zeros((3, 7))
    seen: list[np.ndarray] = []
    try:
        for step in range(start_step, steps):
            batch = to_mesh(next(iterator), mesh, ("data",))
            weights, loss = train_step(weights, batch["agentview"], batch["action"])
            seen.append(np.asarray(batch["action"]))
            if step % 50 == 0:
                print(f"step {step}  loss {float(loss):.4f}")
    finally:
        iterator.close()
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    dataset = EpisodeDataset.raw_load(parser.parse_args().path)
    full = run(dataset, 0, STEPS)
    resumed = run(dataset, RESUME_STEP, STEPS)
    for offset, batch in enumerate(resumed):
        np.testing.assert_array_equal(batch, full[RESUME_STEP + offset])
    print(f"resumed at step {RESUME_STEP}: {len(resumed)} batches match the uninterrupted run")


if __name__ == "__main__":
    main()
