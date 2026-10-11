# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clips per second from an episode dataset against the number of decoding worker processes.

uv run --no-sync python -m experiments.datasets.episodes.throughput PATH --field video --workers 1 4 8 16 24
"""

import argparse
import time

from poorin.episodes.build import EpisodeDataset
from poorin.episodes.read import ClipSpec, batch_iterator, clip_dataset

CLIP_LENGTH = 16
BATCH_SIZE = 32
# Long enough that the workers' prefetch buffers, full after warmup, are a small part of the measurement.
WARMUP_BATCHES = 20
MEASURED_BATCHES = 500
TARGET_CLIPS_PER_SECOND = 1000


def clips_per_second(dataset: EpisodeDataset, field: str, workers: int) -> float:
    clips = clip_dataset(dataset, clip=ClipSpec(length=CLIP_LENGTH, stride=1, skip=1), fields=[field])
    iterator = iter(
        batch_iterator(clips.shuffle(seed=0).repeat(None), batch_size=BATCH_SIZE, start_step=0, num_workers=workers)
    )
    try:
        for _ in range(WARMUP_BATCHES):
            next(iterator)
        start = time.perf_counter()
        for _ in range(MEASURED_BATCHES):
            next(iterator)
        return MEASURED_BATCHES * BATCH_SIZE / (time.perf_counter() - start)
    finally:
        iterator.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--field", required=True)
    parser.add_argument("--workers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    dataset = EpisodeDataset.raw_load(args.path)
    for workers in args.workers:
        rate = clips_per_second(dataset, args.field, workers)
        verdict = "meets" if rate >= TARGET_CLIPS_PER_SECOND else "misses"
        print(f"{workers:3d} workers  {rate:8.0f} clips/s  {verdict} {TARGET_CLIPS_PER_SECOND}")


if __name__ == "__main__":
    main()
