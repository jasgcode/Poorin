# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A of the split tokenize pipeline: NormalizedData → datakit attribute parquet.

Reads one or more :class:`~poorin.datakit.normalize.NormalizedData` sources, runs
the shared tokenization core, and writes co-partitioned attribute parquet shards
with columns ``id``, ``chunk_index`` and ``input_ids``.

Each output shard mirrors the basename of its source shard, which ensures the
datakit invariants hold by construction:

* Same shard count as source (co-partitioned).
* Sorted by ``id`` within each shard (sources are already sorted; the
  tokenize pipeline preserves order). A document above the token limit of one
  Parquet row occupies several adjacent rows that share its ``id``, in
  ``chunk_index`` order, so the sort by ``id`` holds.

Downstream:

* :func:`poorin.processing.tokenize.store_builder.build_levanter_store` consumes
  one or more :class:`TokenizedAttrData` artifacts to produce a Levanter cache
  per split (sharded layout).
* Other datakit attribute consumers (joins, mixing) can use the ``id`` column to
  align tokens with quality scores, dedup flags, etc.
"""

import dataclasses
import logging
import os

import pyarrow as pa
from fray.types import ResourceConfig
from levanter.data.text.formats import LmDatasetFormatBase, TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_file

from poorin.datakit.normalize import NormalizedData
from poorin.datakit.source_key import DatakitArtifactPath, datakit_source_key
from poorin.processing.tokenize._core import (
    CHUNK_INDEX_FIELD,
    resolve_window_and_batch,
    tokenize_batches_with_id,
    tokenize_pipeline,
)

logger = logging.getLogger(__name__)
TOKENIZED_ATTR_DATA_VERSION = 4


class TokenizedAttrData(BaseModel):
    """Per-split datakit attribute datasets produced by :func:`tokenize_attributes`.

    Each split's attribute parquet shards live under ``output_dirs[split]/`` with
    columns ``id: string``, ``chunk_index: int32`` and ``input_ids: list<int>``.
    Shards mirror their source
    :class:`~poorin.datakit.normalize.NormalizedData` partitions 1:1 (same basename,
    same source order, same id range), so the dataset is sorted by ``id`` per
    partition and co-partitioned with the source — both datakit invariants.

    ``id`` is the source document id, and is the join key against the other datakit
    attribute datasets. A document with more tokens than one Parquet row can hold
    occupies several adjacent rows that share that id, in zero-based
    ``chunk_index`` order. An unsplit document is a single row with
    ``chunk_index == 0``. A consumer that needs one row per document must group by
    ``id``. A consumer that walks this dataset positionally against a
    one-row-per-document dataset must not: the row counts differ.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, TokenizedAttrData)``.

    Attributes:
        version: Schema version.
        output_dirs: Map from split name (e.g. ``"train"``, ``"validation"``) to the
            directory containing that split's attribute parquet shards.
        source_keys: Map from split name to the prefix-relative identity of the
            ``NormalizedData.main_output_dir`` whose shards this dataset mirrors.
        tokenizer: Tokenizer name/path used (informational; consumers should re-verify
            against any other inputs they combine this with).
        tokenizer_backend: Tokenizer backend, as ``TokenizerBackend.value``.
        counters: Aggregated zephyr counters per split.
    """

    version: str = f"v{TOKENIZED_ATTR_DATA_VERSION}"
    output_dirs: dict[str, DatakitArtifactPath]
    source_keys: dict[str, str]
    tokenizer: str
    tokenizer_backend: str
    counters: dict[str, dict[str, int | float]]

    def shard_paths(self, split: str) -> list[str]:
        """Return parquet shard paths for ``split`` in order, or ``[]`` if absent."""
        d = self.output_dirs.get(split)
        if d is None:
            return []
        return sorted(str(m) for m in StoragePath(prefix_join(d, "*.parquet")).glob())


@dataclasses.dataclass(frozen=True, kw_only=True)
class TokenizeAttributesConfig:
    """Config for :func:`tokenize_attributes`.

    At least one of ``train_source`` or ``validation_source`` must be provided.
    Each is an independent :class:`NormalizedData` artifact whose shards become
    the corresponding split's co-partitioned attribute parquet output.
    """

    train_source: NormalizedData | None = None
    validation_source: NormalizedData | None = None
    output_path: str
    tokenizer: str
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF
    format: LmDatasetFormatBase = TextLmDatasetFormat()  # noqa: RUF009
    sample_count: int | None = None
    text_field: str = "text"
    max_workers: int = 4096
    worker_resources: ResourceConfig = dataclasses.field(default_factory=lambda: ResourceConfig(ram="10g", disk="5g"))
    zephyr_context: ZephyrContext | None = None

    def __post_init__(self):
        if self.train_source is None and self.validation_source is None:
            raise ValueError("at least one of train_source / validation_source must be provided")


_ATTRIBUTE_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field(CHUNK_INDEX_FIELD, pa.int32()),
        pa.field("input_ids", pa.list_(pa.int32())),
    ]
)


def _tokenize_text_batch(
    batch: pa.RecordBatch,
    *,
    data_format: TextLmDatasetFormat,
    window_size: int,
) -> pa.RecordBatch:
    """Tokenize one projected Arrow batch and return writer-ready Arrow."""
    rows = batch.to_pylist()
    windows = (rows[start : start + window_size] for start in range(0, len(rows), window_size))
    tokenized = tokenize_batches_with_id(data_format=data_format, batches=windows)
    return pa.RecordBatch.from_pylist(list(tokenized), schema=_ATTRIBUTE_SCHEMA)


def _process_split(
    *,
    source: NormalizedData,
    split: str,
    config: TokenizeAttributesConfig,
) -> tuple[str, dict[str, int | float]]:
    """Tokenize one split's NormalizedData into co-partitioned attribute parquet.

    Returns ``(split_output_dir, counters)``.
    """
    source_shards = sorted(str(m) for m in StoragePath(prefix_join(source.main_output_dir, "*.parquet")).glob())
    if not source_shards:
        raise FileNotFoundError(f"No parquet shards found under {source.main_output_dir}")

    split_dir = prefix_join(config.output_path, split)
    output_basenames = tuple(os.path.basename(p) for p in source_shards)

    def _output_path(shard_idx: int, total_shards: int, sd: str = split_dir, bn: tuple = output_basenames) -> str:
        return prefix_join(sd, bn[shard_idx])

    logger.info(
        "Tokenizing %s (split=%s): %d source shards → %s",
        source.main_output_dir,
        split,
        len(source_shards),
        split_dir,
    )

    if isinstance(config.format, TextLmDatasetFormat) and config.sample_count is None:
        window_size, _ = resolve_window_and_batch(source_shards[0], requested_batch_size=None)
        tokenized_ds = (
            Dataset.from_list(source_shards)
            .load_parquet(columns=["id", config.format.text_key], batch_mode=True)
            .map(
                lambda batch, fmt=config.format, size=window_size: _tokenize_text_batch(
                    batch,
                    data_format=fmt,
                    window_size=size,
                )
            )
        )
    else:
        ds = Dataset.from_list(source_shards).flat_map(load_file)
        tokenized_ds, _ = tokenize_pipeline(
            ds,
            data_format=config.format,
            text_field=config.text_field,
            sample_count=config.sample_count,
            # The first source shard is parquet by construction; pass it for the
            # row-group-aware window sizing in the shared core.
            sample_parquet_path=source_shards[0],
            levanter_batch_size=None,
        )

    pipeline = tokenized_ds.write_parquet(
        _output_path,
        schema=_attribute_schema(config.format),
        skip_existing=True,
    )

    ctx = config.zephyr_context or ZephyrContext(
        resources=config.worker_resources,
        max_workers=min(config.max_workers, len(source_shards)),
        name=f"tokenize-attributes-{split}",
    )
    ctx.put("tokenizer_name", config.tokenizer)
    ctx.put("tokenizer_backend", config.tokenizer_backend)

    outcome = ctx.execute(
        pipeline,
        verbose=True,
        map_task_resources=config.worker_resources,
    )
    return split_dir, dict(outcome.counters)


def _attribute_schema(data_format: LmDatasetFormatBase) -> pa.Schema | None:
    """Return the parquet schema for attribute output, or ``None`` to let zephyr infer.

    For text formats we pin ``id: string``, ``chunk_index: int32`` and ``input_ids: list<int32>`` to keep
    files compact and stable across workers. For chat or other multi-output formats,
    we let zephyr infer from the first record so additional columns like
    ``assistant_masks`` flow through unchanged.
    """
    if isinstance(data_format, TextLmDatasetFormat):
        return _ATTRIBUTE_SCHEMA
    return None


def tokenize_attributes(config: TokenizeAttributesConfig) -> TokenizedAttrData:
    """Tokenize :class:`NormalizedData` source(s) into datakit attribute parquet.

    Each split's source shards become co-partitioned attribute parquet files
    sharing basenames with the source. Output rows carry ``{id, chunk_index, input_ids}``
    (plus any extra fields produced by the format processor for non-text formats).

    Args:
        config: See :class:`TokenizeAttributesConfig`.

    Returns:
        A :class:`TokenizedAttrData` describing the per-split output directories,
        source linkage, tokenizer config, and counters.
    """
    output_dirs: dict[str, str] = {}
    source_keys: dict[str, str] = {}
    counters: dict[str, dict[str, int | float]] = {}

    splits: list[tuple[str, NormalizedData]] = []
    if config.train_source is not None:
        splits.append(("train", config.train_source))
    if config.validation_source is not None:
        splits.append(("validation", config.validation_source))

    for split, source in splits:
        split_dir, split_counters = _process_split(source=source, split=split, config=config)
        output_dirs[split] = split_dir
        source_keys[split] = datakit_source_key(source.main_output_dir)
        counters[split] = split_counters

    return TokenizedAttrData(
        output_dirs=output_dirs,
        source_keys=source_keys,
        tokenizer=config.tokenizer,
        tokenizer_backend=config.tokenizer_backend.value,
        counters=counters,
    )
