# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy data builders: tokenized datasets and mixtures as artifact handles.

The library provides *mechanism* — concise builders — while an experiment states
the *policy*: which datasets, at what weights. The split is deliberate, so mixture
weights live in the experiment that chose them, not buried in a catalog constant.

- :func:`tokenized` returns an ``ArtifactStep[TokenizedCache]`` handle. Its raw input is either
  ``source`` (a HuggingFace id or single path) or ``paths`` (raw globs resolved against the run
  prefix). ``pin`` references already-tokenized data at an existing location instead of
  recomputing it.
- :func:`pretokenized` handles an already-tokenized Levanter cache hosted on
  HuggingFace (e.g. the fineweb-edu prebuilt subcaches): it downloads rather than
  re-tokenizes, and is consumed like any other ``TokenizedCache``.
- :func:`mixture` assembles a Levanter ``LmDataConfig`` from ``{handle: weight}``
  training components plus weight-0 ``validation`` handles, reading each cache's
  tokenizer/format from its :class:`~poorin.processing.tokenize.tokenize.TokenizedCache`
  record at run time.
"""

from collections.abc import Callable, Mapping, Sequence

from fray.types import ResourceConfig
from levanter.data.text.datasets import DEFAULT_LM_DATA_SHUFFLE, BlockShuffleConfig, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import TextLmDatasetFormat

from poorin.execution.artifact import resolve_path
from poorin.execution.build_context import resolve_version
from poorin.execution.lazy import ArtifactStep, StepContext
from poorin.execution.remote import remote
from poorin.processing.tokenize.data_configs import dataset_component
from poorin.processing.tokenize.download_pretokenized import PretokenizedCacheDownloadConfig, fetch_pretokenized_cache
from poorin.processing.tokenize.tokenize import (
    HfTokenizeConfig,
    TokenizeConfig,
    TokenizeConfigBase,
    TokenizedCache,
)
from poorin.processing.tokenize.tokenize import tokenize as _tokenize_fn


def _on(fn: Callable[..., object], resources: ResourceConfig | None) -> Callable[..., object]:
    """Run ``fn`` on Fray with ``resources`` (via :func:`remote`), or inline when None.

    Resources ride with the function, so they stay off the step node and out of the
    artifact fingerprint."""
    return remote(fn, resources=resources) if resources is not None else fn


def _looks_like_hf_id(source: str) -> bool:
    """A bare ``org/name`` HuggingFace id (one slash, not a filesystem/URL path)."""
    return source.count("/") == 1 and "://" not in source and not source.startswith("/")


def tokenized(
    name: str,
    *,
    tokenizer: str,
    version: str | None = None,
    source: str | None = None,
    paths: Sequence[str] | None = None,
    validation: bool = False,
    pin: str | None = None,
    text_key: str = "text",
    sample_count: int | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> ArtifactStep[TokenizedCache]:
    """A tokenized-dataset handle.

    Provide exactly one raw input: ``source`` (a HuggingFace id ``org/name`` or a single
    raw path) or ``paths`` (raw globs resolved against the run prefix). ``validation=True`` routes the data
    to the cache's validation split. ``sample_count`` caps the documents tokenized per shard
    (it bears identity — a sampled cache differs from the full one). ``pin`` references
    already-tokenized data at an existing location instead of recomputing it. ``version`` defers
    to the ambient :func:`~poorin.execution.build_context.build_context` when omitted — but a large
    corpus should pin an explicit calendar version, since a mutable (``dev``) tokenize rebuilds the
    whole cache on every run.
    """
    version = resolve_version(name, version)
    if (source is None) == (paths is None):
        raise ValueError(f"{name}: provide exactly one of source or paths")

    fmt = TextLmDatasetFormat(text_key=text_key)

    def build_config(ctx: StepContext) -> TokenizeConfigBase:
        if source is not None and _looks_like_hf_id(source):
            return HfTokenizeConfig(
                id=source,
                cache_path=ctx.output_path,
                tokenizer=tokenizer,
                format=fmt,
                sample_count=sample_count,
                tags=[*tags],
            )
        if paths is not None:
            resolved = [resolve_path(ctx.prefix, p) for p in paths]
        else:
            resolved = [resolve_path(ctx.prefix, source)]
        return TokenizeConfig(
            train_paths=[] if validation else resolved,
            validation_paths=resolved if validation else [],
            cache_path=ctx.output_path,
            tokenizer=tokenizer,
            format=fmt,
            sample_count=sample_count,
            tags=[*tags],
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TokenizedCache,
        run=_on(_tokenize_fn, resources),
        build_config=build_config,
        override_path=pin,
    )


def pretokenized(
    name: str,
    *,
    repo_id: str,
    tokenizer: str,
    version: str | None = None,
    revision: str | None = None,
    pin: str | None = None,
    tags: Sequence[str] = (),
    resources: ResourceConfig | None = None,
) -> ArtifactStep[TokenizedCache]:
    """A handle to an already-tokenized Levanter cache hosted on HuggingFace.

    ``build_config(ctx)`` downloads the HF dataset repo ``repo_id`` into ``ctx.output_path`` as
    a Levanter cache; the handle then reads as a ``TokenizedCache`` with no
    re-tokenization. Use it where a tokenizing :func:`tokenized` handle would be too
    slow — e.g. the fineweb-edu prebuilt subcaches. ``pin`` references an
    already-downloaded cache at an existing location instead of fetching it again. ``version``
    defers to the ambient :func:`~poorin.execution.build_context.build_context` when omitted.
    """
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> PretokenizedCacheDownloadConfig:
        return PretokenizedCacheDownloadConfig(
            cache_path=ctx.output_path,
            tokenizer=tokenizer,
            hf_repo_id=repo_id,
            hf_revision=revision,
            tags=[*tags],
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=TokenizedCache,
        run=_on(fetch_pretokenized_cache, resources),
        build_config=build_config,
        override_path=pin,
    )


def _placeholder_component(cache_dir: str):
    """A fingerprint-time mixture component: just a cache-dir placeholder + constant format.

    At fingerprint time no record exists; the cache dir renders to ``name@version``, so the
    component carries the dataset's identity and nothing read from disk.
    """
    source = UrlDatasetSourceConfig(
        tags=[], train_urls=[], validation_urls=[], cache_dir=cache_dir, format=TextLmDatasetFormat()
    )
    return dataset_component(source)


def mixture(
    ctx: StepContext,
    train: Mapping[ArtifactStep[TokenizedCache], float],
    *,
    validation: Sequence[ArtifactStep[TokenizedCache]] = (),
    shuffle: bool | BlockShuffleConfig = DEFAULT_LM_DATA_SHUFFLE,
) -> LmDataConfig:
    """Assemble an ``LmDataConfig`` from dataset handles.

    ``train`` maps each handle to its mixture weight; ``validation`` handles are added at
    weight 0. The component key is the handle's ``name`` (two handles sharing a name are
    rejected). At run time each component is built from its ``TokenizedCache`` record
    (tokenizer/format/path), never from the producing recipe — so pinned caches
    work the same as freshly tokenized ones. At fingerprint time (no records yet) the data
    contribution is the sorted ``{name@version: weight}`` map; the tokenizer is determined by
    the chosen datasets and verified at run time. Call this inside a consumer's ``build_config``
    and pass the same handles as the step's ``deps`` so they materialize first.
    """
    handles = [*train, *validation]
    if not handles:
        raise ValueError("mixture needs at least one training or validation component")
    names = [dataset.name for dataset in handles]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"mixture components are keyed by name, but these collide: {duplicates}")

    if ctx.is_fingerprint:
        components = {dataset.name: _placeholder_component(ctx.artifact_path(dataset)) for dataset in handles}
        weights = {dataset.name: weight for dataset, weight in train.items()}
        weights.update({dataset.name: 0.0 for dataset in validation})
        return LmDataConfig(
            components=components,
            train_weights=weights,
            tokenizer="<tokenizer>",
            cache_dir=None,
            shuffle=shuffle,
            permutation_type="feistel",
        )

    components = {}
    weights = {}
    tokenizers: set[str] = set()
    for dataset, weight in train.items():
        cache = ctx.resolved(dataset)
        components[dataset.name] = cache.as_component()
        weights[dataset.name] = weight
        tokenizers.add(cache.tokenizer)
    for dataset in validation:
        cache = ctx.resolved(dataset)
        components[dataset.name] = cache.as_component()
        weights[dataset.name] = 0.0
        tokenizers.add(cache.tokenizer)

    if len(tokenizers) != 1:
        raise ValueError(f"mixture components must share one tokenizer, got {sorted(tokenizers)}")

    return LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=tokenizers.pop(),
        cache_dir=None,
        shuffle=shuffle,
        permutation_type="feistel",
    )
