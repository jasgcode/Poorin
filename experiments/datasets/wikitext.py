# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WikiText-2 (detokenized) from the Hugging Face Hub, a 1,000-document sample per shard tokenized
with the Marin tokenizer. The tutorial trains on it; grug base holds out its validation split."""

from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import tokenized
from poorin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def wikitext2_dataset() -> ArtifactStep[TokenizedCache]:
    return tokenized(
        "wikitext",
        tokenizer=marin_tokenizer,
        source="dlwh/wikitext_2_detokenized",
        sample_count=1000,
        version="2026.06.28",
    )
