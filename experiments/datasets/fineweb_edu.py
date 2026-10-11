# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10M pretokenized FineWeb-Edu tokens (about 20 MB) from the Hugging Face Hub, tokenized with the
Marin tokenizer (the Llama 3 vocabulary, 128,256 entries)."""

from poorin.execution.lazy import ArtifactStep
from poorin.experiment.data import pretokenized
from poorin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def fineweb_edu_10m_dataset() -> ArtifactStep[TokenizedCache]:
    return pretokenized(
        "fineweb-edu-10M",
        repo_id="marin-community/fineweb-edu-pretokenized-10M",
        tokenizer=marin_tokenizer,
        version="2026.06.28",
        # The repository's head since 2025-05-15, pinned so a new upload cannot change the data under
        # this version (D104).
        revision="f13fbe24e962f9e20d75c3c6f08830bcfd0606b8",
    )
