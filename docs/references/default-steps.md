# Default Pipeline Steps

Marin provides a set of standard builders for the common stages of an LM experiment:
data download, tokenization, mixture assembly, and training. Reach for these
before writing custom step code.

The download, tokenization, and training builders return lazy `ArtifactStep[T]`
handles (e.g. `ArtifactStep[TokenizedCache]` or `ArtifactStep[LevanterCheckpoint]`) that
`StepRunner` materializes on demand. `mixture` is the exception: it runs inside a step, taking
that step's `StepContext` and returning a Levanter `LmDataConfig`.

## Download

::: poorin.experiment.data.hf_download

::: poorin.experiment.data.raw_download

## Tokenization

::: poorin.experiment.data.tokenized

::: poorin.experiment.data.pretokenized

## Mixture assembly

::: poorin.experiment.data.mixture

## Training

::: poorin.experiment.train.train_lm
