# Default Pipeline Steps

Poorin provides a set of standard builders for the common stages of an LM experiment:
tokenization (or download of a pretokenized cache), mixture assembly, and training. Reach for these
before writing custom step code.

The tokenization and training builders return lazy `ArtifactStep[T]`
handles (e.g. `ArtifactStep[TokenizedCache]` or `ArtifactStep[LevanterCheckpoint]`) that
`StepRunner` materializes on demand. `mixture` is the exception: it runs inside a step, taking
that step's `StepContext` and returning a Levanter `LmDataConfig`.

## Tokenization

::: poorin.experiment.data.tokenized

::: poorin.experiment.data.pretokenized

## Mixture assembly

::: poorin.experiment.data.mixture

## Training

::: poorin.experiment.train.train_lm
