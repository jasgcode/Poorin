# The Language Modeling Pipeline

Our goal is to build strong base and instruction-tuned language models.
There are many components of this pipeline, all of which are represented in Marin:


1. Downloading raw sources
2. Normalizing them into a standard Parquet format
3. Filtering and deduplicating the documents to produce clean data
4. Tokenizing the clean data for training
5. Training a model on the tokenized data
6. Evaluating the resulting model

Currently, we leverage the following open-source tools (thanks to the authors for making them!):

- For model training, it uses [Levanter](https://github.com/stanford-crfm/levanter),
  a Jax-based framework that's legible, scalable, and reproducible.
- For model evaluation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Note that the Marin framework is agnostic to these choices and we can support other tools.

Where possible, we use the same data formats as [Dolma](https://github.com/allenai/dolma). Where not possible, we try to use "natural" extensions that stick to the spirit of the format.
