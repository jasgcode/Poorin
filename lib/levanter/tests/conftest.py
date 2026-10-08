# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from jaxtyping import install_import_hook
from tokenizers import Tokenizer as HfBaseTokenizer

from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import HfMarinTokenizer, load_tokenizer

# Check grug's jaxtyping annotations at run time: every function and dataclass in these packages is
# wrapped with beartype as it is imported, so a wrong shape or dtype fails the test that hit it. The
# hook must be installed before any test imports the packages.
install_import_hook(["levanter.grug"], "beartype.beartype")


@pytest.fixture(scope="session")
def local_gpt2_tokenizer(tmp_path_factory):
    """Load a GPT2 MarinTokenizer from a local directory to avoid network downloads."""
    source_dir = Path(__file__).parent
    output_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    tmpdir = stage_gpt2_tokenizer(source_dir, output_dir)
    return load_tokenizer(str(tmpdir))


@pytest.fixture(scope="session")
def local_gpt2_tokenizer_path(tmp_path_factory) -> str:
    """Path to a local GPT-2 tokenizer directory.

    Lets HF roundtrip tests build an ``HFCheckpointConverter`` with a local
    tokenizer (and no ``reference_checkpoint``) so they never reach the Hub.
    """
    source_dir = Path(__file__).parent
    output_dir = tmp_path_factory.mktemp("gpt2_tokenizer")
    return str(stage_gpt2_tokenizer(source_dir, output_dir))


@pytest.fixture(scope="session")
def local_gpt2_marin_tokenizer(tmp_path_factory) -> HfMarinTokenizer:
    """Load a GPT2 MarinTokenizer from a local JSON file to avoid network downloads."""

    test_dir = Path(__file__).parent
    tmpdir = tmp_path_factory.mktemp("gpt2_marin_tok")
    shutil.copy(test_dir / "gpt2_tokenizer.json", tmpdir / "tokenizer.json")
    shutil.copy(test_dir / "gpt2_tokenizer_config.json", tmpdir / "tokenizer_config.json")

    tok = HfBaseTokenizer.from_file(str(tmpdir / "tokenizer.json"))

    # GPT-2 uses <|endoftext|> as both BOS and EOS
    vocab = tok.get_vocab()
    eos_token = "<|endoftext|>"
    eos_id = vocab.get(eos_token)

    all_special_ids = [eos_id] if eos_id is not None else []

    return HfMarinTokenizer(
        _tokenizer=tok,
        _name_or_path=str(tmpdir),
        _bos_id=None,
        _eos_id=eos_id,
        _pad_id=None,
        _bos_token=None,
        _eos_token=eos_token,
        _chat_template=None,
        _vocab_size=tok.get_vocab_size(),
        _all_special_ids=all_special_ids,
        _id_to_token={v: k for k, v in vocab.items()},
        _vocab=vocab,
    )


@pytest.fixture(autouse=True)
def _configure_poorin_prefix():
    """Set POORIN_PREFIX to a temp directory for tests that rely on it."""
    if "POORIN_PREFIX" in os.environ:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="poorin_prefix") as temp_dir:
        os.environ["POORIN_PREFIX"] = temp_dir
        yield
        del os.environ["POORIN_PREFIX"]
