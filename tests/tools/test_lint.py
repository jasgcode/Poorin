# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.lint's file checks. Ruff and ty are not run here."""

import json
from pathlib import Path

import pytest

from tools.lint import file_problems, is_excluded, lint, ruff_config

MARIN = "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n"
HALIAX = "# Copyright The Levanter Authors\n#\n# SPDX-License-Identifier: Apache-2.0\n"


def write(root: Path, rel: str, content: str | bytes) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return rel


def test_a_missing_license_header_is_reported_and_inserted_after_the_shebang(tmp_path):
    rel = write(tmp_path, "tools/run.py", "#!/usr/bin/env python\nprint('hi')\n")
    assert file_problems(tmp_path, rel, fix=True) == ["missing license header"]
    assert (tmp_path / rel).read_text() == "#!/usr/bin/env python\n" + MARIN + "\nprint('hi')\n"
    assert file_problems(tmp_path, rel, fix=False) == []


def test_each_library_requires_its_own_license_header(tmp_path):
    haliax = write(tmp_path, "lib/haliax/src/haliax/core.py", HALIAX + "X = 1\n")
    marin_header_in_haliax = write(tmp_path, "lib/haliax/src/haliax/other.py", MARIN + "X = 1\n")
    assert file_problems(tmp_path, haliax, fix=False) == []
    assert file_problems(tmp_path, marin_header_in_haliax, fix=False) == ["missing license header"]


def test_ruff_checks_marin_with_its_own_configuration_as_marin_did():
    # Ruff's per-file discovery classifies first-party imports differently from an explicit
    # --config for lib/poorin, so Marin passed it explicitly; everything else uses discovery.
    assert ruff_config("lib/poorin/src/poorin/evaluation/records.py") == "lib/poorin/pyproject.toml"
    assert ruff_config("lib/levanter/src/levanter/main/train_lm.py") is None
    assert ruff_config("tools/lint.py") is None


def test_trailing_whitespace_and_a_missing_final_newline_are_fixed(tmp_path):
    rel = write(tmp_path, "notes.md", "title  \nbody\t\nend")
    assert file_problems(tmp_path, rel, fix=True) == ["trailing whitespace", "no final newline"]
    assert (tmp_path / rel).read_text() == "title\nbody\nend\n"


def test_merge_conflict_markers_are_reported(tmp_path):
    rel = write(tmp_path, "notes.md", f"{'<' * 7} HEAD\nours\n{'=' * 7}\ntheirs\n{'>' * 7} branch\n")
    assert file_problems(tmp_path, rel, fix=False) == ["merge-conflict marker"]


@pytest.mark.parametrize(
    "rel, content",
    [("bad.py", MARIN + "def f(:\n"), ("bad.toml", "a =\n"), ("bad.yaml", "a: [\n"), ("bad.json", "{\n")],
)
def test_files_that_do_not_parse_are_reported(tmp_path, rel, content):
    write(tmp_path, rel, content)
    [problem] = file_problems(tmp_path, rel, fix=False)
    assert problem.startswith("does not parse")


def test_levanter_yaml_include_tags_parse(tmp_path):
    rel = write(tmp_path, "lib/levanter/config/run.yaml", "data: !include data/openwebtext.yaml\n")
    assert file_problems(tmp_path, rel, fix=False) == []


def test_notebook_outputs_are_reported_and_cleared(tmp_path):
    cell = {"cell_type": "code", "source": ["1 + 1"], "metadata": {}, "execution_count": 3, "outputs": [{"x": 1}]}
    notebook = {"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    rel = write(tmp_path, "docs/demo.ipynb", json.dumps(notebook) + "\n")
    assert file_problems(tmp_path, rel, fix=True) == ["notebook has outputs"]
    [cleared] = json.loads((tmp_path / rel).read_text())["cells"]
    assert cleared["outputs"] == [] and cleared["execution_count"] is None


def test_large_files_are_reported(tmp_path):
    rel = write(tmp_path, "data.txt", "x\n" * 256_001)
    assert file_problems(tmp_path, rel, fix=False) == ["larger than 500 KB"]


def test_binary_files_skip_the_text_checks(tmp_path):
    rel = write(tmp_path, "model.bin", b"\xff\xfe trailing \nno final newline")
    assert file_problems(tmp_path, rel, fix=False) == []


def test_a_binary_file_that_decodes_as_utf8_is_left_unchanged_by_fix(tmp_path):
    # NUL bytes are valid UTF-8; a NUL in the first 8 KB marks a binary file, as git decides.
    content = b"\x00\x00\x00weights \nno final newline"
    rel = write(tmp_path, "weights.bin", content)
    assert file_problems(tmp_path, rel, fix=True) == []
    assert (tmp_path / rel).read_bytes() == content


def test_lock_files_and_images_are_excluded_at_any_depth():
    assert is_excluded("uv.lock")
    assert is_excluded("docs/images/plot.png")
    assert not is_excluded("pyproject.toml")


def test_a_run_that_fixed_a_file_still_fails_and_the_next_run_passes(tmp_path):
    rel = write(tmp_path, "notes.md", "x \n")
    assert lint([rel], fix=True, repo_root=tmp_path) == ["notes.md: trailing whitespace"]
    assert lint([rel], fix=False, repo_root=tmp_path) == []


def test_paths_that_no_longer_exist_are_skipped(tmp_path):
    assert lint(["deleted.py"], fix=False, repo_root=tmp_path) == []
