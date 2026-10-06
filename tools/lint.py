# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lint and format Poorin's files. Replaces Marin's ``infra/pre-commit.py``.

Usage (from the repository root):
    uv run --no-sync python -m tools.lint                 # the uncommitted change
    uv run --no-sync python -m tools.lint --fix           # and fix what can be fixed
    uv run --no-sync python -m tools.lint --all-files     # every tracked file
    uv run --no-sync python -m tools.lint PATH ...        # these files

Every file is checked for size, merge-conflict markers, trailing whitespace, and a final newline.
Python, TOML, YAML, and JSON files must parse, and notebooks must have no outputs. Python files
must start with their library's license header and pass ruff and black, and ty checks the
project when any Python file is linted. Ruff and black run at the versions Marin pinned, with
each library's own configuration, so switching from Marin's script reformats nothing; ty runs at
0.0.84.

A check fails when it finds a problem, even when ``--fix`` repaired it, so a fixed file is
reviewed before it is committed.
"""

import argparse
import ast
import fnmatch
import json
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tools.changes import changed_files
from tools.reachability import REPO_ROOT

RUFF = ("uvx", "ruff@0.14.3", "check")
BLACK = ("uvx", "black@25.9.0")
# Errors fail the check; warnings are reported without failing it.
TY = ("uvx", "ty@0.0.84", "check", "--exit-zero-on-warning")
MAX_FILE_BYTES = 500 * 1024
# Built at runtime so this file does not contain the markers it looks for.
CONFLICT_MARKERS = (b"<" * 7, b">" * 7)
EXCLUDED = (
    ".git/**",
    "tests/snapshots/**",
    "**/*_connect.py",
    "**/*_pb2.py",
    "**/*.gz",
    "**/*.pb",
    "**/*.index",
    "**/*.ico",
    "**/*.npy",
    "**/*.lock",
    "**/*.png",
    "**/*.jpg",
    "**/*.html",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.mov",
    "**/*.mp4",
    "**/*.data-*",
    "**/package-lock.json",
    "**/*-template.yaml",
)
MARIN_HEADER = ("Copyright The Marin Authors", "SPDX-License-Identifier: Apache-2.0")
LEVANTER_HEADER = ("Copyright The Levanter Authors", "SPDX-License-Identifier: Apache-2.0")
HALIAX_HEADER = ("Copyright The Levanter Authors", "", "SPDX-License-Identifier: Apache-2.0")


@dataclass(frozen=True)
class PythonStyle:
    """How Python under ``prefix`` is checked: black's configuration, the license header, and an
    explicit ruff configuration, or None to let ruff find the nearest one."""

    prefix: str
    black_config: str
    header: tuple[str, ...]
    ruff_config: str | None = None


# Levanter and Haliax keep the formatting and headers they had as separate projects. Marin's code
# is checked with its own ruff configuration passed explicitly: ruff's discovery would classify its
# first-party imports differently.
PYTHON_STYLES = (
    PythonStyle("lib/levanter/", "lib/levanter/pyproject.toml", LEVANTER_HEADER),
    PythonStyle("lib/haliax/", "lib/haliax/pyproject.toml", HALIAX_HEADER),
    PythonStyle("lib/marin/", "pyproject.toml", MARIN_HEADER, ruff_config="lib/marin/pyproject.toml"),
    PythonStyle("", "pyproject.toml", MARIN_HEADER),
)


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML loader that accepts Levanter's ``!include`` tag without following it."""


def _include(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    return str(loader.construct_scalar(node))


_ConfigLoader.add_constructor("!include", _include)


def python_style(path: str) -> PythonStyle:
    """The style for a repository-relative Python path."""
    return next(style for style in PYTHON_STYLES if path.startswith(style.prefix))


def is_excluded(path: str) -> bool:
    """Whether a repository-relative path is skipped. A leading ``**/`` also matches no directory."""
    return any(
        fnmatch.fnmatch(path, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]))
        for pattern in EXCLUDED
    )


def file_problems(repo_root: Path, rel: str, fix: bool) -> list[str]:
    """Problems in one file. With ``fix``, the fixable ones are repaired and still reported."""
    path = repo_root / rel
    data = path.read_bytes()
    problems = []
    if len(data) > MAX_FILE_BYTES:
        problems.append(f"larger than {MAX_FILE_BYTES // 1024} KB")
    if any(marker in data for marker in CONFLICT_MARKERS):
        problems.append("merge-conflict marker")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return problems  # binary: only the byte-level checks apply
    fixed = text
    if any(line.rstrip("\r").endswith((" ", "\t")) for line in fixed.split("\n")):
        problems.append("trailing whitespace")
        fixed = "\n".join(line.rstrip() for line in fixed.split("\n"))
    if fixed and not fixed.endswith("\n"):
        problems.append("no final newline")
        fixed += "\n"
    syntax_error = _syntax_error(path, fixed)
    if syntax_error is not None:
        problems.append(f"does not parse: {syntax_error}")
    elif path.suffix == ".ipynb":
        notebook = json.loads(fixed)
        code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in code_cells):
            problems.append("notebook has outputs")
            for cell in code_cells:
                cell["outputs"] = []
                cell["execution_count"] = None
            fixed = json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if path.suffix == ".py":
        with_header = _with_header(fixed, python_style(rel).header)
        if with_header is not None:
            problems.append("missing license header")
            fixed = with_header
    if fix and fixed != text:
        path.write_text(fixed, encoding="utf-8")
    return problems


def _syntax_error(path: Path, text: str) -> str | None:
    try:
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        elif path.suffix == ".toml":
            tomllib.loads(text)
        elif path.suffix in (".yaml", ".yml"):
            yaml.load(text, Loader=_ConfigLoader)
        elif path.suffix in (".json", ".ipynb"):
            json.loads(text)
    except (SyntaxError, ValueError, yaml.YAMLError) as error:  # TOML and JSON errors are ValueErrors
        return f"{type(error).__name__}: {error}".splitlines()[0]
    return None


def _with_header(text: str, header: Sequence[str]) -> str | None:
    """``text`` with the license header inserted after any shebang, or None if it has the header.

    The header counts as present when it appears in the leading comment block.
    """
    lines = text.split("\n")
    shebang = lines.pop(0) if text.startswith("#!") else None
    comments = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            if stripped:
                break
            continue
        comments.append(stripped[2:] if stripped.startswith("# ") else stripped[1:])
    if "\n".join(header) in "\n".join(comments):
        return None
    block = "\n".join(f"# {line}" if line else "#" for line in header) + "\n"
    body = "\n".join(lines)
    with_header = f"{block}\n{body}" if body else block
    return with_header if shebang is None else f"{shebang}\n{with_header}"


def _tool_output(command: Sequence[str], repo_root: Path) -> str | None:
    """None when ``command`` succeeds, else its combined output."""
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    return None if result.returncode == 0 else (result.stdout + result.stderr).strip()


def python_tool_problems(paths: Sequence[str], fix: bool, repo_root: Path) -> list[str]:
    """Ruff and black once per library style, and ty over the project."""
    if not paths:
        return []
    problems = []
    for style in PYTHON_STYLES:
        members = [path for path in paths if python_style(path) == style]
        if not members:
            continue
        ruff_config = ("--config", style.ruff_config) if style.ruff_config else ()
        ruff_fix = ("--fix", "--exit-non-zero-on-fix") if fix else ()
        ruff = _tool_output([*RUFF, *ruff_config, *ruff_fix, *members], repo_root)
        if ruff is not None:
            problems.append(f"ruff:\n{ruff}")
        config = ("--config", style.black_config)
        black = _tool_output([*BLACK, "--check", *config, *members], repo_root)
        if black is not None:
            problems.append(f"black ({style.black_config}):\n{black}")
            if fix:
                _tool_output([*BLACK, *config, *members], repo_root)
    ty = _tool_output(TY, repo_root)
    if ty is not None:
        problems.append(f"ty:\n{ty}")
    return problems


def lint(paths: Sequence[str], *, fix: bool, repo_root: Path = REPO_ROOT) -> list[str]:
    """Problems in ``paths`` (repository-relative). Missing and excluded paths are skipped."""
    files = sorted({path for path in paths if (repo_root / path).is_file() and not is_excluded(path)})
    problems = [f"{rel}: {problem}" for rel in files for problem in file_problems(repo_root, rel, fix)]
    return problems + python_tool_problems([rel for rel in files if rel.endswith(".py")], fix, repo_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint and format Poorin's files.")
    parser.add_argument("--fix", action="store_true", help="Fix what can be fixed; the run still fails.")
    parser.add_argument("--all-files", action="store_true", help="Lint every tracked file.")
    parser.add_argument("paths", nargs="*", help="Files to lint, relative to the repository root.")
    args = parser.parse_args(argv)
    if args.all_files:
        tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        paths = tracked.stdout.splitlines()
    else:
        paths = args.paths or changed_files(REPO_ROOT)
    problems = lint(paths, fix=args.fix)
    for problem in problems:
        print(problem)
    print("FAILED" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
