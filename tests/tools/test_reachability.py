# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.reachability on a small synthetic workspace."""

from pathlib import Path

import pytest

from tools.reachability import (
    build_graph,
    closure,
    dominated_lines,
    import_chain,
    line_counts,
    workspace_modules,
)

FILES = {
    "lib/alpha/src/alpha/__init__.py": "",
    "lib/alpha/src/alpha/core.py": "from alpha import util\n",
    "lib/alpha/src/alpha/util.py": "X = 1\n",
    "lib/alpha/src/alpha/lazy.py": (
        "from . import util\n\n\ndef load():\n    import alpha.heavy\n\n    return alpha.heavy\n"
    ),
    "lib/alpha/src/alpha/heavy.py": "Y = 2\nZ = 3\n",
    "lib/alpha/src/alpha/orphan.py": "W = 4\n",
    "lib/alpha/tests/test_core.py": "import alpha.orphan\n",
    "experiments/__init__.py": "",
    "experiments/run.py": "import alpha.core\nfrom alpha import lazy\n",
    "experiments/test_run.py": "import alpha.orphan\n",
}
ROOT = "experiments.run"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_workspace_modules_excludes_test_files(repo):
    modules = workspace_modules(repo)
    assert "alpha.orphan" in modules
    assert "experiments.test_run" not in modules
    assert not any(name.startswith("alpha.tests") for name in modules)


def test_closure_follows_function_local_imports_only_in_the_full_graph(repo):
    modules = workspace_modules(repo)
    full = closure(build_graph(modules, eager=False), [ROOT])
    eager = closure(build_graph(modules, eager=True), [ROOT])
    assert "alpha.heavy" in full
    assert "alpha.heavy" not in eager
    assert "alpha.orphan" not in full


def test_from_and_relative_imports_of_submodules_are_edges(repo):
    graph = build_graph(workspace_modules(repo), eager=True)
    assert "alpha.util" in graph["alpha.core"]  # from alpha import util
    assert "alpha.util" in graph["alpha.lazy"]  # from . import util
    assert "alpha.lazy" in graph[ROOT]  # from alpha import lazy


def test_excluding_a_module_also_excludes_what_only_it_reached(repo):
    graph = build_graph(workspace_modules(repo), eager=False)
    reached = closure(graph, [ROOT], excluded=["alpha.lazy"])
    assert "alpha.lazy" not in reached
    assert "alpha.heavy" not in reached
    assert "alpha.util" in reached  # still reached through alpha.core


def test_import_chain_is_the_shortest_path_from_a_root(repo):
    graph = build_graph(workspace_modules(repo), eager=False)
    assert import_chain(graph, [ROOT], "alpha.heavy") == [ROOT, "alpha.lazy", "alpha.heavy"]
    assert import_chain(graph, [ROOT], "alpha.orphan") is None


def test_dominated_lines_count_only_code_that_becomes_unreachable(repo):
    modules = workspace_modules(repo)
    graph = build_graph(modules, eager=False)
    lines = line_counts(modules)
    weights = {module: weight for weight, module in dominated_lines(graph, [ROOT], lines)}
    # Removing alpha.lazy loses it and alpha.heavy. alpha.util survives through alpha.core.
    assert weights["alpha.lazy"] == lines["alpha.lazy"] + lines["alpha.heavy"]
    assert weights["alpha.core"] == lines["alpha.core"]


def test_dominated_lines_of_a_package_do_not_include_submodules_imported_directly(repo):
    # Removing a package's __init__ does not remove alpha.core or alpha.lazy: the root imports
    # them by name. Ranking must not credit a package with its whole subtree.
    modules = workspace_modules(repo)
    graph = build_graph(modules, eager=False)
    lines = line_counts(modules)
    weights = {module: weight for weight, module in dominated_lines(graph, [ROOT], lines)}
    assert weights["alpha"] == lines["alpha"]
