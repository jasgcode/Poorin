# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static import reachability for the Poorin prune.

Builds the first-party import graph of the workspace (every package under ``lib/*/src``, plus
``experiments`` and ``tools``), computes the closure of the roots in ``tools/roots.toml``,
explains why a module is reachable, and ranks modules by dominated weight: the lines that become
unreachable when that module alone is removed.

The default graph follows every import statement, including imports inside functions, so lazily
imported code is never reported as unreachable. ``--eager`` follows only imports that run when a
module is first imported (statements outside function bodies), which approximates startup cost.

Usage (from the repository root):
    uv run --no-sync python -m tools.reachability
    uv run --no-sync python -m tools.reachability --why levanter.eval_harness
    uv run --no-sync python -m tools.reachability --rank 25
    uv run --no-sync python -m tools.reachability --eager --rank 25
    uv run --no-sync python -m tools.reachability --json
"""

import argparse
import ast
import json
import tomllib
from collections import deque
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOTS_FILE = REPO_ROOT / "tools" / "roots.toml"
TOP_LEVEL_PACKAGES = ("experiments", "tools")
TEST_DIR_NAMES = frozenset({"tests", "test"})
DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

Graph = dict[str, frozenset[str]]


@dataclass(frozen=True)
class RunnableRoot:
    """A root that ``prune_check`` executes: ``python -m <module> <args>``."""

    name: str
    module: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Roots:
    """The roots file: what must keep running, what must keep existing, what is deleted anyway."""

    runnable: tuple[RunnableRoot, ...]
    kept: tuple[str, ...]
    static_only: tuple[str, ...]
    deleted: tuple[str, ...]

    def patterns(self) -> tuple[str, ...]:
        """Every root pattern that contributes to the closure."""
        return (*(root.module for root in self.runnable), *self.kept, *self.static_only)


@dataclass(frozen=True)
class Summary:
    first_party_lines: int
    reachable_lines: int
    reachable_modules: int
    per_package: dict[str, tuple[int, int]]
    """Top-level package -> (reachable lines, total lines)."""


def load_roots(path: Path) -> Roots:
    data = tomllib.loads(path.read_text())
    return Roots(
        runnable=tuple(RunnableRoot(r["name"], r["module"], tuple(r["args"])) for r in data["runnable"]),
        kept=tuple(data["kept"]["modules"]),
        static_only=tuple(data["kept"]["static_only"]),
        deleted=tuple(data["deleted"]["modules"]),
    )


def is_test_file(rel_path: Path) -> bool:
    """Whether a repo-relative path is test code, which never counts as a module."""
    name = rel_path.name
    return (
        any(part in TEST_DIR_NAMES for part in rel_path.parts)
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def _source_roots(repo_root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (package directory, import root), both absolute."""
    for src in sorted(repo_root.glob("lib/*/src")):
        for package in sorted(src.iterdir()):
            if (package / "__init__.py").is_file():
                yield package, src
    for name in TOP_LEVEL_PACKAGES:
        if (repo_root / name / "__init__.py").is_file():
            yield repo_root / name, repo_root


def workspace_modules(repo_root: Path) -> dict[str, Path]:
    """Every non-test first-party module: dotted name -> file."""
    modules: dict[str, Path] = {}
    for package, import_root in _source_roots(repo_root):
        for py in sorted(package.rglob("*.py")):
            rel = py.relative_to(repo_root)
            if "__pycache__" in rel.parts or is_test_file(rel):
                continue
            parts = list(py.relative_to(import_root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            modules[".".join(parts)] = py
    return modules


def _ancestors(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _absolute_base(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """Absolute dotted prefix named by ``from ... import``. From Marin's infra/ci/select_tests.py."""
    if node.level == 0:
        return node.module or ""
    package = module if is_package else module.rsplit(".", 1)[0] if "." in module else ""
    parts = package.split(".") if package else []
    up = node.level - 1
    if up > len(parts):
        return ""
    return ".".join(parts[: len(parts) - up] + (node.module.split(".") if node.module else []))


def _import_nodes(tree: ast.AST, *, eager: bool) -> Iterator[ast.Import | ast.ImportFrom]:
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node
        for child in ast.iter_child_nodes(node):
            if eager and isinstance(child, DEFERRED_SCOPES):
                continue
            stack.append(child)


def imported_modules(path: Path, module: str, known: Collection[str], *, eager: bool) -> set[str]:
    """Known modules that ``module``'s import statements execute, package ancestors included.

    ``from a.b import c`` contributes ``a.b`` and the candidate ``a.b.c``; only names that are
    real modules survive.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    names: set[str] = set()
    for node in _import_nodes(tree, eager=eager):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            continue
        base = _absolute_base(node, module, is_package)
        if base:
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return {ancestor for name in names for ancestor in _ancestors(name) if ancestor in known} - {module}


def build_graph(modules: Mapping[str, Path], *, eager: bool) -> Graph:
    known = set(modules)
    return {name: frozenset(imported_modules(path, name, known, eager=eager)) for name, path in modules.items()}


def expand(patterns: Iterable[str], modules: Collection[str]) -> set[str]:
    """Resolve root patterns to modules. A trailing ``.*`` selects every module under a package.

    A pattern that matches nothing raises: a stale roots file must not silently shrink the closure.
    """
    out: set[str] = set()
    for pattern in patterns:
        if pattern.endswith(".*"):
            prefix = pattern[:-2] + "."
            matched = {module for module in modules if module.startswith(prefix)}
        else:
            matched = {pattern} if pattern in modules else set()
        if not matched:
            raise ValueError(f"root pattern {pattern!r} matches no module")
        out |= matched
    return out


def _excluded_set(graph: Graph, excluded: Collection[str]) -> set[str]:
    return {m for m in graph if any(m == e or m.startswith(e + ".") for e in excluded)}


def closure(
    graph: Graph, roots: Iterable[str], excluded: Collection[str] = (), removed: Collection[str] = ()
) -> set[str]:
    """Modules reachable from ``roots``.

    An ``excluded`` module, and every module under it, is not traversed: this models deleting a
    package. A ``removed`` module is skipped on its own; its submodules stay reachable through
    their own import edges.
    """
    blocked = _excluded_set(graph, excluded) | set(removed)
    seen: set[str] = set()
    stack = [root for root in roots if root not in blocked]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack.extend(dep for dep in graph.get(module, ()) if dep not in seen and dep not in blocked)
    return seen


def import_chain(graph: Graph, roots: Iterable[str], target: str, excluded: Collection[str] = ()) -> list[str] | None:
    """Shortest import path from any root to ``target``, root first, or None if unreachable."""
    blocked = _excluded_set(graph, excluded)
    parent: dict[str, str | None] = {root: None for root in sorted(roots) if root not in blocked}
    queue = deque(parent)
    while queue:
        module = queue.popleft()
        if module == target:
            break
        for dep in sorted(graph.get(module, ())):
            if dep not in parent and dep not in blocked:
                parent[dep] = module
                queue.append(dep)
    if target not in parent:
        return None
    chain = [target]
    while (previous := parent[chain[-1]]) is not None:
        chain.append(previous)
    return chain[::-1]


def dominated_lines(
    graph: Graph, roots: Collection[str], lines: Mapping[str, int], excluded: Collection[str] = ()
) -> list[tuple[int, str]]:
    """For each reachable non-root module, the lines that become unreachable if it alone is removed.

    Only the module itself is removed. A package's submodules that are imported by name stay
    reachable, so a package is credited with its ``__init__`` and what only it reaches, not with
    its whole subtree. Recomputes the closure once per module. About 450 reachable modules take a
    few seconds, so a dominator-tree algorithm is not needed.
    """
    base = closure(graph, roots, excluded)
    weights = []
    for module in sorted(base - set(roots)):
        lost = base - closure(graph, roots, excluded, removed=(module,))
        weights.append((sum(lines[m] for m in lost), module))
    return sorted(weights, reverse=True)


def line_counts(modules: Mapping[str, Path]) -> dict[str, int]:
    """Physical lines per module."""
    return {name: len(path.read_text(encoding="utf-8").splitlines()) for name, path in modules.items()}


def summarize(repo_root: Path, roots: Roots, *, eager: bool = False) -> Summary:
    modules = workspace_modules(repo_root)
    lines = line_counts(modules)
    graph = build_graph(modules, eager=eager)
    reached = closure(graph, expand(roots.patterns(), modules), roots.deleted)
    per_package: dict[str, tuple[int, int]] = {}
    for name, count in lines.items():
        package = name.split(".")[0]
        reachable, total = per_package.get(package, (0, 0))
        per_package[package] = (reachable + (count if name in reached else 0), total + count)
    return Summary(
        first_party_lines=sum(lines.values()),
        reachable_lines=sum(lines[m] for m in reached),
        reachable_modules=len(reached),
        per_package=per_package,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static import reachability for the Poorin prune.")
    parser.add_argument("--eager", action="store_true", help="Follow only imports that run at import time.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--why", metavar="MODULE", help="Print the shortest import chain from a root to MODULE.")
    mode.add_argument("--rank", type=int, metavar="N", help="Print the N modules with the largest dominated weight.")
    mode.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args(argv)

    roots = load_roots(ROOTS_FILE)
    if args.json or (args.why is None and args.rank is None):
        summary = summarize(REPO_ROOT, roots, eager=args.eager)
        if args.json:
            print(json.dumps(asdict(summary), indent=2, sort_keys=True))
            return 0
        reachable, total = summary.reachable_lines, summary.first_party_lines
        print(f"reachable: {summary.reachable_modules} modules, {reachable:,} of {total:,} lines")
        for package, (reachable, total) in sorted(summary.per_package.items(), key=lambda kv: -kv[1][1]):
            print(f"  {package:16s} {reachable:>9,} / {total:>9,}")
        return 0

    modules = workspace_modules(REPO_ROOT)
    graph = build_graph(modules, eager=args.eager)
    root_modules = expand(roots.patterns(), modules)
    if args.why is not None:
        chain = import_chain(graph, root_modules, args.why, roots.deleted)
        print("unreachable" if chain is None else "\n  <- ".join(reversed(chain)))
        return 0
    for weight, module in dominated_lines(graph, root_modules, line_counts(modules), roots.deleted)[: args.rank]:
        print(f"{weight:>8,}  {module}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
