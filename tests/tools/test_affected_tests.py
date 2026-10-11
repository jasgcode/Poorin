# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for tools.affected_tests on a small synthetic workspace."""

from pathlib import Path

import pytest

from tools.affected_tests import (
    LOCAL_SAFE_MARKERS,
    Selection,
    Suite,
    all_suites,
    pytest_runs,
    select,
)

FILES = {
    "pyproject.toml": '[tool.uv.workspace]\nmembers = ["lib/alpha", "lib/beta"]\n',
    "uv.lock": "",
    "lib/alpha/pyproject.toml": "",
    "lib/alpha/src/alpha/__init__.py": "",
    "lib/alpha/src/alpha/core.py": "from alpha import util\n",
    "lib/alpha/src/alpha/util.py": "X = 1\n",
    "lib/alpha/src/alpha/other.py": "Y = 1\n",
    "lib/alpha/tests/conftest.py": "",
    "lib/alpha/tests/helpers.py": "import alpha.other\n\nTHING = 1\n",
    "lib/alpha/tests/test_core.py": "import alpha.core\n",
    "lib/alpha/tests/test_other.py": "from helpers import THING\n",  # pythonpath = ["tests"]
    "lib/alpha/tests/test_relative.py": "from .helpers import THING\n",
    "lib/alpha/tests/data/sample.txt": "x\n",
    "lib/alpha/tests/nested/conftest.py": "",
    "lib/alpha/tests/nested/test_nested.py": "",
    "lib/beta/pyproject.toml": "",
    "lib/beta/src/beta/__init__.py": "",
    "lib/beta/src/beta/api.py": "def load():\n    import alpha.util\n\n    return alpha.util\n",
    "lib/beta/tests/test_api.py": "from beta import api\n",
    "lib/gamma/tests/test_gamma.py": "import alpha.util\n",  # not a workspace member
    "experiments/__init__.py": "",
    "experiments/README.md": "",
    "experiments/run.py": "import alpha.core\n",
    "experiments/test_run.py": "import experiments.run\n",
    "tests/__init__.py": "",
    "tests/support.py": "import beta.api\n\nload = beta.api.load\n",
    "tests/test_root.py": "from tests.support import load\n",
    "docs/index.md": "",
}
ROOT = Suite("", ("tests", "experiments"))
ALPHA = Suite("lib/alpha", ("lib/alpha/tests",))
BETA = Suite("lib/beta", ("lib/beta/tests",))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def test_changing_a_module_selects_tests_that_import_it_directly_lazily_or_through_other_modules(repo):
    # core imports util; beta.api imports it inside a function; experiments.run imports core;
    # tests/support.py (a root-suite helper) imports beta.api.
    assert select(["lib/alpha/src/alpha/util.py"], repo) == [
        Selection(ROOT, ("experiments/test_run.py", "tests/test_root.py")),
        Selection(ALPHA, ("lib/alpha/tests/test_core.py",)),
        Selection(BETA, ("lib/beta/tests/test_api.py",)),
    ]


def test_tests_reach_changed_modules_through_suite_local_helpers(repo):
    # helpers.py is imported by its pythonpath name in one test and relatively in the other.
    assert select(["lib/alpha/src/alpha/other.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_a_changed_helper_selects_the_tests_that_import_it(repo):
    assert select(["lib/alpha/tests/helpers.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_deleting_a_module_selects_the_tests_that_still_import_it(repo):
    (repo / "lib/alpha/src/alpha/other.py").unlink()
    assert select(["lib/alpha/src/alpha/other.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_deleting_a_test_package_selects_nothing(repo):
    assert select(["tests/gone/__init__.py", "tests/gone/test_gone.py"], repo) == []


def test_a_changed_conftest_selects_the_tests_below_it(repo):
    assert select(["lib/alpha/tests/nested/conftest.py"], repo) == [
        Selection(ALPHA, ("lib/alpha/tests/nested/test_nested.py",)),
    ]
    assert select(["lib/alpha/tests/conftest.py"], repo) == [
        Selection(
            ALPHA,
            (
                "lib/alpha/tests/nested/test_nested.py",
                "lib/alpha/tests/test_core.py",
                "lib/alpha/tests/test_other.py",
                "lib/alpha/tests/test_relative.py",
            ),
        ),
    ]


def test_root_manifests_select_nothing_but_a_library_manifest_selects_its_suite(repo):
    # The whole tree runs once per phase with --all (D70), not on every manifest change.
    assert select(["uv.lock"], repo) == []
    assert select(["pyproject.toml"], repo) == []
    assert select(["lib/beta/pyproject.toml"], repo) == [Selection(BETA, BETA.test_dirs)]


def test_all_suites_selects_every_member_suite_whole(repo):
    assert all_suites(repo) == [
        Selection(ROOT, ROOT.test_dirs),
        Selection(ALPHA, ALPHA.test_dirs),
        Selection(BETA, BETA.test_dirs),
    ]


def test_cluster_configs_and_their_test_fixtures_select_every_member_suite(repo):
    # Rigging, Iris, and the root suite read config/; nothing imports a YAML file.
    every_suite = [Selection(ROOT, ROOT.test_dirs), Selection(ALPHA, ALPHA.test_dirs), Selection(BETA, BETA.test_dirs)]
    assert select(["config/local.yaml"], repo) == every_suite
    assert select(["config/testing/marin.yaml"], repo) == every_suite


def test_a_grug_module_selects_the_contract_test_that_imports_variants_by_computed_name(repo):
    # The contract test loads each variant with importlib.import_module, which no import graph sees.
    write_files(
        repo,
        {
            "experiments/grug/__init__.py": "",
            "experiments/grug/base/__init__.py": "",
            "experiments/grug/base/model.py": "X = 1\n",
            "tests/test_grug_variant_contracts.py": (
                "import importlib\n\n\ndef test_variant():\n"
                '    importlib.import_module("experiments.grug." + "base.model")\n'
            ),
        },
    )
    assert select(["experiments/grug/base/model.py"], repo) == [
        Selection(ROOT, ("tests/test_grug_variant_contracts.py",))
    ]


def test_adding_or_deleting_a_module_selects_the_test_that_lists_every_module(repo):
    # The architecture test compares workspace_modules with a document and imports neither.
    write_files(repo, {"lib/alpha/src/alpha/new.py": "", "tests/tools/test_architecture.py": ""})
    architecture = Selection(ROOT, ("tests/tools/test_architecture.py",))
    assert select(["lib/alpha/src/alpha/new.py"], repo, added=["lib/alpha/src/alpha/new.py"]) == [architecture]
    assert select(["docs/poorin/architecture.md"], repo) == [architecture]
    assert architecture not in select(["lib/alpha/src/alpha/util.py"], repo)
    (repo / "lib/alpha/src/alpha/other.py").unlink()
    assert select(["lib/alpha/src/alpha/other.py"], repo) == [
        architecture,
        Selection(ALPHA, ("lib/alpha/tests/test_other.py", "lib/alpha/tests/test_relative.py")),
    ]


def test_test_data_selects_its_suite_but_experiment_files_and_docs_select_nothing(repo):
    assert select(["lib/alpha/tests/data/sample.txt"], repo) == [Selection(ALPHA, ALPHA.test_dirs)]
    assert select(["experiments/README.md", "docs/index.md"], repo) == []


def test_each_suite_runs_on_cpu_in_utc_with_the_complete_local_marker_expression():
    root_run, alpha_run = pytest_runs([Selection(ROOT, ("tests/test_root.py",)), Selection(ALPHA, ALPHA.test_dirs)])
    for run, paths in ((root_run, ("tests/test_root.py",)), (alpha_run, ALPHA.test_dirs)):
        assert run.env == {"JAX_PLATFORMS": "cpu", "TZ": "UTC"}
        assert run.args[run.args.index("-m") + 1] == LOCAL_SAFE_MARKERS
        assert run.args[-len(paths) :] == paths


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
