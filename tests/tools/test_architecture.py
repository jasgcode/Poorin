# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""docs/poorin/architecture.md lists exactly the workspace's first-party modules."""

import re

from tools.reachability import REPO_ROOT, workspace_modules

ARCHITECTURE = REPO_ROOT / "docs" / "poorin" / "architecture.md"
MODULE_ROW = re.compile(r"^\| `([\w.]+)` \|", re.MULTILINE)


def test_architecture_lists_every_workspace_module_once():
    rows = MODULE_ROW.findall(ARCHITECTURE.read_text())
    modules = set(workspace_modules(REPO_ROOT))

    assert sorted(set(rows) - modules) == [], "rows for modules that no longer exist"
    assert sorted(modules - set(rows)) == [], "modules without a row"
    assert len(rows) == len(set(rows)), "a module has more than one row"
