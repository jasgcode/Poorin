# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from poorin.episodes.build import load_reader

from experiments.datasets.episodes import libero, ssv2, vpt


def test_example_readers_resolve_by_import_path():
    for module in (libero, ssv2, vpt):
        assert load_reader(f"{module.__name__}:read") is module.read


def test_the_libero_handle_plans_without_downloading():
    handle = libero.libero_spatial()
    assert handle.name == "examples/libero-spatial"
    assert handle.path().endswith("examples/libero-spatial/2026.10.11")
