# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from fray.device_flops import device_flops_for_jax_device


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [("bf16", 209.5e12), ("amp_bf16", 209.5e12), ("fp16", 209.5e12), ("fp32", 104.8e12)],
)
def test_rtx_5090_device_kind_resolves_to_its_dense_peak_flops(dtype, expected):
    # Regression: JAX reports "NVIDIA GeForce RTX 5090", which had no entry, so training runs on
    # this card logged "Unknown device type" and reported no MFU. MFU divides by NVIDIA's dense
    # rate; the datasheet's sparsity figure is twice as large and would halve every reported MFU.
    assert device_flops_for_jax_device("NVIDIA GeForce RTX 5090", dtype) == expected
