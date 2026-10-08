# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.device_flops import device_flops_for_jax_device


def test_rtx_5090_device_kind_resolves_to_a_peak_flops_entry():
    # Regression: JAX reports "NVIDIA GeForce RTX 5090", which had no entry, so training runs on
    # this card logged "Unknown device type" and reported no MFU.
    flops = device_flops_for_jax_device("NVIDIA GeForce RTX 5090", "bf16")
    assert flops is not None and flops > 0
