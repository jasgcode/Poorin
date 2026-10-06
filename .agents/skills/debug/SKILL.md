---
name: debug
description: Diagnose a stated code, JAX, Levanter, Marin, Zephyr, or GPU fault or startup/performance regression; do not activate for ordinary implementation or optimization without a symptom.
---

# Debug

Keep working notes in the active task. Do not add repository debug-log files.

## GPU faults

For a fault on the local RTX 5090 (cuDNN attention, the compilation cache, attention backends,
missing MFU), read `docs/poorin/spike-0-rtx5090.md` first: it records what is known to fail on
sm_120 and the workaround for each.

## Pipeline faults

For a slow Zephyr pipeline, stragglers, data skew, or worker failures, read
`lib/zephyr/AGENTS.md`, then `lib/zephyr/OPS.md` → Diagnostic Patterns, Observability.

## Code bugs

For code bugs, reproduce the failure, identify the smallest falsifiable
hypothesis, change one cause at a time, and test the behavior that failed. Let
exceptions propagate unless added context changes the diagnosis.
