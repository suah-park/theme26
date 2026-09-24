# 0004 - Clarify collaboration, axis, boundary and configurable defaults

- Date: 2026-09-24
- Status: Accepted

## Context

An audit of the first interface skeleton found mismatches between the collaboration guide, Slurm command, research protocol and code signatures. It also found defaults that had been inserted before the team explicitly selected them.

## Decision

1. Use `main` plus short feature branches and pull requests; do not require a separate `develop` branch.
2. Use `scripts/train.py` and `scripts/infer.py` as command-line entry points.
3. Keep canonical semantic volume order as `[X,Y,Z]`; the sparse interpolation axis is `Z`.
4. Represent preprocessing/evaluation masks as NumPy `[X,Y,Z]` and add batch/channel dimensions only for Torch training `[B,1,X,Y,Z]`.
5. Do not freeze the E1/E2 training-loss mask before the upstream stage-loss audit. Primary evaluation remains restricted to bracketed missing positions.
6. For x4 sampling, exclude missing positions without observed slices on both sides from the primary image metric. With depth 256 and offset 0, this gives 189 primary missing slices; `z=253,254,255` are boundary diagnostics only.
7. E0 may replicate the final observed slice only to complete a full-volume export. Replicated boundary values are not primary image-metric samples.
8. Require soft-clDice skeletonization iterations from config; do not silently default to 10.
9. Treat CPU/RAM, wall time, mixed precision and batch-size values in the Slurm guide as placeholders until the actual server allocation is known.
10. Keep the user-uploaded SAINT paper PDF unchanged.

## Consequences

- Data code must provide a `primary_metric_mask` separately from the full `missing_mask`.
- Learned models and E0 use the same bracketed mask for the headline image comparison.
- Loss implementations support masks, but experiment config selects the training region only after upstream behavior is documented.
- Cluster comments must be resolved before a full training submission.
