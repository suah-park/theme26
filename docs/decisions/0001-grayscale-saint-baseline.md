# 0001 - Grayscale SAINT baseline before topology-aware training

- Date: 2026-09-24
- Status: Accepted
- Owners: Research team
- Approver: `TBD`

## Context

The initial project protocol treated segmented binary pore/solid volumes as model inputs and proposed BCE + Dice training with an early topology-aware extension. That design changed SAINT's original task, output domain and objective simultaneously, making it difficult to distinguish reproduction failures from domain-transfer failures.

The team wants to apply SAINT to digital rock while preserving the original method as closely as possible, then assess whether additional physical or topological constraints are necessary.

## Decision

1. Use grayscale Micro-CT as the model input and target.
2. Use the 11 Sandstones filtered grayscale volumes as the primary dataset.
3. Split at the parent-core level with a target allocation of 8 train, 1 validation and 2 test cores.
4. Begin with x4 z-only sparse sampling.
5. Reproduce the original SAINT architecture and L1 objective before adding new loss terms.
6. Use 2D finite-difference gradient loss as the first isolated ablation after the L1 baseline.
7. Treat provided binary volumes as structure-evaluation references, not training targets.
8. Defer 3D topology-aware loss until the grayscale baseline, frozen segmentation and connectivity analysis are complete.

## Alternatives considered

### Binary input with BCE + Dice

Rejected for the first phase because it changes both the data domain and the loss, and because thresholded predictions hide grayscale reconstruction errors.

### Topology-aware loss from the beginning

Rejected for the first phase because it introduces expensive 3D optimization before upstream SAINT and the digital-rock data path are validated.

### Raw grayscale as the primary dataset

Deferred to robustness testing. Filtered grayscale reduces acquisition noise during pipeline and model validation. Raw grayscale remains important for later external validity.

### Multi-scale training from the beginning

Deferred. x4 is primary because it aligns with the upstream checkpoint and published experiment. x2 and x6 will be added after the x4 baseline is frozen.

## Consequences

- Primary image metrics change from Dice/IoU to missing-slice MAE, PSNR and SSIM.
- A segmentation protocol must be frozen before topology or permeability comparisons.
- Earlier binary-specific BCE/Dice and fixed threshold 0.5 requirements are removed from the main protocol.
- The first implementation milestone is upstream reproduction, not a new architecture.
- Topology-aware modeling becomes a go/no-go decision supported by observed failure modes.

## Required follow-up

- Assign an approver.
- Audit all 11 cores and freeze the exact split manifest.
- Pin the upstream SAINT commit and document its execution environment.
- Resolve upstream code redistribution conditions before copying any source into this repository.

