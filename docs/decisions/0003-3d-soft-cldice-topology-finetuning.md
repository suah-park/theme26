# 0003 - Add 3D soft-clDice topology fine-tuning as E3

- Date: 2026-09-24
- Status: Accepted
- Supersedes: topology-scope parts of 0001 and 0002

## Context

E0-E2 isolate the effects of learned SAINT interpolation and 2D grayscale gradient regularization, but neither objective directly constrains 3D pore connectivity. The submitted research framing includes topology and physical-property preservation, so topology metrics alone would provide evaluation but no topology-aware method contribution.

The additional method must remain feasible on `256^3` blocks, preserve the grayscale SAINT output, and avoid introducing a large architecture search. It also must keep permeability out of the training loop.

## Decision

1. Add E3 after E2 for every headline seed.
2. Initialize E3 from the same seed's E2 best checkpoint.
3. Freeze AMI initially and fine-tune RFN on the assembled 3D grayscale volume.
4. Compose observed input slices and predicted missing slices into a full `256^3` volume before topology conversion.
5. Convert grayscale to soft pore probability with a differentiable sigmoid using a fixed threshold and temperature selected only from training/validation data.
6. Define topology loss as `0.5 * soft Dice + 0.5 * 3D soft-clDice`.
7. Use `L_E3 = L1 + 0.10 * L_grad + 0.10 * L_topo` without a weight sweep.
8. Select the E3 checkpoint by validation total E3 loss. Never use test topology or permeability for selection.
9. Treat soft-clDice as a connectivity-aware surrogate, not a guarantee of all pore topology.
10. Evaluate E0-E3 with the same frozen hard segmentation, morphology, connectivity and permeability protocol.

## Alternatives considered

### Evaluation-only topology

Rejected as the complete study design because it cannot test whether an explicit topology-aware objective adds value beyond E2. Topology and permeability metrics remain necessary as independent evaluation.

### Persistent-homology loss

Deferred because a true 3D implementation and its memory/runtime behavior create more integration risk than soft-clDice within the current experiment. It may be reconsidered after the E3 pipeline is stable.

### Permeability loss

Rejected because differentiating through a flow solver would add substantial computational and implementation complexity and could make the physical evaluation non-independent.

### Hard threshold or connected-component loss

Rejected for training because hard thresholding and ordinary component labeling are not differentiable. They remain evaluation operations.

## Consequences

- Training and validation require aligned grayscale and binary volumes for E3.
- The binary volume is an auxiliary topology target, never the SAINT grayscale output target.
- The data audit must confirm pore polarity and grayscale/binary alignment.
- E3 starts with batch size 1 and may need mixed precision, gradient checkpointing or accumulation.
- Results must report both grayscale fidelity and independent hard topology/property metrics.
- E2 versus E3 is the isolated comparison for the topology constraint.
- Threshold, temperature, clDice iterations and protocol version become required run metadata.

## Required verification

- Constant, disconnected and connected 3D phantom tests for soft Dice and soft-clDice.
- A gradient-flow test confirming that observed slices remain fixed and missing predictions receive gradients.
- Same-seed E2/E3 comparison on identical registered validation and test blocks.
- Independent Euler characteristic, connected-component, z-percolation and permeability evaluation.
