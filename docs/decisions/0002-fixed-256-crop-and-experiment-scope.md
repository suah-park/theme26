# 0002 - Fixed 256-cube crop and primary experiment scope

- Date: 2026-09-24
- Status: Accepted

## Context

The source sandstone volumes are `1000 x 1000 x 1000`. Using a full volume for every data operation is inefficient, while switching between multiple crop sizes would make training and property evaluation difficult to compare. The project also needs a deliberately narrow primary experiment rather than a scale or loss-weight search.

## Decision

1. Use `256 x 256 x 256` as the only spatial crop size for training, validation and test blocks.
2. Draw training crops online after the parent-core split.
3. Freeze validation and test crop coordinates in versioned manifests.
4. Use x4 z-only sparse sampling as the only primary scale.
5. Use SAINT sparse-axis `patch_size=10`; this is an internal 2D strip length and is distinct from the 3D crop size.
6. Compare only E0 linear interpolation, E1 SAINT + L1 and E2 SAINT + L1 + `0.1 * gradient loss` in the primary study.
7. Use seeds 1, 2 and 3 for learned experiments.
8. Keep 3D topology loss, GANs, architecture search, x2/x6 and raw-grayscale robustness outside the current experiment scope.
9. Measure topology and permeability after inference using a frozen segmentation and evaluation protocol.

## Consequences

- All primary results share the same physical block extent.
- No result can be attributed to a crop-size or gradient-weight search.
- The data loader must preserve the parent-core identity and crop coordinates in prediction metadata.
- Property claims apply to the registered `256^3` blocks and should not be presented as whole-core measurements.
