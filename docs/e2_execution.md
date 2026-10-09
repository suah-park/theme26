# E2 L1 + gradient staged execution

E2 uses the E1 target AMI/RFN architecture, frozen Berea split, normalization,
sampling, seed 1, Adam learning rate `1e-4`, batch size 8, and AMP off. Both
models start from new random initialization. E1 checkpoints, caches, and run
directories are not inputs to E2.

For each raw 2D prediction and target, the loss is full-output L1 plus `0.1`
times the sum of the two spatial-axis finite-difference error means. Each
difference is the next pixel minus the current pixel. Neither term uses the
observed/missing mask. `primary_mask` is used only for the clipped validation
bracketed-missing MAE that selects `best.pt`.

From the repository root, set `THEME26_GRAYSCALE_RAW` to the original Berea
grayscale RAW file. These settings record the completed `_02` AMI run and its
checkpoint; the existing output directories are not destinations for a new run.
Use new output paths for any future execution. A fresh run refuses to overwrite
existing results.

## Short GPU checks

```bash
python scripts/train.py --config configs/experiments/e2_ami_gpu_pilot.yaml --device cuda
```

This is a separate two-train-step/two-validation-step diagnostic using batch 8,
not part of the ten-epoch training budget. Its stdout, `progress.yaml`, and
`history.yaml` contain phase progress and loss/timing records. After the
complete E2 AMI cache exists, the separate RFN GPU check is:

```bash
python scripts/train.py --config configs/experiments/e2_rfn_gpu_pilot.yaml --device cuda
```

## Full sequence

```bash
python scripts/train.py --config configs/experiments/e2_ami_seed1_ep10.yaml --device cuda
python scripts/infer.py --config configs/inference/e2_cache_seed1_ep10.yaml \
  --checkpoint runs/e2_ami_seed1_ep10_20261007_02/best.pt --device cuda
python scripts/train.py --config configs/experiments/e2_rfn_seed1_ep10.yaml --device cuda
```

The full AMI run has 10 epochs of 3,072 train and 576 validation batches per
epoch. The RFN run has 10 epochs of 1,536 train and 213 validation batches per
epoch, with `drop_last=False`. The cache is generated only from the E2 AMI
`best.pt` and contains raw float32 directional predictions for the E2 RFN.
The RFN pilot is separate from the full RFN run and does not supply a checkpoint
for it. Full RFN starts from a fresh seed-1 initialization.

For each epoch, stdout and `history.yaml` record train/validation total loss,
L1, gradient loss, and validation primary MAE. The best checkpoint follows
primary MAE, not the total loss. These commands have not been run on GPU here.
