# E1 staged training and inference

## Fixed rules

- AMI and RFN training use raw all-output L1. No test crops enter training/cache generation.
- Best checkpoint uses clipped `[0,1]` validation MAE on bracketed missing voxels only.
- For depth 256/x4, 64 slices are observed, 189 bracketed missing slices are primary, and the last 3 slices are excluded from primary metrics.
- AMI sagittal/coronal share weights. RFN learns from their raw float32 directional predictions, not ground-truth substitutes.
- Final inference clips RFN output and reinserts observed input slices; saved volume is float32 `[X,Y,Z]` in `[0,1]`.
- The source RAW SHA256, protocol identity, AMI checkpoint SHA256 and cache array SHA256 are checked to reject stale/mixed data.
- Cache contains 48 train + 9 validation crops: two float32 `256^3` arrays per crop, about 7.125 GiB. Targets are read from RAW, not duplicated on disk. Hash checks scan the cache at startup.

## 1. Connection and GPU pilot

```bash
export THEME26_GRAYSCALE_RAW=/path/to/Berea_2d25um_grayscale_filtered.raw
python scripts/train.py --config configs/experiments/e1_ami_berea_pilot.yaml --device cpu
python scripts/train.py --config configs/experiments/e1_ami_gpu_pilot.yaml --device cuda
```

CPU pilot uses the small architecture; GPU pilot uses the target architecture, batch 1 and one train/validation step. These are probes, not scientific results. Check `history.yaml` for elapsed time and peak GPU allocated bytes. Allocated bytes exclude some CUDA/driver memory; compare with `nvidia-smi`. A one-step memory probe does not establish convergence or a full-epoch runtime. Before headline runs, measure a longer pilot and perform a one-batch overfit check. AMP/worker/batch changes require a new probe.

## 2. AMI full training

Fill `training.batch_size` and `training.epochs` in `configs/experiments/e1_ami_target.yaml` after inspecting the GPU pilot. These values intentionally remain null until a user decision. They are validated before execution; no epoch/batch budget is guessed. Learning rate remains the existing pilot value `1e-4`. The template keeps workers=0 and AMP off; enable/tune them only after a probe.

```bash
python scripts/train.py --config configs/experiments/e1_ami_target.yaml --device cuda
```

Each run saves `resolved_config.yaml`, `history.yaml`, `last.pt` and `best.pt`. Logs separate raw training L1, raw validation L1 and clipped validation primary MAE. For RFN, the validation loader includes only primary axial slices; AMI covers both directions with an explicit primary mask. Each primary voxel receives equal weight. AMI's two directions are weighted equally because the fixed crops are equal-sized cubes.

## 3. AMI → cache

```bash
python scripts/infer.py --config configs/inference/e1_cache.yaml \
  --checkpoint runs/e1_ami_target/best.pt --device cuda
```

The complete cache is written to `runs/e1_ami_cache/`. AMI runs frozen with gradients disabled in marginal batches. Its raw predictions are not clipped or quantized. `manifest.yaml` is written only when all crops complete. Existing destination directories are refused; choose a new `--output-dir` after a failed or different run and update the RFN config accordingly. No test cache is generated.

## 4. RFN training

Fill `training.batch_size` and `training.epochs` in `configs/experiments/e1_rfn_target.yaml` after a probe. Ensure `data.ami_checkpoint` and `data.cache_dir` refer to the same AMI/cache pair. The RFN GPU pilot uses those same paths, so generate cache before running it.

```bash
python scripts/train.py --config configs/experiments/e1_rfn_gpu_pilot.yaml --device cuda
python scripts/train.py --config configs/experiments/e1_rfn_target.yaml --device cuda
```

RFN input `[B,2,256,256]` combines matching sagittal/coronal axial slices. Target is grayscale `[B,1,256,256]`. AMI is not updated in this stage. Train includes all 256 phases; validation includes only 189 primary phases per crop.

## 5. Resume

```bash
python scripts/train.py --config configs/experiments/e1_ami_target.yaml \
  --resume runs/e1_ami_target/last.pt --epochs TOTAL_EPOCH_BUDGET --device cuda
```

Resume starts at the next saved epoch, not at an interrupted batch. `--epochs` is the total budget, not additional epochs. Use the same run directory and its `last.pt`; best checkpoint and history are preserved. Model, optimizer, AMP scaler, Python/NumPy/Torch RNG and DataLoader shuffle state are restored. Only output directory and epoch budget are excluded from config equality; changing model/data/loss/worker settings requires a new run. GPU bitwise reproducibility is not claimed across devices/software. Old checkpoints without new source/metric/RNG metadata are rejected for resume/cache; old pilot scores are not compared with the new best metric.

## 6. Frozen reconstruction

```bash
python scripts/infer.py --config configs/inference/e1_reconstruct.yaml \
  --checkpoint runs/e1_rfn_target/best.pt \
  --ami-checkpoint runs/e1_ami_target/best.pt --device cuda
```

Default split is validation. After freezing the models, use a separate YAML with `split: test` and a new output directory for final test. `--crop-id` can select one registered crop in that split. Inference processes AMI marginals and RFN axial slices in GPU batches. Each crop saves `prediction.npy`, `primary_mask.npy`, `metadata.yaml`; the completed run saves `manifest.yaml`. Metadata includes axes, coordinates, spacing, normalization, source/checkpoint hashes and primary MAE. This implements MAE for E1 checks; PSNR/SSIM, topology and physics evaluation remain separate evaluation work.

## Slurm

`slurm/train_e1.sbatch` requires `THEME26_REPO`, `THEME26_PYTHON`, `THEME26_CONFIG`, `THEME26_GRAYSCALE_RAW`. Optional `THEME26_RESUME` supplies `last.pt`. Set partition, GPU request syntax, wall time, CPU/RAM and log location as `sbatch` flags based on the actual cluster. No cluster name/resource budget is assumed. Create the log directory before submission.

## Verification scope

CPU regression tests cover chunked-vs-whole inference, clipping/primary masking, miniature RAW → frozen AMI cache → RFN update → observed-consistent final reconstruction, cache identity rejection and uninterrupted-vs-resumed AMI training. They do not establish target-model GPU fit, Berea reconstruction quality, or convergence.
