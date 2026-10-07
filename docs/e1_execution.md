# E1 staged training and inference

실행 명령의 단일 안내입니다. 실험 조건과 모델 수치는 [연구 기준](research_protocol.md), API는 [인터페이스](interface_contract.md)를 참고합니다.

## 설치와 데이터 위치

Python 3.10 이상 개인 환경을 사용합니다. 시스템 Python이나 GPU driver를 바꾸지 않습니다. GPU용 PyTorch는 서버의 driver/GLIBC에 맞는 공식 빌드를 먼저 설치한 뒤 아래를 실행합니다.

```bash
python -m pip install -e '.[training]'
python -m unittest discover -s tests -v
```

CPU tests는 작은 임시 데이터로 코드만 검사합니다. 연구 test 구역이나 GPU 학습 성능 검사가 아닙니다. 설치 없이 검사하려면 Linux는 `export PYTHONPATH=src`, PowerShell은 `$env:PYTHONPATH='src'`를 설정합니다.

데이터 폴더는 Git 저장소 밖에 둡니다. 아래는 예시이며 관리자 지정 경로/개인 quota가 우선입니다.

```text
~/theme/
  theme26/                  # Git clone
  theme-data/Berea/          # grayscale_filtered.raw + binary.raw
~/miniforge3/envs/theme26/   # 개인 환경 예시
```

Grayscale RAW만 E1 학습에 필요합니다. Paired binary는 data audit/QC와 E3/평가에 필요합니다. 코드는 특정 사용자의 절대 경로를 고정하지 않으며 현재 shell/job에서 경로를 전달합니다.

```bash
export THEME26_GRAYSCALE_RAW="$HOME/theme/theme-data/Berea/Berea_2d25um_grayscale_filtered.raw"
ls -lh "$THEME26_GRAYSCALE_RAW"
python -m theme26.data audit --data-root "$HOME/theme/theme-data/Berea" --output-dir outputs/audit
python -m theme26.data qc --data-root "$HOME/theme/theme-data/Berea" --output-dir outputs/spatial_qc
```

PowerShell은 `$env:THEME26_GRAYSCALE_RAW='C:\path\to\Berea_2d25um_grayscale_filtered.raw'`처럼 설정합니다. `export`는 파일을 복사하거나 생성하지 않으며 새 접속에서 다시 설정합니다. 원본은 read-only memmap으로 읽고 fixed crop을 런타임에 생성하므로 별도 대용량 전처리 export가 필요하지 않습니다.

`--data-root`는 두 RAW가 직접 있는 폴더 또는 `Berea/`의 부모를 받습니다. Paired crop 처리 API는 `preprocess_crop`, 데이터 랜덤 sampler는 `TrainRandomCropDataset.sample(rng)`입니다. 후자는 NumPy API이고, 실제 E1 학습은 고정 manifest를 읽는 `ManifestAmiDataset`을 사용합니다.

## GPU 사용 방식 확인

관리자에게 GPU 배정/예약과 직접 실행 또는 Slurm 제출 여부를 확인한 뒤 GPU 명령을 실행합니다. `nvidia-smi`에서 비어 보인다는 이유만으로 사용할 권한이 생기는 것은 아닙니다. `sbatch`가 없으면 Slurm을 직접 설치하지 말고 제출 host/환경을 확인합니다. 아래 `--device cuda` 명령은 사용 권한을 확인한 GPU에서만 실행합니다.

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

For a longer speed and allocated-memory probe on distinct train marginals, use:

```bash
python scripts/train.py --config configs/experiments/e1_ami_gpu_throughput_pilot.yaml --device cuda
```

This separate run uses the target AMI, batch 1, AMP off, 100 train steps and 20
validation steps. The train loader shuffles without replacement within its epoch;
validation is evaluated without optimizer updates. `progress.yaml` and stdout record
start, midpoint and end for each phase, including elapsed wall time, throughput in
samples/second and peak GPU allocated bytes. `history.yaml` also contains separate
train/validation elapsed seconds, samples/second, steps/second and peak bytes. Timing
includes data loading and the phase's progress logging. GPU operations are synchronized
at each recorded point. Allocated bytes exclude CUDA/driver overhead. This pilot is
not the full-training budget; an existing destination is refused.

## Fixed train batch overfit diagnostic

After the GPU pilot, run the separate diagnostic below on an authorized GPU. It selects
`berea_train_000`, sagittal plane 128 from the train manifest and repeats its identical
AMI input and target for 100 optimizer steps. The target architecture and raw full-output
L1 match E1. These 100 steps are a diagnostic, not the full-training budget. The
diagnostic never trains on validation or test records.

```bash
python scripts/train.py --config configs/experiments/e1_ami_fixed_batch_overfit.yaml --device cuda
```

The new run directory contains `resolved_config.yaml`, `metadata.yaml`, `history.yaml`,
`checkpoint.pt`, input/target/primary-mask NPY files, raw prediction NPY files before
and after training, and `comparison.png`. The PNG layout is before | after on the top
row and target | primary mask on the bottom row. Prediction values are clipped to
`[0,1]` only for PNG display. `history.yaml` and stdout report raw L1 on the same
train input at step 0, every 10 steps, and the final step, with elapsed seconds.
The start/end comparison measures fixed-sample fit only; it is neither validation
performance nor evidence of generalization. Choose a fresh `--output-dir` for reruns;
existing directories are refused.

## 2. AMI full training

For one complete epoch from a fresh AMI initialization, use the separate
`configs/experiments/e1_ami_full_epoch1.yaml` config. It processes all 24,576 train
marginals in 3,072 batches and all 4,608 validation marginals in 576 batches at batch
size 8, with no step caps. It logs progress every 100 steps and at each phase's end.
Use its new output directory without `--resume`; the command creates a fresh model
and refuses an existing run directory.

```bash
python scripts/train.py --config configs/experiments/e1_ami_full_epoch1.yaml --device cuda
```

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
When progress logging is enabled, resumed epochs append to `progress.yaml`; previous
progress events remain in place.

## 6. Frozen reconstruction

```bash
python scripts/infer.py --config configs/inference/e1_reconstruct.yaml \
  --checkpoint runs/e1_rfn_target/best.pt \
  --ami-checkpoint runs/e1_ami_target/best.pt --device cuda
```

Default split is validation. After freezing the models, use a separate YAML with `split: test` and a new output directory for final test. `--crop-id` can select one registered crop in that split. Inference processes AMI marginals and RFN axial slices in GPU batches. Each crop saves `prediction.npy`, `primary_mask.npy`, `metadata.yaml`; the completed run saves `manifest.yaml`. Metadata includes axes, coordinates, spacing, normalization, source/checkpoint hashes and primary MAE. This implements MAE for E1 checks; PSNR/SSIM, topology and physics evaluation remain separate evaluation work.

The data CLI owns read-only E1 validation QC. Supply the same Berea RAW file used
for reconstruction and choose a new output directory for each QC run:

```bash
python -m theme26.data e1-validation-crop \
  --result-dir runs/e1_validation_smoke_20261007_01/000 \
  --raw /path/to/Berea_2d25um_grayscale_filtered.raw \
  --output-dir runs/e1_validation_smoke_20261007_01/qc/new_crop_check
python -m theme26.data e1-validation-all \
  --result-dir runs/e1_validation_all_20261007_01 \
  --cache-dir runs/e1_ami_cache_seed1_ep10_20261007_01 \
  --ami-checkpoint runs/e1_ami_seed1_ep10_frozen_20261007_01/best.pt \
  --raw /path/to/Berea_2d25um_grayscale_filtered.raw \
  --output-dir runs/e1_validation_all_20261007_01/qc/new_comparison
```

The first command checks one registered crop and writes comparison figures. The
second checks cache/output/checkpoint identity and compares E0, AMI direction mean,
and RFN on the nine validation blocks from the same core. Existing
`scripts/qc_e1_validation_crop.py` and `scripts/qc_e1_validation_all.py` commands
remain available as compatibility entry points. QC does not update models or
predictions; its output directories must not already exist.

## Slurm

Slurm을 사용하는 서버에서만 `slurm/train_e1.sbatch`을 사용합니다. 이 파일은 자원을 임의로 지정하지 않습니다. 실제 partition/GPU request/CPU/RAM/time은 관리자 제출 예시에 따라 `sbatch` 옵션으로 지정해야 합니다.

```bash
export THEME26_REPO="$PWD"                 # 저장소 루트에서 실행
export THEME26_PYTHON="$(command -v python)" # 활성화한 개인 환경
export THEME26_CONFIG=configs/experiments/e1_ami_gpu_pilot.yaml
# THEME26_GRAYSCALE_RAW도 위에서 지정해야 함
mkdir -p logs
```

제출 형식은 `sbatch <관리자가 안내한 자원 옵션> --output=logs/%x-%j.out slurm/train_e1.sbatch`입니다. `<...>`는 실행 가능한 shell 문법이 아니라 교체할 자리입니다. Slurm이 아닌 서버에서는 관리자 규칙에 따라 GPU 하나를 선택하여 같은 Python 명령을 직접 실행합니다.

재개 제출은 `THEME26_RESUME`에 같은 run의 `last.pt`를 지정합니다. Job 확인/취소는 `squeue -u "$USER"`, `scancel JOB_ID`, 로그는 `tail -f logs/실제로그파일`로 확인합니다. 자원 요청을 모른 상태에서 본학습을 제출하지 않습니다.

## Verification scope

CPU regression tests cover chunked-vs-whole inference, clipping/primary masking, miniature RAW → frozen AMI cache → RFN update → observed-consistent final reconstruction, cache identity rejection and uninterrupted-vs-resumed AMI training. They do not establish target-model GPU fit, Berea reconstruction quality, or convergence.
