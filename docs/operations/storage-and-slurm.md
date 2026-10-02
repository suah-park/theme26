# Data storage and GPU Slurm execution

This document defines the portable server layout. Replace `<group>` and the two cluster-specific Slurm resource lines after receiving the server account; do not hard-code personal workstation paths.

## Persistent storage layout

```text
/project/<group>/theme26/                 # Git clone and small configs
/project/<group>/data/11_sandstones/
  raw/                                    # immutable 1000^3 source volumes
  processed/                              # normalized or chunked representations
  splits/                                 # spatial-region split and fixed crop manifests
/project/<group>/runs/theme26/
  E0_interpolation/
  E1_saint_l1/
  E2_saint_l1_grad/
  E3_saint_l1_grad_topology/
/project/<group>/envs/theme26/             # server Python environment
```

Raw volumes, generated arrays, checkpoints, predictions and solver outputs remain outside Git. Training reads a `256^3` crop from read-only Berea RAW memory maps within its train region and generates sagittal/coronal SAINT samples at runtime. Validation and test use registered crop coordinates.

## Compute allocation

The values below are provisional placeholders only. After the server account is issued, update them to match the actual partition policy, GPU VRAM and CPU/RAM limits before submitting a full job.

```text
GPU: 1
CPU cores: 8
CPU RAM: 64 GB
Initial batch size: 8
Mixed precision: enabled
```

`TODO(cluster)`: confirm the GPU model/VRAM, partition, maximum wall time, CPU allocation, RAM limit and GPU request syntax. The listed batch sizes are first-probe values, not frozen experiment settings. GPU VRAM stores the model and active batch. CPU RAM prepares data; persistent storage holds source volumes and artifacts. If CUDA reports out-of-memory, keep the `256^3` protocol and apply mixed precision, a smaller batch, gradient checkpointing or accumulation before changing the scientific setup.

## Slurm training template

Create the log directory before submitting because Slurm opens the output file before the script begins.

```bash
mkdir -p logs
```

```bash
#!/bin/bash
#SBATCH --job-name=saint-e1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# TODO(cluster): replace partition, GPU syntax, CPU/RAM and wall time after allocation.

set -euo pipefail

cd /project/<group>/theme26

/project/<group>/envs/theme26/bin/python scripts/train.py \
  --config configs/experiments/e1_ami_target.yaml
```

`e1_ami_target.yaml`은 GPU memory probe 뒤에 확정할 본학습 설정 파일명입니다. Pilot/smoke
설정을 본학습에 사용하지 않습니다. Partition name, GPU request syntax와
`/project/<group>` path는 cluster에 맞춰 수정합니다.

## GPU check

```bash
nvidia-smi
```

```bash
/project/<group>/envs/theme26/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The first command reports GPU utilization and VRAM. The Python command must report `True` before training.

## Job commands

```bash
sbatch slurm/train_e1.sbatch
squeue -u "$USER"
scancel <job-id>
tail -f logs/saint-e1-<job-id>.out
```

## Required execution order

```text
data audit and manifests
  -> CPU dry run
  -> one-batch GPU overfit test
  -> E0 interpolation
  -> E1 SAINT + L1
  -> E2 SAINT + L1 + 0.1 gradient
  -> E3 E2-checkpoint + 0.1 topology fine-tuning
  -> frozen inference
  -> grayscale metrics
  -> frozen segmentation
  -> topology and permeability metrics
```

Do not submit a full training job until the one-batch overfit test shows a decreasing loss and the output shape and missing-slice mask have been verified.

Before an E3 headline job, also verify grayscale/binary alignment and pore polarity, freeze the training/validation-derived soft threshold and temperature, and run a 3D phantom gradient-flow test. Test topology or permeability must not be used to tune E3.


## Current implementation status

Only Berea grayscale_filtered/binary RAW files are required for the primary experiment.
Run `python -m theme26.data audit` and `python -m theme26.data qc` with the configured data root.
The E1 training YAML, AMI/RFN model, L1/gradient functions and E1 engine are implemented.
Real Berea AMI/cache/RFN and frozen inference are connected. E2/E3 and full scientific metrics remain pending.
Use `slurm/train_e1.sbatch` with actual cluster flags and environment paths. Full epoch/batch budgets remain null until GPU pilot. See [E1 execution](../e1_execution.md).
Install the training extra (`pip install -e '.[training]'`) before running training scripts.
