# Research Protocol

## 1. Objective

SAINT를 grayscale sandstone Micro-CT에 적용하여 sparse z-slice로부터 dense 3D volume을 복원하고, 영상 품질, pore topology와 z-direction transport property를 함께 평가합니다.

연구는 다음 비교로 구성합니다.

1. Linear interpolation과 learned SAINT 비교
2. L1 baseline과 gradient regularization 비교
3. Gradient baseline과 3D topology-aware fine-tuning 비교

## 2. Scope

### Included

- 11 Sandstones filtered grayscale
- Parent-core train/validation/test split
- `256 x 256 x 256` crop
- Z축 x4 sparse sampling
- SAINT AMI/RFN reimplementation
- L1, 2D gradient와 3D soft Dice + soft-clDice
- Grayscale, morphology, connectivity와 permeability 평가

### Excluded

- x2/x6 primary experiments
- Raw-grayscale robustness
- Binary-only reconstruction
- GAN, diffusion와 architecture search
- Persistent-homology 및 permeability training loss
- 실제 LR/HR scanner pair 주장
- Test set 기반 threshold 또는 hyperparameter 선택

## 3. Dataset

### Source

- Name: 11 Sandstones: Raw, Filtered and Segmented Data
- Portal: <https://digitalporousmedia.org/datasets/317/>
- DOI: <https://doi.org/10.17612/F4H1-W124>
- License: ODC-By 1.0
- Source shape: `1000 x 1000 x 1000`
- Voxel spacing: `2.25 um` isotropic
- Primary grayscale: `*_grayscale_filtered.raw`
- Topology reference: aligned binary volume

Binary volume은 grayscale 모델의 입력 또는 reconstruction target이 아닙니다. E3 training/validation의 auxiliary topology target과 모든 모델의 공통 평가 reference로 사용합니다.

### Provenance

각 source core에 다음 metadata를 저장합니다.

```text
dataset_doi
download_url
download_date
source_filename
checksum_sha256
shape_xyz
dtype
endianness
voxel_spacing_um_xyz
intensity_statistics
artifact_notes
```

Shape, dtype와 endianness는 파일명만 보고 추정하지 않고 metadata와 file size로 확인합니다.

### Split

```text
Train:      8 parent cores
Validation: 1 parent core
Test:       2 parent cores
```

- Parent core를 먼저 분리한 뒤 crop을 생성합니다.
- 동일 core의 crop을 여러 split에 포함하지 않습니다.
- Training crop은 train core 내부에서 추출합니다.
- Validation/test crop과 sampling offset은 manifest에 고정합니다.
- 정확한 core ID는 첫 headline training 전에 동결합니다.

## 4. Data Contract

Canonical axis는 `[X,Y,Z]`입니다.

```text
NumPy dense volume:       [X,Y,Z]
Torch dense batch:        [B,C,X,Y,Z]
NumPy preprocessing mask: [X,Y,Z]
Torch training mask:      [B,1,X,Y,Z]
```

### Grayscale

```text
Source: uint8 [0,255]
Model:  float32 [0,1]
```

Normalization parameters는 training core만으로 추정하고 모든 split과 모델에 동일하게 사용합니다.

### Binary

```text
Internal convention: pore=1, solid=0
```

Loader는 source polarity를 확인해 내부 convention으로 한 번만 변환합니다.

### Crop

- Shape: `[256,256,256]`
- Origin: `[x0,y0,z0]`
- Grayscale과 binary에 같은 origin 적용
- Validation/test origin은 manifest에 기록

## 5. Sparse Sampling

Z축 sampling은 factor 4를 사용합니다.

```text
observed index = offset + 4*k
```

Offset 0, depth 256의 예:

```text
observed: 0, 4, 8, ..., 252       # 64 slices
missing:  all other indices        # 192 slices
primary:  1..251 중 bracketed 값  # 189 slices
boundary: 253, 254, 255            # diagnostic only
```

Primary image metric은 양쪽에 observed slice가 존재하는 `primary_metric_mask`에서 계산합니다. E0는 full-volume export를 위해 마지막 observed slice를 boundary에 복사할 수 있지만, 그 값은 primary image metric에서 제외합니다. Learned model과 E0는 동일한 primary mask로 비교합니다.

## 6. SAINT Model

### AMI

Sparse volume을 sagittal과 coronal marginal slice로 나누어 shared AMI에 입력합니다.

```text
Sagittal: fixed X -> [Y,Z]
Coronal:  fixed Y -> [X,Z]
```

AMI는 다음 요소를 포함합니다.

- Residual-dense feature learner
- Physical spacing을 이용한 Filter Distance Matrix
- Dynamic filter generation
- Z축 anisotropic sub-pixel reconstruction

3 input channels은 RGB가 아니라 인접한 marginal slice 세 장입니다.

### RFN

Sagittal과 coronal 방향에서 생성한 dense volume을 axial grid에 정렬하고 RFN으로 fusion합니다.

```text
I_avg   = (I_sag + I_cor) / 2
I_final = I_avg + RFN_residual(I_sag, I_cor)
```

최종 출력은 dense grayscale volume입니다.

### Upstream reference

- Paper: SAINT, CVPR 2020
- Code: <https://github.com/cpeng93/SAINT>
- Reference commit: `a4a1b8f9c7b74a5c5d00b8b08132ff15a1e65802`
- Documented environment: Python 3.7, PyTorch 1.0

Upstream은 구조, tensor flow와 학습 동작을 확인하기 위한 reference로 사용합니다. 이 저장소에는 현대 PyTorch 기반 독립 구현을 작성하고, 차이가 있는 부분은 PR에서 명시합니다. Upstream source를 그대로 복사하지 않습니다.

논문 식 (5)는 sub-pixel channel 0을 LR 입력으로 정의하지만 공개 구현은 channel 0을 포함한 모든 channel을 dynamic filter로 예측합니다. 이 저장소의 AMI는 공개 구현의 학습 동작을 따르고, 구조·물성 평가용 full volume을 만들 때 observed slice를 입력값으로 다시 삽입하여 data consistency를 보장합니다. 공개 구현의 `512` 고정 반복은 제거하고 실제 입력 shape를 사용합니다.

구현 전 확인 항목:

- Stage 1/2 tensor shape
- x4 output alignment
- FDM coordinate와 spacing semantics
- `n_colors=3`의 의미
- AMI/RFN stage별 L1 적용 범위
- Checkpoint와 implementation의 알려진 차이

## 7. Experiments

| ID | Method | Objective | Seeds |
|---|---|---|---:|
| E0 | Linear z interpolation | None | deterministic |
| E1 | SAINT baseline | L1 | 1, 2, 3 |
| E2 | Gradient ablation | L1 + `0.10 * gradient` | 1, 2, 3 |
| E3 | Topology-aware fine-tuning | L1 + `0.10 * gradient` + `0.10 * topology` | 1, 2, 3 |

### E1

AMI와 RFN을 stage별로 학습합니다. Stage별 L1 region은 upstream audit 후 config에 고정합니다. Primary evaluation mask와 training-loss mask는 별개입니다.

### E2

E1과 architecture, data split, optimizer, training budget와 checkpoint rule을 동일하게 유지하고 gradient term만 추가합니다.

```text
L_grad = |Dx(pred)-Dx(target)| + |Dy(pred)-Dy(target)|
L_E2   = L1 + 0.10 * L_grad
```

Gradient는 역전파 gradient가 아니라 인접 pixel의 grayscale intensity 차이입니다.

### E3

각 seed의 E2 best checkpoint에서 시작하고 AMI를 고정한 상태에서 RFN을 fine-tuning합니다.

Observed slice와 predicted missing slice를 조립한 3D grayscale volume을 soft pore probability로 변환합니다. 낮은 intensity가 pore인 경우:

```text
P_pred = sigmoid((threshold - I_final) / temperature)
```

```text
L_topology = 0.5 * L_softDice + 0.5 * L_soft-clDice
L_E3       = L1 + 0.10 * L_grad + 0.10 * L_topology
```

- Threshold, temperature와 skeletonization iterations는 training/validation에서 정해 config에 고정합니다.
- Observed slice는 상수로 조립하여 해당 위치를 통해 gradient가 흐르지 않게 합니다.
- Soft-clDice는 connectivity surrogate이며 완전한 topology 보장을 의미하지 않습니다.
- Permeability와 hard connected-component 연산은 training graph에 포함하지 않습니다.

## 8. Training and Selection

### Stage order

```text
Data audit
-> CPU smoke test
-> one-batch GPU overfit
-> E0
-> E1 AMI/RFN
-> E2
-> E3
-> frozen inference
-> image/topology/physics evaluation
```

### Checkpoints

- E1/E2: validation primary-mask MAE
- E3: validation total E3 loss
- Test metrics는 checkpoint 선택에 사용하지 않음

### Reproducibility

- Debug seed: 0
- Headline seeds: 1, 2, 3
- Split, validation/test crop과 offset은 seed와 무관하게 고정
- Experiment config와 Git commit 기록

## 9. Evaluation

### Grayscale

Primary mask에서 다음을 계산합니다.

- MAE
- PSNR with data range 1.0
- SSIM with a frozen implementation/window

Supporting metrics:

- Gradient MAE
- Error by relative interpolation position
- Boundary diagnostic
- Inference time와 peak memory

### Full-volume construction

1. Prediction을 HR Z-grid에 정렬합니다.
2. Observed slice를 입력 grayscale로 교체합니다.
3. 등록된 ROI를 적용합니다.
4. 동일한 frozen segmentation operator를 적용합니다.

### Structure and topology

- Dice and IoU
- Absolute porosity error
- Specific surface area error
- Pore-size distribution distance
- Euler characteristic density error
- Connected-component statistics
- Largest connected pore ratio
- z-percolation and failure rate

### Physics

- z-direction tortuosity
- z-direction single-phase permeability
- Permeability log error

Prediction이 non-percolating이면 임의의 작은 permeability를 대입하지 않고 percolation failure로 기록합니다. 모든 모델과 ground truth에 동일한 solver, boundary condition, convergence criterion과 physical spacing을 적용합니다.

### Statistical unit

- Held-out parent core 또는 사전 등록된 non-overlapping block
- Slice와 voxel을 독립 표본으로 간주하지 않음
- 동일 core, crop와 offset의 paired comparison
- Seed와 core/block 변동을 분리해 보고

## 10. Run Metadata

각 prediction과 metric 결과에 다음 정보를 연결합니다.

```text
experiment_id
git_commit
config_name
dataset_doi
source_core_id
split_manifest_version
crop_origin_xyz
crop_shape_xyz
raw_storage_order
normalization_id
sampling_factor
sampling_offset
hr_voxel_spacing_um_xyz
seed
loss_name
loss_weights
checkpoint_epoch
segmentation_protocol_version
topology_protocol_version
soft_threshold
soft_temperature
soft_skeleton_iterations
```

## 11. Values to Freeze Before Full Training

- Exact train/validation/test core IDs
- Normalization method and parameters
- Validation/test crop coordinates and sampling offsets
- AMI/RFN stage loss region
- Soft threshold, temperature and skeletonization iterations
- SSIM implementation/window
- Segmentation operator
- Permeability solver and boundary conditions
- Cluster partition, resources and wall time

## 12. Acceptance Checklist

### Data

- [ ] 11 core checksum, shape, dtype와 endianness 확인
- [ ] Grayscale/binary alignment와 pore polarity 확인
- [ ] Parent-core split manifest 동결
- [ ] Normalization config 동결
- [ ] Aligned `256^3` crop test
- [ ] x4 sampling과 primary mask test

### Model

- [ ] AMI/RFN output shape test
- [ ] FDM physical spacing test
- [ ] One-batch overfit
- [ ] Observed-slice data consistency test
- [ ] E1/E2 seeds 1, 2, 3

### Topology

- [ ] Constant, disconnected와 connected phantom test
- [ ] Soft Dice/soft-clDice backward test
- [ ] Observed slice gradient-flow test
- [ ] E3 seeds 1, 2, 3

### Evaluation

- [ ] Primary-mask MAE/PSNR/SSIM
- [ ] Frozen segmentation
- [ ] Morphology/connectivity metrics
- [ ] Permeability solver consistency
- [ ] Core/block-level result table

## 13. Claim Boundary

권장 표현:

> We evaluate SAINT-based x4 sparse-z reconstruction on held-out grayscale sandstone Micro-CT blocks and compare L1, gradient-regularized and 3D connectivity-aware objectives using image, topology and physical-property metrics.

다음 주장은 하지 않습니다.

- 실제 미관측 pore structure를 유일하게 복원했다.
- PSNR 향상이 permeability 정확도를 보장한다.
- 비교하지 않은 모든 방법보다 우수하다.
- Synthetic sparse sampling이 실제 저해상도 촬영을 완전히 재현한다.
