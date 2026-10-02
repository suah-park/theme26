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
- Berea 한 core의 비중첩 공간 train/validation/test split
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

주 실험은 Berea만 사용한다. 한 core의 train 구역에서 학습하여 같은 core의 별도 구역을
x4 sparse-z 입력으로 복원한다. 새로운 core 일반화는 이번 주 실험의 범위가 아니다.

| Split | X (끝 제외) | Y/Z | Crop |
|---|---|---|---|
| Train | [0,400) | [0,1000) | train 구역 내 random 256³ |
| Validation | [432,688) | [0,1000) | 고정 9개 |
| Test | [720,976) | [0,1000) | 고정 9개 |

구역 사이 32-voxel buffer는 사용하지 않는다. 이 값은 초기 설계이며 공간적 독립성의 증거가 아니다.
평가 Y/Z origins는 {0,372,744} × {0,372,744}이고 각 split의 X 시작에 배치한다.
먼저 구역을 분리한 뒤 그 안에 완전히 포함되는 crop만 생성한다.
Validation/test의 dense grayscale은 선택/평가 정답으로만 사용하고 모델 입력은 sparse다.
Split·평가 좌표·offset은 성능 결과 확인 전에 고정하며 seed마다 변경하지 않는다.

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

Normalization parameters는 train 구역만으로 추정하고 모든 split과 모델에 동일하게 사용합니다.

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
- Documented environment: Python 3.7, PyTorch 1.0

Upstream은 구조, tensor flow와 학습 동작을 확인하기 위한 reference로 사용합니다. 이 저장소에는 현대 PyTorch 기반 독립 구현을 작성하고, 차이가 있는 부분은 PR에서 명시합니다. Upstream source를 그대로 복사하지 않습니다.

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

AMI와 RFN을 stage별로 학습합니다. E1은 두 stage 모두 전체 출력 L1을 사용합니다. Primary evaluation mask와 training-loss mask는 별개입니다. 최종·validation metric은 [0,1] clipping 후 primary mask에서 계산합니다.

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

- 사전 등록된 동일 Berea core의 non-overlapping held-out block
- Slice와 voxel을 독립 표본으로 간주하지 않음
- 동일 core, crop와 offset의 paired comparison
- Seed와 block 변동을 분리해 보고; 동일 core block을 독립 암석 표본으로 간주하지 않음

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

- Exact Berea train/validation/test regions and buffer
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

- [ ] Berea source checksum, shape, dtype와 endianness 확인
- [ ] Grayscale/binary alignment와 pore polarity 확인
- [ ] Berea spatial-region split manifest 동결
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

학습용 dense train 구역 확보 비용을 포함해 스캔 부담을 논의한다. 본 실험은 synthetic x4
subsampling을 사용하며 실제 장비 시간/비용 75% 감소를 측정했다고 주장하지 않는다.
Same-core held-out region 성능만 평가하고 unseen-core 일반화를 주장하지 않는다.


권장 표현:

> We evaluate SAINT-based x4 sparse-z reconstruction on held-out grayscale sandstone Micro-CT blocks and compare L1, gradient-regularized and 3D connectivity-aware objectives using image, topology and physical-property metrics.

다음 주장은 하지 않습니다.

- 실제 미관측 pore structure를 유일하게 복원했다.
- PSNR 향상이 permeability 정확도를 보장한다.
- 비교하지 않은 모든 방법보다 우수하다.
- Synthetic sparse sampling이 실제 저해상도 촬영을 완전히 재현한다.

### Integration status

Train crop은 seed 2026으로 만든 고정 48개를 `configs/data/berea_crops.yaml`에 기록한다. 평가 crop은 각 9개다.
RAW는 모든 경로에서 C-order XYZ로 직접 해석한다. [976,1000) 구역은 사용하지 않는다.
E1은 clipped validation primary-mask MAE로 checkpoint를 선택한다. AMI raw float32 cache를 RFN에 연결하며 최종 출력에는 관측 단면을 재삽입한다. 자세한 실행·재개는 [E1 실행 안내](e1_execution.md)를 따른다.
Gradient mask의 양쪽 voxel 포함 정책은 그대로 유지했으며 경계 정책은 별도 결정한다.
