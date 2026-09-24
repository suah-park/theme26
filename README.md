# Grayscale Sparse-slice Digital Rock Reconstruction with SAINT

SAINT(Spatially Aware Interpolation NeTwork)을 grayscale Micro-CT digital rock에 이전해 z축에서 누락된 단면을 복원하는 연구 프로젝트입니다.

핵심 질문은 다음과 같습니다.

> Grayscale SAINT가 x4 sparse-slice sandstone volume을 단순 보간보다 잘 복원하며, gradient loss와 3D connectivity-aware topology fine-tuning이 pore boundary, topology 및 permeability 보존을 개선하는가?

현재 연구는 원본 SAINT 구조를 유지한 채 L1 기준선, gradient-loss ablation, 그리고 3D soft-clDice topology fine-tuning을 단계적으로 비교합니다. Permeability는 loss가 아니라 동결된 prediction의 최종 평가 지표로 사용합니다.

상세한 정의와 실험 규칙은 [연구 프로토콜](digital_rock_sparse_slice_reconstruction_final_protocol.md)을 따릅니다. 팀별 모듈의 입력·출력 shape, dtype, axis와 공개 함수 signature는 [Interface Contract](docs/interface_contract.md)에 고정합니다.

## 고정된 연구 범위

| 항목 | 설정 |
|---|---|
| Dataset | 11 Sandstones filtered grayscale |
| Source volume | `1000 x 1000 x 1000` |
| Spatial crop | `256 x 256 x 256` |
| Sparse axis | z |
| Scale | x4 only |
| SAINT patch size | `10` |
| Normalization | training data로 동결한 `[0,1]` mapping |
| Learned-model seeds | `1, 2, 3` |
| E1/E2 checkpoint metric | validation missing-slice MAE |
| E3 checkpoint metric | validation E3 total loss |

`256`은 3D 암석 ROI의 크기이고 `10`은 SAINT가 초해상화 축에서 사용하는 내부 2D strip 길이입니다. 두 설정은 대체 관계가 아닙니다.

현재 범위에서 다음 항목은 수행하지 않습니다.

- x2 또는 x6 scale
- raw-grayscale robustness
- Persistent-homology loss 또는 permeability loss
- GAN, diffusion 또는 architecture search
- loss-weight sweep
- test set을 이용한 threshold 또는 hyperparameter 선택

## 데이터

### Primary dataset: 11 Sandstones

- 출처: [Digital Porous Media Portal - 11 Sandstones](https://digitalporousmedia.org/datasets/317/)
- DOI: [10.17612/F4H1-W124](https://doi.org/10.17612/F4H1-W124)
- 라이선스: ODC-By 1.0
- 구성: 서로 다른 sandstone core 11개
- 크기: core당 `1000 x 1000 x 1000` voxels
- voxel spacing: `2.25 um` isotropic
- 제공 형식: raw grayscale, filtered grayscale, binary segmentation

Primary 입력과 grayscale 정답은 `*_grayscale_filtered.raw`입니다. 제공된 binary volume은 모델 입력이나 grayscale reconstruction 정답이 아닙니다. E3에서는 training/validation core의 정렬된 binary volume만 auxiliary topology target으로 사용하고, 최종 평가에서는 모든 모델에 동일한 구조·물성 reference로 사용합니다.

### Split과 crop

- Split 단위는 slice나 crop이 아니라 parent core입니다.
- 목표 split은 train 8 core / validation 1 core / test 2 core입니다.
- Split은 crop 생성보다 먼저 수행합니다.
- Training은 split된 source volume에서 `256^3` crop을 온라인으로 추출합니다.
- Validation/test는 versioned manifest에 기록된 고정 `256^3` 좌표를 사용합니다.
- 동일 core의 crop이 둘 이상의 split에 포함되면 안 됩니다.

## 문제 설정

고해상도 grayscale volume을 다음과 같이 둡니다.

```text
I_HR: [H, W, D], float32, normalized to [0, 1]
```

저해상도 입력은 z축에서만 stride sampling하여 생성합니다.

```python
I_LR = I_HR[:, :, offset::4]
```

원래 voxel spacing이 `(2.25, 2.25, 2.25) um`이면 x4 입력 spacing은 `(2.25, 2.25, 9.00) um`입니다. Axis, offset과 spacing은 모든 prediction metadata에 기록합니다.

## 핵심 실험

| ID | 방법 | Loss | Seeds |
|---|---|---|---:|
| E0 | Linear z interpolation | 없음 | deterministic |
| E1 | Original SAINT on grayscale rock | L1 | 1, 2, 3 |
| E2 | SAINT gradient ablation | L1 + `0.1 * gradient` | 1, 2, 3 |
| E3 | SAINT topology-aware fine-tuning | L1 + `0.1 * gradient` + `0.1 * topology` | 1, 2, 3 |

E1과 E2는 architecture, data split, crop, optimizer, training budget 및 checkpoint rule을 동일하게 유지합니다. E2에서는 loss만 변경합니다.

E3는 각 seed의 E2 best checkpoint에서 시작합니다. 먼저 AMI를 고정하고 RFN을 fine-tuning하며, sagittal/coronal 결과를 조립한 최종 `256^3` grayscale volume에 3D topology loss를 적용합니다. soft-clDice는 pore network의 연결성을 유도하는 surrogate이며 정확한 위상 보존을 보장하는 loss로 해석하지 않습니다.

SAINT의 두 단계 구조를 유지합니다.

```text
Stage 1: sagittal/coronal marginal SR
  -> Stage 1 freeze
  -> directional dense volumes
  -> Stage 2: RFN fusion
  -> frozen test inference
```

## Loss

E1은 원본 SAINT 기준선과 같은 L1을 사용합니다.

```text
L_E1 = mean(abs(pred - target))
```

E2의 gradient loss는 역전파 gradient가 아니라 인접 pixel 사이의 intensity 변화량을 비교합니다.

```text
L_grad = |Dx(pred)-Dx(target)| + |Dy(pred)-Dy(target)|
L_E2   = L1 + 0.1 * L_grad
```

Intensity는 `[0,1]`로 정규화하고 `lambda_grad=0.1`을 고정합니다. 현재 실험에서는 loss-weight sweep을 수행하지 않습니다.

E3에서는 조립된 3D grayscale volume을 미분 가능한 soft pore probability로 바꿉니다. 아래 식은 낮은 intensity가 pore인 경우이며, 데이터 audit에서 polarity를 반드시 확인합니다.

```text
P_pred = sigmoid((t - I_final) / tau)
L_topo = 0.5 * L_softDice(P_pred, B_gt) + 0.5 * L_soft-clDice(P_pred, B_gt)
L_E3   = L1 + 0.1 * L_grad + 0.1 * L_topo
```

`t`와 `tau`는 training/validation 자료에서 정하고 headline E3 전에 동결합니다. `I_final`은 관측 slice는 입력값, 누락 slice는 prediction으로 구성합니다. 따라서 topology loss는 전체 3D volume을 보지만 gradient는 누락 slice prediction으로만 흐릅니다. L1은 누락 grayscale voxel, gradient loss는 누락 위치와 맞닿은 2D intensity 차이, topology loss는 3D soft pore volume에 적용합니다.

## 평가

모든 영상 지표는 원래 관측된 slice를 제외한 missing-slice 위치에서 계산합니다.

### Grayscale

- MAE/L1
- PSNR
- SSIM
- Gradient MAE
- 관측 slice로부터의 상대 위치별 오차

### Structure and physics

E0/E1/E2/E3의 동결된 prediction에 동일한 hard segmentation operator를 적용합니다. E3 학습의 soft conversion과 최종 평가 segmentation은 역할을 분리하며, test 결과를 보고 어느 쪽도 조정하지 않습니다.

- Dice/IoU against provided binary reference
- Absolute porosity error
- Specific surface area error
- Pore-size distribution distance
- Euler characteristic density error
- Largest connected component ratio
- z-percolating porosity and failure rate
- z-direction permeability error

Segmentation threshold와 solver 설정은 training/validation 자료로 동결합니다. Test topology 또는 permeability로 checkpoint, loss weight나 threshold를 선택하지 않습니다.

## 데이터와 실행 결과 저장

Code, data와 run artifact를 분리합니다.

```text
/project/<group>/theme26/                 # Git clone
/project/<group>/data/11_sandstones/      # raw, processed, splits
/project/<group>/runs/theme26/            # checkpoints, logs, predictions
```

원본 volume, 생성된 배열, checkpoint, 전체 prediction과 solver output은 Git에 올리지 않습니다. 실제 경로는 config 또는 환경변수로 전달하고 source code에 개인 경로를 하드코딩하지 않습니다.

GPU 학습은 Slurm에서 GPU 1장을 요청해 실행합니다. 서버마다 다른 partition 이름, GPU resource 표기와 공용 저장소 경로만 바꿉니다. 자세한 구조와 명령은 [Data storage and GPU Slurm execution](docs/operations/storage-and-slurm.md)을 따릅니다.

## SAINT upstream

- 공식 코드: [cpeng93/SAINT](https://github.com/cpeng93/SAINT)
- 공개 구현 환경: Python 3.7, PyTorch 1.0
- upstream training: marginal SR Stage 1과 RFN Stage 2
- 공식 training command의 `patch_size=10`은 3D cube 크기가 아니라 sparse axis의 2D strip 길이

Upstream commit을 고정해 별도 환경에서 먼저 재현한 뒤 현재 PyTorch로 포팅합니다. Upstream repository에 명시적인 라이선스 파일이 확인되지 않으므로, 전체 소스를 이 repository에 복사하거나 수정본을 공개하기 전에 사용 조건을 확인합니다.

E3 topology surrogate는 [clDice 논문](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)과 [공식 구현](https://github.com/jocpae/clDice)을 기준으로 3D soft-skeletonization을 구현합니다.

## 재현성 규칙

각 실험은 최소한 다음 metadata를 저장해야 합니다.

```text
experiment_id
git_commit
upstream_saint_commit
dataset_doi
dataset_variant
source_core_id
split_manifest_version
split_manifest_hash
crop_origin_xyz
crop_size_xyz
normalization_id
sampling_axis
sampling_factor
sampling_offset
hr_voxel_spacing_um
lr_voxel_spacing_um
loss_name
loss_weights
seed
checkpoint_epoch
validation_missing_mae
segmentation_protocol_version
topology_protocol_version
soft_threshold
soft_temperature
soft_skeleton_iterations
```

- 통계 단위는 slice가 아니라 held-out core 또는 사전 등록된 `256^3` block입니다.
- E1/E2/E3는 seeds 1, 2, 3으로 반복합니다.
- Test metric으로 설정을 변경하지 않습니다.
- Split, normalization, crop, loss, spacing, segmentation 또는 solver를 변경하면 decision record를 추가합니다.

## 완료 조건

### Data and reproduction

- [ ] 11 core의 checksum, shape, dtype, endianness 확인
- [ ] Parent-core 8/1/2 split manifest 동결
- [ ] `256^3` crop과 x4 z sampling unit test
- [ ] Upstream SAINT commit과 실행 환경 고정
- [ ] 제공 checkpoint의 x4 inference 확인

### Experiments

- [ ] E0 linear interpolation
- [ ] E1 SAINT + L1, seeds 1/2/3
- [ ] E2 SAINT + L1 + 0.1 gradient, seeds 1/2/3
- [ ] E3 E2-checkpoint + 0.1 topology fine-tuning, seeds 1/2/3
- [ ] 동일한 missing-slice mask로 grayscale metric 계산

### Structure and physics

- [ ] Segmentation operator 동결
- [ ] Morphology와 connectivity 평가
- [ ] `256^3` test block의 permeability 평가
- [ ] 영상·topology·물성 결과를 분리해 보고

## 주장 범위

권장 표현:

> Held-out grayscale sandstone Micro-CT blocks에서 x4 sparse-z SAINT reconstruction과 gradient regularization의 효과를 영상·구조·물성 지표로 평가했다.

E3 결과까지 포함할 때는 다음처럼 표현합니다.

> Held-out grayscale sandstone Micro-CT blocks에서 x4 sparse-z SAINT reconstruction을 수행하고, gradient regularization과 3D connectivity-aware topology fine-tuning의 효과를 영상·구조·물성 지표로 평가했다.

다음 표현은 사용하지 않습니다.

- true pore structure를 완벽히 복원했다
- PSNR 향상이 permeability 정확도를 보장한다
- 최신 방법을 모두 비교하지 않고 SOTA를 달성했다
- synthetic z subsampling이 실제 저해상도 촬영을 완전히 대변한다
