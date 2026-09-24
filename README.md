# Grayscale Sparse-slice Digital Rock Reconstruction with SAINT

SAINT(Spatially Aware Interpolation NeTwork)을 grayscale Micro-CT digital rock에 이전해, z축에서 누락된 단면을 복원하는 연구 프로젝트입니다.

현재 목표는 새로운 모델을 바로 설계하는 것이 아니라 다음 순서로 신뢰할 수 있는 기준선을 확보하는 것입니다.

1. 공개 SAINT 코드를 현재 환경에서 재현합니다.
2. 11 Sandstones의 grayscale volume으로 원본 SAINT + L1 기준선을 학습합니다.
3. Gradient loss가 pore-grain 경계와 z축 연속성에 주는 효과를 분리해 측정합니다.
4. 복원된 grayscale volume을 고정된 segmentation protocol로 평가해 구조와 물성을 확인합니다.
5. 3D topology loss는 위 단계가 완료되고 연결성 실패가 확인된 경우에만 후속 실험으로 추가합니다.

상세한 정의와 실험 규칙은 [연구 프로토콜](digital_rock_sparse_slice_reconstruction_final_protocol.md)을 따릅니다.

## 현재 연구 질문

> 원래의 SAINT 구조와 L1 학습을 grayscale digital rock에 적용했을 때, 선형 보간보다 누락 단면의 영상 품질과 3D pore 구조를 더 잘 보존하는가?

후속 질문은 다음과 같습니다.

> L1에 gradient loss를 추가하면 PSNR/SSIM을 크게 희생하지 않으면서 pore-grain 경계와 z축 연속성을 개선하는가?

Topology-aware loss는 현재 필수 범위가 아닙니다. SAINT baseline의 재현과 grayscale domain adaptation을 먼저 완료합니다.

## 데이터

### Primary dataset: 11 Sandstones

- 출처: [Digital Porous Media Portal - 11 Sandstones](https://digitalporousmedia.org/datasets/317/)
- DOI: [10.17612/F4H1-W124](https://doi.org/10.17612/F4H1-W124)
- 라이선스: ODC-By 1.0
- 구성: 서로 다른 sandstone core 11개
- 크기: core당 `1000 x 1000 x 1000` voxels
- voxel spacing: `2.25 um` isotropic
- 제공 형식: raw grayscale, filtered grayscale, binary segmentation

Primary 입력과 정답은 `*_grayscale_filtered.raw`입니다. Filtered grayscale로 파이프라인을 안정화한 뒤 raw grayscale을 robustness 실험으로 추가합니다. 제공된 binary volume은 모델 입력이나 학습 정답이 아니라, 후속 구조 평가의 reference로만 사용합니다.

원본 데이터와 파생 patch는 Git에 올리지 않습니다.

### Split 원칙

- Split 단위는 slice나 patch가 아니라 parent core입니다.
- 목표 비율은 train 8 core / validation 1 core / test 2 core입니다.
- 정확한 core 배정은 intensity, porosity 및 artifact audit 후 manifest로 동결합니다.
- Split manifest가 동결되기 전에는 headline experiment를 실행하지 않습니다.
- 동일 core에서 나온 patch가 둘 이상의 split에 포함되면 안 됩니다.

## 문제 설정

고해상도 grayscale volume을 다음과 같이 둡니다.

```text
I_HR: [H, W, D], float32, normalized to [0, 1]
```

저해상도 입력은 z축에서만 stride sampling하여 생성합니다.

```python
I_LR = I_HR[:, :, offset::r]
```

공간 해상도는 반드시 함께 기록합니다.

| Scale | HR spacing `(x,y,z)` | LR spacing `(x,y,z)` |
|---:|---|---|
| x2 | `(2.25, 2.25, 2.25) um` | `(2.25, 2.25, 4.50) um` |
| x4 | `(2.25, 2.25, 2.25) um` | `(2.25, 2.25, 9.00) um` |
| x6 | `(2.25, 2.25, 2.25) um` | `(2.25, 2.25, 13.50) um` |

Primary experiment는 공개 checkpoint와 원 논문의 주요 조건에 맞춰 `x4`로 시작합니다. `x2`와 `x6`은 x4 기준선이 고정된 뒤 수행합니다.

## 실험 단계

| 단계 | 실험 | Loss | 상태/우선순위 |
|---|---|---|---|
| R0 | Upstream SAINT inference 재현 | upstream 설정 | 최우선 |
| D0 | 데이터 로더·spacing·z sampling 검증 | 없음 | 최우선 |
| G0 | Linear z interpolation | 없음 | 필수 baseline |
| G1 | Original SAINT on grayscale rock | L1 | 핵심 실험 |
| G2 | SAINT + gradient loss | L1 + gradient | G1 이후 |
| G3 | x2/x6 및 raw grayscale robustness | G1/G2와 동일 | 후속 |
| T1 | SAINT + 3D topology-aware loss | 미정 | 조건부 후속 |

Independent 2D, direct 3D CNN, diffusion model은 현재 최소 실험 범위에 포함하지 않습니다. 필요하면 G1/G2가 완료된 뒤 비교군 확장으로 별도 결정합니다.

## Loss

### G1: 원본 기준선

```text
L_rec = mean(abs(I_pred - I_HR))
```

원본 SAINT와 동일하게 L1을 사용합니다. 모델이 직접 예측하지 않은 observed slice를 metric에 포함해 성능을 부풀리지 않습니다.

### G2: Gradient ablation

```text
L_total = L1 + lambda_grad * L_grad
```

Gradient loss는 역전파 gradient가 아니라 인접 voxel 사이의 intensity 변화량을 비교합니다.

```text
L_grad = |Dx(pred)-Dx(gt)| + |Dy(pred)-Dy(gt)|
```

첫 실험은 SAINT가 처리하는 2D sagittal/coronal output에 적용합니다. 최종 volume의 3D gradient와 z-gradient는 별도 ablation으로 취급합니다. 초기 `lambda_grad` 후보는 `{0.05, 0.1}`이며 validation set에서만 선택합니다.

## 평가

### Primary grayscale metrics

모든 지표는 원래 관측된 slice를 제외한 missing slice 위치에서 계산합니다.

1. MAE/L1
2. PSNR
3. SSIM
4. Gradient MAE
5. 관측 slice로부터의 상대 위치별 오차

Primary checkpoint selection metric은 validation missing-slice MAE입니다. Test set으로 epoch, loss weight, normalization 또는 threshold를 선택하지 않습니다.

### Secondary structure and physics metrics

G1 기준선이 완료된 뒤 동일한 고정 segmentation operator를 GT와 모든 prediction에 적용합니다.

- Dice/IoU against provided binary reference
- Absolute porosity error
- Specific surface area error
- Pore-size distribution distance
- Euler characteristic density error
- Largest connected component ratio
- z-percolating porosity and failure rate
- z-direction permeability log-error

Segmentation threshold나 segmenter는 training/validation 자료만으로 동결하고 모든 test model에 동일하게 적용합니다. 모델별로 test threshold를 최적화하지 않습니다.

## 전체 흐름

```text
11 Sandstones filtered grayscale HR volume
                 |
                 v
       core-level 8/1/2 split
                 |
                 v
        z-only sparse sampling
                 |
        +--------+--------+
        |                 |
        v                 v
  Linear interpolation  SAINT
                          |
                    +-----+------+
                    |            |
                    v            v
                   L1       L1 + gradient
                    |            |
                    +-----+------+
                          |
                          v
             missing-slice image metrics
                          |
                          v
             frozen segmentation operator
                          |
                          v
       morphology / connectivity / permeability
```

## SAINT upstream 사용 방침

- 공식 코드: [cpeng93/SAINT](https://github.com/cpeng93/SAINT)
- 공개 구현은 Python 3.7과 PyTorch 1.0 시대 코드입니다.
- pretrained checkpoint는 x4/x6을 지원합니다.
- AMI와 RFN은 두 단계로 학습합니다.
- RFN 학습 문서는 upstream README에서도 완전히 정리되지 않았습니다.
- upstream repository에 명시적인 라이선스 파일이 확인되지 않으므로, 코드를 복사하거나 수정본을 재배포하기 전에 사용 조건을 확인합니다.

우선 upstream commit을 고정해 별도 환경에서 실행을 재현한 후, 동작을 바꾸지 않는 범위에서 현재 PyTorch로 포팅합니다. 재현 버전과 포팅 버전의 같은 입력에 대한 출력 차이를 기록합니다.

## 팀 역할

### Team A - Data and experiment operations

- 11 Sandstones 다운로드·checksum·metadata 관리
- core audit 및 split manifest 동결
- normalization과 z sampling 구현·검증
- 환경, GPU job, seed, checkpoint 및 log 관리
- prediction volume과 metadata export

### Team B - SAINT reproduction and modeling

- upstream inference 재현
- AMI/RFN tensor shape 및 FDM spacing 검증
- grayscale dataloader와 L1 baseline 포팅
- gradient loss ablation 구현
- upstream/ported output parity test

### Team C - Evaluation and physical validation

- missing-slice MAE/PSNR/SSIM/gradient metric
- segmentation protocol 동결
- morphology, connectivity 및 permeability 평가
- core 단위 통계와 figure 생성

### Integration lead

- interface와 milestone 승인
- split/loss/metric 변경에 대한 decision log 확인
- 재현 가능한 결과만 `main`에 병합

| 역할 | 담당자 | 교차 검토자 |
|---|---|---|
| Team A | `TBD` | `TBD` |
| Team B | `TBD` | `TBD` |
| Team C | `TBD` | `TBD` |
| Integration lead | `TBD` | - |

## 재현성 규칙

각 실험은 최소한 다음 metadata를 저장해야 합니다.

```text
experiment_id
git_commit
upstream_saint_commit
dataset_doi
dataset_variant
split_manifest_version
split_manifest_hash
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
test_core_id
```

- 학습 모델은 최소 3 seeds로 반복합니다.
- 통계 단위는 slice가 아니라 held-out core 또는 사전 등록된 독립 block입니다.
- Test metric을 이용해 설정을 변경하지 않습니다.
- 원본 데이터, checkpoint 및 전체 prediction은 Git에 commit하지 않습니다.

## Branch와 변경 관리

```text
main                 재현 가능한 동결 milestone
develop              통합 개발
infra/<topic>        데이터·환경·실험 운영
model/<topic>        SAINT 재현·모델
eval/<topic>         평가·물성
docs/<topic>         문서·decision record
```

Split, normalization, primary metric, loss, voxel spacing 또는 test core를 변경하면 `docs/decisions/`에 이유와 영향을 기록합니다. 자세한 절차는 [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.

## 현재 완료 조건

### Milestone 0 - Upstream reproduction

- [ ] upstream commit 고정
- [ ] legacy 환경 또는 호환 환경 구성
- [ ] 제공 checkpoint로 x4 inference 실행
- [ ] 입력·출력 tensor와 normalization 기록

### Milestone 1 - Data baseline

- [ ] 11 core checksum 및 shape 확인
- [ ] filtered grayscale intensity/artifact audit
- [ ] 8/1/2 split manifest 동결
- [ ] x4 z sampling 및 spacing unit test
- [ ] 한 core에서 linear interpolation 결과 생성

### Milestone 2 - Grayscale SAINT

- [ ] G1 x4 L1 학습
- [ ] AMI와 RFN best validation checkpoint 확보
- [ ] held-out core prediction 생성
- [ ] missing-slice MAE/PSNR/SSIM 보고

### Milestone 3 - Gradient ablation

- [ ] G2 2D gradient loss 구현
- [ ] lambda 후보를 validation에서만 선택
- [ ] G1/G2를 동일 split·seed로 비교
- [ ] 경계 개선과 노이즈 증폭 여부 분석

### Milestone 4 - Structure and physics

- [ ] segmentation operator 동결
- [ ] morphology/connectivity 평가
- [ ] 대표 test block에서 permeability 계산
- [ ] topology loss 필요성에 대한 go/no-go 결정

## 주장 범위

이 프로젝트는 다음을 주장할 수 있습니다.

> Held-out grayscale Micro-CT core에서 SAINT의 sparse-slice reconstruction 성능과 gradient regularization의 효과를 영상·구조·물성 지표로 평가했다.

다음 표현은 사용하지 않습니다.

- true pore structure를 완벽히 복원했다
- PSNR 향상이 permeability 정확도를 보장한다
- 최신 방법을 모두 재현하지 않고 SOTA를 달성했다
- synthetic z subsampling 결과가 실제 저해상도 촬영을 완전히 대변한다
