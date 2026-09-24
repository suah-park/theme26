# SAINT 기반 Grayscale Digital Rock Sparse-slice Reconstruction 연구 프로토콜

- 개정일: 2026-09-24
- 상태: Phase 0 - upstream 재현 및 데이터 audit
- Primary dataset: 11 Sandstones filtered grayscale Micro-CT
- Primary scale: x4 along z
- Primary model: original SAINT architecture, L1 loss
- First ablation: 2D gradient loss
- 3D topology-aware loss: baseline 완료 후 조건부 확장

---

## 0. 이번 개정에서 확정한 결정

이전 프로토콜의 binary pore-probability reconstruction을 중단하고, SAINT 원 논문과 동일한 grayscale reconstruction으로 문제를 다시 정의한다.

1. 모델 입력과 정답은 grayscale Micro-CT이다.
2. Primary dataset은 11 Sandstones의 filtered grayscale이다.
3. 데이터 split은 parent-core 단위 8/1/2를 사용한다.
4. x4 z-subsampling과 원본 SAINT + L1을 첫 번째 핵심 실험으로 한다.
5. x2, x6, raw grayscale은 x4 기준선 이후에 수행한다.
6. Gradient loss는 L1 기준선 다음의 첫 ablation이다.
7. Binary segmentation은 학습 target이 아니라 구조·물성 평가 reference이다.
8. 3D topology-aware loss는 Phase 1 필수 항목에서 제외한다.
9. Topology loss는 G1/G2에서 실제 연결성 실패가 확인된 경우에만 도입한다.

핵심 연구 질문은 다음과 같다.

> SAINT의 공간 인지형 cross-view interpolation을 grayscale digital rock에 그대로 이전했을 때, 단순 z축 보간보다 누락 단면의 영상 품질과 3D pore 구조를 더 잘 보존하는가?

첫 번째 확장 질문은 다음과 같다.

> L1에 gradient loss를 추가하면 pore-grain 경계와 slice-to-slice 연속성을 개선하는가?

---

## 1. 연구 범위

### 1.1 현재 필수 범위

- 공식 SAINT 코드와 checkpoint의 inference 재현
- 11 Sandstones filtered grayscale 데이터 audit
- core-level split manifest 작성
- z축 x4 sparse sampling 검증
- Linear interpolation baseline
- SAINT AMI/RFN의 grayscale L1 학습
- Missing-slice MAE, PSNR, SSIM 평가
- Gradient-loss ablation
- 동결된 segmentation protocol을 이용한 구조 평가

### 1.2 기준선 완료 후 범위

- x2 및 x6 scale
- raw grayscale robustness
- morphology, connectivity 및 permeability 평가
- Independent 2D 또는 3D CNN 비교군
- 3D topology-aware fine-tuning

### 1.3 현재 제외

- Binary volume을 입력으로 하는 BCE/Dice 학습
- Diffusion model 개발
- 불규칙한 z 간격
- 실제 LR/HR scanner pair 학습
- Test set에 맞춘 threshold 또는 post-processing

---

## 2. 데이터

### 2.1 Primary dataset

**11 Sandstones: Raw, Filtered and Segmented Data**

- Portal: <https://digitalporousmedia.org/datasets/317/>
- DOI: <https://doi.org/10.17612/F4H1-W124>
- License: ODC-By 1.0
- Samples: 11 sandstone cores
- Shape: `1000 x 1000 x 1000` per core
- Voxel spacing: `2.25 um` isotropic
- Variants: raw grayscale, filtered grayscale, binary segmentation

Primary experiment에서는 filtered grayscale 파일을 사용한다.

```text
<sample>_2d25um_grayscale_filtered.raw
```

Binary volume은 grayscale prediction을 평가하기 위한 reference이다. 학습 입력, 모델 출력 또는 loss target으로 사용하지 않는다.

### 2.2 데이터 provenance

각 core에 대해 다음 metadata를 보존한다.

```text
dataset_doi
download_url
download_date
source_filename
checksum_sha256
shape
dtype
endianness
voxel_spacing_um
intensity_min
intensity_max
intensity_percentiles
artifact_notes
```

Shape, dtype 및 endianness를 추정에 의존하지 않는다. Portal metadata와 실제 file size를 함께 확인한다.

### 2.3 데이터 audit

Split 전에 다음을 core별로 확인한다.

- 전체 shape 및 slice 수
- intensity histogram과 percentile
- ring/beam-hardening artifact
- 빈 배경 또는 cylindrical boundary 비율
- binary reference와 grayscale 방향·shape 일치
- x/y/z axis convention
- porosity와 spatial heterogeneity

### 2.4 Split

목표 split은 다음과 같다.

```text
Train:      8 parent cores
Validation: 1 parent core
Test:       2 parent cores
```

정확한 core 이름은 audit 이후 manifest에 기록한다. 다음 규칙을 따른다.

- Split은 patch 생성 전에 수행한다.
- 한 core의 patch는 하나의 split에만 속한다.
- Validation과 test core는 intensity/porosity 범위를 고려해 선택한다.
- Test core 이름은 첫 headline training 전에 동결한다.
- Split 변경 시 새 version과 decision record가 필요하다.

Patch 또는 slice 단위 random split은 금지한다.

---

## 3. 문제 정의

### 3.1 High-resolution volume

정규화된 dense grayscale volume을 다음처럼 정의한다.

\[
I_{HR}\in[0,1]^{H\times W\times D}
\]

원본 voxel spacing은 다음과 같다.

\[
s_{HR}=(2.25,2.25,2.25)\ \mu m
\]

### 3.2 Sparse volume

z축에서 offset \(o\)와 scale \(r\)을 사용해 sampling한다.

\[
I_{LR}^{(r,o)}(x,y,k)=I_{HR}(x,y,o+rk)
\]

LR spacing은 다음과 같다.

\[
s_{LR}=(2.25,2.25,2.25r)\ \mu m
\]

| r | LR z spacing |
|---:|---:|
| 2 | 4.50 um |
| 4 | 9.00 um |
| 6 | 13.50 um |

Primary scale은 `r=4`이다. Offset은 training에서 허용하되 기록해야 하고, validation/test에서는 manifest에 고정한다.

### 3.3 Output

모델 출력은 dense grayscale volume이다.

\[
\hat I\in[0,1]^{H\times W\times D}
\]

전체-volume export에서는 observed slice를 원본 입력으로 다시 넣을 수 있다. 그러나 모델 비교용 missing-slice metric은 모델이 생성한 누락 위치에서만 계산한다.

---

## 4. 전처리

### 4.1 Orientation

공통 array convention은 다음으로 동결한다.

```text
volume[x, y, z]
```

파일 loader가 다른 순서를 반환하면 명시적으로 transpose하고 metadata에 기록한다. Sagittal/coronal/axial 이름보다 axis index를 함께 표기한다.

### 4.2 Normalization

Normalization은 training core만으로 결정하고 version을 부여한다. 초기 후보는 다음과 같다.

1. Training-set global percentile clipping 후 `[0,1]` scaling
2. Scanner intensity가 core 간 일관될 경우 training-set global min/max scaling

Test core별 최적 percentile이나 contrast 조정은 허용하지 않는다. 선택된 방법은 모든 model과 split에 동일하게 적용한다.

### 4.3 Spatial crop

- Rock 외부 배경이 loss를 지배하지 않도록 valid rock ROI를 정의한다.
- ROI mask 생성 규칙은 model output을 보지 않고 동결한다.
- 동일한 test ROI를 모든 model에 사용한다.
- Headline metric은 서로 같은 physical extent에서 계산한다.

### 4.4 Augmentation

허용 후보:

- x/y flip
- x-y plane의 90-degree rotation
- training crop 위치 변화

z축과 x/y축 교환은 rock anisotropy와 reconstruction axis를 바꾸므로 기본적으로 사용하지 않는다. Intensity augmentation은 raw robustness 단계 전까지 사용하지 않는다.

---

## 5. 비교 실험

### G0. Linear interpolation

두 observed slice 사이 상대 위치에 따라 grayscale intensity를 선형 보간한다.

\[
\hat I_{z+j}=\left(1-\frac{j}{r}\right)I_z+\frac{j}{r}I_{z+r}
\]

역할:

- 비학습 sanity baseline
- axis, offset 및 output alignment 검증
- SAINT가 단순 interpolation보다 유효한지 확인

### G1. Original SAINT on grayscale rock

SAINT의 두 단계 구조를 유지한다.

```text
Sparse grayscale volume
        |
        +--> sagittal slices --+
        |                       |
        +--> coronal slices ----+--> shared AMI
                                    |
                           two dense volumes
                                    |
                              axial reformat
                                    |
                                   RFN
                                    |
                           dense grayscale volume
```

#### AMI

- Original residual-dense feature learner
- Filter Distance Matrix(FDM)
- Dynamic filter generation
- Anisotropic sub-pixel convolution
- Sagittal/coronal shared weights

FDM에는 array scale만이 아니라 실제 voxel spacing을 전달한다.

#### RFN

Sagittal/coronal 결과의 평균에 residual correction을 더한다.

\[
I_{fuse}=I_{avg}+F(I_{sag},I_{cor})
\]

#### G1 loss

원 논문 기준선과 같이 L1을 사용한다.

\[
\mathcal L_{L1}=\frac{1}{N}\sum_i|\hat I_i-I_i|
\]

AMI와 RFN의 stage별 loss 정의와 mask 범위는 upstream 코드 확인 후 문서화한다. 포팅 과정에서 loss 범위를 조용히 변경하지 않는다.

### G2. SAINT + gradient loss

G1과 architecture, data, split 및 optimizer를 동일하게 유지하고 loss만 변경한다.

\[
\mathcal L_{G2}=\mathcal L_{L1}+\lambda_g\mathcal L_{grad}
\]

2D finite difference를 다음처럼 정의한다.

\[
D_xI_{i,j}=I_{i+1,j}-I_{i,j}
\]

\[
D_yI_{i,j}=I_{i,j+1}-I_{i,j}
\]

\[
\mathcal L_{grad}^{2D}
=\|D_x\hat I-D_xI\|_1
+\|D_y\hat I-D_yI\|_1
\]

초기 비교 후보는 다음과 같다.

```text
lambda_grad: 0.05, 0.10
```

후보 선택은 validation missing-slice MAE와 gradient MAE를 함께 보고 수행한다. Test 결과로 lambda를 선택하지 않는다.

3D gradient 또는 z-gradient를 추가할 경우 G2와 섞지 않고 `G2-3D`로 별도 실험한다.

### T1. 3D topology-aware SAINT

현재 필수 실험이 아니다. 다음 조건이 모두 충족될 때만 시작한다.

- G1과 G2가 재현 가능하게 완료됨
- segmentation operator가 동결됨
- grayscale 지표는 양호하지만 pore-throat disconnection이 반복됨
- 충분한 GPU 시간과 3D subvolume batch가 확보됨

Loss와 구현 선택은 별도 decision record에서 승인한다.

---

## 6. Upstream SAINT 재현 및 포팅

### 6.1 Source

- Repository: <https://github.com/cpeng93/SAINT>
- Framework base: EDSR-PyTorch
- Documented environment: Python 3.7, PyTorch 1.0
- Public checkpoint: x4/x6
- Training: AMI Stage 1, RFN Stage 2

Upstream repository에는 명시적인 license file이 확인되지 않았다. 다음 원칙을 적용한다.

- 처음에는 별도 clone과 pinned commit으로 재현한다.
- upstream 전체 코드를 이 repository에 복사하지 않는다.
- 재배포나 공개 수정본이 필요하면 저자에게 사용 조건을 확인한다.
- 외부 코드의 commit hash와 local patch를 분리해 기록한다.

### 6.2 Reproduction gate

포팅 전에 다음을 완료한다.

- checkpoint load 성공
- upstream sample 또는 synthetic tensor inference 성공
- stage 1/2 output shape 기록
- scale x4 output alignment 확인
- FDM spacing/coordinate 입력 확인
- normalization과 `n_colors=3` 의미 확인
- 같은 input을 반복했을 때 deterministic output 확인

### 6.3 Modern port gate

- Pinned upstream과 ported model의 parameter mapping 기록
- 같은 checkpoint/input에서 tensor별 최대·평균 차이 기록
- 차이가 발생하면 원인을 decision record에 작성
- parity 승인 전에는 digital-rock headline training을 시작하지 않음

---

## 7. 학습 프로토콜

### 7.1 우선 원칙

G1은 architecture뿐 아니라 loss와 가능한 training behavior도 원 구현에 가깝게 유지한다. 편의를 위해 AdamW, 다른 normalization layer 또는 새로운 scheduler로 교체하지 않는다.

실제 optimizer, beta, scheduler 및 initialization은 upstream 코드를 audit한 뒤 config에 고정한다. 논문 서술과 코드가 다르면 둘 다 기록하고 코드를 우선 재현한다.

### 7.2 Two-stage training

#### Stage 1: AMI

1. Sparse volume에서 sagittal/coronal training slices 생성
2. Shared AMI 학습
3. Validation missing-slice MAE로 best checkpoint 선택
4. Best AMI로 train/validation dense intermediate volume 생성

#### Stage 2: RFN

1. AMI를 고정
2. Sagittal/coronal prediction을 RFN input으로 구성
3. RFN residual 학습
4. Validation missing-slice MAE로 best checkpoint 선택

Average-only fusion은 RFN 학습이 막힐 경우 diagnostic baseline으로 기록하며 RFN 결과로 표기하지 않는다.

### 7.3 Checkpoint selection

Primary selection metric:

\[
\mathrm{Validation\ Missing\text{-}slice\ MAE}
\]

금지:

- 마지막 epoch를 자동 선택
- Test PSNR/SSIM으로 epoch 선택
- Test structure/permeability로 loss weight 선택
- 모델별로 다른 normalization 사용

### 7.4 Seeds

- Pipeline debug: seed 0 한 번
- Headline experiment: 최소 3 seeds
- Data split과 validation/test offset은 seed와 무관하게 고정

---

## 8. 평가 프로토콜

### 8.1 Missing-slice mask

평가 mask는 HR grid에서 observed 위치를 제외한다.

```text
r=4
observed: 1 0 0 0 1 0 0 0 1 ...
missing : 0 1 1 1 0 1 1 1 0 ...
```

모든 primary image metric은 missing mask에서 계산한다.

### 8.2 Grayscale metrics

Primary:

- MAE/L1
- PSNR
- SSIM

Supporting:

- Gradient MAE
- Error by relative position `j/r`
- Intensity histogram distance
- Inference time and peak memory

PSNR의 data range는 normalized `[0,1]`로 고정한다. SSIM window와 implementation version을 기록한다.

### 8.3 Full-volume construction

1. Model prediction을 HR z grid에 정렬한다.
2. Observed slice는 input grayscale로 교체한다.
3. 동일 ROI crop을 적용한다.
4. Grayscale reconstructed volume을 저장한다.
5. Frozen segmentation operator를 적용한다.

Observed slice를 복사한 full volume은 구조·물성 평가에만 사용한다. Primary missing-slice metric에는 사용하지 않는다.

### 8.4 Segmentation protocol

구조 지표 전에 segmentation operator를 동결한다.

초기 원칙:

- 제공 binary volume은 reference label이다.
- Predicted grayscale과 GT grayscale에는 동일 operator를 적용한다.
- Threshold/segmenter calibration은 train/validation core에서만 수행한다.
- Test core나 특정 model의 결과를 보고 조정하지 않는다.
- Segmentation version과 parameters를 prediction metadata에 기록한다.

후보:

1. Training cores에서 고정한 global intensity threshold
2. Training cores에서 학습하고 동결한 lightweight segmenter

Model마다 독립적인 Otsu threshold를 적용하는 방식은 intensity calibration 차이를 가릴 수 있으므로 primary protocol로 사용하지 않는다.

### 8.5 Morphology and topology metrics

- Dice and IoU against provided binary reference
- Absolute porosity error
- Specific surface area error
- Pore-size distribution distance
- Directional two-point correlation distance
- Euler characteristic density error
- Connected-component count/density
- Largest connected component ratio
- z-percolating porosity
- Percolation failure rate

### 8.6 Permeability

Primary physical metric은 z-direction single-phase permeability이다.

\[
E_{k_z}=|\log_{10}\hat k_z-\log_{10}k_z|
\]

모든 model/GT에 다음 조건을 동일하게 적용한다.

- Solver와 version
- Viscosity and pressure gradient
- Inlet/outlet and lateral boundary conditions
- Convergence criterion
- Physical voxel spacing
- Test block physical size

GT는 percolating하지만 prediction이 non-percolating인 경우 임의의 작은 permeability를 대입하지 않는다. Percolation failure로 별도 보고한다.

---

## 9. 실험 매트릭스

### Phase 0 - 재현 및 pipeline

| ID | Data | Scale | Model | Seeds |
|---|---|---:|---|---:|
| R0 | Upstream-compatible sample | x4 | upstream SAINT checkpoint | 1 |
| D0 | One filtered sandstone core | x4 | linear | 1 |

### Phase 1 - 핵심 grayscale baseline

| ID | Data | Scale | Model/Loss | Seeds |
|---|---|---:|---|---:|
| G0-x4 | Filtered grayscale | x4 | linear | deterministic |
| G1-x4 | Filtered grayscale | x4 | SAINT + L1 | 3 |
| G2a-x4 | Filtered grayscale | x4 | SAINT + L1 + 0.05 gradient | 3 |
| G2b-x4 | Filtered grayscale | x4 | SAINT + L1 + 0.10 gradient | 3 |

Validation으로 G2a/G2b 중 하나를 선택한 뒤 test를 한 번 평가한다.

### Phase 2 - robustness

| ID | Data | Scale | Purpose |
|---|---|---:|---|
| G1-x2 | Filtered grayscale | x2 | unseen/easier scale |
| G1-x6 | Filtered grayscale | x6 | larger gap |
| G1-raw-x4 | Raw grayscale | x4 | acquisition-noise robustness |

### Phase 3 - property evaluation

G0/G1/selected G2의 frozen predictions에 동일 segmentation과 solver를 적용한다.

### Phase 4 - conditional topology extension

Go/no-go gate를 통과한 경우 x4에서 먼저 수행한다.

---

## 10. 통계

- 통계 단위: held-out core 또는 사전 등록된 non-overlapping block
- Slice/voxel을 독립 표본으로 간주하지 않음
- 3 seed 결과와 core별 결과를 분리해 제시
- Paired comparison은 동일 core, offset, ROI를 사용
- 평균과 표준편차, 필요 시 median/IQR 보고
- Core/block bootstrap 95% confidence interval 권장

주요 비교:

1. G0 vs G1: 학습 기반 SAINT의 효과
2. G1 vs G2: gradient loss의 효과
3. x2 vs x4 vs x6: slice gap의 영향
4. Grayscale error와 connectivity/permeability error의 관계

---

## 11. 산출물 계약

### 11.1 Prediction volume metadata

```text
experiment_id
model_id
git_commit
upstream_saint_commit
config_name
dataset_doi
dataset_variant
source_core_id
split_manifest_version
split_manifest_hash
normalization_id
sampling_axis
sampling_factor
sampling_offset
hr_voxel_spacing_um
lr_voxel_spacing_um
roi_id
seed
loss_name
loss_weights
checkpoint_epoch
validation_missing_mae
prediction_dtype
prediction_shape
segmentation_protocol_version
```

### 11.2 저장 정책

Git에 포함:

- Source code
- Small configs and manifests
- Checksums and metadata
- Metric summaries
- Reproducible figure scripts
- Decision records

Git에 포함하지 않음:

- Original `.raw` volumes
- Generated patch arrays
- Model checkpoints
- Full prediction volumes
- Solver bulk outputs
- Credentials and local paths

---

## 12. 팀별 인수인계

### Team A to Team B

- Audited dataset metadata
- Frozen split manifest
- Normalization config
- Axis convention and sparse sampling tests
- Train/validation/test access method

### Team B to Team A/C

- Pinned upstream commit
- Model/config entry point
- Tensor shape contract
- Required GPU memory
- Checkpoint and prediction metadata
- Known deviations from upstream

### Team C to all

- Metric implementation/version
- Frozen segmentation protocol
- Raw metric table and summary
- Permeability solver settings/convergence
- Failure cases and interpretation limits

---

## 13. Go/no-go gates

### Gate A - 데이터 준비 완료

- 11 core의 shape/dtype/checksum 확인
- Axis convention 확인
- Split manifest 동결
- Normalization 동결

### Gate B - upstream 재현 완료

- x4 checkpoint inference 성공
- Stage output shape 확인
- FDM spacing semantics 확인
- Pinned commit 기록

### Gate C - G1 완료

- Three-seed x4 L1 training 완료
- Test core prediction 동결
- Missing-slice metrics 산출

### Gate D - G2 완료

- Gradient implementation unit test
- Validation-only lambda 선택
- Same-seed paired G1/G2 comparison

### Gate E - topology 필요성 판단

다음 중 하나가 반복되면 topology extension을 검토한다.

- PSNR/SSIM은 개선되지만 pore connectivity가 악화
- Thin throat disconnection이 대표 failure mode
- Gradient loss가 edge를 선명하게 하지만 false connection을 증가
- Permeability error가 grayscale metric과 낮은 상관을 보임

---

## 14. 리스크와 대응

| Risk | Response |
|---|---|
| Upstream code가 오래된 PyTorch에 의존 | Legacy 재현 후 parity test를 거쳐 포팅 |
| Upstream license가 명시되지 않음 | 코드를 복사하지 않고 commit pin, 재배포 전 확인 |
| Core 수가 11개로 적음 | Core split 유지, patch 수가 아닌 core/block CI 보고 |
| Synthetic z sampling과 실제 scan gap 차이 | 주장 범위 제한, raw/real paired data는 후속 검증 |
| Per-core intensity 차이 | Training-only normalization 동결 |
| Gradient loss가 noise를 강조 | Filtered data부터 시작, raw robustness에서 별도 분석 |
| Segmentation이 결과를 좌우 | Operator/version을 사전 동결하고 모든 model에 동일 적용 |
| PSNR과 물성이 불일치 | 구조·연결성·permeability를 별도 결과로 보고 |
| Patch leakage | Parent-core split을 patch 생성보다 먼저 적용 |

---

## 15. 실행 체크리스트

### Upstream

- [ ] Official repository clone
- [ ] Upstream commit hash 기록
- [ ] License/redistribution 조건 확인
- [ ] Legacy environment 작성
- [ ] x4 checkpoint inference
- [ ] Tensor shape/FDM audit

### Data

- [ ] 11 Sandstones 다운로드 및 citation 기록
- [ ] Checksum, shape, dtype, endianness 확인
- [ ] Grayscale/binary 정렬 확인
- [ ] Artifact/intensity/porosity audit
- [ ] 8/1/2 split manifest 동결
- [ ] Normalization config 동결

### Baseline

- [ ] x4 sampling unit test
- [ ] Linear interpolation alignment test
- [ ] One-core end-to-end dry run
- [ ] G1 AMI training
- [ ] G1 RFN training
- [ ] Three-seed G1 test predictions

### Gradient

- [ ] Finite-difference unit test
- [ ] Constant/ramp/edge phantom test
- [ ] Lambda 0.05/0.10 validation comparison
- [ ] Selected G2 three-seed test predictions

### Evaluation

- [ ] Missing MAE/PSNR/SSIM
- [ ] Relative-position error
- [ ] Segmentation protocol freeze
- [ ] Morphology/connectivity metrics
- [ ] Permeability solver consistency
- [ ] Core/block-level confidence intervals

### Decision

- [ ] Topology extension go/no-go
- [ ] Independent 2D/3D baseline 필요성 재검토
- [ ] x2/x6/raw robustness 우선순위 확정

---

## 16. 주장 범위

권장 표현:

> We evaluate the transfer of SAINT to held-out grayscale sandstone Micro-CT volumes under synthetic sparse-z sampling, first reproducing the original L1 objective and then isolating the effect of gradient regularization.

금지 또는 주의 표현:

- 실제 미관측 pore 구조를 유일하게 복원했다
- 높은 PSNR이 topology/permeability 보존을 보장한다
- 모든 최신 digital-rock reconstruction model보다 우수하다
- Synthetic sparse sampling이 실제 저해상도 촬영을 완전히 재현한다
- Test core에 맞춘 segmentation으로 얻은 물성 결과를 일반화한다

---

## 17. 참고자료

1. Peng, C. et al. SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis. CVPR 2020. <https://doi.org/10.1109/CVPR42600.2020.00777>
2. Official SAINT code. <https://github.com/cpeng93/SAINT>
3. Neumann, R. et al. 11 Sandstones: Raw, Filtered and Segmented Data. <https://doi.org/10.17612/F4H1-W124>
4. Lucas-Oliveira, E. et al. Micro-computed tomography of sandstone rocks: Raw, filtered and segmented datasets. Data in Brief 2022. <https://doi.org/10.1016/j.dib.2022.107893>

---

## 18. 변경 이력

### 2026-09-24

- Primary domain을 binary에서 grayscale로 변경
- 11 Sandstones filtered grayscale을 primary dataset으로 확정
- Core-level 8/1/2 split 원칙 추가
- x4 L1 SAINT를 핵심 baseline으로 변경
- Gradient loss를 첫 ablation으로 추가
- Topology loss를 조건부 후속 단계로 이동
- Primary metric을 missing-slice Dice에서 MAE/PSNR/SSIM으로 변경
- Segmentation을 학습 target에서 frozen evaluation stage로 이동
