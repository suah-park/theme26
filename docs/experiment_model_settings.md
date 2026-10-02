# 실험 조건 및 모델 설정 통합본

이 문서는 2026-10-02 현재 팀에서 논의하고 코드에 반영한 실험 조건과 SAINT 모델 설정을 한곳에 정리한 실행 기준서입니다. 세부 API는 [Interface Contract](interface_contract.md), 연구 원칙은 [Research Protocol](research_protocol.md)을 따릅니다.

상태 표기는 다음과 같습니다.

- **확정**: 본 실험에서 그대로 사용하며 test 결과를 보고 바꾸지 않는 값
- **구현 완료**: 현재 코드와 CPU 테스트로 동작을 확인한 항목
- **GPU pilot 후 확정**: 과학적 조건은 유지하되 GPU 메모리·속도에 맞춰 정할 실행값
- **구현/동결 필요**: 본 실험 전에 코드 구현 또는 train/validation 기반 선택이 필요한 항목

## 1. 연구 목적과 비교 범위

### 1.1 연구 질문

> Berea sandstone의 x4 sparse-z grayscale Micro-CT에서 SAINT 기반 interpolation이 linear interpolation보다 누락 단면을 잘 복원하며, gradient 및 3D topology-aware objective가 pore boundary, connectivity와 물성 보존을 개선하는가?

### 1.2 최종 비교군

| ID | 방법 | 학습 objective | Seed | 상태 |
|---|---|---|---:|---|
| E0 | Z축 linear interpolation | 없음 | deterministic | 구현 완료 |
| E1 | Grayscale SAINT baseline | `L1` | 1, 2, 3 | 기본 학습 엔진 구현 완료 |
| E2 | Gradient ablation | `L1 + 0.10 × gradient` | 1, 2, 3 | gradient 함수 구현, 학습 통합 필요 |
| E3 | Topology-aware RFN fine-tuning | `L1 + 0.10 × gradient + 0.10 × topology` | 1, 2, 3 | 구현/동결 필요 |

E0부터 E3까지 동일한 split, crop, x4 sampling, normalization과 평가 프로토콜을 사용합니다. E2는 E1에서 gradient 항만 추가하고, E3는 같은 seed의 E2 best checkpoint에서 시작합니다.

### 1.3 연구 범위에 포함하지 않는 것

- 다른 sandstone 종류에 대한 일반화 주장
- x2 또는 x6를 primary experiment로 사용하는 실험
- raw grayscale robustness 실험
- binary volume 자체를 reconstruction input/target으로 사용하는 실험
- GAN, diffusion, architecture search
- persistent-homology loss와 permeability loss
- 실제 LR/HR scanner pair를 사용했다는 주장
- test set을 이용한 threshold, loss weight 또는 checkpoint 선택

## 2. 데이터 조건

### 2.1 원본 데이터

| 항목 | 설정 | 상태 |
|---|---|---|
| Source dataset | 11 Sandstones: Raw, Filtered and Segmented Data | 확정 |
| DOI | `10.17612/F4H1-W124` | 확정 |
| License | ODC-By 1.0 | 확정 |
| 실제 사용 sample | Berea sandstone 1개 | 확정 |
| Grayscale | `Berea_2d25um_grayscale_filtered.raw` | 확정 |
| Binary | grayscale와 정렬된 Berea binary raw | 파일 metadata/polarity 동결 필요 |
| Source shape | `1000 × 1000 × 1000` | 확인 완료 |
| Source dtype | `uint8` | 확인 완료 |
| Raw storage order | C-order `XYZ`로 직접 해석, transpose 없음 | 최신 합의 반영 |
| Voxel spacing | `(2.25, 2.25, 2.25) µm` | 확정 |

Grayscale이 모델의 입력과 reconstruction target입니다. Binary는 E3의 auxiliary topology target과 모든 모델의 topology/physics 평가에만 사용합니다.

### 2.2 Canonical 데이터 형식

```text
NumPy volume:       [X,Y,Z]
PyTorch volume:     [B,C,X,Y,Z]
Grayscale dtype:    float32
Grayscale range:    [0,1]
Internal pore mask: bool 또는 uint8, pore=1/True, solid=0/False
Sparse axis:        Z, axis index 2
```

RAW는 직접 XYZ 배열로 읽습니다. 모든 모델·loss·평가 함수에도 같은 XYZ를 전달합니다. Z방향 물성은 이 좌표계의 Z입니다.

### 2.3 공간 분할

단일 Berea volume을 먼저 X축 slab으로 분리한 뒤 각 영역 안에서만 `256³` crop을 만듭니다.

```text
Train:      X=[0,400)      48 random crops
Buffer:     X=[400,432)    unused
Validation: X=[432,688)     9 fixed crops
Buffer:     X=[688,720)    unused
Test:       X=[720,976)     9 fixed crops
Unused:     X=[976,1000)
```

- Crop shape: `256 × 256 × 256`
- Physical crop size: `576 × 576 × 576 µm³`
- Split 간 최소 buffer: 32 voxel
- Train crop seed: 2026
- Validation/test YZ origin grid: `(0, 372, 744) × (0, 372, 744)`
- Train crop끼리는 겹칠 수 있지만 서로 다른 split의 crop은 겹치지 않습니다.
- 고정 manifest: `configs/data/berea_crops.yaml`
- Manifest version: `berea-spatial-xyz-v2`
- 통계적 표본 단위는 voxel이나 slice가 아니라 사전 등록된 3D block입니다.

이 설계는 parent-core generalization이 아니라 **한 Berea volume 내부의 spatial holdout**입니다. 따라서 결론도 Berea sandstone에 한정합니다.

### 2.4 Grayscale normalization

Validation/test를 사용하지 않고 train slab `X=[0,400)`의 400,000,000 voxel만으로 계산했습니다.

| 항목 | 값 |
|---|---:|
| Method | clipped percentile min-max |
| Lower percentile | 0.5 |
| Upper percentile | 99.5 |
| Source lower | uint8 `2` |
| Source upper | uint8 `184` |
| Observed train minimum/maximum | `0 / 255` |
| Normalization ID | `berea-train-xyz-p0.5-p99.5-v2` |

계산식은 다음과 같습니다.

```text
I_clipped = clip(I_raw, 2, 184)
I_model   = (I_clipped - 2) / (184 - 2)
```

정규화 기준 파일은 `configs/data/berea_normalization.yaml`입니다. 모든 split과 E0–E3에 같은 값을 적용하며 개별 crop 또는 test data에 맞춰 다시 계산하지 않습니다.

### 2.5 Sparse-z sampling

| 항목 | 값 | 상태 |
|---|---:|---|
| Sampling axis | Z | 확정 |
| Factor | 4 | 확정 |
| Offset | 0 | 확정 |
| Dense depth | 256 | 확정 |
| Observed slices | 64개: `0,4,8,...,252` | 확정 |
| Missing slices | 192개 | 확정 |
| Primary bracketed missing slices | 189개: `1..251` 중 missing | 확정 |
| Unbracketed tail | `253,254,255` | diagnostic only |
| Boundary mode for E0 | replicate | 확정 |

```text
I_LR = I_HR[:,:,0::4]
```

모델은 HR Z-grid 전체를 예측하지만, 최종 3D 평가 volume에서는 observed slice를 원본 sparse input으로 다시 삽입합니다.

```text
full_prediction[..., observed_indices] = sparse_input
```

이 data-consistency 단계는 AMI 내부가 아니라 추론 후 공통 후처리에서 수행합니다.

## 3. SAINT 모델 설정

### 3.1 전체 흐름

```text
Sparse grayscale [B,1,X,Y,Z_lr]
  ├─ fixed-X sagittal marginals [B·X,3,Y,Z_lr]
  └─ fixed-Y coronal marginals  [B·Y,3,X,Z_lr]
          ↓ shared AMI
Sagittal/Coronal dense volumes [B,1,X,Y,Z_hr]
          ↓ RFN axial fusion
Raw dense grayscale prediction [B,1,X,Y,Z_hr]
          ↓ observed-slice reinsertion
Evaluation-ready full volume
```

AMI의 입력 채널 3개는 RGB가 아니라 현재 marginal plane과 양옆의 이웃 plane입니다. 첫 plane과 마지막 plane에서는 경계를 복제합니다.

모델 기준은 Peng et al., *SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis* (CVPR 2020)와 이전 기준서에 기록된 공개 코드 commit `a4a1b8f9c7b74a5c5d00b8b08132ff15a1e65802` (이번 설정 정리에서 재검증하지 않음)입니다. 공개 코드는 구조와 tensor flow를 확인하는 reference로만 사용하며, 이 저장소는 현대 PyTorch 기반 독립 구현입니다. 공개 코드의 고정 `512` 반복 대신 실제 입력 shape를 사용합니다.

목표 모델과 CPU 검사 모델의 공통 설정 파일은 `configs/model/saint_x4.yaml`입니다.
`ami_target`/`rfn_target`은 본 모델, `ami_cpu`/`rfn_cpu`는 연결 검사 모델입니다.
현재 실행 가능한 CPU YAML은 공통 파일의 CPU profile을 읽습니다.

### 3.2 본 실험 목표 AMI 설정

| 항목 | 값 |
|---|---:|
| Upsampling factor `R` | 4 |
| Input channels | 3 neighboring grayscale marginals |
| Feature channels `C` | 64 |
| Kernel size `K` | 3 |
| Residual-dense blocks | 6 |
| Layers per block | 8 |
| Growth rate | 32 |
| Output channels | 1 grayscale |
| Upsampled dimension | Z only |

AMI는 shallow convolution, residual-dense feature learner, global feature fusion, FDM 기반 dynamic filter와 anisotropic sub-pixel rearrangement로 구성됩니다.

### 3.3 Filter Distance Matrix

- Shape: `[1 또는 B, R, K, K] = [1 또는 B,4,3,3]`
- Unit: µm
- Sagittal marginal은 in-plane spacing으로 `sy`, coronal marginal은 `sx`를 사용합니다.
- 두 방향 모두 upsampling dimension에는 `sz`를 사용합니다.
- HR spacing은 `(2.25,2.25,2.25) µm`, x4 sparse Z 간격은 `9.0 µm`입니다.
- FDM은 subpixel phase별 물리적 거리를 dynamic filter generator에 제공합니다.

### 3.4 Dynamic filter generator

FDM의 각 subpixel phase를 다음 meta-network에 통과시킵니다.

```text
Conv 1→32 → ReLU
Conv 32→64 → ReLU
Conv 64→64 → ReLU
Conv 64→64 → ReLU
Conv 64→C
```

모든 convolution은 `3×3`, padding 1입니다. 출력 shape은 `[B,R,C,K,K]`입니다.

논문 식 (5)의 해석과 달리 공개 구현은 phase 0을 포함한 모든 subpixel phase를 예측합니다. 현재 구현도 공개 코드의 학습 동작을 따르고, observed slice 보존은 최종 volume 조립 단계에서 보장합니다.

### 3.5 본 실험 목표 RFN 설정

| 항목 | 값 |
|---|---:|
| Input channels | 2: sagittal, coronal |
| Feature channels | 64 |
| Kernel size | 3 |
| Residual-dense blocks | 4 |
| Layers per block | 6 |
| Growth rate | 12 |
| Output channels | 1 residual grayscale |

RFN은 Z별 axial slice `[N,2,X,Y]`를 처리합니다.

```text
I_avg   = (I_sagittal + I_coronal) / 2
I_final = I_avg + RFN_residual(I_sagittal, I_coronal)
```

### 3.6 Pilot 모델과 본 모델의 구분

현재 CPU pilot config는 데이터 연결만 빠르게 확인하기 위한 축소 모델입니다.

| 항목 | CPU/Berea pilot | 본 실험 목표 |
|---|---:|---:|
| Feature channels | 8 | 64 |
| AMI blocks | 1 | 6 |
| AMI layers/block | 2 | 8 |
| AMI growth rate | 4 | 32 |
| RFN blocks | 해당 pilot 미실행 | 4 |
| RFN layers/block | 해당 pilot 미실행 | 6 |
| RFN growth rate | 해당 pilot 미실행 | 12 |
| Batch size | 1 | GPU pilot 후 확정 |
| Epochs/steps | 1 epoch, 1 step | GPU pilot 후 확정 |
| Mixed precision | off | GPU에서 on 우선 |

Pilot 결과는 연구 결과나 모델 성능으로 보고하지 않습니다.

## 4. Loss 적용 범위

### 4.1 E1: grayscale L1

```text
L1 = mean(|prediction - target|)
```

- 적용 데이터: 정규화된 2D grayscale AMI/RFN output과 target
- Binary target은 사용하지 않습니다.
- 현재 E1 engine은 batch에 mask가 없으면 output 전체에 L1을 적용합니다.
- **AMI/RFN에서 all-phase loss를 유지할지 missing-phase만 사용할지는 본학습 config 전에 최종 동결해야 합니다.** 평가 mask와 training mask는 서로 다른 개념입니다.

### 4.2 E2: grayscale gradient

```text
L_grad = mean(|Dx(pred)-Dx(target)|) + mean(|Dy(pred)-Dy(target)|)
L_E2   = L1 + 0.10 × L_grad
```

- 적용 데이터: 2D grayscale output
- 의미: grayscale intensity의 인접 pixel 차이, 즉 pore/solid 경계의 선명도
- 이것은 신경망 역전파의 gradient가 아니라 영상의 공간 미분입니다.
- L1은 절대 밝기 오차, gradient loss는 밝기 변화와 경계 오차를 줄입니다.
- 2D와 3D gradient 함수는 구현되어 있지만 E2 training loop 연결은 아직 필요합니다.

### 4.3 E3: 3D topology-aware loss

E3는 E2 best checkpoint에서 시작하고 AMI는 고정하며 RFN만 fine-tuning합니다.

1. RFN이 dense 3D grayscale을 출력합니다.
2. Observed slice를 원본 grayscale로 다시 삽입합니다.
3. Dark-pore 가정에 따라 soft pore probability를 만듭니다.
4. Aligned 3D binary pore target과 비교합니다.

```text
P_pred = sigmoid((threshold - I_final) / temperature)

L_topology = 0.5 × L_softDice + 0.5 × L_soft-clDice
L_E3       = L1 + 0.10 × L_grad + 0.10 × L_topology
```

- L1/gradient는 2D grayscale 수준에서 사용 가능합니다.
- Topology loss는 조립된 3D volume과 aligned binary segmentation이 있어야 사용 가능합니다.
- Observed slice는 상수로 조립하여 해당 위치를 통해 gradient가 흐르지 않게 합니다.
- Soft-clDice는 연결성 surrogate이며 완전한 위상 보장을 의미하지 않습니다.
- Persistent homology는 현재 범위에서 제외합니다.
- Soft threshold, temperature, skeletonization iteration은 train/validation으로만 정합니다.
- Topology loss 함수와 E3 fine-tuning loop는 아직 구현되지 않았습니다.
- E3 학습 후 최종 output도 binary가 아니라 dense 3D grayscale입니다. Binary는 loss reference와 후속 평가에만 사용합니다.

### 4.4 물성은 loss가 아니라 평가

Permeability와 tortuosity는 미분 가능한 training loss로 사용하지 않습니다. 각 모델의 frozen prediction을 동일하게 segmentation한 뒤 외부 solver로 평가합니다. 따라서 E3에서도 물성 계산은 역전파 graph에 들어가지 않습니다.

## 5. 학습 순서와 checkpoint 선택

### 5.1 전체 실행 순서

```text
Raw/binary audit
→ spatial manifest 및 normalization 고정
→ CPU smoke/pilot
→ GPU one-batch overfit
→ E0 linear interpolation
→ E1 AMI 학습
→ E1 AMI inference로 directional volume 생성
→ E1 RFN 학습
→ E2 gradient experiment
→ E3: E2 best에서 AMI freeze, RFN 3D fine-tuning
→ frozen test inference
→ grayscale 평가
→ frozen segmentation
→ topology/physics 평가
```

E1은 AMI와 RFN을 한 번에 처음부터 end-to-end로 학습하는 것이 아니라 stage별로 진행합니다.

E2를 동일 seed로 처음부터 학습할지 E1 checkpoint에서 이어서 학습할지는 아직 동결되지 않았습니다. 공정한 ablation을 위해 초기화 규칙을 본학습 전에 config에 명시하고 모든 seed에 동일하게 적용해야 합니다.

### 5.2 Optimizer 및 실행값

현재 pilot에 실제로 사용한 값은 다음과 같습니다.

| 항목 | Pilot 값 | 본학습 상태 |
|---|---:|---|
| Optimizer | Adam | 유지 예정 |
| Learning rate | `1e-4` | GPU pilot 후 최종 확인 |
| Weight decay | `0` | GPU pilot 후 최종 확인 |
| Gradient clipping | norm `1.0` | 유지 예정 |
| Batch size | 1 | GPU VRAM 측정 후 확정 |
| Mixed precision | off | GPU 본학습에서는 on 우선 |
| Data workers | 0 | 서버 CPU/RAM 확인 후 확정 |
| Epochs | 1 | 본학습 epoch/early stopping 동결 필요 |
| Max steps | train 1 / validation 1 | pilot 전용 |

GPU 계정을 받은 뒤 목표 모델로 10–100 step pilot을 실행하여 다음 실행값만 조정합니다.

- Batch size
- Gradient accumulation 필요 여부
- DataLoader worker 수와 pin memory
- Marginal chunk size
- Wall time와 checkpoint 주기

Crop `256³`, x4 factor, 모델 구조, loss 정의와 split은 GPU 메모리 문제 때문에 바꾸지 않습니다. OOM이면 mixed precision, batch 축소, marginal chunking, gradient accumulation/checkpointing 순서로 대응합니다.

### 5.3 Checkpoint 선택

- E1/E2: validation primary-mask MAE를 기준으로 best checkpoint 선택
- E3: validation total E3 loss를 기준으로 best checkpoint 선택
- Test metric은 checkpoint 또는 hyperparameter 선택에 사용하지 않음
- 각 run에 config, seed, epoch, validation metric과 Git commit을 저장

현재 E1 training engine은 validation L1로 checkpoint를 저장하므로, 본학습 전 primary-mask MAE selection과 stage별 evaluation 연결이 필요합니다.

## 6. 평가 조건

### 6.1 평가용 full volume

모든 방법에서 다음 순서를 동일하게 적용합니다.

1. Raw prediction을 HR Z-grid `[256,256,256]`에 정렬
2. 64개 observed slice를 원본 input으로 교체
3. 동일한 ROI 적용
4. 동일한 frozen hard-segmentation operator 적용
5. Grayscale, topology, physics metric 계산

### 6.2 Grayscale 평가

Primary 189개 bracketed missing slice에서 계산합니다.

- MAE
- PSNR, data range `1.0`
- SSIM: 구현체와 window 동결 필요
- Gradient MAE
- Interpolation 상대 위치별 error: observed slice 사이의 phase 1/2/3
- Unbracketed tail `253–255` boundary diagnostic
- Inference time와 peak GPU memory

### 6.3 Structure/topology 평가

Frozen segmentation 후 3D pore mask에서 계산합니다.

- Dice, IoU
- Absolute porosity error
- Specific surface area error
- Pore-size distribution distance
- Euler characteristic density error
- Connected-component 통계
- Largest connected pore ratio
- Z-direction percolation과 failure rate

Hard segmentation threshold/operator는 train/validation에서만 정하고 모든 E0–E3 prediction과 ground truth에 동일하게 적용합니다.

### 6.4 Physics 평가

- Z-direction tortuosity
- Z-direction single-phase permeability
- Permeability log error
- Percolation failure

Prediction이 non-percolating이면 임의의 작은 permeability를 넣지 않고 failure로 기록합니다. Ground truth와 모든 모델에 같은 physical spacing, solver, inlet/outlet direction, boundary condition, convergence criterion을 적용합니다.

Solver와 상세 boundary condition은 아직 동결되지 않았습니다.

### 6.5 통계와 보고

- Headline learned-model seed: `1,2,3`
- Debug seed: `0`
- Split/crop/offset은 seed와 무관하게 고정
- 같은 test block에 대한 paired comparison
- Slice와 voxel을 독립 표본으로 세지 않음
- Seed별 값과 test block별 값을 보존하고 평균·분산을 보고
- E0는 deterministic이므로 seed 반복 없음

## 7. 저장과 실행 환경

```text
/project/<group>/theme26/              # Git clone
/project/<group>/data/11_sandstones/   # raw/processed data
/project/<group>/runs/theme26/         # checkpoints, predictions, logs
/project/<group>/envs/theme26/         # Python environment
```

- Raw data, checkpoint, full prediction, solver output은 Git에 올리지 않습니다.
- Code, YAML config와 작은 평가 결과만 Git으로 공유합니다.
- GPU: 1장 사용 예정
- GPU model/VRAM, Slurm partition, CPU/RAM, wall time은 계정 발급 후 확인합니다.
- 서버에서는 `THEME26_GRAYSCALE_RAW` 환경 변수로 raw 경로를 전달합니다.

## 8. 재현성을 위해 저장할 metadata

각 run과 prediction에 다음을 연결합니다.

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
solver_name_and_version
```

## 9. 현재 구현 및 검증 상태

### 9.1 구현 완료

- Raw uint8 file-size/dtype/endianness 검사와 memory mapping
- RAW 직접 C-order XYZ 해석; 전처리와 학습의 축 동일
- Train-only normalization YAML
- Aligned grayscale/binary crop 함수
- Berea spatial manifest 생성·읽기·cross-split leakage 검사
- x4 sparse sampling, observed/missing/primary mask
- E0 linear interpolation과 boundary replicate
- Observed-slice data consistency 조립
- Manifest 기반 AMI lazy DataLoader
- AMI marginal sample 수: train `48×(256+256)=24,576`, validation `9×512=4,608`
- FDM, dynamic filter, AMI, RFN, end-to-end SAINT forward
- Masked L1, 2D/3D gradient loss
- E1 AMI/RFN stage training engine, AMP hook, gradient clipping, checkpoint

### 9.2 검증 완료

- 이전 학습 브랜치 CPU unit test 25개 통과 (이전 기록)
- 현재 통합 브랜치 CPU unit test 48개 통과 (공통 YAML profile 일치·오설정 차단 검사 포함)
- Synthetic AMI/RFN forward/backward와 checkpoint smoke test 통과
- 이전 ZYX 학습 브랜치에서 Berea manifest AMI 1-step CPU pilot 통과 (이전 기록)
- Pilot 연결 확인값: train L1 `0.301347`, validation L1 `0.183935`

이 pilot 수치는 이전 ZYX 해석의 축소 모델·각 1 step 기록입니다. 최신 직접 XYZ 설정의 검증 결과나 논문 성능으로 재사용하지 않습니다. 이번 설정 통일 작업에서는 실제 학습을 실행하지 않았습니다.

### 9.3 아직 구현 또는 동결할 항목

1. Berea binary checksum, shape, 저장 순서, grayscale alignment와 pore polarity 확정
2. 목표 64-channel SAINT GPU one-batch overfit
3. 본학습 batch size, epoch/early stopping, worker, AMP 및 Slurm resource 확정
4. E1 AMI inference 결과를 RFN 학습 input으로 만드는 실제-data pipeline
5. AMI/RFN L1 및 gradient training mask 범위 확정
6. E2 initialization 규칙 확정 및 gradient loss를 training loop에 연결
7. Soft pore threshold, temperature와 skeletonization iteration 선택
8. Soft Dice, 3D soft-clDice 및 E3 RFN fine-tuning 구현
9. Image metric의 SSIM 구현/window 동결
10. Hard segmentation protocol 동결
11. Topology metric 구현
12. Tortuosity/permeability solver와 boundary condition 동결 및 adapter 구현
13. Seed 1/2/3 본학습과 frozen test evaluation

## 10. 실행 명령

### 10.1 Berea manifest pilot

PowerShell:

```powershell
$env:THEME26_GRAYSCALE_RAW = "C:\path\to\Berea_2d25um_grayscale_filtered.raw"
python scripts/train.py --config configs/experiments/e1_ami_berea_pilot.yaml --device cpu
```

Linux/Slurm login node:

```bash
export THEME26_GRAYSCALE_RAW=/project/<group>/data/11_sandstones/raw/Berea_2d25um_grayscale_filtered.raw
python scripts/train.py --config configs/experiments/e1_ami_berea_pilot.yaml --device cpu
```

### 10.2 GPU 확인 후 pilot

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python scripts/train.py --config <GPU-pilot-config> --device cuda
```

GPU pilot config와 `slurm/train_e1.sbatch`은 실제 cluster 정보와 VRAM을 확인한 뒤 생성합니다.

## 11. 해석과 주장 범위

가능한 표현:

> Berea sandstone의 spatially held-out Micro-CT block에서 SAINT 기반 x4 sparse-z reconstruction을 평가하고, L1, gradient-regularized, 3D connectivity-aware objective가 영상·위상·물성 지표에 미치는 영향을 비교하였다.

하지 않을 주장:

- 다른 모든 sandstone 또는 rock type에 일반화된다.
- 복원된 pore structure가 실제 미관측 구조의 유일한 정답이다.
- PSNR 향상이 permeability 정확도 향상을 자동으로 보장한다.
- Soft-clDice가 모든 위상 특성을 보존한다.
- Synthetic sparse sampling이 실제 저해상도 촬영 과정을 완전히 재현한다.

## 12. 최신 통합 설정의 적용 범위

- 최신 합의(직접 XYZ 해석)가 이전 문서의 ZYX 변환보다 우선한다.
- `configs/data/`의 YAML 구역·정규화·crop 좌표를 학습 시작 시 함께 검사한다.
- 실제 Berea train 4억 voxel의 직접 XYZ 정규화는 2/184이며 평가 crop 18개 QC와 AMI DataLoader 값 일치를 확인했다.
- 모델 구조는 공통 YAML로 통일했지만 CPU 검증 모델을 본 모델로 바꾸지 않았다.
- Checkpoint primary-mask MAE 선택은 목표 규칙이며 현재 전체 L1 선택과 구분한다.
- Gradient loss의 observed–missing 경계 정책과 AMI/RFN training mask는 아직 동결하지 않았다.
- E2/E3, 실제 RFN 연결, 최종 추론·평가, 목표 모델 GPU 실행은 이번 설정 정리에서 구현하거나 실행하지 않았다.
- 설치: `python -m pip install -e ".[training]"` (Torch 및 PyYAML 포함).
