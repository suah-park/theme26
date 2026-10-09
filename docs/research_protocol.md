# 연구 기준: Berea sparse-Z reconstruction

실험 설계와 모델 설정의 단일 설명 문서입니다. 명령어는 [E1 실행 안내](e1_execution.md), API는 [인터페이스](interface_contract.md)를 따릅니다. 수치 설정의 실행 기준은 `configs/` YAML입니다.

## 1. 연구 질문과 범위

Filtered grayscale Berea Micro-CT의 synthetic x4 sparse-Z 입력에서 SAINT가 linear interpolation보다 누락 단면을 잘 복원하며, gradient 및 3D connectivity-aware objective가 경계·연결성·물성 보존을 개선하는지 평가합니다.

- 데이터: 11 Sandstones: Raw, Filtered and Segmented Data, DOI `10.17612/F4H1-W124`, ODC-By 1.0. 실제 사용 sample은 Berea 한 core입니다.
- 입력과 reconstruction target은 grayscale입니다. Binary는 E3 auxiliary topology target 및 공통 topology/physics 평가 reference입니다.
- 제외: x2/x6 primary 실험, raw-grayscale robustness, binary-only reconstruction, GAN/diffusion/architecture search, persistent-homology 또는 permeability training loss, 실제 LR/HR scanner pair 주장.
- Dense train 구역 확보 비용도 해석에 포함합니다. 실제 장비 시간·비용 75% 감소나 unseen-core 일반화를 입증했다고 주장하지 않습니다.

## 2. 고정 데이터 기준

| 항목 | 기준 |
|---|---|
| Source | `Berea_2d25um_grayscale_filtered.raw`, `Berea_2d25um_binary.raw` |
| Shape / dtype | `1000³`, `uint8` |
| 저장 해석 | C-order `[X,Y,Z]` 직접 읽기, transpose 없음 |
| HR spacing | `(2.25,2.25,2.25)` µm; sparse Z 간격 9.0 µm |
| Crop | `256³`, 물리적 한 변 576 µm |
| 내부 binary | `True/1=pore`, `False/0=solid`; source pore label 0 |
| 정규화 | train slab의 p0.5/p99.5: `clip(raw,2,184)` 후 `(raw-2)/182`, float32 `[0,1]` |
| 정규화 ID | `berea-train-xyz-p0.5-p99.5-v2`; train voxel 400,000,000 |
| 공간 기준 version | `berea-spatial-xyz-v2` |

각 원본의 출처 URL/DOI, 다운로드 시점, 파일명, SHA256, shape/dtype/endianness/spacing, intensity statistics와 artifact notes를 기록합니다. Binary shape·저장 해석·grayscale alignment·pore polarity는 E3/평가 전에 실제 파일에서 확인해야 합니다.

### 공간 분할

| 구역 | X 범위 (끝 제외) | 사용 |
|---|---|---|
| Train | `[0,400)` | crop seed 2026으로 고정한 48개; train crop끼리 겹칠 수 있음 |
| Buffer | `[400,432)` | 사용하지 않음 |
| Validation | `[432,688)` | 고정 9개; 모델 선택 |
| Buffer | `[688,720)` | 사용하지 않음 |
| Test | `[720,976)` | 고정 9개; 최종 평가 |
| 나머지 | `[976,1000)` | 사용하지 않음 |

Y/Z는 `[0,1000)`. 평가 Y/Z origin은 `{0,372,744} × {0,372,744}`이며 X는 각 slab 시작입니다. 먼저 공간을 분리하고 crop이 해당 구역에 완전히 포함되도록 합니다. 32-voxel buffer가 공간적 독립성을 보장한다는 주장은 하지 않습니다.

`configs/data/berea_split.yaml`은 구역, `berea_normalization.yaml`은 정규화, `berea_eval_crops.yaml`은 평가 좌표·metadata, `berea_crops.yaml`은 모델용 train/validation/test record입니다. 겹치는 평가 좌표는 코드가 일치 여부를 검사합니다. Test나 개별 crop에 맞춰 기준을 다시 정하지 않습니다.

### Sparse sampling과 최종 출력

- Z factor 4, offset 0: dense `[256,256,256]` → sparse `[256,256,64]`.
- Observed 64장: `0,4,...,252`. Missing 192장 중 양쪽 관측 사이의 189장만 primary입니다.
- Tail `253,254,255`는 boundary diagnostic. E0는 마지막 observed slice를 replicate합니다.
- AMI/RFN 중간 출력과 학습 loss는 raw prediction을 사용합니다. 최종 추론·영상 평가에서는 `[0,1]`로 clipping합니다.
- 최종 volume에는 관측 단면을 입력값으로 재삽입합니다. 저장 output은 float32 `[X,Y,Z]` grayscale이며 E3도 binary를 최종 output으로 만들지 않습니다.

## 3. SAINT 구조

공유 구조의 실행 기준은 `configs/model/saint_x4.yaml`입니다. 원본 SAINT(CVPR 2020) 공개 코드는 구조·tensor flow의 reference이고 현대 PyTorch로 독립 구현합니다. 문서만으로 upstream과 완전 동일함을 주장하지 않습니다.

| 설정 | AMI target | RFN target | CPU 검사 profile |
|---|---:|---:|---:|
| Feature channels | 64 | 64 | 8 |
| Kernel | 3 | 3 | 3 |
| Residual-dense blocks | 6 | 4 | 1 |
| Layers/block | 8 | 6 | 2 |
| Growth rate | 32 | 12 | 4 |

AMI factor는 4이며 Z만 upsample합니다. Fixed-X sagittal `[Y,Z]`와 fixed-Y coronal `[X,Z]`에 같은 AMI 가중치를 사용합니다. 입력 3채널은 현재 plane과 양옆 이웃이며 가장자리 이웃은 replicate합니다. AMI output은 `[B,1,L,4*Z_lr]`입니다.

FDM `[1 또는 B,4,3,3]`은 µm 단위입니다. Sagittal in-plane spacing은 `sy`, coronal은 `sx`, Z는 `sz`입니다. Dynamic-filter generator는 3×3 padding-1 convolution의 `1→32→64→64→64→C` 구조이며 마지막 이외 단계에 ReLU를 적용하고 `[B,R,C,K,K]`를 만듭니다. Phase 0을 포함한 모든 phase를 예측하며 observed 보존은 최종 조립에서 수행합니다.

RFN은 같은 Z의 두 방향 axial 결과 `[B,2,X,Y]`를 받아 `I_avg + RFN_residual`로 fusion합니다. Target은 grayscale `[B,1,X,Y]`입니다. 전체 3D forward와 큰 볼륨용 batched inference를 구분합니다.

## 4. 비교 실험과 loss

| ID | Objective | 시작·학습 방식 |
|---|---|---|
| E0 | Z linear interpolation | deterministic, 학습 없음 |
| E1 | 전체 raw output L1 | AMI 학습 → frozen 두 방향 cache → RFN 학습 |
| E2 | L1 + 0.10 × gradient | E1과 같은 구조·데이터·budget; seed 1로 AMI/RFN 각각 새 초기화, AMI 10 epoch → 전용 cache → RFN 10 epoch |
| E3 | L1 + 0.10 × gradient + 0.10 × topology | 같은 seed E2 best에서 AMI freeze, RFN fine-tuning |

모든 비교는 같은 split/crop/sampling/normalization을 사용합니다. Headline learned seeds는 1/2/3, debug seed는 0이며 데이터 crop seed 2026과 구분합니다. E0는 seed 반복이 없습니다.

- L1은 절대 밝기 오차입니다. AMI/RFN 모두 전체 2D raw output에 적용하며 binary target은 쓰지 않습니다.
- Gradient loss는 역전파 gradient가 아니라 영상의 인접 픽셀 밝기 차이입니다. 각 2D 공간 축의 전진 차분(`다음 픽셀 - 현재 픽셀`)에 대해 `mean(|Dx(pred)-Dx(target)|)+mean(|Dy(pred)-Dy(target)|)`를 사용합니다. 각 축 전체의 평균을 더하며 2로 나누지 않습니다. E2 AMI/RFN에서는 raw 전체 출력에 적용하고 observed/missing mask를 사용하지 않습니다.
- E3는 assembled 3D grayscale에서 `sigmoid((threshold-I_final)/temperature)`를 만들어 aligned binary pore target과 비교합니다. `topology=0.5 softDice + 0.5 soft-clDice`. Observed 단면은 상수로 조립하고 그 위치로 gradient가 흐르지 않게 합니다.
- Soft-clDice는 connectivity surrogate이며 완전한 위상 보장이 아닙니다. Hard connected-component/물성 solver는 학습 graph에 넣지 않습니다.

## 5. 학습과 모델 선택

- E1 best: `[0,1]` clipping 후 validation bracketed missing-only MAE. Raw train/validation L1은 별도 기록합니다.
- E2 선택 기준: E1과 같은 clipping 후 validation bracketed missing-only primary MAE. Train/validation total, L1, gradient loss를 별도 기록합니다. E3 예정 선택 기준: validation total E3 loss. Test는 선택·tuning에 사용하지 않습니다.
- 기존 실행값: Adam, lr `1e-4`, weight decay 0, gradient clipping norm 1.0. E1 초기 GPU pilot은 target 모델, batch 1, 1 epoch, train/validation 각각 1-step, AMP off, workers 0이었습니다. E2 별도 GPU pilot은 batch 8, train/validation 각각 2-step입니다.
- E1 target 템플릿의 batch/epochs는 `null`이며, E1 실제 실행과 E2 설정은 별도 확정 budget을 사용합니다. AMP/workers/batch를 변경하면 다시 probe합니다. Full validation과 pilot은 구분합니다.
- GPU 자원 문제를 해결하려고 crop·factor·모델·split을 임의로 바꾸지 않습니다. 배치/AMP/chunking을 먼저 검토합니다. Accumulation/checkpointing은 현재 실행 옵션이라고 가정하지 않습니다.
- AMI 실제 sample 수는 train `48×512=24,576`, validation `9×512=4,608`. RFN train은 모든 256 axial phase, validation은 crop당 primary 189 phase입니다.

## 6. 평가와 기록

최종 clipping·observed reinsertion 후 등록 ROI를 사용합니다. Grayscale은 primary mask, topology/physics는 동일한 frozen segmentation의 3D pore mask에서 비교합니다.

| 범주 | 예정 지표 |
|---|---|
| 영상 | MAE, PSNR(range 1), SSIM, gradient MAE, phase 1/2/3별 오차, tail diagnostic, 추론 시간·peak GPU memory |
| 구조·위상 | Dice/IoU, porosity, specific surface area, pore-size distribution, Euler characteristic density, component 통계, largest connected pore ratio, Z-percolation/failure |
| 물성 | Z-tortuosity, single-phase Z-permeability, permeability log error, percolation failure |

동일한 spacing·solver/version·inlet/outlet·boundary condition·convergence를 적용합니다. Non-percolating prediction에 임의의 작은 permeability를 대입하지 않고 failure로 보고합니다. Segmentation operator/threshold는 train/validation에서만 정합니다. 제공된 binary reference와 grayscale의 threshold-derived mask를 혼동하지 않습니다.

동일 held-out block의 paired comparison을 사용하고 seed별·block별 값과 평균/분산을 보존합니다. Slice/voxel이나 같은 core의 block을 독립 암석 표본으로 세지 않습니다.

Run/prediction/metric에는 experiment/config/Git commit, seed, source SHA/DOI/core, split/normalization version, crop 좌표·shape, XYZ 해석·spacing, factor/offset, checkpoint epoch, loss/weights를 연결합니다. E3·평가 구현 시 soft threshold/temperature/iterations, segmentation/topology version, solver/version도 기록합니다. 현재 E1의 metadata 저장이 미래 평가 필드까지 구현했다는 뜻은 아닙니다.

## 7. 남은 구현·결정

- GPU target pilot, 한 batch overfit, 본학습 budget 및 서버 제출 규칙 확인.
- E2 GPU pilot 후 AMI 10 epoch → E2 전용 cache → RFN 10 epoch 실행·검증.
- Binary 실제 alignment/polarity 검증; E3 soft threshold/temperature/iterations 결정, soft Dice/clDice backward와 connected/disconnected/constant phantom 검사, RFN 3D fine-tuning 구현.
- SSIM 구현/window, hard segmentation, topology 지표, tortuosity/permeability solver·조건 및 empty-phase 정책 동결·구현.
- Seeds 1/2/3 본학습, frozen inference와 최종 test evaluation.

E0/E1 코드 경로와 E2 L1+gradient loop는 구현되어 있습니다. E2 GPU 학습과 E3 loop 및 과학적 평가 함수는 아직 수행·구현되지 않았습니다. CPU/QC 성공은 GPU fit·convergence·복원 성능의 증거가 아닙니다. PSNR 향상만으로 permeability 향상이나 실제 미관측 구조의 유일한 정답을 주장하지 않습니다.
