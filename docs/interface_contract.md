# Interface Contract

이 문서는 데이터, 모델, loss, 평가 모듈을 서로 독립적으로 개발하기 위한 코드 계약이다. 구현자는 내부 알고리즘을 자유롭게 작성할 수 있지만, 이 문서의 axis, shape, dtype, 값 범위와 함수 입출력은 임의로 변경하지 않는다.

계약을 바꿔야 하면 관련 구현과 테스트를 작성하기 전에 팀에 공유하고, [Research Protocol](research_protocol.md)과 같은 PR에서 갱신한다.

## 1. 담당 영역

| 담당 | 소유 경로 | 책임 |
|---|---|---|
| 모델·통합·GPU 담당 | `src/theme26/models/saint/`, `src/theme26/training/`, `scripts/train.py`, `scripts/infer.py` | SAINT, 학습, 추론, GPU 실행, 최종 병합 |
| 데이터 담당 | `src/theme26/data/` | raw loading, normalization, aligned crop, x4 sampling, E0 |
| loss 담당 | `src/theme26/losses/` | L1, gradient, soft Dice, 3D soft-clDice |
| 평가 담당 | `src/theme26/evaluation/` | image, topology, physics metrics와 CSV export |

공통 계약은 `src/theme26/contracts.py`에 둔다. 담당 경로 밖의 공개 함수나 공통 계약을 바꾸는 PR은 모델·통합 담당자의 review가 필요하다.

## 2. 공통 데이터 규칙

### 2.1 Axis

연구 프로토콜과 동일하게 의미론적 volume axis는 `XYZ`로 고정한다.

```text
NumPy volume:       [X, Y, Z]
PyTorch volume:     [B, C, X, Y, Z]
Sparse axis:        Z (axis index 2 in a volume, index 4 in a batched tensor)
Sagittal plane:     fixed X -> [Y, Z]
Coronal plane:      fixed Y -> [X, Z]
Axial plane:        fixed Z -> [X, Y]
```

PyTorch `Conv3d`가 차원을 `[D,H,W]`라고 부르더라도 이 프로젝트에서는 그 세 공간 차원의 의미를 `[X,Y,Z]`로 해석한다. RAW는 C-order XYZ 배열로 직접 읽고 축을 transpose하지 않는다.

### 2.2 Shape

```text
Dense crop:                    [256, 256, 256]
Dense batch:             [B, 1, 256, 256, 256]
Compressed x4 sparse:           [256, 256, 64]
Sparse batch:              [B, 1, 256, 256, 64]
Preprocessing masks:             [256, 256, 256]
Training batch masks:      [B, 1, 256, 256, 256]
Binary target:             [B, 1, 256, 256, 256]

Sagittal AMI input:        [B, 3, Y, Z_lr]
Coronal AMI input:         [B, 3, X, Z_lr]
AMI output:                [B, 1, L, Z_hr]
FDM:                       [1 or B, R, K, K]
Directional dense volume: [B, 1, X, Y, Z_hr]
Final dense volume:        [B, 1, X, Y, Z_hr]
```

`L`은 sagittal이면 `Y`, coronal이면 `X`이다. AMI의 3채널은 RGB가 아니라 현재 marginal slice와 양옆의 인접 marginal slice이다.

`SparseSample`과 evaluation 함수는 NumPy 단일 volume이므로 mask가 `[X,Y,Z]`이다. DataLoader가 batch를 만들 때 channel과 batch 차원을 추가하여 Torch `[B,1,X,Y,Z]`로 변환한다.

### 2.3 Dtype와 값 범위

| 데이터 | dtype | 범위/의미 |
|---|---|---|
| 원본 filtered grayscale | `uint8` | `[0,255]` |
| 정규화 grayscale | `float32` | `[0,1]` |
| 모델 입력·출력 | `torch.float32` | `[0,1]` |
| 원본 binary | dataset metadata를 따름 | loader에서 polarity 확인 |
| 내부 binary | `bool` 또는 `uint8` | `pore=1`, `solid=0` |
| observed/missing mask | `bool` | 포함 위치가 `True` |

정규화의 lower/upper 값은 train 공간 구역만으로 추정해 version과 함께 동결한다. 개별 test core에 맞춘 normalization은 금지한다.

### 2.4 Physical spacing

Spacing tuple의 순서는 항상 `(sx, sy, sz)`이고 단위는 `um`이다. 코드에 `2.25`를 직접 하드코딩하지 않고 manifest 또는 config에서 읽는다.

```text
HR spacing: (sx, sy, sz)
x4 LR spacing: (sx, sy, 4*sz)
```

## 3. Sampling과 full-volume 구성

Primary sampling은 다음으로 고정한다.

```text
factor = 4
axis = Z
observed index = offset + factor*k
validation/test offset = manifest에서 고정
```

`SparseSample`은 compressed sparse volume, HR-grid observed/missing mask, primary metric mask와 observed index를 함께 반환한다. `primary_metric_mask`는 양쪽 관측 단면 사이에 있는 bracketed missing 위치만 `True`이며, mask를 함수마다 다시 만들지 않는다.

Learned model은 HR z-grid 전체를 예측한다. 구조·물성 평가용 full volume은 예측에 관측 slice를 다시 삽입해 만든다.

```python
full = prediction.copy()
full[..., observed_indices] = sparse_volume
```

Primary image metric은 `missing_mask` 전체가 아니라 `primary_metric_mask`에서 계산한다. 전체-volume 또는 모든 missing 위치의 metric을 boundary diagnostic으로 저장할 수 있지만 headline 결과로 사용하지 않는다.

E0의 마지막 관측 slice 뒤에 남는 위치는 full-volume shape를 완성하기 위해 `boundary_mode="replicate"`로 채울 수 있다. 그러나 양쪽 관측 단면이 없는 위치는 primary image metric에서 제외한다. `256` 깊이, factor 4, offset 0의 primary mask는 189개 bracketed missing slice를 포함한다.

## 4. 공개 API

아래 signature가 모듈 사이의 계약이다. skeleton의 `NotImplementedError`를 담당자가 구현으로 교체한다.

### 4.1 Data

```python
read_raw_volume(path, *, metadata) -> np.ndarray  # [X,Y,Z], source dtype

normalize_grayscale(volume, *, lower, upper) -> np.ndarray
# [X,Y,Z], float32 [0,1]

to_pore_mask(binary, *, source_pore_value) -> np.ndarray
# [X,Y,Z], bool, pore=True

extract_aligned_crop(grayscale, binary, *, crop) -> tuple[np.ndarray, np.ndarray]
# 두 출력 모두 [crop_x,crop_y,crop_z]

make_sparse_sample(dense, *, sampling) -> SparseSample
# compressed [X,Y,Z_lr] + HR-grid masks/indices

linear_interpolate_z(sample) -> np.ndarray
# [X,Y,Z_hr], float32

compose_full_volume(prediction, sample) -> np.ndarray
# observed slice를 원본 sparse input으로 교체한 [X,Y,Z_hr]
```

### 4.2 SAINT model

```python
build_filter_distance_matrix(...) -> torch.Tensor
# [1,R,K,K], float32, physical distance in um

DynamicFilterGenerator(fdm) -> filters
# fdm [B,R,K,K] -> [B,R,C,K,K]

AnisotropicMetaInterpolator(marginal, fdm) -> dense_marginal
# [B,3,L,Z_lr] -> [B,1,L,R*Z_lr]

ResidualFusionNetwork(sagittal, coronal) -> fused
# two [B,1,X,Y,Z] -> [B,1,X,Y,Z]

SaintReconstructor(sparse_volume, fdm) -> ReconstructionOutput
# [B,1,X,Y,Z_lr] -> final/sagittal/coronal [B,1,X,Y,Z_hr]
```

모델의 raw prediction과 관측 slice를 재삽입한 full volume을 혼동하지 않는다. `ReconstructionOutput.prediction`은 raw prediction이고 data consistency는 별도 함수에서 수행한다.

### 4.3 Loss

모든 loss는 역전파 가능한 scalar `torch.Tensor`를 반환한다.

```python
masked_l1_loss(prediction, target, *, mask=None) -> torch.Tensor

gradient_loss_2d(prediction, target, *, mask=None) -> torch.Tensor

gradient_loss_3d(prediction, target, *, mask=None) -> torch.Tensor

soft_pore_probability(grayscale, *, threshold, temperature) -> torch.Tensor

soft_dice_loss(probability, target, *, eps=1e-6) -> torch.Tensor

soft_cldice_loss_3d(probability, target, *, iterations, eps=1e-6) -> torch.Tensor

topology_loss(probability, target, *, iterations,
              dice_weight=0.5, cldice_weight=0.5) -> torch.Tensor
```

E1/E2 학습 loss의 mask 범위는 upstream stage별 동작을 audit한 뒤 config와 Research Protocol에 동결한다. 평가가 bracketed missing-only라는 이유만으로 학습 loss까지 missing-only로 가정하지 않는다. E3 topology는 observed slice를 입력값으로 교체한 assembled 3D soft pore volume에 적용하며, observed slice를 통해 gradient가 흐르면 안 된다.

### 4.4 Evaluation

평가 함수는 disk I/O와 분리하고, 기본적으로 CPU NumPy array를 받는다.

```python
compute_image_metrics(prediction, target, *, mask, data_range=1.0) -> dict[str, float]

compute_topology_metrics(predicted_pore, target_pore, *, spacing_um_xyz) -> dict[str, float | bool]

compute_physics_metrics(predicted_pore, target_pore, *, spacing_um_xyz,
                        solver) -> dict[str, float | bool | str]
```

입력은 `[X,Y,Z]`이며 grayscale은 `float32 [0,1]`, binary는 `bool`이고 `pore=True`이다. 반환 dictionary의 key는 snake_case이며 JSON/CSV로 직렬화할 수 있는 scalar만 포함한다.

## 5. Validation과 오류 처리

공개 함수는 다음 조건을 조용히 보정하지 말고 `ValueError` 또는 `TypeError`로 거부한다.

- 잘못된 axis 순서 또는 rank
- grayscale 범위가 `[0,1]` 밖인 경우
- binary에 두 label 이외의 값이 있는 경우
- grayscale/binary/mask shape 불일치
- `factor < 2`, 유효 범위 밖 offset 또는 crop
- 음수 또는 0인 voxel spacing
- NaN/Inf
- empty mask에서 metric 계산
- topology/physics 계산에서 empty pore 또는 empty solid

## 6. 아직 config로 동결할 값

다음 항목은 API를 바꾸지 않고 `configs/`의 YAML로 동결한다.

- Berea train/validation/test 공간 범위 및 buffer
- training-only normalization lower/upper와 `normalization_id`
- validation/test crop 좌표와 sampling offset
- soft threshold, temperature와 topology protocol version
- E1/E2 stage별 L1/gradient loss region
- soft-clDice skeletonization iterations
- SSIM implementation/window
- permeability solver와 boundary condition

Test 결과를 본 뒤 위 값을 변경하지 않는다.

## 7. 담당자별 완료 조건

각 담당자는 다음을 PR에 포함한다.

- 자신의 공개 함수 구현
- 정상 입력 unit test
- shape/dtype/range 오류 unit test
- synthetic small volume을 사용한 CPU smoke test
- 입력/출력 예시와 실행 명령
- 개인 절대 경로 및 대용량 데이터가 없다는 확인

GPU 담당자는 병합 후 실제 데이터 one-batch overfit 및 full-volume smoke test를 실행한다. CPU unit test 통과와 GPU 전체 학습 성공은 서로 다른 acceptance gate이다.


## 8. Berea spatial protocol extension

`configs/data/berea_split.yaml`은 동일 Berea core의 서로 겹치지 않는 region_origin_xyz/region_shape_xyz를
각 split에 기록한다. 좌표 끝은 제외한다. `CoreSource.region`은 crop 허용 범위이며
`TrainRandomCropDataset`은 명시적인 train region 없이는 사용할 수 없다.
`preprocess_crop`은 ROI 밖 crop을 거부한다. 기본 crop/SamplingSpec 공개 API는 유지한다.
정규화는 train 구역만으로 계산하고 `normalization_id`, region 좌표와 voxel count를 기록한다.
sample에 `region_id`, `normalization_id`, `voxel_spacing_um_xyz`를 보존한다.
단일 상 crop은 binary label의 허용 집합 {0,1} 내에 있으면 전처리한다.
Source audit와 empty-phase 물성 평가 정책은 별개이다.

### RAW interpretation

`VolumeMetadata.storage_order="XYZ"`만 허용하며 C-order로 직접 읽는다.
전처리와 학습이 같은 좌표를 사용하고 축 transpose는 하지 않는다.
`configs/data/` YAML이 구역, 정규화, train/validation/test crop 좌표의 유일한 기준이다.
