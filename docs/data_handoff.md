# Berea 공간 분할 전처리 인계

## 실험 규칙

주 실험은 Berea 한 core의 비중첩 공간 구역 분할이다. 다른 core는 필요하지 않다.
Train X=[0,400), validation X=[432,688), test X=[720,976).
Y/Z는 [0,1000), 구역 사이 32-voxel buffer는 사용하지 않는다.
Crop이 구역 밖으로 넘어가면 preprocessing API가 오류를 낸다.
Validation/test 고정 crop은 각 9개이며 서로 겹치지 않는다.
구역 비율이나 경계는 성능 결과를 보기 전에 결정하고, 변경 시 설정을 다시 동결한다.

## 정규화와 출처

Train 구역만의 histogram으로 p0.5/p99.5를 계산한다. 현재 Berea에서는 lower=2, upper=184이다.
모든 split에 clip 후 `(x-lower)/(upper-lower)`를 동일 적용한다.
원본 0/1 binary는 source_pore_value=0을 내부 True=pore로 변환한다.
단일 상 crop도 전처리상 유효하다. 물성 계산의 empty-phase 정책은 평가 구현에서 따로 정의한다.
Config의 voxel spacing을 sample에 보존하며 normalization ID와 region ID도 전달한다.

## 모델 연결 예시

```python
from pathlib import Path
import numpy as np
from theme26.data.protocol import load_protocol
from theme26.data.dataset import source_from_entry, TrainRandomCropDataset

config, norm, manifest = load_protocol(Path('config'))
data_root = Path('/path/to/Berea')
sources = [source_from_entry(data_root, config, 'train', entry, norm['normalization_id'])
           for entry in config['splits']['train']]
sampler = TrainRandomCropDataset(sources, normalization_lower=norm['lower'],
                                 normalization_upper=norm['upper'])
sample = sampler.sample(np.random.default_rng(0))
```

`sample.grayscale`: float32 `[256,256,256]`, dense target。
`sample.pore_mask`: bool `[256,256,256]`, E3/평가 reference。
`sample.sparse.sparse_volume`: float32 `[256,256,64]`, sparse input。
`sample.sparse`에 observed/missing/primary masks 및 관측 좌표가 포함된다.
`sample.e0_linear`: dense deterministic E0。
`sample.region_id`, `normalization_id`, `voxel_spacing_um_xyz`를 실험 기록에 연결한다.

PyTorch adapter는 각 sample에 channel을 추가한 후 batch를 묶어 `[B,1,X,Y,Z]`를 만든다.
이 클래스는 PyTorch Dataset이 아니며 `__len__`/`__getitem__`을 제공하지 않는다.
Worker마다 독립된 RNG를 만들고 seed/epoch 재현 정책을 adapter에 기록한다.

Validation/test는 manifest entry와 동일 split/region의 `source_from_entry`를 사용하여
`preprocess_crop(..., crop=CropSpec(...), sampling=SamplingSpec(...))`을 호출한다.
원본 observed slice는 final volume에 `compose_full_volume`으로 다시 삽입한다.

## 검증 단계

1. `tests/`: 작은 합성 배열로 코드 검사. Berea를 읽지 않는다.
2. Source audit: 실제 Berea 파일 크기, labels, 밝기 기반 polarity·alignment 진단.
3. Fixed crop QC: 실제 Berea 18개 crop의 shape/dtype/range/masks/E0 검사.
4. Model 성능: 모델 구현 후 validation checkpoint 선택과 최종 test 지표 계산.

2/3의 통과가 4의 성공을 뜻하지 않는다. 11-core 감사/12-crop QC의 과거 기록은
현재 Berea 공간 분할의 검증 결과로 재사용하지 않는다.

RAW의 직접 XYZ 해석과 train dense 구역 확보 비용은 연구 보고에 명시한다.
같은 core의 공간 구역 평가이므로 새로운 core에 대한 일반화 성능을 주장하지 않는다.

실제 E1 AMI 연결은 `training.ManifestAmiDataset`이 `configs/data/berea_crops.yaml`의 고정 crop을 읽는다.
범용 `TrainRandomCropDataset`은 구역 내 임의 crop 생성 API이며 frozen 48개 학습 경로와 구분한다.
