# 모델 팀 전처리 인계

## 완료 상태

- 11개 parent core의 filtered grayscale/binary RAW 감사가 완료되었다.
- Split은 train 8개, validation은 Berea, test는 Bentheimer·Parker로 동결되었다.
- 모든 core의 source binary polarity는 `source_pore_value=0`이다. Loader는 이를 한 번 변환하여 내부 `pore=1` (`True`), `solid=0`으로 사용한다.
- 고정 validation/test crop 12개 전처리 QC는 12/12 통과했고, unit test는 15개 통과했다.

## 고정 데이터 계약

- 입력과 grayscale reconstruction target은 `*_grayscale_filtered.raw`뿐이다. Binary는 grayscale 모델의 입력이나 reconstruction target이 아니다.
- NumPy volume 축은 `[X,Y,Z]`, PyTorch batch 축은 `[B,C,X,Y,Z]`이다.
- source는 `uint8`, 모델용 grayscale은 `float32 [0,1]`이다. train 8개만 pooling한 histogram의 `p0.5=2`, `p99.5=206`으로 clip한 뒤 `(x - 2) / (206 - 2)`를 적용한다. validation/test 통계는 이 값 산정에 사용되지 않았다.
- 모든 model crop은 `[256,256,256]`이다. Z축 x4, offset 0이므로 observed index는 `0,4,...,252` (64장), sparse input은 `[256,256,64]`이다.
- `primary_metric_mask`는 양쪽 observed slice가 있는 missing z=1..251의 189장만 포함한다. z=253,254,255은 boundary diagnostic이며 headline image metric에서 제외한다.
- Binary pore mask는 E3 topology loss와 구조·물성 평가 reference에만 사용한다.

## 누수 방지 규칙

- Split 단위는 crop이 아니라 parent core다. 동일 core crop을 다른 split에 넣지 않는다.
- normalization, threshold, checkpoint 및 loss weight 선택에 test core를 사용하지 않는다.
- validation/test는 `eval_crop_manifest.json`의 origin, shape, sampling factor와 offset을 그대로 사용한다. seed에 따라 좌표나 offset을 변경하지 않는다.

## 구현 API와 DataLoader 연결

`src/theme26/data/dataset.py`는 PyTorch `Dataset`이나 dict 반환 API를 제공하지 않는다. 실제 반환값은 다음의 `PreprocessedCrop` dataclass다.

| Field | NumPy shape / dtype | 용도 |
|---|---|---|
| `core_id`, `split`, `crop` | metadata | provenance |
| `grayscale` | `[X,Y,Z]`, `float32 [0,1]` | dense grayscale target / observed source |
| `pore_mask` | `[X,Y,Z]`, `bool` | internal pore mask (`True`=pore) |
| `sparse.sparse_volume` | `[X,Y,64]`, `float32 [0,1]` | x4 sparse model input |
| `sparse.observed_mask`, `missing_mask`, `primary_metric_mask` | `[X,Y,Z]`, `bool` | loss/evaluation masks |
| `sparse.observed_indices` | `(0,4,...,252)` | sparse-to-HR alignment |
| `e0_linear` | `[X,Y,Z]`, `float32 [0,1]` | deterministic E0 baseline |

PyTorch adapter는 각 array 앞에 channel과 batch 차원을 추가한다. 예를 들어 dense target/mask는 `[B,1,256,256,256]`, sparse input은 `[B,1,256,256,64]`이다. Batch dict가 필요하면 모델 팀 adapter에서 위 field 이름을 그대로 key로 사용한다.

### Train: on-demand random crop

`TrainRandomCropDataset`은 `split == "train"`인 `CoreSource`만 받아 train 8개 내부에서 core와 aligned origin을 random하게 선택한다. 원본 RAW는 매 sample에서 read-only memory-map으로 열며, 대량 crop NPZ는 만들지 않는다.

```python
import json
from pathlib import Path
import numpy as np
import torch

from theme26.data.dataset import CoreSource, TrainRandomCropDataset

data_root = Path("../data/11_sandstones")  # 프로젝트 배치에 맞는 상대경로
splits = json.loads(Path("config/splits.json").read_text())
normalization = json.loads(Path("config/normalization.json").read_text())

def make_source(split, entry):
    core_dir = data_root / entry["source_core_id"]
    return CoreSource(
        core_id=entry["source_core_id"], split=split,
        grayscale_path=core_dir / entry["grayscale_filtered_filename"],
        binary_path=core_dir / entry["binary_filename"],
        source_pore_value=entry["source_pore_value"],
    )

train_sources = [make_source("train", item) for item in splits["splits"]["train"]]
train_dataset = TrainRandomCropDataset(
    train_sources,
    normalization_lower=normalization["lower"],
    normalization_upper=normalization["upper"],
)
sample = train_dataset.sample(np.random.default_rng(0))

target = torch.from_numpy(sample.grayscale)[None, None]       # [1,1,256,256,256]
sparse = torch.from_numpy(sample.sparse.sparse_volume)[None, None]  # [1,1,256,256,64]
pore = torch.from_numpy(sample.pore_mask)[None, None]        # [1,1,256,256,256], bool
primary_mask = torch.from_numpy(sample.sparse.primary_metric_mask)[None, None]
```

### Validation/test: manifest fixed crop

`preprocess_crop(source, crop=..., normalization_lower=..., normalization_upper=..., sampling=...)`으로 `eval_crop_manifest.json`의 각 entry를 생성한다. `CoreSource`는 위 `make_source`로 구성하고, `CropSpec(tuple(entry["origin_xyz"]), tuple(entry["shape_xyz"]))`, `SamplingSpec(entry["sampling_factor"], entry["sampling_offset"])`를 그대로 전달한다. Manifest에는 validation Berea 4개와 test Bentheimer/Parker 각 4개가 있다.

```python
from theme26.contracts import CropSpec, SamplingSpec
from theme26.data.dataset import preprocess_crop

manifest = json.loads(Path("config/eval_crop_manifest.json").read_text())
entry = manifest["crops"][0]
split, source_entry = "validation", splits["splits"]["validation"][0]
fixed = preprocess_crop(
    make_source(split, source_entry),
    crop=CropSpec(tuple(entry["origin_xyz"]), tuple(entry["shape_xyz"])),
    normalization_lower=normalization["lower"],
    normalization_upper=normalization["upper"],
    sampling=SamplingSpec(entry["sampling_factor"], entry["sampling_offset"]),
)
```

### Final-volume data consistency

모델의 raw HR prediction은 observed 위치까지 포함할 수 있지만, 최종 3D volume에서는 prediction으로 observed slice를 덮어쓰면 안 된다. 제공 함수 `compose_full_volume(prediction, sample.sparse)`는 prediction을 복사한 뒤 `sample.sparse.sparse_volume`의 원본 observed slice를 HR grid에 다시 삽입한다. PyTorch에서 같은 조립을 구현할 때도 observed 위치는 입력 grayscale 상수로 교체한다.

## 파일 위치

| 목적 | 프로젝트 기준 경로 |
|---|---|
| 동결 split/source filename/polarity | `config/splits.json` |
| train-only normalization | `config/normalization.json` |
| fixed validation/test crops | `config/eval_crop_manifest.json` |
| memory-map crop 생성 API | `src/theme26/data/dataset.py` |
| RAW audit API | `src/theme26/data/audit.py` |
| final fixed-crop QC | `outputs/final_preprocessing_qc.md` |
| QC PNG | `outputs/final_preprocessing_qc/` |
| 권장 data root | `../data/11_sandstones/` |

## 학습 시작 전 체크리스트

- [ ] `splits.json`의 core 목록과 `normalization.json`의 ID를 실험 config에 기록했다.
- [ ] filtered grayscale만 reconstruction input/target으로 연결했다.
- [ ] normalization lower/upper를 `2/206`으로 모든 split에 동일하게 적용했다.
- [ ] NumPy/PyTorch 축 순서와 sparse input의 Z depth 64를 확인했다.
- [ ] validation/test는 manifest의 12개 crop·offset 0을 그대로 사용한다.
- [ ] image headline metric은 `primary_metric_mask`의 189 slice만 사용한다.
- [ ] topology/evaluation에서만 `pore_mask`를 사용하고 internal `True=1` convention을 유지한다.
- [ ] full-volume 평가 전 observed slice를 원본 grayscale로 재삽입한다.

## 책임 경계

전처리 담당은 RAW audit, split/normalization/crop manifest 동결, memory-map crop 생성과 QC를 책임진다. 모델 담당은 `PreprocessedCrop`을 PyTorch batch로 collate하고, SAINT tensor flow·loss·checkpoint 선택·inference 및 observed-slice 재삽입 뒤 평가 연결을 책임진다. Split, normalization, 고정 평가 좌표를 모델 코드에서 변경하지 않는다.
