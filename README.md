# SAINT for Grayscale Digital Rock Reconstruction

Sparse한 3D sandstone Micro-CT volume에서 누락된 z축 단면을 복원하고, 영상 품질뿐 아니라 pore topology와 물성이 얼마나 보존되는지 평가하는 연구 프로젝트입니다.

원 논문의 SAINT(Spatially Aware Interpolation NeTwork) 구조를 grayscale digital rock에 맞게 재구현합니다. 최종 모델 출력은 dense grayscale volume이며, binary pore mask는 topology-aware fine-tuning과 구조·물성 평가에만 사용합니다.

## 연구 질문

> x4 sparse-z sandstone volume에서 SAINT 기반 interpolation이 linear interpolation보다 누락 단면을 잘 복원하며, gradient 및 topology loss가 pore boundary, connectivity와 permeability 보존을 개선하는가?

## 고정 실험 범위

| 항목 | 설정 |
|---|---|
| Dataset | 11 Sandstones filtered grayscale |
| Source volume | `1000 x 1000 x 1000` |
| Spatial crop | `256 x 256 x 256` |
| Axis convention | `[X, Y, Z]` |
| Sparse axis | Z |
| Sampling factor | x4 |
| Model output | Dense grayscale volume |
| Internal pore label | `pore=1`, `solid=0` |
| Learned-model seeds | `1, 2, 3` |
| Physical evaluation | z-direction permeability |

현재 범위에는 x2/x6, raw-grayscale robustness, GAN, diffusion, architecture search, persistent-homology loss와 permeability loss를 포함하지 않습니다.

## 데이터

- Dataset: [11 Sandstones: Raw, Filtered and Segmented Data](https://digitalporousmedia.org/datasets/317/)
- DOI: [10.17612/F4H1-W124](https://doi.org/10.17612/F4H1-W124)
- License: ODC-By 1.0
- Voxel spacing: `2.25 um` isotropic
- Grayscale input/target: `*_grayscale_filtered.raw`
- Auxiliary topology target: aligned binary volume

원본 데이터, crop, checkpoint와 전체 prediction은 Git에 저장하지 않습니다. 서버에서는 다음처럼 code, data와 run artifact를 분리합니다.

```text
/project/<group>/theme26/
/project/<group>/data/11_sandstones/
/project/<group>/runs/theme26/
```

### Berea 전처리

`scripts/preprocess_berea.py`는 두 Berea RAW 파일의 byte size/dtype을 검증하고, canonical `[X,Y,Z]` `uint8` volume을 읽어 정규화, aligned `256³` crop, binary pore mask 및 x4 sparse-z/E0 artifact를 만듭니다. `--normalization-lower/--normalization-upper`는 반드시 training core에서 동결한 값이고, `--source-pore-value`는 dataset metadata로 확인한 값이어야 합니다.

```bash
PYTHONPATH=src python scripts/preprocess_berea.py \
  --data-dir /path/to/11_sandstones/Berea \
  --output /path/to/artifacts/berea_crop.npz \
  --crop-origin 372 372 372 \
  --normalization-lower <frozen_train_lower> \
  --normalization-upper <frozen_train_upper> \
  --source-pore-value <0_or_1> \
  --offset 0
```

Artifact에는 normalized grayscale/pore crop, `[256,256,64]` sparse input, HR-grid masks, observed z-indices 및 linear E0 volume이 포함됩니다. 기본 protocol (`offset=0`)에서 observed slice는 64장, headline `primary_metric_mask`는 189장의 bracketed missing slice만 포함합니다.

## 문제 설정

Canonical dense volume은 다음 형식을 사용합니다.

```text
NumPy:   [X, Y, Z]
PyTorch: [B, C, X, Y, Z]
dtype:   float32
range:   [0, 1]
```

Z축에서 네 장마다 한 장을 관측합니다.

```python
I_LR = I_HR[:, :, offset::4]
```

`256` 깊이와 offset 0에서는 64장의 observed slice가 생성됩니다. Primary 영상 평가는 양쪽에 관측 단면이 존재하는 189장의 bracketed missing slice에서 수행하며, 마지막 `z=253,254,255`는 boundary diagnostic으로만 기록합니다.

## 모델

SAINT는 sparse volume을 sagittal 및 coronal 방향의 2D marginal slice로 처리합니다.

```text
Sparse grayscale volume
        |
        +--> sagittal slices --+
        |                       +--> shared AMI --> sagittal volume --+
        +--> coronal slices ----+--> shared AMI --> coronal volume ---+--> RFN --> dense grayscale volume
```

- AMI: feature learning, Filter Distance Matrix, dynamic filter generation, anisotropic sub-pixel reconstruction
- RFN: sagittal/coronal reconstruction의 평균과 residual correction을 이용한 fusion
- AMI 입력의 3채널: RGB가 아니라 인접한 grayscale marginal slice 세 장

## 실험

| ID | 방법 | Loss |
|---|---|---|
| E0 | Linear z interpolation | 없음 |
| E1 | SAINT grayscale baseline | L1 |
| E2 | Gradient ablation | L1 + `0.1 * gradient` |
| E3 | Topology-aware RFN fine-tuning | L1 + `0.1 * gradient` + `0.1 * topology` |

E3는 동일 seed의 E2 best checkpoint에서 시작합니다. AMI를 고정하고 RFN을 fine-tuning하며, topology loss는 assembled 3D grayscale volume을 soft pore probability로 변환해 계산합니다.

```text
L_topology = 0.5 * L_softDice + 0.5 * L_soft-clDice
```

Soft threshold, temperature와 soft-skeletonization iteration은 training/validation에서 정한 뒤 config에 고정합니다. Permeability는 training loss가 아니라 frozen prediction의 평가 지표입니다.

## 평가

### Grayscale

- MAE
- PSNR
- SSIM
- Gradient MAE
- Missing slice의 상대 위치별 오차

### Structure and topology

- Dice/IoU
- Porosity error
- Specific surface area error
- Pore-size distribution distance
- Euler characteristic density error
- Connected-component statistics
- Largest connected pore ratio
- z-percolation

### Physics

- z-direction tortuosity
- z-direction permeability error
- Percolation failure

모든 모델에 동일한 frozen segmentation 및 solver 설정을 적용합니다. Test data로 threshold, checkpoint 또는 loss weight를 선택하지 않습니다.

## 저장소 구조

```text
src/theme26/
├─ contracts.py
├─ data/
│  ├─ raw_volume.py
│  ├─ crop_sampler.py
│  └─ sparse_sampling.py
├─ models/saint/
│  ├─ fdm.py
│  ├─ dynamic_filter.py
│  ├─ ami.py
│  ├─ rfn.py
│  └─ model.py
├─ losses/
│  ├─ reconstruction.py
│  ├─ gradient.py
│  └─ topology.py
└─ evaluation/
   ├─ image_metrics.py
   ├─ topology_metrics.py
   └─ physics_metrics.py

scripts/
├─ train.py
└─ infer.py

tests/
└─ test_contracts.py
```

현재 Python 파일은 팀 분업을 위한 interface skeleton입니다. 실제 알고리즘은 각 모듈의 함수 signature와 데이터 계약을 유지하면서 구현합니다.

## 문서

- [Research protocol](docs/research_protocol.md): 데이터, 학습, 평가 규칙
- [Interface contract](docs/interface_contract.md): 모듈별 입력·출력 shape, dtype과 공개 API
- [Storage and Slurm](docs/operations/storage-and-slurm.md): 서버 저장 구조와 GPU 실행 예시
- [Contributing](CONTRIBUTING.md): branch, PR과 review 규칙

## 참고자료

- Peng et al., *SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis*, CVPR 2020. [DOI](https://doi.org/10.1109/CVPR42600.2020.00777)
- [Official SAINT repository](https://github.com/cpeng93/SAINT)
- Shit et al., *clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
