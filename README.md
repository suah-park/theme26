# SAINT for Grayscale Digital Rock Reconstruction

Sparse한 3D sandstone Micro-CT volume에서 누락된 z축 단면을 복원하고, 영상 품질뿐 아니라 pore topology와 물성이 얼마나 보존되는지 평가하는 연구 프로젝트입니다.

원 논문의 SAINT(Spatially Aware Interpolation NeTwork) 구조를 grayscale digital rock에 맞게 재구현합니다. 최종 모델 출력은 dense grayscale volume이며, binary pore mask는 topology-aware fine-tuning과 구조·물성 평가에만 사용합니다.

## 연구 질문

> x4 sparse-z sandstone volume에서 SAINT 기반 interpolation이 linear interpolation보다 누락 단면을 잘 복원하며, gradient 및 topology loss가 pore boundary, connectivity와 permeability 보존을 개선하는가?

## 고정 실험 범위

| 항목 | 설정 |
|---|---|
| Dataset | Berea sandstone filtered grayscale + aligned binary |
| Source volume | `1000 x 1000 x 1000` |
| Spatial crop | `256 x 256 x 256` |
| Split | One-volume spatial holdout with 32-voxel buffers |
| Axis convention | `[X, Y, Z]` |
| Sparse axis | Z |
| Sampling factor | x4 |
| Model output | Dense grayscale volume |
| Internal pore label | `pore=1`, `solid=0` |
| Learned-model seeds | `1, 2, 3` |
| Physical evaluation | z-direction permeability |

현재 범위에는 x2/x6, raw-grayscale robustness, GAN, diffusion, architecture search, persistent-homology loss와 permeability loss를 포함하지 않습니다.

## 데이터

- Source dataset: [11 Sandstones: Raw, Filtered and Segmented Data](https://digitalporousmedia.org/datasets/317/)
- Selected sample: Berea sandstone only
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

### Berea spatial split

단일 `1000³` Berea volume을 X축 기준으로 분리하며 crop이 split 경계를 넘지 않게 합니다.

```text
Train:      X=[0,400)    48 random crops
Buffer:     X=[400,432)
Validation: X=[432,688)   9 fixed crops
Buffer:     X=[688,720)
Test:       X=[720,976)   9 fixed crops
Unused:     X=[976,1000)
```

고정 좌표는 `configs/data/berea_spatial_split_v1.csv`에 저장되어 있습니다. 이는 한 sandstone 내부의 spatial holdout이므로 다른 암종에 대한 일반화는 주장하지 않습니다.

Grayscale normalization은 train slab의 4억 voxel만 사용해 계산한 0.5/99.5 percentile(`uint8 2/184`)을 모든 split에 공통 적용합니다. 고정 설정은 `configs/data/berea_normalization_v1.yaml`에 저장합니다.

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

SAINT forward, E1 L1 학습 엔진, raw volume 로딩, aligned crop, sparse-z sampling과 기본 loss는 구현되어 있습니다. Topology loss와 evaluation의 `NotImplementedError`는 이후 실험 단계에서 구현할 interface skeleton이며, 공개 함수 signature와 데이터 계약을 유지합니다.

## E1 학습 루프 확인

AMI와 RFN의 E1 L1 학습 루프는 실제 데이터 모듈과 독립적으로 smoke test할 수 있습니다.

```bash
python scripts/train.py --config configs/e1_ami_smoke.yaml --device cpu
python scripts/train.py --config configs/e1_rfn_smoke.yaml --device cpu
```

두 설정은 작은 synthetic tensor로 forward, backward, validation 및 `best.pt`/`last.pt` 저장만 확인합니다. 연구 결과에 사용하지 않습니다. 실제 Digital Rock DataLoader도 같은 training engine을 사용합니다.

실제 Berea 파일과의 연결은 개인 경로를 config에 저장하지 않고 환경 변수로 전달합니다. 아래 설정은 256³ center crop에서 AMI batch 생성과 한 번의 optimizer step만 확인하는 smoke test입니다. `0-255` 정규화와 작은 모델을 사용하므로 연구 결과로 사용하지 않습니다.

```powershell
$env:THEME26_GRAYSCALE_RAW = "C:\path\to\Berea_2d25um_grayscale_filtered.raw"
python scripts/train.py --config configs/e1_ami_real_data_smoke.yaml --device cpu
```

고정 spatial manifest와 train-only normalization을 함께 확인하는 pilot은 다음과 같습니다. 작은 모델로 한 step만 실행하므로 연구 결과가 아니라 GPU 학습 전 연결 검증입니다.

```powershell
$env:THEME26_GRAYSCALE_RAW = "C:\path\to\Berea_2d25um_grayscale_filtered.raw"
python scripts/train.py --config configs/e1_ami_berea_pilot.yaml --device cpu
```

```bash
export THEME26_GRAYSCALE_RAW=/path/to/Berea_2d25um_grayscale_filtered.raw
python scripts/train.py --config configs/e1_ami_real_data_smoke.yaml --device cpu
```

## 문서

- [Experiment and model settings](docs/experiment_model_settings.md): 확정된 실험 조건, 모델·loss·평가 설정과 남은 결정
- [Research protocol](docs/research_protocol.md): 데이터, 학습, 평가 규칙
- [Interface contract](docs/interface_contract.md): 모듈별 입력·출력 shape, dtype과 공개 API
- [Storage and Slurm](docs/operations/storage-and-slurm.md): 서버 저장 구조와 GPU 실행 예시
- [Contributing](CONTRIBUTING.md): branch, PR과 review 규칙

## 참고자료

- Peng et al., *SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis*, CVPR 2020. [DOI](https://doi.org/10.1109/CVPR42600.2020.00777)
- [Official SAINT repository](https://github.com/cpeng93/SAINT)
- Shit et al., *clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
