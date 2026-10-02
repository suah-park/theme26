# SAINT for Berea Sparse-Z Reconstruction

스캔 부담을 줄이기 위한 주 실험: Berea 한 core의 train 구역에서 학습한 모델이
같은 core의 별도 구역에서 네 장마다 한 장만 관측하여 누락 단면을 복원하는지 평가합니다.
Dense Berea에서 synthetic x4 입력을 만드는 실험이며 실제 스캔 비용 절감을 직접 측정한 결과는 아닙니다.
Binary는 grayscale 입력/정답이 아니라 E3 topology와 평가 reference입니다.

## 고정 공간 분할

모든 좌표는 canonical XYZ, 끝 좌표는 제외합니다. 원본은 uint8 `1000³`, model crop은 `256³`입니다.

| Split | X 범위 | Y/Z 범위 | 사용 |
|---|---|---|---|
| Train | `[0,400)` | `[0,1000)` | 고정 seed로 만든 48개 crop, 정규화 통계 |
| Buffer | `[400,432)` | 전체 | 사용하지 않음 |
| Validation | `[432,688)` | `[0,1000)` | 고정 crop 9개, 모델 선택 |
| Buffer | `[688,720)` | 전체 | 사용하지 않음 |
| Test | `[720,976)` | `[0,1000)` | 고정 crop 9개, 최종 평가 |

각 평가 slab에서 Y/Z 시작 좌표는 `{0,372,744} × {0,372,744}`입니다.
32-voxel buffer는 초기 설계 값이며 공간적 독립성을 보장한다고 주장하지 않습니다.
RAW는 C-order XYZ로 직접 해석하며 축을 transpose하지 않습니다. 물성의 Z는 이 좌표계의 Z입니다.

## 데이터와 설치

필요한 파일은 Berea 두 개뿐입니다:

- `Berea_2d25um_grayscale_filtered.raw`
- `Berea_2d25um_binary.raw`

`--data-root`는 두 파일이 직접 있는 디렉터리 또는 `Berea/`의 부모를 받습니다.
원본은 읽기 전용으로 사용하며 대용량 배열·checkpoint는 Git에 넣지 않습니다.

```bash
python -m pip install -e .
# 시각 QC가 필요하면: python -m pip install -e '.[plots]'
# 모델·학습 환경에는: python -m pip install -e '.[training]'
```

## 전처리와 검증 실행

```bash
python -m unittest discover -s tests -v
python -m theme26.data audit --data-root /path/to/Berea --output-dir outputs/audit
python -m theme26.data qc --data-root /path/to/Berea --output-dir outputs/spatial_qc
```

설치하지 않고 실행할 때는 `PYTHONPATH=src`를 설정합니다.
PowerShell에서는 `$env:PYTHONPATH='src'`를 먼저 실행합니다.
정규화는 train 구역만의 histogram으로 p0.5/p99.5를 계산하고 모든 split에 같은 값을 적용합니다.
현재 구역과 제공된 Berea에서 계산한 값은 `2/184`, train voxel 수는 400,000,000입니다.
동결된 값과 crop 좌표는 `configs/data/`의 YAML에 보존합니다. 파일 또는 구역을 바꾸는 것은
새 protocol version이 필요한 연구 설계 변경입니다. 예전 8-core 기준 `2/206`은 사용하지 않습니다.
Source binary의 pore 값은 0이며 내부 mask에서는 `True=pore`입니다.

## tests와 연구 test의 차이

`tests/`는 작은 NumPy 배열과 임시 RAW 파일로 구현을 검증합니다. 실제 Berea가 없어도 실행됩니다.
공간 경계, split 겹침, train-only 통계, 재현성, 단일 상 crop, 잘못된 artifact도 검사합니다.
연구 test 구역은 실제 Berea `[720,976)`이며 학습한 모델의 최종 성능 평가에 사용합니다.
전처리 QC PASS는 데이터 형식/정렬/보간 검증이며 학습 성능이나 복원 품질 결과가 아닙니다.

## 코드 구조와 진행 상태

```text
src/theme26/
  contracts.py
  data/          # RAW, crop, sampler, spatial protocol, audit/QC와 data CLI
  models/saint/  # AMI, FDM, dynamic filter, RFN, 전체 forward
  losses/        # L1/gradient 구현, topology 스켈레톤
  evaluation/    # metric 스켈레톤
  training/      # dataset adapter, engine, checkpoint
scripts/         # train과 infer 실행 진입점만 유지
configs/
  data/          # Berea split, normalization, crop manifest (모두 YAML)
  model/         # 공유 SAINT architecture profile
  experiments/   # 실행 설정; smoke는 experiments/smoke/
tests/           # 합성 데이터 CPU unit tests
```

전처리, SAINT forward, L1/gradient 함수와 E1 AMI/RFN 학습 엔진은 구현되어 있습니다.
실제 Berea는 AMI DataLoader까지 연결되어 있습니다. 실제 AMI 결과를 RFN 학습에 연결하는 단계,
E2 학습 연결, E3 topology, 최종 infer와 성능 metric은 아직 미구현입니다.
현재 pilot은 작은 모델/1-step 연결 검사 설정이며 본 실험 설정이 아닙니다.
`TrainRandomCropDataset`은 NumPy `.sample(rng)` sampler입니다. 모델 담당자가 Torch adapter를 연결합니다.

E0=linear interpolation, E1=SAINT L1, E2=L1+0.1 gradient,
E3=E2에서 AMI를 고정한 RFN topology fine-tuning입니다.
Sparse `[256,256,64]`, observed slice 64장, primary metric 189장,
tail z=253..255은 boundary diagnostic입니다. 최종 volume에 observed slice를 재삽입합니다.

## 문서·출처

- [실험·모델 설정 통합본](docs/experiment_model_settings.md)
- [연구 프로토콜](docs/research_protocol.md)
- [인터페이스](docs/interface_contract.md)
- [전처리 인계](docs/data_handoff.md)
- [데이터](https://digitalporousmedia.org/datasets/317/) / DOI 10.17612/F4H1-W124 / ODC-By 1.0
- [SAINT 원본](https://github.com/cpeng93/SAINT)

## 통합된 E1 연결

```powershell
$env:THEME26_GRAYSCALE_RAW = 'C:\path\to\Berea_2d25um_grayscale_filtered.raw'
python scripts/train.py --config configs/experiments/e1_ami_berea_pilot.yaml --device cpu
```

학습은 `configs/data/`의 YAML만 읽으며 split, 정규화, 48/9/9 crop 좌표와 version을 검사합니다.
옛 ZYX 및 424/456/744 기반 artifact/checkpoint는 이번 XYZ 설정의 결과로 재사용하지 않습니다.
현재 학습 엔진의 checkpoint 기준은 전체 L1이고 gradient 경계 mask는 양쪽 포함 방식입니다.
Primary-mask MAE 선택과 observed–missing 경계 정책은 논의 후 별도 변경합니다.
학습 중 loss와 checkpoint 기준을 혼동하지 않습니다.

## 공통 모델 설정

`configs/model/saint_x4.yaml`의 `ami_target`/`rfn_target`이 문서의 64-channel 목표 모델이다.
`ami_cpu`/`rfn_cpu`는 8-channel 연결 검사 모델이며 기존 CPU 실행 설정이 이 profile을 참조한다.
목표 모델의 GPU 실행 config, batch size와 본학습 기간은 아직 확정하지 않았다.
YAML에서 `model.settings`와 `model.profile`을 선택하면 학습 코드가 공통 설정을 읽고,
checkpoint에는 profile 경로와 실제 적용된 모델 값을 함께 저장한다.
