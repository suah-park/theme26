# SAINT for Berea Sparse-Z Reconstruction

Berea sandstone의 grayscale Micro-CT에서 네 장마다 한 장을 관측하고 누락 단면을 복원합니다. 같은 core의 별도 공간 구역에서 E0 linear, E1 SAINT L1, E2 gradient, E3 topology-aware fine-tuning을 비교합니다. 실제 스캔 시간 절감이나 다른 core로의 일반화를 직접 입증하는 실험은 아닙니다.

## 어디부터 보면 되나요?

| 목적 | 읽을 파일 |
|---|---|
| 실험 조건·모델 설정·아직 결정할 값 | [연구 기준](docs/research_protocol.md) |
| 설치·데이터 위치·pilot·학습·추론 실행 | [E1 실행 안내](docs/e1_execution.md) |
| 코드 구현 시 입력·출력·담당 경로 | [인터페이스 계약](docs/interface_contract.md) |
| 추론 후 FP64 투과도 평가·결과 읽기 | [물성 평가 안내](docs/evaluation_fp64.md) |
| Branch·PR와 코드 공유 규칙 | [기여 안내](CONTRIBUTING.md) |

실험 수치는 `configs/` YAML, 연구 설계 설명은 연구 기준, 명령어는 실행 안내가 기준입니다. RAW·checkpoint·prediction은 Git에 올리지 않습니다.

## 실행에 필요한 파일

```text
scripts/
  train.py                    # AMI/RFN 학습 및 재개
  infer.py                    # AMI cache 생성 및 최종 복원
configs/
  data/                       # 고정 공간 분할·정규화·crop 좌표
  model/saint_x4.yaml          # CPU 검사/목표 모델의 공유 구조
  experiments/
    e1_ami_berea_pilot.yaml    # 작은 모델로 실제 데이터 CPU 1-step
    e1_ami_gpu_pilot.yaml      # 목표 AMI의 GPU 1-step
    e1_rfn_gpu_pilot.yaml      # AMI cache가 필요한 RFN GPU 1-step
    e1_ami_target.yaml         # AMI 본학습 템플릿
    e1_rfn_target.yaml         # RFN 본학습 템플릿
    smoke/                    # 개발용 합성/단일 crop 연결 검사
  inference/                  # cache와 최종 복원 설정
slurm/train_e1.sbatch          # Slurm을 사용하는 서버의 제출 템플릿
```

Pilot은 성능 결과가 아닙니다. 본학습 batch size와 epochs는 GPU pilot 후 결정하며 현재 템플릿에는 `null`입니다. 서버가 Slurm인지 직접 실행 방식인지는 관리자에게 확인합니다.

## 구현 코드는 어디에 있나요?

```text
src/theme26/
  contracts.py    # 공통 metadata·crop·sampling 형식
  data/           # RAW I/O, crop, sampling, 공간 기준, audit/QC
  models/saint/   # AMI, FDM, dynamic filter, RFN, 배치 추론
  training/       # Torch dataset, 학습 engine, cache, checkpoint, 공통 준비
  losses/         # L1/gradient 구현; topology는 구현할 계약
  evaluation/     # FP64 투과도 평가; image/topology는 구현할 계약
tests/            # 작은 CPU 배열로 코드 검증; 연구 test set과 다름
.github/workflows/tests.yml   # GitHub 자동 CPU 검사
```

`data/`는 NumPy 데이터 처리, `training/`은 그 데이터를 모델용 Torch batch로 연결합니다. `__init__.py`는 패키지 연결 파일이며 별도 실험이 아닙니다.

## 빠른 CPU 검사

Python 3.10 이상 환경에서 프로젝트와 학습 라이브러리를 설치합니다. GPU용 PyTorch 빌드는 서버 드라이버·GLIBC에 맞춰 먼저 설치합니다.

```bash
python -m pip install -e '.[training]'
python -m unittest discover -s tests -v
```

실제 Berea 파일 없이 CPU 검사가 가능합니다. 실제 paired RAW가 준비되면 아래 검사도 실행합니다.

```bash
python -m theme26.data audit --data-root /path/to/Berea --output-dir outputs/audit
python -m theme26.data qc --data-root /path/to/Berea --output-dir outputs/spatial_qc
```

## 구현 상태

- 구현: RAW/공간 분할/QC, E0, AMI/RFN forward, L1/gradient 함수, E1 AMI → cache → RFN 학습, 재개·기록·최종 추론.
- 미구현: E2 gradient 학습 연결, E3 topology 및 fine-tuning, PSNR/SSIM·위상·굴곡도 평가.
- 별도 검증 필요: 목표 모델의 GPU pilot·overfit·본학습·실제 복원 성능. CPU 검사나 설치 성공만으로 이를 확인했다고 주장하지 않습니다.

## 선택적 FP64 물성 평가

기존 Python 환경에서 `python scripts/evaluate_physics.py doctor`와
`python scripts/evaluate_physics.py reference-report`로 설치 정보와 기존 기록을 확인합니다.
이 두 명령은 유동을 실행하지 않습니다. 실제 평가는 저장된 `prediction.npy`와
`metadata.yaml`에 연결하며 [물성 평가 안내](docs/evaluation_fp64.md)를 따릅니다.

평가 코드·입출력 연결은 구현되어 있으며, 기존 최종본의 FP64 계산식과 fixed-budget 실행을 유지합니다.
`completed`는 실행 종료이지 정확도 합격이 아닙니다. 이번 통합에서 새 유동 계산을 검증한 것은 아닙니다.
LBM에는 선택적으로 `python -m pip install -e '.[evaluation]'`을 사용하며 기본 학습 의존성은 유지합니다.

## 출처

- [11 Sandstones 데이터](https://digitalporousmedia.org/datasets/317/), DOI `10.17612/F4H1-W124`, ODC-By 1.0.
- [SAINT 원본](https://github.com/cpeng93/SAINT), CVPR 2020. 구조와 학습 동작의 reference이며 이 저장소는 독립 구현입니다.
- 팀이 올린 논문 PDF는 저장소에 그대로 유지합니다.
