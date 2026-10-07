# FP64 투과도 평가: 추론 결과에서 바로 연결하기

이 문서는 `8e7f092`의 저장소 구조에 통합한 **물성 평가 담당 영역**의 안내입니다.
별도 `theme26-evaluation/` 또는 `theme26_eval` 프로젝트를 넣지 않습니다.
모델·loss·학습·데이터 처리 코드는 변경하지 않습니다.

## 1. 무엇이 들어왔나요?

- `src/theme26/evaluation/physics_metrics.py`: 기존 공개 함수의 구현. 결과는 중첩 없는 scalar dict입니다.
- `src/theme26/evaluation/yang_lbm/`: 기존 최종 ZIP의 FP64 계산부와 Python worker.
- `src/theme26/evaluation/physics_io.py`: 팀 `scripts/infer.py`의 저장 결과와 기존 데이터 API를 연결.
- `scripts/evaluate_physics.py`: 설치 확인, 기존 테스트 기록 확인, 저장된 결과 평가.
- `configs/evaluation/physics_fp64.yaml`: 기존 solver 설정 및 동결된 분할 기준.
- `tests/test_evaluation_fp64.py`: 소프트웨어·입출력 테스트. 실제 유동을 실행하지 않습니다.

이미지·위상 지표의 기존 파일과 E2/E3 학습 구현은 이번 변경 범위가 아닙니다.
새 최상위 패키지나 두 번째 `pyproject.toml`을 만들지 않았습니다.

## 2. 기존 Python 환경에서 시작

저장소 루트에서 실행합니다. 기존에 Taichi가 돌아가던 환경을 그대로 사용합니다.

```bash
python scripts/evaluate_physics.py doctor
python scripts/evaluate_physics.py reference-report
```

두 명령은 GPU나 유동 계산을 시작하지 않습니다.
첫 명령은 Python·패키지 정보를, 두 번째는 이미 전달된 FP64 테스트 기록을 보여줍니다.
추가 설치가 필요한 환경에만 선택적으로 `python -m pip install -e '.[evaluation]'`을 사용합니다.
이 extra는 SciPy와 Taichi 1.7.4를 추가하며, PyTorch 의존성을 변경하지 않습니다.
GPU 드라이버·장비에서 실제 실행이 되는지까지 `doctor`가 보장하지는 않습니다.

## 3. 모델 담당자가 넘기는 최종 파일

현재 `scripts/infer.py`의 `reconstruct` 실행 결과를 그대로 사용합니다.

```text
기존 추론 결과 폴더/
  manifest.yaml
  000/
    prediction.npy
    primary_mask.npy
    metadata.yaml
  001/
    ...
```

`prediction.npy`는 `[256,256,256]`, XYZ, float32의 **최종 회색조**입니다.
추론 코드에서 이미 `[0,1]` clipping과 관측 단면 재삽입을 마쳤습니다.
평가에서는 이를 다시 추론하거나 raw cache처럼 다루지 않습니다.
`metadata.yaml`의 crop 좌표·복셀 간격·정규화 ID·checkpoint hash를 결과에 보존합니다.

현재 연구 기준은 **Berea 한 core의 공간 분할**입니다.
Primary 영상 mask는 189개 bracketed missing slice이며, 예전 192장 규칙을 다시 넣지 않습니다.
본 모듈의 투과도는 완성된 256³ 전체 공극 부피를 평가합니다.
`primary_mask.npy`를 공극 mask로 사용하면 안 됩니다.

학습 checkpoint만 있고 위 파일들이 없다면 모델 담당자가 기존 `scripts/infer.py`로
먼저 frozen inference를 수행해야 합니다. 이 평가 코드는 checkpoint를 읽거나 모델을 학습하지 않습니다.

## 4. 추론 결과를 바로 평가

먼저 `configs/evaluation/physics_fp64.yaml`의 `segmentation.threshold`와
`segmentation.version`에 **팀이 이미 결정한 분할 기준**을 입력합니다.
`pore_is_dark`도 데이터 기준에 맞게 확인합니다.
기본 threshold는 임의로 정하지 않았습니다. 이 값은 공극/고체를 나누는 정의이지,
solver의 합격 기준이 아닙니다.

```bash
python scripts/evaluate_physics.py evaluate-inference --prediction-dir outputs/your_inference --binary-raw /path/to/Berea_2d25um_binary.raw --config configs/evaluation/physics_fp64.yaml --experiment-id E1 --seed 1 --output-dir outputs/evaluation/E1_seed1
```

경로와 E1/seed 값은 실제 실행에 맞춥니다. 출력 폴더는 새 이름을 사용합니다.

이 명령은 다음을 수행합니다.

1. 추론 `manifest.yaml`과 각 `metadata.yaml`의 일치를 확인합니다.
2. 저장된 최종 회색조를 공통 기준으로 이진화합니다.
3. 기존 `read_raw_volume`, `extract_volume_crop`, `to_pore_mask`로 같은 좌표의 reference binary를 가져옵니다.
4. 두 부피에 같은 FP64 LBM을 적용하고, 투과도와 원본 대비 오차를 저장합니다.

Berea RAW는 기존 데이터 모듈의 C-order XYZ 해석을 따릅니다.
공극 label은 설정의 `source_pore_value: 0`이며, 내부에서는 True=pore입니다.
실제 binary/grayscale의 정렬과 원본 선택은 데이터 담당자가 확인한 자료를 사용해야 합니다.
메타데이터 형식 검사는 잘못된 실제 파일 선택까지 자동으로 증명하지는 않습니다.

## 5. 이미 공극 binary 두 개가 있을 때

둘 다 같은 위치의 256³, XYZ, bool, True=pore인 `.npy` 파일이면 분할 설정 없이 실행합니다.

```bash
python scripts/evaluate_physics.py evaluate-pair --reference /path/to/reference_pore.npy --prediction /path/to/prediction_pore.npy --spacing-um 2.25 2.25 2.25 --experiment-id E3 --seed 1 --output-dir outputs/evaluation/E3_seed1
```

이 경로의 crop 정렬·reference 선택·분할 기준은 호출자가 보장합니다.
한 실험의 결과를 다른 실험 이름이나 seed로 잘못 표시하지 않습니다.
파일 경로와 큰 배열은 Git에 넣지 않습니다.

## 6. 학습/추론 코드에서 배열로 연결

외부 스크립트가 아니라 함수 호출로 연결할 때도 기존 공개 API를 사용합니다.

```python
from theme26.evaluation.yang_lbm import LBMConfig, YangLBMSolver
from theme26.evaluation.physics_metrics import compute_physics_metrics
from theme26.evaluation.physics_io import segment_full_volume

# full: 기존 compose_full_volume(np.clip(prediction, 0, 1), sample)의 결과
# 또는 scripts/infer.py가 저장한 prediction.npy
# 원래 tensor가 [B,1,X,Y,Z]라면 먼저 [b,0].detach().cpu().numpy()
predicted_pore = segment_full_volume(
    full,
    threshold=frozen_threshold,
    pore_is_dark=True,
)
solver = YangLBMSolver(
    LBMConfig.from_yaml("configs/evaluation/physics_fp64.yaml")
)
metrics = compute_physics_metrics(
    predicted_pore,
    reference_pore,  # 같은 crop, bool, True=pore
    spacing_um_xyz=spacing_from_metadata,
    solver=solver,
)
print(metrics["comparison_status"])
print(metrics.get("prediction_kz_mD"))
print(metrics.get("reference_kz_mD"))
print(metrics.get("kz_log10_abs_error"))
```

`metrics`는 기존 계약대로 scalar dict입니다. 정의되지 않은 값은 0이 아니라 key 생략으로 나타냅니다.
한 부피만 계산하려면 `solver.solve(pore, spacing_um_xyz=spacing)`를 사용합니다.
전체 시간 기록까지 필요한 저장 경로는 `solver.solve_detailed()`를 내부적으로 사용합니다.
기본 실행은 같은 Python의 별도 worker이므로 부모의 Taichi/PyTorch 상태를 reset하지 않습니다.
동시에 돌아가는 학습 job과 GPU 메모리를 공유하지 않도록 실행 시간을 배치합니다.

## 7. 무엇을 읽으면 되나요?

- `results.csv`: crop·seed별 reference/prediction Kz, 상대오차, 로그 오차, 실행 상태, metadata.
- `results.json`: 같은 표의 JSON.
- `cases/000.json` 또는 pair 실행의 `case.json`: 전체 시간 기록과 solver 진단.
- `evaluation_config.yaml`: 해당 실행에 사용한 설정.

대표 열은 `prediction_kz_mD`, `reference_kz_mD`, `relative_error_pct`,
`kz_log10_abs_error`, `prediction_tail_kz_std_mD`, `prediction_status`입니다.
시간 기록의 표준편차는 반복실험 표준편차나 정확도 신뢰구간이 아닙니다.
같은 core의 여러 crop을 독립적인 여러 암석 시료라고 해석하지 않습니다.

## 8. 64비트와 실행 상태

기존 최종본의 설정을 유지합니다: D3Q19 MRT, `tau=3*nu+0.5`, force 계수 3/9,
z방향 force density, 전 축 주기 경계, reservoir 없음, FP64 only.
`max_steps=50000`, `max_seconds=7200`이며 자동 재시험·자동 FP32 전환은 없습니다.

임의의 오차 합격선이나 승인 파일은 사용하지 않습니다.
`completed`는 요청한 step 수 종료, `budget_exhausted`는 시간 예산 종료입니다.
두 경우 모두 얻은 값을 보고하되, 후자를 같은 계산량의 결과로 섞지 않습니다.
`converged=True`를 자동으로 붙이지 않습니다.

NaN, 실행 실패, 메모리 부족은 정상 투과도로 만들지 않습니다.
비어 있지 않은 공극에서 주기 격자를 관통하는 경로가 없으면 `non_percolating`입니다.
공개 평가 API는 저장소 계약에 맞게 공극 또는 고체가 완전히 없는 입력을 거부합니다.
파일 기반 일괄 평가에서는 그런 분할 실패를 상태로 기록하고 다른 사례와 구분합니다.
이 입력 검사는 수치적 정확도의 임의 합격선과는 다릅니다.

## 9. 출처와 확인 범위

Yang, J., Xu, Y., & Yang, L. (2022).
*Taichi-LBM3D: A Single-Phase and Multiphase Lattice Boltzmann Solver on Cross-Platform Multicore CPU/GPUs.*
Fluids 7(8),270. DOI: 10.3390/fluids7080270.

원 코드: `yjhp1016/taichi_LBM3D`,
commit `fe49e3f609b2038cbf93c8bd453ffc5c2bf98e4c`,
`Single_phase/LBM_3D_SinglePhase_Solver.py`.
MIT 저작권과 라이선스는 `yang_lbm/_vendor/LICENSE`에 보존했습니다.

이번 계산부는 원 파일의 byte-identical 복사가 아니라 **이미 전달한 FP64 파생 구현**입니다.
주기 경계만 추출하고, 기존 전달본의 완화식·force 계수 수정과 입력/실행 구조를 유지했습니다.
원 논문의 모든 계산 조건을 재현한 코드라고 표현하지 않습니다.
연결/저장/패키지 경로만 수정했으며 충돌·이동·거시량 계산식을 새로 바꾸지 않았습니다.
해당 출처·변경 사항은 `_vendor/UPSTREAM.json`에도 있습니다.
Guo et al. (2002), DOI 10.1103/PhysRevE.65.046308은 force 식의 별도 근거입니다.

기존 사용자 FP64 기록은 Kz 약 444.72084 mD, 마지막 6개 기록의 표준편차 약
0.0001302 mD입니다. 논문 789 mD와의 산술적 차이는 약 43.63%이고,
유동 방향/구동 조건의 동일성이 확인되지 않아 정확도 오차나 재현 성공으로 부르지 않습니다.
짝수 step만 기록한 한계와 단면 유량 진단도 원 기록에 남겼습니다.
이번 통합에서 해당 시뮬레이션이나 새로운 256³ 유동을 재실행하지 않았습니다.

## 10. 통합 확인

```bash
python -m unittest discover -s tests -p "test_evaluation_fp64.py" -v
```

작은 배열과 모의 solver로 입출력·파일 저장·worker 경로를 확인합니다.
실제 유동·GPU·학습 검증이 아니며, 프로젝트 전체의 기존 테스트 결과로 확대하지 않습니다.
새 코드는 선택적인 evaluation extra로 분리되어 기본 학습 의존성을 바꾸지 않습니다.
