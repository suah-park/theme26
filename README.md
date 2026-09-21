# Sparse-slice Digital Rock 3D Reconstruction

GitHub repository: `theme26`

SAINT 기반 sparse-to-dense reconstruction을 binary Micro-CT digital rock에 적용하고, 영상 정확도뿐 아니라 3D 공극 연결성과 투과도 보존까지 검증하는 연구 프로젝트입니다.

## 1. 연구 목표

z축 방향으로 희소하게 관측된 binary Micro-CT volume이 주어졌을 때 다음 질문을 검증합니다.

> SAINT 기반 cross-view volume reconstruction이 각 누락 단면을 독립적으로 복원하는 2D 모델보다 3차원 공극 연결성과 z 방향 투과도를 더 잘 보존하는가?

이번 포스터 연구는 SOTA 달성을 주장하기보다 동일한 sparse input과 split에서 복원 방식의 차이를 검증하는 데 목적이 있습니다.

상세 프로토콜은 [최종 연구 프로토콜](digital_rock_sparse_slice_reconstruction_final_protocol.md)을 참고하세요.

## 2. 필수 실험 범위

### 비교 모델

| ID | 모델 | 역할 | 우선순위 |
|---|---|---|---|
| A0 | Linear interpolation | 비학습 sanity baseline | 필수 |
| A1 | Independent 2D reconstruction | 누락 axial slice를 독립적으로 예측 | 필수 |
| B | SAINT-Rock | SAINT의 AMI와 RFN을 binary digital rock에 적용 | 필수 |
| C | SAINT-Rock + topology-aware loss | 3D topology constraint 효과 검증 | Phase 2 |

### 관측 조건

- 주 데이터: Binary pore/solid Micro-CT volume
- 저해상도 축: z축만 downsampling
- Sparse input: 관측 단면 5장
- 관측 간격: `r=2`, `r=4`
- 권장 patch: `128 x 128`
- Primary threshold: `0.5`
- 학습 모델 반복: 최소 3 random seeds

### 핵심 평가 지표

1. Missing-slice Dice
2. Absolute porosity error
3. Euler characteristic density error
4. z 방향 percolating porosity error
5. z 방향 permeability log-error

## 3. 전체 연구 흐름

```text
Dense binary Micro-CT volume
             |
             v
Parent-core / spatial-block split
             |
             v
z-axis sparse sampling (r=2, r=4)
             |
        +----+----------------+----------------+
        |                     |                |
        v                     v                v
   A0 Linear          A1 Independent 2D   B SAINT-Rock
        |                     |                |
        +---------------------+----------------+
                              |
                              v
                 Dense reconstructed test block
                              |
           +------------------+------------------+
           |                  |                  |
           v                  v                  v
        Image            Connectivity       Permeability
```

## 4. 팀 구성 및 역할 분배

프로젝트는 세 팀으로 운영합니다. 실제 담당자 이름은 아래 표의 `담당자`와 `교차 검토자` 열에 기록합니다.

### Team A. GPU·실험운영팀

**목표:** 누구나 동일한 조건에서 학습·추론을 재현할 수 있도록 데이터, 환경, GPU 작업과 실험 기록을 관리합니다.

주요 업무:

- 데이터 경로, split manifest 및 sparse sampling 규칙 관리
- 공통 Python/PyTorch 환경과 dependency 고정
- GPU 배정표 및 job queue 관리
- 공통 train/inference entry point와 config 규격 관리
- Seed, checkpoint, log, runtime 및 GPU memory 기록
- 중단된 학습의 resume 및 checkpoint 보존 정책 관리
- 모델별 실험 완료 여부와 실패 원인 기록
- Team B가 전달한 모델을 동일 조건으로 실행
- Team C가 사용할 prediction volume 및 metadata 전달

필수 산출물:

- 고정된 `train/validation/test` split manifest
- `r=2`, `r=4` 데이터 생성 확인 결과
- 환경 설치 문서
- 실험별 config, log, best checkpoint
- Test prediction volume과 실행 metadata

완료 기준:

- 동일 config와 seed로 재실행 가능
- Best checkpoint가 validation 기준으로 선택됨
- Test set을 이용한 checkpoint 선택이 없음
- 모델별 입력 patch와 test block이 동일함

### Team B. 모델·방법론팀

**목표:** 비교 모델을 구현하고 architecture와 loss의 차이를 통제합니다.

주요 업무:

- A0 Linear interpolation 구현
- A1 Independent 2D reconstruction 구현
- SAINT 원 코드와 논문 구조 확인
- B SAINT-Rock의 AMI 및 RFN 구현·포팅
- Binary pore-probability output 적용
- Missing-position BCE + Dice loss 구현
- Observed slice hard-copy 로직 구현
- Unit test 및 tensor shape 검증
- Phase 2 topology-aware loss 후보 검토 및 구현
- 모델별 parameter count 및 inference 요구량 정리

필수 산출물:

- A0/A1/B 실행 코드
- 모델별 입출력 tensor 규격
- Loss 및 observation mask 검증 코드
- AMI Stage 1과 RFN Stage 2 checkpoint
- Architecture diagram 및 방법 설명

완료 기준:

- 모든 모델이 동일 sparse input에서 실행됨
- A1/B에 동일 BCE+Dice와 checkpoint 기준 적용
- 관측 단면이 최종 volume에서 원본값으로 유지됨
- Validation/Test 데이터 누출이 없음

### Team C. 평가·결과정리팀

**목표:** 모델 결과를 동일한 지표와 물리 조건으로 평가하고 포스터용 근거를 만듭니다.

주요 업무:

- Missing-slice Dice, IoU 및 voxel error 계산
- Porosity, specific surface 및 pore-size 지표 계산
- Euler characteristic 및 connected-component 분석
- z 방향 percolating porosity와 disconnection rate 계산
- 동일 solver·경계조건으로 GT/prediction permeability 계산
- `r=2`, `r=4` 및 모델별 paired comparison
- 3 seeds 통합, confidence interval 및 통계 분석
- 구조 오차와 permeability error의 correlation 분석
- 정성 결과, 3D rendering, 그래프 및 표 제작
- 포스터 결과·결론 문구 작성

필수 산출물:

- 평가 코드와 metric definition 문서
- 모델별 raw metric table
- 통합 summary table과 figure
- Permeability solver 설정 및 convergence 기록
- 포스터용 결과 패널과 결과 서술 초안

완료 기준:

- 관측 단면을 제외한 missing-slice metric을 별도 보고
- 충분히 큰 동일 test block에서 물성 비교
- 모든 모델에 threshold 0.5를 동일 적용
- Non-percolating prediction을 임의의 작은 permeability로 대체하지 않음
- 그래프의 모든 값이 raw result로 역추적 가능

## 5. 담당자 배정표

프로젝트 시작 회의에서 이름을 입력합니다. 각 작업에는 주 담당자와 다른 팀의 교차 검토자를 한 명씩 둡니다.

| 팀 | 담당자 | 교차 검토자 | 주요 책임 |
|---|---|---|---|
| Team A: GPU·실험운영 | `TBD` | `TBD` | 환경, split, GPU, config, checkpoint, prediction export |
| Team B: 모델·방법론 | `TBD` | `TBD` | A0/A1/B 구현, loss, SAINT 포팅, topology 확장 |
| Team C: 평가·결과 | `TBD` | `TBD` | 구조·연결성·투과도 평가, 통계, figure, 포스터 |
| Integration lead | `TBD` | - | 일정 조정, interface 승인, 최종 merge 및 포스터 범위 결정 |

## 6. 팀 간 인수인계 규격

### Team A → Team B

- Split manifest
- Sparse sampling config
- Patch tensor shape
- Pore/solid label convention
- Voxel spacing
- Train/validation/test 데이터 접근 방법

### Team B → Team A

- Model entry point
- Default config
- 예상 GPU memory
- Checkpoint selection metric
- Resume 방법
- Prediction export format

### Team A/B → Team C

Prediction마다 다음 metadata를 함께 제공합니다.

```text
model_id
git_commit
config_name
seed
sampling_factor
checkpoint_epoch
validation_missing_dice
threshold
voxel_spacing
test_block_id
```

### Team C → 전체 팀

- Raw metric CSV
- Summary table
- Failure case 목록
- Figure 생성 config
- 결론을 지지하거나 반박하는 핵심 결과

## 7. RACI 요약

`R`: 실행 담당, `A`: 최종 책임, `C`: 검토, `I`: 공유

| 작업 | Team A | Team B | Team C | Integration lead |
|---|:---:|:---:|:---:|:---:|
| 데이터 split·sampling | R | C | C | A |
| 환경·GPU·실험 실행 | R | C | I | A |
| 모델 및 loss 구현 | C | R | I | A |
| Checkpoint 선택 | R | C | I | A |
| 영상 metric | I | C | R | A |
| Connectivity metric | I | C | R | A |
| Permeability | C | I | R | A |
| 통계·figure | I | C | R | A |
| 포스터 문구 | C | C | R | A |
| 최종 범위 결정 | C | C | C | A/R |

## 8. 공통 학습 규칙

- Optimizer 초기값: AdamW
- Initial learning rate: `1e-4`
- Weight decay: `1e-5`
- Maximum epochs: `200`
- Early-stopping patience: `20 epochs`
- 개선 기준: validation missing-slice Dice가 이전 최고값보다 `0.001` 이상 증가
- 최종 모델: 마지막 epoch가 아닌 best validation checkpoint
- Test metric을 이용한 checkpoint 또는 threshold 선택 금지
- `r=2`, `r=4` 모델을 우선 별도로 학습
- Primary threshold는 모든 모델에 `0.5` 적용

자세한 loss와 평가 정의는 [최종 연구 프로토콜](digital_rock_sparse_slice_reconstruction_final_protocol.md)을 따릅니다.

## 9. Git 및 GitHub 작업 규칙

### Branch

```text
main                 안정화 및 포스터 재현 가능 버전
develop              통합 개발 브랜치
infra/<topic>        Team A 작업
model/<topic>        Team B 작업
eval/<topic>         Team C 작업
docs/<topic>         문서 및 포스터 작업
```

예시:

```text
infra/split-manifest
model/independent-2d
model/saint-rock-ami
eval/connectivity-metrics
docs/poster-method
```

### Commit

작고 검토 가능한 단위로 commit합니다.

```text
feat(model): add missing-slice dice loss
feat(eval): add z-percolating porosity metric
fix(data): prevent overlap across spatial splits
docs(protocol): clarify checkpoint selection
```

### Pull request

모든 주요 변경은 PR로 병합합니다.

PR에 포함할 내용:

- 변경 목적
- 입력·출력 또는 config 변경 여부
- 실행한 검증
- 영향받는 실험
- 관련 issue
- 생성된 대용량 파일의 저장 위치

본인이 작성한 주요 코드는 가능하면 다른 팀원이 검토합니다.

## 10. Issue 운영

권장 label:

```text
team:infra
team:model
team:eval
type:experiment
type:bug
type:documentation
priority:high
blocked
needs-review
```

Issue에는 다음 완료 조건을 명시합니다.

```text
Goal
Input
Expected output
Acceptance criteria
Owner
Reviewer
Dependencies
```

## 11. 권장 디렉터리 구조

```text
.
├── README.md
├── CONTRIBUTING.md
├── configs/
│   ├── data/
│   ├── model/
│   └── experiment/
├── docs/
│   ├── decisions/
│   └── figures/
├── digital_rock_sparse_slice_reconstruction_final_protocol.md
├── scripts/
│   ├── train/
│   ├── infer/
│   └── evaluate/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── tests/
└── work/
```

코드가 추가되면서 필요한 폴더를 생성합니다. 빈 구조를 미리 대량으로 만들지는 않습니다.

## 12. 데이터 및 대용량 산출물 정책

다음 파일은 Git에 직접 올리지 않습니다.

- 원본 Micro-CT volume
- 학습/검증/test patch 전체
- Model checkpoint
- TensorBoard/W&B raw log
- 전체 prediction volume
- Permeability solver 대용량 output

Git에는 다음만 포함합니다.

- 데이터 접근 및 전처리 설명
- Split manifest와 작은 metadata
- Config
- Source code
- 작은 summary table
- 포스터용 압축 figure

대용량 파일 저장 위치는 실험 metadata에 명시합니다. 민감하거나 공개할 수 없는 데이터의 경로·샘플·자격증명을 commit하지 않습니다.

## 13. 실행 단계 및 팀별 병렬 작업

### Milestone 0. 협업 기반 확정

- Team A: 환경 및 split 초안
- Team B: SAINT 코드 실행 가능성 확인
- Team C: metric 정의와 permeability workflow 확인
- 전체: tensor shape, label convention, test block 규격 승인

### Milestone 1. Baseline

- Team A: `r=2`, `r=4` 데이터와 공통 config
- Team B: A0, A1 구현
- Team C: Dice, porosity, connectivity metric unit test

완료 조건:

- 동일 test patch에서 A0/A1 prediction 생성
- Missing-slice metric과 full-volume metric 분리 확인

### Milestone 2. SAINT-Rock

- Team A: AMI/RFN 학습 job 운영
- Team B: Binary AMI, RFN 및 hard-copy 구현
- Team C: B 평가 pipeline 연결

완료 조건:

- `r=2`, `r=4` 각각 best validation checkpoint 확보
- 동일 test block의 A0/A1/B 결과 생성

### Milestone 3. 물성 평가

- Team A: 최종 prediction과 metadata 동결
- Team B: 실패 사례의 모델 측 원인 분석
- Team C: connectivity, permeability, 통계, figure 생성

완료 조건:

- GT/A0/A1/B에 동일 solver 조건 적용
- Headline metric table 완성
- 결과 해석 초안 완성

### Milestone 4. Topology 확장 여부 결정

다음 조건에서 C를 우선 수행합니다.

- B가 Dice는 개선하지만 connectivity/permeability를 개선하지 못함
- Pore-throat disconnection이 대표 실패 원인으로 확인됨
- 포스터 일정과 GPU 자원이 충분함

그 외에는 topology-aware loss를 후속 연구로 제시합니다.

## 14. 회의 운영

### 주간 전체 회의

각 팀은 다음 네 항목만 공유합니다.

1. 지난주 완료 결과
2. 재현 가능한 근거 또는 링크
3. 현재 blocker
4. 다음 주 acceptance criteria

### 짧은 실험 동기화

GPU 실험이 집중되는 기간에는 Team A와 Team B가 짧게 다음을 확인합니다.

- 실행 중인 config
- GPU와 예상 종료 시각
- Validation 상태
- 실패·재시작 여부

### Decision log

다음 변경은 `docs/decisions/`에 짧게 기록합니다.

- Split 변경
- Label convention 변경
- Primary metric 변경
- Loss 또는 threshold 변경
- Test block 변경
- Permeability solver 조건 변경

## 15. 포스터 최소 산출물

- 연구 질문 한 문장
- Sparse input → 비교 모델 → 물성 평가 pipeline
- GT/A0/A1/B 정성 비교
- `r=2`, `r=4` headline metric table
- Connected porosity error와 permeability error 관계
- 결론, 한계 및 topology-aware 후속 계획

## 16. 시작 전 합의할 항목

- [ ] 팀별 담당자와 교차 검토자
- [ ] Integration lead
- [ ] 데이터 공개 가능 범위
- [ ] Pore/solid label convention
- [ ] Parent core 또는 spatial block split
- [ ] Voxel spacing 및 target z resolution
- [ ] GPU별 사용 가능 시간과 메모리
- [ ] Permeability solver와 실행 담당자
- [ ] 포스터 제출 일정
