# Contributing

이 프로젝트는 모듈별 담당자가 독립적으로 구현하고, 작은 CPU test를 통과한 변경만 `main`에 병합합니다.

## 작업 영역

| 영역 | 경로 | 주요 책임 |
|---|---|---|
| Model and integration | `src/theme26/models/saint/`, `scripts/`, `config/`, `slurm/` | SAINT, 학습·추론, GPU 실행, 통합 |
| Data | `src/theme26/data/` | Raw I/O, normalization, crop, sparse sampling, E0 |
| Loss | `src/theme26/losses/` | L1, gradient, soft Dice, 3D soft-clDice |
| Evaluation | `src/theme26/evaluation/` | Image, topology, physics metrics |

공통 데이터 규칙과 공개 함수는 [Interface Contract](docs/interface_contract.md)를 따릅니다.

## Branch와 PR

1. 최신 `main`에서 짧은 feature branch를 만듭니다.
2. 자신의 담당 모듈과 관련 test를 함께 수정합니다.
3. 로컬 CPU test와 smoke test를 실행합니다.
4. PR에 입력·출력, 실행 명령과 검증 결과를 기록합니다.
5. Review 후 `main`에 병합합니다.

`main`에는 직접 push하지 않고 별도 `develop` branch는 사용하지 않습니다.

Branch 이름 예시:

```text
data/raw-loader
data/sparse-sampling
model/ami
model/rfn
loss/gradient
loss/topology
eval/image-metrics
eval/topology-metrics
infra/slurm
docs/onboarding
```

Commit 예시:

```text
feat(data): add aligned grayscale and binary crops
feat(model): implement grayscale AMI
feat(loss): add 3d soft-cldice
feat(eval): add missing-slice metrics
fix(data): preserve xyz axis order
docs(protocol): clarify segmentation settings
```

## 공통 규칙

- Canonical NumPy volume은 `[X,Y,Z]`입니다.
- PyTorch volume은 `[B,C,X,Y,Z]`입니다.
- Grayscale은 `float32 [0,1]`입니다.
- 내부 binary는 `pore=1`, `solid=0`입니다.
- Primary crop은 `256 x 256 x 256`입니다.
- Primary sampling은 Z축 x4입니다.
- 원본 volume, checkpoint, prediction과 solver bulk output은 commit하지 않습니다.
- 개인 PC와 서버의 절대 경로를 source code에 넣지 않습니다.
- Test 결과를 보고 threshold, loss weight나 checkpoint rule을 변경하지 않습니다.

## PR 체크리스트

- [ ] 담당한 공개 함수의 signature를 유지했다.
- [ ] 입력·출력 axis, shape, dtype과 값 범위를 설명했다.
- [ ] 정상 입력 test가 있다.
- [ ] 잘못된 shape/dtype/range test가 있다.
- [ ] 작은 synthetic volume의 CPU smoke test가 통과한다.
- [ ] 데이터 split과 sample ID를 보존한다.
- [ ] Voxel spacing, sampling factor와 offset을 기록한다.
- [ ] 대용량 파일, credential 또는 개인 절대 경로가 없다.
- [ ] 연구 프로토콜에 영향을 주는 변경을 문서에 함께 반영했다.

## Review 기준

### Data

- Grayscale과 binary crop이 같은 좌표인가?
- Source polarity를 확인한 뒤 내부 `pore=1`로 변환하는가?
- Berea 공간 split 이후에 구역 내부에서만 crop을 생성하는가?
- Sparse input, observed mask와 primary metric mask가 정렬되는가?

### Model and loss

- Sagittal/coronal AMI가 가중치를 공유하는가?
- FDM이 physical spacing을 사용하는가?
- Spatial size가 `512` 등에 하드코딩되지 않았는가?
- E3 observed slice를 통해 gradient가 흐르지 않는가?
- Empty/constant phantom에서 NaN이 발생하지 않는가?

### Evaluation

- Primary 영상지표가 bracketed missing mask를 사용하는가?
- 모든 모델에 동일한 segmentation과 solver를 사용하는가?
- Slice나 voxel을 독립 통계 표본으로 취급하지 않는가?
- Non-percolating prediction을 임의의 작은 permeability로 대체하지 않는가?

주 실험의 데이터/설정 변경은 Berea spatial protocol을 따른다. 정규화 통계에는 train 구역만 포함한다.
