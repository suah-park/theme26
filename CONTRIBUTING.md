# Contributing

이 프로젝트의 우선순위는 빠른 모델 변경보다 재현 가능한 grayscale SAINT baseline입니다. 모든 변경은 [연구 프로토콜](digital_rock_sparse_slice_reconstruction_final_protocol.md)의 phase와 go/no-go gate를 따라야 합니다.

## 기본 원칙

- Primary data는 11 Sandstones filtered grayscale입니다.
- Split은 parent-core 단위이며 patch 생성보다 먼저 적용합니다.
- Primary experiment는 x4 SAINT + L1입니다.
- Gradient loss는 G1 baseline 완료 후 별도 ablation으로 추가합니다.
- Binary segmentation은 evaluation reference이며 model target이 아닙니다.
- 3D topology loss는 현재 기본 구현 범위가 아닙니다.
- Test 결과로 checkpoint, loss weight, normalization, segmentation threshold 또는 hyperparameter를 선택하지 않습니다.
- 원본 volume, patch, checkpoint, prediction 및 solver output은 Git에 commit하지 않습니다.

## 작업 절차

1. Issue에 goal, input, expected output, acceptance criteria, owner, reviewer, dependency를 작성합니다.
2. `develop`에서 역할에 맞는 branch를 생성합니다.
3. 관련 protocol section과 기존 decision record를 확인합니다.
4. 작은 단위로 구현하고 unit/smoke test를 추가합니다.
5. PR에 재현 명령, 검증 결과, 실험 영향 및 artifact 위치를 기록합니다.
6. 다른 팀의 교차 검토 후 `develop`에 병합합니다.
7. Go/no-go gate를 통과한 milestone만 `main`에 병합합니다.

## Branch 이름

```text
infra/<topic>
model/<topic>
eval/<topic>
docs/<topic>
```

예시:

```text
infra/eleven-sandstones-audit
infra/core-split-manifest
model/upstream-saint-reproduction
model/grayscale-l1
model/gradient-loss
eval/missing-slice-metrics
eval/frozen-segmentation
docs/gradient-ablation-decision
```

## Commit 예시

```text
feat(data): add x4 z-sampling with physical spacing
feat(model): port grayscale SAINT L1 baseline
feat(loss): add finite-difference gradient loss
feat(eval): add missing-slice PSNR and SSIM
fix(data): prevent parent-core split leakage
docs(protocol): record segmentation freeze decision
```

## Pull request 체크리스트

- [ ] 관련 issue와 experiment ID가 연결되어 있다.
- [ ] 입력·출력 tensor의 axis, shape, dtype 및 intensity range를 설명했다.
- [ ] Dataset variant와 split manifest version을 기록했다.
- [ ] Voxel spacing 및 sampling factor/offset을 기록했다.
- [ ] Test core 또는 test metric을 설정 선택에 사용하지 않았다.
- [ ] 최소 실행, unit test 또는 parity test를 수행했다.
- [ ] Upstream SAINT 동작에서 벗어난 부분을 명시했다.
- [ ] 대용량 파일, local path 또는 credential이 포함되지 않았다.
- [ ] 재현에 필요한 config, seed 및 commit을 기록했다.
- [ ] 결과 PR이면 raw metric과 prediction metadata 위치를 기록했다.

## 변경 시 decision record가 필요한 항목

다음 항목을 변경하려면 `docs/decisions/`에 짧은 기록을 추가합니다.

- Parent-core split 또는 test core
- Dataset variant(raw/filtered/binary)
- Axis convention
- Normalization
- Primary scale 또는 sparse sampling rule
- Primary loss 또는 checkpoint metric
- Gradient definition 또는 weight search space
- Segmentation operator
- Test ROI/block
- Permeability solver 또는 boundary condition
- Topology-loss go/no-go

Decision record에는 context, decision, alternatives, consequences, approver를 포함합니다.

## 실험 기록 필수 항목

```text
experiment_id
model_id
git_commit
upstream_saint_commit
config_name
dataset_doi
dataset_variant
source_core_id
split_manifest_version
split_manifest_hash
normalization_id
sampling_axis
sampling_factor
sampling_offset
hr_voxel_spacing_um
lr_voxel_spacing_um
roi_id
seed
loss_name
loss_weights
checkpoint_epoch
validation_missing_mae
prediction_dtype
prediction_shape
segmentation_protocol_version
```

## 외부 코드와 데이터

- 11 Sandstones는 ODC-By 1.0 attribution을 유지합니다.
- SAINT upstream repository의 commit hash를 모든 재현 실험에 기록합니다.
- SAINT upstream에는 명시적인 license file이 확인되지 않았으므로 코드를 이 repository에 그대로 복사하거나 수정본을 재배포하지 않습니다.
- 외부 코드가 필요하면 dependency, submodule, patch 또는 재구현 중 어떤 방식인지 PR에 명확히 적습니다.
- 라이선스가 불명확하면 공개 merge 전에 integration lead와 확인합니다.

## Review 기준

### Data/infra review

- File metadata와 axis가 실제로 검증되었는가?
- Split leakage가 구조적으로 불가능한가?
- Physical voxel spacing이 sampling 후 올바르게 갱신되는가?

### Model review

- G1이 원본 SAINT 동작을 불필요하게 바꾸지 않는가?
- AMI/RFN stage와 FDM input이 문서화되었는가?
- G2가 architecture가 아닌 loss만 바꾸는가?

### Evaluation review

- Observed slice가 missing-slice metric에서 제외되는가?
- Data range와 SSIM implementation이 고정되었는가?
- Segmentation과 permeability 조건이 model 간 동일한가?
- Slice를 독립 통계 표본으로 취급하지 않는가?

