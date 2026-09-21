# Contributing

## 기본 원칙

- 모든 모델은 동일한 데이터 split, sparse sampling 및 test block을 사용합니다.
- Test 결과를 이용해 checkpoint, threshold 또는 hyperparameter를 선택하지 않습니다.
- 대용량 데이터와 checkpoint를 Git에 commit하지 않습니다.
- 모델, 평가 및 데이터 interface 변경은 관련 팀에 먼저 공유합니다.

## 작업 절차

1. GitHub issue를 만들고 담당자와 완료 기준을 지정합니다.
2. `develop`에서 역할에 맞는 branch를 생성합니다.
3. 작은 단위로 구현하고 최소 검증을 추가합니다.
4. PR에 변경 목적, 검증 결과 및 실험 영향을 기록합니다.
5. 다른 팀원의 검토 후 `develop`에 병합합니다.
6. 재현 가능한 milestone만 `main`으로 병합합니다.

## Branch 이름

```text
infra/<topic>
model/<topic>
eval/<topic>
docs/<topic>
```

## PR 체크리스트

- [ ] 관련 issue가 연결되어 있다.
- [ ] 입력·출력 tensor 및 config 변경을 설명했다.
- [ ] 최소 실행 또는 unit test를 수행했다.
- [ ] 데이터 split과 test leakage에 영향을 주지 않는다.
- [ ] 대용량 파일·데이터·자격증명을 포함하지 않는다.
- [ ] 재현에 필요한 seed와 config를 기록했다.
- [ ] 평가 결과라면 raw metric의 위치를 기록했다.

## 실험 기록 필수 항목

```text
model_id
git_commit
config_name
seed
sampling_factor
train_split_version
checkpoint_epoch
validation_missing_dice
threshold
voxel_spacing
test_block_id
```

