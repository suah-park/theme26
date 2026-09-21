# SAINT 기반 Sparse-slice Digital Rock 3D Reconstruction 최종 연구 프로토콜

- 작성일: 2026-09-21
- 연구 단계: 포스터용 최소 검증 연구
- 주 데이터: Binary pore/solid Micro-CT volume
- 핵심 기반 모델: SAINT (Spatially Aware Interpolation NeTwork)
- 최종 물성 지표: z 방향 단상 투과도 보존

---

## 0. 최종 연구 결정 요약

이번 연구에서는 새로운 direct 3D backbone을 처음부터 설계하지 않는다. 먼저 SAINT의 sparse-to-dense volume reconstruction 구조를 유지하면서 binary digital rock에 맞게 입력, loss 및 평가 체계를 수정한다.

필수 비교는 다음과 같다.

1. **A0. Linear interpolation**: 비학습 기준선
2. **A1. Independent 2D reconstruction**: 누락 단면을 한 장씩 독립적으로 복원
3. **B. SAINT-Rock**: SAINT의 sagittal/coronal super-resolution 및 residual fusion 구조를 binary digital rock에 적용
4. **C. SAINT-Rock + topology-aware loss**: B 완료 후 여건이 허용되면 수행하는 확장 실험

필수 연구 질문은 다음과 같다.

> 동일한 z축 희소 3D Micro-CT volume이 주어졌을 때, SAINT 기반 cross-view volume reconstruction이 독립적인 2D 단면 복원보다 3차원 공극 연결성과 투과도를 더 잘 보존하는가?

SOTA를 능가한다고 주장하지 않는다. 이번 포스터의 목적은 **SAINT를 binary digital rock 문제에 적용했을 때 joint/cross-view reconstruction이 구조 및 물성 보존에 실제로 유효한지 검증하는 것**이다.

---

## 1. 연구 배경

고해상도 Micro-CT는 촬영 시간, 저장 용량, 해상도 및 field of view 사이에 trade-off가 있다. 그 결과 z축 방향으로 모든 단면을 촘촘히 획득하기 어려운 상황이 발생할 수 있다.

관측된 단면이 일정한 간격으로 존재할 때, 목표는 단순히 한 장의 가운데 단면을 예측하는 것이 아니라 관측 단면 사이의 모든 누락 단면을 복원해 dense 3D pore volume을 얻는 것이다.

Digital rock에서는 시각적으로 비슷한 영상보다 다음 항목이 더 중요하다.

- 공극률이 보존되는가?
- 작은 pore throat가 끊기거나 가짜로 연결되지 않는가?
- inlet과 outlet 사이의 percolating pore가 유지되는가?
- 복원된 구조에서 계산한 투과도가 원본과 일치하는가?

따라서 본 연구에서는 영상 유사도뿐 아니라 morphology, topology, connectivity 및 permeability를 단계적으로 검증한다.

---

## 2. 연구 목적

### 2.1 가장 큰 목적

> 희소하게 관측된 binary Micro-CT volume으로부터 z축 누락 단면을 복원하고, SAINT 기반 cross-view reconstruction이 독립적인 2D reconstruction보다 3차원 공극 연결성과 투과도 보존에 유리한지 검증한다.

### 2.2 세부 목적

1. Linear interpolation, Independent 2D 및 SAINT-Rock을 동일한 데이터와 관측 조건에서 비교한다.
2. 관측 간격이 증가할 때 복원 성능이 어떻게 저하되는지 확인한다.
3. 높은 missing-slice Dice가 높은 공극 연결성 및 투과도 정확도로 이어지는지 분석한다.
4. Euler characteristic, percolating porosity 등의 구조 지표 중 투과도 오차와 가장 관련성이 높은 지표를 탐색한다.
5. 기본 SAINT-Rock 결과가 확보되면 topology-aware loss의 필요성과 적용 가능성을 판단한다.

### 2.3 연구 가설

- **H1:** SAINT-Rock은 Independent 2D보다 z축 방향의 단면 간 연속성을 더 잘 보존한다.
- **H2:** SAINT-Rock은 Independent 2D보다 percolating porosity 및 z 방향 투과도 오차가 작다.
- **H3:** 관측 간격이 증가하면 모든 모델의 성능이 저하되지만, SAINT-Rock의 구조·물성 오차 증가 폭이 상대적으로 작다.
- **H4:** 투과도 오차는 단순 Dice보다 percolating porosity 또는 topology 오차와 더 밀접하게 연관된다.
- **확장 가설 H5:** Topology-aware loss를 추가하면 Dice를 크게 희생하지 않으면서 연결성과 투과도 보존을 개선할 수 있다.

---

## 3. 선행연구와 연구 포지셔닝

### 3.1 SAINT의 역할

SAINT는 z축 해상도만 낮은 anisotropic CT volume을 입력받아 dense volume을 생성한다. Sparse axial volume을 sagittal 및 coronal 방향의 저해상도 2D 영상으로 재구성한 후, AMI로 z축 super-resolution을 수행하고 RFN으로 두 방향의 결과를 융합한다.

본 연구는 SAINT에서 다음을 가져온다.

- z축만 희소한 3D volume이라는 문제 설정
- 여러 missing slice를 포함한 dense volume 복원
- sagittal/coronal cross-view 정보 활용
- 서로 다른 관측 간격에 대한 확장 가능성

### 3.2 본 연구의 차별점

SAINT 원 논문은 grayscale 의료영상을 대상으로 하며 L1, PSNR 및 SSIM을 중심으로 평가한다. 본 연구는 다음과 같이 디지털 암석 문제로 전환한다.

| 구분 | SAINT 원 연구 | 본 연구 |
|---|---|---|
| 데이터 | Grayscale medical CT | Binary pore/solid Micro-CT |
| 낮은 해상도 축 | z축 | z축 |
| 구조 | AMI + RFN | 동일한 SAINT 기반 구조 |
| 기본 loss | L1 | Missing-position BCE + Dice |
| 영상 평가 | PSNR, SSIM | Missing-slice Dice, IoU |
| 물리 평가 | 없음 | Porosity, topology, connectivity, permeability |
| 최종 목적 | 영상 품질 개선 | 3D 공극 구조 및 물성 보존 검증 |

### 3.3 SOTA에 대한 주장 범위

본 연구는 SOTA를 능가한다고 주장하지 않는다. 의료영상 slice interpolation에는 SAINT 이후 I3Net 등의 최신 모델이 있으며, digital rock의 sparse-slice reconstruction에는 vMuF-DiM 및 GeoTopoDiff와 같은 직접적인 최근 연구가 존재한다.

이번 포스터에서는 이들을 관련 연구로 인용하되, 필수 실험에서 모두 재현하지 않는다. 따라서 다음 수준으로 주장한다.

> SAINT 기반 sparse-to-dense reconstruction을 binary digital rock에 적용하고, 기존 영상 중심 평가를 3D 공극 연결성과 투과도 평가까지 확장하였다.

### 3.4 주요 참고문헌

1. Peng, C., Lin, W.-A., Liao, H., Chellappa, R., & Zhou, S. K. (2020). SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis. *CVPR*. <https://doi.org/10.1109/CVPR42600.2020.00777>
2. Song, H., Mao, X., Yu, J., Li, Q., & Wang, Y. (2024). I3Net: Inter-Intra-slice Interpolation Network for Medical Slice Synthesis. *IEEE Transactions on Medical Imaging*, 43(9), 3306-3318. <https://doi.org/10.1109/TMI.2024.3394033>
3. Hou, S., Hou, Z., & Cao, D. (2026). 3D digital core reconstruction from limited core-scanned images: An improved diffusion model with multi-modal information fusion. *Geoenergy Science and Engineering*, 258, 214309. <https://doi.org/10.1016/j.geoen.2025.214309>
4. Shi, Y., Wang, P., Yu, M., et al. (2026). GeoTopoDiff: Learning Geometry-Topology Graph Priors through Boundary-Constrained Mixed Diffusion for Sparse-Slice 3D Porous Reconstruction. *arXiv:2605.03764*. <https://arxiv.org/abs/2605.03764>
5. Clough, J. R., Byrne, N., Oksuz, I., et al. (2022). A Topological Loss Function for Deep-Learning Based Image Segmentation Using Persistent Homology. *IEEE TPAMI*, 44(12), 8766-8778. <https://doi.org/10.1109/TPAMI.2020.3013679>
6. Mosser, L., Dubrule, O., & Blunt, M. J. (2017). Reconstruction of Three-Dimensional Porous Media Using Generative Adversarial Neural Networks. *Physical Review E*, 96, 043309. <https://doi.org/10.1103/PhysRevE.96.043309>

---

## 4. 문제 정의 및 입출력

### 4.1 Dense volume

원본 binary volume을 다음과 같이 정의한다.

\[
V_{\mathrm{dense}}\in\{0,1\}^{H\times W\times D}
\]

- 1: pore
- 0: solid

### 4.2 Sparse input

z축을 간격 \(r\)로 관측한다.

\[
V_{\mathrm{sparse}}
=V_{\mathrm{dense}}[:,:,0:r:D]
\]

\[
V_{\mathrm{sparse}}
\in\{0,1\}^{H\times W\times D_s}
\]

이번 연구에서는 local sparse volume마다 관측 단면을 5장 사용한다.

#### 관측 간격 \(r=2\)

```text
Sparse input : z0, z2, z4, z6, z8
Dense target : z0, z1, z2, ..., z7, z8
```

\[
5\times H\times W
\rightarrow
9\times H\times W
\]

#### 관측 간격 \(r=4\)

```text
Sparse input : z0, z4, z8, z12, z16
Dense target : z0, z1, z2, ..., z15, z16
```

\[
5\times H\times W
\rightarrow
17\times H\times W
\]

### 4.3 최종 출력

모델은 각 voxel의 pore probability를 출력한다.

\[
\hat P\in[0,1]^{H\times W\times D}
\]

Primary evaluation에서는 모든 모델에 동일한 threshold 0.5를 적용한다.

\[
\hat V=\mathbb{1}(\hat P\ge0.5)
\]

관측 단면은 추론 후 원본 입력값으로 강제 교체한다.

\[
\hat V[:,:,z_{\mathrm{obs}}]
=V_{\mathrm{dense}}[:,:,z_{\mathrm{obs}}]
\]

---

## 5. 데이터 준비 및 분할

### 5.1 Primary setting

- Segmentation이 완료된 binary volume 사용
- 복원 오류와 segmentation 오류를 분리
- Grayscale reconstruction은 후속 robustness experiment로 보류

### 5.2 Train/validation/test 분할

가장 좋은 방법은 parent core 단위 분할이다.

```text
Train      : Core A, B, C, ...
Validation : 별도 Core 또는 별도 spatial block
Test       : 완전히 분리된 Core 또는 spatial block
```

코어 수가 부족하면 전체 volume을 큰 공간 블록으로 먼저 나누고, 블록 사이에 최소 patch 크기 이상의 guard band를 둔다.

금지 사항:

- 전체 volume에서 overlapping patch를 모두 만든 후 무작위 train/test 분할
- 동일 parent volume의 인접 영역을 train과 test에 혼합
- test 결과를 보고 threshold, epoch 또는 loss weight 변경

### 5.3 Patch 생성

초기 권장 크기:

\[
H=W=128
\]

GPU 메모리가 충분하면 \(256\times256\)을 추가 검토한다.

Patch sampling은 다음을 균형 있게 포함한다.

- 일반 무작위 영역
- 공극률이 충분한 영역
- pore throat가 존재하는 영역
- z 방향 연결 공극이 있는 영역

고체가 대부분인 쉬운 patch만 과도하게 포함되지 않도록 한다.

### 5.4 데이터 증강

허용:

- x-y flip
- x-y 평면 90도 회전
- 필요 시 작은 crop 위치 변화

주의:

- 암석이 이방성이면 z축과 x/y축을 교환하지 않는다.
- topology를 인위적으로 바꾸는 elastic deformation은 초기 실험에서 사용하지 않는다.

---

## 6. 비교 모델

### 6.1 A0. Linear interpolation

인접한 관측 단면 사이를 상대 위치에 따라 선형 보간한다.

\[
\hat I_{z+j}
=
\left(1-\frac{j}{r}\right)I_z
+
\frac{j}{r}I_{z+r}
\]

Probability를 생성한 뒤 동일한 threshold 0.5로 이진화한다.

역할:

- 딥러닝을 사용하지 않은 sanity baseline
- 학습 모델이 단순 보간보다 실제로 나은지 확인

### 6.2 A1. Independent 2D reconstruction

각 target axial slice를 독립적으로 출력한다. 공정성을 위해 전체 local sparse stack을 channel로 제공하고 목표 위치 정보를 추가할 수 있다.

입력 예시:

\[
[I_{z_0},I_{z_1},I_{z_2},I_{z_3},I_{z_4},T_j]
\]

- 첫 5개 channel: 관측 단면
- \(T_j\): 복원하려는 target z 위치를 나타내는 coordinate map

출력:

\[
\hat I_j\in[0,1]^{H\times W}
\]

모든 target을 독립적으로 예측한 후 z 순서대로 적층한다.

역할:

- 각 slice를 개별적으로 예측하는 기존 접근의 기준선
- B와 비교하여 cross-view/joint volume reconstruction 효과 확인

### 6.3 B. SAINT-Rock

SAINT의 구조를 유지하되 binary digital rock에 맞게 학습한다.

```text
Sparse binary 3D volume
        ↓
Sagittal / Coronal reformatting
        ↓
Shared AMI
        ↓
Sagittal SR volume + Coronal SR volume
        ↓
Axial reformatting
        ↓
Residual-Fusion Network
        ↓
Dense pore-probability volume
```

#### AMI

Sparse volume을 sagittal 및 coronal 방향에서 z축 해상도가 낮은 2D 영상으로 본다.

\[
I_{\mathrm{sag}}^x
=G_\theta(I_{\downarrow r}^{x})
\]

\[
I_{\mathrm{cor}}^y
=G_\theta(I_{\downarrow r}^{y})
\]

AMI는 다음 구성요소를 포함한다.

1. Residual-dense feature learning
2. 관측 간격 및 물리적 voxel distance를 표현하는 Filter Distance Matrix
3. 목표 위치별 dynamic filter generation
4. z축 anisotropic sub-pixel convolution

#### RFN

Sagittal 및 coronal 결과의 axial slice를 평균한 후 residual을 학습한다.

\[
I_{\mathrm{avg}}^z
=\frac{I_{\mathrm{sag}}^z+I_{\mathrm{cor}}^z}{2}
\]

\[
I_{\mathrm{fuse}}^z
=I_{\mathrm{avg}}^z
+F_\theta(I_{\mathrm{sag}}^z,I_{\mathrm{cor}}^z)
\]

### 6.4 C. SAINT-Rock + topology-aware loss

필수 A0/A1/B 실험이 완료된 후 수행하는 확장 모델이다. Architecture는 B와 동일하고 loss만 변경한다.

\[
\mathcal L_C
=\mathcal L_{\mathrm{rec}}
+\lambda_{\mathrm{topo}}\mathcal L_{\mathrm{topo}}
\]

Topology loss 후보:

- 3D persistent-homology/Betti matching loss
- Cubical-complex 기반 topology discrepancy
- 계산량이 큰 경우 연결성에 특화된 differentiable surrogate

Topology loss는 개별 2D slice가 아니라 최종 3D reconstructed volume에 적용한다.

### 6.5 이번 단계에서 보류하는 모델

Direct 3D encoder-decoder 또는 diffusion model은 이번 포스터의 필수 범위에서 제외한다. SAINT-Rock 결과가 확보된 후 다음 연구 질문으로 확장한다.

> Cross-view 2D super-resolution 기반 SAINT보다 direct 3D reconstruction이 공극 topology와 물성 보존에 유리한가?

---

## 7. Loss 설계

### 7.1 Missing-position mask

관측 단면은 정답을 이미 알고 있으므로 loss는 누락 위치를 중심으로 계산한다.

\(r=4\) 예시:

```text
Observed mask: 1 0 0 0 1 0 0 0 1 ...
Missing mask : 0 1 1 1 0 1 1 1 0 ...
```

### 7.2 Reconstruction loss

\[
\mathcal L_{\mathrm{rec}}
=
\mathcal L_{\mathrm{BCE}}
+\lambda_D\mathcal L_{\mathrm{Dice}}
\]

초기 설정:

\[
\lambda_D=1
\]

두 loss 모두 missing 위치에서 계산한다.

\[
\mathcal L_{\mathrm{BCE}}
=
\mathrm{BCE}
(M_{\mathrm{miss}}\odot\hat P,
M_{\mathrm{miss}}\odot V_{GT})
\]

\[
\mathcal L_{\mathrm{Dice}}
=1-
\frac{
2\sum M_{\mathrm{miss}}\hat P V_{GT}+\epsilon
}{
\sum M_{\mathrm{miss}}\hat P+
\sum M_{\mathrm{miss}}V_{GT}+\epsilon
}
\]

### 7.3 Loss 공정성

Architecture 비교가 목적이므로 A1과 B에는 동일한 BCE+Dice를 사용한다. SAINT 원 논문의 L1 버전은 시간이 허용될 때 보조 ablation으로만 추가한다.

---

## 8. 학습 프로토콜

### 8.1 공통 권장 설정

| 항목 | 초기값 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | \(1\times10^{-4}\) |
| Weight decay | \(1\times10^{-5}\) |
| Batch size | 2-4, GPU 메모리에 따라 조정 |
| Max epochs | 200 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler patience | 7-10 epochs |
| LR reduction factor | 0.5 |
| Minimum LR | \(1\times10^{-6}\) |
| Early-stopping patience | 20 epochs |
| 최소 random seeds | 3 |

Batch size가 작으면 BatchNorm보다 InstanceNorm 또는 GroupNorm을 우선 검토한다.

### 8.2 관측 간격별 학습

초기 포스터에서는 \(r=2\)와 \(r=4\)를 별도 모델로 학습한다.

이유:

- Architecture 효과와 multi-scale training 효과를 분리
- 결과 해석 단순화
- 각 관측 간격에서 충분한 수렴 확인

하나의 모델이 여러 \(r\)을 처리하는 실험은 후속 확장으로 둔다.

### 8.3 A1 학습

1. Sparse stack 및 target coordinate 생성
2. Target axial slice 한 장 출력
3. BCE+Dice 계산
4. 모든 missing target에 대해 반복
5. Validation missing-slice Dice로 checkpoint 선택

### 8.4 B: SAINT-Rock Stage 1 - AMI

1. Sparse volume을 sagittal 및 coronal slice로 재구성
2. 인접 slice context를 포함해 AMI 입력 구성
3. Shared AMI로 두 방향 z축 super-resolution
4. Missing 위치에서 BCE+Dice 계산
5. Validation missing-slice Dice로 early stopping
6. Best AMI checkpoint 저장

### 8.5 B: SAINT-Rock Stage 2 - RFN

1. Best AMI를 고정
2. Train/validation volume의 sagittal 및 coronal SR volume 생성
3. 동일 z 위치의 axial 결과를 RFN 입력으로 구성
4. RFN이 평균 영상에 추가할 residual 학습
5. Final fused output에서 BCE+Dice 계산
6. Validation missing-slice Dice로 early stopping
7. Best RFN checkpoint 저장

### 8.6 C: Topology fine-tuning

1. 학습 완료된 B의 checkpoint에서 시작
2. 낮은 learning rate 사용, 예: \(10^{-5}\)
3. Final fused 3D output에서 topology loss 계산
4. BCE+Dice를 유지하여 voxel fidelity 붕괴 방지
5. \(\lambda_{\mathrm{topo}}\)는 validation set에서 소수 후보만 비교
6. A1/B와 동일한 checkpoint selection rule 적용

---

## 9. 학습 종료 및 checkpoint 선택

### 9.1 Primary selection metric

모든 학습 모델의 공통 기준은 다음이다.

\[
\boxed{\mathrm{Validation\ Missing\text{-}slice\ Dice}}
\]

### 9.2 Early stopping

- 매 epoch 후 validation missing-slice Dice 계산
- 이전 최고값보다 최소 0.001 이상 상승해야 개선으로 간주
- 20 epoch 동안 개선이 없으면 학습 종료
- 마지막 epoch가 아니라 validation Dice가 가장 높았던 checkpoint 사용

### 9.3 금지 사항

- Train loss만 보고 종료
- Test Dice가 좋은 epoch 선택
- Test permeability를 보고 checkpoint 변경
- 모델마다 서로 다른 threshold를 test set에서 탐색

### 9.4 Permeability를 early-stopping에 사용하지 않는 이유

- 계산 비용이 높음
- 모델 선택에 사용하면 최종 물성 검증의 독립성이 약해짐
- 본 연구의 질문 중 하나가 영상 지표로 선택한 모델이 물성도 보존하는지 확인하는 것이기 때문

---

## 10. 추론 및 전체 test volume 구성

학습은 local patch로 수행하지만 물성 평가는 충분히 큰 3D test block에서 수행한다.

1. Dense test block에서 매 \(r\)번째 단면만 유지
2. Sparse local stack을 sliding 방식으로 구성
3. 각 모델로 missing probability 예측
4. x-y overlap 영역은 weighted averaging
5. 관측 단면은 원본값으로 교체
6. Threshold 0.5로 이진화
7. 충분히 큰 dense reconstructed test block 생성

몇 장짜리 얇은 local stack에서 계산한 permeability는 경계 효과가 크므로 headline 결과로 사용하지 않는다. 가능하면 \(128^3\) 이상 크기의 test block을 사용하고 동일 크기의 GT와 비교한다.

---

## 11. 평가 프로토콜

평가는 두 단위로 분리한다.

### 11.1 Missing-slice 평가

제거했던 위치만 평가한다. 관측 단면을 포함하지 않는다.

Primary metric:

\[
\boxed{\mathrm{Dice}_{\mathrm{missing}}}
\]

Supporting metrics:

- IoU
- BCE 또는 voxel error
- SSIM, 필요 시 보조 지표
- 관측 단면에서의 z 거리별 성능

### 11.2 전체 reconstructed 3D volume 평가

관측 단면은 원본을 사용하고 missing 단면은 예측값을 사용한다.

#### Morphology

- Absolute porosity error

\[
|\Delta\phi|=|\hat\phi-\phi|
\]

- Specific surface area error
- Pore-size distribution distance
- x/y/z 방향 two-point correlation distance

#### Topology 및 connectivity

- Euler characteristic density error

\[
|\Delta\chi|
\]

- Number of connected components
- Largest connected component ratio
- z 방향 percolating porosity error

\[
|\Delta\phi_{\mathrm{perc},z}|
\]

- GT는 percolating하지만 prediction은 끊어진 disconnection rate

### 11.3 Permeability 평가

복원 축이 z축이므로 primary physical metric은 z 방향 단상 투과도이다.

\[
\boxed{
E_{k_z}
=
|\log_{10}\hat k_z-\log_{10}k_z|
}
\]

Secondary:

- \(k_x\), \(k_y\)
- Directional anisotropy preservation
- Relative permeability error, 단 \(k_{GT}>0\)이고 안정적으로 정의될 때만 사용

모든 모델에 동일하게 적용할 조건:

- 동일 pore-scale flow solver
- 동일 압력차
- 동일 유체 점도
- 동일 inlet/outlet 및 측면 경계조건
- 동일 convergence criterion
- 동일 test block 크기와 voxel spacing

GT는 z 방향으로 연결되어 있으나 prediction이 연결되지 않는 경우 임의의 작은 \(k\)를 넣지 않는다. 이를 **percolation failure**로 별도 집계하고, 양쪽 모두 유동 가능한 경우에만 조건부 \(E_{k_z}\)를 계산한다.

### 11.4 포스터 headline metrics

포스터 전면에는 다음 다섯 개만 사용한다.

\[
\boxed{
\mathrm{Dice}_{miss},\quad
|\Delta\phi|,\quad
|\Delta\chi|,\quad
|\Delta\phi_{\mathrm{perc},z}|,\quad
|\Delta\log k_z|
}
\]

---

## 12. Threshold 및 robustness

Primary 결과는 모든 모델에 threshold 0.5를 동일하게 적용한다.

추가 robustness:

- threshold 0.4
- threshold 0.5
- threshold 0.6

주요 결론이 threshold 변화에도 유지되는지 확인한다. Test set에서 모델별 최적 threshold를 고르는 것은 금지한다.

---

## 13. 통계 분석

### 13.1 비교 단위

개별 voxel이나 slice가 아니라 **독립 test block 또는 parent core**를 통계 단위로 사용한다.

### 13.2 주요 비교

- A0 vs A1: 학습 기반 복원의 효과
- A1 vs B: SAINT cross-view/joint reconstruction의 효과
- B vs C: topology-aware loss의 효과
- \(r=2\) vs \(r=4\): 관측 간격 효과

### 13.3 보고 형식

- 평균 ± 표준편차
- 중앙값 및 사분위 범위, 분포가 치우친 경우
- Block 또는 core 단위 bootstrap 95% confidence interval
- 동일 test block을 사용한 paired difference
- 필요 시 paired Wilcoxon signed-rank test

### 13.4 구조 지표와 투과도 관계

각 test block에 대해 다음 값을 구성한다.

\[
\{1-\mathrm{Dice},
|\Delta\phi|,
|\Delta\chi|,
|\Delta\phi_{\mathrm{perc},z}|,
|\Delta\log k_z|\}
\]

분석:

1. 각 구조 오차와 \(|\Delta\log k_z|\)의 Spearman correlation
2. 관측 간격별 correlation
3. 암석 종류별 correlation, 데이터가 충분한 경우
4. Dice보다 connected porosity가 permeability error를 더 잘 설명하는지 비교

---

## 14. 최소 실험 매트릭스

### Phase 1: 필수 포스터 실험

| 관측 간격 | A0 Linear | A1 Independent 2D | B SAINT-Rock |
|---:|:---:|:---:|:---:|
| \(r=2\) | O | O | O |
| \(r=4\) | O | O | O |

- A1/B는 최소 3 seeds
- 동일 split 및 sparse sampling 사용
- 모든 모델에 threshold 0.5
- 충분히 큰 test block에서 연결성 및 \(k_z\) 평가

### Phase 2: 시간과 계산 자원이 허용될 때

| 관측 간격 | C SAINT-Rock + Topology |
|---:|:---:|
| \(r=2\) | 선택 |
| \(r=4\) | 우선 수행 |

Topology 효과가 잘 드러날 가능성이 높은 \(r=4\)를 먼저 수행한다.

### 필수 범위에서 제외

- GeoTopoDiff 완전 재현
- vMuF-DiM 완전 재현
- Direct 3D encoder-decoder
- Diffusion model 개발
- Grayscale reconstruction
- 불규칙 z 간격
- 여러 배율을 하나의 모델로 처리

---

## 15. 연구 성공 및 결과 해석 기준

### 15.1 가장 이상적인 결과

- B가 A1보다 missing Dice가 높음
- B가 A1보다 percolating porosity error가 낮음
- B가 A1보다 \(E_{k_z}\)가 낮음

해석:

> SAINT 기반 cross-view reconstruction이 voxel fidelity와 3D 물성 보존을 동시에 개선하였다.

### 15.2 Dice는 비슷하고 물성만 개선

해석:

> 유사한 voxel accuracy에서도 reconstruction 방식에 따라 공극 연결성과 투과도 보존 성능이 달라졌다.

이 결과는 digital rock 관점에서 매우 의미가 있다.

### 15.3 Dice만 개선되고 물성은 개선되지 않음

해석:

> 영상 유사도의 향상이 물성 정확도를 보장하지 않으며, topology-aware constraint가 추가로 필요하다.

이 경우 C 모델의 필요성이 강해진다.

### 15.4 SAINT-Rock이 Independent 2D보다 좋지 않음

가능한 원인:

- Binary pore structure가 grayscale anatomical structure와 다름
- 작은 pore throat가 L1/BCE/Dice에서 충분히 강조되지 않음
- Training data 또는 context depth 부족
- AMI의 cross-view interpolation이 암석의 불규칙 구조에 부적합

해석:

> 의료영상에서 효과적인 cross-view interpolation이 binary pore topology 보존을 자동으로 보장하지 않는다.

이 역시 연구 질문에 대한 유효한 답이다.

---

## 16. 포스터 구성

### 16.1 제목 후보

**Property-aware Evaluation of SAINT-based Sparse-slice Reconstruction for 3D Digital Rocks**

또는

**SAINT-based Dense 3D Digital Rock Reconstruction from Sparse Micro-CT Slices**

### 16.2 Background & Objective

- Micro-CT의 해상도/촬영/저장 trade-off
- Sparse z slices에서 dense volume 복원 필요
- Digital rock에서는 connectivity와 permeability가 중요
- 핵심 연구 질문 한 문장

### 16.3 Method

한 장의 pipeline 그림:

```text
Dense GT
  ↓ z축 sparse sampling
Sparse binary volume
  ↓
Linear / Independent 2D / SAINT-Rock
  ↓
Dense reconstructed volume
  ↓
Image → Morphology → Connectivity → Permeability
```

### 16.4 Results

필수 시각자료:

1. GT/A0/A1/B의 동일 axial slice 비교
2. 3D pore rendering 또는 connected component visualization
3. 다섯 headline metric 비교 그래프
4. Connected porosity error 대 permeability error 산점도

### 16.5 Conclusion & Significance

세 문장 구조:

1. 무엇을 비교했는가?
2. 구조 및 물성 측면에서 무엇을 발견했는가?
3. Sparse Micro-CT 기반 digital rock workflow에 어떤 의미가 있는가?

---

## 17. 결과 서술 템플릿

### B가 A1보다 우수한 경우

> SAINT-Rock은 Independent 2D reconstruction보다 missing-slice overlap과 z축 연속성을 개선하였다. 특히 두 모델의 Dice 차이보다 percolating porosity 및 z 방향 투과도 오차의 차이가 크게 나타났다. 이는 단면을 독립적으로 복원하는 방식보다 cross-view volume reconstruction이 3차원 공극 경로 보존에 유리함을 시사한다.

### Dice만 개선된 경우

> SAINT-Rock은 missing-slice Dice를 개선했지만 permeability error는 유의하게 감소하지 않았다. 이는 voxel-level reconstruction accuracy가 pore-throat connectivity와 transport property preservation을 직접 보장하지 않음을 보여준다.

### 관측 간격 결과

> 관측 간격이 \(r=2\)에서 \(r=4\)로 증가함에 따라 모든 모델의 성능이 저하되었다. 모델 간 차이는 Dice보다 connected porosity와 permeability에서 더 크게 나타났다.

### Topology loss가 효과적인 경우

> Topology-aware fine-tuning은 BCE+Dice만 사용한 SAINT-Rock과 유사한 missing-slice Dice를 유지하면서 Euler characteristic, percolating porosity 및 permeability error를 개선하였다.

---

## 18. 주장 시 주의사항

- Sparse boundary information만으로 실제 interior가 유일하게 결정된다고 주장하지 않는다.
- “True internal structure를 완벽히 복원했다”는 표현을 사용하지 않는다.
- “SOTA를 달성했다”는 표현은 동일 benchmark에서 최신 모델을 재현하지 않는 한 사용하지 않는다.
- 높은 Dice가 높은 permeability accuracy를 보장한다고 가정하지 않는다.
- 2D slice topology와 3D connectivity를 동일하게 해석하지 않는다.
- 작은 local stack에서 계산한 permeability를 전체 암석 물성처럼 주장하지 않는다.
- GeoTopoDiff는 직접 관련 연구이지만 preprint임을 명시한다.

권장 주장:

> Held-out binary Micro-CT volume에서 SAINT 기반 sparse-slice reconstruction의 구조 및 물성 보존 성능을 independent reconstruction과 비교하였다.

---

## 19. 주요 리스크와 대응

| 리스크 | 대응 |
|---|---|
| SAINT 코드가 오래된 PyTorch 환경에 의존 | 먼저 소규모 inference 재현 후 현재 환경으로 단계적 포팅 |
| Binary 학습이 불안정 | BCE+Dice, pore-rich patch sampling, probability output 사용 |
| Patch leakage | Parent volume 또는 spatial block split 선행 |
| 관측 단면 복사가 성능을 부풀림 | Missing-slice metric 별도 계산 |
| 전체 volume permeability 계산 비용 | 대표적인 충분히 큰 test block에서 우선 계산 |
| 예측 volume이 non-percolating | Failure rate 별도 보고, 임의의 작은 k 사용 금지 |
| Threshold에 결과가 민감 | 0.4/0.5/0.6 sensitivity analysis |
| Topology loss 계산량 과다 | Phase 1 완료 후 r=4 중심으로 제한 수행 |
| 비교군이 최신 SOTA가 아님 | Proof-of-concept 및 domain adaptation 연구로 주장 범위 제한 |

---

## 20. 실행 순서 체크리스트

### 데이터

- [ ] Binary Micro-CT volume 목록 및 voxel spacing 정리
- [ ] Parent core/spatial block 단위 split 고정
- [ ] Train/validation/test volume 통계 확인
- [ ] \(r=2,4\) sparse sampling 함수 검증
- [ ] 관측 5장 기준 입력/정답 patch 생성
- [ ] Patch overlap 및 leakage 검사

### Baseline

- [ ] A0 Linear interpolation 구현
- [ ] A1 Independent 2D 학습 및 validation curve 저장
- [ ] A1 best checkpoint test inference

### SAINT-Rock

- [ ] SAINT 원 코드 inference 재현
- [ ] Binary input/output 포맷 적용
- [ ] AMI BCE+Dice 학습
- [ ] AMI best checkpoint 고정
- [ ] Sagittal/coronal SR volume 생성
- [ ] RFN 학습
- [ ] Final fused dense volume 생성

### 평가

- [ ] Missing-slice Dice/IoU 계산
- [ ] Porosity error 계산
- [ ] Euler characteristic density 계산
- [ ] z percolating porosity 및 disconnection 계산
- [ ] 충분히 큰 test block 구성
- [ ] 동일 solver로 GT/prediction \(k_z\) 계산
- [ ] 구조 오차와 permeability error 상관분석
- [ ] 3 seeds 결과 통합 및 confidence interval 계산

### 확장

- [ ] B 결과를 바탕으로 topology loss 필요성 판단
- [ ] r=4에서 C 우선 실험
- [ ] B/C ablation 정리

### 포스터

- [ ] 핵심 질문 한 문장 확정
- [ ] Method pipeline 제작
- [ ] 정성 결과 패널 제작
- [ ] Headline metric 그래프 제작
- [ ] 결과에 맞는 결론 문구 선택
- [ ] 주장 범위 및 한계 명시

---

## 21. 연구실 선배에게 소개할 때의 요약

> 이 연구는 z축 방향으로 희소하게 관측된 binary Micro-CT volume을 dense 3D digital rock으로 복원하는 문제입니다. 의료영상 slice interpolation 모델인 SAINT의 sagittal/coronal reconstruction과 residual fusion 구조를 binary pore reconstruction에 적용합니다. Linear interpolation 및 각 단면을 독립적으로 예측하는 2D 모델과 비교하고, missing-slice Dice뿐 아니라 공극률, Euler characteristic, percolating porosity 및 z 방향 투과도를 평가합니다. 이번 포스터의 핵심은 SOTA 경쟁보다 SAINT 기반 cross-view reconstruction이 실제로 3D 공극 구조와 물성 보존에 도움이 되는지 검증하는 것입니다. 기본 결과가 확보되면 동일 SAINT 구조에 topology-aware loss를 추가하는 방향으로 확장할 계획입니다.

선배에게 받을 피드백:

1. A0/A1/B 비교가 연구 질문 검증에 충분한가?
2. 대표 test block에서 permeability를 계산하는 범위가 현실적인가?
3. 포스터 필수 범위에 topology loss가 필요한가, 후속 확장으로 두어도 되는가?

---

## 22. 최종 한 문장

> 본 연구는 SAINT 기반 cross-view interpolation을 binary digital rock reconstruction에 적용하고, 영상 정확도에서 끝나지 않고 3차원 공극 연결성과 투과도 보존까지 검증하는 포스터 수준의 비교 연구이다.
