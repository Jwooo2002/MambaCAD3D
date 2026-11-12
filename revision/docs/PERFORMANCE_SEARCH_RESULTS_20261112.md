# 구조 유지 성능 개선 — 실행 중 관찰

실행: `outputs/performance_nested_20261112T031500Z_200ep`.
고정 계획: `docs/PERFORMANCE_SEARCH_PLAN_20261112.md`.
2026-11-12 14:52 KST: 첫 fold의 P0/P1/P2 두 seed씩 평가 완료, P3 두 seed 학습 시작. Inner32회 중6회 완료이며 outer test는 아직 평가하지 않았다. 아래 결과는 후보 선택용 내부 validation과 학습 중 진단이며, 전체 k-fold 성능·주 모델 채택·개선 성공의 근거가 아니다.

## 첫 기준 P0의 고정200 epoch validation 결과

같은 정상3명·질환15명, 전체844patch를 두 모델에서 평가했다. 가중치 학습에 들어간 정상12명과는 겹치지 않는다. 질환비율은15/18=0.8333이다. 주 평가 max의 결과는 다음과 같다.

| Seed | AUROC | AP |
| --- | ---: | ---: |
| 42 | 0.311111 | 0.796806 |
| 123 | 0.266667 | 0.755536 |
| 두 seed의 기술적 평균 | 0.288889 | 0.776171 |

**이 fold의 기준 성능은 낮다.** AP도질환비율 기준0.8333보다 낮으며, 약0.8이라는 절대 숫자를 좋은 성능으로 해석하지 않는다. 같은18명의 반복 평가를36명의 독립 관측처럼 취급하지 않는다. 정상3명뿐인 inner holdout 결과이고 다른 fold·후보·outer 결과는 아직 남아 있다.

보조 집계의 AUROC도 seed42/123 각각 mean0.2222/0.2000, top5 0.2444/0.2222,95percentile0.2222/0.2222였다. 이 관찰에서 낮은 순위를 max 집계의 고립된 한 patch에만 귀속할 수 없다. 환자 max 점수의 두 seed 간 순위상관은Spearman0.98349였다. 두 seed가 비슷한 환자를 어렵게 평가했음을 보여주는 기술적 관찰이며, 모든 fold에서 seed 변동이 작다는 뜻은 아니다.

정상 N_8의 max 점수보다 높은 질환 환자는 seed42에서1/15명, seed123에서0/15명이었다. 다른 정상 N_14는8/15명, N_20은5/15명·4/15명이었다. 두 seed의 AUROC를45개 정상–질환 pair의 순위로 보면 각각14/45,12/45다. 복원 오차가 큰 정상 환자가 존재하지만, 이것만으로 그 이유가 해부학·전처리·규제 중 무엇인지 확정하지 않는다. 다음 P1과 같은 환자에서 비교한다. 점수 방향·집계·후보 선택 규칙은 사전 계획 그대로다.

전체844patch의 frozen 환자별 개수, index0…N−1의 정확히1회 포함, label, MSE/SSE 관계, max/mean/top5/percentile 재집계, AUROC/AP 재계산, checkpoint hash, 추론 중 가중치 불변 검사가 모두 통과했다. 별도 독립 순위 계산도 일치했다. 근거: `completed_audits/`, `analysis/results.json`, `first_baseline_validation.json`, 각 run의 patient/patch CSV와 `evaluation_access.json`. 두 기준 모델은13:07 KST에 종료됐고, 큐가 P1_no_decay seed42/123을 이어서 시작했다.

## 첫 fold, 기준 P0의 50 epoch 관찰

2026-11-12 12:30 KST, 두 GPU에서 50 epoch checkpoint가 저장된 후 CPU로 읽었다. 학습이나 저장 가중치를 변경하지 않았고 환자 추론은 하지 않았다. 각 seed는 같은 정상 TRAIN12명·448patch를 epoch마다 전부 사용해 11,200 update를 완료했다. 최종200 epoch 평가와 고정 후보 선택 규칙은 유지한다.

| Seed | Epoch50 TRAIN MSE | 국소 유효 슬롯 수 | 전체 조회 유효 슬롯 수 | Memory 벡터 평균 norm |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.0006053911 | 2.7064 | 3.2622 | 0.132405 |
| 123 | 0.0006396678 | 2.6328 | 3.0599 | 0.143916 |

국소 유효 슬롯은 각 attention 행의 exp(entropy)를 epoch 내 평균한 값이다. 전체 조회 유효 슬롯은 epoch 전체 attention을 누적한 분포의 exp(entropy)이다. 서로 다른 슬롯을 실제로 몇 개 사용했는지 세는 값이 아니다. 두 seed의 zero-memory 행 비율은0이었다. Memory 벡터는 초기 norm≈1에서 평균0.13–0.14로 수축했다. 벡터 간 평균 cosine은0.0156/0.0813이며, memory가 모두 같은 벡터가 됐다는 결과는 아니다.

SSM x_proj/dt_proj 가중치의 심한 수축도 보였다. Seed42의 중간 두 블록 x_proj.weight는 모든 원소가 float32 subnormal 범위였고 L2 norm은 각각3.565e-38/3.557e-38이었다(초기3.639/3.667). 최종 stage의 dt_proj.weight norm은 두 seed 모두 약7.2e-22–8.5e-22였다(초기 약13). 다른 projection에도 작은 값과 subnormal 원소가 섞여 있다. 원소별 비율·norm은 각 `weight_diagnostics/*_epoch50.json`에 보존했다.

이는 기존 coupled Adam 규제 조건에서 관찰했던 projection 수축이 **수정된 ordered_mm_v2 입력에서도** 나타났다는 관찰이다. 현재 라운드에서는 P0만 실행 중이므로 이 차이의 원인을 새 paired 대조로 분리한 결과는 아직 아니다. 또한 가중치만으로 recurrent 기능이나 이상 탐지 성능을 판정할 수 없다. 새 입력에서의 P1/P2/P3 비교, 최종 AUROC/AP와 이후 필요 시 입력 의존성 진단이 남아 있다. 정상 MSE 감소를 모델 전체가 잘 학습됐다는 근거로 삼지 않는다.

## 실행 무결성 확인

- 실제 큐 PID와 두 worker PID의 실행·부모 관계를 호스트에서 확인했다. 관측용 짧은 대기 중에도 같은 프로세스의 start time을 확인했으며 실행을 재시작하지 않았다.
- Frozen 파일 hash, 환자별 학습/평가 역할, 초기 state hash, 실제 seed, epoch/update 수, 전체 학습 로그의 유한값을 확인했다. 질환 평가 접근 기록은0이었다.
- 기준 P0가 최종 선택됐을 때 같은 run을 재사용하고 paired 차이·bootstrap CI가 정확히0이 되는 경계 사례까지 추가 검증했다. 프로토콜/지표 테스트6개가 통과했다. 학습·평가·선택 코드는 변경하지 않았다.
- 최근 실제 속도는 대략16초/epoch, 현재 두 run의 잔여 학습은약40분이다. 이 시점 최대 run 수 기준 전체 잔여 학습 추정은약30시간이며, 변동·검산 여유를 별도로 둔다. 첫 validation 점수 예상은13:10 KST 전후로 보정한다. 11월14일12:16 KST deadline은 그대로다.

관측 코드: `performance_search/audit_running.py` 및 `checkpoint_diagnostics.py`. 학습 프로세스에는 접근해 상태를 읽기만 하며, checkpoint는 CPU에서 읽는다. 관측 기록은 `runtime_audit/`, 가중치 진단은 `weight_diagnostics/`에 분리한다. 기존 frozen 학습·평가 파일은 그대로 보존한다.

## 같은 P0의 100 epoch 관찰

2026-11-12 12:47 KST 기록. 저장된100 epoch 가중치를 CPU에서 읽어50 epoch 현상이 계속되는지 확인했다. 새로운 환자 추론·optimizer update·후보 선택은 없었다.

| Seed | Epoch100 TRAIN MSE | 국소/전체 조회 유효 슬롯 | Memory 평균 norm | Memory 벡터 간 평균 cosine | Centered energy effective rank |
| --- | ---: | --- | ---: | ---: | ---: |
| 42 | 0.0004550729 | 2.1162 / 2.1585 | 0.024024 | 0.82612 | 4.5220 |
| 123 | 0.0006478154 | 2.1803 / 2.3849 | 0.022155 | 0.88762 | 3.7474 |

50 epoch의 memory 평균 norm0.1324/0.1439에서 더 수축했고 벡터 방향도 서로 가까워졌다. 일부 벡터의 norm은1 이상이므로 전체 memory가 같은0벡터라는 뜻은 아니다. Seed42의 TRAIN MSE는50 epoch보다 감소했고 seed123은약1.3% 증가했다. Memory/SSM의 상태와 정상 MSE의 방향을 동일시하지 않는다.

두 seed 모두 여섯 dt_proj.weight의 모든 원소가 float32 subnormal 범위였다. 각 tensor의 float64 계산 L2 norm은약7.9e-39–3.2e-38이다. 각 seed의 x_proj.weight도6개 중5개 tensor의 L2 norm이약2.4e-38–5.5e-38로 작았다. 첫50 epoch에 확인한 수축이 계속되었다는 관찰이며, 기능적 재귀 기여를 개입 추론으로 다시 측정한 결과는 아니다.

기존 학습·평가·선택 코드는 고정된 상태다. 별도 `verify_completed.py`를 준비해 실제 성능 결과가 나오면 frozen 환자별 patch 수, 모든 patch index의 정확히1회 포함, label, MSE/SSE 관계, 모든 집계와 AUROC/AP, checkpoint hash 및 outer 선택 확정 시점을 확인한다. 누락·중복·잘못된 label·잘못된 집계를 거부하는 검사를 포함해 테스트7개가 통과했다. 근거는 `score_audit_readiness.json`이다. 실제 성능 파일 검산은 아직 시작되지 않았다.

## 첫 P0 결과 후 추가한 평균 영상 진단

2026-11-12 13:17 KST, P0의 낮은 validation 결과를 확인한 뒤 추가한 **사후 기전 진단**이다. 후보나 모델 선택 입력으로 추가하지 않았다. Outer·외부 자료는 읽지 않았다. 정상 TRAIN12명·448patch만으로 두 개의 고정 출력을 계산한 후, 이미 평가한 동일 inner18명·844patch에서 오류를 계산했다.

- Scalar mean: 모든 학습 voxel의 평균0.48713633을 모든 위치에 동일하게 출력한다.
- Voxel mean: 학습 patch의 각 voxel 위치별 평균으로 만든 같은64×32×32 영상을 모든 입력에 출력한다.

같은 frozen float32 HU 전처리 뒤 계산은 CPU float64로 수행했다. 검증 label로 평균 영상을 맞추거나 점수 방향을 변경하지 않았다.

| 기준 | Max AUROC | Max AP |
| --- | ---: | ---: |
| P0 seed42 | 0.311111 | 0.796806 |
| P0 seed123 | 0.266667 | 0.755536 |
| 정상 TRAIN scalar mean | 0.644444 | 0.924709 |
| 정상 TRAIN voxel mean | 0.733333 | 0.949545 |

P0와 voxel-mean 기준의 환자 max 점수 순위상관은−0.0320/−0.0836이었다. P0의 오류 점수가 이 특정 평균 영상 기준과 같다는 가설은 지지되지 않는다. 다른 형태의 상수 출력 가능성이나 실제 출력 영상의 입력 의존성을 이 점수 비교만으로 판정하지 않는다.

Voxel-mean 기준 대비 P0의 환자별 평균patch 오류 비율의 중앙값은 정상에서0.0213/0.0271, 질환에서0.0167/0.0218이었다(seed42/123 순서). P0는 질환 환자 입력의 복원 오차도 크게 줄이지만 질환 순위 분리는 더 낮다. 이를 특정 병변 자체를 복원했다는 증거로 표현하지 않는다. 정상 복원 능력, 파라미터 보존, 이상 탐지 성능을 각각 확인해야 한다는 근거다.

평균 영상이 최종 모델보다 좋다고 일반화하거나 채택하지 않는다. 정상3명뿐인 한 inner fold의 사후 진단이며, 반복 CV·외부 성능은 없다. 두 고정 출력 모두 모든patch 수/index/label·집계·지표 검산을 통과했고, scalar 오류의 분해식 Var(x)+(Mean(x)−c)^2도 일치했다.

근거: `constant_reference_diagnostic_f1/protocol.json`, `fit.json`, `results.json`, `verification.json`, patient/patch CSV와 frozen `source.py`. 평균 영상 자체는 로컬 `.npy`에만 저장했고 업로드하지 않았다. 추가 진단의 validation 접근은 같은 폴더의 `evaluation_access.json`으로 별도 기록했다. 기존 학습 큐·후보 선택은 변경하지 않았다.

## P0와 P1의 같은50 epoch 비교

P1은 P0 대비 전체 Adam weight decay만1e-5→0으로 바꿨다. 같은 seed의 초기 state·patch 순서·환자 역할이 정확히 일치하고 config 차이가 weight_decay 한 개뿐임을 확인했다. 두 조건의 첫 fold50 epoch를 비교했다.

| Seed | 조건 | TRAIN MSE | 전체 조회 유효 슬롯 | Memory 평균 norm | dt projection weight norm 범위 |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | P0 | 0.00060539 | 3.2622 | 0.1324 | 7.2e-22–0.00391 |
| 42 | P1 | 0.00059542 | 2.4323 | 1.0041 | 6.0836–13.0962 |
| 123 | P0 | 0.00063967 | 3.0599 | 0.1439 | 7.2e-22–0.00445 |
| 123 | P1 | 0.00060316 | 2.2830 | 1.0029 | 6.6384–13.1361 |

P1에서는 조사한 SSM 파라미터에 subnormal 원소가 없었고 memory 가중치 크기도 초기≈1 수준으로 유지됐다. 하지만 실제 조회의 유효 슬롯은 약2.3–2.4로 여전히 집중돼 있다. 파라미터 크기 보존을 유용한 SSM 학습이나 질환 성능 향상으로 부르지 않는다. P1의 고정200 epoch validation은 아직 진행 전이며, 그 결과를 기다린다.

근거: `P1_pairing_audit.json`, `P0_P1_epoch50_comparison.json`, 네 모델의50 epoch 가중치 진단 JSON. CPU 읽기만 했고 학습 가중치를 변경하지 않았다.

## P0와 P1의 같은100 epoch 비교

2026-11-12 13:34 KST, 저장된 가중치와 TRAIN 기록만 추가 비교했다. P1의 가중치 크기 보존은100 epoch에도 유지되지만 attention 집중은 더 강해졌다.

| Seed | 조건 | TRAIN MSE | 전체 조회 유효 슬롯 | Memory 평균 norm | dt projection weight norm 범위 |
| --- | --- | ---: | ---: | ---: | --- |
| 42 | P0 | 0.00045507 | 2.1585 | 0.02402 | 7.9e-39–3.2e-38 |
| 42 | P1 | 0.00045818 | 1.5844 | 1.00442 | 6.0835–13.0973 |
| 123 | P0 | 0.00064782 | 2.3849 | 0.02216 | 7.9e-39–3.2e-38 |
| 123 | P1 | 0.00046035 | 1.4890 | 1.00459 | 6.6381–13.1434 |

P1의 검사한 SSM 파라미터에는 subnormal 원소가 없었다. 전체 조회 유효 슬롯은50 epoch의2.4323/2.2830에서100 epoch의1.5844/1.4890으로 감소했다. 이는 entropy 기반 분포 집중도이며 실제 사용하는 슬롯 개수를 세는 값이 아니다. **Decay 제거만으로 memory 조회의 집중까지 해결됐다는 근거는 없다.** 다음 P2/P3는 이미 정해 둔 entropy 약화/제거 후보이며 그대로 실행한다. P1의 최종200 epoch AUROC/AP는 아직 없고 정상 MSE·가중치 보존만으로 성능 개선을 선언하지 않는다.

환자 추론 없이 CPU에서 읽었고 checkpoint hash 불변을 확인했다. 근거: `P0_P1_epoch100_comparison.json`, `P1_epoch100_observation.json`, `weight_diagnostics/*_epoch100.json`.

완료된 후보를 P0와 같은 초기 state·patch 순서·실제 seed·환자 역할로 비교하는 별도 집계 `performance_search/compare_inner.py`도 준비했다. AUROC/AP 차이와 정상–질환 쌍의 순위 개선/악화를 기록하되 선택 규칙은 변경하지 않는다. 동률을0.5로 세는 순위 검사를 포함해 프로토콜 테스트8개가 통과했다. 현재 완료된 후보가 없어 paired 결과는0개이며, 새 helper는 frozen 학습/평가 소스에 포함하거나 실행 중 프로세스에 적용하지 않았다.

## P1 최종200 epoch 완료 — 첫 inner fold

P1 seed123은13:59:05 KST, seed42는13:59:18 KST에 학습·평가를 마쳤다. 같은 정상3명·질환15명·전체844patch에서 P0와 비교했다.

| Seed | P0 AUROC | P1 AUROC | 차이 | P0 AP | P1 AP | 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.311111 | 0.355556 | +0.044444 | 0.796806 | 0.781642 | −0.015164 |
| 123 | 0.266667 | 0.355556 | +0.088889 | 0.755536 | 0.781642 | +0.026106 |
| 기술적 평균 | 0.288889 | 0.355556 | +0.066667 | 0.776171 | 0.781642 | +0.005471 |

**Decay 제거로 첫 fold의 AUROC는 올랐지만 성능 문제는 해결되지 않았다.** P1의 AP도 질환비율0.8333보다 낮고 seed42에서는 AP가 하락했다. P1 두 seed의 max AUROC/AP는 같지만 실제 환자 점수와 checkpoint hash는 다르다. 이 결과만으로 여러 fold에서 안정적이거나 개선이 통계적으로 입증됐다고 주장하지 않는다. P2/P3·다른 fold·outer test와 최종 선택은 미완료다.

45개 정상–질환 쌍에서 seed42는4쌍 개선·2쌍 악화, seed123은4쌍 개선·악화0이었다. 정상 N_8은 두 P1 모델 모두 모든15명 질환 환자보다 오류 점수가 높아 여전히 잘못된 순위였다. 다른 정상 N_14/N_20보다 오류가 큰 질환 환자는9/15명,7/15명이다. 보조 mean AUROC도 두 seed 모두0.222222로 낮았다. 점수 방향이나 주 집계를 변경하지 않는다.

최종 TRAIN MSE는0.00045461/0.00045337, 전체 조회 유효 슬롯은1.2876/1.2541, 행별 최대 attention 평균은0.9608/0.9683이다. Memory 평균 norm은1.0055/1.0054로 유지됐고 검사한 SSM 파라미터에 subnormal 원소가 없었다. 즉 가중치 수축은 방지됐지만 조회가 강하게 집중되는 현상은 남아 있다. 유효 슬롯을 실제 슬롯 개수로 해석하거나 특정 병변의 복원 여부를 단정하지 않는다.

두 모델 모두 전체patch 수·index·label, MSE/SSE, 네 집계와 AUROC/AP, checkpoint hash·추론 중 가중치 불변 검산을 통과했다. Frozen 독립 분석과 paired 순위 집계가 일치했다. Inner4/32회 완료, outer0회다. 큐는 P2_weak_entropy 두 seed를 자동 시작했다. P0/P1/P2의 초기값·전체 patch 순서·실제 seed·환자 역할이 같고 P1→P2 설정 차이는 entropy 계수2e-4→2e-5뿐이다. P2/P3 후보와 선택 규칙은 기존 계획 그대로 유지한다.

근거: `P1_validation_observation.json`, `P2_pairing_audit.json`, `analysis/inner_paired.json`, `analysis/results.json`, `completed_audits/inner_f1_s{42,123}_P1_no_decay.json`, 두200 epoch 가중치 진단과 각 run의 원점수 CSV.

## P1과 P2의 같은50 epoch 비교

2026-11-12 14:13 KST, P2 두 seed의50 epoch 가중치와 정상 TRAIN 기록을 읽었다. P1 대비 entropy 계수만2e-4→2e-5로 낮췄고 초기값·학습 순서·환자 역할의 일치는 `P2_pairing_audit.json`에서 확인했다.

| Seed | 조건 | TRAIN MSE | 국소/전체 조회 유효 슬롯 | 행별 최대 attention 평균 | Memory 평균 norm |
| --- | --- | ---: | --- | ---: | ---: |
| 42 | P1 | 0.00059542 | 1.9320 / 2.4323 | 0.7899 | 1.0041 |
| 42 | P2 | 0.00040669 | 29.3190 / 36.8112 | 0.1422 | 1.0097 |
| 123 | P1 | 0.00060316 | 1.9212 / 2.2830 | 0.7858 | 1.0029 |
| 123 | P2 | 0.00040215 | 34.0883 / 46.2175 | 0.1257 | 1.0089 |

같은50 epoch에서 entropy 강도를10분의1로 낮춘 P2는 두 seed 모두 조회 집중이 크게 완화됐다. 정상 TRAIN MSE도 감소했다. 두 조건 모두 zero-memory 행 비율0이고, P2의 검사한 SSM 파라미터에 subnormal 원소가 없었다. 따라서 현재 비교는 단순히 메모리 가중치가 남아 있다는 관찰을 넘어 실제 학습 attention 분포가 달라졌다는 근거다.

단, entropy 기반 유효 슬롯을 실제 서로 다른 슬롯 개수로 부르지 않는다. 이 결과는 첫 fold의 중간 정상 학습 기록이고 질환 입력 추론은 추가하지 않았다. 여러 슬롯을 조회하거나 정상 복원 오차가 낮아졌다는 것만으로 이상 탐지 성능이 좋아졌다고 판단하지 않는다. P2 최종200 epoch AUROC/AP는 아직 미평가다. 후보·학습·선택 규칙은 변경하지 않았다.

CPU 가중치 읽기 전후 checkpoint hash 불변을 확인했다. 근거: `P1_P2_epoch50_comparison.json`, `P2_epoch50_observation.json`, `weight_diagnostics/inner_f1_s{42,123}_P2_weak_entropy_epoch50.json`.

## P1과 P2의 같은100 epoch 비교

2026-11-12 14:26 KST, 두 P2 모델의100 epoch checkpoint를 CPU에서 읽었다. P2는 P1보다 덜 집중돼 있지만 P2 자체에서는50→100 epoch 동안 조회가 다시 좁아졌다.

| Seed | 조건 | TRAIN MSE | 전체 조회 유효 슬롯 | 행별 최대 attention 평균 | Memory 평균 norm |
| --- | --- | ---: | ---: | ---: | ---: |
| 42 | P1 | 0.00045818 | 1.5844 | 0.9212 | 1.0044 |
| 42 | P2 | 0.00023933 | 10.6887 | 0.2573 | 1.0171 |
| 123 | P1 | 0.00046035 | 1.4890 | 0.9353 | 1.0046 |
| 123 | P2 | 0.00024534 | 15.5800 | 0.2518 | 1.0143 |

P2의 전체 조회 유효 슬롯은50 epoch36.8112/46.2175에서100 epoch10.6887/15.5800으로 감소했다. 최대 attention 평균은14.22%/12.57%에서25.73%/25.18%로 증가했다. **Entropy 약화가 조회 집중을 완화했지만 집중이 계속되지 않도록 막았다고 할 수 없다.** 두 조건의 같은100 epoch 비교와 P2의 시간 변화를 구분한다.

P2의 검사한 SSM 파라미터는 subnormal 원소가 없고 zero-memory 행도0이다. 정상 TRAIN MSE 감소나 가중치 보존은 질환 성능 개선의 근거가 아니다. 새 환자 추론·후보 변경 없이 고정200 epoch 평가를 기다린다. 근거: `P1_P2_epoch100_comparison.json`, `P2_epoch100_observation.json`, 두100 epoch 가중치 진단 JSON. 읽기 전후 checkpoint hash 불변 확인.

## Memory2048→300 회상에 대한 합성 연산 진단

과거 실패 원인을 재현한 실험은 아니다. 보존된 legacy와 현재 frozen source의 실제 MemModule 클래스를 추출해 CPU float64에서만 실행했다. 환자·encoder 특징·학습 checkpoint를 사용하지 않고 동일한 unit-norm memory bank,16개 합성 query, query norm0/1/10을 사용했다. Optimizer update나 현재 학습 큐 변경은 없다. 사용자 회상은2048이지만 보존된 모델 생성자의 기본값은2000이므로 둘 다 확인했다. 당시 실패 run의 정확한 설정·로그는 아직 찾지 못했다.

| Memory slots | 균등 softmax 값 | 현재 threshold | 현재 hard-shrink, query=0 | 과거 hardtanh, query=0 |
| --- | ---: | ---: | --- | --- |
| 300 | 0.0033333 | 0.0025 | 전체300슬롯 생존, query/memory gradient 존재 | 전체300슬롯 생존, gradient 존재 |
| 2000 | 0.0005000 | 0.0025 | 모든 행0, 출력과 query/memory gradient0 | 전체2000슬롯 생존, gradient 존재 |
| 2048 | 0.0004883 | 0.0025 | 모든 행0, 출력과 query/memory gradient0 | 전체2048슬롯 생존, gradient 존재 |

Gradient는 고정된 선형 readout probe의 미분이며 실제 reconstruction loss의 학습 gradient가 아니다. 현재 수정본의2000/2048 조건은 합성 query norm1에서도 모든 행0이었으나 norm10에서는 살아남았다. 따라서 큰 memory 자체가 언제나 학습 불가능하다는 뜻은 아니다. 실제 encoder 특징의 크기·분포는 이 진단에 포함하지 않았다.

Finite softmax와300slots에서는 최대값이 적어도1/300>0.0025이므로 모든 슬롯이 제거될 수 없다. 즉 현재 실험의 zero-memory-row=0만으로 memory가 유용하게 학습됐다고 판단할 수 없다. 한 슬롯 집중은 별도 문제다. 반대로 현재 hard-shrink 구현에서 memory만2000/2048로 늘리면 균등 조회 부근에 출력과 메모리 경로의 gradient가0인 구간이 생길 수 있다.

**이 현상을 과거2048 실패의 원인으로 확정하지 않는다.** 과거 hardtanh는 softmax를 실제 threshold로 제거하지 않아 이 dead-region 원인이 적용되지 않는다. 과거 trainer의 entropy는1024 token에서 올바른 slot entropy 대비1024/M배였고, 실제 tensor 계산은300에서3.413333배,2000에서0.512배,2048에서0.5배로 일치했다. 크기를 바꾸면 과거 규제 실효 배율도 달라지므로 용량·규제·특징 크기의 영향을 구분해야 한다.

현재 P0–P3는300slots 고정으로 계속 진행한다. 후속 크기 대조를 하게 되면 threshold와 초기 조회 분포, memory 경로 gradient, entropy 계산을 먼저 확인해야 한다. 이 진단은 새 후보나 성능 결과가 아니다. 근거: `memory_size_boundary_diagnostic/results.json`, 보존된 `source.py`, `performance_search/diagnose_memory_size_boundary.py`. 결과에 두 source와 script hash를 기록했고 source 불변을 확인했다.

## P2 최종200 epoch 완료 — 첫 inner fold

P2 seed42는14:50:16 KST, seed123은14:51:19 KST에 학습·평가를 마쳤다. 같은 정상3명·질환15명·844patch에서 P1과 직접 비교한 결과다.

| Seed | P1 AUROC | P2 AUROC | 차이 | P1 AP | P2 AP | 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.355556 | 0.355556 | 0 | 0.781642 | 0.779434 | −0.002208 |
| 123 | 0.355556 | 0.355556 | 0 | 0.781642 | 0.781642 | 0 |
| 기술적 평균 | 0.355556 | 0.355556 | 0 | 0.781642 | 0.780538 | −0.001104 |

첫 fold의 평균은 P0 AUROC/AP0.288889/0.776171, P1 0.355556/0.781642, P2 0.355556/0.780538이다. **P2는 P1 대비 주 평가 성능 개선이 없었다.** P0 대비 AUROC 평균+0.066667이지만 여전히 낮은 성능이고 AP는 질환비율0.8333보다 낮다. 첫 inner holdout 결과이며 full CV·outer 성능·최종 후보 선택은 아니다.

P1→P2에서 각 seed 모두45개 정상–질환 쌍 중1쌍 개선·1쌍 악화,43쌍 불변으로 AUROC 변화가 상쇄됐다. 정상 N_8은 두 P2에서도 모든15명 질환 환자보다 높은 오류 점수를 받았다. 보조 mean AUROC는0.266667/0.244444로 P1보다0.044444/0.022222 높지만 여전히 낮다. 보조 집계로 주 지표를 바꾸지 않는다.

P2 최종 TRAIN MSE는0.00014519/0.00017715로 P1보다68.06%/60.93% 낮았다. 그러나 전체 조회 유효 슬롯은50epoch36.8112/46.2175→100epoch10.6887/15.5800→200epoch3.1713/3.1483으로 다시 집중됐다. 최종 최대 attention 평균은70.96%/74.98%이고 memory 평균 norm은1.0231/1.0171이다. 검사한 SSM 파라미터의 subnormal 원소는 없었다. 정상 복원 개선·가중치 보존·일시적인 조회 분산이 질환 순위 개선을 보장하지 않는다는 실제 비교 근거다. 특정 병변 복원 여부나 모든 fold의 결과로 일반화하지 않는다.

두 P2 모두 전체patch 수/index/label, MSE/SSE, 네 집계·AUROC/AP, checkpoint hash·추론 중 가중치 불변 검사를 통과했다. Frozen 독립 분석도 일치했다. P0 대비 및 P1 대비 paired 순위 집계를 별도로 보존했다.

Inner6/32회 완료, outer0회. P3_no_entropy seed42는14:50:34 KST, seed123은14:51:34 KST에 시작했다. P0/P1/P2/P3의 같은 seed 초기 state·patch 순서·실제 seed·환자 역할은 같고 P2→P3 config 차이는entropy2e-5→0뿐이다. Memory300·hard shrink·SSM·decoder는 유지하며 기존 후보/선택 규칙을 변경하지 않았다.

근거: `P2_validation_observation.json`, `P3_pairing_audit.json`, `analysis/P2_vs_P1.json`, `analysis/inner_paired.json`, `analysis/results.json`, `completed_audits/inner_f1_s{42,123}_P2_weak_entropy.json`, 각 run의 원점수·학습 기록과200 epoch 가중치 진단.

## P1·P2·P3의 같은50 epoch 비교

2026-11-12 15:05 KST, 두 P3 모델의50 epoch 가중치와 정상 TRAIN 기록을 확인했다. 세 조건은 Adam decay0, 같은 초기값·patch 순서·환자 역할이며 entropy 계수만2e-4/2e-5/0이다.

| 조건 | 전체 조회 유효 슬롯, seed42/123 | 최대 attention 평균, seed42/123 | TRAIN MSE, seed42/123 |
| --- | --- | --- | --- |
| P1, entropy2e-4 | 2.4323 / 2.2830 | 78.99% / 78.58% | 0.00059542 / 0.00060316 |
| P2, entropy2e-5 | 36.8112 / 46.2175 | 14.22% / 12.57% | 0.00040669 / 0.00040215 |
| P3, entropy0 | 121.8728 / 122.5362 | 4.81% / 5.07% | 0.00041900 / 0.00039106 |

두 seed 모두50 epoch에서는 entropy를 약하게 하거나 끌수록 조회가 넓게 분산됐다. P3의 국소 유효 슬롯은90.1321/89.4542, memory 평균 norm은1.0098/1.0123, zero-memory 행 비율0이며 검사한 SSM 파라미터의 subnormal 원소는 없다. 현재까지 기록된 P3의 모든 loss가 TRAIN MSE와 같아 entropy 계수0이 실제 손실에도 반영됐음을 확인했다.

유효 슬롯은 분포의 entropy 기반 수치이며 실제 사용 슬롯 개수가 아니다. P3의 정상 MSE는 P2와 비슷하고 두 seed가 같은 방향으로 더 낮아지지는 않았다. 여러 슬롯을 조회하거나 정상 복원이 잘 된다는 사실만으로 질환 구분 향상을 결론 내리지 않는다. P2에서는50 epoch의 넓은 조회가200 epoch에 다시 좁아졌으므로 P3의 최종 상태도 확인해야 한다. P3 고정200 epoch AUROC/AP는 아직 미평가다.

CPU 가중치 읽기만 수행했고 checkpoint hash 불변, 기존 학습·선택 규칙 유지를 확인했다. 새 환자 추론은 없다. 근거: `P1_P2_P3_epoch50_comparison.json`, `P3_epoch50_observation.json`, 두 P3의50 epoch 가중치 진단 및 TRAIN history.
