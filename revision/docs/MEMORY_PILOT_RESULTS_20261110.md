# Memory 원인 분리 pilot 결과 — 2026-11-10

6/6 실행·총 120 epoch 완료. 14:19 시작, 15:05 종료(KST), 대기열 기준 45.2분. 기존 후보 R6 종료 후 GPU 0이 합류했다.

## 확인 범위

- 첫 fold의 동일한 정상 학습 15명·956개 patch, memory 300개, seed 42/123, 조건별 20 epoch. 같은 seed에서 초기 가중치·배치 순서를 공유했다.
- 모든 소스/설정 해시, 실제 epoch-0 checkpoint 초기 가중치, 마지막 checkpoint의 초기값·배치순서 해시, 20개 epoch의 patch 수 및 9,560 update, scheduler T_max=1000과 마지막 epoch=20을 확인했다.
- 저장 epoch-20 모델의 고정 45-patch 진단을 다시 실행했으며 MSE·attention 지표가 기존 JSON과 정확히 일치했다. 모든 checkpoint 진단 시점 0/1/3/5/10/20의 파일을 확인했다.
- 최종 표는 epoch-20 저장 가중치로 정상 학습 patch 956개 전체를 다시 평가한 결과다. 학습 중 epoch 평균 loss나 고정 45-patch 표와 구분한다. 학습/검증/test를 추가하거나 가중치를 갱신하지 않았다.

## 전체 정상 학습 자료의 마지막 모델 결과

| 조건 | seed | 유효 슬롯 | 출력 차원(99%) | 정상 MSE | 기준 대비 MSE | memory norm 중앙값 |
| --- | --- | --- | --- | --- | --- | --- |
| M0_baseline | 42 | 1.83 | 1 | 0.011604 | 1.000배 | 0.0443 |
| M1_entropy_off | 42 | 115.80 | 3 | 0.006971 | 0.601배 | 0.4193 |
| M2_memory_decay_off | 42 | 1.81 | 1 | 0.011664 | 1.005배 | 1.0142 |
| M0_baseline | 123 | 2.23 | 2 | 0.006767 | 1.000배 | 0.0667 |
| M1_entropy_off | 123 | 74.15 | 5 | 0.006933 | 1.025배 | 0.3904 |
| M2_memory_decay_off | 123 | 1.81 | 1 | 0.011041 | 1.632배 | 1.0119 |

유효 슬롯은 exp(평균 query entropy)이며 독립적인 정상 패턴 수가 아니다. 출력 차원은 memory readout의 전역 중심화 분산 99%를 설명하는 차원 수다. 둘 다 정상 학습 자료상의 기전 지표이며 독립 환자 수나 이상 탐지 성능이 아니다.

## 해석

1. **λ=0은 초기 쏠림을 두 seed 모두 완화했다.** 기준 유효 슬롯 1.83/2.23이 115.80/74.15로 늘고, readout 차원은 1/2에서 3/5로 늘었다. 정상 복원 MSE는 seed 42에서 39.9% 감소했지만 seed 123에서는 2.5% 증가했다. 환자별 MSE도 각각 15/15명, 2/15명에서만 감소했다. 복원 개선이 두 seed에서 일관되었다고 말할 수 없다.
2. **Memory에만 weight decay를 끄는 것으로는 조회 집중이 풀리지 않았다.** memory norm 중앙값은 기준 0.044/0.067에서 1.014/1.012로 유지되고 memory rank99도 121/121이었지만, 유효 슬롯은 약 1.81개, readout 차원은 두 seed 모두 1이었다. MSE는 각각 0.5%, 63.2% 증가했다. 벡터의 크기·다양성과 실제 전달 표현을 분리해야 한다.
3. **λ=0도 해결 완료는 아니다.** 최빈 선택을 받은 슬롯 종류는 5/9개이고 readout 차원도 3/5개다. 높은 유효 슬롯에는 넓은 attention 꼬리도 기여한다. 116개/74개의 독립 패턴을 배웠다는 뜻이 아니다.
4. **이번 20 epoch에서는 기준 M0에도 실제 중복 벡터 군이 아직 없었다.** 모든 조건에서 L2≤1e−6인 최대 이웃 크기는 1이다. 초기 attention 집중과 벡터 수축을 분리했지만, 장기 학습의 296개 중복이 해결됐다고 결론낼 수 없다.
5. **AUROC/AP는 평가하지 않았다.** 학습 정상 MSE가 낮아져도 질환까지 잘 복원하면 이상 탐지에는 불리할 수 있으므로 정상 복원만으로 최종 방법을 채택하지 않는다. 이는 이번 실험의 결과가 아닌 평가상 제한이다.

## 다음 우선순위 — 제안, 미실행

먼저 같은 조건의 M0와 λ=0을 더 길게(예: 사전 고정한 총 100 epoch) 대조하여 집중·수축·중복이 재발하는지 확인하는 편을 권한다. Warm-up은 이후 규제를 다시 켤 이유가 있는지 판단하며 비교한다. M2 단독 대조를 성능 개선 후보로 확대할 근거는 이번 결과에서 없다. 조합·warm-up·memory 크기·no-memory AE의 추가 학습은 이번 턴에 시작하지 않았다.

2048개에서 300개로 줄여야 학습됐다는 사용자 회상은 별도 단서로 유지한다. 이번 실험은 300개 고정이므로 당시 실패 원인이나 최적 memory 크기를 검증하지 않았다. 기존 entropy 계산은 slot 수에 따라 실효 배율도 달라지므로 향후 크기 대조에서는 이 혼합 효과도 명시해야 한다.

## 기존 k-fold 후보 최종 상태

R6도 14:33 KST에 완료되어 기존 후보 24/24 fold가 모두 끝났다. R6 fold 4 AUROC/AP는 0.7600/0.9199, 4-fold 평균은 0.6925/0.9095다. R0 0.7300/0.9173 대비 여전히 낮으며, 여섯 후보 모두 평균 AUROC/AP가 R0보다 낮다. 저장 patch 점수에서 기준 포함 28개 모델-fold의 환자 max·AUROC/AP를 재계산하고 환자 ID/label/patch 수 일치를 확인했다.

## 근거

- 실행: `/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/memory_collapse_pilot_20261110T051850Z`
- 최종 정상 학습 전체 진단: `/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/memory_collapse_pilot_20261110T051850Z/completed_audit.json`
- 요약·고정 진단 시계열: `/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/memory_collapse_pilot_20261110T051850Z/results_summary.json`
- 기존 k-fold 최종 검산: `outputs/legacy_candidate_sweep_20261109/comparison_final_snapshot_20261110.json`
- 후속 검산 코드: `memory_pilot/audit_completed.py`
