# Memory 100-epoch 연장 결과

2026-11-10 21:12 KST 검산. M0 기준과 M1 entropy λ=0, seed 42·123의 네 실행 모두 총 100 epoch를 완료했다. 연장 실행은 16:55에 시작해 18:25에 종료됐으며 실제 소요는 89.27분이었다. 완료 후 해당 학습 프로세스가 종료됐고 두 GPU에 compute process가 없음을 확인했다.

**이번 조건에서 entropy 제거는 심한 메모리 중복·조회 집중을 완화하고 두 seed의 정상 복원을 개선했다. 다만 seed 123의 벡터 수축과 근접 수렴은 남아 있어 붕괴를 완전히 해결했다고 주장하지 않는다. 질환 AUROC/AP는 평가하지 않았다.**

## 동일 비교 조건

각 조건의 기존 epoch-20 모델·Adam·scheduler checkpoint를 이어 총 100 epoch까지 학습했다. 같은 seed에서 초기 가중치와 100 epoch의 명시적 배치 순서를 공유했다. Fold 1 정상 TRAIN 15명·956개 patch를 매 epoch 모두 사용했고 memory 300, legacy kernel-2 모델·entropy 축·clamp, batch 2, scheduler T_max=1000을 유지했다. Validation/test, 조기 종료, checkpoint 선택, 추가 조합은 없다. 부모 checkpoint의 RNG 미보존 및 GPU update의 수치 차이 한계는 실행 README와 verification.json에 기록돼 있다.

## 고정 epoch 100 — 전체 정상 TRAIN 956개 patch

| Seed | 조건 | 정상 MSE ↓ | 유효 슬롯 | 최대 중복 이웃 수 | Readout 차원 | Memory norm 중앙값 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 기준 | 0.00358384 | 1.7607 | 298 | 1 | 0.041205 |
| 42 | λ=0 | 0.00049339 | 47.5991 | 1 | 4 | 0.062915 |
| 123 | 기준 | 0.00238053 | 1.7167 | 297 | 2 | 0.037652 |
| 123 | λ=0 | 0.00165349 | 33.9302 | 1 | 5 | 0.021947 |

MSE는 patch별 평균 제곱 오차를 전체 patch에 걸쳐 평균한 값이다. λ=0의 감소율은 같은 seed 기준 대비 각각 **86.23%, 30.54%**다. 두 seed의 결과를 독립 환자 기반 유의성이나 CI로 해석하지 않는다.

유효 슬롯은 exp(평균 query entropy)이며 독립적인 정상 패턴 개수가 아니다. 최대 중복 이웃 수는 한 벡터로부터 L2≤1e−6 안에 있는 슬롯 수의 최댓값으로 자기 자신을 포함한다. 따라서 1은 이 기준에서 중복 이웃이 없음을 뜻한다. Readout 차원은 전역 중심화 공분산의 99%를 설명하는 차원이다. λ=0에서 argmax로 선택된 서로 다른 슬롯은 각각 6/8개였으며, 넓은 attention 분포의 47.6/33.9를 그대로 독립적 정보량으로 읽을 수 없다.

## 시간 경과와 남은 수렴

기준의 최대 중복 이웃 수는 두 seed 모두 epoch 20에서 1이었지만, epoch 50에서 267/278, epoch 60에서 298/297, epoch 100에서도 298/297이었다. 즉 짧은 pilot에서 관찰되지 않았던 대규모 중복이 더 긴 학습에서 재현됐다. Probe 간격이 있으므로 정확한 최초 발생 epoch를 특정하지 않는다.

λ=0은 저장된 모든 probe 시점에서 L2≤1e−6의 중복 이웃 수가 1이었다. 그러나 최종 checkpoint를 직접 읽어 거리 기준을 바꾸자 seed 간 차이가 확인됐다.

| L2 기준 | 기준 seed 42 | λ=0 seed 42 | 기준 seed 123 | λ=0 seed 123 |
| --- | ---: | ---: | ---: | ---: |
| ≤1e−6 | 298 | 1 | 297 | 1 |
| ≤1e−5 | 298 | 1 | 297 | 113 |
| ≤1e−4 | 298 | 1 | 297 | 268 |
| ≤1e−3 | 298 | 1 | 297 | 286 |

이는 하나의 벡터 주변 이웃 수이며 집합 내 모든 벡터 쌍의 거리가 해당 기준 이하라는 뜻은 아니다. 추가 거리 기준은 최종 해석을 위한 사후 민감도 검사이며 사전 등록한 채택 기준은 아니다.

λ=0의 memory norm 중앙값은 epoch 20→100에 seed 42에서 0.4193→0.0629, seed 123에서 0.3904→0.0219로 줄었다. Seed 123은 작은 차이가 남은 다수의 유사 벡터가 존재한다. 따라서 strict threshold의 중복 수 1만 보고 memory collapse 해결을 선언하면 안 된다. Readout 차원과 실제 사용 분포를 함께 봐야 한다.

## 해석과 다음 판단

- 동일 초기화·배치 순서의 대조에서 λ 제거가 조회 집중·정상 복원·readout 분산 차원의 차이를 만들었다. 현재 legacy 목적식에서 entropy가 강한 집중과 중복에 기여한다는 근거가 강화됐다. 모든 원인이 entropy 하나라고 확정한 것은 아니다.
- 정상 TRAIN MSE 개선은 일반화 또는 이상 탐지 개선의 증거가 아니다. 질환도 더 잘 복원하면 정상/질환 구분이 약해질 수 있다. 질환 AUROC/AP, 정상 validation 일반화, memory 기여는 별도의 사전 고정한 평가로 확인해야 한다.
- 두 seed 중 하나에 근접 수렴이 남고 seed별 MSE 차이도 크므로 최종 모델 채택은 보류한다. 추가 학습·M2 연장·warm-up·새 손실·2048 memory 비교·no-memory AE·질환 평가는 이번 확인에서 실행하지 않았다.

## 검산과 근거

실행: `outputs/memory_collapse_continuation_20261110T075000Z`.

`memory_pilot/audit_continuation.py`로 네 실행의 정상 종료, history 1~100 epoch, 매 epoch 956개 patch·478 updates, 최종 Adam step 47,800, scheduler last_epoch=100/T_max=1000, 부모 checkpoint·초기값·배치순서 hash, 원 소스/설정 hash, RNG 저장을 확인했다. 최종 checkpoint의 memory를 CPU에서 다시 읽어 중복 거리 통계를 계산했고, 저장된 환자별 MSE의 가중 합이 전체 MSE와 일치함을 확인했다. 이번 검산은 저장 자료 기반이며 새로운 forward 평가나 학습을 하지 않았다.

원 수치: 각 조건의 `full_normal_training_epoch_100.json`, `probe_epoch_*.json`, `epoch_100.pt`. 종합 검산: `completed_audit.json`. 기존 20-epoch 결과는 `docs/MEMORY_PILOT_RESULTS_20261110.md`에 별도로 보존한다.
