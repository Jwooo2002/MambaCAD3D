# 현재 재학습 기준의 후보 비교

2026-11-09 사용자 요청에 따라, 진행 중인 `legacy_retrain_20261109T064401Z`를 R0로 삼아 후보 6개를 각각 4-fold로 실행한다. 이전 문서의 **수정 T0와 다른 실험군**이다. 이 작업은 후보 비교용이며 최종 리비전 평가 절차를 바꾸거나 완료한 것이 아니다.

| 실행 순서 | ID | R0 대비 변경 |
| --- | --- | --- |
| 1 | R3_cosine | query/key cosine, 온도 1, raw memory value 유지 |
| 2 | R1_dt | dt projection weight ±rank⁻¹ᐟ² 균등 초기화, softplus(bias) 0.001–0.1 로그 균등 |
| 3 | R2_lambda | 기존 entropy에 곱하는 λ를 0.0002 → 0.0006826667로 증가 |
| 4 | R4_sampling | 정상 환자 균등 확률, 환자 내 patch 균등 확률로 복원추출 |
| 5 | R5_dt_lambda | R1 + R2 |
| 6 | R6_cosine_lambda | R3 + R2 |

- 기존 memory는 실제 hard shrinkage가 없는 clamp이므로, 이전 T7의 shrinkage 끄기는 이 기준에서 생략한다.
- λ 실험은 기존 entropy 축을 그대로 둔 **3.4133배 강도 증가**이다. 올바른 entropy 축으로 바꾼 뒤 강도를 보정하는 이전 T2와 구분한다.
- 원래 입력·정규화·1.72M kernel-2 모델·SSM·entropy·checkpoint 동작·normal test/validation 공유를 유지한다. 변경 효과를 탐색하는 용도이며 최종 일반화 성능을 입증하지 않는다.
- 초기 seed 42와 4-fold 분할·실행 순서를 맞춘다. 원래 학습기는 fold마다 seed를 초기화하지 않는다. 후보별 학습 경로와 조기 종료가 달라지면 이후 fold의 초기화까지 같다고 할 수 없다.
- batch 2, lr 0.0001, 최대 1000 epoch, patience 10을 동일하게 적용한다. Sampling은 epoch당 patch draw와 update 수를 맞춘다. 조기 종료로 실제 총 update 수는 달라질 수 있다.
- 정상/질환 test patch는 모두 평가한다. 이 라운드에 외부 데이터 평가는 추가하지 않는다.
- 기존 평가기가 출력하는 fold 기반 CI는 이 탐색 실험의 환자 단위 불확실성으로 해석하지 않는다. 모든 fold AUROC/AP와 원점수를 보존한다.

## 실행과 상태

처음에는 GPU 1의 순차 큐로 시작했으며, 2026-11-09 18:50 KST 사용자 요청으로 두 GPU를 사용하는 유한 큐로 전환했다. 진행 중인 기준(GPU 0)과 cosine(GPU 1) 학습은 재시작하지 않았다. 기준 학습의 완료·프로세스 종료·GPU 비어 있음을 확인한 뒤 GPU 0에도 대기 후보를 배정한다. GPU별로 후보 하나씩 실행하고, 각 후보의 4-fold 순서는 유지한다. 오류가 발생하면 새 후보 배정을 멈추고 이미 진행 중인 학습은 계속한다.

- `active_run.json`: 큐 프로세스와 출력 경로.
- `outputs/legacy_candidate_sweep_20261109/sweep_manifest.json`: 실행 순서·상태·각 PID.
- `parallel_queue.log`: 두 GPU 배정·완료 기록. `baseline_gate`와 `scheduler_heartbeat_utc`는 기준 종료 대기와 관리자 동작 확인용이다.
- `sweep_manifest_before_parallel_*.json`: 전환 전 큐 상태. 이전 순차 관리자만 종료했으며 진행 중인 학습 프로세스는 인계했다.
- 각 후보의 `run_state.json`, `training.log`, `launcher.log`: 현재 학습 상태.
- 각 후보의 `fold_N_observation.json`: 완료된 fold의 AUROC/AP·모든 환자/patch 점수·학습 기록. 기존 후처리 이후 저장된다.
- 각 후보에 기준 소스의 독립 복사본과 변경 후 해시를 보존했다. 기준 학습과 주 리비전 소스는 수정하지 않았다.

## 실행 전 확인

`verification.json`에 옵션을 끈 모델의 전체 가중치와 실제 입력 출력이 기준 구현과 정확히 같음을 기록했다. 모든 후보에 실제 크기의 입력으로 forward/backward/optimizer update를 실행해 유한한 손실·gradient를 확인했다. 환자별 총 sampling 확률, cosine 크기 불변성, dt 범위, 평가 소스 불변도 확인했다.
