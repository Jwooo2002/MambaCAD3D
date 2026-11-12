# 구조 유지 성능 개선 — 첫 48시간 실험 계획

사용자 승인: 기존 SSM·memory 구조를 유지하고 학습·규제·초기화 범위에서 개선한다. 두 GPU를 사용하되 첫 실행 묶음은 최대 48시간 후 중간 판정한다. 목표는 정상 복원 MSE 자체가 아니라 환자 단위 AUROC/AP 개선과 여러 seed·fold에서의 재현성이다.

실행 폴더: `outputs/performance_nested_20261112T031500Z_200ep`. 이전 SSM·hybrid 실험은 모두 종료된 상태에서 새로 시작한다. 이 계획은 기존 결과와 frozen source를 덮어쓰지 않는다.

## 이번에 바꾸는 것

| 조건 | Optimizer | Weight decay | Entropy 계수 | 해석 및 원고에 남길 내용 |
| --- | --- | ---: | ---: | --- |
| P0 기준 | Adam | 1e-5, 전체 | 2e-4 | 동일 입력·분할·학습량의 새 대조군. 과거 최고값을 기준으로 삼지 않음 |
| P1 | Adam | 0, 전체 | 2e-4 | 규제 변경 및 paired 비교. 구조 변화 없음 |
| P2 | Adam | 0, 전체 | 2e-5 | P1 대비 entropy 강도 10분의 1. 희소성 규제 강도를 보고 |
| P3 | Adam | 0, 전체 | 0 | 명시적 entropy 손실 제거 대조. hard shrinkage·memory는 유지. 학습 목적의 변경이므로 숨기지 않음 |

현재 수정 모델(grouped SSM 파라미터 분리, 실제 hard shrinkage, kernel3 down/up, 2,109,241 parameters), memory300, threshold0.0025, 초기화, decoder, scan은 모든 조건에서 동일하다. Decoder 초기화 변경은 이전 두 seed에서 일관된 개선이 없어 첫 묶음에 넣지 않는다. AdamW, cosine-memory, memory2048, sampling, 새 손실, augmentation, 병변 마스킹, 집계 변경도 이번 후보 탐색에 넣지 않는다.

전처리는 이미 검증·보존한 `ordered_mm_v2`의 연결 순서·물리 규격과 HU clipping [-300,400] → [0,1]을 사용한다. 직전 정상 TRAIN 기전 실험은 legacy 입력을 사용했으므로 그 MSE와 직접 비교하지 않는다. 기존 전처리 수정·구조 공통 오류 수정 자체는 이번 P1–P3의 효과로 귀속하지 않는다. 모든 학습 epoch와 평가에서 해당 환자의 전체 patch를 사용한다.

## 환자 단위 분리와 선택

ASOCA 정상20명·질환20명을 사용한다. 정상 outer fold는 기존 고정 분할을 유지하고 질환도 seed20261112로 4개 fold에 나눈다. 각 환자는 seed당 outer test에 정확히 한 번만 들어간다. 동일 환자의 patch는 역할을 넘나들지 않는다.

1. **Inner 후보 비교: 32회.** Outer fold별 정상 development15명 중 12명만 가중치 학습에 사용한다. 기존 정상 validation3명과 outer development 질환15명은 후보 선택용이다. 후보4개 × outer fold4개 × seed42/123. 정상·질환 validation 점수는 고정 최종200 epoch에서 한 번 평가한다. 정상 학습만 한다는 말과, 질환 label로 hyperparameter를 선택한다는 말을 구분한다.
2. **후보 선택 고정.** 각 outer fold에서 두 seed의 평균 max-AUROC가 가장 높은 후보를 고른다. AP가 기준 P0보다 0.02 넘게 낮은 후보는 제외한다. AUROC 동률은 AP, 다시 동률이면 P0→P1→P2→P3 순서. 0.02는 이번 탐색의 사전 운영 기준이며 임상적 비열등성 한계가 아니다. 네 fold의 선택 파일·hash가 모두 고정될 때까지 outer test 추론을 시작하지 않는다.
3. **Outer 재학습·평가: 최대24회.** 각 fold의 정상 development15명 전체로 P0와 해당 fold 선택 후보를 처음부터 재학습한다. Seeds42/123/2024, 각200 epoch. Outer test는 해당 fold 정상5명·질환5명이다. P0가 선택되면 같은 실행을 중복하지 않고 두 역할에 사용한다. 다른 seed·fold의 높은 값만 골라 보고하지 않는다.

이는 **4-fold outer CV + 한 개 inner holdout**이다. Inner 자체를 다시 k-fold로 반복하는 완전한 중첩 k-fold는 아니다. 정상 validation3명이 작아 후보 선택은 불안정할 수 있다. ASOCA는 이미 과거 개발·평가에 쓰였으므로 새 분할로 과거 미사용 test가 되지 않는다. 이번 외부 ImageCAS-X 평가·튜닝은 큐에 넣지 않는다. 외부 평가를 추가할 때는 선택과 평가 프로토콜을 먼저 다시 고정하고 기존 노출 이력을 유지한다.

## 공통 학습 예산과 재현성

- 고정200 epoch, batch2, Adam lr1e-4, cosine T_max200, eta_min1e-6, gradient clipping1.0. Early stopping 없이 최종 epoch만 주 평가에 사용한다. 50/100/150/200 epoch checkpoint는 진단·복구용이며 높은 질환 점수의 checkpoint를 선택하지 않는다.
- 같은 stage/fold/seed의 모든 후보는 정확히 같은 초기 state와 전체 patch shuffle 순서를 사용한다. Fold마다 seed를 명시적으로 다시 설정한다. 초기 state·200 epoch 순서·모델 소스·설정·입력 파일의 hash를 보존한다.
- Custom CUDA backward의 수치 비결정성까지 제거했다고 주장하지 않는다. 복제 초기값·순서와 여러 seed로 변동을 확인한다.
- 300 epoch 준비본으로 두 GPU에서 정상 patch만 사용한 4조건×12update 검사, 유한 loss/gradient, zero-memory 행 부재, checkpoint 복원 후 출력 일치를 확인했다. 예상 학습만 약46.8시간, 30% 여유 포함60.8시간이었다. **질환 추론 전** 200 epoch로 줄여 새 immutable 실행 폴더를 만들었다. 예상 학습31.2시간, 30% 여유 포함40.5시간. 준비본은 학습 큐에 넣지 않고 보존한다.
- 실제 각 epoch에서 MSE, entropy, local/global 유효 슬롯 수, 최대 attention, zero-memory 비율, 시간·GPU 상태를 기록한다. 추정 시간은 첫 실제 epoch들에서 갱신한다.

## 결과 보고와 판정

주 집계는 전체 patch의 max, 주지표는 환자 AUROC이다. AP는 prevalence와 함께 보조로 보고한다. Inner prevalence는15/18, outer는5/10이므로 AP 절대값을 두 단계끼리 비교하지 않는다. Mean/top5/95percentile은 사전 지정한 보조 결과이며 이 점수로 주 후보를 바꾸지 않는다. 첫 묶음에서는 별도 calibration을 요구하는 민감도·특이도 threshold를 새로 만들지 않는다.

첫 후보 선택용 AUROC/AP는 각 학습이 끝나는 대로, outer test는 선택 고정과 재학습이 끝나는 대로 보고한다. 첫 결과 예상은 실제 epoch 속도로 보정한다. 진행 중 test 결과를 보고 남은 조건·epoch·후보 선택 규칙을 바꾸지 않는다.

최종 비교는 **고정된 후보 선택 절차 vs P0**이다. Fold마다 선택된 설정이 다르면 단일 후보의 검증 점수처럼 표현하지 않는다. 모든 fold×seed AUROC/AP, 평균 paired 차이, seed별·fold별 차이를 제시한다. 환자를 fold·class 안에서 bootstrap하고 같은 환자의 draw를 두 조건 및 세 seed에 공유해 paired 차이의 95% CI를 산출한다. 이 CI는 고정된 모델·선택·분할에 조건부이며 재학습·과거 선택 편향 전체의 불확실성을 포함하지 않는다. Fold/seed SD를 환자 CI로 부르지 않는다.

‘안정적 개선’은 평균 차이만으로 결정하지 않는다. 세 seed의 평균 차이 방향, fold별 악화, AP 손실, paired CI를 함께 본다. CI가0을 포함하거나 seed마다 반대 방향이면 그 한계를 그대로 보고한다. 이 묶음은 자동 원고 주 모델 채택을 하지 않는다. 정상 MSE만 줄고 AUROC/AP가 나빠지면 개선 성공으로 간주하지 않는다.

## 실행·중단과 기록

GPU당 한 학습, 유한 큐. 시작부터48시간 deadline을 넘으면 신규 실행을 멈추고 실행 중인 작업은 현재 상태를 저장해 중단한다. 작업 실패·nonfinite·명시적 STOP도 큐를 중단한다. 다른 프로젝트의 GPU 프로세스를 종료하지 않는다. 실패·중단 결과도 기록한다. Deadline 도달은 목표 완료를 뜻하지 않는다.

주요 시작·실패·완료·새 결과는 기존 Notion 페이지의 새 토글에 기록한다. 영구 감시·예약 보고를 설정한 것은 아니다. 활성 목표를 수행하는 동안 실제 결과를 확인하고 단계별 보고를 갱신한다.

근거 파일: `plan.json`, `splits.json`, `pairs.json`, `frozen_hashes.json`, `preflight_gpu0.json`, `preflight_gpu1.json`, `manifest.json`, `selections.json`(선택 완료 후), 각 실행의 `state.json`, `history.csv`, `evaluation_access.json`, `results.json`, patient/patch 점수 CSV. 분석 코드는 별도 frozen `analyze.py`이며 원점수에서 AUROC/AP와 patch 집계를 독립 검산한다.
