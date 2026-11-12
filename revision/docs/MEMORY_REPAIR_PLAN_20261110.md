# Memory 중복·초기 집중을 줄이는 후속 계획

2026-11-10. **처음에는 제안으로 작성했고, 같은 날 사용자 승인 후 14:19 KST에 M0/M1/M2 pilot을 시작해 15:05에 6/6 실행을 완료했다.** 원래 구조를 유지하는 작은 대조부터 원인을 분리한다. 현재 관찰은 memory 수정을 우선 검토할 근거지만, 낮은 질환 AUROC의 가장 큰 원인이라고 확정한 것은 아니다. Cosine에서 벡터 다양성이 늘어도 AUROC는 높아지지 않았다.

## 추가 확인

- **사용자 회상, 2026-11-10:** 원래 memory가 2048개였던 것으로 기억하며, 300개로 크게 줄여야 학습이 됐다고 설명했다. 당시 설정·실패 로그와 “학습이 되지 않음”의 구체적 양상은 아직 대조하지 않았다. Memory 크기에 대한 민감성의 단서로 보존한다. 현재 슬롯 중복과 같은 원인이라고 단정하지 않으며, 이번 M0/M1/M2는 300개로 고정해 λ·weight decay를 먼저 분리한다. 크기 대조는 후속 후보이고 현재 실행에는 추가하지 않았다.
- 기준 R0의 attention 유효 슬롯 수(exp(평균 entropy), legacy 축 배율 보정)는 fold 1/2/3/4에서 각각 epoch 4/3/4/3에 10개 미만이 됐다. Epoch 5에는 약 2.60/1.96/2.33/2.51개다. 이는 query별 집중의 발생 시점이며, 296개 벡터가 실제로 동일해진 시점을 측정한 것은 아니다.
- Adam은 모든 파라미터에 weight_decay=1e−5를 적용한다. Memory도 포함되며, memory 전용 재초기화나 다양한 슬롯을 쓰게 하는 별도 손실은 없다.
- 최종 checkpoint의 첫 정상 학습 patch에서, 중복군 memory의 행별 복원 gradient 중앙값은 1.38e−8~2.09e−7, 나머지 슬롯은 0.00624~0.0230이었다. 해당 patch에서 중복군에 대한 복원 학습 신호가 훨씬 작았다. 중복군의 Adam decay 항 norm은 복원 gradient보다 약 1.4~16배 컸다.
- 이 gradient 비교는 저장 모델의 국소 진단이다. Adam의 과거 momentum/적응적 스케일이나 전체 학습 경로를 복원하지 않았으므로, weight decay가 중복 발생의 원인이라고 단정하지 않는다. 별도 대조를 할 이유로만 사용한다.
- 각 fold의 정상 학습 환자 15명에서 처음/중간/마지막 patch를 고정 추출했다. 전역 중심화 공분산의 99%를 설명하는 차원 수는 R0 encoder query 15/5/3/8, memory readout 2/3/3/2였다. Cosine은 query 40/33/36/38, readout 4/3/3/4였다. 각 모델에서 46,080개 latent 위치를 분석했으며 독립 환자 수를 뜻하지 않는다. 선형 분산 차원이며 임상 정보량의 측정도 아니다.
- 따라서 memory 벡터가 서로 다르다는 것, query마다 가중치가 집중된다는 것, decoder에 전달되는 표현이 다양한 것은 서로 다른 지표다. 벡터 중복만 없애면 성능 문제가 해결된다고 가정하지 않는다.

근거: `outputs/legacy_candidate_sweep_20261109/diagnostic_memory_causes_20261110.json`, 실행 코드 `candidate_sweep/diagnose_memory_causes.py`.

## 목표

한 patch가 소수의 적합한 memory를 조회하면서도, 정상 학습 자료 전체에서는 서로 다른 정상 패턴이 구별되는 memory와 연결되는 것이 목적이다. 300개를 억지로 균등하게 쓰게 만드는 것을 성공 기준으로 삼지 않는다.

현재 +λ·평균 H(attention)는 개별 query의 분포를 좁히지만, 서로 다른 query가 서로 다른 슬롯을 사용하도록 보장하지 않는다. 모두가 같은 슬롯을 선택해도 entropy 손실은 작다. 원 MemAE도 entropy와 shrinkage를 희소 조회 목적으로 사용하며, memory 없는 AE와 비희소 memory 대조를 제시한다. 이 사실만으로 현재 데이터에서 희소성이 유익하다고 가정할 수 없다. 출처: [MemAE 원 논문](https://arxiv.org/pdf/1904.02639).

## 1차: 작은 변경으로 원인 분리

| ID | 변경 | 목적 | 채택·보고 시 설명 |
| --- | --- | --- | --- |
| M0 | 동일 초기화·배치 순서의 기준 | 재현 대조 | 기존 R0 점수를 새 paired baseline으로 대신하지 않음 |
| M1 | Entropy λ=0, 나머지 그대로 | 초기 집중이 entropy 규제에 의존하는지 | 규제 제거 ablation. 최종 채택값이라는 뜻은 아님 |
| M2 | Memory에만 weight decay=0 | 약하게 선택된 슬롯의 수축 영향 분리 | Memory parameter group의 optimizer 설정 변경 |
| M3, 조건부 | M1+M2 | 두 단독 대조가 함께 필요할 때 상호작용 확인 | 두 변경을 함께 보고하며 효과를 하나에 귀속하지 않음 |
| A0, 필요성 대조 | Memory 없이 동일 encoder/decoder를 처음부터 학습 | Memory의 이상 탐지 기여 확인 | 추론 시 memory를 제거한 기존 진단과 구분 |

기존 checkpoint에는 이미 비슷한 벡터가 같은 gradient를 받을 수 있는 대칭이 생겨 있다. 기본 비교는 새로 초기화한 모델에서 한다. 단지 λ를 낮춰 기존 collapse 모델을 이어 학습하는 것을 우선 해법으로 삼지 않는다.

### 짧은 기전 확인 — 실행 승인됨

- M0/M1/M2를 사전에 고정한 첫 fold의 정상 학습 15명만으로 각각 20 epoch, seed 42와 123에서 비교한다. 3조건×2seed=6개의 짧은 실행이다. 기존 k-fold 후보 실험과 구분한 새 실험이며, 2026-11-10 14:19 KST에 실행을 시작했다.
- 이 단계는 기존 legacy 원인 분리용이다. 같은 kernel 2 모델·원래 입력·batch 2·학습률·scheduler 시간축을 유지한다. 조기 종료 없이 20 epoch를 고정하고, test/normal test를 모델 선택이나 점수 확인에 사용하지 않는다.
- 조건 간 encoder/decoder와 memory 초기값을 동일하게 복제하고, 배치 순서 RNG도 별도로 고정한다. Seed와 fold가 같다는 이름만으로 초기화 일치를 가정하지 않는다.
- Epoch 0/1/3/5/10/20에서 중복군 크기, memory 및 readout의 분산 차원, query별 entropy, 정상 자료 전체의 평균 슬롯 사용 분포, 환자별 사용 양상, reconstruction MSE, memory 행별 gradient/norm을 기록한다.
- Attention 집중이 원래 epoch 3~4에 시작하므로 조기 차이를 관찰할 수 있다. 20 epoch에서 다양성이 유지돼도 이후 다시 붕괴할 수 있어 최종 성공으로 간주하지 않는다.
- 후속 성능 비교는 독립 정상 validation·실제 best checkpoint 복사·fold별 초기화를 모든 비교군에 동일하게 적용한 별도 기준에서 수행한다. 현재 legacy 탐색과 수정 평가를 섞지 않고, 내부·외부 test AUROC로 설정을 반복 선택하지 않는다.

## 2차: 단독 대조의 결과에 따라 선택

### M1에서 초기 집중이 완화될 때: entropy warm-up

처음에는 복원과 서로 다른 memory 표현을 학습하게 두고, 이후 희소 규제를 점진적으로 올리는 방식을 검토한다. 예를 들어 초기 5 epoch λ=0, 다음 10 epoch에 목표값까지 증가시키는 것은 사전 고정할 수 있는 pilot 제안이지 검증된 최적 일정이 아니다. 규제가 켜진 뒤 다시 붕괴하는지도 끝까지 확인한다. Entropy 축을 바꾸면 같은 숫자의 λ가 같은 실효 강도를 뜻하지 않으므로 해당 축과 배율을 함께 명시한다.

### M1/M2만으로 부족할 때: 정상 특징으로 초기화·중복 슬롯 재시드

안정화된 encoder의 정상 학습 특징을 모아 서로 다른 대표 특징으로 memory를 초기화하고, 계속 중복되거나 학습 신호를 받지 못하는 슬롯만 정상 학습 특징으로 다시 설정하는 대조를 검토한다. Test patch나 질환 특징은 사용하지 않는다. 모든 학습 patch는 유지하며, 특징 버퍼의 추출은 입력 patch를 버리는 학습 sampling과 구분한다.

VQ 연구와 EnCodec 구현에 data-dependent initialization 및 미사용 code 교체의 선례가 있다. 다만 현재 soft-attention MemAE와 VQ의 학습식은 다르므로, 그대로 적용하면 된다는 근거는 아니다. 현재 모델에서는 key/value가 같은 tensor이므로 특징과 memory의 norm·좌표 스케일을 맞춰야 하고, 교체한 행의 Adam 상태도 함께 초기화해야 한다. 재시드 시점·판정 기준을 고정하고, 계속 재시드하여 붕괴 지표만 가리는 결과를 피한다.

출처: [Robust Training of Vector Quantized Bottleneck Models](https://arxiv.org/abs/2005.08520), [EnCodec 공식 codebook 구현](https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py).

### 그 뒤의 방법 변경: 개별 희소성과 전체 사용 다양성을 함께 다루는 손실

추가 손실을 도입한다면 다음 형태가 기전상 직접적이다.

`L = reconstruction + α · mean_i H(a_i) − β · H(mean_i a_i)`

첫 항은 복원, 두 번째는 query별 희소성, 세 번째는 정상 자료 전체에서 같은 슬롯만 고르는 것을 완화한다. 이는 현재 결과에서 도출한 후속 설계 제안이며 이 프로젝트에서 검증되지 않았다. Batch 2의 위치들이 강하게 상관되어 있어 한 batch의 균등 사용을 강요하면 안 된다. 여러 정상 환자를 포함한 사용 통계를 관찰하고, 계수와 추정 방법을 별도로 설계해야 한다. Attention이 이미 동일한 벡터에 대한 대칭 상태라면 이 항만으로 벡터가 다시 갈라진다고 보장할 수도 없다.

이 항은 새로운 학습 목적이므로 원고에 조용히 숨길 내부 교체로 취급하지 않는다. 이번 리비전의 첫 대조에는 넣지 않는다. Memory 300개를 전부 직교시키는 규제도 128차원에서 성립하지 않으므로 권하지 않는다.

## 채택 기준과 리비전 범위

붕괴 감소, 정상 복원 유지, 여러 seed에서의 지속성을 먼저 확인한다. 이후 사전에 정한 평가에서 no-memory AE 및 동일 기준 모델 대비 이상 탐지 효과를 확인한다. 슬롯 수나 정상 MSE 하나만 좋아져도 채택하는 기준은 아니다.

Memory optimizer 설정·entropy 일정은 구조를 유지하더라도 학습 동작을 바꾼다. 특히 λ=0은 학습 목적과 명시적 희소성 유도를 바꾸므로 실질적인 방법 변경으로 다뤄야 한다. 앞서 이를 비교적 좁은 변경으로만 설명한 판단은 2026-11-10에 정정했다. 정상 특징 초기화, 재시드 규칙, 새로운 diversity 손실도 추가 방법 변경으로 명시하고 대조해야 한다. 어느 변경도 저널의 공식 'minor revision' 판정이나 수용을 보장하지 않는다.

처음 제안한 순서는 **새 초기화로 M0/M1/M2의 짧은 기전 비교 → 필요한 경우 warm-up/정상 특징 초기화 → 고정한 방법과 no-memory AE의 정식 비교**였다. 첫 단계를 완료했고 기존 R6도 종료됐다. Pilot 결과상 다음 우선순위는 M0와 λ=0의 더 긴 대조이며, 아래 결과 문서에서 근거를 설명한다. Warm-up, 재시드, 새 다양성 손실 및 no-memory AE 학습은 아직 실행하지 않았다.

## 실행 기록 — 2026-11-10 14:19 KST

- 실행 경로: `outputs/memory_collapse_pilot_20261110T051850Z`.
- 정상 학습 15명의 956개 patch를 매 epoch 모두 사용한다. 같은 seed의 세 조건은 저장된 초기 가중치와 20개 epoch의 배치 순서를 공유한다. 진단용 45개 patch만 고정 추출하며, 학습 patch를 제외하는 sampling은 하지 않는다.
- 기존 kernel-2 모델 소스와 설정을 해시로 고정했다. 학습률 scheduler의 T_max=1000, 기존 entropy 축과 memory clamp를 유지한다. 총 20 epoch를 고정하므로 validation/test/조기 종료는 사용하지 않는다.
- 실행 전 실제 GPU에서 전처리의 원 구현 일치, 동일 초기 출력, 세 조건의 유한한 gradient/update, optimizer parameter group, 진단 전후 가중치·RNG 불변, 모든 epoch의 전체 patch 포함을 검증했다. 진단의 분산이 정확히 0인 경우 rank=0으로 처리함도 확인했다. 근거는 실행 폴더의 `verification.json`이다.
- GPU 1에서 M0 seed 42를 시작했다. 기존 R6가 완료되고 GPU 0의 학습 프로세스가 종료되면 남은 실행을 GPU 0에도 배정한다. 여섯 실행이 끝나면 대기열도 종료되며, 실패하면 신규 실행을 멈춘다.
- 진행은 실행 폴더의 `manifest.json`과 조건별 `run_state.json`에 기록한다. 고정 epoch checkpoint와 `probe_epoch_*.json`을 보존한다. 최종 epoch에는 optimizer/scheduler 상태도 저장한다.

## 완료 후 판단 — 2026-11-10

여섯 실행이 15:05 KST에 모두 종료됐다. 정상 학습 956개 patch 전체의 최종 checkpoint 재평가에서 λ=0은 두 seed의 초기 조회 집중을 완화했지만, 복원 MSE는 각각 39.9% 감소와 2.5% 증가였다. Memory 전용 decay 제거는 벡터 norm을 유지해도 조회 집중과 출력 병목을 해소하지 못했다. 20 epoch에서는 기준에도 실제 중복 벡터 군이 아직 없었으므로 장기 중복 해결이나 질환 성능 향상을 주장하지 않는다.

상세 표·검산·한계는 [MEMORY_PILOT_RESULTS_20261110.md](MEMORY_PILOT_RESULTS_20261110.md)에 기록했다. 완료 직후 M0/λ=0의 더 긴 고정 대조를 우선 제안했고, 이후 사용자 승인에 따라 아래 연장 실행을 시작했다.

## 리비전 범위 재평가 — 2026-11-10

사용자가 entropy 제거가 주요 방법 변경에 해당하는지 우려했다. λ=0을 진단용 ablation으로 추가하는 것과 최종 주 모델을 λ=0으로 교체하는 것은 구분한다. 현재 pilot만으로 최종 교체를 권하지 않으며, 100-epoch 후속 제안도 채택을 전제하지 않는 진단이다. 구조 유지·낮은 구현 난도가 작은 과학적 변경을 의미하지 않는다.

보관된 편집자 결정문 `/home/jiwoong/.codex/attachments/75a697aa-35d3-46ab-ae02-3400a2fbf0c5/붙여넣은 텍스트.txt`의 5행은 MAJOR REVISION이며, major revision은 한 번만 허용하고 재심에서 다시 major가 필요하면 자동 거절된다고 명시한다. 11행은 모든 변경을 원고에 표시하고 별도 point-by-point 답변에 설명하도록 요구한다. 보관 편지에 대한 확인이며 이후 별도 결정의 존재나 변경 수용을 단정하지 않는다. 일반 절차: https://journals.ieeeauthorcenter.ieee.org/submit-your-article-for-peer-review/understanding-the-decision-process/

현재 권고는 기존 방법을 공통 오류 수정·적절한 평가 아래 검증하고, λ=0은 우선 보조 대조로 두는 것이다. 주 모델 교체는 장기 안정성·질환 구분·memory 기여와 이전 결과 변화를 설명할 근거가 있을 때 검토한다. 기존 주장이 검증되지 않으면 단지 변경을 작게 보이게 하려고 유지하지 않는다. 변경 범위가 크면 구체적인 변경 요약을 준비하여 편집자 범위 문의를 검토한다. 아래 진단 연장은 시작했으며, 편집자 연락·최종 모델 채택은 실행하지 않았다.

## 연장 실행 — 2026-11-10 16:55 KST

사용자의 “그래 일단 걍 해봐” 승인에 따라 M0 기준과 M1 entropy λ=0만 seed 42·123에서 각각 총 100 epoch까지 이어간다. 네 모델의 기존 20 epoch에서 80 epoch씩, 합계 320 epoch를 추가한다. M2, warm-up, 새 손실, memory 크기 변경, no-memory AE는 추가하지 않는다.

- 실행 경로: `outputs/memory_collapse_continuation_20261110T075000Z`. 실행 포인터는 `memory_pilot/active_continuation.json`; 완료된 기존 20-epoch 실행은 그대로 보존했다.
- 정상 TRAIN 15명·956개 patch만 사용한다. 모델·Adam·scheduler 상태를 epoch 20에서 정확히 복원했다. 두 seed 각각의 배치 순서 생성 흐름을 100 epoch까지 연장하고 첫 20 epoch가 부모 실행과 일치하는지 확인했다. Memory 300, 기존 entropy 축·clamp·모델 소스, 학습률 시간축 T_max=1000을 유지한다.
- 부모 checkpoint에는 RNG 상태가 없어 중단 없이 100 epoch를 돌린 실행과 비트 단위로 같다고 주장하지 않는다. 현재 update가 RNG를 소비하지 않고 활성 dropout이 없음을 검사했으며, 연장 단계 RNG를 seed별로 고정했다. 이번 checkpoint부터 RNG 상태도 보존한다.
- 재개 검사에서 가중치·optimizer·scheduler 복원, 같은 입력의 출력·손실은 정확히 일치했다. 최초의 엄격한 update 일치 검사는 실패했다. 별도로 복원한 대조를 추가한 결과 GPU 학습 반복에서도 같은 규모의 차이가 발생했다. 네 조건의 다음 update에서 최대 가중치 차이는 1.86e−9~3.73e−9였고, 정해 둔 수치 허용 범위 안에 있었다. 상세 수치와 대조 결과는 `verification.json`에 기록했다. 원 학습 계산은 변경하지 않았다.
- Epoch 20/25/30/40/50/60/75/100에서 조회 집중, memory 벡터 중복·norm, query/readout 분산 차원, 정상 복원, gradient 진단을 기록한다. Epoch 100은 전체 정상 학습 patch에서도 평가한다. Validation/test·질환 AUROC/AP·조기 종료·결과 기반 최종 모델 선택은 하지 않는다.
- GPU 1에서 M0 seed 42, GPU 0에서 M1 seed 42가 실제 epoch 21 학습에 들어간 것을 확인했다. 끝나는 GPU에 나머지 seed 123 두 실행을 배정하며 총 네 실행 후 대기열이 종료된다. 실패하면 신규 실행을 중단한다. 시작 시 예상 소요는 약 1.5~2시간이며 종료 시각을 보장하지 않는다.

이번 결과는 집중·수축·중복이 더 긴 학습에서 재발하는지를 판단하는 근거다. 정상 복원이나 유효 슬롯 수의 개선만으로 이상 탐지 성능 향상 또는 최종 채택을 선언하지 않는다. Notion의 기존 memory pilot 토글과 현재 진행에 실행 시작을 반영했다.

## 연장 완료 후 판단 — 2026-11-10 21:12 KST 확인

네 실행 모두 18:25 KST에 총 100 epoch를 완료했다(89.27분). 기준은 두 seed에서 298/297개 메모리 중복 이웃이 재현됐다. λ=0은 정상 TRAIN MSE를 각각 86.2%/30.5% 줄이고 readout 차원을 1→4, 2→5로 늘렸다. 다만 seed 123은 L2≤1e−6 기준의 중복은 없어도 L2≤1e−4에서 최대 이웃 수가 268개이며 벡터 norm 수축이 남았다. 완전한 붕괴 해결이나 질환 성능 개선으로 해석하지 않는다. 추가 실행·질환 평가·최종 채택은 하지 않았다. 상세 표와 거리 민감도 검사는 [MEMORY_CONTINUATION_RESULTS_20261110.md](MEMORY_CONTINUATION_RESULTS_20261110.md)에 기록했고 Notion 기존 토글에도 완료 결과를 반영했다.

## 100-epoch 결과를 반영한 학습 전략 — 제안, 미실행

사용자가 제대로 학습시키는 방법을 요청했다. 새 학습을 시작하는 요청으로 확대하지 않고, 코드·실행 결과·선행 연구를 바탕으로 다음 우선순위를 정했다. 앞의 warm-up 우선 제안보다 이 순서를 우선한다.

### 첫 대조: M3, entropy 제거와 memory decay 제거의 조합

M1은 λ=0에서도 모든 파라미터에 Adam weight_decay=1e−5가 남아 있다. M2는 memory decay만 제거했지만 entropy는 켜져 있었고 20 epoch만 관찰했다. 따라서 M2의 부진은 M3의 효과를 배제하지 않는다. 기존 M1을 대조로 **λ=0 + memory에만 decay=0**을 비교한다. Encoder/decoder decay, memory 300, 모델·입력·scheduler·전체 patch 사용은 유지한다.

Seed 42/123, 기존과 같은 저장 초기값·100-epoch 배치 순서에서 처음부터 고정 100 epoch를 제안한다. 이미 수렴한 epoch-100 모델을 이어서 수정하는 것은 예방 효과와 회복 효과를 섞으므로 첫 비교에 쓰지 않는다. 새 조건 두 실행은 두 GPU 동시 사용 시 관측 속도 기준 약 1시간으로 추정한다. 실행 코드는 아직 추가하지 않았으며 소요를 보장하지 않는다. M2도 장기 연장해야 완전한 2×2 상호작용 분석이 가능하므로 현재 M3 대조만으로 상호작용 전체를 정량화했다고 표현하지 않는다.

이는 decay가 남은 수렴의 원인이라고 확정해서 하는 처방이 아니다. 유사한 key/value는 비슷한 attention과 gradient를 받는 대칭 상태에 가까워질 수 있고, 복원 목적 자체가 소수의 유용한 슬롯과 다수의 유사한 슬롯을 허용한다. Decay 제거는 이 중 수축 요인의 기여를 분리한다. 무조건 norm이 커지거나 슬롯 사용이 많아지는 것을 성공으로 보지 않는다.

### 잔여 수렴 시: 정상 특징으로 초기화하고 단계적으로 학습

무작위 memory와 변화 중인 encoder를 동시에 적응시키는 어려움을 줄이는 후속 제안이다. 먼저 정상 TRAIN 복원으로 encoder/decoder를 사전학습하고, 정상 TRAIN 특징의 여러 대표값으로 memory를 초기화한다. Encoder를 잠시 고정한 상태에서 memory와 decoder가 적응한 다음 전체를 미세 조정하는 순서를 검토한다. 최종 encoder→memory→decoder 구조를 유지해도 추가 사전학습·초기화·동결 단계는 학습 방법 변경이다.

초기화용 특징은 여러 정상 환자에서 얻고, 필요한 경우 특징 버퍼를 제한하되 모든 학습 입력 patch와 모든 test patch는 그대로 처리한다. 원본 학습 입력을 버리는 sampling과 다르다. 대표 벡터와 decoder 입력의 스케일을 확인하고 교체된 memory의 Adam 상태를 초기화해야 한다. 단순히 encoder 특징을 그대로 복사하면 norm과 decoder 입력 분포가 바뀔 수 있다. 단계별 epoch·동결 해제·스케일 처리 규칙은 별도 실행 전에 고정해야 한다.

정상 특징 기반 초기화·재초기화 및 encoder 특징의 변화가 codebook 학습에 미치는 문제는 [Robust Training of Vector Quantized Bottleneck Models](https://arxiv.org/abs/2005.08520)에서 다룬다. [EnCodec 공식 구현](https://github.com/facebookresearch/encodec/blob/main/encodec/quantization/core_vq.py)에도 k-means 초기화와 dead-code 교체가 있다. 이것은 원 MemAE의 필수 절차가 아니며, 현재 soft-attention memory와 VQ의 학습식은 다르다. 현재 모델에서 성공한다는 직접 증거는 아니다. 일반 AE 사전학습이 이상 탐지에 적합한 특징을 보장하지도 않는다.

### 그 이후의 조건부 후보와 제외할 해석

학습이 안정되면 원 방법의 희소성 목적을 유지할 수 있는지 약한 entropy를 사전에 정한 값/일정으로 다시 비교할 수 있다. Warm-up만으로 해결됐다고 가정하지 않는다. 계속 같은 슬롯만 조회하면 여러 정상 환자의 사용 통계에 기반한 다양성 제약을 검토할 수 있지만, batch 2에서 300개 균등 사용을 강제하지 않는다. 새 목적식은 별도 방법 변경으로 보고해야 한다. Memory 크기를 즉시 2048로 늘리거나 매번 reset하여 진단 수치만 개선하는 방식도 첫 조치로 권하지 않는다.

기존 cosine 후보의 `candidate_sweep/candidate_hooks.py:memory_logits`에는 온도/배율 없이 정규화된 내적을 softmax에 넣었다. Logit이 [−1,1] 범위이고 슬롯이 300개이면 단일 슬롯 가중치 상한은 exp(2)/(299+exp(2))≈0.02412다. 이는 코드와 softmax 식에서 직접 도출한 한계이며, 당시 AUROC 저하의 원인을 입증한 것은 아니다. 조회용 정규화+온도를 나중에 검토한다면 기존 cosine 실험과 구분한 새 대조로 다뤄야 한다. 조회 key의 정규화가 decoder에 전달하는 value의 수축을 자동 해결하지도 않는다.

### 학습 성공과 최종 채택 기준

엄격한 중복 기준 하나 대신 여러 절대 거리 기준과 norm 대비 상대 거리, readout의 입력 의존성·분산 차원, 환자별 사용 양상, 정상 validation 일반화를 함께 본다. 고정/서로 바꾼 memory 출력을 넣었을 때의 변화는 실제 입력 의존성을 확인하는 진단이며, 그 자체로 이상 탐지 기여를 증명하지 않는다. 최종 후보는 추가 seed와 사전 고정된 평가, no-memory AE 대조로 확인한다. 정상 MSE가 작아져도 질환을 함께 잘 복원할 수 있다([MemAE 원 논문](https://arxiv.org/abs/1904.02639)).

원인 분리 대조에서는 legacy 조건을 고정한다. 최종 리비전 비교는 공통 구현 오류와 정상 validation 분리를 적용한 같은 조건에서 모든 비교군을 재학습·평가해야 한다. Legacy probe 숫자와 수정된 평가 결과를 직접 섞지 않는다. 이번 전략은 제안이며 새 학습·질환 test 조회·최종 채택은 하지 않았다. Notion의 기존 memory pilot 토글에 동일 우선순위를 반영했다.

## M3 실행 시작 — 2026-11-10 21:30 KST

위 제안 후 사용자의 “그래 해봐” 승인에 따라 **첫 번째 M3 대조만** 시작했다. λ=0, memory의 Adam decay=0, 그 외 파라미터 decay=1e−5인 두 seed(42/123)를 원래 저장 초기값에서 각각 100 epoch까지 학습한다. 100개 배치 순서 파일은 완료된 M1 연장 실행과 hash까지 동일하다. Memory 300, 정상 TRAIN 15명·956개 전체 patch, 기존 모델·전처리·scheduler T_max=1000을 유지한다. 기존 M1 결과는 대조로 보존한다.

실행 폴더는 `outputs/memory_m3_pilot_20261110T123000Z`, 포인터는 `memory_pilot/active_m3.json`, 실행 코드는 `memory_pilot/run_m3.py`다. GPU 1에 seed 42, GPU 0에 seed 123을 배정했다. 실제 첫 epoch 학습을 확인했다. 두 실행은 총 200 epoch 후 종료되며, 실패하면 신규 실행을 중단한다. 시작 시 예상 소요는 약 1시간이다.

실행 전 두 seed의 초기 가중치·출력과 epoch-zero 진단이 M1과 일치함을 확인했다. Optimizer가 모든 파라미터를 중복 없이 포함하고 memory에만 decay가 없음을 검사했다. 동일한 복원 gradient를 넣으면 nonmemory update 차이는 정확히 0이고 memory update만 달랐다. 유한한 실제 update, RNG 불변, 저장·복원 상태 일치, 100 epoch 전체 patch 포함도 검증했다. `verification.json`에 근거를 기록했다. 기존 GPU 수치 오차와 M1의 분할 실행에 관한 비교 한계는 그대로 적용한다.

진단 시 여러 절대 거리 기준과 pair별 평균 norm으로 나눈 상대 거리를 함께 기록한다. 같은 거리 지표를 기존 M1 최종 checkpoint에도 계산해 `control_reference`에 보존했다. Epoch 0/1/3/5/10/20/25/30/40/50/60/75/100의 가중치·optimizer·scheduler·RNG를 저장하며, 최종 epoch은 정상 TRAIN 전체에서도 평가한다. 단계적 초기화·warm-up·추가 손실·memory 크기 변경·질환 평가·최종 채택은 시작하지 않았다. Notion 기존 토글과 현재 진행에 M3 실행을 반영했다.

## M3 완료 후 판단 — 2026-11-10 23:46 KST 확인

M3 두 seed가 모두 22:27 KST에 고정 100 epoch를 완료했다(56.57분). 절대·상대 거리 기준 모두에서 기록된 probe 시점의 근접 중복이 없었고 memory norm 중앙값은 두 seed 모두 약 1.19였다. M1 대비 readout 차원은 4→9, 5→8로 증가했다. 그러나 정상 TRAIN MSE는 seed 42에서 3.73배로 악화, seed 123에서 3.56% 개선됐다. λ=0 조건에서 memory decay가 잔여 수축에 기여한다는 근거는 강화됐지만, 이상 탐지 성능이나 최종 채택은 미확인이다. 후속 초기화·새 손실을 자동 실행하지 않으며 일반화·memory 기여·고정 평가를 다음 판단 대상으로 둔다. 상세 표와 검산은 [MEMORY_M3_RESULTS_20261110.md](MEMORY_M3_RESULTS_20261110.md), 실행 폴더의 `completed_audit.json`에 보존했고 Notion 기존 토글에 반영했다.

## M3 결과 이후의 최소 다음 단계 — 제안, 미실행

저장된 M0/M1/M3 × seed 42/123의 epoch-100 checkpoint 6개를 고정하여 한 차례 탐색 평가하는 것을 우선 권고한다. 각 frozen 모델·전처리를 유지하고 같은 환자의 전체 patch를 처리한다. 환자 점수 max와 주지표 AUROC를 유지하며 AP 및 정상/질환 점수 분포를 함께 보고한다. 모델·epoch·seed·집계법의 최고값을 다시 고르는 평가가 아니다.

실제 분할을 확인한 결과 정상 TRAIN 15명과 겹치지 않는 정상 5명, 질환 20명이 해당 fold의 기존 평가 대상이다. 현재 pilot은 별도 validation 없이 고정 epoch로 학습했다. 이 환자들은 과거 legacy 실험에서 이미 평가됐으므로 새로운 independent validation이나 미사용 최종 test로 부르면 안 된다. 현재 학습 환자를 사후에 validation으로 바꾸는 것도 유효한 검증이 아니다. 정식 비교에서는 공통 오류 수정을 모든 조건에 동일하게 적용하고 train/validation을 재학습 전에 분리해야 한다. 질환 label을 사용한 모델 선택은 개발/평가 분리와 기존 데이터 노출을 명시한다. 이미 본 데이터를 다시 나눠도 과거 미사용 자료가 되지는 않는다. 근거: [모델 선택 편향 연구](https://www.jmlr.org/papers/v11/cawley10a.html).

M3의 수축 완화가 실제 정상/질환 점수 분리에 연결되는지부터 확인한다. 정상 복원 오차만으로 이상 탐지 우열을 확정할 수 없다([MemAE 원 논문](https://arxiv.org/abs/1904.02639)). 결과가 엇갈리면 최종 승자를 선언하지 않고 동일 조건 정식 비교로 확인한다. 이번 턴은 계획만 기록했으며 평가 점수 조회·새 학습은 시작하지 않았다.

## 고정 평가 완료 — 2026-11-11 00:01 KST

위 계획 후 사용자 승인으로 M0/M1/M3 × 두 seed의 100-epoch 모델 6개를 평가했다. 동일한 정상 5명·질환 20명의 전체 1,547 patch에서 환자 max 점수의 AUROC/AP를 계산했고 원점수 검산을 통과했다. AUROC는 seed 42에서 M0/M1/M3=0.59/0.64/0.75, seed 123에서 0.77/0.72/0.72다. 두 seed 기술적 평균은 0.680/0.680/0.735로 M3가 높지만, 기준 대비 변화가 +0.16/−0.05로 엇갈렸다. 메모리 수축을 막은 관찰을 일관된 질환 탐지 개선으로 확대하지 않는다. 새 학습·최종 채택은 하지 않았다. 이전에 평가했던 자료의 탐색 결과라는 한계는 유지한다.

전체 표·점수 분포·검산: [MEMORY_FIXED_EVAL_RESULTS_20261111.md](MEMORY_FIXED_EVAL_RESULTS_20261111.md). 실행 근거: `outputs/memory_fixed_epoch100_eval_20261110T150000Z`.

## 작은 학습 설정 변경 제안 — 2026-11-11, 미실행

첫 권고는 기존 M3의 memory decay=0을 유지하고 entropy를 고정된 약한 양수로 복원하는 대조다. Frozen legacy 단위에서 λ=0.00002와 0.000002(원래 0.0002의 1/10, 1/100)를 같은 seed 42/123·저장 초기값·전체 patch 순서로 처음부터 학습하는 두 후보를 제안한다. Encoder/decoder decay=1e−5와 그 밖의 조건은 고정한다. 기존 M3(λ=0)를 대조로 삼으며, 기존 M0 대비는 λ와 memory decay가 함께 달라지는 조합이다. 기존 R2는 λ를 3.4133배 올린 실험이므로 약화 방향의 결과를 대신하지 않는다. 양수 λ 자체가 실제 희소 동작이나 탐지 성능 향상을 보장하지 않으며 attention/readout과 일반화를 따로 확인해야 한다.

후순위로 100 epoch를 학습 예산으로 사용할 때 cosine T_max=100과 기존 T_max=1000의 단독 대조를 검토한다. 기존 진단은 의도적으로 원래 1000-epoch 시간축을 보존하여 epoch 100 이후 LR=9.757729755661011e−5(초기 1e−4의 97.58%)다. 이것을 새 구현 버그라고 부르지 않는다. 수정 trainer에는 이미 T_max=config["epochs"]가 적용되어 있다. 모든 비교군에 같은 예산·스케줄 조건을 적용하고 이전 100-epoch 수치와 바로 공정 비교라고 부르지 않는다. 약한 entropy와 스케줄 변경을 한꺼번에 적용해 원인을 섞지 않는다.

구체적인 λ 두 값은 frozen legacy 진단 단위다. 이 구현의 H_legacy=(1024/300)×H_query이므로 수정 entropy에서 같은 계수 강도에 대응하는 값은 6.8266667e−5 및 6.8266667e−6이다. 이 항의 환산이 다른 공통 오류 수정까지 포함한 학습 동등성을 뜻하지 않는다. 정식 비교는 수정 구현의 대조들을 동일 조건으로 다시 고정한다.

계수·parameter group·학습률 변경은 구조·손실 항을 유지하는 학습 설정 조정으로 설명할 수 있으나, 성능에 영향을 준 값·선정 절차·변경 사실은 학습 설정과 리뷰어 답변에 명시한다. 기존 오류 수정의 보고 필요성은 별개이며 minor revision 수용을 보장하지 않는다. 질환 점수를 반복 확인해 최고 설정을 택하지 않는다. 이번 턴은 아이디어와 우선순위만 기록했으며 코드 변경·새 학습·질환 평가를 시작하지 않았다.

근거: [MemAE 목적식](https://arxiv.org/html/1904.02639v2), [PyTorch cosine schedule](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), `docs/CANDIDATE_FAILURE_DIAGNOSIS_20261110.md`, M0/M1/M2/M3 결과와 frozen config/history. Notion 기존 memory 토글에 반영했다.


## 2026-11-11 SSM 복구 장시간 실행 시작

사용자 요청으로 수정 모델을 공통 기준으로 한 8조건×2seed×200epoch(16회, 총3,200epoch)를 GPU 두 장에서 시작했다. 두 seed 첫 epoch 완료; 실측 약34–36초/epoch, 전체 약16시간 예상. SSM decay 제외 범위, AdamW, dt bias 초기화 및 상호작용을 비교한다. 기존 정상 TRAIN 입력/λ=0/memory decay=0을 고정한 기전 실험이며 정식 k-fold 성능·최종 채택을 뜻하지 않는다. 세부 사전 고정 설계: `docs/SSM_REPAIR_RUNS_20261111.md`. 실행 근거: `outputs/ssm_repair_20261110T155100Z/manifest.json`, `verification.json`, `launch_record.json`. 기존 노션 토글 동기화 완료.


## 2026-11-11 SSM 복구 중간 인과 분석 (14/16회 완료)

S0–S6를 같은epoch200에서 비교했다. dt_proj만 decay를 제외한 S1은 가중치는 살아도 x_proj가 소멸해 기능은 회복되지 않았다. S2/S3/S4/S6는 두seed 모두 Δ 입력 의존성과 재귀 경로를 보존했고, 초기화 단독 S5는 실패했다. S3와 S4는 정상 TRAIN 복원도 두seed 모두 개선했다. AdamW는 동일 숫자의 decay가 같은 규제강도를 뜻하지 않아 SSM 밖 규제 약화가 섞여 있다. 질환 성능·최종 채택은 미확인이고 S7는 계속 실행한다. 자세한 수치·한계: `docs/SSM_REPAIR_INTERIM_20261111.md`; 재현 근거: `outputs/ssm_repair_analysis_20261111T051100Z`. 기존 노션 토글에 반영 완료.
