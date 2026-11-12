# SSM 수축 원인 분리 및 복구 실험 — 2026-11-11

사용자가 모델 감사에서 발견한 문제를 해결할 후보들을 준비해 10시간을 넘는 분량으로 실행하도록 요청했다. 아래 16회는 학습 전에 고정한 기전 실험이다. 최종 논문 모델 채택이나 탐지 성능 향상을 뜻하지 않는다.

실행 폴더: `outputs/ssm_repair_20261110T155100Z`.
시작·완료·실패의 실제 상태는 이 폴더의 `manifest.json`, 각 job의 `run_state.json`과 `history.json`을 기준으로 확인한다. 00:53 KST에 큐 PID 56255로 시작했다. 두 기준 seed의 첫 epoch 완료를 확인했고 이후 조건 14회가 대기한다. 16회 사전 GPU update/저장 복원과 기존 모델 수정 테스트3개를 통과했다. 첫 epoch 실측33.77/36.17초로 진단·저장을 포함해 약16시간, 11월11일17시 전후 완료 예상이다. 추정 시간이며 경합·오류 시 달라진다. Notion의 기존 Memory 붕괴 대응 토글에 조건표와 실행 기록을 반영했다.

## 공통 조건과 비교 범위

- 8조건 × seed 42/123 × 고정 200 epoch = 총 3,200 epoch, 16회. GPU 0/1 각각 한 job, 유한 대기열. 예전 약 34초/epoch 기준 약 15.1시간 벽시계 시간(약 30.2 GPU-hours); 실제 수정 모델 속도로 다시 추정한다.
- **새 공통 기준 모델**: 현재 수정 `main_files/model.py`를 복제하여 고정. 방향별 dt/B/C packing, 실제 hard shrinkage, kernel3 down/up을 모든 조건에 동일하게 적용한다. Legacy M3의 가중치를 이어받지 않고 새로 초기화한다. 원본 모델 소스는 수정하지 않는다.
- 원인 분리 배경: entropy 계수=0, memory weight decay=0을 모든 조건에 고정한다. 이는 기존 M3와 같은 두 설정을 사용하는 진단 대조이며, 원 논문 손실을 그대로 유지한 최종 주 모델이라고 표현하지 않는다. Hard shrink threshold는 0.0025로 유지한다.
- 기존 frozen 정상 TRAIN 15명/956patch, fold1의 입력 manifest와 SHA256, 전처리 float32 `(x+300)/700`을 유지한다. 전처리까지 수정한 전체 파이프라인의 k-fold 재평가는 아니다. 질환·validation·test를 로드하지 않는다.
- 같은 seed의 모든 조건은 같은 기본 가중치와 전체 patch 순서를 사용한다. 초기화 조건은 **6개 dt_proj.bias만** 다르다. 200회 순열 모두 전체 patch를 한 번씩 포함하며 앞 100회는 기존 M3 순열과 동일하다. 학습 sampling을 추가하지 않는다.
- Memory 300, dims 32/64/128, depths 2/2/2, state8, patch64×32×32, batch2. 초기 LR 1e-4, eta_min 1e-6, cosine T_max=1000, gradient clipping=1, float32. Scheduler는 기존 기전 대조 시간축을 유지하며 이 배치의 탐색 변수가 아니다.
- 기존 checkpoint에서 dt weight가 감소한 시점보다 긴 200 epoch 동안 보존·재수축을 확인한다. 동일 고정 예산이며 조기 종료나 최고 checkpoint 선택은 하지 않는다.

## 사전 고정한 대조

| ID | S0에서 바꾸는 설정 | 확인할 원인 | 채택 시 기록할 내용 |
| --- | --- | --- | --- |
| S0 | 없음: Adam, memory 외 decay=1e-5 | 수정 모델에서 현상 재현 | 공통 구현·진단 배경·학습 설정 |
| S1 | dt_proj weight/bias decay=0 | Δ 최종 투영에 대한 decay | 제외한 파라미터 명시 |
| S2 | x_proj·dt_proj·A_log·D decay=0 | Δ 입력, B/C, 시간척도까지의 수축 | x_proj에는 B/C도 포함됨을 명시 |
| S3 | SS3D 내부 전체 decay=0 | 입력/출력 투영·conv·norm까지의 수축 | S2보다 넓은 제외 범위 |
| S4 | Adam→AdamW, decay 숫자는 동일 | 결합된 Adam decay와 분리된 decay의 차이 | optimizer 교체. 동일 숫자가 동일 규제 강도라는 뜻은 아님 |
| S5 | dt bias만 초기화 | 초기 시간척도와 학습 신호 | softplus(bias)가 log-uniform [0.001,0.1] |
| S6 | S2 + S5 | 동역학 decay 제외와 초기화의 상호작용 | S2/S5/S0와 비교 |
| S7 | S3 + S5 | 더 넓은 보존 범위와 초기화의 상호작용 | S3/S5/S6와 비교 |

S1의 dt 투영만 제외했는데 upstream x_proj가 수축하면 입력 의존성이 회복되지 않을 수 있다. 따라서 S2를 별도로 둔다. S3에는 SS3D의 in/out projection, depthwise conv 및 out_norm도 포함되지만 VSSBlock의 외부 ln_1·stem·decoder는 포함되지 않는다. Memory decay는 모든 조건에서 0이다.

초기화는 bias만 바꿔 효과를 분리한다. 원래 Conv1d dt weight의 균등분포 범위 ±1/√dt_rank는 보존한다. 공식 Mamba의 모든 초기화·parameter-group 관행을 통째로 복제했다고 표현하지 않는다. A_log와 D의 초기값도 모든 조건에서 같다.

## 진단과 해석 기준

Epoch 0/1/3/5/10/20/30/50/75/100/150/200에서:

1. SSM 각 파라미터의 float64 norm·max·nonzero·subnormal 수, reconstruction gradient 및 clipping 후 크기, decay×parameter, Adam moment를 저장한다. AdamW의 decay는 task gradient에 더해지는 항이 아님을 구분한다.
2. 첫 3개 정상 TRAIN 환자의 고정 middle patch에서 6블록의 Δ가 bias-only와 같은지, 입력 기여·token 변동이 남는지 기록한다. 전체 입력 분포에 대한 증명으로 확대하지 않는다.
3. scan 출력의 재귀 성분과 D·u 성분 RMS, A 시간척도를 함께 관찰한다. LayerNorm 이전 비율이므로 최종 성능 기여도로 등치하지 않는다.
4. 정상 TRAIN 15명의 고정 first/middle/last 45patch에서 memory 중복·norm·사용량, query/readout rank99, 정상 MSE를 기록한다. 최종 epoch200에는 전체956patch를 진단한다.

**큰 가중치가 남는 것만으로 해결이라고 판단하지 않는다.** Δ의 실제 입력 의존성, task gradient, 재귀 경로, memory/query 변화와 복원을 함께 비교하고 두 seed의 결과를 모두 보고한다. 정상 복원이 좋아져도 질환 탐지가 좋아진다고 단정하지 않는다. 이 배치에는 AUROC/AP 기반 선택이나 자동 주 모델 교체가 없다. 후속 정식 평가는 공통 구현·사전 고정 선택 절차 아래 별도로 설계한다.

## 실행 보호와 재현 자료

- 실행 코드, 수정 모델 소스, config, 초기값, 입력 및 배치 순서를 snapshot/SHA256으로 고정했다.
- 각 조건에서 실제 GPU forward/backward/update, finite gradient, optimizer 전체 파라미터 중복/누락 여부, checkpoint 저장·복원을 실행 전에 검사한다. 같은 seed의 초기 상태 차이가 의도한 bias에만 있는지 확인한다.
- 진단 hook 전후의 모델 상태·출력·RNG 불변을 검사한다. 진단은 optimizer update를 하지 않는다.
- 각 epoch의 loss·시간·완료 수, 100 step마다 heartbeat. Epoch0/20/50/100/150/200에는 모델·optimizer·scheduler·RNG checkpoint를 atomic 저장한다.
- 각 GPU가 비어 있을 때 한 job만 시작하고 실패가 있으면 새 job 시작을 중단한다. 이미 실행 중인 다른 job은 마친다. 무한 재시도·추가 조건 자동 생성은 없다.
- 큐 프로세스는 터미널과 분리해서 시작한다. 컴퓨터 종료·프로세스 강제 종료·전원 장애를 넘어 자동 재개하는 서비스는 아니다. 완료 후 `results_summary.json`을 생성한다.
- 원본 CT를 외부에 올리지 않는다. Notion에는 실험 설정·시작 상태만 남긴다.

감사 근거: `docs/MODEL_STRUCTURE_AUDIT_20261111.md`와 `outputs/model_structure_audit_20261110T152700Z`.
