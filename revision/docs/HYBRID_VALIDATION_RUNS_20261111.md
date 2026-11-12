# CNN–memory–SSM 학습 검증

사용자는 `all stop` 뒤 재확인에 **“기존 실행 종료 후 새 검증 시작”**으로 답했다. 기존 SSM 실험을 종료 상태로 정리하고 아래 유한한 비교 실험을 시작했다. 최종 모델 채택은 아직 하지 않았다.

## 기존 실행 종료

- `outputs/ssm_repair_20261110T155100Z`: 15개 완료. `S7_all_ssm_no_decay_dt_init_seed123`은 epoch167 완료 후 중단.
- 호스트 프로세스와 GPU 작업을 확인했을 때 기존 실행은 이미 없었다. 정확한 종료 시각은 확인할 수 없으므로 추정하지 않았다.
- 실행 중으로 남았던 manifest/run_state를 `stopped`로 정정했다. 정정 전 상태는 `stop_reconciliation.json`에 보존했다. 완료된 결과와 checkpoint는 변경하지 않았다.

## 새 실행

- 루트: `outputs/hybrid_validation_20261111T055500Z`.
- 시작: **2026-11-11 15:58:21 KST**. Queue PID7386, 최초 학습 PID7397/7398, GPU1/0.
- **6조건 × seed42/123 × 고정200epoch = 12회, 2400epoch**. GPU당 한 학습을 순서대로 실행한다.
- 첫 epoch 실측 약34초. 기존 실행을 근거로 총 약11–12시간을 예상하되, 진단 비용·후속 조건 속도에 따라 갱신한다. 잠정 완료 예상은 11월12일 새벽3–4시 KST.
- 모든 작업이 끝나면 queue가 종료한다. 실패가 발생하면 다음 작업의 새 시작을 멈춘다.

| 조건 | 설정 | 분리하는 질문 |
| --- | --- | --- |
| V0 | 기존 S3와 같은 Adam, SS3D 전체·memory decay=0 | 같은 초기값·순서에서 S3 결과를 재현하는가? |
| V1 | V0 + decoder 가중치의 fan-in 초기화 | decoder 초기 스케일이 memory/encoder의 학습 신호와 수렴에 영향을 주는가? |
| V2 | Adam, 전체 weight decay=0 | 기존 S4의 이득이 AdamW 자체보다 결합 L2 규제 제거에 가까운가? |
| V3 | AdamW, memory·1차원 파라미터/bias·A_log decay=0; 나머지 1e-5 | 명시적 parameter group 구분이 기존 S4와 차이를 만드는가? |
| V4 | V0 + 학습 처음부터 재귀 성분 제외, D·u 유지 | 기본 초기화에서 재귀 경로를 학습시키는 것이 도움이 되는가? |
| V5 | V1 + 학습 처음부터 재귀 성분 제외, D·u 유지 | 초기화를 바꾼 경우에도 재귀 기여가 남는가? |

V0/V1/V4/V5는 초기화×재귀 사용의 2×2 대조다. V4/V5는 기여 확인용이며 논문의 주 모델에서 SSM을 제거하자는 결정이 아니다. Conv3d, gate, LayerNorm, D·u, 바깥 residual, memory는 유지한다. 이 조건에서 x_proj/dt/A의 task gradient가 연결되지 않는 것은 의도된 동작이며 경로가 보존됐다고 해석하지 않는다.

## 고정 조건과 범위

정상 TRAIN 15명/956patch, batch2, lr1e-4, cosine T_max1000, eta_min1e-6, entropy0, memory decay0, memory300, shrink_thres0.0025. 기존 SSM 실험과 같은 초기 가중치와 200epoch 전체 patch 순서를 파일 단위로 복사·검증했다. 학습 예산·평가 epoch는 고정하며 validation/test의 최고 점수로 선택하지 않는다.

모델은 방향 packing·hard shrink 수정 및 kernel3가 적용된 frozen 모델이다. 입력과 변환은 이전 기전 실험의 legacy 유래 TRAIN 자료를 유지한다. 정식 수정 전처리·전체 k-fold 평가를 대신하지 않는다. Entropy0·T_max1000도 원인 분리를 위한 공통 조건이며 최종 논문 설정으로 자동 채택하지 않는다.

Fan-in 조건은 **decoder 가중치9개만** 기존 난수 방향을 유지하며 rescale한다. 중간 Conv3d gain√2, ConvTranspose3d와 최종 출력 Conv3d gain1, PyTorch tensor fan-in 기준이다. Bias·encoder·memory 초기값은 같다. U-Mamba의 초기화에서 착안한 후보이며 U-Mamba 전체 구조 재현이나 GELU 최적 초기화라는 주장이 아니다.

## 시작 전 검증과 관찰

12개 모두 실제 GPU forward/backward, 3step optimizer update, 유한값, optimizer 누락/중복 없음, checkpoint/optimizer 재로딩과 복원 출력 일치 검증을 통과했다. 진단은 상태·RNG·출력을 바꾸지 않는다. No-op 재귀 개입의 출력은 정확히 같았다. V4/V5는 재귀-off 개입 전후 동일하며 dt 투영 gradient가 분리됨을 확인했다. V0와 과거 S3의 사전점검 loss는 두 seed에서 정확히 같다. 장기 학습의 bitwise 재현까지 보장하지는 않는다. 실제 첫 epoch의 V0 MSE는 과거 S3 대비 seed42 −1.03%, seed123 +0.089% 차이가 있었다. 초기 가중치·순서·optimizer group은 일치하므로 정확한 원인을 단정하지 않는다. CUDA scan backward 반올림 비결정성이 가능한 요인이며, 현재 V0를 동시 비교 기준으로 유지한다. 근거는 `replay_epoch1_check.json`이다.

기존 S0/S3/S4의 **고정 epoch200** 모델6개를 변경 없이 측정했다. 정상 TRAIN 각 환자의 앞/중간/뒤 patch 총45개에서 모든 raw scan의 재귀 성분만 끄고 D·u를 유지했다. 아래 값은 45개 patch 평균 MSE의 비율이며, patch별 비율 평균이나 임상 성능이 아니다.

| 기존 조건 | seed42: 재귀-off MSE 변화 | seed123: 재귀-off MSE 변화 | 해석 범위 |
| --- | --- | --- | --- |
| S0 · 기존 Adam 규제 | 0%, 45/45 출력 bitwise 동일 | 0%, 45/45 출력 bitwise 동일 | 측정한 패치에서 재귀를 꺼도 복원이 바뀌지 않음 |
| S3 · SS3D 전체 decay0 | +34.25% | +5.05% | 두 seed 모두 정상 복원에 재귀가 기여함 |
| S4 · AdamW1e-5 | +1.03% | −0.26% | 낮은 복원 MSE만으로 재귀 활용도가 높다고 할 수 없음 |

층별 개입에서는 S3의 마지막 stage 효과가 가장 컸다. 층별 효과는 비선형 상호작용이 있으므로 합산하지 않는다. 추론 시 재귀 제거는 현재 의존도를 측정하며, 처음부터 제외하고 재학습했을 때의 차이는 V4/V5로 따로 확인한다.

초기 seed42의 고정 정상 patch3개에서 V1은 V0보다 memory readout/query의 MSE gradient RMS가 약84배 컸다. 최종 출력 gradient는 거의 같았다. Decoder 스케일이 초기 학습 신호를 바꾼다는 근거이며, 이것만으로 학습 실패의 단일 원인이나 성능 개선을 확정하지 않는다. 같은 시점의 재귀-off MSE는 세 patch 모두 float32에서 같았다.

원자료: `verification.json`, 각 seed42의 `verification_probe.json`, `parent_checkpoint_audit/*.json`, `pretraining_diagnostic_summary.json`. 요약 재생성: `memory_pilot/summarize_hybrid_audit.py --root outputs/hybrid_validation_20261111T055500Z`.

## 완료 후 판단

1. 12개 실행의 고정 epoch200, 전체 정상956patch MSE와 seed별 값을 모두 제시한다. 중간 최고 epoch를 고르지 않는다.
2. V0와 기존 S3의 재현 차이를 먼저 확인하고, V1−V0/V5−V4 초기화 차이와 V4−V0/V5−V1 재귀 차이를 함께 본다.
3. 가중치 크기·입력 의존 Δ·재귀 성분뿐 아니라 decoder→memory→encoder task gradient, memory readout/attention 변화 및 실제 출력 개입을 함께 해석한다.
4. V2와 과거 S4 비교에서 동일 숫자 1e-5의 Adam/AdamW가 같은 규제 강도가 아님을 명시한다. 초기 AdamW lr×wd=1e-9의 직접 수축은 매우 작으며 float32에서 반올림될 수 있다.
5. 정상 복원이 좋아졌다는 사실을 AUROC/AP 개선으로 표현하지 않는다. 질환 탐지 검증·정식 k-fold 및 주 모델 채택은 별도 단계로 남긴다. 적용한다면 초기화·optimizer group·실제 설정 변경을 원고와 답변에 간결히 기록한다.

현재 상태: **12/12회 학습 및 최종45patch 진단 완료**. 고정200epoch 전체 결과, 2,160개 개입, 동일 패치·유한 수치·상태 불변과 프로세스 종료를 검증했다. 전체 비교·해석·리뷰어 보고 사항은 `docs/HYBRID_VALIDATION_FINAL_20261112.md`에 정리했다. 질환 성능·정식 k-fold·주 모델 채택은 이번 검증 범위 밖이다.

### 최종 집계와 완료 판정

`memory_pilot/analyze_hybrid_validation.py`는 frozen 파일 hash, 12개 조건/seed 조합, 전체200epoch/956patch/478step 이력, checkpoint의 epoch·seed·조건·config·유한 가중치·95600회 optimizer update, 필수 probe 및 전체 TRAIN audit를 확인한다. 재귀를 제외한 조건은 실제 재귀0·dt gradient 단절·재귀-off 출력 동일도 확인한다. 모든 작업의 성공 종료가 확인돼야 전체 완료로 표시한다.

예정된 epoch0/20/50/100/150/200 checkpoint 파일의 존재·비어 있지 않음도 확인한다. 200개 epoch의 실제 LR을 고정 cosine 시간축과 대조하고, 최종 scheduler의 last_epoch·T_max·eta_min·초기 LR 및 optimizer LR 일치를 확인한다. 가중치 전체의 유한값·metadata·update 수 검사는 최종200epoch checkpoint에 적용한다.

최종 확인 명령은 다음과 같다. 진행 중에는 `--require-complete` 없이 중간 진단을 모을 수 있지만 최종 표의 미완료 값은 채우지 않는다. 미완료 상태에서 완료 요구 옵션이 실제로 실패하는 것을 확인했고, 기존 완료 checkpoint4개는 CPU에서 metadata·유한 가중치·update 수 확인을 통과했다.

```bash
/home/jiwoong/anaconda3/envs/VMamba/bin/python -B memory_pilot/analyze_hybrid_validation.py \
  --root outputs/hybrid_validation_20261111T055500Z \
  --output outputs/hybrid_validation_20261111T055500Z/analysis \
  --require-complete --require-final-audit
```

최종 JSON/표는 각 seed의 전체 TRAIN MSE, 사전 정의한 paired 차이와 초기화×재귀 상호작용을 기록한다. 45patch의 memory 진단과 3patch의 재귀/gradient 진단은 별도 필드로 남긴다. 두 seed의 값으로 유의성이나 임상 CI를 만들지 않는다.

후보 결과를 보기 전 최종 확인을 보강했다. 학습12회가 모두 끝나고 GPU가 비었을 때 `memory_pilot/audit_hybrid_final.py --root outputs/hybrid_validation_20261111T055500Z --gpu 0`으로 최종 모델12개에 기존 parent와 같은 정상45patch 재귀 개입을 수행한다. 전체 재귀 및 stage별 재귀 제거, no-op 일치, checkpoint/state/RNG 불변을 확인한다. 이때 재학습하거나 질환 자료를 조회하지 않는다. 학습 도중3patch 결과를 전체15명에게 일반화하지 않기 위한 최종 확인이다. 최종 분석은 45patch 진단도 별도 완료 판정하며, 위 엄격 옵션은 두 단계가 모두 끝나야 통과한다.

실행을 확인하는 한시적 monitor는 현재 queue만 관찰한다. 학습 완료 시 CPU 기반 학습 결과 검증을 실행하고 종료한다. 이후 위45patch 진단과 최종 통합 해석이 남는다. 상시 감시나 새 실험 자동 탐색을 설정한 것이 아니다.

## 첫 완료 결과: V0 기준 재현 2/12회

V0 두 seed의 200epoch, 전체 정상 956patch 평가, 저장된 유한 가중치, 정확히 95,600회 optimizer update, 필수 진단·성공 종료를 확인했다. 두 GPU는 다음 V1 decoder 초기화 조건으로 넘어갔다.

| seed | 기존 S3 전체 TRAIN MSE | 이번 V0 전체 TRAIN MSE | 변화 |
| --- | --- | --- | --- |
| 42 | 0.000447692 | 0.000406215 | −9.26% |
| 123 | 0.000398234 | 0.000441889 | +10.96% |

두 seed 평균은 약+0.26% 차이지만 개별 seed는 약±10% 달랐다. 같은 초기값·순서·optimizer group이 장기 bitwise 재현을 보장하지 않으며, 이 한 번의 재현 차이를 모든 조건의 분산 추정치나 허용 오차로 일반화하지 않는다. 현재 V0를 주 비교 기준으로 사용한다. 두 모델의 고정3patch 진단은 Δ bias-only0/18이고, 재귀-off 시 평균 MSE가 각각15.21%·2.91% 증가했다. 최종45patch 재귀 진단은 아직 수행 전이다. 질환 성능 결과가 아니다.

저장된 환자별 평가에서도 같은 방향을 확인했다. Seed42는 정상15명 모두 MSE 감소, seed123은15명 모두 증가했다. 환자별 평균에 동일 가중치를 주면 −9.36%/+11.12%로, patch 가중 평균과 방향·크기가 비슷했다. 가장 큰 한 환자의 절대 차이 기여도는 각각10.65%/9.08%였다. 이 재현 차이를 특정 한 환자나 patch 수 편중만으로 설명할 수 없다는 관찰이며, 변동의 정확한 원인이나 임상 일반화를 증명하지 않는다. 주 비교 지표는 고정200epoch 전체 TRAIN MSE로 유지한다. 근거: `v0_replay_patient_differences.json`.

### AdamW 직접 감쇠의 수치 확인

후속 V2/V3 결과를 보기 전에 초기 seed42의 비-memory float32 텐서2,070,841개로 AdamW 직접 감쇠만 분리했다. Gradient를0으로 둔 단일 optimizer step에서 첫/마지막 학습 LR(1e-4, 약9.06375e-5)과 foreach False/True의 네 CPU 계산 모두 **바뀐 원소0, 최대 차이0**이었다. 이때 lr×wd는 약1e-9다. 근거는 `adamw_shrink_rounding_check.json`이다.

이 검사는 직접 수축이 반올림으로 사라질 수 있다는 실측이다. CPU/GPU 전체 학습 궤적의 동일성을 증명하지는 않는다. 현재 설정에서 V2/V3/과거 S4를 독립적인 유효 규제 강도 차이로 해석하거나, 작은 MSE 차이를 parameter group의 이득이라고 단정하지 않는다. V2는 기존 Adam의 결합 L2 규제 제거라는 가설을 확인하고, V3의 차이는 수치 경로·재현 변동과 함께 해석한다. 사전 설정과12개 실행은 유지한다.

### 감시 도구 수정

첫 V1 작업이 시작될 때 queue의 PID 기록과 자식의 run_state 생성 사이 간격 때문에 초기 monitor가 종료됐다. 학습 queue PID7386과 새 학습 PID20737/20823은 계속 실행됐고 학습 재시작은 없었다. `monitor_hybrid_validation.py`는 살아 있는 자식의 상태 파일 생성 전을 `starting`으로 처리한다. 이 시작 간격·파일 생성·프로세스 종료 관찰을 확인하고 monitor만 재시작했다. 세션69705→90531, 근거 `monitor_recovery.json`. Frozen 학습 코드·config·모델·순서는 변경하지 않았다.

## V1 decoder 초기화 완료: 4/12회

2026-11-11 19:45 KST 기준 V0·V1 네 실행의 성공 종료와 고정200epoch, 전체 정상956patch 평가, 모든 예정 checkpoint·LR 이력, 최종 유한 가중치·95,600회 optimizer update를 확인했다. V2 Adam 전체 decay0 두 seed가 이어서 시작됐다. 비교 기준은 현재 V0이며 과거 S3의 더 유리한 값을 대신 사용하지 않는다.

| seed | V0 정상 TRAIN MSE | V1 decoder fan-in MSE | V1 변화 |
| --- | --- | --- | --- |
| 42 | 0.000406215 | 0.000424928 | +4.61% |
| 123 | 0.000441889 | 0.000767974 | +73.79% |

**이번 고정200epoch 예산에서 decoder 초기화만 바꾸는 것은 정상 복원을 개선하지 않았다.** 두 seed 각각 정상15명 모두 MSE가 높았고 환자 균등 평균도 +4.81%/+73.38%였다. Seed123의 고정45patch MSE는 epoch100/150/200에 0.000768165/0.000767318/0.000768234로 비슷했다. 마지막 epoch에만 생긴 급등으로 설명되지 않는다. Seed42는 같은 probe가 epoch150의0.001285579에서200의0.000415078로 뒤늦게 낮아졌지만, 사전 지정한 최종 비교에서 V0보다 낮지는 않았다. 저장된 이력의 사후 진단이며 최고 checkpoint를 재선택하지 않는다. 모든 초기화 방식이나 더 긴 학습 예산으로 일반화하지 않는다.

입력 의존 Δ는 V1 두 seed 모두 유지됐다(bias-only0/18, dt task-gradient 연결6/6). 전체 TRAIN memory readout rank99는 V0의16/15에서 V1의15/11로, effective global slots는215.8/238.1에서152.3/143.9로 바뀌었다. 이 지표만으로 memory가 임상적으로 유용해졌거나 붕괴했다고 판정하지 않는다. 초기 seed42의 약84배 gradient 증가는 최종 이득을 보장하지 않았다.

고정3patch에서 재귀 제거 시 MSE 변화는 V0의+15.21%/+2.91%에서 V1의+32.53%/+2.56%였다. Seed42의 의존도는 커졌으나 두 seed에서 일관된 증가가 아니며, 정상 복원 자체도 개선되지 않았다. 현재 의존도와 처음부터 재귀 없이 학습한 대조의 이득을 구분한다. 최종45patch 진단 및 V4/V5 재학습 결과는 아직 없다.

**판단:** V1을 개선안으로 채택할 근거는 현재 없다. 초기 gradient를 키우는 것만으로 충분하다는 가설은 이번 결과가 지지하지 않는다. 사전 지정한 V2/V3 규제 비교와 V4/V5 초기화×재귀 대조는 그대로 완료한다. 질환 AUROC/AP·정식 k-fold·논문 주 모델 변경은 이 결과로 결정하지 않는다. 적용할 경우 decoder 초기화와 optimizer 설정은 실제 변경 내용으로 기록한다.

근거: `analysis/analysis.json`, `v1_completed_diagnostics.json`, 각 V0/V1의 `full_normal_training_epoch_200.json`, `probe_epoch_200.json`, `history.json`. 진단 파일의 마지막20epoch 학습 평균·범위는 사후 참고치이며 고정200epoch 전체956patch 최종값을 대체하지 않는다.

## V2 Adam 전체 decay0 완료: 6/12회

2026-11-11 21:40 KST 기준 V0·V1·V2 여섯 실행의 고정200epoch, 전체 정상956patch 평가, 예정 checkpoint·LR 이력, 최종 유한 가중치·95,600회 optimizer update와 성공 종료를 확인했다. V3 AdamW 명시적 parameter group 조건 두 seed가 실행 중이다.

| 조건 | seed42 정상 TRAIN MSE | seed123 정상 TRAIN MSE | 비교에서의 위치 |
| --- | --- | --- | --- |
| V0 · SS3D와 memory만 decay0 | 0.000406215 | 0.000441889 | 이번 suite 기준 |
| V2 · Adam 전체 decay0 | 0.000042409 | 0.000094840 | 이번 완료 조건 |
| 과거 S4 · AdamW1e-5, memory decay0 | 0.000050277 | 0.000088359 | 사전 지정한 역사적 비교 기준 |

사전 지정한 V2−과거 S4 비교에서 변화는 seed42 **−15.65%**, seed123 **+7.34%**로 방향이 다르다. 동등성 검정이나 V2 우월성의 근거로 표현하지 않는다. 동일한 초기값·순서에도 장기 재현 변동이 있고, 현재 AdamW lr×wd≈1e-9의 직접 수축은 float32에서 반올림될 수 있다는 기존 수치 확인과 함께 본다.

현재 V0와의 **추가 서술 비교**에서는 V2가 **−89.56%/−78.54%**였다. 두 seed 각각 정상15명 모두 MSE가 낮았으며 환자 균등 평균도 −89.48%/−78.40%였다. 이 추가 비교는 새로운 checkpoint 선택 기준이나 통계 검정이 아니다. 고정45patch probe도 epoch100→150→200에 V2 seed42는0.000137759→0.000070767→0.000042069, seed123은0.000219027→0.000132058→0.000094014로 줄었다. 주 지표는 전체956patch의 고정200epoch 결과로 유지한다.

**해석:** 낮은 정상 복원 MSE는 AdamW에서만 얻을 수 있는 결과가 아니었다. Adam의 결합 L2 규제 제거가 기존 AdamW의 개선을 설명하는 중요한 요인이라는 가설을 지지한다. V0는 이미 SS3D·memory의 decay가0이므로 V2와의 차이는 SSM 밖에 남아 있던 규제까지 제거한 것이다. Stem·downsample·decoder 등 여러 파라미터가 함께 바뀌므로 decoder 하나의 영향으로 좁히거나 단일 원인을 확정하지 않는다.

V2는 epoch100과200 두 seed 모두 Δ bias-only0/18, dt task-gradient 연결6/6, 모든 dt 투영이 subnormal인 블록0/6이었다. 최종 전체 TRAIN query rank99는 V0의7/8에서 V2의31/29, readout rank99는16/15에서20/19로 바뀌었다. 반면 effective global slots는215.8/238.1에서27.3/26.1로 줄었다. 더 많은 슬롯 사용이나 더 낮은 정상 MSE 중 하나만으로 memory의 질환 탐지 유용성을 판정하지 않는다.

고정3patch에서 재귀 제거 시 MSE 증가는 V2 **+1.71%/+3.06%**였다(V0 +15.21%/+2.91%). 입력 의존 Δ와 raw 재귀 성분은 살아 있어도 최종 복원이 그 경로에 강하게 의존한다고 자동으로 결론 내릴 수 없다. 전체45patch 최종 개입은 아직 전이며, V4/V5의 처음부터 재귀를 제외한 재학습도 남아 있다. 현재 값은 질환 성능이나 장거리 의존성의 증명이 아니다.

**판단·보고:** 규제 설정이 정상 학습에 미치는 영향은 초기화만 바꾼 V1보다 훨씬 컸다. 주 모델 자동 채택은 하지 않고 V3 및 V4/V5를 완료한다. 적용할 때는 실제 optimizer와 parameter별 decay 설정을 재현 가능하게 명시한다. 이번 조건은 entropy0·legacy 유래 정상 TRAIN 등 기전 대조이며, 정식 전처리·k-fold·AUROC/AP 검증은 별개다.

근거: `analysis/analysis.json`, `v2_completed_diagnostics.json`, 각 V2의 `probe_epoch_100.json`·`probe_epoch_200.json`·`full_normal_training_epoch_200.json`·`epoch_200.pt`. 추가 서술 비교와 사전 지정 비교를 진단 파일에서 구분했다.

## V3 AdamW 명시적 parameter group 완료: 8/12회

2026-11-11 23:33 KST 기준 V0–V3 여덟 실행의 고정200epoch, 전체 정상956patch 평가, 예정 checkpoint·LR 이력, 최종 유한 가중치·95,600회 optimizer update와 성공 종료를 확인했다. V4 재귀 제외 대조 두 seed가 실행 중이다.

| 조건 | seed42 정상 TRAIN MSE | seed123 정상 TRAIN MSE | 비교에서의 위치 |
| --- | --- | --- | --- |
| V2 · Adam 전체 decay0 | 0.000042409 | 0.000094840 | 이번 suite의 추가 서술 비교 |
| V3 · AdamW 명시적 group | 0.000048352 | 0.000050920 | 이번 완료 결과 |
| 과거 S4 · AdamW1e-5, memory decay0 | 0.000050277 | 0.000088359 | 사전 지정한 역사적 비교 기준 |

사전 지정한 V3−과거 S4의 MSE 변화는 **−3.83%/−42.37%**였다. V2와의 추가 서술 비교에서는 **+14.01%/−46.31%**로 방향이 달랐다. Seed123의 차이가 크다는 관찰을 보존하되, 이것을 parameter grouping의 인과 효과나 두 seed 사이 안정성 개선의 증명으로 해석하지 않는다. 현재 lr×wd≈1e-9의 AdamW 직접 수축은 기존 CPU float32 분리 검사에서 반올림되어 원소가 바뀌지 않았다. 이 결과는 전체 GPU 학습 궤적의 동일성을 보장하지 않지만, 현재 비교로 실효 규제 강도 차이를 분리하기 어렵다는 제한이다. 같은 초기값·순서의 V0 재현 변동도 있었으므로 작은 차이만 문제라고 보거나, 모든 차이를 특정 수치 오차 하나로 설명하지 않는다.

V3 두 seed 모두 Δ bias-only0/18, dt task-gradient 연결6/6, dt 전체 subnormal 블록0/6이었다. 전체 TRAIN query rank99는28/32, readout rank99는18/18, effective global slots는26.1/25.7이었다. 고정3patch 재귀 제거 시 MSE는 **+3.06%/+0.72%** 증가했다. 낮은 정상 복원 MSE를 얻었다는 사실과 강한 재귀 의존도·질환 탐지 유용성은 구분한다. 최종45patch 개입은 아직 수행 전이다.

**판단·보고:** V3도 낮은 정상 복원 오차를 보였으나, 현재 비교만으로 grouping 자체를 성능 개선의 원인으로 채택하지 않는다. 실제 적용 시 optimizer·각 group의 decay·수치 정밀도를 명시한다. V4/V5와 최종45patch 진단을 마친 뒤 전체 결과를 해석한다. 질환 성능·정식 k-fold·주 모델 자동 채택은 아직 없다.

### V4 실제 학습 시작 확인과 남은 시간

V4 seed42의 실제 epoch1 저장 진단에서 dt gradient 연결0/6, raw 재귀0/18, 재귀-off 개입 전후 복원3/3 bitwise 동일과 no-op 일치를 확인했다. 이는 의도된 대조이며 학습 실패가 아니다. Conv3d·gate·LayerNorm·D·u·바깥 residual·memory는 유지한다. 보존된 dt 가중치나 계산된 Δ만으로 활성 재귀라고 해석하지 않는다. 두 seed의 최종200epoch 대조 검증은 여전히 남아 있다.

Seed42의 첫31epoch 기준 학습 시간 중앙값은 약28.83초/epoch였다. 23:34 KST 시점에서 남은 V4/V5 학습은 대략3시간 반과 최종 추론 진단 시간이 더 필요할 것으로 본다. 한 GPU의 초기 실측 추정이며 두 GPU의 속도·진단 비용에 따라 달라질 수 있다. 학습 예산이나 순서는 변경하지 않았다.

근거: `analysis/analysis.json`, `v3_completed_diagnostics.json`, 각 V3의 최종 checkpoint·평가·진단, V4 seed42의 `probe_epoch_001.json`·`history.json`.

## V4 첫 seed 완료: 9/12회 · 부분 결과

2026-11-12 00:53 KST 기준 V4 seed42의 고정200epoch·전체 정상956patch 평가·유한 가중치·95,600회 update·예정 checkpoint와 LR 이력·성공 종료를 검증했다. 전체 TRAIN MSE는 **0.000419971**, 같은 seed의 현재 V0(0.000406215) 대비 **+3.39%**였다. Seed123은 아직 학습 중이므로 재귀의 일반적 이득·불필요성·동등성에 대한 결론은 내리지 않는다. V5 seed42가 순서대로 시작됐다.

의도된 대조 동작도 확인했다. 최종3patch 진단에서 dt task-gradient 연결0/6, raw 재귀0/18, 재귀-off 전후 출력3/3 bitwise 동일과 no-op 일치를 확인했다. 두 seed의 epoch100에서도 같은 대조 조건을 확인했다. Δ 계산이나 투영 가중치 보존은 활성 재귀의 근거가 아니므로 자동 생성 표에서는 V4/V5의 Δ 칸을 ‘재귀 제외’로 표시한다. 학습 코드·조건은 변경하지 않았다.

근거: `analysis/analysis.json`, `v4_first_seed_diagnostics.json`, `v4_epoch100_control_checks.json`. 최종45patch 진단·질환 AUROC/AP·정식 k-fold·모델 채택은 미완료다.

## V4 재귀 제외 재학습 완료: 10/12회

2026-11-12 01:09 KST 기준 V0–V4 열 실행의 고정200epoch, 전체 정상956patch 평가, 예정 checkpoint·LR 이력, 최종 유한 가중치·95,600회 optimizer update와 성공 종료를 검증했다. 마지막 V5 초기화 변경+재귀 제외 조건의 두 seed가 실행 중이다.

| seed | V0 · 재귀 사용 정상 TRAIN MSE | V4 · 처음부터 재귀 제외 MSE | V4 변화 |
| --- | --- | --- | --- |
| 42 | 0.000406215 | 0.000419971 | +3.39% |
| 123 | 0.000441889 | 0.000404230 | −8.52% |

**현재 기본 초기화·고정200epoch 조건에서는 재귀를 학습시키는 일관된 정상 복원 이득이 확인되지 않았다.** 재귀를 제외하면 seed42에서는 오차가 늘고 seed123에서는 줄었다. 추가 환자별 서술 비교에서도 seed42는15명 모두 증가, seed123은15명 모두 감소했으며 환자 균등 평균 변화는 +3.44%/−8.61%였다. 두 seed의 반대 방향을 평균만으로 숨기거나 동등성으로 해석하지 않는다. 기존 V0 재현 변동이 있었지만 그 한 번의 차이를 일반적 잡음 한계나 통계적 유의성 기준으로 적용하지 않는다.

**학습된 모델의 의존도와 재학습 시 이득은 다르다.** 현재 V0의 고정3patch에서 재귀를 추론 시에 끄면 MSE가 +15.21%/+2.91% 늘었지만, V4처럼 처음부터 없이 학습하면 다른 경로가 적응할 수 있다. 두 관찰은 모순이 아니며, V4 결과가 V0의 재귀 경로가 작동하지 않았다는 뜻도 아니다. Conv3d·D·u·gate·LayerNorm·바깥 residual·memory를 유지한 대조이므로 전체 SSM 블록 제거 또는 CNN-only 모델로 부르지 않는다. V5가 끝나야 초기화 조건과의 상호작용까지 비교할 수 있다.

V4 두 seed 모두 최종3patch에서 dt task-gradient 연결0/6, raw 재귀0/18, 재귀-off 전후 출력3/3 bitwise 동일과 no-op 일치를 통과했다. Epoch100에서도 두 seed의 같은 대조 동작을 확인했다. Δ/투영 가중치가 남아 있어도 재귀가 활성화됐다고 해석하지 않는다. 최종 전체 TRAIN query rank99는6/8, readout rank99는16/14, effective global slots는236.0/218.8이었다. 이 지표만으로 memory의 질환 탐지 유용성을 판단하지 않는다.

**판단·보고:** 현재로서는 재귀가 일관된 학습 이득을 준다는 근거가 부족하다. SSM 불필요성이나 질환 AUROC/AP 동등성이 증명된 것은 아니다. V5·최종45patch 개입을 마친 뒤 전체 결과를 해석한다. 논문에 이 대조를 쓴다면 ‘동일 예산·두 seed, 학습 처음부터 재귀만 제외, 나머지 경로 유지’와 seed별 결과를 명시한다. 주 모델에서 재귀를 제거하는 변경은 자동 채택하지 않는다.

근거: `analysis/analysis.json`, `v4_completed_diagnostics.json`, `v4_epoch100_control_checks.json`, 각 V4의 `full_normal_training_epoch_200.json`·`probe_epoch_200.json`·`epoch_200.pt`. 이전9/12 부분 결과도 위에 보존했다.

## V5 첫 seed 완료: 11/12회 · 부분 결과

2026-11-12 02:28 KST 기준 V5 seed42의 고정200epoch·전체 정상956patch 평가·95,600회 update·유한 가중치·예정 checkpoint와 LR 이력·성공 종료를 검증했다. 전체 TRAIN MSE는 **0.000406092**다. 같은 seed의 V4 대비 **−3.30%**(재귀 없이 decoder 초기화 비교), V1 대비 **−4.43%**(fan-in 초기화에서 재귀 제외 비교)였다. Seed123은 아직 학습 중이므로 일반적 초기화 효과나 재귀 이득의 결론은 보류한다.

최종3patch에서 dt task-gradient 연결0/6, raw 재귀0/18, 재귀-off 전후 출력3/3 bitwise 동일과 no-op 일치를 확인했다. 먼저 저장된 중간 진단(seed42 epoch50, seed123 epoch20)에서도 의도된 대조 동작과 유한 학습 이력을 확인했다. 최종45patch 진단은 두 seed 학습 종료 뒤 수행하며, 현재 결과는 질환 성능이 아니다.

근거: `analysis/analysis.json`, `v5_first_seed_diagnostics.json`, `v5_early_control_checks.json`, V5 seed42의 최종 checkpoint·평가·진단.

## 최종 완료: 12/12회 + 45patch 진단

2026-11-12 02:45 KST 학습 queue가 성공 종료됐고, 02:46 KST 최종12개 모델 진단도 완료됐다. 마지막 V5 seed123 정상 TRAIN MSE는0.000666327로 V4 대비+64.84%, V1 대비−13.24%였다. V5 두 seed와 전체2×2 대조, 모든 조건의 최종45patch 결과를 `docs/HYBRID_VALIDATION_FINAL_20261112.md`에 통합했다. 이전 부분 결과를 삭제하지 않았다.

엄격한 최종 검사와 별도의 parent 동일45patch·2,160개 개입 유한값 검사를 통과했다. 마지막 별도 집계에서 summary.json을 개별 parent 모델로 읽던 파일 선택을 바로잡았으며 학습/추론 실험은 재실행하지 않았다. 완료 후 호스트 확인에서 queue·모든 학습 PID가 없고 GPU 계산 작업도 없었다. 남아 있던 active pointer를 완료로 정정했다.

판단: 구조 유지+Adam 전체 decay0를 다음 검증 후보로 우선하되 질환 성능 개선이나 주 모델 채택은 아니다. Decoder fan-in은 일관된 개선이 없었고, 재귀 사용의 정상 재학습 이득도 일관되지 않았다. Memory2048 문제는 미검증이다. 원점수·한계·리뷰어 보고 내용은 최종 문서를 기준으로 한다.
