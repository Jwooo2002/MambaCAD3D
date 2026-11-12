# CNN–memory–SSM 직접 검증 최종 결과

2026-11-12 KST. **6조건×2seed×고정200epoch, 12/12회 학습과 최종45patch 진단을 완료했다.** 학습·대기열·진단 프로세스가 종료됐고 GPU 계산 작업은 없다. 기존 중단 실험을 재개하거나 주 모델을 자동 변경하지 않았다.

아래는 정상 TRAIN15명/956patch의 고정200epoch 복원 MSE다. 낮을수록 정상 복원 오차가 작으며 AUROC/AP가 아니다. 재귀-off 열은 같은 정상45patch에서 재학습 없이 재귀만 끈 평균 MSE 변화율이다. 전체956patch 값이나 patch별 비율 평균과 다르다. 최종 표에는 이전 중간3patch 수치 대신45patch 결과를 사용했다.

| 조건 | seed42 정상 TRAIN MSE | seed123 정상 TRAIN MSE | 45patch 재귀-off MSE 변화: 42 / 123 | 비고·리뷰어 보고 |
| --- | --- | --- | --- | --- |
| V0 · 기준 Adam/SS3D decay0 | 0.000406215 | 0.000441889 | +14.82% / +1.96% | 동시 기준; 과거 S3 재현 차이 −9.26%/+10.96% |
| V1 · V0+decoder fan-in | 0.000424928 | 0.000767974 | +27.59% / +1.32% | V0 대비 +4.61%/+73.79%; 초기화 이득 미확인 |
| V2 · Adam 전체 decay0 | 0.000042409 | 0.000094840 | +1.90% / +3.06% | V0 대비 −89.56%/−78.54%; optimizer/decay 명시 |
| V3 · AdamW 명시적 group | 0.000048352 | 0.000050920 | +2.77% / +0.83% | 낮은 정상 MSE; grouping 인과 효과 미분리 |
| V4 · V0+재귀 제외 재학습 | 0.000419971 | 0.000404230 | 0% / 0% (의도된 대조) | V0 대비 +3.39%/−8.52%; 유지한 경로·seed별 값 명시 |
| V5 · V1+재귀 제외 재학습 | 0.000406092 | 0.000666327 | 0% / 0% (의도된 대조) | V1 대비 −4.43%/−13.24%; 주 모델 채택 아님 |

## 확인된 문제와 해석

**규제 설정이 중요한 학습 제약이었다.** 기존 S0에서는 Adam의 결합 L2가 약한 SSM task gradient에 비해 강해 x_proj/dt 투영이 subnormal로 축소됐고, 기존45patch 검사에서 두 seed 모두 재귀를 꺼도 출력이 정확히 같았다. SS3D 전체 decay0인 V0에서는 입력 의존 Δ·task gradient·실제 재귀 의존도가 회복됐다. SSM 밖의 규제까지 제거한 V2는 V0보다 정상 MSE가78.54–89.56% 낮았다. Stem/downsample/decoder 등 여러 파라미터가 함께 바뀌므로 decoder 하나의 원인으로 좁히지 않는다.

**Decoder 초기화만으로 해결되지 않았다.** V1의 초기 memory 관련 gradient는 약84배 커졌지만 두 seed의 최종 결과는 V0보다 나빴다. 재귀 없이 초기화만 비교한 V5−V4도 −3.30%/+64.84%로 일관된 개선이 아니다. 이번 fan-in 후보를 개선안으로 채택할 근거는 부족하다.

**재귀가 작동하는 것과 재귀를 학습시키는 이득은 다르다.** V0/V1은 추론 시 재귀를 끄면 오차가 늘지만, 처음부터 제외한 V4−V0는 +3.39%/−8.52%, V5−V1은 −4.43%/−13.24%였다. 현재 조건에서 재귀 학습이 더 좋은 정상 복원을 준다는 일관된 근거는 얻지 못했다. 다른 경로가 적응할 수 있지만 정확한 보상 기전이나 SSM 불필요성을 증명한 것은 아니다. V4/V5는 Conv3d·D·u·gate·LayerNorm·바깥 residual·memory를 유지하며 CNN-only 모델이 아니다.

**낮은 정상 MSE가 강한 재귀 활용이나 질환 탐지 개선을 뜻하지 않았다.** V2/V3의45patch 재귀 제거 효과는 각각 +1.90%/+3.06%, +2.77%/+0.83%다. Query rank99는 V0의7/8에서 V2의31/29로 늘었지만 effective global slots는215.8/238.1에서27.3/26.1로 줄었다. 높은 rank·많은 슬롯·낮은 MSE 중 하나만으로 memory의 질환 유용성을 판단하지 않는다.

**AdamW grouping의 인과 효과는 분리되지 않았다.** V2−과거S4는 −15.65%/+7.34%, V3−과거S4는 −3.83%/−42.37%였다. 현재 lr×wd≈1e-9의 직접 수축은 분리한 CPU float32 검사에서 반올림됐다. 이 검사가 전체 GPU 학습 궤적의 동일성을 증명하지는 않는다. Seed123의 큰 차이도 보존하되 grouping의 이득·동등성·seed 안정성의 증명으로 표현하지 않는다.

## 2×2 대조와 재현 변동

원시 MSE의 상호작용 `(V5−V1)−(V4−V0)`는 seed42 −0.000032591674, seed123 −0.000063988064다. 이번 실행에서 fan-in 초기화 쪽의 재귀 제거 효과가 더 유리한 방향이었다는 서술이며 유의성 검정이 아니다. 모든 사전 지정 비교의 원점수·차이와 stage별45patch 결과는 `analysis/analysis.json`에 보존했다. 층별 제거 효과는 비선형 상호작용이 있으므로 합산하지 않는다.

V0와 과거S3는 같은 초기값·순서·설정에서도 −9.26%/+10.96% 차이가 있었다. 이 한 번의 재현 차이를 일반적 분산 추정치나 유의성 기준으로 쓰지 않는다. 두 seed의 반대 방향을 평균으로 숨기거나 좋은 seed·중간 checkpoint를 재선택하지 않았다. V2−V0 등 사후 추가 서술 비교는 원래 지정한 역사적 비교와 구분했다.

## 적용 판단과 리뷰어 보고

**다음 검증 후보는 구조를 유지한 Adam 전체 decay0(V2)를 우선한다.** 이번 정상 학습 결과에 따른 권고이며 임상 최종 모델 선정은 아니다. Decoder fan-in은 우선 채택하지 않는다. Mamba3 교체나 SSM 제거를 이 결과만으로 채택할 근거도 없다. 정식 전처리와 사전에 고정한 학습·모델 선택 기준의 k-fold AUROC/AP를 확인해야 실제 탐지 성능을 판단할 수 있다. 이 후속 질환 실험은 이번 suite에 포함되지 않았고 새로 실행하지 않았다.

실제 적용한 optimizer·parameter별 weight decay·초기화·학습 시간축은 원고와 답변에 재현 가능하게 명시한다. V4/V5를 원인 분리 ablation으로 보고한다면 학습 처음부터 재귀만 제외했다는 점, 유지한 경로, 동일 예산과 seed별 결과를 적는다. 진단용 대조와 논문 주 모델에서 재귀를 제거하는 변경을 구분한다.

## 범위와 완료 검증

Entropy0, memory300, shrink_thres0.0025, lr1e-4, cosine T_max1000/eta_min1e-6, batch2를 공통 고정했다. 방향 packing·hard shrink 수정/kernel3 frozen 모델에 legacy 유래 정상 TRAIN 입력을 사용했다. 논문의 최종 설정을 자동 대체하지 않는다. 사용자가 기억한 memory2048→300 문제는 정확한 기존 설정·실패 로그가 확인되지 않았고 이번 suite는300만 사용했다. 따라서 해결됐다고 주장하지 않으며 과거 hardtanh와 수정 hard-shrink를 혼동해 실패 원인을 단정하지 않는다.

12개 모두200epoch·956patch/478step 이력, 95,600회 update, 예정 checkpoint, LR/scheduler, 유한 최종 가중치, frozen hash 검증을 통과했다. 최종12모델×45patch×4개 조건=2,160개 개입에서 유한 수치, no-op 일치, checkpoint/state/RNG 불변을 확인했다. 기존6개 parent 모델과 같은45patch 인덱스인지도 대조했다. V4/V5는 모든 개입에서 복원 출력이 정확히 같았다. `--require-complete --require-final-audit`도 통과했다. 마지막 별도 기록 검사에서는 parent summary.json을 개별 모델과 구분해 검사했으며 실험 재실행은 없었다.

근거 루트: `outputs/hybrid_validation_20261111T055500Z`. 핵심 파일은 `analysis/analysis.json`, `final_checkpoint_audit/summary.json`, `completion_audit.json`, `final_host_process_check.json`, `v5_completed_diagnostics.json`이다. 설정·과정과 부분 결과는 `docs/HYBRID_VALIDATION_RUNS_20261111.md`에 보존했다. 공식 GitHub 비교는 `docs/SSM_OFFICIAL_COMPARISON_20261111.md`, `docs/UMAMBA_COMPARISON_20261111.md`에 있다.
