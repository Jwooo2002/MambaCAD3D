# 고정 epoch-100 메모리 대조 평가 — 2026-11-11 KST

사용자의 “ㄱㄱ” 승인 후 저장된 M0/M1/M3 × seed 42/123을 같은 평가 계획으로 실행했다. 새 학습, memory 개입, epoch/seed/집계법 탐색, 최종 채택은 하지 않았다. 실행: 00:00:57–00:01:55 KST, 약 57.7초(검산 별도).

## 사전 고정 조건

- Frozen legacy 모델과 float32 `(x+300)/700` 전처리, clipping 없음. 실제 입력 patch 규격은 64×32×32이며 batch 2로 평가했다.
- Fold 1의 정상 TRAIN 15명과 겹치지 않는 정상 5명(329 patch), 질환 20명(1,218 patch), 전체 1,547 patch. 모델 6개 모두 동일 환자·patch 순서.
- Patch 점수는 전체 voxel MSE, 환자 점수는 전체 patch의 max. 주지표 AUROC, 보조 AP와 정상/질환 점수 분포. 판정 threshold를 선택하지 않았다.
- Protocol, checkpoint·소스·데이터 hash는 질환 추론 전에 고정했다. 최초 준비 검사에서 임시로 적었던 64×16×16 형상 가정이 실패하여 실제 자료의 64×32×32로 검사 조건을 바로잡았다. 자료·모델·전처리는 변경하지 않았으며 그 시점에는 어떤 추론도 시작하지 않았다.

## 전체 결과

| Seed | 조건 | AUROC ↑ | AP ↑ |
| --- | --- | --- | --- |
| 42 | M0 · 기준 | 0.590 | 0.8778 |
| 42 | M1 · λ=0 | 0.640 | 0.8758 |
| 42 | M3 · λ=0 + memory decay=0 | 0.750 | 0.9305 |
| 123 | M0 · 기준 | 0.770 | 0.9409 |
| 123 | M1 · λ=0 | 0.720 | 0.9221 |
| 123 | M3 · λ=0 + memory decay=0 | 0.720 | 0.9221 |

두 seed의 기술적 평균 AUROC/AP는 M0 **0.680/0.9093**, M1 **0.680/0.8990**, M3 **0.735/0.9263**이다. 같은 환자를 반복 평가한 평균이며 독립 표본의 신뢰구간이 아니다.

M3는 M0 대비 seed 42에서 AUROC +0.16, seed 123에서 −0.05다. M1 대비는 +0.11/0.00이다. 평균 향상은 관찰됐지만 두 seed 모두에서 기준을 개선하지는 않았다. M1의 entropy 제거만으로도 평균 AUROC는 개선되지 않았다. M3의 메모리 수축·근접 수렴 완화와 질환 구분 성능 향상을 구분하며, 이번 결과로 최종 주 모델을 교체하지 않는다.

## 점수 분포

아래는 환자 max MSE ×1,000이다. Q1/Q3는 환자 점수의 25/75 percentile이다.

| Seed | 조건 | 정상 max 점수 중앙값 [Q1–Q3] | 질환 max 점수 중앙값 [Q1–Q3] |
| --- | --- | --- | --- |
| 42 | M0 | 5.880 [4.823–6.085] | 6.066 [4.591–7.717] |
| 42 | M1 | 0.816 [0.600–0.888] | 0.950 [0.736–1.103] |
| 42 | M3 | 2.807 [1.987–3.343] | 3.445 [2.775–4.039] |
| 123 | M0 | 3.435 [2.827–3.640] | 4.337 [3.462–5.696] |
| 123 | M1 | 2.543 [1.908–2.891] | 3.289 [2.344–4.782] |
| 123 | M3 | 2.371 [1.993–2.740] | 3.177 [2.223–4.641] |

Seed 42에서 M1은 정상과 질환의 복원 오차를 함께 크게 줄였고 두 점수 분포는 계속 겹친다. M3는 M1보다 정상 TRAIN MSE가 3.73배였음에도 AUROC가 0.64→0.75로 높았다. 정상 복원 오차만으로 이상 탐지 우열을 정할 수 없다는 관찰이다. Seed 123의 M1/M3는 원점수가 서로 다르지만 내림차순 정상/질환 label 순서가 같아 AUROC와 AP가 정확히 같다. 모든 조건에서 정상·질환 점수 범위가 겹친다.

![환자별 max 점수 분포](../outputs/memory_fixed_epoch100_eval_20261110T150000Z/patient_score_distributions.png)

## 검산과 범위

여섯 저장 모델의 frozen Python 소스 일치와 원 소스·설정·checkpoint·입력 파일 hash를 확인했다. 25명 각각의 첫/마지막 patch 전처리는 원 VideoDataset과 정확히 일치했다. 여섯 모델 모두 정상 TRAIN 고정 probe 45개에서 환자별 MSE가 과거 기록과 정확히 같았다(최대 절대 차이 0). 추론 전후 모델 상태와 checkpoint는 불변이다. 모든 환자의 전체 patch max를 재집계하고, 정상–질환 100쌍의 순위로 AUROC를, 순위별 precision으로 AP를 독립 검산해 일치했다. 해당 GPU 프로세스는 종료됐다.

정상 5명·질환 20명인 한 fold와 두 seed의 탐색 결과다. 이 환자들은 과거 legacy 평가에 이미 사용됐으며 새로운 독립 validation/미사용 test가 아니다. 질환 비율이 80%이므로 AP의 절대값을 그 비율과 함께 해석한다. 100 epoch는 고정 진단 시점이며 최적 checkpoint나 수렴 완료를 뜻하지 않는다. 기존 여러 fold의 최고/평균값을 이번 한 fold의 합격선으로 삼지 않는다.

현재 단계에서 새 손실·memory 초기화·warm-up 등을 추가하지 않았다. 정식 비교는 공통 구현 오류 수정, 재학습 전 분리한 정상 validation, label을 이용한 모델 선택과 평가의 분리, 모든 fold/seed 보고를 같은 조건으로 적용해야 한다. 기존에 본 자료의 노출 이력도 유지한다. Memory 기여와 최종 채택은 별도 검증이 남아 있다.

## 근거

- 실행 폴더: `/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/memory_fixed_epoch100_eval_20261110T150000Z`
- `protocol.json`, `frozen_hashes.json`, `input_manifest.json`, `results_summary.json`, `completed_audit.json`
- 조건·seed별 JSON: 모든 patch MSE와 환자 max, 원 checkpoint 경로/hash, probe 재현 검사 기록. 원영상은 업로드하지 않았다.
- 그림: `patient_score_distributions.png` / `.pdf`
- 실행/검산 코드: `memory_pilot/evaluate_fixed_checkpoints.py`, `memory_pilot/audit_fixed_evaluation.py`
