# 투고 당시 구현 복원 및 재현

확인일: 2026-11-09. 체크포인트 추론 재현은 완료했고, 초기화부터의 재학습은 진행 중이다.

## 어떤 실험을 찾았는가

기존 `evaluate_model.py`, `evaluate_model_v2.py`, `evaluate_and_visualize_multiview.py`가 모두 다음 실험의 첫 fold 체크포인트를 직접 선택한다.

```
/mnt/sda2/Jiwoong/Results/VMamba/MemAE_kfold_simplescan/res/trainResults/results_data_root_bs2_d32_64_128/data_0_05_8_32_CTCA_64/data_0_05_8_32_CTCA_64_20251023_231210
```

이 체크포인트는 원고의 AUROC 0.860 및 1.72M 파라미터와 일치하는 **투고 실험 후보**다. 원고의 모든 표·절제실험이 이 실행에서 나왔다는 뜻은 아니다. 원래 2025년 10월의 코드 사본은 찾지 못했다.

## 복원한 차이

처음 보존한 현재 원본 폴더의 모델은 resize convolution에 kernel 3을 사용해 2,109,241개 파라미터를 갖는다. 역사적 체크포인트는 같은 위치에 kernel 2를 사용하며 1,720,121개다. 차이 389,120개는 해당 가중치의 크기로 설명된다. 원본에 주석으로 남은 생성자와 다른 모델 폴더의 구현도 kernel 2를 뒷받침한다.

| 연산 | 보존된 현재 소스 | 체크포인트에 맞춰 복원 |
| --- | --- | --- |
| Encoder downsampling | Conv3d(k=3, s=2, p=1) | Conv3d(k=2, s=2, p=0) |
| Decoder upsampling | ConvTranspose3d(k=3, s=2, p=1, output_padding=1) | ConvTranspose3d(k=2, s=2, p=0, output_padding=0) |

복원본은 `/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/legacy_reproduction/source`에 별도로 두었다. 보존된 모델에서 생성자 두 곳만 바꾸고, SSM 방향 분리와 memory 연산은 당시 동작 그대로 유지했다. 정확한 차이와 SHA-256은 같은 디렉터리 상위의 `model_recovery.diff`, `recovery_manifest.json`에 있다. 원본 및 수정 버전 v2를 덮어쓰지 않았다.

따라서 앞서 완료한 수정 버전 v2와 투고 후보 실험의 차이에는 이 구조 차이도 포함된다. 현재 성능 차이를 특정 오류 수정의 효과로 해석할 수 없다.

## 저장 체크포인트 추론 재현 — 완료

원래 v11 패치, HU 변환, 평가 코드, 환자별 max 집계를 사용했다. 4개 체크포인트 모두 109개 state-dict key가 strict load에 성공했고, 총 6,157개 패치 점수가 저장 점수와 허용 오차 내에서 일치했다. 비교 기준은 `rtol=1e-4, atol=1e-7`, 최대 절대 MSE 차이는 3.451e-7이다. 환자별 순위 및 아래 AUROC/AP도 저장된 원점수에서 다시 계산한 결과와 일치한다.

| Fold | 정상 / 질환 | AUROC | AP |
| --- | --- | --- | --- |
| 1 | 5 / 20 | 0.860 | 0.963426 |
| 2 | 5 / 20 | 0.770 | 0.931187 |
| 3 | 5 / 20 | 0.630 | 0.883932 |
| 4 | 5 / 20 | 0.760 | 0.906443 |
| 단순 평균 | fold별 지표의 평균 | 0.755 | 0.921247 |

원고의 0.860은 이 후보 실행에서 첫 fold 값과 일치한다. 전체 4-fold 평균으로 보고하면 안 된다. 첫 fold AP는 0.963426이므로 원고의 AP 0.980은 여전히 확인·정정이 필요하다. AP는 질환 양성 환자의 `average_precision_score`로 계산했다. 같은 질환 20명이 fold마다 반복되므로 전체 행을 독립 환자로 합치지 않았다.

2026-11-09 후속 확인: 같은 첫 fold 점수의 사다리꼴 PR-AUC도 0.962385로, AP와 PR-AUC의 계산 방식 차이만으로 원고의 0.980을 설명하지 못한다. 4개 fold의 AP는 양성 환자 순위별 precision 평균으로 독립 검산해 일치했다. 기존 84개 실행의 요약 JSON(324개 fold)은 AUROC만 제공하며 환자 점수 배열과 AP가 없어, 그 요약만으로 AP 0.980의 출처를 확인할 수는 없다. 추가 확인 결과와 원점수 파일 해시는 `outputs/legacy_ap_provenance_audit_20261109.json`에 보존했다.

검증 결과:

```
/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/legacy_checkpoint_audit_20261109T063847Z/audit.json
```

같은 폴더의 CSV에 원래 점수와 복원 점수가 보존되어 있다. 실행 코드는 `legacy_reproduction/audit_checkpoints.py`다. **이는 저장 가중치의 추론 재현이며, 초기화부터의 학습 재현 성공을 의미하지 않는다.**

## 초기화부터의 재학습 — 진행 중

별도 실행:

```
/mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision/outputs/legacy_retrain_20261109T064401Z
```

GPU 0에서 원래 실험 관리자로 4개 fold를 순차 실행한다. 처음에 seed 42를 한 번 설정하고 fold 사이에 임의로 초기화하지 않는다. 원래 설정은 dims 32/64/128, depths 2/2/2, d_state 8, memory 300, batch 2, lr 1e-4, 최대 1,000 epochs, patience 10, entropy weight 0.0002다. 원래 v11 데이터를 사용한다.

원래 학습·평가 동작을 추적하기 위해 test normal을 early stopping에 쓰는 동작, shallow-copy checkpoint, 기존 entropy 축, SSM·memory 연산, 원래 그림·보고서 생성 순서까지 유지했다. **이 실행은 역사적 결과의 원인 확인용이며, 독립 validation을 갖춘 리비전 최종 평가가 아니다.**

실행에 필요한 변경은 데이터·출력의 절대 경로, 별도 작업 폴더, W&B 로컬 저장과 그림의 화면 없는 생성, 나중에 추가된 CLI의 4-stage 제한을 우회한 저장 3-stage 설정 직접 로딩이다. 학습기는 보존된 2025년 11월 소스이므로 정확한 2025년 10월 학습 경로가 같다고 아직 보증할 수 없다. Python/Torch/timm 버전은 역사적 메타데이터와 같지만 NumPy는 당시 2.2.6, 현재 1.26.4다.

각 실행 폴더에 원래/유효 설정, 실행 소스 사본과 해시, 당시 첫 epoch 기준값, 실행 상태 및 학습 로그를 남긴다. 이후 진행은 `legacy_reproduction/active_run.json`과 실행 폴더의 `run_state.json`, `training.log`를 읽어 확인한다. 초기 수치가 다른 경우 차이를 기록하고 완료로 표현하지 않는다.

시작 확인: 첫 fold의 첫 epoch가 유한한 손실로 완료되었다. 기록상 train loss는 당시 0.071618652, 이번 0.071641이고 validation loss는 당시 0.027229094, 이번 0.027694다. 출력 로그의 여섯 자리 정밀도에서 비교한 값이다. 첫 epoch부터 완전히 같은 학습 경로는 아니며 원인은 아직 분리하지 않았다. 세부 비교를 실행 폴더의 `first_epoch_comparison.json`에 보존했다. 이후 epoch도 진행 중이다. 원본 21개 소스와 보존 사본 21개의 SHA-256이 최초 manifest와 모두 일치함을 확인했다.


## 2026-11-10 재학습 완료 확인

당시 입력·복원 구현으로 수행한 새 4-fold 학습이 완료됐다. AUROC는 0.840 / 0.730 / 0.650 / 0.700, 평균 0.7300이며 평균 AP는 0.9173343이다. 이는 기존 저장 가중치의 평균 AUROC 0.755와 구분한다. 저장된 환자별 patch 점수에서 지표를 다시 계산했다. 역사적 학습 동작을 유지한 재학습이므로 최종 독립 검증 결과로 해석하지 않는다.

근거: outputs/legacy_candidate_sweep_20261109/comparison_snapshot_20261110.json.
